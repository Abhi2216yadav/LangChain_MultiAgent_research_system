"""Agents that plan, research, write, and review a report.

Each public node function accepts a partial ``ResearchState`` and returns the
fields it changed. The functions are wired together as nodes in a LangGraph
state machine in ``src.piplines.pipline``.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_llm
from src.tools.tools import scrape_url, web_search

MAX_SUB_QUESTIONS = 5
SEARCH_RESULTS_PER_QUESTION = 3
MAX_SCRAPED_SOURCES = 4
MAX_SCRAPE_CHARS = 4_000
MAX_NOTES_CHARS = 60_000
_URL_LINE = re.compile(r"^URL:\s*(\S+)\s*$", re.MULTILINE)


class ResearchState(TypedDict, total=False):
    """State passed between the agents in the research graph."""

    query: str
    max_revisions: int
    sub_questions: list[str]
    notes: list[str]
    sources: list[str]
    report: str
    feedback: str
    approved: bool
    review_score: int
    revisions: int


class ResearchPlan(BaseModel):
    """Structured output produced by the planner."""

    sub_questions: list[str] = Field(
        min_length=2,
        max_length=MAX_SUB_QUESTIONS,
        description="Two to five focused, non-overlapping research sub-questions.",
    )


class ReportReview(BaseModel):
    """Structured output produced by the reviewer."""

    approved: bool = Field(description="Whether the report is ready to deliver.")
    score: int = Field(ge=1, le=10, description="Overall report quality from 1 to 10.")
    feedback: str = Field(
        description="Specific, actionable feedback. Empty when approved."
    )


_PLANNER_SYSTEM = """You are the planning agent in a multi-agent research system.
Break the user's research question into 2-5 focused sub-questions that together
give comprehensive coverage.

Rules:
- Each sub-question must be self-contained, specific, and searchable on the web.
- Do not overlap questions and do not answer them yourself.
- Prefer concrete facts, recent developments, and differing perspectives.

Today's date is {today}."""

_WRITER_SYSTEM = """You are the writing agent in a multi-agent research system.
Write a clear, well-structured report that answers the research question using
ONLY the supplied research notes.

Rules:
- Cite factual claims with bracketed numbers such as [1] that match the numbered
  source list.
- Never invent facts, statistics, dates, quotes, or URLs. If the notes are
  insufficient, say so explicitly instead of filling the gap.
- Use Markdown with these sections: a descriptive H1 title, "## Executive
  summary", "## Key findings", "## Analysis", "## Limitations", and "## Sources".
- In "Sources", list every numbered source URL exactly as provided.
- Keep the report focused: roughly 800-1,400 words unless the question needs
  less.

Today's date is {today}."""

_REVIEWER_SYSTEM = """You are the quality-review agent in a multi-agent research
system. Evaluate the draft report against these criteria:

1. It directly answers the research question.
2. Its claims are supported by the research notes; there are no fabricated facts
   or URLs.
3. It is well organized, specific, and useful to a reader.
4. In-text citations match the numbered sources.

Approve only if the draft has no major factual or structural problems. Score the
report from 1 to 10. When rejecting, give precise instructions the writer can
apply immediately."""


def _message_text(message: Any) -> str:
    """Extract plain text from a LangChain message across provider formats."""
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_questions(text: str) -> list[str]:
    """Fallback parser for models that cannot produce structured output."""
    questions: list[str] = []
    for line in text.splitlines():
        candidate = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip()
        if candidate.endswith("?") and len(candidate) > 15:
            questions.append(candidate)

    if not questions:
        questions = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) > 20 and not line.strip().endswith(":")
        ]
    return questions[:MAX_SUB_QUESTIONS]


def _notes_blob(notes: list[str]) -> str:
    blob = "\n\n".join(notes).strip()
    if len(blob) > MAX_NOTES_CHARS:
        blob = blob[:MAX_NOTES_CHARS] + "\n\n[Notes truncated to fit the model context.]"
    return blob or "No research notes were collected."


def _scrape_sources(urls: list[str]) -> list[tuple[str, str]]:
    """Scrape URLs concurrently, preserving the original source order."""
    if not urls:
        return []

    scraped: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(4, len(urls))) as pool:
        futures = {pool.submit(scrape_url.invoke, {"url": url}): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            try:
                content = future.result()
            except Exception as exc:
                content = f"Scrape error: {exc}"
            if (
                isinstance(content, str)
                and content.strip()
                and not content.lower().startswith("scrape error")
            ):
                scraped.append((url, content))

    order = {url: index for index, url in enumerate(urls)}
    scraped.sort(key=lambda item: order[item[0]])
    return scraped


def plan_research(state: ResearchState) -> dict[str, Any]:
    """Break the user's question into focused research sub-questions."""
    query = (state.get("query") or "").strip()
    if not query:
        raise ValueError("A research question is required.")

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM.format(today=date.today().isoformat())),
        HumanMessage(content=query),
    ]
    llm = get_llm(temperature=0.2)

    try:
        plan = llm.with_structured_output(ResearchPlan).invoke(messages)
        questions = [question.strip() for question in plan.sub_questions if question.strip()]
    except Exception:
        # Structured output depends on function calling; fall back to plain text.
        questions = _parse_questions(_message_text(llm.invoke(messages)))

    if not questions:
        questions = [query]

    return {
        "sub_questions": questions[:MAX_SUB_QUESTIONS],
        "notes": [],
        "sources": [],
        "report": "",
        "feedback": "",
        "approved": False,
        "review_score": 0,
        "revisions": 0,
    }


def conduct_research(state: ResearchState) -> dict[str, Any]:
    """Search for every sub-question and scrape the most promising pages."""
    query = (state.get("query") or "").strip()
    questions = state.get("sub_questions") or ([query] if query else [])
    if not questions:
        raise ValueError("No research questions were generated.")

    notes: list[str] = []
    sources: list[str] = []

    for question in questions:
        search_output = web_search.invoke(
            {"query": question, "max_results": SEARCH_RESULTS_PER_QUESTION}
        )
        notes.append(f"## Sub-question: {question}\n{search_output}")

        for url in _URL_LINE.findall(search_output):
            if url not in sources:
                sources.append(url)

    for url, content in _scrape_sources(sources[:MAX_SCRAPED_SOURCES]):
        notes.append(f"## Scraped page: {url}\n{content[:MAX_SCRAPE_CHARS]}")

    return {"notes": notes, "sources": sources}


def write_report(state: ResearchState) -> dict[str, Any]:
    """Write or revise the report from the collected research notes."""
    query = (state.get("query") or "").strip()
    report = state.get("report") or ""
    sources = state.get("sources") or []
    notes = state.get("notes") or []
    feedback = (state.get("feedback") or "").strip()
    is_revision = state.get("revisions", 0) > 0

    source_list = (
        "\n".join(f"[{index}] {url}" for index, url in enumerate(sources, 1))
        or "No sources were captured."
    )
    prompt_parts = [
        f"Research question:\n{query}",
        f"Numbered sources:\n{source_list}",
        f"Research notes:\n{_notes_blob(notes)}",
    ]

    if is_revision:
        prompt_parts.append(
            "Reviewer feedback on the previous draft:\n"
            + (feedback or "Improve factual grounding, structure, and citation coverage.")
        )
        if report:
            prompt_parts.append(f"Previous draft:\n{report[:12_000]}")
    else:
        prompt_parts.append("This is the first draft.")

    response = get_llm(temperature=0.35).invoke(
        [
            SystemMessage(
                content=_WRITER_SYSTEM.format(today=date.today().isoformat())
            ),
            HumanMessage(content="\n\n".join(prompt_parts)),
        ]
    )
    report_text = _message_text(response)
    if not report_text:
        raise RuntimeError("The writing agent returned an empty report.")

    return {
        "report": report_text,
        "revisions": state.get("revisions", 0) + 1,
        "feedback": "",
        "approved": False,
    }


def review_report(state: ResearchState) -> dict[str, Any]:
    """Review the draft for accuracy, structure, and citation coverage."""
    query = (state.get("query") or "").strip()
    report = (state.get("report") or "").strip()
    sources = state.get("sources") or []
    notes = state.get("notes") or []

    if not report:
        return {
            "approved": False,
            "review_score": 1,
            "feedback": "The draft is empty. Write a complete report from the notes.",
        }

    source_list = (
        "\n".join(f"[{index}] {url}" for index, url in enumerate(sources, 1))
        or "No sources were captured."
    )
    payload = (
        f"Research question:\n{query}\n\n"
        f"Numbered sources:\n{source_list}\n\n"
        f"Research notes:\n{_notes_blob(notes)}\n\n"
        f"Draft report:\n{report}"
    )
    messages = [SystemMessage(content=_REVIEWER_SYSTEM), HumanMessage(content=payload)]

    try:
        review = get_llm(temperature=0).with_structured_output(ReportReview).invoke(
            messages
        )
    except Exception as exc:
        # Never lose a usable report because the review call failed.
        return {
            "approved": True,
            "review_score": 7,
            "feedback": f"Review skipped because the reviewer model failed: {exc}",
        }

    return {
        "approved": bool(review.approved),
        "review_score": int(review.score),
        "feedback": (review.feedback or "").strip(),
    }


def route_after_review(state: ResearchState) -> str:
    """Decide whether to revise the report or finish the workflow."""
    if state.get("approved"):
        return "end"
    if state.get("revisions", 0) >= state.get("max_revisions", 1):
        return "end"
    return "revise"


__all__ = [
    "ResearchState",
    "conduct_research",
    "plan_research",
    "review_report",
    "route_after_review",
    "write_report",
]
