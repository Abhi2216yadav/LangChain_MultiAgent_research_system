"""A guided Streamlit interface for the four-stage research pipeline.

Run with ``streamlit run app.py``.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import streamlit as st

from src.config import ConfigurationError, get_llm, get_model_name
from src.piplines.pipline import stream_research
from src.utils import slugify

st.set_page_config(page_title="Field Notes", page_icon="✦", layout="wide")

MODEL_OPTIONS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
PHASES = {
    "planner": ("01", "Frame the question", "Turning the brief into focused research angles."),
    "researcher": ("02", "Gather evidence", "Finding sources and pulling out the useful material."),
    "writer": ("03", "Shape the draft", "Writing the report from the evidence collected."),
    "reviewer": ("04", "Check the work", "Reviewing accuracy, structure, and citations."),
}


def model_choices() -> list[str]:
    configured = get_model_name()
    return [configured] + [item for item in MODEL_OPTIONS if item != configured]


def make_readme(state: dict[str, Any], report: str) -> str:
    """Create a portable, human-readable handoff for the finished research."""
    query = state.get("query", "Research brief")
    sources = state.get("sources") or []
    questions = state.get("sub_questions") or []
    score = state.get("review_score") or 0
    score_line = f"{score}/10" if score else "Not scored"
    created = datetime.now().strftime("%d %B %Y, %H:%M")
    plan = "\n".join(f"{index}. {item}" for index, item in enumerate(questions, 1)) or "No planning notes recorded."
    source_list = "\n".join(f"{index}. {url}" for index, url in enumerate(sources, 1)) or "No source URLs recorded."
    return f"""# Research handoff

## Brief

{query}

## Final report

{report}

## Research plan

{plan}

## Source list

{source_list}

## Run notes

- Prepared: {created}
- Review score: {score_line}
- Writing passes: {state.get('revisions', 1)}
- Workflow: planning → research → writing → review

This file contains the final report and the practical trail behind it, so it can be shared, checked, or picked up by another reader without reopening the app.
"""


def render_phase(node: str, update: dict[str, Any]) -> None:
    number, title, description = PHASES[node]
    st.markdown(f"<div class='phase-kicker'>{number} · {title}</div>", unsafe_allow_html=True)
    st.caption(description)

    if node == "planner":
        questions = update.get("sub_questions") or []
        if questions:
            for question in questions:
                st.markdown(f"- {question}")
        else:
            st.caption("The planner did not return sub-questions.")
    elif node == "researcher":
        sources = update.get("sources") or []
        notes = update.get("notes") or []
        st.metric("Sources collected", len(sources))
        if sources:
            st.markdown("**Source trail**")
            for url in sources:
                st.markdown(f"- [{url}]({url})")
        with st.expander(f"Research notes ({len(notes)})"):
            st.markdown("\n\n---\n\n".join(notes) if notes else "No notes were captured.")
    elif node == "writer":
        draft = update.get("report") or ""
        st.caption(f"Writing pass {update.get('revisions', 1)} · {len(draft):,} characters")
        st.markdown(draft)
    else:
        approved = bool(update.get("approved"))
        score = update.get("review_score", 0)
        verdict = "Ready to publish" if approved else "Revision requested"
        st.markdown(f"**{verdict}** · quality check: **{score}/10**")
        feedback = (update.get("feedback") or "").strip()
        if feedback:
            st.info(feedback)
        elif approved:
            st.caption("The reviewer found no changes that need to be made.")


def render_result(result: dict[str, Any]) -> None:
    """Render a stored or freshly completed research result."""
    state = result["state"]
    report = result["report"]

    st.divider()
    st.markdown("<div class='final-label'>FINAL OUTPUT</div>", unsafe_allow_html=True)
    st.header("Ready to share")
    st.caption(
        f"Completed in {result['elapsed']:.0f} seconds · "
        f"review score {state.get('review_score', 0)}/10"
    )
    st.markdown(report)

    with st.expander("Research trail"):
        questions = state.get("sub_questions") or []
        sources = state.get("sources") or []
        notes = state.get("notes") or []
        plan_tab, sources_tab, notes_tab = st.tabs(["Plan", "Sources", "Notes"])

        with plan_tab:
            if questions:
                for index, question in enumerate(questions, 1):
                    st.markdown(f"{index}. {question}")
            else:
                st.caption("No planning notes recorded.")

        with sources_tab:
            if sources:
                for index, url in enumerate(sources, 1):
                    st.markdown(f"{index}. [{url}]({url})")
            else:
                st.caption("No source URLs recorded.")

        with notes_tab:
            if notes:
                st.markdown("\n\n---\n\n".join(notes))
            else:
                st.caption("No research notes recorded.")

    readme = make_readme(state, report)
    filename = slugify(result["query"]) or "research-report"
    left, right = st.columns(2)
    left.download_button(
        "Download report (.md)",
        report,
        f"{filename}.md",
        "text/markdown",
        width="stretch",
        key="download_report",
    )
    right.download_button(
        "Download project README (.md)",
        readme,
        f"{filename}-README.md",
        "text/markdown",
        width="stretch",
        key="download_readme",
    )


def run_brief(query: str, model: str, max_revisions: int) -> None:
    st.session_state["last_result"] = None
    os.environ["GEMINI_MODEL"] = model
    get_llm.cache_clear()
    state: dict[str, Any] = {"query": query}
    slots = {name: st.empty() for name in PHASES}
    started = time.perf_counter()

    progress = st.empty()
    progress.info("The research desk is working through the brief…")
    try:
        for chunk in stream_research(query, max_revisions=max_revisions):
            for node, update in chunk.items():
                if not isinstance(update, dict) or node not in PHASES:
                    continue
                state.update(update)
                with slots[node].container(border=True):
                    render_phase(node, update)
    except ConfigurationError as exc:
        st.error(str(exc), title="Configuration needed")
        return
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}", title="The research run stopped")
        return
    finally:
        progress.empty()

    report = (state.get("report") or "").strip()
    if not report:
        st.error("The pipeline finished without a final report.")
        return

    result = {
        "query": query,
        "state": state,
        "report": report,
        "elapsed": time.perf_counter() - started,
    }
    st.session_state["last_result"] = result
    render_result(result)


st.session_state.setdefault("last_result", None)

st.markdown(
   """<style>
    .stApp { background: #f6f3ed; color: #25221d; }
    .block-container { max-width: 1120px; padding-top: 3.5rem; padding-bottom: 5rem; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #25221d; }
    [data-testid='stSidebar'] { background: #eae5da; }
    [data-testid='stSidebar'], [data-testid='stSidebar'] * { color: #171512; }
    [data-testid='stSidebar'] [data-testid='stSelectbox'] div[data-baseweb='select'] *,
    [data-testid='stSidebar'] [data-testid='stSelectbox'] svg { color: #ffffff; }
    .eyebrow, .phase-kicker, .final-label { color: #9d4525; font-weight: 700; letter-spacing: .12em; font-size: .78rem; text-transform: uppercase; }
    .intro { font-size: 1.15rem; max-width: 44rem; color: #5d564d; }
    [data-testid='stVerticalBlockBorderWrapper'] { background: #fffdf9; border-color: #ded7cb; border-radius: 4px; padding: .65rem .9rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 3px; border: 1px solid #9d4525; color: #9d4525; background: transparent; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #9d4525; color: #fffdf9; }
    [data-testid='stTextArea'] textarea { background: #292724; color: #ffffff !important; caret-color: #ffffff; }
    [data-testid='stTextArea'] textarea::placeholder { color: #c9c3b8; opacity: 1; }
    </style>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Field Notes")
    st.caption("A small research desk powered by your four-stage workflow.")
    st.divider()
    model = st.selectbox("Model", model_choices())
    max_revisions = st.slider("Writing passes", 1, 3, 1)
    st.caption("The reviewer can send the draft back for another pass within this limit.")
    st.divider()
    st.markdown("**Workflow**\n\n01 Plan\n\n02 Research\n\n03 Write\n\n04 Review")

st.markdown("<div class='eyebrow'>Research workspace</div>", unsafe_allow_html=True)
st.title("Start with a good question.")
st.markdown("<p class='intro'>Give the team a research brief. You’ll see each phase complete in sequence, followed by a clean final report you can take with you.</p>", unsafe_allow_html=True)

with st.form("research_brief", clear_on_submit=False):
    query = st.text_area(
        "What would you like to understand?",
        placeholder="For example: How are small language models changing the economics of customer support in 2026?",
        height=125,
    )
    submitted = st.form_submit_button("Run the research", type="primary")

if submitted:
    query = query.strip()
    if not query:
        st.warning("Add a research question before starting.")
    else:
        st.markdown("---")
        st.markdown("<div class='eyebrow'>Your brief</div>", unsafe_allow_html=True)
        st.subheader(query)
        run_brief(query, model, max_revisions)
elif st.session_state.get("last_result"):
    render_result(st.session_state["last_result"])
