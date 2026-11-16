"""LangGraph workflow that wires the research agents together.

Flow:
    planner -> researcher -> writer -> reviewer
                  ^                       |
                  +------ revise ---------+

The reviewer either approves the report or sends it back to the writer. The
number of writing passes is bounded by ``max_revisions``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.agents.agent import (
    ResearchState,
    conduct_research,
    plan_research,
    review_report,
    route_after_review,
    write_report,
)

NODE_LABELS = {
    "planner": "Planner",
    "researcher": "Researcher",
    "writer": "Writer",
    "reviewer": "Reviewer",
}


def build_research_graph():
    """Build and compile the multi-agent research graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", plan_research)
    graph.add_node("researcher", conduct_research)
    graph.add_node("writer", write_report)
    graph.add_node("reviewer", review_report)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"revise": "writer", "end": END},
    )

    return graph.compile()


def stream_research(
    query: str, max_revisions: int = 1
) -> Iterator[dict[str, dict[str, Any]]]:
    """Yield one update per completed graph node.

    Each item has the shape ``{"node_name": {state_changes}}``, which makes it
    easy to show live progress in the CLI or Streamlit UI.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("A research question is required.")

    passes = max(1, int(max_revisions))
    initial: ResearchState = {
        "query": query,
        "max_revisions": passes,
        "sub_questions": [],
        "notes": [],
        "sources": [],
        "report": "",
        "feedback": "",
        "approved": False,
        "review_score": 0,
        "revisions": 0,
    }
    config = {"recursion_limit": 10 + (2 * passes)}

    yield from build_research_graph().stream(
        initial,
        stream_mode="updates",
        config=config,
    )


def run_research(query: str, max_revisions: int = 1) -> ResearchState:
    """Run the graph to completion and return the final state."""
    final: dict[str, Any] = {
        "query": (query or "").strip(),
        "max_revisions": max(1, int(max_revisions)),
    }
    for chunk in stream_research(query, max_revisions):
        for update in chunk.values():
            if isinstance(update, dict):
                final.update(update)
    return cast(ResearchState, final)


__all__ = ["NODE_LABELS", "build_research_graph", "run_research", "stream_research"]
