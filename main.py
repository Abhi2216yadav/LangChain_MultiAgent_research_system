"""Command-line entry point for the multi-agent research system.

Examples:
    python main.py "What are the latest developments in AI agents?"
    python main.py "How is AI regulation changing?" --max-revisions 2 --save
"""

from __future__ import annotations

import argparse
import os
import sys

# Legacy Windows consoles default to cp1252, which cannot encode characters
# such as box-drawing diagrams or bullets that models often put in reports.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel

    from src.config import (
        ConfigurationError,
        get_llm,
        get_max_revisions,
        get_model_name,
    )
    from src.piplines.pipline import stream_research
    from src.utils import save_report
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(
        f"\nMissing Python package: {missing}\n"
        f"Interpreter: {sys.executable}\n\n"
        "Install the dependencies into this interpreter, then try again:\n"
        "    python -m pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multi-agent research workflow from the command line."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Research question. If omitted, you will be prompted for one.",
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=get_max_revisions(),
        metavar="N",
        help="Maximum number of writing passes (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        help="Override GEMINI_MODEL for this run (for example gemini-3.5-flash).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the finished report under ./reports/.",
    )
    return parser


def _summarize(node: str, update: dict) -> str:
    if node == "planner":
        return f"planned {len(update.get('sub_questions') or [])} sub-questions"
    if node == "researcher":
        return (
            f"collected {len(update.get('notes') or [])} note blocks from "
            f"{len(update.get('sources') or [])} sources"
        )
    if node == "writer":
        return (
            f"draft {update.get('revisions', 1)} ready "
            f"({len(update.get('report') or ''):,} characters)"
        )
    if node == "reviewer":
        verdict = "approved" if update.get("approved") else "requested a revision"
        return f"{verdict} (quality {update.get('review_score', 0)}/10)"
    return "done"


def _run(query: str, max_revisions: int) -> dict:
    final: dict = {"query": query, "max_revisions": max(1, max_revisions)}

    for chunk in stream_research(query, max_revisions=max_revisions):
        for node, update in chunk.items():
            if isinstance(update, dict):
                final.update(update)
            console.print(
                f"[green]OK[/green] [bold]{node.title()}[/bold] - "
                f"{_summarize(node, update or {})}"
            )
            if node == "planner":
                for question in (update or {}).get("sub_questions") or []:
                    console.print(f"   [dim]- {question}[/dim]")

    return final


def main() -> int:
    args = _build_parser().parse_args()

    if args.model:
        os.environ["GEMINI_MODEL"] = args.model
        get_llm.cache_clear()

    query = " ".join(args.query).strip()
    if not query:
        query = console.input("[bold]Research question:[/bold] ").strip()
    if not query:
        console.print("[red]No research question was provided.[/red]")
        return 2

    max_revisions = max(1, args.max_revisions)
    console.print(
        Panel(
            f"[bold]{query}[/bold]\n\n"
            f"[dim]Model: {get_model_name()} | Writing passes: {max_revisions}[/dim]",
            title="Research request",
            border_style="cyan",
        )
    )

    try:
        final = _run(query, max_revisions)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research cancelled.[/yellow]")
        return 130
    except ConfigurationError as exc:
        console.print(Panel(str(exc), title="Configuration error", border_style="red"))
        return 1
    except Exception:
        console.print("[red]The research run failed:[/red]")
        console.print_exception(show_locals=False)
        return 1

    report = (final.get("report") or "").strip()
    if not report:
        console.print("[red]The workflow finished without producing a report.[/red]")
        return 1

    console.print()
    console.rule("[bold]Research report[/bold]")
    console.print(Markdown(report))

    sources = final.get("sources") or []
    if sources:
        console.print()
        console.print("[bold]Sources[/bold]")
        for index, url in enumerate(sources, 1):
            console.print(f"  [{index}] {url}", style="dim")

    if args.save:
        path = save_report(query, report)
        console.print(f"\n[green]Saved report:[/green] {path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
