"""Small shared helpers used by the CLI and the Streamlit app."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify(text: str, max_length: int = 60) -> str:
    """Turn a research question into a safe, readable file name."""
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_length].rstrip("-") or "research"


def save_report(
    query: str, report: str, directory: str | Path = "reports"
) -> Path:
    """Write a Markdown report to disk and return its path."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = Path(directory) / f"{timestamp}-{slugify(query)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<!-- Research question: {query} -->\n\n{report}\n",
        encoding="utf-8",
    )
    return path


__all__ = ["save_report", "slugify"]
