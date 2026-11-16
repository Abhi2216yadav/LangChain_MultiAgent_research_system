"""Web search and page-scraping tools used by the research agents."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from readability import Document

from src.config import get_tavily_api_key

REQUEST_TIMEOUT_SECONDS = 20
MAX_PAGE_CHARS = 5_000
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
_REMOVABLE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form")


@lru_cache(maxsize=1)
def _tavily_client() -> Any:
    """Create the Tavily client lazily so importing this module never fails."""
    from tavily import TavilyClient

    return TavilyClient(api_key=get_tavily_api_key())


@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the web for recent, reliable information about a topic.

    Returns each result as a title, URL, and short snippet. Call this before
    scraping a page so you can choose useful sources.
    """
    query = (query or "").strip()
    if not query:
        return "Search error: the query is empty."

    try:
        limit = max(1, min(int(max_results), 8))
        results = _tavily_client().search(
            query=query,
            max_results=limit,
            search_depth="advanced",
        )
    except Exception as exc:  # Network and API errors are reported to the agent.
        return f"Search error for '{query}': {exc}"

    rows: list[str] = []
    for item in results.get("results", []):
        title = (item.get("title") or "(untitled)").strip()
        url = (item.get("url") or "").strip()
        snippet = re.sub(r"\s+", " ", item.get("content") or "").strip()
        rows.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet[:400]}")

    return "\n---\n".join(rows) if rows else f"No search results found for '{query}'."


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _soup_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_REMOVABLE_TAGS):
        tag.decompose()
    return _clean_text(soup.get_text(separator=" "))


@tool
def scrape_url(url: str) -> str:
    """Download a web page and return its clean, readable text.

    The result is capped at 5,000 characters and falls back through several
    extraction strategies so news articles, blogs, and documentation pages
    all work reasonably well.
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Scrape error: the URL must start with http:// or https://."

    headers = {
        "User-Agent": _USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        return f"Scrape error for {url}: {exc}"

    if response.status_code >= 400:
        return f"Scrape error for {url}: HTTP {response.status_code}."

    html = response.text
    if not html.strip():
        return f"Scrape error for {url}: the server returned an empty page."

    # Strategy 1: trafilatura (best for articles and blogs).
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False,
        )
    except Exception:
        extracted = None
    if extracted and len(extracted.strip()) > 200:
        return _clean_text(extracted)[:MAX_PAGE_CHARS]

    # Strategy 2: readability-lxml.
    try:
        summary_html = Document(html).summary()
        readable = _soup_text(summary_html)
    except Exception:
        readable = ""
    if len(readable) > 200:
        return readable[:MAX_PAGE_CHARS]

    # Strategy 3: raw BeautifulSoup text.
    raw_text = _soup_text(html)
    if raw_text:
        return raw_text[:MAX_PAGE_CHARS]

    return f"Scrape error for {url}: could not extract meaningful content."


__all__ = ["scrape_url", "web_search"]
