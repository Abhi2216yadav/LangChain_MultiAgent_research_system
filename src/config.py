"""Central configuration for the multi-agent research system.

All secrets are read from the project ``.env`` file (or the process
environment). Nothing in this module writes secrets to disk or logs them.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


class ConfigurationError(RuntimeError):
    """Raised when a required API key or setting is missing."""


def _clean(value: str | None) -> str:
    return (value or "").strip()


def get_gemini_api_key() -> str:
    """Return the Gemini API key or explain exactly how to configure it."""
    key = _clean(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    if not key:
        raise ConfigurationError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your "
            "Google Gemini API key from https://aistudio.google.com/apikey"
        )
    return key


def get_tavily_api_key() -> str:
    """Return the Tavily API key or explain exactly how to configure it."""
    key = _clean(os.getenv("TAVILY_API_KEY"))
    if not key:
        raise ConfigurationError(
            "TAVILY_API_KEY is missing. Copy .env.example to .env and add your "
            "Tavily API key from https://app.tavily.com"
        )
    return key


def get_model_name() -> str:
    """Return the configured Gemini model name."""
    return _clean(os.getenv("GEMINI_MODEL")) or DEFAULT_GEMINI_MODEL


def get_max_revisions() -> int:
    """Return the default number of writing passes (always at least 1)."""
    try:
        return max(1, int(_clean(os.getenv("MAX_REVISIONS")) or "1"))
    except ValueError:
        return 1


@lru_cache(maxsize=8)
def get_llm(temperature: float = 0.2) -> Any:
    """Return a cached chat model configured from the environment.

    Set ``LLM_PROVIDER=openai`` to use OpenAI instead of Gemini. Gemini is the
    default because the bundled ``.env.example`` targets Gemini.
    """
    provider = (_clean(os.getenv("LLM_PROVIDER")) or "google").lower()

    if provider in {"google", "gemini"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=get_model_name(),
            temperature=temperature,
            api_key=get_gemini_api_key(),
            max_retries=2,
            timeout=180,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = _clean(os.getenv("OPENAI_API_KEY"))
        if not api_key:
            raise ConfigurationError(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY is missing. "
                "Add it to .env or set LLM_PROVIDER=google."
            )
        return ChatOpenAI(
            model=_clean(os.getenv("OPENAI_MODEL")) or "gpt-4o-mini",
            temperature=temperature,
            api_key=api_key,
            timeout=180,
            max_retries=2,
        )

    raise ConfigurationError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'google' or 'openai'."
    )
