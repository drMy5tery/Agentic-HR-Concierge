"""Central configuration for the HR Concierge demo.

Every tunable is read from an environment variable (loaded from a local ``.env``
via python-dotenv) with a sensible default. Switching LLM provider or model is
therefore a one-line change in ``.env`` and never a code change.

No secrets live in this file — only variable names and non-secret defaults.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env if present. Real environment variables always take precedence.
load_dotenv()


def _get_float(name: str, default: float) -> float:
    """Read a float env var, falling back to ``default`` on missing/invalid."""
    try:
        value = os.getenv(name)
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    """Read an int env var, falling back to ``default`` on missing/invalid."""
    try:
        value = os.getenv(name)
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


# --- LLM provider / models --------------------------------------------------
# Provider is "groq" (primary) or "gemini" (backup). Both sit behind a single
# call_llm() adapter, so this flag is the only switch needed to swap backends.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").strip().lower()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Low temperature keeps the JSON router deterministic.
LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.0)

# --- Retrieval (RAG) --------------------------------------------------------
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K: int = _get_int("RETRIEVAL_TOP_K", 4)
# Minimum cosine similarity for a chunk to count as relevant. Below this, the
# agent treats the question as uncovered and escalates rather than guessing.
RETRIEVAL_THRESHOLD: float = _get_float("RETRIEVAL_THRESHOLD", 0.30)

# --- Agent loop -------------------------------------------------------------
# Hard cap on tool-call iterations per user turn; exceeding it escalates.
MAX_AGENT_ITERATIONS: int = _get_int("MAX_AGENT_ITERATIONS", 6)

# Where the policy markdown lives (used by the RAG ingest step).
POLICIES_DIR: str = os.getenv(
    "POLICIES_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "policies"),
)
