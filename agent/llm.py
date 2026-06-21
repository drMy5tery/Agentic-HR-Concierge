"""The single model-agnostic LLM adapter: ``call_llm()``.

Two interchangeable backends sit behind one function — Groq (primary) and
Google Gemini (backup) — selected by ``config.LLM_PROVIDER``. Both are driven in
**JSON mode** at low temperature, and both take the same normalised message list
``[{"role": "system"|"user"|"assistant", "content": str}, ...]`` and return the
model's raw text (expected to be a single JSON object — parsing happens in the
agent loop, not here).

This is a JSON router, **not** native function calling: the model decides the
next action by emitting JSON, and our dispatcher executes it. The provider SDKs
are imported lazily so importing this module never requires both to be installed.
"""
from __future__ import annotations

from typing import Optional

import config


class LLMError(RuntimeError):
    """Raised when a backend cannot be called (missing key, SDK, or API error)."""


def call_llm(
    messages: list[dict[str, str]],
    *,
    temperature: Optional[float] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Send ``messages`` to the configured backend and return the raw reply text.

    Args:
        messages: normalised chat messages (a leading ``system`` message is
            supported and mapped appropriately per provider).
        temperature: overrides ``config.LLM_TEMPERATURE`` when given.
        provider: overrides ``config.LLM_PROVIDER`` ("groq" or "gemini").
        model: overrides the provider's default model id.

    Raises:
        LLMError: on misconfiguration or any backend failure.
    """
    provider = (provider or config.LLM_PROVIDER).strip().lower()
    temperature = config.LLM_TEMPERATURE if temperature is None else temperature

    if provider == "groq":
        return _call_groq(messages, temperature, model or config.GROQ_MODEL)
    if provider == "gemini":
        return _call_gemini(messages, temperature, model or config.GEMINI_MODEL)
    raise LLMError(
        f"Unknown LLM_PROVIDER '{provider}'. Set it to 'groq' or 'gemini'."
    )


def _call_groq(messages: list[dict[str, str]], temperature: float, model: str) -> str:
    """Groq backend via its OpenAI-compatible chat completions API, JSON mode."""
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LLMError("The 'groq' package is not installed (pip install groq).") from exc

    try:
        client = Groq(api_key=config.GROQ_API_KEY)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - surface as a uniform LLMError
        raise LLMError(f"Groq request failed: {exc}") from exc


def _call_gemini(messages: list[dict[str, str]], temperature: float, model: str) -> str:
    """Google Gemini backend via the current ``google-genai`` SDK, JSON mode.

    The leading system message becomes ``system_instruction``; remaining messages
    are mapped to Gemini ``contents`` (role 'assistant' -> 'model').
    """
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise LLMError(
            "The 'google-genai' package is not installed (pip install google-genai)."
        ) from exc

    system_text = "\n\n".join(
        m["content"] for m in messages if m.get("role") == "system"
    )
    contents = [
        types.Content(
            role="model" if m.get("role") == "assistant" else "user",
            parts=[types.Part(text=m.get("content", ""))],
        )
        for m in messages
        if m.get("role") != "system"
    ]

    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_text or None,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""
    except Exception as exc:  # noqa: BLE001 - surface as a uniform LLMError
        raise LLMError(f"Gemini request failed: {exc}") from exc
