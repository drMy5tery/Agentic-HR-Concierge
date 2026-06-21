"""Deterministic fake LLM backends for keyless testing and the CLI self-test.

These satisfy the same call signature as ``call_llm`` — ``fn(messages) -> str`` —
so they can be injected anywhere the real adapter is used. They let us verify the
agent loop and the write-gate without any API key or network call.
"""
from __future__ import annotations

import json
from typing import Callable


class ScriptedLLM:
    """Returns a pre-set sequence of raw replies, one per call.

    Once the script is exhausted it returns a benign ``respond`` action (so a
    runaway loop ends gracefully rather than raising), and records how many calls
    were made for assertions.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"action": "respond",
                           "args": {"text": "(scripted backend exhausted)",
                                    "citation_ids": []}})


def action(name: str, *, confirm: bool = False, **args) -> str:
    """Helper to build a raw JSON action string for scripts/tests."""
    return json.dumps({"action": name, "args": args, "confirm": confirm})


def respond(text: str, citation_ids: list[str] | None = None) -> str:
    """Helper to build a raw JSON 'respond' action string."""
    return json.dumps({"action": "respond",
                       "args": {"text": text, "citation_ids": citation_ids or []},
                       "confirm": False})


def always(raw: str) -> Callable[[list[dict[str, str]]], str]:
    """A fake LLM that always returns the same raw reply (e.g. to test the cap)."""
    return lambda messages: raw
