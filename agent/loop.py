"""The JSON-router agent loop and the human-in-the-loop write-gate.

Streamlit reruns top-to-bottom, so the loop is modelled as a small set of pure
functions over an explicit, caller-owned state (``conversation`` message list and
``seen_chunks`` map). A single user turn proceeds:

1. ``start_turn`` — a code-side sensitivity pre-check runs first; an obviously
   sensitive message is escalated immediately to a (gated) confidential ticket.
   Otherwise the loop runs.
2. ``step`` — reads execute and chain freely; the moment a WRITE tool is
   proposed the loop STOPS and returns a ``pending`` result (it does NOT execute
   the write). A ``respond`` ends the turn with text + resolved citations.
3. ``confirm_write`` / ``cancel_write`` — the caller renders a confirm card; on
   confirm the write executes and the loop resumes to produce the final message,
   on cancel a 'cancelled' result is fed back and the agent acknowledges.

Safety properties enforced here, never trusting the model:
- The write-gate is driven by the registry ``kind``, not the model's ``confirm``.
- Reference ids and citations are only ever surfaced from tool/chunk data.
- Malformed JSON (after one retry), an exceeded iteration cap, or any backend
  error all degrade to raising a ticket — the turn never crashes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import config
from agent.llm import LLMError, call_llm
from agent.tools import confirm_message, execute_tool

LLMFn = Callable[[list[dict[str, str]]], str]
TraceFn = Callable[[dict[str, Any]], None]

# Code-side sensitivity pre-check (point 6): obvious cases are caught here in
# addition to the system prompt's escalation instruction. Substrings only.
_HARASSMENT_KEYWORDS = (
    "harass", "bully", "bullie", "sexual", "molest", "assault", "abuse",
    "threaten", "threat", "retaliat", "stalk",
)
_DISCRIMINATION_KEYWORDS = (
    "discriminat", "racist", "racism", "casteist", "sexist", "homophob",
)
_SENSITIVE_SUMMARY = (
    "Confidential concern reported by the employee; requires sensitive handling."
)


@dataclass
class AgentResult:
    """Outcome of a loop step: either a finished turn or a pending write."""

    kind: str  # "final" | "pending"
    # final:
    text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    escalated: bool = False
    ticket: Optional[dict[str, Any]] = None
    # pending (a write awaiting confirmation):
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    confirm_prompt: str = ""
    sensitive: bool = False


# --- sensitivity pre-check --------------------------------------------------
def detect_sensitive(text: str) -> Optional[tuple[str, str]]:
    """Return ``(category, summary)`` if the text is obviously sensitive, else None."""
    lowered = (text or "").lower()
    if any(kw in lowered for kw in _HARASSMENT_KEYWORDS):
        return "harassment", _SENSITIVE_SUMMARY
    if any(kw in lowered for kw in _DISCRIMINATION_KEYWORDS):
        return "discrimination", _SENSITIVE_SUMMARY
    return None


# --- JSON parsing -----------------------------------------------------------
def _parse_action(text: str) -> Optional[dict[str, Any]]:
    """Parse one JSON action object from raw model text, tolerantly.

    Strips code fences and extracts the outermost ``{...}``. Returns None if the
    result is not a dict with an ``action`` key.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped[:4].lower() == "json":
            stripped = stripped[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "action" not in obj:
        return None
    if not isinstance(obj.get("args"), dict):
        obj["args"] = {}
    return obj


class _ParseError(Exception):
    """Raised when the model fails to return valid JSON, even after a retry."""


def _request_action(system_prompt: str, conversation: list[dict[str, str]], llm: LLMFn) -> tuple[dict[str, Any], str]:
    """Call the LLM, parse one action; retry once with a corrective nudge."""
    messages = [{"role": "system", "content": system_prompt}] + conversation
    raw = llm(messages)
    parsed = _parse_action(raw)
    if parsed is not None:
        return parsed, raw

    retry = messages + [{
        "role": "user",
        "content": "Your previous reply was not a single valid JSON object. "
                   "Reply with ONLY the JSON object, no prose or code fences.",
    }]
    raw = llm(retry)
    parsed = _parse_action(raw)
    if parsed is not None:
        return parsed, raw
    raise _ParseError()


# --- escalation fallback ----------------------------------------------------
def _escalate(state, reason: str, *, category: str = "policy", summary: Optional[str] = None) -> AgentResult:
    """Raise a fail-safe ticket and return a final result. Used when the loop
    cannot safely continue (bad JSON, exceeded cap, backend error)."""
    try:
        result = state.raise_ticket(category, summary or reason)
    except Exception:  # noqa: BLE001 - even the fail-safe must not crash
        result = {"reference": "", "team": "the HR team"}
    ref = result.get("reference", "")
    team = result.get("team", "the HR team")
    return AgentResult(
        kind="final",
        text=f"{reason} I've raised ticket {ref} so {team} can help you directly.",
        escalated=True,
        ticket=result,
    )


def _service_unavailable(exc: Exception) -> AgentResult:
    """LLM backend error (e.g. a rate limit) — tell the user and ask them to retry.

    Crucially this does NOT raise an HR ticket: a backend outage is an
    infrastructure issue, not something to route to a person.
    """
    text = str(exc)
    cause = "a rate limit" if ("429" in text or "RESOURCE_EXHAUSTED" in text) else "a temporary error"
    return AgentResult(
        kind="final",
        text=("The assistant is temporarily unavailable — the language model "
              f"returned {cause}. Please try again in a moment, or switch the LLM "
              "provider in the sidebar. (No ticket was raised.)"),
        escalated=False,
    )


# --- citation resolution ----------------------------------------------------
def _resolve_citations(citation_ids: Any, seen_chunks: dict[str, dict]) -> list[dict[str, Any]]:
    """Resolve model-supplied ids against chunks actually returned by search.

    Citations are built from chunk metadata, never from free text — any id the
    model invents (not in ``seen_chunks``) is dropped.
    """
    resolved: list[dict[str, Any]] = []
    for cid in citation_ids or []:
        chunk = seen_chunks.get(cid)
        if chunk:
            resolved.append({
                "id": cid,
                "source": chunk.get("source"),
                "section": chunk.get("section"),
            })
    return resolved


# --- the loop ---------------------------------------------------------------
def step(
    state,
    conversation: list[dict[str, str]],
    seen_chunks: dict[str, dict],
    registry: dict[str, dict],
    system_prompt: str,
    *,
    llm: LLMFn = call_llm,
    max_iters: int = config.MAX_AGENT_ITERATIONS,
    trace: Optional[TraceFn] = None,
) -> AgentResult:
    """Run reads until a write is proposed, a reply is produced, or we escalate."""
    grounding_nudged = False
    for _ in range(max_iters):
        try:
            action, raw = _request_action(system_prompt, conversation, llm)
        except LLMError as exc:
            # Backend unavailable (rate limit / network / auth): retry message, no ticket.
            return _service_unavailable(exc)
        except _ParseError:
            return _escalate(state, "I couldn't read that reliably.")
        except Exception:  # noqa: BLE001 - last-resort safety net for unexpected bugs
            return _escalate(state, "I hit an unexpected problem.")

        if trace:
            trace({"action": action})
        name = action.get("action")
        args = action.get("args") or {}

        if name == "respond":
            claimed = args.get("citation_ids") or []
            citations = _resolve_citations(claimed, seen_chunks)
            # Grounding guard: the model cited ids that no search_policy result
            # contains — i.e. it fabricated grounding (or answered from memory).
            # Reject once and force a real search before it may answer.
            if claimed and not citations and not grounding_nudged:
                grounding_nudged = True
                conversation.append({"role": "assistant", "content": raw})
                conversation.append({"role": "user", "content": json.dumps({
                    "error": "Those citation ids did not come from a search_policy "
                             "result. Call search_policy first and cite only ids it "
                             "returns; do not answer a policy question from memory."})})
                continue
            conversation.append({"role": "assistant", "content": raw})
            text = str(args.get("text", "")).strip() or "I'm not sure how to help with that."
            return AgentResult(kind="final", text=text, citations=citations)

        spec = registry.get(name)
        if spec is None:
            # Unknown action: feed the error back and let the model retry.
            conversation.append({"role": "assistant", "content": raw})
            conversation.append({"role": "user",
                                  "content": json.dumps({"error": f"Unknown action '{name}'."})})
            continue

        if spec["kind"] == "write":
            # WRITE-GATE: stop without executing; await human confirmation.
            return AgentResult(kind="pending", tool=name, args=args,
                               confirm_prompt=confirm_message(name, args))

        # READ tool: execute now and continue the loop.
        conversation.append({"role": "assistant", "content": raw})
        result = execute_tool(registry, name, args)
        if name == "search_policy" and isinstance(result, dict):
            for chunk in result.get("chunks", []):
                if isinstance(chunk, dict) and "id" in chunk:
                    seen_chunks[chunk["id"]] = chunk
        if trace:
            trace({"tool": name, "result": result})
        conversation.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(result)})

    return _escalate(state, "This needed more steps than allowed, so I've escalated it.")


def start_turn(
    state,
    conversation: list[dict[str, str]],
    seen_chunks: dict[str, dict],
    registry: dict[str, dict],
    system_prompt: str,
    user_message: str,
    *,
    llm: LLMFn = call_llm,
    max_iters: int = config.MAX_AGENT_ITERATIONS,
    trace: Optional[TraceFn] = None,
) -> AgentResult:
    """Begin a user turn: run the sensitivity pre-check, then the loop."""
    conversation.append({"role": "user", "content": user_message})

    sensitive = detect_sensitive(user_message)
    if sensitive is not None:
        category, summary = sensitive
        args = {"category": category, "summary": summary}
        return AgentResult(kind="pending", tool="raise_ticket", args=args,
                           confirm_prompt=confirm_message("raise_ticket", args),
                           sensitive=True)

    return step(state, conversation, seen_chunks, registry, system_prompt,
                llm=llm, max_iters=max_iters, trace=trace)


def confirm_write(
    state,
    conversation: list[dict[str, str]],
    seen_chunks: dict[str, dict],
    registry: dict[str, dict],
    system_prompt: str,
    pending: AgentResult,
    *,
    llm: LLMFn = call_llm,
    max_iters: int = config.MAX_AGENT_ITERATIONS,
    trace: Optional[TraceFn] = None,
) -> AgentResult:
    """Execute a confirmed write, then resume the loop for the final message."""
    name, args = pending.tool, pending.args
    conversation.append({"role": "assistant",
                         "content": json.dumps({"action": name, "args": args, "confirm": True})})
    result = execute_tool(registry, name, args)
    if trace:
        trace({"executed": name, "result": result})
    conversation.append({"role": "user", "content": "TOOL_RESULT " + json.dumps(result)})

    # Sensitive escalations get a fixed, careful acknowledgement — we do not ask
    # the model to phrase anything about a sensitive matter.
    if pending.sensitive:
        ref = result.get("reference", "")
        team = result.get("team", "People Ops")
        text = (f"Thank you for telling me. I've raised a confidential ticket "
                f"({ref}) with {team}, who handle these matters discreetly and "
                f"will follow up with you. This is informational support, not "
                f"legal or HR advice.")
        return AgentResult(kind="final", text=text, escalated=True, ticket=result)

    return step(state, conversation, seen_chunks, registry, system_prompt,
                llm=llm, max_iters=max_iters, trace=trace)


def cancel_write(
    state,
    conversation: list[dict[str, str]],
    seen_chunks: dict[str, dict],
    registry: dict[str, dict],
    system_prompt: str,
    pending: AgentResult,
    *,
    llm: LLMFn = call_llm,
    max_iters: int = config.MAX_AGENT_ITERATIONS,
    trace: Optional[TraceFn] = None,
) -> AgentResult:
    """Record a cancelled write and let the agent acknowledge it."""
    name, args = pending.tool, pending.args
    conversation.append({"role": "assistant",
                         "content": json.dumps({"action": name, "args": args, "confirm": True})})
    conversation.append({"role": "user",
                         "content": json.dumps({"cancelled": True,
                                                "note": "The employee cancelled this action; "
                                                        "it was NOT performed."})})
    if pending.sensitive:
        return AgentResult(kind="final",
                           text="Understood — I won't raise that ticket. If you change your "
                                "mind, I can raise it confidentially with People Ops at any time.")

    return step(state, conversation, seen_chunks, registry, system_prompt,
                llm=llm, max_iters=max_iters, trace=trace)
