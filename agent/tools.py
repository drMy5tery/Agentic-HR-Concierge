"""Tool registry, dispatch, and the write-gate primitives.

Every tool is tagged ``"read"`` or ``"write"``. The agent loop enforces the
human-in-the-loop gate **from this ``kind`` tag, in code** — it never trusts the
model's ``confirm`` field. Read tools may execute and chain freely; write tools
are held for explicit confirmation before they run.

The registry is built per session and bound to a concrete :class:`KekaState`
(and, from Stage 3, a policy searcher), so the tool functions close over the
state they mutate.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from mock.keka import LEAVE_TYPES, route_category

# Type alias: a tool function takes the model-supplied args dict and returns a
# JSON-serialisable result dict.
ToolFn = Callable[[dict[str, Any]], dict[str, Any]]


def build_registry(state, searcher: Optional[Callable[[str], dict]] = None) -> dict[str, dict]:
    """Build the tool registry bound to ``state`` (and an optional policy searcher).

    ``search_policy`` is only registered when a ``searcher`` is supplied (wired in
    Stage 3), so the prompt's advertised tools always match what is callable.
    """
    registry: dict[str, dict] = {
        "check_leave": {
            "kind": "read",
            "fn": lambda args: state.check_leave(),
            "args": {},
            "description": "Get the employee's current leave balances "
                           "(casual, sick, earned).",
        },
        "get_payslip": {
            "kind": "read",
            "fn": lambda args: state.get_payslip(args.get("month")),
            "args": {"month": "optional 'YYYY-MM'; omit for the latest payslip"},
            "description": "Fetch a payslip for a given month, or the latest one.",
        },
        "apply_leave": {
            "kind": "write",
            "fn": lambda args: state.apply_leave(
                args.get("leave_type"), args.get("days"), args.get("start_date")
            ),
            "args": {
                "leave_type": f"one of {', '.join(LEAVE_TYPES)}",
                "days": "whole number of days, >= 1",
                "start_date": "YYYY-MM-DD",
            },
            "description": "Apply for leave on the employee's behalf. "
                           "State-changing — requires confirmation.",
        },
        "raise_ticket": {
            "kind": "write",
            "fn": lambda args: state.raise_ticket(
                args.get("category"), args.get("summary")
            ),
            "args": {
                "category": "harassment | discrimination | payroll | it | "
                            "policy | benefits | general",
                "summary": "a short, respectful description of the issue",
            },
            "description": "Raise a routed support ticket. Use this to escalate "
                           "sensitive matters and questions the policies don't "
                           "cover. State-changing — requires confirmation.",
        },
    }

    if searcher is not None:
        registry["search_policy"] = {
            "kind": "read",
            "fn": lambda args: searcher(args.get("query", "")),
            "args": {"query": "the policy question, as a search string"},
            "description": "Search the HR policy documents. Returns relevant "
                           "chunks with ids; answer ONLY from these and cite "
                           "their ids.",
        }
    return registry


def catalogue_text(registry: dict[str, dict]) -> str:
    """Render the registry as a human-readable catalogue for the system prompt."""
    lines: list[str] = []
    for name, spec in registry.items():
        args = spec.get("args") or {}
        arg_text = (
            ", ".join(f'"{k}" ({v})' for k, v in args.items()) if args else "none"
        )
        lines.append(
            f'- {name} [{spec["kind"].upper()}]: {spec["description"]} '
            f"Args: {arg_text}."
        )
    return "\n".join(lines)


def is_write(registry: dict[str, dict], name: str) -> bool:
    """True if ``name`` is a registered write (state-changing) tool."""
    spec = registry.get(name)
    return bool(spec) and spec.get("kind") == "write"


def execute_tool(registry: dict[str, dict], name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name, never raising — tool errors become a result dict.

    The write-gate is *not* enforced here; the loop decides whether a write may
    be executed. This function is the single dispatch point once that decision
    has been made (for reads, or for a confirmed write).
    """
    spec = registry.get(name)
    if spec is None:
        return {"ok": False, "error": f"Unknown tool '{name}'."}
    if not isinstance(args, dict):
        args = {}
    try:
        result = spec["fn"](args)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        return {"ok": False, "error": f"Tool '{name}' failed: {exc}"}
    return result if isinstance(result, dict) else {"ok": True, "result": result}


def confirm_message(name: str, args: dict[str, Any]) -> str:
    """Plain-English confirmation prompt for a pending write action."""
    if name == "apply_leave":
        lt = args.get("leave_type", "?")
        days = args.get("days", "?")
        start = args.get("start_date", "?")
        return f"About to apply {days} day(s) of {lt} leave starting {start}. Confirm?"
    if name == "raise_ticket":
        category = str(args.get("category", "general"))
        summary = str(args.get("summary", "")).strip()
        team, confidential = route_category(category, summary)
        routing = (
            "routed confidentially to People Ops"
            if confidential
            else f"routed to {team}"
        )
        shown = f': "{summary}"' if summary else ""
        return f"About to raise a {category} ticket ({routing}){shown}. Confirm?"
    return f"About to run {name} with {args}. Confirm?"
