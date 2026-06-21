"""Tiny CLI harness for the agent loop (no Streamlit, no UI).

Modes
-----
Interactive REPL against the configured LLM (needs a key in ``.env``)::

    python -m agent.cli

Deterministic self-test that needs NO API key — prints each emitted JSON action
and demonstrates the write-gate, the sensitivity escalation, and the
JSON-failure fallback::

    python -m agent.cli --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from agent.fake_llm import ScriptedLLM, action, respond
from agent.loop import cancel_write, confirm_write, start_turn
from agent.prompts import build_system_prompt
from agent.tools import build_registry
from mock.keka import new_state


def _new_session(searcher=None):
    """Build a fresh mock state + registry + system prompt for one CLI session."""
    state = new_state()
    registry = build_registry(state, searcher)
    system_prompt = build_system_prompt(registry, state.employee, date.today().isoformat())
    return state, registry, system_prompt


def _print_event(event: dict) -> None:
    """Trace callback: surface every emitted action / tool result."""
    if "action" in event:
        print("   -> action:", json.dumps(event["action"]))
    elif "executed" in event:
        print(f"   ** executed {event['executed']}:", json.dumps(event["result"]))
    elif "tool" in event:
        print(f"   <- {event['tool']} result:", json.dumps(event["result"]))


# --- interactive REPL -------------------------------------------------------
def interactive() -> int:
    state, registry, system_prompt = _new_session()
    emp = state.employee
    print("HR Concierge CLI — type 'quit' to exit.")
    print(f"Acting for {emp['name']} ({emp['employee_id']}). "
          f"Balances: {state.balances}\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", ":q"}:
            break

        conversation: list[dict] = []
        seen_chunks: dict = {}
        result = start_turn(state, conversation, seen_chunks, registry,
                            system_prompt, user, trace=_print_event)

        while result.kind == "pending":
            print(f"\n[confirm] {result.confirm_prompt}")
            answer = input("Confirm? [y/N] ").strip().lower()
            if answer in {"y", "yes"}:
                result = confirm_write(state, conversation, seen_chunks, registry,
                                       system_prompt, result, trace=_print_event)
            else:
                result = cancel_write(state, conversation, seen_chunks, registry,
                                      system_prompt, result, trace=_print_event)

        print(f"\nHR Concierge> {result.text}")
        if result.citations:
            cites = "; ".join(f"{c['source']} / {c['section']}" for c in result.citations)
            print(f"   (sources: {cites})")
        print(f"   [balances: {state.balances}]\n")
    return 0


# --- deterministic self-test ------------------------------------------------
class _Checks:
    def __init__(self):
        self.failures = 0

    def expect(self, label: str, condition: bool) -> None:
        mark = "PASS" if condition else "FAIL"
        if not condition:
            self.failures += 1
        print(f"   [{mark}] {label}")


def selftest() -> int:
    checks = _Checks()

    # Scenario 1 — read tool executes, no state change.
    print("\n=== 1. Read: 'What is my leave balance?' ===")
    state, registry, sp = _new_session()
    fake = ScriptedLLM([action("check_leave"), respond("Your balances are shown above.")])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "What is my leave balance?",
                     llm=fake, trace=_print_event)
    checks.expect("ended with a reply", res.kind == "final")
    checks.expect("balances unchanged by a read", state.balances == {"casual": 8, "sick": 10, "earned": 15})

    # Scenario 2 — write is gated, not executed, until confirmed.
    print("\n=== 2. Write-gate: 'Apply 2 days casual leave from 2030-06-25' ===")
    state, registry, sp = _new_session()
    fake = ScriptedLLM([
        action("apply_leave", confirm=True, leave_type="casual", days=2, start_date="2030-06-25"),
        respond("Done — your casual leave has been applied."),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "Apply 2 days casual leave from 2030-06-25", llm=fake, trace=_print_event)
    checks.expect("write proposed as PENDING (gated)", res.kind == "pending" and res.tool == "apply_leave")
    checks.expect("NOT executed before confirm (balance still 8)", state.balances["casual"] == 8)
    print("   ... user confirms ...")
    res = confirm_write(state, conv, seen, registry, sp, res, llm=fake, trace=_print_event)
    checks.expect("executed after confirm (balance now 6)", state.balances["casual"] == 6)
    checks.expect("a real LEAVE reference was recorded",
                  any(r["reference"].startswith("LEAVE-") for r in state.leave_records))

    # Scenario 3 — sensitive message escalates WITHOUT calling the LLM.
    print("\n=== 3. Sensitive: 'I think my manager is harassing me' ===")
    state, registry, sp = _new_session()
    fake = ScriptedLLM([])  # must not be needed
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "I think my manager is harassing me", llm=fake, trace=_print_event)
    checks.expect("escalated to a gated ticket, no policy answer",
                  res.kind == "pending" and res.tool == "raise_ticket" and res.sensitive)
    checks.expect("no LLM call was made for the sensitive pre-check", fake.calls == 0)
    print("   ... user confirms ...")
    res = confirm_write(state, conv, seen, registry, sp, res, llm=fake, trace=_print_event)
    checks.expect("routed confidentially to People Ops",
                  res.ticket and res.ticket["team"] == "People Ops (Confidential)" and res.ticket["confidential"])

    # Scenario 4 — malformed JSON twice -> fail-safe escalation (never crashes).
    print("\n=== 4. Fail-safe: model returns invalid JSON ===")
    state, registry, sp = _new_session()
    fake = ScriptedLLM(["not json at all", "still not json"])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "Tell me a joke about work",
                     llm=fake, trace=_print_event)
    checks.expect("degraded to an escalation ticket", res.kind == "final" and res.escalated)
    checks.expect("parsed once and retried once (2 calls)", fake.calls == 2)

    print(f"\nSelf-test complete: {'all checks passed' if checks.failures == 0 else str(checks.failures) + ' FAILED'}.")
    return 1 if checks.failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HR Concierge agent-loop CLI harness.")
    parser.add_argument("--selftest", action="store_true",
                        help="run the deterministic, keyless verification scenarios")
    args = parser.parse_args(argv)
    return selftest() if args.selftest else interactive()


if __name__ == "__main__":
    sys.exit(main())
