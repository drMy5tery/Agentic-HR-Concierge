"""Deterministic tests for the agent loop and write-gate, using a scripted fake
LLM (no API key, no network). These pin down the safety-critical behaviour:
the gate, escalation, citation integrity, and the never-crash fallbacks.
"""
import pytest

from agent.fake_llm import ScriptedLLM, action, always, respond
from agent.loop import (
    AgentResult,
    _parse_action,
    cancel_write,
    confirm_write,
    detect_sensitive,
    start_turn,
)
from agent.prompts import build_system_prompt
from agent.tools import build_registry
from mock.keka import new_state

TODAY = "2026-06-21"


def session(searcher=None):
    """Fresh (state, registry, system_prompt) for one scripted turn."""
    state = new_state()
    registry = build_registry(state, searcher)
    sp = build_system_prompt(registry, state.employee, TODAY)
    return state, registry, sp


# --- JSON parsing -----------------------------------------------------------
@pytest.mark.parametrize("raw,expected_action", [
    ('{"action":"respond","args":{"text":"hi"}}', "respond"),
    ('```json\n{"action":"check_leave","args":{}}\n```', "check_leave"),
    ('Sure! {"action":"check_leave","args":{}} done.', "check_leave"),
])
def test_parse_action_tolerant(raw, expected_action):
    parsed = _parse_action(raw)
    assert parsed is not None and parsed["action"] == expected_action
    assert isinstance(parsed["args"], dict)


@pytest.mark.parametrize("raw", ["not json", "", "{\"foo\": 1}", "{bad json"])
def test_parse_action_rejects_invalid(raw):
    assert _parse_action(raw) is None


def test_parse_action_coerces_bad_args():
    parsed = _parse_action('{"action":"x","args":"oops"}')
    assert parsed["args"] == {}


# --- sensitivity pre-check --------------------------------------------------
@pytest.mark.parametrize("text,category", [
    ("I think my manager is harassing me", "harassment"),
    ("There is bullying in my team", "harassment"),
    ("I'm being discriminated against", "discrimination"),
])
def test_detect_sensitive_flags(text, category):
    out = detect_sensitive(text)
    assert out is not None and out[0] == category


def test_detect_sensitive_ignores_benign():
    assert detect_sensitive("How many casual leave days do I get?") is None


# --- read tools chain freely ------------------------------------------------
def test_read_tool_executes_and_responds():
    state, registry, sp = session()
    fake = ScriptedLLM([action("check_leave"), respond("Here are your balances.")])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "what's my balance?", llm=fake)
    assert res.kind == "final"
    assert state.balances == {"casual": 8, "sick": 10, "earned": 15}  # unchanged
    assert any("TOOL_RESULT" in m["content"] for m in conv)


def test_read_executes_even_when_model_sets_confirm_true():
    # confirm is only a hint; a READ runs regardless.
    state, registry, sp = session()
    fake = ScriptedLLM([action("check_leave", confirm=True), respond("ok")])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "balance please", llm=fake)
    assert res.kind == "final"


# --- the write-gate ---------------------------------------------------------
def test_write_is_gated_not_executed():
    state, registry, sp = session()
    fake = ScriptedLLM([action("apply_leave", confirm=True,
                              leave_type="casual", days=2, start_date="2030-06-25")])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "apply 2 casual days from 2030-06-25", llm=fake)
    assert res.kind == "pending" and res.tool == "apply_leave"
    assert state.balances["casual"] == 8  # NOT executed


def test_write_gate_ignores_model_confirm_false():
    # The gate is enforced from the registry 'kind', never the model's confirm.
    state, registry, sp = session()
    fake = ScriptedLLM([action("apply_leave", confirm=False,
                              leave_type="casual", days=1, start_date="2030-06-25")])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "apply 1 casual day from 2030-06-25", llm=fake)
    assert res.kind == "pending"
    assert state.balances["casual"] == 8


def test_reads_then_write_gates_after_chaining():
    state, registry, sp = session()
    fake = ScriptedLLM([
        action("check_leave"),
        action("apply_leave", confirm=True, leave_type="casual", days=2, start_date="2030-06-25"),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "check my balance then apply 2 casual days from 2030-06-25", llm=fake)
    assert res.kind == "pending"
    assert any("TOOL_RESULT" in m["content"] for m in conv)  # the read ran
    assert state.balances["casual"] == 8                      # the write did not


def test_confirm_executes_write_and_finishes():
    state, registry, sp = session()
    fake = ScriptedLLM([
        action("apply_leave", confirm=True, leave_type="casual", days=2, start_date="2030-06-25"),
        respond("Done — casual leave applied."),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "apply 2 casual days from 2030-06-25", llm=fake)
    res = confirm_write(state, conv, seen, registry, sp, res, llm=fake)
    assert res.kind == "final"
    assert state.balances["casual"] == 6
    assert any(r["reference"].startswith("LEAVE-") for r in state.leave_records)


def test_cancel_does_not_execute_write():
    state, registry, sp = session()
    fake = ScriptedLLM([
        action("apply_leave", confirm=True, leave_type="casual", days=2, start_date="2030-06-25"),
        respond("No problem, I've cancelled that."),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "apply 2 casual days from 2030-06-25", llm=fake)
    res = cancel_write(state, conv, seen, registry, sp, res, llm=fake)
    assert res.kind == "final"
    assert state.balances["casual"] == 8  # unchanged
    assert any("cancelled" in m["content"] for m in conv)


def test_confirmed_write_validation_failure_is_read_back():
    state, registry, sp = session()
    fake = ScriptedLLM([
        action("apply_leave", confirm=True, leave_type="casual", days=999, start_date="2030-06-25"),
        respond("Sorry, you only have 8 casual days available."),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "apply 999 casual days from 2030-06-25", llm=fake)
    res = confirm_write(state, conv, seen, registry, sp, res, llm=fake)
    assert res.kind == "final"
    assert state.balances["casual"] == 8  # validation blocked the mutation
    assert any("insufficient_balance" in m["content"] for m in conv)


# --- sensitive escalation end-to-end ----------------------------------------
def test_sensitive_message_escalates_without_llm():
    state, registry, sp = session()
    fake = ScriptedLLM([])  # must not be called
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "I think my manager is harassing me", llm=fake)
    assert res.kind == "pending" and res.tool == "raise_ticket" and res.sensitive
    assert fake.calls == 0
    res = confirm_write(state, conv, seen, registry, sp, res, llm=fake)
    assert res.kind == "final" and res.escalated
    assert res.ticket["team"] == "People Ops (Confidential)" and res.ticket["confidential"]
    assert fake.calls == 0  # the sensitive reply is canned, not model-authored


# --- citation integrity -----------------------------------------------------
def test_citations_resolve_only_real_chunk_ids():
    def fake_searcher(query):
        return {"ok": True, "chunks": [
            {"id": "leave-1", "source": "Leave & PTO", "section": "Entitlement",
             "text": "Casual leave is 12 days per year."},
        ]}

    state, registry, sp = session(searcher=fake_searcher)
    fake = ScriptedLLM([
        action("search_policy", query="casual leave days"),
        respond("You get 12 casual days a year.", citation_ids=["leave-1", "fabricated-99"]),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "how many casual days?", llm=fake)
    assert res.kind == "final"
    # The fabricated id is dropped; only the real chunk is cited.
    assert [c["id"] for c in res.citations] == ["leave-1"]
    assert res.citations[0]["source"] == "Leave & PTO"


# --- never-crash fallbacks --------------------------------------------------
def test_malformed_json_twice_escalates():
    state, registry, sp = session()
    fake = ScriptedLLM(["not json", "still not json"])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "tell me a joke about work", llm=fake)
    assert res.kind == "final" and res.escalated
    assert res.ticket["reference"].startswith("TICKET-")
    assert fake.calls == 2  # parsed once, retried once


def test_iteration_cap_escalates():
    # A model that loops forever on a read must be stopped and escalated.
    state, registry, sp = session()
    fake = always(action("check_leave"))
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "loop forever please",
                     llm=fake, max_iters=3)
    assert res.kind == "final" and res.escalated


def test_backend_unavailable_does_not_raise_ticket():
    # A rate limit / backend outage must NOT mint an HR ticket — it asks to retry.
    from agent.llm import LLMError

    def rate_limited(messages):
        raise LLMError("Gemini request failed: 429 RESOURCE_EXHAUSTED")

    state, registry, sp = session()
    n_tickets = len(state.tickets)
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "what is earned leave?", llm=rate_limited)
    assert res.kind == "final"
    assert res.escalated is False
    assert res.ticket is None
    assert len(state.tickets) == n_tickets
    assert "temporarily unavailable" in res.text.lower()


def test_unexpected_exception_still_escalates():
    # A genuinely unexpected (non-LLM) error keeps the fail-safe ticket.
    def boom(messages):
        raise RuntimeError("unexpected bug")

    state, registry, sp = session()
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "anything", llm=boom)
    assert res.kind == "final" and res.escalated and res.ticket
