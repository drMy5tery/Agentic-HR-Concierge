"""End-to-end UI tests for the Streamlit app via the official ``AppTest`` harness.

A rule-based fake LLM is patched onto ``agent.llm.call_llm`` so the four demo
moments can be driven deterministically — through the real chat input, confirm
gate, button clicks, and live panel — with no browser and no API key. We assert
on ``session_state`` (the source of truth) and on the mock backend.
"""
import json
import pathlib
import re

import pytest

APP = str(pathlib.Path(__file__).resolve().parents[1] / "app.py")


def _tool_results(messages):
    """Extract parsed TOOL_RESULT payloads from the conversation messages."""
    out = []
    for m in messages:
        content = m.get("content", "")
        if m.get("role") == "user" and content.startswith("TOOL_RESULT "):
            try:
                out.append(json.loads(content[len("TOOL_RESULT "):]))
            except json.JSONDecodeError:
                pass
    return out


def _rule_based_llm(messages, **kwargs):
    """A deterministic stand-in 'model' that picks actions from the question.

    Accepts (and ignores) the same keyword args as the real ``call_llm`` (e.g.
    ``provider``), so it is a drop-in replacement when patched in.
    """
    non_system = [m for m in messages if m.get("role") != "system"]
    question = non_system[0]["content"] if non_system else ""
    ql = question.lower()
    results = _tool_results(messages)

    def emit(name, **args):
        return json.dumps({"action": name, "args": args, "confirm": name in ("apply_leave", "raise_ticket")})

    def reply(text, citation_ids=None):
        return json.dumps({"action": "respond",
                           "args": {"text": text, "citation_ids": citation_ids or []},
                           "confirm": False})

    # A cancelled action was fed back: acknowledge instead of re-proposing it.
    if any('"cancelled": true' in m.get("content", "") for m in messages):
        return reply("No problem — I won't proceed with that action.")

    if "apply" in ql and "leave" in ql:
        if any("balance_after" in r for r in results):
            r = results[-1]
            return reply(f"Done — applied {r['days']} day(s) of {r['leave_type']} leave, "
                        f"reference {r['reference']}; balance now {r['balance_after']}.")
        days = int(re.search(r"(\d+)", ql).group(1)) if re.search(r"(\d+)", ql) else 1
        lt = ("casual" if "casual" in ql else "sick" if "sick" in ql
              else "earned" if ("earned" in ql or "privilege" in ql) else "casual")
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", question)
        start = date_match.group(0) if date_match else "2030-06-25"
        return emit("apply_leave", leave_type=lt, days=days, start_date=start)

    if "payslip" in ql:
        if results:
            s = results[-1]
            return reply(f"Your latest payslip ({s.get('month')}): net {s.get('net')} "
                        f"{s.get('currency')}.")
        return emit("get_payslip")

    if "balance" in ql:
        if results:
            return reply(f"Your balances: {results[-1].get('balances')}.")
        return emit("check_leave")

    # default: treat as a policy question -> search, then ground or escalate
    if results:
        chunks = results[-1].get("chunks", [])
        if chunks:
            return reply(f"Based on the policy: {chunks[0]['text'][:60]}…",
                        citation_ids=[chunks[0]["id"]])
        return emit("raise_ticket", category="policy",
                    summary="Question not covered by the policy documents.")
    return emit("search_policy", query=question)


def _click(at, label):
    for button in at.button:
        if button.label == label:
            button.click().run()
            return
    raise AssertionError(f"button {label!r} not found")


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setattr("agent.llm.call_llm", _rule_based_llm)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, f"app raised on startup: {at.exception}"
    return at


def test_app_starts_clean(app):
    assert app.session_state.pending is None
    assert app.session_state.keka.balances == {"casual": 8, "sick": 10, "earned": 15}


def test_moment_1_grounded_qa_with_citation(app):
    app.chat_input[0].set_value("How many casual leave days do I get?").run()
    assert not app.exception
    assert app.session_state.pending is None  # a Q&A needs no confirmation
    last = app.session_state.chat[-1]
    assert last["role"] == "assistant"
    assert last["citations"], "a grounded answer must carry a citation"
    assert last["citations"][0]["source"] == "Leave & Paid Time Off Policy"


def test_moment_4_read_payslip(app):
    app.chat_input[0].set_value("Show my latest payslip").run()
    assert not app.exception
    assert app.session_state.pending is None
    assert "payslip" in app.session_state.chat[-1]["text"].lower()


def test_moment_3_sensitive_escalates_with_confirm(app):
    app.chat_input[0].set_value("I think my manager is harassing me").run()
    pending = app.session_state.pending
    assert pending is not None and pending.tool == "raise_ticket" and pending.sensitive
    n_tickets_before = len(app.session_state.keka.tickets)

    _click(app, "Confirm")
    assert app.session_state.pending is None  # cleared before rerun
    tickets = app.session_state.keka.tickets
    assert len(tickets) == n_tickets_before + 1
    new = tickets[-1]
    assert new["team"] == "People Ops (Confidential)" and new["confidential"] is True
    assert app.session_state.chat[-1]["escalated"]


def test_moment_2_apply_leave_gated_then_executes(app):
    app.chat_input[0].set_value("Apply 2 days casual leave from 2026-06-25").run()
    # Gated: proposed but NOT executed.
    assert app.session_state.pending is not None
    assert app.session_state.pending.tool == "apply_leave"
    assert app.session_state.keka.balances["casual"] == 8

    _click(app, "Confirm")
    # Executed only after confirmation.
    assert app.session_state.pending is None
    assert app.session_state.keka.balances["casual"] == 6
    assert any(r["reference"].startswith("LEAVE-")
               for r in app.session_state.keka.leave_records)
    assert "reference" in app.session_state.chat[-1]["text"].lower()


def test_cancel_does_not_execute(app):
    app.chat_input[0].set_value("Apply 3 days sick leave from 2026-06-25").run()
    assert app.session_state.pending is not None
    _click(app, "Cancel")
    assert app.session_state.pending is None
    assert app.session_state.keka.balances["sick"] == 10  # unchanged


def test_conversation_memory_threads_prior_turns(monkeypatch):
    # A follow-up turn must carry the previous turn's question and answer as context.
    seen = []

    def recorder(messages, **kwargs):
        seen.append(messages)
        return json.dumps({"action": "respond",
                           "args": {"text": "Noted.", "citation_ids": []}})

    monkeypatch.setattr("agent.llm.call_llm", recorder)
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.chat_input[0].set_value("How many casual leave days do I get?").run()
    at.chat_input[0].set_value("what about earned leave?").run()
    assert not at.exception

    joined = " ".join(m["content"] for m in seen[-1])  # 2nd turn's messages
    assert "How many casual leave days" in joined   # prior question threaded as context
    assert "what about earned leave" in joined       # current question
