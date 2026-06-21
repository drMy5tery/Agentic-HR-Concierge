"""HR Concierge — Streamlit UI.

Chat on the left, a live mock-state panel on the right. Because Streamlit reruns
the whole script top-to-bottom, the human-in-the-loop write-gate is modelled as a
session-state machine:

* All state mutations happen inside event handlers (a chat submission, or a
  Confirm/Cancel click), each of which ends by calling ``st.rerun()``.
* The "draw" portion runs every pass and always reflects the latest state, so the
  panel updates the instant an action executes.
* A proposed write is stored as ``pending`` and the input is disabled until the
  employee Confirms or Cancels. On Confirm the tool executes and ``pending`` is
  cleared *before* the rerun, so an action can never be submitted twice.

The agent's reasoning and tool-selection are real; the "Keka" backend is simulated.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import config
from agent import llm as llm_backend
from agent.loop import cancel_write, confirm_write, start_turn
from agent.prompts import build_system_prompt
from agent.tools import build_registry
from mock.keka import LEAVE_TYPES, new_state
from rag.search import build_index, make_searcher

st.set_page_config(page_title="HR Concierge", page_icon="🧑‍💼", layout="wide")


# --- cached, shared resources ----------------------------------------------
@st.cache_resource(show_spinner="Loading policy index (first run downloads the embedding model)…")
def get_index():
    """Build the policy embedding index once per process."""
    return build_index()


# --- session-state initialisation ------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    if "keka" not in ss:
        ss.keka = new_state()
    ss.setdefault("chat", [])          # display messages
    ss.setdefault("pending", None)     # AgentResult awaiting confirmation
    ss.setdefault("work_conv", [])     # in-flight agent conversation (scratch)
    ss.setdefault("seen_chunks", {})   # chunks retrieved this session (for citations)
    ss.setdefault("provider", config.LLM_PROVIDER)
    ss.setdefault("prev_balances", None)


# How many recent USER questions to thread in as context for follow-ups.
_MAX_HISTORY_QUESTIONS = 3

# Referential cues that mark a message as a follow-up needing prior context.
_FOLLOWUP_CUES = (
    "that ", "those", "this ", " it", "them", "same", "earlier", "previous",
    "what about", "how about", "and what", "before that", "after that", "the other",
)


def _is_followup(text: str) -> bool:
    """True if the message likely refers back to an earlier turn.

    Threading prior context degrades this free model's tool-calling on
    self-contained questions, so we only do it when the message actually needs
    it — a referential cue, or a very short utterance.
    """
    t = " " + text.lower().strip() + " "
    if len(t.split()) <= 3:
        return True
    return any(cue in t for cue in _FOLLOWUP_CUES)


def _recent_history() -> list[dict]:
    """Thread recent USER questions (not prior answers) as lightweight context.

    Just the questions — deliberately NOT the prior grounded answers: feeding the
    model its own prose replies biases it towards answering directly instead of
    calling tools (search_policy / get_payslip). The questions alone give enough
    continuity to resolve follow-ups like "what about earned leave?" or "that month".
    """
    prior = [m["text"] for m in st.session_state.chat
             if m["role"] == "user" and m.get("text")][-_MAX_HISTORY_QUESTIONS:]
    if not prior:
        return []
    joined = " | ".join(prior)
    return [{"role": "user",
             "content": f"(Context only — earlier questions in this chat: {joined}. "
                        f"Now answer my new question below, choosing the right tool for it.)"}]


def _llm_fn():
    """Bind call_llm to the provider currently selected in the sidebar."""
    provider = st.session_state.provider
    # Resolve call_llm at call time (via the module) so it stays patchable in tests.
    return lambda messages: llm_backend.call_llm(messages, provider=provider)


def _session_objects():
    """Rebuild the (registry, system_prompt) bound to the live mock state."""
    keka = st.session_state.keka
    searcher = make_searcher(get_index())
    registry = build_registry(keka, searcher=searcher)
    system_prompt = build_system_prompt(registry, keka.employee, date.today().isoformat())
    return registry, system_prompt


def _append_assistant(result) -> None:
    st.session_state.chat.append({
        "role": "assistant",
        "text": result.text,
        "citations": result.citations,
        "escalated": result.escalated,
        "ticket": result.ticket,
    })


# --- event handlers (mutate, then caller reruns) ----------------------------
def _handle_user_message(text: str) -> None:
    # Only thread prior context for genuine follow-ups; keep self-contained
    # questions clean so the model reliably calls the right tool.
    conv = _recent_history() if _is_followup(text) else []
    st.session_state.chat.append({"role": "user", "text": text})
    registry, system_prompt = _session_objects()
    seen = st.session_state.seen_chunks  # persists across turns so citations resolve
    result = start_turn(st.session_state.keka, conv, seen, registry, system_prompt,
                        text, llm=_llm_fn())
    st.session_state.work_conv = conv
    if result.kind == "pending":
        st.session_state.pending = result
    else:
        _append_assistant(result)
        st.session_state.pending = None


def _handle_confirm() -> None:
    pending = st.session_state.pending
    registry, system_prompt = _session_objects()
    result = confirm_write(st.session_state.keka, st.session_state.work_conv,
                           st.session_state.seen_chunks, registry, system_prompt,
                           pending, llm=_llm_fn())
    # Clear the pending action BEFORE the rerun to avoid a double-submit.
    st.session_state.pending = result if result.kind == "pending" else None
    if result.kind == "final":
        _append_assistant(result)


def _handle_cancel() -> None:
    pending = st.session_state.pending
    registry, system_prompt = _session_objects()
    result = cancel_write(st.session_state.keka, st.session_state.work_conv,
                          st.session_state.seen_chunks, registry, system_prompt,
                          pending, llm=_llm_fn())
    st.session_state.pending = result if result.kind == "pending" else None
    if result.kind == "final":
        _append_assistant(result)


# --- sidebar: live mock-state panel ----------------------------------------
def _render_sidebar() -> None:
    keka = st.session_state.keka
    emp = keka.employee
    with st.sidebar:
        st.subheader("Live HR state")
        st.caption("Simulated 'Keka' backend — updates as actions execute.")

        st.markdown(f"**{emp['name']}** · {emp['employee_id']}")
        st.caption(f"{emp['designation']}, {emp['department']} · manager {emp['manager']}")

        st.markdown("**Leave balances**")
        prev = st.session_state.prev_balances
        cols = st.columns(len(LEAVE_TYPES))
        for col, lt in zip(cols, LEAVE_TYPES):
            delta = None
            if prev is not None:
                diff = keka.balances[lt] - prev[lt]
                delta = diff if diff != 0 else None
            col.metric(lt.capitalize(), keka.balances[lt], delta=delta)
        # Remember balances so the next render can show the change as a delta.
        st.session_state.prev_balances = dict(keka.balances)

        st.markdown("**Tickets**")
        if keka.tickets:
            st.dataframe(
                pd.DataFrame(keka.tickets)[
                    ["reference", "category", "team", "confidential", "status"]
                ],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No tickets yet.")

        st.markdown("**Recent leave records**")
        if keka.leave_records:
            st.dataframe(
                pd.DataFrame(keka.recent_leave_records())[
                    ["reference", "leave_type", "days", "start_date", "status"]
                ],
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No leave records yet.")

        st.divider()
        st.selectbox("LLM provider", options=["groq", "gemini"], key="provider",
                     help="Switch the backend live — the only change needed is this flag.")
        model = config.GROQ_MODEL if st.session_state.provider == "groq" else config.GEMINI_MODEL
        st.caption(f"Model: `{model}`")
        if st.button("Reset demo", width="stretch"):
            st.session_state.keka = new_state()
            st.session_state.chat = []
            st.session_state.pending = None
            st.session_state.prev_balances = None
            st.session_state.seen_chunks = {}
            st.session_state.work_conv = []
            st.rerun()


# --- main: chat + confirm gate ----------------------------------------------
def _render_chat() -> None:
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("citations"):
                sources = "; ".join(f"{c['source']} — {c['section']}" for c in msg["citations"])
                st.caption(f"📄 Sources: {sources}")
            elif msg.get("escalated") and msg.get("ticket"):
                ref = msg["ticket"].get("reference", "")
                team = msg["ticket"].get("team", "")
                st.caption(f"🎫 Escalated: {ref} → {team}")


def _render_pending_gate() -> None:
    pending = st.session_state.pending
    if not pending:
        return
    with st.chat_message("assistant"):
        icon = "🔒" if pending.sensitive else "⚠️"
        st.markdown(f"{icon} **Confirmation needed** — {pending.confirm_prompt}")
        confirm_col, cancel_col = st.columns(2)
        if confirm_col.button("Confirm", type="primary", width="stretch"):
            _handle_confirm()
            st.rerun()
        if cancel_col.button("Cancel", width="stretch"):
            _handle_cancel()
            st.rerun()


def main() -> None:
    _init_state()
    _render_sidebar()

    st.title("HR Concierge")
    st.caption("Grounded HR Q&A with citations, and HR actions with a confirm step. "
               "Informational support only — not legal or HR advice. The downstream "
               "HR system is simulated.")

    # Friendly note if no key is configured for the selected provider.
    key = config.GROQ_API_KEY if st.session_state.provider == "groq" else config.GEMINI_API_KEY
    if not key:
        st.info(
            f"No API key set for **{st.session_state.provider}**. Add one to `.env` "
            "for grounded answers and actions. (Sensitive-topic escalation works "
            "without a key; other requests will safely escalate to a ticket.)",
            icon="🔑",
        )

    _render_chat()
    _render_pending_gate()

    user_input = st.chat_input(
        "Ask about HR policies, or request an action…",
        disabled=st.session_state.pending is not None,
    )
    if user_input:
        _handle_user_message(user_input)
        st.rerun()


if __name__ == "__main__":
    main()
