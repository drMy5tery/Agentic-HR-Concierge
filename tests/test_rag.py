"""Tests for policy ingestion and retrieval.

Chunk-loading tests need no model. The retrieval tests build a real
``all-MiniLM-L6-v2`` index once (module-scoped) and assert that relevant queries
retrieve the right document/section above threshold while off-topic queries
return nothing — and that the loop produces a grounded citation end-to-end.
"""
import pytest

from rag.ingest import load_chunks

EXPECTED_SOURCES = {
    "Leave & Paid Time Off Policy",
    "Benefits & Insurance Policy",
    "Code of Conduct",
    "Expense & Reimbursement Policy",
    "Remote Work Policy",
}


# --- chunking (no model) ----------------------------------------------------
def test_load_chunks_structure_and_sources():
    chunks = load_chunks()
    assert len(chunks) >= 20
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))  # ids are unique
    for chunk in chunks:
        assert set(chunk) == {"id", "source", "section", "text"}
        assert chunk["text"].strip()
    assert {c["source"] for c in chunks} == EXPECTED_SOURCES


def test_chunk_for_casual_leave_entitlement_exists():
    chunks = load_chunks()
    match = [c for c in chunks
             if c["source"] == "Leave & Paid Time Off Policy"
             and "Entitlement" in c["section"]]
    assert match and "casual" in match[0]["text"].lower()


# --- retrieval (real embeddings) --------------------------------------------
@pytest.fixture(scope="module")
def index():
    from rag.search import build_index
    return build_index()


def test_relevant_query_retrieves_correct_section(index):
    from config import RETRIEVAL_THRESHOLD
    hits = index.search("How many casual leave days do I get?", top_k=4,
                        threshold=RETRIEVAL_THRESHOLD)
    assert hits, "a relevant query should return at least one chunk"
    top = hits[0]
    assert top["source"] == "Leave & Paid Time Off Policy"
    assert top["section"] == "Leave Types and Annual Entitlement"
    assert top["score"] >= 0.5


def test_work_from_home_query_retrieves_remote_policy(index):
    hits = index.search("Can I work from home and how many days?", top_k=4, threshold=0.30)
    assert hits[0]["source"] == "Remote Work Policy"


def test_off_topic_query_returns_nothing(index):
    from rag.search import make_searcher
    searcher = make_searcher(index)  # uses configured top_k / threshold
    result = searcher("What is the airspeed velocity of an unladen swallow?")
    assert result["ok"] is True
    assert result["chunks"] == []  # below threshold -> uncovered -> escalate


# --- grounded answer end-to-end through the loop ----------------------------
def test_loop_produces_grounded_citation(index):
    from datetime import date

    from agent.fake_llm import ScriptedLLM, action, respond
    from agent.loop import start_turn
    from agent.prompts import build_system_prompt
    from agent.tools import build_registry
    from rag.search import make_searcher
    from mock.keka import new_state

    state = new_state()
    registry = build_registry(state, searcher=make_searcher(index))
    sp = build_system_prompt(registry, state.employee, date.today().isoformat())
    fake = ScriptedLLM([
        action("search_policy", query="casual leave annual entitlement"),
        respond("You receive 12 days of casual leave per year.",
                citation_ids=["leave-and-pto-1"]),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp, "How many casual leave days do I get?",
                     llm=fake)
    assert res.kind == "final"
    assert [c["id"] for c in res.citations] == ["leave-and-pto-1"]
    assert res.citations[0]["source"] == "Leave & Paid Time Off Policy"


def test_loop_escalates_uncovered_question(index):
    from datetime import date

    from agent.fake_llm import ScriptedLLM, action
    from agent.loop import start_turn
    from agent.prompts import build_system_prompt
    from agent.tools import build_registry
    from rag.search import make_searcher
    from mock.keka import new_state

    state = new_state()
    registry = build_registry(state, searcher=make_searcher(index))
    sp = build_system_prompt(registry, state.employee, date.today().isoformat())
    # Search returns nothing relevant -> the agent raises a policy ticket.
    fake = ScriptedLLM([
        action("search_policy", query="company stance on time travel"),
        action("raise_ticket", confirm=True, category="policy",
               summary="Employee asked something the policies don't cover."),
    ])
    conv, seen = [], {}
    res = start_turn(state, conv, seen, registry, sp,
                     "What is the company policy on time travel?", llm=fake)
    assert res.kind == "pending" and res.tool == "raise_ticket"
