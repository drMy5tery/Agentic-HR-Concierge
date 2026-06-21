# HR Concierge — an agentic HR helpdesk assistant

A take-home demo built for **Connect and Heal**. It is a grounded, tool-using HR
helpdesk assistant: it answers employee questions **only** from a small set of
HR policy documents (with citations), and it can take actions on the employee's
behalf — checking a leave balance, applying for leave, raising a ticket, or
fetching a payslip — through tools, with a **human-in-the-loop confirmation
before any state-changing action**.

> **The agent's reasoning and tool-selection are real. The downstream HR system
> ("Keka") is simulated** — an in-memory mock, not a real integration. This is
> informational support, not legal or HR advice.

The app runs entirely on **free LLM tiers** (Groq or Google Gemini) with all
integrations mocked, and uses **local embeddings** for retrieval (no vector
database, no LangChain/LlamaIndex).

---

## What it does

- **Grounded Q&A.** Answers HR questions only from the policy docs and cites the
  source document and section. It never uses outside knowledge for policy
  answers; if the docs don't cover a question, it escalates instead of guessing.
- **Actions via tools.** Check leave balance, apply for leave, raise a ticket,
  and fetch a payslip — all against the mock "Keka" backend.
- **Human-in-the-loop.** Any state-changing (write) action — applying leave,
  raising a ticket — must be confirmed by the employee before it executes.
- **Escalation.** Sensitive topics (harassment, discrimination, medical, pay
  disputes, anything legal) and questions the docs don't cover are **not**
  answered. The agent raises a routed ticket instead.
- **Live side panel.** Leave balances, tickets, and recent leave records are
  shown alongside the chat and update the moment an action executes.

---

## Architecture (in words)

**Model-agnostic JSON router, not native function calling.** A single
`call_llm()` adapter wraps two interchangeable backends — Groq
(`openai/gpt-oss-120b`, primary) and Google Gemini (`gemini-2.5-flash`, backup).
Both are asked to return **one JSON action per turn** in JSON mode. The provider
and model come from `.env`, so switching backends is a one-line change.

**The agent loop.** Each turn the model returns a single JSON object:

```json
{ "action": "<tool name or 'respond'>", "args": { }, "confirm": false }
```

- `respond` ends the turn with `args: {"text": "...", "citation_ids": ["..."]}`.
- **Read** tools (`search_policy`, `check_leave`, `get_payslip`) execute
  immediately; their result is appended and the loop continues.
- **Write** tools (`apply_leave`, `raise_ticket`) are **never executed without
  explicit human approval**.
- The loop is capped at ~6 iterations; if exceeded it raises a ticket rather
  than spinning. The model runs at low temperature; a malformed JSON reply is
  retried once, and if it still fails the turn degrades to raising a ticket — so
  the app never crashes on a bad model response.

**The write-gate.** The gate is enforced **in code, from the tool registry's
`kind` (`read`/`write`) — never from the model's `confirm` field**, which is
only a phrasing hint. When the loop proposes a write, it is stored as a
`pending_action` and the loop stops. Streamlit then renders a plain-English
confirm card ("About to apply 2 days casual leave from 25 Jun — confirm?") with
**Confirm / Cancel** buttons. On confirm, the tool executes, the result is
appended, the pending action is cleared *before* the rerun (to avoid
double-submit), and the loop re-enters to produce the final message. On cancel,
a "cancelled" result is appended and the agent acknowledges.

**Retrieval as a read tool.** `search_policy(query)` embeds the query with a
local `sentence-transformers` model (`all-MiniLM-L6-v2`), scores every policy
chunk by cosine similarity in NumPy, and returns the top-k chunks above a
relevance threshold as `[{id, source, section, text}]`. The agent answers
**only** from the returned chunks and cites their ids. If nothing clears the
threshold, the question is treated as uncovered and escalated. Citations are
rendered from chunk metadata, never from free text the model writes.

**The mock "Keka" layer.** An in-memory store (held in Streamlit
`session_state`) with the employee record, balances (`casual`, `sick`,
`earned`), tickets, leave records, and payslips. **The mock generates all
reference IDs** (`LEAVE-000n`, `TICKET-000n`); the agent only ever relays an id
the mock returned and never fabricates one. `apply_leave` validates first (valid
type, sufficient balance, sane future date) and returns a structured error the
agent reads back — it never overdraws or mutates on invalid input.
`raise_ticket` resolves the category to a team **in code**
(harassment/discrimination → People Ops (Confidential), payroll → Finance,
IT → IT, uncovered policy → HR generalist).

---

## Safety design

- **Writes are gated in code.** The dispatcher decides read-vs-write from the
  registry, so the model cannot self-authorise a state change.
- **ID integrity.** Only the mock layer mints reference IDs; the agent relays
  them. It cannot invent a `LEAVE-` or `TICKET-` number.
- **Citation integrity.** Citations come from retrieved chunk metadata, not from
  model free text, so a cited source always exists.
- **Escalation over guessing.** Sensitive queries and uncovered questions raise
  a routed ticket instead of receiving a policy-style answer. A lightweight
  keyword pre-check plus the system prompt drives this.
- **Fail safe.** Bad model output, a tool error, or hitting the iteration cap all
  degrade to raising a ticket — never a crash.

---

## Demo moments (exact prompts to type)

1. **Grounded Q&A** — type: `How many casual leave days do I get?`
   → a cited answer drawn from the Leave & PTO policy.
2. **Action with confirm** — type: `Apply 2 days casual leave from 25 Jun 2026`
   → a confirm card; on **Confirm**, the casual balance drops and a
   `LEAVE-000n` reference is shown (and appears in the side panel).
3. **Sensitive → escalate** — type: `I think my manager is harassing me`
   → no policy answer; a **confidential** ticket routed to **People Ops** is
   raised and shown in the panel.
4. **Read action** — type: `Show my latest payslip`
   → the latest payslip is fetched from the mock and summarised.

---

## Setup & run

Requires Python 3.10+ and a free API key from **Groq** (primary) and/or
**Google AI Studio** (Gemini, backup).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure keys
cp .env.example .env          # then edit .env
#   set GROQ_API_KEY=...      (get one free at https://console.groq.com)
#   and/or GEMINI_API_KEY=... (get one free at https://aistudio.google.com)
#   set LLM_PROVIDER=groq or gemini

# 4. Run the app
streamlit run app.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) once.

A small CLI harness (`python -m agent.cli`) is also included to exercise the
agent loop and the write-gate without the UI.

---

## Tech choices and why

- **Model-agnostic JSON router (not native function calling).** Keeps the agent
  portable across providers and free tiers: the same prompt and parser work for
  Groq and Gemini, and switching is one line of config. JSON mode + an in-code
  dispatcher also let us enforce the write-gate ourselves rather than trusting a
  provider's tool-calling semantics.
- **Local embeddings, no vector DB.** With only a handful of short policy docs,
  a brute-force cosine search over NumPy arrays is exact, instant, and has zero
  infrastructure. `sentence-transformers` runs locally and free.
- **No LangChain / LlamaIndex.** The retrieval and the agent loop are
  hand-rolled so the control flow — especially the safety-critical write-gate
  and the escalation logic — is explicit and easy to audit.
- **Mocked downstream.** The brief is to demonstrate agentic reasoning safely;
  a real HR integration is out of scope, so "Keka" is an in-memory simulation.

---

## Limitations

- The HR backend is simulated and in-memory; state resets when the app restarts.
- Answers are only as good and current as the sample policy docs in
  `data/policies/`; they are illustrative, not Connect and Heal's real policies.
- Retrieval quality depends on the threshold and the small document set; the
  threshold is tunable in `.env`.
- This is a demo for informational support, not a system of record and not legal
  or HR advice.

---

## Repository layout

```
Agentic-HR-Concierge/
  app.py                  # Streamlit UI: chat, confirm gate, side panel
  config.py               # provider/model/threshold/top_k from env, with defaults
  agent/
    llm.py                # call_llm(): Groq + Gemini behind one adapter
    tools.py              # tool registry (read/write) + dispatch + write-gate
    loop.py               # JSON-router agent loop
    prompts.py            # system prompt(s)
    cli.py                # tiny CLI harness for the loop (no UI)
  mock/
    keka.py               # in-memory state, 4 functions, IDs, validation, routing
  rag/
    ingest.py             # load + chunk policy docs, build embeddings
    search.py             # search_policy: cosine top-k + threshold
  data/policies/          # sample HR policy markdown (Leave, Benefits, Conduct, …)
  tests/test_keka.py      # unit tests for the mock layer
```
