# Banking Customer Support AI Agent — Multi-Agent Architecture

A multi-agent GenAI system for banking customer support built with **Python** and the
**LangChain** chat-model abstraction. It classifies incoming customer messages,
routes them to the right specialist agent, and manages support tickets in SQLite.

## Architecture

```
User message
    │
    ▼
┌─────────────────┐
│ Classifier Agent │  → POSITIVE_FEEDBACK / NEGATIVE_FEEDBACK / QUERY
└─────────────────┘
    │
    ├── POSITIVE_FEEDBACK ──► Feedback Agent (thank-you branch)
    ├── NEGATIVE_FEEDBACK ──► Feedback Agent (ticket-creation branch) ──► SQLite
    └── QUERY             ──► Query Agent (ticket lookup) ────────────► SQLite
```

The **Orchestrator** (`orchestrator.py`) is the supervisor: it calls the
Classifier, then hands off to exactly one downstream agent — the same
hand-off pattern used in LangGraph supervisor architectures, applied here
without a graph runtime since the workflow is linear.

## Project layout

```
banking_ai_agent/
├── agents/
│   ├── classifier_agent.py   # message → (label, used_fallback)
│   ├── feedback_agent.py     # thank-you / ticket-creation
│   └── query_agent.py        # ticket-number extraction + lookup
├── database/
│   └── db.py                 # SQLite: support_tickets, interaction_logs, feedback_log
├── ui/
│   └── app.py                # Streamlit dashboard (5 tabs, see below)
├── tests/
│   ├── test_agents.py        # pytest suite (runs offline)
│   └── eval_dataset.json     # labeled test cases for evaluation.py
├── llm_provider.py           # picks Anthropic / OpenAI / offline demo model
├── orchestrator.py           # supervisor / router + prompt-trace logging
├── evaluation.py             # QA-based scoring, classifier accuracy, routing success rate
├── demo_cli.py                # terminal walkthrough of the 3 sample flows
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Choosing a model

The system works with a real LLM or in a keyless offline demo mode:

| Mode | How to enable |
|---|---|
| Anthropic (Claude) | `export ANTHROPIC_API_KEY=sk-...` |
| OpenAI | `export OPENAI_API_KEY=sk-...` |
| Offline demo (no key needed) | `export LLM_PROVIDER=offline` |

If no key and no `LLM_PROVIDER` are set, it automatically falls back to the
offline rule-based model so the project always runs.

## Running it

**Streamlit dashboard** (all 5 tabs described below):
```bash
streamlit run ui/app.py
```

**Terminal demo (reproduces the 3 sample flows from the spec):**
```bash
python demo_cli.py
```

**Evaluation suite (classifier accuracy + QA response scoring + routing success rate):**
```bash
python evaluation.py
```
Writes a full JSON report to `eval_report.json`.

**Tests:**
```bash
python -m pytest tests/ -v
```

## What each agent does

1. **Classifier Agent** — one constrained LLM call that labels the message
   `POSITIVE_FEEDBACK`, `NEGATIVE_FEEDBACK`, or `QUERY`. Returns `(label,
   used_fallback)` — `used_fallback=True` means the model's raw output didn't
   match a valid label and the classifier had to default, which is logged as
   a routing failure.
2. **Feedback Agent** —
   - *Positive:* generates a short personalized thank-you.
   - *Negative:* creates a new 6-digit ticket in `support_tickets`, then
     generates an empathetic acknowledgement including the ticket number.
3. **Query Agent** — extracts a 6-digit ticket number with a regex and
   reports its current status from the database.

## Streamlit dashboard tabs (Part 2 requirements, mapped directly)

| Tab | Brief requirement it satisfies |
|---|---|
| 💬 Live Agent | Accept user input, simulate agent routing, display classification/response/DB interaction. Each response also has a 👍/👎 button — the "agent improvement loop." |
| 🎫 Tickets | View and manually update ticket status in `support_tickets`. |
| 🧭 Test Scenarios | "Test scenarios for each agent role" — run the Classifier, Feedback Handler, or Query Handler individually, bypassing the full pipeline, to isolate a bug to one agent. |
| 🧪 Evaluation | "Model Evaluation" — runs `evaluation.py`'s QA-based scoring and classifier test-case coverage, shows routing success rate. |
| 🪵 Logs & Debugging | Prompt traces, classification output, ticket actions, agent success/failure rate, and a flagged-for-review queue populated by 👎 feedback. |

## Evaluation approach (`evaluation.py`)

- **Classifier evaluation**: runs `tests/eval_dataset.json` (9 labeled cases
  across all three classes) through `ClassifierAgent` and reports overall and
  per-label accuracy — this is the "test case coverage" requirement.
- **Response quality (QA rubric)**: runs the same dataset through the full
  `Orchestrator` and scores each response against explicit, explainable
  checks (e.g., does a thank-you include the customer's name; does an
  apology include the ticket number and read as empathetic; does a status
  reply clearly report found/not-found). Each check is a yes/no rubric item
  — that's what "QA-based scoring" means here, rather than a hidden ML score.
- **Routing success rate**: read directly from `interaction_logs` via
  `database.db.routing_success_rate()`, reflecting real traffic (any request,
  not just the curated test set) — a message only counts as a routing
  failure if the classifier had to fall back to a default label.

## Notes on the offline demo model

`llm_provider.OfflineRuleBasedChatModel` is a keyword-based stand-in for a
real chat model, implementing the same `invoke()` interface LangChain chat
models expose. It exists purely so graders/reviewers can run the whole
pipeline — including the evaluation suite — without an API key. Swap in
`ANTHROPIC_API_KEY` for production-quality classification and generation —
no other code changes needed.
