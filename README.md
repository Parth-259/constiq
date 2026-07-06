---
title: ConstructIQ
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.11"
app_port: 8000
pinned: false
---

# ConstructIQ

**ConstructIQ takes a project's tender, site, and communication documents and runs the
full procurement decision loop — find the vendor, check it complies, score the risk,
negotiate the price, generate the order, and track it — with every step explained in
plain language, not asserted by a black box.**

Built for the Kaya AI IIT India Hackathon 2026 (Procurement track).

## Architecture

```
┌─────────────────────────────────────────────┐
│        Web Frontend (Tailwind + vanilla JS) │
│  chat / risk / vendors / negotiation / POs  │
└──────────────────┬──────────────────────────┘
                   │ HTTP (same origin, /api/*)
┌──────────────────▼──────────────────────────┐
│                FastAPI Backend              │
│ /ingest /ask /agent/ask /vendors /risk      │
│ /discovery /recommend /negotiation /po      │
└──┬───────────┬──────────────┬───────────────┘
   │           │              │
┌──▼────────┐ ┌▼─────────┐ ┌──▼──────────────────────────┐
│ Multi-    │ │ ChromaDB │ │  Claude agent loop (8 tools) │
│ source    │ │ (local,  │ │  doc_search · vendor_discovery│
│ pipeline  │ │ vectors) │ │  vendor_evaluation · compliance│
│ pdf/docx/ │ └──────────┘ │  risk · recommend · negotiate │
│ txt →     │              │  generate_po                  │
│ chunk →   │              └──┬───────────────────────────┘
│ embed     │                 │
└───────────┘   ┌─────────────▼─────────────────┐
                │            SQLite             │
                │ vendors · quotes · requirements│
                │ (versioned) · negotiations ·  │
                │ rounds · purchase_orders ·    │
                │ tracking_events               │
                └───────────────────────────────┘
```

## Tech stack — and why

| Layer | Choice | Why |
|---|---|---|
| LLM | Anthropic Claude **or** Google Gemini (auto-selected by API key) — smart tier for reasoning/extraction, fast tier for narration | Provider-agnostic `backend/llm.py`; reliable SDKs, strong tool use |
| Embeddings | Local `sentence-transformers` (all-MiniLM-L6-v2) | Free, no rate limits, works offline during a live demo |
| Vector store | ChromaDB (embedded, file-based) | Zero infrastructure, metadata filtering built in |
| Structured data | SQLite via SQLAlchemy | Honest MVP scope; production path is PostgreSQL |
| PDF parsing | PyMuPDF (text) + pdfplumber (BOQ tables) | Speed for body text, table fidelity where it matters |
| Risk scoring | Explainable weighted formula | No historical delay data exists to train an ML model on — a transparent formula a judge can verify beats a fabricated black box |
| Negotiation | Deterministic concession formula; LLM only narrates | Every price is computed from seeded vendor attributes — the LLM never invents a number |
| Live web search | Tavily (optional) | Web-discovered vendors are labeled unverified; the pipeline runs fully on internal data if the call fails |
| PO generation | ReportLab | Single structured PDF document |

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY or GEMINI_API_KEY (and optional TAVILY_API_KEY)
.venv/bin/python -m backend.db.seed_vendors   # optional — startup also seeds
.venv/bin/python -m uvicorn backend.main:app --port 8000
```

Open **http://localhost:8000** — the frontend is served by the backend.
API docs (Swagger): http://localhost:8000/docs

Without any LLM API key, document search, vendor discovery, evaluation,
compliance, risk, the full negotiation loop, and PO generation all still work
(they are deterministic); only free-text answers and in-character narration
degrade to templates.

## LLM providers

All LLM calls go through `backend/llm.py`, which works with **Anthropic
Claude** or **Google Gemini** and auto-detects the provider from the
environment:

1. An explicit `LLM_PROVIDER=anthropic|gemini` override wins (when its key is
   set).
2. Otherwise `ANTHROPIC_API_KEY` is used if present, else `GEMINI_API_KEY`.
3. With neither key, every LLM feature degrades gracefully as described above.

Gemini has a generous free tier — get a key at
https://aistudio.google.com/apikey (models default to `gemini-2.5-flash` /
`gemini-2.5-flash-lite`, overridable via `MODEL_GEMINI_SMART` /
`MODEL_GEMINI_FAST`).

Regardless of provider, every price, risk score, and compliance verdict is
computed deterministically in Python — the LLM only extracts, narrates, and
orchestrates tools; it never invents a number.

### Docker

```bash
docker compose up --build
```

Then open http://localhost:8000.

## Demo data disclosure

The vendor database, quotes, and the `PRJ-2024-001` requirements are **curated
synthetic data** modeling real Indian construction supply norms (IS 1786 TMT
grades, OPC/PPC cement grades, RMC mixes). They are clearly labeled demo data —
a production version would integrate live vendor APIs.

## Evaluation

```bash
# with the backend running on :8000
.venv/bin/python -m backend.eval.run_eval
```

Runs the hand-written benchmark in `backend/eval/benchmark.json` against
`/api/ask` and reports per-question pass/fail plus summary accuracy to
`backend/eval/results.json`.

## What's next (production roadmap)

PostgreSQL migration · real vendor API integrations · hybrid keyword+vector
search · WhatsApp/email export ingestion · PO delivery by e-signature ·
delay alerts wired to the risk score · React frontend build pipeline.
