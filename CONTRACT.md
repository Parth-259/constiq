# ConstructIQ — Interface Contract (source of truth for all modules)

Every module MUST conform to these exact signatures and JSON shapes. Files
`backend/config.py`, `backend/db/models.py`, `backend/db/session.py` already
exist — READ them before writing code; import from them, never redefine them.

Project root: `/Users/parth/constructiq`. Python package root is `backend`
(imports like `from backend import config`, `from backend.db.models import Vendor`).
Plain JavaScript + Tailwind-CDN HTML frontend lives in `frontend/`.

Conventions:
- Python 3.11, type hints everywhere, `logging.getLogger(__name__)` (no prints).
- All LLM calls via `anthropic` SDK, model names from `config.MODEL_SONNET` /
  `config.MODEL_HAIKU`. Every LLM-dependent function MUST degrade gracefully
  when `config.ANTHROPIC_API_KEY` is empty: deterministic logic still runs,
  narration falls back to a formatted template string.
- Money is INR, floats. Percentages are 0-100 floats unless stated.

---

## backend/pipeline/pdf_loader.py
```python
class PDFExtractionError(Exception): ...
def extract_pdf(file_path: str) -> list[dict]
# returns one dict per page:
# {"page_number": int (1-based), "text": str, "tables": list[list[list[str]]],
#  "source_file": str (basename)}
```
PyMuPDF (`fitz`) for text; `pdfplumber` (separate open) for tables. Empty page
=> warning log, keep entry. FileNotFoundError / parse errors => raise
`PDFExtractionError` (chained).

## backend/pipeline/chunking.py
```python
def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 100) -> list[dict]
# each chunk: {"chunk_id": str(uuid4), "text": str, "page_number": int,
#              "source_file": str, "chunk_type": "text"|"table"}
```
Recursive split: paragraphs -> sentences -> hard limit; never mid-word.
Each table serialized as one markdown-table chunk, never split.

## backend/pipeline/embedding.py
```python
def get_model()  # module-level cached SentenceTransformer(config.EMBEDDING_MODEL)
def get_collection()  # chromadb.PersistentClient(path=config.CHROMA_PATH)
                      # .get_or_create_collection(config.CHROMA_COLLECTION)
def index_chunks(chunks: list[dict], project_id: str, source_type: str = "tender") -> int
# upsert (id=chunk_id, metadata={project_id, page_number, source_file,
# chunk_type, source_type}); returns count indexed
def retrieve(query: str, project_id: str, top_k: int = 5) -> dict
# {"results": [{"text": str, "page_number": int, "source_file": str,
#   "chunk_type": str, "source_type": str, "distance": float}],
#  "low_confidence": bool}   # True when best distance > config.MIN_CONFIDENCE_DISTANCE
```
Lazy-load model & chroma inside functions (import of module must not load torch).

## backend/pipeline/extraction.py
```python
class RequirementModel(pydantic.BaseModel):  # material, grade|None, quantity|None,
    # unit|None, deadline|None, certification|None, source_page:int,
    # confidence: Literal["high","medium","low"]
def extract_requirements(chunks: list[dict], project_id: str, db_session) -> list[ExtractedRequirement]
# LLM (tool-use forced JSON) per batch of text chunks; "never infer a missing
# field"; validates via pydantic; persists rows; skips invalid responses.
def process_change_request(chunks: list[dict], project_id: str, db_session) -> list[ExtractedRequirement]
# extraction, then for each new req: find existing non-superseded row with same
# project + same material family (case-insensitive substring either way, grade
# may differ) -> INSERT new row, set old.superseded_by = new.id. Else fresh insert.
def get_current_requirements(project_id: str, db_session) -> list[ExtractedRequirement]
# rows where superseded_by IS NULL
```

## backend/pipeline/multi_source_loader.py
```python
def load_document(file_path: str, project_id: str, source_type: str, db_session) -> dict
# source_type in {"tender","meeting_notes","inspection_report","change_request"}
# .pdf via pdf_loader; .docx via python-docx; .txt plain read (single pseudo-page).
# Always: chunk_pages -> index_chunks(source_type=...).
# tender -> extract_requirements; change_request -> process_change_request;
# inspection_report -> LLM-extract InspectionFinding rows (location,
#   defect_description, severity) when API key present; meeting_notes -> index only.
# Records an IngestedDocument row. Returns
# {"project_id", "filename", "source_type", "pages_processed": int,
#  "chunks_indexed": int, "tables_found": int, "requirements_extracted": int}
```

## backend/db/seed_vendors.py
SYNTHETIC demo data (stated in file header comment). Idempotent
(`seed_all(db_session)` skips if vendors exist).
```python
def seed_vendors(db_session) -> int      # 16-18 Vendors, realistic Indian construction
# materials (Fe415/Fe500/Fe500D TMT per IS 1786; OPC 33/43/53, PPC; RMC M20-M60;
# structural steel, cement, aggregates, glass, HVAC). Varied on_time (68-98),
# delivery days (3-45), price_index (0.92-1.15), negotiation_flexibility
# (0.05-0.8), regions across India, plausible fictional names, ratings 3.0-5.0,
# typical_min_qty/typical_max_qty consistent with typical_order_size text.
def seed_quotes(db_session) -> int       # 2-3 VendorQuote rows per vendor for its
# materials, prices varied around plausible INR market rates, project_id="PRJ-2024-001"
def seed_demo_requirements(db_session) -> int  # 4 ExtractedRequirement rows for
# project "PRJ-2024-001" (source_file="demo_seed"): Fe500D TMT steel 120 tonne
# deadline "2026-08-10"; OPC 53 cement 800 bag; M40 ready-mix concrete 350 cum;
# Double-glazed curtain wall glass 1200 sqm.
def seed_all(db_session) -> dict         # {"vendors": n, "quotes": n, "requirements": n}
```
Runnable: `python -m backend.db.seed_vendors`.

## backend/agent/tools.py  (pure functions; the agent + API both call these)
```python
def vendor_lookup(material: str, grade: str | None, db_session) -> list[dict]      # Vendor.to_dict()s whose materials match
def get_market_reference_price(material: str, grade: str | None, db_session) -> float | None  # avg VendorQuote.quoted_price; None if no quotes
def check_compliance(requirement_id: int, db_session, vendor_id: int | None = None) -> dict
# {"requirement": {...}, "status": "compliant"|"non_compliant_alternate_available"|"no_vendor_found",
#  "matching_vendors": [vendor dicts], "explanation": str plain-English}
# exact grade match => compliant; same material family diff grade => alternate.
# If vendor_id given, evaluate that vendor only.
def calculate_risk(requirement_id: int, vendor_id: int, deadline_days_remaining: int, db_session) -> dict
# factors: lead_time_pressure = avg_delivery_days / max(days,1)  (cap 2.0)
#          reliability_factor = (100 - on_time_pct)/100
#          order_size_factor  = 0 if qty within [typical_min,typical_max]; else
#                               min(1, relative distance outside the range)
# score = clamp(lead*RISK_WEIGHT_LEAD_TIME + rel*RISK_WEIGHT_RELIABILITY
#               + size*RISK_WEIGHT_ORDER_SIZE, 0, 100) -> int
# label: 0-33 LOW, 34-66 MEDIUM, 67-100 HIGH
# {"score": int, "label": str, "factors": {raw numbers}, "explanation": str with real numbers}
def vendor_discovery(material: str, grade: str | None, location_hint: str, db_session) -> dict
# internal first (verified True); ONE Tavily call, fixed template
# f"{material} {grade or ''} supplier {location_hint} India"; web results
# verified False with source_url; Tavily failure/missing key => caught,
# {"internal_matches":[...], "web_matches":[], "web_search_succeeded": False}
def vendor_evaluation(candidates: list[dict], material: str, grade: str | None,
                      quantity: float | None, db_session) -> list[dict]
# scores verified candidates only: reliability = on_time/100;
# price = clamp(1 - |vendor_avg_quote - market_ref| / market_ref, 0, 1) (0.5 + note if no data);
# capacity = 1.0 in range else scaled. evaluation_score = 0.4/0.4/0.2 weights
# (config constants). Sorted desc, each with sub-scores + one-sentence summary.
# Unverified candidates appended unscored with note.
def recommend_vendor(requirement_id: int, deadline_days_remaining: int, db_session) -> dict
# discovery -> evaluation -> compliance per candidate -> risk (compliant/near only)
# drop no-match candidates; rank: compliant > alternate, then evaluation_score,
# then lower risk. Returns {"recommended_vendor": {...}|None, "compliance_status",
# "evaluation_summary", "risk_score", "risk_label", "risk_explanation",
# "overall_reason", "alternatives": [up to 2 runner-ups]} or
# {"recommended_vendor": None, "reason": "no_recommendation_possible: ..."}
```

## backend/agent/negotiation.py
```python
def start_negotiation(requirement_id: int, vendor_id: int, db_session) -> Negotiation
# market_ref = get_market_reference_price (fallback: vendor avg quote; else error ValueError)
# vendor_asking_price = market_ref * vendor.price_index
# opening_offer = market_ref * (1 - NEGOTIATION_OPENING_DISCOUNT)
# risk = calculate_risk(...) with DEFAULT_DEADLINE_DAYS
# target_price = market_ref * (1 + 0.03 * risk_score/100)  # urgency concedes more
# creates Negotiation + round 1 (buyer, opening_offer, narrated message)
def run_negotiation_round(negotiation_id: int, db_session) -> NegotiationRound
# alternates actor. DETERMINISTIC pricing:
#   vendor: new = prev_vendor_price - gap * negotiation_flexibility
#           (first vendor turn: vendor_asking_price)
#   buyer:  new = prev_buyer_offer + gap * 0.4, capped at target_price
# gap = abs(last_vendor_price - last_buyer_offer)
# LLM ONLY narrates the computed number (one line, in character); template
# fallback without API key. Stop conditions after each round:
#   gap/opening_offer < NEGOTIATION_CONVERGENCE_PCT -> status pending_approval,
#     final_price = midpoint rounded to 2dp
#   vendor turns used >= max_rounds without convergence -> status "stalled"
def run_full_negotiation(negotiation_id: int, db_session) -> dict
# loops rounds until status != in_progress; returns get_negotiation_state
def approve_negotiation(negotiation_id: int, db_session) -> Negotiation  # pending_approval -> accepted else ValueError
def decline_negotiation(negotiation_id: int, db_session) -> Negotiation  # -> declined
def get_negotiation_state(negotiation_id: int, db_session) -> dict
# {"negotiation": Negotiation.to_dict() + "vendor_name", "rounds": [round dicts]}
```

## backend/agent/purchase_order.py
```python
def generate_po(negotiation_id: int, db_session) -> PurchaseOrder
# ValueError unless negotiation.status == "accepted". po_number =
# f"PO-{project_id}-{seq:04d}". ReportLab Platypus PDF to
# config.PO_DIR / f"{po_number}.pdf"; footer: "Generated by ConstructIQ —
# demo prototype, not a legally binding document". delivery_date = today +
# vendor.avg_delivery_days. payment_terms from vendor's matching quote else "Net 30".
# Creates initial TrackingEvent(status="draft").
```

## backend/agent/tracking.py
```python
class InvalidTransitionError(ValueError): ...
def update_tracking_status(po_id: int, new_status: str, note: str | None, db_session) -> dict
# forward-only per models.PO_STATUS_ORDER; "cancelled" from any non-completed;
# invalid -> InvalidTransitionError with plain-English reason. Appends TrackingEvent.
def get_po_timeline(po_id: int, db_session) -> list[dict]  # TrackingEvent dicts by timestamp asc
```

## backend/agent/agent.py
```python
def run_agent(question: str, project_id: str, db_session) -> dict
# {"answer": str, "tool_calls": [{"tool": str, "input": dict, "output_summary": str}], "final": bool}
```
Raw anthropic tool-use loop (client.messages.create, tools=[...8 tool schemas],
loop while stop_reason=="tool_use", max config.AGENT_MAX_TOOL_CALLS).
Tools: doc_search (embedding.retrieve), vendor_discovery, vendor_evaluation,
compliance_checker, risk_calculator, recommend_vendor, negotiate
(start+run_full, returns state; refuses to approve — human gate),
generate_po (only for accepted negotiations; surface the ValueError politely).
System prompt: full-lifecycle procurement assistant; always cite tools used;
NEVER state a price/compliance/risk number that didn't come from a tool.
No API key => {"answer": "Agent unavailable: ANTHROPIC_API_KEY not set...", "tool_calls": [], "final": False}

## backend/api/  (FastAPI routers; main.py mounts all under /api)
All endpoints: try/except -> log traceback, return generic HTTP 500 detail
"Internal error — see server logs". Pydantic request/response models.
`GET /api/health` -> {"status": "ok"}

ingest.py:  POST /api/ingest  multipart(file: UploadFile, project_id: str Form,
  source_type: str Form default "tender") -> multi_source_loader.load_document dict.
  400 if extension not in {pdf,docx,txt} or (pdf and bad magic bytes). Temp file cleanup in finally.
ask.py:     POST /api/ask {project_id, question} ->
  {"answer": str, "sources": [{"source_file","page_number","snippet"}], "low_confidence": bool}
  retrieval + Claude answer-only-from-context w/ citations; graceful 200 with
  explanatory answer if no API key; 503 on Anthropic API errors.
agent_routes.py: POST /api/agent/ask {project_id, question} -> run_agent dict
vendors.py: GET /api/vendors -> {"vendors": [Vendor.to_dict()...]}
            GET /api/requirements?project_id= -> {"requirements": [...current only...]}
            GET /api/documents?project_id= -> {"documents": [IngestedDocument dicts]}
            POST /api/discovery {material, grade?, location_hint?} -> vendor_discovery dict
            POST /api/recommend {requirement_id, deadline_days_remaining=30} -> recommend dict
risk.py:    GET /api/risk/{project_id} -> {"project_id", "cards": [per current
  requirement: {"requirement": {...}, "vendor": {...}|None, "score", "label",
  "factors", "explanation", "est_value": float|None (quantity x market reference
  price when both known), "est_delivery": "YYYY-MM-DD"|None (today +
  vendor.avg_delivery_days)} using best vendor from recommend-lite
  (compliance-first, else best evaluation); requirements with no vendor get
  label "NO_VENDOR", score 0, explanation saying no vendor found],
  "total_risk_score": int avg of scored cards, "active_mitigations": int count of HIGH}
negotiation_routes.py:
  POST /api/negotiation/start {requirement_id, vendor_id} -> state dict
  POST /api/negotiation/{id}/run -> run_full_negotiation state
  POST /api/negotiation/{id}/round -> single round then state
  GET  /api/negotiation/{id} -> state
  GET  /api/negotiations?project_id= -> {"negotiations": [neg dicts + vendor_name]}
  POST /api/negotiation/{id}/approve -> state (auto-generates PO after approve; include "po": po dict)
  POST /api/negotiation/{id}/decline -> state
po_routes.py:
  GET  /api/po?project_id= -> {"purchase_orders": [po dict + vendor_name]}
  GET  /api/po/{id}/download -> FileResponse(pdf)
  GET  /api/po/{id}/timeline -> {"timeline": [...]}
  POST /api/po/{id}/status {status, note?} -> updated po dict (400 on InvalidTransitionError with reason)
  GET  /api/stats?project_id= -> {"total_po_value": float, "pending_approval": int
  (negotiations pending_approval), "in_transit": int (POs sent/accepted),
  "completion_rate": float pct of POs completed/delivered, "po_count": int}

main.py: FastAPI(title="ConstructIQ API"); CORS allow config.FRONTEND_ORIGIN +
"http://localhost:8000"; include all routers; startup: init_db() + seed_all() +
log Chroma path; mount StaticFiles(directory="frontend", html=True) at "/" LAST.

## frontend/ (plain HTML + Tailwind CDN + vanilla JS)
Pages (adapted from `designs/*/code.html`, keep the Industrial Intelligence look):
  index.html (redirect to chat.html), chat.html, risk.html, vendors.html,
  negotiation.html, purchase_orders.html, js/api.js
Shared js/api.js: `const API = ""` (same origin); helpers apiGet/apiPost/apiUpload;
localStorage-persisted `projectId` (default "PRJ-2024-001"); INR currency
formatter (₹, en-IN); `wireNav()` marking active page.
Top-nav links across ALL pages: chat.html, risk.html, vendors.html,
negotiation.html, purchase_orders.html (same labels as designs).
All dynamic data from the API endpoints above — no hardcoded mock rows.
Errors: visible toast/inline message, never silent console-only.
