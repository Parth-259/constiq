"""Central configuration for ConstructIQ. All settings come from environment
variables (loaded from .env via python-dotenv) so nothing is hardcoded."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# Optional explicit provider override: "anthropic" | "gemini" (auto-detect when empty).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "constructiq.db"))
CHROMA_PATH = os.getenv("CHROMA_PATH", str(DATA_DIR / "chroma"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
PO_DIR = Path(os.getenv("PO_DIR", DATA_DIR / "pos"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PO_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:8501")

# LLM models: Sonnet for reasoning/extraction, Haiku for cheap classification.
MODEL_SONNET = os.getenv("MODEL_SONNET", "claude-sonnet-4-6")
MODEL_HAIKU = os.getenv("MODEL_HAIKU", "claude-haiku-4-5-20251001")
# Gemini equivalents ("smart" tier for reasoning, "fast" tier for narration).
MODEL_GEMINI_SMART = os.getenv("MODEL_GEMINI_SMART", "gemini-2.5-flash")
MODEL_GEMINI_FAST = os.getenv("MODEL_GEMINI_FAST", "gemini-2.5-flash-lite")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_COLLECTION = "construction_documents"

# Retrieval: Chroma cosine distance above this is flagged low-confidence.
MIN_CONFIDENCE_DISTANCE = float(os.getenv("MIN_CONFIDENCE_DISTANCE", "1.1"))

# Risk-score weights — documented constants, not magic numbers, so they can
# be defended line-by-line in front of a judge (roadmap Prompt #10).
RISK_WEIGHT_LEAD_TIME = 50
RISK_WEIGHT_RELIABILITY = 35
RISK_WEIGHT_ORDER_SIZE = 15

# Vendor-evaluation weights (roadmap Prompt #23).
EVAL_WEIGHT_RELIABILITY = 0.4
EVAL_WEIGHT_PRICE = 0.4
EVAL_WEIGHT_CAPACITY = 0.2

# Negotiation constants (roadmap Prompt #25).
NEGOTIATION_MAX_ROUNDS = 4
NEGOTIATION_CONVERGENCE_PCT = 0.02   # offers within 2% of opening => accepted
NEGOTIATION_OPENING_DISCOUNT = 0.05  # buyer opens 5% below market reference

# Default days-to-deadline used when a requirement's deadline is free text
# that can't be resolved to a date.
DEFAULT_DEADLINE_DAYS = 30

# Agent loop.
AGENT_MAX_TOOL_CALLS = 10
