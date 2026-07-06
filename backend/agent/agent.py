"""ConstructIQ procurement agent — thin wrapper over ``backend.llm.tools_loop``.

``run_agent`` builds the 8 tool specs (schema + ``fn`` closure over
``project_id`` / ``db_session``) and hands them to
:func:`backend.llm.tools_loop`, which drives the provider-native tool-use loop
(Anthropic messages loop or Gemini function calling). The loop is hard-capped
at ``config.AGENT_MAX_TOOL_CALLS`` tool executions, after which a graceful
"tool budget exhausted" answer is returned.

Design rules (per CONTRACT.md):
- ``db_session`` and ``project_id`` are injected server-side into every tool
  call and are never exposed in the tool JSON schemas.
- Each tool result is truncated to ~2000 chars for the model transcript, but
  the full output is logged; ``tool_calls`` records the first 200 chars.
- No configured LLM provider => deterministic fallback dict, never a crash.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend import config, llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are ConstructIQ's procurement agent: a full-lifecycle construction "
    "procurement assistant for Indian construction projects (all money is INR). "
    "You cover tender/document search, vendor discovery and evaluation, "
    "compliance checking, delivery-risk scoring, automated price negotiation, "
    "and purchase-order generation — always by calling your tools.\n\n"
    "Non-negotiable rules:\n"
    "1. NEVER state a price, quantity, compliance status, risk score, or any "
    "other number that did not come from a tool result in this conversation. "
    "If you have not called a tool for it, you do not know it.\n"
    "2. Always cite which tools your answer is based on (e.g. 'per doc_search' "
    "or 'per risk_calculator').\n"
    "3. Negotiation approval is a HUMAN-ONLY gate. You may start and run "
    "negotiations with the negotiate tool, but you must REFUSE any request to "
    "approve, accept, or finalize a negotiation yourself — a human must "
    "approve it in the ConstructIQ interface.\n"
    "4. Purchase orders can only be generated for negotiations a human has "
    "already approved (status 'accepted'). If generate_po reports that a "
    "negotiation is not accepted, relay that message politely and explain the "
    "human approval step.\n"
    "5. If tool results are empty, low-confidence, or insufficient, say so "
    "plainly instead of guessing."
)

# --- 8 tool schemas (db_session / project_id are injected server-side) -------
TOOLS: list[dict[str, Any]] = [
    {
        "name": "doc_search",
        "description": (
            "Semantic search over this project's ingested documents (tenders, "
            "meeting notes, inspection reports, change requests); call it "
            "whenever the answer depends on document content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to retrieve (default 5).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "vendor_discovery",
        "description": (
            "Find suppliers for a material — verified internal vendors first, "
            "plus unverified web matches; call it when you need candidate "
            "vendors for a material."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "material": {
                    "type": "string",
                    "description": "Material name, e.g. 'TMT steel' or 'OPC 53 cement'.",
                },
                "grade": {
                    "type": ["string", "null"],
                    "description": "Grade such as 'Fe500D' or 'OPC 53'; null if unknown.",
                },
                "location_hint": {
                    "type": "string",
                    "description": "Region or city to focus the search on (default 'India').",
                },
            },
            "required": ["material"],
            "additionalProperties": False,
        },
    },
    {
        "name": "vendor_evaluation",
        "description": (
            "Score a list of candidate vendors on reliability, price "
            "competitiveness, and capacity for a material and quantity; call "
            "it after vendor_discovery to rank the candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Candidate vendor dicts (from vendor_discovery).",
                },
                "material": {
                    "type": "string",
                    "description": "Material the vendors are evaluated for.",
                },
                "grade": {
                    "type": ["string", "null"],
                    "description": "Required grade; null if not specified.",
                },
                "quantity": {
                    "type": ["number", "null"],
                    "description": "Required quantity in the requirement's unit; null if unknown.",
                },
            },
            "required": ["candidates", "material"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compliance_checker",
        "description": (
            "Check whether vendors can supply the exact grade a requirement "
            "demands (compliant / alternate grade available / no vendor); call "
            "it to verify grade compliance for a stored requirement."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "integer",
                    "description": "ID of the stored extracted requirement.",
                },
                "vendor_id": {
                    "type": ["integer", "null"],
                    "description": "Evaluate only this vendor when given; null for all vendors.",
                },
            },
            "required": ["requirement_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "risk_calculator",
        "description": (
            "Compute a 0-100 delivery-risk score (LOW/MEDIUM/HIGH) for one "
            "vendor against one requirement given the days remaining to "
            "deadline; call it before recommending or negotiating with a vendor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "integer",
                    "description": "ID of the stored extracted requirement.",
                },
                "vendor_id": {
                    "type": "integer",
                    "description": "ID of the vendor to score.",
                },
                "deadline_days_remaining": {
                    "type": "integer",
                    "description": "Days remaining until the requirement deadline (default 30).",
                },
            },
            "required": ["requirement_id", "vendor_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recommend_vendor",
        "description": (
            "Run the full pipeline (discovery -> evaluation -> compliance -> "
            "risk) and return the single best vendor with alternatives for a "
            "requirement; call it when asked which vendor to pick."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "integer",
                    "description": "ID of the stored extracted requirement.",
                },
                "deadline_days_remaining": {
                    "type": "integer",
                    "description": "Days remaining until the requirement deadline (default 30).",
                },
            },
            "required": ["requirement_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "negotiate",
        "description": (
            "Start and run a full automated buyer-vendor price negotiation for "
            "a requirement with a chosen vendor and return the round-by-round "
            "state; the final approval is a human-only action you must never "
            "perform."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "requirement_id": {
                    "type": "integer",
                    "description": "ID of the stored extracted requirement.",
                },
                "vendor_id": {
                    "type": "integer",
                    "description": "ID of the vendor to negotiate with.",
                },
            },
            "required": ["requirement_id", "vendor_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "generate_po",
        "description": (
            "Generate the purchase-order PDF and record for a negotiation that "
            "a human has already approved (status 'accepted'); call it only "
            "after the user confirms the negotiation was approved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "negotiation_id": {
                    "type": "integer",
                    "description": "ID of the accepted negotiation to turn into a PO.",
                },
            },
            "required": ["negotiation_id"],
            "additionalProperties": False,
        },
    },
]


def _dispatch(name: str, tool_input: dict[str, Any], project_id: str, db_session: Any) -> Any:
    """Execute one tool against the real backend functions.

    Dependency modules are imported lazily so that importing this module never
    pulls in heavy dependencies (and so unit tests can patch the callees).
    """
    if name == "doc_search":
        from backend.pipeline import embedding

        return embedding.retrieve(
            query=str(tool_input["query"]),
            project_id=project_id,
            top_k=int(tool_input.get("top_k") or 5),
        )
    if name == "vendor_discovery":
        from backend.agent import tools

        return tools.vendor_discovery(
            str(tool_input["material"]),
            tool_input.get("grade"),
            str(tool_input.get("location_hint") or "India"),
            db_session,
        )
    if name == "vendor_evaluation":
        from backend.agent import tools

        quantity = tool_input.get("quantity")
        return tools.vendor_evaluation(
            list(tool_input.get("candidates") or []),
            str(tool_input["material"]),
            tool_input.get("grade"),
            float(quantity) if quantity is not None else None,
            db_session,
        )
    if name == "compliance_checker":
        from backend.agent import tools

        vendor_id = tool_input.get("vendor_id")
        return tools.check_compliance(
            int(tool_input["requirement_id"]),
            db_session,
            vendor_id=int(vendor_id) if vendor_id is not None else None,
        )
    if name == "risk_calculator":
        from backend.agent import tools

        return tools.calculate_risk(
            int(tool_input["requirement_id"]),
            int(tool_input["vendor_id"]),
            int(tool_input.get("deadline_days_remaining") or config.DEFAULT_DEADLINE_DAYS),
            db_session,
        )
    if name == "recommend_vendor":
        from backend.agent import tools

        return tools.recommend_vendor(
            int(tool_input["requirement_id"]),
            int(tool_input.get("deadline_days_remaining") or config.DEFAULT_DEADLINE_DAYS),
            db_session,
        )
    if name == "negotiate":
        from backend.agent import negotiation

        started = negotiation.start_negotiation(
            int(tool_input["requirement_id"]),
            int(tool_input["vendor_id"]),
            db_session,
        )
        return negotiation.run_full_negotiation(started.id, db_session)
    if name == "generate_po":
        from backend.agent import purchase_order

        po = purchase_order.generate_po(int(tool_input["negotiation_id"]), db_session)
        return po.to_dict()
    raise ValueError(f"Unknown tool: {name}")


def _execute_tool(
    name: str, tool_input: dict[str, Any], project_id: str, db_session: Any
) -> tuple[str, bool]:
    """Run one tool; return ``(full_output_text, is_error)``.

    ValueErrors (e.g. generate_po on a non-accepted negotiation) become the
    tool result text so the model can relay the message; any other exception is
    caught and reported as an error result — the loop must never crash.
    """
    logger.info("agent tool call: %s input=%s", name, tool_input)
    try:
        result = _dispatch(name, tool_input, project_id, db_session)
    except ValueError as exc:
        logger.warning("agent tool %s returned ValueError: %s", name, exc)
        return str(exc), True
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the loop
        logger.exception("agent tool %s failed", name)
        return f"Tool '{name}' failed: {exc}", True
    full_text = json.dumps(result, ensure_ascii=False, default=str)
    # Full (untruncated) output goes to the log; the transcript gets ~2000 chars.
    logger.debug("agent tool %s full output: %s", name, full_text)
    return full_text, False


def run_agent(question: str, project_id: str, db_session: Any) -> dict:
    """Answer a procurement question by driving the LLM tool-use loop.

    Returns ``{"answer": str, "tool_calls": [{"tool", "input",
    "output_summary"}], "final": bool}``.
    """
    if not llm.is_configured():
        logger.warning("run_agent called without a configured LLM provider; returning fallback")
        return {
            "answer": (
                "Agent unavailable: ANTHROPIC_API_KEY not set — configure the "
                "key (or GEMINI_API_KEY) in the environment to enable the "
                "procurement agent."
            ),
            "tool_calls": [],
            "final": False,
        }

    def _make_fn(tool_name: str) -> Any:
        def fn(tool_input: dict) -> tuple[str, bool]:
            return _execute_tool(tool_name, tool_input, project_id, db_session)

        return fn

    specs = [{**tool, "fn": _make_fn(tool["name"])} for tool in TOOLS]
    result = llm.tools_loop(
        SYSTEM_PROMPT, question, specs, max_calls=config.AGENT_MAX_TOOL_CALLS
    )
    logger.info(
        "agent finished for project %s: %d tool call(s), final=%s",
        project_id,
        len(result.get("tool_calls", [])),
        result.get("final"),
    )
    return result
