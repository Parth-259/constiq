"""Provider-agnostic LLM access layer for ConstructIQ.

This is the ONLY module that talks to LLM SDKs. Every other module goes
through :func:`complete`, :func:`extract_json` or :func:`tools_loop` and never
imports ``anthropic`` or ``google-genai`` directly.

Provider selection (:func:`provider`):
- an explicit ``config.LLM_PROVIDER`` override ("anthropic" | "gemini") wins
  when the matching API key is configured;
- otherwise "anthropic" when ``config.ANTHROPIC_API_KEY`` is set, else
  "gemini" when ``config.GEMINI_API_KEY`` is set, else "none".

Model tiers: ``tier="smart"`` maps to ``MODEL_SONNET`` / ``MODEL_GEMINI_SMART``
(reasoning, extraction, agent loop); ``tier="fast"`` maps to ``MODEL_HAIKU`` /
``MODEL_GEMINI_FAST`` (cheap narration).

Both SDKs are imported lazily INSIDE functions so importing this module stays
dependency-light and tests can substitute the clients.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from backend import config

logger = logging.getLogger(__name__)

# --- transcript / bookkeeping limits (tools_loop) -----------------------------
TRANSCRIPT_RESULT_MAX_CHARS = 2000  # per tool result, in the model transcript
SUMMARY_MAX_CHARS = 200             # per tool result, in the returned tool_calls
_TOOLS_LOOP_MAX_TOKENS = 4096
_EXTRACT_MAX_TOKENS = 4096

NO_PROVIDER_MESSAGE = "No LLM provider configured"

_GEMINI_TYPES = {"string", "number", "integer", "boolean", "array", "object"}


class LLMError(Exception):
    """Raised when an LLM call fails or no provider is configured."""


# --- provider selection -------------------------------------------------------

def provider() -> str:
    """Return the active provider: "anthropic" | "gemini" | "none".

    An explicit ``config.LLM_PROVIDER`` override wins when its key exists;
    otherwise auto-detect by which API key is configured (Anthropic first).
    """
    override = (config.LLM_PROVIDER or "").strip().lower()
    if override == "anthropic" and config.ANTHROPIC_API_KEY:
        return "anthropic"
    if override == "gemini" and config.GEMINI_API_KEY:
        return "gemini"
    if config.ANTHROPIC_API_KEY:
        return "anthropic"
    if config.GEMINI_API_KEY:
        return "gemini"
    return "none"


def is_configured() -> bool:
    """True when at least one LLM provider can be used."""
    return provider() != "none"


# --- lazy SDK clients ---------------------------------------------------------

def _anthropic_client() -> Any:
    import anthropic

    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def _gemini_client() -> Any:
    from google import genai

    return genai.Client(api_key=config.GEMINI_API_KEY)


def _anthropic_model(tier: str) -> str:
    return config.MODEL_SONNET if tier == "smart" else config.MODEL_HAIKU


def _gemini_model(tier: str) -> str:
    return config.MODEL_GEMINI_SMART if tier == "smart" else config.MODEL_GEMINI_FAST


def _gemini_thinking_budget(model: str) -> int | None:
    """Thinking budget override for Gemini 2.5 models.

    2.5 models think by default and those tokens count against
    ``max_output_tokens`` — on a long RAG prompt the whole budget can be spent
    thinking, yielding an empty response. Flash/flash-lite accept 0 (off);
    2.5-pro cannot fully disable thinking, so it gets the minimum (128).
    Non-2.5 models don't accept the parameter at all -> None.
    """
    if not model.startswith("gemini-2.5"):
        return None
    return 128 if "pro" in model else 0


def _gemini_text(response: Any) -> str:
    """Join all text parts of the first candidate; '' when there are none.

    Avoids ``response.text``, which raises on candidates that contain no text
    part (e.g. when thinking consumed the whole output budget).
    """
    candidates = getattr(response, "candidates", None) or []
    content = getattr(candidates[0], "content", None) if candidates else None
    parts = getattr(content, "parts", None) or []
    text = "".join(str(getattr(p, "text", "") or "") for p in parts).strip()
    if text:
        return text
    try:  # fallback: .text raises on candidates without text parts
        return str(getattr(response, "text", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# --- schema conversion (JSON Schema -> genai-accepted subset) ------------------

def to_gemini_schema(schema: Any) -> Any:
    """Convert a JSON schema to the subset google-genai accepts.

    Strips unsupported keys (e.g. ``additionalProperties``), maps union types
    like ``["string", "null"]`` to a nullable single type, and keeps
    ``required`` / ``enum`` / ``description`` plus recursive ``properties`` /
    ``items``.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}

    stype = schema.get("type")
    nullable = False
    if isinstance(stype, list):
        non_null = [t for t in stype if t != "null"]
        nullable = "null" in stype
        stype = non_null[0] if non_null else "string"
    if isinstance(stype, str) and stype.lower() in _GEMINI_TYPES:
        out["type"] = stype.upper()
    if nullable:
        out["nullable"] = True

    if isinstance(schema.get("description"), str):
        out["description"] = schema["description"]
    if isinstance(schema.get("enum"), list):
        out["enum"] = [str(v) for v in schema["enum"]]
    if isinstance(schema.get("required"), list):
        out["required"] = list(schema["required"])

    properties = schema.get("properties")
    if isinstance(properties, dict):
        out["properties"] = {
            key: to_gemini_schema(value) for key, value in properties.items()
        }
    if isinstance(schema.get("items"), dict):
        out["items"] = to_gemini_schema(schema["items"])
    return out


# --- plain completion ----------------------------------------------------------

def _anthropic_text(response: Any) -> str:
    """Concatenate the text of an Anthropic response's content blocks."""
    return "".join(
        str(getattr(block, "text", "") or "")
        for block in (getattr(response, "content", None) or [])
    ).strip()


def complete(system: str, user: str, max_tokens: int = 1024, tier: str = "smart") -> str:
    """Single-turn completion via the active provider; returns the text.

    Raises :class:`LLMError` (chained) on any API failure, and
    ``LLMError("No LLM provider configured")`` when no provider is set up.
    """
    active = provider()
    if active == "none":
        raise LLMError(NO_PROVIDER_MESSAGE)
    try:
        if active == "anthropic":
            client = _anthropic_client()
            response = client.messages.create(
                model=_anthropic_model(tier),
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return _anthropic_text(response)
        client = _gemini_client()
        model = _gemini_model(tier)
        gen_config: dict[str, Any] = {
            "system_instruction": system,
            "max_output_tokens": max_tokens,
        }
        budget = _gemini_thinking_budget(model)
        if budget is not None:
            gen_config["thinking_config"] = {"thinking_budget": budget}
        response = client.models.generate_content(
            model=model, contents=user, config=gen_config
        )
        text = _gemini_text(response)
        if not text:
            finish = getattr(
                (getattr(response, "candidates", None) or [None])[0],
                "finish_reason",
                None,
            )
            raise LLMError(f"gemini returned no text (finish_reason={finish})")
        return text
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalize every SDK failure
        raise LLMError(f"{active} completion failed: {exc}") from exc


# --- forced structured output ---------------------------------------------------

def extract_json(
    system: str, user: str, schema: dict, tool_name: str = "record"
) -> dict | None:
    """Forced structured output: return a dict matching ``schema``, or None.

    - anthropic: forced tool-use (``tool_choice={"type": "tool", ...}``); the
      ``tool_use`` block's ``input`` dict is returned.
    - gemini: ``response_mime_type="application/json"`` with the schema
      converted to the genai-accepted subset; the text is ``json.loads``-ed.

    Any failure (API error, missing block, invalid JSON, missing provider) is
    logged as a warning and returns ``None`` — callers pydantic-validate the
    payload anyway.
    """
    active = provider()
    if active == "none":
        logger.warning("extract_json skipped: no LLM provider configured")
        return None
    try:
        if active == "anthropic":
            client = _anthropic_client()
            response = client.messages.create(
                model=_anthropic_model("smart"),
                max_tokens=_EXTRACT_MAX_TOKENS,
                system=system,
                tools=[
                    {
                        "name": tool_name,
                        "description": (
                            "Record the extracted data as structured JSON "
                            "matching the input schema."
                        ),
                        "input_schema": schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
                messages=[{"role": "user", "content": user}],
            )
            for block in getattr(response, "content", None) or []:
                if (
                    getattr(block, "type", None) == "tool_use"
                    and getattr(block, "name", None) == tool_name
                    and isinstance(getattr(block, "input", None), dict)
                ):
                    return block.input
            logger.warning(
                "extract_json: model response contained no %s tool_use block",
                tool_name,
            )
            return None

        client = _gemini_client()
        model = _gemini_model("smart")
        gen_config: dict[str, Any] = {
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": to_gemini_schema(schema),
        }
        budget = _gemini_thinking_budget(model)
        if budget is not None:
            gen_config["thinking_config"] = {"thinking_budget": budget}
        response = client.models.generate_content(
            model=model, contents=user, config=gen_config
        )
        text = _gemini_text(response)
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            logger.warning("extract_json: gemini returned non-object JSON: %r", parsed)
            return None
        return parsed
    except Exception:  # noqa: BLE001 — extraction must degrade, never crash
        logger.warning("extract_json failed via %s provider", active, exc_info=True)
        return None


# --- agentic tool-use loop -------------------------------------------------------

def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to ``limit`` chars, appending a marker when clipped."""
    if len(text) <= limit:
        return text
    return text[:limit] + " …[truncated]"


def _run_tool_fn(name: str, fn: Any, tool_input: dict) -> tuple[str, bool]:
    """Execute one tool fn; return ``(full_output_text, is_error)``.

    ``fn`` may return a plain string or a ``(text, is_error)`` tuple; any
    exception is caught and reported as an error result — the loop must never
    crash.
    """
    logger.info("llm tool call: %s input=%s", name, tool_input)
    try:
        result = fn(tool_input)
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the loop
        logger.exception("llm tool %s failed", name)
        return f"Tool '{name}' failed: {exc}", True
    if isinstance(result, tuple):
        text, is_error = result
        text = str(text)
    else:
        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, default=str
        )
        is_error = False
    # Full (untruncated) output goes to the log; transcripts get ~2000 chars.
    logger.debug("llm tool %s full output: %s", name, text)
    return text, bool(is_error)


def _budget_answer(max_calls: int) -> str:
    return (
        f"Tool budget exhausted: I stopped after {max_calls} tool calls "
        "without reaching a final answer. Please ask a narrower question or "
        "re-run to continue the investigation."
    )


_EMPTY_ANSWER = (
    "The agent finished without producing a text answer — please try rephrasing."
)


def tools_loop(system: str, user: str, tools: list[dict], max_calls: int) -> dict:
    """Drive a provider-native tool-use loop until a final text answer.

    ``tools`` items: ``{"name", "description", "input_schema", "fn"}`` where
    ``fn(input: dict) -> str`` (or ``(str, is_error)``) executes the tool.
    Returns ``{"answer": str, "tool_calls": [{"tool", "input",
    "output_summary"}], "final": bool}``. Raises :class:`LLMError` when no
    provider is configured, and :class:`LLMError` (chained) on any provider
    API failure — raw SDK exceptions never escape this module.
    """
    active = provider()
    if active == "none":
        raise LLMError(NO_PROVIDER_MESSAGE)
    try:
        if active == "anthropic":
            return _anthropic_tools_loop(system, user, tools, max_calls)
        return _gemini_tools_loop(system, user, tools, max_calls)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001 — normalize every SDK failure
        raise LLMError(f"{active} tools loop failed: {exc}") from exc


def _anthropic_tools_loop(
    system: str, user: str, tools: list[dict], max_calls: int
) -> dict:
    """Raw Anthropic messages.create tool-use loop (no framework)."""
    import anthropic  # noqa: F401 — lazy import; keep module import light

    client = _anthropic_client()
    api_tools = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t["input_schema"],
        }
        for t in tools
    ]
    fns = {t["name"]: t["fn"] for t in tools}

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    tool_calls: list[dict[str, Any]] = []
    budget_exhausted = False

    def _create() -> Any:
        try:
            return client.messages.create(
                model=_anthropic_model("smart"),
                max_tokens=_TOOLS_LOOP_MAX_TOKENS,
                system=system,
                tools=api_tools,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"anthropic tools loop failed: {exc}") from exc

    response = _create()
    while response.stop_reason == "tool_use":
        tool_use_blocks = [
            block for block in response.content if getattr(block, "type", "") == "tool_use"
        ]
        if not tool_use_blocks:  # defensive: malformed response, avoid spinning
            logger.warning("stop_reason=tool_use but no tool_use blocks; stopping")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            if len(tool_calls) >= max_calls:
                budget_exhausted = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": "Tool budget exhausted — no further tool calls are permitted.",
                        "is_error": True,
                    }
                )
                continue
            tool_input = dict(block.input or {})
            fn = fns.get(block.name)
            if fn is None:
                full_text, is_error = f"Unknown tool: {block.name}", True
            else:
                full_text, is_error = _run_tool_fn(block.name, fn, tool_input)
            tool_calls.append(
                {
                    "tool": block.name,
                    "input": tool_input,
                    "output_summary": full_text[:SUMMARY_MAX_CHARS],
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": _truncate(full_text, TRANSCRIPT_RESULT_MAX_CHARS),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

        if budget_exhausted or len(tool_calls) >= max_calls:
            budget_exhausted = True
            break

        response = _create()

    if budget_exhausted:
        logger.warning("tools_loop stopped: tool budget of %d exhausted", max_calls)
        return {"answer": _budget_answer(max_calls), "tool_calls": tool_calls, "final": False}

    answer = "".join(
        str(getattr(block, "text", "") or "")
        for block in (getattr(response, "content", None) or [])
        if getattr(block, "type", "") == "text"
    ).strip()
    if not answer:
        logger.warning(
            "tools_loop finished (stop_reason=%s) with no text answer",
            getattr(response, "stop_reason", None),
        )
        answer = _EMPTY_ANSWER
    logger.info(
        "tools_loop finished via anthropic: %d tool call(s), stop_reason=%s",
        len(tool_calls),
        getattr(response, "stop_reason", None),
    )
    return {"answer": answer, "tool_calls": tool_calls, "final": True}


def _gemini_tools_loop(
    system: str, user: str, tools: list[dict], max_calls: int
) -> dict:
    """Manual google-genai function-calling loop (automatic calling disabled)."""
    from google.genai import types

    client = _gemini_client()
    fns = {t["name"]: t["fn"] for t in tools}
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters=to_gemini_schema(t["input_schema"]),
        )
        for t in tools
    ]
    model = _gemini_model("smart")
    budget = _gemini_thinking_budget(model)
    gen_config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=_TOOLS_LOOP_MAX_TOKENS,
        tools=[types.Tool(function_declarations=declarations)],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=(
            types.ThinkingConfig(thinking_budget=budget)
            if budget is not None
            else None
        ),
    )
    contents: list[Any] = [
        types.Content(role="user", parts=[types.Part(text=user)])
    ]
    tool_calls: list[dict[str, Any]] = []
    budget_exhausted = False

    while True:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"gemini tools loop failed: {exc}") from exc
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        parts = list(getattr(content, "parts", None) or [])
        function_calls = [
            part.function_call
            for part in parts
            if getattr(part, "function_call", None) is not None
        ]

        if not function_calls:  # plain text => final answer
            answer = "".join(
                str(getattr(part, "text", "") or "") for part in parts
            ).strip()
            if not answer:
                logger.warning("tools_loop (gemini) finished with no text answer")
                answer = _EMPTY_ANSWER
            logger.info(
                "tools_loop finished via gemini: %d tool call(s)", len(tool_calls)
            )
            return {"answer": answer, "tool_calls": tool_calls, "final": True}

        contents.append(content)
        response_parts: list[Any] = []
        for call in function_calls:
            if len(tool_calls) >= max_calls:
                budget_exhausted = True
                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={
                            "error": "Tool budget exhausted — no further tool calls are permitted."
                        },
                    )
                )
                continue
            tool_input = dict(getattr(call, "args", None) or {})
            fn = fns.get(call.name)
            if fn is None:
                full_text, is_error = f"Unknown tool: {call.name}", True
            else:
                full_text, is_error = _run_tool_fn(call.name, fn, tool_input)
            tool_calls.append(
                {
                    "tool": call.name,
                    "input": tool_input,
                    "output_summary": full_text[:SUMMARY_MAX_CHARS],
                }
            )
            key = "error" if is_error else "result"
            response_parts.append(
                types.Part.from_function_response(
                    name=call.name,
                    response={key: _truncate(full_text, TRANSCRIPT_RESULT_MAX_CHARS)},
                )
            )
        contents.append(types.Content(role="user", parts=response_parts))

        if budget_exhausted or len(tool_calls) >= max_calls:
            logger.warning(
                "tools_loop (gemini) stopped: tool budget of %d exhausted", max_calls
            )
            return {
                "answer": _budget_answer(max_calls),
                "tool_calls": tool_calls,
                "final": False,
            }
