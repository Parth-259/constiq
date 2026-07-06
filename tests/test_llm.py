"""Hermetic tests for backend.llm — ALL SDK clients are mocked, no real calls."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from backend import config, llm


def _set_keys(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_key: str = "",
    gemini_key: str = "",
    override: str = "",
) -> None:
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setattr(config, "GEMINI_API_KEY", gemini_key)
    monkeypatch.setattr(config, "LLM_PROVIDER", override)


# --------------------------------------------------- provider() selection ----


@pytest.mark.parametrize(
    ("anthropic_key", "gemini_key", "override", "expected"),
    [
        ("ant-key", "", "", "anthropic"),            # anthropic only
        ("", "gem-key", "", "gemini"),               # gemini only
        ("ant-key", "gem-key", "", "anthropic"),     # both -> anthropic wins
        ("ant-key", "gem-key", "gemini", "gemini"),  # explicit override wins
        ("ant-key", "gem-key", "anthropic", "anthropic"),
        ("ant-key", "", "gemini", "anthropic"),      # override ignored w/o its key
        ("", "gem-key", "anthropic", "gemini"),      # override ignored w/o its key
        ("", "", "", "none"),                        # nothing configured
        ("", "", "gemini", "none"),                  # override without any key
    ],
)
def test_provider_selection_matrix(
    monkeypatch: pytest.MonkeyPatch,
    anthropic_key: str,
    gemini_key: str,
    override: str,
    expected: str,
) -> None:
    _set_keys(monkeypatch, anthropic_key, gemini_key, override)
    assert llm.provider() == expected
    assert llm.is_configured() is (expected != "none")


# --------------------------------------------------------------- complete ----


def test_complete_raises_llmerror_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch)
    with pytest.raises(llm.LLMError, match="No LLM provider configured"):
        llm.complete("system", "user")


def test_complete_wraps_api_failure_in_llmerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, anthropic_key="ant-key")
    client = mock.MagicMock()
    client.messages.create.side_effect = RuntimeError("overloaded")
    with mock.patch("anthropic.Anthropic", return_value=client):
        with pytest.raises(llm.LLMError) as excinfo:
            llm.complete("system", "user")
    assert isinstance(excinfo.value.__cause__, RuntimeError)  # chained


def test_complete_anthropic_tiers_select_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, anthropic_key="ant-key")
    client = mock.MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")]
    )
    with mock.patch("anthropic.Anthropic", return_value=client):
        assert llm.complete("sys", "user", tier="fast") == "hello"
        assert llm.complete("sys", "user", tier="smart") == "hello"
    models = [c.kwargs["model"] for c in client.messages.create.call_args_list]
    assert models == [config.MODEL_HAIKU, config.MODEL_SONNET]


# ------------------------------------------------------------ extract_json ----

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "grade": {"type": ["string", "null"]},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["material"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requirements"],
    "additionalProperties": False,
}

PAYLOAD = {"requirements": [{"material": "OPC 53 cement", "grade": None, "confidence": "high"}]}


def test_extract_json_anthropic_returns_tool_use_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, anthropic_key="ant-key")
    block = SimpleNamespace(
        type="tool_use", name="record_requirements", input=PAYLOAD, id="toolu_1"
    )
    client = mock.MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[block], stop_reason="tool_use"
    )
    with mock.patch("anthropic.Anthropic", return_value=client) as ctor:
        result = llm.extract_json(
            "sys", "user", SCHEMA, tool_name="record_requirements"
        )

    ctor.assert_called_once_with(api_key="ant-key")
    assert result == PAYLOAD
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "record_requirements"}
    assert kwargs["tools"][0]["name"] == "record_requirements"
    assert kwargs["tools"][0]["input_schema"] is SCHEMA
    assert kwargs["messages"][0]["content"] == "user"


def test_extract_json_anthropic_failure_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, anthropic_key="ant-key")
    client = mock.MagicMock()
    client.messages.create.side_effect = RuntimeError("boom")
    with mock.patch("anthropic.Anthropic", return_value=client):
        assert llm.extract_json("sys", "user", SCHEMA) is None


def test_extract_json_gemini_parses_json_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, gemini_key="gem-key")
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(
        text=json.dumps(PAYLOAD)
    )
    with mock.patch("google.genai.Client", return_value=client) as ctor:
        result = llm.extract_json("sys", "user", SCHEMA, tool_name="record")

    ctor.assert_called_once_with(api_key="gem-key")
    assert result == PAYLOAD
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == config.MODEL_GEMINI_SMART
    assert kwargs["contents"] == "user"
    gen_config = kwargs["config"]
    assert gen_config["response_mime_type"] == "application/json"
    converted = gen_config["response_schema"]
    # Unsupported JSON-schema keys are stripped; the rest maps through.
    assert "additionalProperties" not in json.dumps(converted)
    item_schema = converted["properties"]["requirements"]["items"]
    assert item_schema["properties"]["grade"] == {"type": "STRING", "nullable": True}
    assert item_schema["properties"]["confidence"]["enum"] == ["high", "medium", "low"]
    assert item_schema["required"] == ["material"]
    assert converted["type"] == "OBJECT"


def test_extract_json_gemini_invalid_json_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, gemini_key="gem-key")
    client = mock.MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="not json {")
    with mock.patch("google.genai.Client", return_value=client):
        assert llm.extract_json("sys", "user", SCHEMA) is None


def test_extract_json_returns_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch)
    assert llm.extract_json("sys", "user", SCHEMA) is None


# -------------------------------------------------------------- tools_loop ----


def _gemini_fc_response(name: str, args: dict) -> SimpleNamespace:
    part = SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part], role="model"))]
    )


def _gemini_text_response(text: str) -> SimpleNamespace:
    part = SimpleNamespace(function_call=None, text=text)
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part], role="model"))]
    )


def test_tools_loop_gemini_executes_fn_then_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, gemini_key="gem-key")
    fn = mock.MagicMock(return_value='{"results": ["OPC 53 on p.3"]}')
    tools = [
        {
            "name": "doc_search",
            "description": "Search the docs.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            "fn": fn,
        }
    ]
    client = mock.MagicMock()
    client.models.generate_content.side_effect = [
        _gemini_fc_response("doc_search", {"query": "cement grade"}),
        _gemini_text_response("Per doc_search, the tender requires OPC 53."),
    ]
    with mock.patch("google.genai.Client", return_value=client):
        result = llm.tools_loop("sys", "What cement grade?", tools, max_calls=5)

    fn.assert_called_once_with({"query": "cement grade"})
    assert result["final"] is True
    assert result["answer"] == "Per doc_search, the tender requires OPC 53."
    assert result["tool_calls"] == [
        {
            "tool": "doc_search",
            "input": {"query": "cement grade"},
            "output_summary": '{"results": ["OPC 53 on p.3"]}',
        }
    ]
    assert client.models.generate_content.call_count == 2
    # Second call carries the function response back in the transcript.
    contents = client.models.generate_content.call_args.kwargs["contents"]
    fn_response_part = contents[-1].parts[0]
    assert fn_response_part.function_response.name == "doc_search"
    assert "OPC 53 on p.3" in str(fn_response_part.function_response.response)


def test_tools_loop_gemini_stops_at_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_keys(monkeypatch, gemini_key="gem-key")
    fn = mock.MagicMock(return_value="result")
    tools = [
        {
            "name": "echo",
            "description": "",
            "input_schema": {"type": "object", "properties": {}},
            "fn": fn,
        }
    ]
    client = mock.MagicMock()
    client.models.generate_content.return_value = _gemini_fc_response("echo", {})
    with mock.patch("google.genai.Client", return_value=client):
        result = llm.tools_loop("sys", "loop forever", tools, max_calls=3)

    assert fn.call_count == 3
    assert len(result["tool_calls"]) == 3
    assert result["final"] is False
    assert "budget" in result["answer"].lower()


def test_tools_loop_raises_llmerror_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch)
    with pytest.raises(llm.LLMError, match="No LLM provider configured"):
        llm.tools_loop("sys", "user", [], max_calls=1)


def test_tools_loop_anthropic_wraps_api_failure_in_llmerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, anthropic_key="ant-key")
    client = mock.MagicMock()
    client.messages.create.side_effect = RuntimeError("overloaded")
    with mock.patch("anthropic.Anthropic", return_value=client):
        with pytest.raises(llm.LLMError, match="tools loop failed") as excinfo:
            llm.tools_loop("sys", "user", [], max_calls=1)
    assert isinstance(excinfo.value.__cause__, RuntimeError)  # chained


def test_tools_loop_gemini_wraps_api_failure_in_llmerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_keys(monkeypatch, gemini_key="gem-key")
    client = mock.MagicMock()
    client.models.generate_content.side_effect = RuntimeError("quota exceeded")
    with mock.patch("google.genai.Client", return_value=client):
        with pytest.raises(llm.LLMError, match="tools loop failed") as excinfo:
            llm.tools_loop("sys", "user", [], max_calls=1)
    assert isinstance(excinfo.value.__cause__, RuntimeError)  # chained
