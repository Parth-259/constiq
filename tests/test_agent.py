"""Hermetic tests for the agent tool-use loop (backend/agent/agent.py).

The Anthropic client is always faked (never a real API call); doc_search's
backing function ``backend.pipeline.embedding.retrieve`` is patched so no
model/Chroma artifacts are ever loaded; the DB is in-memory SQLite.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterator
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.agent import agent
from backend.db.models import Base


@pytest.fixture()
def db_session() -> Iterator[Any]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()
    engine.dispose()


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=tool_input)


def _response(stop_reason: str, content: list) -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeMessages:
    """Scripted messages.create: pops responses; repeats the last one forever."""

    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


class FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.messages = FakeMessages(responses)


RETRIEVAL = {
    "results": [
        {
            "text": "Supply OPC 53 cement, 800 bags, delivery by 2026-08-10.",
            "page_number": 3,
            "source_file": "tender.pdf",
            "chunk_type": "text",
            "source_type": "tender",
            "distance": 0.21,
        }
    ],
    "low_confidence": False,
}


def test_doc_search_then_answer(monkeypatch: pytest.MonkeyPatch, db_session: Any) -> None:
    """One doc_search tool_use then an end_turn answer."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    fake_client = FakeClient(
        [
            _response(
                "tool_use",
                [_tool_use_block("toolu_01", "doc_search", {"query": "cement grade"})],
            ),
            _response(
                "end_turn",
                [_text_block("Per doc_search, the tender requires OPC 53 cement.")],
            ),
        ]
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client) as fake_ctor, mock.patch(
        "backend.pipeline.embedding.retrieve", return_value=RETRIEVAL
    ) as fake_retrieve:
        result = agent.run_agent("What cement grade is required?", "PRJ-2024-001", db_session)

    fake_ctor.assert_called_once()
    fake_retrieve.assert_called_once()
    assert fake_retrieve.call_args.kwargs["query"] == "cement grade"
    assert fake_retrieve.call_args.kwargs["project_id"] == "PRJ-2024-001"

    assert result["final"] is True
    assert result["answer"] == "Per doc_search, the tender requires OPC 53 cement."
    assert len(result["tool_calls"]) == 1
    entry = result["tool_calls"][0]
    assert entry["tool"] == "doc_search"
    assert entry["input"] == {"query": "cement grade"}
    assert "OPC 53" in entry["output_summary"]
    assert len(entry["output_summary"]) <= 200

    # Two API round-trips; the second carries the tool_result transcript.
    assert len(fake_client.messages.calls) == 2
    transcript = fake_client.messages.calls[1]["messages"]
    tool_result = transcript[-1]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_01"
    assert tool_result["is_error"] is False
    # Model config per contract.
    assert fake_client.messages.calls[0]["model"] == config.MODEL_SONNET
    # The API tool specs mirror agent.TOOLS (schemas pass through unchanged).
    assert fake_client.messages.calls[0]["tools"] == [
        {
            "name": tool["name"],
            "description": tool["description"],
            "input_schema": tool["input_schema"],
        }
        for tool in agent.TOOLS
    ]


def test_stops_at_tool_budget(monkeypatch: pytest.MonkeyPatch, db_session: Any) -> None:
    """A scripted infinite tool_use loop stops at AGENT_MAX_TOOL_CALLS."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    fake_client = FakeClient(
        [
            _response(
                "tool_use",
                [_tool_use_block("toolu_loop", "doc_search", {"query": "steel"})],
            )
        ]
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client), mock.patch(
        "backend.pipeline.embedding.retrieve",
        return_value={"results": [], "low_confidence": True},
    ) as fake_retrieve:
        result = agent.run_agent("Loop forever", "PRJ-2024-001", db_session)

    assert len(result["tool_calls"]) == config.AGENT_MAX_TOOL_CALLS
    assert fake_retrieve.call_count == config.AGENT_MAX_TOOL_CALLS
    assert result["final"] is False
    assert "budget" in result["answer"].lower()


def test_empty_api_key_fallback(monkeypatch: pytest.MonkeyPatch, db_session: Any) -> None:
    """Empty ANTHROPIC_API_KEY => contract fallback dict, no client constructed."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    with mock.patch("anthropic.Anthropic") as fake_ctor:
        result = agent.run_agent("Anything", "PRJ-2024-001", db_session)

    fake_ctor.assert_not_called()
    assert result["tool_calls"] == []
    assert result["final"] is False
    assert result["answer"].startswith("Agent unavailable: ANTHROPIC_API_KEY not set")


def test_tool_result_truncated_in_transcript(
    monkeypatch: pytest.MonkeyPatch, db_session: Any
) -> None:
    """Huge tool outputs are clipped to ~2000 chars in the transcript."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    huge = {"results": [{"text": "x" * 6000}], "low_confidence": False}
    fake_client = FakeClient(
        [
            _response(
                "tool_use",
                [_tool_use_block("toolu_big", "doc_search", {"query": "everything"})],
            ),
            _response("end_turn", [_text_block("done")]),
        ]
    )
    with mock.patch("anthropic.Anthropic", return_value=fake_client), mock.patch(
        "backend.pipeline.embedding.retrieve", return_value=huge
    ):
        result = agent.run_agent("Summarize everything", "PRJ-2024-001", db_session)

    transcript = fake_client.messages.calls[1]["messages"]
    content = transcript[-1]["content"][0]["content"]
    assert len(content) <= 2000 + len(" …[truncated]")
    assert content.endswith("…[truncated]")
    assert len(result["tool_calls"][0]["output_summary"]) == 200


def test_tool_schemas_hide_server_side_params() -> None:
    """8 tools, contract names, db_session/project_id never exposed."""
    names = sorted(tool["name"] for tool in agent.TOOLS)
    assert names == sorted(
        [
            "doc_search",
            "vendor_discovery",
            "vendor_evaluation",
            "compliance_checker",
            "risk_calculator",
            "recommend_vendor",
            "negotiate",
            "generate_po",
        ]
    )
    for tool in agent.TOOLS:
        assert tool["description"], tool["name"]
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert "db_session" not in schema["properties"]
        assert "project_id" not in schema["properties"]
