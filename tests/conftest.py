"""Shared test fixtures.

The developer's real ``.env`` may configure a live LLM provider (e.g. a
GEMINI_API_KEY). Tests must stay hermetic — provider selection is driven only
by what each test monkeypatches, so ambient keys/overrides are cleared here.
"""
from __future__ import annotations

import pytest

from backend import config


@pytest.fixture(autouse=True)
def _no_ambient_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GEMINI_API_KEY", "", raising=False)
    monkeypatch.setattr(config, "LLM_PROVIDER", "", raising=False)
