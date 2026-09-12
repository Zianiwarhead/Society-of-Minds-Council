"""Executor tests — prompt/diff helpers + client behavior (mocked HTTP)."""
from __future__ import annotations

import io
import json
import urllib.error
from unittest import mock

import pytest

from council import executor as ex
from council.executor import (
    ExecutorError,
    build_prompt,
    extract_diff,
    safe_call,
)
from council.registry import ModelEntry


def _model(**overrides) -> ModelEntry:
    base = dict(
        id="x/model", provider="openrouter",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key_env="TEST_COUNCIL_KEY", cost_tier="free",
        capabilities=["code_generation"],
    )
    base.update(overrides)
    return ModelEntry(**base)


def test_build_prompt_contains_task_and_files():
    prompt = build_prompt("fix it", {"a.py": "print(1)"})
    assert "fix it" in prompt
    assert "a.py" in prompt
    assert "unified diff" in prompt


def test_extract_diff_fenced_block():
    text = "here you go\n```diff\n--- a/x\n+++ b/x\n@@\n-old\n+new\n```\ndone"
    assert extract_diff(text) == "--- a/x\n+++ b/x\n@@\n-old\n+new"


def test_extract_diff_bare_diff():
    text = "--- a/x\n+++ b/x\n@@\n-old\n+new\n"
    assert extract_diff(text) is not None


def test_extract_diff_none_for_chat():
    assert extract_diff("I think the bug is on line 3, looks fine otherwise.") is None


def _http_response(payload: dict):
    data = json.dumps(payload).encode()
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    resp.read.return_value = data
    # json.load(resp) needs a file-like; patch json.load instead
    return resp


def test_call_model_success(monkeypatch):
    monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
    payload = {"choices": [{"message": {"content": "```diff\n--- a\n```"}}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    with mock.patch.object(ex.urllib.request, "urlopen") as urlopen, \
         mock.patch.object(ex.json, "load", return_value=payload):
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        urlopen.return_value = resp
        result = ex.call_model(_model(), "do it", timeout_seconds=5)
    assert result.ok
    assert "diff" in result.text
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.estimated_cost_usd(3.0, 15.0) == pytest.approx(10 / 1000 * 3.0 + 5 / 1000 * 15.0)


def test_call_model_missing_key(monkeypatch):
    monkeypatch.delenv("TEST_COUNCIL_KEY", raising=False)
    with pytest.raises(ExecutorError, match="TEST_COUNCIL_KEY"):
        ex.call_model(_model(), "do it")


def test_call_model_http_error(monkeypatch):
    monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
    err = urllib.error.HTTPError(
        "https://x", 429, "Too Many Requests", {}, io.BytesIO(b"rate limited"))
    with mock.patch.object(ex.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(ExecutorError, match="429"):
            ex.call_model(_model(), "do it")


def test_safe_call_returns_failed_result(monkeypatch):
    monkeypatch.delenv("TEST_COUNCIL_KEY", raising=False)
    result = safe_call(_model(), "do it")
    assert not result.ok
    assert result.error is not None
