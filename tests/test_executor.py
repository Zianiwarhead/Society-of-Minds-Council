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
    build_create_prompt,
    build_critique_prompt,
    build_prompt,
    build_review_prompt,
    extract_code_block,
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
    assert result.latency_seconds >= 0


def test_transient_oserror_retried_once(monkeypatch):
    monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
    payload = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
    with mock.patch.object(ex.urllib.request, "urlopen") as urlopen, \
         mock.patch.object(ex.json, "load", return_value=payload), \
         mock.patch.object(ex.time, "sleep") as sleep:
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        urlopen.side_effect = [OSError("conn reset"), resp]
        result = ex.call_model(_model(), "do it", timeout_seconds=5)
    assert result.ok
    assert urlopen.call_count == 2
    sleep.assert_called_once()


def test_non_transient_http_error_not_retried(monkeypatch):
    monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
    err = urllib.error.HTTPError(
        "https://x", 401, "Unauthorized", {}, io.BytesIO(b"bad key"))
    with mock.patch.object(ex.urllib.request, "urlopen", side_effect=err) as urlopen:
        with pytest.raises(ExecutorError, match="401"):
            ex.call_model(_model(), "do it")
        assert urlopen.call_count == 1


def test_unexpected_shape_includes_preview(monkeypatch):
    monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
    payload = {"error": {"message": "model overloaded"}}
    with mock.patch.object(ex.urllib.request, "urlopen") as urlopen, \
         mock.patch.object(ex.json, "load", return_value=payload):
        resp = mock.MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        urlopen.return_value = resp
        with pytest.raises(ExecutorError, match="model overloaded"):
            ex.call_model(_model(), "do it")


def test_build_create_prompt_names_file_and_task():
    prompt = build_create_prompt("write tetris", "game.py")
    assert "write tetris" in prompt
    assert "game.py" in prompt
    assert "standard library" in prompt


def test_extract_code_block_prefers_python():
    text = "here:\n```python\nprint(1)\n```\ndone"
    assert extract_code_block(text) == "print(1)"


def test_extract_code_block_falls_back_to_plain_fence():
    text = "here:\n```\nx = 1\n```"
    assert extract_code_block(text) == "x = 1"


def test_extract_code_block_none_for_chat():
    assert extract_code_block("Tetris is a fun game with blocks!") is None


def test_build_review_prompt_names_peer_and_code():
    prompt = build_review_prompt("write x", "game.py", "print(1)", "free-b")
    assert "REVIEW a peer" in prompt
    assert "free-b" in prompt
    assert "print(1)" in prompt
    assert "numbered" in prompt
    assert "corrections only" in prompt


def test_build_critique_prompt_carries_peer_reviews():
    from council.executor import build_critique_prompt
    prompt = build_critique_prompt("t", "g.py", "code", peer_reviews="fix loop")
    assert "PEER REVIEW SAID" in prompt
    assert "fix loop" in prompt


def test_build_critique_prompt_demands_disposition():
    from council.executor import build_critique_prompt
    prompt = build_critique_prompt("t", "g.py", "code")
    assert "FIXED" in prompt and "KEPT" in prompt


def _opencode_model(**overrides) -> ModelEntry:
    base = dict(
        id="ocMind", provider="opencode", endpoint="anthropic/claude-sonnet-4-5",
        api_key_env="none", cost_tier="free", capabilities=["code_generation"],
        backend="opencode",
    )
    base.update(overrides)
    return ModelEntry(**base)


def test_opencode_routes_by_backend(monkeypatch):
    from council import executor as ex_mod
    monkeypatch.setattr(ex_mod.shutil, "which", lambda _: "/usr/bin/opencode")
    with mock.patch.object(ex_mod.subprocess, "run") as run:
        proc = mock.MagicMock()
        proc.returncode = 0
        proc.stdout = '```python\nprint(1)\n```'
        proc.stderr = ""
        run.return_value = proc
        result = ex_mod.call_model(_opencode_model(), "write x", timeout_seconds=5)
    assert result.ok
    assert "print(1)" in result.text
    cmd = run.call_args[0][0]
    assert cmd[:4] == ["opencode", "run", "--model", "anthropic/claude-sonnet-4-5"]


def test_opencode_missing_binary(monkeypatch):
    from council import executor as ex_mod
    monkeypatch.setattr(ex_mod.shutil, "which", lambda _: None)
    with pytest.raises(ExecutorError, match="opencode CLI on PATH"):
        ex_mod.call_model(_opencode_model(), "write x")


def test_opencode_nonzero_exit_is_error(monkeypatch):
    from council import executor as ex_mod
    monkeypatch.setattr(ex_mod.shutil, "which", lambda _: "/usr/bin/opencode")
    with mock.patch.object(ex_mod.subprocess, "run") as run:
        proc = mock.MagicMock()
        proc.returncode = 1
        proc.stdout = ""
        proc.stderr = "Error: rate limited"
        run.return_value = proc
        with pytest.raises(ExecutorError, match="rate limited"):
            ex_mod.call_model(_opencode_model(), "write x")


def test_parse_opencode_json_events():
    from council.executor import _parse_opencode_output
    events = '\n'.join([
        json.dumps({"type": "step", "text": "thinking out loud here okay!"}),
        json.dumps({"type": "text", "text": "```python\nprint(1)\n```"}),
    ])
    out = _parse_opencode_output(events)
    assert "print(1)" in out
    # raw non-JSON falls through untouched
    assert _parse_opencode_output("```python\nx=1\n```") == "```python\nx=1\n```"
    assert _parse_opencode_output("   ") == ""


def _backend_model(backend, **overrides) -> ModelEntry:
    base = dict(
        id="x/model", provider="test",
        endpoint="https://example.com/v1/chat/completions",
        api_key_env="TEST_COUNCIL_KEY", cost_tier="free",
        capabilities=["code_generation"], backend=backend,
    )
    base.update(overrides)
    return ModelEntry(**base)


def _mock_http(monkeypatch, payload):
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        from council import executor as ex_mod
        monkeypatch.setenv("TEST_COUNCIL_KEY", "k")
        with mock.patch.object(ex_mod.urllib.request, "urlopen") as urlopen, \
             mock.patch.object(ex_mod.json, "load", return_value=payload):
            resp = mock.MagicMock()
            resp.__enter__.return_value = resp
            resp.__exit__.return_value = False
            urlopen.return_value = resp
            yield urlopen
    return _ctx()


def test_openai_compatible_backend_uses_entry_endpoint(monkeypatch):
    from council import executor as ex_mod
    payload = {"choices": [{"message": {"content": "hi"}}], "usage": {}}
    with _mock_http(monkeypatch, payload) as urlopen:
        result = ex_mod.call_model(
            _backend_model("openai_compatible",
                           endpoint="https://api.groq.com/openai/v1/chat/completions",
                           id="llama-3.3-70b-versatile"),
            "do it", timeout_seconds=5)
    assert result.ok
    assert result.text == "hi"
    req = urlopen.call_args[0][0]
    assert req.full_url == "https://api.groq.com/openai/v1/chat/completions"
    sent = json.loads(req.data.decode())
    assert sent["model"] == "llama-3.3-70b-versatile"


def test_anthropic_backend_headers_and_parse(monkeypatch):
    from council import executor as ex_mod
    payload = {"content": [{"type": "text", "text": "hello"}],
               "usage": {"input_tokens": 3, "output_tokens": 2}}
    with _mock_http(monkeypatch, payload) as urlopen:
        result = ex_mod.call_model(
            _backend_model("anthropic",
                           endpoint="https://api.anthropic.com/v1/messages",
                           id="claude-sonnet-4-5"),
            "do it", timeout_seconds=5)
    assert result.ok
    assert result.text == "hello"
    assert (result.prompt_tokens, result.completion_tokens) == (3, 2)
    req = urlopen.call_args[0][0]
    assert req.get_header("X-api-key") == "k"
    assert req.get_header("Anthropic-version") == "2023-06-01"
    sent = json.loads(req.data.decode())
    assert sent["model"] == "claude-sonnet-4-5"
    assert sent["max_tokens"] == 4096


def test_google_backend_url_and_parse(monkeypatch):
    from council import executor as ex_mod
    payload = {"candidates": [{"content": {"parts": [{"text": "yo"}]}}],
               "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1}}
    with _mock_http(monkeypatch, payload) as urlopen:
        result = ex_mod.call_model(
            _backend_model("google",
                           endpoint="https://generativelanguage.googleapis.com/v1beta/models",
                           id="gemini-2.0-flash"),
            "do it", timeout_seconds=5)
    assert result.ok
    assert result.text == "yo"
    assert (result.prompt_tokens, result.completion_tokens) == (4, 1)
    req = urlopen.call_args[0][0]
    assert req.full_url.endswith("/gemini-2.0-flash:generateContent")
    assert req.get_header("X-goog-api-key") == "k"
    assert "key=" not in req.full_url  # key in header, never the URL


def test_unknown_backend_id_rejected_by_registry(tmp_path):
    # belt-and-braces: registry validation is the gate, executor trusts it
    import yaml
    from council.registry import RegistryError, load_registry
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([{
        "id": "m", "provider": "x", "endpoint": "https://x",
        "api_key_env": "K", "cost_tier": "free",
        "capabilities": ["code_generation"], "backend": "telepathy"}]))
    import pytest as _pytest
    with _pytest.raises(RegistryError, match="backend"):
        load_registry(p)
