"""Council mode tests — write -> critique -> revise (model calls mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from council.collab import run_council, write_council_report
from council.compare import CompareReport
from council.config import CouncilConfig, SandboxConfig, VerifyStep
from council.executor import ExecutorResult, build_critique_prompt
from council.registry import load_registry


def _registry(tmp_path: Path, monkeypatch) -> object:
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([
        {"id": "free-a", "provider": "openrouter",
         "endpoint": "https://openrouter.ai/api/v1/chat/completions",
         "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
         "capabilities": ["code_generation"]},
        {"id": "free-b", "provider": "openrouter",
         "endpoint": "https://openrouter.ai/api/v1/chat/completions",
         "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
         "capabilities": ["code_generation"]},
    ]))
    return load_registry(p)


def _project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    return proj


def _config() -> CouncilConfig:
    return CouncilConfig(
        verify=[VerifyStep(name="compiles",
                           run="python -m py_compile game.py")],
        sandbox=SandboxConfig(image="python:3.11-slim"))


def test_critique_prompt_carries_code_and_notes():
    prompt = build_critique_prompt("write x", "game.py", "print(1)", "boom")
    assert "write x" in prompt
    assert "print(1)" in prompt
    assert "boom" in prompt


def test_critique_prompt_carries_human_notes():
    prompt = build_critique_prompt("write x", "game.py", "print(1)",
                                   human_notes="pieces fall too fast")
    assert "pieces fall too fast" in prompt
    assert "PLAYTESTED" in prompt


def test_council_revision_wins(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    models = [reg.get("free-a"), reg.get("free-b")]
    calls = {"n": 0}

    def fake_safe_call(model, prompt, timeout_seconds=180, workdir=None):
        calls["n"] += 1
        if "REVIEW a peer" in prompt:
            return ExecutorResult(model.id, "The code looks buggy, fix the loop.",
                                  5, 5, 0.5)
        if "peer model wrote" in prompt:
            # revise round: proper file
            return ExecutorResult(model.id, "```python\nprint('fixed')\n```",
                                  5, 5, 0.5)
        # write round: broken file
        return ExecutorResult(model.id, "```python\ndef broken(:\n```", 5, 5, 0.5)

    with mock.patch("council.collab._executor.safe_call", side_effect=fake_safe_call):
        report = run_council(models, "game.py", "write a game", proj,
                             _config(), "create-prompt")
    assert calls["n"] == 6  # 2 writes + 2 reviews + 2 revisions
    assert any(r.verdict == "PASS" for r in report.rows)
    assert all("(rev)" in r.model_id for r in report.rows)
    assert "phase1-free-a" in report.raw_responses
    assert "phase2-free-a" in report.raw_responses
    assert "review-free-a-on-free-b" in report.raw_responses
    assert "review-free-b-on-free-a" in report.raw_responses


def test_council_passes_human_notes_to_revise_round(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    seen_prompts = []

    def fake_safe_call(model, prompt, timeout_seconds=180, workdir=None):
        seen_prompts.append(prompt)
        if "REVIEW a peer" in prompt:
            return ExecutorResult(model.id, "fine, keep it.", 1, 1, 0.1)
        return ExecutorResult(model.id, "```python\nx = 1\n```", 1, 1, 0.1)

    with mock.patch("council.collab._executor.safe_call", side_effect=fake_safe_call):
        run_council([reg.get("free-a")], "game.py", "write", proj,
                    _config(), "p", human_notes="too slow, add levels")
    revise_prompts = [p for p in seen_prompts if "peer model wrote" in p]
    assert revise_prompts
    assert all("too slow, add levels" in p for p in revise_prompts)


def test_council_stops_when_nothing_to_build_on(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)

    def fake_safe_call(model, prompt, timeout_seconds=180, workdir=None):
        return ExecutorResult(model.id, "", error="429 nope")

    with mock.patch("council.collab._executor.safe_call", side_effect=fake_safe_call):
        report = run_council([reg.get("free-a")], "game.py", "write", proj,
                             _config(), "p")
    assert report.rows[0].verdict == "CALL_FAIL"
    assert "phase2" not in "".join(report.raw_responses)


def test_write_council_report_goes_to_councils(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)

    def fake_safe_call(model, prompt, timeout_seconds=180, workdir=None):
        if "REVIEW a peer" in prompt:
            return ExecutorResult(model.id, "looks good.", 1, 1, 0.1)
        return ExecutorResult(model.id, "```python\nx = 1\n```", 1, 1, 0.1)

    with mock.patch("council.collab._executor.safe_call", side_effect=fake_safe_call):
        report = run_council([reg.get("free-a")], "game.py", "write", proj,
                             _config(), "p")
    outdir = write_council_report(proj, report)
    assert outdir.parent.name == "councils"
    assert (outdir / "summary.md").exists()


def test_revise_round_feeds_peer_reviews(tmp_path, monkeypatch):
    from council.collab import run_revise_round
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    seen = []

    def fake_safe_call(model, prompt, timeout_seconds=180, workdir=None):
        seen.append(prompt)
        return ExecutorResult(model.id, "```python\ny = 2\n```", 1, 1, 0.1)

    models = [reg.get("free-a")]
    with mock.patch("council.collab._executor.safe_call", side_effect=fake_safe_call):
        rows, new_codes = run_revise_round(
            models, {"free-a": "x = 1"}, {"free-a": ["fix the loop"]},
            "game.py", "write", proj, _config(), round_tag="r1")
    assert rows[0].verdict == "PASS"
    assert new_codes["free-a"] == "y = 2"
    assert any("fix the loop" in p for p in seen)


def test_max_reviewers_caps_critics(tmp_path, monkeypatch):
    from council.collab import run_council
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([
        {"id": f"m{i}", "provider": "openrouter",
         "endpoint": "https://openrouter.ai/api/v1/chat/completions",
         "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
         "capabilities": ["code_generation"],
         "quality_score": {"value": 0.1 * i, "source": "user_override"}}
        for i in range(4)
    ]))
    reg = load_registry(p)
    proj = _project(tmp_path)
    models = [reg.get(f"m{i}") for i in range(4)]
    review_calls = []

    def fake(model, prompt, timeout_seconds=180, workdir=None):
        if "REVIEW a peer" in prompt:
            review_calls.append(model.id)
            return ExecutorResult(model.id, "1. x — y — fix it.", 1, 1, 0.1)
        return ExecutorResult(model.id, "```python\nx = 1\n```", 1, 1, 0.1)

    with mock.patch("council.collab._executor.safe_call", side_effect=fake):
        run_council(models, "game.py", "write", proj, _config(), "p",
                    max_reviewers=1)
    # 4 authors x 1 reviewer each
    assert len(review_calls) == 4
