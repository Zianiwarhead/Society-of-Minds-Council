"""Tests for `council run`'s model-calling loop (all model I/O mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from council.cli import _run_model_attempt, main
from council.executor import ExecutorResult
from council.task import build_task


def _setup_proj(tmp_path: Path, monkeypatch, verify_run: str,
                target_body: str = "def add(a, b):\n    return a - b\n") -> Path:
    """A project with one buggy file, one verifier step, two free models."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".council.yaml").write_text(
        f"verify:\n  - name: \"check\"\n    run: '{verify_run}'\n    required: true\n"
        "sandbox:\n  image: \"python:3.11-slim\"\n")
    (proj / "models.yaml").write_text(yaml.safe_dump([
        {"id": "free-a", "provider": "openrouter",
         "endpoint": "https://openrouter.ai/api/v1/chat/completions",
         "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
         "capabilities": ["code_generation"],
         "quality_score": {"value": 0.7, "source": "user_override"}},
        {"id": "free-b", "provider": "openrouter",
         "endpoint": "https://openrouter.ai/api/v1/chat/completions",
         "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
         "capabilities": ["code_generation"],
         "quality_score": {"value": 0.5, "source": "user_override"}},
    ]))
    (proj / "calc.py").write_text(target_body)
    return proj


def _task(proj: Path):
    return build_task(path="calc.py", diff_ref=None, description="fix add",
                      verifier_command="check", cwd=proj)


def _config(proj: Path):
    from council.config import load_config
    return load_config(proj)


def test_attempt_pass(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(0)\"")
    from council.registry import load_registry
    reg = load_registry(proj / "models.yaml")
    good = "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n```"

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, good, 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        attempt, report, diff, err = _run_model_attempt(
            reg.get("free-a"), _task(proj), proj, _config(proj), 30)
    assert attempt.passed is True
    assert diff is not None and "return a + b" in diff
    assert err is None


def test_attempt_no_diff_is_fail_not_verify_error(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(0)\"")
    from council.registry import load_registry
    reg = load_registry(proj / "models.yaml")

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, "looks fine, ship it", 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        attempt, report, diff, err = _run_model_attempt(
            reg.get("free-a"), _task(proj), proj, _config(proj), 30)
    assert attempt.passed is False
    assert attempt.verify_error is False  # must escalate, not short-circuit
    assert diff is None


def test_attempt_unappliable_diff_is_fail_not_verify_error(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(0)\"")
    from council.registry import load_registry
    reg = load_registry(proj / "models.yaml")
    garbage = "```diff\n--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n```"

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, garbage, 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        attempt, report, diff, err = _run_model_attempt(
            reg.get("free-a"), _task(proj), proj, _config(proj), 30)
    assert attempt.passed is False
    assert attempt.verify_error is False
    assert diff is not None  # preserved for the review record


def test_attempt_call_failure_is_fail_not_verify_error(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(0)\"")
    from council.registry import load_registry
    reg = load_registry(proj / "models.yaml")

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, "", error="429 rate limited")

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        attempt, report, diff, err = _run_model_attempt(
            reg.get("free-a"), _task(proj), proj, _config(proj), 30)
    assert attempt.passed is False
    assert attempt.verify_error is False


def test_run_as_is_pass_calls_no_model(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(0)\"")
    calls = []

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        calls.append(model.id)
        return ExecutorResult(model.id, "x", 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        code = main(["run", "calc.py", "--task", "fix add",
                     "--project-dir", str(proj), "--registry", str(proj / "models.yaml")])
    assert code == 0
    assert calls == []


def test_run_loop_fix_then_done_leaves_tree_alone(tmp_path, monkeypatch):
    check = ("python -c \"from calc import add; import sys; "
             "sys.exit(0 if add(2, 3) == 5 else 1)\"")
    proj = _setup_proj(tmp_path, monkeypatch, check)
    good = "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n```"

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, good, 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        code = main(["run", "calc.py", "--task", "fix add",
                     "--project-dir", str(proj), "--registry", str(proj / "models.yaml")])
    assert code == 0
    assert (proj / "calc.py").read_text() == "def add(a, b):\n    return a - b\n"
    runs = list((proj / ".council" / "runs").glob("*/final.diff"))
    assert len(runs) == 1
    assert "return a + b" in runs[0].read_text(encoding="utf-8")


def test_run_loop_apply_writes_tree(tmp_path, monkeypatch):
    check = ("python -c \"from calc import add; import sys; "
             "sys.exit(0 if add(2, 3) == 5 else 1)\"")
    proj = _setup_proj(tmp_path, monkeypatch, check)
    good = "```diff\n--- a/calc.py\n+++ b/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n```"

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, good, 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        code = main(["run", "calc.py", "--task", "fix add", "--apply",
                     "--project-dir", str(proj), "--registry", str(proj / "models.yaml")])
    assert code == 0
    assert "return a + b" in (proj / "calc.py").read_text()


def test_run_loop_exhaustion_is_human_review(tmp_path, monkeypatch):
    proj = _setup_proj(tmp_path, monkeypatch, "python -c \"import sys; sys.exit(1)\"")
    garbage = "```diff\n--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n```"

    def fake(model, prompt, timeout_seconds=120, workdir=None):
        return ExecutorResult(model.id, garbage, 1, 1, 0.1)

    with mock.patch("council.cli._executor.safe_call", side_effect=fake):
        code = main(["run", "calc.py", "--task", "fix add",
                     "--project-dir", str(proj), "--registry", str(proj / "models.yaml")])
    # 2 free FAILs -> forced paid escalation -> no paid model -> HUMAN_REVIEW
    assert code == 4
    reviews = list((proj / ".council" / "reviews").glob("*/summary.md"))
    assert len(reviews) == 1
    diffs = list((proj / ".council" / "reviews").glob("*/attempt-*.diff"))
    assert len(diffs) == 2
