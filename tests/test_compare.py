"""Compare harness tests — verdicts + isolation (model calls mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from council.compare import (
    collect_context,
    run_compare,
    write_compare_report,
)
from council.config import CouncilConfig, SandboxConfig, VerifyStep
from council.executor import ExecutorResult
from council.registry import load_registry
from council.task import Task


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
    (proj / ".council.yaml").write_text(
        "verify:\n"
        "  - name: \"check\"\n"
        "    run: \"python -c \\\"import sys; sys.exit(0)\\\"\"\n"
        "    required: true\n"
        "sandbox:\n"
        "  image: \"python:3.11-slim\"\n")
    (proj / "a.py").write_text("print(1)\n")
    return proj


def _task(proj: Path) -> Task:
    return Task(id="t", description="do it", target_paths=[str(proj / "a.py")],
                diff=None, files_touched=1, lines_changed=0,
                cross_file_dependencies=False, is_retry=False, retry_count=0,
                verifier_command="check", created_at="2026-09-12T10:00:00Z")


def _config() -> CouncilConfig:
    return CouncilConfig(
        verify=[VerifyStep(name="check",
                           run="python -c \"import sys; sys.exit(0)\"")],
        sandbox=SandboxConfig(image="python:3.11-slim"))


def test_collect_context_reads_targets(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    ctx = collect_context(_task(proj), proj)
    assert "a.py" in ctx
    assert "print(1)" in ctx["a.py"]


def test_compare_pass_and_no_diff(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    models = [reg.get("free-a"), reg.get("free-b")]
    good_diff = ("```diff\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-print(1)\n+print(2)\n```")

    def fake_safe_call(model, prompt, timeout_seconds=120):
        if model.id == "free-a":
            return ExecutorResult(model.id, good_diff, 10, 5, 0.5)
        return ExecutorResult(model.id, "looks fine to me, no changes needed", 10, 5, 0.4)

    with mock.patch("council.compare._executor.safe_call", side_effect=fake_safe_call):
        report = run_compare(models, _task(proj), proj, _config(), "prompt")
    by_id = {r.model_id: r for r in report.rows}
    assert by_id["free-a"].verdict == "PASS"
    assert by_id["free-b"].verdict == "NO_DIFF"
    assert set(report.raw_responses) == {"free-a", "free-b"}


def test_compare_call_failure_recorded(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)

    def fake_safe_call(model, prompt, timeout_seconds=120):
        return ExecutorResult(model.id, "", error="boom: 429 rate limited")

    with mock.patch("council.compare._executor.safe_call", side_effect=fake_safe_call):
        report = run_compare([reg.get("free-a")], _task(proj), proj, _config(), "p")
    assert report.rows[0].verdict == "CALL_FAIL"


def test_compare_does_not_touch_working_tree(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)
    before = (proj / "a.py").read_text()
    good_diff = ("```diff\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-print(1)\n+print(2)\n```")

    def fake_safe_call(model, prompt, timeout_seconds=120):
        return ExecutorResult(model.id, good_diff, 1, 1, 0.1)

    with mock.patch("council.compare._executor.safe_call", side_effect=fake_safe_call):
        run_compare([reg.get("free-a")], _task(proj), proj, _config(), "p")
    assert (proj / "a.py").read_text() == before


def test_write_compare_report(tmp_path, monkeypatch):
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)

    def fake_safe_call(model, prompt, timeout_seconds=120):
        return ExecutorResult(model.id, "no diff here", 1, 1, 0.1)

    with mock.patch("council.compare._executor.safe_call", side_effect=fake_safe_call):
        report = run_compare([reg.get("free-a")], _task(proj), proj, _config(), "p")
    outdir = write_compare_report(proj, report)
    assert (outdir / "summary.md").exists()
    assert (outdir / "summary.json").exists()
    assert (outdir / "free-a.md").exists()
