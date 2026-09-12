"""Compare harness tests — verdicts + isolation (model calls mocked)."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import yaml

from council.compare import (
    _apply_diff,
    _normalize_diff_paths,
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


def test_apply_diff_tolerates_wrong_hunk_counts(tmp_path):
    # Live evidence 2026-09-12: models miscount `@@ -a,b +c,d @@` headers.
    # --recount must save the apply when the content itself is right.
    workdir = tmp_path / "w"
    workdir.mkdir()
    (workdir / "a.py").write_text("print(1)\nprint(2)\n", encoding="utf-8")
    bad_counts = ("--- a/a.py\n+++ b/a.py\n"
                  "@@ -1,99 +1,99 @@\n-print(1)\n+print(9)\n print(2)\n")
    assert _apply_diff(workdir, bad_counts) is None
    assert (workdir / "a.py").read_text(encoding="utf-8") == "print(9)\nprint(2)\n"


def test_apply_diff_rejects_content_mismatch(tmp_path):
    # Live evidence 2026-09-12 (nemotron): dedented `-` lines that don't
    # match the file must stay FAIL, not be fuzzed into place.
    workdir = tmp_path / "w"
    workdir.mkdir()
    (workdir / "a.py").write_text('    """indented"""\n', encoding="utf-8")
    hallucinated = ('--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-"""indented"""\n+"""changed"""\n')
    assert _apply_diff(workdir, hallucinated) is not None
    assert (workdir / "a.py").read_text(encoding="utf-8") == '    """indented"""\n'


def test_normalize_diff_paths_fixes_backslashes():
    # Live evidence 2026-09-12 (lfm): `--- a/council\\task.py`.
    raw = "--- a/council\\task.py\n+++ b/council\\task.py\n@@ -1 +1 @@\n-x\n+x\n"
    fixed = _normalize_diff_paths(raw)
    assert "--- a/council/task.py" in fixed
    assert "+++ b/council/task.py" in fixed
    # Body backslashes (e.g. in code) are untouched.
    body = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-print(\"a\\nb\")\n+print(1)\n"
    assert _normalize_diff_paths(body) == body.rstrip("\n")


def test_write_compare_report_is_utf8(tmp_path, monkeypatch):
    # Live evidence 2026-09-12: a cp1252 byte (0x96) landed in a report
    # file because writes used the platform default encoding on Windows.
    reg = _registry(tmp_path, monkeypatch)
    proj = _project(tmp_path)

    def fake_safe_call(model, prompt, timeout_seconds=120):
        return ExecutorResult(model.id, "em dash: \u2014 and en dash: \u2013",
                              1, 1, 0.1)

    with mock.patch("council.compare._executor.safe_call", side_effect=fake_safe_call):
        report = run_compare([reg.get("free-a")], _task(proj), proj, _config(), "p")
    outdir = write_compare_report(proj, report)
    raw = (outdir / "free-a.md").read_bytes()
    assert raw.decode("utf-8") == "em dash: \u2014 and en dash: \u2013"
