"""Verifier tests — the 5 cases from the README."""
from __future__ import annotations

from pathlib import Path

from council.config import CouncilConfig, SandboxConfig, VerifyStep
from council.verifier import run_verifier


def _config(*steps: VerifyStep, timeout: int = 30) -> CouncilConfig:
    return CouncilConfig(verify=list(steps), timeout_seconds=timeout,
                         sandbox=SandboxConfig(image="python:3.11-slim"))


def test_all_steps_passing(tmp_path: Path):
    cfg = _config(VerifyStep(name="a", run="python -c \"import sys; sys.exit(0)\""),
                  VerifyStep(name="b", run="python -c \"import sys; sys.exit(0)\""))
    report = run_verifier(cfg, tmp_path)
    assert report.overall_pass is True
    assert report.verify_error is False
    assert all(s.passed for s in report.steps)


def test_required_failure_blocks_rest(tmp_path: Path):
    cfg = _config(VerifyStep(name="fail", run="python -c \"import sys; sys.exit(1)\""),
                  VerifyStep(name="never", run="python -c \"import sys; sys.exit(0)\""))
    report = run_verifier(cfg, tmp_path)
    assert report.overall_pass is False
    assert report.verify_error is False
    assert [s.name for s in report.steps] == ["fail"]
    assert report.failing_step().name == "fail"


def test_non_required_failure_does_not_block(tmp_path: Path):
    cfg = _config(
        VerifyStep(name="style", run="python -c \"import sys; sys.exit(1)\"",
                   required=False),
        VerifyStep(name="test", run="python -c \"import sys; sys.exit(0)\""))
    report = run_verifier(cfg, tmp_path)
    assert report.overall_pass is True
    assert [s.name for s in report.steps] == ["style", "test"]


def test_missing_command_is_verify_error(tmp_path: Path):
    cfg = _config(VerifyStep(name="gone",
                             run="definitely-not-a-real-command-xyz-123"))
    report = run_verifier(cfg, tmp_path)
    assert report.overall_pass is False
    assert report.verify_error is True
    assert report.steps[0].error is not None


def test_timeout_is_verify_error(tmp_path: Path):
    cfg = _config(VerifyStep(name="slow",
                             run="python -c \"import time; time.sleep(30)\"",
                             timeout_seconds=1))
    report = run_verifier(cfg, tmp_path, )
    assert report.overall_pass is False
    assert report.verify_error is True
    assert report.steps[0].timed_out is True
