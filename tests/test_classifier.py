"""Classifier tests — the 7 scenarios from the README."""
from __future__ import annotations

import pytest
import yaml

from council.classifier import (
    Attempt,
    NoEligibleModelError,
    TaskState,
    classify_complexity,
    complexity_score,
    decide_next,
    required_capabilities,
    select_initial_model,
)
from council.registry import load_registry
from council.task import Task


def _task(**overrides) -> Task:
    base = dict(
        id="t", description="add input validation", target_paths=["a.py"],
        diff=None, files_touched=1, lines_changed=0,
        cross_file_dependencies=False, is_retry=False, retry_count=0,
        verifier_command="pytest", created_at="2026-09-12T10:00:00Z",
    )
    base.update(overrides)
    return Task(**base)


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([
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
        {"id": "paid-c", "provider": "anthropic",
         "endpoint": "https://api.anthropic.com/v1/messages",
         "api_key_env": "ANTHROPIC_API_KEY", "cost_tier": "paid",
         "capabilities": ["code_generation", "cross_file_architecture",
                          "security_audit"]},
    ]))
    return load_registry(p)


def test_simple_task_routes_free(registry):
    model = select_initial_model(registry, _task())
    assert model.cost_tier == "free"
    assert model.id == "free-a"  # higher quality wins


def test_complex_security_task_skips_to_paid(registry):
    t = _task(description="fix auth security vulnerability",
              files_touched=8, lines_changed=400,
              cross_file_dependencies=True)
    assert classify_complexity(t) == "complex"
    assert "security_audit" in required_capabilities(t)
    model = select_initial_model(registry, t)
    assert model.cost_tier == "paid"


def test_single_free_failure_retries_other_free(registry):
    t = _task()
    nxt = decide_next(t, registry, [Attempt("free-a", passed=False)])
    assert nxt.state == TaskState.RUNNING
    assert nxt.model.id == "free-b"


def test_two_free_failures_force_paid_escalation(registry):
    t = _task()
    nxt = decide_next(t, registry, [
        Attempt("free-a", passed=False), Attempt("free-b", passed=False)])
    assert nxt.state == TaskState.ESCALATED
    assert nxt.model.cost_tier == "paid"


def test_paid_failure_goes_human_review(registry):
    t = _task()
    nxt = decide_next(t, registry, [Attempt("paid-c", passed=False)])
    assert nxt.state == TaskState.HUMAN_REVIEW
    assert nxt.model is None


def test_verify_error_short_circuits(registry):
    t = _task()
    nxt = decide_next(t, registry, [Attempt("free-a", passed=False, verify_error=True)])
    assert nxt.state == TaskState.VERIFY_ERROR


def test_pass_goes_done(registry):
    t = _task()
    nxt = decide_next(t, registry, [Attempt("free-a", passed=True)])
    assert nxt.state == TaskState.DONE


def test_no_eligible_model_raises(registry):
    t = _task(description="encrypt the auth flow")  # needs security_audit
    # free tier has no security_audit-capable model
    with pytest.raises(NoEligibleModelError):
        from council.classifier import select_model
        select_model(registry, ["security_audit", "nonexistent-cap"], tier="free")


def test_complexity_buckets():
    assert classify_complexity(_task()) == "simple"
    assert classify_complexity(_task(files_touched=5, lines_changed=200,
                                     cross_file_dependencies=True)) == "moderate"
    assert complexity_score(_task(is_retry=True, cross_file_dependencies=True,
                                  files_touched=10, lines_changed=500)) >= 50
