"""Registry loader tests — validation + query primitives."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from council.registry import (
    ModelRegistry,
    RegistryError,
    load_registry,
)


def _write_models(tmp_path: Path, entries: list) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(entries))
    return p


def _entry(model_id: str, **overrides) -> dict:
    base = {
        "id": model_id,
        "provider": "openrouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY",
        "cost_tier": "free",
        "capabilities": ["code_generation"],
    }
    base.update(overrides)
    return base


def test_load_valid_registry(tmp_path):
    p = _write_models(tmp_path, [
        _entry("free-a", quality_score={"value": 0.7, "source": "user_override"}),
        _entry("paid-b", cost_tier="paid", api_key_env="OTHER_KEY",
               capabilities=["code_generation", "security_audit"],
               escalates_to=None),
    ])
    reg = load_registry(p)
    assert len(reg.all()) == 2
    assert reg.get("free-a").quality_score.value == 0.7


def test_duplicate_ids_rejected(tmp_path):
    p = _write_models(tmp_path, [_entry("dup"), _entry("dup")])
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(p)


def test_invalid_cost_tier_rejected(tmp_path):
    p = _write_models(tmp_path, [_entry("m", cost_tier="freemium")])
    with pytest.raises(RegistryError, match="cost_tier"):
        load_registry(p)


def test_scraper_quality_source_rejected(tmp_path):
    p = _write_models(tmp_path, [
        _entry("m", quality_score={"value": 0.9, "source": "scraper"})
    ])
    with pytest.raises(RegistryError, match="quality_score.source"):
        load_registry(p)


def test_dangling_escalates_to_rejected(tmp_path):
    p = _write_models(tmp_path, [_entry("m", escalates_to="ghost")])
    with pytest.raises(RegistryError, match="escalates_to"):
        load_registry(p)


def test_escalation_cycle_rejected(tmp_path):
    p = _write_models(tmp_path, [
        _entry("a", escalates_to="b"),
        _entry("b", escalates_to="a"),
    ])
    with pytest.raises(RegistryError, match="cycle"):
        load_registry(p)


def test_eligible_for_respects_caps_and_exclusions(tmp_path):
    p = _write_models(tmp_path, [
        _entry("coder"),
        _entry("guarded", not_trusted_for=["security_audit"],
               capabilities=["code_generation", "security_audit"]),
        _entry("auditor", capabilities=["code_generation", "security_audit"]),
    ])
    reg = load_registry(p)
    eligible = reg.eligible_for(["code_generation", "security_audit"])
    assert {m.id for m in eligible} == {"auditor"}
    free_only = reg.eligible_for(["code_generation"], cost_tier="free")
    assert {m.id for m in free_only} == {"coder", "guarded", "auditor"}


def test_available_and_missing_keys(monkeypatch, tmp_path):
    p = _write_models(tmp_path, [
        _entry("a", api_key_env="KEY_A_SET"),
        _entry("b", api_key_env="KEY_B_MISSING"),
    ])
    monkeypatch.setenv("KEY_A_SET", "x")
    monkeypatch.delenv("KEY_B_MISSING", raising=False)
    reg = load_registry(p)
    assert [m.id for m in reg.available()] == ["a"]
    assert reg.missing_keys() == ["KEY_B_MISSING"]


def test_missing_file_raises(tmp_path):
    with pytest.raises(RegistryError, match="no models.yaml"):
        load_registry(tmp_path / "does-not-exist.yaml")


def test_unknown_model_id_raises(tmp_path):
    p = _write_models(tmp_path, [_entry("a")])
    reg = load_registry(p)
    with pytest.raises(RegistryError, match="no model registered"):
        reg.get("ghost")


def test_keyless_local_backend_always_available(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    p = _write_models(tmp_path, [
        _entry("local-mind", provider="opencode", endpoint="anthropic/claude-sonnet-4-5",
               api_key_env="none", backend="opencode"),
    ])
    reg = load_registry(p)
    assert reg.get("local-mind").backend == "opencode"
    assert [m.id for m in reg.available()] == ["local-mind"]
    assert reg.missing_keys() == []


def test_invalid_backend_rejected(tmp_path):
    p = _write_models(tmp_path, [_entry("m", backend="telepathy")])
    with pytest.raises(RegistryError, match="backend"):
        load_registry(p)
