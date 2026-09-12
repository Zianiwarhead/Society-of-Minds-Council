"""Discovery tests — cache + merge, no network (except one opt-in live test)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import yaml

from council.discovery import (
    DiscoveryError,
    _is_free,
    discovered_to_entries,
    load_merged_registry,
    merge_registries,
    read_cache,
    sync_openrouter,
    write_cache,
)
from council.registry import load_registry


def test_is_free_handles_string_pricing():
    assert _is_free({"prompt": "0", "completion": "0"}) is True
    assert _is_free({"prompt": "0.0000002", "completion": "0"}) is False
    assert _is_free({}) is True
    assert _is_free({"prompt": "nonsense", "completion": "0"}) is False


def test_discovered_entries_have_safe_defaults():
    entries = discovered_to_entries([
        {"id": "x/free-model", "context_length": 64000,
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "x/paid-model",
         "pricing": {"prompt": "0.003", "completion": "0.015"}},
    ])
    free, paid = entries
    assert free.cost_tier == "free"
    assert paid.cost_tier == "paid"
    assert free.capabilities == ["code_generation"]
    assert free.quality_score.value is None  # unranked, never wins ties
    assert free.context_window == 64000
    assert free.api_key_env == "OPENROUTER_API_KEY"


def test_merge_keeps_overrides_and_adds_new(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([{
        "id": "x/known", "provider": "openrouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "paid",
        "capabilities": ["code_generation", "security_audit"],
        "quality_score": {"value": 0.8, "source": "user_override"},
    }]))
    overrides = load_registry(p)
    discovered = discovered_to_entries([
        {"id": "x/known", "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "x/brand-new", "pricing": {"prompt": "0", "completion": "0"}},
    ])
    merged, n_new, n_total = merge_registries(overrides, discovered)
    assert n_new == 1
    assert n_total == 2
    # local override untouched even though live says free
    assert merged.get("x/known").cost_tier == "paid"
    assert merged.get("x/known").quality_score.value == 0.8
    assert merged.get("x/brand-new").cost_tier == "free"


def test_cache_roundtrip_and_stale(tmp_path):
    cache = tmp_path / "openrouter-models.json"
    write_cache(cache, [{"id": "a"}])
    models, stale, fetched_at = read_cache(cache)
    assert models == [{"id": "a"}]
    assert stale is False
    assert fetched_at is not None


def test_cache_missing_raises(tmp_path):
    with pytest.raises(DiscoveryError, match="no cached catalog"):
        read_cache(tmp_path / "missing.json")


def test_sync_serves_fresh_cache_without_network(tmp_path):
    cache = tmp_path / "openrouter-models.json"
    write_cache(cache, [{"id": "a", "pricing": {"prompt": "0", "completion": "0"}}])
    with mock.patch("council.discovery.fetch_openrouter_models") as fetch:
        result = sync_openrouter(cache_dir=tmp_path, refresh=False)
        fetch.assert_not_called()
    assert result.cached is True
    assert result.total == 1
    assert result.free == 1


def test_sync_falls_back_to_stale_cache_on_network_failure(tmp_path):
    cache = tmp_path / "openrouter-models.json"
    cache.write_text(json.dumps({
        "fetched_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        "models": [{"id": "a", "pricing": {"prompt": "1", "completion": "1"}}],
    }))
    with mock.patch("council.discovery.fetch_openrouter_models",
                     side_effect=RuntimeError("no network")):
        result = sync_openrouter(cache_dir=tmp_path, refresh=True)
    assert result.cached is True
    assert result.stale is True
    assert result.total == 1


def test_sync_no_cache_and_no_network_raises(tmp_path):
    with mock.patch("council.discovery.fetch_openrouter_models",
                     side_effect=RuntimeError("no network")):
        with pytest.raises(DiscoveryError, match="could not fetch"):
            sync_openrouter(cache_dir=tmp_path, refresh=True)


def test_load_merged_registry_needs_cache(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump([{
        "id": "a", "provider": "openrouter",
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "api_key_env": "OPENROUTER_API_KEY", "cost_tier": "free",
        "capabilities": ["code_generation"],
    }]))
    with pytest.raises(DiscoveryError, match="run `models sync` first"):
        load_merged_registry(p, tmp_path)
