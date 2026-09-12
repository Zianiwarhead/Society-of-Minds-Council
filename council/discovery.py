"""Live model discovery — cache + merge (OpenRouter first).

`models.yaml` stays small: it holds only local overrides (cost_tier,
capabilities, not_trusted_for, quality_score, escalates_to). The live
catalog comes from the provider handshake (`GET /models`), is cached on
disk with a TTL, and merged at load time. Unknown IDs default to
eligible-but-unranked and never win ties (spec Section 8).
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from council.registry import ModelEntry, ModelRegistry

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

CACHE_TTL = timedelta(hours=24)


class DiscoveryError(Exception):
    """Raised when the live catalog can't be fetched and no usable cache exists."""


@dataclass
class SyncResult:
    provider: str
    total: int
    free: int
    paid: int
    cached: bool  # True if served from cache without a network fetch
    stale: bool  # True if the cache is older than the TTL
    cache_path: Path


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "council"


def _cache_path(cache_dir: Optional[Path], provider: str = "openrouter") -> Path:
    base = cache_dir or default_cache_dir()
    return base / f"{provider}-models.json"


def _is_free(pricing: Dict[str, Any]) -> bool:
    """OpenRouter quotes pricing as decimal strings per token. Zero/zero
    (or missing/blank) means a free endpoint."""
    try:
        prompt = float(pricing.get("prompt") or 0)
        completion = float(pricing.get("completion") or 0)
    except (TypeError, ValueError):
        return False
    return prompt == 0 and completion == 0


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def fetch_openrouter_models(timeout_seconds: int = 30) -> List[Dict[str, Any]]:
    req = urllib.request.Request(
        OPENROUTER_MODELS_URL,
        headers={"Accept": "application/json", "User-Agent": "ai-council/0.1"},
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        payload = json.load(resp)
    models = payload.get("data", [])
    if not isinstance(models, list):
        raise DiscoveryError("unexpected /models response shape (no data list)")
    return models


def write_cache(cache_path: Path, models: List[Dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat(), "models": models}),
        encoding="utf-8",
    )
    tmp.replace(cache_path)


def read_cache(cache_path: Path) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    """Returns (models, stale, fetched_at). Missing file raises DiscoveryError."""
    if not cache_path.exists():
        raise DiscoveryError(f"no cached catalog at {cache_path} — run `models sync` first")
    raw = json.loads(cache_path.read_text())
    models = raw.get("models", [])
    fetched_at = raw.get("fetched_at")
    stale = True
    if fetched_at:
        try:
            fetched = datetime.fromisoformat(fetched_at)
            stale = datetime.now(timezone.utc) - fetched > CACHE_TTL
        except ValueError:
            stale = True
    return models, stale, fetched_at


def sync_openrouter(
    cache_dir: Optional[Path] = None, refresh: bool = False
) -> SyncResult:
    """Refresh the cached OpenRouter catalog unless a fresh cache exists.

    A failed fetch with a usable (even stale) cache falls back to the
    cache instead of blocking — a dead network should never block a run.
    """
    path = _cache_path(cache_dir)
    if not refresh and path.exists():
        try:
            models, stale, _ = read_cache(path)
            if not stale:
                free = sum(1 for m in models if _is_free(m.get("pricing", {})))
                return SyncResult("openrouter", len(models), free, len(models) - free,
                                  True, False, path)
        except (DiscoveryError, ValueError, OSError):
            pass  # fall through to a live fetch

    try:
        models = fetch_openrouter_models()
    except Exception as exc:
        if path.exists():
            try:
                cached, stale, _ = read_cache(path)
                free = sum(1 for m in cached if _is_free(m.get("pricing", {})))
                return SyncResult("openrouter", len(cached), free, len(cached) - free,
                                  True, True, path)
            except (DiscoveryError, ValueError, OSError):
                pass
        raise DiscoveryError(f"could not fetch {OPENROUTER_MODELS_URL}: {exc}") from exc

    write_cache(path, models)
    free = sum(1 for m in models if _is_free(m.get("pricing", {})))
    return SyncResult("openrouter", len(models), free, len(models) - free,
                      False, False, path)


def discovered_to_entries(raw_models: List[Dict[str, Any]]) -> List[ModelEntry]:
    """Map the live catalog to registry entries with safe defaults: minimal
    capabilities, unranked quality, tier inferred from pricing only."""
    entries = []
    for m in raw_models:
        model_id = m.get("id")
        if not model_id:
            continue
        pricing = m.get("pricing", {}) or {}
        entries.append(
            ModelEntry(
                id=model_id,
                provider="openrouter",
                endpoint=OPENROUTER_CHAT_URL,
                api_key_env=OPENROUTER_API_KEY_ENV,
                cost_tier="free" if _is_free(pricing) else "paid",
                capabilities=["code_generation"],
                context_window=m.get("context_length"),
                last_verified=_today(),
            )
        )
    return entries


def merge_registries(
    overrides: ModelRegistry, discovered: List[ModelEntry]
) -> Tuple[ModelRegistry, int, int]:
    """Merge live entries under local overrides. Returns (merged, n_new, n_total).

    Any discovered ID present in `models.yaml` keeps the local entry
    untouched; discovered IDs not overridden are added with defaults.
    YAML-only IDs (local/direct providers) are always kept.
    """
    from council.registry import _validate_escalation_graph

    by_id = {m.id: m for m in overrides.all()}
    n_new = 0
    for entry in discovered:
        if entry.id not in by_id:
            by_id[entry.id] = entry
            n_new += 1
    merged = list(by_id.values())
    _validate_escalation_graph(merged)
    return ModelRegistry(merged), n_new, len(merged)


def load_merged_registry(
    registry_path: Path,
    cache_dir: Optional[Path] = None,
) -> Tuple[ModelRegistry, int, int, bool]:
    """Load `models.yaml` overrides + cached live catalog. Returns
    (merged, n_new, n_total, cache_stale). Raises DiscoveryError if no
    cache exists — run `models sync` first."""
    from council.registry import load_registry

    overrides = load_registry(registry_path)
    path = _cache_path(cache_dir)
    raw, stale, _ = read_cache(path)
    merged, n_new, n_total = merge_registries(overrides, discovered_to_entries(raw))
    return merged, n_new, n_total, stale
