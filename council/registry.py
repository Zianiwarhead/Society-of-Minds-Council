"""Model registry loader — parses and validates `models.yaml` (Section 1).

This module only loads and validates the registry, and exposes query
primitives (`eligible_for`, `available`) that the classifier/model-selection
code (Section 6) will build on next. It does not itself pick a model for a
task — that's the next piece.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

VALID_COST_TIERS = {"free", "paid", "hybrid"}
VALID_BACKENDS = {"openrouter", "opencode"}
VALID_QUALITY_SOURCES = {"user_override", "measured"}  # no scraper/judge — manual only (Section 8)
REQUIRED_FIELDS = ("id", "provider", "endpoint", "api_key_env", "cost_tier", "capabilities")


class RegistryError(Exception):
    """Raised when models.yaml is missing, malformed, or internally inconsistent
    (dangling escalates_to reference, escalation cycle, duplicate id, etc.)."""


@dataclass
class QualityScore:
    value: Optional[float] = None
    source: Optional[str] = None       # "user_override" | "measured" | None
    last_updated: Optional[str] = None


@dataclass
class ModelEntry:
    id: str
    provider: str
    endpoint: str
    api_key_env: str
    cost_tier: str                      # free | paid | hybrid
    capabilities: List[str] = field(default_factory=list)
    not_trusted_for: List[str] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    context_window: Optional[int] = None
    quality_score: QualityScore = field(default_factory=QualityScore)
    latency_class: Optional[str] = None
    rate_limit_rpm: Optional[int] = None
    max_concurrent: Optional[int] = None
    escalates_to: Optional[str] = None
    last_verified: Optional[str] = None
    backend: str = "openrouter"         # openrouter | opencode

    def has_api_key(self) -> bool:
        """Whether this model's key is actually set in the environment — the
        fail-fast check from Section 5. Doesn't read the value, just checks
        presence, so a key is never logged or echoed anywhere.

        Local backends (opencode CLI, which owns its own auth) set
        `api_key_env: "none"` and are always available.
        """
        if self.api_key_env.strip().lower() in ("", "none", "-"):
            return True
        return bool(os.environ.get(self.api_key_env))


class ModelRegistry:
    def __init__(self, models: List[ModelEntry]):
        self._models: Dict[str, ModelEntry] = {m.id: m for m in models}

    def all(self) -> List[ModelEntry]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelEntry:
        try:
            return self._models[model_id]
        except KeyError:
            raise RegistryError(f"no model registered with id '{model_id}'")

    def eligible_for(self, required_caps: List[str], cost_tier: Optional[str] = None) -> List[ModelEntry]:
        """Models that have every required capability and aren't explicitly
        excluded from any of them — the filter step the classifier (Section 6)
        builds its selection on top of."""
        result = []
        for m in self._models.values():
            if not all(c in m.capabilities for c in required_caps):
                continue
            if any(c in m.not_trusted_for for c in required_caps):
                continue
            if cost_tier is not None and m.cost_tier != cost_tier:
                continue
            result.append(m)
        return result

    def available(self, models: Optional[List[ModelEntry]] = None) -> List[ModelEntry]:
        """Filters to models whose API key is actually set — the fail-fast
        check from Section 5, applied to whatever candidate pool is passed in
        (or the whole registry if none is given)."""
        candidates = models if models is not None else self._models.values()
        return [m for m in candidates if m.has_api_key()]

    def missing_keys(self, models: Optional[List[ModelEntry]] = None) -> List[str]:
        """Names of env vars that are missing, for whichever models are
        actually candidates for the task at hand — not the whole registry
        by default, since a shared/public registry will list far more
        models than any one user has keys for."""
        candidates = models if models is not None else self._models.values()
        missing = sorted({m.api_key_env for m in candidates if not m.has_api_key()})
        return missing


def _parse_quality_score(raw: Optional[dict]) -> QualityScore:
    if not raw:
        return QualityScore()
    source = raw.get("source")
    if source is not None and source not in VALID_QUALITY_SOURCES:
        raise RegistryError(
            f"invalid quality_score.source '{source}' — must be one of "
            f"{sorted(VALID_QUALITY_SOURCES)} (no scraper/judge sources, see Section 8)"
        )
    return QualityScore(
        value=raw.get("value"),
        source=source,
        last_updated=raw.get("last_updated"),
    )


def _parse_entry(raw: dict) -> ModelEntry:
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise RegistryError(f"model entry missing required field(s) {missing}: {raw}")

    if raw["cost_tier"] not in VALID_COST_TIERS:
        raise RegistryError(
            f"model '{raw['id']}' has invalid cost_tier '{raw['cost_tier']}' "
            f"— must be one of {sorted(VALID_COST_TIERS)}"
        )

    backend = raw.get("backend", "openrouter")
    if backend not in VALID_BACKENDS:
        raise RegistryError(
            f"model '{raw['id']}' has invalid backend '{backend}' "
            f"— must be one of {sorted(VALID_BACKENDS)}"
        )

    return ModelEntry(
        id=raw["id"],
        provider=raw["provider"],
        endpoint=raw["endpoint"],
        api_key_env=raw["api_key_env"],
        cost_tier=raw["cost_tier"],
        capabilities=raw.get("capabilities", []),
        not_trusted_for=raw.get("not_trusted_for", []),
        cost_per_1k_input=raw.get("cost_per_1k_input", 0.0),
        cost_per_1k_output=raw.get("cost_per_1k_output", 0.0),
        context_window=raw.get("context_window"),
        quality_score=_parse_quality_score(raw.get("quality_score")),
        latency_class=raw.get("latency_class"),
        rate_limit_rpm=raw.get("rate_limit_rpm"),
        max_concurrent=raw.get("max_concurrent"),
        escalates_to=raw.get("escalates_to"),
        last_verified=raw.get("last_verified"),
        backend=backend,
    )


def _validate_escalation_graph(models: List[ModelEntry]) -> None:
    ids = {m.id for m in models}
    for m in models:
        if m.escalates_to is None:
            continue
        if m.escalates_to not in ids:
            raise RegistryError(
                f"model '{m.id}' escalates_to '{m.escalates_to}', which isn't "
                f"in the registry"
            )

    # Cycle detection: walk each model's escalation chain; a chain that
    # revisits a node before running out means a loop that would never
    # reach HUMAN_REVIEW.
    for start in models:
        seen = set()
        current = start.id
        by_id = {m.id: m for m in models}
        while current is not None:
            if current in seen:
                raise RegistryError(
                    f"escalation cycle detected starting from '{start.id}' "
                    f"(revisits '{current}')"
                )
            seen.add(current)
            current = by_id[current].escalates_to


def load_registry(path: Path) -> ModelRegistry:
    registry_path = path if path.is_file() else path / "models.yaml"
    if not registry_path.exists():
        raise RegistryError(f"no models.yaml found at {registry_path}")

    raw = yaml.safe_load(registry_path.read_text()) or []
    if not isinstance(raw, list):
        raise RegistryError("models.yaml must be a list of model entries")
    if not raw:
        raise RegistryError("models.yaml has no model entries")

    entries = [_parse_entry(e) for e in raw]

    ids = [e.id for e in entries]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise RegistryError(f"duplicate model id(s) in models.yaml: {sorted(duplicates)}")

    _validate_escalation_graph(entries)

    return ModelRegistry(entries)
