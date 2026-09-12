"""Task classifier — complexity scoring, model selection, and the escalation
state machine. See spec Section 6 (Task Classifier Logic).

This module is deliberately boring: complexity scoring is a handful of
cheap, deterministic signals, not an LLM call — that's the whole point of
the token-saving design. It calls straight into the registry's query
primitives (`eligible_for`, `available`) built in Section 1/registry.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from council.registry import ModelEntry, ModelRegistry
from council.task import Task

SECURITY_KEYWORDS = ("security", "auth", "cve", "encrypt", "vulnerab", "exploit")


class NoEligibleModelError(Exception):
    """Raised when no registered, available model has the required
    capabilities (and, if specified, the required cost tier)."""


def _has_security_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(k in lowered for k in SECURITY_KEYWORDS)


def complexity_score(task: Task) -> int:
    """The raw score behind the complexity bucket — kept separate so it can
    be logged/inspected, not just the final label."""
    score = 0
    score += min(task.files_touched, 10) * 2
    score += min(task.lines_changed // 20, 10)
    score += 15 if task.cross_file_dependencies else 0
    score += 20 if task.is_retry else 0
    score += 25 if _has_security_keywords(task.description) else 0
    return score


def classify_complexity(task: Task) -> str:
    score = complexity_score(task)
    if score < 20:
        return "simple"
    elif score < 50:
        return "moderate"
    else:
        return "complex"


def required_capabilities(task: Task) -> List[str]:
    """What a model needs to be trusted for, derived from the same cheap
    signals as the complexity score — no LLM call needed here either."""
    caps = ["code_generation"]
    if task.cross_file_dependencies:
        caps.append("cross_file_architecture")
    if _has_security_keywords(task.description):
        caps.append("security_audit")
    return caps


def _score_key(model: ModelEntry):
    # Unranked (quality_score.value is None) models never win a tie-break
    # against a scored model — see Section 8. -1 sorts below any real 0..1 score.
    quality = model.quality_score.value if model.quality_score.value is not None else -1
    return (quality, -model.cost_per_1k_input)


def select_model(registry: ModelRegistry, required_caps: List[str], tier: Optional[str] = None) -> ModelEntry:
    """Picks the best eligible, available model for the given capabilities
    and (optional) cost tier. Highest quality_score wins; ties broken by
    lowest cost."""
    candidates = registry.available(registry.eligible_for(required_caps, cost_tier=tier))
    if not candidates:
        raise NoEligibleModelError(
            f"no available model with capabilities {required_caps}"
            + (f" in tier '{tier}'" if tier else "")
        )
    return max(candidates, key=_score_key)


def select_initial_model(registry: ModelRegistry, task: Task) -> ModelEntry:
    """The Rule 1 'free-first baseline' decision: simple and moderate tasks
    get a free-tier model first; only tasks scored 'complex' (cross-file +
    security-flagged + already-a-retry, roughly) skip straight to paid,
    per the original free-vs-paid routing diagram."""
    complexity = classify_complexity(task)
    caps = required_capabilities(task)
    tier = "paid" if complexity == "complex" else "free"

    try:
        return select_model(registry, caps, tier=tier)
    except NoEligibleModelError:
        if tier == "free":
            # No free model fits — fall back to the open pool rather than
            # failing a task outright just because free-tier coverage is thin.
            return select_model(registry, caps, tier=None)
        raise


class TaskState(str, Enum):
    RUNNING = "RUNNING"
    ESCALATED = "ESCALATED"
    DONE = "DONE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    VERIFY_ERROR = "VERIFY_ERROR"


@dataclass
class Attempt:
    model_id: str
    passed: bool
    verify_error: bool = False  # True: the verifier itself couldn't run — not a merit-based failure


@dataclass
class NextStep:
    state: TaskState
    model: Optional[ModelEntry] = None
    reason: str = ""


def decide_next(task: Task, registry: ModelRegistry, attempts: List[Attempt]) -> NextStep:
    """The escalation state machine from Section 6:

        NEW -> (free model) -> RUNNING
        RUNNING -> PASS -> DONE
        RUNNING -> FAIL (1st free attempt) -> RUNNING (try another free model)
        RUNNING -> FAIL (2nd free attempt) -> ESCALATED (force paid tier)
        ESCALATED -> FAIL -> HUMAN_REVIEW (never loop a paid model indefinitely)

    A VERIFY_ERROR on the latest attempt short-circuits everything above —
    see Section 3/7: an environment problem isn't evidence the model's code
    was wrong, so it must never count toward the free/paid attempt tally.
    """
    if attempts and attempts[-1].verify_error:
        return NextStep(TaskState.VERIFY_ERROR, reason="verifier could not run — not a model failure")

    if attempts and attempts[-1].passed:
        return NextStep(TaskState.DONE)

    caps = required_capabilities(task)
    free_attempts = [a for a in attempts if registry.get(a.model_id).cost_tier == "free"]
    paid_attempts = [a for a in attempts if registry.get(a.model_id).cost_tier != "free"]

    if paid_attempts:
        # We're already in ESCALATED and the paid attempt just failed —
        # per spec, this goes straight to human review, no further escalation.
        return NextStep(TaskState.HUMAN_REVIEW, reason="paid-tier attempt failed; not looping a paid model")

    if not attempts:
        model = select_initial_model(registry, task)
        state = TaskState.ESCALATED if model.cost_tier != "free" else TaskState.RUNNING
        return NextStep(state, model=model, reason="initial selection")

    if len(free_attempts) == 1:
        tried_ids = {a.model_id for a in free_attempts}
        try:
            pool_models = [
                m for m in registry.available(registry.eligible_for(caps, cost_tier="free"))
                if m.id not in tried_ids
            ]
            if pool_models:
                model = max(pool_models, key=_score_key)
                return NextStep(TaskState.RUNNING, model=model, reason="retrying with a different free model")
        except NoEligibleModelError:
            pass
        # no other free model available — escalate now rather than stall
        model = select_model(registry, caps, tier="paid")
        return NextStep(TaskState.ESCALATED, model=model, reason="no free-tier alternative left")

    # len(free_attempts) >= 2 and no paid attempt yet: force escalation
    model = select_model(registry, caps, tier="paid")
    return NextStep(TaskState.ESCALATED, model=model, reason="two free-tier failures — forcing paid tier")
