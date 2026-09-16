"""Rate limiter for the model fan-out.

Keyed by `api_key_env`, not model id — a provider's rate limit applies per
API key, shared across every model called with it, not per individual
model. Firing N parallel calls at N different free models that all share
one OpenRouter key hits ONE shared limit, not N separate ones.

Token-bucket, stdlib only (matches executor.py's "stdlib only" design),
thread-safe — this is called from inside the ThreadPoolExecutor fan-outs
in compare.py and collab.py.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional

# Used ONLY when nothing in the registry specifies rate_limit_rpm for a
# given api_key_env. Most free tiers sit somewhere near this, but it's a
# fallback, not a substitute for checking your provider's actual docs and
# setting rate_limit_rpm explicitly in models.yaml.
DEFAULT_RPM_FALLBACK = 20


class _Bucket:
    """One token bucket per api_key_env. Refills continuously at rpm/60
    tokens/sec, capped at `rpm` tokens — allows a small burst up to the
    per-minute limit, then paces every call after that."""

    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.capacity = float(self.rpm)
        self.tokens = float(self.rpm)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        rate_per_sec = self.rpm / 60.0
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                self.last_refill = now
                self.tokens = min(self.capacity, self.tokens + elapsed * rate_per_sec)
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                deficit = 1 - self.tokens
                wait = deficit / rate_per_sec
            time.sleep(max(wait, 0.05))


class RateLimiter:
    """One bucket per api_key_env, created lazily on first use. Call
    `.acquire(model)` immediately before every model call in a fan-out —
    it blocks the calling thread until a slot is free. It never drops a
    call and never queues silently forever; a stuck bucket just means the
    configured rpm is being honestly enforced."""

    def __init__(self, default_rpm: int = DEFAULT_RPM_FALLBACK):
        self._buckets: Dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._default_rpm = default_rpm

    def _bucket_for(self, key_env: str, rpm_hint: Optional[int]) -> _Bucket:
        with self._lock:
            bucket = self._buckets.get(key_env)
            if bucket is None:
                bucket = _Bucket(rpm_hint or self._default_rpm)
                self._buckets[key_env] = bucket
            return bucket

    def acquire(self, model) -> None:
        """Blocks until it's safe to call `model` without exceeding the
        pacing for its api_key_env. Reads `model.rate_limit_rpm` when the
        registry sets it; falls back to DEFAULT_RPM_FALLBACK otherwise."""
        bucket = self._bucket_for(model.api_key_env, getattr(model, "rate_limit_rpm", None))
        bucket.acquire()


_singleton: Optional[RateLimiter] = None
_singleton_lock = threading.Lock()


def get_limiter() -> RateLimiter:
    """Process-wide limiter shared across every fan-out in one CLI
    invocation — a global here is deliberate: this tool runs once per
    invocation, it isn't a library serving multiple concurrent callers,
    and every fan-out in the same run needs to share the same pacing
    state or the whole point of rate-limiting is defeated."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = RateLimiter()
    return _singleton


def reset() -> None:
    """Clears the singleton's bucket state. Real CLI runs never call
    this — persisting pacing state for the whole process is correct
    there. It exists for test isolation: without it, fake models reused
    across unrelated test cases inherit depleted tokens from whichever
    test ran first, and tests slow down for a reason that has nothing
    to do with what they're actually testing."""
    global _singleton
    with _singleton_lock:
        _singleton = None
