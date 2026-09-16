"""Rate limiter tests — bucket math with a fake clock (no real waiting)."""
from __future__ import annotations

from council import ratelimit as rl
from council.ratelimit import RateLimiter, _Bucket


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.slept += seconds
        self.now += seconds


def _patch_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(rl.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(rl.time, "sleep", clock.sleep)
    return clock


def _model(key_env="K", rpm=None):
    return type("M", (), {"api_key_env": key_env, "rate_limit_rpm": rpm})()


def test_burst_then_pace(monkeypatch):
    clock = _patch_clock(monkeypatch)
    bucket = _Bucket(rpm=6)  # capacity 6, refill 1 per 10s
    for _ in range(6):
        bucket.acquire()
    assert clock.slept == 0
    bucket.acquire()  # 7th waits ~10s
    assert clock.slept == 10.0


def test_refill_over_time(monkeypatch):
    clock = _patch_clock(monkeypatch)
    bucket = _Bucket(rpm=60)  # 1 token/sec
    for _ in range(60):
        bucket.acquire()
    bucket.acquire()
    assert clock.slept == 1.0
    clock.now += 30  # 30s idle refills 30 tokens
    for _ in range(30):
        bucket.acquire()
    assert clock.slept == 1.0  # no further waiting


def test_buckets_keyed_by_api_key_env(monkeypatch):
    _patch_clock(monkeypatch)
    limiter = RateLimiter(default_rpm=1)
    a = _model("KEY_A")
    b = _model("KEY_B")
    limiter.acquire(a)  # drains KEY_A bucket
    limiter.acquire(b)  # KEY_B bucket untouched — no wait
    assert set(limiter._buckets) == {"KEY_A", "KEY_B"}


def test_shared_key_shares_bucket(monkeypatch):
    clock = _patch_clock(monkeypatch)
    limiter = RateLimiter(default_rpm=1)
    limiter.acquire(_model("SHARED"))
    limiter.acquire(_model("SHARED"))  # same key -> must wait ~60s
    assert clock.slept == 60.0


def test_registry_rpm_hint_wins_over_fallback(monkeypatch):
    _patch_clock(monkeypatch)
    limiter = RateLimiter(default_rpm=1)
    fast = _model("K", rpm=600)
    for _ in range(10):
        limiter.acquire(fast)
    assert limiter._buckets["K"].rpm == 600


def test_singleton_reset(monkeypatch):
    _patch_clock(monkeypatch)
    first = rl.get_limiter()
    assert rl.get_limiter() is first
    rl.reset()
    assert rl.get_limiter() is not first
