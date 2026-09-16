import pytest

from council import ratelimit as _ratelimit


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Every test starts with a fresh rate limiter. Without this, tests
    that fan out to fake models with the same id as an earlier test's
    inherit that bucket's depleted tokens and pace-wait for no reason
    connected to what they're actually testing — the rate limiter is
    process-global by design for real CLI runs (see ratelimit.get_limiter),
    but a test suite is not one continuous CLI run."""
    _ratelimit.reset()
    yield
    _ratelimit.reset()
