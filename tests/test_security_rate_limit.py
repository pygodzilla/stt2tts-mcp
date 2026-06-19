"""Unit tests for the rate limiter (token bucket)."""

from __future__ import annotations

import pytest

from stt2tts_mcp.security.errors import RateLimitError
from stt2tts_mcp.security.rate_limit import RateLimiter


def test_first_call_always_passes() -> None:
    """Bucket starts full so the first call never blocks."""
    rl = RateLimiter({"speak": (3.0, 0.1)})
    for _ in range(3):
        rl.consume("speak")  # should not raise


def test_burst_then_429() -> None:
    """Burst of 3 then 4th must raise with retry_after."""
    rl = RateLimiter({"speak": (3.0, 0.001)})  # very slow refill
    for _ in range(3):
        rl.consume("speak")
    with pytest.raises(RateLimitError) as exc:
        rl.consume("speak")
    assert exc.value.code == "rate_limited"
    assert exc.value.details["tool"] == "speak"
    assert exc.value.details["retry_after"] > 0


def test_unthrottled_tools_never_block() -> None:
    """Tools not in the bucket map are not throttled."""
    rl = RateLimiter({"speak": (1.0, 0.0001)})
    # health_check isn't in the bucket — 1000 calls should all succeed.
    for _ in range(1000):
        rl.consume("health_check")


def test_buckets_are_per_tool() -> None:
    """Exhausting speak must NOT block transcribe."""
    rl = RateLimiter(
        {
            "speak": (1.0, 0.0001),
            "transcribe": (5.0, 0.0001),
        }
    )
    rl.consume("speak")
    with pytest.raises(RateLimitError):
        rl.consume("speak")
    # transcribe bucket still has tokens.
    for _ in range(5):
        rl.consume("transcribe")


def test_refill_restores_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    """After enough wall-clock time, the bucket refills."""
    # 2 tokens, 10 per second refill → ~100ms refill of 1 token.
    rl = RateLimiter({"speak": (2.0, 10.0)})
    rl.consume("speak")
    rl.consume("speak")
    with pytest.raises(RateLimitError):
        rl.consume("speak")
    # Force refill by resetting (testing convenience).
    rl.reset("speak")
    rl.consume("speak")
    rl.consume("speak")


def test_snapshot_exposes_state() -> None:
    rl = RateLimiter({"speak": (5.0, 0.5), "transcribe": (10.0, 1.0)})
    snap = rl.snapshot()
    assert snap["speak"]["capacity"] == 5.0
    assert snap["speak"]["refill_per_second"] == 0.5
    assert snap["transcribe"]["capacity"] == 10.0


def test_default_limits_cover_speak_and_transcribe() -> None:
    rl = RateLimiter()  # use defaults
    snap = rl.snapshot()
    assert "speak" in snap
    assert "transcribe" in snap


def test_consume_n_tokens_at_once() -> None:
    """Spending multiple tokens at once (e.g., for batch ops)."""
    rl = RateLimiter({"speak": (5.0, 0.0001)})
    rl.consume("speak", n=5)
    with pytest.raises(RateLimitError):
        rl.consume("speak")  # bucket is now empty


def test_zero_refill_returns_long_retry_after() -> None:
    """Pathological config: bucket that never refills."""
    rl = RateLimiter({"speak": (1.0, 0.0)})
    rl.consume("speak")
    with pytest.raises(RateLimitError) as exc:
        rl.consume("speak")
    # Should suggest a long retry (not 0, not infinity).
    assert exc.value.details["retry_after"] > 100
