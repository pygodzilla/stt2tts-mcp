"""Token-bucket rate limiter for stt2tts-mcp tools.

Design notes:
- One bucket per tool name. STT and TTS have separate quotas so a heavy
  transcribe user can't starve speak (and vice versa).
- Thread-safe via a single lock — tool calls in this server are sequential
  on a single event loop, but the lock keeps things honest if the server is
  ever embedded in a multi-threaded runtime.
- Lazy refill: capacity is checked at consume time, so we don't need a
  background thread.
- Bounded memory: one small struct per tool (currently 6 tools).
- Bypassed for `health_check`, `reload_config`, `list_*` — those are cheap,
  idempotent, and should always succeed. Only expensive or side-effecting
  tools are throttled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from stt2tts_mcp.security.errors import RateLimitError


@dataclass
class _Bucket:
    """Sliding-window-ish bucket: refills continuously up to `capacity`."""

    capacity: float
    refill_per_second: float
    tokens: float
    last_refill: float

    def consume(self, n: float = 1.0) -> tuple[bool, float]:
        """Try to consume `n` tokens. Returns (ok, retry_after_seconds).

        On success, retry_after is 0.0. On failure, retry_after is the
        wall-clock seconds until enough tokens would be available.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(
                self.capacity, self.tokens + elapsed * self.refill_per_second
            )
            self.last_refill = now
        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        # Not enough tokens: how long until we have `n`?
        deficit = n - self.tokens
        if self.refill_per_second <= 0:
            # No refill — would never recover. Return a large retry hint.
            return False, 3600.0
        return False, deficit / self.refill_per_second


class RateLimiter:
    """Per-tool token-bucket rate limiter.

    Usage:
        limiter = RateLimiter({
            "transcribe": (60, 1.0),    # 60 burst, 1 req/sec sustained
            "speak":      (30, 0.5),
        })
        limiter.consume("transcribe") -> raise RateLimitError on exhaustion
    """

    def __init__(self, limits: dict[str, tuple[float, float]] | None = None) -> None:
        # default limits are conservative: STT is cheap (~1s), TTS is heavier
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        for name, (capacity, per_sec) in (limits or self.default_limits()).items():
            self._buckets[name] = _Bucket(
                capacity=float(capacity),
                refill_per_second=float(per_sec),
                tokens=float(capacity),  # start full
                last_refill=time.monotonic(),
            )

    @staticmethod
    def default_limits() -> dict[str, tuple[float, float]]:
        """Default per-tool quotas.

        Format: {tool_name: (burst_capacity, sustained_per_second)}.
        - transcribe: 30 burst, 0.5/sec sustained (one every 2s average)
        - speak:      15 burst, 0.25/sec (one every 4s average)
        """
        return {
            "transcribe": (30.0, 0.5),
            "speak": (15.0, 0.25),
        }

    def consume(self, tool: str, n: float = 1.0) -> None:
        """Try to consume a token for `tool`. Raises `RateLimitError` on exhaustion."""
        # Tools not in the bucket map are unthrottled (cheap reads like
        # health_check, list_*_models, reload_config).
        if tool not in self._buckets:
            return
        with self._lock:
            bucket = self._buckets[tool]
            ok, retry_after = bucket.consume(n)
        if not ok:
            raise RateLimitError(tool=tool, retry_after=retry_after)

    def reset(self, tool: str | None = None) -> None:
        """Refill a bucket (useful for tests)."""
        with self._lock:
            targets = [tool] if tool else list(self._buckets)
            for name in targets:
                b = self._buckets.get(name)
                if b:
                    b.tokens = b.capacity
                    b.last_refill = time.monotonic()

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Read-only view of bucket state — for /health or tests."""
        with self._lock:
            return {
                name: {
                    "capacity": b.capacity,
                    "refill_per_second": b.refill_per_second,
                    "tokens": b.tokens,
                }
                for name, b in self._buckets.items()
            }


__all__ = ["RateLimiter"]
