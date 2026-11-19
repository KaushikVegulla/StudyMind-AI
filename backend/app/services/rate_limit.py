"""Per-IP rate limiting: in-memory token buckets.

One bucket per client IP: capacity ``limit`` tokens, refilling at
``limit`` tokens per minute. A request costs one token; an empty
bucket means 429 with a Retry-After derived from the refill rate.

In-memory and per-process by design — this is a single-instance
free-tier deployment, and the limiter's job is to stop one client
from burning the shared LLM quota, not to be a distributed quota
system. The clock is injectable so tests never wait.
"""
from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, capacity: float, refill_per_second: float,
                 now=time.monotonic) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._now = now
        self.tokens = capacity
        self._updated = now()

    def _refill(self) -> None:
        current = self._now()
        elapsed = max(0.0, current - self._updated)
        self.tokens = min(self.capacity,
                          self.tokens + elapsed * self.refill_per_second)
        self._updated = current

    def allow(self) -> bool:
        """Take one token if available."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def retry_after_seconds(self) -> float:
        """Seconds until at least one token is available again."""
        self._refill()
        if self.tokens >= 1:
            return 0.0
        if self.refill_per_second <= 0:
            return 60.0
        return (1 - self.tokens) / self.refill_per_second


class RateLimiter:
    """A TokenBucket per key (client IP), created on first use."""

    def __init__(self, limit_per_minute: int, now=time.monotonic) -> None:
        self.limit_per_minute = limit_per_minute
        self._now = now
        self._buckets: dict[str, TokenBucket] = {}

    def _bucket(self, key: str) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(capacity=float(self.limit_per_minute),
                                 refill_per_second=self.limit_per_minute / 60,
                                 now=self._now)
            self._buckets[key] = bucket
        return bucket

    def allow(self, key: str) -> bool:
        if self.limit_per_minute <= 0:
            return True
        return self._bucket(key).allow()

    def retry_after_seconds(self, key: str) -> int:
        if self.limit_per_minute <= 0:
            return 0
        return max(1, round(self._bucket(key).retry_after_seconds()))
