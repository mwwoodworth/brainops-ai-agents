"""
Lightweight observability helpers for the BrainOps AI Agents service.

Provides:
- RequestMetrics: in-memory request/latency tracking with quantiles
- TTLCache: simple async-aware TTL cache with hit/miss accounting
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter, deque
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable, Deque, Optional


@dataclass
class RequestSample:
    """Single HTTP request observation."""

    path: str
    method: str
    status: int
    duration_ms: float
    timestamp: float


class RequestMetrics:
    """Collects rolling HTTP request metrics for observability endpoints."""

    def __init__(self, window: int = 500) -> None:
        self._samples: Deque[RequestSample] = deque(maxlen=window)
        self._lock = asyncio.Lock()
        # Note: counters computed from samples to avoid memory leaks

    async def record(self, path: str, method: str, status: int, duration_ms: float) -> None:
        """Record a request/response observation."""
        sample = RequestSample(
            path=path,
            method=method,
            status=status,
            duration_ms=duration_ms,
            timestamp=time.time(),
        )
        async with self._lock:
            self._samples.append(sample)

    def snapshot(self) -> dict[str, Any]:
        """Return aggregated metrics over the rolling window."""
        samples = list(self._samples)
        durations = [s.duration_ms for s in samples]
        count = len(samples)
        # Compute errors from current window only (fixes memory leak)
        errors = sum(1 for s in samples if s.status >= 500)

        def _quantile(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            try:
                return statistics.quantiles(values, n=100)[int(q * 100) - 1]
            except statistics.StatisticsError:
                return statistics.mean(values)

        # Compute hot paths from current window (avoids memory leak)
        path_counter: Counter = Counter()
        for s in samples:
            path_counter[(s.method, s.path)] += 1
        top_paths = []
        for (method, path), hits in path_counter.most_common(5):
            top_paths.append({"method": method, "path": path, "hits": hits})

        return {
            "sample_size": count,
            "error_rate": (errors / count) if count else 0.0,
            "latency_ms": {
                "avg": (sum(durations) / count) if count else 0.0,
                "p50": _quantile(durations, 0.50),
                "p95": _quantile(durations, 0.95),
                "p99": _quantile(durations, 0.99),
            },
            "recent_errors": errors,
            "hot_paths": top_paths,
        }


class TTLCache:
    """Minimal async-friendly TTL cache with hit/miss tracking."""

    def __init__(self, max_size: int = 256) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get a value if it has not expired."""
        now = time.monotonic()
        async with self._lock:
            value = self._store.get(key)
            if not value:
                self._misses += 1
                return None
            payload, expires_at = value
            if expires_at < now:
                self._store.pop(key, None)
                self._key_locks.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return payload

    def _prune_expired_locked(self) -> None:
        """Remove expired entries (call while holding lock)."""
        now = time.monotonic()
        expired_keys = [k for k, (_, exp) in self._store.items() if exp < now]
        for k in expired_keys:
            self._store.pop(k, None)
            self._key_locks.pop(k, None)

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Set a cache entry with TTL."""
        expires_at = time.monotonic() + ttl_seconds
        async with self._lock:
            # Prune expired entries first to free space
            self._prune_expired_locked()
            if len(self._store) >= self._max_size:
                # Evict oldest by insertion order
                oldest_key = next(iter(self._store.keys()))
                self._store.pop(oldest_key, None)
                self._key_locks.pop(oldest_key, None)
            self._store[key] = (value, expires_at)

    async def get_or_set(
        self,
        key: str,
        ttl_seconds: float,
        loader: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, bool]:
        """
        Get a cached value or compute it via loader.

        Returns tuple of (value, from_cache).
        """
        async with await self._get_key_lock(key):
            cached = await self.get(key)
            if cached is not None:
                return cached, True

            value = await loader()
            await self.set(key, value, ttl_seconds)
            return value, False

    async def _get_key_lock(self, key: str) -> asyncio.Lock:
        """Return a per-key lock to avoid thundering-herd cache misses."""
        lock = self._key_locks.get(key)
        if lock is not None:
            return lock
        async with self._lock:
            return self._key_locks.setdefault(key, asyncio.Lock())

    def snapshot(self) -> dict[str, Any]:
        """Return cache stats (non-mutating)."""
        live = 0
        now = time.monotonic()
        for _, (_, expires_at) in list(self._store.items()):
            if expires_at >= now:
                live += 1
        return {
            "size": live,
            "capacity": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
        }
