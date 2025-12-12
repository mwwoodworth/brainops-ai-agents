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
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple


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
        self._counters = Counter()
        self._error_count = 0
        self._total_duration_ms = 0.0

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
            self._counters[(method, path)] += 1
            self._total_duration_ms += duration_ms
            if status >= 500:
                self._error_count += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return aggregated metrics over the rolling window."""
        samples = list(self._samples)
        durations = [s.duration_ms for s in samples]
        count = len(samples)
        errors = self._error_count

        def _quantile(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            try:
                return statistics.quantiles(values, n=100)[int(q * 100) - 1]
            except statistics.StatisticsError:
                return statistics.mean(values)

        top_paths = []
        for (method, path), hits in self._counters.most_common(5):
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
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get a value if it has not expired."""
        now = time.monotonic()
        async with self._lock:
            value = self._store.get(key)
            if not value:
                return None
            payload, expires_at = value
            if expires_at < now:
                self._store.pop(key, None)
                return None
            self._hits += 1
            return payload

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        """Set a cache entry with TTL."""
        expires_at = time.monotonic() + ttl_seconds
        async with self._lock:
            if len(self._store) >= self._max_size:
                # Evict oldest by insertion order
                oldest_key = next(iter(self._store.keys()))
                self._store.pop(oldest_key, None)
            self._store[key] = (value, expires_at)

    async def get_or_set(
        self,
        key: str,
        ttl_seconds: float,
        loader: Callable[[], Awaitable[Any]],
    ) -> Tuple[Any, bool]:
        """
        Get a cached value or compute it via loader.

        Returns tuple of (value, from_cache).
        """
        cached = await self.get(key)
        if cached is not None:
            return cached, True

        self._misses += 1
        value = await loader()
        await self.set(key, value, ttl_seconds)
        return value, False

    def snapshot(self) -> Dict[str, Any]:
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

