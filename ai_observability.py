#!/usr/bin/env python3
"""
AI OBSERVABILITY LAYER - Unified Metrics, Tracing, and Monitoring
==================================================================
Perfect observability across all bleeding-edge AI modules.

CAPABILITIES:
1. Prometheus-style metrics (counters, gauges, histograms)
2. Distributed tracing with correlation IDs
3. Real-time event streaming
4. Cross-module aggregation
5. SLA tracking with percentiles (p50, p95, p99)
6. Anomaly detection on metrics
7. Dashboard-ready data export

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import json
import time
import asyncio
import logging
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import hashlib
import uuid

logger = logging.getLogger(__name__)

# Database persistence (optional)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_values
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.info("psycopg2 not available, database persistence disabled")


# =============================================================================
# DATABASE PERSISTENCE ADAPTER
# =============================================================================

class ObservabilityPersistence:
    """
    Optional database persistence for observability data.
    Writes metrics, events, and traces to observability.* tables.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> Optional["ObservabilityPersistence"]:
        if not DB_AVAILABLE:
            return None
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        cls._instance = cls()
                    except Exception as e:
                        logger.error(f"Failed to initialize ObservabilityPersistence: {e}")
                        return None
        return cls._instance

    def __init__(self):
        self._db_config = self._get_db_config()
        self._write_buffer_metrics: deque = deque(maxlen=1000)
        self._write_buffer_events: deque = deque(maxlen=1000)
        self._write_buffer_traces: deque = deque(maxlen=1000)
        self._flush_interval = 10  # seconds
        self._last_flush = time.time()
        self._enabled = self._db_config is not None
        if self._enabled:
            logger.info("ObservabilityPersistence initialized with database connection")

    def _get_db_config(self) -> Optional[Dict[str, Any]]:
        """Get database configuration from environment"""
        host = os.environ.get("DB_HOST")
        database = os.environ.get("DB_NAME")
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASSWORD")
        port = int(os.environ.get("DB_PORT", "5432"))

        if all([host, database, user, password]):
            return {
                "host": host,
                "database": database,
                "user": user,
                "password": password,
                "port": port,
                "sslmode": "require"
            }
        return None

    def _get_connection(self):
        """Get a database connection"""
        if not self._db_config:
            return None
        try:
            return psycopg2.connect(**self._db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    def record_metric(self, name: str, value: float, metric_type: str,
                      labels: Dict[str, str] = None, module: str = ""):
        """Buffer a metric for persistence"""
        if not self._enabled:
            return
        self._write_buffer_metrics.append({
            "name": name,
            "value": value,
            "metric_type": metric_type,
            "labels": json.dumps(labels or {}),
            "module": module,
            "timestamp": datetime.now(timezone.utc)
        })
        self._maybe_flush()

    def record_event(self, event_type: str, source_module: str,
                     payload: Dict[str, Any], correlation_id: str = ""):
        """Buffer an event for persistence"""
        if not self._enabled:
            return
        self._write_buffer_events.append({
            "event_type": event_type,
            "source_module": source_module,
            "payload": json.dumps(payload),
            "correlation_id": correlation_id,
            "timestamp": datetime.now(timezone.utc)
        })
        self._maybe_flush()

    def record_trace(self, request_id: str, span_id: str, parent_span_id: str,
                     operation: str, module: str, duration_ms: float, status: str = "ok"):
        """Buffer a trace span for persistence"""
        if not self._enabled:
            return
        self._write_buffer_traces.append({
            "request_id": request_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "operation": operation,
            "module": module,
            "duration_ms": duration_ms,
            "status": status,
            "timestamp": datetime.now(timezone.utc)
        })
        self._maybe_flush()

    def _maybe_flush(self):
        """Flush buffers if interval has elapsed"""
        now = time.time()
        if now - self._last_flush >= self._flush_interval:
            self.flush()
            self._last_flush = now

    def flush(self):
        """Flush all buffered data to database"""
        if not self._enabled:
            return

        conn = self._get_connection()
        if not conn:
            return

        try:
            cur = conn.cursor()

            # Flush metrics
            if self._write_buffer_metrics:
                metrics_to_write = list(self._write_buffer_metrics)
                self._write_buffer_metrics.clear()
                self._insert_metrics(cur, metrics_to_write)

            # Flush events
            if self._write_buffer_events:
                events_to_write = list(self._write_buffer_events)
                self._write_buffer_events.clear()
                self._insert_events(cur, events_to_write)

            # Flush traces
            if self._write_buffer_traces:
                traces_to_write = list(self._write_buffer_traces)
                self._write_buffer_traces.clear()
                self._insert_traces(cur, traces_to_write)

            conn.commit()
            cur.close()
        except Exception as e:
            logger.error(f"Failed to flush observability data: {e}")
            conn.rollback()
        finally:
            conn.close()

    def _insert_metrics(self, cur, metrics: List[Dict]):
        """Insert metrics into observability.metrics table"""
        if not metrics:
            return
        try:
            execute_values(
                cur,
                """
                INSERT INTO observability.metrics (name, value, metric_type, labels, module, timestamp)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                [(m["name"], m["value"], m["metric_type"], m["labels"],
                  m["module"], m["timestamp"]) for m in metrics],
                template="(%s, %s, %s, %s, %s, %s)"
            )
        except Exception as e:
            logger.error(f"Failed to insert metrics: {e}")

    def _insert_events(self, cur, events: List[Dict]):
        """Insert events into observability.logs table"""
        if not events:
            return
        try:
            execute_values(
                cur,
                """
                INSERT INTO observability.logs (event_type, source_module, payload, correlation_id, timestamp)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                [(e["event_type"], e["source_module"], e["payload"],
                  e["correlation_id"], e["timestamp"]) for e in events],
                template="(%s, %s, %s, %s, %s)"
            )
        except Exception as e:
            logger.error(f"Failed to insert events: {e}")

    def _insert_traces(self, cur, traces: List[Dict]):
        """Insert traces into observability.traces table"""
        if not traces:
            return
        try:
            execute_values(
                cur,
                """
                INSERT INTO observability.traces (request_id, span_id, parent_span_id, operation, module, duration_ms, status, timestamp)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                [(t["request_id"], t["span_id"], t["parent_span_id"], t["operation"],
                  t["module"], t["duration_ms"], t["status"], t["timestamp"]) for t in traces],
                template="(%s, %s, %s, %s, %s, %s, %s, %s)"
            )
        except Exception as e:
            logger.error(f"Failed to insert traces: {e}")


# =============================================================================
# METRIC TYPES
# =============================================================================

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """Single metric value with metadata"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    module: str = ""

    def to_prometheus(self) -> str:
        """Export in Prometheus format"""
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        if label_str:
            return f'{self.name}{{{label_str}}} {self.value}'
        return f'{self.name} {self.value}'


@dataclass
class HistogramBucket:
    """Histogram bucket for latency tracking"""
    le: float  # less than or equal
    count: int = 0


class Histogram:
    """
    Histogram for tracking latency distributions.
    Provides p50, p95, p99 percentiles.
    """
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, labels: Dict[str, str] = None, buckets: List[float] = None):
        self.name = name
        self.labels = labels or {}
        self.buckets = [HistogramBucket(le=b) for b in (buckets or self.DEFAULT_BUCKETS)]
        self.buckets.append(HistogramBucket(le=float('inf')))  # +Inf bucket
        self._sum = 0.0
        self._count = 0
        self._values: deque = deque(maxlen=1000)  # For percentile calculation
        self._lock = threading.Lock()

    def observe(self, value: float):
        """Record an observation"""
        with self._lock:
            self._sum += value
            self._count += 1
            self._values.append(value)
            for bucket in self.buckets:
                if value <= bucket.le:
                    bucket.count += 1

    def get_percentile(self, p: float) -> float:
        """Get percentile value (0-100)"""
        with self._lock:
            if not self._values:
                return 0.0
            sorted_vals = sorted(self._values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

    @property
    def p50(self) -> float:
        return self.get_percentile(50)

    @property
    def p95(self) -> float:
        return self.get_percentile(95)

    @property
    def p99(self) -> float:
        return self.get_percentile(99)

    @property
    def avg(self) -> float:
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0

    def to_prometheus(self) -> List[str]:
        """Export in Prometheus format"""
        lines = []
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        base_labels = f"{{{label_str}}}" if label_str else ""

        for bucket in self.buckets:
            le_label = f'le="{bucket.le}"' if bucket.le != float('inf') else 'le="+Inf"'
            full_labels = f"{{{label_str},{le_label}}}" if label_str else f"{{{le_label}}}"
            lines.append(f'{self.name}_bucket{full_labels} {bucket.count}')

        lines.append(f'{self.name}_sum{base_labels} {self._sum}')
        lines.append(f'{self.name}_count{base_labels} {self._count}')
        return lines


# =============================================================================
# CORRELATION CONTEXT - Distributed Tracing
# =============================================================================

@dataclass
class CorrelationContext:
    """
    Correlation context for distributed tracing.
    Flows through all module calls.
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    tenant_id: str = ""
    source_module: str = ""
    parent_span_id: Optional[str] = None
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def child_span(self, module: str) -> "CorrelationContext":
        """Create child span for nested calls"""
        return CorrelationContext(
            request_id=self.request_id,
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            source_module=module,
            parent_span_id=self.span_id,
            metadata=self.metadata.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "source_module": self.source_module,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time.isoformat(),
            "metadata": self.metadata
        }


# Thread-local storage for current context
_context_local = threading.local()


def get_current_context() -> Optional[CorrelationContext]:
    """Get current correlation context from thread-local storage"""
    return getattr(_context_local, 'context', None)


def set_current_context(ctx: CorrelationContext):
    """Set current correlation context in thread-local storage"""
    _context_local.context = ctx


@contextmanager
def traced_operation(name: str, module: str):
    """Context manager for traced operations"""
    parent_ctx = get_current_context()
    if parent_ctx:
        ctx = parent_ctx.child_span(module)
    else:
        ctx = CorrelationContext(source_module=module)

    set_current_context(ctx)
    start_time = time.time()
    status = "ok"

    try:
        yield ctx
    except Exception as e:
        status = "error"
        raise
    finally:
        duration_ms = (time.time() - start_time) * 1000
        MetricsRegistry.get_instance().observe_histogram(
            f"{module}_operation_duration_ms",
            duration_ms,
            {"operation": name}
        )

        # Persist trace span to database
        persistence = ObservabilityPersistence.get_instance()
        if persistence:
            persistence.record_trace(
                request_id=ctx.request_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id or "",
                operation=name,
                module=module,
                duration_ms=duration_ms,
                status=status
            )

        if parent_ctx:
            set_current_context(parent_ctx)


# =============================================================================
# EVENT BUS - Cross-Module Communication
# =============================================================================

class EventType(Enum):
    # OODA Events
    OBSERVATION_COMPLETE = "ooda.observation_complete"
    DECISION_MADE = "ooda.decision_made"
    ACTION_EXECUTED = "ooda.action_executed"

    # Hallucination Events
    VALIDATION_COMPLETE = "hallucination.validation_complete"
    HALLUCINATION_DETECTED = "hallucination.detected"
    CLAIM_VERIFIED = "hallucination.claim_verified"

    # Memory Events
    MEMORY_STORED = "memory.stored"
    MEMORY_RETRIEVED = "memory.retrieved"
    CONTRADICTION_DETECTED = "memory.contradiction_detected"
    WISDOM_CRYSTALLIZED = "memory.wisdom_crystallized"

    # Consciousness Events
    AWARENESS_LEVEL_CHANGED = "consciousness.awareness_changed"
    INTENTION_GENERATED = "consciousness.intention_generated"
    VALUE_VIOLATION = "consciousness.value_violation"

    # Dependability Events
    GUARD_VIOLATION = "dependability.guard_violation"
    STATE_TRANSITION = "dependability.state_transition"
    RECOVERY_ATTEMPTED = "dependability.recovery_attempted"

    # Circuit Breaker Events
    CIRCUIT_STATE_CHANGED = "circuit.state_changed"
    DEADLOCK_DETECTED = "circuit.deadlock_detected"
    CASCADE_TRIGGERED = "circuit.cascade_triggered"

    # System Events
    HEALTH_CHECK = "system.health_check"
    ERROR_OCCURRED = "system.error"
    METRIC_THRESHOLD_EXCEEDED = "system.metric_threshold"


@dataclass
class Event:
    """Event for cross-module communication"""
    event_type: EventType
    source_module: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source_module": self.source_module,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id
        }


class EventBus:
    """
    Central event bus for cross-module communication.
    Supports sync and async handlers.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EventBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=1000)
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._subscribers: Set[Callable] = set()  # Catch-all subscribers
        self._dead_letter_queue: deque = deque(maxlen=100)
        self._lock = threading.Lock()

    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to specific event type"""
        with self._lock:
            self._handlers[event_type].append(handler)

    def subscribe_async(self, event_type: EventType, handler: Callable):
        """Subscribe async handler to event type"""
        with self._lock:
            self._async_handlers[event_type].append(handler)

    def subscribe_all(self, handler: Callable):
        """Subscribe to all events"""
        with self._lock:
            self._subscribers.add(handler)

    def publish(self, event: Event):
        """Publish event synchronously"""
        ctx = get_current_context()
        if ctx:
            event.correlation_id = ctx.request_id

        with self._lock:
            self._event_history.append(event)
            self._event_counts[event.event_type.value] += 1

        # Call sync handlers
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type}: {e}")
                self._dead_letter_queue.append((event, str(e)))

        # Call catch-all subscribers
        for handler in self._subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Subscriber error for {event.event_type}: {e}")

        # Track metric
        MetricsRegistry.get_instance().increment_counter(
            "event_bus_events_total",
            {"event_type": event.event_type.value, "source": event.source_module}
        )

        # Persist to database
        persistence = ObservabilityPersistence.get_instance()
        if persistence:
            persistence.record_event(
                event_type=event.event_type.value,
                source_module=event.source_module,
                payload=event.payload,
                correlation_id=event.correlation_id
            )

    async def publish_async(self, event: Event):
        """Publish event asynchronously"""
        ctx = get_current_context()
        if ctx:
            event.correlation_id = ctx.request_id

        with self._lock:
            self._event_history.append(event)
            self._event_counts[event.event_type.value] += 1

        # Call async handlers
        handlers = self._async_handlers.get(event.event_type, [])
        if handlers:
            results = await asyncio.gather(
                *[handler(event) for handler in handlers],
                return_exceptions=True
            )
            # Log any exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Async handler error for {event.event_type}: {result}")
                    self._dead_letter_queue.append((event, str(result)))

        # Call sync handlers in executor
        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
                self._dead_letter_queue.append((event, str(e)))

        # Call catch-all subscribers (was missing in async version)
        for handler in self._subscribers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Subscriber error for {event.event_type}: {e}")

        # Track metric (was missing in async version)
        MetricsRegistry.get_instance().increment_counter(
            "event_bus_events_total",
            {"event_type": event.event_type.value, "source": event.source_module, "async": "true"}
        )

        # Persist to database
        persistence = ObservabilityPersistence.get_instance()
        if persistence:
            persistence.record_event(
                event_type=event.event_type.value,
                source_module=event.source_module,
                payload=event.payload,
                correlation_id=event.correlation_id
            )

    def get_event_counts(self) -> Dict[str, int]:
        """Get event counts by type"""
        with self._lock:
            return dict(self._event_counts)

    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent events"""
        with self._lock:
            return [e.to_dict() for e in list(self._event_history)[-limit:]]

    def get_dead_letters(self) -> List[Tuple[Dict, str]]:
        """Get failed events"""
        with self._lock:
            return [(e.to_dict(), err) for e, err in self._dead_letter_queue]


# =============================================================================
# METRICS REGISTRY - Central Metrics Collection
# =============================================================================

class MetricsRegistry:
    """
    Central registry for all metrics across modules.
    Provides Prometheus-compatible export.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "MetricsRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._counters: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Histogram] = {}
        self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._thresholds: Dict[str, Tuple[float, float]] = {}  # min, max
        self._threshold_handlers: List[Callable] = []
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)

    def increment_counter(self, name: str, labels: Dict[str, str] = None, value: float = 1.0):
        """Increment a counter"""
        label_key = self._labels_to_key(labels or {})
        with self._lock:
            self._counters[name][label_key] += value
            self._check_threshold(name, self._counters[name][label_key])

    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge value"""
        label_key = self._labels_to_key(labels or {})
        with self._lock:
            self._gauges[name][label_key] = value
            self._metric_history[f"{name}:{label_key}"].append((datetime.now(timezone.utc), value))
            self._check_threshold(name, value)

    def observe_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram observation"""
        label_key = self._labels_to_key(labels or {})
        full_name = f"{name}:{label_key}"

        with self._lock:
            if full_name not in self._histograms:
                self._histograms[full_name] = Histogram(name, labels)
            self._histograms[full_name].observe(value)

    def get_histogram(self, name: str, labels: Dict[str, str] = None) -> Optional[Histogram]:
        """Get histogram by name and optional labels"""
        label_key = self._labels_to_key(labels or {})
        full_name = f"{name}:{label_key}"
        return self._histograms.get(full_name)

    def get_histograms_by_name(self, name: str) -> List[Histogram]:
        """Get all histograms matching a base name (ignoring labels)"""
        with self._lock:
            matching = []
            for full_name, histogram in self._histograms.items():
                if histogram.name == name or full_name.startswith(f"{name}:"):
                    matching.append(histogram)
            return matching

    def set_threshold(self, name: str, min_val: float = None, max_val: float = None):
        """Set threshold for metric alerts"""
        self._thresholds[name] = (min_val, max_val)

    def add_threshold_handler(self, handler: Callable):
        """Add handler for threshold violations"""
        self._threshold_handlers.append(handler)

    def _check_threshold(self, name: str, value: float):
        """Check if value exceeds threshold"""
        if name in self._thresholds:
            min_val, max_val = self._thresholds[name]
            violated = False
            if min_val is not None and value < min_val:
                violated = True
            if max_val is not None and value > max_val:
                violated = True

            if violated:
                for handler in self._threshold_handlers:
                    try:
                        handler(name, value, self._thresholds[name])
                    except Exception as e:
                        logger.error(f"Threshold handler error: {e}")

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """Convert labels dict to hashable key"""
        return json.dumps(labels, sort_keys=True)

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """Convert key back to labels dict"""
        return json.loads(key) if key else {}

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus format"""
        lines = []

        # Counters
        for name, values in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            for label_key, value in values.items():
                labels = self._key_to_labels(label_key)
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    lines.append(f'{name}{{{label_str}}} {value}')
                else:
                    lines.append(f'{name} {value}')

        # Gauges
        for name, values in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            for label_key, value in values.items():
                labels = self._key_to_labels(label_key)
                label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
                if label_str:
                    lines.append(f'{name}{{{label_str}}} {value}')
                else:
                    lines.append(f'{name} {value}')

        # Histograms
        for full_name, histogram in self._histograms.items():
            lines.append(f"# TYPE {histogram.name} histogram")
            lines.extend(histogram.to_prometheus())

        return "\n".join(lines)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary"""
        with self._lock:
            metrics = {
                "counters": {},
                "gauges": {},
                "histograms": {},
                "uptime_seconds": (datetime.now(timezone.utc) - self._start_time).total_seconds()
            }

            # Counters - group by name, then by label key string
            for name, values in self._counters.items():
                if name not in metrics["counters"]:
                    metrics["counters"][name] = {}
                for label_key, value in values.items():
                    # Use the label key string directly, or "default" for empty labels
                    key_str = label_key if label_key and label_key != "{}" else "default"
                    metrics["counters"][name][key_str] = value

            # Gauges - same approach
            for name, values in self._gauges.items():
                if name not in metrics["gauges"]:
                    metrics["gauges"][name] = {}
                for label_key, value in values.items():
                    key_str = label_key if label_key and label_key != "{}" else "default"
                    metrics["gauges"][name][key_str] = value

            # Histograms - group by base name with labels preserved
            for full_name, histogram in self._histograms.items():
                base_name = histogram.name
                if base_name not in metrics["histograms"]:
                    metrics["histograms"][base_name] = {}

                # Use labels as key identifier
                label_key = json.dumps(histogram.labels, sort_keys=True) if histogram.labels else "default"
                metrics["histograms"][base_name][label_key] = {
                    "labels": histogram.labels,
                    "count": histogram._count,
                    "sum": histogram._sum,
                    "avg": histogram.avg,
                    "p50": histogram.p50,
                    "p95": histogram.p95,
                    "p99": histogram.p99
                }

            return metrics

    def get_metric_history(self, name: str, labels: Dict[str, str] = None,
                          since: datetime = None) -> List[Tuple[datetime, float]]:
        """Get metric history for time-series analysis"""
        label_key = self._labels_to_key(labels or {})
        full_name = f"{name}:{label_key}"

        with self._lock:
            history = list(self._metric_history.get(full_name, []))
            if since:
                history = [(ts, v) for ts, v in history if ts >= since]
            return history


# =============================================================================
# MODULE-SPECIFIC METRICS
# =============================================================================

class OODAMetrics:
    """OODA-specific metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def record_observation(self, duration_ms: float, source: str):
        self.registry.observe_histogram("ooda_observation_duration_ms", duration_ms, {"source": source})
        self.registry.increment_counter("ooda_observations_total", {"source": source})

    def record_decision(self, confidence: float, decision_type: str):
        self.registry.observe_histogram("ooda_decision_confidence", confidence, {"type": decision_type})
        self.registry.increment_counter("ooda_decisions_total", {"type": decision_type})

    def record_action(self, duration_ms: float, success: bool, action_type: str):
        self.registry.observe_histogram("ooda_action_duration_ms", duration_ms, {"type": action_type})
        self.registry.increment_counter("ooda_actions_total", {"type": action_type, "success": str(success)})

    def set_cycle_latency(self, latency_ms: float):
        self.registry.set_gauge("ooda_cycle_latency_ms", latency_ms)


class HallucinationMetrics:
    """Hallucination prevention metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def record_validation(self, duration_ms: float, result: str, method: str):
        self.registry.observe_histogram("hallucination_validation_duration_ms", duration_ms, {"method": method})
        self.registry.increment_counter("hallucination_validations_total", {"result": result, "method": method})

    def record_detection(self, hallucination_type: str, confidence: float):
        self.registry.increment_counter("hallucinations_detected_total", {"type": hallucination_type})
        self.registry.observe_histogram("hallucination_detection_confidence", confidence, {"type": hallucination_type})

    def set_cache_hit_rate(self, rate: float):
        self.registry.set_gauge("hallucination_cache_hit_rate", rate)

    def record_model_call(self, model: str, duration_ms: float, success: bool):
        self.registry.observe_histogram("hallucination_model_call_duration_ms", duration_ms, {"model": model})
        self.registry.increment_counter("hallucination_model_calls_total", {"model": model, "success": str(success)})


class MemoryMetrics:
    """Memory brain metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def record_store(self, duration_ms: float, memory_type: str):
        self.registry.observe_histogram("memory_store_duration_ms", duration_ms, {"type": memory_type})
        self.registry.increment_counter("memory_stores_total", {"type": memory_type})

    def record_retrieve(self, duration_ms: float, hit: bool):
        self.registry.observe_histogram("memory_retrieve_duration_ms", duration_ms)
        self.registry.increment_counter("memory_retrieves_total", {"hit": str(hit)})

    def set_memory_count(self, count: int, memory_type: str):
        self.registry.set_gauge("memory_count", count, {"type": memory_type})

    def record_contradiction(self, resolution: str):
        self.registry.increment_counter("memory_contradictions_total", {"resolution": resolution})

    def record_prediction(self, accurate: bool):
        self.registry.increment_counter("memory_predictions_total", {"accurate": str(accurate)})


class ConsciousnessMetrics:
    """Consciousness emergence metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def set_awareness_level(self, level: str, numeric: float):
        self.registry.set_gauge("consciousness_awareness_level", numeric, {"level": level})

    def set_consciousness_level(self, level: float):
        self.registry.set_gauge("consciousness_level", level)

    def record_intention(self, intention_type: str):
        self.registry.increment_counter("consciousness_intentions_total", {"type": intention_type})

    def record_value_alignment(self, aligned: bool, value: str):
        self.registry.increment_counter("consciousness_value_checks_total", {"aligned": str(aligned), "value": value})

    def record_experience(self, duration_ms: float):
        self.registry.observe_histogram("consciousness_experience_duration_ms", duration_ms)


class DependabilityMetrics:
    """Dependability framework metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def record_guard_check(self, guard_type: str, duration_ms: float, passed: bool):
        self.registry.observe_histogram("dependability_guard_duration_ms", duration_ms, {"guard": guard_type})
        self.registry.increment_counter("dependability_guard_checks_total", {"guard": guard_type, "passed": str(passed)})

    def set_system_state(self, state: str, numeric: int):
        self.registry.set_gauge("dependability_system_state", numeric, {"state": state})

    def record_recovery(self, success: bool, action: str):
        self.registry.increment_counter("dependability_recoveries_total", {"success": str(success), "action": action})

    def set_health_score(self, score: float):
        self.registry.set_gauge("dependability_health_score", score)


class CircuitBreakerMetrics:
    """Circuit breaker metrics"""
    def __init__(self):
        self.registry = MetricsRegistry.get_instance()

    def set_circuit_state(self, component: str, state: str, numeric: int):
        self.registry.set_gauge("circuit_breaker_state", numeric, {"component": component, "state": state})

    def record_request(self, component: str, duration_ms: float, success: bool):
        self.registry.observe_histogram("circuit_breaker_request_duration_ms", duration_ms, {"component": component})
        self.registry.increment_counter("circuit_breaker_requests_total", {"component": component, "success": str(success)})

    def set_failure_rate(self, component: str, rate: float):
        self.registry.set_gauge("circuit_breaker_failure_rate", rate, {"component": component})

    def set_health_score(self, component: str, score: float):
        self.registry.set_gauge("circuit_breaker_health_score", score, {"component": component})

    def record_state_transition(self, component: str, from_state: str, to_state: str):
        self.registry.increment_counter("circuit_breaker_transitions_total",
                                        {"component": component, "from": from_state, "to": to_state})


# =============================================================================
# OBSERVABILITY CONTROLLER
# =============================================================================

class ObservabilityController:
    """
    Central controller for all observability features.
    Aggregates metrics, manages events, provides dashboards.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ObservabilityController":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.metrics = MetricsRegistry.get_instance()
        self.events = EventBus.get_instance()

        # Module-specific metrics
        self.ooda = OODAMetrics()
        self.hallucination = HallucinationMetrics()
        self.memory = MemoryMetrics()
        self.consciousness = ConsciousnessMetrics()
        self.dependability = DependabilityMetrics()
        self.circuit_breaker = CircuitBreakerMetrics()

        # Anomaly detection
        self._anomaly_thresholds: Dict[str, Dict] = {}
        self._anomaly_handlers: List[Callable] = []

        # Dashboard state
        self._dashboard_data: Dict[str, Any] = {}
        self._last_dashboard_update = datetime.now(timezone.utc)

        logger.info("ObservabilityController initialized")

    def get_prometheus_metrics(self) -> str:
        """Get all metrics in Prometheus format"""
        return self.metrics.export_prometheus()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get aggregated data for dashboard"""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "events": {
                "counts": self.events.get_event_counts(),
                "recent": self.events.get_recent_events(50),
                "dead_letters": len(self.events.get_dead_letters())
            },
            "health": self._calculate_overall_health(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall system health from metrics"""
        all_metrics = self.metrics.get_all_metrics()

        health_factors = []

        # Check circuit breaker health scores
        for key, value in all_metrics.get("gauges", {}).items():
            if "health_score" in key:
                if isinstance(value, dict):
                    for v in value.values():
                        health_factors.append(v / 100.0)
                else:
                    health_factors.append(value / 100.0)

        # Check error rates
        total_requests = 0
        failed_requests = 0
        for key, value in all_metrics.get("counters", {}).items():
            if "requests_total" in key or "calls_total" in key:
                if isinstance(value, dict):
                    for label_key, count in value.items():
                        total_requests += count
                        if "false" in str(label_key).lower():
                            failed_requests += count

        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        health_factors.append(1.0 - error_rate)

        overall = sum(health_factors) / len(health_factors) if health_factors else 1.0

        return {
            "overall_score": round(overall * 100, 2),
            "status": "healthy" if overall >= 0.8 else "degraded" if overall >= 0.5 else "critical",
            "factors": len(health_factors),
            "error_rate": round(error_rate * 100, 2)
        }

    def add_anomaly_detection(self, metric_name: str,
                               min_threshold: float = None,
                               max_threshold: float = None,
                               std_dev_threshold: float = None):
        """Add anomaly detection for a metric"""
        self._anomaly_thresholds[metric_name] = {
            "min": min_threshold,
            "max": max_threshold,
            "std_dev": std_dev_threshold
        }

        if min_threshold is not None or max_threshold is not None:
            self.metrics.set_threshold(metric_name, min_threshold, max_threshold)

    def on_anomaly(self, handler: Callable):
        """Register anomaly handler"""
        self._anomaly_handlers.append(handler)
        self.metrics.add_threshold_handler(
            lambda name, val, thresh: self._handle_anomaly(name, val, thresh)
        )

    def _handle_anomaly(self, metric_name: str, value: float, thresholds: Tuple[float, float]):
        """Handle detected anomaly"""
        for handler in self._anomaly_handlers:
            try:
                handler(metric_name, value, thresholds)
            except Exception as e:
                logger.error(f"Anomaly handler error: {e}")

        # Publish event
        self.events.publish(Event(
            event_type=EventType.METRIC_THRESHOLD_EXCEEDED,
            source_module="observability",
            payload={
                "metric": metric_name,
                "value": value,
                "thresholds": {"min": thresholds[0], "max": thresholds[1]}
            }
        ))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_observability() -> ObservabilityController:
    """Get the global observability controller"""
    return ObservabilityController.get_instance()


def get_event_bus() -> EventBus:
    """Get the global event bus"""
    return EventBus.get_instance()


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry"""
    return MetricsRegistry.get_instance()


def publish_event(event_type: EventType, source: str, payload: Dict[str, Any]):
    """Convenience function to publish event"""
    EventBus.get_instance().publish(Event(
        event_type=event_type,
        source_module=source,
        payload=payload
    ))


async def publish_event_async(event_type: EventType, source: str, payload: Dict[str, Any]):
    """Convenience function to publish event asynchronously"""
    await EventBus.get_instance().publish_async(Event(
        event_type=event_type,
        source_module=source,
        payload=payload
    ))


# Export all public classes and functions
def get_persistence() -> Optional[ObservabilityPersistence]:
    """Get the global observability persistence instance"""
    return ObservabilityPersistence.get_instance()


def flush_persistence():
    """Flush all pending observability data to database"""
    persistence = ObservabilityPersistence.get_instance()
    if persistence:
        persistence.flush()


__all__ = [
    # Core classes
    "MetricsRegistry",
    "EventBus",
    "ObservabilityController",
    "ObservabilityPersistence",
    "CorrelationContext",
    "Event",
    "EventType",
    "Histogram",

    # Module metrics
    "OODAMetrics",
    "HallucinationMetrics",
    "MemoryMetrics",
    "ConsciousnessMetrics",
    "DependabilityMetrics",
    "CircuitBreakerMetrics",

    # Convenience functions
    "get_observability",
    "get_event_bus",
    "get_metrics",
    "get_persistence",
    "flush_persistence",
    "publish_event",
    "publish_event_async",
    "traced_operation",
    "get_current_context",
    "set_current_context"
]
