#!/usr/bin/env python3
"""
Monitoring Dashboard Data Provider
System monitoring, metrics collection, and dashboard data API
"""

import asyncio
import json
import logging
import os
from urllib.parse import urlparse as _urlparse
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServiceHealth(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class TimeRange(Enum):
    """Predefined time ranges"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MetricValue:
    """A single metric value"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class MetricSeries:
    """Time series of metric values"""
    name: str
    values: list[tuple[datetime, float]] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    aggregation: str = "avg"  # avg, sum, min, max, count


@dataclass
class ServiceStatus:
    """Status of a service"""
    service_name: str
    health: ServiceHealth
    uptime_seconds: float
    last_check: datetime
    response_time_ms: float
    error_rate: float
    request_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardAlert:
    """Dashboard alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    resolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardPanel:
    """A dashboard panel configuration"""
    panel_id: str
    title: str
    panel_type: str  # chart, stat, table, gauge, alert
    metrics: list[str]
    query: Optional[str] = None
    refresh_interval: int = 60  # seconds
    options: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# MONITORING DASHBOARD PROVIDER
# ============================================================================

class MonitoringDashboard:
    """
    Provides monitoring data for dashboards
    Collects metrics, tracks health, and generates alerts
    """

    def __init__(self):
        self._initialized = False
        self._db_config = None

        # Metrics storage
        self._metrics: dict[str, list[MetricValue]] = defaultdict(list)
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)

        # Service tracking
        self._services: dict[str, ServiceStatus] = {}
        self._service_history: dict[str, list[ServiceStatus]] = defaultdict(list)

        # Alerts
        self._alerts: list[DashboardAlert] = []
        self._alert_rules: dict[str, dict[str, Any]] = {}

        # Dashboard configuration
        self._panels: dict[str, DashboardPanel] = {}

        # Cache
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = 60  # seconds

        self._lock = asyncio.Lock()
        self._collection_task: Optional[asyncio.Task] = None

    def _get_db_config(self) -> dict[str, Any]:
        """Get database configuration lazily with validation"""
        if not self._db_config:
            required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                
        # DATABASE_URL fallback
        _db_url = os.getenv('DATABASE_URL', '')
        if _db_url:
            try:
                _p = _urlparse(_db_url)
                globals().update({'_DB_HOST': _p.hostname, '_DB_NAME': _p.path.lstrip('/'), '_DB_USER': _p.username, '_DB_PASSWORD': _p.password, '_DB_PORT': str(_p.port or 5432)})
            except: pass
        missing = [v for v in required_vars if not os.getenv(v) and not globals().get('_' + v)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

            self._db_config = {
                'host': os.getenv('DB_HOST'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', '5432'))
            }
        return self._db_config

    async def initialize(self):
        """Initialize the monitoring dashboard"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                await self._initialize_database()
                await self._setup_default_panels()
                await self._setup_alert_rules()
                self._collection_task = asyncio.create_task(self._collection_loop())
                self._initialized = True
                logger.info("Monitoring dashboard initialized")
            except Exception as e:
                logger.error(f"Failed to initialize monitoring dashboard: {e}")

    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_dashboard_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    metric_name VARCHAR(255) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    metric_type VARCHAR(50) DEFAULT 'gauge',
                    labels JSONB DEFAULT '{}'::jsonb,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Service status table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_dashboard_services (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    service_name VARCHAR(255) NOT NULL,
                    health VARCHAR(50) NOT NULL,
                    uptime_seconds FLOAT DEFAULT 0,
                    last_check TIMESTAMPTZ DEFAULT NOW(),
                    response_time_ms FLOAT DEFAULT 0,
                    error_rate FLOAT DEFAULT 0,
                    request_count INT DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_dashboard_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    alert_id VARCHAR(255) UNIQUE NOT NULL,
                    severity VARCHAR(50) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    message TEXT NOT NULL,
                    source VARCHAR(255),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    acknowledged BOOLEAN DEFAULT false,
                    acknowledged_at TIMESTAMPTZ,
                    resolved BOOLEAN DEFAULT false,
                    resolved_at TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Dashboard panels table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_dashboard_panels (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    panel_id VARCHAR(255) UNIQUE NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    panel_type VARCHAR(50) NOT NULL,
                    metrics JSONB DEFAULT '[]'::jsonb,
                    query TEXT,
                    refresh_interval INT DEFAULT 60,
                    options JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_dashboard_metrics_name_time
                ON ai_dashboard_metrics(metric_name, timestamp DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_dashboard_alerts_severity
                ON ai_dashboard_alerts(severity, created_at DESC)
            """)

            # Create partition for metrics (optional, for high-volume)
            # This is a simplified version - production would use proper partitioning

            conn.commit()
            conn.close()
            logger.info("Monitoring dashboard tables initialized")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    async def _setup_default_panels(self):
        """Set up default dashboard panels"""
        default_panels = [
            DashboardPanel(
                panel_id="system_health",
                title="System Health",
                panel_type="gauge",
                metrics=["system.health_score"],
                options={"thresholds": [50, 80, 95]}
            ),
            DashboardPanel(
                panel_id="agent_status",
                title="Agent Status",
                panel_type="stat",
                metrics=["agents.active", "agents.total", "agents.busy"],
                options={"layout": "horizontal"}
            ),
            DashboardPanel(
                panel_id="task_throughput",
                title="Task Throughput",
                panel_type="chart",
                metrics=["tasks.completed", "tasks.failed"],
                options={"chartType": "line", "timeRange": "24h"}
            ),
            DashboardPanel(
                panel_id="error_rate",
                title="Error Rate",
                panel_type="chart",
                metrics=["errors.rate", "errors.count"],
                options={"chartType": "area", "fill": True}
            ),
            DashboardPanel(
                panel_id="response_times",
                title="Response Times",
                panel_type="histogram",
                metrics=["response.time_ms"],
                options={"buckets": [10, 50, 100, 250, 500, 1000]}
            ),
            DashboardPanel(
                panel_id="active_alerts",
                title="Active Alerts",
                panel_type="alert",
                metrics=[],
                options={"showResolved": False}
            ),
            DashboardPanel(
                panel_id="database_stats",
                title="Database Statistics",
                panel_type="table",
                metrics=["db.connections", "db.size_mb", "db.queries_per_sec"],
                options={"columns": ["Metric", "Value", "Change"]}
            ),
            DashboardPanel(
                panel_id="memory_usage",
                title="Memory Usage",
                panel_type="gauge",
                metrics=["memory.usage_percent"],
                options={"max": 100, "unit": "%"}
            )
        ]

        for panel in default_panels:
            self._panels[panel.panel_id] = panel

    async def _setup_alert_rules(self):
        """Set up default alert rules"""
        self._alert_rules = {
            "high_error_rate": {
                "metric": "errors.rate",
                "condition": "gt",
                "threshold": 0.1,
                "severity": AlertSeverity.ERROR,
                "message": "Error rate exceeded 10%"
            },
            "low_health_score": {
                "metric": "system.health_score",
                "condition": "lt",
                "threshold": 50,
                "severity": AlertSeverity.CRITICAL,
                "message": "System health score below 50%"
            },
            "high_response_time": {
                "metric": "response.time_ms",
                "condition": "gt",
                "threshold": 5000,
                "severity": AlertSeverity.WARNING,
                "message": "Response time exceeded 5 seconds"
            },
            "database_connection_limit": {
                "metric": "db.connections",
                "condition": "gt",
                "threshold": 80,
                "severity": AlertSeverity.WARNING,
                "message": "Database connections approaching limit"
            },
            "high_memory_usage": {
                "metric": "memory.usage_percent",
                "condition": "gt",
                "threshold": 90,
                "severity": AlertSeverity.CRITICAL,
                "message": "Memory usage exceeded 90%"
            }
        }

    async def _collection_loop(self):
        """Background loop to collect metrics"""
        while True:
            try:
                await asyncio.sleep(60)
                await self._collect_system_metrics()
                await self._check_alert_rules()
                await self._cleanup_old_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")

    # ========================================================================
    # METRIC COLLECTION
    # ========================================================================

    async def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[dict[str, str]] = None
    ):
        """Record a metric value"""
        await self.initialize()

        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {}
        )

        self._metrics[name].append(metric)

        # Update quick access structures
        if metric_type == MetricType.COUNTER:
            self._counters[name] += value
        elif metric_type == MetricType.GAUGE:
            self._gauges[name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self._histograms[name].append(value)

        # Keep only recent values in memory
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]

        # Persist periodically
        if len(self._metrics[name]) % 100 == 0:
            await self._persist_metrics(name)

    async def increment_counter(self, name: str, value: float = 1, labels: Optional[dict[str, str]] = None):
        """Increment a counter metric"""
        await self.record_metric(name, value, MetricType.COUNTER, labels)

    async def set_gauge(self, name: str, value: float, labels: Optional[dict[str, str]] = None):
        """Set a gauge metric"""
        await self.record_metric(name, value, MetricType.GAUGE, labels)

    async def observe_histogram(self, name: str, value: float, labels: Optional[dict[str, str]] = None):
        """Record a histogram observation"""
        await self.record_metric(name, value, MetricType.HISTOGRAM, labels)

    async def get_metric(self, name: str, time_range: TimeRange = TimeRange.LAST_HOUR) -> Optional[MetricSeries]:
        """Get metric time series"""
        if name not in self._metrics:
            return None

        # Calculate cutoff time
        now = datetime.now(timezone.utc)
        cutoff = self._get_cutoff_time(time_range)

        # Filter values
        values = [
            (m.timestamp, m.value)
            for m in self._metrics[name]
            if m.timestamp >= cutoff
        ]

        if not values:
            return None

        return MetricSeries(
            name=name,
            values=values,
            labels=self._metrics[name][-1].labels if self._metrics[name] else {}
        )

    async def get_metric_value(self, name: str) -> Optional[float]:
        """Get current metric value"""
        if name in self._gauges:
            return self._gauges[name]
        elif name in self._counters:
            return self._counters[name]
        elif name in self._metrics and self._metrics[name]:
            return self._metrics[name][-1].value
        return None

    async def get_metric_stats(
        self,
        name: str,
        time_range: TimeRange = TimeRange.LAST_HOUR
    ) -> dict[str, float]:
        """Get statistical summary of a metric"""
        series = await self.get_metric(name, time_range)
        if not series or not series.values:
            return {}

        values = [v[1] for v in series.values]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "sum": sum(values),
            "latest": values[-1] if values else 0
        }

    async def _persist_metrics(self, name: str):
        """Persist metrics to database"""
        try:
            import psycopg2

            if name not in self._metrics:
                return

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Get last 100 metrics to persist
            metrics_to_persist = self._metrics[name][-100:]

            for metric in metrics_to_persist:
                cur.execute("""
                    INSERT INTO ai_dashboard_metrics (
                        metric_name, metric_value, metric_type, labels, timestamp
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    metric.name,
                    metric.value,
                    metric.metric_type.value,
                    json.dumps(metric.labels),
                    metric.timestamp
                ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")

    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Database statistics
            cur.execute("SELECT COUNT(*) as count FROM pg_stat_activity")
            result = cur.fetchone()
            await self.set_gauge("db.connections", result['count'])

            cur.execute("SELECT pg_database_size('postgres') / 1024 / 1024 as size_mb")
            result = cur.fetchone()
            await self.set_gauge("db.size_mb", result['size_mb'])

            # Agent statistics
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active
                FROM ai_agents
            """)
            result = cur.fetchone()
            await self.set_gauge("agents.total", result['total'] or 0)
            await self.set_gauge("agents.active", result['active'] or 0)

            # Task statistics
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            result = cur.fetchone()
            await self.set_gauge("tasks.total_hour", result['total'] or 0)
            await self.set_gauge("tasks.completed", result['completed'] or 0)
            await self.set_gauge("tasks.failed", result['failed'] or 0)

            # Calculate error rate
            total = (result['completed'] or 0) + (result['failed'] or 0)
            if total > 0:
                error_rate = (result['failed'] or 0) / total
                await self.set_gauge("errors.rate", error_rate)

            # Error count
            cur.execute("""
                SELECT COUNT(*) as count
                FROM ai_error_events
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            result = cur.fetchone()
            await self.set_gauge("errors.count", result['count'] or 0)

            conn.close()

            # Calculate overall health score
            health_score = await self._calculate_health_score()
            await self.set_gauge("system.health_score", health_score)

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        weights = {
            "error_rate": 30,
            "response_time": 20,
            "agent_availability": 25,
            "database_health": 25
        }

        # Error rate impact
        error_rate = await self.get_metric_value("errors.rate") or 0
        if error_rate > 0.1:
            score -= weights["error_rate"]
        elif error_rate > 0.05:
            score -= weights["error_rate"] * 0.5

        # Response time impact (would need actual response time tracking)
        response_time = await self.get_metric_value("response.time_ms") or 0
        if response_time > 5000:
            score -= weights["response_time"]
        elif response_time > 1000:
            score -= weights["response_time"] * 0.5

        # Agent availability
        total_agents = await self.get_metric_value("agents.total") or 1
        active_agents = await self.get_metric_value("agents.active") or 0
        agent_ratio = active_agents / max(total_agents, 1)
        if agent_ratio < 0.5:
            score -= weights["agent_availability"]
        elif agent_ratio < 0.8:
            score -= weights["agent_availability"] * 0.5

        # Database health (connection usage)
        connections = await self.get_metric_value("db.connections") or 0
        if connections > 90:
            score -= weights["database_health"]
        elif connections > 70:
            score -= weights["database_health"] * 0.5

        return max(0, min(100, score))

    async def _cleanup_old_metrics(self):
        """Clean up old metrics from memory and database"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        # Clean memory
        for name in list(self._metrics.keys()):
            self._metrics[name] = [
                m for m in self._metrics[name]
                if m.timestamp >= cutoff
            ]
            if not self._metrics[name]:
                del self._metrics[name]

        # Clean database
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                DELETE FROM ai_dashboard_metrics
                WHERE timestamp < NOW() - INTERVAL '30 days'
            """)

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to cleanup old metrics: {e}")

    def _get_cutoff_time(self, time_range: TimeRange) -> datetime:
        """Get cutoff time for a time range"""
        now = datetime.now(timezone.utc)
        deltas = {
            TimeRange.LAST_HOUR: timedelta(hours=1),
            TimeRange.LAST_6_HOURS: timedelta(hours=6),
            TimeRange.LAST_24_HOURS: timedelta(hours=24),
            TimeRange.LAST_7_DAYS: timedelta(days=7),
            TimeRange.LAST_30_DAYS: timedelta(days=30)
        }
        return now - deltas.get(time_range, timedelta(hours=1))

    # ========================================================================
    # SERVICE MONITORING
    # ========================================================================

    async def update_service_status(
        self,
        service_name: str,
        health: ServiceHealth,
        response_time_ms: float = 0,
        error_rate: float = 0,
        request_count: int = 0,
        metadata: Optional[dict[str, Any]] = None
    ):
        """Update service status"""
        await self.initialize()

        # Get or create service status
        if service_name not in self._services:
            self._services[service_name] = ServiceStatus(
                service_name=service_name,
                health=health,
                uptime_seconds=0,
                last_check=datetime.now(timezone.utc),
                response_time_ms=response_time_ms,
                error_rate=error_rate,
                request_count=request_count,
                metadata=metadata or {}
            )
        else:
            status = self._services[service_name]
            previous_check = status.last_check
            current_time = datetime.now(timezone.utc)

            # Update uptime if healthy
            if status.health == ServiceHealth.HEALTHY:
                status.uptime_seconds += (current_time - previous_check).total_seconds()

            status.health = health
            status.last_check = current_time
            status.response_time_ms = response_time_ms
            status.error_rate = error_rate
            status.request_count = request_count
            if metadata:
                status.metadata.update(metadata)

        # Store in history
        self._service_history[service_name].append(self._services[service_name])

        # Keep history limited
        if len(self._service_history[service_name]) > 100:
            self._service_history[service_name] = self._service_history[service_name][-100:]

        # Record metrics
        await self.set_gauge(f"service.{service_name}.response_time", response_time_ms)
        await self.set_gauge(f"service.{service_name}.error_rate", error_rate)
        await self.set_gauge(
            f"service.{service_name}.health",
            1 if health == ServiceHealth.HEALTHY else 0
        )

    async def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """Get current service status"""
        return self._services.get(service_name)

    async def get_all_service_statuses(self) -> dict[str, ServiceStatus]:
        """Get all service statuses"""
        return self._services.copy()

    async def check_service_health(self, service_name: str, url: str) -> ServiceHealth:
        """Check health of a service by URL"""
        try:
            import aiohttp

            start_time = datetime.now(timezone.utc)
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    response_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                    if response.status == 200:
                        health = ServiceHealth.HEALTHY
                    elif response.status < 500:
                        health = ServiceHealth.DEGRADED
                    else:
                        health = ServiceHealth.UNHEALTHY

                    await self.update_service_status(
                        service_name=service_name,
                        health=health,
                        response_time_ms=response_time
                    )

                    return health

        except Exception as e:
            logger.error(f"Service health check failed for {service_name}: {e}")
            await self.update_service_status(
                service_name=service_name,
                health=ServiceHealth.UNHEALTHY,
                metadata={"error": str(e)}
            )
            return ServiceHealth.UNHEALTHY

    # ========================================================================
    # ALERTS
    # ========================================================================

    async def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Optional[dict[str, Any]] = None
    ) -> DashboardAlert:
        """Create a new alert"""
        await self.initialize()

        alert_id = f"alert_{datetime.now().timestamp()}"

        alert = DashboardAlert(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )

        self._alerts.append(alert)

        # Keep alerts limited
        if len(self._alerts) > 1000:
            self._alerts = self._alerts[-1000:]

        await self._persist_alert(alert)

        logger.warning(f"Alert created: [{severity.value}] {title}")
        return alert

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                await self._update_alert(alert)
                return True
        return False

    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                await self._update_alert(alert)
                return True
        return False

    async def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None
    ) -> list[DashboardAlert]:
        """Get active (unresolved) alerts"""
        alerts = [a for a in self._alerts if not a.resolved]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    async def get_alert_count(self) -> dict[str, int]:
        """Get alert counts by severity"""
        counts = defaultdict(int)
        for alert in self._alerts:
            if not alert.resolved:
                counts[alert.severity.value] += 1
        return dict(counts)

    async def _check_alert_rules(self):
        """Check alert rules and create alerts if needed"""
        for rule_name, rule in self._alert_rules.items():
            value = await self.get_metric_value(rule["metric"])
            if value is None:
                continue

            threshold = rule["threshold"]
            condition = rule["condition"]
            triggered = False

            if condition == "gt" and value > threshold:
                triggered = True
            elif condition == "lt" and value < threshold:
                triggered = True
            elif condition == "eq" and value == threshold:
                triggered = True

            if triggered:
                # Check if similar alert already exists
                existing = [
                    a for a in self._alerts
                    if a.source == rule_name and not a.resolved
                ]
                if not existing:
                    await self.create_alert(
                        severity=rule["severity"],
                        title=f"Alert: {rule_name}",
                        message=rule["message"],
                        source=rule_name,
                        metadata={"rule": rule, "value": value}
                    )

    async def _persist_alert(self, alert: DashboardAlert):
        """Persist alert to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_dashboard_alerts (
                    alert_id, severity, title, message, source,
                    created_at, acknowledged, resolved, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_id) DO NOTHING
            """, (
                alert.alert_id,
                alert.severity.value,
                alert.title,
                alert.message,
                alert.source,
                alert.created_at,
                alert.acknowledged,
                alert.resolved,
                json.dumps(alert.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")

    async def _update_alert(self, alert: DashboardAlert):
        """Update alert in database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                UPDATE ai_dashboard_alerts
                SET acknowledged = %s,
                    acknowledged_at = CASE WHEN %s THEN NOW() ELSE acknowledged_at END,
                    resolved = %s,
                    resolved_at = CASE WHEN %s THEN NOW() ELSE resolved_at END
                WHERE alert_id = %s
            """, (
                alert.acknowledged,
                alert.acknowledged,
                alert.resolved,
                alert.resolved,
                alert.alert_id
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update alert: {e}")

    # ========================================================================
    # DASHBOARD DATA
    # ========================================================================

    async def get_dashboard_data(self) -> dict[str, Any]:
        """Get complete dashboard data"""
        await self.initialize()

        # Check cache
        cache_key = "dashboard_data"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now(timezone.utc) - cached_time).seconds < self._cache_ttl:
                return cached_data

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_score": await self.get_metric_value("system.health_score") or 100,
            "summary": {
                "total_agents": await self.get_metric_value("agents.total") or 0,
                "active_agents": await self.get_metric_value("agents.active") or 0,
                "tasks_completed": await self.get_metric_value("tasks.completed") or 0,
                "tasks_failed": await self.get_metric_value("tasks.failed") or 0,
                "error_rate": await self.get_metric_value("errors.rate") or 0,
                "error_count": await self.get_metric_value("errors.count") or 0
            },
            "services": {
                name: asdict(status)
                for name, status in self._services.items()
            },
            "alerts": {
                "counts": await self.get_alert_count(),
                "recent": [
                    asdict(a) for a in (await self.get_active_alerts())[:10]
                ]
            },
            "panels": {
                panel_id: asdict(panel)
                for panel_id, panel in self._panels.items()
            }
        }

        # Cache the result
        self._cache[cache_key] = (datetime.now(timezone.utc), data)

        return data

    async def get_panel_data(
        self,
        panel_id: str,
        time_range: TimeRange = TimeRange.LAST_HOUR
    ) -> dict[str, Any]:
        """Get data for a specific panel"""
        if panel_id not in self._panels:
            return {}

        panel = self._panels[panel_id]

        data = {
            "panel": asdict(panel),
            "metrics": {}
        }

        for metric_name in panel.metrics:
            series = await self.get_metric(metric_name, time_range)
            if series:
                data["metrics"][metric_name] = {
                    "values": [
                        {"timestamp": v[0].isoformat(), "value": v[1]}
                        for v in series.values
                    ],
                    "stats": await self.get_metric_stats(metric_name, time_range)
                }

        return data

    async def get_health_status(self) -> dict[str, Any]:
        """Get monitoring system health status"""
        health_score = await self.get_metric_value("system.health_score") or 100

        return {
            "status": "healthy" if health_score >= 80 else ("degraded" if health_score >= 50 else "unhealthy"),
            "initialized": self._initialized,
            "health_score": health_score,
            "metrics_count": len(self._metrics),
            "services_count": len(self._services),
            "active_alerts": len([a for a in self._alerts if not a.resolved]),
            "panels_count": len(self._panels)
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_dashboard_instance: Optional[MonitoringDashboard] = None


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get or create the monitoring dashboard instance"""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = MonitoringDashboard()
    return _dashboard_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def record_metric(
    name: str,
    value: float,
    metric_type: MetricType = MetricType.GAUGE
):
    """Record a metric"""
    dashboard = get_monitoring_dashboard()
    await dashboard.record_metric(name, value, metric_type)


async def create_alert(
    severity: AlertSeverity,
    title: str,
    message: str,
    source: str
) -> DashboardAlert:
    """Create an alert"""
    dashboard = get_monitoring_dashboard()
    return await dashboard.create_alert(severity, title, message, source)


async def get_dashboard_data() -> dict[str, Any]:
    """Get dashboard data"""
    dashboard = get_monitoring_dashboard()
    return await dashboard.get_dashboard_data()


async def update_service_status(
    service_name: str,
    health: ServiceHealth,
    response_time_ms: float = 0
):
    """Update service status"""
    dashboard = get_monitoring_dashboard()
    await dashboard.update_service_status(service_name, health, response_time_ms)


if __name__ == "__main__":
    async def test():
        dashboard = get_monitoring_dashboard()
        await dashboard.initialize()

        # Record some metrics
        await dashboard.set_gauge("test.gauge", 42)
        await dashboard.increment_counter("test.counter", 1)
        await dashboard.observe_histogram("test.histogram", 100)

        print(f"Gauge value: {await dashboard.get_metric_value('test.gauge')}")
        print(f"Counter value: {await dashboard.get_metric_value('test.counter')}")

        # Update service status
        await dashboard.update_service_status(
            service_name="test_service",
            health=ServiceHealth.HEALTHY,
            response_time_ms=50
        )

        # Create alert
        alert = await dashboard.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            source="test"
        )
        print(f"Alert created: {alert.alert_id}")

        # Get dashboard data
        data = await dashboard.get_dashboard_data()
        print(f"Dashboard data: {json.dumps(data, indent=2, default=str)}")

        # Get health status
        health = await dashboard.get_health_status()
        print(f"Health status: {json.dumps(health, indent=2)}")

    asyncio.run(test())
