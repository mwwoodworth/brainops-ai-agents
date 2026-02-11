#!/usr/bin/env python3
"""
ALWAYS-KNOW OBSERVABILITY BRAIN
================================
Comprehensive observability that makes the AI OS truly self-aware.
Never need to query - the system KNOWS its state at all times.

Features:
1. Continuous state collection (every 30 seconds)
2. Persistent state storage in database
3. Slack webhook alerting for critical issues
4. Automated UI testing on schedule
5. ChatGPT-Agent-level frontend testing
6. Real-time anomaly detection
7. Proactive issue reporting

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum

from safe_task import create_safe_task
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

# Configuration
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
PAGERDUTY_KEY = os.getenv("PAGERDUTY_ROUTING_KEY", "")
STATE_COLLECTION_INTERVAL = 30  # seconds
UI_TEST_INTERVAL = 3600  # 1 hour
ALERT_COOLDOWN = 300  # 5 minutes between same alerts


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SystemComponent(Enum):
    AI_AGENTS = "ai_agents"
    BACKEND = "backend"
    MCP_BRIDGE = "mcp_bridge"
    DATABASE = "database"
    AUREA = "aurea"
    SELF_HEALING = "self_healing"
    MEMORY = "memory"
    FRONTEND_MRG = "frontend_mrg"
    FRONTEND_ERP = "frontend_erp"
    FRONTEND_COMMAND_CENTER = "frontend_command_center"
    FRONTEND_BRAINSTACK_STUDIO = "frontend_brainstack_studio"


@dataclass
class SystemState:
    """Current state of the entire AI OS"""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Service health
    ai_agents_healthy: bool = False
    ai_agents_version: str = ""
    ai_agents_systems: int = 0

    backend_healthy: bool = False
    backend_version: str = ""

    mcp_bridge_healthy: bool = False
    mcp_servers: int = 0
    mcp_tools: int = 0

    database_connected: bool = False

    # AUREA metrics
    aurea_operational: bool = False
    aurea_ooda_cycles: int = 0
    aurea_decisions: int = 0
    aurea_active_agents: int = 0

    # Memory metrics
    embedded_memories: int = 0
    brain_records: int = 0

    # Error metrics
    errors_last_hour: int = 0
    errors_last_24h: int = 0

    # Performance
    response_time_ms: float = 0.0

    # Frontend health
    mrg_healthy: bool = False
    mrg_response_time_ms: float = 0.0
    erp_healthy: bool = False
    erp_response_time_ms: float = 0.0
    command_center_healthy: bool = False
    command_center_response_time_ms: float = 0.0
    brainstack_studio_healthy: bool = False
    brainstack_studio_response_time_ms: float = 0.0

    # Business metrics
    customers_total: int = 0
    jobs_total: int = 0
    revenue_leads: int = 0


@dataclass
class Alert:
    """An alert that needs attention"""
    alert_id: str
    severity: AlertSeverity
    component: SystemComponent
    title: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved: bool = False
    sent_to_slack: bool = False
    sent_to_pagerduty: bool = False


class AlwaysKnowBrain:
    """
    The Always-Know Brain continuously monitors all systems
    and maintains a real-time state cache.
    """

    def __init__(self):
        self.current_state: SystemState = SystemState()
        self.state_history: list[SystemState] = []
        self.alerts: dict[str, Alert] = {}
        self.alert_cooldowns: dict[str, float] = {}
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_ui_test: float = 0
        self._db_pool = None

    async def initialize(self):
        """Initialize the brain"""
        # Longer timeout - Render services can be slow when cold
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

        # Try to get database pool
        try:
            from database.sync_pool import get_sync_pool
            self._db_pool = get_sync_pool()
        except Exception as e:
            logger.warning(f"Database pool not available: {e}")

        logger.info("Always-Know Brain initialized")

    async def close(self):
        """Cleanup resources"""
        self._running = False
        if self._session:
            await self._session.close()

    async def start_continuous_monitoring(self):
        """Start the continuous monitoring loop"""
        self._running = True
        logger.info("Starting continuous state monitoring...")

        while self._running:
            try:
                # Collect current state
                await self._collect_state()

                # Check for anomalies and alert
                await self._check_anomalies()

                # Store state in database
                await self._persist_state()

                # Run UI tests if interval elapsed
                if time.time() - self._last_ui_test > UI_TEST_INTERVAL:
                    create_safe_task(self._run_ui_tests())
                    self._last_ui_test = time.time()

            except Exception as e:
                logger.error(f"State collection error: {e}")

            await asyncio.sleep(STATE_COLLECTION_INTERVAL)

    async def _collect_state(self):
        """Collect state from all systems"""
        state = SystemState()

        # Parallel collection
        tasks = [
            self._check_ai_agents(state),
            self._check_backend(state),
            self._check_mcp_bridge(state),
            self._check_database(state),
            self._check_aurea(state),
            self._check_frontends(state),
            self._check_errors(state),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        self.current_state = state
        self.state_history.append(state)

        # Keep only last 1000 states
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]

        logger.debug(f"State collected: {state.ai_agents_healthy}/{state.backend_healthy}/{state.database_connected}")

    async def _check_ai_agents(self, state: SystemState):
        """Check AI Agents service"""
        try:
            start = time.time()
            async with self._session.get(
                "https://brainops-ai-agents.onrender.com/health",
                headers={"X-API-Key": os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY")}
            ) as resp:
                state.response_time_ms = (time.time() - start) * 1000
                if resp.status == 200:
                    data = await resp.json()
                    state.ai_agents_healthy = data.get("status") == "healthy"
                    state.ai_agents_version = data.get("version", "")
                    state.ai_agents_systems = data.get("system_count", 0)
                    state.embedded_memories = data.get("embedded_memory_stats", {}).get("total_memories", 0)
        except Exception as e:
            logger.warning(f"AI Agents check failed: {e}")
            state.ai_agents_healthy = False

    async def _check_backend(self, state: SystemState):
        """Check Backend service"""
        try:
            async with self._session.get(
                "https://brainops-backend-prod.onrender.com/health"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    state.backend_healthy = data.get("status") == "healthy"
                    state.backend_version = data.get("version", "")
        except Exception as e:
            logger.warning(f"Backend check failed: {e}")
            state.backend_healthy = False

    async def _check_mcp_bridge(self, state: SystemState):
        """Check MCP Bridge service"""
        try:
            async with self._session.get(
                "https://brainops-mcp-bridge.onrender.com/health"
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    state.mcp_bridge_healthy = data.get("status") == "healthy"
                    state.mcp_servers = data.get("mcpServers", 0)
                    state.mcp_tools = data.get("totalTools", 0)
        except Exception as e:
            logger.warning(f"MCP Bridge check failed: {e}")
            state.mcp_bridge_healthy = False

    async def _check_database(self, state: SystemState):
        """Check database connection"""
        try:
            if self._db_pool:
                # Use pool's fetchone method for simple queries
                result = self._db_pool.fetchone("SELECT 1 as test")
                if result:
                    state.database_connected = True

                    # Get counts
                    customers = self._db_pool.fetchone("SELECT COUNT(*) as cnt FROM customers")
                    if customers:
                        state.customers_total = customers.get("cnt", 0)

                    jobs = self._db_pool.fetchone("SELECT COUNT(*) as cnt FROM jobs")
                    if jobs:
                        state.jobs_total = jobs.get("cnt", 0)

                    # Get revenue leads count
                    leads = self._db_pool.fetchone("SELECT COUNT(*) as cnt FROM revenue_leads")
                    if leads:
                        state.revenue_leads = leads.get("cnt", 0)
                else:
                    state.database_connected = False
            else:
                state.database_connected = False
        except Exception as e:
            logger.warning(f"Database check failed: {e}")
            state.database_connected = False

    async def _check_aurea(self, state: SystemState):
        """Check AUREA orchestrator"""
        try:
            async with self._session.get(
                "https://brainops-ai-agents.onrender.com/aurea/status",
                headers={"X-API-Key": os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY")}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # AUREA is operational if status is operational/idle/active AND aurea_available is True
                    status = data.get("status", "").lower()
                    state.aurea_operational = (
                        status in ("operational", "idle", "active", "processing")
                        and data.get("aurea_available", False)
                    )
                    state.aurea_ooda_cycles = data.get("ooda_cycles_last_5min", 0)
                    state.aurea_decisions = data.get("decisions_last_hour", 0)
                    state.aurea_active_agents = data.get("active_agents", 0)
        except Exception as e:
            logger.warning(f"AUREA check failed: {e!r}")
            state.aurea_operational = False

    async def _check_frontends(self, state: SystemState):
        """Check frontend applications"""
        # MyRoofGenius - check API health endpoint
        try:
            start = time.time()
            async with self._session.get("https://myroofgenius.com/api/health") as resp:
                state.mrg_response_time_ms = (time.time() - start) * 1000
                # Accept any 2xx status (200=OK, 206=Partial Content for degraded)
                if 200 <= resp.status < 300:
                    data = await resp.json()
                    # Check if core services are healthy (DB must be healthy, Stripe optional)
                    db_healthy = data.get("services", {}).get("database", {}).get("status") == "healthy"
                    # MRG is healthy if DB is connected, even if Stripe isn't configured
                    state.mrg_healthy = db_healthy
                    if not state.mrg_healthy:
                        logger.warning(f"MRG unhealthy: {data.get('services', {})}")
                else:
                    logger.warning(f"MRG returned non-2xx status: {resp.status}")
                    state.mrg_healthy = False
        except Exception as e:
            logger.warning(f"MRG check failed: {e}")
            state.mrg_healthy = False

        # Weathercraft ERP - check root (health endpoint requires secret)
        try:
            start = time.time()
            async with self._session.get("https://weathercraft-erp.vercel.app/") as resp:
                state.erp_response_time_ms = (time.time() - start) * 1000
                state.erp_healthy = resp.status == 200
        except Exception as e:
            logger.warning(f"ERP check failed: {e}")
            state.erp_healthy = False

        # BrainOps Command Center - check public health endpoint
        # NOTE: /api/unified-health requires auth; use /api/health instead
        try:
            start = time.time()
            async with self._session.get("https://brainops-command-center.vercel.app/api/health") as resp:
                state.command_center_response_time_ms = (time.time() - start) * 1000
                if resp.status == 200:
                    try:
                        data = await resp.json()
                        # Check status field from health endpoint
                        status = data.get("status", "")
                        state.command_center_healthy = status == "ok"
                    except Exception:
                        # If response is 200 but can't parse, still consider healthy
                        state.command_center_healthy = True
                else:
                    state.command_center_healthy = False
        except Exception as e:
            logger.warning(f"Command Center check failed: {e}")
            state.command_center_healthy = False

        # Brainstack Studio (marketing) - simple availability check
        try:
            start = time.time()
            async with self._session.get("https://brainstack-studio.vercel.app/") as resp:
                state.brainstack_studio_response_time_ms = (time.time() - start) * 1000
                state.brainstack_studio_healthy = resp.status == 200
        except Exception as e:
            logger.warning(f"Brainstack Studio check failed: {e}")
            state.brainstack_studio_healthy = False

    async def _check_errors(self, state: SystemState):
        """Check error counts"""
        try:
            if self._db_pool:
                result = self._db_pool.fetchone("""
                    SELECT
                        COUNT(*) FILTER (WHERE occurred_at > NOW() - INTERVAL '1 hour') as errors_1h,
                        COUNT(*) FILTER (WHERE occurred_at > NOW() - INTERVAL '24 hours') as errors_24h
                    FROM ai_error_logs
                    WHERE occurred_at > NOW() - INTERVAL '24 hours'
                """)
                if result:
                    state.errors_last_hour = result.get("errors_1h", 0) or 0
                    state.errors_last_24h = result.get("errors_24h", 0) or 0
        except Exception as e:
            logger.warning(f"Error check failed: {e}")

    async def _resolve_recovered_alerts(self):
        """Auto-resolve alerts when services recover"""
        state = self.current_state

        # Map of component -> health check
        health_checks = {
            SystemComponent.AI_AGENTS.value: state.ai_agents_healthy,
            SystemComponent.BACKEND.value: state.backend_healthy,
            SystemComponent.DATABASE.value: state.database_connected,
            SystemComponent.FRONTEND_MRG.value: state.mrg_healthy,
            SystemComponent.FRONTEND_ERP.value: state.erp_healthy,
            SystemComponent.FRONTEND_COMMAND_CENTER.value: state.command_center_healthy,
            SystemComponent.FRONTEND_BRAINSTACK_STUDIO.value: state.brainstack_studio_healthy,
            SystemComponent.AUREA.value: state.aurea_operational,
        }

        # Find alerts to resolve
        alerts_to_resolve = []
        for alert_id, alert in list(self.alerts.items()):
            if alert.resolved:
                continue
            component_key = alert.component.value
            if component_key in health_checks and health_checks[component_key]:
                alerts_to_resolve.append(alert_id)

        # Resolve alerts
        for alert_id in alerts_to_resolve:
            if alert_id in self.alerts:
                self.alerts[alert_id].resolved = True
                logger.info(f"Alert auto-resolved: {self.alerts[alert_id].title}")

    async def _check_anomalies(self):
        """Check for anomalies and create alerts"""
        # First, resolve any alerts for recovered services
        await self._resolve_recovered_alerts()

        state = self.current_state

        # Critical: Service down
        if not state.ai_agents_healthy:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                SystemComponent.AI_AGENTS,
                "AI Agents Service Down",
                f"AI Agents service is not responding. Last version: {state.ai_agents_version}"
            )

        if not state.backend_healthy:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                SystemComponent.BACKEND,
                "Backend Service Down",
                "Backend service is not responding."
            )

        if not state.database_connected:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                SystemComponent.DATABASE,
                "Database Connection Lost",
                "Cannot connect to PostgreSQL database."
            )

        # Warning: High errors
        if state.errors_last_hour > 10:
            await self._create_alert(
                AlertSeverity.WARNING,
                SystemComponent.AI_AGENTS,
                "High Error Rate",
                f"{state.errors_last_hour} errors in the last hour"
            )

        # Warning: Slow response
        if state.response_time_ms > 5000:
            await self._create_alert(
                AlertSeverity.WARNING,
                SystemComponent.AI_AGENTS,
                "Slow Response Time",
                f"AI Agents response time: {state.response_time_ms:.0f}ms"
            )

        # Warning: Frontend issues
        if not state.mrg_healthy:
            await self._create_alert(
                AlertSeverity.ERROR,
                SystemComponent.FRONTEND_MRG,
                "MyRoofGenius Down",
                "MyRoofGenius website is not responding."
            )

        if not state.erp_healthy:
            await self._create_alert(
                AlertSeverity.ERROR,
                SystemComponent.FRONTEND_ERP,
                "Weathercraft ERP Down",
                "Weathercraft ERP is not responding."
            )

        if not state.command_center_healthy:
            await self._create_alert(
                AlertSeverity.ERROR,
                SystemComponent.FRONTEND_COMMAND_CENTER,
                "Command Center Down",
                "BrainOps Command Center is not responding (or reporting critical status)."
            )

        if not state.brainstack_studio_healthy:
            await self._create_alert(
                AlertSeverity.WARNING,
                SystemComponent.FRONTEND_BRAINSTACK_STUDIO,
                "Brainstack Studio Down",
                "Brainstack Studio marketing site is not responding."
            )

        # Warning: AUREA not operational
        if not state.aurea_operational:
            await self._create_alert(
                AlertSeverity.WARNING,
                SystemComponent.AUREA,
                "AUREA Not Operational",
                "AUREA orchestrator is not in operational state."
            )

    async def _create_alert(
        self,
        severity: AlertSeverity,
        component: SystemComponent,
        title: str,
        message: str
    ):
        """Create and send an alert"""
        alert_key = f"{component.value}:{title}"

        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if time.time() - self.alert_cooldowns[alert_key] < ALERT_COOLDOWN:
                return

        alert_id = hashlib.md5(f"{alert_key}{time.time()}".encode()).hexdigest()[:12]

        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            component=component,
            title=title,
            message=message
        )

        self.alerts[alert_id] = alert
        self.alert_cooldowns[alert_key] = time.time()

        logger.warning(f"ALERT [{severity.value.upper()}]: {title} - {message}")

        # Send to Slack
        if SLACK_WEBHOOK_URL and severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
            await self._send_slack_alert(alert)

        # Send to PagerDuty for critical
        if PAGERDUTY_KEY and severity == AlertSeverity.CRITICAL:
            await self._send_pagerduty_alert(alert)

    async def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        try:
            color = {
                AlertSeverity.CRITICAL: "#FF0000",
                AlertSeverity.ERROR: "#FF6600",
                AlertSeverity.WARNING: "#FFCC00",
                AlertSeverity.INFO: "#0066FF"
            }.get(alert.severity, "#808080")

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Component", "value": alert.component.value, "short": True},
                        {"title": "Time", "value": alert.timestamp, "short": True}
                    ],
                    "footer": "BrainOps AI OS Monitoring"
                }]
            }

            async with self._session.post(SLACK_WEBHOOK_URL, json=payload) as resp:
                if resp.status == 200:
                    alert.sent_to_slack = True
                    logger.info(f"Slack alert sent: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    async def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        try:
            payload = {
                "routing_key": PAGERDUTY_KEY,
                "event_action": "trigger",
                "dedup_key": f"brainops-{alert.component.value}-{alert.title}",
                "payload": {
                    "summary": f"{alert.title}: {alert.message}",
                    "severity": "critical",
                    "source": "brainops-ai-os",
                    "component": alert.component.value
                }
            }

            async with self._session.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            ) as resp:
                if resp.status == 202:
                    alert.sent_to_pagerduty = True
                    logger.info(f"PagerDuty alert sent: {alert.title}")

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")

    async def _persist_state(self):
        """Persist current state to database.

        NOTE: DDL (CREATE TABLE/INDEX) removed because agent_worker role
        (app_agent_role) has no DDL permissions by design (P0-LOCK security).
        Tables must be created via migrations. This method verifies tables
        exist and degrades gracefully if they are missing.
        """
        try:
            if not self._db_pool:
                return

            with self._db_pool.get_connection() as conn:
                if not conn:
                    return

                cur = conn.cursor()

                # Verify required table exists (no DDL - agent_worker has no CREATE permissions)
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                    (['always_know_state'],)
                )
                row = cur.fetchone()
                table_count = row[0] if row else 0
                if table_count < 1:
                    logger.error(
                        "Required table always_know_state missing (found %s/1). "
                        "Run migrations to create it.", table_count
                    )
                    cur.close()
                    return

                # Insert current state
                state_dict = asdict(self.current_state)
                cur.execute("""
                    INSERT INTO always_know_state
                    (state_json, ai_agents_healthy, backend_healthy, database_connected,
                     aurea_operational, errors_last_hour, response_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    json.dumps(state_dict),
                    self.current_state.ai_agents_healthy,
                    self.current_state.backend_healthy,
                    self.current_state.database_connected,
                    self.current_state.aurea_operational,
                    self.current_state.errors_last_hour,
                    self.current_state.response_time_ms
                ))

                conn.commit()
                cur.close()

                # Cleanup old states (keep last 7 days)
                cur = conn.cursor()
                cur.execute("""
                    DELETE FROM always_know_state
                    WHERE timestamp < NOW() - INTERVAL '7 days'
                """)
                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to persist state: {e}")

    async def _run_ui_tests(self):
        """Run comprehensive UI tests"""
        logger.info("Starting scheduled UI tests...")

        try:
            from ai_ui_testing import (
                AIUITestingEngine,
                MRG_ROUTES,
                ERP_ROUTES,
                COMMAND_CENTER_ROUTES,
                BRAINSTACK_STUDIO_ROUTES,
            )

            engine = AIUITestingEngine()
            await engine.initialize()

            mode = (os.getenv("ALWAYS_KNOW_UI_TEST_MODE") or "smoke").strip().lower()
            if mode == "full":
                mrg_routes = MRG_ROUTES
                erp_routes = ERP_ROUTES
            else:
                mrg_routes = ["/", "/login", "/tools", "/pricing", "/aurea"]
                erp_routes = ["/", "/login"]

            # Test MyRoofGenius
            mrg_results = await engine.test_application(
                base_url="https://myroofgenius.com",
                routes=mrg_routes,
                app_name="MyRoofGenius"
            )

            # Test ERP
            erp_results = await engine.test_application(
                base_url="https://weathercraft-erp.vercel.app",
                routes=erp_routes,
                app_name="Weathercraft ERP"
            )

            # Test Command Center
            command_center_results = await engine.test_application(
                base_url="https://brainops-command-center.vercel.app",
                routes=COMMAND_CENTER_ROUTES,
                app_name="BrainOps Command Center",
            )

            # Test Brainstack Studio
            brainstack_results = await engine.test_application(
                base_url="https://brainstack-studio.vercel.app",
                routes=BRAINSTACK_STUDIO_ROUTES,
                app_name="Brainstack Studio",
            )

            await engine.close()

            # Create alerts for failures
            if mrg_results.get("failed", 0) > 0:
                await self._create_alert(
                    AlertSeverity.ERROR,
                    SystemComponent.FRONTEND_MRG,
                    "MyRoofGenius UI Test Failures",
                    f"{mrg_results['failed']} UI tests failed out of {mrg_results['total_tests']}"
                )

            if erp_results.get("failed", 0) > 0:
                await self._create_alert(
                    AlertSeverity.ERROR,
                    SystemComponent.FRONTEND_ERP,
                    "ERP UI Test Failures",
                    f"{erp_results['failed']} UI tests failed out of {erp_results['total_tests']}"
                )

            if command_center_results.get("failed", 0) > 0:
                await self._create_alert(
                    AlertSeverity.ERROR,
                    SystemComponent.FRONTEND_COMMAND_CENTER,
                    "Command Center UI Test Failures",
                    f"{command_center_results['failed']} UI tests failed out of {command_center_results['total_tests']}",
                )

            if brainstack_results.get("failed", 0) > 0:
                await self._create_alert(
                    AlertSeverity.WARNING,
                    SystemComponent.FRONTEND_BRAINSTACK_STUDIO,
                    "Brainstack Studio UI Test Failures",
                    f"{brainstack_results['failed']} UI tests failed out of {brainstack_results['total_tests']}",
                )

            # Persist test results
            await self._persist_ui_test_results(
                [
                    ("MyRoofGenius", mrg_results),
                    ("Weathercraft ERP", erp_results),
                    ("BrainOps Command Center", command_center_results),
                    ("Brainstack Studio", brainstack_results),
                ]
            )

            logger.info(
                "UI tests complete. "
                "MRG: %s/%s passed. ERP: %s/%s passed. Command Center: %s/%s passed. Brainstack: %s/%s passed.",
                mrg_results.get("passed", 0),
                mrg_results.get("total_tests", 0),
                erp_results.get("passed", 0),
                erp_results.get("total_tests", 0),
                command_center_results.get("passed", 0),
                command_center_results.get("total_tests", 0),
                brainstack_results.get("passed", 0),
                brainstack_results.get("total_tests", 0),
            )

        except Exception as e:
            logger.error(f"UI tests failed: {e}\n{traceback.format_exc()}")

    async def _persist_ui_test_results(self, app_results: list[tuple[str, dict]]):
        """Persist UI test results to database.

        NOTE: DDL (CREATE TABLE) removed because agent_worker role
        (app_agent_role) has no DDL permissions by design (P0-LOCK security).
        Tables must be created via migrations. This method verifies tables
        exist and degrades gracefully if they are missing.
        """
        try:
            if not self._db_pool:
                return

            with self._db_pool.get_connection() as conn:
                if not conn:
                    return

                cur = conn.cursor()

                # Verify required table exists (no DDL - agent_worker has no CREATE permissions)
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                    (['ui_test_history'],)
                )
                row = cur.fetchone()
                table_count = row[0] if row else 0
                if table_count < 1:
                    logger.error(
                        "Required table ui_test_history missing (found %s/1). "
                        "Run migrations to create it.", table_count
                    )
                    cur.close()
                    return

                for app_name, results in app_results:
                    cur.execute("""
                        INSERT INTO ui_test_history
                        (application, total_tests, passed, failed, warnings, duration_seconds, results_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        app_name,
                        results.get("total_tests", 0),
                        results.get("passed", 0),
                        results.get("failed", 0),
                        results.get("warnings", 0),
                        results.get("duration_seconds", 0),
                        json.dumps(results)
                    ))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to persist UI test results: {e}")

    def get_current_state(self) -> dict[str, Any]:
        """Get current system state as dict"""
        return asdict(self.current_state)

    def get_state_summary(self) -> str:
        """Get a human-readable state summary"""
        s = self.current_state

        # Calculate overall health
        services_up = sum([
            s.ai_agents_healthy,
            s.backend_healthy,
            s.mcp_bridge_healthy,
            s.database_connected,
            s.mrg_healthy,
            s.erp_healthy,
            s.command_center_healthy,
            s.brainstack_studio_healthy,
        ])
        services_total = 8
        health_pct = (services_up / services_total) * 100

        return f"""
=== BRAINOPS AI OS STATE ===
Timestamp: {s.timestamp}
Overall Health: {health_pct:.0f}% ({services_up}/{services_total} services)

SERVICES:
  AI Agents: {'✅' if s.ai_agents_healthy else '❌'} v{s.ai_agents_version} ({s.ai_agents_systems} systems)
  Backend: {'✅' if s.backend_healthy else '❌'} v{s.backend_version}
  MCP Bridge: {'✅' if s.mcp_bridge_healthy else '❌'} ({s.mcp_servers} servers, {s.mcp_tools} tools)
  Database: {'✅' if s.database_connected else '❌'}
  MyRoofGenius: {'✅' if s.mrg_healthy else '❌'} ({s.mrg_response_time_ms:.0f}ms)
  Weathercraft ERP: {'✅' if s.erp_healthy else '❌'} ({s.erp_response_time_ms:.0f}ms)
  Command Center: {'✅' if s.command_center_healthy else '❌'} ({s.command_center_response_time_ms:.0f}ms)
  Brainstack Studio: {'✅' if s.brainstack_studio_healthy else '❌'} ({s.brainstack_studio_response_time_ms:.0f}ms)

AUREA:
  Status: {'Operational' if s.aurea_operational else 'Not Operational'}
  OODA Cycles: {s.aurea_ooda_cycles}/5min
  Decisions: {s.aurea_decisions}/hr
  Active Agents: {s.aurea_active_agents}

METRICS:
  Embedded Memories: {s.embedded_memories:,}
  Customers: {s.customers_total:,}
  Jobs: {s.jobs_total:,}
  Errors (1h): {s.errors_last_hour}
  Errors (24h): {s.errors_last_24h}
  Response Time: {s.response_time_ms:.0f}ms

ACTIVE ALERTS: {len([a for a in self.alerts.values() if not a.resolved])}
"""

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get all active (unresolved) alerts"""
        return [
            {
                "alert_id": a.alert_id,
                "severity": a.severity.value,
                "component": a.component.value,
                "title": a.title,
                "message": a.message,
                "timestamp": a.timestamp
            }
            for a in self.alerts.values()
            if not a.resolved
        ]


# =============================================================================
# SINGLETON
# =============================================================================

_brain: Optional[AlwaysKnowBrain] = None


def get_always_know_brain() -> AlwaysKnowBrain:
    """Get or create the Always-Know Brain singleton"""
    global _brain
    if _brain is None:
        _brain = AlwaysKnowBrain()
    return _brain


async def initialize_always_know_brain():
    """Initialize and start the Always-Know Brain"""
    brain = get_always_know_brain()
    await brain.initialize()

    # Start monitoring in background
    create_safe_task(brain.start_continuous_monitoring())

    logger.info("Always-Know Brain started - continuous monitoring active")
    return brain


# =============================================================================
# API ENDPOINTS (to be added to router)
# =============================================================================

async def get_system_state() -> dict[str, Any]:
    """API: Get current system state"""
    brain = get_always_know_brain()
    return brain.get_current_state()


async def get_system_summary() -> str:
    """API: Get system summary"""
    brain = get_always_know_brain()
    return brain.get_state_summary()


async def get_alerts() -> list[dict[str, Any]]:
    """API: Get active alerts"""
    brain = get_always_know_brain()
    return brain.get_active_alerts()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    async def main():
        brain = AlwaysKnowBrain()
        await brain.initialize()

        # Single state collection
        await brain._collect_state()

        print(brain.get_state_summary())

        await brain.close()

    asyncio.run(main())
