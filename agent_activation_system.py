"""
Agent Activation System
Manages the lifecycle and activation of AI agents.
Implements real database operations for agent state management.
"""

import json
import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# NOTE: This system runs in build-out mode for many tenants. We must be safe-by-default:
# - Respect dry_run_outreach flags from AUREA.
# - Prevent task floods via simple DB-level dedupe windows (no migrations).

# Try to use sync pool, fall back to direct connections
try:
    from database.sync_pool import get_sync_pool
    _SYNC_POOL_AVAILABLE = True
except ImportError:
    _SYNC_POOL_AVAILABLE = False
    logger.warning("sync_pool not available, using direct connections")


def json_safe_serialize(obj: Any) -> Any:
    """Recursively convert datetime/Decimal/Enum/bytes objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif hasattr(obj, '__dataclass_fields__'):
        return {k: json_safe_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {str(k): json_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(json_safe_serialize(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return json_safe_serialize(obj.__dict__)
    else:
        return str(obj)

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}


class BusinessEventType(Enum):
    """Types of business events that can trigger agent activation"""
    # Original events
    CUSTOMER_INQUIRY = "customer_inquiry"
    LEAD_CREATED = "lead_created"
    DEAL_CLOSED = "deal_closed"
    SUPPORT_REQUEST = "support_request"
    REVENUE_OPPORTUNITY = "revenue_opportunity"
    SYSTEM_ALERT = "system_alert"
    SCHEDULED_TASK = "scheduled_task"
    USER_ACTION = "user_action"
    INTEGRATION_EVENT = "integration_event"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    # AUREA orchestration events
    NEW_CUSTOMER = "new_customer"
    ESTIMATE_REQUESTED = "estimate_requested"
    INVOICE_OVERDUE = "invoice_overdue"
    SCHEDULING_CONFLICT = "scheduling_conflict"
    SYSTEM_HEALTH_CHECK = "system_health_check"
    CUSTOMER_CHURN_RISK = "customer_churn_risk"
    QUOTE_REQUESTED = "quote_requested"
    PAYMENT_RECEIVED = "payment_received"
    JOB_SCHEDULED = "job_scheduled"


def _coerce_uuid(value: Any) -> str | None:
    if not value:
        return None
    try:
        return str(uuid.UUID(str(value)))
    except Exception:
        return None


def _activation_dedupe_window_seconds(event_type: "BusinessEventType") -> int:
    """
    Best-effort idempotency/cooldown window for agent activation.

    We do NOT rely on DB constraints/migrations here. This is a safety rail to prevent
    runaway task floods (seen in production with AUREA loops).
    """
    env_override = os.getenv(f"AGENT_ACTIVATION_DEDUPE_SECONDS_{event_type.value.upper()}")
    if env_override:
        try:
            return max(0, int(env_override))
        except Exception:
            pass

    try:
        default_window = max(0, int(os.getenv("AGENT_ACTIVATION_DEDUPE_SECONDS", "900")))
    except Exception:
        default_window = 900

    per_event: dict[BusinessEventType, int] = {
        BusinessEventType.INVOICE_OVERDUE: 86400,        # daily
        BusinessEventType.CUSTOMER_CHURN_RISK: 86400,    # daily
        BusinessEventType.QUOTE_REQUESTED: 3600,         # hourly
        BusinessEventType.NEW_CUSTOMER: 3600,            # hourly
        BusinessEventType.SYSTEM_HEALTH_CHECK: 300,      # 5 minutes
    }
    return per_event.get(event_type, default_window)


# Singleton instances per tenant
_activation_systems: dict[str, "AgentActivationSystem"] = {}


def get_activation_system(tenant_id: str = "default") -> "AgentActivationSystem":
    """
    Get or create an AgentActivationSystem instance for the given tenant.
    Uses singleton pattern to reuse instances.
    """
    if not tenant_id:
        tenant_id = "default"

    if tenant_id not in _activation_systems:
        _activation_systems[tenant_id] = AgentActivationSystem(tenant_id)
        logger.info(f"Created new AgentActivationSystem for tenant: {tenant_id}")

    return _activation_systems[tenant_id]


class AgentActivationSystem:
    """Real agent activation system with database persistence"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for AgentActivationSystem")
        self.tenant_id = tenant_id

    def _get_db_connection(self):
        """Get database connection from pool or direct"""
        if _SYNC_POOL_AVAILABLE:
            return get_sync_pool().get_connection()
        else:
            @contextmanager
            def _fallback():
                conn = None
                try:
                    conn = psycopg2.connect(**DB_CONFIG)
                    yield conn
                except Exception as e:
                    logger.error(f"Database connection failed: {e}")
                    yield None
                finally:
                    if conn and not conn.closed:
                        conn.close()
            return _fallback()

    async def activate_agent(self, agent_id: str) -> dict[str, Any]:
        """Activate an agent with real database update"""
        result = {
            "agent_id": agent_id,
            "action": "activate",
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    result["success"] = False
                    result["error"] = "Database connection failed"
                    return result

                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Update agent status to active
                cur.execute("""
                    UPDATE ai_agents
                    SET status = 'active',
                        updated_at = NOW(),
                        last_active = NOW()
                    WHERE id = %s
                    RETURNING id, name, status, type
                """, (agent_id,))

                updated = cur.fetchone()

                if updated:
                    result["success"] = True
                    result["agent"] = dict(updated)
                    logger.info(f"Activated agent {agent_id} for tenant {self.tenant_id}")

                    # Log activation event
                    cur.execute("""
                        INSERT INTO agent_activation_log (agent_id, tenant_id, action, created_at)
                        VALUES (%s, %s, 'activate', NOW())
                    """, (agent_id, self.tenant_id))
                else:
                    result["success"] = False
                    result["error"] = "Agent not found"

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to activate agent {agent_id}: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    async def deactivate_agent(self, agent_id: str) -> dict[str, Any]:
        """Deactivate an agent with real database update"""
        result = {
            "agent_id": agent_id,
            "action": "deactivate",
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    result["success"] = False
                    result["error"] = "Database connection failed"
                    return result

                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Update agent status to inactive
                cur.execute("""
                    UPDATE ai_agents
                    SET status = 'inactive',
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING id, name, status, type
                """, (agent_id,))

                updated = cur.fetchone()

                if updated:
                    result["success"] = True
                    result["agent"] = dict(updated)
                    logger.info(f"Deactivated agent {agent_id} for tenant {self.tenant_id}")

                    # Log deactivation event
                    cur.execute("""
                        INSERT INTO agent_activation_log (agent_id, tenant_id, action, created_at)
                        VALUES (%s, %s, 'deactivate', NOW())
                    """, (agent_id, self.tenant_id))
                else:
                    result["success"] = False
                    result["error"] = "Agent not found"

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to deactivate agent {agent_id}: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    async def get_agent_status(self, agent_id: str) -> dict[str, Any]:
        """Get current status of an agent from database"""
        result = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    result["status"] = "unknown"
                    result["error"] = "Database connection failed"
                    return result

                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                    SELECT
                        id, name, type, status,
                        last_active, total_executions,
                        created_at, updated_at
                    FROM ai_agents
                    WHERE id = %s
                """, (agent_id,))

                agent = cur.fetchone()

                if agent:
                    result["status"] = agent['status']
                    result["agent"] = {
                        k: str(v) if isinstance(v, datetime) else v
                        for k, v in dict(agent).items()
                    }
                else:
                    result["status"] = "not_found"
                    result["error"] = "Agent not found"

                cur.close()

        except Exception as e:
            logger.error(f"Failed to get agent status {agent_id}: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        return result

    async def get_all_agents_status(self) -> dict[str, Any]:
        """Get status of all agents"""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": [],
            "summary": {}
        }

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    result["error"] = "Database connection failed"
                    return result

                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get all agents
                cur.execute("""
                    SELECT
                        id, name, type, status,
                        last_active, total_executions
                    FROM ai_agents
                    ORDER BY status, name
                """)

                agents = cur.fetchall()
                result["agents"] = [
                    {k: str(v) if isinstance(v, datetime) else v for k, v in dict(a).items()}
                    for a in agents
                ] if agents else []

                # Summary
                cur.execute("""
                    SELECT
                        status,
                        COUNT(*) as count
                    FROM ai_agents
                    GROUP BY status
                """)
                summary = cur.fetchall()
                result["summary"] = {row['status']: row['count'] for row in summary} if summary else {}
                result["total"] = len(agents) if agents else 0

                cur.close()

        except Exception as e:
            logger.error(f"Failed to get all agents status: {e}")
            result["error"] = str(e)

        return result

    async def trigger_agent_by_event(self, event_type: BusinessEventType, event_data: dict[str, Any]) -> dict[str, Any]:
        """Trigger appropriate agents based on business event"""
        result = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_agents": []
        }

        tenant_id_value = _coerce_uuid(self.tenant_id)
        dedupe_window_seconds = _activation_dedupe_window_seconds(event_type)

        # Map event types to agent types (includes AUREA orchestration events)
        # UPDATED to match actual types in database
        event_agent_mapping = {
            BusinessEventType.CUSTOMER_INQUIRY: ["interface", "support"],
            BusinessEventType.LEAD_CREATED: ["lead_generation", "discovery", "qualification", "LeadScoringAgent"],
            BusinessEventType.DEAL_CLOSED: ["closing", "proposal"],
            BusinessEventType.SUPPORT_REQUEST: ["support", "customer_success"],
            BusinessEventType.REVENUE_OPPORTUNITY: ["analytics", "optimizer", "pricing"],
            BusinessEventType.SYSTEM_ALERT: ["monitor", "system_improvement"],
            BusinessEventType.SCHEDULED_TASK: ["analytics", "monitor"],
            BusinessEventType.PERFORMANCE_THRESHOLD: ["system_improvement", "monitor"],
            # AUREA orchestration events
            BusinessEventType.NEW_CUSTOMER: ["analytics", "discovery"],
            BusinessEventType.ESTIMATE_REQUESTED: ["EstimationAgent", "universal", "proposal"],
            BusinessEventType.INVOICE_OVERDUE: ["InvoiceAgent", "analytics"],
            BusinessEventType.SCHEDULING_CONFLICT: ["SchedulingAgent", "universal", "system_improvement"],
            BusinessEventType.SYSTEM_HEALTH_CHECK: ["monitor", "system_improvement", "MonitoringAgent"],
            BusinessEventType.CUSTOMER_CHURN_RISK: ["analytics"],
            BusinessEventType.QUOTE_REQUESTED: ["pricing", "revenue", "proposal"],
            BusinessEventType.PAYMENT_RECEIVED: ["revenue"],
            BusinessEventType.JOB_SCHEDULED: ["system_improvement", "analytics"],
        }

        # Map event types to SPECIFIC agent names (for agents with generic 'workflow' type)
        event_agent_names_mapping = {
            BusinessEventType.INVOICE_OVERDUE: ["InvoicingAgent", "RevenueOptimizer"],
            BusinessEventType.NEW_CUSTOMER: ["CustomerAgent", "LeadGenerationAgent", "CustomerIntelligence"],
            BusinessEventType.SCHEDULING_CONFLICT: ["DispatchAgent", "Scheduler", "IntelligentScheduler"],
            BusinessEventType.QUOTE_REQUESTED: ["RevenueProposalAgent", "ContractGenerator"],
            BusinessEventType.PAYMENT_RECEIVED: ["InvoicingAgent"],
            BusinessEventType.JOB_SCHEDULED: ["DispatchAgent", "RoutingAgent"],
            BusinessEventType.CUSTOMER_CHURN_RISK: ["CustomerIntelligence", "CustomerAgent"],
            BusinessEventType.ESTIMATE_REQUESTED: ["Elena", "EstimationAgent", "ProposalGenerator"],
            BusinessEventType.LEAD_CREATED: ["LeadGenerationAgent", "LeadScorer", "LeadQualificationAgent"],
            BusinessEventType.SYSTEM_HEALTH_CHECK: ["HealthMonitor", "LearningFeedbackLoop", "SystemMonitor"],
        }

        target_types = event_agent_mapping.get(event_type, [])
        target_names = event_agent_names_mapping.get(event_type, [])

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    result["error"] = "Database connection failed"
                    return result

                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Find active agents matching event type OR specific names
                # We use specific names to target agents that have generic 'workflow' types
                cur.execute("""
                    SELECT id, name, type
                    FROM ai_agents
                    WHERE status = 'active'
                    AND (
                        type = ANY(%s)
                        OR
                        name = ANY(%s)
                    )
                """, (target_types, target_names))

                agents = cur.fetchall()

                for agent in agents:
                    # Flood protection: if a similar task already exists recently for this agent/event, skip.
                    try:
                        task_type = f"{event_type.value}_handler"
                        if dedupe_window_seconds > 0:
                            if tenant_id_value:
                                cur.execute(
                                    """
                                    SELECT id
                                    FROM ai_autonomous_tasks
                                    WHERE agent_id = %s
                                      AND tenant_id = %s
                                      AND task_type = %s
                                      AND status IN ('pending', 'processing', 'in_progress')
                                      AND created_at > NOW() - (%s * INTERVAL '1 second')
                                    ORDER BY created_at DESC
                                    LIMIT 1
                                    """,
                                    (agent["id"], tenant_id_value, task_type, dedupe_window_seconds),
                                )
                            else:
                                cur.execute(
                                    """
                                    SELECT id
                                    FROM ai_autonomous_tasks
                                    WHERE agent_id = %s
                                      AND tenant_id IS NULL
                                      AND task_type = %s
                                      AND status IN ('pending', 'processing', 'in_progress')
                                      AND created_at > NOW() - (%s * INTERVAL '1 second')
                                    ORDER BY created_at DESC
                                    LIMIT 1
                                    """,
                                    (agent["id"], task_type, dedupe_window_seconds),
                                )

                            existing = cur.fetchone()
                            if existing:
                                logger.info(
                                    "Skipping duplicate activation for %s (%s): existing_task=%s within %ss",
                                    agent.get("name"),
                                    event_type.value,
                                    existing.get("id"),
                                    dedupe_window_seconds,
                                )
                                continue
                    except Exception as dedupe_exc:
                        logger.warning("Activation dedupe check failed (continuing): %s", dedupe_exc)

                    # Queue task for each matching agent
                    cur.execute("""
                        INSERT INTO ai_autonomous_tasks
                        (task_type, priority, status, trigger_type, trigger_condition, agent_id, tenant_id, created_at)
                        VALUES (%s, %s, 'pending', %s, %s, %s, %s, NOW())
                        RETURNING id
                    """, (
                        task_type,
                        'high' if event_type in [BusinessEventType.SYSTEM_ALERT, BusinessEventType.SUPPORT_REQUEST] else 'medium',
                        event_type.value,
                        json.dumps(json_safe_serialize(event_data)),
                        agent['id'],
                        tenant_id_value,
                    ))
                    task = cur.fetchone()

                    result["triggered_agents"].append({
                        "agent_id": agent['id'],
                        "agent_name": agent['name'],
                        "task_id": str(task['id']) if task else None
                    })

                conn.commit()
                cur.close()

                result["success"] = True
                result["agents_triggered"] = len(result["triggered_agents"])
                logger.info(f"Triggered {len(result['triggered_agents'])} agents for event {event_type.value}")

        except Exception as e:
            logger.error(f"Failed to trigger agents for event {event_type.value}: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def get_agent_stats(self) -> dict[str, Any]:
        """Get statistics about agents for this tenant"""
        stats = {
            "total_agents": 0,
            "active_agents": 0,
            "inactive_agents": 0,
            "recent_activations": 0,
            "tenant_id": self.tenant_id
        }

        try:
            with self._get_db_connection() as conn:
                if not conn:
                    return stats

                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get total and active agent counts
                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) as active,
                        COUNT(CASE WHEN status = 'inactive' THEN 1 END) as inactive
                    FROM ai_agents
                """)
                counts = cur.fetchone()
                if counts:
                    stats["total_agents"] = counts["total"] or 0
                    stats["active_agents"] = counts["active"] or 0
                    stats["inactive_agents"] = counts["inactive"] or 0

                # Get recent activations (last 24h)
                cur.execute("""
                    SELECT COUNT(*) as recent
                    FROM agent_activation_log
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)
                recent = cur.fetchone()
                if recent:
                    stats["recent_activations"] = recent["recent"] or 0

                cur.close()
        except Exception as e:
            logger.error(f"Failed to get agent stats: {e}")

        return stats

    async def handle_business_event(
        self,
        event_type: BusinessEventType,
        event_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle business events from AUREA orchestration.
        This is the main entry point for AUREA decisions to trigger agent activation.

        Supports verification_mode and dry_run flags to prevent real outreach
        when testing with seeded data.
        """
        # Check for verification/dry_run mode
        verification_mode = event_data.get("verification_mode", False)
        dry_run_outreach = event_data.get("dry_run_outreach", False)
        dry_run = event_data.get("dry_run", False) or dry_run_outreach
        aurea_initiated = event_data.get("aurea_initiated", False)

        result = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "verification_mode": verification_mode,
            "dry_run": dry_run,
            "dry_run_outreach": dry_run_outreach,
            "aurea_initiated": aurea_initiated,
            "actions_taken": [],
            "success": True
        }

        if verification_mode or dry_run:
            # In verification mode, log what WOULD happen but don't execute outreach
            logger.info(f"🔍 VERIFICATION MODE: Would handle {event_type.value} event")
            result["message"] = f"Verification mode: {event_type.value} event processed but no real actions taken"
            result["would_trigger"] = self._get_agent_types_for_event(event_type)
            return result

        # Full execution - trigger agents for this event
        try:
            trigger_result = await self.trigger_agent_by_event(event_type, event_data)
            result.update(trigger_result)
            result["actions_taken"] = trigger_result.get("triggered_agents", [])

            # Log to database for observability
            self._log_business_event(event_type, event_data, result)

        except Exception as e:
            logger.error(f"Failed to handle business event {event_type.value}: {e}")
            result["success"] = False
            result["error"] = str(e)

        return result

    def _get_agent_types_for_event(self, event_type: BusinessEventType) -> list[str]:
        """Get the agent types that would be triggered for an event"""
        event_agent_mapping = {
            BusinessEventType.CUSTOMER_INQUIRY: ["customer_success", "support"],
            BusinessEventType.LEAD_CREATED: ["lead_generation", "revenue"],
            BusinessEventType.DEAL_CLOSED: ["revenue", "customer_success"],
            BusinessEventType.SUPPORT_REQUEST: ["support", "customer_success"],
            BusinessEventType.REVENUE_OPPORTUNITY: ["revenue", "pricing"],
            BusinessEventType.SYSTEM_ALERT: ["system_improvement", "devops_optimization"],
            BusinessEventType.SCHEDULED_TASK: ["analytics", "monitoring"],
            BusinessEventType.PERFORMANCE_THRESHOLD: ["system_improvement", "devops_optimization"],
            BusinessEventType.NEW_CUSTOMER: ["customer_success", "lead_generation"],
            BusinessEventType.ESTIMATE_REQUESTED: ["pricing", "revenue"],
            BusinessEventType.INVOICE_OVERDUE: ["revenue", "customer_success"],
            BusinessEventType.SCHEDULING_CONFLICT: ["system_improvement", "devops_optimization"],
            BusinessEventType.SYSTEM_HEALTH_CHECK: ["system_improvement", "devops_optimization"],
            BusinessEventType.CUSTOMER_CHURN_RISK: ["customer_success", "revenue"],
            BusinessEventType.QUOTE_REQUESTED: ["pricing", "revenue"],
            BusinessEventType.PAYMENT_RECEIVED: ["revenue", "customer_success"],
            BusinessEventType.JOB_SCHEDULED: ["system_improvement", "analytics"],
        }
        return event_agent_mapping.get(event_type, [])

    def _log_business_event(
        self,
        event_type: BusinessEventType,
        event_data: dict[str, Any],
        result: dict[str, Any]
    ):
        """Log business event handling to database for observability"""
        try:
            with self._get_db_connection() as conn:
                if not conn:
                    return

                cur = conn.cursor()
                # Serialize to ensure all datetime/Decimal objects are converted
                safe_data = json_safe_serialize({
                    "event_data": event_data,
                    "result": result
                })
                cur.execute("""
                    INSERT INTO agent_activation_log
                    (agent_id, agent_name, tenant_id, event_type, event_data, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (
                    'system',  # System-level agent ID for business events
                    'BusinessEventLogger',  # Agent name
                    self.tenant_id,
                    f"business_event:{event_type.value}",
                    json.dumps(safe_data)
                ))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.warning(f"Failed to log business event: {e}")
