"""
Agent Activation System
Manages the lifecycle and activation of AI agents.
Implements real database operations for agent state management.
"""

import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

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


# Singleton instances per tenant
_activation_systems: Dict[str, "AgentActivationSystem"] = {}


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
        """Get database connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def activate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Activate an agent with real database update"""
        result = {
            "agent_id": agent_id,
            "action": "activate",
            "timestamp": datetime.utcnow().isoformat()
        }

        conn = self._get_db_connection()
        if not conn:
            result["success"] = False
            result["error"] = "Database connection failed"
            return result

        try:
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
                    INSERT INTO agent_activation_log (agent_id, tenant_id, action, timestamp)
                    VALUES (%s, %s, 'activate', NOW())
                """, (agent_id, self.tenant_id))
            else:
                result["success"] = False
                result["error"] = "Agent not found"

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to activate agent {agent_id}: {e}")
            result["success"] = False
            result["error"] = str(e)
            if conn:
                conn.close()

        return result

    async def deactivate_agent(self, agent_id: str) -> Dict[str, Any]:
        """Deactivate an agent with real database update"""
        result = {
            "agent_id": agent_id,
            "action": "deactivate",
            "timestamp": datetime.utcnow().isoformat()
        }

        conn = self._get_db_connection()
        if not conn:
            result["success"] = False
            result["error"] = "Database connection failed"
            return result

        try:
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
                    INSERT INTO agent_activation_log (agent_id, tenant_id, action, timestamp)
                    VALUES (%s, %s, 'deactivate', NOW())
                """, (agent_id, self.tenant_id))
            else:
                result["success"] = False
                result["error"] = "Agent not found"

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to deactivate agent {agent_id}: {e}")
            result["success"] = False
            result["error"] = str(e)
            if conn:
                conn.close()

        return result

    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current status of an agent from database"""
        result = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat()
        }

        conn = self._get_db_connection()
        if not conn:
            result["status"] = "unknown"
            result["error"] = "Database connection failed"
            return result

        try:
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
            conn.close()

        except Exception as e:
            logger.error(f"Failed to get agent status {agent_id}: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            if conn:
                conn.close()

        return result

    async def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "agents": [],
            "summary": {}
        }

        conn = self._get_db_connection()
        if not conn:
            result["error"] = "Database connection failed"
            return result

        try:
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
            conn.close()

        except Exception as e:
            logger.error(f"Failed to get all agents status: {e}")
            result["error"] = str(e)
            if conn:
                conn.close()

        return result

    async def trigger_agent_by_event(self, event_type: BusinessEventType, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger appropriate agents based on business event"""
        result = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "triggered_agents": []
        }

        # Map event types to agent types
        event_agent_mapping = {
            BusinessEventType.CUSTOMER_INQUIRY: ["customer_success", "support"],
            BusinessEventType.LEAD_CREATED: ["lead_generation", "revenue"],
            BusinessEventType.DEAL_CLOSED: ["revenue", "customer_success"],
            BusinessEventType.SUPPORT_REQUEST: ["support", "customer_success"],
            BusinessEventType.REVENUE_OPPORTUNITY: ["revenue", "pricing"],
            BusinessEventType.SYSTEM_ALERT: ["system_improvement", "devops_optimization"],
            BusinessEventType.SCHEDULED_TASK: ["analytics", "monitoring"],
            BusinessEventType.PERFORMANCE_THRESHOLD: ["system_improvement", "devops_optimization"],
        }

        target_types = event_agent_mapping.get(event_type, [])

        conn = self._get_db_connection()
        if not conn:
            result["error"] = "Database connection failed"
            return result

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Find active agents matching event type
            cur.execute("""
                SELECT id, name, type
                FROM ai_agents
                WHERE status = 'active'
                AND type = ANY(%s)
            """, (target_types,))

            agents = cur.fetchall()

            for agent in agents:
                # Queue task for each matching agent
                cur.execute("""
                    INSERT INTO ai_autonomous_tasks
                    (task_type, priority, status, trigger_type, trigger_condition, agent_id, created_at)
                    VALUES (%s, %s, 'pending', %s, %s, %s, NOW())
                    RETURNING id
                """, (
                    f"{event_type.value}_handler",
                    'high' if event_type in [BusinessEventType.SYSTEM_ALERT, BusinessEventType.SUPPORT_REQUEST] else 'medium',
                    event_type.value,
                    json.dumps(event_data),
                    agent['id']
                ))
                task = cur.fetchone()

                result["triggered_agents"].append({
                    "agent_id": agent['id'],
                    "agent_name": agent['name'],
                    "task_id": str(task['id']) if task else None
                })

            conn.commit()
            cur.close()
            conn.close()

            result["success"] = True
            result["agents_triggered"] = len(result["triggered_agents"])
            logger.info(f"Triggered {len(result['triggered_agents'])} agents for event {event_type.value}")

        except Exception as e:
            logger.error(f"Failed to trigger agents for event {event_type.value}: {e}")
            result["success"] = False
            result["error"] = str(e)
            if conn:
                conn.close()

        return result
