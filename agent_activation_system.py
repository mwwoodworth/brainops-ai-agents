"""
Agent Activation System
Manages the lifecycle and activation of AI agents.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BusinessEventType(Enum):
    """Types of business events that can trigger agent activation"""
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
    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for AgentActivationSystem")
        self.tenant_id = tenant_id

    async def activate_agent(self, agent_id: str) -> bool:
        """Activate an agent"""
        logger.info(f"Activating agent {agent_id} for tenant {self.tenant_id}")
        return True

    async def deactivate_agent(self, agent_id: str) -> bool:
        """Deactivate an agent"""
        logger.info(f"Deactivating agent {agent_id} for tenant {self.tenant_id}")
        return True

    async def get_agent_status(self, agent_id: str) -> str:
        """Get current status of an agent"""
        return "active"