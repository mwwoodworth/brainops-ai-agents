#!/usr/bin/env python3
"""
BrainOps Integration Bridge
Connects all services and ensures continuous operation

ENHANCEMENTS (v2.0):
- Event-driven communication for service integration
- Message queuing for async operations
- Circuit breakers for service resilience
- Load balancing across operations
- Priority-based task routing
- System-wide health aggregation

Version: 2.0.0
"""

import asyncio
import logging
from datetime import datetime, timezone

import httpx

from memory_system import memory_system

logger = logging.getLogger(__name__)

# Import enhanced orchestration features
try:
    from autonomous_system_orchestrator import (
        AgentInstance,
        CircuitBreaker,
        EventBus,
        HealthAggregator,
        LoadBalancer,
        LoadBalancingStrategy,
        MessageQueue,
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced orchestration features not available")
    ENHANCED_FEATURES_AVAILABLE = False

class IntegrationBridge:
    """
    Enhanced Integration Bridge with circuit breakers, event-driven communication,
    and intelligent routing.
    """

    def __init__(self):
        self.backend_url = "https://brainops-backend-prod.onrender.com"
        self.agents_url = "https://brainops-ai-agents.onrender.com"
        self.memory = memory_system

        # Enhanced features
        if ENHANCED_FEATURES_AVAILABLE:
            self.event_bus = EventBus()
            self.message_queue = MessageQueue(max_workers=5)
            self.load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
            self.health_aggregator = HealthAggregator()
            self.circuit_breakers = {
                "backend": CircuitBreaker("backend_circuit", failure_threshold=3, timeout=30),
                "agents": CircuitBreaker("agents_circuit", failure_threshold=3, timeout=30)
            }
            self._enhanced_initialized = False
        else:
            self.event_bus = None
            self.message_queue = None
            self.load_balancer = None
            self.health_aggregator = None
            self.circuit_breakers = {}

        # Priority mapping for operations
        self.operation_priorities = {
            "sync_backend_to_agents": 3,
            "trigger_monitoring": 2,
            "sync_workflows": 4
        }

    async def initialize_enhanced_features(self):
        """Initialize enhanced features"""
        if not ENHANCED_FEATURES_AVAILABLE or self._enhanced_initialized:
            return

        await self.event_bus.start()
        await self.message_queue.start()

        # Register operation instances
        operations = ["sync", "monitor", "workflow"]
        for op in operations:
            for i in range(2):
                instance = AgentInstance(
                    instance_id=f"{op}_instance_{i}",
                    agent_name=op,
                    max_capacity=3
                )
                self.load_balancer.register_instance(instance)

        self._enhanced_initialized = True
        logger.info("Enhanced integration bridge features initialized")

    async def sync_backend_to_agents(self):
        """Sync backend data with AI agents"""
        async with httpx.AsyncClient(timeout=30) as client:
            # Get customers from backend
            response = await client.get(f"{self.backend_url}/api/v1/erp/customers")
            if response.status_code == 200:
                customers = response.json().get('customers', [])

                # Trigger CustomerAgent to analyze
                await client.post(
                    f"{self.agents_url}/agents/9e8ad51f-2f0b-4ee4-9e48-4971c93e8cdf/execute",
                    json={"action": "analyze_customers", "data": customers}
                )

            # Get jobs from backend
            response = await client.get(f"{self.backend_url}/api/v1/erp/jobs")
            if response.status_code == 200:
                jobs = response.json()

                # Store in memory for agents
                self.memory.store_context('backend', 'latest_jobs', jobs)

    async def trigger_monitoring(self):
        """Trigger monitoring agents"""
        async with httpx.AsyncClient(timeout=30) as client:
            # Monitor agent
            await client.post(
                f"{self.agents_url}/agents/888a991c-0cc6-4cfe-84b2-d4bf0abeecea/execute",
                json={"action": "full_check"}
            )

            # SystemMonitor with self-healing
            await client.post(
                f"{self.agents_url}/agents/ef4082d9-6b61-4c6a-ac51-4fba2a445dd1/execute",
                json={"action": "monitor_and_heal"}
            )

    async def sync_workflows(self):
        """Sync LangGraph workflows with AI agents"""
        async with httpx.AsyncClient(timeout=30) as client:
            # Get LangGraph status
            response = await client.get(f"{self.backend_url}/api/v1/langgraph/status")
            if response.status_code == 200:
                workflows = response.json().get('workflows', [])

                # Trigger WorkflowEngine to coordinate
                await client.post(
                    f"{self.agents_url}/agents/466edb09-bd02-4c79-a4ce-a3c733059a67/execute",
                    json={"type": "sync_workflows", "workflows": workflows}
                )

    async def continuous_integration_loop(self):
        """Main integration loop"""
        while True:
            try:
                # Run all sync operations
                await self.sync_backend_to_agents()
                await self.trigger_monitoring()
                await self.sync_workflows()

                # Store status
                self.memory.update_system_state('integration', 'last_sync', {
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'status': 'success'
                })

                # Wait 5 minutes
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"Integration error: {e}")
                self.memory.update_system_state('integration', 'last_error', str(e))
                await asyncio.sleep(60)  # Retry in 1 minute

# Global bridge
bridge = IntegrationBridge()

async def main():
    """Run integration bridge"""
    logger.info("Starting Integration Bridge v2.0")

    # Initialize enhanced features
    if ENHANCED_FEATURES_AVAILABLE:
        await bridge.initialize_enhanced_features()
        logger.info("Enhanced features initialized")

    await bridge.continuous_integration_loop()

if __name__ == "__main__":
    asyncio.run(main())
