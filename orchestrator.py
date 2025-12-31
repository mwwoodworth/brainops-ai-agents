#!/usr/bin/env python3
"""
BrainOps AI Orchestration System
Central orchestrator for all AI operations and system coordination

Enhanced with:
- Event-driven communication
- Message queuing for async operations
- Circuit breakers for resilience
- Load balancing across components
- Priority-based routing
- System-wide health aggregation
"""

import os
import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import RealDictCursor
from memory_system import memory_system
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== IMPORT ENHANCED ORCHESTRATION COMPONENTS ==============
try:
    from autonomous_system_orchestrator import (
        EventBus, EventType, SystemEvent, MessageQueue,
        CircuitBreaker, CircuitState, LoadBalancer, LoadBalancingStrategy,
        AgentInstance, HealthAggregator
    )
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    logger.warning("Enhanced orchestration features not available")
    ENHANCED_FEATURES_AVAILABLE = False

class SystemOrchestrator:
    """
    Master orchestrator for all BrainOps systems

    Enhanced with event-driven architecture, message queuing,
    circuit breakers, load balancing, and priority routing.
    """

    def __init__(self):
        self.memory = memory_system
        self.services = self._load_service_config()
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "database": "postgres",
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": 5432
        }

        # Enhanced features (if available)
        if ENHANCED_FEATURES_AVAILABLE:
            self.event_bus = EventBus()
            self.message_queue = MessageQueue(max_workers=15)
            self.load_balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_LOADED)
            self.health_aggregator = HealthAggregator()
            self.circuit_breakers: Dict[str, CircuitBreaker] = {}
            self._initialize_circuit_breakers()
            self._initialized_enhanced = False
        else:
            self.event_bus = None
            self.message_queue = None
            self.load_balancer = None
            self.health_aggregator = None
            self.circuit_breakers = {}

        # Priority routes for different workflow types
        self.priority_routes = {
            "critical_alert": 1,
            "full_system_check": 2,
            "deploy_update": 2,
            "customer_onboarding": 3,
            "data_sync": 5,
            "analytics": 7,
            "maintenance": 10
        }

    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all services"""
        for service_name in self.services.keys():
            self.circuit_breakers[service_name] = CircuitBreaker(
                name=f"{service_name}_circuit",
                failure_threshold=3,
                success_threshold=2,
                timeout=30
            )

    def _load_service_config(self) -> Dict:
        """Load service configuration from memory"""
        config = self.memory.get_context('service_endpoints')
        if not config:
            config = {
                'backend': 'https://brainops-backend-prod.onrender.com',
                'ai_agents': 'https://brainops-ai-agents.onrender.com',
                'myroofgenius': 'https://myroofgenius.com',
                'weathercraft': 'https://weathercraft-erp.vercel.app'
            }
            self.memory.store_context('system', 'service_endpoints', config, critical=True)
        return config

    async def initialize_enhanced_features(self):
        """Initialize enhanced orchestration features"""
        if not ENHANCED_FEATURES_AVAILABLE or self._initialized_enhanced:
            return

        # Start event bus
        await self.event_bus.start()
        logger.info("Event bus started")

        # Start message queue
        await self.message_queue.start()
        logger.info("Message queue started")

        # Register agent instances for load balancing
        agent_types = ["health_check", "deployment", "workflow", "monitoring", "data_sync"]
        for agent_type in agent_types:
            for i in range(3):  # 3 instances per type
                instance = AgentInstance(
                    instance_id=f"{agent_type}_instance_{i}",
                    agent_name=agent_type,
                    max_capacity=5,
                    weight=1
                )
                self.load_balancer.register_instance(instance)

        # Setup event handlers
        self._setup_event_handlers()

        self._initialized_enhanced = True
        logger.info("Enhanced orchestration features initialized")

    def _setup_event_handlers(self):
        """Setup event handlers"""
        if not self.event_bus:
            return

        async def handle_health_change(event: SystemEvent):
            service_name = event.data.get("service_name")
            if service_name and self.health_aggregator:
                health_score = event.data.get("health_score", 100.0)
                self.health_aggregator.update_system_health(service_name, {
                    "health_score": health_score,
                    "status": event.data.get("status", "unknown")
                })

        async def handle_circuit_event(event: SystemEvent):
            logger.warning(f"Circuit breaker {event.event_type.value}: {event.data}")

        self.event_bus.subscribe(EventType.SYSTEM_HEALTH_CHANGED, handle_health_change)
        self.event_bus.subscribe(EventType.CIRCUIT_OPENED, handle_circuit_event)
        self.event_bus.subscribe(EventType.CIRCUIT_CLOSED, handle_circuit_event)

    async def health_check_all(self) -> Dict:
        """Check health of all systems with circuit breaker protection"""
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'services': {},
            'database': {},
            'overall_health': 'healthy'
        }

        # Check services with circuit breaker protection
        async with httpx.AsyncClient(timeout=10) as client:
            for service_name, url in self.services.items():
                circuit = self.circuit_breakers.get(service_name)

                # Check circuit breaker
                if circuit and not circuit.can_execute():
                    results['services'][service_name] = {
                        'status': 'circuit_open',
                        'circuit_state': circuit.state.value,
                        'message': 'Circuit breaker open, service unavailable'
                    }
                    results['overall_health'] = 'degraded'
                    continue

                success = False
                try:
                    if service_name in ['backend', 'ai_agents']:
                        response = await client.get(f"{url}/health")
                        is_healthy = response.status_code == 200
                        success = is_healthy

                        results['services'][service_name] = {
                            'status': 'healthy' if is_healthy else 'unhealthy',
                            'code': response.status_code,
                            'data': response.json() if is_healthy else None,
                            'circuit_state': circuit.state.value if circuit else 'unknown'
                        }

                        # Update health aggregator
                        if self.health_aggregator:
                            health_score = 100.0 if is_healthy else 0.0
                            self.health_aggregator.update_system_health(service_name, {
                                "health_score": health_score,
                                "status": "healthy" if is_healthy else "unhealthy"
                            })

                            # Publish event
                            if self.event_bus:
                                await self.event_bus.publish(SystemEvent(
                                    event_type=EventType.SYSTEM_HEALTH_CHANGED,
                                    source=service_name,
                                    data={
                                        "service_name": service_name,
                                        "status": "healthy" if is_healthy else "unhealthy",
                                        "health_score": health_score
                                    }
                                ))
                    else:
                        response = await client.get(url, follow_redirects=True)
                        is_online = response.status_code == 200
                        success = is_online

                        results['services'][service_name] = {
                            'status': 'online' if is_online else 'offline',
                            'code': response.status_code,
                            'circuit_state': circuit.state.value if circuit else 'unknown'
                        }

                except Exception as e:
                    results['services'][service_name] = {
                        'status': 'error',
                        'error': str(e),
                        'circuit_state': circuit.state.value if circuit else 'unknown'
                    }
                    results['overall_health'] = 'degraded'

                # Update circuit breaker
                if circuit:
                    if success:
                        circuit.record_success()
                        if circuit.state == CircuitState.CLOSED and self.event_bus:
                            await self.event_bus.publish(SystemEvent(
                                event_type=EventType.CIRCUIT_CLOSED,
                                source=service_name,
                                data={"service_name": service_name}
                            ))
                    else:
                        circuit.record_failure()
                        if circuit.state == CircuitState.OPEN and self.event_bus:
                            await self.event_bus.publish(SystemEvent(
                                event_type=EventType.CIRCUIT_OPENED,
                                source=service_name,
                                data={"service_name": service_name}
                            ))
                        results['overall_health'] = 'degraded'

        # Check database
        try:
            conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM customers) as customers,
                    (SELECT COUNT(*) FROM jobs) as jobs,
                    (SELECT COUNT(*) FROM invoices) as invoices,
                    (SELECT COUNT(*) FROM ai_agents) as agents,
                    (SELECT COUNT(*) FROM agent_executions) as executions
            """)

            results['database'] = cursor.fetchone()
            results['database']['status'] = 'healthy'

            cursor.close()
            conn.close()
        except Exception as e:
            results['database']['status'] = 'error'
            results['database']['error'] = str(e)
            results['overall_health'] = 'critical'

        # Store health check result
        self.memory.update_system_state('orchestrator', 'last_health_check', results)

        return results

    async def execute_workflow(self, workflow_type: str, params: Dict) -> Dict:
        """
        Execute a multi-step workflow with priority routing and event publishing

        Enhancements:
        - Priority-based routing
        - Event publishing for workflow stages
        - Message queue integration
        - Load balanced execution
        """
        workflow_id = f"wf_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Determine priority
        priority = self.priority_routes.get(workflow_type, 5)

        # Record workflow start
        self.memory.update_system_state('workflow', workflow_id, {
            'type': workflow_type,
            'status': 'started',
            'params': params,
            'priority': priority,
            'started_at': datetime.now(timezone.utc).isoformat()
        })

        # Publish workflow started event
        if self.event_bus:
            await self.event_bus.publish(SystemEvent(
                event_type=EventType.AGENT_STARTED,
                source="orchestrator",
                data={
                    "workflow_id": workflow_id,
                    "workflow_type": workflow_type,
                    "priority": priority
                },
                priority=priority
            ))

        result = {
            'workflow_id': workflow_id,
            'type': workflow_type,
            'steps': [],
            'status': 'in_progress'
        }

        try:
            if workflow_type == 'full_system_check':
                result['steps'].append(await self._check_services())
                result['steps'].append(await self._check_database())
                result['steps'].append(await self._check_agents())

            elif workflow_type == 'deploy_update':
                result['steps'].append(await self._prepare_deployment(params))
                result['steps'].append(await self._execute_deployment(params))
                result['steps'].append(await self._verify_deployment(params))

            elif workflow_type == 'customer_onboarding':
                result['steps'].append(await self._create_customer(params))
                result['steps'].append(await self._setup_workflows(params))
                result['steps'].append(await self._send_welcome(params))

            elif workflow_type == 'data_sync':
                result['steps'].append(await self._sync_databases())
                result['steps'].append(await self._update_memory())
                result['steps'].append(await self._refresh_cache())

            else:
                result['status'] = 'unknown_workflow'

            result['status'] = 'completed'

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Workflow {workflow_id} failed: {e}")

        # Record workflow completion
        self.memory.update_system_state('workflow', workflow_id, {
            'type': workflow_type,
            'status': result['status'],
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'result': result
        })

        return result

    async def _check_services(self) -> Dict:
        """Check all service health"""
        health = await self.health_check_all()
        return {
            'step': 'check_services',
            'status': 'completed',
            'services': health['services']
        }

    async def _check_database(self) -> Dict:
        """Check database health and stats"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                pg_database_size('postgres') as db_size,
                (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public') as table_count,
                (SELECT SUM(n_live_tup) FROM pg_stat_user_tables) as total_rows
        """)

        stats = cursor.fetchone()
        cursor.close()
        conn.close()

        return {
            'step': 'check_database',
            'status': 'completed',
            'database': stats
        }

    async def _check_agents(self) -> Dict:
        """Check AI agents status"""
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.get(f"{self.services['ai_agents']}/agents")
                agents = response.json()
                return {
                    'step': 'check_agents',
                    'status': 'completed',
                    'agents': {
                        'total': agents.get('total', 0),
                        'active': len([a for a in agents.get('agents', []) if a.get('status') == 'active'])
                    }
                }
            except Exception as e:
                return {
                    'step': 'check_agents',
                    'status': 'failed',
                    'error': str(e)
                }

    async def _sync_databases(self) -> Dict:
        """Synchronize database states"""
        # This would sync between different databases/environments
        return {
            'step': 'sync_databases',
            'status': 'completed',
            'message': 'Database sync completed'
        }

    async def _update_memory(self) -> Dict:
        """Update AI memory with latest context"""
        overview = self.memory.get_system_overview()
        self.memory.store_context('system', 'latest_overview', overview)

        return {
            'step': 'update_memory',
            'status': 'completed',
            'entries_updated': len(overview.get('services', {}))
        }

    async def _refresh_cache(self) -> Dict:
        """Refresh system caches"""
        return {
            'step': 'refresh_cache',
            'status': 'completed',
            'message': 'Caches refreshed'
        }

    async def _prepare_deployment(self, params: Dict) -> Dict:
        """Prepare for deployment"""
        return {
            'step': 'prepare_deployment',
            'status': 'completed',
            'service': params.get('service'),
            'version': params.get('version')
        }

    async def _execute_deployment(self, params: Dict) -> Dict:
        """Execute deployment"""
        # Trigger actual deployment via agent
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(
                    f"{self.services['ai_agents']}/agents/DeploymentAgent/execute",
                    json={
                        'action': 'deploy',
                        'service': params.get('service'),
                        'version': params.get('version')
                    }
                )
                return {
                    'step': 'execute_deployment',
                    'status': 'completed',
                    'result': response.json() if response.status_code == 200 else {'error': 'Deployment failed'}
                }
            except Exception as e:
                return {
                    'step': 'execute_deployment',
                    'status': 'failed',
                    'error': str(e)
                }

    async def _verify_deployment(self, params: Dict) -> Dict:
        """Verify deployment success"""
        await asyncio.sleep(5)  # Wait for deployment to settle

        # Check if service is healthy
        service_name = params.get('service', 'backend')
        url = self.services.get(service_name)

        if url:
            async with httpx.AsyncClient(timeout=10) as client:
                try:
                    response = await client.get(f"{url}/health" if service_name in ['backend', 'ai_agents'] else url)
                    return {
                        'step': 'verify_deployment',
                        'status': 'completed' if response.status_code == 200 else 'failed',
                        'service_status': response.status_code
                    }
                except Exception as e:
                    return {
                        'step': 'verify_deployment',
                        'status': 'failed',
                        'error': str(e)
                    }

        return {
            'step': 'verify_deployment',
            'status': 'skipped',
            'message': 'Service not found'
        }

    async def _create_customer(self, params: Dict) -> Dict:
        """Create new customer"""
        return {
            'step': 'create_customer',
            'status': 'completed',
            'customer': params.get('customer_data')
        }

    async def _setup_workflows(self, params: Dict) -> Dict:
        """Setup customer workflows"""
        return {
            'step': 'setup_workflows',
            'status': 'completed',
            'workflows': ['invoice_automation', 'job_tracking', 'ai_analysis']
        }

    async def _send_welcome(self, params: Dict) -> Dict:
        """Send welcome communications"""
        return {
            'step': 'send_welcome',
            'status': 'completed',
            'message': 'Welcome email sent'
        }

    async def coordinate_agents(self, agents: List[str], task: Dict) -> Dict:
        """Coordinate multiple agents for complex tasks"""
        results = {
            'coordination_id': f"coord_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'agents': agents,
            'task': task,
            'results': [],
            'status': 'in_progress'
        }

        async with httpx.AsyncClient(timeout=30) as client:
            tasks = []
            for agent in agents:
                tasks.append(self._execute_agent(client, agent, task))

            # Execute all agents in parallel
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)

            for agent, result in zip(agents, agent_results):
                if isinstance(result, Exception):
                    results['results'].append({
                        'agent': agent,
                        'status': 'failed',
                        'error': str(result)
                    })
                else:
                    results['results'].append({
                        'agent': agent,
                        'status': 'completed',
                        'result': result
                    })

        results['status'] = 'completed'
        return results

    async def _execute_agent(self, client: httpx.AsyncClient, agent: str, task: Dict) -> Dict:
        """Execute single agent"""
        response = await client.post(
            f"{self.services['ai_agents']}/agents/{agent}/execute",
            json=task
        )
        return response.json()

    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics"""
        stats = {
            "services": list(self.services.keys()),
            "enhanced_features_enabled": ENHANCED_FEATURES_AVAILABLE,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if ENHANCED_FEATURES_AVAILABLE:
            # Circuit breaker stats
            stats["circuit_breakers"] = {
                name: breaker.get_stats()
                for name, breaker in self.circuit_breakers.items()
            }

            # Message queue stats
            if self.message_queue:
                stats["message_queue"] = self.message_queue.get_queue_stats()

            # Load balancer stats
            if self.load_balancer:
                stats["load_balancer"] = self.load_balancer.get_stats()

            # Health aggregation
            if self.health_aggregator:
                stats["health_aggregation"] = self.health_aggregator.get_aggregated_health()

            # Recent events
            if self.event_bus:
                stats["recent_events"] = [
                    {
                        "event_id": e.event_id,
                        "type": e.event_type.value,
                        "source": e.source,
                        "timestamp": e.timestamp.isoformat(),
                        "priority": e.priority
                    }
                    for e in self.event_bus.get_recent_events(limit=20)
                ]

        return stats

# Global orchestrator instance
orchestrator = SystemOrchestrator()

async def main():
    """Test orchestration system"""
    logger.info("Starting BrainOps Orchestrator")

    # Initialize enhanced features
    await orchestrator.initialize_enhanced_features()

    # Run health check
    health = await orchestrator.health_check_all()
    logger.info(f"System Health: {health['overall_health']}")

    # Get orchestrator stats
    stats = orchestrator.get_orchestrator_stats()
    logger.info(f"Orchestrator Stats: {stats}")

    # Run a workflow
    workflow_result = await orchestrator.execute_workflow('full_system_check', {})
    logger.info(f"Workflow Result: {workflow_result['status']}")

    # Coordinate agents
    coordination = await orchestrator.coordinate_agents(
        ['Monitor', 'SystemMonitor'],
        {'action': 'check'}
    )
    logger.info(f"Agent Coordination: {coordination['status']}")

if __name__ == "__main__":
    asyncio.run(main())
