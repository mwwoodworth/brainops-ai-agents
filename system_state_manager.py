"""
System State Management Module
Tracks and manages overall system state, health, and operational status
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"

class SystemComponent(Enum):
    DATABASE = "database"
    AI_AGENTS = "ai_agents"
    VECTOR_MEMORY = "vector_memory"
    REVENUE_SYSTEM = "revenue_system"
    PRICING_ENGINE = "pricing_engine"
    ACQUISITION_SYSTEM = "acquisition_system"
    NOTEBOOK_LM = "notebook_lm"
    CONVERSATION_MEMORY = "conversation_memory"
    ORCHESTRATOR = "orchestrator"
    API_GATEWAY = "api_gateway"
    WEB_FRONTEND = "web_frontend"
    TASK_SCHEDULER = "task_scheduler"
    MONITORING = "monitoring"

@dataclass
class ComponentState:
    component: str
    status: ServiceStatus
    last_check: datetime
    uptime_seconds: int
    error_count: int
    success_rate: float
    latency_ms: float
    metadata: Dict[str, Any]
    dependencies: List[str]
    health_score: float

@dataclass
class SystemSnapshot:
    snapshot_id: str
    timestamp: datetime
    overall_status: ServiceStatus
    health_score: float
    active_components: int
    failed_components: int
    warning_count: int
    error_count: int
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    active_sessions: int
    pending_tasks: int
    completed_tasks: int

class SystemStateManager:
    """Manages overall system state and health monitoring"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
            'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
            'port': os.getenv('DB_PORT', 5432)
        }
        self.components = {}
        self.current_state = None
        self.state_history = []
        self.alerts = []
        self.recovery_procedures = {}
        self._initialize_database()
        self._load_recovery_procedures()

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)

    def _initialize_database(self):
        """Initialize database tables for state management"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Create system state table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_system_state (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    snapshot_id VARCHAR(255) UNIQUE NOT NULL,
                    timestamp TIMESTAMPTZ DEFAULT NOW(),
                    overall_status VARCHAR(50) NOT NULL,
                    health_score FLOAT DEFAULT 0.0,
                    active_components INT DEFAULT 0,
                    failed_components INT DEFAULT 0,
                    warning_count INT DEFAULT 0,
                    error_count INT DEFAULT 0,
                    performance_metrics JSONB DEFAULT '{}'::jsonb,
                    resource_usage JSONB DEFAULT '{}'::jsonb,
                    active_sessions INT DEFAULT 0,
                    pending_tasks INT DEFAULT 0,
                    completed_tasks INT DEFAULT 0,
                    snapshot_data JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create component state table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_component_state (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    component VARCHAR(100) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    last_check TIMESTAMPTZ DEFAULT NOW(),
                    uptime_seconds INT DEFAULT 0,
                    error_count INT DEFAULT 0,
                    success_rate FLOAT DEFAULT 100.0,
                    latency_ms FLOAT DEFAULT 0.0,
                    health_score FLOAT DEFAULT 100.0,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    dependencies TEXT[] DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(component)
                )
            """)

            # Create state transitions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_state_transitions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    component VARCHAR(100) NOT NULL,
                    from_state VARCHAR(50),
                    to_state VARCHAR(50) NOT NULL,
                    trigger VARCHAR(255),
                    reason TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    transition_time TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create alerts table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_system_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    alert_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    component VARCHAR(100),
                    message TEXT NOT NULL,
                    details JSONB DEFAULT '{}'::jsonb,
                    status VARCHAR(50) DEFAULT 'active',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    acknowledged_at TIMESTAMPTZ,
                    resolved_at TIMESTAMPTZ,
                    resolution_notes TEXT
                )
            """)

            # Create recovery actions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_recovery_actions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    component VARCHAR(100) NOT NULL,
                    error_type VARCHAR(255) NOT NULL,
                    action_taken TEXT NOT NULL,
                    success BOOLEAN DEFAULT false,
                    execution_time_ms INT,
                    error_message TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    executed_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            conn.commit()
            logger.info("System state tables initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing state tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def _load_recovery_procedures(self):
        """Load automated recovery procedures"""
        self.recovery_procedures = {
            SystemComponent.DATABASE: [
                {"error": "connection_failed", "action": "reconnect", "max_retries": 3},
                {"error": "slow_query", "action": "analyze_tables", "threshold": 5000}
            ],
            SystemComponent.AI_AGENTS: [
                {"error": "execution_timeout", "action": "kill_and_restart", "timeout": 300},
                {"error": "memory_exceeded", "action": "garbage_collect", "threshold": 0.9}
            ],
            SystemComponent.VECTOR_MEMORY: [
                {"error": "embedding_failed", "action": "retry_with_fallback", "max_retries": 2},
                {"error": "search_timeout", "action": "optimize_index", "threshold": 10000}
            ],
            SystemComponent.API_GATEWAY: [
                {"error": "rate_limit", "action": "throttle_requests", "window": 60},
                {"error": "5xx_errors", "action": "circuit_breaker", "threshold": 10}
            ]
        }

    async def check_component_health(self, component: SystemComponent) -> ComponentState:
        """Check health of a specific component"""
        try:
            if component == SystemComponent.DATABASE:
                return await self._check_database_health()
            elif component == SystemComponent.AI_AGENTS:
                return await self._check_agents_health()
            elif component == SystemComponent.VECTOR_MEMORY:
                return await self._check_vector_memory_health()
            elif component == SystemComponent.API_GATEWAY:
                return await self._check_api_health()
            else:
                return await self._generic_health_check(component)

        except Exception as e:
            logger.error(f"Error checking {component.value} health: {e}")
            return ComponentState(
                component=component.value,
                status=ServiceStatus.UNKNOWN,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=1,
                success_rate=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)},
                dependencies=[],
                health_score=0.0
            )

    async def _check_database_health(self) -> ComponentState:
        """Check database health"""
        start_time = datetime.now()
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check basic connectivity
            cur.execute("SELECT 1")

            # Check database size and connections
            cur.execute("""
                SELECT
                    pg_database_size('postgres') as db_size,
                    (SELECT COUNT(*) FROM pg_stat_activity) as connections,
                    (SELECT COUNT(*) FROM pg_stat_activity WHERE state = 'active') as active_queries
            """)
            stats = cur.fetchone()

            # Check table counts
            cur.execute("""
                SELECT COUNT(*) as table_count
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            table_stats = cur.fetchone()

            latency = (datetime.now() - start_time).total_seconds() * 1000

            conn.close()

            return ComponentState(
                component=SystemComponent.DATABASE.value,
                status=ServiceStatus.HEALTHY,
                last_check=datetime.now(),
                uptime_seconds=0,  # Would need to track this separately
                error_count=0,
                success_rate=100.0,
                latency_ms=latency,
                metadata={
                    "db_size_mb": stats['db_size'] / (1024*1024),
                    "connections": stats['connections'],
                    "active_queries": stats['active_queries'],
                    "table_count": table_stats['table_count']
                },
                dependencies=[],
                health_score=100.0
            )

        except Exception as e:
            return ComponentState(
                component=SystemComponent.DATABASE.value,
                status=ServiceStatus.DOWN,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=1,
                success_rate=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)},
                dependencies=[],
                health_score=0.0
            )

    async def _check_agents_health(self) -> ComponentState:
        """Check AI agents health"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check agent executions
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration
                FROM agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            stats = cur.fetchone()

            conn.close()

            success_rate = 0.0
            if stats and stats['total'] > 0:
                success_rate = (stats['completed'] / stats['total']) * 100

            status = ServiceStatus.HEALTHY
            if success_rate < 50:
                status = ServiceStatus.DEGRADED
            elif success_rate < 20:
                status = ServiceStatus.DOWN

            return ComponentState(
                component=SystemComponent.AI_AGENTS.value,
                status=status,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=stats['failed'] if stats else 0,
                success_rate=success_rate,
                latency_ms=stats['avg_duration'] * 1000 if stats and stats['avg_duration'] else 0,
                metadata={
                    "total_executions": stats['total'] if stats else 0,
                    "running": stats['running'] if stats else 0
                },
                dependencies=[SystemComponent.DATABASE.value],
                health_score=success_rate
            )

        except Exception as e:
            return ComponentState(
                component=SystemComponent.AI_AGENTS.value,
                status=ServiceStatus.UNKNOWN,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=1,
                success_rate=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)},
                dependencies=[SystemComponent.DATABASE.value],
                health_score=0.0
            )

    async def _check_vector_memory_health(self) -> ComponentState:
        """Check vector memory system health"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check vector memory table
            cur.execute("""
                SELECT
                    COUNT(*) as total_memories,
                    AVG(importance_score) as avg_importance,
                    MAX(last_accessed) as last_access
                FROM ai_persistent_memory
            """)
            stats = cur.fetchone()

            conn.close()

            # Calculate health based on recent activity
            health_score = 100.0
            if stats and stats['last_access']:
                hours_since_access = (datetime.now() - stats['last_access']).total_seconds() / 3600
                if hours_since_access > 24:
                    health_score = 50.0
                elif hours_since_access > 72:
                    health_score = 20.0

            return ComponentState(
                component=SystemComponent.VECTOR_MEMORY.value,
                status=ServiceStatus.HEALTHY if health_score > 50 else ServiceStatus.DEGRADED,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=0,
                success_rate=100.0,
                latency_ms=0.0,
                metadata={
                    "total_memories": stats['total_memories'] if stats else 0,
                    "avg_importance": float(stats['avg_importance']) if stats and stats['avg_importance'] else 0
                },
                dependencies=[SystemComponent.DATABASE.value],
                health_score=health_score
            )

        except Exception as e:
            return ComponentState(
                component=SystemComponent.VECTOR_MEMORY.value,
                status=ServiceStatus.UNKNOWN,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=1,
                success_rate=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)},
                dependencies=[SystemComponent.DATABASE.value],
                health_score=0.0
            )

    async def _check_api_health(self) -> ComponentState:
        """Check API gateway health"""
        try:
            # Check the Render API endpoint
            async with aiohttp.ClientSession() as session:
                start_time = datetime.now()
                async with session.get('https://brainops-ai-agents.onrender.com/health', timeout=5) as response:
                    latency = (datetime.now() - start_time).total_seconds() * 1000

                    if response.status == 200:
                        data = await response.json()
                        return ComponentState(
                            component=SystemComponent.API_GATEWAY.value,
                            status=ServiceStatus.HEALTHY,
                            last_check=datetime.now(),
                            uptime_seconds=0,
                            error_count=0,
                            success_rate=100.0,
                            latency_ms=latency,
                            metadata=data,
                            dependencies=[],
                            health_score=100.0
                        )
                    else:
                        return ComponentState(
                            component=SystemComponent.API_GATEWAY.value,
                            status=ServiceStatus.DEGRADED,
                            last_check=datetime.now(),
                            uptime_seconds=0,
                            error_count=1,
                            success_rate=50.0,
                            latency_ms=latency,
                            metadata={"status_code": response.status},
                            dependencies=[],
                            health_score=50.0
                        )

        except Exception as e:
            return ComponentState(
                component=SystemComponent.API_GATEWAY.value,
                status=ServiceStatus.DOWN,
                last_check=datetime.now(),
                uptime_seconds=0,
                error_count=1,
                success_rate=0.0,
                latency_ms=0.0,
                metadata={"error": str(e)},
                dependencies=[],
                health_score=0.0
            )

    async def _generic_health_check(self, component: SystemComponent) -> ComponentState:
        """Generic health check for components"""
        return ComponentState(
            component=component.value,
            status=ServiceStatus.UNKNOWN,
            last_check=datetime.now(),
            uptime_seconds=0,
            error_count=0,
            success_rate=100.0,
            latency_ms=0.0,
            metadata={"note": "Generic health check"},
            dependencies=[],
            health_score=75.0
        )

    async def perform_full_system_check(self) -> SystemSnapshot:
        """Perform complete system health check"""
        snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check all components
        components_to_check = [
            SystemComponent.DATABASE,
            SystemComponent.AI_AGENTS,
            SystemComponent.VECTOR_MEMORY,
            SystemComponent.API_GATEWAY,
            SystemComponent.REVENUE_SYSTEM,
            SystemComponent.PRICING_ENGINE,
            SystemComponent.ACQUISITION_SYSTEM,
            SystemComponent.NOTEBOOK_LM,
            SystemComponent.CONVERSATION_MEMORY
        ]

        component_states = []
        for component in components_to_check:
            state = await self.check_component_health(component)
            component_states.append(state)
            self.components[component.value] = state

        # Calculate overall metrics
        active_components = sum(1 for s in component_states if s.status == ServiceStatus.HEALTHY)
        failed_components = sum(1 for s in component_states if s.status == ServiceStatus.DOWN)
        warning_count = sum(1 for s in component_states if s.status == ServiceStatus.DEGRADED)

        # Calculate overall health score
        total_health = sum(s.health_score for s in component_states)
        avg_health = total_health / len(component_states) if component_states else 0

        # Determine overall status
        if failed_components > len(components_to_check) / 2:
            overall_status = ServiceStatus.DOWN
        elif warning_count > len(components_to_check) / 3:
            overall_status = ServiceStatus.DEGRADED
        else:
            overall_status = ServiceStatus.HEALTHY

        # Get task metrics
        task_metrics = self._get_task_metrics()

        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            overall_status=overall_status,
            health_score=avg_health,
            active_components=active_components,
            failed_components=failed_components,
            warning_count=warning_count,
            error_count=sum(s.error_count for s in component_states),
            performance_metrics={
                "avg_latency": sum(s.latency_ms for s in component_states) / len(component_states) if component_states else 0,
                "avg_success_rate": sum(s.success_rate for s in component_states) / len(component_states) if component_states else 0
            },
            resource_usage={
                "database_connections": self._get_db_connections(),
                "memory_usage": 0.0  # Would need system monitoring
            },
            active_sessions=0,
            pending_tasks=task_metrics['pending'],
            completed_tasks=task_metrics['completed']
        )

        # Store snapshot
        self._store_snapshot(snapshot, component_states)
        self.current_state = snapshot
        self.state_history.append(snapshot)

        return snapshot

    def _get_task_metrics(self) -> Dict[str, int]:
        """Get task execution metrics"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress
                FROM ai_development_tasks
            """)
            result = cur.fetchone()

            conn.close()

            return {
                'completed': result['completed'] or 0,
                'pending': result['pending'] or 0,
                'in_progress': result['in_progress'] or 0
            }

        except Exception:
            return {'completed': 0, 'pending': 0, 'in_progress': 0}

    def _get_db_connections(self) -> int:
        """Get current database connection count"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM pg_stat_activity")
            count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def _store_snapshot(self, snapshot: SystemSnapshot, component_states: List[ComponentState]):
        """Store system snapshot to database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Store main snapshot
            cur.execute("""
                INSERT INTO ai_system_state (
                    snapshot_id, overall_status, health_score,
                    active_components, failed_components, warning_count,
                    error_count, performance_metrics, resource_usage,
                    active_sessions, pending_tasks, completed_tasks,
                    snapshot_data
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                snapshot.snapshot_id,
                snapshot.overall_status.value,
                snapshot.health_score,
                snapshot.active_components,
                snapshot.failed_components,
                snapshot.warning_count,
                snapshot.error_count,
                json.dumps(snapshot.performance_metrics),
                json.dumps(snapshot.resource_usage),
                snapshot.active_sessions,
                snapshot.pending_tasks,
                snapshot.completed_tasks,
                json.dumps(asdict(snapshot), default=str)
            ))

            # Store component states
            for state in component_states:
                cur.execute("""
                    INSERT INTO ai_component_state (
                        component, status, last_check, uptime_seconds,
                        error_count, success_rate, latency_ms, health_score,
                        metadata, dependencies
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (component) DO UPDATE SET
                        status = EXCLUDED.status,
                        last_check = EXCLUDED.last_check,
                        uptime_seconds = EXCLUDED.uptime_seconds,
                        error_count = EXCLUDED.error_count,
                        success_rate = EXCLUDED.success_rate,
                        latency_ms = EXCLUDED.latency_ms,
                        health_score = EXCLUDED.health_score,
                        metadata = EXCLUDED.metadata,
                        dependencies = EXCLUDED.dependencies,
                        updated_at = NOW()
                """, (
                    state.component,
                    state.status.value,
                    state.last_check,
                    state.uptime_seconds,
                    state.error_count,
                    state.success_rate,
                    state.latency_ms,
                    state.health_score,
                    json.dumps(state.metadata),
                    state.dependencies
                ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    async def trigger_recovery(self, component: SystemComponent, error_type: str) -> bool:
        """Trigger automated recovery for a component"""
        if component not in self.recovery_procedures:
            return False

        procedures = self.recovery_procedures[component]
        for procedure in procedures:
            if procedure['error'] == error_type:
                return await self._execute_recovery_action(component, procedure)

        return False

    async def _execute_recovery_action(self, component: SystemComponent, procedure: Dict) -> bool:
        """Execute a recovery action"""
        start_time = datetime.now()
        success = False
        error_msg = None

        try:
            action = procedure['action']

            if action == "reconnect":
                success = await self._reconnect_database()
            elif action == "kill_and_restart":
                success = await self._restart_component(component)
            elif action == "garbage_collect":
                success = await self._cleanup_resources(component)
            elif action == "optimize_index":
                success = await self._optimize_indexes()
            elif action == "circuit_breaker":
                success = await self._activate_circuit_breaker(component)
            else:
                success = False
                error_msg = f"Unknown recovery action: {action}"

        except Exception as e:
            success = False
            error_msg = str(e)

        # Log recovery action
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        self._log_recovery_action(
            component.value,
            procedure['error'],
            procedure['action'],
            success,
            execution_time,
            error_msg
        )

        return success

    async def _reconnect_database(self) -> bool:
        """Attempt to reconnect to database"""
        try:
            conn = self._get_connection()
            conn.close()
            return True
        except Exception:
            return False

    async def _restart_component(self, component: SystemComponent) -> bool:
        """Restart a system component"""
        # This would need actual implementation based on deployment
        return True

    async def _cleanup_resources(self, component: SystemComponent) -> bool:
        """Clean up resources for a component"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Clean up old executions
            cur.execute("""
                UPDATE agent_executions
                SET status = 'failed'
                WHERE status = 'running'
                AND created_at < NOW() - INTERVAL '1 hour'
            """)

            conn.commit()
            conn.close()
            return True

        except Exception:
            return False

    async def _optimize_indexes(self) -> bool:
        """Optimize database indexes"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("ANALYZE")

            conn.commit()
            conn.close()
            return True

        except Exception:
            return False

    async def _activate_circuit_breaker(self, component: SystemComponent) -> bool:
        """Activate circuit breaker for a component"""
        # This would need actual implementation
        return True

    def _log_recovery_action(self, component: str, error_type: str, action: str,
                            success: bool, execution_time: float, error_msg: Optional[str]):
        """Log recovery action to database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_recovery_actions (
                    component, error_type, action_taken, success,
                    execution_time_ms, error_message
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (component, error_type, action, success, execution_time, error_msg))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error logging recovery action: {e}")

    def create_alert(self, alert_type: str, severity: str, component: Optional[str],
                    message: str, details: Dict = None):
        """Create a system alert"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_system_alerts (
                    alert_type, severity, component, message, details
                ) VALUES (%s, %s, %s, %s, %s)
            """, (alert_type, severity, component, message, json.dumps(details or {})))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error creating alert: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        if not self.current_state:
            return {
                "status": "unknown",
                "message": "No system state available"
            }

        return {
            "snapshot_id": self.current_state.snapshot_id,
            "timestamp": self.current_state.timestamp.isoformat(),
            "overall_status": self.current_state.overall_status.value,
            "health_score": self.current_state.health_score,
            "components": {
                name: {
                    "status": state.status.value,
                    "health_score": state.health_score,
                    "last_check": state.last_check.isoformat()
                }
                for name, state in self.components.items()
            },
            "metrics": {
                "active_components": self.current_state.active_components,
                "failed_components": self.current_state.failed_components,
                "warning_count": self.current_state.warning_count,
                "pending_tasks": self.current_state.pending_tasks,
                "completed_tasks": self.current_state.completed_tasks
            }
        }

# Singleton instance
_system_state_manager = None

def get_system_state_manager():
    """Get or create system state manager instance"""
    global _system_state_manager
    if _system_state_manager is None:
        _system_state_manager = SystemStateManager()
    return _system_state_manager

# Async helper functions
async def check_system_health():
    """Check overall system health"""
    manager = get_system_state_manager()
    return await manager.perform_full_system_check()

async def monitor_component(component: SystemComponent):
    """Monitor specific component health"""
    manager = get_system_state_manager()
    return await manager.check_component_health(component)

async def trigger_system_recovery(component: SystemComponent, error_type: str):
    """Trigger recovery for a component"""
    manager = get_system_state_manager()
    return await manager.trigger_recovery(component, error_type)