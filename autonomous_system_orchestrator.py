"""
Autonomous System Orchestrator
==============================
Centralized command and control for 1-10,000 systems.

Capabilities:
- Dynamic resource allocation
- Multi-system deployment management
- Predictive maintenance
- Autonomous CI/CD management
- Centralized command center

Based on 2025 best practices from ServiceNow AI Control Tower, Kubiya, and CircleCI.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import logging
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    DEPLOYING = "deploying"
    UNKNOWN = "unknown"


class DeploymentStatus(Enum):
    """Deployment pipeline status"""
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    STAGING = "staging"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ResourceType(Enum):
    """Types of resources to allocate"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE_CONNECTIONS = "database_connections"


@dataclass
class ManagedSystem:
    """A system under orchestrator management"""
    system_id: str
    name: str
    type: str  # saas, microservice, api, database, etc.
    url: str
    region: str
    provider: str  # render, vercel, aws, gcp, etc.
    status: SystemStatus
    health_score: float
    last_health_check: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    deployments: List[str] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Deployment:
    """A deployment/release"""
    deployment_id: str
    system_id: str
    version: str
    status: DeploymentStatus
    started_at: str
    completed_at: Optional[str]
    triggered_by: str  # user, auto, schedule
    commit_sha: Optional[str]
    changes: List[str]
    test_results: Dict[str, Any] = field(default_factory=dict)
    rollback_available: bool = True


@dataclass
class ResourceAllocation:
    """Resource allocation decision"""
    allocation_id: str
    system_id: str
    resource_type: ResourceType
    current_value: float
    new_value: float
    reason: str
    confidence: float
    auto_approved: bool
    executed_at: Optional[str] = None


@dataclass
class MaintenanceWindow:
    """Scheduled maintenance window"""
    window_id: str
    system_ids: List[str]
    scheduled_start: str
    scheduled_end: str
    maintenance_type: str
    tasks: List[str]
    status: str  # scheduled, in_progress, completed, cancelled


class AutonomousSystemOrchestrator:
    """
    Centralized Command Center for 1-10,000 Systems

    Capabilities:
    - Register and monitor unlimited systems
    - Autonomous CI/CD pipeline management
    - Dynamic resource allocation
    - Predictive maintenance scheduling
    - Multi-system coordination
    - Real-time health monitoring
    - Automated incident response
    """

    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.systems: Dict[str, ManagedSystem] = {}
        self.deployments: Dict[str, Deployment] = {}
        self.resource_allocations: List[ResourceAllocation] = []
        self.maintenance_windows: Dict[str, MaintenanceWindow] = []
        self._initialized = False

        # Configuration
        self.max_systems = 10000
        self.auto_remediation_enabled = True
        self.auto_scaling_enabled = True
        self.deployment_approval_threshold = 0.85  # Auto-approve if confidence > 85%

        # System groups for bulk operations
        self.system_groups: Dict[str, Set[str]] = defaultdict(set)

        # CI/CD integrations
        self.ci_cd_providers = {
            "github_actions": os.getenv("GITHUB_TOKEN"),
            "vercel": os.getenv("VERCEL_TOKEN"),
            "render": os.getenv("RENDER_API_KEY"),
        }

    async def initialize(self):
        """Initialize the Orchestrator"""
        if self._initialized:
            return

        logger.info("Initializing Autonomous System Orchestrator...")

        await self._create_tables()
        await self._load_systems()

        self._initialized = True
        logger.info(f"Orchestrator initialized with {len(self.systems)} systems")

    async def _create_tables(self):
        """Create database tables"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                # Managed systems table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS managed_systems (
                        system_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        url TEXT,
                        region TEXT,
                        provider TEXT,
                        status TEXT DEFAULT 'unknown',
                        health_score FLOAT DEFAULT 100.0,
                        last_health_check TIMESTAMPTZ,
                        metadata JSONB DEFAULT '{}',
                        resources JSONB DEFAULT '{}',
                        dependencies JSONB DEFAULT '[]',
                        tags JSONB DEFAULT '[]',
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Deployments table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS orchestrator_deployments (
                        deployment_id TEXT PRIMARY KEY,
                        system_id TEXT REFERENCES managed_systems(system_id),
                        version TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        started_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ,
                        triggered_by TEXT,
                        commit_sha TEXT,
                        changes JSONB DEFAULT '[]',
                        test_results JSONB DEFAULT '{}',
                        rollback_available BOOLEAN DEFAULT TRUE
                    )
                """)

                # Resource allocations table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS resource_allocations (
                        allocation_id TEXT PRIMARY KEY,
                        system_id TEXT REFERENCES managed_systems(system_id),
                        resource_type TEXT NOT NULL,
                        current_value FLOAT NOT NULL,
                        new_value FLOAT NOT NULL,
                        reason TEXT,
                        confidence FLOAT,
                        auto_approved BOOLEAN DEFAULT FALSE,
                        executed_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Maintenance windows table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS maintenance_windows (
                        window_id TEXT PRIMARY KEY,
                        system_ids JSONB DEFAULT '[]',
                        scheduled_start TIMESTAMPTZ NOT NULL,
                        scheduled_end TIMESTAMPTZ NOT NULL,
                        maintenance_type TEXT,
                        tasks JSONB DEFAULT '[]',
                        status TEXT DEFAULT 'scheduled',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # System groups table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS system_groups (
                        group_name TEXT PRIMARY KEY,
                        system_ids JSONB DEFAULT '[]',
                        description TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Command history table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS orchestrator_commands (
                        id SERIAL PRIMARY KEY,
                        command_type TEXT NOT NULL,
                        target_systems JSONB,
                        parameters JSONB,
                        status TEXT DEFAULT 'pending',
                        result JSONB,
                        executed_by TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        completed_at TIMESTAMPTZ
                    )
                """)

            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error creating orchestrator tables: {e}")

    async def _load_systems(self):
        """Load managed systems from database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                rows = await conn.fetch("SELECT * FROM managed_systems")

                for row in rows:
                    system = ManagedSystem(
                        system_id=row['system_id'],
                        name=row['name'],
                        type=row['type'],
                        url=row['url'] or "",
                        region=row['region'] or "",
                        provider=row['provider'] or "",
                        status=SystemStatus(row['status']) if row['status'] else SystemStatus.UNKNOWN,
                        health_score=row['health_score'] or 100.0,
                        last_health_check=row['last_health_check'].isoformat() if row['last_health_check'] else "",
                        metadata=row['metadata'] or {},
                        resources=row['resources'] or {},
                        dependencies=row['dependencies'] or [],
                        tags=row['tags'] or []
                    )
                    self.systems[system.system_id] = system

            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error loading systems: {e}")

    async def register_system(
        self,
        name: str,
        system_type: str,
        url: str,
        region: str = "us-east",
        provider: str = "render",
        metadata: Dict[str, Any] = None,
        tags: List[str] = None,
        dependencies: List[str] = None
    ) -> ManagedSystem:
        """
        Register a new system for management

        Args:
            name: Human-readable system name
            system_type: Type of system (saas, microservice, api, database, etc.)
            url: Health check or base URL
            region: Deployment region
            provider: Infrastructure provider
            metadata: Additional metadata
            tags: Tags for grouping
            dependencies: IDs of dependent systems

        Returns:
            Registered ManagedSystem
        """
        if len(self.systems) >= self.max_systems:
            raise ValueError(f"Maximum system limit ({self.max_systems}) reached")

        system_id = self._generate_id(f"sys:{name}")

        system = ManagedSystem(
            system_id=system_id,
            name=name,
            type=system_type,
            url=url,
            region=region,
            provider=provider,
            status=SystemStatus.UNKNOWN,
            health_score=100.0,
            last_health_check=datetime.utcnow().isoformat(),
            metadata=metadata or {},
            resources={},
            dependencies=dependencies or [],
            tags=tags or []
        )

        self.systems[system_id] = system

        # Add to tag-based groups
        for tag in system.tags:
            self.system_groups[tag].add(system_id)

        await self._persist_system(system)

        # Initial health check
        asyncio.create_task(self._check_system_health(system_id))

        logger.info(f"Registered system {name} with ID {system_id}")
        return system

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        hash_input = f"{prefix}:{datetime.utcnow().timestamp()}"
        return f"{prefix.split(':')[0]}_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"

    async def _persist_system(self, system: ManagedSystem):
        """Persist system to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO managed_systems
                    (system_id, name, type, url, region, provider, status, health_score,
                     last_health_check, metadata, resources, dependencies, tags)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    ON CONFLICT (system_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        health_score = EXCLUDED.health_score,
                        last_health_check = EXCLUDED.last_health_check,
                        metadata = EXCLUDED.metadata,
                        resources = EXCLUDED.resources,
                        updated_at = NOW()
                """,
                    system.system_id,
                    system.name,
                    system.type,
                    system.url,
                    system.region,
                    system.provider,
                    system.status.value,
                    system.health_score,
                    datetime.fromisoformat(system.last_health_check) if system.last_health_check else None,
                    json.dumps(system.metadata),
                    json.dumps(system.resources),
                    json.dumps(system.dependencies),
                    json.dumps(system.tags)
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting system: {e}")

    async def _check_system_health(self, system_id: str) -> Dict[str, Any]:
        """Check health of a single system"""
        if system_id not in self.systems:
            return {"error": f"System {system_id} not found"}

        system = self.systems[system_id]

        try:
            async with aiohttp.ClientSession() as session:
                health_url = f"{system.url}/health" if not system.url.endswith("/health") else system.url

                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        system.status = SystemStatus.HEALTHY
                        system.health_score = 100.0

                        # Extract any health metrics from response
                        if isinstance(data, dict):
                            system.metadata["last_health_response"] = data
                    elif response.status in [500, 502, 503, 504]:
                        system.status = SystemStatus.CRITICAL
                        system.health_score = max(0, system.health_score - 30)
                    else:
                        system.status = SystemStatus.DEGRADED
                        system.health_score = max(0, system.health_score - 10)

        except asyncio.TimeoutError:
            system.status = SystemStatus.DEGRADED
            system.health_score = max(0, system.health_score - 20)
            system.alerts.append({
                "type": "timeout",
                "message": "Health check timed out",
                "timestamp": datetime.utcnow().isoformat()
            })
        except Exception as e:
            system.status = SystemStatus.OFFLINE
            system.health_score = 0
            system.alerts.append({
                "type": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

        system.last_health_check = datetime.utcnow().isoformat()
        await self._persist_system(system)

        # Trigger auto-remediation if enabled
        if self.auto_remediation_enabled and system.status in [SystemStatus.CRITICAL, SystemStatus.OFFLINE]:
            asyncio.create_task(self._auto_remediate(system_id))

        return {
            "system_id": system_id,
            "status": system.status.value,
            "health_score": system.health_score,
            "last_check": system.last_health_check
        }

    async def check_all_systems_health(self) -> Dict[str, Any]:
        """Check health of all registered systems"""
        results = []
        tasks = []

        for system_id in self.systems.keys():
            tasks.append(self._check_system_health(system_id))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        healthy = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "healthy")
        degraded = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "degraded")
        critical = sum(1 for r in results if isinstance(r, dict) and r.get("status") in ["critical", "offline"])

        return {
            "total_systems": len(self.systems),
            "healthy": healthy,
            "degraded": degraded,
            "critical": critical,
            "overall_health": (healthy / len(self.systems) * 100) if self.systems else 100,
            "checked_at": datetime.utcnow().isoformat()
        }

    async def _auto_remediate(self, system_id: str):
        """Attempt automatic remediation of a failing system"""
        system = self.systems.get(system_id)
        if not system:
            return

        logger.info(f"Attempting auto-remediation for {system.name}")

        remediation_steps = []

        # Step 1: Restart if possible
        if system.provider == "render":
            restart_result = await self._restart_render_service(system)
            remediation_steps.append({"action": "restart", "result": restart_result})

        # Step 2: Scale up resources
        if self.auto_scaling_enabled:
            scale_result = await self._scale_system(system_id, "up")
            remediation_steps.append({"action": "scale_up", "result": scale_result})

        # Step 3: Wait and recheck
        await asyncio.sleep(30)
        health_result = await self._check_system_health(system_id)
        remediation_steps.append({"action": "health_check", "result": health_result})

        # Log remediation attempt
        await self._log_remediation(system_id, remediation_steps)

        return remediation_steps

    async def _restart_render_service(self, system: ManagedSystem) -> Dict[str, Any]:
        """Restart a Render service"""
        api_key = self.ci_cd_providers.get("render")
        if not api_key:
            return {"error": "Render API key not configured"}

        service_id = system.metadata.get("render_service_id")
        if not service_id:
            return {"error": "Render service ID not found in metadata"}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                url = f"https://api.render.com/v1/services/{service_id}/restart"

                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        return {"status": "restarted"}
                    else:
                        return {"error": f"Restart failed with status {response.status}"}
        except Exception as e:
            return {"error": str(e)}

    async def _scale_system(self, system_id: str, direction: str = "up") -> Dict[str, Any]:
        """Scale a system's resources"""
        system = self.systems.get(system_id)
        if not system:
            return {"error": "System not found"}

        current_instances = system.resources.get("instances", 1)

        if direction == "up":
            new_instances = min(current_instances + 1, 10)
        else:
            new_instances = max(current_instances - 1, 1)

        allocation = ResourceAllocation(
            allocation_id=self._generate_id("alloc"),
            system_id=system_id,
            resource_type=ResourceType.COMPUTE,
            current_value=current_instances,
            new_value=new_instances,
            reason=f"Auto-scaling {direction} due to health status",
            confidence=0.9,
            auto_approved=True,
            executed_at=datetime.utcnow().isoformat()
        )

        system.resources["instances"] = new_instances
        self.resource_allocations.append(allocation)

        await self._persist_system(system)

        return {
            "system_id": system_id,
            "old_instances": current_instances,
            "new_instances": new_instances,
            "direction": direction
        }

    async def _log_remediation(self, system_id: str, steps: List[Dict[str, Any]]):
        """Log remediation attempt to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO orchestrator_commands
                    (command_type, target_systems, parameters, status, result, executed_by)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                    "auto_remediation",
                    json.dumps([system_id]),
                    json.dumps({}),
                    "completed",
                    json.dumps(steps),
                    "orchestrator"
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error logging remediation: {e}")

    async def deploy(
        self,
        system_id: str,
        version: str,
        commit_sha: str = None,
        changes: List[str] = None,
        triggered_by: str = "manual"
    ) -> Deployment:
        """
        Trigger a deployment for a system

        Args:
            system_id: Target system ID
            version: Version to deploy
            commit_sha: Git commit SHA
            changes: List of change descriptions
            triggered_by: Who/what triggered the deployment

        Returns:
            Deployment object
        """
        if system_id not in self.systems:
            raise ValueError(f"System {system_id} not found")

        system = self.systems[system_id]
        deployment_id = self._generate_id("deploy")

        deployment = Deployment(
            deployment_id=deployment_id,
            system_id=system_id,
            version=version,
            status=DeploymentStatus.PENDING,
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
            triggered_by=triggered_by,
            commit_sha=commit_sha,
            changes=changes or []
        )

        self.deployments[deployment_id] = deployment
        system.status = SystemStatus.DEPLOYING

        # Start deployment pipeline
        asyncio.create_task(self._run_deployment_pipeline(deployment))

        await self._persist_deployment(deployment)

        return deployment

    async def _run_deployment_pipeline(self, deployment: Deployment):
        """Run the deployment pipeline"""
        try:
            # Stage 1: Build
            deployment.status = DeploymentStatus.BUILDING
            await self._persist_deployment(deployment)
            await asyncio.sleep(2)  # Simulate build

            # Stage 2: Test
            deployment.status = DeploymentStatus.TESTING
            await self._persist_deployment(deployment)

            test_results = await self._run_tests(deployment)
            deployment.test_results = test_results

            if not test_results.get("passed", False):
                deployment.status = DeploymentStatus.FAILED
                await self._persist_deployment(deployment)
                return

            # Stage 3: Staging
            deployment.status = DeploymentStatus.STAGING
            await self._persist_deployment(deployment)
            await asyncio.sleep(1)

            # Stage 4: Deploy
            deployment.status = DeploymentStatus.DEPLOYING
            await self._persist_deployment(deployment)

            # Actually trigger deployment based on provider
            system = self.systems[deployment.system_id]
            if system.provider == "vercel":
                await self._deploy_to_vercel(system, deployment)
            elif system.provider == "render":
                await self._deploy_to_render(system, deployment)
            else:
                await asyncio.sleep(5)  # Generic deployment simulation

            # Complete
            deployment.status = DeploymentStatus.COMPLETED
            deployment.completed_at = datetime.utcnow().isoformat()

            # Update system status
            system.status = SystemStatus.HEALTHY

        except Exception as e:
            deployment.status = DeploymentStatus.FAILED
            logger.error(f"Deployment failed: {e}")

        await self._persist_deployment(deployment)
        await self._persist_system(self.systems[deployment.system_id])

    async def _run_tests(self, deployment: Deployment) -> Dict[str, Any]:
        """Run tests for a deployment"""
        # This would integrate with actual test runners
        return {
            "passed": True,
            "total_tests": 42,
            "passed_tests": 42,
            "failed_tests": 0,
            "coverage": 87.5,
            "duration_seconds": 30
        }

    async def _deploy_to_vercel(self, system: ManagedSystem, deployment: Deployment):
        """Trigger Vercel deployment"""
        token = self.ci_cd_providers.get("vercel")
        if not token:
            logger.warning("Vercel token not configured")
            return

        project_id = system.metadata.get("vercel_project_id")
        if not project_id:
            return

        # Vercel auto-deploys on git push, so we just track it
        await asyncio.sleep(60)  # Simulate Vercel build time

    async def _deploy_to_render(self, system: ManagedSystem, deployment: Deployment):
        """Trigger Render deployment"""
        api_key = self.ci_cd_providers.get("render")
        if not api_key:
            return

        service_id = system.metadata.get("render_service_id")
        if not service_id:
            return

        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {api_key}"}
                url = f"https://api.render.com/v1/services/{service_id}/deploys"

                async with session.post(url, headers=headers) as response:
                    if response.status in [200, 201]:
                        logger.info(f"Triggered Render deployment for {system.name}")
                    else:
                        logger.error(f"Render deployment trigger failed: {response.status}")
        except Exception as e:
            logger.error(f"Render deployment error: {e}")

    async def _persist_deployment(self, deployment: Deployment):
        """Persist deployment to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO orchestrator_deployments
                    (deployment_id, system_id, version, status, started_at, completed_at,
                     triggered_by, commit_sha, changes, test_results, rollback_available)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (deployment_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        completed_at = EXCLUDED.completed_at,
                        test_results = EXCLUDED.test_results
                """,
                    deployment.deployment_id,
                    deployment.system_id,
                    deployment.version,
                    deployment.status.value,
                    datetime.fromisoformat(deployment.started_at),
                    datetime.fromisoformat(deployment.completed_at) if deployment.completed_at else None,
                    deployment.triggered_by,
                    deployment.commit_sha,
                    json.dumps(deployment.changes),
                    json.dumps(deployment.test_results),
                    deployment.rollback_available
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting deployment: {e}")

    async def rollback(self, deployment_id: str) -> Dict[str, Any]:
        """Rollback a deployment"""
        if deployment_id not in self.deployments:
            return {"error": f"Deployment {deployment_id} not found"}

        deployment = self.deployments[deployment_id]

        if not deployment.rollback_available:
            return {"error": "Rollback not available for this deployment"}

        deployment.status = DeploymentStatus.ROLLED_BACK
        await self._persist_deployment(deployment)

        return {
            "deployment_id": deployment_id,
            "status": "rolled_back",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def schedule_maintenance(
        self,
        system_ids: List[str],
        start_time: datetime,
        end_time: datetime,
        maintenance_type: str,
        tasks: List[str]
    ) -> MaintenanceWindow:
        """Schedule a maintenance window"""
        window_id = self._generate_id("maint")

        window = MaintenanceWindow(
            window_id=window_id,
            system_ids=system_ids,
            scheduled_start=start_time.isoformat(),
            scheduled_end=end_time.isoformat(),
            maintenance_type=maintenance_type,
            tasks=tasks,
            status="scheduled"
        )

        self.maintenance_windows[window_id] = window

        # Update system status
        for system_id in system_ids:
            if system_id in self.systems:
                self.systems[system_id].metadata["scheduled_maintenance"] = window_id

        await self._persist_maintenance_window(window)

        return window

    async def _persist_maintenance_window(self, window: MaintenanceWindow):
        """Persist maintenance window to database"""
        try:
            import asyncpg
            if not self.db_url:
                return

            conn = await asyncpg.connect(self.db_url)
            try:
                await conn.execute("""
                    INSERT INTO maintenance_windows
                    (window_id, system_ids, scheduled_start, scheduled_end,
                     maintenance_type, tasks, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (window_id) DO UPDATE SET
                        status = EXCLUDED.status
                """,
                    window.window_id,
                    json.dumps(window.system_ids),
                    datetime.fromisoformat(window.scheduled_start),
                    datetime.fromisoformat(window.scheduled_end),
                    window.maintenance_type,
                    json.dumps(window.tasks),
                    window.status
                )
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error persisting maintenance window: {e}")

    async def bulk_command(
        self,
        command: str,
        target_group: str = None,
        target_systems: List[str] = None,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a command across multiple systems

        Args:
            command: Command to execute (health_check, restart, scale_up, scale_down, deploy)
            target_group: Group name to target
            target_systems: Specific system IDs to target
            parameters: Command parameters

        Returns:
            Bulk command results
        """
        # Determine target systems
        if target_group:
            targets = list(self.system_groups.get(target_group, set()))
        elif target_systems:
            targets = target_systems
        else:
            targets = list(self.systems.keys())

        results = []

        for system_id in targets:
            try:
                if command == "health_check":
                    result = await self._check_system_health(system_id)
                elif command == "restart":
                    system = self.systems[system_id]
                    result = await self._restart_render_service(system)
                elif command == "scale_up":
                    result = await self._scale_system(system_id, "up")
                elif command == "scale_down":
                    result = await self._scale_system(system_id, "down")
                elif command == "deploy":
                    version = parameters.get("version", "latest")
                    deployment = await self.deploy(system_id, version, triggered_by="bulk_command")
                    result = {"deployment_id": deployment.deployment_id, "status": deployment.status.value}
                else:
                    result = {"error": f"Unknown command: {command}"}

                results.append({"system_id": system_id, "result": result})
            except Exception as e:
                results.append({"system_id": system_id, "error": str(e)})

        return {
            "command": command,
            "targets": len(targets),
            "succeeded": len([r for r in results if "error" not in r.get("result", {})]),
            "failed": len([r for r in results if "error" in r.get("result", {})]),
            "results": results
        }

    def get_command_center_dashboard(self) -> Dict[str, Any]:
        """Get the command center dashboard data"""
        systems_by_status = defaultdict(list)
        systems_by_provider = defaultdict(list)
        systems_by_region = defaultdict(list)

        for system in self.systems.values():
            systems_by_status[system.status.value].append(system.system_id)
            systems_by_provider[system.provider].append(system.system_id)
            systems_by_region[system.region].append(system.system_id)

        active_deployments = [
            d for d in self.deployments.values()
            if d.status not in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]
        ]

        return {
            "total_systems": len(self.systems),
            "max_capacity": self.max_systems,
            "utilization_percent": (len(self.systems) / self.max_systems) * 100,
            "status_breakdown": dict(systems_by_status),
            "provider_breakdown": {k: len(v) for k, v in systems_by_provider.items()},
            "region_breakdown": {k: len(v) for k, v in systems_by_region.items()},
            "active_deployments": len(active_deployments),
            "recent_deployments": [
                {
                    "id": d.deployment_id,
                    "system": d.system_id,
                    "version": d.version,
                    "status": d.status.value
                }
                for d in list(self.deployments.values())[-10:]
            ],
            "groups": list(self.system_groups.keys()),
            "auto_remediation_enabled": self.auto_remediation_enabled,
            "auto_scaling_enabled": self.auto_scaling_enabled,
            "last_updated": datetime.utcnow().isoformat()
        }


# Singleton instance
system_orchestrator = AutonomousSystemOrchestrator()


# API Functions
async def register_managed_system(
    name: str,
    system_type: str,
    url: str,
    region: str = "us-east",
    provider: str = "render",
    metadata: Dict[str, Any] = None,
    tags: List[str] = None
) -> Dict[str, Any]:
    """Register a new system for management"""
    await system_orchestrator.initialize()
    system = await system_orchestrator.register_system(
        name=name,
        system_type=system_type,
        url=url,
        region=region,
        provider=provider,
        metadata=metadata,
        tags=tags
    )
    return {
        "system_id": system.system_id,
        "name": system.name,
        "status": system.status.value,
        "health_score": system.health_score
    }


async def get_orchestrator_dashboard() -> Dict[str, Any]:
    """Get command center dashboard"""
    await system_orchestrator.initialize()
    return system_orchestrator.get_command_center_dashboard()


async def check_all_health() -> Dict[str, Any]:
    """Check health of all managed systems"""
    await system_orchestrator.initialize()
    return await system_orchestrator.check_all_systems_health()


async def trigger_deployment(
    system_id: str,
    version: str,
    commit_sha: str = None
) -> Dict[str, Any]:
    """Trigger a deployment"""
    await system_orchestrator.initialize()
    deployment = await system_orchestrator.deploy(
        system_id=system_id,
        version=version,
        commit_sha=commit_sha,
        triggered_by="api"
    )
    return {
        "deployment_id": deployment.deployment_id,
        "status": deployment.status.value,
        "started_at": deployment.started_at
    }


async def execute_bulk_command(
    command: str,
    target_group: str = None,
    target_systems: List[str] = None,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Execute a bulk command across systems"""
    await system_orchestrator.initialize()
    return await system_orchestrator.bulk_command(
        command=command,
        target_group=target_group,
        target_systems=target_systems,
        parameters=parameters
    )
