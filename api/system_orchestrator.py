"""
Autonomous System Orchestrator API Router
==========================================
API endpoints for centralized command and control of 1-10,000+ systems.
Dynamic resource allocation, deployments, and predictive maintenance.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["System Orchestrator"])

# Lazy initialization
_engine = None


def _get_engine():
    """Lazy load the System Orchestrator"""
    global _engine
    if _engine is None:
        try:
            from autonomous_system_orchestrator import AutonomousSystemOrchestrator
            _engine = AutonomousSystemOrchestrator()
        except Exception as e:
            logger.error(f"Failed to initialize System Orchestrator: {e}")
            raise HTTPException(status_code=503, detail="System Orchestrator not available")
    return _engine


class RegisterSystemRequest(BaseModel):
    system_name: str
    system_type: str  # SAAS_APP, MICROSERVICE, DATABASE, API_GATEWAY, etc.
    provider: str  # render, vercel, aws, gcp, azure, on_prem
    endpoint: str
    health_endpoint: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    auto_scaling: bool = True
    auto_remediation: bool = True


class DeploymentRequest(BaseModel):
    system_id: str
    version: str
    deployment_type: str = "rolling"  # rolling, blue_green, canary, immediate
    rollback_on_failure: bool = True
    pre_deploy_checks: Optional[List[str]] = None


class ResourceAllocationRequest(BaseModel):
    system_id: str
    resource_type: str  # cpu, memory, instances, storage
    amount: float
    reason: str


class BulkCommandRequest(BaseModel):
    system_ids: Optional[List[str]] = None  # None = all systems
    group: Optional[str] = None
    command: str  # health_check, restart, scale_up, scale_down, update, rollback
    parameters: Optional[Dict[str, Any]] = None


class MaintenanceWindowRequest(BaseModel):
    system_ids: List[str]
    start_time: str  # ISO format
    end_time: str
    maintenance_type: str
    description: str


@router.get("/status")
async def get_orchestrator_status():
    """Get Orchestrator system status"""
    engine = _get_engine()
    return {
        "system": "autonomous_system_orchestrator",
        "status": "operational",
        "initialized": engine._initialized if hasattr(engine, '_initialized') else True,
        "managed_systems": len(engine.managed_systems) if hasattr(engine, 'managed_systems') else 0,
        "capabilities": [
            "centralized_control",
            "dynamic_scaling",
            "auto_deployment",
            "health_monitoring",
            "predictive_maintenance",
            "bulk_operations",
            "resource_optimization"
        ],
        "scale_capacity": "1-10,000 systems"
    }


@router.post("/systems")
async def register_system(request: RegisterSystemRequest):
    """Register a new system for orchestration"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.register_system(
        system_name=request.system_name,
        system_type=request.system_type,
        provider=request.provider,
        endpoint=request.endpoint,
        health_endpoint=request.health_endpoint,
        metadata=request.metadata or {},
        auto_scaling=request.auto_scaling,
        auto_remediation=request.auto_remediation
    )
    return result


@router.get("/systems")
async def list_systems(
    provider: Optional[str] = None,
    system_type: Optional[str] = None,
    status: Optional[str] = None,
    group: Optional[str] = None
):
    """List all managed systems"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    systems = await engine.list_systems(
        provider=provider,
        system_type=system_type,
        status=status,
        group=group
    )
    return {"systems": systems, "total": len(systems)}


@router.get("/systems/{system_id}")
async def get_system(system_id: str):
    """Get details of a specific managed system"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    system = await engine.get_system(system_id)
    if not system:
        raise HTTPException(status_code=404, detail=f"System {system_id} not found")
    return system


@router.delete("/systems/{system_id}")
async def deregister_system(system_id: str):
    """Remove a system from orchestration"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.deregister_system(system_id)
    return result


@router.post("/systems/{system_id}/health-check")
async def check_system_health(system_id: str):
    """Run a health check on a specific system"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.check_health(system_id)
    return result


@router.post("/health-check/all")
async def check_all_health():
    """Run health checks on all managed systems"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    results = await engine.check_all_health()
    return {
        "checked_at": results.get("timestamp"),
        "systems_checked": len(results.get("results", [])),
        "healthy": sum(1 for r in results.get("results", []) if r.get("healthy")),
        "unhealthy": sum(1 for r in results.get("results", []) if not r.get("healthy")),
        "results": results.get("results", [])
    }


@router.post("/deploy")
async def trigger_deployment(request: DeploymentRequest):
    """Trigger a deployment for a system"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.deploy(
        system_id=request.system_id,
        version=request.version,
        deployment_type=request.deployment_type,
        rollback_on_failure=request.rollback_on_failure,
        pre_deploy_checks=request.pre_deploy_checks
    )
    return result


@router.get("/deployments")
async def list_deployments(
    system_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500)
):
    """List deployment history"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    deployments = await engine.get_deployments(
        system_id=system_id,
        status=status,
        limit=limit
    )
    return {"deployments": deployments}


@router.post("/resources/allocate")
async def allocate_resources(request: ResourceAllocationRequest):
    """Allocate resources to a system"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.allocate_resources(
        system_id=request.system_id,
        resource_type=request.resource_type,
        amount=request.amount,
        reason=request.reason
    )
    return result


@router.get("/resources")
async def get_resource_allocations(system_id: Optional[str] = None):
    """Get current resource allocations"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    allocations = await engine.get_resource_allocations(system_id=system_id)
    return {"allocations": allocations}


@router.post("/commands/bulk")
async def execute_bulk_command(request: BulkCommandRequest):
    """Execute a command on multiple systems"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.execute_bulk_command(
        system_ids=request.system_ids,
        group=request.group,
        command=request.command,
        parameters=request.parameters or {}
    )
    return result


@router.post("/maintenance")
async def schedule_maintenance(request: MaintenanceWindowRequest):
    """Schedule a maintenance window"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.schedule_maintenance(
        system_ids=request.system_ids,
        start_time=request.start_time,
        end_time=request.end_time,
        maintenance_type=request.maintenance_type,
        description=request.description
    )
    return result


@router.get("/maintenance")
async def list_maintenance_windows(
    system_id: Optional[str] = None,
    upcoming_only: bool = True
):
    """List maintenance windows"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    windows = await engine.get_maintenance_windows(
        system_id=system_id,
        upcoming_only=upcoming_only
    )
    return {"maintenance_windows": windows}


@router.get("/groups")
async def list_system_groups():
    """List all system groups"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    groups = await engine.get_groups() if hasattr(engine, 'get_groups') else []
    return {"groups": groups}


@router.post("/groups/{group_name}")
async def create_or_update_group(group_name: str, system_ids: List[str]):
    """Create or update a system group"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.set_group(group_name, system_ids)
    return result


@router.get("/dashboard")
async def get_orchestrator_dashboard():
    """Get a comprehensive orchestrator dashboard"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    dashboard = await engine.get_dashboard() if hasattr(engine, 'get_dashboard') else {}

    total_systems = len(engine.managed_systems) if hasattr(engine, 'managed_systems') else 0

    return {
        "overview": {
            "total_systems": total_systems,
            "healthy_systems": dashboard.get("healthy", 0),
            "warning_systems": dashboard.get("warning", 0),
            "critical_systems": dashboard.get("critical", 0),
            "active_deployments": dashboard.get("active_deployments", 0),
            "scheduled_maintenance": dashboard.get("scheduled_maintenance", 0)
        },
        "by_provider": dashboard.get("by_provider", {}),
        "by_type": dashboard.get("by_type", {}),
        "recent_events": dashboard.get("recent_events", [])[:20],
        "resource_utilization": dashboard.get("resource_utilization", {}),
        "scale_capacity": {
            "current": total_systems,
            "max_supported": 10000,
            "utilization_percent": (total_systems / 10000) * 100
        }
    }
