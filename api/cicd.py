"""
CI/CD Management API Endpoints
==============================
RESTful API for autonomous CI/CD management.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from autonomous_cicd_management import (
    DeploymentPlatform,
    get_cicd_engine,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cicd", tags=["CI/CD Management"])


class RegisterServiceRequest(BaseModel):
    name: str
    platform: str  # render, vercel, docker_hub, etc.
    repository: str
    branch: str = "main"
    environment: str = "production"
    health_endpoint: str = "/health"
    deployment_url: Optional[str] = None
    platform_config: dict[str, Any] = {}
    dependencies: list[str] = []


class DeployServiceRequest(BaseModel):
    version: Optional[str] = None
    triggered_by: str = "manual"


@router.get("/services")
async def list_services() -> dict[str, Any]:
    """List all registered services"""
    engine = get_cicd_engine()
    await engine.initialize()

    return {
        "total": len(engine.services),
        "services": [
            {
                "service_id": s.service_id,
                "name": s.name,
                "platform": s.platform.value,
                "repository": s.repository,
                "current_version": s.current_version,
                "health_status": s.health_status,
                "last_deployed": s.last_deployed,
            }
            for s in engine.services.values()
        ],
    }


@router.post("/services")
async def register_service(request: RegisterServiceRequest) -> dict[str, Any]:
    """Register a new service for CI/CD management"""
    engine = get_cicd_engine()
    await engine.initialize()

    try:
        platform = DeploymentPlatform(request.platform)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid platform: {request.platform}. Valid: {[p.value for p in DeploymentPlatform]}"
        )

    service = await engine.register_service(
        name=request.name,
        platform=platform,
        repository=request.repository,
        branch=request.branch,
        environment=request.environment,
        health_endpoint=request.health_endpoint,
        deployment_url=request.deployment_url,
        platform_config=request.platform_config,
        dependencies=request.dependencies,
    )

    return {
        "status": "registered",
        "service_id": service.service_id,
        "name": service.name,
        "platform": service.platform.value,
    }


@router.get("/services/{service_id}")
async def get_service(service_id: str) -> dict[str, Any]:
    """Get details for a specific service"""
    engine = get_cicd_engine()
    await engine.initialize()

    service = engine.services.get(service_id)
    if not service:
        raise HTTPException(status_code=404, detail=f"Service not found: {service_id}")

    return {
        "service_id": service.service_id,
        "name": service.name,
        "platform": service.platform.value,
        "repository": service.repository,
        "branch": service.branch,
        "environment": service.environment,
        "current_version": service.current_version,
        "health_endpoint": service.health_endpoint,
        "deployment_url": service.deployment_url,
        "dependencies": service.dependencies,
        "health_status": service.health_status,
        "last_deployed": service.last_deployed,
    }


@router.post("/services/{service_id}/deploy")
async def deploy_service(
    service_id: str,
    request: DeployServiceRequest,
    background_tasks: BackgroundTasks
) -> dict[str, Any]:
    """Deploy a service"""
    engine = get_cicd_engine()
    await engine.initialize()

    if service_id not in engine.services:
        raise HTTPException(status_code=404, detail=f"Service not found: {service_id}")

    # Start deployment in background
    deployment = await engine.deploy_service(
        service_id=service_id,
        version=request.version,
        triggered_by=request.triggered_by,
    )

    if not deployment:
        raise HTTPException(status_code=500, detail="Failed to start deployment")

    return {
        "status": "started",
        "deployment_id": deployment.deployment_id,
        "service_id": service_id,
        "version": deployment.version,
        "deployment_status": deployment.status.value,
    }


@router.get("/deployments")
async def list_deployments(
    service_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(default=50, le=200),
) -> dict[str, Any]:
    """List recent deployments"""
    engine = get_cicd_engine()
    await engine.initialize()

    deployments = list(engine.deployments.values())

    if service_id:
        deployments = [d for d in deployments if d.service_id == service_id]

    if status:
        deployments = [d for d in deployments if d.status.value == status]

    # Sort by started_at descending
    deployments.sort(key=lambda d: d.started_at, reverse=True)
    deployments = deployments[:limit]

    return {
        "total": len(deployments),
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "service_id": d.service_id,
                "version": d.version,
                "status": d.status.value,
                "started_at": d.started_at,
                "completed_at": d.completed_at,
                "triggered_by": d.triggered_by,
                "duration_seconds": d.duration_seconds,
                "error": d.error,
            }
            for d in deployments
        ],
    }


@router.get("/deployments/{deployment_id}")
async def get_deployment(deployment_id: str) -> dict[str, Any]:
    """Get details for a specific deployment"""
    engine = get_cicd_engine()
    await engine.initialize()

    deployment = engine.deployments.get(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail=f"Deployment not found: {deployment_id}")

    return {
        "deployment_id": deployment.deployment_id,
        "service_id": deployment.service_id,
        "version": deployment.version,
        "status": deployment.status.value,
        "started_at": deployment.started_at,
        "completed_at": deployment.completed_at,
        "triggered_by": deployment.triggered_by,
        "commit_sha": deployment.commit_sha,
        "build_logs": deployment.build_logs,
        "test_results": deployment.test_results,
        "rollback_version": deployment.rollback_version,
        "duration_seconds": deployment.duration_seconds,
        "error": deployment.error,
    }


@router.post("/deploy-all")
async def deploy_all_services(
    triggered_by: str = "manual",
) -> dict[str, Any]:
    """Deploy all services (coordinated multi-service deployment)"""
    engine = get_cicd_engine()
    await engine.initialize()

    if not engine.services:
        raise HTTPException(status_code=400, detail="No services registered")

    deployments = await engine.deploy_all(triggered_by=triggered_by)

    return {
        "status": "completed",
        "total_deployments": len(deployments),
        "successful": len([d for d in deployments if d.status.value == "live"]),
        "failed": len([d for d in deployments if d.status.value == "failed"]),
        "deployments": [
            {
                "deployment_id": d.deployment_id,
                "service_id": d.service_id,
                "status": d.status.value,
            }
            for d in deployments
        ],
    }


@router.get("/health")
async def check_all_health() -> dict[str, Any]:
    """Check health of all registered services"""
    engine = get_cicd_engine()
    await engine.initialize()

    health_results = await engine.check_all_health()

    healthy = sum(1 for r in health_results.values() if r.get("status") == "healthy")
    unhealthy = sum(1 for r in health_results.values() if r.get("status") == "unhealthy")
    errors = sum(1 for r in health_results.values() if r.get("status") == "error")

    return {
        "summary": {
            "total": len(health_results),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "errors": errors,
        },
        "services": health_results,
    }


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get CI/CD metrics"""
    engine = get_cicd_engine()
    await engine.initialize()

    return await engine.get_deployment_metrics()


@router.post("/seed-brainops-services")
async def seed_brainops_services() -> dict[str, Any]:
    """Seed the BrainOps services for CI/CD management"""
    engine = get_cicd_engine()
    await engine.initialize()

    services_to_register = [
        {
            "name": "brainops-backend-prod",
            "platform": "render",
            "repository": "mwwoodworth/brainops-backend",
            "health_endpoint": "/api/v1/health",
            "deployment_url": "https://brainops-backend-prod.onrender.com",
            "platform_config": {"service_id": "srv-d1tfs4idbo4c73di6k00"},
        },
        {
            "name": "brainops-ai-agents",
            "platform": "render",
            "repository": "mwwoodworth/brainops-ai-agents",
            "health_endpoint": "/health",
            "deployment_url": "https://brainops-ai-agents.onrender.com",
            "platform_config": {"service_id": "srv-ai-agents"},
        },
        {
            "name": "brainops-mcp-bridge",
            "platform": "render",
            "repository": "mwwoodworth/brainops-mcp-bridge",
            "health_endpoint": "/health",
            "deployment_url": "https://brainops-mcp-bridge.onrender.com",
            "platform_config": {"service_id": "srv-mcp-bridge"},
        },
        {
            "name": "weathercraft-erp",
            "platform": "vercel",
            "repository": "mwwoodworth/weathercraft-erp",
            "health_endpoint": "/api/health",
            "deployment_url": "https://weathercraft-erp.vercel.app",
        },
        {
            "name": "myroofgenius",
            "platform": "vercel",
            "repository": "mwwoodworth/myroofgenius-app",
            "health_endpoint": "/api/health",
            "deployment_url": "https://myroofgenius.com",
        },
    ]

    registered = []
    for svc in services_to_register:
        try:
            platform = DeploymentPlatform(svc["platform"])
            service = await engine.register_service(
                name=svc["name"],
                platform=platform,
                repository=svc["repository"],
                health_endpoint=svc.get("health_endpoint", "/health"),
                deployment_url=svc.get("deployment_url"),
                platform_config=svc.get("platform_config", {}),
            )
            registered.append(service.service_id)
        except Exception as e:
            logger.warning(f"Failed to register {svc['name']}: {e}")

    return {
        "status": "seeded",
        "registered": len(registered),
        "service_ids": registered,
    }
