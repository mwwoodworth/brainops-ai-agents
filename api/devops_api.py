"""
DEVOPS AUTOMATION API
Endpoints for automated deployment, healing, and knowledge management.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    from fastapi import HTTPException
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/devops",
    tags=["devops-automation"],
    dependencies=[Depends(verify_api_key)]
)


class DeployRequest(BaseModel):
    """Request to trigger a deployment"""
    service: str
    version: Optional[str] = None
    triggered_by: str = "api"


class HealRequest(BaseModel):
    """Request to heal a service"""
    service: str


class LearningRequest(BaseModel):
    """Request to record a learning"""
    service: str
    incident_type: str
    resolution: str
    root_cause: Optional[str] = None


@router.get("/health")
async def get_all_service_health():
    """
    Get health status of all managed services.
    Quick overview of the entire infrastructure.
    """
    try:
        from devops_automation import get_all_service_health
        health = await get_all_service_health()

        # Calculate overall status
        unhealthy_count = sum(1 for h in health.values() if h.get("status") == "unhealthy")
        overall = "healthy" if unhealthy_count == 0 else "degraded" if unhealthy_count < 3 else "critical"

        return {
            "overall": overall,
            "services": health,
            "unhealthy_count": unhealthy_count,
            "total_services": len(health)
        }
    except Exception as e:
        logger.error(f"Failed to get service health: {e}")
        return {"overall": "unknown", "error": str(e)}


@router.post("/deploy")
async def trigger_deployment(request: DeployRequest):
    """
    Trigger a deployment for a service.
    Supports Render and Vercel services.
    """
    try:
        from devops_automation import trigger_service_deploy
        result = await trigger_service_deploy(
            service=request.service,
            triggered_by=request.triggered_by
        )
        return result
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return {"status": "failed", "error": str(e)}


@router.post("/heal")
async def auto_heal_service(request: HealRequest):
    """
    Attempt to auto-heal an unhealthy service.
    Will try redeploy and verify recovery.
    """
    try:
        from devops_automation import auto_heal
        result = await auto_heal(request.service)
        return result
    except Exception as e:
        logger.error(f"Auto-heal failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/deployments")
async def get_deployment_history(
    service: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get deployment history.
    Filter by service or get all deployments.
    """
    try:
        from devops_automation import get_devops_automation
        automation = get_devops_automation()
        history = await automation.get_deployment_history(service=service, limit=limit)
        return {
            "count": len(history),
            "deployments": history
        }
    except Exception as e:
        logger.error(f"Failed to get deployment history: {e}")
        return {"count": 0, "deployments": [], "error": str(e)}


@router.post("/learn")
async def record_learning(request: LearningRequest):
    """
    Record a learning from an incident.
    Knowledge is persisted permanently.
    """
    try:
        from devops_automation import get_devops_automation
        automation = get_devops_automation()
        success = await automation.learn_from_incident(
            service=request.service,
            incident_type=request.incident_type,
            resolution=request.resolution,
            root_cause=request.root_cause
        )
        return {"success": success, "message": "Learning recorded" if success else "Failed to record"}
    except Exception as e:
        logger.error(f"Failed to record learning: {e}")
        return {"success": False, "error": str(e)}


@router.get("/knowledge/{key}")
async def get_knowledge(key: str):
    """
    Retrieve knowledge by key.
    """
    try:
        from devops_automation import get_devops_automation
        automation = get_devops_automation()
        value = await automation.get_knowledge(key)
        if value is None:
            return {"found": False, "key": key}
        return {"found": True, "key": key, "value": value}
    except Exception as e:
        logger.error(f"Failed to get knowledge: {e}")
        return {"found": False, "error": str(e)}


@router.get("/stats")
async def get_devops_stats():
    """
    Get DevOps automation statistics.
    """
    try:
        from devops_automation import get_devops_automation
        automation = get_devops_automation()
        return automation.get_stats()
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"error": str(e)}


@router.get("/services")
async def list_managed_services():
    """
    List all managed services with their configuration.
    """
    try:
        from devops_automation import get_devops_automation
        automation = get_devops_automation()
        services = {}
        for name, config in automation.services.items():
            services[name] = {
                "type": config["type"].value,
                "health_url": config["health_url"],
                "critical": config.get("critical", False)
            }
        return {"services": services, "count": len(services)}
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        return {"services": {}, "error": str(e)}
