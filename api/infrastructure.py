
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
import os
import logging

router = APIRouter(prefix="/infrastructure", tags=["Infrastructure"])
logger = logging.getLogger(__name__)

RENDER_API_KEY = os.getenv("RENDER_API_KEY")
RENDER_API_URL = "https://api.render.com/v1"

class ScaleRequest(BaseModel):
    service_id: str
    num_instances: int

class ServiceActionRequest(BaseModel):
    service_id: str

@router.post("/scale")
async def scale_service(request: ScaleRequest):
    """
    Self-Provisioning: Scale a Render service (e.g. for heavy jobs).
    """
    if not RENDER_API_KEY:
        raise HTTPException(status_code=503, detail="RENDER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{RENDER_API_URL}/services/{request.service_id}/scale",
                headers=headers,
                json={"numInstances": request.num_instances}
            )
            
            if response.status_code != 200:
                logger.error(f"Render scale error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            return response.json()
    except Exception as e:
        logger.error(f"Infrastructure error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/restart")
async def restart_service(request: ServiceActionRequest):
    """
    Self-Healing: Restart a service via API.
    """
    if not RENDER_API_KEY:
        raise HTTPException(status_code=503, detail="RENDER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {RENDER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{RENDER_API_URL}/services/{request.service_id}/restart",
                headers=headers
            )
            
            if response.status_code != 202: # Render returns 202 Accepted for restart
                logger.error(f"Render restart error: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
                
            return {"status": "restart_initiated", "service_id": request.service_id}
    except Exception as e:
        logger.error(f"Infrastructure error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
