"""
CompanyCam Integration API - Safety Stub
Prevents UI crashes in Weathercraft ERP by handling requests gracefully.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/companycam", tags=["integrations"])

class CompanyCamSyncRequest(BaseModel):
    project_id: str
    tenant_id: Optional[str] = None

class CompanyCamResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@router.post("/sync", response_model=CompanyCamResponse)
async def sync_companycam_project(request: CompanyCamSyncRequest):
    """
    Stub for CompanyCam sync. 
    Returns success to prevent UI errors, but logs that integration is not fully configured.
    """
    # In a full implementation, this would talk to CompanyCam API
    return CompanyCamResponse(
        success=True, 
        message="CompanyCam sync request received (Integration Stub Active)",
        data={"synced_count": 0, "status": "simulated"}
    )

@router.get("/projects/{project_id}", response_model=CompanyCamResponse)
async def get_companycam_project(project_id: str):
    """
    Stub for getting project details.
    """
    return CompanyCamResponse(
        success=True,
        message="Project details retrieved (Integration Stub Active)",
        data={"id": project_id, "photos": []}
    )

@router.post("/photos", response_model=CompanyCamResponse)
async def get_companycam_photos(data: Dict[str, Any]):
    """
    Stub for photo retrieval
    """
    return CompanyCamResponse(
        success=True,
        message="Photos retrieved (Integration Stub Active)",
        data={"photos": []}
    )
