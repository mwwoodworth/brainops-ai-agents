"""
Unified Brain API Endpoints
Single source of truth for ALL BrainOps memory
"""
import logging
from datetime import datetime
from typing import Any, Optional, Dict, List

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/brain", tags=["brain"])

# Import unified brain with fallback
try:
    from unified_brain import brain
    BRAIN_AVAILABLE = True
    logger.info("✅ Unified Brain loaded")
except ImportError as e:
    BRAIN_AVAILABLE = False
    brain = None
    logger.warning(f"⚠️ Unified Brain not available: {e}")


class BrainEntry(BaseModel):
    """Brain entry model"""
    key: str = Field(..., description="Unique key for this context")
    value: Any = Field(..., description="The actual data")
    category: str = Field("general", description="Category: system, session, architecture, deployment, issue")
    priority: str = Field("medium", description="Priority: critical, high, medium, low")
    source: str = Field("api", description="Source: claude_code, codex, api, manual, automated")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BrainQuery(BaseModel):
    """Brain search query"""
    query: str = Field(..., description="Search query")
    limit: int = Field(20, ge=1, le=100)


@router.get("/context", response_model=Dict[str, Any])
async def get_full_context():
    """
    Get COMPLETE system context for Claude Code session initialization
    This is THE endpoint that runs at session start
    """
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(
            status_code=503,
            detail="Unified Brain not available. System initializing..."
        )

    try:
        context = brain.get_full_context()
        return context
    except Exception as e:
        logger.error(f"Failed to get full context: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve context: {str(e)}"
        )


@router.get("/critical", response_model=List[Dict[str, Any]])
async def get_critical_context():
    """Get ALL critical context across all categories"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return brain.get_all_critical()
    except Exception as e:
        logger.error(f"Failed to get critical context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/category/{category}", response_model=List[Dict[str, Any]])
async def get_by_category(
    category: str,
    limit: int = Query(100, ge=1, le=500)
):
    """Get all context in a specific category"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return brain.get_by_category(category, limit)
    except Exception as e:
        logger.error(f"Failed to get category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{key}", response_model=Dict[str, Any])
async def get_context(key: str):
    """Retrieve a specific piece of context"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        result = brain.get(key)
        if not result:
            raise HTTPException(status_code=404, detail=f"Key not found: {key}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store", response_model=Dict[str, str])
async def store_context(entry: BrainEntry):
    """Store or update a piece of context"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        entry_id = brain.store(
            key=entry.key,
            value=entry.value,
            category=entry.category,
            priority=entry.priority,
            source=entry.source,
            metadata=entry.metadata
        )
        return {"id": entry_id, "key": entry.key, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=List[Dict[str, Any]])
async def search_context(query: BrainQuery):
    """Search across all context"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return brain.search(query.query, query.limit)
    except Exception as e:
        logger.error(f"Failed to search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", response_model=Dict[str, str])
async def record_session(
    session_id: str = Body(..., embed=True),
    summary: Dict[str, Any] = Body(...)
):
    """Record a Claude Code session summary"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        brain.record_session_summary(session_id, summary)
        return {"session_id": session_id, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployment", response_model=Dict[str, str])
async def record_deployment(
    service: str = Body(...),
    version: str = Body(...),
    status: str = Body(...),
    metadata: Optional[Dict] = Body(None)
):
    """Record a deployment"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        brain.record_deployment(service, version, status, metadata)
        return {"service": service, "version": version, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system-state", response_model=Dict[str, str])
async def update_system_state(
    component: str = Body(...),
    state: Dict[str, Any] = Body(...)
):
    """Update current system state"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        brain.update_system_state(component, state)
        return {"component": component, "status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def brain_health():
    """Check unified brain health"""
    return {
        "status": "operational" if BRAIN_AVAILABLE else "unavailable",
        "brain_available": BRAIN_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }
