"""
Unified Brain API Endpoints
Single source of truth for ALL BrainOps memory
"""
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# API Key Security - use centralized config
from config import config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# All endpoints require API key authentication
router = APIRouter(
    prefix="/brain",
    tags=["brain"],
    dependencies=[Depends(verify_api_key)]
)

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
    """Brain entry model with enhanced features"""
    key: str = Field(..., description="Unique key for this context")
    value: Any = Field(..., description="The actual data")
    category: str = Field("general", description="Category: system, session, architecture, deployment, issue")
    priority: str = Field("medium", description="Priority: critical, high, medium, low")
    source: str = Field("api", description="Source: claude_code, codex, api, manual, automated")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    ttl_hours: Optional[int] = Field(None, description="Time-to-live in hours (for temporary data)")


class BrainQuery(BaseModel):
    """Brain search query with semantic options"""
    query: str = Field(..., description="Search query")
    limit: int = Field(20, ge=1, le=100)
    use_semantic: bool = Field(True, description="Use semantic vector search if available")


@router.get("/context")
async def get_full_context():
    """
    Get COMPLETE system context for Claude Code session initialization
    This is THE endpoint that runs at session start
    """
    if not BRAIN_AVAILABLE or not brain:
        logger.warning("Unified Brain not available for /brain/context")
        return {
            "status": "unavailable",
            "message": "Unified Brain is initializing or not configured",
            "context": {},
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        context = await brain.get_full_context()
        return {
            "status": "ok",
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get full context: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "context": {},
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/critical")
async def get_critical_context():
    """Get ALL critical context across all categories"""
    if not BRAIN_AVAILABLE or not brain:
        logger.warning("Unified Brain not available for /brain/critical")
        return {
            "status": "unavailable",
            "message": "Unified Brain is initializing or not configured",
            "critical_items": [],
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        critical_items = await brain.get_all_critical()
        return {
            "status": "ok",
            "critical_items": critical_items,
            "count": len(critical_items),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get critical context: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "critical_items": [],
            "timestamp": datetime.utcnow().isoformat()
        }


@router.get("/category/{category}", response_model=list[dict[str, Any]])
async def get_by_category(
    category: str,
    limit: int = Query(100, ge=1, le=500)
):
    """Get all context in a specific category"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return await brain.get_by_category(category, limit)
    except Exception as e:
        logger.error(f"Failed to get category: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get/{key}", response_model=dict[str, Any])
async def get_context(
    key: str,
    include_related: bool = Query(False, description="Include related entries")
):
    """Retrieve a specific piece of context with optional related entries"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        result = await brain.get(key, include_related=include_related)
        if not result:
            raise HTTPException(status_code=404, detail=f"Key not found: {key}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/store", response_model=dict[str, str])
async def store_context(entry: BrainEntry):
    """Store or update a piece of context with enhanced features (embeddings, TTL, etc.)"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        entry_id = await brain.store(
            key=entry.key,
            value=entry.value,
            category=entry.category,
            priority=entry.priority,
            source=entry.source,
            metadata=entry.metadata,
            ttl_hours=entry.ttl_hours
        )
        return {"id": entry_id, "key": entry.key, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=list[dict[str, Any]])
async def search_context(query: BrainQuery):
    """Search across all context using semantic search, tags, and full-text"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return await brain.search(query.query, query.limit, use_semantic=query.use_semantic)
    except Exception as e:
        logger.error(f"Failed to search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session", response_model=dict[str, str])
async def record_session(
    session_id: str = Body(..., embed=True),
    summary: dict[str, Any] = Body(...)
):
    """Record a Claude Code session summary"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.record_session_summary(session_id, summary)
        return {"session_id": session_id, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployment", response_model=dict[str, str])
async def record_deployment(
    service: str = Body(...),
    version: str = Body(...),
    status: str = Body(...),
    metadata: Optional[dict] = Body(None)
):
    """Record a deployment"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.record_deployment(service, version, status, metadata)
        return {"service": service, "version": version, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system-state", response_model=dict[str, str])
async def update_system_state(
    component: str = Body(...),
    state: dict[str, Any] = Body(...)
):
    """Update current system state"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.update_system_state(component, state)
        return {"component": component, "status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update system state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=dict[str, Any])
async def brain_health():
    """Check unified brain health"""
    return {
        "status": "operational" if BRAIN_AVAILABLE else "unavailable",
        "brain_available": BRAIN_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/statistics", response_model=dict[str, Any])
async def get_statistics():
    """Get comprehensive statistics about the brain"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        stats = await brain.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{key}", response_model=list[dict[str, Any]])
async def find_similar(
    key: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar entries to return")
):
    """Find entries similar to the given key using vector similarity"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        results = await brain.find_similar(key, limit)
        return results
    except Exception as e:
        logger.error(f"Failed to find similar entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/related/{key}", response_model=list[dict[str, Any]])
async def get_related(
    key: str,
    max_depth: int = Query(2, ge=1, le=5, description="Maximum depth of relationship traversal")
):
    """Get related entries recursively up to max_depth"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        results = await brain.get_related_entries(key, max_depth)
        return results
    except Exception as e:
        logger.error(f"Failed to get related entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup-expired", response_model=dict[str, Any])
async def cleanup_expired():
    """Remove expired entries and return count of deleted items"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        count = await brain.cleanup_expired()
        return {
            "status": "ok",
            "deleted_count": count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to cleanup expired entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-reference", response_model=dict[str, str])
async def add_reference(
    from_key: str = Body(...),
    to_key: str = Body(...),
    reference_type: str = Body("related", description="Type: related, superseded, depends_on, derived_from"),
    strength: float = Body(1.0, ge=0.0, le=1.0, description="Strength of relationship (0-1)")
):
    """Add a cross-reference between two entries"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain._add_reference(from_key, to_key, reference_type, strength)
        return {
            "status": "ok",
            "from_key": from_key,
            "to_key": to_key,
            "reference_type": reference_type
        }
    except Exception as e:
        logger.error(f"Failed to add reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))
