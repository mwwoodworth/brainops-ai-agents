"""
Unified Brain API Endpoints
Single source of truth for ALL BrainOps memory
"""
import logging
import os
import re
from datetime import datetime
import time
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
router = APIRouter(prefix="/brain", tags=["brain"], dependencies=[Depends(verify_api_key)])

# Import unified brain with fallback
try:
    from unified_brain import brain

    BRAIN_AVAILABLE = True
    logger.info("✅ Unified Brain loaded")
except ImportError as e:
    BRAIN_AVAILABLE = False
    brain = None
    logger.warning(f"⚠️ Unified Brain not available: {e}")


_SECRET_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"brainops_[A-Za-z0-9_]*key_[A-Za-z0-9_]*", re.IGNORECASE),
        "<REDACTED_BRAINOPS_API_KEY>",
    ),
    (re.compile(r"\brnd_[A-Za-z0-9]{10,}\b"), "rnd_<REDACTED>"),
    (re.compile(r"\b***REMOVED***_[A-Za-z0-9]{10,}\b"), "***REMOVED***_<REDACTED>"),
    (re.compile(r"\b***REMOVED***_[A-Za-z0-9]{10,}\b"), "***REMOVED***_<REDACTED>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), "<REDACTED_JWT>"),
    # Redact DB URL passwords (keep scheme/user/host/path)
    (
        re.compile(r"(postgres(?:ql)?://[^:/\\s]+:)([^@\\s]+)(@)", re.IGNORECASE),
        r"\\1<REDACTED>\\3",
    ),
]


def _redact_string(text: str) -> str:
    redacted = text
    for rx, replacement in _SECRET_REPLACEMENTS:
        redacted = rx.sub(replacement, redacted)
    return redacted


def redact_secrets(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    return value


def _sanitize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(entry)
    if "value" in sanitized:
        sanitized["value"] = redact_secrets(sanitized["value"])
    if "metadata" in sanitized and sanitized["metadata"] is not None:
        sanitized["metadata"] = redact_secrets(sanitized["metadata"])
    return sanitized


class BrainEntry(BaseModel):
    """Brain entry model with enhanced features"""

    key: str = Field(..., description="Unique key for this context")
    value: Any = Field(..., description="The actual data")
    category: str = Field(
        "general", description="Category: system, session, architecture, deployment, issue"
    )
    priority: str = Field("medium", description="Priority: critical, high, medium, low")
    source: str = Field("api", description="Source: claude_code, codex, api, manual, automated")
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    ttl_hours: Optional[int] = Field(None, description="Time-to-live in hours (for temporary data)")


class BrainQuery(BaseModel):
    """Brain search query with semantic options"""

    query: str = Field(..., description="Search query")
    limit: int = Field(20, ge=1, le=100)
    use_semantic: bool = Field(True, description="Use semantic vector search if available")


_BRAIN_STATUS_CACHE: tuple[dict[str, Any], float] | None = None
_BRAIN_STATUS_CACHE_TTL_S = float(os.getenv("BRAIN_STATUS_CACHE_TTL_S", "30"))


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
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        context = redact_secrets(await brain.get_full_context())
        return {"status": "ok", "context": context, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Failed to get full context: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "context": {},
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/ping")
async def brain_ping():
    """Ultra-fast brain availability check - no database queries."""
    return {
        "status": "ok" if BRAIN_AVAILABLE else "unavailable",
        "brain_loaded": BRAIN_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/status")
async def get_brain_status():
    """Lightweight health/status check for the unified brain."""
    import asyncio

    if not BRAIN_AVAILABLE or not brain:
        logger.warning("Unified Brain not available for /brain/status")
        return {
            "status": "unavailable",
            "message": "Unified Brain is initializing or not configured",
            "timestamp": datetime.utcnow().isoformat(),
        }

    global _BRAIN_STATUS_CACHE
    now = time.monotonic()
    if _BRAIN_STATUS_CACHE is not None:
        cached_payload, expires_at = _BRAIN_STATUS_CACHE
        if expires_at > now:
            return {**cached_payload, "cached": True}

    try:
        # Add timeout to prevent hanging
        await asyncio.wait_for(brain._ensure_pool(), timeout=5.0)
        from database.async_connection import get_pool

        pool = get_pool()
        stats = await asyncio.wait_for(
            pool.fetchrow(
                """
                SELECT
                    -- Avoid COUNT(*) full-table scans in hot-path health checks.
                    COALESCE(
                        (SELECT n_live_tup::bigint FROM pg_stat_user_tables WHERE relname = 'unified_brain' LIMIT 1),
                        0
                    ) as total_entries,
                    -- Prefer index-backed lookup when available.
                    (SELECT last_updated
                     FROM unified_brain
                     ORDER BY last_updated DESC NULLS LAST
                     LIMIT 1) as last_update
            """
            ),
            timeout=5.0,
        )

        payload = {
            "status": "ok",
            "total_entries": stats["total_entries"] if stats else 0,
            "last_update": stats["last_update"].isoformat()
            if stats and stats["last_update"]
            else None,
            "approximate_total": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
        _BRAIN_STATUS_CACHE = (payload, now + _BRAIN_STATUS_CACHE_TTL_S)
        return payload
    except asyncio.TimeoutError:
        logger.error("Brain status check timed out")
        return {
            "status": "timeout",
            "message": "Database query timed out after 5s",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Brain status check failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.get("/critical")
async def get_critical_context():
    """Get ALL critical context across all categories"""
    import asyncio

    if not BRAIN_AVAILABLE or not brain:
        logger.warning("Unified Brain not available for /brain/critical")
        return {
            "status": "unavailable",
            "message": "Unified Brain is initializing or not configured",
            "critical_items": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    try:
        # Add timeout to prevent 502 errors
        raw_items = await asyncio.wait_for(brain.get_all_critical(), timeout=10.0)
        critical_items = [_sanitize_entry(item) for item in raw_items]
        return {
            "status": "ok",
            "critical_items": critical_items,
            "count": len(critical_items),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except asyncio.TimeoutError:
        logger.error("Critical context query timed out")
        return {
            "status": "timeout",
            "message": "Database query timed out after 10s",
            "critical_items": [],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get critical context: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "critical_items": [],
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/category/{category}", response_model=list[dict[str, Any]])
async def get_by_category(category: str, limit: int = Query(100, ge=1, le=500)):
    """Get all context in a specific category"""
    import asyncio

    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        items = await asyncio.wait_for(brain.get_by_category(category, limit), timeout=10.0)
        return [_sanitize_entry(item) for item in items]
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Database query timed out")
    except Exception as e:
        logger.error(f"Failed to get category: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/get/{key}", response_model=dict[str, Any])
async def get_context(
    key: str, include_related: bool = Query(False, description="Include related entries")
):
    """Retrieve a specific piece of context with optional related entries"""
    import asyncio

    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        result = await asyncio.wait_for(
            brain.get(key, include_related=include_related), timeout=10.0
        )
        if not result:
            raise HTTPException(status_code=404, detail=f"Key not found: {key}")
        return _sanitize_entry(result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Database query timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/store", response_model=dict[str, str])
async def store_context(entry: BrainEntry):
    """Store or update a piece of context with enhanced features (embeddings, TTL, etc.)"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        safe_value = redact_secrets(entry.value)
        safe_metadata = redact_secrets(entry.metadata or {})
        if safe_value != entry.value or safe_metadata != (entry.metadata or {}):
            safe_metadata = {**(safe_metadata or {}), "secrets_redacted": True}

        entry_id = await brain.store(
            key=entry.key,
            value=safe_value,
            category=entry.category,
            priority=entry.priority,
            source=entry.source,
            metadata=safe_metadata,
            ttl_hours=entry.ttl_hours,
        )
        return {"id": entry_id, "key": entry.key, "status": "stored"}
    except Exception as e:
        logger.error(f"Failed to store context: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/search", response_model=list[dict[str, Any]])
async def search_context(query: BrainQuery):
    """Search across all context using semantic search, tags, and full-text"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return [
            _sanitize_entry(item)
            for item in await brain.search(
                query.query, query.limit, use_semantic=query.use_semantic
            )
        ]
    except Exception as e:
        logger.error(f"Failed to search: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class RecallQuery(BaseModel):
    """Query model for memory recall (vector similarity search on unified_ai_memory)"""

    query: str = Field(..., description="Natural language query for semantic recall")
    limit: int = Field(10, ge=1, le=50, description="Max results to return")
    memory_type: Optional[str] = Field(
        None, description="Filter by memory type (e.g. semantic, episodic, procedural)"
    )
    context: Optional[str] = Field(None, description="Context ID to filter results")


@router.post("/recall", response_model=dict[str, Any])
async def recall_memory(query: RecallQuery):
    """
    Recall memories via vector similarity search on unified_ai_memory.
    Uses embeddings for semantic matching — the RAG retrieval endpoint.
    """
    import asyncio

    try:
        from unified_memory_manager import MemoryType, UnifiedMemoryManager, get_memory_manager
    except ImportError:
        raise HTTPException(status_code=503, detail="Unified Memory Manager not available")

    try:
        memory = get_memory_manager()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Memory Manager not initialized: {e}")

    # Resolve optional memory type filter
    mem_type = None
    if query.memory_type:
        try:
            mem_type = MemoryType(query.memory_type.lower())
        except ValueError:
            pass  # Ignore invalid type, search all

    try:
        # Resolve tenant_id from config (recall requires it for RLS)
        tenant_id = memory.tenant_id or config.tenant.default_tenant_id
        if not tenant_id:
            raise HTTPException(status_code=500, detail="No tenant_id configured for memory recall")

        # UnifiedMemoryManager.recall() is synchronous (psycopg2) — run in thread.
        # For lightweight test doubles, call directly to avoid creating teardown-blocking executors.
        if isinstance(memory, UnifiedMemoryManager):
            results = await asyncio.to_thread(
                memory.recall,
                query.query,
                tenant_id=tenant_id,
                context=query.context,
                limit=query.limit,
                memory_type=mem_type,
            )
        else:
            results = memory.recall(
                query.query,
                tenant_id=tenant_id,
                context=query.context,
                limit=query.limit,
                memory_type=mem_type,
            )

        # Sanitize results
        sanitized = []
        for r in results or []:
            entry = dict(r) if hasattr(r, "keys") else r
            # Remove raw embedding vectors from response
            entry.pop("embedding", None)
            sanitized.append(redact_secrets(entry))

        return {
            "status": "ok",
            "query": query.query,
            "results": sanitized,
            "count": len(sanitized),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Memory recall failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/session", response_model=dict[str, str])
async def record_session(
    session_id: str = Body(..., embed=True), summary: dict[str, Any] = Body(...)
):
    """Record a Claude Code session summary"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.record_session_summary(session_id, summary)
        return {"session_id": session_id, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record session: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/deployment", response_model=dict[str, str])
async def record_deployment(
    service: str = Body(...),
    version: str = Body(...),
    status: str = Body(...),
    metadata: Optional[dict] = Body(None),
):
    """Record a deployment"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.record_deployment(service, version, status, metadata)
        return {"service": service, "version": version, "status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/system-state", response_model=dict[str, str])
async def update_system_state(component: str = Body(...), state: dict[str, Any] = Body(...)):
    """Update current system state"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        await brain.update_system_state(component, state)
        return {"component": component, "status": "updated"}
    except Exception as e:
        logger.error(f"Failed to update system state: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health", response_model=dict[str, Any])
async def brain_health():
    """Check unified brain health"""
    return {
        "status": "operational" if BRAIN_AVAILABLE else "unavailable",
        "brain_available": BRAIN_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
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
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/similar/{key}", response_model=list[dict[str, Any]])
async def find_similar(
    key: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar entries to return"),
):
    """Find entries similar to the given key using vector similarity"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return [_sanitize_entry(item) for item in await brain.find_similar(key, limit)]
    except Exception as e:
        logger.error(f"Failed to find similar entries: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/related/{key}", response_model=list[dict[str, Any]])
async def get_related(
    key: str,
    max_depth: int = Query(2, ge=1, le=5, description="Maximum depth of relationship traversal"),
):
    """Get related entries recursively up to max_depth"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        return [_sanitize_entry(item) for item in await brain.get_related_entries(key, max_depth)]
    except Exception as e:
        logger.error(f"Failed to get related entries: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cleanup-expired", response_model=dict[str, Any])
async def cleanup_expired():
    """Remove expired entries and return count of deleted items"""
    if not BRAIN_AVAILABLE or not brain:
        raise HTTPException(status_code=503, detail="Unified Brain not available")

    try:
        count = await brain.cleanup_expired()
        return {"status": "ok", "deleted_count": count, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"Failed to cleanup expired entries: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/add-reference", response_model=dict[str, str])
async def add_reference(
    from_key: str = Body(...),
    to_key: str = Body(...),
    reference_type: str = Body(
        "related", description="Type: related, superseded, depends_on, derived_from"
    ),
    strength: float = Body(1.0, ge=0.0, le=1.0, description="Strength of relationship (0-1)"),
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
            "reference_type": reference_type,
        }
    except Exception as e:
        logger.error(f"Failed to add reference: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/operational-truth", response_model=dict[str, Any])
async def get_operational_truth():
    """
    Get the TRUTH about operational status.
    Distinguishes REAL data from demo/test data.
    Critical for understanding actual business state.
    """
    from database.async_connection import get_pool, using_fallback

    if using_fallback():
        return {"status": "error", "message": "Database unavailable"}

    pool = get_pool()

    try:
        # Get real revenue data
        gumroad_real = (
            await pool.fetchval(
                "SELECT COALESCE(SUM(price), 0) FROM gumroad_sales WHERE is_test = false"
            )
            or 0
        )

        gumroad_test = (
            await pool.fetchval("SELECT COUNT(*) FROM gumroad_sales WHERE is_test = true") or 0
        )

        # MRG subscriptions with Stripe (real payments)
        mrg_real = await pool.fetchrow(
            """
            SELECT COUNT(*) as count, COALESCE(SUM(amount), 0) as mrr
            FROM mrg_subscriptions
            WHERE stripe_subscription_id IS NOT NULL AND status = 'active'
        """
        )

        mrg_demo = (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM mrg_subscriptions
            WHERE stripe_subscription_id IS NULL
        """
            )
            or 0
        )

        # Acquisition targets
        targets = await pool.fetchrow(
            """
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE status = 'contacted') as contacted,
                   COUNT(*) FILTER (WHERE status = 'converted') as converted
            FROM acquisition_targets
        """
        )

        # Revenue leads
        leads = await pool.fetchrow(
            """
            SELECT COUNT(*) as total,
                   COUNT(*) FILTER (WHERE stage = 'won') as won,
                   COALESCE(SUM(estimated_value) FILTER (WHERE stage = 'won'), 0) as won_value
            FROM revenue_leads
        """
        )

        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "truth": {
                "real_revenue": {
                    "gumroad_total": float(gumroad_real),
                    "mrg_mrr": float(mrg_real["mrr"] if mrg_real else 0),
                    "total_real_mrr": float(mrg_real["mrr"] if mrg_real else 0),
                    "has_paying_customers": float(mrg_real["mrr"] if mrg_real else 0) > 0,
                },
                "demo_data": {
                    "gumroad_test_sales": gumroad_test,
                    "mrg_demo_subscriptions": mrg_demo,
                    "warning": "Demo data exists - do not report as real revenue",
                },
                "acquisition": {
                    "targets_total": targets["total"] if targets else 0,
                    "targets_contacted": targets["contacted"] if targets else 0,
                    "targets_converted": targets["converted"] if targets else 0,
                },
                "pipeline": {
                    "leads_total": leads["total"] if leads else 0,
                    "leads_won": leads["won"] if leads else 0,
                    "won_value": float(leads["won_value"] if leads else 0),
                },
            },
            "action_required": float(mrg_real["mrr"] if mrg_real else 0) == 0,
            "priority_action": "Acquire first paying customer"
            if float(mrg_real["mrr"] if mrg_real else 0) == 0
            else "Scale revenue",
        }
    except Exception as e:
        logger.error(f"Failed to get operational truth: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
