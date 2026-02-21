"""Platform router — Wave 2C/2D extraction from app.py.

Routes: /, /email/*, /api/v1/knowledge/*, /api/v1/erp/analyze, /systems/usage
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr

from config import config
from database.async_connection import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["platform"])


# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------


def _get_app():
    import app as _app

    return _app.app


def _ai_available():
    import app as _app

    return getattr(_app, "AI_AVAILABLE", False)


def _ai_core():
    import app as _app

    return getattr(_app, "ai_core", None)


def _scheduler_available():
    import app as _app

    return getattr(_app, "SCHEDULER_AVAILABLE", False)


def _brain_available():
    import app as _app

    return getattr(_app, "BRAIN_AVAILABLE", False)


def _memory_available():
    import app as _app

    return getattr(_app, "MEMORY_AVAILABLE", False)


def _learning_available():
    import app as _app

    return getattr(_app, "LEARNING_AVAILABLE", False)


def _customer_success_available():
    import app as _app

    return getattr(_app, "CUSTOMER_SUCCESS_AVAILABLE", False)


def _version():
    import app as _app

    return getattr(_app, "VERSION", "unknown")


def _build_time():
    import app as _app

    return getattr(_app, "BUILD_TIME", "unknown")


def _response_cache():
    import app as _app

    return getattr(_app, "RESPONSE_CACHE", None)


def _cache_ttls():
    import app as _app

    return getattr(_app, "CACHE_TTLS", {})


def _limiter():
    import app as _app

    return getattr(_app, "limiter", None)


async def _get_agent_usage(pool) -> dict[str, Any]:
    """Fetch recent agent usage, trying both legacy and new table names."""
    queries: list[tuple[str, str, str, str]] = [
        ("ai_agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
        ("agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
    ]
    errors: list[str] = []
    for agents_table, executions_table, join_cond, time_col in queries:
        try:
            rows = await pool.fetch(
                f"""
                SELECT
                    a.id::text AS id, a.name,
                    COALESCE(a.category, 'other') AS category,
                    COALESCE(a.enabled, true) AS enabled,
                    COUNT(e.id) AS executions_last_30d,
                    MAX({time_col}) AS last_execution,
                    AVG(e.execution_time_ms) FILTER (WHERE e.execution_time_ms IS NOT NULL) AS avg_duration_ms
                FROM {agents_table} a
                LEFT JOIN {executions_table} e ON {join_cond} AND {time_col} >= NOW() - INTERVAL '30 days'
                GROUP BY a.id, a.name, a.category, a.enabled
                ORDER BY executions_last_30d DESC, last_execution DESC NULLS LAST
                LIMIT 20
            """
            )
            usage = []
            for row in rows:
                data = row if isinstance(row, dict) else dict(row)
                usage.append(
                    {
                        "id": str(data.get("id")),
                        "name": data.get("name"),
                        "category": data.get("category"),
                        "enabled": bool(data.get("enabled", True)),
                        "executions_last_30d": int(data.get("executions_last_30d") or 0),
                        "last_execution": data.get("last_execution").isoformat()
                        if data.get("last_execution")
                        else None,
                        "avg_duration_ms": float(data.get("avg_duration_ms") or 0),
                    }
                )
            return {"agents": usage, "table": agents_table, "executions_table": executions_table}
        except Exception as exc:
            errors.append(f"{agents_table}/{executions_table}: {exc}")
            continue
    return {"agents": [], "warning": "No agent usage data available", "errors": errors[:2]}


async def _get_schedule_usage(pool) -> dict[str, Any]:
    """Fetch scheduler schedule rows with resiliency."""
    schedules: list[dict[str, Any]] = []
    try:
        rows = await pool.fetch(
            """
            SELECT
                s.id::text AS id, s.agent_id::text AS agent_id,
                s.enabled, s.frequency_minutes, s.created_at,
                COALESCE(a.name, s.agent_id::text) AS agent_name
            FROM public.agent_schedules s
            LEFT JOIN ai_agents a ON a.id = s.agent_id
            ORDER BY s.enabled DESC, s.created_at DESC NULLS LAST
            LIMIT 50
        """
        )
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            schedules.append(
                {
                    "id": data.get("id"),
                    "agent_id": data.get("agent_id"),
                    "agent_name": data.get("agent_name"),
                    "enabled": bool(data.get("enabled", True)),
                    "frequency_minutes": data.get("frequency_minutes"),
                    "created_at": data.get("created_at").isoformat()
                    if data.get("created_at")
                    else None,
                }
            )
        return {"schedules": schedules, "table": "public.agent_schedules"}
    except Exception as exc:
        logger.error("Failed to load schedule usage: %s", exc)
        return {"schedules": schedules, "error": str(exc)}


async def _memory_stats_snapshot(pool) -> dict[str, Any]:
    """Fast snapshot of memory/learning health."""
    try:
        existing_tables = await pool.fetch(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """
        )
        if not existing_tables:
            return {"status": "not_configured"}
        table_names = [t["table_name"] for t in existing_tables]
        preferred = next(
            (t for t in ("ai_persistent_memory", "memory_entries", "memories") if t in table_names),
            table_names[0],
        )
        stats = await pool.fetchrow(f"SELECT COUNT(*) AS total FROM {preferred}")
        return {
            "status": "operational",
            "table": preferred,
            "total_records": stats["total"] if stats else 0,
        }
    except Exception as exc:
        logger.error("Failed to fetch memory stats: %s", exc)
        return {"status": "error", "error": str(exc)}


def _collect_active_systems() -> list[str]:
    """Return a list of systems that are initialized and active."""
    import app as _app

    active = []
    _flags = [
        ("AUREA_AVAILABLE", "aurea", "AUREA Orchestrator"),
        ("SELF_HEALING_AVAILABLE", "healer", "Self-Healing Recovery"),
        ("MEMORY_AVAILABLE", "memory", "Memory Manager"),
        ("EMBEDDED_MEMORY_AVAILABLE", "embedded_memory", "Embedded Memory (RAG)"),
        ("TRAINING_AVAILABLE", "training", "Training Pipeline"),
        ("LEARNING_AVAILABLE", "learning", "Learning System"),
        ("SCHEDULER_AVAILABLE", "scheduler", "Agent Scheduler"),
        ("NERVE_CENTER_AVAILABLE", "nerve_center", "NerveCenter (Operational Coordinator)"),
        ("OPERATIONAL_MONITOR_AVAILABLE", "operational_monitor", "Operational Monitor"),
        ("SYSTEM_IMPROVEMENT_AVAILABLE", "system_improvement", "System Improvement Agent"),
        ("DEVOPS_AGENT_AVAILABLE", "devops_agent", "DevOps Optimization Agent"),
        ("CODE_QUALITY_AVAILABLE", "code_quality", "Code Quality Agent"),
        ("CUSTOMER_SUCCESS_AVAILABLE", "customer_success", "Customer Success Agent"),
        ("COMPETITIVE_INTEL_AVAILABLE", "competitive_intel", "Competitive Intelligence Agent"),
        ("VISION_ALIGNMENT_AVAILABLE", "vision_alignment", "Vision Alignment Agent"),
        ("RECONCILER_AVAILABLE", "reconciler", "Self-Healing Reconciler"),
    ]
    app_obj = _app.app
    for flag_name, state_attr, label in _flags:
        if getattr(_app, flag_name, False) and getattr(app_obj.state, state_attr, None):
            active.append(label)
    if getattr(_app, "AI_AVAILABLE", False) and getattr(_app, "ai_core", None):
        active.append("AI Core")
    if getattr(_app, "BLEEDING_EDGE_AVAILABLE", False):
        active.append(
            "Bleeding Edge AI (OODA, Hallucination, Memory, Dependability, Consciousness, Circuit Breaker)"
        )
    if getattr(_app, "AUTONOMOUS_RESOLVER_AVAILABLE", False):
        active.append("Autonomous Issue Resolver (Detects AND FIXES AI OS Issues)")
    if getattr(_app, "MEMORY_ENFORCEMENT_AVAILABLE", False):
        active.append("Memory Enforcement (RBA/WBA, Verification, Audit)")
    if getattr(_app, "MEMORY_HYGIENE_AVAILABLE", False):
        active.append("Memory Hygiene (Deduplication, Conflicts, Decay)")
    if getattr(_app, "WORKFLOWS_AVAILABLE", False):
        active.append("Advanced Workflow Engine (LangGraph, OODA, HITL, Checkpoints)")
    return active


def _scheduler_snapshot() -> dict[str, Any]:
    """Return scheduler status with safe defaults."""
    import app as _app

    app_obj = _app.app
    scheduler = getattr(app_obj.state, "scheduler", None)
    if not (getattr(_app, "SCHEDULER_AVAILABLE", False) and scheduler):
        return {"enabled": False, "message": "Scheduler not available"}
    apscheduler_jobs = scheduler.scheduler.get_jobs()
    return {
        "enabled": True,
        "running": scheduler.scheduler.running,
        "registered_jobs_count": len(scheduler.registered_jobs),
        "apscheduler_jobs_count": len(apscheduler_jobs),
        "next_jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in apscheduler_jobs[:5]
        ],
    }


def _aurea_status() -> dict[str, Any]:
    import app as _app

    app_obj = _app.app
    aurea = getattr(app_obj.state, "aurea", None)
    if not (getattr(_app, "AUREA_AVAILABLE", False) and aurea):
        return {"available": False, "running": False}
    try:
        return {**aurea.get_status(), "available": True}
    except Exception as exc:
        logger.error("Failed to read AUREA status: %s", exc)
        return {"available": True, "running": False, "error": str(exc)}


def _self_healing_status() -> dict[str, Any]:
    import app as _app

    app_obj = _app.app
    healer = getattr(app_obj.state, "healer", None)
    if not (getattr(_app, "SELF_HEALING_AVAILABLE", False) and healer):
        return {"available": False}
    try:
        circuit_breakers = getattr(healer, "circuit_breakers", None) or {}
        breaker_total = len(circuit_breakers) if isinstance(circuit_breakers, dict) else 0
        rules = getattr(healer, "healing_rules", None) or []
        active_rules = len(rules) if hasattr(rules, "__len__") else None
        return {
            "available": True,
            "report_available": hasattr(healer, "get_health_report"),
            "circuit_breakers_total": breaker_total,
            "active_healing_rules": active_rules,
        }
    except Exception as exc:
        logger.error("Failed to read self-healing status: %s", exc)
        return {"available": True, "error": str(exc)}


def _ai_generate(prompt, **kwargs):
    import app as _app

    return _app.ai_generate(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------


def _parse_csv_env(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _is_allowlisted_recipient(recipient: str | None) -> bool:
    if not recipient:
        return False
    lowered = recipient.strip().lower()
    allowlist_recipients = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST", ""))
    allowlist_domains = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", ""))
    if lowered in allowlist_recipients:
        return True
    if "@" not in lowered:
        return False
    domain = lowered.split("@", 1)[1]
    if domain in allowlist_domains:
        return True
    return any(domain.endswith(f".{allowed}") for allowed in allowlist_domains)


def _require_allowlist_in_live_mode() -> bool:
    return os.getenv("OUTBOUND_EMAIL_REQUIRE_ALLOWLIST_IN_LIVE", "true").strip().lower() == "true"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SendEmailPayload(BaseModel):
    recipient: EmailStr
    subject: str
    html: str
    metadata: dict[str, Any] = {}


class KnowledgeStoreRequest(BaseModel):
    """Request payload for storing knowledge/memory entries."""

    content: str
    memory_type: str = "knowledge"
    source_system: Optional[str] = None
    source_agent: Optional[str] = None
    created_by: Optional[str] = None
    importance: float = 0.5
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class KnowledgeQueryRequest(BaseModel):
    """Request payload for querying unified memory/knowledge."""

    query: str
    limit: int = 10
    memory_type: Optional[str] = None
    min_importance: float = 0.0


class ErpAnalyzeRequest(BaseModel):
    """Request payload for ERP job analysis."""

    tenant_id: Optional[str] = None
    job_ids: Optional[list[str]] = None
    limit: int = 20


# ---------------------------------------------------------------------------
# Email Endpoints
# ---------------------------------------------------------------------------


@router.post("/email/send")
async def send_email_endpoint(request: Request, payload: SendEmailPayload):
    """Send a one-off email (admin-only, API-key protected).

    Safety rail:
    - non-live modes always require allowlist
    - live mode requires allowlist unless OUTBOUND_EMAIL_REQUIRE_ALLOWLIST_IN_LIVE=false
    """
    mode = os.getenv("OUTBOUND_EMAIL_MODE", "disabled").strip().lower()
    require_allowlist = mode != "live" or _require_allowlist_in_live_mode()
    if require_allowlist and not _is_allowlisted_recipient(payload.recipient):
        raise HTTPException(
            status_code=403, detail="Recipient is not allowlisted for outbound email"
        )

    from email_sender import send_email

    # Avoid blocking the HTTP loop with synchronous HTTP (Resend) / SMTP operations.
    subject = payload.subject.strip()[:200] if payload.subject else "BrainOps AI"
    success, message = await asyncio.to_thread(
        send_email,
        str(payload.recipient),
        subject,
        payload.html,
        payload.metadata or {},
    )

    recipient_masked = (
        str(payload.recipient).split("@", 1)[0][:3]
        + "***@"
        + str(payload.recipient).split("@", 1)[1]
    )
    logger.info("One-off email send requested -> %s (success=%s)", recipient_masked, success)
    return {"success": success, "message": message}


@router.get("/email/status")
async def email_queue_status():
    """Get email queue status - shows pending, sent, failed counts."""
    try:
        from email_sender import get_queue_status

        return get_queue_status()
    except ImportError:
        return {"error": "email_sender module not available"}
    except Exception as e:
        return {"error": str(e)}


@router.post("/email/process")
async def process_email_queue_endpoint(
    request: Request,
    batch_size: int = Query(default=10, ge=1, le=50),
    dry_run: bool = Query(default=False),
):
    """
    Manually trigger email queue processing.
    - batch_size: Number of emails to process (1-50)
    - dry_run: If true, don't actually send, just report what would be sent
    """
    try:
        from email_sender import process_email_queue

        result = process_email_queue(batch_size=batch_size, dry_run=dry_run)
        return result
    except ImportError:
        return {"error": "email_sender module not available"}
    except Exception as e:
        logger.error(f"Email processing failed: {e}")
        return {"error": str(e)}


@router.post("/email/test")
async def test_email_sending(
    request: Request,
    recipient: str = Query(..., description="Email address to send test to"),
):
    """Send a test email to verify email configuration is working."""
    try:
        from email_sender import send_email

        success, message = send_email(
            recipient,
            "Test Email from BrainOps AI",
            "<h1>Test Email</h1><p>This is a test email from the BrainOps AI email system.</p><p>If you received this, email sending is working correctly.</p>",
            {},
        )
        return {"success": success, "message": message, "recipient": recipient}
    except ImportError:
        return {"error": "email_sender module not available", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# ---------------------------------------------------------------------------
# Knowledge Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/v1/knowledge/store-legacy")
async def store_knowledge(request: Request):
    """
    Legacy knowledge endpoint kept for backward compatibility.
    Use POST /api/v1/knowledge/store for the canonical memory-backed path.
    """
    try:
        body = await request.json()
        key = body.get("key", "")
        value = body.get("value", {})
        category = body.get("category", "external")

        if not key:
            return {"success": False, "error": "Key required"}

        # Store in brain context
        if _brain_available():
            try:
                from api.brain import brain as unified_brain

                if unified_brain:
                    await unified_brain.store(
                        key=key,
                        value=value,
                        category=category,
                        priority="medium",
                        source="legacy_api_v1_knowledge_store",
                    )
                else:
                    raise RuntimeError("Unified brain instance unavailable")
                return {"success": True, "key": key}
            except Exception as brain_err:
                logger.warning(f"Brain store failed: {brain_err}")

        return {"success": False, "error": "Brain not available"}
    except Exception as e:
        logger.error(f"Knowledge store error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/api/v1/knowledge/store")
async def api_v1_knowledge_store(payload: KnowledgeStoreRequest, request: Request):
    """
    Store a knowledge/memory entry in the unified memory system.
    """
    _app = _get_app()
    embedded_memory = getattr(_app.state, "embedded_memory", None)

    metadata: dict[str, Any] = dict(payload.metadata or {})
    if payload.source_system:
        metadata.setdefault("source_system", payload.source_system)
    if payload.created_by:
        metadata.setdefault("created_by", payload.created_by)
    if payload.tags:
        metadata.setdefault("tags", payload.tags)

    memory_id = str(uuid.uuid4())

    if embedded_memory:
        try:
            success = embedded_memory.store_memory(
                memory_id=memory_id,
                memory_type=payload.memory_type,
                source_agent=payload.source_agent or "system",
                content=payload.content,
                metadata=metadata,
                importance_score=payload.importance,
            )
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to store memory in embedded backend"
                )
        except Exception as exc:
            logger.error("Embedded memory store failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to store memory") from exc
    elif _memory_available() and hasattr(_app.state, "memory") and _app.state.memory:
        try:
            from unified_memory_manager import Memory, MemoryType

            mem = Memory(
                memory_type=MemoryType.SEMANTIC,
                content={"text": payload.content, "metadata": metadata},
                source_system=payload.source_system or "brainops-core",
                source_agent=payload.source_agent or "system",
                created_by=payload.created_by or "system",
                importance_score=payload.importance,
                tags=payload.tags or [],
                metadata=metadata,
            )
            memory_id = _app.state.memory.store(mem)
        except Exception as exc:
            logger.error("UnifiedMemoryManager store failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to store memory") from exc
    else:
        raise HTTPException(status_code=503, detail="No memory backend available")

    return {
        "success": True,
        "id": memory_id,
        "memory_type": payload.memory_type,
    }


@router.post("/api/v1/knowledge/query")
async def api_v1_knowledge_query(payload: KnowledgeQueryRequest, request: Request):
    """
    Query the unified memory / knowledge store.
    """
    _app = _get_app()
    embedded_memory = getattr(_app.state, "embedded_memory", None)
    results: list[dict[str, Any]] = []

    if embedded_memory:
        try:
            results = embedded_memory.search_memories(
                query=payload.query,
                limit=payload.limit,
                memory_type=payload.memory_type,
                min_importance=payload.min_importance,
            )
        except Exception as exc:
            logger.error("Embedded memory query failed: %s", exc)
            results = []

    if (
        (not results)
        and _memory_available()
        and hasattr(_app.state, "memory")
        and _app.state.memory
    ):
        try:
            memory_type_enum = None
            if payload.memory_type:
                from unified_memory_manager import MemoryType

                try:
                    memory_type_enum = MemoryType(payload.memory_type)
                except Exception as exc:
                    logger.debug("Invalid memory type %s: %s", payload.memory_type, exc)
                    memory_type_enum = None

            results = _app.state.memory.recall(
                query=payload.query,
                context=None,
                limit=payload.limit,
                memory_type=memory_type_enum,
            )
        except Exception as exc:
            logger.error("UnifiedMemoryManager recall failed: %s", exc)
            raise HTTPException(status_code=500, detail="Memory query failed") from exc

    normalized: list[dict[str, Any]] = []
    for item in results:
        data = dict(item)
        content = data.get("content")
        if isinstance(content, str):
            try:
                content_parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError, ValueError):
                content_parsed = content
        else:
            content_parsed = content

        normalized.append(
            {
                "id": str(data.get("id")),
                "memory_type": data.get("memory_type"),
                "source_agent": data.get("source_agent"),
                "source_system": data.get("source_system"),
                "importance_score": float(data.get("importance_score", 0.0))
                if data.get("importance_score") is not None
                else None,
                "tags": data.get("tags"),
                "metadata": data.get("metadata"),
                "content": content_parsed,
                "created_at": data.get("created_at"),
                "last_accessed": data.get("last_accessed"),
                "similarity_score": data.get("similarity_score"),
                "combined_score": data.get("combined_score"),
            }
        )

    return {
        "success": True,
        "query": payload.query,
        "results": normalized,
        "count": len(normalized),
    }


@router.get("/api/v1/knowledge/graph/stats")
async def api_v1_knowledge_graph_stats():
    """
    Get knowledge graph statistics - node counts, edge counts, extraction status.
    """
    _app = _get_app()
    try:
        pool = get_pool()

        nodes_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_nodes") or 0
        edges_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_edges") or 0
        graph_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_graph") or 0

        node_types = await pool.fetch(
            """
            SELECT node_type, COUNT(*) as count
            FROM ai_knowledge_nodes
            GROUP BY node_type
            ORDER BY count DESC
        """
        )

        recent_graph = await pool.fetchrow(
            """
            SELECT node_data, updated_at
            FROM ai_knowledge_graph
            WHERE node_type = 'graph_metadata'
            ORDER BY updated_at DESC
            LIMIT 1
        """
        )

        extraction_stats = {}
        last_extraction = None
        if recent_graph:
            node_data = recent_graph.get("node_data", {})
            if isinstance(node_data, str):
                node_data = json.loads(node_data)
            extraction_stats = node_data.get("extraction_stats", {})
            last_extraction = recent_graph.get("updated_at")

        extractor_stats = {}
        if hasattr(_app.state, "knowledge_extractor") and _app.state.knowledge_extractor:
            extractor_stats = _app.state.knowledge_extractor.extraction_stats

        return {
            "success": True,
            "total_nodes": nodes_count,
            "total_edges": edges_count,
            "graph_entries": graph_count,
            "node_types": [dict(row) for row in node_types],
            "last_extraction": last_extraction.isoformat() if last_extraction else None,
            "extraction_stats": extraction_stats or extractor_stats,
            "extractor_active": hasattr(_app.state, "knowledge_extractor")
            and _app.state.knowledge_extractor is not None,
        }
    except Exception as e:
        logger.error(f"Knowledge graph stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/v1/knowledge/graph/extract")
async def api_v1_knowledge_graph_extract(hours_back: int = 24):
    """
    Manually trigger knowledge graph extraction.
    """
    try:
        from knowledge_graph_extractor import get_knowledge_extractor

        extractor = get_knowledge_extractor()
        await extractor.initialize()

        result = await extractor.run_extraction(hours_back=hours_back)

        return {
            "success": result.get("success", False),
            "message": f"Extracted {result.get('nodes_stored', 0)} nodes and {result.get('edges_stored', 0)} edges",
            "details": result,
        }
    except Exception as e:
        logger.error(f"Knowledge graph extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# ERP Analysis
# ---------------------------------------------------------------------------


@router.post("/api/v1/erp/analyze")
async def api_v1_erp_analyze(payload: ErpAnalyzeRequest):
    """
    Analyze ERP jobs using centralized BrainOps Core.
    """
    pool = get_pool()

    try:
        filters = ["j.status = ANY($1::text[])"]
        params: list[Any] = [["in_progress", "scheduled"]]

        has_tenant_id = False
        try:
            has_tenant_id = await pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'jobs'
                      AND column_name = 'tenant_id'
                )
                """
            )
        except Exception as column_exc:
            logger.warning("Unable to inspect jobs.tenant_id column: %s", column_exc)

        if payload.tenant_id and has_tenant_id:
            filters.append(f"j.tenant_id = ${len(params) + 1}::uuid")
            params.append(payload.tenant_id)
        elif payload.tenant_id and not has_tenant_id:
            logger.warning(
                "Tenant filter requested but jobs.tenant_id column not found; returning unscoped jobs"
            )

        if payload.job_ids:
            filters.append(f"j.id = ANY(${len(params) + 1}::uuid[])")
            params.append(payload.job_ids)

        limit_param_index = len(params) + 1
        params.append(payload.limit or 20)

        query = f"""
            SELECT
                j.id,
                j.job_number,
                j.title,
                j.status,
                j.scheduled_start,
                j.scheduled_end,
                j.actual_start,
                j.actual_end,
                j.completion_percentage,
                j.estimated_revenue,
                j.created_at,
                c.name AS customer_name
            FROM jobs j
            LEFT JOIN customers c ON c.id = j.customer_id
            WHERE {' AND '.join(filters)}
            ORDER BY j.scheduled_start NULLS LAST, j.created_at DESC
            LIMIT ${limit_param_index}
        """

        rows = await pool.fetch(query, *params)

        def _to_naive(dt):
            if not dt:
                return None
            if hasattr(dt, "tzinfo"):
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                from datetime import date

                if isinstance(dt, date):
                    return datetime.combine(dt, datetime.min.time())
                return dt

        now = datetime.utcnow()
        jobs_intel: list[dict[str, Any]] = []

        for row in rows:
            data = row if isinstance(row, dict) else dict(row)

            planned_start = data.get("scheduled_start") or data.get("actual_start")
            planned_end = data.get("scheduled_end") or data.get("actual_end")

            days_in_progress = 0
            if planned_start:
                delta = now - _to_naive(planned_start)
                days_in_progress = max(0, delta.days)

            total_duration = 30
            if planned_start and planned_end:
                delta_total = _to_naive(planned_end) - _to_naive(planned_start)
                total_duration = max(1, delta_total.days)

            completion_pct = data.get("completion_percentage")
            if completion_pct is None:
                if total_duration:
                    completion_pct = min(100, round((days_in_progress / total_duration) * 100))
                else:
                    completion_pct = 0
            else:
                completion_pct = min(100, completion_pct)

            on_track = completion_pct <= 100 and days_in_progress <= total_duration

            risk_level: str = "low"
            risk_score: int = 20
            predicted_delay = 0

            if completion_pct > 100 or days_in_progress > total_duration:
                risk_level = "critical"
                risk_score = 90
                predicted_delay = max(0, days_in_progress - total_duration)
            elif completion_pct > 80:
                risk_level = "high"
                risk_score = 70
                predicted_delay = 3
            elif completion_pct > 60:
                risk_level = "medium"
                risk_score = 50
                predicted_delay = 1

            job_name = data.get("title") or data.get("job_number") or "Job"
            customer_name = data.get("customer_name") or "Unknown"

            ai_commentary: Optional[str] = None
            if _ai_available() and _ai_core():
                try:
                    summary_prompt = (
                        f"Job '{job_name}' for customer '{customer_name}' has status '{data.get('status')}', "
                        f"completion {completion_pct}% after {days_in_progress} days "
                        f"with planned duration {total_duration} days. "
                        f"Risk level is {risk_level} with score {risk_score}."
                    )
                    commentary = await _ai_generate(
                        f"Provide a concise, 2-3 sentence risk summary and recommended next action for this roofing job:\n\n{summary_prompt}",
                        model="gpt-4-turbo-preview",
                        temperature=0.3,
                        max_tokens=160,
                    )
                    ai_commentary = commentary
                except Exception as exc:
                    logger.warning("AI commentary failed for job %s: %s", data.get("id"), exc)

            base_change_prob = 25.0
            if completion_pct < 25:
                base_change_prob += 15.0
            if data.get("total_amount", 0) > 15000:
                base_change_prob += 10.0
            if risk_level in ("high", "critical"):
                base_change_prob += 15.0
            change_prob = min(base_change_prob, 85.0)

            job_value = float(data.get("total_amount", 10000) or 10000)
            estimated_impact = int(job_value * (0.05 + (change_prob / 100) * 0.15))

            jobs_intel.append(
                {
                    "job_id": str(data.get("id")),
                    "job_name": job_name,
                    "customer_name": customer_name,
                    "status": data.get("status"),
                    "ai_source": "brainops-core",
                    "delay_risk": {
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "delay_factors": [
                            (
                                f"Job {days_in_progress} days in progress vs {total_duration} days planned"
                                if planned_start and planned_end
                                else "Limited schedule data available"
                            ),
                            "Weather delays possible",
                            "Material delivery timing critical",
                            ("Behind schedule" if not on_track else "On schedule"),
                        ],
                        "mitigation_strategies": [
                            "Add 1-2 crew members to accelerate",
                            "Schedule overtime for critical path tasks",
                            "Pre-order materials to avoid delays",
                            "Daily progress check-ins with foreman",
                        ],
                        "predicted_delay_days": predicted_delay,
                    },
                    "progress_tracking": {
                        "completion_percentage": completion_pct,
                        "on_track": on_track,
                        "milestones_completed": completion_pct // 25,
                        "milestones_total": 4,
                        "ai_progress_assessment": (
                            f"Job progressing well - {completion_pct}% complete on schedule"
                            if on_track
                            else f"Job needs attention - {predicted_delay} days behind schedule"
                        ),
                    },
                    "resource_optimization": {
                        "current_crew_size": 4,
                        "optimal_crew_size": 6 if risk_level in ("high", "critical") else 4,
                        "resource_utilization": 85 if on_track else 110,
                        "recommendations": (
                            [
                                "Increase crew size by 2 workers",
                                "Reassign experienced technician from another job",
                                "Schedule weekend work if customer approves",
                                "Focus resources on critical path items",
                            ]
                            if risk_level in ("high", "critical")
                            else [
                                "Current crew size is optimal",
                                "Resource utilization healthy at 85%",
                                "Maintain current staffing levels",
                            ]
                        ),
                    },
                    "change_order_intelligence": {
                        "probability_of_change": change_prob,
                        "potential_change_areas": [
                            "Additional valley flashing may be needed",
                            "Customer may upgrade shingle quality",
                            "Possible deck repair if rot discovered",
                        ],
                        "estimated_impact": estimated_impact,
                        "ai_recommendations": [
                            "Pre-approve deck inspection with customer",
                            "Have upgrade options ready to present",
                            "Document any rot/damage immediately",
                        ],
                    },
                    "next_action": {
                        "action": (
                            "Schedule emergency crew meeting"
                            if risk_level == "critical"
                            else (
                                "Add crew members"
                                if risk_level == "high"
                                else "Continue monitoring"
                            )
                        ),
                        "priority": (
                            "urgent"
                            if risk_level == "critical"
                            else ("high" if risk_level == "high" else "medium")
                        ),
                        "reasoning": [
                            (
                                "Job at risk of delay - immediate intervention needed"
                                if risk_level in ("critical", "high")
                                else "Job progressing normally"
                            ),
                            f"Current completion: {completion_pct}%",
                            ("On schedule" if on_track else f"{predicted_delay} days behind"),
                            "Weather forecast favorable for next 7 days",
                            ai_commentary or "",
                        ],
                    },
                }
            )

        return {
            "success": True,
            "jobs": jobs_intel,
            "count": len(jobs_intel),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERP analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="ERP analysis failed") from exc


# ---------------------------------------------------------------------------
# Systems Usage
# ---------------------------------------------------------------------------


@router.get("/systems/usage")
async def systems_usage():
    """Report which AI systems are being used plus scheduler and memory effectiveness."""
    _app = _get_app()

    async def _load_usage() -> dict[str, Any]:
        pool = get_pool()
        agent_usage = await _get_agent_usage(pool)
        schedule_usage = await _get_schedule_usage(pool)
        memory_usage = await _memory_stats_snapshot(pool)

        customer_success_preview = None
        if _customer_success_available() and getattr(_app.state, "customer_success", None):
            try:
                customer_success_preview = (
                    await _app.state.customer_success.generate_onboarding_plan(
                        customer_id="sample-customer",
                        plan_type="value-check",
                    )
                )
            except Exception as exc:
                customer_success_preview = {"error": str(exc)}

        return {
            "active_systems": _collect_active_systems(),
            "agents": agent_usage,
            "schedules": {**schedule_usage, "scheduler_runtime": _scheduler_snapshot()},
            "memory": memory_usage,
            "learning": {
                "available": _learning_available()
                and getattr(_app.state, "learning", None) is not None,
                "notes": "Notebook LM+ initialized"
                if getattr(_app.state, "learning", None)
                else "Learning system not initialized",
            },
            "aurea": _aurea_status(),
            "self_healing": _self_healing_status(),
            "customer_success": {
                "available": _customer_success_available()
                and getattr(_app.state, "customer_success", None) is not None,
                "sample_plan": customer_success_preview,
            },
        }

    cache = _response_cache()
    ttls = _cache_ttls()
    if cache:
        usage, from_cache = await cache.get_or_set(
            "systems_usage",
            ttls.get("systems_usage", 60),
            _load_usage,
        )
        return {**usage, "cached": from_cache}

    usage = await _load_usage()
    return {**usage, "cached": False}
