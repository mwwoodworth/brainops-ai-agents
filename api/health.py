"""
Health, Readiness, and Status API Routes.

Extracted from app.py as part of Phase 2 architecture decomposition.
These endpoints serve container orchestration (Render), monitoring,
and diagnostic purposes.

Routes:
  GET  /health       — Full health check (auth-gated diagnostics)
  GET  /healthz      — Lightweight container probe (no DB)
  GET  /ready        — Dependency-aware readiness check
  GET  /alive        — Alive status with system vitals
  GET  /capabilities — Authenticated capability registry
  GET  /diagnostics  — Deep diagnostics endpoint
  GET  /system/alerts — System alerts query
"""
import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from config import config
from database.async_connection import get_pool, using_fallback
from services.db_health import attempt_db_pool_init_once, pool_roundtrip_healthy
from services.system_status import (
    collect_active_systems,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ---------------------------------------------------------------------------
# Auth helper (diagnostics-only, does NOT use the standard verify_api_key)
# ---------------------------------------------------------------------------


def _require_diagnostics_key(request: Request) -> None:
    api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("Authorization")
        or ""
    )
    if api_key.startswith("ApiKey "):
        api_key = api_key.split(" ", 1)[1]
    if api_key.startswith("Bearer "):
        api_key = api_key.split(" ", 1)[1]

    if not api_key or api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Forbidden")


# ---------------------------------------------------------------------------
# Constants (read from env at import time, same as original app.py)
# ---------------------------------------------------------------------------

HEALTH_CACHE_TTL_S = float(os.getenv("HEALTH_CACHE_TTL_S", "30"))
HEALTH_PAYLOAD_TIMEOUT_S = float(os.getenv("HEALTH_PAYLOAD_TIMEOUT_S", "5"))


# ---------------------------------------------------------------------------
# /health — main health endpoint
# ---------------------------------------------------------------------------


@router.get("/health")
async def health_check(
    request: Request,
    force_refresh: bool = Query(False, description="Bypass cache and force live health checks"),
):
    """Health check endpoint.

    Unauthenticated requests receive a minimal {"status": "ok", "version": "..."} response.
    Authenticated requests (valid X-API-Key) receive the full diagnostic payload.
    """
    # Lazy import to avoid circular import at module load
    import app as _app

    VERSION = _app.VERSION
    BUILD_TIME = _app.BUILD_TIME
    RESPONSE_CACHE = _app.RESPONSE_CACHE
    CACHE_TTLS = _app.CACHE_TTLS

    # --- Unauthenticated callers get a minimal response ---
    _api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("X-Api-Key")
    )
    _master = getattr(config.security, "master_api_key", None) or os.getenv("MASTER_API_KEY")
    _is_authenticated = False
    if _api_key and _api_key.strip() in config.security.valid_api_keys:
        _is_authenticated = True
    elif _master and _api_key and _api_key.strip() == _master:
        _is_authenticated = True

    if not _is_authenticated:
        return {"status": "ok", "version": VERSION}

    # --- Authenticated callers get full diagnostics ---

    async def _build_health_payload() -> dict[str, Any]:
        db_timeout = float(os.getenv("DB_HEALTH_TIMEOUT_S", "4.0"))
        pool_metrics: dict[str, Any] | None = None
        pool: Any | None = None
        try:
            pool = get_pool()
        except RuntimeError as e:
            if "not initialized" in str(e):
                lazy_ready = await attempt_db_pool_init_once(
                    _app.app.state, "health_lazy_init", timeout=max(2.0, db_timeout)
                )
                if not lazy_ready:
                    return {
                        "status": "starting",
                        "version": VERSION,
                        "build": BUILD_TIME,
                        "database": "initializing",
                        "database_error": getattr(_app.app.state, "db_init_error", None),
                        "message": "Service is starting up, database pool initializing...",
                    }
                pool = get_pool()
            else:
                raise
        db_healthy = await pool_roundtrip_healthy(pool, timeout=max(2.0, db_timeout))
        db_status = (
            "fallback" if using_fallback() else ("connected" if db_healthy else "disconnected")
        )
        auth_configured = config.security.auth_configured

        try:
            raw_pool = getattr(pool, "pool", None)
            if raw_pool is not None:
                pool_metrics = {
                    "min_size": getattr(raw_pool, "get_min_size", lambda: None)(),
                    "max_size": getattr(raw_pool, "get_max_size", lambda: None)(),
                    "size": getattr(raw_pool, "get_size", lambda: None)(),
                    "idle": getattr(raw_pool, "get_idle_size", lambda: None)(),
                }
        except Exception as exc:
            pool_metrics = {"error": str(exc)}

        active_systems = collect_active_systems(_app.app.state)

        # Embedded memory stats
        EMBEDDED_MEMORY_AVAILABLE = getattr(_app, "EMBEDDED_MEMORY_AVAILABLE", False)
        get_embedded_memory = getattr(_app, "get_embedded_memory", None)

        embedded_memory_stats = None
        embedded_memory_error = getattr(_app.app.state, "embedded_memory_error", None)
        if EMBEDDED_MEMORY_AVAILABLE and getattr(_app.app.state, "embedded_memory", None) is None:
            try:
                _app.app.state.embedded_memory = await asyncio.wait_for(
                    get_embedded_memory(), timeout=5.0
                )
                _app.app.state.embedded_memory_error = None
                embedded_memory_error = None
                logger.info("✅ Embedded Memory System lazily initialized from /health")
            except Exception as exc:
                embedded_memory_error = str(exc)
                _app.app.state.embedded_memory_error = embedded_memory_error
                logger.warning("Embedded memory lazy init failed during /health: %s", exc)

        if EMBEDDED_MEMORY_AVAILABLE and getattr(_app.app.state, "embedded_memory", None):
            try:
                embedded_memory_stats = _app.app.state.embedded_memory.get_stats()
            except Exception as exc:
                logger.warning("Failed to read embedded memory stats: %s", exc, exc_info=True)
                embedded_memory_stats = {"status": "error"}

        # Circuit breakers
        get_circuit_breaker_health = getattr(_app, "get_circuit_breaker_health", None)
        SERVICE_CIRCUIT_BREAKERS_AVAILABLE = getattr(
            _app, "SERVICE_CIRCUIT_BREAKERS_AVAILABLE", False
        )

        return {
            "status": "healthy" if db_healthy and auth_configured else "degraded",
            "version": VERSION,
            "build": BUILD_TIME,
            "database": db_status,
            "database_error": getattr(_app.app.state, "db_init_error", None),
            "db_pool": pool_metrics,
            "active_systems": active_systems,
            "system_count": len(active_systems),
            "embedded_memory_active": EMBEDDED_MEMORY_AVAILABLE
            and hasattr(_app.app.state, "embedded_memory")
            and _app.app.state.embedded_memory is not None,
            "embedded_memory_stats": embedded_memory_stats,
            "embedded_memory_error": embedded_memory_error,
            "capabilities": {
                "aurea_orchestrator": getattr(_app, "AUREA_AVAILABLE", False),
                "self_healing": getattr(_app, "SELF_HEALING_AVAILABLE", False),
                "memory_manager": getattr(_app, "MEMORY_AVAILABLE", False),
                "embedded_memory": EMBEDDED_MEMORY_AVAILABLE,
                "training_pipeline": getattr(_app, "TRAINING_AVAILABLE", False),
                "learning_system": getattr(_app, "LEARNING_AVAILABLE", False),
                "agent_scheduler": getattr(_app, "SCHEDULER_AVAILABLE", False),
                "ai_core": getattr(_app, "AI_AVAILABLE", False),
                "system_improvement": getattr(_app, "SYSTEM_IMPROVEMENT_AVAILABLE", False),
                "devops_optimization": getattr(_app, "DEVOPS_AGENT_AVAILABLE", False),
                "code_quality": getattr(_app, "CODE_QUALITY_AVAILABLE", False),
                "customer_success": getattr(_app, "CUSTOMER_SUCCESS_AVAILABLE", False),
                "competitive_intelligence": getattr(_app, "COMPETITIVE_INTEL_AVAILABLE", False),
                "vision_alignment": getattr(_app, "VISION_ALIGNMENT_AVAILABLE", False),
                "digital_twin": True,
                "market_intelligence": True,
                "system_orchestrator": True,
                "enhanced_self_healing": True,
                "reconciliation_loop": getattr(_app, "RECONCILER_AVAILABLE", False),
                "bleeding_edge_ooda": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "hallucination_prevention": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "live_memory_brain": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "dependability_framework": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "consciousness_emergence": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "enhanced_circuit_breaker": getattr(_app, "BLEEDING_EDGE_AVAILABLE", False),
                "ai_observability": getattr(_app, "AI_OBSERVABILITY_AVAILABLE", False),
                "cross_module_integration": getattr(_app, "AI_OBSERVABILITY_AVAILABLE", False),
                "unified_metrics": getattr(_app, "AI_OBSERVABILITY_AVAILABLE", False),
                "learning_feedback_loops": getattr(_app, "AI_OBSERVABILITY_AVAILABLE", False),
                "module_health_scoring": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "realtime_alerting": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "event_correlation": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "auto_recovery": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "websocket_streaming": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "enhanced_learning": getattr(_app, "AI_ENHANCEMENTS_AVAILABLE", False),
                "unified_awareness": getattr(_app, "UNIFIED_AWARENESS_AVAILABLE", False),
                "self_reporting": getattr(_app, "UNIFIED_AWARENESS_AVAILABLE", False),
                "service_circuit_breakers": SERVICE_CIRCUIT_BREAKERS_AVAILABLE,
            },
            "circuit_breakers": get_circuit_breaker_health()
            if SERVICE_CIRCUIT_BREAKERS_AVAILABLE and get_circuit_breaker_health
            else {"status": "unavailable"},
            "config": {
                "environment": config.environment,
                "security": {
                    "auth_required": config.security.auth_required,
                    "dev_mode": config.security.dev_mode,
                    "auth_configured": auth_configured,
                    "api_keys_configured": len(config.security.valid_api_keys),
                },
            },
            "missing_systems": getattr(_app.app.state, "missing_systems", []),
        }

    async def _build_health_payload_safe() -> dict[str, Any]:
        """Bound health work so transient stalls don't bubble up as 502s from Render."""
        try:
            return await asyncio.wait_for(_build_health_payload(), timeout=HEALTH_PAYLOAD_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("Health payload build timed out after %.2fs", HEALTH_PAYLOAD_TIMEOUT_S)
            return {
                "status": "degraded",
                "version": _app.VERSION,
                "build": _app.BUILD_TIME,
                "database": "timeout",
                "message": "Health payload timed out",
            }
        except Exception as exc:
            logger.error("Health payload build failed: %s", exc, exc_info=True)
            return {
                "status": "degraded",
                "version": _app.VERSION,
                "build": _app.BUILD_TIME,
                "database": "error",
                "message": "Health payload error",
            }

    if force_refresh:
        return await _build_health_payload_safe()

    payload, from_cache = await RESPONSE_CACHE.get_or_set(
        "health_status",
        CACHE_TTLS["health"],
        _build_health_payload_safe,
    )
    return {**payload, "cached": from_cache}


# ---------------------------------------------------------------------------
# /healthz — lightweight container probe
# ---------------------------------------------------------------------------


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    """Lightweight health endpoint for container checks (no DB calls)."""
    import app as _app

    return {
        "status": "ok",
        "version": _app.VERSION,
        "build": _app.BUILD_TIME,
    }


# ---------------------------------------------------------------------------
# /ready — dependency-aware readiness
# ---------------------------------------------------------------------------


@router.get("/ready")
async def readiness_check(request: Request):
    """Dependency-aware readiness check."""
    try:
        pool = get_pool()
        db_healthy = await pool.test_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database not ready: {exc}")

    if not db_healthy:
        raise HTTPException(status_code=503, detail="Database not ready")

    if config.security.auth_required and not config.security.auth_configured:
        raise HTTPException(status_code=503, detail="Auth not configured")

    return {
        "status": "ready",
        "database": "connected",
        "auth_configured": config.security.auth_configured,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# /alive — alive status with vitals
# ---------------------------------------------------------------------------


@router.get("/alive")
async def alive_status(request: Request):
    """Get the alive status of the AI OS based on core system health."""
    import app as _app

    active_systems = collect_active_systems(_app.app.state)
    db_ok = False
    try:
        pool = get_pool()
        row = await pool.fetchval("SELECT 1")
        db_ok = row == 1
    except Exception:
        pass

    uptime = time.time() - getattr(_app.app.state, "_start_time", time.time())
    is_alive = db_ok and len(active_systems) > 0

    status = {
        "alive": is_alive,
        "uptime_seconds": int(uptime),
        "database": db_ok,
        "active_systems": len(active_systems),
        "nerve_center": None,
        "operational_monitor": None,
    }

    nc_value = getattr(_app.app.state, "nerve_center", None)
    if nc_value:
        try:
            nc_status = nc_value.get_status()
            status["nerve_center"] = nc_status
            status["uptime_seconds"] = nc_status.get("uptime_seconds", uptime)
        except Exception:
            pass

    monitor_value = getattr(_app.app.state, "operational_monitor", None)
    if monitor_value:
        try:
            status["operational_monitor"] = monitor_value.get_status()
        except Exception:
            pass

    return status


# ---------------------------------------------------------------------------
# /capabilities — authenticated capability registry
# ---------------------------------------------------------------------------


@router.get("/capabilities")
async def capabilities(request: Request):
    """Authenticated capability registry."""
    _require_diagnostics_key(request)
    import app as _app

    routes = []
    for route in _app.app.routes:
        methods = sorted(getattr(route, "methods", []) or [])
        routes.append({"path": route.path, "methods": methods})

    return {
        "service": config.service_name,
        "version": _app.VERSION,
        "environment": config.environment,
        "active_systems": collect_active_systems(_app.app.state),
        "routes": routes,
        "ai_enabled": getattr(_app, "AI_AVAILABLE", False),
        "scheduler_enabled": getattr(_app, "SCHEDULER_AVAILABLE", False),
    }


# ---------------------------------------------------------------------------
# /diagnostics — deep diagnostics
# ---------------------------------------------------------------------------


@router.get("/diagnostics")
async def diagnostics(request: Request):
    """Authenticated deep diagnostics."""
    _require_diagnostics_key(request)
    import app as _app

    missing_env = [
        key
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_KEY",
        )
        if not os.getenv(key)
    ]

    db_status: dict[str, Any] = {"ready": False}
    last_error = None
    try:
        pool = get_pool()
        db_ready = await pool.test_connection()
        db_status["ready"] = bool(db_ready)
    except Exception as exc:
        db_status["error"] = str(exc)

    try:
        pool = get_pool()
        last_error_row = await pool.fetchrow(
            """
            SELECT error_type, error_message, severity, component, timestamp
            FROM ai_error_logs
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        if last_error_row:
            last_error = dict(last_error_row)
    except Exception as exc:
        last_error = {"error": str(exc)}

    nerve_center_status = None
    try:
        if hasattr(_app.app.state, "nerve_center") and _app.app.state.nerve_center:
            nerve_center_status = _app.app.state.nerve_center.get_status()
    except Exception as exc:
        nerve_center_status = {"error": str(exc)}

    operational_monitor_status = None
    try:
        if hasattr(_app.app.state, "operational_monitor") and _app.app.state.operational_monitor:
            operational_monitor_status = _app.app.state.operational_monitor.get_status()
    except Exception as exc:
        operational_monitor_status = {"error": str(exc)}

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "missing_env": missing_env,
        "fallback_mode": using_fallback(),
        "database": db_status,
        "active_systems": collect_active_systems(_app.app.state),
        "last_error": last_error,
        "nerve_center": nerve_center_status,
        "operational_monitor": operational_monitor_status,
    }


# ---------------------------------------------------------------------------
# /system/alerts — system alerts
# ---------------------------------------------------------------------------


@router.get("/system/alerts")
async def get_system_alerts(request: Request, limit: int = 50, unresolved_only: bool = True):
    """Get system alerts that need attention."""
    pool = get_pool()
    try:
        if unresolved_only:
            alerts = await pool.fetch(
                """
                SELECT alert_type, severity, message, details, created_at
                FROM brainops_alerts
                WHERE resolved = false
                ORDER BY
                    CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
                    created_at DESC
                LIMIT $1
            """,
                limit,
            )
        else:
            alerts = await pool.fetch(
                """
                SELECT alert_type, severity, message, details, created_at, resolved
                FROM brainops_alerts
                ORDER BY created_at DESC
                LIMIT $1
            """,
                limit,
            )

        return {"count": len(alerts), "alerts": [dict(a) for a in alerts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
