"""
System Health & Observability API Endpoints — WAVE 2A extraction from app.py.

Routes: /system/awareness, /awareness, /awareness/report, /awareness/pulse,
        /truth, /truth/quick, /alive/thoughts, /observability/metrics,
        /debug/database, /debug/aurea, /debug/scheduler,
        /api/v1/telemetry/events
"""
import json
import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from config import config
from database.async_connection import get_pool, using_fallback

logger = logging.getLogger(__name__)

router = APIRouter(tags=["observability"])

# Re-use centralized auth
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Depends(API_KEY_HEADER)) -> str:
    """Verify API key — mirrors app-level verify_api_key."""
    if not api_key or api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# /system/awareness — deep self-awareness scan
# ---------------------------------------------------------------------------


@router.get("/system/awareness", dependencies=[Depends(verify_api_key)])
async def system_awareness():
    """
    CRITICAL: Self-awareness endpoint that reports what's actually broken.
    This is the AI OS telling you its problems — listen to it!
    """
    import app as _app

    pool = get_pool()
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    healthy: list[str] = []

    try:
        # Check Gumroad revenue (truthful: exclude test rows)
        gumroad_real = await pool.fetchval(
            "SELECT COUNT(*) FROM gumroad_sales WHERE NOT COALESCE(is_test, FALSE)"
        )
        gumroad_test = await pool.fetchval(
            "SELECT COUNT(*) FROM gumroad_sales WHERE COALESCE(is_test, FALSE)"
        )
        if gumroad_real == 0:
            issues.append(
                {
                    "category": "REVENUE",
                    "problem": "Zero REAL Gumroad sales recorded",
                    "impact": "No personal revenue from digital products (test rows do not count as revenue)",
                    "fix": "Verify Gumroad webhook is receiving real purchases at /gumroad/webhook",
                }
            )
            if gumroad_test and gumroad_test > 0:
                warnings.append(
                    {
                        "category": "REVENUE",
                        "problem": "Gumroad test rows present",
                        "impact": "Webhook wiring may be OK, but there is still zero real revenue recorded",
                        "fix": "Ensure production Gumroad products are live and receiving real purchases",
                    }
                )
        else:
            healthy.append(f"Gumroad: {gumroad_real} real sales recorded")

        # Check MRG subscriptions
        mrg_active = await pool.fetchval(
            "SELECT COUNT(*) FROM mrg_subscriptions WHERE status = 'active'"
        )
        mrg_mrr = await pool.fetchval(
            "SELECT COALESCE(SUM(price_amount), 0) FROM mrg_subscriptions WHERE status = 'active'"
        )
        if mrg_active == 0:
            issues.append(
                {
                    "category": "REVENUE",
                    "problem": "Zero active MRG subscriptions",
                    "impact": "No recurring SaaS revenue",
                    "fix": "Verify Stripe webhook processing and subscription creation flow",
                }
            )
        else:
            healthy.append(f"MRG: {mrg_active} active subscriptions (MRR: ${float(mrg_mrr):.2f})")

        # Check alerts
        unresolved_alerts = await pool.fetchval(
            "SELECT COUNT(*) FROM brainops_alerts WHERE resolved_at IS NULL"
        )
        if unresolved_alerts and unresolved_alerts > 10:
            warnings.append(
                {
                    "category": "ALERTS",
                    "problem": f"{unresolved_alerts} unresolved alerts",
                    "impact": "Potential system issues being ignored",
                    "fix": "Review and resolve alerts at /system/alerts",
                }
            )
        elif unresolved_alerts:
            healthy.append(f"Alerts: {unresolved_alerts} unresolved (manageable)")
        else:
            healthy.append("Alerts: All resolved")

        # Check invariant violations
        unresolved_violations = await pool.fetchval(
            "SELECT COUNT(*) FROM invariant_violations WHERE resolved_at IS NULL"
        )
        if unresolved_violations and unresolved_violations > 0:
            issues.append(
                {
                    "category": "INTEGRITY",
                    "problem": f"{unresolved_violations} unresolved invariant violations",
                    "impact": "Database integrity rules being violated",
                    "fix": "Run invariant check: SELECT * FROM invariant_violations WHERE resolved_at IS NULL",
                }
            )
        else:
            healthy.append("Invariants: Zero unresolved violations")

        # Check database size
        db_size_mb = await pool.fetchval(
            "SELECT pg_database_size(current_database()) / (1024*1024)"
        )
        if db_size_mb and db_size_mb > 8000:
            warnings.append(
                {
                    "category": "DATABASE",
                    "problem": f"Database is {db_size_mb}MB",
                    "impact": "Approaching Supabase limits",
                    "fix": "Review large tables, archive old data",
                }
            )
        else:
            healthy.append(f"Database: {db_size_mb}MB")

        # Check agent execution health
        recent_failures = await pool.fetchval(
            """
            SELECT COUNT(*) FROM ai_agent_executions
            WHERE status = 'failed'
            AND created_at > NOW() - INTERVAL '24 hours'
        """
        )
        if recent_failures and recent_failures > 10:
            warnings.append(
                {
                    "category": "AGENTS",
                    "problem": f"{recent_failures} agent execution failures in last 24h",
                    "impact": "AI agents may be malfunctioning",
                    "fix": "Check agent logs and error patterns",
                }
            )
        else:
            healthy.append(f"Agent failures (24h): {recent_failures or 0}")

        # Check email queue
        stuck_emails = await pool.fetchval(
            """
            SELECT COUNT(*) FROM ai_email_queue
            WHERE status = 'queued'
            AND scheduled_for < NOW() - INTERVAL '1 hour'
        """
        )
        if stuck_emails and stuck_emails > 0:
            warnings.append(
                {
                    "category": "EMAIL",
                    "problem": f"{stuck_emails} emails stuck in queue (overdue > 1h)",
                    "impact": "Email delivery delayed",
                    "fix": "Check email daemon status and Resend API connectivity",
                }
            )
        else:
            healthy.append("Email queue: No stuck emails")

        # Determine overall status
        if issues:
            overall_status = "needs_attention"
        elif warnings:
            overall_status = "mostly_healthy"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "version": getattr(_app, "VERSION", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "warnings": warnings,
            "healthy": healthy,
            "summary": {
                "issues_count": len(issues),
                "warnings_count": len(warnings),
                "healthy_count": len(healthy),
            },
        }

    except Exception as e:
        logger.error(f"System awareness check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to run system awareness check",
        }


# ---------------------------------------------------------------------------
# /alive/thoughts — legacy stub
# ---------------------------------------------------------------------------


@router.get("/alive/thoughts", dependencies=[Depends(verify_api_key)])
async def get_recent_thoughts():
    """Legacy endpoint retained after thought-stream deprecation."""
    return {
        "thoughts": [],
        "message": "Thought stream is disabled; use /alive and /diagnostics for operational status.",
    }


# ---------------------------------------------------------------------------
# /awareness — unified AI awareness endpoints
# ---------------------------------------------------------------------------


@router.get("/awareness", dependencies=[Depends(verify_api_key)])
async def get_awareness_status():
    """Quick status check — AI OS reports its current state."""
    import app as _app

    if not getattr(_app, "UNIFIED_AWARENESS_AVAILABLE", False):
        return {"status": "unavailable", "message": "Unified awareness system not loaded"}

    try:
        from unified_awareness import check_status

        return {"status": check_status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/awareness/report", dependencies=[Depends(verify_api_key)])
async def get_full_awareness_report():
    """Full self-reporting status — AI OS tells you everything it knows about itself."""
    import app as _app

    if not getattr(_app, "UNIFIED_AWARENESS_AVAILABLE", False):
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        from unified_awareness import get_status_report

        return get_status_report()
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.get("/awareness/pulse", dependencies=[Depends(verify_api_key)])
async def get_system_pulse():
    """Real-time system pulse — the AI's heartbeat and vital signs."""
    import app as _app

    if not getattr(_app, "UNIFIED_AWARENESS_AVAILABLE", False):
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        from unified_awareness import get_unified_awareness

        awareness = get_unified_awareness()
        pulse = awareness.get_system_pulse()
        return pulse.to_dict()
    except Exception as e:
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# /truth — live system truth
# ---------------------------------------------------------------------------


@router.get("/truth", dependencies=[Depends(verify_api_key)])
async def get_truth():
    """
    THE TRUTH — Complete live system truth from database.
    No static docs, no outdated info — just live truth.
    """
    import app as _app

    if not getattr(_app, "TRUE_AWARENESS_AVAILABLE", False):
        return {"available": False, "message": "True self-awareness not loaded"}

    try:
        from true_self_awareness import get_system_truth

        truth = await get_system_truth()
        return truth
    except Exception as e:
        logger.error(f"Error getting truth: {e}")
        return {"available": False, "error": str(e)}


@router.get("/truth/quick", dependencies=[Depends(verify_api_key)])
async def get_truth_quick():
    """Quick human-readable system truth. Shows what's real vs demo, what's working vs broken."""
    import app as _app

    if not getattr(_app, "TRUE_AWARENESS_AVAILABLE", False):
        return {"available": False, "message": "True self-awareness not loaded"}

    try:
        from true_self_awareness import get_quick_status

        status = await get_quick_status()
        return {"status": status}
    except Exception as e:
        logger.error(f"Error getting quick truth: {e}")
        return {"available": False, "error": str(e)}


# ---------------------------------------------------------------------------
# /api/v1/telemetry/events — neural event ingestion
# ---------------------------------------------------------------------------


@router.post("/api/v1/telemetry/events")
async def receive_telemetry_events(request: Request, authenticated: bool = Depends(verify_api_key)):
    """
    Receive telemetry events from external systems (ERP, MRG, etc.).
    Connects the 'nervous system' — allowing external apps to send
    events to the AI brain for processing and awareness.
    """
    try:
        body = await request.json()
        events = body.get("events", [body])

        logger.info(f"Received {len(events)} telemetry events")

        stored_count = 0
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                for event in events:
                    await conn.execute(
                        """
                        INSERT INTO ai_nerve_signals (
                            source, event_type, payload, metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT DO NOTHING
                    """,
                        event.get("source", "unknown"),
                        event.get("type", "telemetry"),
                        json.dumps(event.get("data", {})),
                        json.dumps(event.get("metadata", {})),
                    )
                    stored_count += 1
        except Exception as db_err:
            logger.warning(f"Failed to store telemetry: {db_err}")

        return {
            "success": True,
            "received": len(events),
            "stored": stored_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Telemetry ingestion error: {e}")
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# /observability/metrics — lightweight monitoring
# ---------------------------------------------------------------------------


@router.get("/observability/metrics", dependencies=[Depends(verify_api_key)])
async def observability_metrics():
    """Lightweight monitoring endpoint for request, cache, DB, and orchestrator health."""
    import app as _app
    from services.system_status import aurea_status, scheduler_snapshot, self_healing_status

    pool = get_pool()
    db_probe_ms = None
    db_error = None
    start = time.perf_counter()
    try:
        await pool.fetchval("SELECT 1")
        db_probe_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        db_error = str(exc)

    return {
        "requests": _app.REQUEST_METRICS.snapshot(),
        "cache": _app.RESPONSE_CACHE.snapshot(),
        "database": {
            "using_fallback": using_fallback(),
            "probe_latency_ms": db_probe_ms,
            "error": db_error,
        },
        "scheduler": scheduler_snapshot(_app.app.state),
        "aurea": aurea_status(_app.app.state),
        "self_healing": self_healing_status(_app.app.state),
    }


# ---------------------------------------------------------------------------
# /debug/* — diagnostic endpoints
# ---------------------------------------------------------------------------


@router.get("/debug/database", dependencies=[Depends(verify_api_key)])
async def debug_database():
    """Diagnostic endpoint for database connection issues."""
    import psycopg2

    results: dict[str, Any] = {
        "async_pool": {"using_fallback": using_fallback(), "status": "unknown"},
        "sync_psycopg2": {"status": "unknown"},
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password_set": bool(config.database.password),
            "ssl": config.database.ssl,
            "ssl_verify": config.database.ssl_verify,
        },
    }

    try:
        pool = get_pool()
        start = time.perf_counter()
        result = await pool.fetchval("SELECT 1")
        latency = (time.perf_counter() - start) * 1000
        results["async_pool"]["status"] = "connected" if result == 1 else "query_failed"
        results["async_pool"]["latency_ms"] = latency
        results["async_pool"]["test_query"] = result
    except Exception as e:
        results["async_pool"]["status"] = "error"
        results["async_pool"]["error"] = str(e)

    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password,
            sslmode="require",
        )
        cur = conn.cursor()
        start = time.perf_counter()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        latency = (time.perf_counter() - start) * 1000
        cur.close()
        conn.close()
        results["sync_psycopg2"]["status"] = "connected"
        results["sync_psycopg2"]["latency_ms"] = latency
        results["sync_psycopg2"]["test_query"] = result[0] if result else None
    except Exception as e:
        results["sync_psycopg2"]["status"] = "error"
        results["sync_psycopg2"]["error"] = str(e)

    return results


@router.get("/debug/aurea", dependencies=[Depends(verify_api_key)])
async def debug_aurea():
    """Diagnostic endpoint for AUREA orchestrator status."""
    import app as _app

    aurea = getattr(_app.app.state, "aurea", None)
    if not aurea:
        return {"status": "not_initialized", "available": getattr(_app, "AUREA_AVAILABLE", False)}

    try:
        status = aurea.get_status()
        return {
            "status": "running" if status.get("running") else "stopped",
            "details": status,
            "available": True,
            "cycle_count": getattr(aurea, "cycle_count", 0),
            "autonomy_level": str(getattr(aurea, "autonomy_level", "unknown")),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "available": True}


@router.get("/debug/scheduler", dependencies=[Depends(verify_api_key)])
async def debug_scheduler():
    """Diagnostic endpoint for agent scheduler status."""
    import app as _app

    scheduler = getattr(_app.app.state, "scheduler", None)
    if not scheduler:
        return {
            "status": "not_initialized",
            "available": getattr(_app, "SCHEDULER_AVAILABLE", False),
        }

    try:
        jobs = scheduler.scheduler.get_jobs() if hasattr(scheduler, "scheduler") else []
        return {
            "status": "running" if scheduler.scheduler.running else "stopped",
            "total_jobs": len(jobs),
            "next_10_jobs": [
                {
                    "id": str(job.id),
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in sorted(jobs, key=lambda x: x.next_run_time or datetime.max)[:10]
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
