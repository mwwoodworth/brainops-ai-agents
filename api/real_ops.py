"""
Real Operations API - The POWER endpoints
==========================================
These endpoints do REAL things. No theater, no stubs.

Endpoints:
- POST /ops/health-check      - Check all 10 services, return real status
- POST /ops/ooda-cycle         - Run observe->decide->act cycle
- POST /ops/restart/{service}  - Restart a specific Render service
- POST /ops/alert              - Send an alert email
- POST /ops/db-maintenance     - Run VACUUM on bloated tables
- GET  /ops/briefing           - Get daily operational intelligence briefing
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ops", tags=["real-operations"])


def _get_async_pool():
    """Get the async connection pool (sync call, not a coroutine)."""
    from database.async_connection import get_pool

    return get_pool()


@router.post("/health-check")
async def run_health_check(
    service: Optional[str] = Query(None, description="Specific service name, or omit for all"),
) -> dict[str, Any]:
    """Check health of all 10 deployed services with real HTTP requests."""
    from real_action_engine import check_service_health

    return await check_service_health(service)


@router.post("/ooda-cycle")
async def run_ooda_cycle() -> dict[str, Any]:
    """
    Run a complete OODA cycle: observe real system state, decide on actions, execute them.
    This is the REAL OODA loop — it observes, decides, and ACTS.
    """
    from real_action_engine import check_service_health, decide_and_act

    # Phase 1: OBSERVE - check all services
    health = await check_service_health()

    # Convert health data to observations
    observations = []
    for svc_name, svc_data in health["services"].items():
        if not svc_data["healthy"]:
            observations.append(
                {
                    "type": "service_down",
                    "severity": "critical",
                    "data": {"service": svc_name, **svc_data},
                }
            )
        elif svc_data.get("response_ms", 0) > 5000:
            observations.append(
                {
                    "type": "slow_response",
                    "severity": "warning",
                    "data": {"service": svc_name, **svc_data},
                }
            )

    # Also observe DB health
    try:
        pool = _get_async_pool()
        async with pool.acquire() as conn:
            slow = await conn.fetchval(
                """
                SELECT count(*) FROM pg_stat_activity
                WHERE state = 'active'
                  AND query_start < now() - interval '30 seconds'
                  AND usename NOT IN ('supabase_admin', 'postgres')
            """
            )
            if slow and slow > 0:
                observations.append(
                    {
                        "type": "db_slow_query",
                        "severity": "warning",
                        "data": {"slow_query_count": slow},
                    }
                )

            bloated = await conn.fetch(
                """
                SELECT relname, n_dead_tup
                FROM pg_stat_user_tables
                WHERE schemaname = 'public' AND n_dead_tup > 10000
                ORDER BY n_dead_tup DESC LIMIT 3
            """
            )
            if bloated:
                for row in bloated:
                    observations.append(
                        {
                            "type": "db_bloat",
                            "severity": "info",
                            "data": {"table": row["relname"], "dead_tuples": row["n_dead_tup"]},
                        }
                    )
    except Exception as e:
        observations.append(
            {
                "type": "db_connection_failure",
                "severity": "critical",
                "data": {"error": str(e)},
            }
        )

    # Phase 2+3: DECIDE and ACT
    result = await decide_and_act(observations)
    result["health_summary"] = {
        "healthy": health["healthy"],
        "total": health["total"],
        "all_healthy": health["all_healthy"],
    }

    return result


@router.post("/restart/{service_name}")
async def restart_service_endpoint(service_name: str) -> dict[str, Any]:
    """Restart a specific Render service via API."""
    from real_action_engine import restart_service

    result = await restart_service(service_name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Restart failed"))
    return result


@router.post("/alert")
async def send_alert_endpoint(
    subject: str = Body(...),
    body: str = Body(""),
    severity: str = Body("info"),
) -> dict[str, Any]:
    """Send a real alert email via Resend."""
    from real_action_engine import send_alert

    return await send_alert(subject, body or f"<p>{subject}</p>", severity)


@router.post("/db-maintenance")
async def run_db_maintenance_endpoint(
    table: Optional[str] = Body(None),
) -> dict[str, Any]:
    """Run VACUUM ANALYZE on bloated tables."""
    from real_action_engine import run_db_maintenance

    return await run_db_maintenance(table)


@router.get("/briefing")
async def get_daily_briefing() -> dict[str, Any]:
    """
    Real operational intelligence briefing.
    Aggregates REAL data from all systems into a single picture of truth.
    """
    from real_action_engine import check_service_health

    briefing = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sections": {},
    }

    # Section 1: Service Health
    try:
        health = await check_service_health()
        briefing["sections"]["service_health"] = {
            "healthy": health["healthy"],
            "total": health["total"],
            "all_healthy": health["all_healthy"],
            "unhealthy": [name for name, data in health["services"].items() if not data["healthy"]],
        }
    except Exception as e:
        briefing["sections"]["service_health"] = {"error": str(e)}

    # Get pool once for all DB sections
    pool = None
    try:
        pool = _get_async_pool()
    except Exception as e:
        logger.error(f"Failed to get async pool for briefing: {e}")

    # Section 2: Revenue (REAL - from gumroad_sales and mrg_subscriptions)
    if pool:
        try:
            async with pool.acquire() as conn:
                gumroad = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total_sales,
                        coalesce(sum(price), 0) as total_revenue,
                        count(*) FILTER (WHERE sale_timestamp > now() - interval '30 days') as sales_30d,
                        coalesce(sum(price) FILTER (WHERE sale_timestamp > now() - interval '30 days'), 0) as revenue_30d
                    FROM gumroad_sales
                    WHERE is_test = false
                """
                )

                mrg = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total_subs,
                        count(*) FILTER (WHERE status = 'active') as active_subs,
                        count(*) FILTER (WHERE created_at > now() - interval '30 days') as new_subs_30d
                    FROM mrg_subscriptions
                """
                )

                leads = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total_leads,
                        count(*) FILTER (WHERE status = 'new' OR status = 'contacted') as active_leads,
                        count(*) FILTER (WHERE created_at > now() - interval '30 days') as new_leads_30d
                    FROM revenue_leads
                    WHERE is_test = false AND is_demo = false
                """
                )

                briefing["sections"]["revenue"] = {
                    "gumroad": {
                        "total_sales": gumroad["total_sales"] if gumroad else 0,
                        "total_revenue": float(gumroad["total_revenue"]) if gumroad else 0,
                        "sales_30d": gumroad["sales_30d"] if gumroad else 0,
                        "revenue_30d": float(gumroad["revenue_30d"]) if gumroad else 0,
                    },
                    "mrg_subscriptions": {
                        "total": mrg["total_subs"] if mrg else 0,
                        "active": mrg["active_subs"] if mrg else 0,
                        "new_30d": mrg["new_subs_30d"] if mrg else 0,
                    },
                    "leads": {
                        "total": leads["total_leads"] if leads else 0,
                        "active": leads["active_leads"] if leads else 0,
                        "new_30d": leads["new_leads_30d"] if leads else 0,
                    },
                }
        except Exception as e:
            briefing["sections"]["revenue"] = {"error": str(e)}
    else:
        briefing["sections"]["revenue"] = {"error": "Database pool unavailable"}

    # Section 3: Database Health
    if pool:
        try:
            async with pool.acquire() as conn:
                db_size = await conn.fetchval(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                table_count = await conn.fetchval(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
                )
                memory_count = await conn.fetchval("SELECT count(*) FROM unified_ai_memory")
                violations = await conn.fetchval(
                    "SELECT count(*) FROM invariant_violations WHERE resolved = false"
                )
                alerts = await conn.fetchval(
                    "SELECT count(*) FROM brainops_alerts WHERE resolved = false"
                )

                briefing["sections"]["database"] = {
                    "size": db_size,
                    "tables": table_count,
                    "memory_entries": memory_count,
                    "unresolved_violations": violations,
                    "unresolved_alerts": alerts,
                }
        except Exception as e:
            briefing["sections"]["database"] = {"error": str(e)}
    else:
        briefing["sections"]["database"] = {"error": "Database pool unavailable"}

    # Section 4: Agent Activity (last 24h)
    if pool:
        try:
            async with pool.acquire() as conn:
                executions = await conn.fetchrow(
                    """
                    SELECT
                        count(*) as total,
                        count(*) FILTER (WHERE status = 'completed') as completed,
                        count(*) FILTER (WHERE status = 'failed') as failed
                    FROM ai_agent_executions
                    WHERE created_at > now() - interval '24 hours'
                """
                )

                briefing["sections"]["agent_activity"] = {
                    "last_24h": {
                        "total": executions["total"] if executions else 0,
                        "completed": executions["completed"] if executions else 0,
                        "failed": executions["failed"] if executions else 0,
                    },
                }
        except Exception as e:
            briefing["sections"]["agent_activity"] = {"error": str(e)}
    else:
        briefing["sections"]["agent_activity"] = {"error": "Database pool unavailable"}

    # Section 5: Email Activity
    if pool:
        try:
            async with pool.acquire() as conn:
                emails = await conn.fetchrow(
                    """
                    SELECT
                        count(*) FILTER (WHERE status = 'sent') as sent_24h,
                        count(*) FILTER (WHERE status = 'queued') as queued,
                        count(*) FILTER (WHERE status = 'failed') as failed_24h
                    FROM ai_email_queue
                    WHERE created_at > now() - interval '24 hours'
                """
                )

                briefing["sections"]["email"] = {
                    "last_24h": {
                        "sent": emails["sent_24h"] if emails else 0,
                        "queued": emails["queued"] if emails else 0,
                        "failed": emails["failed_24h"] if emails else 0,
                    },
                }
        except Exception as e:
            briefing["sections"]["email"] = {"error": str(e)}
    else:
        briefing["sections"]["email"] = {"error": "Database pool unavailable"}

    return briefing
