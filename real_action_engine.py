"""
Real Action Engine - The POWER layer of BrainOps AI OS
=====================================================
This module takes REAL actions in response to REAL observations.
No theater. No stubs. Every function here does something in the real world.

Actions:
1. Service restart via Render API
2. Email alerts via Resend
3. Database maintenance (VACUUM, connection pool recovery)
4. Health verification (HTTP checks against all 10 services)
5. Memory storage (persist decisions and outcomes for learning)

Used by: OODA loop decide+act, invariant monitor escalation, scheduled health checks
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

RENDER_API_KEY = os.getenv("RENDER_API_KEY", "")
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
RESEND_FROM_EMAIL = os.getenv("RESEND_FROM_EMAIL", "BrainOps Alerts <matt@myroofgenius.com>")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "matthew@weathercraft.net")
BRAINOPS_API_KEY = os.getenv("BRAINOPS_API_KEY", "")

# Service registry: name -> (render_service_id, health_url)
RENDER_SERVICES = {
    "ai_agents": ("srv-d413iu75r7bs738btc10", "https://brainops-ai-agents.onrender.com/health"),
    "backend": ("srv-d1tfs4idbo4c73di6k00", "https://brainops-backend-prod.onrender.com/health"),
    "mcp_bridge": ("srv-d4rhvg63jp1c73918770", "https://brainops-mcp-bridge.onrender.com/health"),
}

VERCEL_SERVICES = {
    "erp": "https://weathercraft-erp.vercel.app",
    "mrg": "https://myroofgenius.com",
    "command_center": "https://brainops-command-center.vercel.app",
    "guardian": "https://weathercraft-guardian.vercel.app",
    "bss": "https://brainstack-studio.vercel.app",
    "vaulted_slabs": "https://vaulted-slabs-marketplace.vercel.app",
    "wc_share": "https://wc-docs.vercel.app",
}


# ─── Action: Service Health Check ────────────────────────────────────────────


async def check_service_health(service_name: Optional[str] = None) -> dict[str, Any]:
    """Check health of one or all services. Returns real HTTP status codes."""
    results = {}
    timeout = httpx.Timeout(15.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        services = {}
        if service_name and service_name in RENDER_SERVICES:
            sid, url = RENDER_SERVICES[service_name]
            services[service_name] = url
        elif service_name and service_name in VERCEL_SERVICES:
            services[service_name] = VERCEL_SERVICES[service_name]
        else:
            for name, (_, url) in RENDER_SERVICES.items():
                services[name] = url
            for name, url in VERCEL_SERVICES.items():
                services[name] = url

        async def _check(name: str, url: str):
            try:
                headers = {}
                if name in RENDER_SERVICES and BRAINOPS_API_KEY:
                    headers["X-API-Key"] = BRAINOPS_API_KEY
                resp = await client.get(url, headers=headers, follow_redirects=True)
                return name, {
                    "status": resp.status_code,
                    "healthy": 200 <= resp.status_code < 400,
                    "url": url,
                    "response_ms": int(resp.elapsed.total_seconds() * 1000),
                }
            except Exception as e:
                return name, {
                    "status": 0,
                    "healthy": False,
                    "url": url,
                    "error": str(e),
                }

        checks = await asyncio.gather(*[_check(n, u) for n, u in services.items()])
        for name, result in checks:
            results[name] = result

    healthy_count = sum(1 for r in results.values() if r["healthy"])
    return {
        "services": results,
        "healthy": healthy_count,
        "total": len(results),
        "all_healthy": healthy_count == len(results),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Action: Restart Service via Render API ──────────────────────────────────


async def restart_service(service_name: str) -> dict[str, Any]:
    """Restart a Render service. Returns real API response."""
    if not RENDER_API_KEY:
        return {"success": False, "error": "RENDER_API_KEY not configured"}

    if service_name not in RENDER_SERVICES:
        return {"success": False, "error": f"Unknown service: {service_name}"}

    service_id, health_url = RENDER_SERVICES[service_name]

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(
                f"https://api.render.com/v1/services/{service_id}/restart",
                headers={
                    "Authorization": f"Bearer {RENDER_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            success = resp.status_code in (200, 202)

            result = {
                "success": success,
                "service_name": service_name,
                "service_id": service_id,
                "status_code": resp.status_code,
                "action": "restart",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Log to memory
            await _log_action("service_restart", result)
            logger.info(f"Service restart {'succeeded' if success else 'FAILED'}: {service_name}")
            return result

    except Exception as e:
        logger.error(f"Service restart error for {service_name}: {e}")
        return {"success": False, "error": str(e), "service_name": service_name}


# ─── Action: Send Alert Email via Resend ─────────────────────────────────────


async def send_alert(
    subject: str,
    body_html: str,
    severity: str = "warning",
    recipient: Optional[str] = None,
) -> dict[str, Any]:
    """Send a real alert email via Resend API."""
    if not RESEND_API_KEY:
        logger.warning(f"Alert not sent (no RESEND_API_KEY): {subject}")
        return {"success": False, "error": "RESEND_API_KEY not configured"}

    to_email = recipient or ALERT_EMAIL

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
            resp = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                json={
                    "from": RESEND_FROM_EMAIL,
                    "to": [to_email],
                    "subject": f"[BrainOps {severity.upper()}] {subject}",
                    "html": body_html,
                },
            )
            success = 200 <= resp.status_code < 300
            result = {
                "success": success,
                "status_code": resp.status_code,
                "recipient": to_email,
                "severity": severity,
                "subject": subject,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if not success:
                result["response"] = resp.text

            await _log_action("alert_sent", result)
            return result

    except Exception as e:
        logger.error(f"Alert email failed: {e}")
        return {"success": False, "error": str(e)}


# ─── Action: Database Maintenance ────────────────────────────────────────────


async def run_db_maintenance(table_name: Optional[str] = None) -> dict[str, Any]:
    """Run VACUUM ANALYZE on a table or the most bloated tables."""
    try:
        from database.async_connection import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            if table_name:
                await conn.execute(f"VACUUM ANALYZE {table_name}")
                return {
                    "success": True,
                    "action": "vacuum_analyze",
                    "table": table_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            else:
                # Find top 5 tables by dead tuple ratio
                rows = await conn.fetch(
                    """
                    SELECT schemaname, relname,
                           n_dead_tup,
                           n_live_tup,
                           CASE WHEN n_live_tup > 0
                                THEN round(n_dead_tup::numeric / n_live_tup * 100, 1)
                                ELSE 0 END as dead_pct
                    FROM pg_stat_user_tables
                    WHERE schemaname = 'public'
                      AND n_dead_tup > 1000
                    ORDER BY n_dead_tup DESC
                    LIMIT 5
                """
                )

                maintained = []
                for row in rows:
                    tbl = row["relname"]
                    await conn.execute(f"VACUUM ANALYZE {tbl}")
                    maintained.append(
                        {
                            "table": tbl,
                            "dead_tuples": row["n_dead_tup"],
                            "dead_pct": float(row["dead_pct"]),
                        }
                    )

                return {
                    "success": True,
                    "action": "vacuum_analyze_auto",
                    "tables_maintained": maintained,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

    except Exception as e:
        logger.error(f"DB maintenance failed: {e}")
        return {"success": False, "error": str(e)}


# ─── OODA Decision Engine ────────────────────────────────────────────────────


async def decide_and_act(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    The REAL decide+act phase of the OODA loop.

    Takes observations from the observe phase and:
    1. Categorizes issues by severity
    2. Decides on concrete actions
    3. Executes those actions
    4. Returns results

    Decision rules (simple, deterministic, no AI needed):
    - Service down -> restart it
    - Service slow (>5s response) -> alert + log
    - DB connection failure -> alert
    - Error spike detected -> alert + restart if service-level
    - All healthy -> store positive status in memory
    """
    decisions = []
    actions_taken = []
    start_time = datetime.now(timezone.utc)

    for obs in observations:
        obs_type = obs.get("type", obs.get("observation", "unknown"))
        severity = obs.get("severity", "info")
        data = obs.get("data", obs)

        # Decision: Service is DOWN
        if obs_type in ("service_down", "backend_unhealthy") or (
            obs_type == "health_check" and not data.get("healthy", True)
        ):
            service_name = data.get("service", data.get("service_name", ""))
            if service_name and service_name in RENDER_SERVICES:
                decisions.append(
                    {
                        "action": "restart_service",
                        "service": service_name,
                        "reason": f"Service {service_name} is unhealthy",
                        "severity": "critical",
                    }
                )

        # Decision: Slow response (>5000ms)
        elif obs_type == "slow_response" or (data.get("response_ms", 0) > 5000):
            service_name = data.get("service", "unknown")
            decisions.append(
                {
                    "action": "alert",
                    "subject": f"Slow response: {service_name} ({data.get('response_ms', '?')}ms)",
                    "severity": "warning",
                }
            )

        # Decision: Error spike
        elif obs_type in ("error_spike", "high_error_rate"):
            decisions.append(
                {
                    "action": "alert",
                    "subject": f"Error spike detected: {data.get('error_count', '?')} errors",
                    "severity": "critical",
                }
            )

        # Decision: DB issues
        elif obs_type in ("db_connection_failure", "db_slow_query"):
            decisions.append(
                {
                    "action": "alert",
                    "subject": f"Database issue: {obs_type}",
                    "severity": "critical",
                }
            )
            if obs_type == "db_slow_query":
                decisions.append(
                    {
                        "action": "db_maintenance",
                        "table": data.get("table"),
                        "reason": "Slow queries detected",
                    }
                )

        # Decision: Overdue invoices (business intelligence)
        elif obs_type == "overdue_invoices" and data.get("count", 0) > 0:
            decisions.append(
                {
                    "action": "alert",
                    "subject": f"{data['count']} overdue invoices (${data.get('total', 0):.0f} total)",
                    "severity": "info",
                }
            )

        # Decision: Churn risk
        elif obs_type == "churn_risks" and data.get("count", 0) > 0:
            decisions.append(
                {
                    "action": "alert",
                    "subject": f"{data['count']} customers at churn risk (>90 days inactive)",
                    "severity": "warning",
                }
            )

    # Execute decisions
    for decision in decisions:
        action = decision["action"]
        result = None

        if action == "restart_service":
            result = await restart_service(decision["service"])
            # Also send alert about the restart
            await send_alert(
                subject=f"Auto-restarted {decision['service']}",
                body_html=f"""
                <h2>Service Auto-Restart</h2>
                <p><strong>Service:</strong> {decision['service']}</p>
                <p><strong>Reason:</strong> {decision['reason']}</p>
                <p><strong>Result:</strong> {'Success' if result.get('success') else 'FAILED'}</p>
                <p><strong>Time:</strong> {datetime.now(timezone.utc).isoformat()}</p>
                """,
                severity="critical",
            )

        elif action == "alert":
            result = await send_alert(
                subject=decision["subject"],
                body_html=f"""
                <h2>BrainOps Alert</h2>
                <p><strong>Issue:</strong> {decision['subject']}</p>
                <p><strong>Severity:</strong> {decision.get('severity', 'info')}</p>
                <p><strong>Time:</strong> {datetime.now(timezone.utc).isoformat()}</p>
                """,
                severity=decision.get("severity", "warning"),
            )

        elif action == "db_maintenance":
            result = await run_db_maintenance(decision.get("table"))

        actions_taken.append(
            {
                "decision": decision,
                "result": result,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    elapsed_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

    summary = {
        "observations_processed": len(observations),
        "decisions_made": len(decisions),
        "actions_taken": len(actions_taken),
        "actions": actions_taken,
        "elapsed_ms": elapsed_ms,
        "timestamp": start_time.isoformat(),
    }

    # Store the decision cycle in memory for learning
    await _log_action("ooda_decision_cycle", summary)

    return summary


# ─── Daily Briefing Email ─────────────────────────────────────────────────────


async def send_daily_briefing() -> dict[str, Any]:
    """Generate and email the daily operational intelligence briefing.

    Called by the agent scheduler every morning. Fetches the real briefing
    data from /ops/briefing logic and formats it as an HTML email.
    """
    from api.real_ops import get_daily_briefing

    try:
        briefing = await get_daily_briefing()
        sections = briefing.get("sections", {})

        # Build HTML email
        health = sections.get("service_health", {})
        revenue = sections.get("revenue", {})
        db = sections.get("database", {})
        agents = sections.get("agent_activity", {})
        email_sec = sections.get("email", {})

        healthy_count = health.get("healthy", "?")
        total_count = health.get("total", "?")
        unhealthy = health.get("unhealthy", [])

        gumroad = revenue.get("gumroad", {})
        mrg = revenue.get("mrg_subscriptions", {})
        leads = revenue.get("leads", {})

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #1a1a2e; border-bottom: 2px solid #16213e;">BrainOps Daily Briefing</h2>
            <p style="color: #666;">{briefing.get('generated_at', 'Unknown')}</p>

            <h3 style="color: #16213e;">Service Health: {healthy_count}/{total_count}</h3>
            {'<p style="color: #27ae60;">All systems operational</p>' if not unhealthy else f'<p style="color: #e74c3c;">Unhealthy: {", ".join(unhealthy)}</p>'}

            <h3 style="color: #16213e;">Revenue</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f8f9fa;">
                    <td style="padding: 8px; border: 1px solid #dee2e6;">Gumroad Revenue</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">${gumroad.get('total_revenue', 0):.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #dee2e6;">MRG Active Subs</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">{mrg.get('active', 0)}</td>
                </tr>
                <tr style="background: #f8f9fa;">
                    <td style="padding: 8px; border: 1px solid #dee2e6;">Active Leads</td>
                    <td style="padding: 8px; border: 1px solid #dee2e6; text-align: right;">{leads.get('active', 0)} / {leads.get('total', 0)}</td>
                </tr>
            </table>

            <h3 style="color: #16213e;">Database</h3>
            <p>Size: {db.get('size', '?')} | Tables: {db.get('tables', '?')} | Memory: {db.get('memory_entries', '?'):,} entries</p>
            <p>Violations: {db.get('unresolved_violations', '?')} | Alerts: {db.get('unresolved_alerts', '?')}</p>

            <h3 style="color: #16213e;">Agent Activity (24h)</h3>
            <p>{'No errors' if isinstance(agents, dict) and 'error' not in agents else agents}</p>

            <h3 style="color: #16213e;">Email (24h)</h3>
            <p>Sent: {email_sec.get('last_24h', {}).get('sent', 0)} | Queued: {email_sec.get('last_24h', {}).get('queued', 0)}</p>

            <hr style="border: 1px solid #eee; margin: 20px 0;">
            <p style="color: #999; font-size: 12px;">BrainOps AI OS v11.25.1 | Automated daily briefing</p>
        </div>
        """

        result = await send_alert(
            subject="Daily Operational Briefing",
            body_html=html,
            severity="info",
        )
        result["briefing_data"] = briefing
        return result

    except Exception as e:
        logger.error(f"Daily briefing email failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ─── Memory logging ──────────────────────────────────────────────────────────


async def _log_action(action_type: str, data: dict[str, Any]) -> None:
    """Log an action to unified_ai_memory for learning and audit."""
    try:
        from unified_memory_manager import get_memory_manager, MemoryType

        mm = get_memory_manager()
        content = f"[{action_type}] {json.dumps(data, default=str)[:500]}"
        mm.store(
            content=content,
            memory_type=MemoryType.EPISODIC,
            source_system="real_action_engine",
            source_agent="ActionEngine",
            metadata={"action_type": action_type, "success": data.get("success")},
        )
    except Exception as e:
        # Don't let memory failures block real actions
        logger.debug(f"Failed to log action to memory: {e}")
