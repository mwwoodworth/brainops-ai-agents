"""
Invariant Monitor — Production Safety Sentinel
================================================
Runs every 5 minutes via agent_scheduler. Detects logical corruption,
security bypasses, RLS gaps, and financial integrity issues.

Features:
- Persists violations to invariant_violations table
- Sends Slack alerts for critical/high severity violations
- RLS coverage check (tables with RLS enabled but 0 policies)
- Financial integrity checks (orphaned invoices, mismatches)
- Runtime privilege self-audit (agent_worker identity enforcement)
- Synthetic canary drill (proves detection pipeline is alive)
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def _persist_violation(pool, check_name: str, severity: str,
                             message: str, details: dict | None = None) -> str | None:
    """Insert a violation row into invariant_violations and return its ID."""
    try:
        row_id = await pool.fetchval("""
            INSERT INTO invariant_violations (check_name, severity, message, details)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING id::text
        """, check_name, severity, message, json.dumps(details or {}))
        return row_id
    except Exception as exc:
        logger.error("Failed to persist violation to invariant_violations: %s", exc)
        # Fallback: log to unified_brain_logs
        try:
            await pool.execute("""
                INSERT INTO unified_brain_logs (system, action, data, created_at)
                VALUES ($1, $2, $3, NOW())
            """, "invariant_monitor", "violation_detected", json.dumps({
                "check": check_name, "severity": severity, "message": message
            }))
        except Exception:
            pass
        return None


async def _send_slack_alert(severity: str, title: str, message: str,
                            fields: dict | None = None):
    """Fire Slack alert through the existing SlackNotifier singleton."""
    try:
        from slack_notifications import get_slack_notifier
        notifier = get_slack_notifier()
        if not notifier.enabled:
            logger.debug("Slack not configured, skipping alert")
            return
        await notifier.send_alert(
            title=title,
            message=message,
            severity=severity,
            fields=fields,
        )
    except Exception as exc:
        logger.warning("Slack alert failed (non-fatal): %s", exc)


async def check_invariants() -> dict:
    """
    Run all invariant checks. Returns a structured summary dict.

    Each check:
      1. Queries the DB for a specific invariant.
      2. If violated, persists to invariant_violations table.
      3. If critical/high, fires a Slack alert.
    """
    from database.async_connection import get_pool
    pool = get_pool()
    violations: list[dict] = []
    checks_run = 0

    async def record(check_name: str, severity: str, msg: str,
                     details: dict | None = None):
        violations.append({"check": check_name, "severity": severity, "message": msg})
        vid = await _persist_violation(pool, check_name, severity, msg, details)
        if severity in ("critical", "high"):
            await _send_slack_alert(
                severity=severity,
                title=f"Invariant Violation: {check_name}",
                message=msg,
                fields={"severity": severity, "violation_id": vid or "unknown"},
            )

    # ── Check 1: Runtime Privilege Self-Audit ──────────────────────────
    checks_run += 1
    try:
        current_user = await pool.fetchval("SELECT current_user")
        if current_user != "agent_worker":
            await record(
                "runtime_identity", "critical",
                f"Agents running as '{current_user}', expected 'agent_worker'",
                {"current_user": current_user},
            )
        can_delete = await pool.fetchval(
            "SELECT has_table_privilege($1, 'users', 'DELETE')", current_user
        )
        if can_delete:
            await record(
                "privilege_escalation", "critical",
                f"User '{current_user}' has DELETE on users table",
            )
    except Exception as exc:
        logger.error("Privilege check failed: %s", exc)

    # ── Check 2: Orphaned Invoices (NULL tenant_id) ───────────────────
    checks_run += 1
    try:
        orphans = await pool.fetchval(
            "SELECT count(*) FROM invoices WHERE tenant_id IS NULL"
        )
        if orphans and orphans > 0:
            await record(
                "orphaned_invoices", "high",
                f"{orphans} invoices with NULL tenant_id",
                {"count": orphans},
            )
    except Exception as exc:
        logger.error("Orphaned invoices check failed: %s", exc)

    # ── Check 3: Cross-Tenant Job/Customer Mismatch ───────────────────
    checks_run += 1
    try:
        mismatches = await pool.fetchval("""
            SELECT count(*)
            FROM jobs j
            JOIN customers c ON j.customer_id = c.id
            WHERE j.tenant_id != c.tenant_id
        """)
        if mismatches and mismatches > 0:
            await record(
                "cross_tenant_mismatch", "critical",
                f"{mismatches} jobs linked to cross-tenant customers",
                {"count": mismatches},
            )
    except Exception as exc:
        logger.error("Cross-tenant check failed: %s", exc)

    # ── Check 4: Stuck Webhooks ───────────────────────────────────────
    checks_run += 1
    try:
        stuck = await pool.fetchval("""
            SELECT count(*)
            FROM stripe_webhook_events
            WHERE status = 'processing'
              AND created_at < NOW() - INTERVAL '1 hour'
        """)
        if stuck and stuck > 0:
            await record(
                "stuck_webhooks", "warning",
                f"{stuck} webhooks stuck in 'processing' for >1 hour",
                {"count": stuck},
            )
    except Exception as exc:
        logger.error("Stuck webhooks check failed: %s", exc)

    # ── Check 5: RLS Coverage — tables with RLS but ZERO policies ─────
    checks_run += 1
    try:
        uncovered = await pool.fetch("""
            SELECT c.relname as table_name
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relrowsecurity = true
              AND NOT EXISTS (
                  SELECT 1 FROM pg_policies p
                  WHERE p.schemaname = 'public' AND p.tablename = c.relname
              )
        """)
        if uncovered:
            table_names = [r["table_name"] for r in uncovered]
            await record(
                "rls_no_policies", "high",
                f"{len(uncovered)} tables have RLS enabled with ZERO policies",
                {"tables": table_names[:20]},
            )
    except Exception as exc:
        logger.error("RLS coverage check failed: %s", exc)

    # ── Check 6: Financial Integrity — Invoice/Line-Item mismatch ─────
    checks_run += 1
    try:
        financial_mismatches = await pool.fetchval("""
            SELECT count(*)
            FROM (
                SELECT i.id
                FROM invoices i
                LEFT JOIN invoice_items ii ON i.id = ii.invoice_id
                GROUP BY i.id, i.total_amount
                HAVING ABS(i.total_amount - COALESCE(SUM(ii.total_price), 0)) > 0.01
            ) AS mismatches
        """)
        if financial_mismatches and financial_mismatches > 0:
            await record(
                "invoice_line_item_mismatch", "high",
                f"{financial_mismatches} invoices have total_amount != sum of line items",
                {"count": financial_mismatches},
            )
    except Exception as exc:
        logger.error("Financial integrity check failed: %s", exc)

    # ── Check 7: Financial Integrity — Negative Invoice Amounts ───────
    checks_run += 1
    try:
        negative = await pool.fetchval(
            "SELECT count(*) FROM invoices WHERE total_amount < 0"
        )
        if negative and negative > 0:
            await record(
                "negative_invoices", "high",
                f"{negative} invoices have negative total_amount",
                {"count": negative},
            )
    except Exception as exc:
        logger.error("Negative invoices check failed: %s", exc)

    # ── Check 8: Revenue Demo Flag — unmarked rows ────────────────────
    checks_run += 1
    try:
        unmarked = await pool.fetchval(
            "SELECT count(*) FROM real_revenue_tracking WHERE is_demo IS NULL"
        )
        if unmarked and unmarked > 0:
            await record(
                "revenue_demo_flag_missing", "warning",
                f"{unmarked} rows in real_revenue_tracking have NULL is_demo flag",
                {"count": unmarked},
            )
    except Exception as exc:
        logger.error("Revenue demo flag check failed: %s", exc)

    # ── Check 9: Synthetic Canary Drill ───────────────────────────────
    # Insert a known canary violation, then verify we can read it back.
    # This proves the full pipeline: DB write → read → persistence works.
    checks_run += 1
    canary_ok = False
    try:
        canary_id = await pool.fetchval("""
            INSERT INTO invariant_violations
                (check_name, severity, message, details, resolved, resolved_at)
            VALUES ('synthetic_canary', 'info',
                    'Synthetic drill — monitoring pipeline alive',
                    '{"drill": true}'::jsonb, true, NOW())
            RETURNING id::text
        """)
        readback = await pool.fetchval(
            "SELECT id::text FROM invariant_violations WHERE id = $1::uuid",
            canary_id,
        )
        if readback != canary_id:
            await record(
                "canary_readback_failed", "critical",
                "Synthetic canary inserted but could not be read back",
            )
        else:
            canary_ok = True
            logger.info("Canary drill passed (id=%s)", canary_id)
    except Exception as exc:
        logger.error("Canary drill failed: %s", exc)
        await record(
            "canary_drill_error", "critical",
            f"Synthetic canary drill threw an exception: {exc}",
        )

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks_run": checks_run,
        "violations_found": len(violations),
        "violations": violations,
        "canary_ok": canary_ok,
        "status": "clean" if not violations else "violations_detected",
    }

    if violations:
        for v in violations:
            logger.error("VIOLATION [%s] %s: %s", v["severity"], v["check"], v["message"])
    else:
        logger.info("All %d invariant checks passed.", checks_run)

    return summary


if __name__ == "__main__":
    from database.async_connection import init_pool, PoolConfig

    logging.basicConfig(level=logging.INFO)

    async def run():
        config = PoolConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            database=os.getenv("DB_NAME", "postgres"),
            ssl=False,
        )
        os.environ["ALLOW_INMEMORY_FALLBACK"] = "1"
        try:
            await init_pool(config)
            result = await check_invariants()
            print(json.dumps(result, indent=2))
        except Exception as exc:
            logger.error("Invariant check failed: %s", exc)
            exit(1)

    try:
        asyncio.run(run())
    except Exception as exc:
        logger.error("Runner failed: %s", exc)
        exit(1)
