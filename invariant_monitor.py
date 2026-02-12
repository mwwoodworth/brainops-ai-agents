"""
Invariant Engine — Production Safety Daemon
=============================================
Continuous monitoring daemon that detects logical corruption,
security bypasses, RLS gaps, drift, and financial integrity issues.

Architecture:
  - InvariantEngine class encapsulates all state and check logic.
  - Persists violations to invariant_violations table.
  - Fires Slack alerts for critical/high severity.
  - Maintains run_history for trend analysis.
  - Backward-compatible check_invariants() function for scheduler.

Checks (14 total):
  1.  runtime_identity       — Agent running as expected DB user
  2.  privilege_escalation   — No DELETE on protected tables
  3.  orphaned_invoices      — No NULL tenant_id on invoices
  4.  cross_tenant_mismatch  — Jobs linked to correct tenant customers
  5.  stuck_webhooks         — No webhooks stuck in processing >1h
  6.  rls_no_policies        — No tables with RLS enabled but 0 policies
  7.  rls_disabled_new       — Detect tables with RLS disabled entirely
  8.  invoice_line_item_mismatch — Invoice total matches line items
  9.  negative_invoices      — No negative invoice amounts
  10. revenue_demo_flag      — All revenue rows have is_demo flag
  11. role_drift             — Agent role membership unchanged
  12. correlation_missing    — Recent agent executions have correlation_id
  13. financial_double_count — No duplicate revenue tracking rows
  14. synthetic_canary       — Write/read pipeline liveness
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class InvariantEngine:
    """
    Stateful invariant monitoring daemon.

    Maintains run history, persists violations, and fires alerts.
    Designed to be instantiated once and called repeatedly by the scheduler.
    """

    EXPECTED_AGENT_USER = "agent_worker"
    EXPECTED_ROLE_MEMBERSHIPS = {"app_agent_role"}

    def __init__(self) -> None:
        self.run_count = 0
        self.last_run: str | None = None
        self.last_violations: list[dict] = []
        self.consecutive_clean = 0
        self.total_violations_detected = 0
        self._pool = None

    def _get_pool(self):
        """Get the database pool (system-level, not tenant-scoped)."""
        if self._pool is None:
            from database.async_connection import get_pool
            self._pool = get_pool()
        return self._pool

    async def _persist_violation(self, check_name: str, severity: str,
                                 message: str, details: dict | None = None) -> str | None:
        """Insert a violation row and return its ID."""
        pool = self._get_pool()
        try:
            row_id = await pool.fetchval("""
                INSERT INTO invariant_violations (check_name, severity, message, details)
                VALUES ($1, $2, $3, $4::jsonb)
                RETURNING id::text
            """, check_name, severity, message, json.dumps(details or {}))
            return row_id
        except Exception as exc:
            logger.error("Failed to persist violation: %s", exc)
            try:
                await pool.execute("""
                    INSERT INTO unified_brain_logs (system, action, data, created_at)
                    VALUES ($1, $2, $3, NOW())
                """, "invariant_engine", "violation_detected", json.dumps({
                    "check": check_name, "severity": severity, "message": message
                }))
            except Exception:
                pass
            return None

    async def _send_slack_alert(self, severity: str, title: str,
                                message: str, fields: dict | None = None) -> None:
        """Fire Slack alert for critical/high violations."""
        try:
            from slack_notifications import get_slack_notifier
            notifier = get_slack_notifier()
            if not notifier.enabled:
                return
            await notifier.send_alert(
                title=title, message=message,
                severity=severity, fields=fields,
            )
        except Exception as exc:
            logger.warning("Slack alert failed (non-fatal): %s", exc)

    async def _record(self, violations: list, check_name: str,
                      severity: str, msg: str, details: dict | None = None) -> None:
        """Record a violation: persist + alert + append to run list."""
        violations.append({
            "check": check_name, "severity": severity, "message": msg,
        })
        vid = await self._persist_violation(check_name, severity, msg, details)
        if severity in ("critical", "high"):
            await self._send_slack_alert(
                severity=severity,
                title=f"Invariant Violation: {check_name}",
                message=msg,
                fields={"severity": severity, "violation_id": vid or "unknown"},
            )

    # ════════════════════════════════════════════════════════════════════
    #  INDIVIDUAL CHECKS
    # ════════════════════════════════════════════════════════════════════

    async def _check_runtime_identity(self, pool, violations: list) -> None:
        """Check 1: Agent running as expected DB user."""
        try:
            current_user = await pool.fetchval("SELECT current_user")
            if current_user != self.EXPECTED_AGENT_USER:
                await self._record(violations,
                    "runtime_identity", "critical",
                    f"Agents running as '{current_user}', expected '{self.EXPECTED_AGENT_USER}'",
                    {"current_user": current_user},
                )
        except Exception as exc:
            logger.error("Runtime identity check failed: %s", exc)

    async def _check_privilege_escalation(self, pool, violations: list) -> None:
        """Check 2: No DELETE privilege on protected tables."""
        try:
            current_user = await pool.fetchval("SELECT current_user")
            can_delete = await pool.fetchval(
                "SELECT has_table_privilege($1, 'users', 'DELETE')", current_user
            )
            if can_delete:
                await self._record(violations,
                    "privilege_escalation", "critical",
                    f"User '{current_user}' has DELETE on users table",
                )
        except Exception as exc:
            logger.error("Privilege escalation check failed: %s", exc)

    async def _check_orphaned_invoices(self, pool, violations: list) -> None:
        """Check 3: No invoices with NULL tenant_id."""
        try:
            orphans = await pool.fetchval(
                "SELECT count(*) FROM invoices WHERE tenant_id IS NULL"
            )
            if orphans and orphans > 0:
                await self._record(violations,
                    "orphaned_invoices", "high",
                    f"{orphans} invoices with NULL tenant_id",
                    {"count": orphans},
                )
        except Exception as exc:
            logger.error("Orphaned invoices check failed: %s", exc)

    async def _check_cross_tenant_mismatch(self, pool, violations: list) -> None:
        """Check 4: Jobs linked to correct tenant customers."""
        try:
            mismatches = await pool.fetchval("""
                SELECT count(*)
                FROM jobs j
                JOIN customers c ON j.customer_id = c.id
                WHERE j.tenant_id != c.tenant_id
            """)
            if mismatches and mismatches > 0:
                await self._record(violations,
                    "cross_tenant_mismatch", "critical",
                    f"{mismatches} jobs linked to cross-tenant customers",
                    {"count": mismatches},
                )
        except Exception as exc:
            logger.error("Cross-tenant mismatch check failed: %s", exc)

    async def _check_stuck_webhooks(self, pool, violations: list) -> None:
        """Check 5: No webhooks stuck in processing > 1 hour."""
        try:
            stuck = await pool.fetchval("""
                SELECT count(*)
                FROM stripe_webhook_events
                WHERE status = 'processing'
                  AND created_at < NOW() - INTERVAL '1 hour'
            """)
            if stuck and stuck > 0:
                await self._record(violations,
                    "stuck_webhooks", "warning",
                    f"{stuck} webhooks stuck in 'processing' for >1 hour",
                    {"count": stuck},
                )
        except Exception as exc:
            logger.error("Stuck webhooks check failed: %s", exc)

    async def _check_rls_no_policies(self, pool, violations: list) -> None:
        """Check 6: No tables with RLS enabled but zero policies."""
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
                await self._record(violations,
                    "rls_no_policies", "high",
                    f"{len(uncovered)} tables have RLS enabled with ZERO policies",
                    {"tables": table_names[:20]},
                )
        except Exception as exc:
            logger.error("RLS no-policies check failed: %s", exc)

    async def _check_rls_disabled(self, pool, violations: list) -> None:
        """Check 7: Detect NEW tables with RLS disabled (drift detection).

        Maintains a baseline of known-unprotected tables. Alerts only when
        a NEW table appears without RLS that is not on the known-safe list.
        """
        # Known tables without RLS (ERP feature tables, PostGIS system table)
        known_no_rls = {
            "spatial_ref_sys",
            "analytics_events", "api_metrics", "automation_sequences",
            "communication_preferences", "custom_field_values", "equipment_usage",
            "estimate_versions", "failed_payments", "follow_ups",
            "installability_windows", "insurance_tracking", "inventory_usage",
            "maintenance_predictions", "maintenance_records", "pdf_templates",
            "pipeline_health", "quality_inspections", "quickbooks_entity_mappings",
            "refunds", "safety_checklist_templates", "safety_incidents",
            "schedule_plans", "sms_messages", "takeoff_features", "tax_settings",
            "thread_participants", "usage_alerts", "usage_records",
            "vendor_bill_payments", "vendor_evaluations", "weather_forecast_cache",
        }
        try:
            no_rls = await pool.fetch("""
                SELECT c.relname as table_name
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relrowsecurity = false
            """)
            new_tables = [
                r["table_name"] for r in no_rls
                if r["table_name"] not in known_no_rls
            ]
            if new_tables:
                await self._record(violations,
                    "rls_disabled_new", "high",
                    f"{len(new_tables)} NEW table(s) without RLS detected",
                    {"tables": new_tables[:20]},
                )
        except Exception as exc:
            logger.error("RLS disabled check failed: %s", exc)

    async def _check_invoice_line_item_mismatch(self, pool, violations: list) -> None:
        """Check 8: Invoice total matches sum of line items."""
        try:
            mismatches = await pool.fetchval("""
                SELECT count(*)
                FROM (
                    SELECT i.id
                    FROM invoices i
                    LEFT JOIN invoice_items ii ON i.id = ii.invoice_id
                    GROUP BY i.id, i.total_amount
                    HAVING ABS(i.total_amount - COALESCE(SUM(ii.total_price), 0)) > 0.01
                ) AS mismatches
            """)
            if mismatches and mismatches > 0:
                await self._record(violations,
                    "invoice_line_item_mismatch", "high",
                    f"{mismatches} invoices have total_amount != sum of line items",
                    {"count": mismatches},
                )
        except Exception as exc:
            logger.error("Invoice line-item mismatch check failed: %s", exc)

    async def _check_negative_invoices(self, pool, violations: list) -> None:
        """Check 9: No negative invoice amounts."""
        try:
            negative = await pool.fetchval(
                "SELECT count(*) FROM invoices WHERE total_amount < 0"
            )
            if negative and negative > 0:
                await self._record(violations,
                    "negative_invoices", "high",
                    f"{negative} invoices have negative total_amount",
                    {"count": negative},
                )
        except Exception as exc:
            logger.error("Negative invoices check failed: %s", exc)

    async def _check_revenue_demo_flag(self, pool, violations: list) -> None:
        """Check 10: All revenue tracking rows have is_demo flag set."""
        try:
            unmarked = await pool.fetchval(
                "SELECT count(*) FROM real_revenue_tracking WHERE is_demo IS NULL"
            )
            if unmarked and unmarked > 0:
                await self._record(violations,
                    "revenue_demo_flag_missing", "warning",
                    f"{unmarked} rows in real_revenue_tracking have NULL is_demo flag",
                    {"count": unmarked},
                )
        except Exception as exc:
            logger.error("Revenue demo flag check failed: %s", exc)

    async def _check_role_drift(self, pool, violations: list) -> None:
        """Check 11: agent_worker role memberships unchanged.

        Detects if the agent_worker role gains unexpected memberships
        (e.g., someone grants it superuser or additional roles).
        """
        try:
            memberships = await pool.fetch("""
                SELECT r.rolname
                FROM pg_auth_members m
                JOIN pg_roles r ON r.oid = m.roleid
                JOIN pg_roles u ON u.oid = m.member
                WHERE u.rolname = 'agent_worker'
            """)
            current_roles = {r["rolname"] for r in memberships}
            unexpected = current_roles - self.EXPECTED_ROLE_MEMBERSHIPS
            if unexpected:
                await self._record(violations,
                    "role_drift", "critical",
                    f"agent_worker has unexpected role memberships: {unexpected}",
                    {"expected": list(self.EXPECTED_ROLE_MEMBERSHIPS),
                     "actual": list(current_roles),
                     "unexpected": list(unexpected)},
                )
            missing = self.EXPECTED_ROLE_MEMBERSHIPS - current_roles
            if missing:
                await self._record(violations,
                    "role_drift", "high",
                    f"agent_worker MISSING expected role memberships: {missing}",
                    {"expected": list(self.EXPECTED_ROLE_MEMBERSHIPS),
                     "actual": list(current_roles),
                     "missing": list(missing)},
                )
        except Exception as exc:
            logger.error("Role drift check failed: %s", exc)

    async def _check_correlation_missing(self, pool, violations: list) -> None:
        """Check 12: Recent agent executions should have correlation_id.

        Checks ai_agent_executions from the last hour for missing correlation data.
        This is a SOFT check (warning) since correlation_id propagation is
        being rolled out incrementally.
        """
        try:
            # Check if ai_agent_executions has a correlation_id column
            has_col = await pool.fetchval("""
                SELECT count(*) FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'ai_agent_executions'
                  AND column_name = 'correlation_id'
            """)
            if not has_col:
                # Column doesn't exist yet — skip silently
                return

            total = await pool.fetchval("""
                SELECT count(*) FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            missing = await pool.fetchval("""
                SELECT count(*) FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour'
                  AND (correlation_id IS NULL OR correlation_id = '')
            """)
            if total and total > 0 and missing and missing > 0:
                pct = round(missing / total * 100, 1)
                if pct > 50:
                    await self._record(violations,
                        "correlation_missing", "warning",
                        f"{missing}/{total} ({pct}%) agent executions missing correlation_id in last hour",
                        {"total": total, "missing": missing, "percent": pct},
                    )
        except Exception as exc:
            logger.error("Correlation check failed: %s", exc)

    async def _check_financial_double_count(self, pool, violations: list) -> None:
        """Check 13: No duplicate revenue tracking rows (double-counting)."""
        try:
            duplicates = await pool.fetchval("""
                SELECT count(*) FROM (
                    SELECT source_event_id, count(*)
                    FROM real_revenue_tracking
                    WHERE source_event_id IS NOT NULL
                    GROUP BY source_event_id
                    HAVING count(*) > 1
                ) AS dupes
            """)
            if duplicates and duplicates > 0:
                await self._record(violations,
                    "financial_double_count", "high",
                    f"{duplicates} duplicate source_event_id entries in revenue tracking",
                    {"count": duplicates},
                )
        except Exception as exc:
            # Column may not exist — that's OK
            if "column" in str(exc).lower() and "does not exist" in str(exc).lower():
                return
            logger.error("Financial double-count check failed: %s", exc)

    async def _check_synthetic_canary(self, pool, violations: list) -> bool:
        """Check 14: Synthetic canary drill — proves write/read pipeline alive."""
        try:
            canary_id = await pool.fetchval("""
                INSERT INTO invariant_violations
                    (check_name, severity, message, details, resolved, resolved_at)
                VALUES ('synthetic_canary', 'info',
                        'Synthetic drill -- monitoring pipeline alive',
                        '{"drill": true}'::jsonb, true, NOW())
                RETURNING id::text
            """)
            readback = await pool.fetchval(
                "SELECT id::text FROM invariant_violations WHERE id = $1::uuid",
                canary_id,
            )
            if readback != canary_id:
                await self._record(violations,
                    "canary_readback_failed", "critical",
                    "Synthetic canary inserted but could not be read back",
                )
                return False
            logger.info("Canary drill passed (id=%s)", canary_id)
            return True
        except Exception as exc:
            logger.error("Canary drill failed: %s", exc)
            await self._record(violations,
                "canary_drill_error", "critical",
                f"Synthetic canary drill threw an exception: {exc}",
            )
            return False

    # ════════════════════════════════════════════════════════════════════
    #  MAIN RUN METHOD
    # ════════════════════════════════════════════════════════════════════

    async def _resolve_stale_violations(self, pool) -> int:
        """Resolve all open violations before re-running checks.

        Each run creates fresh violations for currently-failing checks.
        Previously-open violations are auto-resolved so they don't accumulate.
        """
        try:
            resolved = await pool.fetchval("""
                UPDATE invariant_violations
                SET resolved = true, resolved_at = NOW()
                WHERE resolved = false
                  AND check_name != 'synthetic_canary'
                RETURNING count(*)
            """)
            # fetchval on UPDATE RETURNING count(*) may not work; use execute + status
            return 0  # placeholder
        except Exception:
            pass

        try:
            result = await pool.execute("""
                UPDATE invariant_violations
                SET resolved = true, resolved_at = NOW()
                WHERE resolved = false
                  AND check_name != 'synthetic_canary'
            """)
            # asyncpg returns 'UPDATE N' string
            if result and result.startswith("UPDATE "):
                return int(result.split()[-1])
            return 0
        except Exception as exc:
            logger.warning("Failed to resolve stale violations: %s", exc)
            return 0

    async def run(self) -> dict:
        """Execute all invariant checks. Returns structured summary."""
        pool = self._get_pool()
        violations: list[dict] = []
        checks_run = 0

        # Auto-resolve previous violations; fresh ones are created below
        stale_resolved = await self._resolve_stale_violations(pool)

        checks = [
            self._check_runtime_identity,
            self._check_privilege_escalation,
            self._check_orphaned_invoices,
            self._check_cross_tenant_mismatch,
            self._check_stuck_webhooks,
            self._check_rls_no_policies,
            self._check_rls_disabled,
            self._check_invoice_line_item_mismatch,
            self._check_negative_invoices,
            self._check_revenue_demo_flag,
            self._check_role_drift,
            self._check_correlation_missing,
            self._check_financial_double_count,
        ]

        for check_fn in checks:
            checks_run += 1
            await check_fn(pool, violations)

        # Synthetic canary is special (returns bool)
        checks_run += 1
        canary_ok = await self._check_synthetic_canary(pool, violations)

        # Update daemon state
        self.run_count += 1
        self.last_run = datetime.now(timezone.utc).isoformat()
        self.last_violations = violations
        self.total_violations_detected += len(violations)
        if not violations:
            self.consecutive_clean += 1
        else:
            self.consecutive_clean = 0

        summary = {
            "timestamp": self.last_run,
            "checks_run": checks_run,
            "violations_found": len(violations),
            "stale_resolved": stale_resolved,
            "violations": violations,
            "canary_ok": canary_ok,
            "status": "clean" if not violations else "violations_detected",
            "run_count": self.run_count,
            "consecutive_clean": self.consecutive_clean,
        }

        if violations:
            for v in violations:
                logger.error("VIOLATION [%s] %s: %s",
                             v["severity"], v["check"], v["message"])
        else:
            logger.info("All %d invariant checks passed (run #%d, %d consecutive clean).",
                        checks_run, self.run_count, self.consecutive_clean)

        return summary

    def get_status(self) -> dict:
        """Return daemon status for API/dashboard consumption."""
        return {
            "run_count": self.run_count,
            "last_run": self.last_run,
            "last_violations": self.last_violations,
            "consecutive_clean": self.consecutive_clean,
            "total_violations_detected": self.total_violations_detected,
            "checks_registered": 14,
        }


# ════════════════════════════════════════════════════════════════════════
#  SINGLETON + BACKWARD COMPATIBILITY
# ════════════════════════════════════════════════════════════════════════

_engine: InvariantEngine | None = None


def get_invariant_engine() -> InvariantEngine:
    """Get or create the singleton InvariantEngine."""
    global _engine
    if _engine is None:
        _engine = InvariantEngine()
    return _engine


async def check_invariants() -> dict:
    """Backward-compatible entry point for agent_scheduler."""
    engine = get_invariant_engine()
    return await engine.run()


if __name__ == "__main__":
    from database.async_connection import init_pool, PoolConfig

    logging.basicConfig(level=logging.INFO)

    def _as_bool(value: str | None) -> bool:
        return (value or "").strip().lower() in {"1", "true", "yes"}

    def _pool_config_from_env() -> PoolConfig:
        database_url = (os.getenv("DATABASE_URL") or "").strip()
        if database_url:
            parsed = urlparse(database_url)
            db_name = parsed.path.lstrip("/")
            if not parsed.hostname or not parsed.username or parsed.password is None or not db_name:
                raise RuntimeError(
                    "DATABASE_URL must include host, database, username, and password for invariant_monitor."
                )
            return PoolConfig(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=db_name,
                ssl=not _as_bool(os.getenv("DB_SSL_DISABLE")),
            )

        required = ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD")
        missing = [key for key in required if not (os.getenv(key) or "").strip()]
        if missing:
            raise RuntimeError(
                "Missing required DB env vars for invariant_monitor: "
                + ", ".join(missing)
                + ". Set DATABASE_URL or explicit DB_HOST/DB_NAME/DB_USER/DB_PASSWORD."
            )

        return PoolConfig(
            host=(os.getenv("DB_HOST") or "").strip(),
            port=int((os.getenv("DB_PORT") or "5432").strip()),
            user=(os.getenv("DB_USER") or "").strip(),
            password=(os.getenv("DB_PASSWORD") or "").strip(),
            database=(os.getenv("DB_NAME") or "").strip(),
            ssl=not _as_bool(os.getenv("DB_SSL_DISABLE")),
        )

    async def run():
        env = (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "production").strip().lower()
        if env in {"production", "prod"} and _as_bool(os.getenv("ALLOW_INMEMORY_FALLBACK")):
            raise RuntimeError("ALLOW_INMEMORY_FALLBACK is forbidden for production invariant checks.")

        config = _pool_config_from_env()

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
