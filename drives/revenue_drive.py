#!/usr/bin/env python3
"""
Revenue Drive - Autonomous revenue-oriented scanning and activation.

Scans for stale leads, overdue invoices, and upsell opportunities,
then triggers agent activation via business events.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

from agent_activation_system import BusinessEventType, get_activation_system

logger = logging.getLogger(__name__)


def _build_db_config() -> dict[str, Any]:
    """Build database config, supporting DATABASE_URL fallback."""
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    if not (host and user and password):
        database_url = os.getenv("DATABASE_URL", "")
        if database_url:
            parsed = urlparse(database_url)
            return {
                "host": parsed.hostname or "",
                "database": parsed.path.lstrip("/") if parsed.path else "postgres",
                "user": parsed.username or "",
                "password": parsed.password or "",
                "port": parsed.port or 5432,
            }
        raise RuntimeError("DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL required")

    return {
        "host": host,
        "database": os.getenv("DB_NAME", "postgres"),
        "user": user,
        "password": password,
        "port": int(os.getenv("DB_PORT", "5432")),
    }


@contextmanager
def _get_connection():
    """Get database connection from shared pool when available."""
    try:
        from database.sync_pool import get_sync_pool
        pool = get_sync_pool()
        with pool.get_connection() as conn:
            yield conn
        return
    except Exception:
        pass

    conn = psycopg2.connect(**_build_db_config())
    try:
        yield conn
    finally:
        if conn and not conn.closed:
            conn.close()


class RevenueDrive:
    """Background revenue drive that triggers autonomous revenue actions."""

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        activation_system: Optional[Any] = None,
    ) -> None:
        self.tenant_id = tenant_id or os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID")
        environment = os.getenv("ENVIRONMENT", "production").lower()
        dry_run_env = os.getenv("REVENUE_DRIVE_DRY_RUN")
        self.dry_run = (
            dry_run_env.lower() in ("1", "true", "yes")
            if dry_run_env is not None
            else environment != "production"
        )
        self.stale_days = int(os.getenv("REVENUE_DRIVE_STALE_DAYS", "7"))
        self.overdue_days = int(os.getenv("REVENUE_DRIVE_OVERDUE_DAYS", "14"))
        self.upsell_days = int(os.getenv("REVENUE_DRIVE_UPSELL_DAYS", "30"))
        self.max_items = int(os.getenv("REVENUE_DRIVE_MAX_ITEMS", "25"))
        self.cooldown_hours = int(os.getenv("REVENUE_DRIVE_COOLDOWN_HOURS", "24"))

        self.activation_system = activation_system
        if self.activation_system is None and self.tenant_id:
            self.activation_system = get_activation_system(self.tenant_id)

    def run(self) -> Any:
        """Sync entrypoint for schedulers."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async())
        return loop.create_task(self.run_async())

    async def run_async(self) -> dict[str, Any]:
        """Run the revenue drive scan and trigger events."""
        if not self.activation_system:
            logger.warning("RevenueDrive missing tenant_id; skipping run.")
            return {"status": "skipped", "reason": "tenant_id_missing"}

        stale_leads = self._fetch_stale_leads()
        overdue_invoices = self._fetch_overdue_invoices()
        upsell_candidates = self._fetch_upsell_candidates()

        events_triggered = []
        for lead in stale_leads:
            event = await self._trigger_event(
                BusinessEventType.REVENUE_OPPORTUNITY,
                {
                    "entity_id": lead.get("entity_id"),
                    "entity_type": "lead",
                    "reason": "stale_lead",
                    "lead": lead,
                },
            )
            if event:
                events_triggered.append(event)

        for invoice in overdue_invoices:
            event = await self._trigger_event(
                BusinessEventType.INVOICE_OVERDUE,
                {
                    "entity_id": invoice.get("entity_id"),
                    "entity_type": "invoice",
                    "reason": "overdue_invoice",
                    "invoice": invoice,
                },
            )
            if event:
                events_triggered.append(event)

        for lead in upsell_candidates:
            event = await self._trigger_event(
                BusinessEventType.REVENUE_OPPORTUNITY,
                {
                    "entity_id": lead.get("entity_id"),
                    "entity_type": "lead",
                    "reason": "upsell_candidate",
                    "lead": lead,
                },
            )
            if event:
                events_triggered.append(event)

        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "dry_run": self.dry_run,
            "stale_leads": len(stale_leads),
            "overdue_invoices": len(overdue_invoices),
            "upsell_candidates": len(upsell_candidates),
            "events_triggered": events_triggered,
        }

    async def _trigger_event(self, event_type: BusinessEventType, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Trigger a business event if cooldown allows."""
        entity_id = payload.get("entity_id")
        if entity_id and self._recent_task_exists(event_type.value, str(entity_id)):
            return None

        event_data = {
            **payload,
            "tenant_id": self.tenant_id,
            "verification_mode": self.dry_run,
            "dry_run": self.dry_run,
            "aurea_initiated": True,
        }
        result = await self.activation_system.handle_business_event(event_type, event_data)
        return {"event_type": event_type.value, "result": result}

    def _get_columns(self, table: str) -> set[str]:
        with _get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (table,),
            )
            columns = {row[0] for row in cur.fetchall()}
            cur.close()
            return columns

    def _select_column(self, columns: set[str], column: Optional[str], alias: str) -> sql.SQL:
        if column and column in columns:
            return sql.SQL("{} AS {}").format(sql.Identifier(column), sql.Identifier(alias))
        return sql.SQL("NULL AS {}").format(sql.Identifier(alias))

    def _recent_task_exists(self, trigger_type: str, entity_id: str) -> bool:
        try:
            with _get_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT COUNT(*) FROM ai_autonomous_tasks
                    WHERE trigger_type = %s
                      AND COALESCE(trigger_condition ->> 'entity_id', '') = %s
                      AND created_at > NOW() - INTERVAL %s
                    """,
                    (trigger_type, entity_id, f"{self.cooldown_hours} hours"),
                )
                count = cur.fetchone()[0]
                cur.close()
                return int(count or 0) > 0
        except Exception as exc:
            logger.warning("RevenueDrive cooldown check failed: %s", exc)
            return False

    def _fetch_stale_leads(self) -> list[dict[str, Any]]:
        columns = self._get_columns("revenue_leads")
        if not columns:
            return []

        id_col = "id" if "id" in columns else "lead_id" if "lead_id" in columns else None
        if not id_col:
            logger.warning("RevenueDrive: revenue_leads missing id/lead_id")
            return []

        status_col = "stage" if "stage" in columns else "status" if "status" in columns else None
        last_contact_col = None
        for candidate in ("last_contact", "contacted_at", "updated_at", "created_at"):
            if candidate in columns:
                last_contact_col = candidate
                break

        if not last_contact_col:
            logger.warning("RevenueDrive: revenue_leads missing contact timestamp")
            return []

        company_col = "company_name" if "company_name" in columns else "company" if "company" in columns else None
        email_col = "email" if "email" in columns else None
        value_col = "value_estimate" if "value_estimate" in columns else (
            "estimated_value" if "estimated_value" in columns else None
        )

        fields = [
            self._select_column(columns, id_col, "entity_id"),
            self._select_column(columns, company_col, "company_name"),
            self._select_column(columns, email_col, "email"),
            self._select_column(columns, status_col, "status"),
            self._select_column(columns, last_contact_col, "last_contact"),
            self._select_column(columns, value_col, "value_estimate"),
        ]

        where_clauses = [
            sql.SQL("{} < NOW() - INTERVAL %s").format(sql.Identifier(last_contact_col)),
        ]
        params: list[Any] = [f"{self.stale_days} days"]

        if status_col:
            where_clauses.append(
                sql.SQL("{} NOT IN ('won','lost','closed_won','closed_lost')").format(
                    sql.Identifier(status_col)
                )
            )
        if self.tenant_id and "tenant_id" in columns:
            where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier("tenant_id")))
            params.append(self.tenant_id)

        query = sql.SQL(
            "SELECT {fields} FROM revenue_leads WHERE {where} ORDER BY {order_col} ASC LIMIT %s"
        ).format(
            fields=sql.SQL(", ").join(fields),
            where=sql.SQL(" AND ").join(where_clauses),
            order_col=sql.Identifier(last_contact_col),
        )
        params.append(self.max_items)

        with _get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            return rows

    def _fetch_overdue_invoices(self) -> list[dict[str, Any]]:
        columns = self._get_columns("invoices")
        if not columns:
            return []

        id_col = "id" if "id" in columns else None
        due_col = None
        for candidate in ("due_date", "overdue_date", "invoice_date", "issue_date"):
            if candidate in columns:
                due_col = candidate
                break

        if not id_col or not due_col:
            logger.warning("RevenueDrive: invoices missing id or due_date")
            return []

        status_col = "status" if "status" in columns else None
        payment_status_col = "payment_status" if "payment_status" in columns else None
        balance_col = None
        for candidate in ("balance_due", "amount_due", "balance_cents", "balance"):
            if candidate in columns:
                balance_col = candidate
                break

        number_col = "invoice_number" if "invoice_number" in columns else None
        total_col = "total_amount" if "total_amount" in columns else (
            "total" if "total" in columns else "total_cents" if "total_cents" in columns else None
        )

        fields = [
            self._select_column(columns, id_col, "entity_id"),
            self._select_column(columns, number_col, "invoice_number"),
            self._select_column(columns, total_col, "total_amount"),
            self._select_column(columns, balance_col, "balance_due"),
            self._select_column(columns, due_col, "due_date"),
            self._select_column(columns, status_col, "status"),
            self._select_column(columns, payment_status_col, "payment_status"),
        ]

        where_clauses = [
            sql.SQL("{} < CURRENT_DATE - INTERVAL %s").format(sql.Identifier(due_col)),
        ]
        params: list[Any] = [f"{self.overdue_days} days"]

        if balance_col:
            where_clauses.append(sql.SQL("COALESCE({}, 0) > 0").format(sql.Identifier(balance_col)))
        if status_col:
            where_clauses.append(
                sql.SQL("{} NOT IN ('paid','void','canceled','cancelled')").format(
                    sql.Identifier(status_col)
                )
            )
        if payment_status_col and payment_status_col != status_col:
            where_clauses.append(
                sql.SQL("{} NOT IN ('paid','void','canceled','cancelled')").format(
                    sql.Identifier(payment_status_col)
                )
            )
        if self.tenant_id and "tenant_id" in columns:
            where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier("tenant_id")))
            params.append(self.tenant_id)

        query = sql.SQL(
            "SELECT {fields} FROM invoices WHERE {where} ORDER BY {order_col} ASC LIMIT %s"
        ).format(
            fields=sql.SQL(", ").join(fields),
            where=sql.SQL(" AND ").join(where_clauses),
            order_col=sql.Identifier(due_col),
        )
        params.append(self.max_items)

        with _get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            return rows

    def _fetch_upsell_candidates(self) -> list[dict[str, Any]]:
        columns = self._get_columns("revenue_leads")
        if not columns:
            return []

        id_col = "id" if "id" in columns else "lead_id" if "lead_id" in columns else None
        status_col = "stage" if "stage" in columns else "status" if "status" in columns else None
        last_contact_col = None
        for candidate in ("last_contact", "contacted_at", "updated_at", "created_at"):
            if candidate in columns:
                last_contact_col = candidate
                break

        if not id_col or not status_col or not last_contact_col:
            return []

        company_col = "company_name" if "company_name" in columns else "company" if "company" in columns else None
        email_col = "email" if "email" in columns else None
        value_col = "value_estimate" if "value_estimate" in columns else (
            "estimated_value" if "estimated_value" in columns else None
        )

        fields = [
            self._select_column(columns, id_col, "entity_id"),
            self._select_column(columns, company_col, "company_name"),
            self._select_column(columns, email_col, "email"),
            self._select_column(columns, status_col, "status"),
            self._select_column(columns, last_contact_col, "last_contact"),
            self._select_column(columns, value_col, "value_estimate"),
        ]

        where_clauses = [
            sql.SQL("{} IN ('won','closed_won','customer')").format(sql.Identifier(status_col)),
            sql.SQL("{} < NOW() - INTERVAL %s").format(sql.Identifier(last_contact_col)),
        ]
        params: list[Any] = [f"{self.upsell_days} days"]

        if self.tenant_id and "tenant_id" in columns:
            where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier("tenant_id")))
            params.append(self.tenant_id)

        query = sql.SQL(
            "SELECT {fields} FROM revenue_leads WHERE {where} ORDER BY {order_col} ASC LIMIT %s"
        ).format(
            fields=sql.SQL(", ").join(fields),
            where=sql.SQL(" AND ").join(where_clauses),
            order_col=sql.Identifier(last_contact_col),
        )
        params.append(self.max_items)

        with _get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            return rows
