"""
Notification System (ERP bridge support)

The ERP Event Bridge references `get_notification_system()` for producing internal
notifications (job scheduled, invoice created, etc).

This module intentionally:
- Does NOT send outbound email/SMS directly (outbound is fail-closed elsewhere).
- Writes durable, in-app notification records into the ERP database tables.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NotificationPayload:
    title: str
    message: str
    priority: str = "medium"


class NotificationSystem:
    """Durable, in-app notifications only (no outbound sending)."""

    async def send_job_scheduled_notification(
        self,
        *,
        job_id: str | None,
        tenant_id: str,
        scheduled_start: str | None = None,
        scheduled_end: str | None = None,
    ) -> None:
        if not job_id:
            return

        payload = NotificationPayload(
            title="Job scheduled",
            message=f"Job {job_id} scheduled. Start={scheduled_start or 'unknown'} End={scheduled_end or 'unknown'}.",
            priority="medium",
        )
        await self._insert_job_notification(
            job_id=job_id,
            tenant_id=tenant_id,
            notification_type="job_scheduled",
            payload=payload,
            metadata={
                "scheduled_start": scheduled_start,
                "scheduled_end": scheduled_end,
            },
        )

    async def send_invoice_notification(
        self,
        *,
        invoice_id: str | None,
        customer_id: str | None,
        amount: str | int | float | None,
        tenant_id: str,
    ) -> None:
        if not invoice_id:
            return

        payload = NotificationPayload(
            title="Invoice created",
            message=f"Invoice {invoice_id} created for customer {customer_id or 'unknown'} amount={amount or 'unknown'}.",
            priority="medium",
        )

        await self._insert_system_notification(
            notification_type="invoice_created",
            payload=payload,
            entity_type="invoice",
            entity_id=invoice_id,
            target_roles=["accounting"],
            metadata={
                "tenant_id": tenant_id,
                "customer_id": customer_id,
                "amount": amount,
            },
        )

    async def _insert_job_notification(
        self,
        *,
        job_id: str,
        tenant_id: str,
        notification_type: str,
        payload: NotificationPayload,
        metadata: dict,
    ) -> None:
        pool = get_pool()
        try:
            await pool.execute(
                """
                insert into public.job_notifications
                  (job_id, type, priority, title, message, metadata, channels, recipient_ids, created_by, tenant_id, created_at)
                values
                  ($1::uuid, $2, $3, $4, $5, $6::jsonb, '["in_app"]'::jsonb, '[]'::jsonb, null, $7::uuid, now())
                """,
                job_id,
                notification_type,
                payload.priority,
                payload.title,
                payload.message,
                json.dumps(metadata),
                tenant_id,
            )
        except Exception as exc:
            logger.warning("NotificationSystem: failed to insert job_notification: %s", exc)

    async def _insert_system_notification(
        self,
        *,
        notification_type: str,
        payload: NotificationPayload,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        target_user_ids: Optional[list[str]] = None,
        target_roles: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        pool = get_pool()
        now = datetime.now(timezone.utc)
        try:
            await pool.execute(
                """
                insert into public.system_notifications
                  (notification_type, priority, title, message, target_user_ids, target_roles, entity_type, entity_id, action_required, action_url, source_agent, created_at, sent_at)
                values
                  ($1, $2, $3, $4, $5::uuid[], $6::varchar[], $7, $8::uuid, false, null, 'ai_agents.notification_system', $9, $9)
                """,
                notification_type,
                payload.priority,
                payload.title,
                payload.message
                + (f"\n\nmetadata={json.dumps(metadata, default=str)[:4000]}" if metadata else ""),
                target_user_ids or [],
                target_roles or [],
                entity_type,
                entity_id,
                now,
            )
        except Exception as exc:
            logger.warning("NotificationSystem: failed to insert system_notification: %s", exc)


_notification_system: Optional[NotificationSystem] = None


def get_notification_system() -> NotificationSystem:
    global _notification_system
    if _notification_system is None:
        _notification_system = NotificationSystem()
    return _notification_system

