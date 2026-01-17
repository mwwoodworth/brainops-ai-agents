"""
AI Task Queue Consumer
======================
Processes tasks from public.ai_task_queue and executes real workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Optional

from database.async_connection import get_pool

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = (
    os.getenv("DEFAULT_TENANT_ID")
    or os.getenv("TENANT_ID")
    or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
)


def _is_test_email(email: str | None) -> bool:
    if not email:
        return True
    lowered = email.lower().strip()
    return any(
        lowered.endswith(suffix)
        for suffix in (".test", ".example", ".invalid")
    ) or any(
        token in lowered
        for token in ("@example.", "@test.", "@demo.", "@invalid.")
    )


class AITaskQueueConsumer:
    """Consumes ai_task_queue and dispatches real tasks."""

    def __init__(self, poll_interval: int = 30, batch_size: int = 10):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._running = False
        self._executor = None
        self._nurture_agent = None
        self._stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_skipped": 0,
            "last_poll": None,
        }

    async def start(self):
        if self._running:
            logger.warning("AITaskQueueConsumer already running")
            return

        self._running = True
        logger.info("🧠 AI Task Queue Consumer started")

        try:
            from agent_executor import AgentExecutor

            self._executor = AgentExecutor()
        except Exception as exc:
            logger.error("Failed to initialize AgentExecutor: %s", exc)
            self._executor = None

        try:
            from revenue_pipeline_agents import NurtureExecutorAgentReal

            self._nurture_agent = NurtureExecutorAgentReal()
        except Exception as exc:
            logger.error("Failed to initialize NurtureExecutorAgentReal: %s", exc)
            self._nurture_agent = None

        await self._consume_loop()

    async def stop(self):
        self._running = False
        logger.info("🛑 AI Task Queue Consumer stopped")

    async def _consume_loop(self):
        while self._running:
            try:
                await self._reconcile_stale_processing_tasks()
                tasks = await self._fetch_pending_tasks()
                if tasks:
                    logger.info("📥 Fetched %s ai_task_queue tasks", len(tasks))
                    for task in tasks:
                        await self._process_task(task)
                self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()
            except Exception as exc:
                logger.error("Error in ai_task_queue consume loop: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _reconcile_stale_processing_tasks(self) -> None:
        """Fail or requeue tasks stuck in processing for too long (self-healing)."""
        stale_after_seconds = int(os.getenv("AI_TASK_QUEUE_STALE_AFTER_SECONDS", "600"))
        if stale_after_seconds <= 0:
            return

        pool = get_pool()
        async with pool.acquire() as conn:
            stale_rows = await conn.fetch(
                """
                SELECT id, task_type
                FROM ai_task_queue
                WHERE status = 'processing'
                  AND started_at IS NOT NULL
                  AND started_at < NOW() - ($1 * INTERVAL '1 second')
                LIMIT 100
                """,
                stale_after_seconds,
            )

            if not stale_rows:
                return

            quality_ids = [row["id"] for row in stale_rows if (row["task_type"] or "").lower() == "quality_check"]
            other_ids = [row["id"] for row in stale_rows if row["id"] not in quality_ids]

            requeued = 0
            failed = 0

            if quality_ids:
                requeued = await conn.execute(
                    """
                    UPDATE ai_task_queue
                    SET status = 'pending',
                        started_at = NULL,
                        updated_at = NOW(),
                        error_message = 'stale_processing_requeued'
                    WHERE id = ANY($1)
                    """,
                    quality_ids,
                )

            if other_ids:
                failed = await conn.execute(
                    """
                    UPDATE ai_task_queue
                    SET status = 'failed',
                        completed_at = NOW(),
                        updated_at = NOW(),
                        error_message = 'stale_processing_timeout'
                    WHERE id = ANY($1)
                    """,
                    other_ids,
                )

        # asyncpg returns "UPDATE <n>" strings
        try:
            requeued_n = int(str(requeued).split()[-1]) if requeued else 0
        except Exception:
            requeued_n = 0
        try:
            failed_n = int(str(failed).split()[-1]) if failed else 0
        except Exception:
            failed_n = 0

        if requeued_n or failed_n:
            logger.warning(
                "🧹 Reconciled stale ai_task_queue processing tasks | requeued=%s failed=%s stale_after_seconds=%s",
                requeued_n,
                failed_n,
                stale_after_seconds,
            )

    async def _fetch_pending_tasks(self) -> list[dict[str, Any]]:
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, tenant_id, task_type, payload, input_data, priority, status, created_at
                    FROM ai_task_queue
                    WHERE status = 'pending'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT $1
                    FOR UPDATE SKIP LOCKED
                    """,
                    self.batch_size,
                )

                if rows:
                    task_ids = [row["id"] for row in rows]
                    await conn.execute(
                        """
                        UPDATE ai_task_queue
                        SET status = 'processing', started_at = NOW(), updated_at = NOW()
                        WHERE id = ANY($1)
                        """,
                        task_ids,
                    )

                return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("Failed to fetch ai_task_queue tasks: %s", exc)
            return []

    async def _process_task(self, task: dict[str, Any]):
        task_id = str(task.get("id"))
        task_type = (task.get("task_type") or "").lower()
        payload = task.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = {}
        if not isinstance(payload, dict):
            payload = {}

        input_data = task.get("input_data") or {}
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                input_data = {}
        if not isinstance(input_data, dict):
            input_data = {}

        # Normalize: payload is the "active" envelope; input_data is often used by DB triggers.
        # Prefer explicit payload keys over trigger-provided input_data keys.
        payload = {**input_data, **payload}

        tenant_raw = task.get("tenant_id") or DEFAULT_TENANT_ID
        tenant_id = str(tenant_raw) if tenant_raw else DEFAULT_TENANT_ID

        started_at = time.time()

        try:
            timeout_seconds = float(os.getenv("AI_TASK_QUEUE_EXECUTION_TIMEOUT_SECONDS", "60"))

            if task_type == "lead_nurturing":
                result = await asyncio.wait_for(
                    self._handle_lead_nurturing(task_id, tenant_id, payload),
                    timeout=timeout_seconds,
                )
            elif task_type == "quality_check":
                result = await asyncio.wait_for(
                    self._handle_quality_check(task_id, payload),
                    timeout=timeout_seconds,
                )
            else:
                result = {"status": "skipped", "reason": "unsupported_task_type", "task_type": task_type}
                self._stats["tasks_skipped"] += 1

            duration_ms = int((time.time() - started_at) * 1000)
            outcome = (result or {}).get("status")
            if outcome in ("failed", "error"):
                status = "failed"
                self._stats["tasks_failed"] += 1
                logger.error("❌ ai_task_queue task %s failed", task_id)
            elif outcome == "skipped":
                status = "skipped"
                self._stats["tasks_skipped"] += 1
                logger.info("⚠️ ai_task_queue task %s skipped", task_id)
            else:
                status = "completed"
                self._stats["tasks_processed"] += 1
                logger.info("✅ ai_task_queue task %s completed", task_id)

            await self._update_task_status(task_id, status, result, duration_ms)
        except asyncio.TimeoutError:
            duration_ms = int((time.time() - started_at) * 1000)
            result = {
                "status": "failed",
                "error": "timeout",
                "timeout_seconds": float(os.getenv("AI_TASK_QUEUE_EXECUTION_TIMEOUT_SECONDS", "60")),
                "task_type": task_type,
            }
            await self._update_task_status(task_id, "failed", result, duration_ms)
            self._stats["tasks_failed"] += 1
            logger.error("⏱️ ai_task_queue task %s timed out after %ss", task_id, result["timeout_seconds"])
        except Exception as exc:
            duration_ms = int((time.time() - started_at) * 1000)
            await self._update_task_status(task_id, "failed", {"error": str(exc)}, duration_ms)
            self._stats["tasks_failed"] += 1
            logger.error("❌ ai_task_queue task %s failed: %s", task_id, exc)

    async def _handle_lead_nurturing(
        self,
        task_id: str,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._nurture_agent:
            raise RuntimeError("NurtureExecutorAgentReal not available")

        email = payload.get("customer_email") or payload.get("email")
        company_name = (
            payload.get("customer_name")
            or payload.get("company_name")
            or payload.get("company")
            or "Unknown Lead"
        )
        contact_name = payload.get("contact_name") or payload.get("customer_name") or company_name
        phone = payload.get("customer_phone") or payload.get("phone")

        if not email:
            return {"status": "skipped", "reason": "missing_email"}

        is_test = bool(payload.get("is_test")) or _is_test_email(email)

        pool = get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id FROM revenue_leads WHERE email = $1 ORDER BY created_at DESC LIMIT 1",
                email,
            )
            if existing:
                lead_id = str(existing["id"])
            else:
                row = await conn.fetchrow(
                    """
                    INSERT INTO revenue_leads (
                        company_name, contact_name, email, phone,
                        stage, status, source, metadata, created_at, updated_at, is_test
                    ) VALUES ($1, $2, $3, $4, 'new', 'new', $5, $6, NOW(), NOW(), $7)
                    RETURNING id
                    """,
                    company_name,
                    contact_name,
                    email,
                    phone,
                    "ai_task_queue",
                    json.dumps(
                        {
                            "task_id": task_id,
                            "tenant_id": tenant_id,
                            "payload": payload,
                            "lead_type": payload.get("lead_type", "nurture"),
                        },
                        default=str,
                    ),
                    is_test,
                )
                lead_id = str(row["id"]) if row else None

        if not lead_id:
            return {"status": "failed", "reason": "lead_insert_failed"}

        sequence_type = payload.get("sequence_type", "reengagement")
        result = await self._nurture_agent.execute(
            {"action": "create_sequence", "lead_id": lead_id, "sequence_type": sequence_type}
        )

        if isinstance(result, dict) and result.get("status") == "error":
            return {
                "status": "failed",
                "lead_id": lead_id,
                "sequence_type": sequence_type,
                "error": result.get("error"),
                "nurture_result": result,
            }

        return {
            "status": "completed",
            "lead_id": lead_id,
            "sequence_type": sequence_type,
            "nurture_result": result,
        }

    async def _handle_quality_check(self, task_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self._executor:
            raise RuntimeError("AgentExecutor not available")
        result = await self._executor.execute(
            "QualityAgent",
            {"task_id": task_id, "action": "quality_check", "payload": payload, "use_graph_context": False},
        )
        return {"status": "completed", "result": result}

    async def _update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[dict[str, Any]],
        duration_ms: int,
    ):
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE ai_task_queue
                    SET status = $1,
                        result = $2,
                        error_message = $3,
                        processing_time_ms = $4,
                        completed_at = CASE WHEN $1 IN ('completed', 'failed', 'skipped') THEN NOW() ELSE completed_at END,
                        updated_at = NOW()
                    WHERE id = $5
                    """,
                    status,
                    json.dumps(result, default=str) if result else None,
                    result.get("error") if result else None,
                    duration_ms,
                    task_id,
                )
        except Exception as exc:
            logger.error("Failed to update ai_task_queue status: %s", exc)


_ai_task_queue_consumer: Optional[AITaskQueueConsumer] = None
_ai_task_queue_consumer_task: Optional[asyncio.Task] = None


def get_ai_task_queue_consumer() -> AITaskQueueConsumer:
    global _ai_task_queue_consumer
    if _ai_task_queue_consumer is None:
        poll_interval = int(os.getenv("AI_TASK_QUEUE_POLL_INTERVAL", "30"))
        batch_size = int(os.getenv("AI_TASK_QUEUE_BATCH_SIZE", "10"))
        _ai_task_queue_consumer = AITaskQueueConsumer(
            poll_interval=poll_interval,
            batch_size=batch_size,
        )
    return _ai_task_queue_consumer


async def start_ai_task_queue_consumer():
    global _ai_task_queue_consumer_task
    consumer = get_ai_task_queue_consumer()

    if _ai_task_queue_consumer_task is None or _ai_task_queue_consumer_task.done():
        _ai_task_queue_consumer_task = asyncio.create_task(consumer.start())
        logger.info("📋 AI task queue consumer started as background task")

    return consumer


async def stop_ai_task_queue_consumer():
    global _ai_task_queue_consumer_task
    consumer = get_ai_task_queue_consumer()
    await consumer.stop()

    if _ai_task_queue_consumer_task:
        _ai_task_queue_consumer_task.cancel()
        try:
            await _ai_task_queue_consumer_task
        except asyncio.CancelledError:
            logger.debug("AI task queue consumer task cancelled")
        _ai_task_queue_consumer_task = None
