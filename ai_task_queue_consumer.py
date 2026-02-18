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

from safe_task import create_safe_task
from database.async_connection import get_pool, get_tenant_pool

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = (
    os.getenv("DEFAULT_TENANT_ID")
    or os.getenv("TENANT_ID")
    or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
)


def _env_flag(name: str, default: bool = False) -> bool:
    """Return whether an environment-variable feature flag is enabled."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


ENABLE_AI_TASK_PRIORITY_PREEMPTION = _env_flag("ENABLE_AI_TASK_PRIORITY_PREEMPTION", False)
ENABLE_AI_TASK_EXECUTION_BUDGETS = _env_flag("ENABLE_AI_TASK_EXECUTION_BUDGETS", False)
ENABLE_AI_TASK_MULTI_AGENT_ROUTING = _env_flag("ENABLE_AI_TASK_MULTI_AGENT_ROUTING", False)
CRITICAL_TASK_PRIORITY_THRESHOLD = int(os.getenv("AI_TASK_QUEUE_CRITICAL_PRIORITY", "90"))


def _priority_value(priority: Any) -> int:
    """Normalize queue priority into comparable integer value."""
    if isinstance(priority, (int, float)):
        return int(priority)
    if isinstance(priority, str):
        lowered = priority.strip().lower()
        if lowered in {"critical", "urgent", "p0"}:
            return 100
        if lowered in {"high", "p1"}:
            return 75
        if lowered in {"medium", "normal", "p2"}:
            return 50
        if lowered in {"low", "p3"}:
            return 25
        try:
            return int(float(lowered))
        except Exception:
            return 0
    return 0


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
        """Main consumption loop with advisory lock to prevent race conditions"""
        # Advisory lock key for ai_task_queue (different from TaskQueueConsumer)
        ADVISORY_LOCK_KEY = 8675309002  # Unique key for this consumer

        while self._running:
            try:
                pool = get_pool()

                tasks: list[dict[str, Any]] = []
                lock_acquired = False

                # Advisory locks are session-scoped. Acquire/release the lock quickly to avoid
                # holding a database connection while executing long-running tasks (LLM calls).
                async with pool.acquire() as lock_conn:
                    lock_acquired = bool(
                        await lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", ADVISORY_LOCK_KEY)
                    )

                    if lock_acquired:
                        try:
                            # IMPORTANT: Do not acquire additional pool connections while holding
                            # the advisory lock connection. Nested acquisitions can deadlock the pool
                            # under load and starve HTTP endpoints.
                            await self._reconcile_stale_processing_tasks(lock_conn)
                            tasks = await self._fetch_pending_tasks(lock_conn)
                            self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()
                        finally:
                            try:
                                await lock_conn.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_KEY)
                            except Exception as unlock_exc:
                                logger.warning("AITaskQueueConsumer: failed to release advisory lock: %s", unlock_exc)

                if not lock_acquired:
                    logger.debug("AITaskQueueConsumer: Another instance holds the lock, waiting...")
                    await asyncio.sleep(5)
                    continue

                if tasks:
                    logger.info("📥 Fetched %s ai_task_queue tasks", len(tasks))
                    for task in tasks:
                        await self._process_task(task)

            except Exception as exc:
                logger.error("Error in ai_task_queue consume loop: %s", exc)
            await asyncio.sleep(self.poll_interval)

    async def _reconcile_stale_processing_tasks(self, conn: Any | None = None) -> None:
        """Fail or requeue tasks stuck in processing for too long (self-healing)."""
        stale_after_seconds = int(os.getenv("AI_TASK_QUEUE_STALE_AFTER_SECONDS", "600"))
        if stale_after_seconds <= 0:
            return

        if conn is None:
            pool = get_pool()
            async with pool.acquire() as pooled_conn:
                await self._reconcile_stale_processing_tasks(pooled_conn)
            return

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

    async def _fetch_pending_tasks(self, conn: Any | None = None) -> list[dict[str, Any]]:
        try:
            if conn is None:
                pool = get_pool()
                async with pool.acquire() as pooled_conn:
                    return await self._fetch_pending_tasks(pooled_conn)

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

    async def _has_higher_priority_pending(self, current_priority: int) -> bool:
        """Return True when a higher-priority pending task exists."""
        if not ENABLE_AI_TASK_PRIORITY_PREEMPTION:
            return False
        if current_priority >= CRITICAL_TASK_PRIORITY_THRESHOLD:
            return False

        try:
            tenant_pool = get_tenant_pool(DEFAULT_TENANT_ID)
            conn = await tenant_pool.acquire()
            try:
                higher = await conn.fetchval(
                    """
                    SELECT EXISTS(
                        SELECT 1
                        FROM ai_task_queue
                        WHERE status = 'pending'
                          AND priority >= $1
                    )
                    """,
                    CRITICAL_TASK_PRIORITY_THRESHOLD,
                )
                return bool(higher)
            finally:
                await tenant_pool.release(conn)
        except Exception as exc:
            logger.warning("Priority preemption check failed: %s", exc)
            return False

    async def _requeue_for_preemption(self, task_id: str, current_priority: int) -> bool:
        """Requeue a task to allow higher-priority work to preempt."""
        if not ENABLE_AI_TASK_PRIORITY_PREEMPTION:
            return False

        should_preempt = await self._has_higher_priority_pending(current_priority)
        if not should_preempt:
            return False

        try:
            tenant_pool = get_tenant_pool(DEFAULT_TENANT_ID)
            conn = await tenant_pool.acquire()
            try:
                await conn.execute(
                    """
                    UPDATE ai_task_queue
                    SET status = 'pending',
                        started_at = NULL,
                        updated_at = NOW(),
                        error_message = $1
                    WHERE id = $2
                    """,
                    "preempted_by_critical_priority",
                    task_id,
                )
            finally:
                await tenant_pool.release(conn)
            logger.info("⏭️ Preempted ai_task_queue task %s (priority=%s)", task_id, current_priority)
            return True
        except Exception as exc:
            logger.warning("Failed to requeue preempted task %s: %s", task_id, exc)
            return False

    def _resolve_task_timeout(self, task_type: str, task_priority: int) -> float:
        """Resolve execution timeout with optional priority-aware budgets."""
        timeout_seconds = float(os.getenv("AI_TASK_QUEUE_EXECUTION_TIMEOUT_SECONDS", "60"))
        if task_type == "self_build":
            timeout_seconds = float(os.getenv("AI_TASK_QUEUE_SELF_BUILD_TIMEOUT_SECONDS", "900"))
        elif task_type == "revenue_prompt_compile":
            timeout_seconds = float(os.getenv("AI_TASK_QUEUE_DSPY_COMPILE_TIMEOUT_SECONDS", "600"))

        if ENABLE_AI_TASK_EXECUTION_BUDGETS:
            if task_priority >= CRITICAL_TASK_PRIORITY_THRESHOLD:
                timeout_seconds = min(
                    timeout_seconds,
                    float(os.getenv("AI_TASK_QUEUE_CRITICAL_TIMEOUT_SECONDS", "45")),
                )
            elif task_priority <= 25:
                timeout_seconds = min(
                    timeout_seconds,
                    float(os.getenv("AI_TASK_QUEUE_LOW_PRIORITY_TIMEOUT_SECONDS", "30")),
                )
        return timeout_seconds

    async def _process_task(self, task: dict[str, Any]):
        task_id = str(task.get("id"))
        task_type = (task.get("task_type") or "").lower()
        task_priority = _priority_value(task.get("priority"))
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
        timeout_seconds = self._resolve_task_timeout(task_type, task_priority)

        try:
            if await self._requeue_for_preemption(task_id, task_priority):
                self._stats["tasks_skipped"] += 1
                return

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
            elif task_type == "self_build":
                result = await asyncio.wait_for(
                    self._handle_self_build(task_id, tenant_id, payload),
                    timeout=timeout_seconds,
                )
            elif task_type == "revenue_prompt_compile":
                result = await asyncio.wait_for(
                    self._handle_revenue_prompt_compile(task_id, tenant_id, payload),
                    timeout=timeout_seconds,
                )
            elif task_type == "multi_agent_collaboration":
                result = await asyncio.wait_for(
                    self._handle_multi_agent_collaboration(task_id, tenant_id, payload),
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
                "timeout_seconds": timeout_seconds,
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

    async def _handle_self_build(
        self,
        task_id: str,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._executor:
            raise RuntimeError("AgentExecutor not available")

        enabled = os.getenv("ENABLE_SELF_CODING_ENGINE", "false").strip().lower() in ("true", "1", "yes")
        if not enabled:
            return {"status": "skipped", "reason": "self_coding_engine_disabled"}

        proposal_id = str(payload.get("proposal_id") or "").strip()
        if not proposal_id:
            return {"status": "skipped", "reason": "missing_proposal_id"}

        builder_task: dict[str, Any] = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "action": payload.get("action") or "implement_proposal",
            "proposal_id": proposal_id,
            "repo_path": payload.get("repo_path"),
            "github_repo": payload.get("github_repo"),
            "base_branch": payload.get("base_branch"),
            "test_command": payload.get("test_command"),
            "use_graph_context": bool(payload.get("use_graph_context", True)),
        }

        result = await self._executor.execute("SelfBuilder", builder_task)
        if isinstance(result, dict) and result.get("status") in {"error", "failed", "skipped", "blocked"}:
            return result
        return {"status": "completed", "result": result}

    async def _handle_revenue_prompt_compile(
        self,
        task_id: str,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Phase 2: Revenue reinforcement.

        Recompile the DSPy revenue optimizer using the latest REAL outcomes.
        """
        try:
            from optimization.revenue_prompt_optimizer import get_revenue_prompt_optimizer

            optimizer = get_revenue_prompt_optimizer()
        except Exception as exc:
            return {"status": "skipped", "reason": f"optimizer_import_failed:{exc}"}

        if not optimizer.enabled():
            return {"status": "skipped", "reason": "dspy_disabled_or_unavailable"}

        force = bool(payload.get("force", True))
        lead_id = payload.get("lead_id")
        reason = payload.get("reason")

        pool = get_pool()
        result = await optimizer.ensure_compiled(pool=pool, force=force)
        if not isinstance(result, dict):
            result = {"status": "error", "error": "unexpected_optimizer_result"}

        # Attach context for auditability in ai_task_queue.result.
        result["task_id"] = task_id
        result["tenant_id"] = tenant_id
        if lead_id:
            result["lead_id"] = lead_id
        if reason:
            result["reason"] = reason
        return result

    async def _handle_multi_agent_collaboration(
        self,
        task_id: str,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Dispatch multi-agent collaboration payloads into AgentExecutor."""
        if not self._executor:
            raise RuntimeError("AgentExecutor not available")
        if not ENABLE_AI_TASK_MULTI_AGENT_ROUTING:
            return {"status": "skipped", "reason": "multi_agent_routing_disabled"}

        primary_agent = payload.get("primary_agent") or payload.get("agent") or "WorkflowEngine"
        collaborative_task = {
            "task_id": task_id,
            "tenant_id": tenant_id,
            "action": payload.get("action") or "collaborative_task",
            "description": payload.get("description"),
            "multi_agent": True,
            "collaborative": True,
            "delegation_plan": payload.get("delegation_plan"),
            "delegate_to": payload.get("delegate_to"),
            "required_capabilities": payload.get("required_capabilities"),
            "aggregation_strategy": payload.get("aggregation_strategy", "merge_dict"),
            "execution_time_budget_seconds": payload.get("execution_time_budget_seconds"),
            "token_budget": payload.get("token_budget"),
            "_skip_langchain_runtime": True,
        }
        if isinstance(payload.get("task"), dict):
            collaborative_task.update(payload["task"])

        result = await self._executor.execute(primary_agent, collaborative_task)
        if isinstance(result, dict) and result.get("status") in {"failed", "error"}:
            return {"status": "failed", "error": result.get("error"), "result": result}
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
        _ai_task_queue_consumer_task = create_safe_task(consumer.start())
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
