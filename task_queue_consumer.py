"""
Task Queue Consumer
====================
Consumes tasks from ai_autonomous_tasks table that were
previously inserted but never processed (the "dead queue" problem).

This bridges the gap between:
- agent_activation_system.py (which creates tasks)
- agent_executor.py (which executes agents)

Author: BrainOps AI System
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            return {
                'host': parsed.hostname or '',
                'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
                'user': parsed.username or '',
                'password': parsed.password or '',
                'port': int(str(parsed.port)) if parsed.port else 5432
            }
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        'host': os.getenv('DB_HOST'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
    }


class TaskQueueConsumer:
    """
    Consumes and executes tasks from the ai_autonomous_tasks queue.

    Fixes the "dead queue" problem where tasks are created by
    agent_activation_system but never consumed.
    """

    def __init__(self, poll_interval: int = 30, batch_size: int = 1):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self._running = False
        self._executor = None
        self._stats = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "tasks_skipped": 0,
            "last_poll": None
        }

    async def start(self):
        """Start the task queue consumer"""
        if self._running:
            logger.warning("TaskQueueConsumer already running")
            return

        self._running = True
        logger.info("🚀 TaskQueueConsumer started")

        # Lazy load executor
        try:
            from agent_executor import AgentExecutor
            self._executor = AgentExecutor()
            logger.info("✅ AgentExecutor loaded for task execution")
        except Exception as e:
            logger.error(f"Failed to load AgentExecutor: {e}")
            self._executor = None

        # Start consumption loop
        await self._consume_loop()

    async def stop(self):
        """Stop the task queue consumer"""
        self._running = False
        logger.info("🛑 TaskQueueConsumer stopped")

    async def _consume_loop(self):
        """Main consumption loop with advisory lock to prevent race conditions"""
        # Advisory lock key for ai_autonomous_tasks queue (unique identifier)
        ADVISORY_LOCK_KEY = 8675309001  # Unique key for this consumer

        while self._running:
            try:
                from database.async_connection import get_pool
                pool = get_pool()

                # Try to acquire advisory lock (non-blocking)
                async with pool.acquire() as conn:
                    lock_acquired = await conn.fetchval(
                        "SELECT pg_try_advisory_lock($1)",
                        ADVISORY_LOCK_KEY
                    )

                if not lock_acquired:
                    # Another instance is processing, wait and retry
                    logger.debug("TaskQueueConsumer: Another instance holds the lock, waiting...")
                    await asyncio.sleep(5)  # Short wait before retry
                    continue

                try:
                    await self._reconcile_stale_processing_tasks()
                    tasks = await self._fetch_pending_tasks()

                    if tasks:
                        logger.info(f"📥 Fetched {len(tasks)} pending tasks")
                        for task in tasks:
                            await self._process_task(task)

                    self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()
                finally:
                    # Release advisory lock
                    async with pool.acquire() as conn:
                        await conn.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_KEY)

            except Exception as e:
                logger.error(f"Error in consume loop: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _reconcile_stale_processing_tasks(self) -> None:
        """Requeue tasks stuck in processing for too long (self-healing)."""
        stale_after_seconds = int(os.getenv("TASK_QUEUE_STALE_AFTER_SECONDS", "600"))
        max_retries = int(os.getenv("TASK_QUEUE_MAX_RETRIES", "3"))

        if stale_after_seconds <= 0:
            return

        try:
            from database.async_connection import get_pool

            pool = get_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    stale_rows = await conn.fetch(
                        """
                        SELECT id, retry_count
                        FROM ai_autonomous_tasks
                        WHERE status = 'processing'
                          AND started_at IS NOT NULL
                          AND started_at < NOW() - ($1 * INTERVAL '1 second')
                        LIMIT 100
                        FOR UPDATE SKIP LOCKED
                        """,
                        stale_after_seconds,
                    )

                    if not stale_rows:
                        return

                    requeue_ids: list[str] = []
                    fail_ids: list[str] = []

                    for row in stale_rows:
                        task_id = str(row["id"])
                        retry_count = int(row.get("retry_count") or 0)
                        if retry_count >= max_retries:
                            fail_ids.append(task_id)
                        else:
                            requeue_ids.append(task_id)

                    requeued = "UPDATE 0"
                    failed = "UPDATE 0"

                    if requeue_ids:
                        requeued = await conn.execute(
                            """
                            UPDATE ai_autonomous_tasks
                            SET status = 'pending',
                                started_at = NULL,
                                retry_count = COALESCE(retry_count, 0) + 1,
                                error_log = COALESCE(error_log, '') || 'stale_processing_requeued\n'
                            WHERE id = ANY($1::uuid[])
                            """,
                            requeue_ids,
                        )

                    if fail_ids:
                        failed = await conn.execute(
                            """
                            UPDATE ai_autonomous_tasks
                            SET status = 'failed',
                                completed_at = NOW(),
                                error_log = COALESCE(error_log, '') || 'stale_processing_max_retries\n'
                            WHERE id = ANY($1::uuid[])
                            """,
                            fail_ids,
                        )

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
                    "🧹 Reconciled stale ai_autonomous_tasks processing tasks | requeued=%s failed=%s stale_after_seconds=%s max_retries=%s",
                    requeued_n,
                    failed_n,
                    stale_after_seconds,
                    max_retries,
                )

        except Exception as e:
            logger.error("Failed to reconcile stale tasks: %s", e)

    async def _fetch_pending_tasks(self) -> list[dict[str, Any]]:
        """Fetch pending tasks from database with row locking"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Lock and fetch pending tasks
                    rows = await conn.fetch(
                        """
                        SELECT
                            t.id,
                            t.task_type,
                            t.priority,
                            t.status,
                            t.trigger_type,
                            t.trigger_condition,
                            t.agent_id,
                            a.name AS agent_name,
                            t.created_at
                        FROM ai_autonomous_tasks t
                        LEFT JOIN ai_agents a ON a.id = t.agent_id
                        WHERE t.status = 'pending'
                        ORDER BY
                            CASE priority
                                WHEN 'critical' THEN 1
                                WHEN 'high' THEN 2
                                WHEN 'medium' THEN 3
                                WHEN 'low' THEN 4
                                ELSE 5
                            END,
                            created_at ASC
                        LIMIT $1
                        FOR UPDATE OF t SKIP LOCKED
                        """,
                        self.batch_size,
                    )

                    # Mark as processing
                    if rows:
                        task_ids = [str(row["id"]) for row in rows]
                        await conn.execute(
                            """
                            UPDATE ai_autonomous_tasks
                            SET status = 'processing',
                                started_at = NOW()
                            WHERE id = ANY($1::uuid[])
                            """,
                            task_ids,
                        )

                    return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to fetch tasks: {e}")
            return []

    async def _process_task(self, task: dict[str, Any]):
        """Process a single task"""
        task_id = task['id']
        agent_id = task.get('agent_id')
        agent_name = task.get('agent_name')
        task_type = task.get('task_type', 'unknown')

        logger.info(f"⚙️ Processing task {task_id}: {task_type}")

        try:
            trigger_condition = task.get("trigger_condition")
            if isinstance(trigger_condition, str):
                try:
                    trigger_condition = json.loads(trigger_condition)
                except json.JSONDecodeError:
                    trigger_condition = {"raw": trigger_condition}
            if not isinstance(trigger_condition, dict):
                trigger_condition = {"raw": str(trigger_condition)}

            # Execute the task via AgentExecutor (bounded by timeout for queue health)
            if self._executor and agent_name:
                timeout_seconds = float(os.getenv("TASK_QUEUE_EXECUTION_TIMEOUT_SECONDS", "60"))
                try:
                    result = await asyncio.wait_for(
                        self._executor.execute(
                            agent_name=str(agent_name),
                            task={
                                "task_id": str(task_id),
                                "agent_id": str(agent_id) if agent_id else None,
                                "action": task_type,
                                "task_type": task_type,
                                "trigger_type": task.get("trigger_type"),
                                "trigger_condition": trigger_condition,
                                # Production performance: ERP/unified_event handlers should not
                                # trigger codebase graph scans by default.
                                "use_graph_context": False,
                            },
                        ),
                        timeout=timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    await self._update_task_status(
                        task_id,
                        "failed",
                        {
                            "error": "timeout",
                            "timeout_seconds": timeout_seconds,
                            "agent_name": agent_name,
                            "task_type": task_type,
                        },
                    )
                    self._stats["tasks_failed"] += 1
                    logger.error("❌ Task %s timed out after %ss", task_id, timeout_seconds)
                    return

                # Mark as completed
                await self._update_task_status(task_id, 'completed', result)
                self._stats["tasks_processed"] += 1
                logger.info(f"✅ Task {task_id} completed")

            else:
                # No executor or agent - skip
                await self._update_task_status(
                    task_id,
                    'skipped',
                    {
                        "reason": "missing_agent_name_or_executor",
                        "agent_id": str(agent_id) if agent_id else None,
                        "agent_name": agent_name,
                    },
                )
                self._stats["tasks_skipped"] += 1
                logger.warning(f"⏭️ Task {task_id} skipped - no executor")

        except Exception as e:
            # Mark as failed
            await self._update_task_status(task_id, 'failed', {"error": str(e)})
            self._stats["tasks_failed"] += 1
            logger.error(f"❌ Task {task_id} failed: {e}")

    async def _update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[dict[str, Any]] = None
    ):
        """Update task status in database"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE ai_autonomous_tasks
                    SET status = $1::text,
                        completed_at = CASE
                            WHEN $1::text IN ('completed', 'failed', 'skipped') THEN NOW()
                            ELSE completed_at
                        END,
                        result = $2::jsonb,
                        error_log = CASE
                            WHEN $1::text = 'failed' THEN COALESCE(($2::jsonb ->> 'error'), error_log)
                            ELSE error_log
                        END
                    WHERE id = $3::uuid
                """, status, json.dumps(result, default=str) if result else None, task_id)

        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get consumer statistics"""
        return {
            **self._stats,
            "running": self._running,
            "poll_interval": self.poll_interval,
            "batch_size": self.batch_size
        }


# Singleton instance
_consumer: Optional[TaskQueueConsumer] = None
_consumer_task: Optional[asyncio.Task] = None


def get_task_queue_consumer() -> TaskQueueConsumer:
    """Get or create the TaskQueueConsumer singleton"""
    global _consumer
    if _consumer is None:
        poll_interval = int(os.getenv("TASK_QUEUE_POLL_INTERVAL", "30"))
        batch_size = int(os.getenv("TASK_QUEUE_BATCH_SIZE", "1"))
        _consumer = TaskQueueConsumer(poll_interval=poll_interval, batch_size=batch_size)
    return _consumer


async def start_task_queue_consumer():
    """Start the task queue consumer as a background task"""
    global _consumer_task
    consumer = get_task_queue_consumer()

    if _consumer_task is None or _consumer_task.done():
        _consumer_task = asyncio.create_task(consumer.start())
        logger.info("📋 Task queue consumer started as background task")

    return consumer


async def stop_task_queue_consumer():
    """Stop the task queue consumer"""
    global _consumer_task
    consumer = get_task_queue_consumer()
    await consumer.stop()

    if _consumer_task:
        _consumer_task.cancel()
        try:
            await _consumer_task
        except asyncio.CancelledError:
            logger.debug("Task queue consumer task cancelled")
        _consumer_task = None
