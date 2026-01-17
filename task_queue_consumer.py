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

    def __init__(self, poll_interval: int = 30, batch_size: int = 10):
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
        """Main consumption loop"""
        while self._running:
            try:
                tasks = await self._fetch_pending_tasks()

                if tasks:
                    logger.info(f"📥 Fetched {len(tasks)} pending tasks")
                    for task in tasks:
                        await self._process_task(task)

                self._stats["last_poll"] = datetime.now(timezone.utc).isoformat()

            except Exception as e:
                logger.error(f"Error in consume loop: {e}")

            await asyncio.sleep(self.poll_interval)

    async def _fetch_pending_tasks(self) -> list[dict[str, Any]]:
        """Fetch pending tasks from database with row locking"""
        try:
            from database.async_connection import get_pool
            pool = get_pool()

            async with pool.acquire() as conn:
                # Lock and fetch pending tasks
                rows = await conn.fetch("""
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
                    FOR UPDATE SKIP LOCKED
                """, self.batch_size)

                # Mark as processing
                if rows:
                    task_ids = [row['id'] for row in rows]
                    await conn.execute("""
                        UPDATE ai_autonomous_tasks
                        SET status = 'processing',
                            started_at = NOW()
                        WHERE id = ANY($1)
                    """, task_ids)

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

            # Execute the task via AgentExecutor
            if self._executor and agent_name:
                result = await self._executor.execute(
                    agent_name=str(agent_name),
                    task={
                        "task_id": str(task_id),
                        "agent_id": str(agent_id) if agent_id else None,
                        "action": task_type,
                        "task_type": task_type,
                        "trigger_type": task.get("trigger_type"),
                        "trigger_condition": trigger_condition,
                    },
                )

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
                    SET status = $1,
                        completed_at = CASE WHEN $1 IN ('completed', 'failed', 'skipped') THEN NOW() ELSE completed_at END,
                        result = $2
                    WHERE id = $3
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
        _consumer = TaskQueueConsumer()
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
