import logging
import uuid
import json
import os
from typing import Any, Dict
from datetime import datetime
from database.async_connection import get_pool
from unified_brain import UnifiedBrain


class MemoryLossError(Exception):
    """Both persistence stores failed — execution data would be lost."""


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.type = agent_type
        self.logger = logging.getLogger(f"Agent.{name}")

    async def execute(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute agent task - override in subclasses"""
        raise NotImplementedError(f"Agent {self.name} must implement execute method")

    async def log_execution(self, task: dict, result: dict):
        """Log execution to database and Unified Brain.

        Raises MemoryLossError if BOTH stores fail so the caller's retry
        loop can attempt persistence again.
        """
        exec_id = str(uuid.uuid4())
        tenant_id = str(
            task.get("tenant_id")
            or os.getenv("DEFAULT_TENANT_ID")
            or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
        )

        legacy_ok = False
        brain_ok = False

        # 1. Log to legacy table (keep for backward compatibility)
        try:
            pool = get_pool()
            await pool.execute(
                """
                INSERT INTO agent_executions (
                    id, task_execution_id, agent_type, prompt,
                    response, status, created_at, completed_at, tenant_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), NOW(), $7::uuid)
            """,
                exec_id,
                exec_id,
                self.type,
                json.dumps(task),
                json.dumps(result),
                result.get("status", "completed"),
                tenant_id,
            )
            legacy_ok = True
        except Exception as e:
            self.logger.error(f"Legacy logging failed: {e}")

        # 2. Log to Unified Brain (New System)
        try:
            brain = UnifiedBrain(lazy_init=True)
            brain.store(
                key=f"exec_{exec_id}",
                value={
                    "task": task,
                    "result": result,
                    "agent": self.name,
                    "type": self.type,
                    "status": result.get("status", "completed"),
                },
                category="agent_execution",
                priority="medium" if result.get("status") == "completed" else "high",
                source=f"agent_{self.name}",
                metadata={"execution_id": exec_id, "timestamp": datetime.now().isoformat()},
            )
            brain_ok = True
        except Exception as e:
            self.logger.warning(f"Failed to store in UnifiedBrain: {e}")

        if not legacy_ok and not brain_ok:
            self.logger.critical(
                "EXECUTION_LOST: agent=%s exec=%s — both stores failed",
                self.name,
                exec_id,
            )
            raise MemoryLossError(f"Agent {self.name} exec {exec_id}: both stores failed")
