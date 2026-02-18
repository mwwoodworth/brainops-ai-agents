#!/usr/bin/env python3
"""
LangChain enhancement runtime for BrainOps AI Agents.

All features are feature-flagged and default to OFF.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from database.async_connection import get_tenant_pool

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


ENABLE_LANGCHAIN_TOOL_ABSTRACTION = _env_flag("ENABLE_LANGCHAIN_TOOL_ABSTRACTION", False)
ENABLE_LANGCHAIN_CHAIN_COMPOSITION = _env_flag("ENABLE_LANGCHAIN_CHAIN_COMPOSITION", False)
ENABLE_LANGCHAIN_MEMORY_INTEGRATION = _env_flag("ENABLE_LANGCHAIN_MEMORY_INTEGRATION", False)
ENABLE_LANGCHAIN_STRUCTURED_OUTPUT = _env_flag("ENABLE_LANGCHAIN_STRUCTURED_OUTPUT", False)
ENABLE_LANGCHAIN_CALLBACK_LOGGING = _env_flag("ENABLE_LANGCHAIN_CALLBACK_LOGGING", False)


def _resolve_tenant_id(candidate: Optional[str]) -> str:
    """Resolve tenant id for scoped operations."""
    tenant_id = (candidate or "").strip()
    return tenant_id or DEFAULT_TENANT_ID


try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - optional dependency
    BaseCallbackHandler = object

try:
    from langchain_core.tools import StructuredTool
except Exception:  # pragma: no cover - optional dependency
    StructuredTool = None


class AgentToolInput(BaseModel):
    """Input schema for BrainOps agent tools."""

    task: dict[str, Any] = Field(default_factory=dict)


class UnifiedBrainLogCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that writes chain telemetry to unified_brain_logs."""

    def __init__(self, tenant_id: str = DEFAULT_TENANT_ID, system_name: str = "langchain_runtime"):
        self.tenant_id = _resolve_tenant_id(tenant_id)
        self.system_name = system_name

    async def _log_event(self, action: str, payload: dict[str, Any]) -> None:
        """Persist callback event into unified_brain_logs (best-effort)."""
        try:
            tenant_pool = get_tenant_pool(self.tenant_id)
            conn = await tenant_pool.acquire()
            try:
                await conn.execute(
                    """
                    INSERT INTO unified_brain_logs (system, action, data, created_at)
                    VALUES ($1, $2, $3::jsonb, NOW())
                    """,
                    self.system_name,
                    action,
                    json.dumps(payload, default=str),
                )
            finally:
                await tenant_pool.release(conn)
        except Exception as exc:  # pragma: no cover - logging must never break runtime
            logger.warning("LangChain callback log failed (%s): %s", action, exc)

    def _schedule_log(self, action: str, payload: dict[str, Any]) -> None:
        """Schedule callback logging without blocking caller execution."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._log_event(action, payload))
        except RuntimeError:
            logger.debug("No running event loop for callback logging: %s", action)

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Handle chain-start events."""
        if not ENABLE_LANGCHAIN_CALLBACK_LOGGING:
            return
        self._schedule_log(
            "chain_start",
            {
                "serialized": serialized,
                "inputs": inputs,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Handle chain-end events."""
        if not ENABLE_LANGCHAIN_CALLBACK_LOGGING:
            return
        self._schedule_log(
            "chain_end",
            {"outputs": outputs, "timestamp": datetime.now(timezone.utc).isoformat()},
        )

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        """Handle chain-error events."""
        if not ENABLE_LANGCHAIN_CALLBACK_LOGGING:
            return
        self._schedule_log(
            "chain_error",
            {"error": str(error), "timestamp": datetime.now(timezone.utc).isoformat()},
        )


class UnifiedAIMemoryBridge:
    """Bridge LangChain-style memory operations to the unified_ai_memory table."""

    def __init__(self, tenant_id: str = DEFAULT_TENANT_ID):
        self.tenant_id = _resolve_tenant_id(tenant_id)

    async def load_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Load contextual memories relevant to a query."""
        if not ENABLE_LANGCHAIN_MEMORY_INTEGRATION:
            return []

        try:
            tenant_pool = get_tenant_pool(self.tenant_id)
            conn = await tenant_pool.acquire()
            try:
                rows = await conn.fetch(
                    """
                    SELECT id, content, metadata, created_at
                    FROM unified_ai_memory
                    WHERE tenant_id = $1
                      AND (search_text ILIKE $2 OR content::text ILIKE $2)
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    self.tenant_id,
                    f"%{query}%",
                    max(1, int(limit)),
                )
            finally:
                await tenant_pool.release(conn)
        except Exception as exc:
            logger.warning("Memory load failed: %s", exc)
            return []

        memory_rows: list[dict[str, Any]] = []
        for row in rows:
            content_value = row["content"]
            metadata_value = row["metadata"]
            parsed_content = (
                json.loads(content_value) if isinstance(content_value, str) else (content_value or {})
            )
            parsed_metadata = (
                json.loads(metadata_value)
                if isinstance(metadata_value, str)
                else (metadata_value or {})
            )
            memory_rows.append(
                {
                    "id": str(row["id"]),
                    "content": parsed_content,
                    "metadata": parsed_metadata,
                    "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                }
            )
        return memory_rows

    async def save_exchange(
        self,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[str]:
        """Persist chain execution exchange as episodic memory."""
        if not ENABLE_LANGCHAIN_MEMORY_INTEGRATION:
            return None

        memory_metadata = metadata or {}
        content = {
            "input": input_payload,
            "output": output_payload,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        search_text = (
            f"{input_payload.get('action', '')} "
            f"{input_payload.get('description', '')} "
            f"{output_payload.get('status', '')}"
        ).strip()

        try:
            tenant_pool = get_tenant_pool(self.tenant_id)
            conn = await tenant_pool.acquire()
            try:
                row = await conn.fetchrow(
                    """
                    INSERT INTO unified_ai_memory (
                        memory_type, content, source_system, source_agent,
                        created_by, metadata, search_text, tenant_id
                    )
                    VALUES ($1, $2::jsonb, $3, $4, $5, $6::jsonb, $7, $8)
                    RETURNING id
                    """,
                    "episodic",
                    json.dumps(content, default=str),
                    "langchain_runtime",
                    memory_metadata.get("source_agent", "LangChain"),
                    "langchain_runtime",
                    json.dumps(memory_metadata, default=str),
                    search_text,
                    self.tenant_id,
                )
            finally:
                await tenant_pool.release(conn)
            if row and row["id"]:
                return str(row["id"])
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)
        return None


class StructuredOutputValidator:
    """Validate and normalize structured LLM output payloads."""

    @staticmethod
    def parse_output(raw_output: Any, schema_model: type[BaseModel]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        """Parse model output using a required Pydantic schema."""
        try:
            payload = raw_output
            if isinstance(raw_output, str):
                payload = json.loads(raw_output)
            validated = schema_model.model_validate(payload)
            return validated.model_dump(), None
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return None, str(exc)


class BrainOpsToolRegistry:
    """Tool abstraction layer that exposes BrainOps agents as LangChain tools."""

    def __init__(self, executor: Any):
        self.executor = executor

    def _get_agent_names(self) -> list[str]:
        """Return available agent names from the executor."""
        if hasattr(self.executor, "agents") and isinstance(self.executor.agents, dict):
            return sorted(self.executor.agents.keys())
        return []

    def get_tools(self, include_agents: Optional[list[str]] = None) -> list[Any]:
        """Build tool wrappers for each selected agent."""
        if not ENABLE_LANGCHAIN_TOOL_ABSTRACTION:
            return []

        tools: list[Any] = []
        selected = include_agents or self._get_agent_names()

        for agent_name in selected:
            tool_name = f"agent_{agent_name.lower()}"

            async def _invoke_agent_tool(task: dict[str, Any], _agent_name: str = agent_name) -> dict[str, Any]:
                payload = dict(task or {})
                payload["_skip_langchain_runtime"] = True
                return await self.executor.execute(_agent_name, payload)

            if StructuredTool is not None:
                try:
                    tool = StructuredTool.from_function(
                        name=tool_name,
                        description=f"Execute BrainOps agent '{agent_name}' with task payload",
                        args_schema=AgentToolInput,
                        func=lambda task, _tool=tool_name: {
                            "status": "error",
                            "error": f"Synchronous execution not supported for {_tool}",
                        },
                        coroutine=_invoke_agent_tool,
                    )
                    tools.append(tool)
                    continue
                except Exception as exc:
                    logger.warning("Failed to build StructuredTool for %s: %s", agent_name, exc)

            tools.append(
                {
                    "name": tool_name,
                    "description": f"Execute BrainOps agent '{agent_name}'",
                    "coroutine": _invoke_agent_tool,
                    "agent_name": agent_name,
                }
            )

        return tools


class BrainOpsChainFactory:
    """Reusable chain composition patterns for common BrainOps flows."""

    def __init__(self, executor: Any, tenant_id: str = DEFAULT_TENANT_ID):
        self.executor = executor
        self.tenant_id = _resolve_tenant_id(tenant_id)
        self.memory_bridge = UnifiedAIMemoryBridge(self.tenant_id)
        self.callback_handler = UnifiedBrainLogCallbackHandler(self.tenant_id)

    async def run_research_analyze_act(
        self,
        task: dict[str, Any],
        default_agent: str,
    ) -> dict[str, Any]:
        """Run the reusable research -> analyze -> act chain."""
        if not ENABLE_LANGCHAIN_CHAIN_COMPOSITION:
            return {"status": "skipped", "reason": "langchain_chain_composition_disabled"}

        chain_inputs = {
            "task_id": task.get("task_id"),
            "default_agent": default_agent,
            "action": task.get("action"),
        }
        self.callback_handler.on_chain_start({"name": "research_analyze_act"}, chain_inputs)

        try:
            query = (
                task.get("description")
                or task.get("prompt")
                or task.get("action")
                or "brainops_chain_task"
            )

            memory_context = await self.memory_bridge.load_context(query=query, limit=5)

            research_agent = task.get("research_agent") or "Knowledge"
            analyze_agent = task.get("analysis_agent") or "PredictiveAnalyzer"
            action_agent = task.get("action_agent") or default_agent

            research_task = {
                **task,
                "action": task.get("research_action", "research"),
                "query": query,
                "memory_context": memory_context,
                "_skip_langchain_runtime": True,
            }
            research_result = await self.executor.execute(research_agent, research_task)

            analyze_task = {
                **task,
                "action": task.get("analysis_action", "analyze"),
                "research_result": research_result,
                "memory_context": memory_context,
                "_skip_langchain_runtime": True,
            }
            analysis_result = await self.executor.execute(analyze_agent, analyze_task)

            act_task = {
                **task,
                "action": task.get("action_action", task.get("action", "act")),
                "analysis_result": analysis_result,
                "research_result": research_result,
                "_skip_langchain_runtime": True,
            }
            action_result = await self.executor.execute(action_agent, act_task)

            structured_result: Optional[dict[str, Any]] = None
            structured_error: Optional[str] = None
            output_schema = task.get("output_schema_model")
            if ENABLE_LANGCHAIN_STRUCTURED_OUTPUT and output_schema and isinstance(output_schema, type):
                structured_result, structured_error = StructuredOutputValidator.parse_output(
                    raw_output=action_result,
                    schema_model=output_schema,
                )

            combined = {
                "status": "completed",
                "pattern": "research_analyze_act",
                "agents": {
                    "research_agent": research_agent,
                    "analysis_agent": analyze_agent,
                    "action_agent": action_agent,
                },
                "result": {
                    "research": research_result,
                    "analysis": analysis_result,
                    "action": action_result,
                    "structured_output": structured_result,
                    "structured_output_error": structured_error,
                },
                "memory_context_count": len(memory_context),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            await self.memory_bridge.save_exchange(
                input_payload=chain_inputs,
                output_payload=combined,
                metadata={"source_agent": action_agent, "pattern": "research_analyze_act"},
            )

            self.callback_handler.on_chain_end(combined)
            return combined

        except Exception as exc:
            self.callback_handler.on_chain_error(exc)
            return {
                "status": "failed",
                "pattern": "research_analyze_act",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def execute_pattern(
        self,
        pattern: str,
        task: dict[str, Any],
        default_agent: str,
    ) -> dict[str, Any]:
        """Execute a registered chain pattern."""
        normalized = (pattern or "").strip().lower()
        if normalized in {"research_analyze_act", "research->analyze->act"}:
            return await self.run_research_analyze_act(task=task, default_agent=default_agent)
        return {"status": "skipped", "reason": f"unsupported_chain_pattern:{pattern}"}


class LangChainEnhancementRuntime:
    """Facade for LangChain enhancements used by AgentExecutor."""

    def __init__(self, executor: Any):
        self.executor = executor
        self._tool_registry = BrainOpsToolRegistry(executor)

    def enabled(self) -> bool:
        """Return True when any LangChain enhancement flag is enabled."""
        return any(
            (
                ENABLE_LANGCHAIN_TOOL_ABSTRACTION,
                ENABLE_LANGCHAIN_CHAIN_COMPOSITION,
                ENABLE_LANGCHAIN_MEMORY_INTEGRATION,
                ENABLE_LANGCHAIN_STRUCTURED_OUTPUT,
                ENABLE_LANGCHAIN_CALLBACK_LOGGING,
            )
        )

    def get_tools(self, include_agents: Optional[list[str]] = None) -> list[Any]:
        """Return LangChain tool wrappers for selected agents."""
        return self._tool_registry.get_tools(include_agents=include_agents)

    async def run_chain_pattern(
        self,
        pattern: str,
        task: dict[str, Any],
        default_agent: str,
    ) -> dict[str, Any]:
        """Execute a named LangChain composition pattern."""
        tenant_id = _resolve_tenant_id(task.get("tenant_id") if isinstance(task, dict) else None)
        chain_factory = BrainOpsChainFactory(self.executor, tenant_id=tenant_id)
        return await chain_factory.execute_pattern(pattern=pattern, task=task, default_agent=default_agent)
