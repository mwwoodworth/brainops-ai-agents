"""
Agent CRUD, execution, dispatch, and health endpoints.

Extracted from app.py during Phase 2 Wave 2B.
Covers agent listing, individual agent ops, execution (single, generic, scheduled, v1),
health monitoring, and AUREA event dispatch.
"""

import asyncio
import inspect
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from config import config
from database.async_connection import DatabaseUnavailableError, get_pool
from models.agent import Agent, AgentExecution, AgentList
from observability import TTLCache
from services.agent_helpers import row_to_agent

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])


# ---------------------------------------------------------------------------
# Pydantic request models (moved from app.py inline definitions)
# ---------------------------------------------------------------------------
class ProductRequest(BaseModel):
    concept: str


class AgentExecuteRequest(BaseModel):
    """Request payload for executing an agent via v1 API."""

    agent_id: Optional[str] = None
    id: Optional[str] = None
    payload: dict[str, Any] = {}


class AgentActivateRequest(BaseModel):
    """Request payload for activating or deactivating an agent."""

    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    enabled: bool = True


class AUREAEventRequest(BaseModel):
    """Request model for AUREA event execution"""

    event_id: str
    topic: str
    source: str
    payload: dict[str, Any]
    target_agent: dict[str, Any]  # {name, role, capabilities}
    routing_metadata: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Module-level cache for agents listing
# ---------------------------------------------------------------------------
_AGENTS_CACHE = TTLCache(max_size=64)
_CACHE_TTLS = {"agents": 30.0}


# ---------------------------------------------------------------------------
# Lazy accessors for app-level singletons
# ---------------------------------------------------------------------------
def _get_app():
    import app as _app

    return _app.app


def _get_limiter():
    import app as _app

    return _app.limiter


def _get_local_executions():
    import app as _app

    return _app.LOCAL_EXECUTIONS


def _get_response_cache():
    import app as _app

    return _app.RESPONSE_CACHE


def _agents_available():
    import app as _app

    return getattr(_app, "AGENTS_AVAILABLE", False)


def _get_agent_executor():
    import app as _app

    return getattr(_app, "AGENT_EXECUTOR", None)


def _health_monitor_available():
    import app as _app

    return getattr(_app, "HEALTH_MONITOR_AVAILABLE", False)


def _scheduler_available():
    import app as _app

    return getattr(_app, "SCHEDULER_AVAILABLE", False)


def _integration_layer_available():
    import app as _app

    return getattr(_app, "INTEGRATION_LAYER_AVAILABLE", False)


def _ai_available():
    import app as _app

    return getattr(_app, "AI_AVAILABLE", False)


def _get_ai_core():
    import app as _app

    return getattr(_app, "ai_core", None)


def _safe_json_dumps(obj, **kwargs):
    import app as _app

    return _app.safe_json_dumps(obj, **kwargs)


def _resolve_tenant(request):
    from services.tenant_helpers import resolve_tenant_uuid_from_request

    return resolve_tenant_uuid_from_request(request)


def _product_agent_available():
    import app as _app

    return getattr(_app, "PRODUCT_AGENT_AVAILABLE", False)


def _get_product_agent_graph():
    import app as _app

    return getattr(_app, "product_agent_graph", None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/agents/product/run", tags=["Agents"])
async def run_product_agent(request: ProductRequest):
    """
    Run the LangGraph Product Agent to generate product specs, code, and QA.
    """
    if not _product_agent_available():
        raise HTTPException(
            status_code=503,
            detail="Product Agent is not available - required dependencies not installed",
        )
    try:
        from langchain_core.messages import HumanMessage

        graph = _get_product_agent_graph()
        result = graph.invoke({"messages": [HumanMessage(content=request.concept)]})
        last_message = result["messages"][-1].content
        return {
            "status": "success",
            "result": last_message,
            "trace": [m.content for m in result["messages"]],
        }
    except Exception as e:
        logger.error(f"Product Agent Failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents", response_model=AgentList)
async def get_agents(
    category: Optional[str] = None,
    enabled: Optional[bool] = True,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_capabilities: bool = Query(True),
    include_configuration: bool = Query(False),
) -> AgentList:
    """Get list of available agents"""
    try:
        cache_key = (
            f"agents:{category or 'all'}:{enabled}:"
            f"{limit}:{offset}:{int(include_capabilities)}:{int(include_configuration)}"
        )

        async def _load_agents() -> AgentList:
            try:
                pool = get_pool()

                params: list[Any] = []
                where_sql = "WHERE 1=1"

                if enabled is not None:
                    where_sql += f" AND a.enabled = ${len(params) + 1}"
                    params.append(enabled)

                if category:
                    where_sql += f" AND a.category = ${len(params) + 1}"
                    params.append(category)

                select_cols = [
                    "a.id",
                    "a.name",
                    "a.category",
                    "a.enabled",
                    "a.status",
                    "a.type",
                    "a.created_at",
                    "a.updated_at",
                    ("a.capabilities" if include_capabilities else "'[]'::jsonb AS capabilities"),
                    (
                        "a.configuration"
                        if include_configuration
                        else "'{}'::jsonb AS configuration"
                    ),
                ]
                select_sql = ", ".join(select_cols)

                total = (
                    await pool.fetchval(f"SELECT COUNT(*) FROM agents a {where_sql}", *params) or 0
                )

                query = f"""
                    SELECT {select_sql}
                    FROM agents a
                    {where_sql}
                    ORDER BY a.category, a.name
                    LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """

                rows = await pool.fetch(query, *params, limit, offset)

                exec_stats_by_agent_name: dict[str, dict[str, Any]] = {}
                try:
                    agent_names = [
                        (row.get("name") if isinstance(row, dict) else dict(row).get("name"))
                        for row in rows
                    ]
                    agent_names = [name for name in agent_names if isinstance(name, str) and name]
                    if agent_names:
                        exec_rows = await pool.fetch(
                            """
                            SELECT agent_name,
                                   COUNT(*) as exec_count,
                                   MAX(created_at) as last_exec
                            FROM ai_agent_executions
                            WHERE agent_name = ANY($1::text[])
                            GROUP BY agent_name
                            """,
                            agent_names,
                        )
                        exec_stats_by_agent_name = {
                            str(r.get("agent_name")): (r if isinstance(r, dict) else dict(r))
                            for r in exec_rows
                            if (
                                r.get("agent_name")
                                if isinstance(r, dict)
                                else dict(r).get("agent_name")
                            )
                        }
                except Exception as stats_error:
                    logger.warning("Failed to load agent execution stats: %s", stats_error)

                agents: list[Agent] = []
                for row in rows:
                    data = row if isinstance(row, dict) else dict(row)
                    stats = exec_stats_by_agent_name.get(str(data.get("name") or ""), {})
                    data["total_executions"] = int(stats.get("exec_count") or 0)
                    data["last_active"] = stats.get("last_exec") or None
                    agents.append(row_to_agent(data))

                return AgentList(
                    agents=agents,
                    total=int(total),
                    page=(offset // limit) + 1,
                    page_size=limit,
                )
            except DatabaseUnavailableError as exc:
                logger.error("Database unavailable while loading agents", exc_info=True)
                raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
            except Exception as e:
                logger.error(f"Failed to get agents from database: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to load agents") from e

        response_cache = _get_response_cache()
        agents_response, _ = await response_cache.get_or_set(
            cache_key, _CACHE_TTLS["agents"], _load_agents
        )
        return agents_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agents (outer): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agents") from e


@router.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, request: Request, authenticated: bool = True):
    """Execute an agent"""
    pool = get_pool()
    tenant_uuid = (
        _resolve_tenant(request)
        or config.tenant.default_tenant_id
        or os.getenv("DEFAULT_TENANT_ID")
        or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
    )

    try:
        # Resolve agent aliases before database lookup (ERP compatibility)
        resolved_agent_id = agent_id
        agent_executor = _get_agent_executor()
        if _agents_available() and agent_executor and hasattr(agent_executor, "AGENT_ALIASES"):
            if agent_id in agent_executor.AGENT_ALIASES:
                resolved_agent_id = agent_executor.AGENT_ALIASES[agent_id]
                logger.info(f"Resolved agent alias: {agent_id} -> {resolved_agent_id}")

        # Get agent by UUID (text comparison) or legacy slug
        agent = await pool.fetchrow(
            """SELECT id, name, type, enabled, description, capabilities, configuration,
                      schedule_hours, created_at, updated_at
               FROM agents WHERE id::text = $1 OR name = $1""",
            resolved_agent_id,
        )
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found (resolved: {resolved_agent_id})",
            )

        if not agent["enabled"]:
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is disabled")

        # Get request body
        body = await request.json()

        # Generate execution ID
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()

        # Log execution start
        agent_uuid = str(agent["id"])
        agent_name = agent["name"]
        try:
            await pool.execute(
                """
                INSERT INTO ai_agent_executions
                    (id, agent_name, task_type, input_data, status, tenant_id)
                VALUES ($1, $2, $3, $4, $5, $6::uuid)
            """,
                execution_id,
                agent_name,
                "execute",
                json.dumps(body),
                "running",
                tenant_uuid,
            )
            logger.info(f"Logged execution start for {agent_name}: {execution_id}")
        except Exception as insert_error:
            logger.warning("Failed to persist execution start: %s", insert_error)

        # Execute agent logic
        result = {"status": "completed", "message": "Agent executed successfully"}
        task = body.get("task", {})
        exec_task = task
        if isinstance(exec_task, dict):
            exec_task = dict(exec_task)
            exec_task["_skip_ai_agent_log"] = True
        else:
            exec_task = {"task": exec_task, "_skip_ai_agent_log": True}

        if _agents_available() and agent_executor:
            try:
                agent_result = await agent_executor.execute(agent_name, exec_task)
                result = (
                    agent_result
                    if isinstance(agent_result, dict)
                    else {"status": "completed", "result": agent_result}
                )
                result["agent_executed"] = True
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                result["status"] = "error"
                result["error"] = str(e)
                result["agent_executed"] = False
        elif _ai_available() and _get_ai_core():
            try:
                ai_core = _get_ai_core()
                prompt = f"Execute {agent['name']}: {task}"
                if inspect.iscoroutinefunction(ai_core.generate):
                    ai_result = await ai_core.generate(prompt)
                else:
                    ai_result = await asyncio.to_thread(ai_core.generate, prompt)
                result["ai_response"] = ai_result
                result["agent_executed"] = False
            except Exception as e:
                logger.error(f"AI execution failed: {e}")
                result["ai_response"] = None

        # Update execution record
        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        try:
            await pool.execute(
                """
                UPDATE ai_agent_executions
                SET status = $1, output_data = $2, execution_time_ms = $3
                WHERE id = $4
            """,
                "completed",
                _safe_json_dumps(result),
                duration_ms,
                execution_id,
            )
        except Exception as update_error:
            logger.warning("Failed to persist execution completion: %s", update_error)

        local_record = {
            "execution_id": execution_id,
            "agent_id": agent_uuid,
            "agent_name": agent["name"],
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "error": None,
        }
        _get_local_executions().appendleft(local_record)

        return AgentExecution(
            agent_id=agent_uuid,
            agent_name=agent["name"],
            execution_id=execution_id,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            input_data=body,
            output_data=result,
            duration_ms=duration_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        if "execution_id" in locals():
            try:
                await pool.execute(
                    """
                    UPDATE ai_agent_executions
                    SET status = $1, error_message = $2
                    WHERE id = $3
                """,
                    "failed",
                    str(e),
                    execution_id,
                )
            except Exception as fail_error:
                logger.warning("Failed to persist failed execution: %s", fail_error)

            _get_local_executions().appendleft(
                {
                    "execution_id": execution_id,
                    "agent_id": agent_uuid if "agent_uuid" in locals() else agent_id,
                    "agent_name": agent["name"] if "agent" in locals() else agent_id,
                    "status": "failed",
                    "started_at": locals().get("started_at"),
                    "completed_at": datetime.utcnow(),
                    "duration_ms": None,
                    "error": str(e),
                }
            )

        logger.error("Internal server error: %s", fail_error)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/agents/status")
async def get_all_agents_status():
    """
    Get comprehensive status of all agents including health metrics,
    execution statistics, and current state
    """
    if not _health_monitor_available():
        pool = get_pool()
        try:
            result = await pool.fetch(
                """
                SELECT
                    a.id,
                    a.name,
                    a.type,
                    a.status,
                    a.last_active,
                    a.total_executions,
                    s.enabled as scheduled,
                    s.frequency_minutes,
                    s.last_execution,
                    s.next_execution
                FROM ai_agents a
                LEFT JOIN agent_schedules s ON s.agent_id = a.id
                ORDER BY a.name
            """
            )

            agents = [dict(row) for row in result]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "health_monitoring": False,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error("Internal server error: %s", e)
            raise HTTPException(status_code=500, detail="Internal server error") from e

    try:
        from agent_health_monitor import get_health_monitor

        health_monitor = get_health_monitor()
        health_summary = health_monitor.check_all_agents_health()
        detailed_summary = health_monitor.get_agent_health_summary()

        total_agents = health_summary.get("total_agents", 0)
        agents_list = health_summary.get("agents", [])
        if total_agents == 0:
            try:
                pool = get_pool()
                db_agents = await pool.fetch(
                    "SELECT id, name, type, status FROM ai_agents ORDER BY name"
                )
                total_agents = len(db_agents)
                agents_list = [dict(row) for row in db_agents]
            except Exception:
                pass

        return {
            "total_agents": total_agents,
            "health_summary": {
                "healthy": health_summary.get("healthy", 0),
                "degraded": health_summary.get("degraded", 0),
                "critical": health_summary.get("critical", 0),
                "unknown": health_summary.get("unknown", 0),
            },
            "agents": agents_list,
            "critical_agents": detailed_summary.get("critical_agents", []),
            "active_alerts": detailed_summary.get("active_alerts", []),
            "recent_restarts": detailed_summary.get("recent_restarts", []),
            "health_monitoring": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> Agent:
    """Get a specific agent"""
    pool = get_pool()

    try:
        agent = await pool.fetchrow(
            "SELECT * FROM agents WHERE id::text = $1 OR name = $1",
            agent_id,
        )
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        data = agent if isinstance(agent, dict) else dict(agent)
        return row_to_agent(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    limit: int = Query(50, ge=1, le=500),
):
    """Get execution history for a specific agent."""
    pool = get_pool()

    try:
        agent = await pool.fetchrow(
            "SELECT id, name FROM agents WHERE id::text = $1 OR name = $1",
            agent_id,
        )
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent_name = agent["name"]
        rows = await pool.fetch(
            """
            SELECT id, status, task_type, input_data, output_data, error_message,
                   execution_time_ms, created_at
            FROM ai_agent_executions
            WHERE agent_name = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            agent_name,
            limit,
        )

        history = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            history.append(
                {
                    "execution_id": str(data.get("id")),
                    "status": data.get("status"),
                    "task_type": data.get("task_type"),
                    "input_data": data.get("input_data"),
                    "output_data": data.get("output_data"),
                    "error": data.get("error_message"),
                    "duration_ms": data.get("execution_time_ms"),
                    "created_at": data["created_at"].isoformat()
                    if data.get("created_at")
                    else None,
                }
            )

        return {
            "agent_id": str(agent["id"]),
            "agent_name": agent_name,
            "history": history,
            "count": len(history),
        }
    except HTTPException:
        raise
    except DatabaseUnavailableError as exc:
        logger.error("Database unavailable while loading agent history", exc_info=True)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable") from exc
    except Exception as e:
        logger.error("Failed to get agent history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent history") from e


@router.post("/execute")
async def execute_scheduled_agents(request: Request):
    """Execute scheduled agents (called by cron)"""
    app = _get_app()
    if not _scheduler_available() or not app.state.scheduler:
        return {"status": "scheduler_disabled", "message": "Agent scheduler not available"}

    try:
        pool = get_pool()
        tenant_uuid = (
            _resolve_tenant(request)
            or config.tenant.default_tenant_id
            or os.getenv("DEFAULT_TENANT_ID")
            or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
        )

        current_hour = datetime.utcnow().hour

        agents = await pool.fetch(
            """
            SELECT id, name, type, enabled, description, capabilities, configuration,
                   schedule_hours, created_at, updated_at
            FROM agents
            WHERE enabled = true
            AND schedule_hours @> ARRAY[$1]::integer[]
            LIMIT 50
        """,
            current_hour,
        )

        hour_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        already_ran_rows = await pool.fetch(
            """
            SELECT DISTINCT agent_name
            FROM ai_agent_executions
            WHERE task_type = 'scheduled_run'
              AND created_at >= $1
            """,
            hour_start,
        )
        already_ran_names = {
            (r.get("agent_name") if isinstance(r, dict) else dict(r).get("agent_name"))
            for r in already_ran_rows
        }
        already_ran_names = {name for name in already_ran_names if isinstance(name, str) and name}

        agent_executor = _get_agent_executor()
        results = []
        for agent in agents:
            try:
                execution_id = str(uuid.uuid4())
                agent_name = agent.get("name", "unknown")

                if agent_name in already_ran_names:
                    results.append(
                        {
                            "agent_id": str(agent.get("id")),
                            "agent_name": agent_name,
                            "execution_id": None,
                            "status": "skipped",
                            "reason": "already_ran_this_hour",
                        }
                    )
                    continue

                await pool.execute(
                    """
                    INSERT INTO agent_executions (id, agent_type, status, prompt, tenant_id)
                    VALUES ($1, $2, $3, $4, $5::uuid)
                """,
                    execution_id,
                    agent.get("type", "scheduled"),
                    "running",
                    json.dumps({"scheduled": True, "agent_name": agent_name}),
                    tenant_uuid,
                )

                result = {"status": "skipped", "message": "No executor available"}
                if agent_executor:
                    try:
                        task_data = {
                            "action": "scheduled_run",
                            "agent_id": agent["id"],
                            "scheduled": True,
                            "execution_id": execution_id,
                            "context": agent.get("configuration") or {},
                        }
                        timeout_env = os.getenv("SCHEDULED_AGENT_TIMEOUT_SECONDS", "180")
                        try:
                            timeout_s = int(timeout_env)
                        except (TypeError, ValueError):
                            timeout_s = 180
                        result = await asyncio.wait_for(
                            agent_executor.execute(agent_name, task_data),
                            timeout=timeout_s,
                        )
                        result["scheduled_execution"] = True
                        logger.info(f"Agent {agent_name} executed successfully")
                    except asyncio.TimeoutError:
                        logger.error(
                            "Agent %s timed out after %ss (scheduled_run)",
                            agent_name,
                            timeout_s,
                        )
                        result = {
                            "status": "timeout",
                            "error": f"Timed out after {timeout_s}s",
                            "scheduled_execution": True,
                        }
                    except NotImplementedError:
                        logger.warning(
                            f"Agent {agent_name} has no execute method (NotImplementedError)"
                        )
                        result = {
                            "status": "not_implemented",
                            "message": f"Agent {agent_name} missing execute implementation",
                            "scheduled_execution": True,
                            "warning": "This agent needs implementation",
                        }
                    except Exception as exec_err:
                        logger.error(f"Agent {agent_name} execution error: {exec_err}")
                        result = {
                            "status": "error",
                            "error": str(exec_err),
                            "scheduled_execution": True,
                        }
                else:
                    result = {
                        "status": "completed",
                        "message": "AgentExecutor not available - logged only",
                        "scheduled_execution": True,
                    }

                result_status = (result.get("status") or "").lower()
                final_status = (
                    "failed"
                    if result_status in {"error", "failed", "timeout", "not_implemented"}
                    else "completed"
                )
                await pool.execute(
                    """
                    UPDATE agent_executions
                    SET completed_at = $1, status = $2, response = $3
                    WHERE id = $4
                """,
                    datetime.utcnow(),
                    final_status,
                    _safe_json_dumps(result),
                    execution_id,
                )

                results.append(
                    {
                        "agent_id": str(agent["id"]),
                        "agent_name": agent["name"],
                        "execution_id": execution_id,
                        "status": final_status,
                        "result_status": result.get("status"),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to execute agent {agent['id']}: {e}")
                results.append(
                    {
                        "agent_id": str(agent["id"]),
                        "agent_name": agent["name"],
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return {
            "status": "completed",
            "executed": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Scheduled execution failed: {e}")
        return {"status": "failed", "error": str(e)}


@router.post("/agents/health/check")
async def check_agents_health():
    """Manually trigger health check for all agents"""
    if not _health_monitor_available():
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    try:
        from agent_health_monitor import get_health_monitor

        health_monitor = get_health_monitor()
        result = health_monitor.check_all_agents_health()
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/agents/{agent_id}/restart")
async def restart_agent(agent_id: str):
    """Manually restart a specific agent"""
    if not _health_monitor_available():
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    pool = get_pool()
    try:
        agent = await pool.fetchrow("SELECT name FROM ai_agents WHERE id::text = $1", agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        from agent_health_monitor import get_health_monitor

        health_monitor = get_health_monitor()
        result = health_monitor.restart_failed_agent(agent_id, agent["name"])

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Restart failed"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/agents/health/auto-restart")
async def auto_restart_critical_agents():
    """Automatically restart all agents in critical state"""
    if not _health_monitor_available():
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    try:
        from agent_health_monitor import get_health_monitor

        health_monitor = get_health_monitor()
        result = health_monitor.auto_restart_critical_agents()
        return result
    except Exception as e:
        logger.error(f"Auto-restart failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/agents/execute")
async def execute_agent_generic(
    request: Request,
    agent_type: str = Body("general", embed=True),
    task: str = Body("", embed=True),
    parameters: dict = Body({}, embed=True),
):
    """
    Generic agent execution endpoint - executes an agent by type without requiring agent_id.
    """
    pool = get_pool()
    execution_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    tenant_uuid = (
        _resolve_tenant(request)
        or config.tenant.default_tenant_id
        or os.getenv("DEFAULT_TENANT_ID")
        or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"
    )

    try:
        agent = await pool.fetchrow(
            """
            SELECT id, name, type
            FROM agents
            WHERE LOWER(type) = LOWER($1) AND status = 'active'
            ORDER BY RANDOM() LIMIT 1
        """,
            agent_type,
        )

        if not agent:
            agent = await pool.fetchrow(
                """
                SELECT id, name, type
                FROM agents
                WHERE LOWER(name) LIKE LOWER($1) AND status = 'active'
                ORDER BY RANDOM() LIMIT 1
            """,
                f"%{agent_type}%",
            )

        agent_id = str(agent["id"]) if agent else "system"
        agent_name = agent["name"] if agent else agent_type

        await pool.execute(
            """
            INSERT INTO ai_agent_executions
                (id, agent_name, task_type, status, input_data, created_at, tenant_id)
            VALUES ($1, $2, $3, 'running', $4, $5, $6::uuid)
        """,
            execution_id,
            agent_name,
            agent_type,
            json.dumps({"type": agent_type, "task": task, "parameters": parameters}),
            started_at,
            tenant_uuid,
        )

        execution_status = "completed"
        error_message = None
        http_status = 200

        agent_executor = _get_agent_executor()
        if _agents_available() and agent_executor:
            try:
                exec_task = {"task": task, **parameters, "_skip_ai_agent_log": True}
                result = await agent_executor.execute(agent_name, exec_task)
            except Exception as exec_error:
                logger.error(f"AgentExecutor failed: {exec_error}")
                execution_status = "failed"
                error_message = str(exec_error)
                http_status = 500
                result = {
                    "status": "error",
                    "message": f"Agent {agent_name} failed to execute task",
                    "agent_type": agent_type,
                    "error": error_message,
                }
        else:
            execution_status = "failed"
            error_message = "Agent executor not available"
            http_status = 503
            result = {
                "status": "error",
                "message": "Agent executor not available",
                "agent_type": agent_type,
            }

        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        await pool.execute(
            """
            UPDATE ai_agent_executions
            SET status = $1, execution_time_ms = $2, output_data = $3, error_message = $4
            WHERE id = $5
        """,
            execution_status,
            duration_ms,
            json.dumps(jsonable_encoder(result)),
            error_message,
            execution_id,
        )

        response_payload = {
            "success": execution_status == "completed",
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": completed_at.isoformat(),
        }

        if execution_status != "completed":
            raise HTTPException(status_code=http_status, detail=response_payload)

        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generic agent execution failed: {e}")
        await pool.execute(
            """
            UPDATE ai_agent_executions SET status = 'failed', error_message = $1 WHERE id = $2
        """,
            str(e),
            execution_id,
        )
        logger.error("Internal server error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/ai/tasks/execute/{task_id}")
async def execute_ai_task(task_id: str):
    """Manually trigger execution of a specific task (requires auth - CRITICAL)"""
    app = _get_app()
    integration_layer = getattr(app.state, "integration_layer", None)
    if not _integration_layer_available() or integration_layer is None:
        raise HTTPException(
            status_code=503, detail="AI Integration Layer not available or not initialized"
        )

    try:
        task = await integration_layer.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        await integration_layer._execute_task(task)

        return {"success": True, "message": "Task execution triggered", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/api/v1/agents/execute")
async def api_v1_agents_execute(
    request: Request,
    payload: AgentExecuteRequest,
):
    """
    Execute an agent via the v1 API surface.

    Body: { "agent_id" | "id": string, "payload": object }
    Internally delegates to the existing /agents/{agent_id}/execute endpoint.
    """
    agent_id = payload.agent_id or payload.id
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")

    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/agents/{agent_id}/execute",
        "headers": [],
    }
    from starlette.requests import Request as StarletteRequest

    async def receive() -> dict[str, Any]:
        body_bytes = json.dumps(payload.payload or {}).encode("utf-8")
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    delegated_request: Request = StarletteRequest(scope, receive)  # type: ignore[arg-type]

    return await execute_agent(agent_id=agent_id, request=delegated_request, authenticated=True)


@router.post("/api/v1/agents/activate")
async def api_v1_agents_activate(payload: AgentActivateRequest):
    """
    Activate or deactivate an agent via the v1 API surface.
    """
    if not payload.agent_id and not payload.agent_name:
        raise HTTPException(status_code=400, detail="agent_id or agent_name is required")

    pool = get_pool()

    try:
        row = None

        if payload.agent_id:
            row = await pool.fetchrow(
                """
                UPDATE agents
                SET enabled = $1, updated_at = NOW()
                WHERE id::text = $2
                RETURNING id, name, category, enabled
                """,
                payload.enabled,
                payload.agent_id,
            )

        if not row and payload.agent_name:
            row = await pool.fetchrow(
                """
                UPDATE agents
                SET enabled = $1, updated_at = NOW()
                WHERE name = $2
                RETURNING id, name, category, enabled
                """,
                payload.enabled,
                payload.agent_name,
            )

        if not row:
            raise HTTPException(status_code=404, detail="Agent not found")

        data = dict(row)
        return {
            "success": True,
            "agent": {
                "id": str(data.get("id")),
                "name": data.get("name"),
                "category": data.get("category"),
                "enabled": data.get("enabled"),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Agent activation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent activation failed") from exc


@router.post("/api/v1/aurea/execute-event")
async def execute_aurea_event(
    request: AUREAEventRequest,
):
    """
    Execute event with specified AI agent via AUREA orchestration.
    Called by Event Router daemon to process events from brainops_core.event_bus.
    """
    logger.info(
        f"AUREA Event: {request.event_id} ({request.topic}) -> {request.target_agent['name']}"
    )

    pool = get_pool()

    try:
        agent_row = await pool.fetchrow(
            """
            SELECT id, name, category, enabled
            FROM agents
            WHERE name = $1 AND enabled = TRUE
            """,
            request.target_agent["name"],
        )

        if not agent_row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{request.target_agent['name']}' not found or disabled",
            )

        str(agent_row["id"])
        agent_name = agent_row["name"]

        result = {
            "status": "acknowledged",
            "agent": agent_name,
            "event_id": request.event_id,
            "topic": request.topic,
            "action": "processed",
        }

        try:
            await pool.execute(
                """
                UPDATE brainops_core.agents
                SET last_active_at = NOW()
                WHERE name = $1
                """,
                agent_name,
            )
        except Exception as exc:
            logger.debug("Failed to update agent heartbeat: %s", exc, exc_info=True)

        app = _get_app()
        embedded_memory = getattr(app.state, "embedded_memory", None)
        if embedded_memory:
            try:
                embedded_memory.store_memory(
                    memory_id=str(uuid.uuid4()),
                    memory_type="episodic",
                    source_agent=agent_name,
                    content=f"Processed event: {request.topic}",
                    metadata={
                        "event_id": request.event_id,
                        "topic": request.topic,
                        "source": request.source,
                    },
                    importance_score=0.7,
                )
            except Exception as e:
                logger.warning(f"Could not store in embedded memory: {e}")

        logger.info(f"AUREA Event {request.event_id} executed by {agent_name}")

        return {
            "success": True,
            "event_id": request.event_id,
            "agent": agent_name,
            "topic": request.topic,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AUREA Event {request.event_id} failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
