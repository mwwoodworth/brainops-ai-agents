"""Operational router — Wave 2C extraction from app.py.

Routes: /executions, /self-heal/*, /aurea/status, /aurea/command/natural_language,
/training/capture-interaction, /training/stats
"""

import json
import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database.async_connection import DatabaseUnavailableError, get_pool
from services.tenant_helpers import fetchval_with_tenant_context, resolve_tenant_uuid_from_request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["operational"])


# ---------------------------------------------------------------------------
# Lazy helpers – avoid circular imports; resolve at call time
# ---------------------------------------------------------------------------


def _get_app():
    import app as _app

    return _app.app


def _local_executions():
    import app as _app

    return getattr(_app, "LOCAL_EXECUTIONS", [])


def _training_available():
    import app as _app

    return getattr(_app, "TRAINING_AVAILABLE", False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AureaCommandRequest(BaseModel):
    command_text: str


# ---------------------------------------------------------------------------
# Executions
# ---------------------------------------------------------------------------


@router.get("/executions")
async def get_executions(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    """Get agent executions"""
    try:
        pool = get_pool()
        query = """
            SELECT e.id, e.agent_type, e.status, e.started_at, e.completed_at,
                   e.duration_ms, e.error_message, e.model_name,
                   e.tokens_input, e.tokens_output
            FROM agent_executions e
            WHERE 1=1
        """
        params = []

        if agent_id:
            query += f" AND e.agent_type = ${len(params) + 1}"
            params.append(agent_id)

        if status:
            query += f" AND e.status = ${len(params) + 1}"
            params.append(status)

        query += f" ORDER BY e.started_at DESC NULLS LAST LIMIT ${len(params) + 1}"
        params.append(limit)

        try:
            rows = await pool.fetch(query, *params)
        except Exception as primary_error:
            logger.error("Execution query failed: %s", primary_error, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Execution history unavailable; database query failed.",
            ) from primary_error

        executions = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            execution = {
                "execution_id": str(data.get("id")),
                "agent_id": data.get("agent_type"),
                "agent_name": data.get("agent_type"),
                "status": data.get("status"),
                "started_at": data["started_at"].isoformat() if data.get("started_at") else None,
                "completed_at": data["completed_at"].isoformat()
                if data.get("completed_at")
                else None,
                "duration_ms": data.get("duration_ms"),
                "error": data.get("error_message"),
            }
            executions.append(execution)

        seen_ids = {item["execution_id"] for item in executions if item.get("execution_id")}
        for entry in list(_local_executions()):
            exec_id = entry.get("execution_id")
            if exec_id in seen_ids:
                continue
            executions.insert(
                0,
                {
                    "execution_id": exec_id,
                    "agent_id": entry.get("agent_id"),
                    "agent_name": entry.get("agent_name"),
                    "status": entry.get("status"),
                    "started_at": entry.get("started_at").isoformat()
                    if entry.get("started_at")
                    else None,
                    "completed_at": entry.get("completed_at").isoformat()
                    if entry.get("completed_at")
                    else None,
                    "duration_ms": entry.get("duration_ms"),
                    "error": entry.get("error"),
                },
            )

        return {"executions": executions, "total": len(executions)}

    except HTTPException:
        raise
    except DatabaseUnavailableError as exc:
        logger.error("Database unavailable while loading executions", exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        logger.error(f"Failed to get executions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve executions") from e


# ---------------------------------------------------------------------------
# Self-Healing Endpoints
# ---------------------------------------------------------------------------


@router.post("/self-heal/trigger")
async def trigger_self_healing():
    """
    Trigger self-healing check and remediation.

    This endpoint can be called by cron jobs to proactively check for issues
    and trigger healing actions, bypassing the need for AUREA's main loop.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "issues_detected": [],
        "actions_taken": [],
        "status": "completed",
    }

    try:
        pool = get_pool()

        # 1. Check for failed AUREA decisions and retry
        failed_decisions = await pool.fetch(
            """
            SELECT id, decision_type, context
            FROM aurea_decisions
            WHERE execution_status = 'failed'
            AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 10
        """
        )

        for decision in failed_decisions:
            results["issues_detected"].append(
                {
                    "type": "failed_decision",
                    "id": str(decision["id"]),
                    "decision_type": decision["decision_type"],
                }
            )
            # Reset for retry
            await pool.execute(
                """
                UPDATE aurea_decisions
                SET execution_status = 'pending',
                    execution_result = NULL
                WHERE id = $1
            """,
                decision["id"],
            )
            results["actions_taken"].append(
                {"action": "reset_for_retry", "target": str(decision["id"])}
            )

        # 2. Check healing rules and match against recent errors
        recent_errors = (
            await pool.fetch(
                """
            SELECT DISTINCT error_type, error_message, component
            FROM ai_error_logs
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            LIMIT 20
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_error_logs')"
            )
            else []
        )

        healing_rules = await pool.fetch(
            """
            SELECT id, component, error_pattern, fix_action, confidence
            FROM ai_healing_rules
            WHERE enabled = true
        """
        )

        for error in recent_errors:
            for rule in healing_rules:
                if rule["error_pattern"] in str(error.get("error_message", "")) or rule[
                    "error_pattern"
                ] in str(error.get("error_type", "")):
                    results["issues_detected"].append(
                        {
                            "type": "matched_error",
                            "error_type": error.get("error_type"),
                            "matched_rule": str(rule["id"]),
                        }
                    )
                    results["actions_taken"].append(
                        {
                            "action": rule["fix_action"],
                            "component": rule["component"],
                            "confidence": float(rule["confidence"] or 0),
                        }
                    )
                    # Update rule usage
                    await pool.execute(
                        """
                        UPDATE ai_healing_rules
                        SET success_count = success_count + 1, updated_at = NOW()
                        WHERE id = $1
                    """,
                        rule["id"],
                    )

        # 3. Check for stalled agents (use correct column: type not agent_type)
        stalled_agents = (
            await pool.fetch(
                """
            SELECT id, name, type
            FROM ai_agents
            WHERE enabled = true
            AND last_execution_at < NOW() - INTERVAL '2 hours'
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_agents' AND column_name = 'last_execution_at')"
            )
            else []
        )

        for agent in stalled_agents:
            results["issues_detected"].append(
                {"type": "stalled_agent", "agent_id": str(agent["id"]), "agent_name": agent["name"]}
            )

        # 4. Log healing run
        await pool.execute(
            """
            INSERT INTO remediation_history (action_type, target_component, result, success, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """,
            "self_heal_trigger",
            "system",
            json.dumps(results),
            True,
            json.dumps(
                {
                    "issues_count": len(results["issues_detected"]),
                    "actions_count": len(results["actions_taken"]),
                }
            ),
        ) if await pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'remediation_history')"
        ) else None

        logger.info(
            f"Self-healing check complete: {len(results['issues_detected'])} issues, {len(results['actions_taken'])} actions"
        )

    except Exception as e:
        logger.error(f"Self-healing trigger failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


@router.get("/self-heal/check")
async def check_self_healing():
    """
    Check self-healing status without triggering remediation.
    Returns current health state and any detected issues.
    """
    pool = get_pool()
    _app = _get_app()

    try:
        issues = []
        health_score = 100.0

        # Check database connectivity
        db_healthy = True
        try:
            await pool.fetchval("SELECT 1")
        except Exception:
            db_healthy = False
            issues.append(
                {
                    "type": "database",
                    "severity": "critical",
                    "message": "Database connection failed",
                }
            )
            health_score -= 30

        # Check for stalled agents (use created_at, not started_at)
        stalled_agents = await pool.fetch(
            """
            SELECT id, agent_name, status, created_at
            FROM ai_agent_executions
            WHERE status = 'running'
            AND created_at < NOW() - INTERVAL '30 minutes'
            LIMIT 10
        """
        )

        if stalled_agents:
            for agent in stalled_agents:
                issues.append(
                    {
                        "type": "stalled_agent",
                        "severity": "high",
                        "agent_name": agent["agent_name"],
                        "started_at": agent["created_at"].isoformat()
                        if agent["created_at"]
                        else None,
                    }
                )
            health_score -= len(stalled_agents) * 5

        # Check for failed executions in last hour (use created_at, not started_at)
        failed_count = (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM ai_agent_executions
            WHERE status = 'failed' AND created_at > NOW() - INTERVAL '1 hour'
        """
            )
            or 0
        )

        if failed_count > 5:
            issues.append(
                {
                    "type": "high_failure_rate",
                    "severity": "medium",
                    "message": f"{failed_count} failed executions in last hour",
                }
            )
            health_score -= min(failed_count, 20)

        # Check healer status
        healer = getattr(_app.state, "healer", None)
        healer_active = healer is not None

        # Get recent remediation history
        remediation_history = (
            await pool.fetch(
                """
            SELECT action_type, target_component, success, created_at
            FROM remediation_history
            ORDER BY created_at DESC LIMIT 5
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'remediation_history')"
            )
            else []
        )

        return {
            "health_score": max(0, health_score),
            "status": "healthy"
            if health_score >= 80
            else "degraded"
            if health_score >= 50
            else "critical",
            "issues_count": len(issues),
            "issues": issues,
            "self_healer_active": healer_active,
            "database_healthy": db_healthy,
            "stalled_agents": len(stalled_agents),
            "failed_executions_last_hour": failed_count,
            "recent_remediation": [
                {
                    "action": r["action_type"],
                    "target": r["target_component"],
                    "success": r["success"],
                    "at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in remediation_history
            ]
            if remediation_history
            else [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Self-healing check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# AUREA Endpoints
# ---------------------------------------------------------------------------


@router.get("/aurea/status")
async def get_aurea_status(request: Request):
    """
    Get AUREA operational status - checks actual OODA loop activity (requires auth).
    This endpoint verifies real AUREA activity in the database.
    """
    try:
        pool = get_pool()
        if not pool:
            return {
                "status": "initializing",
                "aurea_available": False,
                "message": "Database pool not available",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints": {
                    "full_status": "/aurea/chat/status",
                    "chat": "/aurea/chat/message",
                    "websocket": "/aurea/chat/ws/{session_id}",
                },
            }

        tenant_uuid = resolve_tenant_uuid_from_request(request)

        recent_cycles = await fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM aurea_state
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """,
            tenant_uuid=tenant_uuid,
        )

        recent_decisions = await fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """,
            tenant_uuid=tenant_uuid,
        )

        active_agents = await fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM ai_agents WHERE status = 'active'
        """,
            tenant_uuid=tenant_uuid,
        )

        # AUREA is operational if we have recent OODA cycles
        aurea_operational = recent_cycles > 0

        return {
            "status": "operational" if aurea_operational else "idle",
            "aurea_available": True,
            "ooda_cycles_last_5min": recent_cycles,
            "decisions_last_hour": recent_decisions,
            "active_agents": active_agents,
            "tenant_id": tenant_uuid,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "full_status": "/aurea/chat/status",
                "chat": "/aurea/chat/message",
                "websocket": "/aurea/chat/ws/{session_id}",
            },
        }
    except DatabaseUnavailableError as e:
        logger.warning("AUREA status requested while database unavailable: %s", e)
        return JSONResponse(
            status_code=503,
            content={
                "status": "initializing",
                "aurea_available": False,
                "message": "Database pool not initialized yet",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
    except Exception as e:
        logger.error(f"Failed to get AUREA status: {e!r}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@router.post("/aurea/command/natural_language")
async def execute_aurea_nl_command(request: Request, payload: AureaCommandRequest = Body(...)):
    """
    Execute a natural language command through AUREA's NLU processor (requires auth - CRITICAL).
    Founder-level authority for natural language system control.

    Examples:
    - "Create a high priority task to deploy the new feature"
    - "Show me all tasks that are in progress"
    - "Get AUREA status"
    - "Execute task abc-123"
    """
    _app = _get_app()
    if not hasattr(_app.state, "aurea_nlu") or not _app.state.aurea_nlu:
        logger.warning(
            "AUREA NLU processor unavailable; using /aurea/chat/command fallback for '%s'",
            payload.command_text,
        )
        try:
            from api.aurea_chat import NLCommand, execute_natural_language_command

            fallback_response = await execute_natural_language_command(
                NLCommand(command=payload.command_text)
            )

            if isinstance(fallback_response, JSONResponse):
                try:
                    parsed = json.loads(fallback_response.body.decode("utf-8"))
                except Exception:
                    parsed = {"detail": fallback_response.body.decode("utf-8", errors="ignore")}
                if fallback_response.status_code >= 400:
                    raise HTTPException(
                        status_code=fallback_response.status_code,
                        detail=parsed.get("error")
                        or parsed.get("detail")
                        or "AUREA fallback failed",
                    )
                return {
                    "success": True,
                    "command": payload.command_text,
                    "result": parsed,
                    "processor": "chat-command-fallback",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            return {
                "success": True,
                "command": payload.command_text,
                "result": fallback_response,
                "processor": "chat-command-fallback",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except HTTPException:
            raise
        except Exception as fallback_error:
            logger.error(
                "AUREA fallback command execution failed: %s", fallback_error, exc_info=True
            )
            raise HTTPException(
                status_code=503,
                detail=f"AUREA NLU Processor not available; fallback failed: {fallback_error}",
            ) from fallback_error

    try:
        command_text = payload.command_text
        nlu = _app.state.aurea_nlu
        result = await nlu.execute_natural_language_command(command_text)

        return {
            "success": True,
            "command": command_text,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Natural language command execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Training Endpoints
# ---------------------------------------------------------------------------


@router.post("/training/capture-interaction")
async def capture_interaction(interaction_data: dict[str, Any] = Body(...)):
    """
    Capture customer interaction for AI training.

    This is CRITICAL for the learning system - without captured interactions,
    the AI cannot learn and improve.
    """
    _app = _get_app()
    if not _training_available() or not hasattr(_app.state, "training") or not _app.state.training:
        raise HTTPException(status_code=503, detail="Training pipeline not available")

    try:
        from ai_training_pipeline import InteractionType

        training_pipeline = _app.state.training
        interaction_id = await training_pipeline.capture_interaction(
            customer_id=interaction_data.get("customer_id"),
            interaction_type=InteractionType[interaction_data.get("type", "EMAIL").upper()],
            content=interaction_data.get("content"),
            channel=interaction_data.get("channel"),
            context=interaction_data.get("context", {}),
            outcome=interaction_data.get("outcome"),
            value=interaction_data.get("value"),
        )

        logger.info(f"Captured interaction {interaction_id} for training")
        return {"interaction_id": interaction_id, "status": "captured"}

    except Exception as e:
        logger.error(f"Failed to capture interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/training/stats")
async def get_training_stats():
    """Get training pipeline statistics (requires auth)"""
    _app = _get_app()
    if not _training_available() or not hasattr(_app.state, "training") or not _app.state.training:
        return {"available": False, "message": "Training pipeline not available"}

    try:
        pool = get_pool()
        stats = await pool.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM ai_customer_interactions) as total_interactions,
                (SELECT MAX(created_at) FROM ai_customer_interactions) as last_interaction,
                (SELECT COUNT(*) FROM ai_training_data) as training_samples,
                (SELECT COUNT(*) FROM ai_learning_insights) as insights_generated
        """
        )
        return {
            "available": True,
            "total_interactions": stats["total_interactions"],
            "last_interaction": stats["last_interaction"].isoformat()
            if stats["last_interaction"]
            else None,
            "training_samples": stats["training_samples"],
            "insights_generated": stats["insights_generated"],
        }
    except Exception as e:
        return {"available": True, "error": str(e)}
