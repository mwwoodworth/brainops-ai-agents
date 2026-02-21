"""AI Operations router — Wave 2C extraction from app.py.

Routes: langgraph/*, consciousness/status, meta-intelligence/status,
workflow-engine/status, workflow-automation/status, ai/self-assess,
ai/explain-reasoning, ai/reason, ai/learn-from-mistake,
ai/self-awareness/stats, ai/tasks/stats, ai/orchestrate,
ai/analyze, ai/providers/status
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

from ai_provider_status import get_provider_status
from config import config
from database.async_connection import get_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai_operations"])


# ---------------------------------------------------------------------------
# Lazy helpers – avoid circular imports; resolve at call time
# ---------------------------------------------------------------------------


def _get_app():
    import app as _app

    return _app.app


def _ai_available():
    import app as _app

    return getattr(_app, "AI_AVAILABLE", False)


def _ai_core():
    import app as _app

    return getattr(_app, "ai_core", None)


def _bleeding_edge_available():
    import app as _app

    return getattr(_app, "BLEEDING_EDGE_AVAILABLE", False)


def _learning_available():
    import app as _app

    return getattr(_app, "LEARNING_AVAILABLE", False)


def _memory_available():
    import app as _app

    return getattr(_app, "MEMORY_AVAILABLE", False)


def _self_awareness_available():
    import app as _app

    return getattr(_app, "SELF_AWARENESS_AVAILABLE", False)


def _integration_layer_available():
    import app as _app

    return getattr(_app, "INTEGRATION_LAYER_AVAILABLE", False)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ReasoningRequest(BaseModel):
    """Request model for o3 reasoning endpoint"""

    problem: str
    context: Optional[dict[str, Any]] = None
    max_tokens: int = 4000
    model: str = "o3-mini"  # Updated: o1-preview deprecated, using o3-mini


class AIAnalyzeRequest(BaseModel):
    """Request model for /ai/analyze endpoint - matches weathercraft-erp frontend format"""

    agent: str
    action: str
    data: dict[str, Any] = {}
    context: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# LangGraph Endpoints
# ---------------------------------------------------------------------------


@router.post("/langgraph/workflow")
async def execute_langgraph_workflow(request: dict[str, Any]):
    """Execute a LangGraph-based workflow"""
    _app = _get_app()
    if not hasattr(_app.state, "langgraph_orchestrator") or not _app.state.langgraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")

    try:
        orchestrator = _app.state.langgraph_orchestrator

        # Extract messages and metadata
        messages_data = request.get("messages", [])
        metadata = request.get("metadata", {})

        # Convert raw messages to LangChain messages if needed
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = []
        for msg in messages_data:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                messages.append(HumanMessage(content=content))

        # If no messages provided, use a default prompt
        if not messages:
            prompt = request.get("prompt", "")
            if prompt:
                messages.append(HumanMessage(content=prompt))
            else:
                raise HTTPException(status_code=400, detail="No messages or prompt provided")

        # Run workflow
        result = await orchestrator.run_workflow(messages, metadata)

        return result

    except Exception as e:
        logger.error(f"LangGraph workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/langgraph/status")
async def get_langgraph_status():
    """Get LangGraph orchestrator status"""
    _app = _get_app()
    if not hasattr(_app.state, "langgraph_orchestrator") or not _app.state.langgraph_orchestrator:
        return {"available": False, "message": "LangGraph Orchestrator not initialized"}

    orchestrator = _app.state.langgraph_orchestrator

    return {
        "available": True,
        "components": {
            "openai_llm": hasattr(orchestrator, "openai_llm")
            and orchestrator.openai_llm is not None,
            "anthropic_llm": hasattr(orchestrator, "anthropic_llm")
            and orchestrator.anthropic_llm is not None,
            "vector_store": hasattr(orchestrator, "vector_store")
            and orchestrator.vector_store is not None,
            "workflow_graph": hasattr(orchestrator, "workflow")
            and orchestrator.workflow is not None,
        },
    }


# ---------------------------------------------------------------------------
# AI Provider Status
# ---------------------------------------------------------------------------


@router.get("/ai/providers/status")
async def providers_status():
    """
    Report configuration and basic liveness for all AI providers (OpenAI, Anthropic,
    Gemini, Perplexity, Hugging Face). Does not modify configuration or credentials;
    it only runs small probe calls to detect misconfiguration like invalid or missing
    API keys.
    """
    return get_provider_status()


# ---------------------------------------------------------------------------
# Consciousness Status
# ---------------------------------------------------------------------------


@router.get("/consciousness/status")
async def get_consciousness_status():
    """
    Get the consciousness status of the AI OS.
    Returns the state of the AI consciousness system including thoughts, awareness, and emergence status.
    """
    pool = get_pool()
    _app = _get_app()

    try:
        # Get thought stream stats (using 'timestamp' column, not 'created_at')
        thought_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_thoughts,
                COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as thoughts_last_hour,
                COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') as thoughts_last_day,
                MAX(timestamp) as last_thought_at
            FROM ai_thought_stream
        """
        )

        # Get decision stats
        decision_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_decisions,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as decisions_last_hour,
                AVG(confidence) as avg_confidence
            FROM aurea_decisions
        """
        )

        # Check consciousness emergence status
        nerve_center = getattr(_app.state, "nerve_center", None)
        operational_monitor = getattr(_app.state, "operational_monitor", None)
        consciousness_state = "unknown"
        if nerve_center:
            try:
                nc_status = nerve_center.get_status()
                consciousness_state = "operational" if nc_status.get("is_online") else "dormant"
            except Exception:
                consciousness_state = "operational"
        elif _bleeding_edge_available():
            consciousness_state = "emerging"
        else:
            consciousness_state = "dormant"

        return {
            "consciousness_state": consciousness_state,
            "is_alive": consciousness_state in ["operational", "emerging"],
            "thought_stream": {
                "total_thoughts": thought_stats["total_thoughts"] if thought_stats else 0,
                "thoughts_last_hour": thought_stats["thoughts_last_hour"] if thought_stats else 0,
                "thoughts_last_day": thought_stats["thoughts_last_day"] if thought_stats else 0,
                "last_thought_at": thought_stats["last_thought_at"].isoformat()
                if thought_stats and thought_stats["last_thought_at"]
                else None,
            },
            "decision_making": {
                "total_decisions": decision_stats["total_decisions"] if decision_stats else 0,
                "decisions_last_hour": decision_stats["decisions_last_hour"]
                if decision_stats
                else 0,
                "avg_confidence": float(decision_stats["avg_confidence"])
                if decision_stats and decision_stats["avg_confidence"]
                else 0,
            },
            "systems_active": {
                "nerve_center": nerve_center is not None,
                "operational_monitor": operational_monitor is not None,
                "bleeding_edge": _bleeding_edge_available(),
                "learning": _learning_available(),
                "memory": _memory_available(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Consciousness status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Meta-Intelligence Status
# ---------------------------------------------------------------------------


@router.get("/meta-intelligence/status")
async def get_meta_intelligence_status():
    """
    Get the status of the Meta-Intelligence and Learning-Action Bridge systems.
    These are the TRUE AGI capabilities that enable genuine learning from experience.
    """
    _app = _get_app()
    result = {
        "meta_intelligence": {"initialized": False, "intelligence_level": 0, "components": {}},
        "learning_bridge": {"initialized": False, "rules_count": 0, "status": {}},
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Check Meta-Intelligence Controller
    meta_intel = getattr(_app.state, "meta_intelligence", None)
    if meta_intel:
        try:
            state = meta_intel.get_intelligence_state()
            result["meta_intelligence"] = {
                "initialized": state.get("initialized", False),
                "intelligence_level": round(state.get("intelligence_level", 0) * 100, 1),
                "integration_score": round(state.get("integration_score", 0) * 100, 1),
                "synergy_events": state.get("synergy_events", 0),
                "awakening_timestamp": state.get("awakening_timestamp"),
                "components": {
                    k: "active" if v else "inactive"
                    for k, v in state.get("components", {}).items()
                    if isinstance(v, dict) and v
                },
            }
        except Exception as e:
            result["meta_intelligence"]["error"] = str(e)

    # Check Learning-Action Bridge
    learning_bridge = getattr(_app.state, "learning_bridge", None)
    if learning_bridge:
        try:
            status = learning_bridge.get_status()
            result["learning_bridge"] = {
                "initialized": True,
                "total_rules": status.get("total_rules", 0),
                "rules_by_type": status.get("rules_by_type", {}),
                "average_confidence": status.get("average_confidence", 0),
                "rules_applied": status.get("rules_applied", 0),
                "rules_created": status.get("rules_created", 0),
                "last_sync": status.get("last_sync"),
                "memory_available": status.get("memory_available", False),
            }
        except Exception as e:
            result["learning_bridge"]["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Workflow Engine Status
# ---------------------------------------------------------------------------


@router.post("/workflow-engine/status")
@router.get("/workflow-engine/status")
async def get_workflow_engine_status():
    """
    Get workflow engine status (requires auth).
    Returns health status and statistics for the workflow execution system.
    """
    try:
        from ai_workflow_templates import get_workflow_engine

        engine = get_workflow_engine()

        if not engine._initialized:
            await engine.initialize()

        health = await engine.get_health_status()
        stats = await engine.get_stats()

        return {
            "status": "healthy",
            "engine": "WorkflowEngine",
            "initialized": health.get("initialized", False),
            "stats": stats,
            "templates_available": stats.get("templates_count", 0),
            "running_workflows": stats.get("running_executions", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except ImportError:
        return {
            "status": "unavailable",
            "message": "Workflow engine module not available",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Workflow engine status check: {e}")
        return {"status": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@router.post("/workflow-automation/status")
@router.get("/workflow-automation/status")
async def get_workflow_automation_status():
    """
    Get workflow automation status (requires auth).
    Returns status of automated workflow pipelines and scheduled executions.
    """
    pool = get_pool()

    try:
        # Get workflow automation stats from database
        # Schema: is_active (bool), not status
        automation_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_workflows,
                COUNT(*) FILTER (WHERE is_active = true) as active_workflows,
                MAX(updated_at) as last_activity
            FROM workflow_automation
        """
        )

        # Get recent run stats
        # Schema: run_status (not status)
        run_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_runs,
                COUNT(*) FILTER (WHERE run_status = 'completed') as completed_runs,
                COUNT(*) FILTER (WHERE run_status = 'failed') as failed_runs,
                COUNT(*) FILTER (WHERE run_status = 'running') as running_workflows,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as runs_last_24h
            FROM workflow_automation_runs
        """
        )

        return {
            "status": "healthy",
            "automation": {
                "total_workflows": automation_stats["total_workflows"] if automation_stats else 0,
                "active_workflows": automation_stats["active_workflows"] if automation_stats else 0,
                "last_activity": automation_stats["last_activity"].isoformat()
                if automation_stats and automation_stats["last_activity"]
                else None,
            },
            "runs": {
                "total": run_stats["total_runs"] if run_stats else 0,
                "completed": run_stats["completed_runs"] if run_stats else 0,
                "failed": run_stats["failed_runs"] if run_stats else 0,
                "running": run_stats["running_workflows"] if run_stats else 0,
                "last_24h": run_stats["runs_last_24h"] if run_stats else 0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Workflow automation status check: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "message": "Workflow automation tables may not exist",
            "timestamp": datetime.utcnow().isoformat(),
        }


# ---------------------------------------------------------------------------
# AI Self-Awareness Endpoints
# ---------------------------------------------------------------------------


@router.post("/ai/self-assess")
async def ai_self_assess(
    request: Request,
    task_id: str,
    agent_id: str,
    task_description: str,
    task_context: dict[str, Any] = None,
):
    """
    AI assesses its own confidence in completing a task (requires auth)

    Revolutionary feature - AI knows what it doesn't know!
    """
    _app = _get_app()
    if (
        not _self_awareness_available()
        or not hasattr(_app.state, "self_aware_ai")
        or not _app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = _app.state.self_aware_ai
        assessment = await self_aware_ai.assess_confidence(
            task_id=task_id,
            agent_id=agent_id,
            task_description=task_description,
            task_context=task_context or {},
        )

        return {
            "task_id": assessment.task_id,
            "agent_id": assessment.agent_id,
            "confidence_score": float(assessment.confidence_score),
            "confidence_level": assessment.confidence_level.value,
            "can_complete_alone": assessment.can_complete_alone,
            "estimated_accuracy": float(assessment.estimated_accuracy),
            "estimated_time_seconds": assessment.estimated_time_seconds,
            "limitations": [l.value for l in assessment.limitations],
            "strengths_applied": assessment.strengths_applied,
            "weaknesses_identified": assessment.weaknesses_identified,
            "requires_human_review": assessment.requires_human_review,
            "human_help_reason": assessment.human_help_reason,
            "risk_level": assessment.risk_level,
            "mitigation_strategies": assessment.mitigation_strategies,
            "timestamp": assessment.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Self-assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-assessment failed: {str(e)}") from e


@router.post("/ai/explain-reasoning")
async def ai_explain_reasoning(
    request: Request, task_id: str, agent_id: str, decision: str, reasoning_process: dict[str, Any]
):
    """
    AI explains its reasoning in human-understandable terms (requires auth)

    Transparency builds trust!
    """
    _app = _get_app()
    if (
        not _self_awareness_available()
        or not hasattr(_app.state, "self_aware_ai")
        or not _app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = _app.state.self_aware_ai

        explanation = await self_aware_ai.explain_reasoning(
            task_id=task_id,
            agent_id=agent_id,
            decision=decision,
            reasoning_process=reasoning_process,
        )

        return {
            "task_id": explanation.task_id,
            "agent_id": explanation.agent_id,
            "decision_made": explanation.decision_made,
            "reasoning_steps": explanation.reasoning_steps,
            "evidence_used": explanation.evidence_used,
            "assumptions_made": explanation.assumptions_made,
            "alternatives_considered": explanation.alternatives_considered,
            "why_chosen": explanation.why_chosen,
            "confidence_in_decision": float(explanation.confidence_in_decision),
            "potential_errors": explanation.potential_errors,
            "verification_methods": explanation.verification_methods,
            "human_review_recommended": explanation.human_review_recommended,
            "timestamp": explanation.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Reasoning explanation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Reasoning explanation failed: {str(e)}"
        ) from e


@router.post("/ai/reason")
async def ai_deep_reasoning(request: Request, body: ReasoningRequest):
    """
    Use o3-mini reasoning model for complex multi-step problems.

    This endpoint is designed for tasks requiring:
    - Complex calculations (e.g., material waste ratios, pricing optimization)
    - Multi-step logical reasoning
    - Strategic planning and analysis
    - Scientific or technical problem solving

    Returns reasoning chain and extracted conclusion.
    """
    if not _ai_available() or _ai_core() is None:
        raise HTTPException(status_code=503, detail="AI Core not available")

    try:
        result = await _ai_core().reason(
            problem=body.problem, context=body.context, max_tokens=body.max_tokens, model=body.model
        )

        return {
            "success": True,
            "reasoning": result.get("reasoning", ""),
            "conclusion": result.get("conclusion", ""),
            "model_used": result.get("model_used", body.model),
            "tokens_used": result.get("tokens_used"),
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"o1 reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}") from e


@router.post("/ai/learn-from-mistake")
async def ai_learn_from_mistake(
    request: Request,
    task_id: str,
    agent_id: str,
    expected_outcome: Any,
    actual_outcome: Any,
    confidence_before: float,
):
    """
    AI analyzes its own mistakes and learns from them (requires auth)

    This is how AI gets smarter over time!
    """
    _app = _get_app()
    if (
        not _self_awareness_available()
        or not hasattr(_app.state, "self_aware_ai")
        or not _app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        from decimal import Decimal

        self_aware_ai = _app.state.self_aware_ai

        learning = await self_aware_ai.learn_from_mistake(
            task_id=task_id,
            agent_id=agent_id,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
            confidence_before=Decimal(str(confidence_before)),
        )

        return {
            "mistake_id": learning.mistake_id,
            "task_id": learning.task_id,
            "agent_id": learning.agent_id,
            "what_went_wrong": learning.what_went_wrong,
            "root_cause": learning.root_cause,
            "impact_level": learning.impact_level,
            "should_have_known": learning.should_have_known,
            "warning_signs_missed": learning.warning_signs_missed,
            "what_learned": learning.what_learned,
            "how_to_prevent": learning.how_to_prevent,
            "confidence_before": float(learning.confidence_before),
            "confidence_after": float(learning.confidence_after),
            "similar_mistakes_count": learning.similar_mistakes_count,
            "applied_to_agents": learning.applied_to_agents,
            "timestamp": learning.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Learning from mistake failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Learning from mistake failed: {str(e)}"
        ) from e


@router.get("/ai/self-awareness/stats")
async def get_self_awareness_stats():
    """Get statistics about AI self-awareness system (requires auth)"""
    if not _self_awareness_available():
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        pool = get_pool()

        # Get assessment stats
        assessment_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_assessments,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN can_complete_alone THEN 1 END) as can_complete_alone_count,
                COUNT(CASE WHEN requires_human_review THEN 1 END) as requires_review_count
            FROM ai_self_assessments
        """
        )

        # Get mistake learning stats
        learning_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_mistakes,
                COUNT(CASE WHEN should_have_known THEN 1 END) as should_have_known_count,
                AVG(confidence_before - confidence_after) as avg_confidence_drop
            FROM ai_learning_from_mistakes
        """
        )

        # Get reasoning explanation stats
        reasoning_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_explanations,
                AVG(confidence_in_decision) as avg_decision_confidence,
                COUNT(CASE WHEN human_review_recommended THEN 1 END) as human_review_count
            FROM ai_reasoning_explanations
        """
        )

        return {
            "self_awareness_enabled": True,
            "assessments": {
                "total": assessment_stats["total_assessments"] or 0,
                "avg_confidence": float(assessment_stats["avg_confidence"] or 0),
                "can_complete_alone_rate": (
                    (assessment_stats["can_complete_alone_count"] or 0)
                    / max(assessment_stats["total_assessments"] or 1, 1)
                    * 100
                ),
                "requires_review_rate": (
                    (assessment_stats["requires_review_count"] or 0)
                    / max(assessment_stats["total_assessments"] or 1, 1)
                    * 100
                ),
            },
            "learning": {
                "total_mistakes_analyzed": learning_stats["total_mistakes"] or 0,
                "should_have_known_rate": (
                    (learning_stats["should_have_known_count"] or 0)
                    / max(learning_stats["total_mistakes"] or 1, 1)
                    * 100
                ),
                "avg_confidence_adjustment": float(learning_stats["avg_confidence_drop"] or 0),
            },
            "reasoning": {
                "total_explanations": reasoning_stats["total_explanations"] or 0,
                "avg_decision_confidence": float(reasoning_stats["avg_decision_confidence"] or 0),
                "human_review_rate": (
                    (reasoning_stats["human_review_count"] or 0)
                    / max(reasoning_stats["total_explanations"] or 1, 1)
                    * 100
                ),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get self-awareness stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}") from e


# ---------------------------------------------------------------------------
# AI Task Stats & Orchestration
# ---------------------------------------------------------------------------


@router.get("/ai/tasks/stats")
async def get_task_stats():
    """Get AI task system statistics (requires auth)"""
    _app = _get_app()
    integration_layer = getattr(_app.state, "integration_layer", None)
    if not _integration_layer_available() or integration_layer is None:
        raise HTTPException(
            status_code=503, detail="AI Integration Layer not available or not initialized"
        )

    try:
        # Get all tasks
        all_tasks = await integration_layer.list_tasks(limit=1000)

        # Calculate stats
        stats = {
            "total": len(all_tasks),
            "by_status": {},
            "by_priority": {},
            "agents_active": len(integration_layer.agents_registry),
            "execution_queue_size": integration_layer.execution_queue.qsize(),
        }

        for task in all_tasks:
            # Count by status
            status = task.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by priority
            priority = task.get("priority", "unknown")
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

        return {
            "success": True,
            "stats": stats,
            "system_status": "operational",
            "task_executor_running": True,
        }

    except Exception as e:
        logger.error(f"Failed to get task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ai/orchestrate")
async def orchestrate_complex_workflow(
    request: Request, task_description: str, context: dict[str, Any] = {}
):
    """
    Execute complex multi-stage workflow using LangGraph orchestration
    This is for sophisticated tasks that need multi-agent coordination
    """
    _app = _get_app()
    if not hasattr(_app.state, "langgraph_orchestrator") or not _app.state.langgraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")

    try:
        orchestrator = _app.state.langgraph_orchestrator

        result = await orchestrator.execute(task_description=task_description, context=context)

        return {"success": True, "result": result, "message": "Workflow orchestrated successfully"}

    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/ai/analyze")
async def ai_analyze(request: Request, payload: AIAnalyzeRequest = Body(...)):
    """
    AI analysis endpoint for weathercraft-erp and other frontends.
    Accepts JSON body with agent, action, data, and context fields.
    Routes to the appropriate agent or orchestrator.
    """
    _app = _get_app()
    try:
        agent_name = payload.agent
        action = payload.action
        data = payload.data
        context = payload.context

        # Build task description from agent and action
        task_description = f"{agent_name}: {action}"
        if data:
            task_description += f" with data: {json.dumps(data)[:200]}"

        # Try to use LangGraph orchestrator if available
        if hasattr(_app.state, "langgraph_orchestrator") and _app.state.langgraph_orchestrator:
            orchestrator = _app.state.langgraph_orchestrator
            result = await orchestrator.execute(
                agent_name=agent_name,
                prompt=task_description,
                context={**context, "action": action, "data": data},
            )
            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": result,
                "message": "Analysis completed via orchestrator",
            }

        # Fallback: Use module-level agent executor singleton
        try:
            from agent_executor import executor as agent_executor_singleton

            if agent_executor_singleton:
                result = await agent_executor_singleton.execute(
                    agent_name=agent_name, task={"action": action, "data": data, "context": context}
                )
                return {
                    "success": True,
                    "agent": agent_name,
                    "action": action,
                    "result": result,
                    "message": "Analysis completed via agent executor",
                }
        except (ImportError, Exception) as e:
            logger.warning(f"Agent executor fallback failed: {e}")

        # No orchestrator available - queue task for later processing instead of mock response
        logger.warning(
            f"No orchestrator/executor available for agent {agent_name}, queueing for async processing"
        )

        # Queue the task to ai_autonomous_tasks for later execution
        try:
            pool = get_pool()
            task_id = str(uuid.uuid4())
            tenant_id = (
                request.headers.get(config.tenant.header_name)
                or context.get("tenant_id")
                or config.tenant.default_tenant_id
            )
            tenant_uuid: str | None = None
            if tenant_id:
                try:
                    tenant_uuid = str(uuid.UUID(str(tenant_id)))
                except (ValueError, TypeError, AttributeError):
                    tenant_uuid = None

            agent_row = await pool.fetchrow(
                "SELECT id FROM ai_agents WHERE id::text = $1 OR name = $1 LIMIT 1",
                agent_name,
            )
            agent_uuid = str(agent_row["id"]) if agent_row else None

            await pool.execute(
                """
                INSERT INTO ai_autonomous_tasks (
                    id,
                    title,
                    task_type,
                    priority,
                    status,
                    trigger_type,
                    trigger_condition,
                    agent_id,
                    tenant_id,
                    created_at
                )
                VALUES ($1, $2, $3, $4, 'pending', $5, $6::jsonb, $7, $8, NOW())
                """,
                task_id,
                f"{agent_name}.{action}",
                "ai_analyze",
                "medium",
                "ai_analyze",
                json.dumps(
                    {
                        "agent": agent_name,
                        "action": action,
                        "data": data,
                        "context": context,
                    },
                    default=str,
                ),
                agent_uuid,
                tenant_uuid,
            )

            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": {
                    "status": "queued",
                    "task_id": task_id,
                    "message": f"Request queued for async processing (task: {task_id})",
                },
                "message": "Request queued - orchestrator temporarily unavailable",
            }
        except Exception as queue_error:
            logger.error(f"Failed to queue task: {queue_error}")
            raise HTTPException(
                status_code=503,
                detail=f"AI orchestrator unavailable and task queueing failed: {str(queue_error)}",
            ) from queue_error

    except Exception as e:
        logger.error(f"AI analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
