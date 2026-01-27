#!/usr/bin/env python3
"""
PREDICTIVE EXECUTION API
========================

API endpoints for the Predictive Task Executor system.

Endpoints:
- POST /predictive/execute - Execute a single prediction
- POST /predictive/execute-batch - Execute multiple predictions
- POST /predictive/evaluate - Evaluate a prediction without executing
- GET /predictive/stats - Get execution statistics
- GET /predictive/recent - Get recent executions
- POST /predictive/learn - Record accuracy feedback
- POST /predictive/from-ooda - Execute OODA speculations
- POST /predictive/from-proactive - Execute proactive recommendations
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from predictive_executor import (
    PredictedTask,
    PredictionSource,
    get_predictive_executor,
    get_ooda_integration,
    get_proactive_intel_integration,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictive", tags=["Predictive Execution"])


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictionRequest(BaseModel):
    """Request to execute a prediction"""
    task_type: str = Field(..., description="Type of task to execute")
    description: str = Field(default="", description="Human-readable description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    source: str = Field(default="api", description="Source of prediction")
    dependencies: list[str] = Field(default_factory=list, description="Dependency task IDs")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BatchPredictionRequest(BaseModel):
    """Request to execute multiple predictions"""
    predictions: list[PredictionRequest]


class OODASpeculationsRequest(BaseModel):
    """Request to execute OODA speculations"""
    speculations: list[dict[str, Any]]


class ProactiveRecommendationsRequest(BaseModel):
    """Request to execute proactive recommendations"""
    recommendations: list[dict[str, Any]]


class AccuracyFeedbackRequest(BaseModel):
    """Request to record accuracy feedback"""
    execution_id: str = Field(..., description="Execution ID to update")
    was_accurate: bool = Field(..., description="Whether prediction was accurate")
    feedback: Optional[str] = Field(None, description="Optional feedback text")


class ExecutionResponse(BaseModel):
    """Response from execution request"""
    executed: bool
    execution_id: Optional[str] = None
    skipped: bool = False
    decision: Optional[str] = None
    reason: Optional[str] = None
    result: Optional[dict] = None
    execution_time_ms: Optional[int] = None
    evaluation: Optional[dict] = None
    error: Optional[str] = None


class StatsResponse(BaseModel):
    """Response with execution statistics"""
    period: str
    overall: dict[str, Any]
    by_task_type: list[dict[str, Any]]
    by_decision: list[dict[str, Any]]
    in_memory: dict[str, Any]


# =============================================================================
# Helper Functions
# =============================================================================

def _source_from_string(source: str) -> PredictionSource:
    """Convert string to PredictionSource enum"""
    source_map = {
        "api": PredictionSource.PATTERN_MATCH,
        "ooda": PredictionSource.OODA_SPECULATION,
        "ooda_speculation": PredictionSource.OODA_SPECULATION,
        "proactive": PredictionSource.PROACTIVE_INTEL,
        "proactive_intel": PredictionSource.PROACTIVE_INTEL,
        "pattern": PredictionSource.PATTERN_MATCH,
        "pattern_match": PredictionSource.PATTERN_MATCH,
        "schedule": PredictionSource.SCHEDULE_BASED,
        "schedule_based": PredictionSource.SCHEDULE_BASED,
        "user": PredictionSource.USER_BEHAVIOR,
        "user_behavior": PredictionSource.USER_BEHAVIOR,
        "system": PredictionSource.SYSTEM_STATE,
        "system_state": PredictionSource.SYSTEM_STATE,
    }
    return source_map.get(source.lower(), PredictionSource.PATTERN_MATCH)


def _request_to_predicted_task(request: PredictionRequest, task_id: str) -> PredictedTask:
    """Convert API request to PredictedTask"""
    return PredictedTask(
        id=task_id,
        source=_source_from_string(request.source),
        task_type=request.task_type,
        description=request.description,
        confidence=request.confidence,
        parameters=request.parameters,
        dependencies=request.dependencies,
        metadata=request.metadata,
    )


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/execute", response_model=ExecutionResponse)
async def execute_prediction(request: PredictionRequest):
    """
    Execute a single predicted task.

    The prediction will be evaluated for safety before execution.
    Blocked actions (delete, send_email, etc.) will never be executed.

    Example:
    ```json
    {
        "task_type": "fetch_customer_data",
        "description": "Prefetch customer data",
        "confidence": 0.92,
        "parameters": {"customer_id": 12345},
        "source": "pattern"
    }
    ```
    """
    try:
        executor = get_predictive_executor()

        import uuid
        task_id = f"api_{uuid.uuid4().hex[:12]}"
        prediction = _request_to_predicted_task(request, task_id)

        result = await executor.execute_prediction(prediction)
        return ExecutionResponse(**result)

    except Exception as e:
        logger.error(f"Prediction execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-batch")
async def execute_batch_predictions(request: BatchPredictionRequest):
    """
    Execute multiple predicted tasks in batch.

    Tasks are sorted by confidence and executed respecting rate limits.
    """
    try:
        executor = get_predictive_executor()

        import uuid
        predictions = [
            _request_to_predicted_task(req, f"batch_{uuid.uuid4().hex[:8]}")
            for req in request.predictions
        ]

        results = await executor.execute_batch(predictions)

        return {
            "total": len(results),
            "executed": sum(1 for r in results if r.get("executed")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "failed": sum(1 for r in results if r.get("error")),
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_prediction(request: PredictionRequest):
    """
    Evaluate a prediction without executing it.

    Returns the safety evaluation including:
    - Whether it would be executed
    - Risk level and score
    - Safety checks performed
    """
    try:
        executor = get_predictive_executor()

        import uuid
        task_id = f"eval_{uuid.uuid4().hex[:12]}"
        prediction = _request_to_predicted_task(request, task_id)

        evaluation = await executor.evaluate_prediction(prediction)

        return {
            "prediction": prediction.to_dict(),
            "evaluation": evaluation.to_dict(),
            "would_execute": evaluation.should_execute
        }

    except Exception as e:
        logger.error(f"Prediction evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_execution_stats():
    """
    Get predictive execution statistics.

    Returns metrics including:
    - Total executions, completions, failures, skips
    - Accuracy rate (when feedback is provided)
    - Average confidence and risk scores
    - Breakdown by task type and decision
    """
    try:
        executor = get_predictive_executor()
        stats = await executor.get_execution_stats()
        return stats

    except Exception as e:
        logger.error(f"Stats retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recent")
async def get_recent_executions(
    limit: int = Query(default=20, ge=1, le=100, description="Number of records to return")
):
    """
    Get recent predictive executions.

    Returns the most recent executions with their results and evaluations.
    """
    try:
        executor = get_predictive_executor()
        executions = await executor.get_recent_executions(limit)

        return {
            "count": len(executions),
            "executions": executions
        }

    except Exception as e:
        logger.error(f"Recent executions retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learn")
async def record_accuracy_feedback(request: AccuracyFeedbackRequest):
    """
    Record accuracy feedback for a predictive execution.

    This helps the system learn which predictions are actually useful.
    Call this endpoint when you know whether a prediction matched
    the actual need.

    Example:
    ```json
    {
        "execution_id": "pexec_abc123",
        "was_accurate": true,
        "feedback": "User did request this data"
    }
    ```
    """
    try:
        executor = get_predictive_executor()
        await executor.learn_from_outcome(
            request.execution_id,
            request.was_accurate,
            request.feedback
        )

        return {
            "status": "recorded",
            "execution_id": request.execution_id,
            "was_accurate": request.was_accurate
        }

    except Exception as e:
        logger.error(f"Feedback recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-ooda")
async def execute_ooda_speculations(request: OODASpeculationsRequest):
    """
    Execute speculations from the OODA system.

    Converts OODA speculations to predictions and executes them
    through the safety evaluation pipeline.

    Example:
    ```json
    {
        "speculations": [
            {
                "type": "fetch_system_health",
                "probability": 0.85,
                "params": {"service": "api"}
            }
        ]
    }
    ```
    """
    try:
        integration = get_ooda_integration()
        results = await integration.execute_ooda_speculations(request.speculations)

        return {
            "source": "ooda_speculation",
            "total": len(results),
            "executed": sum(1 for r in results if r.get("executed")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "results": results
        }

    except Exception as e:
        logger.error(f"OODA speculation execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-proactive")
async def execute_proactive_recommendations(request: ProactiveRecommendationsRequest):
    """
    Execute recommendations from the Proactive Intelligence system.

    Converts proactive recommendations to predictions and executes them
    through the safety evaluation pipeline.

    Example:
    ```json
    {
        "recommendations": [
            {
                "action": "scale_monitoring",
                "reason": "CPU trending up",
                "confidence": 0.82,
                "prediction_id": "pred_123"
            }
        ]
    }
    ```
    """
    try:
        integration = get_proactive_intel_integration()
        results = await integration.execute_proactive_recommendations(
            request.recommendations
        )

        return {
            "source": "proactive_intelligence",
            "total": len(results),
            "executed": sum(1 for r in results if r.get("executed")),
            "skipped": sum(1 for r in results if r.get("skipped")),
            "results": results
        }

    except Exception as e:
        logger.error(f"Proactive recommendation execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/config")
async def get_executor_config():
    """
    Get the current predictive executor configuration.

    Returns thresholds and limits that control execution behavior.
    """
    try:
        executor = get_predictive_executor()

        return {
            "confidence_threshold": executor.confidence_threshold,
            "risk_threshold": executor.risk_threshold,
            "max_concurrent_executions": executor.max_concurrent_executions,
            "rate_limit_per_minute": executor.rate_limit_per_minute,
            "blocked_actions": list(executor.BLOCKED_ACTIONS),
            "safe_actions": list(executor.SAFE_ACTIONS),
            "registered_executors": list(executor.task_executors.keys())
        }

    except Exception as e:
        logger.error(f"Config retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def predictive_executor_health():
    """
    Health check for the predictive execution system.
    """
    try:
        executor = get_predictive_executor()

        return {
            "status": "healthy",
            "active_executions": len(executor.active_executions),
            "pending_queue": len(executor.pending_queue),
            "total_executions": executor.total_executions,
            "accuracy": {
                "accurate": executor.accurate_predictions,
                "inaccurate": executor.inaccurate_predictions
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
