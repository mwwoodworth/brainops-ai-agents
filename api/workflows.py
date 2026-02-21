#!/usr/bin/env python3
"""
Advanced Workflow Engine API
============================
REST API endpoints for the LangGraph-based workflow engine.

Provides:
- Workflow execution (start, resume, cancel)
- Human-in-the-loop approval management
- Workflow status and monitoring
- OODA loop integration
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["Advanced Workflows"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class WorkflowStartRequest(BaseModel):
    """Request to start a new workflow."""

    workflow_type: str = Field(..., description="Type of workflow to start")
    initial_state: dict[str, Any] = Field(default_factory=dict, description="Initial state data")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenancy")
    timeout_seconds: int = Field(300, ge=10, le=3600, description="Execution timeout in seconds")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_type": "customer_onboarding",
                "initial_state": {
                    "customer_id": "cust-123",
                    "customer_name": "Acme Corp",
                    "customer_email": "contact@acme.com",
                },
                "tenant_id": "tenant-abc",
                "timeout_seconds": 300,
            }
        }
    )


class WorkflowResumeRequest(BaseModel):
    """Request to resume a paused workflow."""

    workflow_id: str = Field(..., description="ID of workflow to resume")
    additional_state: Optional[dict[str, Any]] = Field(
        None, description="Additional state to merge"
    )
    timeout_seconds: int = Field(300, ge=10, le=3600, description="Execution timeout")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "workflow_id": "wf-abc123",
                "additional_state": {"metadata": {"human_decision": "approve"}},
                "timeout_seconds": 300,
            }
        }
    )


class ApprovalSubmitRequest(BaseModel):
    """Request to submit human approval."""

    request_id: str = Field(..., description="Approval request ID")
    response: str = Field(..., description="Approval response (must match available options)")
    responded_by: str = Field(..., description="User/system submitting the approval")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "apr-xyz789",
                "response": "approve",
                "responded_by": "user@example.com",
            }
        }
    )


class WorkflowResponse(BaseModel):
    """Standard workflow response."""

    workflow_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    errors: list[dict[str, Any]] = []
    execution_summary: Optional[dict[str, Any]] = None
    can_resume: bool = False
    error: Optional[str] = None


class ApprovalRequest(BaseModel):
    """Human approval request."""

    request_id: str
    workflow_id: str
    node_name: str
    description: str
    context: dict[str, Any] = {}
    options: list[str] = []
    timeout_minutes: int
    created_at: Optional[str] = None


class WorkflowStatusResponse(BaseModel):
    """Workflow status response."""

    workflow_id: str
    workflow_type: Optional[str] = None
    status: str
    current_node: Optional[str] = None
    requires_approval: bool = False
    approval_request_id: Optional[str] = None
    errors: list[dict[str, Any]] = []
    last_updated: Optional[str] = None
    message: Optional[str] = None


class WorkflowTypeInfo(BaseModel):
    """Information about a workflow type."""

    type: str
    breakpoints: list[str] = []


# =============================================================================
# LAZY ENGINE IMPORT (avoid import errors if deps missing)
# =============================================================================

_engine = None


def get_engine():
    """Lazy import and get workflow engine instance."""
    global _engine
    if _engine is None:
        try:
            from langgraph_workflow_engine import get_workflow_engine

            _engine = get_workflow_engine()
        except ImportError as e:
            logger.error(f"Failed to import workflow engine: {e}")
            raise HTTPException(
                status_code=503,
                detail="Workflow engine not available. LangGraph dependencies may be missing.",
            )
    return _engine


# =============================================================================
# WORKFLOW MANAGEMENT ENDPOINTS
# =============================================================================


@router.get("/types", response_model=list[WorkflowTypeInfo])
async def list_workflow_types():
    """
    List available workflow types.

    Returns all registered workflow templates with their breakpoint information.
    """
    try:
        engine = get_engine()
        types = engine.get_available_workflows()
        return [WorkflowTypeInfo(**t) for t in types]
    except Exception as e:
        logger.error(f"Failed to list workflow types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=WorkflowResponse)
async def start_workflow(request: WorkflowStartRequest):
    """
    Start a new workflow execution.

    Initiates a workflow of the specified type with the provided initial state.
    The workflow will execute until completion, a breakpoint, or timeout.

    Supported workflow types:
    - customer_onboarding: Customer onboarding process
    - invoice_collection: Invoice collection workflow
    - lead_qualification: BANT-based lead qualification
    - system_healing: OODA-based error recovery
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.start_workflow(
            workflow_type=request.workflow_type,
            initial_state=request.initial_state,
            tenant_id=request.tenant_id,
            timeout_seconds=request.timeout_seconds,
        )

        if "error" in result and result.get("status") == "failed":
            raise HTTPException(status_code=400, detail=result["error"])

        return WorkflowResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=WorkflowResponse)
async def resume_workflow(request: WorkflowResumeRequest):
    """
    Resume a paused or checkpointed workflow.

    Continues execution from the last saved checkpoint.
    Use this after submitting human approvals or recovering from timeouts.
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.resume_workflow(
            workflow_id=request.workflow_id,
            additional_state=request.additional_state,
            timeout_seconds=request.timeout_seconds,
        )

        if "error" in result and result.get("status") == "failed":
            raise HTTPException(status_code=404, detail=result["error"])

        return WorkflowResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get current status of a workflow.

    Returns the workflow's current state, including whether it needs human approval.
    """
    try:
        engine = get_engine()
        await engine.initialize()

        status = await engine.get_workflow_status(workflow_id)
        return WorkflowStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HUMAN-IN-THE-LOOP ENDPOINTS
# =============================================================================


@router.get("/approvals/pending", response_model=list[ApprovalRequest])
async def get_pending_approvals(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant"),
    workflow_type: Optional[str] = Query(None, description="Filter by workflow type"),
):
    """
    Get all pending human approval requests.

    Returns approval requests that are waiting for human decision.
    Use this to build approval dashboards and notification systems.
    """
    try:
        engine = get_engine()
        await engine.initialize()

        approvals = await engine.get_pending_approvals(
            tenant_id=tenant_id, workflow_type=workflow_type
        )

        return [ApprovalRequest(**a) for a in approvals]

    except Exception as e:
        logger.error(f"Failed to get pending approvals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approvals/submit")
async def submit_approval(request: ApprovalSubmitRequest):
    """
    Submit a human approval decision.

    Responds to a pending approval request with the specified decision.
    The response must be one of the options provided in the approval request.

    After approval, use /workflows/resume to continue the workflow.
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.submit_approval(
            request_id=request.request_id,
            response=request.response,
            responded_by=request.responded_by,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=400, detail=result.get("error", "Failed to submit approval")
            )

        return {
            "success": True,
            "message": f"Approval submitted: {request.response}",
            "workflow_id": result.get("workflow_id"),
            "next_step": "Use /workflows/resume to continue the workflow",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit approval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# WORKFLOW TEMPLATE ENDPOINTS
# =============================================================================


@router.post("/onboarding/start", response_model=WorkflowResponse)
async def start_customer_onboarding(
    customer_id: str,
    customer_name: str,
    customer_email: str,
    tenant_id: Optional[str] = None,
    has_import_data: bool = False,
    training_requested: bool = True,
):
    """
    Start a customer onboarding workflow.

    Convenience endpoint for the customer onboarding workflow type.

    Stages:
    1. Welcome email sent
    2. Account setup
    3. Data import (optional)
    4. Configuration
    5. Training scheduling
    6. First project creation
    7. Success verification
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.start_workflow(
            workflow_type="customer_onboarding",
            initial_state={
                "customer_id": customer_id,
                "customer_name": customer_name,
                "customer_email": customer_email,
                "metadata": {
                    "has_import_data": has_import_data,
                    "training_requested": training_requested,
                },
            },
            tenant_id=tenant_id,
        )

        return WorkflowResponse(**result)

    except Exception as e:
        logger.error(f"Failed to start onboarding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection/start", response_model=WorkflowResponse)
async def start_invoice_collection(
    invoice_id: str, customer_id: str, amount: float, due_date: str, tenant_id: Optional[str] = None
):
    """
    Start an invoice collection workflow.

    Convenience endpoint for the invoice collection workflow type.

    Collection stages (based on days overdue):
    - 1-7 days: Gentle reminder
    - 8-14 days: Follow-up
    - 15-30 days: Payment plan offer
    - 30+ days: Human review/escalation
    - Final notice
    - Collections handoff
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.start_workflow(
            workflow_type="invoice_collection",
            initial_state={
                "invoice_id": invoice_id,
                "customer_id": customer_id,
                "amount": amount,
                "due_date": due_date,
            },
            tenant_id=tenant_id,
        )

        return WorkflowResponse(**result)

    except Exception as e:
        logger.error(f"Failed to start collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qualification/start", response_model=WorkflowResponse)
async def start_lead_qualification(
    lead_id: str,
    lead_name: str,
    lead_email: str,
    company_name: Optional[str] = None,
    company_size: Optional[str] = None,
    budget_range: Optional[str] = None,
    timeline: Optional[str] = None,
    lead_source: str = "website",
    tenant_id: Optional[str] = None,
):
    """
    Start a lead qualification workflow.

    Convenience endpoint for the BANT-based lead qualification workflow.

    BANT Framework:
    - Budget: Can they afford it?
    - Authority: Are they the decision maker?
    - Need: Do they have the problem we solve?
    - Timeline: When do they need it?

    Outcomes:
    - SQL: Sales Qualified Lead (score >= 100)
    - MQL: Marketing Qualified Lead (score >= 60)
    - Disqualified: Does not meet criteria
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.start_workflow(
            workflow_type="lead_qualification",
            initial_state={
                "lead_id": lead_id,
                "lead_name": lead_name,
                "lead_email": lead_email,
                "company_name": company_name,
                "company_size": company_size,
                "budget_range": budget_range,
                "timeline": timeline,
                "lead_source": lead_source,
            },
            tenant_id=tenant_id,
        )

        return WorkflowResponse(**result)

    except Exception as e:
        logger.error(f"Failed to start qualification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/healing/start", response_model=WorkflowResponse)
async def start_system_healing(
    error_id: str,
    error_type: str,
    error_message: str,
    component: str,
    severity: str = "medium",
    tenant_id: Optional[str] = None,
):
    """
    Start a system healing workflow.

    OODA-based automatic error recovery workflow.

    OODA Loop:
    - Observe: Gather error information
    - Orient: Diagnose root cause
    - Decide: Select healing strategy
    - Act: Execute fix
    - Verify: Confirm healing
    - Learn: Record pattern

    Supported error types:
    - connection_error: Network connectivity issues
    - timeout: Resource exhaustion
    - authentication_error: Credential issues
    - database_error: Database connectivity
    - memory_error: Memory exhaustion
    """
    try:
        engine = get_engine()
        await engine.initialize()

        result = await engine.start_workflow(
            workflow_type="system_healing",
            initial_state={
                "error_id": error_id,
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "severity": severity,
            },
            tenant_id=tenant_id,
        )

        return WorkflowResponse(**result)

    except Exception as e:
        logger.error(f"Failed to start healing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# HEALTH & MONITORING
# =============================================================================


@router.get("/health")
async def workflow_engine_health():
    """
    Check workflow engine health.

    Returns the status of the workflow engine and its dependencies.
    """
    try:
        engine = get_engine()
        initialized = await engine.initialize()

        # Check for pending approvals as a sign of active workflows
        pending = await engine.get_pending_approvals()

        return {
            "status": "healthy" if initialized else "degraded",
            "engine_initialized": initialized,
            "pending_approvals": len(pending),
            "available_workflow_types": list(engine._templates.keys()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@router.get("/metrics")
async def workflow_metrics(
    tenant_id: Optional[str] = Query(None), hours: int = Query(24, ge=1, le=168)
):
    """
    Get workflow execution metrics.

    Returns aggregated metrics for workflow executions.
    """
    try:
        from langgraph_workflow_engine import get_workflow_pool

        pool = await get_workflow_pool()
        if not pool:
            return {"error": "Database not available"}

        async with pool.acquire() as conn:
            # Get checkpoint stats (active workflows)
            checkpoint_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as active_workflows,
                    COUNT(DISTINCT workflow_type) as workflow_types
                FROM workflow_checkpoints
                WHERE created_at > NOW() - INTERVAL '1 hour' * $1
            """,
                hours,
            )

            # Get approval stats
            approval_stats = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'approved') as approved,
                    COUNT(*) FILTER (WHERE status = 'rejected') as rejected,
                    COUNT(*) FILTER (WHERE status = 'timeout') as timeout
                FROM workflow_approval_requests
                WHERE created_at > NOW() - INTERVAL '1 hour' * $1
            """,
                hours,
            )

            return {
                "period_hours": hours,
                "active_workflows": checkpoint_stats["active_workflows"] if checkpoint_stats else 0,
                "workflow_types_in_use": checkpoint_stats["workflow_types"]
                if checkpoint_stats
                else 0,
                "approvals": {
                    "pending": approval_stats["pending"] if approval_stats else 0,
                    "approved": approval_stats["approved"] if approval_stats else 0,
                    "rejected": approval_stats["rejected"] if approval_stats else 0,
                    "timeout": approval_stats["timeout"] if approval_stats else 0,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {"error": str(e)}
