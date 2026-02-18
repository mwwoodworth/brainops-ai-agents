#!/usr/bin/env python3
"""
LangGraph Advanced Workflow Engine
==================================
Enhanced LangGraph orchestrator with cutting-edge workflow patterns:
- State machine patterns for complex workflows
- Conditional branching based on AI decisions
- Checkpoint and resume capability for long-running tasks
- OODA (Observe, Orient, Decide, Act) loop integration
- Human-in-the-loop breakpoints for critical decisions

Workflow Templates:
- Customer Onboarding
- Invoice Collection
- Lead Qualification
- System Healing
"""

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, TypedDict

from database.async_connection import get_tenant_pool

# LangGraph imports (conditional)
try:
    from langgraph.graph import END, StateGraph
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "END"
    BaseCheckpointSaver = object

logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

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


FEATURE_LANGGRAPH_PARALLEL_NODES = _env_flag("ENABLE_LANGGRAPH_PARALLEL_NODES", False)
FEATURE_LANGGRAPH_WORKFLOW_COMPOSITION = _env_flag(
    "ENABLE_LANGGRAPH_WORKFLOW_COMPOSITION", False
)
FEATURE_LANGGRAPH_ERROR_RECOVERY = _env_flag("ENABLE_LANGGRAPH_ERROR_RECOVERY", False)
FEATURE_LANGGRAPH_WORKFLOW_VERSIONING = _env_flag(
    "ENABLE_LANGGRAPH_WORKFLOW_VERSIONING", False
)
FEATURE_LANGGRAPH_STATE_CHECKPOINTS = _env_flag(
    "ENABLE_LANGGRAPH_STATE_CHECKPOINTS", False
)


def _resolve_tenant_id(candidate: Optional[str] = None) -> str:
    """Resolve tenant id for scoped DB operations."""
    tenant_id = (candidate or "").strip()
    return tenant_id or DEFAULT_TENANT_ID


async def get_workflow_pool():
    """Get tenant-scoped pool used by the workflow engine."""
    return get_tenant_pool(_resolve_tenant_id())


@asynccontextmanager
async def get_workflow_connection(tenant_id: Optional[str] = None):
    """Get a tenant-scoped database connection for workflow persistence."""
    scoped_pool = get_tenant_pool(_resolve_tenant_id(tenant_id))
    conn = await scoped_pool.acquire()
    try:
        yield conn
    finally:
        await scoped_pool.release(conn)


# =============================================================================
# WORKFLOW STATE DEFINITIONS
# =============================================================================

class WorkflowStatus(str, Enum):
    """Status of a workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"  # Waiting for human input
    WAITING = "waiting"  # Waiting for external event
    CHECKPOINT = "checkpoint"  # Saved checkpoint
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OODAPhase(str, Enum):
    """OODA Loop phases."""
    OBSERVE = "observe"
    ORIENT = "orient"
    DECIDE = "decide"
    ACT = "act"


class HumanApprovalStatus(str, Enum):
    """Status of human approval requests."""
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"


@dataclass
class WorkflowCheckpoint:
    """Checkpoint for workflow state persistence."""
    checkpoint_id: str
    workflow_id: str
    workflow_type: str
    state_data: dict
    current_node: str
    created_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class HumanApprovalRequest:
    """Request for human approval at a breakpoint."""
    request_id: str
    workflow_id: str
    node_name: str
    description: str
    context: dict
    options: list[str]  # e.g., ["approve", "reject", "modify"]
    timeout_minutes: int
    created_at: datetime
    status: HumanApprovalStatus = HumanApprovalStatus.PENDING
    response: Optional[str] = None
    responded_by: Optional[str] = None
    responded_at: Optional[datetime] = None


# =============================================================================
# BASE WORKFLOW STATE
# =============================================================================

class BaseWorkflowState(TypedDict, total=False):
    """Base state shared by all workflows."""
    # Core identification
    workflow_id: str
    workflow_type: str
    workflow_version: str
    tenant_id: str
    parent_workflow_id: Optional[str]
    child_workflow_ids: list[str]

    # Status tracking
    status: str
    current_node: str
    previous_nodes: list[str]

    # OODA loop state
    ooda_phase: str
    observations: list[dict]
    orientation: dict
    decisions: list[dict]
    actions_taken: list[dict]

    # Human-in-the-loop
    requires_approval: bool
    approval_status: str
    approval_request_id: Optional[str]

    # Error handling
    errors: list[dict]
    retry_count: int
    max_retries: int
    error_recovery_node: Optional[str]
    error_recovery_action: Optional[str]

    # Timing
    started_at: str
    last_updated_at: str
    deadline: Optional[str]

    # Results
    result: Optional[dict]
    parallel_results: dict[str, Any]
    metadata: dict


# =============================================================================
# CHECKPOINT SAVER FOR WORKFLOW PERSISTENCE
# =============================================================================

class PostgresCheckpointSaver:
    """
    Checkpoint saver that persists workflow state to PostgreSQL.
    Enables pause/resume of long-running workflows.
    """

    def __init__(self):
        self._initialized = False

    async def initialize(self) -> bool:
        """Verify checkpoint tables exist (no DDL - agent_worker has no DDL perms)."""
        if self._initialized:
            return True

        try:
            pool = await get_workflow_pool()
            if not pool:
                logger.warning("No database pool available for checkpoints")
                return False

            # Verify required tables exist instead of creating them
            from database.verify_tables import verify_tables_async
            tables_ok = await verify_tables_async(
                ["workflow_checkpoints", "workflow_approval_requests"],
                pool,
                module_name="langgraph_workflow_engine",
            )
            if not tables_ok:
                logger.error(
                    "Workflow tables missing - run migrations to create "
                    "workflow_checkpoints and workflow_approval_requests"
                )
                return False

            self._initialized = True
            logger.info("PostgresCheckpointSaver initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize checkpoint saver: {e}")
            return False

    async def save_checkpoint(
        self,
        workflow_id: str,
        workflow_type: str,
        state: dict,
        current_node: str,
        metadata: dict = None,
        tenant_id: str | None = None,
    ) -> Optional[str]:
        """Save workflow checkpoint."""
        if not await self.initialize():
            return None

        try:
            resolved_tenant = _resolve_tenant_id(
                tenant_id or state.get("tenant_id") or (metadata or {}).get("tenant_id")
            )
            async with get_workflow_connection(resolved_tenant) as conn:
                # Upsert checkpoint
                row = await conn.fetchrow("""
                    INSERT INTO workflow_checkpoints
                        (workflow_id, workflow_type, state_data, current_node, metadata)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (workflow_id) DO UPDATE SET
                        state_data = EXCLUDED.state_data,
                        current_node = EXCLUDED.current_node,
                        metadata = EXCLUDED.metadata,
                        created_at = NOW()
                    RETURNING id
                """, workflow_id, workflow_type, json.dumps(state, default=str),
                    current_node, json.dumps(metadata or {}))

                checkpoint_id = str(row['id'])
                logger.debug(f"Saved checkpoint {checkpoint_id} for workflow {workflow_id}")
                return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    async def load_checkpoint(
        self,
        workflow_id: str,
        tenant_id: str | None = None,
    ) -> Optional[WorkflowCheckpoint]:
        """Load workflow checkpoint."""
        if not await self.initialize():
            return None

        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                row = await conn.fetchrow("""
                    SELECT id, workflow_id, workflow_type, state_data,
                           current_node, created_at, metadata
                    FROM workflow_checkpoints
                    WHERE workflow_id = $1
                """, workflow_id)

                if not row:
                    return None

                raw_state_data = row["state_data"]
                raw_metadata = row["metadata"]
                parsed_state = (
                    json.loads(raw_state_data)
                    if isinstance(raw_state_data, str)
                    else (raw_state_data or {})
                )
                parsed_metadata = (
                    json.loads(raw_metadata)
                    if isinstance(raw_metadata, str)
                    else (raw_metadata or {})
                )

                return WorkflowCheckpoint(
                    checkpoint_id=str(row['id']),
                    workflow_id=row['workflow_id'],
                    workflow_type=row['workflow_type'],
                    state_data=parsed_state,
                    current_node=row['current_node'],
                    created_at=row['created_at'],
                    metadata=parsed_metadata,
                )

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def delete_checkpoint(self, workflow_id: str, tenant_id: str | None = None) -> bool:
        """Delete workflow checkpoint after completion."""
        if not await self.initialize():
            return False

        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                await conn.execute("""
                    DELETE FROM workflow_checkpoints WHERE workflow_id = $1
                """, workflow_id)
                return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False


# =============================================================================
# HUMAN-IN-THE-LOOP MANAGER
# =============================================================================

class HumanInTheLoopManager:
    """
    Manages human approval breakpoints in workflows.
    Allows pausing workflows for human review and resuming after approval.
    """

    def __init__(self, checkpoint_saver: PostgresCheckpointSaver):
        self.checkpoint_saver = checkpoint_saver

    async def request_approval(
        self,
        workflow_id: str,
        node_name: str,
        description: str,
        context: dict,
        options: list[str] = None,
        timeout_minutes: int = 60,
        tenant_id: str | None = None,
    ) -> HumanApprovalRequest:
        """Create an approval request and pause the workflow."""
        options = options or ["approve", "reject"]

        try:
            resolved_tenant = _resolve_tenant_id(tenant_id or context.get("tenant_id"))
            async with get_workflow_connection(resolved_tenant) as conn:
                row = await conn.fetchrow("""
                    INSERT INTO workflow_approval_requests
                        (workflow_id, node_name, description, context, options, timeout_minutes)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id, created_at
                """, workflow_id, node_name, description,
                    json.dumps(context, default=str), json.dumps(options), timeout_minutes)

                return HumanApprovalRequest(
                    request_id=str(row['id']),
                    workflow_id=workflow_id,
                    node_name=node_name,
                    description=description,
                    context=context,
                    options=options,
                    timeout_minutes=timeout_minutes,
                    created_at=row['created_at']
                )

        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            raise

    async def get_approval_status(
        self,
        request_id: str,
        tenant_id: str | None = None,
    ) -> Optional[HumanApprovalRequest]:
        """Get current status of an approval request."""
        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM workflow_approval_requests WHERE id = $1
                """, request_id)

                if not row:
                    return None

                # Check for timeout
                if row['status'] == 'pending':
                    created = row['created_at']
                    timeout = timedelta(minutes=row['timeout_minutes'])
                    if datetime.now(timezone.utc) > created + timeout:
                        await self._update_approval_status(
                            request_id,
                            "timeout",
                            tenant_id=tenant_id,
                        )
                        return await self.get_approval_status(request_id, tenant_id=tenant_id)

                parsed_context = (
                    json.loads(row["context"])
                    if isinstance(row["context"], str)
                    else (row["context"] or {})
                )
                parsed_options = (
                    json.loads(row["options"])
                    if isinstance(row["options"], str)
                    else (row["options"] or [])
                )

                return HumanApprovalRequest(
                    request_id=str(row['id']),
                    workflow_id=row['workflow_id'],
                    node_name=row['node_name'],
                    description=row['description'],
                    context=parsed_context,
                    options=parsed_options,
                    timeout_minutes=row['timeout_minutes'],
                    created_at=row['created_at'],
                    status=HumanApprovalStatus(row['status']),
                    response=row['response'],
                    responded_by=row['responded_by'],
                    responded_at=row['responded_at']
                )

        except Exception as e:
            logger.error(f"Failed to get approval status: {e}")
            return None

    async def submit_approval(
        self,
        request_id: str,
        response: str,
        responded_by: str,
        tenant_id: str | None = None,
    ) -> bool:
        """Submit human response to an approval request."""
        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                # Validate response against options
                row = await conn.fetchrow("""
                    SELECT options FROM workflow_approval_requests WHERE id = $1
                """, request_id)

                if not row:
                    logger.error(f"Approval request {request_id} not found")
                    return False

                options = (
                    json.loads(row["options"])
                    if isinstance(row["options"], str)
                    else (row["options"] or [])
                )
                if options and response not in options:
                    logger.error(f"Invalid response '{response}'. Options: {options}")
                    return False

                # Update the request
                status = 'approved' if response == 'approve' else 'rejected'
                await conn.execute("""
                    UPDATE workflow_approval_requests
                    SET status = $1, response = $2, responded_by = $3, responded_at = NOW()
                    WHERE id = $4
                """, status, response, responded_by, request_id)

                logger.info(f"Approval request {request_id} {status} by {responded_by}")
                return True

        except Exception as e:
            logger.error(f"Failed to submit approval: {e}")
            return False

    async def _update_approval_status(
        self,
        request_id: str,
        status: str,
        tenant_id: str | None = None,
    ) -> None:
        """Update approval request status."""
        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                await conn.execute("""
                    UPDATE workflow_approval_requests SET status = $1 WHERE id = $2
                """, status, request_id)
        except Exception as e:
            logger.error(f"Failed to update approval status: {e}")

    async def get_pending_approvals(
        self,
        tenant_id: str = None,
        workflow_type: str = None,
        limit: int = 50
    ) -> list[HumanApprovalRequest]:
        """Get all pending approval requests."""
        try:
            async with get_workflow_connection(_resolve_tenant_id(tenant_id)) as conn:
                query = """
                    SELECT ar.*, wc.workflow_type
                    FROM workflow_approval_requests ar
                    LEFT JOIN workflow_checkpoints wc ON ar.workflow_id = wc.workflow_id
                    WHERE ar.status = 'pending'
                """
                params = []

                if workflow_type:
                    query += f" AND wc.workflow_type = ${len(params) + 1}"
                    params.append(workflow_type)

                query += f" ORDER BY ar.created_at ASC LIMIT ${len(params) + 1}"
                params.append(limit)

                rows = await conn.fetch(query, *params)

                return [
                    HumanApprovalRequest(
                        request_id=str(row['id']),
                        workflow_id=row['workflow_id'],
                        node_name=row['node_name'],
                        description=row['description'],
                        context=(
                            json.loads(row["context"])
                            if isinstance(row["context"], str)
                            else (row["context"] or {})
                        ),
                        options=(
                            json.loads(row["options"])
                            if isinstance(row["options"], str)
                            else (row["options"] or [])
                        ),
                        timeout_minutes=row['timeout_minutes'],
                        created_at=row['created_at'],
                        status=HumanApprovalStatus(row['status'])
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get pending approvals: {e}")
            return []


# =============================================================================
# OODA LOOP INTEGRATION
# =============================================================================

class OODALoopExecutor:
    """
    Implements the OODA (Observe, Orient, Decide, Act) loop pattern.
    This provides a structured approach to decision-making in workflows.
    """

    def __init__(self, ai_core=None):
        self.ai_core = ai_core

    async def observe(self, state: BaseWorkflowState, context: dict) -> dict:
        """
        OBSERVE phase: Gather information about the current situation.
        Returns observations about the environment, data, and constraints.
        """
        observations = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "workflow_status": state.get("status", "unknown"),
            "current_node": state.get("current_node", "unknown"),
            "retry_count": state.get("retry_count", 0),
            "context": context,
            "errors": state.get("errors", [])
        }

        # If AI core available, use it for deeper observation
        if self.ai_core:
            try:
                ai_observations = await self.ai_core.analyze_context(context)
                observations["ai_insights"] = ai_observations
            except Exception as e:
                logger.warning(f"AI observation failed: {e}")

        return observations

    async def orient(self, state: BaseWorkflowState, observations: dict) -> dict:
        """
        ORIENT phase: Analyze observations and form understanding.
        Returns orientation analysis including threats, opportunities, priorities.
        """
        orientation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "workflow_orientation",
            "observations_count": len(observations),
            "priorities": [],
            "risks": [],
            "opportunities": []
        }

        # Analyze errors for risks
        errors = observations.get("errors", [])
        if errors:
            orientation["risks"].append({
                "type": "execution_errors",
                "severity": "high" if len(errors) > 2 else "medium",
                "count": len(errors)
            })

        # Analyze retry count
        retry_count = observations.get("retry_count", 0)
        if retry_count > 0:
            orientation["risks"].append({
                "type": "retry_risk",
                "severity": "high" if retry_count >= 3 else "low",
                "count": retry_count
            })

        # If AI core available, use it for deeper analysis
        if self.ai_core:
            try:
                ai_orientation = await self.ai_core.synthesize_analysis(observations)
                orientation["ai_analysis"] = ai_orientation
            except Exception as e:
                logger.warning(f"AI orientation failed: {e}")

        return orientation

    async def decide(
        self,
        state: BaseWorkflowState,
        orientation: dict,
        available_actions: list[str]
    ) -> dict:
        """
        DECIDE phase: Select best course of action based on orientation.
        Returns decision with selected action and reasoning.
        """
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "available_actions": available_actions,
            "selected_action": None,
            "reasoning": "",
            "confidence": 0.0
        }

        # Simple heuristic decision making
        risks = orientation.get("risks", [])
        high_risks = [r for r in risks if r.get("severity") == "high"]

        if high_risks:
            # High risk situation - be conservative
            if "escalate" in available_actions:
                decision["selected_action"] = "escalate"
                decision["reasoning"] = f"High risk detected: {high_risks[0].get('type')}"
                decision["confidence"] = 0.85
            elif "retry" in available_actions:
                decision["selected_action"] = "retry"
                decision["reasoning"] = "Attempting retry before escalation"
                decision["confidence"] = 0.6
            else:
                decision["selected_action"] = available_actions[0] if available_actions else "fail"
                decision["reasoning"] = "No safe action available, using default"
                decision["confidence"] = 0.3
        else:
            # Normal situation - proceed
            if "proceed" in available_actions:
                decision["selected_action"] = "proceed"
                decision["reasoning"] = "No significant risks detected"
                decision["confidence"] = 0.9
            elif "complete" in available_actions:
                decision["selected_action"] = "complete"
                decision["reasoning"] = "Ready for completion"
                decision["confidence"] = 0.95
            else:
                decision["selected_action"] = available_actions[0] if available_actions else "proceed"
                decision["reasoning"] = "Using first available action"
                decision["confidence"] = 0.7

        # If AI core available, use it for better decisions
        if self.ai_core:
            try:
                ai_decision = await self.ai_core.make_decision(
                    context={
                        "state": state,
                        "orientation": orientation,
                        "available_actions": available_actions
                    }
                )
                if ai_decision.get("confidence", 0) > decision["confidence"]:
                    decision = ai_decision
            except Exception as e:
                logger.warning(f"AI decision failed, using heuristic: {e}")

        return decision

    async def act(
        self,
        state: BaseWorkflowState,
        decision: dict,
        action_handlers: dict[str, Callable]
    ) -> dict:
        """
        ACT phase: Execute the decided action.
        Returns action result.
        """
        action = decision.get("selected_action")

        if not action:
            return {
                "success": False,
                "error": "No action selected",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        handler = action_handlers.get(action)

        if not handler:
            return {
                "success": False,
                "error": f"No handler for action: {action}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        try:
            result = await handler(state, decision)
            return {
                "success": True,
                "action": action,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return {
                "success": False,
                "action": action,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    async def execute_loop(
        self,
        state: BaseWorkflowState,
        context: dict,
        available_actions: list[str],
        action_handlers: dict[str, Callable]
    ) -> tuple[dict, BaseWorkflowState]:
        """
        Execute full OODA loop cycle.
        Returns (action_result, updated_state).
        """
        # Track phases in state
        state["ooda_phase"] = OODAPhase.OBSERVE.value

        # OBSERVE
        observations = await self.observe(state, context)
        if "observations" not in state:
            state["observations"] = []
        state["observations"].append(observations)

        # ORIENT
        state["ooda_phase"] = OODAPhase.ORIENT.value
        orientation = await self.orient(state, observations)
        state["orientation"] = orientation

        # DECIDE
        state["ooda_phase"] = OODAPhase.DECIDE.value
        decision = await self.decide(state, orientation, available_actions)
        if "decisions" not in state:
            state["decisions"] = []
        state["decisions"].append(decision)

        # ACT
        state["ooda_phase"] = OODAPhase.ACT.value
        action_result = await self.act(state, decision, action_handlers)
        if "actions_taken" not in state:
            state["actions_taken"] = []
        state["actions_taken"].append(action_result)

        return action_result, state


# =============================================================================
# WORKFLOW TEMPLATE BASE CLASS
# =============================================================================

class WorkflowTemplate(ABC):
    """
    Base class for workflow templates.
    Provides standard patterns for building LangGraph workflows.
    """

    def __init__(
        self,
        workflow_type: str,
        checkpoint_saver: PostgresCheckpointSaver = None,
        hitl_manager: HumanInTheLoopManager = None,
        ooda_executor: OODALoopExecutor = None,
        workflow_registry: Optional[dict[str, type["WorkflowTemplate"]]] = None,
    ):
        self.workflow_type = workflow_type
        self.checkpoint_saver = checkpoint_saver or PostgresCheckpointSaver()
        self.hitl_manager = hitl_manager or HumanInTheLoopManager(self.checkpoint_saver)
        self.ooda_executor = ooda_executor or OODALoopExecutor()
        self.workflow_registry = workflow_registry or {}
        self._graph = None

    @abstractmethod
    def define_nodes(self) -> dict[str, Callable]:
        """Define workflow nodes. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def define_edges(self) -> list[tuple]:
        """Define workflow edges. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def define_conditional_edges(self) -> list[dict]:
        """Define conditional edges. Must be implemented by subclasses."""
        pass

    def get_entry_point(self) -> str:
        """Get the entry point node name."""
        return "start"

    def get_breakpoints(self) -> list[str]:
        """Get list of nodes that require human approval."""
        return []

    def define_error_recovery_nodes(self) -> dict[str, Callable[[BaseWorkflowState], Awaitable[BaseWorkflowState]]]:
        """Define optional workflow error recovery nodes."""
        return {"error_recovery": self._default_error_recovery_node}

    def get_error_recovery_node(self) -> Optional[str]:
        """Get the default error recovery node name."""
        nodes = self.define_error_recovery_nodes()
        if not nodes:
            return None
        if "error_recovery" in nodes:
            return "error_recovery"
        return next(iter(nodes))

    def get_subworkflow_bindings(self) -> dict[str, str]:
        """Map node name -> child workflow type for composition."""
        return {}

    def define_parallel_fanouts(self) -> list[dict[str, Any]]:
        """Define optional fan-out/fan-in topology metadata."""
        return []

    async def _default_error_recovery_node(self, state: BaseWorkflowState) -> BaseWorkflowState:
        """Default error-recovery node that retries or checkpoints."""
        state["status"] = WorkflowStatus.CHECKPOINT.value
        state["error_recovery_action"] = "checkpoint_for_retry"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        if state.get("retry_count", 0) > state.get("max_retries", 3):
            state["status"] = WorkflowStatus.FAILED.value
            state["error_recovery_action"] = "max_retries_exceeded"

        return state

    async def _resolve_workflow_version(
        self,
        tenant_id: str,
        fallback: str = "1.0.0",
    ) -> str:
        """Resolve current workflow version from langgraph_workflows when enabled."""
        if not FEATURE_LANGGRAPH_WORKFLOW_VERSIONING:
            return fallback

        try:
            async with get_workflow_connection(tenant_id) as conn:
                row = await conn.fetchrow(
                    """
                    SELECT version
                    FROM langgraph_workflows
                    WHERE LOWER(name) = LOWER($1)
                       OR LOWER(COALESCE(graph_type, '')) = LOWER($1)
                    ORDER BY updated_at DESC NULLS LAST, created_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    self.workflow_type,
                )
                if row and row["version"]:
                    return str(row["version"])
        except Exception as exc:
            logger.warning(
                "Workflow version lookup failed for %s: %s",
                self.workflow_type,
                exc,
            )
        return fallback

    def _resolve_error_handler(self) -> Optional[Callable[[BaseWorkflowState], Awaitable[BaseWorkflowState]]]:
        """Resolve the active error handler callable when feature is enabled."""
        if not FEATURE_LANGGRAPH_ERROR_RECOVERY:
            return None
        error_node_name = self.get_error_recovery_node()
        if not error_node_name:
            return None
        return self.define_error_recovery_nodes().get(error_node_name)

    def _wrap_node_handler(
        self,
        node_name: str,
        handler: Callable[[BaseWorkflowState], Awaitable[BaseWorkflowState]],
        error_handler: Optional[Callable[[BaseWorkflowState], Awaitable[BaseWorkflowState]]],
    ) -> Callable[[BaseWorkflowState], Awaitable[BaseWorkflowState]]:
        """Wrap a node with state bookkeeping, checkpointing, and error handling."""

        async def _wrapped(state: BaseWorkflowState) -> BaseWorkflowState:
            state.setdefault("previous_nodes", [])
            state.setdefault("errors", [])
            state.setdefault("metadata", {})
            state.setdefault("parallel_results", {})
            state.setdefault("child_workflow_ids", [])

            state["current_node"] = node_name
            state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
            if not state["previous_nodes"] or state["previous_nodes"][-1] != node_name:
                state["previous_nodes"].append(node_name)

            try:
                updated_state = await handler(state)

                if FEATURE_LANGGRAPH_WORKFLOW_COMPOSITION:
                    subworkflow_type = self.get_subworkflow_bindings().get(node_name)
                    if subworkflow_type:
                        sub_result = await self.execute_subworkflow(
                            subworkflow_type=subworkflow_type,
                            parent_state=updated_state,
                        )
                        updated_state.setdefault("metadata", {}).setdefault(
                            "subworkflow_results", {}
                        )[subworkflow_type] = sub_result
                        child_id = sub_result.get("workflow_id")
                        if child_id:
                            updated_state.setdefault("child_workflow_ids", []).append(child_id)
                        if sub_result.get("status") == WorkflowStatus.FAILED.value:
                            updated_state.setdefault("errors", []).append(
                                {
                                    "node": node_name,
                                    "error": f"Subworkflow '{subworkflow_type}' failed",
                                    "subworkflow_result": sub_result,
                                }
                            )
                return updated_state

            except Exception as exc:
                state["retry_count"] = state.get("retry_count", 0) + 1
                state["errors"].append(
                    {
                        "node": node_name,
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                state["metadata"]["last_error"] = str(exc)
                if error_handler:
                    state["error_recovery_node"] = self.get_error_recovery_node()
                    return await error_handler(state)
                raise

            finally:
                if FEATURE_LANGGRAPH_STATE_CHECKPOINTS:
                    try:
                        await self.checkpoint_saver.save_checkpoint(
                            workflow_id=state.get("workflow_id"),
                            workflow_type=self.workflow_type,
                            state=state,
                            current_node=node_name,
                            metadata={
                                "tenant_id": state.get("tenant_id"),
                                "checkpoint_reason": "node_completion",
                                "workflow_version": state.get("workflow_version"),
                            },
                            tenant_id=state.get("tenant_id"),
                        )
                    except Exception as exc:
                        logger.warning(
                            "Checkpoint save failed for node %s in %s: %s",
                            node_name,
                            self.workflow_type,
                            exc,
                        )

        return _wrapped

    async def run_parallel_nodes(
        self,
        state: BaseWorkflowState,
        node_names: list[str],
        *,
        group_name: str = "default",
    ) -> BaseWorkflowState:
        """Execute node handlers concurrently and capture fan-out/fan-in results."""
        if not FEATURE_LANGGRAPH_PARALLEL_NODES:
            return state

        nodes = self.define_nodes()
        state.setdefault("parallel_results", {})

        async def _run_node(node_name: str) -> tuple[str, dict[str, Any]]:
            handler = nodes.get(node_name)
            if not handler:
                return node_name, {"status": "missing_node", "error": "node_not_registered"}
            branch_state: BaseWorkflowState = {
                **state,
                "current_node": node_name,
                "metadata": dict(state.get("metadata", {})),
            }
            try:
                branch_result = await handler(branch_state)
                return node_name, {
                    "status": "completed",
                    "result": branch_result.get("result"),
                    "metadata": branch_result.get("metadata", {}),
                }
            except Exception as exc:
                return node_name, {"status": "failed", "error": str(exc)}

        results = await asyncio.gather(*[_run_node(node) for node in node_names])
        state["parallel_results"][group_name] = {name: output for name, output in results}
        return state

    async def execute_subworkflow(
        self,
        subworkflow_type: str,
        parent_state: BaseWorkflowState,
    ) -> dict[str, Any]:
        """Execute a child workflow and return its result payload."""
        if not FEATURE_LANGGRAPH_WORKFLOW_COMPOSITION:
            return {"status": "skipped", "reason": "workflow_composition_disabled"}

        template_class = self.workflow_registry.get(subworkflow_type)
        if not template_class:
            return {
                "status": WorkflowStatus.FAILED.value,
                "error": f"Unknown subworkflow type: {subworkflow_type}",
            }

        composed_state = {
            "parent_workflow_id": parent_state.get("workflow_id"),
            "tenant_id": parent_state.get("tenant_id"),
            "metadata": {
                **parent_state.get("metadata", {}),
                "composed_by": self.workflow_type,
            },
        }

        template = template_class(
            checkpoint_saver=self.checkpoint_saver,
            hitl_manager=self.hitl_manager,
            ooda_executor=self.ooda_executor,
            workflow_registry=self.workflow_registry,
        )
        return await template.execute(
            initial_state=composed_state,
            tenant_id=parent_state.get("tenant_id"),
            timeout_seconds=int(parent_state.get("metadata", {}).get("subworkflow_timeout_seconds", 180)),
        )

    def build_graph(self) -> Optional[StateGraph]:
        """Build the LangGraph workflow graph."""
        if not LANGGRAPH_AVAILABLE:
            logger.warning("LangGraph not available, using fallback execution")
            return None

        # Create state graph
        graph = StateGraph(BaseWorkflowState)

        error_handler = self._resolve_error_handler()

        # Add nodes
        defined_nodes = self.define_nodes()
        for name, handler in defined_nodes.items():
            wrapped = self._wrap_node_handler(name, handler, error_handler)
            graph.add_node(name, wrapped)

        if FEATURE_LANGGRAPH_ERROR_RECOVERY:
            for recovery_name, recovery_handler in self.define_error_recovery_nodes().items():
                if recovery_name not in defined_nodes:
                    graph.add_node(recovery_name, recovery_handler)

        # Add edges
        for source, target in self.define_edges():
            if target == "END":
                graph.add_edge(source, END)
            else:
                graph.add_edge(source, target)

        # Add conditional edges
        for edge_def in self.define_conditional_edges():
            source = edge_def["source"]
            condition = edge_def["condition"]
            mapping = edge_def["mapping"]

            # Convert "END" strings to actual END
            mapping = {k: END if v == "END" else v for k, v in mapping.items()}
            graph.add_conditional_edges(source, condition, mapping)

        if FEATURE_LANGGRAPH_PARALLEL_NODES:
            for fanout in self.define_parallel_fanouts():
                source = fanout.get("source")
                branches = fanout.get("branches", [])
                join = fanout.get("join")
                if not source or not branches:
                    continue
                for branch in branches:
                    graph.add_edge(source, branch)
                if join:
                    for branch in branches:
                        graph.add_edge(branch, join)

        # Set entry point
        graph.set_entry_point(self.get_entry_point())

        self._graph = graph.compile()
        return self._graph

    async def execute(
        self,
        initial_state: dict,
        tenant_id: str = None,
        timeout_seconds: int = 300
    ) -> dict:
        """Execute the workflow with checkpointing support."""
        workflow_id = initial_state.get("workflow_id") or str(uuid.uuid4())
        tenant = _resolve_tenant_id(tenant_id or initial_state.get("tenant_id"))
        workflow_version = await self._resolve_workflow_version(
            tenant_id=tenant,
            fallback=str(initial_state.get("workflow_version") or "1.0.0"),
        )

        # Build state
        state: BaseWorkflowState = {
            "workflow_id": workflow_id,
            "workflow_type": self.workflow_type,
            "workflow_version": workflow_version,
            "tenant_id": tenant,
            "parent_workflow_id": initial_state.get("parent_workflow_id"),
            "child_workflow_ids": [],
            "status": WorkflowStatus.RUNNING.value,
            "current_node": self.get_entry_point(),
            "previous_nodes": [],
            "ooda_phase": OODAPhase.OBSERVE.value,
            "observations": [],
            "orientation": {},
            "decisions": [],
            "actions_taken": [],
            "requires_approval": False,
            "approval_status": HumanApprovalStatus.NOT_REQUIRED.value,
            "approval_request_id": None,
            "errors": [],
            "retry_count": 0,
            "max_retries": 3,
            "error_recovery_node": self.get_error_recovery_node(),
            "error_recovery_action": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "deadline": None,
            "result": None,
            "parallel_results": {},
            "metadata": {},
            **initial_state
        }

        try:
            # Check for existing checkpoint
            checkpoint = await self.checkpoint_saver.load_checkpoint(
                workflow_id,
                tenant_id=tenant,
            )
            if checkpoint:
                logger.info(f"Resuming workflow {workflow_id} from checkpoint")
                state = {**state, **checkpoint.state_data}
                state["current_node"] = checkpoint.current_node

            if FEATURE_LANGGRAPH_PARALLEL_NODES:
                parallel_entry_nodes = state.get("metadata", {}).get("parallel_entry_nodes", [])
                if parallel_entry_nodes:
                    state = await self.run_parallel_nodes(
                        state,
                        node_names=list(parallel_entry_nodes),
                        group_name="entry",
                    )

            # Build and execute graph
            graph = self.build_graph()
            if graph:
                final_state = await asyncio.wait_for(
                    graph.ainvoke(state),
                    timeout=timeout_seconds
                )
            else:
                # Fallback execution without LangGraph
                final_state = await self._fallback_execute(state)

            if final_state.get("requires_approval"):
                final_state["status"] = WorkflowStatus.PAUSED.value
                await self.checkpoint_saver.save_checkpoint(
                    workflow_id=workflow_id,
                    workflow_type=self.workflow_type,
                    state=final_state,
                    current_node=final_state.get("current_node", "approval_wait"),
                    metadata={
                        "tenant_id": tenant,
                        "workflow_version": final_state.get("workflow_version"),
                        "approval_request_id": final_state.get("approval_request_id"),
                    },
                    tenant_id=tenant,
                )

            # Clean up checkpoint on success
            if final_state.get("status") == WorkflowStatus.COMPLETED.value:
                await self.checkpoint_saver.delete_checkpoint(workflow_id, tenant_id=tenant)

            return {
                "workflow_id": workflow_id,
                "workflow_version": final_state.get("workflow_version", workflow_version),
                "status": final_state.get("status"),
                "result": final_state.get("result"),
                "errors": final_state.get("errors", []),
                "parallel_results": final_state.get("parallel_results", {}),
                "child_workflow_ids": final_state.get("child_workflow_ids", []),
                "approval_request_id": final_state.get("approval_request_id"),
                "can_resume": final_state.get("status") in {
                    WorkflowStatus.CHECKPOINT.value,
                    WorkflowStatus.PAUSED.value,
                },
                "execution_summary": {
                    "nodes_visited": final_state.get("previous_nodes", []),
                    "decisions_made": len(final_state.get("decisions", [])),
                    "actions_taken": len(final_state.get("actions_taken", []))
                }
            }

        except asyncio.TimeoutError:
            # Save checkpoint before timeout
            await self.checkpoint_saver.save_checkpoint(
                workflow_id,
                self.workflow_type,
                state,
                state.get("current_node", "unknown"),
                {
                    "timeout": True,
                    "timeout_seconds": timeout_seconds,
                    "tenant_id": tenant,
                    "workflow_version": workflow_version,
                },
                tenant_id=tenant,
            )
            return {
                "workflow_id": workflow_id,
                "workflow_version": workflow_version,
                "status": WorkflowStatus.CHECKPOINT.value,
                "error": f"Workflow timed out after {timeout_seconds}s, checkpoint saved",
                "can_resume": True
            }

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            if FEATURE_LANGGRAPH_ERROR_RECOVERY:
                state.setdefault("errors", []).append(
                    {
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "node": state.get("current_node"),
                    }
                )
                state["retry_count"] = state.get("retry_count", 0) + 1
                recovery_state = await self._default_error_recovery_node(state)
                if recovery_state.get("status") == WorkflowStatus.CHECKPOINT.value:
                    await self.checkpoint_saver.save_checkpoint(
                        workflow_id=workflow_id,
                        workflow_type=self.workflow_type,
                        state=recovery_state,
                        current_node=recovery_state.get("current_node", "error_recovery"),
                        metadata={
                            "tenant_id": tenant,
                            "workflow_version": workflow_version,
                            "error_recovery_action": recovery_state.get("error_recovery_action"),
                        },
                        tenant_id=tenant,
                    )
                    return {
                        "workflow_id": workflow_id,
                        "workflow_version": workflow_version,
                        "status": WorkflowStatus.CHECKPOINT.value,
                        "error": str(e),
                        "errors": recovery_state.get("errors", []),
                        "can_resume": True,
                    }
            return {
                "workflow_id": workflow_id,
                "workflow_version": workflow_version,
                "status": WorkflowStatus.FAILED.value,
                "error": str(e),
                "errors": state.get("errors", []) + [{"error": str(e)}]
            }

    async def _fallback_execute(self, state: BaseWorkflowState) -> BaseWorkflowState:
        """Fallback execution when LangGraph is not available."""
        error_handler = self._resolve_error_handler()
        nodes = {
            name: self._wrap_node_handler(name, handler, error_handler)
            for name, handler in self.define_nodes().items()
        }
        edges = self.define_edges()
        conditional_edges = {
            edge["source"]: (edge["condition"], edge["mapping"])
            for edge in self.define_conditional_edges()
        }

        current = self.get_entry_point()
        visited = []

        while current and current != "END":
            if current in nodes:
                visited.append(current)
                state["current_node"] = current
                state["previous_nodes"] = visited.copy()

                # Execute node
                handler = nodes[current]
                state = await handler(state)

                # Find next node (conditional first, then static edge)
                next_node = None
                conditional = conditional_edges.get(current)
                if conditional:
                    condition_fn, mapping = conditional
                    route = condition_fn(state)
                    next_node = mapping.get(route)

                if not next_node:
                    for source, target in edges:
                        if source == current:
                            next_node = target
                            break

                current = next_node
            else:
                break

        if state.get("status") not in {
            WorkflowStatus.FAILED.value,
            WorkflowStatus.PAUSED.value,
            WorkflowStatus.CHECKPOINT.value,
        }:
            state["status"] = WorkflowStatus.COMPLETED.value
        return state


# =============================================================================
# CUSTOMER ONBOARDING WORKFLOW TEMPLATE
# =============================================================================

class CustomerOnboardingState(BaseWorkflowState):
    """State specific to customer onboarding workflow."""
    customer_id: str
    customer_name: str
    customer_email: str
    onboarding_stage: str
    welcome_email_sent: bool
    setup_completed: bool
    training_scheduled: bool
    first_project_created: bool


class CustomerOnboardingWorkflow(WorkflowTemplate):
    """
    Customer Onboarding Workflow
    ============================
    Stages:
    1. Welcome & Account Setup
    2. Data Import (if applicable)
    3. Initial Configuration
    4. Training Scheduling
    5. First Project Setup
    6. Success Check
    """

    def __init__(self, **kwargs):
        super().__init__(workflow_type="customer_onboarding", **kwargs)

    def define_nodes(self) -> dict[str, Callable]:
        return {
            "start": self._start_onboarding,
            "send_welcome": self._send_welcome_email,
            "setup_account": self._setup_account,
            "import_data": self._import_data,
            "configure_settings": self._configure_settings,
            "schedule_training": self._schedule_training,
            "create_first_project": self._create_first_project,
            "success_check": self._success_check,
            "complete": self._complete_onboarding,
            "escalate": self._escalate_to_human
        }

    def define_edges(self) -> list[tuple]:
        return [
            ("start", "send_welcome"),
            ("send_welcome", "setup_account"),
            ("setup_account", "import_data"),
            ("configure_settings", "schedule_training"),
            ("create_first_project", "success_check"),
            ("complete", "END"),
            ("escalate", "END")
        ]

    def define_conditional_edges(self) -> list[dict]:
        return [
            {
                "source": "import_data",
                "condition": self._check_import_result,
                "mapping": {
                    "success": "configure_settings",
                    "skip": "configure_settings",
                    "fail": "escalate"
                }
            },
            {
                "source": "schedule_training",
                "condition": self._check_training_scheduled,
                "mapping": {
                    "scheduled": "create_first_project",
                    "declined": "create_first_project",
                    "needs_approval": "escalate"
                }
            },
            {
                "source": "success_check",
                "condition": self._check_success,
                "mapping": {
                    "success": "complete",
                    "needs_attention": "escalate",
                    "retry": "configure_settings"
                }
            }
        ]

    def get_breakpoints(self) -> list[str]:
        return ["import_data", "schedule_training"]

    async def _start_onboarding(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Initialize onboarding process."""
        state["onboarding_stage"] = "started"
        state["welcome_email_sent"] = False
        state["setup_completed"] = False
        state["training_scheduled"] = False
        state["first_project_created"] = False
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Starting onboarding for customer {state.get('customer_id')}")
        return state

    async def _send_welcome_email(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Send welcome email to new customer."""
        state["onboarding_stage"] = "welcome_sent"
        state["welcome_email_sent"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # In real implementation, this would send actual email
        logger.info(f"Welcome email sent to {state.get('customer_email')}")
        return state

    async def _setup_account(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Set up customer account with defaults."""
        state["onboarding_stage"] = "account_setup"
        state["setup_completed"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Account setup completed for {state.get('customer_id')}")
        return state

    async def _import_data(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Import customer data if provided."""
        state["onboarding_stage"] = "data_import"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Check if data import is needed
        metadata = state.get("metadata", {})
        if metadata.get("has_import_data"):
            # Simulate import
            state["metadata"]["import_status"] = "success"
        else:
            state["metadata"]["import_status"] = "skip"

        return state

    async def _configure_settings(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Configure customer-specific settings."""
        state["onboarding_stage"] = "configuration"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Settings configured for {state.get('customer_id')}")
        return state

    async def _schedule_training(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Schedule training session."""
        state["onboarding_stage"] = "training"

        # Check if customer wants training
        metadata = state.get("metadata", {})
        if metadata.get("training_requested", True):
            state["training_scheduled"] = True
            state["metadata"]["training_status"] = "scheduled"
        else:
            state["metadata"]["training_status"] = "declined"

        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    async def _create_first_project(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Create a sample/first project for the customer."""
        state["onboarding_stage"] = "first_project"
        state["first_project_created"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"First project created for {state.get('customer_id')}")
        return state

    async def _success_check(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Verify onboarding success."""
        state["onboarding_stage"] = "success_check"

        # Check all critical steps completed
        checks = [
            state.get("welcome_email_sent", False),
            state.get("setup_completed", False),
            state.get("first_project_created", False)
        ]

        state["metadata"]["success_check_result"] = "success" if all(checks) else "needs_attention"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    async def _complete_onboarding(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Complete the onboarding process."""
        state["onboarding_stage"] = "completed"
        state["status"] = WorkflowStatus.COMPLETED.value
        state["result"] = {
            "success": True,
            "customer_id": state.get("customer_id"),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Onboarding completed for {state.get('customer_id')}")
        return state

    async def _escalate_to_human(self, state: CustomerOnboardingState) -> CustomerOnboardingState:
        """Escalate to human for review."""
        state["status"] = WorkflowStatus.PAUSED.value
        state["requires_approval"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Create approval request
        request = await self.hitl_manager.request_approval(
            workflow_id=state.get("workflow_id"),
            node_name="escalate",
            description=f"Customer onboarding needs attention: {state.get('customer_name')}",
            context={
                "customer_id": state.get("customer_id"),
                "stage": state.get("onboarding_stage"),
                "issues": state.get("errors", [])
            },
            options=["approve_continue", "assign_to_agent", "cancel"],
            tenant_id=state.get("tenant_id"),
        )

        state["approval_request_id"] = request.request_id
        return state

    def _check_import_result(self, state: CustomerOnboardingState) -> str:
        metadata = state.get("metadata", {})
        return metadata.get("import_status", "skip")

    def _check_training_scheduled(self, state: CustomerOnboardingState) -> str:
        metadata = state.get("metadata", {})
        return metadata.get("training_status", "scheduled")

    def _check_success(self, state: CustomerOnboardingState) -> str:
        metadata = state.get("metadata", {})
        return metadata.get("success_check_result", "success")


# =============================================================================
# INVOICE COLLECTION WORKFLOW TEMPLATE
# =============================================================================

class InvoiceCollectionState(BaseWorkflowState):
    """State specific to invoice collection workflow."""
    invoice_id: str
    customer_id: str
    amount: float
    due_date: str
    days_overdue: int
    collection_stage: str
    reminder_count: int
    payment_plan_offered: bool
    escalated: bool


class InvoiceCollectionWorkflow(WorkflowTemplate):
    """
    Invoice Collection Workflow
    ===========================
    Stages:
    1. Assessment - Evaluate invoice status
    2. Gentle Reminder - First contact
    3. Follow-up - Second contact
    4. Payment Plan - Offer alternatives
    5. Escalation - Human intervention
    6. Final Notice - Last warning
    7. Collections - External handoff
    """

    def __init__(self, **kwargs):
        super().__init__(workflow_type="invoice_collection", **kwargs)

    def define_nodes(self) -> dict[str, Callable]:
        return {
            "start": self._assess_invoice,
            "gentle_reminder": self._send_gentle_reminder,
            "follow_up": self._send_follow_up,
            "payment_plan": self._offer_payment_plan,
            "human_review": self._request_human_review,
            "final_notice": self._send_final_notice,
            "collections": self._send_to_collections,
            "payment_received": self._mark_paid,
            "close": self._close_workflow
        }

    def define_edges(self) -> list[tuple]:
        return [
            ("payment_received", "close"),
            ("collections", "close"),
            ("close", "END")
        ]

    def define_conditional_edges(self) -> list[dict]:
        return [
            {
                "source": "start",
                "condition": self._route_by_overdue_days,
                "mapping": {
                    "not_overdue": "close",
                    "gentle": "gentle_reminder",
                    "follow_up": "follow_up",
                    "payment_plan": "payment_plan",
                    "escalate": "human_review"
                }
            },
            {
                "source": "gentle_reminder",
                "condition": self._check_payment_status,
                "mapping": {
                    "paid": "payment_received",
                    "continue": "follow_up",
                    "wait": "close"
                }
            },
            {
                "source": "follow_up",
                "condition": self._check_payment_status,
                "mapping": {
                    "paid": "payment_received",
                    "continue": "payment_plan",
                    "wait": "close"
                }
            },
            {
                "source": "payment_plan",
                "condition": self._check_plan_response,
                "mapping": {
                    "accepted": "payment_received",
                    "rejected": "human_review",
                    "no_response": "final_notice"
                }
            },
            {
                "source": "human_review",
                "condition": self._check_human_decision,
                "mapping": {
                    "continue": "final_notice",
                    "forgive": "close",
                    "collections": "collections"
                }
            },
            {
                "source": "final_notice",
                "condition": self._check_payment_status,
                "mapping": {
                    "paid": "payment_received",
                    "continue": "collections",
                    "wait": "close"
                }
            }
        ]

    def get_breakpoints(self) -> list[str]:
        return ["payment_plan", "human_review", "collections"]

    async def _assess_invoice(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Assess invoice and determine collection strategy."""
        state["collection_stage"] = "assessment"
        state["reminder_count"] = state.get("reminder_count", 0)
        state["payment_plan_offered"] = False
        state["escalated"] = False
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Calculate days overdue
        if state.get("due_date"):
            due = datetime.fromisoformat(state["due_date"].replace("Z", "+00:00"))
            days_overdue = (datetime.now(timezone.utc) - due).days
            state["days_overdue"] = max(0, days_overdue)
        else:
            state["days_overdue"] = 0

        logger.info(f"Assessing invoice {state.get('invoice_id')}: {state['days_overdue']} days overdue")
        return state

    async def _send_gentle_reminder(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Send first gentle payment reminder."""
        state["collection_stage"] = "gentle_reminder"
        state["reminder_count"] = state.get("reminder_count", 0) + 1
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # In real implementation, send email
        logger.info(f"Gentle reminder sent for invoice {state.get('invoice_id')}")
        return state

    async def _send_follow_up(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Send follow-up reminder."""
        state["collection_stage"] = "follow_up"
        state["reminder_count"] = state.get("reminder_count", 0) + 1
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Follow-up sent for invoice {state.get('invoice_id')}")
        return state

    async def _offer_payment_plan(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Offer payment plan to customer."""
        state["collection_stage"] = "payment_plan"
        state["payment_plan_offered"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Request approval for payment plan terms
        request = await self.hitl_manager.request_approval(
            workflow_id=state.get("workflow_id"),
            node_name="payment_plan",
            description=f"Approve payment plan for invoice {state.get('invoice_id')}",
            context={
                "invoice_id": state.get("invoice_id"),
                "amount": state.get("amount"),
                "customer_id": state.get("customer_id"),
                "suggested_plan": "3 monthly payments"
            },
            options=["approve_plan", "modify_plan", "skip_to_final"],
            tenant_id=state.get("tenant_id"),
        )

        state["approval_request_id"] = request.request_id
        state["requires_approval"] = True
        return state

    async def _request_human_review(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Request human review for difficult cases."""
        state["collection_stage"] = "human_review"
        state["escalated"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        request = await self.hitl_manager.request_approval(
            workflow_id=state.get("workflow_id"),
            node_name="human_review",
            description=f"Review collection strategy for invoice {state.get('invoice_id')}",
            context={
                "invoice_id": state.get("invoice_id"),
                "amount": state.get("amount"),
                "days_overdue": state.get("days_overdue"),
                "reminder_count": state.get("reminder_count"),
                "customer_id": state.get("customer_id")
            },
            options=["continue_collection", "forgive_debt", "send_to_collections"],
            timeout_minutes=1440,  # 24 hours
            tenant_id=state.get("tenant_id"),
        )

        state["approval_request_id"] = request.request_id
        state["requires_approval"] = True
        return state

    async def _send_final_notice(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Send final notice before collections."""
        state["collection_stage"] = "final_notice"
        state["reminder_count"] = state.get("reminder_count", 0) + 1
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Final notice sent for invoice {state.get('invoice_id')}")
        return state

    async def _send_to_collections(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Hand off to external collections."""
        state["collection_stage"] = "collections"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Invoice {state.get('invoice_id')} sent to collections")
        return state

    async def _mark_paid(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Mark invoice as paid."""
        state["collection_stage"] = "paid"
        state["status"] = WorkflowStatus.COMPLETED.value
        state["result"] = {
            "success": True,
            "outcome": "paid",
            "invoice_id": state.get("invoice_id")
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Invoice {state.get('invoice_id')} marked as paid")
        return state

    async def _close_workflow(self, state: InvoiceCollectionState) -> InvoiceCollectionState:
        """Close the collection workflow."""
        state["status"] = WorkflowStatus.COMPLETED.value
        if not state.get("result"):
            state["result"] = {
                "success": True,
                "outcome": state.get("collection_stage"),
                "invoice_id": state.get("invoice_id")
            }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    def _route_by_overdue_days(self, state: InvoiceCollectionState) -> str:
        days = state.get("days_overdue", 0)
        if days <= 0:
            return "not_overdue"
        elif days <= 7:
            return "gentle"
        elif days <= 14:
            return "follow_up"
        elif days <= 30:
            return "payment_plan"
        else:
            return "escalate"

    def _check_payment_status(self, state: InvoiceCollectionState) -> str:
        # In real implementation, check database for payment
        metadata = state.get("metadata", {})
        if metadata.get("payment_received"):
            return "paid"
        # Continue to next stage based on reminder count
        if state.get("reminder_count", 0) >= 3:
            return "continue"
        return "wait"

    def _check_plan_response(self, state: InvoiceCollectionState) -> str:
        metadata = state.get("metadata", {})
        return metadata.get("plan_response", "no_response")

    def _check_human_decision(self, state: InvoiceCollectionState) -> str:
        metadata = state.get("metadata", {})
        return metadata.get("human_decision", "continue")


# =============================================================================
# LEAD QUALIFICATION WORKFLOW TEMPLATE
# =============================================================================

class LeadQualificationState(BaseWorkflowState):
    """State specific to lead qualification workflow."""
    lead_id: str
    lead_name: str
    lead_email: str
    lead_source: str
    company_name: str
    company_size: str
    budget_range: str
    timeline: str
    score: int
    qualification_stage: str
    is_qualified: bool
    disqualification_reason: Optional[str]


class LeadQualificationWorkflow(WorkflowTemplate):
    """
    Lead Qualification Workflow (BANT Framework)
    =============================================
    Stages:
    1. Initial Scoring - Basic fit assessment
    2. Budget Qualification - Can they afford it?
    3. Authority Check - Decision maker?
    4. Need Assessment - Do they need it?
    5. Timeline Evaluation - When do they need it?
    6. Final Scoring - Overall qualification
    7. Routing - MQL/SQL/Disqualify
    """

    def __init__(self, **kwargs):
        super().__init__(workflow_type="lead_qualification", **kwargs)

    def define_nodes(self) -> dict[str, Callable]:
        return {
            "start": self._initial_scoring,
            "budget_check": self._check_budget,
            "authority_check": self._check_authority,
            "need_assessment": self._assess_need,
            "timeline_check": self._check_timeline,
            "final_scoring": self._calculate_final_score,
            "route_mql": self._route_to_mql,
            "route_sql": self._route_to_sql,
            "disqualify": self._disqualify_lead,
            "human_review": self._request_human_review,
            "complete": self._complete_qualification
        }

    def define_edges(self) -> list[tuple]:
        return [
            ("start", "budget_check"),
            ("route_mql", "complete"),
            ("route_sql", "complete"),
            ("disqualify", "complete"),
            ("complete", "END")
        ]

    def define_conditional_edges(self) -> list[dict]:
        return [
            {
                "source": "budget_check",
                "condition": self._check_budget_result,
                "mapping": {
                    "qualified": "authority_check",
                    "uncertain": "human_review",
                    "disqualified": "disqualify"
                }
            },
            {
                "source": "authority_check",
                "condition": self._check_authority_result,
                "mapping": {
                    "qualified": "need_assessment",
                    "uncertain": "need_assessment",
                    "disqualified": "disqualify"
                }
            },
            {
                "source": "need_assessment",
                "condition": self._check_need_result,
                "mapping": {
                    "high": "timeline_check",
                    "medium": "timeline_check",
                    "low": "disqualify"
                }
            },
            {
                "source": "timeline_check",
                "condition": self._check_timeline_result,
                "mapping": {
                    "immediate": "final_scoring",
                    "short_term": "final_scoring",
                    "long_term": "final_scoring",
                    "no_timeline": "human_review"
                }
            },
            {
                "source": "final_scoring",
                "condition": self._route_by_score,
                "mapping": {
                    "sql": "route_sql",
                    "mql": "route_mql",
                    "disqualify": "disqualify",
                    "review": "human_review"
                }
            },
            {
                "source": "human_review",
                "condition": self._check_human_decision,
                "mapping": {
                    "sql": "route_sql",
                    "mql": "route_mql",
                    "disqualify": "disqualify"
                }
            }
        ]

    def get_breakpoints(self) -> list[str]:
        return ["human_review"]

    async def _initial_scoring(self, state: LeadQualificationState) -> LeadQualificationState:
        """Initial lead scoring based on available data."""
        state["qualification_stage"] = "initial"
        state["score"] = 0
        state["is_qualified"] = False
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Basic scoring from lead source
        source_scores = {
            "website": 10,
            "referral": 25,
            "paid_ad": 15,
            "organic": 20,
            "cold_outreach": 5
        }
        state["score"] += source_scores.get(state.get("lead_source", ""), 10)

        # Company size scoring
        size_scores = {
            "enterprise": 30,
            "mid_market": 25,
            "smb": 15,
            "startup": 10
        }
        state["score"] += size_scores.get(state.get("company_size", ""), 10)

        logger.info(f"Initial score for lead {state.get('lead_id')}: {state['score']}")
        return state

    async def _check_budget(self, state: LeadQualificationState) -> LeadQualificationState:
        """Check if lead has appropriate budget."""
        state["qualification_stage"] = "budget"

        budget = state.get("budget_range", "unknown")
        budget_scores = {
            "enterprise": (40, "qualified"),
            "growth": (30, "qualified"),
            "starter": (15, "uncertain"),
            "unknown": (5, "uncertain"),
            "none": (0, "disqualified")
        }

        score_add, result = budget_scores.get(budget, (5, "uncertain"))
        state["score"] += score_add
        state["metadata"]["budget_result"] = result
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        return state

    async def _check_authority(self, state: LeadQualificationState) -> LeadQualificationState:
        """Check if lead has decision-making authority."""
        state["qualification_stage"] = "authority"

        # In real implementation, this would analyze job title, etc.
        metadata = state.get("metadata", {})
        is_decision_maker = metadata.get("is_decision_maker", None)

        if is_decision_maker is True:
            state["score"] += 25
            state["metadata"]["authority_result"] = "qualified"
        elif is_decision_maker is False:
            state["metadata"]["authority_result"] = "disqualified"
        else:
            state["score"] += 10
            state["metadata"]["authority_result"] = "uncertain"

        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    async def _assess_need(self, state: LeadQualificationState) -> LeadQualificationState:
        """Assess the lead's need for the product."""
        state["qualification_stage"] = "need"

        # In real implementation, analyze interaction history, form data, etc.
        metadata = state.get("metadata", {})
        pain_points = metadata.get("pain_points", [])

        if len(pain_points) >= 3:
            state["score"] += 30
            state["metadata"]["need_result"] = "high"
        elif len(pain_points) >= 1:
            state["score"] += 15
            state["metadata"]["need_result"] = "medium"
        else:
            state["metadata"]["need_result"] = "low"

        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    async def _check_timeline(self, state: LeadQualificationState) -> LeadQualificationState:
        """Evaluate the lead's purchase timeline."""
        state["qualification_stage"] = "timeline"

        timeline = state.get("timeline", "unknown")
        timeline_scores = {
            "immediate": (30, "immediate"),
            "this_quarter": (25, "short_term"),
            "this_year": (15, "long_term"),
            "exploring": (5, "long_term"),
            "unknown": (0, "no_timeline")
        }

        score_add, result = timeline_scores.get(timeline, (0, "no_timeline"))
        state["score"] += score_add
        state["metadata"]["timeline_result"] = result
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        return state

    async def _calculate_final_score(self, state: LeadQualificationState) -> LeadQualificationState:
        """Calculate final qualification score."""
        state["qualification_stage"] = "final_scoring"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        score = state.get("score", 0)

        if score >= 100:
            state["metadata"]["final_route"] = "sql"
            state["is_qualified"] = True
        elif score >= 60:
            state["metadata"]["final_route"] = "mql"
            state["is_qualified"] = True
        elif score >= 40:
            state["metadata"]["final_route"] = "review"
        else:
            state["metadata"]["final_route"] = "disqualify"

        logger.info(f"Final score for lead {state.get('lead_id')}: {score}")
        return state

    async def _route_to_mql(self, state: LeadQualificationState) -> LeadQualificationState:
        """Route as Marketing Qualified Lead."""
        state["qualification_stage"] = "mql"
        state["is_qualified"] = True
        state["result"] = {
            "qualification": "MQL",
            "score": state.get("score"),
            "lead_id": state.get("lead_id")
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Lead {state.get('lead_id')} qualified as MQL (score: {state.get('score')})")
        return state

    async def _route_to_sql(self, state: LeadQualificationState) -> LeadQualificationState:
        """Route as Sales Qualified Lead."""
        state["qualification_stage"] = "sql"
        state["is_qualified"] = True
        state["result"] = {
            "qualification": "SQL",
            "score": state.get("score"),
            "lead_id": state.get("lead_id")
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Lead {state.get('lead_id')} qualified as SQL (score: {state.get('score')})")
        return state

    async def _disqualify_lead(self, state: LeadQualificationState) -> LeadQualificationState:
        """Disqualify the lead."""
        state["qualification_stage"] = "disqualified"
        state["is_qualified"] = False
        state["disqualification_reason"] = state.get("metadata", {}).get(
            "disqualification_reason", "Did not meet qualification criteria"
        )
        state["result"] = {
            "qualification": "Disqualified",
            "reason": state.get("disqualification_reason"),
            "score": state.get("score"),
            "lead_id": state.get("lead_id")
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Lead {state.get('lead_id')} disqualified")
        return state

    async def _request_human_review(self, state: LeadQualificationState) -> LeadQualificationState:
        """Request human review for uncertain cases."""
        state["qualification_stage"] = "human_review"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        request = await self.hitl_manager.request_approval(
            workflow_id=state.get("workflow_id"),
            node_name="human_review",
            description=f"Review lead qualification: {state.get('lead_name')}",
            context={
                "lead_id": state.get("lead_id"),
                "company": state.get("company_name"),
                "score": state.get("score"),
                "budget": state.get("budget_range"),
                "timeline": state.get("timeline")
            },
            options=["qualify_sql", "qualify_mql", "disqualify"],
            timeout_minutes=480,  # 8 hours
            tenant_id=state.get("tenant_id"),
        )

        state["approval_request_id"] = request.request_id
        state["requires_approval"] = True
        return state

    async def _complete_qualification(self, state: LeadQualificationState) -> LeadQualificationState:
        """Complete the qualification process."""
        state["status"] = WorkflowStatus.COMPLETED.value
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        return state

    def _check_budget_result(self, state: LeadQualificationState) -> str:
        return state.get("metadata", {}).get("budget_result", "uncertain")

    def _check_authority_result(self, state: LeadQualificationState) -> str:
        return state.get("metadata", {}).get("authority_result", "uncertain")

    def _check_need_result(self, state: LeadQualificationState) -> str:
        return state.get("metadata", {}).get("need_result", "medium")

    def _check_timeline_result(self, state: LeadQualificationState) -> str:
        return state.get("metadata", {}).get("timeline_result", "long_term")

    def _route_by_score(self, state: LeadQualificationState) -> str:
        return state.get("metadata", {}).get("final_route", "review")

    def _check_human_decision(self, state: LeadQualificationState) -> str:
        metadata = state.get("metadata", {})
        decision = metadata.get("human_decision", "mql")
        if decision == "qualify_sql":
            return "sql"
        elif decision == "qualify_mql":
            return "mql"
        return "disqualify"


# =============================================================================
# SYSTEM HEALING WORKFLOW TEMPLATE
# =============================================================================

class SystemHealingState(BaseWorkflowState):
    """State specific to system healing workflow."""
    error_id: str
    error_type: str
    error_message: str
    component: str
    severity: str
    healing_stage: str
    diagnosis: dict
    attempted_fixes: list[dict]
    healed: bool
    requires_manual_intervention: bool


class SystemHealingWorkflow(WorkflowTemplate):
    """
    System Self-Healing Workflow
    ============================
    OODA-based approach to automatic error recovery:
    1. Observe - Detect and gather error information
    2. Orient - Diagnose root cause
    3. Decide - Select healing strategy
    4. Act - Execute fix
    5. Verify - Confirm healing
    6. Learn - Store pattern for future
    """

    def __init__(self, **kwargs):
        super().__init__(workflow_type="system_healing", **kwargs)

    def define_nodes(self) -> dict[str, Callable]:
        return {
            "start": self._observe_error,
            "diagnose": self._diagnose_cause,
            "select_strategy": self._select_healing_strategy,
            "attempt_fix": self._attempt_fix,
            "verify_fix": self._verify_fix,
            "escalate": self._escalate_to_human,
            "record_learning": self._record_learning,
            "complete": self._complete_healing
        }

    def define_edges(self) -> list[tuple]:
        return [
            ("start", "diagnose"),
            ("diagnose", "select_strategy"),
            ("record_learning", "complete"),
            ("complete", "END")
        ]

    def define_conditional_edges(self) -> list[dict]:
        return [
            {
                "source": "select_strategy",
                "condition": self._check_can_auto_fix,
                "mapping": {
                    "auto_fix": "attempt_fix",
                    "manual_required": "escalate"
                }
            },
            {
                "source": "attempt_fix",
                "condition": self._check_fix_result,
                "mapping": {
                    "success": "verify_fix",
                    "partial": "verify_fix",
                    "failed": "escalate"
                }
            },
            {
                "source": "verify_fix",
                "condition": self._check_verification,
                "mapping": {
                    "healed": "record_learning",
                    "not_healed": "select_strategy",
                    "worse": "escalate"
                }
            },
            {
                "source": "escalate",
                "condition": self._check_escalation_response,
                "mapping": {
                    "resolved": "record_learning",
                    "retry": "attempt_fix",
                    "accept": "complete"
                }
            }
        ]

    def get_breakpoints(self) -> list[str]:
        return ["escalate"]

    async def _observe_error(self, state: SystemHealingState) -> SystemHealingState:
        """Observe and gather error information."""
        state["healing_stage"] = "observing"
        state["ooda_phase"] = OODAPhase.OBSERVE.value
        state["attempted_fixes"] = []
        state["healed"] = False
        state["requires_manual_intervention"] = False
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Gather observations
        observations = {
            "error_type": state.get("error_type"),
            "error_message": state.get("error_message"),
            "component": state.get("component"),
            "severity": state.get("severity"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        state["observations"] = [observations]
        logger.info(f"Observing error {state.get('error_id')}: {state.get('error_type')}")
        return state

    async def _diagnose_cause(self, state: SystemHealingState) -> SystemHealingState:
        """Diagnose the root cause of the error."""
        state["healing_stage"] = "diagnosing"
        state["ooda_phase"] = OODAPhase.ORIENT.value
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # Simple diagnosis based on error type
        error_type = state.get("error_type", "unknown")
        diagnoses = {
            "connection_error": {
                "root_cause": "network_connectivity",
                "likely_fixes": ["retry_connection", "check_firewall", "restart_service"],
                "auto_fixable": True
            },
            "timeout": {
                "root_cause": "resource_exhaustion",
                "likely_fixes": ["increase_timeout", "scale_resources", "optimize_query"],
                "auto_fixable": True
            },
            "authentication_error": {
                "root_cause": "credentials_issue",
                "likely_fixes": ["refresh_token", "rotate_credentials"],
                "auto_fixable": False
            },
            "database_error": {
                "root_cause": "database_issue",
                "likely_fixes": ["reconnect", "clear_pool", "restart_connection"],
                "auto_fixable": True
            },
            "memory_error": {
                "root_cause": "memory_exhaustion",
                "likely_fixes": ["gc_collect", "restart_service", "scale_up"],
                "auto_fixable": True
            },
            "unknown": {
                "root_cause": "unidentified",
                "likely_fixes": ["restart_service"],
                "auto_fixable": False
            }
        }

        diagnosis = diagnoses.get(error_type, diagnoses["unknown"])
        state["diagnosis"] = diagnosis
        state["orientation"] = diagnosis

        logger.info(f"Diagnosed error {state.get('error_id')}: {diagnosis['root_cause']}")
        return state

    async def _select_healing_strategy(self, state: SystemHealingState) -> SystemHealingState:
        """Select appropriate healing strategy."""
        state["healing_stage"] = "deciding"
        state["ooda_phase"] = OODAPhase.DECIDE.value
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        diagnosis = state.get("diagnosis", {})
        attempted = state.get("attempted_fixes", [])
        likely_fixes = diagnosis.get("likely_fixes", [])

        # Find next untried fix
        tried_fixes = {f.get("fix_type") for f in attempted}
        available_fixes = [f for f in likely_fixes if f not in tried_fixes]

        if available_fixes and diagnosis.get("auto_fixable", False):
            state["metadata"]["selected_fix"] = available_fixes[0]
            state["metadata"]["can_auto_fix"] = True
        else:
            state["metadata"]["can_auto_fix"] = False

        # Check retry limit
        if len(attempted) >= 3:
            state["metadata"]["can_auto_fix"] = False

        return state

    async def _attempt_fix(self, state: SystemHealingState) -> SystemHealingState:
        """Attempt to fix the error."""
        state["healing_stage"] = "acting"
        state["ooda_phase"] = OODAPhase.ACT.value
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        fix_type = state.get("metadata", {}).get("selected_fix", "restart_service")

        # Simulate fix execution
        fix_result = {
            "fix_type": fix_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": True,  # In real implementation, actually attempt the fix
            "details": f"Attempted {fix_type}"
        }

        state["attempted_fixes"].append(fix_result)
        state["metadata"]["last_fix_result"] = "success" if fix_result["success"] else "failed"

        logger.info(f"Attempted fix {fix_type} for error {state.get('error_id')}")
        return state

    async def _verify_fix(self, state: SystemHealingState) -> SystemHealingState:
        """Verify that the fix worked."""
        state["healing_stage"] = "verifying"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # In real implementation, actually verify the component is healthy
        last_fix = state.get("attempted_fixes", [{}])[-1]

        if last_fix.get("success"):
            state["metadata"]["verification_result"] = "healed"
            state["healed"] = True
        else:
            attempts = len(state.get("attempted_fixes", []))
            if attempts >= 3:
                state["metadata"]["verification_result"] = "worse"
            else:
                state["metadata"]["verification_result"] = "not_healed"

        return state

    async def _escalate_to_human(self, state: SystemHealingState) -> SystemHealingState:
        """Escalate to human operator."""
        state["healing_stage"] = "escalated"
        state["requires_manual_intervention"] = True
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        request = await self.hitl_manager.request_approval(
            workflow_id=state.get("workflow_id"),
            node_name="escalate",
            description=f"System healing failed for {state.get('component')}: {state.get('error_type')}",
            context={
                "error_id": state.get("error_id"),
                "error_type": state.get("error_type"),
                "error_message": state.get("error_message"),
                "component": state.get("component"),
                "diagnosis": state.get("diagnosis"),
                "attempted_fixes": state.get("attempted_fixes")
            },
            options=["manual_fix_applied", "retry_auto_fix", "accept_degraded"],
            timeout_minutes=30,
            tenant_id=state.get("tenant_id"),
        )

        state["approval_request_id"] = request.request_id
        state["requires_approval"] = True
        return state

    async def _record_learning(self, state: SystemHealingState) -> SystemHealingState:
        """Record the healing pattern for future learning."""
        state["healing_stage"] = "learning"
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        # In real implementation, store in knowledge base
        learning = {
            "error_type": state.get("error_type"),
            "component": state.get("component"),
            "diagnosis": state.get("diagnosis"),
            "successful_fix": state.get("attempted_fixes", [{}])[-1] if state.get("healed") else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        state["metadata"]["learning_recorded"] = learning
        logger.info(f"Recorded healing pattern for {state.get('error_type')}")
        return state

    async def _complete_healing(self, state: SystemHealingState) -> SystemHealingState:
        """Complete the healing workflow."""
        state["status"] = WorkflowStatus.COMPLETED.value
        state["healing_stage"] = "completed"
        state["result"] = {
            "healed": state.get("healed", False),
            "attempts": len(state.get("attempted_fixes", [])),
            "manual_intervention": state.get("requires_manual_intervention", False),
            "error_id": state.get("error_id")
        }
        state["last_updated_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Healing workflow completed for error {state.get('error_id')}: healed={state.get('healed')}")
        return state

    def _check_can_auto_fix(self, state: SystemHealingState) -> str:
        if state.get("metadata", {}).get("can_auto_fix"):
            return "auto_fix"
        return "manual_required"

    def _check_fix_result(self, state: SystemHealingState) -> str:
        return state.get("metadata", {}).get("last_fix_result", "failed")

    def _check_verification(self, state: SystemHealingState) -> str:
        return state.get("metadata", {}).get("verification_result", "not_healed")

    def _check_escalation_response(self, state: SystemHealingState) -> str:
        metadata = state.get("metadata", {})
        response = metadata.get("escalation_response", "accept")
        if response == "manual_fix_applied":
            return "resolved"
        elif response == "retry_auto_fix":
            return "retry"
        return "accept"


# =============================================================================
# WORKFLOW ENGINE - MAIN ORCHESTRATOR
# =============================================================================

class AdvancedWorkflowEngine:
    """
    Advanced Workflow Engine with LangGraph integration.
    Provides a unified interface for managing complex workflows with:
    - State machine patterns
    - Checkpoint/resume capability
    - Human-in-the-loop support
    - OODA loop integration
    """

    def __init__(self):
        self.checkpoint_saver = PostgresCheckpointSaver()
        self.hitl_manager = HumanInTheLoopManager(self.checkpoint_saver)
        self.ooda_executor = OODALoopExecutor()

        # Register workflow templates
        self._templates: dict[str, type[WorkflowTemplate]] = {
            "customer_onboarding": CustomerOnboardingWorkflow,
            "invoice_collection": InvoiceCollectionWorkflow,
            "lead_qualification": LeadQualificationWorkflow,
            "system_healing": SystemHealingWorkflow
        }

        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the workflow engine."""
        if self._initialized:
            return True

        try:
            await self.checkpoint_saver.initialize()
            self._initialized = True
            logger.info("AdvancedWorkflowEngine initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AdvancedWorkflowEngine: {e}")
            return False

    def register_workflow(self, name: str, template_class: type[WorkflowTemplate]) -> None:
        """Register a custom workflow template."""
        self._templates[name] = template_class

    async def start_workflow(
        self,
        workflow_type: str,
        initial_state: dict,
        tenant_id: str = None,
        timeout_seconds: int = 300
    ) -> dict:
        """Start a new workflow execution."""
        if not await self.initialize():
            return {"error": "Engine not initialized", "status": "failed"}

        template_class = self._templates.get(workflow_type)
        if not template_class:
            return {
                "error": f"Unknown workflow type: {workflow_type}",
                "available_types": list(self._templates.keys()),
                "status": "failed"
            }

        # Create template instance with shared components
        template = template_class(
            checkpoint_saver=self.checkpoint_saver,
            hitl_manager=self.hitl_manager,
            ooda_executor=self.ooda_executor,
            workflow_registry=self._templates,
        )

        # Execute workflow
        result = await template.execute(
            initial_state=initial_state,
            tenant_id=tenant_id,
            timeout_seconds=timeout_seconds
        )

        return result

    async def resume_workflow(
        self,
        workflow_id: str,
        additional_state: dict = None,
        timeout_seconds: int = 300
    ) -> dict:
        """Resume a paused or checkpointed workflow."""
        if not await self.initialize():
            return {"error": "Engine not initialized", "status": "failed"}

        # Load checkpoint
        tenant_hint = (additional_state or {}).get("tenant_id")
        checkpoint = await self.checkpoint_saver.load_checkpoint(
            workflow_id,
            tenant_id=tenant_hint,
        )
        if not checkpoint:
            return {
                "error": f"No checkpoint found for workflow: {workflow_id}",
                "status": "failed"
            }

        # Get template class
        template_class = self._templates.get(checkpoint.workflow_type)
        if not template_class:
            return {
                "error": f"Unknown workflow type: {checkpoint.workflow_type}",
                "status": "failed"
            }

        # Merge additional state
        state = checkpoint.state_data
        if additional_state:
            state.update(additional_state)

        # Resume execution
        template = template_class(
            checkpoint_saver=self.checkpoint_saver,
            hitl_manager=self.hitl_manager,
            ooda_executor=self.ooda_executor,
            workflow_registry=self._templates,
        )

        result = await template.execute(
            initial_state=state,
            tenant_id=state.get("tenant_id"),
            timeout_seconds=timeout_seconds
        )

        return result

    async def submit_approval(
        self,
        request_id: str,
        response: str,
        responded_by: str
    ) -> dict:
        """Submit human approval for a workflow breakpoint."""
        if not await self.initialize():
            return {"error": "Engine not initialized", "success": False}

        # Get the approval request
        request = await self.hitl_manager.get_approval_status(request_id)
        if not request:
            return {"error": "Approval request not found", "success": False}

        # Submit the response
        success = await self.hitl_manager.submit_approval(
            request_id,
            response,
            responded_by,
        )

        if success:
            # Update workflow state with the decision
            checkpoint = await self.checkpoint_saver.load_checkpoint(
                request.workflow_id,
                tenant_id=request.context.get("tenant_id") if isinstance(request.context, dict) else None,
            )
            if checkpoint:
                state = checkpoint.state_data
                state.setdefault("metadata", {})["human_decision"] = response
                state["approval_status"] = HumanApprovalStatus.APPROVED.value if response == "approve" else HumanApprovalStatus.REJECTED.value
                state["requires_approval"] = False

                await self.checkpoint_saver.save_checkpoint(
                    request.workflow_id,
                    checkpoint.workflow_type,
                    state,
                    checkpoint.current_node,
                    tenant_id=state.get("tenant_id"),
                )

        return {
            "success": success,
            "workflow_id": request.workflow_id,
            "response": response
        }

    async def get_pending_approvals(
        self,
        tenant_id: str = None,
        workflow_type: str = None
    ) -> list[dict]:
        """Get all pending human approval requests."""
        if not await self.initialize():
            return []

        requests = await self.hitl_manager.get_pending_approvals(
            tenant_id=tenant_id,
            workflow_type=workflow_type
        )

        return [
            {
                "request_id": r.request_id,
                "workflow_id": r.workflow_id,
                "node_name": r.node_name,
                "description": r.description,
                "options": r.options,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "timeout_minutes": r.timeout_minutes
            }
            for r in requests
        ]

    async def get_workflow_status(self, workflow_id: str) -> dict:
        """Get current status of a workflow."""
        if not await self.initialize():
            return {"error": "Engine not initialized"}

        checkpoint = await self.checkpoint_saver.load_checkpoint(workflow_id)
        if not checkpoint:
            return {
                "workflow_id": workflow_id,
                "status": "not_found",
                "message": "Workflow completed or does not exist"
            }

        state = checkpoint.state_data
        return {
            "workflow_id": workflow_id,
            "workflow_type": checkpoint.workflow_type,
            "status": state.get("status", "unknown"),
            "current_node": checkpoint.current_node,
            "requires_approval": state.get("requires_approval", False),
            "approval_request_id": state.get("approval_request_id"),
            "errors": state.get("errors", []),
            "last_updated": checkpoint.created_at.isoformat() if checkpoint.created_at else None
        }

    def get_available_workflows(self) -> list[dict]:
        """Get list of available workflow templates."""
        return [
            {
                "type": name,
                "breakpoints": cls(
                    checkpoint_saver=self.checkpoint_saver,
                    hitl_manager=self.hitl_manager,
                    workflow_registry=self._templates,
                ).get_breakpoints()
            }
            for name, cls in self._templates.items()
        ]


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_workflow_engine: Optional[AdvancedWorkflowEngine] = None


def get_workflow_engine() -> AdvancedWorkflowEngine:
    """Get singleton workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = AdvancedWorkflowEngine()
    return _workflow_engine


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    async def main():
        engine = get_workflow_engine()
        await engine.initialize()

        if len(sys.argv) > 1:
            command = sys.argv[1]

            if command == "list":
                workflows = engine.get_available_workflows()
                print(json.dumps(workflows, indent=2))

            elif command == "start" and len(sys.argv) > 2:
                workflow_type = sys.argv[2]
                result = await engine.start_workflow(
                    workflow_type=workflow_type,
                    initial_state={
                        "customer_id": "test-123",
                        "customer_name": "Test Customer",
                        "customer_email": "test@example.com"
                    }
                )
                print(json.dumps(result, indent=2, default=str))

            elif command == "approvals":
                approvals = await engine.get_pending_approvals()
                print(json.dumps(approvals, indent=2, default=str))

            else:
                print("Usage: python langgraph_workflow_engine.py [list|start <type>|approvals]")
        else:
            # Run demo
            print("Running Customer Onboarding demo...")
            result = await engine.start_workflow(
                workflow_type="customer_onboarding",
                initial_state={
                    "customer_id": "demo-customer",
                    "customer_name": "Demo Corp",
                    "customer_email": "demo@example.com",
                    "metadata": {
                        "has_import_data": False,
                        "training_requested": True
                    }
                }
            )
            print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())
