#!/usr/bin/env python3
"""
Tests for Advanced LangGraph Workflow Engine
=============================================
Tests workflow templates, state machines, checkpointing,
and human-in-the-loop patterns.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_workflow_engine import (
    AdvancedWorkflowEngine,
    CustomerOnboardingWorkflow,
    InvoiceCollectionWorkflow,
    LeadQualificationWorkflow,
    SystemHealingWorkflow,
    PostgresCheckpointSaver,
    HumanInTheLoopManager,
    OODALoopExecutor,
    WorkflowStatus,
    OODAPhase,
    HumanApprovalStatus,
    BaseWorkflowState
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def checkpoint_saver():
    """Create a mock checkpoint saver."""
    saver = PostgresCheckpointSaver()
    saver._initialized = True  # Skip DB init
    return saver


@pytest.fixture
def hitl_manager(checkpoint_saver):
    """Create a mock HITL manager."""
    return HumanInTheLoopManager(checkpoint_saver)


@pytest.fixture
def ooda_executor():
    """Create an OODA loop executor."""
    return OODALoopExecutor()


@pytest.fixture
def workflow_engine():
    """Create workflow engine with mocked dependencies."""
    engine = AdvancedWorkflowEngine()
    engine._initialized = True
    return engine


# =============================================================================
# WORKFLOW ENGINE TESTS
# =============================================================================

class TestAdvancedWorkflowEngine:
    """Tests for the main workflow engine."""

    def test_engine_initialization(self):
        """Test engine initializes with default workflows."""
        engine = AdvancedWorkflowEngine()

        assert "customer_onboarding" in engine._templates
        assert "invoice_collection" in engine._templates
        assert "lead_qualification" in engine._templates
        assert "system_healing" in engine._templates

    def test_get_available_workflows(self, workflow_engine):
        """Test getting available workflow types."""
        workflows = workflow_engine.get_available_workflows()

        assert len(workflows) == 4
        types = [w["type"] for w in workflows]
        assert "customer_onboarding" in types
        assert "invoice_collection" in types
        assert "lead_qualification" in types
        assert "system_healing" in types

    def test_register_custom_workflow(self, workflow_engine):
        """Test registering a custom workflow template."""
        class CustomWorkflow(CustomerOnboardingWorkflow):
            pass

        workflow_engine.register_workflow("custom", CustomWorkflow)

        assert "custom" in workflow_engine._templates
        workflows = workflow_engine.get_available_workflows()
        types = [w["type"] for w in workflows]
        assert "custom" in types


# =============================================================================
# OODA LOOP TESTS
# =============================================================================

class TestOODALoopExecutor:
    """Tests for OODA loop execution."""

    @pytest.mark.asyncio
    async def test_observe_phase(self, ooda_executor):
        """Test the OBSERVE phase gathers correct information."""
        state: BaseWorkflowState = {
            "workflow_id": "test-123",
            "status": "running",
            "current_node": "test_node",
            "retry_count": 0,
            "errors": []
        }
        context = {"test_key": "test_value"}

        observations = await ooda_executor.observe(state, context)

        assert "timestamp" in observations
        assert observations["workflow_status"] == "running"
        assert observations["current_node"] == "test_node"
        assert observations["retry_count"] == 0
        assert observations["context"] == context

    @pytest.mark.asyncio
    async def test_orient_phase(self, ooda_executor):
        """Test the ORIENT phase analyzes observations."""
        state: BaseWorkflowState = {
            "workflow_id": "test-123",
            "errors": [{"error": "test error"}]
        }
        observations = {
            "errors": [{"error": "test error"}],
            "retry_count": 0
        }

        orientation = await ooda_executor.orient(state, observations)

        assert "timestamp" in orientation
        assert "risks" in orientation
        assert len(orientation["risks"]) > 0  # Should detect error risk

    @pytest.mark.asyncio
    async def test_decide_phase_with_risks(self, ooda_executor):
        """Test DECIDE phase selects appropriate action for risky situation."""
        state: BaseWorkflowState = {"workflow_id": "test-123"}
        orientation = {
            "risks": [{"type": "execution_errors", "severity": "high"}]
        }
        available_actions = ["proceed", "escalate", "retry"]

        decision = await ooda_executor.decide(state, orientation, available_actions)

        assert decision["selected_action"] == "escalate"
        assert decision["confidence"] >= 0.5

    @pytest.mark.asyncio
    async def test_decide_phase_no_risks(self, ooda_executor):
        """Test DECIDE phase proceeds normally without risks."""
        state: BaseWorkflowState = {"workflow_id": "test-123"}
        orientation = {"risks": []}
        available_actions = ["proceed", "escalate", "retry"]

        decision = await ooda_executor.decide(state, orientation, available_actions)

        assert decision["selected_action"] == "proceed"
        assert decision["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_act_phase(self, ooda_executor):
        """Test ACT phase executes the decided action."""
        state: BaseWorkflowState = {"workflow_id": "test-123"}
        decision = {"selected_action": "test_action"}

        async def test_handler(s, d):
            return {"handled": True}

        action_handlers = {"test_action": test_handler}

        result = await ooda_executor.act(state, decision, action_handlers)

        assert result["success"] is True
        assert result["action"] == "test_action"
        assert result["result"]["handled"] is True

    @pytest.mark.asyncio
    async def test_full_ooda_loop(self, ooda_executor):
        """Test complete OODA loop execution."""
        state: BaseWorkflowState = {
            "workflow_id": "test-123",
            "status": "running",
            "current_node": "test",
            "retry_count": 0,
            "errors": []
        }
        context = {"mission": "test"}
        available_actions = ["proceed", "escalate"]

        async def proceed_handler(s, d):
            return {"proceeded": True}

        action_handlers = {"proceed": proceed_handler}

        result, updated_state = await ooda_executor.execute_loop(
            state, context, available_actions, action_handlers
        )

        assert result["success"] is True
        assert "observations" in updated_state
        assert "orientation" in updated_state
        assert "decisions" in updated_state
        assert "actions_taken" in updated_state


# =============================================================================
# CUSTOMER ONBOARDING WORKFLOW TESTS
# =============================================================================

class TestCustomerOnboardingWorkflow:
    """Tests for customer onboarding workflow."""

    def test_workflow_nodes_defined(self, checkpoint_saver, hitl_manager):
        """Test all required nodes are defined."""
        workflow = CustomerOnboardingWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        nodes = workflow.define_nodes()

        required_nodes = [
            "start", "send_welcome", "setup_account", "import_data",
            "configure_settings", "schedule_training", "create_first_project",
            "success_check", "complete", "escalate"
        ]

        for node in required_nodes:
            assert node in nodes, f"Missing node: {node}"

    def test_workflow_breakpoints(self, checkpoint_saver, hitl_manager):
        """Test breakpoints are correctly defined."""
        workflow = CustomerOnboardingWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        breakpoints = workflow.get_breakpoints()

        assert "import_data" in breakpoints
        assert "schedule_training" in breakpoints

    @pytest.mark.asyncio
    async def test_start_onboarding_node(self, checkpoint_saver, hitl_manager):
        """Test the start onboarding node initializes state correctly."""
        workflow = CustomerOnboardingWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        state = {
            "customer_id": "test-123",
            "customer_name": "Test Corp",
            "customer_email": "test@example.com"
        }

        result = await workflow._start_onboarding(state)

        assert result["onboarding_stage"] == "started"
        assert result["welcome_email_sent"] is False
        assert result["setup_completed"] is False
        assert result["training_scheduled"] is False
        assert result["first_project_created"] is False


# =============================================================================
# INVOICE COLLECTION WORKFLOW TESTS
# =============================================================================

class TestInvoiceCollectionWorkflow:
    """Tests for invoice collection workflow."""

    def test_workflow_nodes_defined(self, checkpoint_saver, hitl_manager):
        """Test all required nodes are defined."""
        workflow = InvoiceCollectionWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        nodes = workflow.define_nodes()

        required_nodes = [
            "start", "gentle_reminder", "follow_up", "payment_plan",
            "human_review", "final_notice", "collections",
            "payment_received", "close"
        ]

        for node in required_nodes:
            assert node in nodes, f"Missing node: {node}"

    def test_route_by_overdue_days(self, checkpoint_saver, hitl_manager):
        """Test routing logic based on days overdue."""
        workflow = InvoiceCollectionWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        # Test different overdue scenarios
        assert workflow._route_by_overdue_days({"days_overdue": 0}) == "not_overdue"
        assert workflow._route_by_overdue_days({"days_overdue": 5}) == "gentle"
        assert workflow._route_by_overdue_days({"days_overdue": 10}) == "follow_up"
        assert workflow._route_by_overdue_days({"days_overdue": 20}) == "payment_plan"
        assert workflow._route_by_overdue_days({"days_overdue": 45}) == "escalate"


# =============================================================================
# LEAD QUALIFICATION WORKFLOW TESTS
# =============================================================================

class TestLeadQualificationWorkflow:
    """Tests for BANT lead qualification workflow."""

    def test_workflow_nodes_defined(self, checkpoint_saver, hitl_manager):
        """Test all required nodes are defined."""
        workflow = LeadQualificationWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        nodes = workflow.define_nodes()

        required_nodes = [
            "start", "budget_check", "authority_check", "need_assessment",
            "timeline_check", "final_scoring", "route_mql", "route_sql",
            "disqualify", "human_review", "complete"
        ]

        for node in required_nodes:
            assert node in nodes, f"Missing node: {node}"

    @pytest.mark.asyncio
    async def test_initial_scoring(self, checkpoint_saver, hitl_manager):
        """Test initial lead scoring logic."""
        workflow = LeadQualificationWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        state = {
            "lead_id": "lead-123",
            "lead_source": "referral",  # Should add 25 points
            "company_size": "enterprise"  # Should add 30 points
        }

        result = await workflow._initial_scoring(state)

        assert result["score"] == 55  # 25 + 30
        assert result["qualification_stage"] == "initial"


# =============================================================================
# SYSTEM HEALING WORKFLOW TESTS
# =============================================================================

class TestSystemHealingWorkflow:
    """Tests for OODA-based system healing workflow."""

    def test_workflow_nodes_defined(self, checkpoint_saver, hitl_manager):
        """Test all required nodes are defined."""
        workflow = SystemHealingWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        nodes = workflow.define_nodes()

        required_nodes = [
            "start", "diagnose", "select_strategy", "attempt_fix",
            "verify_fix", "escalate", "record_learning", "complete"
        ]

        for node in required_nodes:
            assert node in nodes, f"Missing node: {node}"

    @pytest.mark.asyncio
    async def test_diagnose_connection_error(self, checkpoint_saver, hitl_manager):
        """Test diagnosis of connection errors."""
        workflow = SystemHealingWorkflow(
            checkpoint_saver=checkpoint_saver,
            hitl_manager=hitl_manager
        )

        state = {
            "error_id": "err-123",
            "error_type": "connection_error",
            "error_message": "Connection refused",
            "component": "database",
            "observations": []
        }

        result = await workflow._diagnose_cause(state)

        assert result["diagnosis"]["root_cause"] == "network_connectivity"
        assert result["diagnosis"]["auto_fixable"] is True
        assert "retry_connection" in result["diagnosis"]["likely_fixes"]


# =============================================================================
# WORKFLOW STATUS TESTS
# =============================================================================

class TestWorkflowStatus:
    """Tests for workflow status enum."""

    def test_status_values(self):
        """Test all expected status values exist."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.PAUSED.value == "paused"
        assert WorkflowStatus.WAITING.value == "waiting"
        assert WorkflowStatus.CHECKPOINT.value == "checkpoint"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


class TestOODAPhase:
    """Tests for OODA phase enum."""

    def test_phase_values(self):
        """Test all OODA phases exist."""
        assert OODAPhase.OBSERVE.value == "observe"
        assert OODAPhase.ORIENT.value == "orient"
        assert OODAPhase.DECIDE.value == "decide"
        assert OODAPhase.ACT.value == "act"


class TestHumanApprovalStatus:
    """Tests for human approval status enum."""

    def test_approval_status_values(self):
        """Test all approval status values exist."""
        assert HumanApprovalStatus.NOT_REQUIRED.value == "not_required"
        assert HumanApprovalStatus.PENDING.value == "pending"
        assert HumanApprovalStatus.APPROVED.value == "approved"
        assert HumanApprovalStatus.REJECTED.value == "rejected"
        assert HumanApprovalStatus.TIMEOUT.value == "timeout"


# =============================================================================
# INTEGRATION TESTS (Mock Database)
# =============================================================================

class TestWorkflowEngineIntegration:
    """Integration tests with mocked database."""

    @pytest.mark.asyncio
    async def test_start_workflow_unknown_type(self, workflow_engine):
        """Test error handling for unknown workflow type."""
        result = await workflow_engine.start_workflow(
            workflow_type="nonexistent",
            initial_state={}
        )

        assert result["status"] == "failed"
        assert "Unknown workflow type" in result["error"]
        assert "available_types" in result


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
