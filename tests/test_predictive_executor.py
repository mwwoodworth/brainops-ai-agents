#!/usr/bin/env python3
"""
Tests for the Predictive Executor module.

Tests cover:
- Safety evaluation (blocked actions, confidence, risk)
- Execution flow
- Learning from outcomes
- Integration with OODA and ProactiveIntelligence
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictive_executor import (
    ExecutionDecision,
    ExecutionEvaluation,
    PredictedTask,
    PredictionSource,
    PredictiveExecution,
    PredictiveExecutor,
    RiskLevel,
    get_predictive_executor,
    get_ooda_integration,
    get_proactive_intel_integration,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def executor():
    """Create a fresh PredictiveExecutor instance."""
    return PredictiveExecutor(
        confidence_threshold=0.80,
        risk_threshold=0.30,
        max_concurrent_executions=5,
        rate_limit_per_minute=30
    )


@pytest.fixture
def safe_prediction():
    """Create a safe prediction that should be executed."""
    return PredictedTask(
        id="test_safe_1",
        source=PredictionSource.PATTERN_MATCH,
        task_type="fetch_customer_data",
        description="Prefetch customer data",
        confidence=0.92,
        parameters={"customer_id": 12345}
    )


@pytest.fixture
def low_confidence_prediction():
    """Create a prediction with low confidence."""
    return PredictedTask(
        id="test_low_conf",
        source=PredictionSource.USER_BEHAVIOR,
        task_type="analyze_data",
        description="Low confidence analysis",
        confidence=0.65,  # Below 0.80 threshold
        parameters={}
    )


@pytest.fixture
def blocked_prediction():
    """Create a prediction with a blocked action type."""
    return PredictedTask(
        id="test_blocked",
        source=PredictionSource.OODA_SPECULATION,
        task_type="send_email",  # Blocked action
        description="Send notification email",
        confidence=0.95,
        parameters={"to": "user@example.com"}
    )


@pytest.fixture
def delete_prediction():
    """Create a prediction that tries to delete data."""
    return PredictedTask(
        id="test_delete",
        source=PredictionSource.SYSTEM_STATE,
        task_type="delete_old_records",  # Blocked action
        description="Delete old records",
        confidence=0.99,
        parameters={"older_than_days": 30}
    )


# =============================================================================
# Safety Evaluation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_safe_action_passes_evaluation(executor, safe_prediction):
    """Test that safe actions pass evaluation."""
    # Mock database connection to avoid real DB calls
    with patch.object(executor, '_get_connection', return_value=MagicMock()):
        with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
            evaluation = await executor.evaluate_prediction(safe_prediction)

    assert evaluation.should_execute is True
    assert evaluation.decision == ExecutionDecision.EXECUTE
    assert evaluation.confidence == 0.92
    assert evaluation.risk_level == RiskLevel.NONE
    assert "confidence_check: PASSED" in evaluation.safety_checks
    assert "blocked_action_check: PASSED" in evaluation.safety_checks


@pytest.mark.asyncio
async def test_low_confidence_fails_evaluation(executor, low_confidence_prediction):
    """Test that low confidence predictions are skipped."""
    evaluation = await executor.evaluate_prediction(low_confidence_prediction)

    assert evaluation.should_execute is False
    assert evaluation.decision == ExecutionDecision.SKIP_LOW_CONFIDENCE
    assert "confidence_check: FAILED" in evaluation.safety_checks


@pytest.mark.asyncio
async def test_blocked_action_fails_evaluation(executor, blocked_prediction):
    """Test that blocked actions (send_email) are rejected."""
    evaluation = await executor.evaluate_prediction(blocked_prediction)

    assert evaluation.should_execute is False
    assert evaluation.decision == ExecutionDecision.SKIP_BLOCKED_ACTION
    assert evaluation.risk_level == RiskLevel.CRITICAL
    assert any("send_email" in check for check in evaluation.safety_checks)


@pytest.mark.asyncio
async def test_delete_action_fails_evaluation(executor, delete_prediction):
    """Test that delete actions are rejected."""
    evaluation = await executor.evaluate_prediction(delete_prediction)

    assert evaluation.should_execute is False
    assert evaluation.decision == ExecutionDecision.SKIP_BLOCKED_ACTION
    assert evaluation.risk_level == RiskLevel.CRITICAL


@pytest.mark.asyncio
async def test_all_blocked_actions_rejected(executor):
    """Test that all blocked action types are properly rejected."""
    blocked_types = [
        "delete_user",
        "drop_table",
        "truncate_data",
        "send_notification",
        "publish_message",
        "process_payment",
        "reset_password",
        "shutdown_service",
    ]

    for action_type in blocked_types:
        prediction = PredictedTask(
            id=f"test_{action_type}",
            source=PredictionSource.PATTERN_MATCH,
            task_type=action_type,
            description=f"Test {action_type}",
            confidence=0.99,
            parameters={}
        )
        evaluation = await executor.evaluate_prediction(prediction)
        assert evaluation.should_execute is False, f"Action '{action_type}' should be blocked"
        assert evaluation.decision == ExecutionDecision.SKIP_BLOCKED_ACTION


# =============================================================================
# Risk Assessment Tests
# =============================================================================

@pytest.mark.asyncio
async def test_read_only_action_has_no_risk(executor):
    """Test that read-only actions have no risk."""
    prediction = PredictedTask(
        id="test_query",
        source=PredictionSource.PATTERN_MATCH,
        task_type="query_database",
        description="Query database",
        confidence=0.90,
        parameters={}
    )

    with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
        risk_level, risk_score = await executor._assess_risk(prediction)

    assert risk_level == RiskLevel.NONE
    assert risk_score == 0.0


@pytest.mark.asyncio
async def test_external_target_increases_risk(executor):
    """Test that external targets increase risk score."""
    prediction = PredictedTask(
        id="test_external",
        source=PredictionSource.PATTERN_MATCH,
        task_type="modify_config",  # Not a safe action type
        description="Modify config with webhook",
        confidence=0.90,
        parameters={"webhook_url": "https://example.com/hook"}
    )

    with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
        risk_level, risk_score = await executor._assess_risk(prediction)

    # External URL should increase risk - modify_config + external URL = 0.2 + 0.3 = 0.5
    assert risk_score >= 0.5


@pytest.mark.asyncio
async def test_sensitive_data_increases_risk(executor):
    """Test that sensitive data patterns increase risk."""
    prediction = PredictedTask(
        id="test_sensitive",
        source=PredictionSource.PATTERN_MATCH,
        task_type="update_config",
        description="Update configuration",
        confidence=0.90,
        parameters={"api_token": "secret123"}
    )

    with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
        risk_level, risk_score = await executor._assess_risk(prediction)

    # Sensitive data pattern should increase risk
    assert risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)
    assert risk_score > 0.4


# =============================================================================
# Execution Tests
# =============================================================================

@pytest.mark.asyncio
async def test_execute_safe_prediction(executor, safe_prediction):
    """Test executing a safe prediction."""
    # Mock database operations
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
            result = await executor.execute_prediction(safe_prediction)

    assert result["executed"] is True
    assert "execution_id" in result
    assert result["evaluation"]["should_execute"] is True


@pytest.mark.asyncio
async def test_skip_blocked_prediction(executor, blocked_prediction):
    """Test that blocked predictions are skipped, not executed."""
    # Mock database operations
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        result = await executor.execute_prediction(blocked_prediction)

    assert result["executed"] is False
    assert result["skipped"] is True
    assert result["decision"] == ExecutionDecision.SKIP_BLOCKED_ACTION.value


@pytest.mark.asyncio
async def test_custom_executor_called(executor, safe_prediction):
    """Test that custom executors are called when provided."""
    custom_result = {"custom": "result", "status": "success"}
    custom_executor = AsyncMock(return_value=custom_result)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
            result = await executor.execute_prediction(safe_prediction, executor=custom_executor)

    assert result["executed"] is True
    custom_executor.assert_called_once_with(
        safe_prediction.task_type,
        safe_prediction.parameters
    )


# =============================================================================
# Rate Limiting Tests
# =============================================================================

@pytest.mark.asyncio
async def test_rate_limiting(executor, safe_prediction):
    """Test that rate limiting prevents excess executions."""
    # Fill up the rate limit
    from datetime import datetime
    executor.execution_timestamps = [datetime.utcnow() for _ in range(30)]

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
            evaluation = await executor.evaluate_prediction(safe_prediction)

    assert evaluation.should_execute is False
    assert evaluation.decision == ExecutionDecision.SKIP_RATE_LIMITED


# =============================================================================
# Learning Tests
# =============================================================================

@pytest.mark.asyncio
async def test_learn_from_accurate_outcome(executor):
    """Test learning from accurate predictions."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        await executor.learn_from_outcome(
            execution_id="test_exec_123",
            was_accurate=True,
            feedback="Prediction matched actual need"
        )

    assert executor.accurate_predictions == 1
    assert executor.inaccurate_predictions == 0


@pytest.mark.asyncio
async def test_learn_from_inaccurate_outcome(executor):
    """Test learning from inaccurate predictions."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        await executor.learn_from_outcome(
            execution_id="test_exec_456",
            was_accurate=False,
            feedback="User did not need this"
        )

    assert executor.accurate_predictions == 0
    assert executor.inaccurate_predictions == 1


# =============================================================================
# Integration Tests
# =============================================================================

def test_ooda_integration_creation():
    """Test OODA integration can be created."""
    integration = get_ooda_integration()
    assert integration is not None
    assert hasattr(integration, 'execute_ooda_speculations')


def test_proactive_intel_integration_creation():
    """Test ProactiveIntelligence integration can be created."""
    integration = get_proactive_intel_integration()
    assert integration is not None
    assert hasattr(integration, 'execute_proactive_recommendations')


@pytest.mark.asyncio
async def test_ooda_integration_converts_speculations():
    """Test that OODA speculations are properly converted."""
    integration = get_ooda_integration()

    speculations = [
        {
            "type": "fetch_health",
            "probability": 0.85,
            "params": {"service": "api"}
        },
        {
            "action_type": "query_metrics",
            "confidence": 0.90,
            "parameters": {"period": "1h"}
        }
    ]

    # Mock the executor's execute_batch
    with patch.object(integration.executor, 'execute_batch', new_callable=AsyncMock) as mock_batch:
        mock_batch.return_value = [
            {"executed": True, "execution_id": "test1"},
            {"executed": True, "execution_id": "test2"}
        ]

        results = await integration.execute_ooda_speculations(speculations)

    # Verify execute_batch was called with PredictedTasks
    assert mock_batch.called
    call_args = mock_batch.call_args
    predictions = call_args[0][0]
    assert len(predictions) == 2
    assert all(isinstance(p, PredictedTask) for p in predictions)


# =============================================================================
# Edge Cases
# =============================================================================

@pytest.mark.asyncio
async def test_empty_batch_execution(executor):
    """Test executing an empty batch."""
    results = await executor.execute_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_prediction_with_dependencies(executor):
    """Test prediction with dependencies is handled."""
    prediction = PredictedTask(
        id="test_deps",
        source=PredictionSource.PATTERN_MATCH,
        task_type="fetch_data",
        description="Fetch dependent data",
        confidence=0.90,
        parameters={},
        dependencies=["dep_1", "dep_2"]
    )

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch.object(executor, '_get_connection', return_value=mock_conn):
        with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
            evaluation = await executor.evaluate_prediction(prediction)

    # Dependencies don't block execution (they're informational)
    assert evaluation.should_execute is True


@pytest.mark.asyncio
async def test_concurrent_execution_limit(executor, safe_prediction):
    """Test that concurrent execution limit is respected."""
    # Fill up active executions
    for i in range(5):
        executor.active_executions[f"exec_{i}"] = PredictiveExecution(
            id=f"exec_{i}",
            prediction_id=f"pred_{i}",
            task_type="test",
            parameters={},
            status="executing"
        )

    with patch.object(executor, '_get_historical_failure_rate', return_value=0.0):
        evaluation = await executor.evaluate_prediction(safe_prediction)

    assert evaluation.should_execute is False
    assert evaluation.decision == ExecutionDecision.DEFER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
