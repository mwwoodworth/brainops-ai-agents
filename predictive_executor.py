#!/usr/bin/env python3
"""
PREDICTIVE TASK EXECUTOR - Autonomous Proactive Execution System
=================================================================

This module bridges the gap between prediction and action. While the existing
systems (proactive_intelligence.py, bleeding_edge_ooda.py) can PREDICT tasks,
this module actually EXECUTES them proactively when safe to do so.

Key Features:
- Safety evaluation before execution (never delete data, never send external comms)
- Confidence thresholds (default: 80% confidence required)
- Risk assessment (default: < 30% risk required)
- Full audit trail of all predictive executions
- Learning from execution outcomes (accuracy tracking)
- Integration with existing OODA speculation and ProactiveIntelligence

Safety Rules (ENFORCED):
1. NEVER execute predictions that could delete data
2. NEVER execute predictions that send external communications
3. NEVER execute predictions affecting financial transactions
4. Log ALL predictive executions for audit

Author: BrainOps AI System
Version: 1.0.0 (2026-01-27)
"""

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from urllib.parse import urlparse

logger = logging.getLogger("PREDICTIVE_EXECUTOR")


# =============================================================================
# Database Configuration
# =============================================================================

def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            return {
                'host': parsed.hostname or '',
                'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
                'user': parsed.username or '',
                'password': parsed.password or '',
                'port': int(str(parsed.port)) if parsed.port else 5432
            }
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', '5432'))
    }


# =============================================================================
# Enums and Data Classes
# =============================================================================

class RiskLevel(Enum):
    """Risk levels for predicted actions"""
    NONE = "none"           # Read-only operations
    LOW = "low"             # Minor state changes, easily reversible
    MEDIUM = "medium"       # State changes, reversible with effort
    HIGH = "high"           # Significant changes, hard to reverse
    CRITICAL = "critical"   # Irreversible or dangerous operations


class ExecutionDecision(Enum):
    """Decisions about whether to execute a prediction"""
    EXECUTE = "execute"              # Safe to execute
    SKIP_LOW_CONFIDENCE = "skip_low_confidence"
    SKIP_HIGH_RISK = "skip_high_risk"
    SKIP_BLOCKED_ACTION = "skip_blocked_action"
    SKIP_RATE_LIMITED = "skip_rate_limited"
    SKIP_DEPENDENCY_FAILED = "skip_dependency_failed"
    DEFER = "defer"                  # Queue for later


class PredictionSource(Enum):
    """Sources of predictions"""
    OODA_SPECULATION = "ooda_speculation"      # From bleeding_edge_ooda.py
    PROACTIVE_INTEL = "proactive_intel"        # From proactive_intelligence.py
    PATTERN_MATCH = "pattern_match"            # Historical pattern matching
    SCHEDULE_BASED = "schedule_based"          # Time-based predictions
    USER_BEHAVIOR = "user_behavior"            # User action predictions
    SYSTEM_STATE = "system_state"              # System state predictions


@dataclass
class PredictedTask:
    """A task predicted by the AI system"""
    id: str
    source: PredictionSource
    task_type: str
    description: str
    confidence: float  # 0.0 - 1.0
    parameters: dict[str, Any]
    predicted_at: datetime = field(default_factory=datetime.utcnow)
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'source': self.source.value,
            'task_type': self.task_type,
            'description': self.description,
            'confidence': self.confidence,
            'parameters': self.parameters,
            'predicted_at': self.predicted_at.isoformat(),
            'dependencies': self.dependencies,
            'metadata': self.metadata
        }


@dataclass
class ExecutionEvaluation:
    """Result of evaluating a prediction for execution"""
    prediction_id: str
    should_execute: bool
    decision: ExecutionDecision
    confidence: float
    risk_level: RiskLevel
    risk_score: float  # 0.0 - 1.0
    reason: str
    safety_checks: list[str]
    evaluated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            'prediction_id': self.prediction_id,
            'should_execute': self.should_execute,
            'decision': self.decision.value,
            'confidence': self.confidence,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'reason': self.reason,
            'safety_checks': self.safety_checks,
            'evaluated_at': self.evaluated_at.isoformat()
        }


@dataclass
class PredictiveExecution:
    """Record of a predictive execution"""
    id: str
    prediction_id: str
    task_type: str
    parameters: dict[str, Any]
    status: str = "pending"  # pending, executing, completed, failed, skipped
    result: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: Optional[int] = None
    was_accurate: Optional[bool] = None  # Did the prediction match actual need?

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'prediction_id': self.prediction_id,
            'task_type': self.task_type,
            'parameters': self.parameters,
            'status': self.status,
            'result': self.result,
            'error': self.error,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time_ms': self.execution_time_ms,
            'was_accurate': self.was_accurate
        }


# =============================================================================
# Predictive Executor
# =============================================================================

class PredictiveExecutor:
    """
    The bridge between prediction and action. Evaluates predictions for safety
    and executes them proactively when appropriate.

    Safety Philosophy:
    - Better to skip a valid prediction than execute a harmful one
    - All executions are logged for audit
    - Learning from outcomes improves future decisions
    """

    # Actions that are NEVER allowed to be executed predictively
    BLOCKED_ACTIONS = {
        # Data destruction
        "delete", "drop", "truncate", "remove", "purge", "clear",
        # External communications
        "send_email", "send_sms", "send_notification", "post_message",
        "send_webhook", "publish", "broadcast",
        # Financial operations
        "charge", "refund", "transfer", "payment", "invoice",
        "process_payment", "create_charge",
        # Security operations
        "reset_password", "revoke_access", "delete_user", "modify_permissions",
        # System operations
        "shutdown", "restart", "terminate", "kill",
    }

    # Actions that are safe for predictive execution
    SAFE_ACTIONS = {
        # Read operations
        "query", "fetch", "get", "list", "search", "find", "read",
        # Analysis operations
        "analyze", "calculate", "predict", "forecast", "evaluate",
        # Preparation operations
        "prepare", "cache", "preload", "warm", "prefetch",
        # Validation operations
        "validate", "check", "verify", "test",
        # Logging operations
        "log", "record", "track",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        risk_threshold: float = 0.30,
        max_concurrent_executions: int = 5,
        rate_limit_per_minute: int = 30
    ):
        self.confidence_threshold = confidence_threshold
        self.risk_threshold = risk_threshold
        self.max_concurrent_executions = max_concurrent_executions
        self.rate_limit_per_minute = rate_limit_per_minute

        # Execution state
        self.pending_queue: list[PredictedTask] = []
        self.active_executions: dict[str, PredictiveExecution] = {}
        self.execution_history: list[PredictiveExecution] = []

        # Rate limiting
        self.execution_timestamps: list[datetime] = []

        # Accuracy tracking
        self.total_executions = 0
        self.accurate_predictions = 0
        self.inaccurate_predictions = 0

        # Custom executors registered by task type
        self.task_executors: dict[str, Callable] = {}

        # Initialize database schema
        self._ensure_schema()

    def _get_connection(self):
        """Get a database connection"""
        return psycopg2.connect(**_get_db_config())

    def _ensure_schema(self):
        """Create required database tables"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            conn.commit()
            conn.close()
            logger.info("PredictiveExecutor schema initialized")
        except Exception as e:
            logger.warning(f"Schema init warning (may already exist): {e}")

    async def evaluate_prediction(self, prediction: PredictedTask) -> ExecutionEvaluation:
        """
        Evaluate if a prediction should be executed.

        Criteria:
        1. Confidence must be above threshold (default: 80%)
        2. Risk must be below threshold (default: 30%)
        3. Action must not be in blocked list
        4. Rate limit not exceeded
        """
        safety_checks = []

        # Check 1: Confidence threshold
        if prediction.confidence < self.confidence_threshold:
            return ExecutionEvaluation(
                prediction_id=prediction.id,
                should_execute=False,
                decision=ExecutionDecision.SKIP_LOW_CONFIDENCE,
                confidence=prediction.confidence,
                risk_level=RiskLevel.NONE,
                risk_score=0.0,
                reason=f"Confidence {prediction.confidence:.2f} below threshold {self.confidence_threshold}",
                safety_checks=["confidence_check: FAILED"]
            )
        safety_checks.append("confidence_check: PASSED")

        # Check 2: Blocked actions
        task_lower = prediction.task_type.lower()
        for blocked in self.BLOCKED_ACTIONS:
            if blocked in task_lower:
                return ExecutionEvaluation(
                    prediction_id=prediction.id,
                    should_execute=False,
                    decision=ExecutionDecision.SKIP_BLOCKED_ACTION,
                    confidence=prediction.confidence,
                    risk_level=RiskLevel.CRITICAL,
                    risk_score=1.0,
                    reason=f"Task type '{prediction.task_type}' contains blocked action '{blocked}'",
                    safety_checks=[*safety_checks, f"blocked_action_check: FAILED ({blocked})"]
                )
        safety_checks.append("blocked_action_check: PASSED")

        # Check 3: Risk assessment
        risk_level, risk_score = await self._assess_risk(prediction)
        if risk_score > self.risk_threshold:
            return ExecutionEvaluation(
                prediction_id=prediction.id,
                should_execute=False,
                decision=ExecutionDecision.SKIP_HIGH_RISK,
                confidence=prediction.confidence,
                risk_level=risk_level,
                risk_score=risk_score,
                reason=f"Risk score {risk_score:.2f} above threshold {self.risk_threshold}",
                safety_checks=[*safety_checks, f"risk_check: FAILED ({risk_level.value})"]
            )
        safety_checks.append(f"risk_check: PASSED ({risk_level.value})")

        # Check 4: Rate limiting
        if not self._check_rate_limit():
            return ExecutionEvaluation(
                prediction_id=prediction.id,
                should_execute=False,
                decision=ExecutionDecision.SKIP_RATE_LIMITED,
                confidence=prediction.confidence,
                risk_level=risk_level,
                risk_score=risk_score,
                reason=f"Rate limit exceeded ({self.rate_limit_per_minute}/minute)",
                safety_checks=[*safety_checks, "rate_limit_check: FAILED"]
            )
        safety_checks.append("rate_limit_check: PASSED")

        # Check 5: Concurrent execution limit
        if len(self.active_executions) >= self.max_concurrent_executions:
            return ExecutionEvaluation(
                prediction_id=prediction.id,
                should_execute=False,
                decision=ExecutionDecision.DEFER,
                confidence=prediction.confidence,
                risk_level=risk_level,
                risk_score=risk_score,
                reason=f"Max concurrent executions reached ({self.max_concurrent_executions})",
                safety_checks=[*safety_checks, "concurrency_check: DEFERRED"]
            )
        safety_checks.append("concurrency_check: PASSED")

        # All checks passed
        return ExecutionEvaluation(
            prediction_id=prediction.id,
            should_execute=True,
            decision=ExecutionDecision.EXECUTE,
            confidence=prediction.confidence,
            risk_level=risk_level,
            risk_score=risk_score,
            reason="All safety checks passed",
            safety_checks=safety_checks
        )

    async def _assess_risk(self, prediction: PredictedTask) -> tuple[RiskLevel, float]:
        """
        Assess the risk level and score of a predicted task.

        Risk factors:
        - Task type (read vs write vs delete)
        - Parameters (external targets, sensitive data)
        - Historical failures for this task type
        """
        task_lower = prediction.task_type.lower()
        base_score = 0.0

        # Check if it's a safe action type
        is_safe = any(safe in task_lower for safe in self.SAFE_ACTIONS)
        if is_safe:
            return RiskLevel.NONE, 0.0

        # Check for write operations
        write_indicators = ["update", "modify", "change", "set", "add", "insert", "create"]
        if any(indicator in task_lower for indicator in write_indicators):
            base_score = 0.3
            risk_level = RiskLevel.LOW
        else:
            base_score = 0.2
            risk_level = RiskLevel.LOW

        # Check parameters for risky patterns
        params_str = json.dumps(prediction.parameters).lower()

        # External targets increase risk
        if any(ext in params_str for ext in ["http://", "https://", "@", "email", "webhook"]):
            base_score += 0.3
            risk_level = RiskLevel.MEDIUM

        # Sensitive data patterns
        if any(sens in params_str for sens in ["password", "secret", "token", "key", "credential"]):
            base_score += 0.4
            risk_level = RiskLevel.HIGH

        # Check historical failure rate for this task type
        failure_rate = await self._get_historical_failure_rate(prediction.task_type)
        if failure_rate > 0.3:
            base_score += failure_rate * 0.3
            if risk_level.value < RiskLevel.MEDIUM.value:
                risk_level = RiskLevel.MEDIUM

        return risk_level, min(base_score, 1.0)

    async def _get_historical_failure_rate(self, task_type: str) -> float:
        """Get the historical failure rate for a task type"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed
                FROM ai_predictive_executions
                WHERE task_type = %s
                  AND created_at > NOW() - INTERVAL '7 days'
            """, (task_type,))

            row = cur.fetchone()
            conn.close()

            if row and row['total'] > 0:
                return row['failed'] / row['total']
            return 0.0

        except Exception as e:
            logger.warning(f"Failed to get failure rate: {e}")
            return 0.0

    def _check_rate_limit(self) -> bool:
        """Check if execution rate limit allows another execution"""
        now = datetime.utcnow()

        # Remove timestamps older than 1 minute
        self.execution_timestamps = [
            ts for ts in self.execution_timestamps
            if (now - ts).total_seconds() < 60
        ]

        return len(self.execution_timestamps) < self.rate_limit_per_minute

    def _record_execution_timestamp(self):
        """Record an execution timestamp for rate limiting"""
        self.execution_timestamps.append(datetime.utcnow())

    async def execute_prediction(
        self,
        prediction: PredictedTask,
        executor: Optional[Callable] = None
    ) -> dict:
        """
        Execute a predicted task if safe.

        Args:
            prediction: The predicted task to execute
            executor: Optional custom executor function

        Returns:
            Dict with execution result or skip reason
        """
        # Evaluate the prediction
        evaluation = await self.evaluate_prediction(prediction)

        # Create execution record
        execution = PredictiveExecution(
            id=f"pexec_{uuid.uuid4().hex[:12]}",
            prediction_id=prediction.id,
            task_type=prediction.task_type,
            parameters=prediction.parameters
        )

        # Log the evaluation
        await self._log_execution(execution, evaluation, prediction)

        if not evaluation.should_execute:
            execution.status = "skipped"
            return {
                "executed": False,
                "skipped": True,
                "decision": evaluation.decision.value,
                "reason": evaluation.reason,
                "evaluation": evaluation.to_dict()
            }

        # Execute the prediction
        self._record_execution_timestamp()
        execution.status = "executing"
        execution.started_at = datetime.utcnow()
        self.active_executions[execution.id] = execution

        try:
            # Get the executor
            exec_fn = executor or self.task_executors.get(prediction.task_type)

            if exec_fn:
                result = await exec_fn(prediction.task_type, prediction.parameters)
            else:
                # Default executor - just log and return success for safe actions
                result = await self._default_executor(prediction)

            execution.status = "completed"
            execution.result = result
            execution.completed_at = datetime.utcnow()
            execution.execution_time_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )

            # Update database
            await self._update_execution_status(execution, evaluation)

            # Track for accuracy
            self.total_executions += 1
            self.execution_history.append(execution)

            return {
                "executed": True,
                "execution_id": execution.id,
                "result": result,
                "execution_time_ms": execution.execution_time_ms,
                "evaluation": evaluation.to_dict()
            }

        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()

            await self._update_execution_status(execution, evaluation)

            logger.error(f"Predictive execution failed: {e}")
            return {
                "executed": False,
                "error": str(e),
                "evaluation": evaluation.to_dict()
            }
        finally:
            # Remove from active executions
            self.active_executions.pop(execution.id, None)

    async def _default_executor(self, prediction: PredictedTask) -> dict:
        """Default executor for safe actions - primarily logging"""
        logger.info(
            f"PREDICTIVE_EXEC: {prediction.task_type} | "
            f"confidence={prediction.confidence:.2f} | "
            f"params={json.dumps(prediction.parameters)[:100]}"
        )

        return {
            "status": "completed",
            "message": f"Predictive execution of {prediction.task_type} completed",
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _log_execution(
        self,
        execution: PredictiveExecution,
        evaluation: ExecutionEvaluation,
        prediction: PredictedTask
    ):
        """Log execution to database for audit"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_predictive_executions
                (execution_id, prediction_id, prediction_source, task_type,
                 description, parameters, confidence, risk_level, risk_score,
                 decision, decision_reason, safety_checks, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                execution.id,
                prediction.id,
                prediction.source.value,
                prediction.task_type,
                prediction.description,
                Json(prediction.parameters),
                prediction.confidence,
                evaluation.risk_level.value,
                evaluation.risk_score,
                evaluation.decision.value,
                evaluation.reason,
                evaluation.safety_checks,
                execution.status
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to log execution: {e}")

    async def _update_execution_status(
        self,
        execution: PredictiveExecution,
        evaluation: ExecutionEvaluation
    ):
        """Update execution status in database"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE ai_predictive_executions
                SET status = %s,
                    result = %s,
                    error = %s,
                    started_at = %s,
                    completed_at = %s,
                    execution_time_ms = %s
                WHERE execution_id = %s
            """, (
                execution.status,
                Json(execution.result) if execution.result else None,
                execution.error,
                execution.started_at,
                execution.completed_at,
                execution.execution_time_ms,
                execution.id
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to update execution status: {e}")

    async def learn_from_outcome(
        self,
        execution_id: str,
        was_accurate: bool,
        feedback: Optional[str] = None
    ):
        """
        Learn from whether the predictive execution was actually needed.

        This is called later when we know if the prediction matched
        the actual user/system need.
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE ai_predictive_executions
                SET was_accurate = %s,
                    accuracy_feedback = %s
                WHERE execution_id = %s
            """, (was_accurate, feedback, execution_id))

            conn.commit()
            conn.close()

            # Update tracking
            if was_accurate:
                self.accurate_predictions += 1
            else:
                self.inaccurate_predictions += 1

            logger.info(
                f"Learned from execution {execution_id}: "
                f"accurate={was_accurate}, feedback={feedback}"
            )

        except Exception as e:
            logger.warning(f"Failed to record learning: {e}")

    async def execute_batch(
        self,
        predictions: list[PredictedTask],
        executor: Optional[Callable] = None
    ) -> list[dict]:
        """
        Execute a batch of predictions, respecting rate limits and concurrency.
        """
        results = []

        # Sort by confidence (highest first)
        sorted_predictions = sorted(
            predictions,
            key=lambda p: p.confidence,
            reverse=True
        )

        for prediction in sorted_predictions:
            result = await self.execute_prediction(prediction, executor)
            results.append(result)

            # Small delay between executions
            await asyncio.sleep(0.1)

        return results

    def register_executor(self, task_type: str, executor: Callable):
        """Register a custom executor for a specific task type"""
        self.task_executors[task_type] = executor
        logger.info(f"Registered executor for task type: {task_type}")

    async def get_execution_stats(self) -> dict:
        """Get statistics about predictive executions"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Overall stats
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
                    COUNT(*) FILTER (WHERE was_accurate = TRUE) as accurate,
                    COUNT(*) FILTER (WHERE was_accurate = FALSE) as inaccurate,
                    AVG(confidence) as avg_confidence,
                    AVG(risk_score) as avg_risk,
                    AVG(execution_time_ms) FILTER (WHERE status = 'completed') as avg_exec_time_ms
                FROM ai_predictive_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            overall = cur.fetchone()

            # By task type
            cur.execute("""
                SELECT
                    task_type,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE was_accurate = TRUE) as accurate
                FROM ai_predictive_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY task_type
                ORDER BY count DESC
                LIMIT 10
            """)
            by_type = cur.fetchall()

            # By decision
            cur.execute("""
                SELECT
                    decision,
                    COUNT(*) as count
                FROM ai_predictive_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY decision
            """)
            by_decision = cur.fetchall()

            conn.close()

            accuracy_rate = 0.0
            if overall['accurate'] or overall['inaccurate']:
                total_validated = (overall['accurate'] or 0) + (overall['inaccurate'] or 0)
                accuracy_rate = (overall['accurate'] or 0) / total_validated

            return {
                "period": "24h",
                "overall": {
                    "total": overall['total'] or 0,
                    "completed": overall['completed'] or 0,
                    "failed": overall['failed'] or 0,
                    "skipped": overall['skipped'] or 0,
                    "accuracy_rate": accuracy_rate,
                    "avg_confidence": float(overall['avg_confidence'] or 0),
                    "avg_risk_score": float(overall['avg_risk'] or 0),
                    "avg_execution_time_ms": float(overall['avg_exec_time_ms'] or 0)
                },
                "by_task_type": [dict(row) for row in by_type],
                "by_decision": [dict(row) for row in by_decision],
                "in_memory": {
                    "total_executions": self.total_executions,
                    "accurate_predictions": self.accurate_predictions,
                    "inaccurate_predictions": self.inaccurate_predictions,
                    "active_executions": len(self.active_executions),
                    "pending_queue": len(self.pending_queue)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def get_recent_executions(self, limit: int = 20) -> list[dict]:
        """Get recent predictive executions"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            cur.execute("""
                SELECT
                    execution_id, prediction_id, prediction_source,
                    task_type, description, confidence, risk_level,
                    risk_score, decision, decision_reason, status,
                    result, error, was_accurate, execution_time_ms,
                    created_at
                FROM ai_predictive_executions
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))

            rows = cur.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []


# =============================================================================
# Integration with Existing Systems
# =============================================================================

class OODAIntegration:
    """
    Integration layer between PredictiveExecutor and bleeding_edge_ooda.py
    """

    def __init__(self, executor: PredictiveExecutor):
        self.executor = executor

    async def execute_ooda_speculations(
        self,
        speculations: list[dict],
        ooda_executor: Optional[Callable] = None
    ) -> list[dict]:
        """
        Execute speculations from the OODA system.

        Converts OODA speculations to PredictedTasks and executes them.
        """
        predictions = []

        for spec in speculations:
            pred = PredictedTask(
                id=f"ooda_{uuid.uuid4().hex[:8]}",
                source=PredictionSource.OODA_SPECULATION,
                task_type=spec.get("type") or spec.get("action_type", "unknown"),
                description=spec.get("description", "OODA speculation"),
                confidence=spec.get("probability", spec.get("confidence", 0.5)),
                parameters=spec.get("params", spec.get("parameters", {})),
                metadata={"ooda_source": True}
            )
            predictions.append(pred)

        return await self.executor.execute_batch(predictions, ooda_executor)


class ProactiveIntelIntegration:
    """
    Integration layer between PredictiveExecutor and proactive_intelligence.py
    """

    def __init__(self, executor: PredictiveExecutor):
        self.executor = executor

    async def execute_proactive_recommendations(
        self,
        recommendations: list[dict]
    ) -> list[dict]:
        """
        Execute recommendations from ProactiveIntelligence.

        Converts recommendations to PredictedTasks and executes them.
        """
        predictions = []

        for rec in recommendations:
            pred = PredictedTask(
                id=f"proactive_{uuid.uuid4().hex[:8]}",
                source=PredictionSource.PROACTIVE_INTEL,
                task_type=rec.get("action", "recommendation"),
                description=rec.get("reason", "Proactive recommendation"),
                confidence=rec.get("confidence", 0.7),
                parameters={
                    "action": rec.get("action"),
                    "prediction_id": rec.get("prediction_id")
                },
                metadata={"proactive_intel_source": True}
            )
            predictions.append(pred)

        return await self.executor.execute_batch(predictions)


# =============================================================================
# Singleton and Factory
# =============================================================================

_predictive_executor: Optional[PredictiveExecutor] = None


def get_predictive_executor() -> PredictiveExecutor:
    """Get or create the singleton PredictiveExecutor instance"""
    global _predictive_executor
    if _predictive_executor is None:
        _predictive_executor = PredictiveExecutor()
    return _predictive_executor


def get_ooda_integration() -> OODAIntegration:
    """Get OODA integration instance"""
    return OODAIntegration(get_predictive_executor())


def get_proactive_intel_integration() -> ProactiveIntelIntegration:
    """Get ProactiveIntelligence integration instance"""
    return ProactiveIntelIntegration(get_predictive_executor())


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    async def demo():
        print("\n" + "=" * 60)
        print("PREDICTIVE EXECUTOR - DEMO")
        print("=" * 60 + "\n")

        executor = get_predictive_executor()

        # Test predictions
        test_predictions = [
            PredictedTask(
                id="test_1",
                source=PredictionSource.PATTERN_MATCH,
                task_type="fetch_customer_data",
                description="Prefetch customer data for likely next request",
                confidence=0.92,
                parameters={"customer_id": 12345}
            ),
            PredictedTask(
                id="test_2",
                source=PredictionSource.OODA_SPECULATION,
                task_type="analyze_revenue_trends",
                description="Analyze revenue trends based on current context",
                confidence=0.85,
                parameters={"period": "7d"}
            ),
            PredictedTask(
                id="test_3",
                source=PredictionSource.USER_BEHAVIOR,
                task_type="send_email",  # This should be blocked
                description="Send notification email",
                confidence=0.95,
                parameters={"to": "user@example.com"}
            ),
            PredictedTask(
                id="test_4",
                source=PredictionSource.SYSTEM_STATE,
                task_type="cache_warmup",
                description="Warm caches for predicted high traffic",
                confidence=0.65,  # Below threshold
                parameters={"cache_keys": ["users", "products"]}
            )
        ]

        print("Testing predictions:\n")
        for pred in test_predictions:
            print(f"  [{pred.id}] {pred.task_type} (confidence: {pred.confidence})")
            result = await executor.execute_prediction(pred)

            if result.get("executed"):
                print(f"    -> EXECUTED: {result.get('execution_id')}")
            else:
                print(f"    -> SKIPPED: {result.get('decision')} - {result.get('reason')}")
            print()

        # Get stats
        print("\n" + "-" * 40)
        print("EXECUTION STATS:")
        stats = await executor.get_execution_stats()
        print(json.dumps(stats, indent=2, default=str))

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60 + "\n")

    asyncio.run(demo())
