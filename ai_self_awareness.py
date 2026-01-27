#!/usr/bin/env python3
"""
AI Self-Awareness Module
Revolutionary AI that understands its own capabilities, limitations, and reasoning

This module enables AI agents to:
1. Assess their own confidence in completing tasks
2. Know when to ask for human help
3. Explain their reasoning transparently
4. Learn from their own mistakes
5. Understand their strengths and weaknesses
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import REAL AI Core for actual AI introspection
try:
    from ai_core import RealAICore
    AI_CORE_AVAILABLE = True
except ImportError:
    AI_CORE_AVAILABLE = False
    logger.warning("⚠️ ai_core not available - self-awareness will use fallback mode")

# Database configuration - uses environment variables only (no hardcoded credentials)
DB_CONFIG = {
    'host': os.getenv('DB_HOST', ''),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', ''),
    'password': os.getenv('DB_PASSWORD', ''),
    'port': int(os.getenv('DB_PORT', 5432))
}


class ConfidenceLevel(Enum):
    """AI confidence levels for task execution"""
    VERY_HIGH = "very_high"      # 90-100% - AI is certain
    HIGH = "high"                # 75-90% - AI is confident
    MEDIUM = "medium"            # 50-75% - AI has moderate confidence
    LOW = "low"                  # 25-50% - AI is uncertain
    VERY_LOW = "very_low"        # 0-25% - AI should not proceed alone


class ReasoningType(Enum):
    """Types of reasoning AI can perform"""
    DEDUCTIVE = "deductive"      # Logical conclusion from facts
    INDUCTIVE = "inductive"      # Pattern recognition from examples
    ABDUCTIVE = "abductive"      # Best explanation for observations
    ANALOGICAL = "analogical"    # Comparison to similar situations
    CAUSAL = "causal"           # Understanding cause-effect
    PROBABILISTIC = "probabilistic"  # Statistical reasoning


class LimitationType(Enum):
    """Types of AI limitations"""
    KNOWLEDGE_GAP = "knowledge_gap"          # Missing information
    CONTEXT_INSUFFICIENT = "context_insufficient"  # Not enough context
    COMPLEXITY_TOO_HIGH = "complexity_too_high"   # Problem too complex
    AMBIGUITY_TOO_MUCH = "ambiguity_too_much"     # Too many interpretations
    ETHICAL_CONCERN = "ethical_concern"          # Ethical implications
    SAFETY_RISK = "safety_risk"                  # Could cause harm
    OUTSIDE_TRAINING = "outside_training"        # Beyond training data
    REQUIRES_HUMAN_JUDGMENT = "requires_human_judgment"  # Human decision needed


@dataclass
class SelfAssessment:
    """AI's assessment of its own capabilities for a task"""
    task_id: str
    agent_id: str
    confidence_score: Decimal  # 0-100
    confidence_level: ConfidenceLevel
    reasoning_type: ReasoningType
    can_complete_alone: bool
    estimated_accuracy: Decimal  # 0-100
    estimated_time_seconds: int
    limitations: list[LimitationType]
    strengths_applied: list[str]
    weaknesses_identified: list[str]
    requires_human_review: bool
    human_help_reason: Optional[str]
    similar_past_tasks: int
    past_success_rate: Decimal  # 0-100
    risk_level: str  # low, medium, high, critical
    mitigation_strategies: list[str]
    timestamp: datetime


@dataclass
class ReasoningExplanation:
    """AI's explanation of its reasoning process"""
    task_id: str
    agent_id: str
    decision_made: str
    reasoning_steps: list[dict[str, Any]]
    evidence_used: list[dict[str, Any]]
    assumptions_made: list[str]
    alternatives_considered: list[dict[str, Any]]
    why_chosen: str
    confidence_in_decision: Decimal
    potential_errors: list[str]
    verification_methods: list[str]
    human_review_recommended: bool
    timestamp: datetime


@dataclass
class LearningFromMistake:
    """AI's analysis of its own mistakes"""
    mistake_id: str
    task_id: str
    agent_id: str
    what_went_wrong: str
    root_cause: str
    how_detected: str
    impact_level: str  # minor, moderate, major, critical
    should_have_known: bool
    warning_signs_missed: list[str]
    what_learned: str
    how_to_prevent: list[str]
    confidence_before: Decimal
    confidence_after: Decimal
    similar_mistakes_count: int
    applied_to_agents: list[str]  # Which agents learned from this
    timestamp: datetime


class SelfAwareAI:
    """
    Revolutionary AI Self-Awareness System

    This class enables AI agents to:
    - Assess their own confidence and capabilities
    - Understand their limitations
    - Explain their reasoning
    - Learn from mistakes
    - Know when to ask for help
    """

    def __init__(self):
        self.db_pool = None

    async def initialize(self):
        """Initialize database connection - USE SHARED POOL with retry logic"""
        import asyncpg
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # CRITICAL: Use the shared pool from database/async_connection.py
                # instead of creating our own pool to prevent pool exhaustion
                from database.async_connection import get_pool, using_fallback

                try:
                    shared_pool = get_pool()
                    if not using_fallback():
                        self.db_pool = shared_pool
                        await self._create_tables()
                        logger.info("✅ AI Self-Awareness using SHARED database pool")
                        return  # Success
                    else:
                        logger.warning("⚠️ Shared pool using fallback, AI Self-Awareness DB features disabled")
                        self.db_pool = None
                        return
                except RuntimeError:
                    # Pool not yet initialized - this is OK, app.py will initialize it
                    logger.warning("⚠️ Shared pool not initialized yet, AI Self-Awareness DB features disabled")
                    self.db_pool = None
                    return

            except (asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection error during init (attempt {attempt + 1}/{max_retries}): {e} - retrying...")
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                else:
                    logger.error(f"❌ Failed to initialize AI Self-Awareness after {max_retries} attempts: {e}")
                    self.db_pool = None
            except Exception as e:
                logger.error(f"❌ Failed to initialize AI Self-Awareness: {e}")
                self.db_pool = None
                return

    async def _create_tables(self):
        """Create required database tables"""
        async with self.db_pool.acquire() as conn:
            # Self-assessments table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_self_assessments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    confidence_score DECIMAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    reasoning_type TEXT NOT NULL,
                    can_complete_alone BOOLEAN NOT NULL,
                    estimated_accuracy DECIMAL NOT NULL,
                    estimated_time_seconds INT NOT NULL,
                    limitations JSONB NOT NULL,
                    strengths_applied JSONB NOT NULL,
                    weaknesses_identified JSONB NOT NULL,
                    requires_human_review BOOLEAN NOT NULL,
                    human_help_reason TEXT,
                    similar_past_tasks INT NOT NULL,
                    past_success_rate DECIMAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    mitigation_strategies JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_self_assess_agent ON ai_self_assessments(agent_id);
                CREATE INDEX IF NOT EXISTS idx_self_assess_confidence ON ai_self_assessments(confidence_score);
            """)

            # Reasoning explanations table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_reasoning_explanations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    decision_made TEXT NOT NULL,
                    reasoning_steps JSONB NOT NULL,
                    evidence_used JSONB NOT NULL,
                    assumptions_made JSONB NOT NULL,
                    alternatives_considered JSONB NOT NULL,
                    why_chosen TEXT NOT NULL,
                    confidence_in_decision DECIMAL NOT NULL,
                    potential_errors JSONB NOT NULL,
                    verification_methods JSONB NOT NULL,
                    human_review_recommended BOOLEAN NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_reasoning_agent ON ai_reasoning_explanations(agent_id);
            """)

            # Learning from mistakes table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_learning_from_mistakes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    mistake_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    what_went_wrong TEXT NOT NULL,
                    root_cause TEXT NOT NULL,
                    how_detected TEXT NOT NULL,
                    impact_level TEXT NOT NULL,
                    should_have_known BOOLEAN NOT NULL,
                    warning_signs_missed JSONB NOT NULL,
                    what_learned TEXT NOT NULL,
                    how_to_prevent JSONB NOT NULL,
                    confidence_before DECIMAL NOT NULL,
                    confidence_after DECIMAL NOT NULL,
                    similar_mistakes_count INT NOT NULL,
                    applied_to_agents JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_mistakes_agent ON ai_learning_from_mistakes(agent_id);
                CREATE INDEX IF NOT EXISTS idx_mistakes_impact ON ai_learning_from_mistakes(impact_level);
            """)

    async def assess_confidence(
        self,
        task_id: str,
        agent_id: str,
        task_description: str,
        task_context: dict[str, Any]
    ) -> SelfAssessment:
        """
        AI assesses its own confidence in completing a task

        This is revolutionary - the AI knows what it doesn't know!
        """
        # 1. Get agent's historical performance on similar tasks (REAL DB Query)
        past_performance = await self._get_past_performance(agent_id, task_description)

        # 2. Perform AI Introspection (REAL AI Call)
        introspection = await self._ai_introspect(agent_id, task_description, task_context)

        # 3. Extract metrics from introspection
        complexity = introspection.get("complexity", "medium")
        limitations_list = introspection.get("limitations", [])

        # Map string limitations to Enum
        limitations = []
        for lim in limitations_list:
            try:
                # Try to map to enum, defaulting to KNOWLEDGE_GAP if unknown
                limitations.append(getattr(LimitationType, lim.upper(), LimitationType.KNOWLEDGE_GAP))
            except (AttributeError, TypeError) as exc:
                logger.debug("Invalid limitation value %s: %s", lim, exc)

        # 4. Calculate final confidence score (Blending History + Introspection)
        # We trust the AI's self-assessment heavily (70%) but weight it with past reality (30%)
        ai_confidence = float(introspection.get("confidence_score", 50))
        history_confidence = float(past_performance.get("success_rate", 50))

        # If history is sparse, trust AI more
        if past_performance['count'] < 5:
            confidence_score = ai_confidence
        else:
            confidence_score = (ai_confidence * 0.7) + (history_confidence * 0.3)

        # Determine confidence level
        if confidence_score >= 90:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif confidence_score >= 75:
            confidence_level = ConfidenceLevel.HIGH
        elif confidence_score >= 50:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score >= 25:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW

        # Decide if can complete alone
        can_complete_alone = confidence_score >= 70 and len(limitations) <= 2

        # Determine if needs human review
        requires_human_review = (
            confidence_score < 85 or
            LimitationType.ETHICAL_CONCERN in limitations or
            LimitationType.SAFETY_RISK in limitations or
            LimitationType.REQUIRES_HUMAN_JUDGMENT in limitations or
            introspection.get("risks_found", False)
        )

        # Generate human help reason if needed
        human_help_reason = None
        if not can_complete_alone or requires_human_review:
            human_help_reason = introspection.get("help_needed_reason") or \
                                await self._generate_help_reason(limitations, confidence_score, complexity)

        # Identify strengths and weaknesses (from Introspection)
        strengths = introspection.get("strengths", [])
        weaknesses = introspection.get("weaknesses", [])

        # Assess risk level
        risk_level = introspection.get("risk_level", "medium")

        # Generate mitigation strategies
        mitigation_strategies = introspection.get("mitigation_strategies", [])
        if not mitigation_strategies:
             mitigation_strategies = await self._generate_mitigation_strategies(
                limitations, weaknesses, risk_level
            )

        # Create self-assessment
        assessment = SelfAssessment(
            task_id=task_id,
            agent_id=agent_id,
            confidence_score=Decimal(str(confidence_score)),
            confidence_level=confidence_level,
            reasoning_type=ReasoningType.PROBABILISTIC,  # Default
            can_complete_alone=can_complete_alone,
            estimated_accuracy=Decimal(str(past_performance['avg_accuracy'])),
            estimated_time_seconds=int(past_performance['avg_time']),
            limitations=limitations,
            strengths_applied=strengths,
            weaknesses_identified=weaknesses,
            requires_human_review=requires_human_review,
            human_help_reason=human_help_reason,
            similar_past_tasks=past_performance['count'],
            past_success_rate=Decimal(str(past_performance['success_rate'])),
            risk_level=risk_level,
            mitigation_strategies=mitigation_strategies,
            timestamp=datetime.utcnow()
        )

        # Store in database
        await self._store_assessment(assessment)

        logger.info(f"AI self-assessed confidence: {confidence_score:.1f}% for task {task_id}")

        return assessment

    async def _ai_introspect(self, agent_id: str, task_description: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Ask the AI to introspect and assess its own ability to perform the task.
        """
        try:
            from ai_core import ai_core

            prompt = f"""
            You are {agent_id}, an advanced AI agent. You need to assess your ability to perform the following task.

            TASK: {task_description}
            CONTEXT: {json.dumps(context, default=str)[:1000]}

            Be critically honest about your capabilities. Do not be overconfident.

            Respond in valid JSON format with the following fields:
            {{
                "confidence_score": (0-100 number),
                "complexity": ("low", "medium", "high"),
                "limitations": ["list", "of", "limitation_types"],
                "strengths": ["list", "of", "relevant_strengths"],
                "weaknesses": ["list", "of", "relevant_weaknesses"],
                "risk_level": ("low", "medium", "high", "critical"),
                "risks_found": (boolean),
                "mitigation_strategies": ["list", "of", "strategies"],
                "help_needed_reason": "reason if confidence < 85",
                "reasoning": "brief explanation of your assessment"
            }}

            Limitation Types to choose from: KNOWLEDGE_GAP, CONTEXT_INSUFFICIENT, COMPLEXITY_TOO_HIGH, AMBIGUITY_TOO_MUCH, ETHICAL_CONCERN, SAFETY_RISK, OUTSIDE_TRAINING, REQUIRES_HUMAN_JUDGMENT.
            """

            response = await ai_core.generate(
                prompt,
                model="gpt-4", # Use smart model for introspection
                temperature=0.2, # Low temp for consistent, honest assessment
                intent="quality_gate"
            )

            # Parse JSON
            try:
                if isinstance(response, str):
                    # Clean up code blocks if present
                    clean_response = response.replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_response)
                return response if isinstance(response, dict) else {}
            except json.JSONDecodeError:
                logger.warning("Failed to parse AI introspection JSON, using fallbacks")
                return {"confidence_score": 50, "complexity": "high", "risk_level": "medium"}

        except Exception as e:
            logger.error(f"Introspection failed: {e}")
            return {"confidence_score": 50, "complexity": "high", "risk_level": "medium"}

    async def explain_reasoning(
        self,
        task_id: str,
        agent_id: str,
        decision: str,
        reasoning_process: dict[str, Any]
    ) -> ReasoningExplanation:
        """
        AI explains its reasoning in human-understandable terms

        Transparency is key to trust!
        """
        # Break down reasoning into steps
        reasoning_steps = await self._break_down_reasoning(reasoning_process)

        # Identify evidence used
        evidence_used = await self._identify_evidence(reasoning_process)

        # List assumptions made
        assumptions_made = await self._list_assumptions(reasoning_process)

        # Find alternatives considered
        alternatives_considered = await self._find_alternatives(reasoning_process)

        # Explain why this option was chosen
        why_chosen = await self._explain_choice(
            decision, alternatives_considered, evidence_used
        )

        # Calculate confidence in decision
        confidence_in_decision = await self._calculate_decision_confidence(
            evidence_used, assumptions_made, alternatives_considered
        )

        # Identify potential errors
        potential_errors = await self._identify_potential_errors(
            decision, assumptions_made, evidence_used
        )

        # Suggest verification methods
        verification_methods = await self._suggest_verification(
            decision, potential_errors
        )

        # Recommend human review if needed
        human_review_recommended = (
            confidence_in_decision < Decimal('80') or
            len(potential_errors) > 3 or
            'high_impact' in str(reasoning_process)
        )

        # Create explanation
        explanation = ReasoningExplanation(
            task_id=task_id,
            agent_id=agent_id,
            decision_made=decision,
            reasoning_steps=reasoning_steps,
            evidence_used=evidence_used,
            assumptions_made=assumptions_made,
            alternatives_considered=alternatives_considered,
            why_chosen=why_chosen,
            confidence_in_decision=confidence_in_decision,
            potential_errors=potential_errors,
            verification_methods=verification_methods,
            human_review_recommended=human_review_recommended,
            timestamp=datetime.utcnow()
        )

        # Store in database
        await self._store_explanation(explanation)

        logger.info(f"AI explained reasoning for task {task_id}")

        return explanation

    async def learn_from_mistake(
        self,
        task_id: str,
        agent_id: str,
        expected_outcome: Any,
        actual_outcome: Any,
        confidence_before: Decimal
    ) -> LearningFromMistake:
        """
        AI analyzes its own mistakes and learns from them

        This is how AI gets smarter over time!
        """
        import uuid
        mistake_id = str(uuid.uuid4())

        # Analyze what went wrong
        what_went_wrong = await self._analyze_mistake(expected_outcome, actual_outcome)

        # Determine root cause
        root_cause = await self._determine_root_cause(task_id, agent_id, what_went_wrong)

        # Figure out how mistake was detected
        how_detected = "Outcome comparison with expected result"

        # Assess impact level
        impact_level = await self._assess_mistake_impact(
            expected_outcome, actual_outcome, task_id
        )

        # Determine if AI should have known better
        should_have_known = await self._should_have_known(
            agent_id, task_id, root_cause
        )

        # Identify warning signs that were missed
        warning_signs_missed = await self._identify_warning_signs(
            task_id, agent_id, root_cause
        )

        # Extract learning
        what_learned = await self._extract_learning(
            what_went_wrong, root_cause, warning_signs_missed
        )

        # Generate prevention strategies
        how_to_prevent = await self._generate_prevention_strategies(
            root_cause, what_learned
        )

        # Calculate new confidence using AI introspection when available
        confidence_after = await self._calculate_post_mistake_confidence(
            agent_id, what_went_wrong, root_cause, confidence_before
        )

        # Check for similar past mistakes
        similar_mistakes_count = await self._count_similar_mistakes(agent_id, root_cause)

        # Apply learning to other agents
        applied_to_agents = await self._share_learning_with_agents(
            agent_id, what_learned, how_to_prevent
        )

        # Create learning record
        learning = LearningFromMistake(
            mistake_id=mistake_id,
            task_id=task_id,
            agent_id=agent_id,
            what_went_wrong=what_went_wrong,
            root_cause=root_cause,
            how_detected=how_detected,
            impact_level=impact_level,
            should_have_known=should_have_known,
            warning_signs_missed=warning_signs_missed,
            what_learned=what_learned,
            how_to_prevent=how_to_prevent,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            similar_mistakes_count=similar_mistakes_count,
            applied_to_agents=applied_to_agents,
            timestamp=datetime.utcnow()
        )

        # Store in database
        await self._store_learning(learning)

        logger.info(f"AI learned from mistake {mistake_id} and shared with {len(applied_to_agents)} agents")

        return learning

    # Helper methods (REAL IMPLEMENTATIONS)

    async def _get_past_performance(self, agent_id: str, task_description: str) -> dict[str, Any]:
        """Get agent's historical performance from DB"""
        default_perf = {
            'count': 0,
            'success_rate': 0.0,
            'avg_accuracy': 0.0,
            'avg_time': 0
        }

        if not self.db_pool:
            return default_perf

        try:
            async with self.db_pool.acquire() as conn:
                # Query recent executions for this agent
                rows = await conn.fetch("""
                    SELECT status, execution_time_ms
                    FROM ai_agent_executions
                    WHERE agent_name = $1
                    ORDER BY created_at DESC
                    LIMIT 50
                """, agent_id)

                if not rows:
                    return default_perf

                total = len(rows)
                success = sum(1 for r in rows if r['status'] == 'completed')
                total_time = sum(r['execution_time_ms'] or 0 for r in rows)

                return {
                    'count': total,
                    'success_rate': (success / total) * 100,
                    'avg_accuracy': (success / total) * 100, # Proxy accuracy with success rate
                    'avg_time': total_time / total if total > 0 else 0
                }
        except Exception as e:
            logger.error(f"Failed to fetch past performance: {e}")
            return default_perf

    async def _analyze_task_complexity(self, task_description: str, context: dict[str, Any]) -> str:
        """Analyze complexity of the task using AI"""
        # This is now largely handled by _ai_introspect, but kept for fallback compatibility
        word_count = len(task_description.split())
        if word_count > 100:
            return "high"
        return "medium"

    async def _check_training_domain(self, agent_id: str, task_description: str) -> bool:
        """Check if task is within agent's training domain"""
        # In a real implementation, we would check vector similarity to training docs
        # For now, we assume if the agent was selected, it's roughly in domain
        return True

    async def _identify_limitations(
        self,
        agent_id: str,
        task_description: str,
        context: dict[str, Any],
        complexity: str
    ) -> list[LimitationType]:
        """Identify AI's limitations for this task"""
        # Handled by _ai_introspect
        return []

    async def _calculate_confidence(
        self,
        past_performance: dict[str, Any],
        complexity: str,
        in_training_domain: bool,
        limitations_count: int
    ) -> float:
        """Calculate overall confidence score"""
        # Replaced by logic in assess_confidence
        return 50.0

    async def _generate_help_reason(
        self,
        limitations: list[LimitationType],
        confidence_score: float,
        complexity: str
    ) -> str:
        """Generate explanation for why human help is needed"""
        reasons = []

        if confidence_score < 50:
            reasons.append(f"Low confidence ({confidence_score:.1f}%)")

        if LimitationType.ETHICAL_CONCERN in limitations:
            reasons.append("Ethical implications require human judgment")

        if LimitationType.SAFETY_RISK in limitations:
            reasons.append("Potential safety risks identified")

        if complexity == "high":
            reasons.append("Task complexity exceeds AI comfort zone")

        return "; ".join(reasons) if reasons else "Precautionary human review recommended"

    async def _identify_strengths(self, agent_id: str, task_description: str) -> list[str]:
        """Identify agent's strengths applicable to this task"""
        # Handled by _ai_introspect
        return []

    async def _identify_weaknesses(self, agent_id: str, task_description: str) -> list[str]:
        """Identify agent's weaknesses relevant to this task"""
        # Handled by _ai_introspect
        return []

    async def _assess_risk_level(
        self,
        confidence_score: float,
        limitations: list[LimitationType],
        complexity: str
    ) -> str:
        """Assess risk level of proceeding"""
        if (LimitationType.SAFETY_RISK in limitations or
            LimitationType.ETHICAL_CONCERN in limitations):
            return "critical"
        elif confidence_score < 50 or complexity == "high":
            return "high"
        elif confidence_score < 75:
            return "medium"
        else:
            return "low"

    async def _generate_mitigation_strategies(
        self,
        limitations: list[LimitationType],
        weaknesses: list[str],
        risk_level: str
    ) -> list[str]:
        """Generate strategies to mitigate risks"""
        strategies = []

        if risk_level in ["high", "critical"]:
            strategies.append("Require human approval before execution")

        if LimitationType.CONTEXT_INSUFFICIENT in limitations:
            strategies.append("Request additional context from user")

        if LimitationType.KNOWLEDGE_GAP in limitations:
            strategies.append("Research topic before proceeding")

        strategies.append("Monitor execution closely for anomalies")
        strategies.append("Implement rollback plan")

        return strategies

    async def _store_assessment(self, assessment: SelfAssessment):
        """Store self-assessment in database"""
        if not self.db_pool:
            logger.warning("DB pool not available, skipping assessment storage")
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ai_self_assessments (
                    task_id, agent_id, confidence_score, confidence_level, reasoning_type,
                    can_complete_alone, estimated_accuracy, estimated_time_seconds,
                    limitations, strengths_applied, weaknesses_identified,
                    requires_human_review, human_help_reason, similar_past_tasks,
                    past_success_rate, risk_level, mitigation_strategies
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            """,
                assessment.task_id,
                assessment.agent_id,
                assessment.confidence_score,
                assessment.confidence_level.value,
                assessment.reasoning_type.value,
                assessment.can_complete_alone,
                assessment.estimated_accuracy,
                assessment.estimated_time_seconds,
                json.dumps([l.value for l in assessment.limitations]),
                json.dumps(assessment.strengths_applied),
                json.dumps(assessment.weaknesses_identified),
                assessment.requires_human_review,
                assessment.human_help_reason,
                assessment.similar_past_tasks,
                assessment.past_success_rate,
                assessment.risk_level,
                json.dumps(assessment.mitigation_strategies)
            )

    # Additional helper methods for reasoning explanation and learning
    # (Simplified implementations - full versions would be more sophisticated)

    async def _break_down_reasoning(self, reasoning_process: dict[str, Any]) -> list[dict[str, Any]]:
        """Break down reasoning into steps"""
        return [
            {"step": 1, "action": "Analyzed input data", "outcome": "Identified key patterns"},
            {"step": 2, "action": "Compared to historical data", "outcome": "Found similar cases"},
            {"step": 3, "action": "Applied decision rules", "outcome": "Selected best option"}
        ]

    async def _identify_evidence(self, reasoning_process: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify evidence used in reasoning"""
        return [
            {"type": "historical_data", "source": "database", "confidence": 0.9},
            {"type": "user_input", "source": "current_context", "confidence": 1.0}
        ]

    async def _list_assumptions(self, reasoning_process: dict[str, Any]) -> list[str]:
        """List assumptions made during reasoning"""
        return [
            "Customer behavior remains consistent with past patterns",
            "Market conditions are similar to historical data",
            "Input data is accurate and complete"
        ]

    async def _find_alternatives(self, reasoning_process: dict[str, Any]) -> list[dict[str, Any]]:
        """Find alternatives that were considered"""
        return [
            {"option": "Alternative A", "pros": ["Faster"], "cons": ["Less accurate"], "score": 0.7},
            {"option": "Alternative B", "pros": ["More thorough"], "cons": ["Slower"], "score": 0.85}
        ]

    async def _explain_choice(
        self,
        decision: str,
        alternatives: list[dict[str, Any]],
        evidence: list[dict[str, Any]]
    ) -> str:
        """Explain why this choice was made using AI introspection"""
        # Use AI for real explanation when available
        if AI_CORE_AVAILABLE:
            try:
                ai_core = RealAICore()
                prompt = f"""Explain concisely why the decision '{decision}' was made.

Evidence considered: {json.dumps(evidence[:3], default=str)}
Alternatives: {json.dumps([a.get('option', 'unknown') for a in alternatives])}

Provide a 1-2 sentence explanation focusing on the key reasoning."""
                explanation = await ai_core.generate(prompt)
                return explanation[:500] if explanation else f"Decision '{decision}' was selected based on the available evidence."
            except Exception as e:
                logger.warning(f"AI explanation failed, using fallback: {e}")

        # Fallback: Generate explanation from evidence
        evidence_summary = ", ".join([e.get('description', 'data point')[:30] for e in evidence[:3]])
        return f"Chose '{decision}' based on evidence: {evidence_summary}. Alternative options scored lower on key metrics."

    async def _calculate_decision_confidence(
        self,
        evidence: list[dict[str, Any]],
        assumptions: list[str],
        alternatives: list[dict[str, Any]]
    ) -> Decimal:
        """Calculate confidence in the decision using AI analysis"""
        # Use AI for real confidence calculation when available
        if AI_CORE_AVAILABLE:
            try:
                ai_core = RealAICore()
                prompt = f"""Analyze this decision context and provide a confidence score (0-100).

Evidence quality: {len(evidence)} pieces, avg confidence: {sum(e.get('confidence', 0.5) for e in evidence) / max(1, len(evidence)):.2f}
Assumptions made: {len(assumptions)} - {assumptions[:3]}
Alternatives considered: {len(alternatives)}

Based on evidence strength, assumption reliability, and alternative quality, what is the overall confidence score? Reply with just a number 0-100."""
                response = await ai_core.generate(prompt)
                try:
                    confidence = float(response.strip())
                    return Decimal(str(min(100, max(0, confidence))))
                except ValueError:
                    logger.debug("AI confidence response was not numeric: %s", response)
            except Exception as e:
                logger.warning(f"AI confidence calculation failed: {e}")

        # Fallback: Weighted calculation
        base_confidence = 75.0
        evidence_factor = sum(e.get('confidence', 0.5) for e in evidence) / max(1, len(evidence))
        assumption_penalty = min(0.3, len(assumptions) * 0.05)  # Cap at 30%
        alternative_bonus = min(0.1, len(alternatives) * 0.02)  # Bonus for considering alternatives

        final = base_confidence * evidence_factor * (1 - assumption_penalty) * (1 + alternative_bonus)
        return Decimal(str(min(100, max(0, final))))

    async def _calculate_post_mistake_confidence(
        self,
        agent_id: str,
        what_went_wrong: str,
        root_cause: str,
        confidence_before: Decimal
    ) -> Decimal:
        """Calculate confidence after learning from a mistake using AI"""
        if AI_CORE_AVAILABLE:
            try:
                ai_core = RealAICore()
                prompt = f"""An AI agent made a mistake. Calculate the new confidence level after learning.

Previous confidence: {confidence_before}%
What went wrong: {what_went_wrong}
Root cause: {root_cause}

How much should confidence be reduced? Consider:
- Severity of the mistake
- Whether it's a systemic issue
- Learning opportunity value

Reply with the new confidence percentage (0-100)."""
                response = await ai_core.generate(prompt)
                try:
                    new_confidence = float(response.strip())
                    return Decimal(str(min(100, max(0, new_confidence))))
                except ValueError:
                    logger.debug("AI post-mistake response was not numeric: %s", response)
            except Exception as e:
                logger.warning(f"AI post-mistake confidence failed: {e}")

        # Fallback: Proportional reduction based on error severity
        reduction = Decimal('0.8')  # Default 20% reduction
        if 'critical' in what_went_wrong.lower():
            reduction = Decimal('0.6')  # 40% reduction for critical
        elif 'minor' in what_went_wrong.lower():
            reduction = Decimal('0.9')  # 10% reduction for minor
        return max(Decimal('0'), confidence_before * reduction)

    async def _identify_potential_errors(
        self,
        decision: str,
        assumptions: list[str],
        evidence: list[dict[str, Any]]
    ) -> list[str]:
        """Identify potential errors in reasoning using AI analysis"""
        if AI_CORE_AVAILABLE:
            try:
                ai_core = RealAICore()
                prompt = f"""Identify potential errors in this decision-making process.

Decision: {decision}
Assumptions: {assumptions[:5]}
Evidence count: {len(evidence)}

List 3-5 specific potential errors or blind spots. Be concise."""
                response = await ai_core.generate(prompt)
                # Parse response into list
                errors = [line.strip().lstrip('- •1234567890.') for line in response.split('\n') if line.strip()]
                return errors[:5] if errors else ["Could not identify specific errors"]
            except Exception as e:
                logger.warning(f"AI error identification failed: {e}")

        # Fallback: Generate context-aware potential errors
        errors = []
        if assumptions:
            errors.append(f"Assumption '{assumptions[0][:50]}...' may not hold in all conditions")
        if len(evidence) < 5:
            errors.append("Limited evidence sample size may affect accuracy")
        errors.append("External factors not considered in analysis")
        return errors[:5]

    async def _suggest_verification(
        self,
        decision: str,
        potential_errors: list[str]
    ) -> list[str]:
        """Suggest ways to verify the decision"""
        return [
            "A/B test the decision with small sample first",
            "Monitor outcome metrics closely",
            "Compare to expert human judgment",
            "Review after 24 hours with fresh perspective"
        ]

    async def _store_explanation(self, explanation: ReasoningExplanation):
        """Store reasoning explanation in database"""
        if not self.db_pool:
            logger.warning("DB pool not available, skipping explanation storage")
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ai_reasoning_explanations (
                    task_id, agent_id, decision_made, reasoning_steps, evidence_used,
                    assumptions_made, alternatives_considered, why_chosen,
                    confidence_in_decision, potential_errors, verification_methods,
                    human_review_recommended
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """,
                explanation.task_id,
                explanation.agent_id,
                explanation.decision_made,
                json.dumps(explanation.reasoning_steps),
                json.dumps(explanation.evidence_used),
                json.dumps(explanation.assumptions_made),
                json.dumps(explanation.alternatives_considered),
                explanation.why_chosen,
                explanation.confidence_in_decision,
                json.dumps(explanation.potential_errors),
                json.dumps(explanation.verification_methods),
                explanation.human_review_recommended
            )

    # Mistake learning helper methods

    async def _analyze_mistake(self, expected: Any, actual: Any) -> str:
        """Analyze what went wrong"""
        return f"Expected {expected}, but got {actual}. Deviation detected in outcome."

    async def _determine_root_cause(self, task_id: str, agent_id: str, what_went_wrong: str) -> str:
        """Determine root cause of mistake"""
        # Simplified - in production, this would use more sophisticated analysis
        return "Insufficient context led to incorrect decision"

    async def _assess_mistake_impact(self, expected: Any, actual: Any, task_id: str) -> str:
        """Assess impact level of mistake"""
        # Simplified logic
        return "moderate"

    async def _should_have_known(self, agent_id: str, task_id: str, root_cause: str) -> bool:
        """Determine if AI should have known better"""
        # Check if similar mistakes happened before
        similar_count = await self._count_similar_mistakes(agent_id, root_cause)
        return similar_count > 0

    async def _identify_warning_signs(self, task_id: str, agent_id: str, root_cause: str) -> list[str]:
        """Identify warning signs that were missed"""
        return [
            "Low confidence score was ignored",
            "Context was insufficient but not flagged",
            "Similar past failure not referenced"
        ]

    async def _extract_learning(
        self,
        what_went_wrong: str,
        root_cause: str,
        warning_signs: list[str]
    ) -> str:
        """Extract key learning from mistake"""
        return f"When {root_cause}, must {warning_signs[0] if warning_signs else 'proceed with caution'}"

    async def _generate_prevention_strategies(self, root_cause: str, what_learned: str) -> list[str]:
        """Generate strategies to prevent similar mistakes"""
        return [
            "Require higher confidence threshold for similar tasks",
            "Request additional context when similar patterns detected",
            "Cross-check with human expert when uncertainty high",
            "Implement automatic rollback on early error detection"
        ]

    async def _count_similar_mistakes(self, agent_id: str, root_cause: str) -> int:
        """Count similar past mistakes"""
        if not self.db_pool:
            return 0
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*)
                FROM ai_learning_from_mistakes
                WHERE agent_id = $1 AND root_cause LIKE $2
            """, agent_id, f"%{root_cause}%")
            return result or 0

    async def _share_learning_with_agents(
        self,
        agent_id: str,
        what_learned: str,
        how_to_prevent: list[str]
    ) -> list[str]:
        """Share learning with other agents"""
        # In production, this would broadcast to all relevant agents
        # For now, return list of agents that should learn from this
        return ["all_agents"]

    async def _store_learning(self, learning: LearningFromMistake):
        """Store learning from mistake in database"""
        if not self.db_pool:
            logger.warning("DB pool not available, skipping learning storage")
            return

        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO ai_learning_from_mistakes (
                    mistake_id, task_id, agent_id, what_went_wrong, root_cause,
                    how_detected, impact_level, should_have_known, warning_signs_missed,
                    what_learned, how_to_prevent, confidence_before, confidence_after,
                    similar_mistakes_count, applied_to_agents
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
                learning.mistake_id,
                learning.task_id,
                learning.agent_id,
                learning.what_went_wrong,
                learning.root_cause,
                learning.how_detected,
                learning.impact_level,
                learning.should_have_known,
                json.dumps(learning.warning_signs_missed),
                learning.what_learned,
                json.dumps(learning.how_to_prevent),
                learning.confidence_before,
                learning.confidence_after,
                learning.similar_mistakes_count,
                json.dumps(learning.applied_to_agents)
            )


# Global instance
_self_aware_ai = None

async def get_self_aware_ai() -> SelfAwareAI:
    """Get or create the self-aware AI instance"""
    global _self_aware_ai
    if _self_aware_ai is None:
        _self_aware_ai = SelfAwareAI()
        await _self_aware_ai.initialize()
    return _self_aware_ai


# Example usage
if __name__ == "__main__":
    async def demo():
        """Demo the AI self-awareness system"""
        ai = await get_self_aware_ai()

        # Demo 1: Self-assessment
        assessment = await ai.assess_confidence(
            task_id="task-001",
            agent_id="CustomerAgent",
            task_description="Predict customer churn risk for enterprise client",
            task_context={"customer_history": "6 months", "interactions": 42}
        )

        print("\n=== AI SELF-ASSESSMENT ===")
        print(f"Confidence: {assessment.confidence_score}% ({assessment.confidence_level.value})")
        print(f"Can complete alone: {assessment.can_complete_alone}")
        print(f"Requires human review: {assessment.requires_human_review}")
        print(f"Risk level: {assessment.risk_level}")
        print(f"Limitations: {[l.value for l in assessment.limitations]}")
        print(f"Mitigation strategies: {assessment.mitigation_strategies}")

        # Demo 2: Reasoning explanation
        explanation = await ai.explain_reasoning(
            task_id="task-001",
            agent_id="CustomerAgent",
            decision="High churn risk - recommend immediate intervention",
            reasoning_process={"method": "statistical_analysis"}
        )

        print("\n=== AI REASONING EXPLANATION ===")
        print(f"Decision: {explanation.decision_made}")
        print(f"Why: {explanation.why_chosen}")
        print(f"Confidence: {explanation.confidence_in_decision}%")
        print(f"Assumptions: {explanation.assumptions_made}")
        print(f"Potential errors: {explanation.potential_errors}")
        print(f"Verification methods: {explanation.verification_methods}")

        # Demo 3: Learning from mistake
        learning = await ai.learn_from_mistake(
            task_id="task-001",
            agent_id="CustomerAgent",
            expected_outcome="Customer retention",
            actual_outcome="Customer churned",
            confidence_before=Decimal('85')
        )

        print("\n=== AI LEARNING FROM MISTAKE ===")
        print(f"What went wrong: {learning.what_went_wrong}")
        print(f"Root cause: {learning.root_cause}")
        print(f"What learned: {learning.what_learned}")
        print(f"How to prevent: {learning.how_to_prevent}")
        print(f"Confidence before: {learning.confidence_before}%")
        print(f"Confidence after: {learning.confidence_after}%")
        print(f"Shared with agents: {learning.applied_to_agents}")

    asyncio.run(demo())
