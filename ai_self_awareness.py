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

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
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
    limitations: List[LimitationType]
    strengths_applied: List[str]
    weaknesses_identified: List[str]
    requires_human_review: bool
    human_help_reason: Optional[str]
    similar_past_tasks: int
    past_success_rate: Decimal  # 0-100
    risk_level: str  # low, medium, high, critical
    mitigation_strategies: List[str]
    timestamp: datetime


@dataclass
class ReasoningExplanation:
    """AI's explanation of its reasoning process"""
    task_id: str
    agent_id: str
    decision_made: str
    reasoning_steps: List[Dict[str, Any]]
    evidence_used: List[Dict[str, Any]]
    assumptions_made: List[str]
    alternatives_considered: List[Dict[str, Any]]
    why_chosen: str
    confidence_in_decision: Decimal
    potential_errors: List[str]
    verification_methods: List[str]
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
    warning_signs_missed: List[str]
    what_learned: str
    how_to_prevent: List[str]
    confidence_before: Decimal
    confidence_after: Decimal
    similar_mistakes_count: int
    applied_to_agents: List[str]  # Which agents learned from this
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
        """Initialize database connection"""
        self.db_pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=1,
            max_size=2,  # Reduced to prevent pool exhaustion
            statement_cache_size=0,
            max_inactive_connection_lifetime=60  # Recycle idle connections
        )
        await self._create_tables()
        logger.info("AI Self-Awareness System initialized")

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
        task_context: Dict[str, Any]
    ) -> SelfAssessment:
        """
        AI assesses its own confidence in completing a task

        This is revolutionary - the AI knows what it doesn't know!
        """
        # Get agent's historical performance on similar tasks
        past_performance = await self._get_past_performance(agent_id, task_description)

        # Analyze task complexity
        complexity = await self._analyze_task_complexity(task_description, task_context)

        # Check if task is within training domain
        in_training_domain = await self._check_training_domain(agent_id, task_description)

        # Identify potential limitations
        limitations = await self._identify_limitations(
            agent_id, task_description, task_context, complexity
        )

        # Calculate confidence score (0-100)
        confidence_score = await self._calculate_confidence(
            past_performance,
            complexity,
            in_training_domain,
            len(limitations)
        )

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
            LimitationType.REQUIRES_HUMAN_JUDGMENT in limitations
        )

        # Generate human help reason if needed
        human_help_reason = None
        if not can_complete_alone or requires_human_review:
            human_help_reason = await self._generate_help_reason(
                limitations, confidence_score, complexity
            )

        # Identify strengths and weaknesses
        strengths = await self._identify_strengths(agent_id, task_description)
        weaknesses = await self._identify_weaknesses(agent_id, task_description)

        # Assess risk level
        risk_level = await self._assess_risk_level(
            confidence_score, limitations, complexity
        )

        # Generate mitigation strategies
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
            limitations=[l for l in limitations],
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

        logger.info(f"AI self-assessed confidence: {confidence_score}% for task {task_id}")

        return assessment

    async def explain_reasoning(
        self,
        task_id: str,
        agent_id: str,
        decision: str,
        reasoning_process: Dict[str, Any]
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

        # Calculate new confidence
        confidence_after = max(Decimal('0'), confidence_before * Decimal('0.8'))  # Reduce by 20%

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

    # Helper methods (simplified implementations)

    async def _get_past_performance(self, agent_id: str, task_description: str) -> Dict[str, Any]:
        """Get agent's historical performance on similar tasks"""
        # Simplified - in production, this would use semantic similarity
        return {
            'count': 10,
            'success_rate': 85.0,
            'avg_accuracy': 92.0,
            'avg_time': 30
        }

    async def _analyze_task_complexity(self, task_description: str, context: Dict[str, Any]) -> str:
        """Analyze complexity of the task"""
        word_count = len(task_description.split())
        context_size = len(str(context))

        if word_count > 100 or context_size > 1000:
            return "high"
        elif word_count > 50 or context_size > 500:
            return "medium"
        else:
            return "low"

    async def _check_training_domain(self, agent_id: str, task_description: str) -> bool:
        """Check if task is within agent's training domain"""
        # Simplified - in production, this would use embeddings
        return True

    async def _identify_limitations(
        self,
        agent_id: str,
        task_description: str,
        context: Dict[str, Any],
        complexity: str
    ) -> List[LimitationType]:
        """Identify AI's limitations for this task"""
        limitations = []

        if complexity == "high":
            limitations.append(LimitationType.COMPLEXITY_TOO_HIGH)

        if len(context) < 3:
            limitations.append(LimitationType.CONTEXT_INSUFFICIENT)

        if "ethical" in task_description.lower():
            limitations.append(LimitationType.ETHICAL_CONCERN)

        return limitations

    async def _calculate_confidence(
        self,
        past_performance: Dict[str, Any],
        complexity: str,
        in_training_domain: bool,
        limitations_count: int
    ) -> float:
        """Calculate overall confidence score"""
        base_confidence = past_performance['success_rate']

        # Adjust for complexity
        if complexity == "high":
            base_confidence *= 0.8
        elif complexity == "medium":
            base_confidence *= 0.9

        # Adjust for training domain
        if not in_training_domain:
            base_confidence *= 0.7

        # Adjust for limitations
        base_confidence *= (1 - (limitations_count * 0.1))

        return max(0, min(100, base_confidence))

    async def _generate_help_reason(
        self,
        limitations: List[LimitationType],
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

    async def _identify_strengths(self, agent_id: str, task_description: str) -> List[str]:
        """Identify agent's strengths applicable to this task"""
        # Simplified
        return ["pattern recognition", "data analysis", "quick processing"]

    async def _identify_weaknesses(self, agent_id: str, task_description: str) -> List[str]:
        """Identify agent's weaknesses relevant to this task"""
        # Simplified
        return ["limited context window", "no real-world experience", "potential bias"]

    async def _assess_risk_level(
        self,
        confidence_score: float,
        limitations: List[LimitationType],
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
        limitations: List[LimitationType],
        weaknesses: List[str],
        risk_level: str
    ) -> List[str]:
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

    async def _break_down_reasoning(self, reasoning_process: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Break down reasoning into steps"""
        return [
            {"step": 1, "action": "Analyzed input data", "outcome": "Identified key patterns"},
            {"step": 2, "action": "Compared to historical data", "outcome": "Found similar cases"},
            {"step": 3, "action": "Applied decision rules", "outcome": "Selected best option"}
        ]

    async def _identify_evidence(self, reasoning_process: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify evidence used in reasoning"""
        return [
            {"type": "historical_data", "source": "database", "confidence": 0.9},
            {"type": "user_input", "source": "current_context", "confidence": 1.0}
        ]

    async def _list_assumptions(self, reasoning_process: Dict[str, Any]) -> List[str]:
        """List assumptions made during reasoning"""
        return [
            "Customer behavior remains consistent with past patterns",
            "Market conditions are similar to historical data",
            "Input data is accurate and complete"
        ]

    async def _find_alternatives(self, reasoning_process: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find alternatives that were considered"""
        return [
            {"option": "Alternative A", "pros": ["Faster"], "cons": ["Less accurate"], "score": 0.7},
            {"option": "Alternative B", "pros": ["More thorough"], "cons": ["Slower"], "score": 0.85}
        ]

    async def _explain_choice(
        self,
        decision: str,
        alternatives: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]]
    ) -> str:
        """Explain why this choice was made"""
        return f"Chose {decision} because it scored highest (0.92) based on accuracy, speed, and risk factors. Alternative options had lower scores."

    async def _calculate_decision_confidence(
        self,
        evidence: List[Dict[str, Any]],
        assumptions: List[str],
        alternatives: List[Dict[str, Any]]
    ) -> Decimal:
        """Calculate confidence in the decision"""
        base_confidence = 85.0

        # Adjust based on evidence quality
        avg_evidence_confidence = sum(e.get('confidence', 0.5) for e in evidence) / len(evidence)
        base_confidence *= avg_evidence_confidence

        # Adjust based on assumptions count
        base_confidence *= (1 - (len(assumptions) * 0.05))

        return Decimal(str(min(100, max(0, base_confidence))))

    async def _identify_potential_errors(
        self,
        decision: str,
        assumptions: List[str],
        evidence: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify potential errors in reasoning"""
        return [
            "Assumption about market conditions may not hold",
            "Historical data may not perfectly predict future",
            "Sample size of evidence could be larger"
        ]

    async def _suggest_verification(
        self,
        decision: str,
        potential_errors: List[str]
    ) -> List[str]:
        """Suggest ways to verify the decision"""
        return [
            "A/B test the decision with small sample first",
            "Monitor outcome metrics closely",
            "Compare to expert human judgment",
            "Review after 24 hours with fresh perspective"
        ]

    async def _store_explanation(self, explanation: ReasoningExplanation):
        """Store reasoning explanation in database"""
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

    async def _identify_warning_signs(self, task_id: str, agent_id: str, root_cause: str) -> List[str]:
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
        warning_signs: List[str]
    ) -> str:
        """Extract key learning from mistake"""
        return f"When {root_cause}, must {warning_signs[0] if warning_signs else 'proceed with caution'}"

    async def _generate_prevention_strategies(self, root_cause: str, what_learned: str) -> List[str]:
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
        how_to_prevent: List[str]
    ) -> List[str]:
        """Share learning with other agents"""
        # In production, this would broadcast to all relevant agents
        # For now, return list of agents that should learn from this
        return ["all_agents"]

    async def _store_learning(self, learning: LearningFromMistake):
        """Store learning from mistake in database"""
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
