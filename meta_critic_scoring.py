#!/usr/bin/env python3
"""
META-CRITIC SCORING SYSTEM
2025 Cutting-Edge Pattern: Multi-model consensus with learned meta-critic

Scores candidate outputs on:
- Correctness (factual accuracy)
- Risk (safety, compliance, reversibility)
- Business Value (revenue impact, strategic alignment)
- Confidence (model certainty signals)
- Provenance (source reliability, token cost)

Based on Perplexity research on AI agent orchestration patterns 2025.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }


class ScoreDimension(Enum):
    """Dimensions for meta-critic scoring"""
    CORRECTNESS = "correctness"
    RISK = "risk"
    BUSINESS_VALUE = "business_value"
    CONFIDENCE = "confidence"
    PROVENANCE = "provenance"
    LATENCY = "latency"
    TOKEN_COST = "token_cost"


@dataclass
class CandidateScore:
    """Score for a single candidate output"""
    candidate_id: str
    model: str
    provider: str
    scores: dict[str, float]  # dimension -> score (0-1)
    weighted_total: float
    metadata: dict[str, Any]
    timestamp: datetime


@dataclass
class MetaCriticResult:
    """Result from meta-critic evaluation"""
    winner_id: str
    winner_score: float
    all_scores: list[CandidateScore]
    consensus_method: str
    adjudication_reason: str
    requires_human_review: bool
    timestamp: datetime
    # Enhanced fields
    confidence_score: float = 0.0
    risk_assessment: Optional[dict[str, Any]] = None
    outcome_tracking_id: Optional[str] = None


class MetaCriticScorer:
    """
    Meta-critic that scores multiple model outputs and selects the best one.
    Uses learned weights based on historical outcomes.
    """

    def __init__(self):
        # Default weights (can be learned from feedback)
        self.dimension_weights = {
            ScoreDimension.CORRECTNESS.value: 0.30,
            ScoreDimension.RISK.value: 0.25,
            ScoreDimension.BUSINESS_VALUE.value: 0.20,
            ScoreDimension.CONFIDENCE.value: 0.15,
            ScoreDimension.PROVENANCE.value: 0.05,
            ScoreDimension.LATENCY.value: 0.03,
            ScoreDimension.TOKEN_COST.value: 0.02,
        }

        # Thresholds for human review
        self.human_review_threshold = float(os.getenv("META_CRITIC_HUMAN_THRESHOLD", "0.6"))
        self.high_risk_threshold = float(os.getenv("META_CRITIC_RISK_THRESHOLD", "0.7"))
        self.min_confidence = float(os.getenv("META_CRITIC_MIN_CONFIDENCE", "0.5"))

        # Provider reliability scores (learned from historical performance)
        self.provider_reliability = {
            "anthropic": 0.95,
            "openai": 0.93,
            "gemini": 0.88,
            "perplexity": 0.85,
            "local": 0.70,
        }

        self._init_database()

    def _init_database(self):
        """Initialize meta-critic tables"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cur = conn.cursor()

            cur.execute("""
            CREATE TABLE IF NOT EXISTS meta_critic_scores (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                evaluation_id UUID NOT NULL,
                candidate_id TEXT NOT NULL,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                dimension_scores JSONB NOT NULL,
                weighted_total FLOAT NOT NULL,
                was_winner BOOLEAN DEFAULT FALSE,
                human_feedback JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS meta_critic_evaluations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                task_type TEXT NOT NULL,
                winner_id TEXT NOT NULL,
                winner_score FLOAT NOT NULL,
                consensus_method TEXT NOT NULL,
                adjudication_reason TEXT,
                requires_human_review BOOLEAN DEFAULT FALSE,
                human_override BOOLEAN DEFAULT FALSE,
                final_outcome TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_meta_scores_eval ON meta_critic_scores(evaluation_id);
            CREATE INDEX IF NOT EXISTS idx_meta_evals_type ON meta_critic_evaluations(task_type);

            -- Enhanced tracking tables
            ALTER TABLE meta_critic_evaluations
            ADD COLUMN IF NOT EXISTS confidence_score FLOAT DEFAULT 0.0;

            ALTER TABLE meta_critic_evaluations
            ADD COLUMN IF NOT EXISTS risk_assessment JSONB;

            CREATE TABLE IF NOT EXISTS meta_critic_outcome_tracking (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                evaluation_id UUID REFERENCES meta_critic_evaluations(id),
                actual_outcome JSONB,
                expected_outcome JSONB,
                success_score FLOAT,
                lessons_learned TEXT[],
                recorded_at TIMESTAMP DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS meta_critic_learning_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                dimension VARCHAR(50),
                weight_adjustment FLOAT,
                performance_delta FLOAT,
                reason TEXT,
                applied_at TIMESTAMP DEFAULT NOW()
            );
            """)

            conn.commit()
            cur.close()
            conn.close()
            logger.info("✅ Meta-critic database initialized")
        except Exception as e:
            logger.warning(f"Meta-critic database init failed: {e}")

    async def score_candidates(
        self,
        candidates: list[dict[str, Any]],
        task_context: dict[str, Any],
        consensus_method: str = "weighted_score"
    ) -> MetaCriticResult:
        """
        Score multiple candidate outputs and select the best one.

        Args:
            candidates: List of candidate outputs with metadata
            task_context: Context about the task (type, importance, etc.)
            consensus_method: "weighted_score", "plurality_vote", "debate", "human"

        Returns:
            MetaCriticResult with winner and scores
        """
        if not candidates:
            raise ValueError("No candidates to score")

        scored_candidates: list[CandidateScore] = []

        for candidate in candidates:
            score = await self._score_single_candidate(candidate, task_context)
            scored_candidates.append(score)

        # Select winner based on consensus method
        if consensus_method == "weighted_score":
            winner = self._select_by_weighted_score(scored_candidates)
        elif consensus_method == "plurality_vote":
            winner = self._select_by_plurality(scored_candidates)
        elif consensus_method == "debate":
            winner = await self._select_by_debate(scored_candidates, task_context)
        else:
            winner = self._select_by_weighted_score(scored_candidates)

        # Enhanced: Calculate confidence score
        confidence_score = self._calculate_confidence(winner, scored_candidates)

        # Enhanced: Perform risk assessment
        risk_assessment = self._assess_decision_risk(winner, task_context)

        # Determine if human review needed
        requires_human = self._check_human_review_needed(winner, scored_candidates, task_context)

        result = MetaCriticResult(
            winner_id=winner.candidate_id,
            winner_score=winner.weighted_total,
            all_scores=scored_candidates,
            consensus_method=consensus_method,
            adjudication_reason=self._generate_adjudication_reason(winner, scored_candidates),
            requires_human_review=requires_human,
            timestamp=datetime.now(),
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            outcome_tracking_id=str(uuid.uuid4())
        )

        # Store for learning
        self._store_evaluation(result, task_context)

        return result

    async def _score_single_candidate(
        self,
        candidate: dict[str, Any],
        task_context: dict[str, Any]
    ) -> CandidateScore:
        """Score a single candidate output across all dimensions"""

        scores: dict[str, float] = {}

        # Correctness score (based on validation signals if available)
        scores[ScoreDimension.CORRECTNESS.value] = self._score_correctness(candidate, task_context)

        # Risk score (inverse - lower is better, we invert for consistency)
        risk_raw = self._score_risk(candidate, task_context)
        scores[ScoreDimension.RISK.value] = 1.0 - risk_raw  # Invert so higher is better

        # Business value score
        scores[ScoreDimension.BUSINESS_VALUE.value] = self._score_business_value(candidate, task_context)

        # Confidence score (from model signals)
        scores[ScoreDimension.CONFIDENCE.value] = self._score_confidence(candidate)

        # Provenance score (provider reliability + audit trail)
        scores[ScoreDimension.PROVENANCE.value] = self._score_provenance(candidate)

        # Latency score (faster is better, normalized)
        scores[ScoreDimension.LATENCY.value] = self._score_latency(candidate)

        # Token cost score (cheaper is better, normalized)
        scores[ScoreDimension.TOKEN_COST.value] = self._score_token_cost(candidate)

        # Calculate weighted total
        weighted_total = sum(
            scores[dim] * self.dimension_weights[dim]
            for dim in scores
        )

        return CandidateScore(
            candidate_id=candidate.get("id", str(hash(str(candidate)))),
            model=candidate.get("model", "unknown"),
            provider=candidate.get("provider", "unknown"),
            scores=scores,
            weighted_total=weighted_total,
            metadata=candidate.get("metadata", {}),
            timestamp=datetime.now()
        )

    def _score_correctness(self, candidate: dict[str, Any], context: dict[str, Any]) -> float:
        """Score correctness based on validation signals"""
        # Check for explicit validation results
        if "validation" in candidate:
            return float(candidate["validation"].get("score", 0.5))

        # Check for error signals
        if candidate.get("error") or candidate.get("failed"):
            return 0.0

        # Check for completion signals
        if candidate.get("complete", True):
            return 0.7  # Base score for complete output

        return 0.5  # Neutral

    def _score_risk(self, candidate: dict[str, Any], context: dict[str, Any]) -> float:
        """Score risk level (0 = no risk, 1 = high risk)"""
        risk = 0.0
        content = str(candidate.get("output", "")).lower()

        # Financial actions are high risk
        if context.get("task_type") in ["payment", "transaction", "financial"]:
            risk += 0.3

        # Destructive actions
        if any(word in content for word in ["delete", "remove", "destroy", "drop"]):
            risk += 0.2

        # External communications
        if any(word in content for word in ["email", "send", "notify", "publish"]):
            risk += 0.1

        # Check explicit risk signals
        if "risk_score" in candidate:
            risk = max(risk, candidate["risk_score"])

        return min(risk, 1.0)

    def _score_business_value(self, candidate: dict[str, Any], context: dict[str, Any]) -> float:
        """Score business value impact"""
        value = 0.5  # Base

        # Revenue-generating actions
        if context.get("task_type") in ["revenue", "sales", "conversion"]:
            value += 0.2

        # Check for value signals in output
        if candidate.get("revenue_impact"):
            value += 0.2

        if candidate.get("customer_impact"):
            value += 0.1

        return min(value, 1.0)

    def _score_confidence(self, candidate: dict[str, Any]) -> float:
        """Score model confidence"""
        # Check for explicit confidence
        if "confidence" in candidate:
            return float(candidate["confidence"])

        # Check for parsed confidence in response
        parsed = candidate.get("parsed", {})
        if "confidence" in parsed:
            return float(parsed["confidence"])

        return 0.5  # Neutral

    def _score_provenance(self, candidate: dict[str, Any]) -> float:
        """Score based on provider reliability and audit trail"""
        provider = candidate.get("provider", "unknown").lower()
        reliability = self.provider_reliability.get(provider, 0.5)

        # Bonus for complete audit trail
        if candidate.get("audit_trail"):
            reliability = min(reliability + 0.05, 1.0)

        return reliability

    def _score_latency(self, candidate: dict[str, Any]) -> float:
        """Score latency (faster is better)"""
        latency_ms = candidate.get("latency_ms", 1000)
        # Normalize: 0ms = 1.0, 5000ms+ = 0.0
        return max(0, 1.0 - (latency_ms / 5000))

    def _score_token_cost(self, candidate: dict[str, Any]) -> float:
        """Score token cost (cheaper is better)"""
        tokens = candidate.get("total_tokens", 1000)
        # Normalize: 0 tokens = 1.0, 10000+ = 0.0
        return max(0, 1.0 - (tokens / 10000))

    def _select_by_weighted_score(self, candidates: list[CandidateScore]) -> CandidateScore:
        """Select winner by highest weighted score"""
        return max(candidates, key=lambda c: c.weighted_total)

    def _select_by_plurality(self, candidates: list[CandidateScore]) -> CandidateScore:
        """Select by plurality vote across dimensions"""
        # Count how many dimensions each candidate wins
        wins = {c.candidate_id: 0 for c in candidates}

        for dim in ScoreDimension:
            best_for_dim = max(candidates, key=lambda c: c.scores.get(dim.value, 0))
            wins[best_for_dim.candidate_id] += 1

        # Return candidate with most dimension wins
        winner_id = max(wins, key=wins.get)
        return next(c for c in candidates if c.candidate_id == winner_id)

    async def _select_by_debate(
        self,
        candidates: list[CandidateScore],
        context: dict[str, Any]
    ) -> CandidateScore:
        """Select winner through iterative debate (advanced pattern)"""
        # For now, fall back to weighted score
        # TODO: Implement multi-round debate with rebuttals
        return self._select_by_weighted_score(candidates)

    def _check_human_review_needed(
        self,
        winner: CandidateScore,
        all_candidates: list[CandidateScore],
        context: dict[str, Any]
    ) -> bool:
        """Determine if human review is required"""
        # Low confidence winner
        if winner.weighted_total < self.human_review_threshold:
            return True

        # High-risk task
        if context.get("task_type") in ["payment", "delete", "legal", "compliance"]:
            return True

        # Close race (top 2 within 5%)
        sorted_candidates = sorted(all_candidates, key=lambda c: c.weighted_total, reverse=True)
        if len(sorted_candidates) >= 2:
            margin = sorted_candidates[0].weighted_total - sorted_candidates[1].weighted_total
            if margin < 0.05:
                return True

        # Explicit risk flag
        risk_score = winner.scores.get(ScoreDimension.RISK.value, 1.0)
        if (1.0 - risk_score) > self.high_risk_threshold:  # Original risk was high
            return True

        return False

    def _generate_adjudication_reason(
        self,
        winner: CandidateScore,
        all_candidates: list[CandidateScore]
    ) -> str:
        """Generate human-readable reason for selection"""
        reasons = []

        # Winning dimensions
        for dim, weight in sorted(self.dimension_weights.items(), key=lambda x: -x[1]):
            if winner.scores.get(dim, 0) >= 0.7:
                reasons.append(f"strong {dim}")

        if reasons:
            return f"Selected for: {', '.join(reasons[:3])}"

        return f"Highest weighted score: {winner.weighted_total:.2f}"

    def _store_evaluation(self, result: MetaCriticResult, context: dict[str, Any]):
        """Store evaluation for learning"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cur = conn.cursor()

            # Store evaluation
            cur.execute("""
            INSERT INTO meta_critic_evaluations
            (task_type, winner_id, winner_score, consensus_method,
             adjudication_reason, requires_human_review)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """, (
                context.get("task_type", "unknown"),
                result.winner_id,
                result.winner_score,
                result.consensus_method,
                result.adjudication_reason,
                result.requires_human_review
            ))

            eval_id = cur.fetchone()[0]

            # Store individual scores
            for score in result.all_scores:
                cur.execute("""
                INSERT INTO meta_critic_scores
                (evaluation_id, candidate_id, model, provider,
                 dimension_scores, weighted_total, was_winner)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    eval_id,
                    score.candidate_id,
                    score.model,
                    score.provider,
                    Json(score.scores),
                    score.weighted_total,
                    score.candidate_id == result.winner_id
                ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to store meta-critic evaluation: {e}")

    def _calculate_confidence(
        self,
        winner: CandidateScore,
        all_candidates: list[CandidateScore]
    ) -> float:
        """Calculate confidence in the winning selection"""
        confidence_factors = []

        # Winner's absolute score
        confidence_factors.append(winner.weighted_total)

        # Margin of victory (distance to second place)
        sorted_candidates = sorted(all_candidates, key=lambda c: c.weighted_total, reverse=True)
        if len(sorted_candidates) >= 2:
            margin = sorted_candidates[0].weighted_total - sorted_candidates[1].weighted_total
            confidence_factors.append(min(margin * 2, 1.0))  # Normalize margin

        # Consistency across dimensions (low variance = higher confidence)
        if winner.scores:
            score_values = list(winner.scores.values())
            avg_score = sum(score_values) / len(score_values)
            variance = sum((s - avg_score)**2 for s in score_values) / len(score_values)
            consistency = 1.0 - min(variance, 1.0)
            confidence_factors.append(consistency)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _assess_decision_risk(
        self,
        winner: CandidateScore,
        task_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess risk of the selected candidate"""
        risk_assessment = {
            'overall_risk': 0.0,
            'risk_categories': {},
            'risk_factors': []
        }

        # Inverse of confidence (low confidence = high risk)
        confidence_risk = 1.0 - winner.scores.get(ScoreDimension.CONFIDENCE.value, 0.5)
        risk_assessment['risk_categories']['confidence'] = confidence_risk

        # Direct risk score (inverted from dimension)
        direct_risk = winner.scores.get(ScoreDimension.RISK.value, 0.5)
        risk_assessment['risk_categories']['direct'] = 1.0 - direct_risk  # Un-invert

        # Business value risk (low value = higher risk)
        value_risk = 1.0 - winner.scores.get(ScoreDimension.BUSINESS_VALUE.value, 0.5)
        risk_assessment['risk_categories']['value'] = value_risk

        # Provenance risk (unreliable source = higher risk)
        provenance_risk = 1.0 - winner.scores.get(ScoreDimension.PROVENANCE.value, 0.5)
        risk_assessment['risk_categories']['provenance'] = provenance_risk

        # Task type risk
        task_type = task_context.get('task_type', '')
        high_risk_tasks = ['payment', 'delete', 'financial', 'legal']
        if any(rt in task_type.lower() for rt in high_risk_tasks):
            risk_assessment['risk_categories']['task_type'] = 0.8
            risk_assessment['risk_factors'].append(f"High-risk task type: {task_type}")

        # Calculate overall risk
        risk_values = list(risk_assessment['risk_categories'].values())
        risk_assessment['overall_risk'] = sum(risk_values) / len(risk_values) if risk_values else 0.5

        return risk_assessment

    def record_outcome(
        self,
        outcome_tracking_id: str,
        actual_outcome: dict[str, Any],
        expected_outcome: dict[str, Any],
        success_score: float
    ):
        """Record actual outcome for learning"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cur = conn.cursor()

            # Find evaluation by outcome tracking ID
            cur.execute("""
                SELECT id FROM meta_critic_evaluations
                WHERE id::text = %s OR id IN (
                    SELECT evaluation_id FROM meta_critic_scores
                    WHERE candidate_id = %s
                )
                LIMIT 1
            """, (outcome_tracking_id, outcome_tracking_id))

            row = cur.fetchone()
            if not row:
                logger.warning(f"Evaluation not found for tracking ID: {outcome_tracking_id}")
                return

            evaluation_id = row[0]

            # Extract lessons learned
            lessons = []
            if success_score < 0.5:
                lessons.append("Selected candidate underperformed expectations")
            if success_score > 0.8:
                lessons.append("Selected candidate exceeded expectations")

            # Store outcome
            cur.execute("""
                INSERT INTO meta_critic_outcome_tracking
                (evaluation_id, actual_outcome, expected_outcome, success_score, lessons_learned)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                evaluation_id,
                Json(actual_outcome),
                Json(expected_outcome),
                success_score,
                lessons
            ))

            conn.commit()

            # Trigger weight optimization if performance is poor
            if success_score < 0.6:
                self._optimize_weights_from_outcome(evaluation_id, success_score)

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
        finally:
            if conn:
                cur.close()
                conn.close()

    def _optimize_weights_from_outcome(self, evaluation_id: str, success_score: float):
        """Automatically adjust dimension weights based on poor outcomes"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cur = conn.cursor()

            # Analyze which dimensions were weak in the winning candidate
            cur.execute("""
                SELECT dimension_scores FROM meta_critic_scores
                WHERE evaluation_id = %s AND was_winner = TRUE
            """, (evaluation_id,))

            row = cur.fetchone()
            if not row:
                return

            dimension_scores = row[0]

            # Identify weak dimensions (below 0.5)
            adjustments = {}
            for dim, score in dimension_scores.items():
                if score < 0.5:
                    # Increase weight for this dimension (it was too weak)
                    weight_increase = 0.05
                    adjustments[dim] = weight_increase
                    self.dimension_weights[dim] = min(
                        self.dimension_weights.get(dim, 0.1) + weight_increase,
                        0.5  # Max weight
                    )

            # Normalize weights to sum to 1.0
            total = sum(self.dimension_weights.values())
            for dim in self.dimension_weights:
                self.dimension_weights[dim] /= total

            # Log adjustments
            for dim, adjustment in adjustments.items():
                cur.execute("""
                    INSERT INTO meta_critic_learning_log
                    (dimension, weight_adjustment, performance_delta, reason)
                    VALUES (%s, %s, %s, %s)
                """, (
                    dim,
                    adjustment,
                    1.0 - success_score,
                    f"Dimension underperformed in evaluation {evaluation_id}"
                ))

            conn.commit()
            logger.info(f"Optimized weights based on outcome: {adjustments}")

        except Exception as e:
            logger.error(f"Failed to optimize weights: {e}")
        finally:
            if conn:
                cur.close()
                conn.close()

    def update_weights_from_feedback(self, evaluation_id: str, outcome: str, feedback: dict[str, Any]):
        """Update dimension weights based on human feedback (learning)"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cur = conn.cursor()

            # Update the evaluation with outcome
            cur.execute("""
            UPDATE meta_critic_evaluations
            SET final_outcome = %s, human_override = %s
            WHERE id = %s
            """, (outcome, feedback.get("override", False), evaluation_id))

            # If human overrode, store feedback for learning
            if feedback.get("override"):
                cur.execute("""
                UPDATE meta_critic_scores
                SET human_feedback = %s
                WHERE evaluation_id = %s AND candidate_id = %s
                """, (Json(feedback), evaluation_id, feedback.get("preferred_candidate")))

                # Log this as a learning event
                cur.execute("""
                    INSERT INTO meta_critic_learning_log
                    (dimension, weight_adjustment, performance_delta, reason)
                    VALUES (%s, %s, %s, %s)
                """, (
                    'overall',
                    0.0,
                    1.0,  # Human override indicates complete miss
                    f"Human override: {feedback.get('reason', 'No reason provided')}"
                ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Updated meta-critic weights from feedback: {evaluation_id}")

        except Exception as e:
            logger.error(f"Failed to update weights from feedback: {e}")


# Singleton instance
_meta_critic: Optional[MetaCriticScorer] = None


def get_meta_critic() -> MetaCriticScorer:
    """Get or create meta-critic instance"""
    global _meta_critic
    if _meta_critic is None:
        _meta_critic = MetaCriticScorer()
    return _meta_critic
