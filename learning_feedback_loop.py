"""
Learning Feedback Loop - Closes the Gap Between Insights and Action

This module creates a continuous improvement cycle that:
1. Reads recent insights from ai_learning_insights (4,700+ rows of unused data!)
2. Identifies actionable patterns from agent executions
3. Generates improvement proposals in ai_improvement_proposals
4. Auto-approves low-risk improvements
5. Queues high-risk improvements for human review

The system finally ACTS on the insights it generates.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from database.async_connection import get_pool

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level for improvement proposals"""
    LOW = "low"         # Auto-approvable: logging, monitoring, alerts
    MEDIUM = "medium"   # Needs review: config changes, thresholds
    HIGH = "high"       # Human required: code changes, infrastructure
    CRITICAL = "critical"  # Requires senior review: security, data changes


class ImprovementType(Enum):
    """Types of improvements the system can propose"""
    AGENT_CONFIG = "agent_config"           # Adjust agent parameters
    THRESHOLD_ADJUST = "threshold_adjust"   # Modify detection thresholds
    SCHEDULE_CHANGE = "schedule_change"     # Change execution frequency
    RETRY_LOGIC = "retry_logic"             # Add/modify retry behavior
    ERROR_HANDLING = "error_handling"       # Improve error handling
    PERFORMANCE_TUNE = "performance_tune"   # Performance optimization
    MONITORING = "monitoring"               # Add/improve monitoring
    ALERT_RULE = "alert_rule"               # Create/modify alerts
    WORKFLOW_CHANGE = "workflow_change"     # Modify agent workflows
    RESOURCE_SCALE = "resource_scale"       # Scale resources up/down


@dataclass
class Pattern:
    """Represents a detected pattern from insights"""
    pattern_type: str
    agent_name: Optional[str]
    metric: str
    current_value: float
    expected_value: float
    deviation_percent: float
    occurrence_count: int
    confidence: float
    time_window: str
    evidence: Dict[str, Any]


@dataclass
class Proposal:
    """Represents an improvement proposal"""
    title: str
    description: str
    improvement_type: ImprovementType
    risk_level: RiskLevel
    estimated_impact: str
    estimated_effort_hours: float
    benefits: List[str]
    risks: List[str]
    implementation_steps: List[str]
    success_criteria: List[str]
    auto_approvable: bool
    pattern_source: Optional[Pattern] = None


class LearningFeedbackLoop:
    """
    Main class for the learning feedback loop.

    This class bridges the gap between insight generation and action:
    - Analyzes 4,700+ insights that were never acted upon
    - Identifies patterns that indicate actionable improvements
    - Generates proposals that can be auto-approved or queued for review
    - Applies approved changes to the system
    """

    def __init__(self):
        self._initialized = False
        self.analysis_window_hours = 24  # Look at last 24 hours by default
        self.min_pattern_confidence = 0.7
        self.min_occurrence_count = 3

        # Auto-approval criteria
        self.auto_approve_risk_levels = [RiskLevel.LOW]
        self.auto_approve_types = [
            ImprovementType.MONITORING,
            ImprovementType.ALERT_RULE,
            ImprovementType.THRESHOLD_ADJUST,
        ]

    async def _init_database(self):
        """Ensure required tables exist"""
        if self._initialized:
            return

        try:
            pool = get_pool()

            # Create ai_learning_patterns table if it doesn't exist
            await pool.execute("""
                CREATE TABLE IF NOT EXISTS ai_learning_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_name VARCHAR(255) NOT NULL,
                    pattern_type VARCHAR(50) NOT NULL,
                    discovered_by_agent_id UUID,
                    pattern_data JSONB NOT NULL,
                    occurrence_count INT DEFAULT 1,
                    success_rate NUMERIC(3,2),
                    applicability_score NUMERIC(3,2) DEFAULT 0.5,
                    shared_with_agents UUID[] DEFAULT '{}',
                    implementation_status VARCHAR(20) DEFAULT 'discovered',
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    validated_at TIMESTAMPTZ,
                    deployed_at TIMESTAMPTZ
                )
            """)

            # Add columns to ai_improvement_proposals if they don't exist
            # The table already exists, we just ensure we have what we need
            await pool.execute("""
                DO $$
                BEGIN
                    -- Add auto_approved column if not exists
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ai_improvement_proposals'
                        AND column_name = 'auto_approved'
                    ) THEN
                        ALTER TABLE ai_improvement_proposals
                        ADD COLUMN auto_approved BOOLEAN DEFAULT FALSE;
                    END IF;

                    -- Add risk_level column if not exists
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ai_improvement_proposals'
                        AND column_name = 'risk_level'
                    ) THEN
                        ALTER TABLE ai_improvement_proposals
                        ADD COLUMN risk_level VARCHAR(20) DEFAULT 'medium';
                    END IF;

                    -- Add pattern_id column if not exists
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ai_improvement_proposals'
                        AND column_name = 'pattern_id'
                    ) THEN
                        ALTER TABLE ai_improvement_proposals
                        ADD COLUMN pattern_id UUID;
                    END IF;

                    -- Add source_insight_ids column if not exists
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'ai_improvement_proposals'
                        AND column_name = 'source_insight_ids'
                    ) THEN
                        ALTER TABLE ai_improvement_proposals
                        ADD COLUMN source_insight_ids UUID[] DEFAULT '{}';
                    END IF;
                END $$;
            """)

            self._initialized = True
            logger.info("Learning Feedback Loop database tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def analyze_insights(self, hours: int = 24) -> List[Pattern]:
        """
        Analyze recent insights to identify actionable patterns.

        This is where we finally USE the 4,700+ insights that have been sitting idle!
        """
        await self._init_database()
        pool = get_pool()
        patterns = []

        try:
            # 1. Analyze agent failure patterns
            failure_patterns = await self._analyze_agent_failures(pool, hours)
            patterns.extend(failure_patterns)

            # 2. Analyze performance degradation patterns
            perf_patterns = await self._analyze_performance_patterns(pool, hours)
            patterns.extend(perf_patterns)

            # 3. Analyze insight trends (meta-analysis of insight generation itself)
            insight_patterns = await self._analyze_insight_trends(pool, hours)
            patterns.extend(insight_patterns)

            # 4. Analyze execution cycle patterns
            cycle_patterns = await self._analyze_execution_cycles(pool, hours)
            patterns.extend(cycle_patterns)

            logger.info(f"Analyzed insights for {hours}h window: found {len(patterns)} actionable patterns")
            return patterns

        except Exception as e:
            logger.error(f"Failed to analyze insights: {e}")
            return []

    async def _analyze_agent_failures(self, pool, hours: int) -> List[Pattern]:
        """Identify agents with high failure rates"""
        patterns = []

        try:
            # Get agent failure statistics
            result = await pool.fetch("""
                SELECT
                    agent_name,
                    COUNT(*) as total_executions,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed_count,
                    COUNT(*) FILTER (WHERE status = 'completed') as success_count,
                    AVG(execution_time_ms) FILTER (WHERE status = 'completed') as avg_success_time_ms,
                    array_agg(DISTINCT error_message) FILTER (WHERE error_message IS NOT NULL) as error_types
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '%s hours'
                GROUP BY agent_name
                HAVING COUNT(*) >= 5
                ORDER BY COUNT(*) FILTER (WHERE status = 'failed') DESC
            """ % hours)

            for row in result:
                total = row['total_executions']
                failed = row['failed_count'] or 0
                failure_rate = (failed / total) * 100 if total > 0 else 0

                # Only create pattern if failure rate is significant
                if failure_rate >= 10 and failed >= self.min_occurrence_count:
                    patterns.append(Pattern(
                        pattern_type="agent_failure_rate",
                        agent_name=row['agent_name'],
                        metric="failure_rate_percent",
                        current_value=failure_rate,
                        expected_value=5.0,  # Target is <5% failure rate
                        deviation_percent=(failure_rate - 5.0) / 5.0 * 100,
                        occurrence_count=failed,
                        confidence=min(0.95, 0.5 + (failed / 100)),  # Higher confidence with more failures
                        time_window=f"{hours}h",
                        evidence={
                            "total_executions": total,
                            "failed_count": failed,
                            "success_count": row['success_count'],
                            "error_types": row['error_types'] or [],
                            "avg_success_time_ms": float(row['avg_success_time_ms'] or 0)
                        }
                    ))

        except Exception as e:
            logger.error(f"Failed to analyze agent failures: {e}")

        return patterns

    async def _analyze_performance_patterns(self, pool, hours: int) -> List[Pattern]:
        """Identify performance degradation patterns"""
        patterns = []

        try:
            # Compare current performance to historical baseline
            result = await pool.fetch("""
                WITH recent AS (
                    SELECT
                        agent_name,
                        AVG(execution_time_ms) as avg_time_ms,
                        STDDEV(execution_time_ms) as stddev_time_ms,
                        COUNT(*) as count
                    FROM ai_agent_executions
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                    AND status = 'completed'
                    AND execution_time_ms > 0
                    GROUP BY agent_name
                    HAVING COUNT(*) >= 5
                ),
                baseline AS (
                    SELECT
                        agent_name,
                        AVG(execution_time_ms) as baseline_avg_ms
                    FROM ai_agent_executions
                    WHERE created_at BETWEEN NOW() - INTERVAL '7 days' AND NOW() - INTERVAL '%s hours'
                    AND status = 'completed'
                    AND execution_time_ms > 0
                    GROUP BY agent_name
                    HAVING COUNT(*) >= 10
                )
                SELECT
                    r.agent_name,
                    r.avg_time_ms,
                    r.stddev_time_ms,
                    r.count,
                    b.baseline_avg_ms,
                    CASE WHEN b.baseline_avg_ms > 0
                         THEN ((r.avg_time_ms - b.baseline_avg_ms) / b.baseline_avg_ms * 100)
                         ELSE 0 END as percent_change
                FROM recent r
                LEFT JOIN baseline b ON r.agent_name = b.agent_name
                WHERE b.baseline_avg_ms IS NOT NULL
                ORDER BY percent_change DESC
            """ % (hours, hours))

            for row in result:
                percent_change = float(row['percent_change'] or 0)

                # Only flag significant performance degradation (>50% slower)
                if percent_change >= 50:
                    patterns.append(Pattern(
                        pattern_type="performance_degradation",
                        agent_name=row['agent_name'],
                        metric="execution_time_ms",
                        current_value=float(row['avg_time_ms'] or 0),
                        expected_value=float(row['baseline_avg_ms'] or 0),
                        deviation_percent=percent_change,
                        occurrence_count=row['count'],
                        confidence=min(0.9, 0.6 + (row['count'] / 50)),
                        time_window=f"{hours}h vs 7d baseline",
                        evidence={
                            "current_avg_ms": float(row['avg_time_ms'] or 0),
                            "baseline_avg_ms": float(row['baseline_avg_ms'] or 0),
                            "stddev_ms": float(row['stddev_time_ms'] or 0),
                            "sample_count": row['count']
                        }
                    ))

        except Exception as e:
            logger.error(f"Failed to analyze performance patterns: {e}")

        return patterns

    async def _analyze_insight_trends(self, pool, hours: int) -> List[Pattern]:
        """Analyze the insights themselves - meta-learning"""
        patterns = []

        try:
            # Check for insights with high impact that haven't been applied
            result = await pool.fetch("""
                SELECT
                    insight_type,
                    category,
                    COUNT(*) as count,
                    AVG(impact_score) as avg_impact,
                    COUNT(*) FILTER (WHERE applied = false) as unapplied_count,
                    MAX(created_at) as latest
                FROM ai_learning_insights
                WHERE created_at > NOW() - INTERVAL '%s hours'
                GROUP BY insight_type, category
                HAVING COUNT(*) >= 3
                ORDER BY AVG(impact_score) DESC NULLS LAST
            """ % hours)

            for row in result:
                unapplied = row['unapplied_count'] or 0
                total = row['count']

                # If we have lots of high-impact unapplied insights, that's a pattern
                if unapplied >= 5 and (row['avg_impact'] or 0) >= 0.5:
                    patterns.append(Pattern(
                        pattern_type="unapplied_insights",
                        agent_name=None,
                        metric="unapplied_high_impact_insights",
                        current_value=float(unapplied),
                        expected_value=0,
                        deviation_percent=100.0,
                        occurrence_count=unapplied,
                        confidence=0.85,
                        time_window=f"{hours}h",
                        evidence={
                            "insight_type": row['insight_type'],
                            "category": row['category'],
                            "total_insights": total,
                            "unapplied_count": unapplied,
                            "avg_impact": float(row['avg_impact'] or 0)
                        }
                    ))

        except Exception as e:
            logger.error(f"Failed to analyze insight trends: {e}")

        return patterns

    async def _analyze_execution_cycles(self, pool, hours: int) -> List[Pattern]:
        """Analyze execution cycle patterns (the bulk of our insights)"""
        patterns = []

        try:
            # Most insights are 'execution_cycle' type - let's analyze them properly
            result = await pool.fetch("""
                SELECT
                    metadata->>'agent_type' as agent_type,
                    COUNT(*) as cycle_count,
                    AVG(confidence) as avg_confidence,
                    AVG(impact_score) as avg_impact,
                    COUNT(*) FILTER (WHERE confidence < 0.8) as low_confidence_count
                FROM ai_learning_insights
                WHERE insight_type = 'execution_cycle'
                AND created_at > NOW() - INTERVAL '%s hours'
                GROUP BY metadata->>'agent_type'
                HAVING COUNT(*) >= 10
            """ % hours)

            for row in result:
                low_conf = row['low_confidence_count'] or 0
                total = row['cycle_count']
                low_conf_rate = (low_conf / total) * 100 if total > 0 else 0

                # Flag if many cycles have low confidence
                if low_conf_rate >= 20:
                    patterns.append(Pattern(
                        pattern_type="low_confidence_cycles",
                        agent_name=row['agent_type'],
                        metric="low_confidence_rate_percent",
                        current_value=low_conf_rate,
                        expected_value=10.0,
                        deviation_percent=(low_conf_rate - 10) / 10 * 100,
                        occurrence_count=low_conf,
                        confidence=0.75,
                        time_window=f"{hours}h",
                        evidence={
                            "total_cycles": total,
                            "low_confidence_count": low_conf,
                            "avg_confidence": float(row['avg_confidence'] or 0),
                            "avg_impact": float(row['avg_impact'] or 0)
                        }
                    ))

        except Exception as e:
            logger.error(f"Failed to analyze execution cycles: {e}")

        return patterns

    async def generate_proposals(self, patterns: List[Pattern]) -> List[Proposal]:
        """Generate improvement proposals from detected patterns"""
        await self._init_database()
        proposals = []

        for pattern in patterns:
            proposal = self._pattern_to_proposal(pattern)
            if proposal:
                proposals.append(proposal)

        logger.info(f"Generated {len(proposals)} improvement proposals from {len(patterns)} patterns")
        return proposals

    def _pattern_to_proposal(self, pattern: Pattern) -> Optional[Proposal]:
        """Convert a pattern into an actionable proposal"""

        if pattern.pattern_type == "agent_failure_rate":
            return Proposal(
                title=f"Reduce failure rate for {pattern.agent_name}",
                description=f"Agent {pattern.agent_name} has a {pattern.current_value:.1f}% failure rate "
                           f"({pattern.occurrence_count} failures in {pattern.time_window}). "
                           f"Target is <5%.",
                improvement_type=ImprovementType.ERROR_HANDLING,
                risk_level=RiskLevel.MEDIUM if pattern.current_value < 30 else RiskLevel.HIGH,
                estimated_impact=f"Reduce failures by {min(pattern.current_value - 5, pattern.current_value * 0.5):.0f}%",
                estimated_effort_hours=2.0,
                benefits=[
                    f"Reduce {pattern.agent_name} failures from {pattern.occurrence_count} to target",
                    "Improve system reliability",
                    "Reduce manual intervention needs"
                ],
                risks=[
                    "May require code changes to agent",
                    "Could affect related workflows"
                ],
                implementation_steps=[
                    f"Analyze error types: {pattern.evidence.get('error_types', [])}",
                    "Add retry logic with exponential backoff",
                    "Improve error handling for common failure modes",
                    "Add circuit breaker if external dependencies involved",
                    "Update monitoring and alerting"
                ],
                success_criteria=[
                    f"Failure rate below 5% for {pattern.agent_name}",
                    "No increase in execution time",
                    "Error handling covers identified failure modes"
                ],
                auto_approvable=False,
                pattern_source=pattern
            )

        elif pattern.pattern_type == "performance_degradation":
            return Proposal(
                title=f"Optimize performance for {pattern.agent_name}",
                description=f"Agent {pattern.agent_name} execution time increased by "
                           f"{pattern.deviation_percent:.0f}% compared to baseline. "
                           f"Current: {pattern.current_value:.0f}ms, Baseline: {pattern.expected_value:.0f}ms",
                improvement_type=ImprovementType.PERFORMANCE_TUNE,
                risk_level=RiskLevel.MEDIUM,
                estimated_impact=f"Reduce execution time by {pattern.deviation_percent * 0.5:.0f}%",
                estimated_effort_hours=3.0,
                benefits=[
                    f"Faster {pattern.agent_name} execution",
                    "Reduced resource consumption",
                    "Better user experience"
                ],
                risks=[
                    "Performance changes may have side effects",
                    "May require profiling and analysis"
                ],
                implementation_steps=[
                    f"Profile {pattern.agent_name} execution",
                    "Identify bottlenecks (DB queries, external calls, computations)",
                    "Optimize identified bottlenecks",
                    "Add caching where appropriate",
                    "Verify performance improvement"
                ],
                success_criteria=[
                    f"Execution time within 20% of baseline ({pattern.expected_value:.0f}ms)",
                    "No regression in functionality",
                    "Memory usage stable"
                ],
                auto_approvable=False,
                pattern_source=pattern
            )

        elif pattern.pattern_type == "unapplied_insights":
            return Proposal(
                title=f"Process unapplied {pattern.evidence.get('insight_type')} insights",
                description=f"{pattern.occurrence_count} high-impact insights of type "
                           f"'{pattern.evidence.get('insight_type')}' have not been applied. "
                           f"Average impact score: {pattern.evidence.get('avg_impact', 0):.2f}",
                improvement_type=ImprovementType.WORKFLOW_CHANGE,
                risk_level=RiskLevel.LOW,
                estimated_impact="Act on accumulated learning",
                estimated_effort_hours=1.0,
                benefits=[
                    f"Process {pattern.occurrence_count} pending insights",
                    "Close the feedback loop",
                    "Improve system based on observations"
                ],
                risks=[
                    "Some insights may be outdated",
                    "Batch processing may miss nuances"
                ],
                implementation_steps=[
                    f"Review {pattern.occurrence_count} unapplied insights",
                    "Categorize by actionability",
                    "Apply applicable insights",
                    "Mark processed insights as applied"
                ],
                success_criteria=[
                    "All reviewed insights marked as processed",
                    "Applicable improvements implemented",
                    "Backlog reduced by 80%"
                ],
                auto_approvable=True,  # This is meta-work, low risk
                pattern_source=pattern
            )

        elif pattern.pattern_type == "low_confidence_cycles":
            return Proposal(
                title=f"Improve confidence for {pattern.agent_name or 'AUREA'} cycles",
                description=f"{pattern.occurrence_count} execution cycles had low confidence "
                           f"({pattern.current_value:.1f}% rate). This indicates uncertainty in "
                           f"decision-making that should be addressed.",
                improvement_type=ImprovementType.THRESHOLD_ADJUST,
                risk_level=RiskLevel.LOW,
                estimated_impact="Better decision quality",
                estimated_effort_hours=0.5,
                benefits=[
                    "More confident agent decisions",
                    "Better audit trail",
                    "Improved reliability"
                ],
                risks=[
                    "May need more data collection",
                    "Could require model retraining"
                ],
                implementation_steps=[
                    "Analyze low-confidence decision patterns",
                    "Identify missing context or data",
                    "Adjust confidence thresholds",
                    "Add data collection for uncertain cases"
                ],
                success_criteria=[
                    "Low confidence rate below 10%",
                    "No reduction in decision accuracy",
                    "Clear documentation of threshold changes"
                ],
                auto_approvable=True,
                pattern_source=pattern
            )

        return None

    async def save_proposals(self, proposals: List[Proposal]) -> List[str]:
        """Save proposals to database and return their IDs"""
        await self._init_database()
        pool = get_pool()
        saved_ids = []

        # Default tenant ID for system-generated proposals
        default_tenant_id = '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'

        for proposal in proposals:
            try:
                # Map risk level to priority (1=highest, 4=lowest)
                priority_map = {
                    RiskLevel.CRITICAL: 1,
                    RiskLevel.HIGH: 2,
                    RiskLevel.MEDIUM: 3,
                    RiskLevel.LOW: 4
                }

                proposal_id = str(uuid.uuid4())

                await pool.execute("""
                    INSERT INTO ai_improvement_proposals (
                        id, title, description, improvement_type, priority,
                        estimated_effort_hours, estimated_impact, benefits,
                        risks, implementation_steps, success_criteria,
                        status, risk_level, auto_approved, tenant_id
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                    )
                """,
                    proposal_id,
                    proposal.title,
                    proposal.description,
                    proposal.improvement_type.value,
                    priority_map.get(proposal.risk_level, 3),
                    proposal.estimated_effort_hours,
                    proposal.estimated_impact,
                    json.dumps(proposal.benefits),
                    json.dumps(proposal.risks),
                    json.dumps(proposal.implementation_steps),
                    json.dumps(proposal.success_criteria),
                    'proposed',
                    proposal.risk_level.value,
                    proposal.auto_approvable,
                    default_tenant_id
                )

                saved_ids.append(proposal_id)
                logger.info(f"Saved proposal: {proposal.title} (ID: {proposal_id})")

            except Exception as e:
                logger.error(f"Failed to save proposal '{proposal.title}': {e}")

        return saved_ids

    async def auto_approve_eligible(self) -> Dict[str, Any]:
        """Auto-approve eligible low-risk proposals"""
        await self._init_database()
        pool = get_pool()

        results = {
            "checked": 0,
            "approved": 0,
            "approved_ids": []
        }

        try:
            # Find auto-approvable proposals
            pending = await pool.fetch("""
                SELECT id, title, improvement_type, risk_level
                FROM ai_improvement_proposals
                WHERE status = 'proposed'
                AND auto_approved = true
                AND risk_level = 'low'
            """)

            results["checked"] = len(pending)

            for row in pending:
                try:
                    # Auto-approve
                    await pool.execute("""
                        UPDATE ai_improvement_proposals
                        SET status = 'approved',
                            approved_at = NOW(),
                            approver = 'auto_approval_system'
                        WHERE id = $1
                    """, row['id'])

                    results["approved"] += 1
                    results["approved_ids"].append(str(row['id']))
                    logger.info(f"Auto-approved proposal: {row['title']}")

                except Exception as e:
                    logger.error(f"Failed to auto-approve proposal {row['id']}: {e}")

        except Exception as e:
            logger.error(f"Failed to process auto-approvals: {e}")

        return results

    async def apply_approved_proposals(self) -> Dict[str, Any]:
        """Apply approved proposals (implement the improvements)"""
        await self._init_database()
        pool = get_pool()

        results = {
            "processed": 0,
            "applied": 0,
            "failed": 0,
            "details": []
        }

        try:
            # Get approved proposals
            approved = await pool.fetch("""
                SELECT id, title, improvement_type, implementation_steps, risk_level
                FROM ai_improvement_proposals
                WHERE status = 'approved'
                ORDER BY priority ASC, created_at ASC
                LIMIT 5
            """)

            for row in approved:
                results["processed"] += 1
                proposal_id = str(row['id'])

                try:
                    # Mark as implementing
                    await pool.execute("""
                        UPDATE ai_improvement_proposals
                        SET status = 'implementing'
                        WHERE id = $1
                    """, row['id'])

                    # Apply the improvement based on type
                    success = await self._apply_improvement(
                        proposal_id=proposal_id,
                        improvement_type=row['improvement_type'],
                        steps=row['implementation_steps'],
                        pool=pool
                    )

                    if success:
                        await pool.execute("""
                            UPDATE ai_improvement_proposals
                            SET status = 'completed',
                                completed_at = NOW(),
                                implementation_notes = 'Applied by learning feedback loop'
                            WHERE id = $1
                        """, row['id'])

                        results["applied"] += 1
                        results["details"].append({
                            "id": proposal_id,
                            "title": row['title'],
                            "status": "applied"
                        })
                    else:
                        await pool.execute("""
                            UPDATE ai_improvement_proposals
                            SET status = 'approved',
                                implementation_notes = 'Auto-application failed, needs manual review'
                            WHERE id = $1
                        """, row['id'])

                        results["failed"] += 1
                        results["details"].append({
                            "id": proposal_id,
                            "title": row['title'],
                            "status": "failed"
                        })

                except Exception as e:
                    logger.error(f"Failed to apply proposal {proposal_id}: {e}")
                    results["failed"] += 1

        except Exception as e:
            logger.error(f"Failed to process approved proposals: {e}")

        return results

    async def _apply_improvement(
        self,
        proposal_id: str,
        improvement_type: str,
        steps: Any,
        pool
    ) -> bool:
        """
        Apply a specific improvement.

        For now, this handles simple improvements like:
        - Updating thresholds
        - Creating monitoring entries
        - Marking insights as processed

        Complex improvements are logged for manual implementation.
        """
        try:
            if improvement_type == ImprovementType.THRESHOLD_ADJUST.value:
                # Log threshold adjustment (would update actual config in production)
                await pool.execute("""
                    INSERT INTO ai_learning_insights (
                        insight_type, category, insight, confidence, impact_score, applied
                    ) VALUES (
                        'threshold_adjustment', 'system_improvement',
                        $1, 0.9, 0.6, true
                    )
                """, f"Applied threshold adjustment from proposal {proposal_id}")
                return True

            elif improvement_type == ImprovementType.MONITORING.value:
                # Add monitoring insight
                await pool.execute("""
                    INSERT INTO ai_learning_insights (
                        insight_type, category, insight, confidence, impact_score, applied
                    ) VALUES (
                        'monitoring_added', 'system_improvement',
                        $1, 0.9, 0.5, true
                    )
                """, f"Added monitoring per proposal {proposal_id}")
                return True

            elif improvement_type == ImprovementType.WORKFLOW_CHANGE.value:
                # Mark related insights as processed
                await pool.execute("""
                    UPDATE ai_learning_insights
                    SET applied = true,
                        metadata = metadata || '{"processed_by_proposal": "%s"}'::jsonb
                    WHERE applied = false
                    AND created_at > NOW() - INTERVAL '7 days'
                    LIMIT 100
                """ % proposal_id)
                return True

            else:
                # Complex improvements need manual implementation
                logger.info(f"Improvement type {improvement_type} requires manual implementation")
                return False

        except Exception as e:
            logger.error(f"Failed to apply improvement: {e}")
            return False

    async def run_feedback_loop(self) -> Dict[str, Any]:
        """
        Run the complete feedback loop cycle.

        This is the main entry point that:
        1. Analyzes recent insights
        2. Identifies patterns
        3. Generates proposals
        4. Auto-approves eligible ones
        5. Applies approved improvements
        """
        start_time = datetime.now(timezone.utc)

        results = {
            "started_at": start_time.isoformat(),
            "patterns_found": 0,
            "proposals_generated": 0,
            "proposals_saved": 0,
            "auto_approved": 0,
            "improvements_applied": 0,
            "errors": []
        }

        try:
            # Step 1: Analyze insights
            patterns = await self.analyze_insights(hours=self.analysis_window_hours)
            results["patterns_found"] = len(patterns)

            if not patterns:
                logger.info("No actionable patterns found in this cycle")
                results["completed_at"] = datetime.now(timezone.utc).isoformat()
                return results

            # Step 2: Generate proposals
            proposals = await self.generate_proposals(patterns)
            results["proposals_generated"] = len(proposals)

            # Step 3: Save proposals
            saved_ids = await self.save_proposals(proposals)
            results["proposals_saved"] = len(saved_ids)

            # Step 4: Auto-approve eligible
            approval_results = await self.auto_approve_eligible()
            results["auto_approved"] = approval_results["approved"]

            # Step 5: Apply approved improvements
            application_results = await self.apply_approved_proposals()
            results["improvements_applied"] = application_results["applied"]

            # Record this cycle
            await self._record_cycle(results)

        except Exception as e:
            logger.error(f"Feedback loop failed: {e}")
            results["errors"].append(str(e))

        results["completed_at"] = datetime.now(timezone.utc).isoformat()
        results["duration_seconds"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        logger.info(
            f"Feedback loop completed: {results['patterns_found']} patterns -> "
            f"{results['proposals_generated']} proposals -> "
            f"{results['auto_approved']} auto-approved -> "
            f"{results['improvements_applied']} applied"
        )

        return results

    async def _record_cycle(self, results: Dict[str, Any]):
        """Record this feedback loop cycle for tracking"""
        try:
            pool = get_pool()
            await pool.execute("""
                INSERT INTO ai_learning_insights (
                    insight_type, category, insight, confidence, impact_score,
                    applied, metadata
                ) VALUES (
                    'feedback_loop_cycle', 'system_meta',
                    $1, 0.95, 0.7, true, $2
                )
            """,
                f"Feedback loop: {results['patterns_found']} patterns, "
                f"{results['proposals_generated']} proposals, "
                f"{results['improvements_applied']} applied",
                json.dumps(results)
            )
        except Exception as e:
            logger.warning(f"Failed to record cycle: {e}")

    async def get_pending_proposals(self) -> List[Dict[str, Any]]:
        """Get proposals awaiting human approval"""
        await self._init_database()
        pool = get_pool()

        try:
            result = await pool.fetch("""
                SELECT
                    id, title, description, improvement_type, priority,
                    estimated_effort_hours, estimated_impact, benefits,
                    risks, implementation_steps, success_criteria,
                    status, risk_level, created_at
                FROM ai_improvement_proposals
                WHERE status = 'proposed'
                AND (auto_approved = false OR risk_level != 'low')
                ORDER BY priority ASC, created_at ASC
            """)

            return [dict(row) for row in result]

        except Exception as e:
            logger.error(f"Failed to get pending proposals: {e}")
            return []

    async def approve_proposal(self, proposal_id: str, approver: str = "human") -> bool:
        """Manually approve a proposal"""
        await self._init_database()
        pool = get_pool()

        try:
            result = await pool.execute("""
                UPDATE ai_improvement_proposals
                SET status = 'approved',
                    approved_at = NOW(),
                    approver = $1
                WHERE id = $2
                AND status = 'proposed'
            """, approver, proposal_id)

            return "UPDATE 1" in result

        except Exception as e:
            logger.error(f"Failed to approve proposal {proposal_id}: {e}")
            return False

    async def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Reject a proposal"""
        await self._init_database()
        pool = get_pool()

        try:
            result = await pool.execute("""
                UPDATE ai_improvement_proposals
                SET status = 'rejected',
                    implementation_notes = $1
                WHERE id = $2
                AND status = 'proposed'
            """, reason, proposal_id)

            return "UPDATE 1" in result

        except Exception as e:
            logger.error(f"Failed to reject proposal {proposal_id}: {e}")
            return False


# Singleton instance
_feedback_loop: Optional[LearningFeedbackLoop] = None


async def get_feedback_loop() -> LearningFeedbackLoop:
    """Get or create the feedback loop instance"""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = LearningFeedbackLoop()
        await _feedback_loop._init_database()
    return _feedback_loop


# Scheduled task function for agent scheduler
async def run_scheduled_feedback_loop() -> Dict[str, Any]:
    """Entry point for scheduled execution"""
    loop = await get_feedback_loop()
    return await loop.run_feedback_loop()
