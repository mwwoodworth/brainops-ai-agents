"""
Conversion Analytics Engine
Track and optimize conversion rates across the entire funnel

Features:
- End-to-end funnel metrics
- Stage velocity analysis
- Win/loss analysis
- Attribution tracking
- Predictive conversion modeling
- Revenue forecasting
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, field
from enum import Enum

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class FunnelStage(Enum):
    """Sales funnel stages"""
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATING = "negotiating"
    WON = "won"
    LOST = "lost"


@dataclass
class StageMetrics:
    """Metrics for a single funnel stage"""
    stage: FunnelStage
    total_leads: int = 0
    converted: int = 0
    conversion_rate: float = 0.0
    avg_days_in_stage: float = 0.0
    avg_deal_size: float = 0.0
    total_value: float = 0.0


@dataclass
class FunnelMetrics:
    """Complete funnel metrics"""
    stages: List[StageMetrics]
    overall_conversion: float = 0.0
    total_pipeline_value: float = 0.0
    total_won_value: float = 0.0
    avg_sales_cycle_days: float = 0.0
    velocity_score: float = 0.0  # 0-100
    calculated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class WinLossAnalysis:
    """Win/loss analysis results"""
    won_count: int = 0
    lost_count: int = 0
    win_rate: float = 0.0
    won_reasons: List[Dict] = field(default_factory=list)
    lost_reasons: List[Dict] = field(default_factory=list)
    competitor_wins: Dict = field(default_factory=dict)
    avg_won_deal_size: float = 0.0
    avg_lost_deal_size: float = 0.0


class ConversionAnalyticsEngine:
    """
    Ultimate Conversion Analytics Engine

    Features:
    - Real-time funnel tracking
    - Stage-by-stage conversion analysis
    - Bottleneck identification
    - Predictive modeling
    - Revenue forecasting
    - Actionable optimization recommendations
    """

    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
        self._initialized = False

        # Stage progression order
        self.stage_order = [
            FunnelStage.NEW,
            FunnelStage.CONTACTED,
            FunnelStage.QUALIFIED,
            FunnelStage.PROPOSAL_SENT,
            FunnelStage.NEGOTIATING,
            FunnelStage.WON
        ]

        # Benchmark conversion rates (industry average)
        self.benchmark_rates = {
            ('new', 'contacted'): 0.70,
            ('contacted', 'qualified'): 0.45,
            ('qualified', 'proposal_sent'): 0.60,
            ('proposal_sent', 'negotiating'): 0.50,
            ('negotiating', 'won'): 0.40
        }

    def _get_connection(self):
        """Get database connection"""
        if not self.db_url:
            raise ValueError("DATABASE_URL not configured")
        return psycopg2.connect(self.db_url)

    async def initialize_tables(self):
        """Create analytics tables"""
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Funnel snapshots
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversion_funnel_snapshots (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        snapshot_date DATE NOT NULL,
                        stage VARCHAR(50) NOT NULL,
                        leads_in_stage INT DEFAULT 0,
                        leads_entered INT DEFAULT 0,
                        leads_converted INT DEFAULT 0,
                        leads_lost INT DEFAULT 0,
                        conversion_rate FLOAT,
                        avg_days_in_stage FLOAT,
                        total_value FLOAT DEFAULT 0,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(snapshot_date, stage)
                    )
                """)

                # Stage transitions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS stage_transitions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        lead_id UUID NOT NULL,
                        from_stage VARCHAR(50),
                        to_stage VARCHAR(50) NOT NULL,
                        transition_reason VARCHAR(255),
                        days_in_previous_stage FLOAT,
                        deal_value FLOAT,
                        transitioned_at TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Win/loss records
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS win_loss_records (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        lead_id UUID NOT NULL UNIQUE,
                        outcome VARCHAR(10) NOT NULL,
                        deal_value FLOAT,
                        primary_reason VARCHAR(255),
                        secondary_reasons JSONB DEFAULT '[]',
                        competitor_involved VARCHAR(255),
                        sales_cycle_days INT,
                        touchpoints INT,
                        decision_makers_engaged INT,
                        closed_at TIMESTAMPTZ DEFAULT NOW(),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Conversion predictions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversion_predictions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        lead_id UUID NOT NULL,
                        current_stage VARCHAR(50),
                        predicted_outcome VARCHAR(10),
                        probability_won FLOAT,
                        probability_lost FLOAT,
                        predicted_close_date DATE,
                        predicted_value FLOAT,
                        confidence FLOAT,
                        factors JSONB DEFAULT '{}',
                        predicted_at TIMESTAMPTZ DEFAULT NOW(),
                        actual_outcome VARCHAR(10),
                        outcome_date DATE,
                        prediction_accuracy FLOAT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                # Revenue forecasts
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS revenue_forecasts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        forecast_date DATE NOT NULL,
                        forecast_period VARCHAR(20),
                        pipeline_value FLOAT DEFAULT 0,
                        weighted_pipeline FLOAT DEFAULT 0,
                        committed_value FLOAT DEFAULT 0,
                        best_case_value FLOAT DEFAULT 0,
                        forecast_value FLOAT DEFAULT 0,
                        actual_value FLOAT,
                        accuracy FLOAT,
                        methodology VARCHAR(50),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE(forecast_date, forecast_period)
                    )
                """)

                # Create indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_transitions_lead
                    ON stage_transitions(lead_id)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_transitions_date
                    ON stage_transitions(transitioned_at DESC)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_winloss_outcome
                    ON win_loss_records(outcome)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_predictions_lead
                    ON conversion_predictions(lead_id)
                """)

                conn.commit()
                self._initialized = True
                logger.info("Conversion analytics tables initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize analytics tables: {e}")
            raise
        finally:
            conn.close()

    async def calculate_funnel_metrics(self, days: int = 30) -> FunnelMetrics:
        """
        Calculate comprehensive funnel metrics

        Returns detailed conversion rates and metrics for each stage
        """
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stage_metrics = []

                for i, stage in enumerate(self.stage_order):
                    # Get leads in this stage
                    cur.execute("""
                        SELECT
                            COUNT(*) as total,
                            COALESCE(SUM(expected_revenue), 0) as total_value,
                            COALESCE(AVG(expected_revenue), 0) as avg_value
                        FROM revenue_leads
                        WHERE stage = %s
                        AND created_at > NOW() - INTERVAL '%s days'
                    """, (stage.value, days))
                    stage_data = cur.fetchone()

                    # Get conversion to next stage
                    converted = 0
                    if i < len(self.stage_order) - 1:
                        next_stage = self.stage_order[i + 1]
                        cur.execute("""
                            SELECT COUNT(*) as converted
                            FROM stage_transitions
                            WHERE from_stage = %s AND to_stage = %s
                            AND transitioned_at > NOW() - INTERVAL '%s days'
                        """, (stage.value, next_stage.value, days))
                        converted = cur.fetchone()['converted']

                    # Calculate days in stage
                    cur.execute("""
                        SELECT COALESCE(AVG(days_in_previous_stage), 0) as avg_days
                        FROM stage_transitions
                        WHERE from_stage = %s
                        AND transitioned_at > NOW() - INTERVAL '%s days'
                    """, (stage.value, days))
                    avg_days = cur.fetchone()['avg_days']

                    total = stage_data['total'] or 0
                    conversion_rate = converted / total if total > 0 else 0

                    stage_metrics.append(StageMetrics(
                        stage=stage,
                        total_leads=total,
                        converted=converted,
                        conversion_rate=conversion_rate,
                        avg_days_in_stage=avg_days,
                        avg_deal_size=stage_data['avg_value'],
                        total_value=stage_data['total_value']
                    ))

                # Calculate overall metrics
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE stage = 'new') as new_leads,
                        COUNT(*) FILTER (WHERE stage = 'won') as won_leads,
                        COALESCE(SUM(expected_revenue), 0) as pipeline_value,
                        COALESCE(SUM(expected_revenue) FILTER (WHERE stage = 'won'), 0) as won_value
                    FROM revenue_leads
                    WHERE created_at > NOW() - INTERVAL '%s days'
                """, (days,))
                overall = cur.fetchone()

                # Calculate average sales cycle
                cur.execute("""
                    SELECT COALESCE(AVG(sales_cycle_days), 0) as avg_cycle
                    FROM win_loss_records
                    WHERE closed_at > NOW() - INTERVAL '%s days'
                    AND outcome = 'won'
                """, (days,))
                avg_cycle = cur.fetchone()['avg_cycle']

                # Calculate velocity score (based on benchmark comparison)
                velocity_score = self._calculate_velocity_score(stage_metrics)

                new_leads = overall['new_leads'] or 1
                won_leads = overall['won_leads'] or 0

                return FunnelMetrics(
                    stages=stage_metrics,
                    overall_conversion=won_leads / new_leads,
                    total_pipeline_value=overall['pipeline_value'],
                    total_won_value=overall['won_value'],
                    avg_sales_cycle_days=avg_cycle,
                    velocity_score=velocity_score
                )

        finally:
            conn.close()

    def _calculate_velocity_score(self, stage_metrics: List[StageMetrics]) -> float:
        """Calculate velocity score compared to benchmarks"""
        scores = []

        for i, metrics in enumerate(stage_metrics[:-1]):  # Exclude WON stage
            from_stage = metrics.stage.value
            to_stage = self.stage_order[i + 1].value
            benchmark = self.benchmark_rates.get((from_stage, to_stage), 0.5)

            if benchmark > 0:
                ratio = metrics.conversion_rate / benchmark
                score = min(100, ratio * 100)
                scores.append(score)

        return sum(scores) / len(scores) if scores else 50

    async def identify_bottlenecks(self) -> List[Dict]:
        """
        Identify funnel bottlenecks and optimization opportunities
        """
        funnel = await self.calculate_funnel_metrics()
        bottlenecks = []

        for i, stage in enumerate(funnel.stages[:-1]):
            from_stage = stage.stage.value
            to_stage = self.stage_order[i + 1].value
            benchmark = self.benchmark_rates.get((from_stage, to_stage), 0.5)

            if stage.conversion_rate < benchmark * 0.8:  # More than 20% below benchmark
                gap = benchmark - stage.conversion_rate
                impact = stage.total_value * gap

                bottlenecks.append({
                    'stage': from_stage,
                    'current_rate': round(stage.conversion_rate * 100, 1),
                    'benchmark_rate': round(benchmark * 100, 1),
                    'gap_percentage': round(gap * 100, 1),
                    'revenue_impact': round(impact, 2),
                    'avg_days_in_stage': round(stage.avg_days_in_stage, 1),
                    'severity': 'critical' if gap > 0.2 else 'high' if gap > 0.1 else 'medium',
                    'recommendations': self._get_stage_recommendations(from_stage, stage)
                })

        # Sort by revenue impact
        bottlenecks.sort(key=lambda x: x['revenue_impact'], reverse=True)

        return bottlenecks

    def _get_stage_recommendations(self, stage: str, metrics: StageMetrics) -> List[str]:
        """Get recommendations for improving a specific stage"""
        recommendations = {
            'new': [
                'Improve lead qualification criteria',
                'Implement faster initial outreach (< 5 min)',
                'Add more personalized first touch',
                'Use multi-channel initial contact'
            ],
            'contacted': [
                'Increase follow-up frequency',
                'Add value-driven content to outreach',
                'Implement call-to-action optimization',
                'Test different messaging approaches'
            ],
            'qualified': [
                'Reduce time to proposal',
                'Improve discovery call process',
                'Add social proof at this stage',
                'Implement deal scoring for prioritization'
            ],
            'proposal_sent': [
                'Add proposal tracking and alerts',
                'Implement proposal follow-up automation',
                'Test different pricing presentations',
                'Add urgency elements to proposals'
            ],
            'negotiating': [
                'Reduce decision-maker access time',
                'Improve objection handling training',
                'Add competitive comparison materials',
                'Implement deal-closing incentives'
            ]
        }

        return recommendations.get(stage, ['Analyze stage-specific friction points'])

    async def analyze_win_loss(self, days: int = 90) -> WinLossAnalysis:
        """
        Comprehensive win/loss analysis
        """
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get win/loss counts
                cur.execute("""
                    SELECT
                        outcome,
                        COUNT(*) as count,
                        COALESCE(AVG(deal_value), 0) as avg_value,
                        COALESCE(SUM(deal_value), 0) as total_value
                    FROM win_loss_records
                    WHERE closed_at > NOW() - INTERVAL '%s days'
                    GROUP BY outcome
                """, (days,))
                outcomes = {row['outcome']: row for row in cur.fetchall()}

                won = outcomes.get('won', {'count': 0, 'avg_value': 0})
                lost = outcomes.get('lost', {'count': 0, 'avg_value': 0})

                total = won['count'] + lost['count']
                win_rate = won['count'] / total if total > 0 else 0

                # Get won reasons
                cur.execute("""
                    SELECT primary_reason, COUNT(*) as count
                    FROM win_loss_records
                    WHERE outcome = 'won'
                    AND closed_at > NOW() - INTERVAL '%s days'
                    GROUP BY primary_reason
                    ORDER BY count DESC
                    LIMIT 10
                """, (days,))
                won_reasons = [{'reason': r['primary_reason'], 'count': r['count']}
                              for r in cur.fetchall()]

                # Get lost reasons
                cur.execute("""
                    SELECT primary_reason, COUNT(*) as count
                    FROM win_loss_records
                    WHERE outcome = 'lost'
                    AND closed_at > NOW() - INTERVAL '%s days'
                    GROUP BY primary_reason
                    ORDER BY count DESC
                    LIMIT 10
                """, (days,))
                lost_reasons = [{'reason': r['primary_reason'], 'count': r['count']}
                               for r in cur.fetchall()]

                # Get competitor analysis
                cur.execute("""
                    SELECT
                        competitor_involved,
                        COUNT(*) FILTER (WHERE outcome = 'won') as wins,
                        COUNT(*) FILTER (WHERE outcome = 'lost') as losses
                    FROM win_loss_records
                    WHERE competitor_involved IS NOT NULL
                    AND closed_at > NOW() - INTERVAL '%s days'
                    GROUP BY competitor_involved
                """, (days,))
                competitor_wins = {}
                for row in cur.fetchall():
                    total_deals = row['wins'] + row['losses']
                    competitor_wins[row['competitor_involved']] = {
                        'wins': row['wins'],
                        'losses': row['losses'],
                        'win_rate': row['wins'] / total_deals if total_deals > 0 else 0
                    }

                return WinLossAnalysis(
                    won_count=won['count'],
                    lost_count=lost['count'],
                    win_rate=win_rate,
                    won_reasons=won_reasons,
                    lost_reasons=lost_reasons,
                    competitor_wins=competitor_wins,
                    avg_won_deal_size=won['avg_value'],
                    avg_lost_deal_size=lost['avg_value']
                )

        finally:
            conn.close()

    async def predict_conversion(self, lead_id: str) -> Dict:
        """
        Predict conversion probability for a specific lead
        """
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get lead data
                cur.execute("""
                    SELECT * FROM revenue_leads WHERE id = %s
                """, (lead_id,))
                lead = cur.fetchone()

                if not lead:
                    return {'error': 'Lead not found'}

                # Get advanced scoring if available
                cur.execute("""
                    SELECT * FROM advanced_lead_metrics WHERE lead_id = %s
                """, (lead_id,))
                scoring = cur.fetchone()

                # Get historical conversion rates for this stage
                stage = lead.get('stage', 'new')
                stage_idx = next((i for i, s in enumerate(self.stage_order)
                                 if s.value == stage), 0)

                # Calculate probability based on stage and score
                base_probability = 1.0
                for i in range(stage_idx, len(self.stage_order) - 1):
                    from_stage = self.stage_order[i].value
                    to_stage = self.stage_order[i + 1].value
                    stage_rate = self.benchmark_rates.get((from_stage, to_stage), 0.5)
                    base_probability *= stage_rate

                # Adjust for lead score
                score_multiplier = 1.0
                if scoring:
                    composite = scoring.get('composite_score', 50)
                    score_multiplier = 0.5 + (composite / 100)  # 0.5 to 1.5x

                probability_won = min(0.95, base_probability * score_multiplier)

                # Estimate close date
                avg_days_remaining = (len(self.stage_order) - 1 - stage_idx) * 5  # 5 days per stage avg
                predicted_close = datetime.utcnow() + timedelta(days=avg_days_remaining)

                # Store prediction
                cur.execute("""
                    INSERT INTO conversion_predictions (
                        lead_id, current_stage, predicted_outcome,
                        probability_won, probability_lost, predicted_close_date,
                        predicted_value, confidence, factors
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    lead_id,
                    stage,
                    'won' if probability_won > 0.5 else 'lost',
                    probability_won,
                    1 - probability_won,
                    predicted_close.date(),
                    lead.get('expected_revenue', 0),
                    0.75,  # Confidence level
                    json.dumps({
                        'stage': stage,
                        'score': scoring.get('composite_score') if scoring else None,
                        'base_probability': base_probability,
                        'score_multiplier': score_multiplier
                    })
                ))
                conn.commit()

                return {
                    'lead_id': lead_id,
                    'current_stage': stage,
                    'probability_won': round(probability_won, 3),
                    'probability_lost': round(1 - probability_won, 3),
                    'predicted_outcome': 'won' if probability_won > 0.5 else 'lost',
                    'predicted_close_date': predicted_close.date().isoformat(),
                    'predicted_value': lead.get('expected_revenue', 0),
                    'confidence': 0.75,
                    'factors': {
                        'stage_position': f"{stage_idx + 1}/{len(self.stage_order)}",
                        'lead_score': scoring.get('composite_score') if scoring else 'N/A',
                        'lead_tier': scoring.get('tier') if scoring else 'N/A'
                    }
                }

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to predict conversion: {e}")
            return {'error': str(e)}
        finally:
            conn.close()

    async def forecast_revenue(self, period: str = 'month') -> Dict:
        """
        Generate revenue forecast

        period: 'week', 'month', 'quarter'
        """
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get pipeline by stage
                cur.execute("""
                    SELECT
                        stage,
                        COUNT(*) as count,
                        COALESCE(SUM(expected_revenue), 0) as value
                    FROM revenue_leads
                    WHERE stage NOT IN ('won', 'lost')
                    GROUP BY stage
                """)
                pipeline_by_stage = {row['stage']: row for row in cur.fetchall()}

                # Calculate weighted pipeline
                weighted_pipeline = 0
                pipeline_value = 0

                for stage, data in pipeline_by_stage.items():
                    stage_idx = next((i for i, s in enumerate(self.stage_order)
                                     if s.value == stage), 0)

                    # Probability increases with stage progression
                    probability = (stage_idx + 1) / len(self.stage_order) * 0.8

                    pipeline_value += data['value']
                    weighted_pipeline += data['value'] * probability

                # Get committed (negotiating stage)
                committed_value = pipeline_by_stage.get('negotiating', {}).get('value', 0)

                # Best case (all pipeline closes)
                best_case = pipeline_value

                # Calculate forecast based on historical win rate
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE outcome = 'won') as won,
                        COUNT(*) as total
                    FROM win_loss_records
                    WHERE closed_at > NOW() - INTERVAL '90 days'
                """)
                historical = cur.fetchone()
                win_rate = historical['won'] / historical['total'] if historical['total'] > 0 else 0.3

                forecast_value = weighted_pipeline * (1 + win_rate) / 2  # Blend weighted and win rate

                # Store forecast
                cur.execute("""
                    INSERT INTO revenue_forecasts (
                        forecast_date, forecast_period, pipeline_value,
                        weighted_pipeline, committed_value, best_case_value,
                        forecast_value, methodology
                    ) VALUES (CURRENT_DATE, %s, %s, %s, %s, %s, %s, 'weighted_probability')
                    ON CONFLICT (forecast_date, forecast_period) DO UPDATE SET
                        pipeline_value = EXCLUDED.pipeline_value,
                        weighted_pipeline = EXCLUDED.weighted_pipeline,
                        committed_value = EXCLUDED.committed_value,
                        best_case_value = EXCLUDED.best_case_value,
                        forecast_value = EXCLUDED.forecast_value
                """, (period, pipeline_value, weighted_pipeline, committed_value,
                      best_case, forecast_value))
                conn.commit()

                return {
                    'period': period,
                    'forecast_date': datetime.utcnow().date().isoformat(),
                    'pipeline_value': round(pipeline_value, 2),
                    'weighted_pipeline': round(weighted_pipeline, 2),
                    'committed_value': round(committed_value, 2),
                    'best_case_value': round(best_case, 2),
                    'forecast_value': round(forecast_value, 2),
                    'historical_win_rate': round(win_rate, 3),
                    'pipeline_by_stage': pipeline_by_stage,
                    'methodology': 'weighted_probability'
                }

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to forecast revenue: {e}")
            return {'error': str(e)}
        finally:
            conn.close()

    async def record_stage_transition(self, lead_id: str, from_stage: str,
                                       to_stage: str, reason: str = None,
                                       deal_value: float = None):
        """Record a stage transition for analytics"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Calculate days in previous stage
                cur.execute("""
                    SELECT transitioned_at FROM stage_transitions
                    WHERE lead_id = %s
                    ORDER BY transitioned_at DESC
                    LIMIT 1
                """, (lead_id,))
                last_transition = cur.fetchone()

                days_in_stage = 0
                if last_transition:
                    days_in_stage = (datetime.utcnow() - last_transition[0]).days

                cur.execute("""
                    INSERT INTO stage_transitions (
                        lead_id, from_stage, to_stage, transition_reason,
                        days_in_previous_stage, deal_value
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (lead_id, from_stage, to_stage, reason, days_in_stage, deal_value))

                conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record transition: {e}")
        finally:
            conn.close()

    async def record_win_loss(self, lead_id: str, outcome: str,
                              deal_value: float, primary_reason: str,
                              competitor: str = None, sales_cycle_days: int = None):
        """Record win/loss for analytics"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO win_loss_records (
                        lead_id, outcome, deal_value, primary_reason,
                        competitor_involved, sales_cycle_days
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (lead_id) DO UPDATE SET
                        outcome = EXCLUDED.outcome,
                        deal_value = EXCLUDED.deal_value,
                        primary_reason = EXCLUDED.primary_reason,
                        competitor_involved = EXCLUDED.competitor_involved,
                        sales_cycle_days = EXCLUDED.sales_cycle_days,
                        closed_at = NOW()
                """, (lead_id, outcome, deal_value, primary_reason,
                      competitor, sales_cycle_days))
                conn.commit()

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record win/loss: {e}")
        finally:
            conn.close()

    async def get_summary(self) -> Dict:
        """Get analytics summary"""
        funnel = await self.calculate_funnel_metrics()
        bottlenecks = await self.identify_bottlenecks()
        win_loss = await self.analyze_win_loss()
        forecast = await self.forecast_revenue()

        return {
            'funnel': {
                'overall_conversion': round(funnel.overall_conversion * 100, 2),
                'pipeline_value': funnel.total_pipeline_value,
                'won_value': funnel.total_won_value,
                'avg_sales_cycle': round(funnel.avg_sales_cycle_days, 1),
                'velocity_score': round(funnel.velocity_score, 1)
            },
            'stages': [{
                'stage': s.stage.value,
                'leads': s.total_leads,
                'conversion_rate': round(s.conversion_rate * 100, 1),
                'avg_days': round(s.avg_days_in_stage, 1),
                'value': round(s.total_value, 2)
            } for s in funnel.stages],
            'bottlenecks': bottlenecks[:3],  # Top 3
            'win_loss': {
                'win_rate': round(win_loss.win_rate * 100, 1),
                'avg_won_deal': round(win_loss.avg_won_deal_size, 2),
                'top_win_reason': win_loss.won_reasons[0] if win_loss.won_reasons else None,
                'top_loss_reason': win_loss.lost_reasons[0] if win_loss.lost_reasons else None
            },
            'forecast': {
                'period': forecast.get('period'),
                'forecast_value': forecast.get('forecast_value'),
                'pipeline_value': forecast.get('pipeline_value')
            },
            'timestamp': datetime.utcnow().isoformat()
        }


# Singleton instance
_analytics_engine = None

def get_analytics_engine() -> ConversionAnalyticsEngine:
    """Get or create analytics engine instance"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = ConversionAnalyticsEngine()
    return _analytics_engine
