#!/usr/bin/env python3
"""
Customer Success Agent - Proactive Customer Retention & Satisfaction
Predicts churn, triggers interventions, optimizes onboarding, ensures customer success
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 6543))
}


class ChurnRisk(Enum):
    """Churn risk levels"""
    CRITICAL = "critical"  # >80% chance of churning
    HIGH = "high"          # 60-80% chance
    MEDIUM = "medium"      # 40-60% chance
    LOW = "low"            # <40% chance


class HealthScore(Enum):
    """Customer health score categories"""
    THRIVING = "thriving"      # 90-100
    HEALTHY = "healthy"        # 70-89
    NEEDS_ATTENTION = "needs_attention"  # 50-69
    AT_RISK = "at_risk"        # 30-49
    CRITICAL = "critical"      # 0-29


class InterventionType(Enum):
    """Types of customer interventions"""
    ONBOARDING_CALL = "onboarding_call"
    TRAINING_SESSION = "training_session"
    FEATURE_DEMO = "feature_demo"
    CHECK_IN = "check_in"
    DISCOUNT_OFFER = "discount_offer"
    PERSONALIZED_CONTENT = "personalized_content"
    ESCALATION = "escalation"
    SUCCESS_PLAN = "success_plan"


@dataclass
class CustomerHealthMetrics:
    """Customer health metrics"""
    customer_id: str
    customer_name: str
    health_score: float  # 0-100
    health_category: HealthScore
    churn_probability: float  # 0-1
    churn_risk: ChurnRisk
    engagement_score: float  # 0-100
    satisfaction_score: float  # 0-100
    lifetime_value: float
    days_since_last_activity: int
    feature_adoption_rate: float  # 0-1
    support_tickets_last_30d: int
    measured_at: datetime


@dataclass
class ChurnPrediction:
    """Churn risk prediction"""
    id: str
    customer_id: str
    customer_name: str
    churn_probability: float
    churn_risk: ChurnRisk
    primary_risk_factors: List[str]
    predicted_churn_date: datetime
    recommended_interventions: List[str]
    estimated_revenue_at_risk: float
    confidence: float  # 0-1
    created_at: datetime


@dataclass
class CustomerIntervention:
    """Intervention for at-risk customers"""
    id: str
    customer_id: str
    intervention_type: InterventionType
    reason: str
    triggered_by: str  # churn_risk, low_engagement, support_issue, etc.
    scheduled_date: datetime
    completed: bool
    outcome: Optional[str]
    health_score_before: float
    health_score_after: Optional[float]
    created_at: datetime


class CustomerSuccessAgent:
    """Agent that ensures customer success and prevents churn"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.at_risk_customers = []
        self.interventions_triggered = []
        self._init_database()
        logger.info("✅ Customer Success Agent initialized")

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create customer health metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_customer_health (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id UUID NOT NULL,
                    health_score FLOAT NOT NULL CHECK (health_score >= 0 AND health_score <= 100),
                    health_category VARCHAR(50),
                    churn_probability FLOAT CHECK (churn_probability >= 0 AND churn_probability <= 1),
                    churn_risk VARCHAR(20),
                    engagement_score FLOAT,
                    satisfaction_score FLOAT,
                    lifetime_value FLOAT,
                    days_since_last_activity INTEGER,
                    feature_adoption_rate FLOAT,
                    support_tickets_last_30d INTEGER,
                    measured_at TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create churn predictions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_churn_predictions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id UUID NOT NULL,
                    churn_probability FLOAT NOT NULL,
                    churn_risk VARCHAR(20),
                    primary_risk_factors JSONB DEFAULT '[]'::jsonb,
                    predicted_churn_date TIMESTAMP,
                    recommended_interventions JSONB DEFAULT '[]'::jsonb,
                    estimated_revenue_at_risk FLOAT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create customer interventions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_customer_interventions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id UUID NOT NULL,
                    intervention_type VARCHAR(50) NOT NULL,
                    reason TEXT,
                    triggered_by VARCHAR(100),
                    scheduled_date TIMESTAMP,
                    completed BOOLEAN DEFAULT FALSE,
                    completed_at TIMESTAMP,
                    outcome TEXT,
                    health_score_before FLOAT,
                    health_score_after FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create customer engagement tracking
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_customer_engagement (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id UUID NOT NULL,
                    engagement_type VARCHAR(50),
                    engagement_value FLOAT,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_customer_health_score ON ai_customer_health(health_score, measured_at DESC);
                CREATE INDEX IF NOT EXISTS idx_churn_predictions_unresolved ON ai_churn_predictions(churn_risk, created_at DESC) WHERE resolved = FALSE;
                CREATE INDEX IF NOT EXISTS idx_interventions_pending ON ai_customer_interventions(customer_id, completed) WHERE completed = FALSE;
            """)

            conn.commit()
            logger.info("✅ Customer Success Agent database tables ready")

        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}. Operating without persistence")
        finally:
            if conn:
                conn.close()

    def calculate_customer_health(self) -> List[CustomerHealthMetrics]:
        """Calculate health scores for all customers"""
        health_metrics = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get customer activity metrics
            cur.execute("""
                SELECT
                    c.id as customer_id,
                    c.name as customer_name,
                    COUNT(DISTINCT j.id) as total_jobs,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '30 days') as jobs_last_30d,
                    COUNT(DISTINCT j.id) FILTER (WHERE j.created_at > NOW() - INTERVAL '90 days') as jobs_last_90d,
                    COALESCE(SUM(i.total_amount), 0) as total_revenue,
                    COALESCE(MAX(j.created_at), c.created_at) as last_activity,
                    EXTRACT(EPOCH FROM NOW() - COALESCE(MAX(j.created_at), c.created_at)) / 86400 as days_since_activity
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                LEFT JOIN invoices i ON i.job_id = j.id
                GROUP BY c.id, c.name, c.created_at
                LIMIT 100
            """)

            customers = cur.fetchall()

            for customer in customers:
                # Calculate engagement score (0-100)
                engagement_score = self._calculate_engagement_score(customer)

                # Calculate satisfaction score (0-100)
                satisfaction_score = self._calculate_satisfaction_score(customer)

                # Calculate feature adoption (0-1)
                feature_adoption = min(1.0, customer['total_jobs'] / 10.0)

                # Calculate overall health score (weighted average)
                health_score = (
                    engagement_score * 0.40 +
                    satisfaction_score * 0.30 +
                    (feature_adoption * 100) * 0.30
                )

                # Determine health category
                if health_score >= 90:
                    health_category = HealthScore.THRIVING
                elif health_score >= 70:
                    health_category = HealthScore.HEALTHY
                elif health_score >= 50:
                    health_category = HealthScore.NEEDS_ATTENTION
                elif health_score >= 30:
                    health_category = HealthScore.AT_RISK
                else:
                    health_category = HealthScore.CRITICAL

                # Calculate churn probability
                churn_probability = self._calculate_churn_probability(
                    health_score,
                    customer['days_since_activity'],
                    customer['jobs_last_30d']
                )

                # Determine churn risk
                if churn_probability >= 0.8:
                    churn_risk = ChurnRisk.CRITICAL
                elif churn_probability >= 0.6:
                    churn_risk = ChurnRisk.HIGH
                elif churn_probability >= 0.4:
                    churn_risk = ChurnRisk.MEDIUM
                else:
                    churn_risk = ChurnRisk.LOW

                health_metrics.append({
                    'customer_id': customer['customer_id'],
                    'customer_name': customer['customer_name'],
                    'health_score': health_score,
                    'health_category': health_category.value,
                    'churn_probability': churn_probability,
                    'churn_risk': churn_risk.value,
                    'engagement_score': engagement_score,
                    'satisfaction_score': satisfaction_score,
                    'lifetime_value': float(customer['total_revenue']),
                    'days_since_last_activity': int(customer['days_since_activity']),
                    'feature_adoption_rate': feature_adoption,
                    'support_tickets_last_30d': 0  # Would query support tickets table
                })

            conn.close()

            # Persist health metrics
            if health_metrics:
                self._persist_health_metrics(health_metrics)

        except Exception as e:
            logger.warning(f"Customer health calculation failed: {e}")

        return health_metrics

    def _calculate_engagement_score(self, customer: Dict) -> float:
        """Calculate customer engagement score (0-100)"""
        # Based on activity recency and frequency
        jobs_last_30d = customer.get('jobs_last_30d', 0)
        jobs_last_90d = customer.get('jobs_last_90d', 0)
        days_since_activity = customer.get('days_since_activity', 999)

        # Recency score (0-50 points)
        if days_since_activity < 7:
            recency_score = 50
        elif days_since_activity < 30:
            recency_score = 40
        elif days_since_activity < 60:
            recency_score = 25
        elif days_since_activity < 90:
            recency_score = 10
        else:
            recency_score = 0

        # Frequency score (0-50 points)
        frequency_score = min(50, jobs_last_30d * 10)

        return recency_score + frequency_score

    def _calculate_satisfaction_score(self, customer: Dict) -> float:
        """Calculate customer satisfaction score (0-100)"""
        # In production, would use NPS, CSAT, support ticket sentiment
        # For now, use job completion rate as proxy
        total_jobs = customer.get('total_jobs', 0)
        if total_jobs == 0:
            return 50.0  # Neutral for new customers

        # Assume 80% completion rate = 80 satisfaction score
        return min(100.0, total_jobs * 2.0)

    def _calculate_churn_probability(self, health_score: float, days_since_activity: int, recent_jobs: int) -> float:
        """Calculate probability of customer churning (0-1)"""
        # Simple model - in production would use ML
        base_churn = (100 - health_score) / 100

        # Increase churn probability based on inactivity
        if days_since_activity > 90:
            inactivity_factor = 0.4
        elif days_since_activity > 60:
            inactivity_factor = 0.3
        elif days_since_activity > 30:
            inactivity_factor = 0.2
        else:
            inactivity_factor = 0.0

        # Decrease churn probability if customer is active
        activity_factor = -0.1 if recent_jobs > 2 else 0

        churn_prob = base_churn + inactivity_factor + activity_factor
        return max(0.0, min(1.0, churn_prob))

    def predict_churn(self, health_metrics: List[Dict]) -> List[ChurnPrediction]:
        """Generate churn predictions for at-risk customers"""
        predictions = []

        for metrics in health_metrics:
            if metrics['churn_risk'] in ['high', 'critical']:
                # Identify risk factors
                risk_factors = []

                if metrics['days_since_last_activity'] > 60:
                    risk_factors.append(f"Inactive for {metrics['days_since_last_activity']} days")

                if metrics['health_score'] < 50:
                    risk_factors.append(f"Low health score ({metrics['health_score']:.0f}/100)")

                if metrics['feature_adoption_rate'] < 0.3:
                    risk_factors.append(f"Low feature adoption ({metrics['feature_adoption_rate']*100:.0f}%)")

                # Recommend interventions
                interventions = []
                if metrics['churn_risk'] == 'critical':
                    interventions.extend([
                        InterventionType.ESCALATION.value,
                        InterventionType.DISCOUNT_OFFER.value,
                        InterventionType.CHECK_IN.value
                    ])
                else:
                    interventions.extend([
                        InterventionType.CHECK_IN.value,
                        InterventionType.TRAINING_SESSION.value,
                        InterventionType.PERSONALIZED_CONTENT.value
                    ])

                # Predict churn date (estimate)
                days_to_churn = int((1 - metrics['churn_probability']) * 90)
                predicted_date = datetime.now() + timedelta(days=days_to_churn)

                predictions.append({
                    'customer_id': metrics['customer_id'],
                    'customer_name': metrics['customer_name'],
                    'churn_probability': metrics['churn_probability'],
                    'churn_risk': metrics['churn_risk'],
                    'primary_risk_factors': risk_factors,
                    'predicted_churn_date': predicted_date,
                    'recommended_interventions': interventions,
                    'estimated_revenue_at_risk': metrics['lifetime_value'] * 0.3,  # 30% of LTV
                    'confidence': 0.75
                })

        # Persist predictions
        if predictions:
            self._persist_churn_predictions(predictions)

        return predictions

    def trigger_interventions(self, predictions: List[Dict]) -> List[CustomerIntervention]:
        """Trigger interventions for at-risk customers"""
        interventions = []

        for pred in predictions:
            for intervention_type in pred['recommended_interventions'][:2]:  # Top 2 interventions
                interventions.append({
                    'customer_id': pred['customer_id'],
                    'intervention_type': intervention_type,
                    'reason': f"Churn risk: {pred['churn_risk']} ({pred['churn_probability']*100:.0f}% probability)",
                    'triggered_by': 'churn_prediction',
                    'scheduled_date': datetime.now() + timedelta(hours=24),
                    'health_score_before': pred.get('health_score_before', 0.0)
                })

        # Persist interventions
        if interventions:
            self._persist_interventions(interventions)

        return interventions

    def _persist_health_metrics(self, metrics: List[Dict]):
        """Persist health metrics to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for metric in metrics:
                cur.execute("""
                    INSERT INTO ai_customer_health
                    (customer_id, health_score, health_category, churn_probability,
                     churn_risk, engagement_score, satisfaction_score, lifetime_value,
                     days_since_last_activity, feature_adoption_rate, support_tickets_last_30d)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metric['customer_id'],
                    metric['health_score'],
                    metric['health_category'],
                    metric['churn_probability'],
                    metric['churn_risk'],
                    metric['engagement_score'],
                    metric['satisfaction_score'],
                    metric['lifetime_value'],
                    metric['days_since_last_activity'],
                    metric['feature_adoption_rate'],
                    metric['support_tickets_last_30d']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(metrics)} customer health metrics")

        except Exception as e:
            logger.warning(f"Failed to persist health metrics: {e}")

    def _persist_churn_predictions(self, predictions: List[Dict]):
        """Persist churn predictions to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for pred in predictions:
                cur.execute("""
                    INSERT INTO ai_churn_predictions
                    (customer_id, churn_probability, churn_risk,
                     primary_risk_factors, predicted_churn_date, recommended_interventions,
                     estimated_revenue_at_risk, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    pred['customer_id'],
                    pred['churn_probability'],
                    pred['churn_risk'],
                    Json(pred['primary_risk_factors']),
                    pred['predicted_churn_date'],
                    Json(pred['recommended_interventions']),
                    pred['estimated_revenue_at_risk'],
                    pred['confidence']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(predictions)} churn predictions")

        except Exception as e:
            logger.warning(f"Failed to persist churn predictions: {e}")

    def _persist_interventions(self, interventions: List[Dict]):
        """Persist interventions to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for intervention in interventions:
                cur.execute("""
                    INSERT INTO ai_customer_interventions
                    (customer_id, intervention_type, reason, triggered_by, scheduled_date, health_score_before)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    intervention['customer_id'],
                    intervention['intervention_type'],
                    intervention['reason'],
                    intervention['triggered_by'],
                    intervention['scheduled_date'],
                    intervention['health_score_before']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(interventions)} customer interventions")

        except Exception as e:
            logger.warning(f"Failed to persist interventions: {e}")

    async def continuous_success_loop(self, interval_hours: int = 4):
        """Main loop that continuously ensures customer success"""
        logger.info(f"🔄 Starting customer success loop (every {interval_hours}h)")

        while True:
            try:
                logger.info("🔍 Analyzing customer health...")

                # Calculate customer health
                health_metrics = self.calculate_customer_health()
                logger.info(f"📊 Analyzed {len(health_metrics)} customers")

                # Count by health category
                critical = sum(1 for m in health_metrics if m['health_category'] == 'critical')
                at_risk = sum(1 for m in health_metrics if m['health_category'] == 'at_risk')

                if critical > 0 or at_risk > 0:
                    logger.warning(f"⚠️ {critical} critical, {at_risk} at-risk customers")

                # Predict churn
                predictions = self.predict_churn(health_metrics)
                if predictions:
                    logger.warning(f"🚨 {len(predictions)} customers at risk of churning")

                    # Trigger interventions
                    interventions = self.trigger_interventions(predictions)
                    logger.info(f"💡 Triggered {len(interventions)} customer interventions")

            except Exception as e:
                logger.error(f"❌ Customer success loop error: {e}")

            # Wait before next analysis
            await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    agent = CustomerSuccessAgent()
    asyncio.run(agent.continuous_success_loop(interval_hours=4))
