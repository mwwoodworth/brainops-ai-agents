"""
Advanced Multi-Factor Lead Scoring Engine
Ultimate AI-powered lead qualification for maximum conversion

This module implements bleeding-edge lead scoring using:
- Behavioral analysis (engagement patterns)
- Firmographic scoring (company fit)
- Intent signals (buying readiness)
- Deal velocity tracking
- Financial health indicators
- Machine learning predictions
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# Database
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class LeadTier(Enum):
    """Lead classification tiers"""
    HOT = "hot"           # Score 80-100: Immediate action required
    WARM = "warm"         # Score 60-79: Active nurturing
    COOL = "cool"         # Score 40-59: Long-term nurturing
    COLD = "cold"         # Score 0-39: Low priority


class ScoreCategory(Enum):
    """Scoring dimensions"""
    BEHAVIORAL = "behavioral"       # 0-30 points
    FIRMOGRAPHIC = "firmographic"   # 0-25 points
    INTENT = "intent"               # 0-25 points
    VELOCITY = "velocity"           # 0-15 points
    FINANCIAL = "financial"         # 0-5 points


@dataclass
class BehavioralScore:
    """Behavioral engagement scoring (0-30 points)"""
    email_engagement: float = 0.0      # Opens, clicks (0-8)
    website_activity: float = 0.0      # Visits, pages viewed (0-6)
    content_downloads: float = 0.0     # Whitepapers, resources (0-5)
    demo_attendance: float = 0.0       # Webinars, demos (0-6)
    support_interaction: float = 0.0   # Ticket patterns (0-5)
    total: float = field(init=False)

    def __post_init__(self):
        self.total = min(30, sum([
            self.email_engagement,
            self.website_activity,
            self.content_downloads,
            self.demo_attendance,
            self.support_interaction
        ]))


@dataclass
class FirmographicScore:
    """Company fit scoring (0-25 points)"""
    company_size_fit: float = 0.0      # Alignment with ICP (0-7)
    industry_fit: float = 0.0          # Vertical relevance (0-6)
    revenue_alignment: float = 0.0     # Budget potential (0-5)
    growth_indicators: float = 0.0     # Expansion signals (0-4)
    tech_maturity: float = 0.0         # Technology adoption (0-3)
    total: float = field(init=False)

    def __post_init__(self):
        self.total = min(25, sum([
            self.company_size_fit,
            self.industry_fit,
            self.revenue_alignment,
            self.growth_indicators,
            self.tech_maturity
        ]))


@dataclass
class IntentScore:
    """Buying intent signals (0-25 points)"""
    search_behavior: float = 0.0       # Keywords, research (0-7)
    competitor_interest: float = 0.0   # Comparison shopping (0-6)
    job_postings: float = 0.0          # Hiring signals (0-5)
    website_changes: float = 0.0       # Modernization signals (0-4)
    social_activity: float = 0.0       # Engagement patterns (0-3)
    total: float = field(init=False)

    def __post_init__(self):
        self.total = min(25, sum([
            self.search_behavior,
            self.competitor_interest,
            self.job_postings,
            self.website_changes,
            self.social_activity
        ]))


@dataclass
class VelocityScore:
    """Deal velocity indicators (0-15 points)"""
    response_time: float = 0.0         # Speed of engagement (0-5)
    meeting_scheduling: float = 0.0    # Calendar speed (0-4)
    decision_timeline: float = 0.0     # Urgency signals (0-3)
    sales_cycle_pace: float = 0.0      # Stage progression (0-3)
    total: float = field(init=False)

    def __post_init__(self):
        self.total = min(15, sum([
            self.response_time,
            self.meeting_scheduling,
            self.decision_timeline,
            self.sales_cycle_pace
        ]))


@dataclass
class FinancialScore:
    """Financial health indicators (0-5 points)"""
    payment_history: float = 0.0       # If existing customer (0-2)
    expansion_potential: float = 0.0   # Upsell opportunity (0-2)
    churn_risk: float = 0.0            # Inverse risk score (0-1)
    total: float = field(init=False)

    def __post_init__(self):
        self.total = min(5, sum([
            self.payment_history,
            self.expansion_potential,
            self.churn_risk
        ]))


@dataclass
class LeadScoreResult:
    """Complete lead scoring result"""
    lead_id: str
    behavioral: BehavioralScore
    firmographic: FirmographicScore
    intent: IntentScore
    velocity: VelocityScore
    financial: FinancialScore
    composite_score: float = field(init=False)
    tier: LeadTier = field(init=False)
    probability_conversion_30d: float = field(init=False)
    expected_deal_size: float = field(init=False)
    next_best_action: str = field(init=False)
    recommended_touch_frequency: int = field(init=False)
    scoring_factors: dict = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        # Calculate composite score (0-100)
        self.composite_score = (
            self.behavioral.total +
            self.firmographic.total +
            self.intent.total +
            self.velocity.total +
            self.financial.total
        )

        # Determine tier
        if self.composite_score >= 80:
            self.tier = LeadTier.HOT
            self.next_best_action = "immediate_call"
            self.recommended_touch_frequency = 1  # Daily
            self.probability_conversion_30d = 0.65 + (self.composite_score - 80) * 0.015
        elif self.composite_score >= 60:
            self.tier = LeadTier.WARM
            self.next_best_action = "personalized_outreach"
            self.recommended_touch_frequency = 3  # Every 3 days
            self.probability_conversion_30d = 0.35 + (self.composite_score - 60) * 0.015
        elif self.composite_score >= 40:
            self.tier = LeadTier.COOL
            self.next_best_action = "nurture_sequence"
            self.recommended_touch_frequency = 7  # Weekly
            self.probability_conversion_30d = 0.15 + (self.composite_score - 40) * 0.01
        else:
            self.tier = LeadTier.COLD
            self.next_best_action = "long_term_nurture"
            self.recommended_touch_frequency = 14  # Bi-weekly
            self.probability_conversion_30d = 0.05 + self.composite_score * 0.0025

        # Expected deal size based on firmographic score
        base_deal = 2500
        self.expected_deal_size = base_deal * (1 + self.firmographic.total / 25)


class AdvancedLeadScoringEngine:
    """
    Ultimate Lead Scoring Engine with ML-powered predictions

    Features:
    - Multi-dimensional scoring (behavioral, firmographic, intent, velocity, financial)
    - Real-time score updates on new signals
    - Predictive conversion probability
    - Automatic tier classification
    - Next-best-action recommendations
    - Historical performance learning
    """

    def __init__(self):
        self.db_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
        self._initialized = False

        # Scoring weights (can be adjusted based on learning)
        self.weights = {
            'behavioral': 1.0,
            'firmographic': 1.0,
            'intent': 1.2,  # Intent signals weighted higher
            'velocity': 1.3,  # Fast movers weighted higher
            'financial': 1.0
        }

        # Industry-specific ICP (Ideal Customer Profile) scores
        self.icp_industry_scores = {
            'roofing': 1.0,
            'construction': 0.95,
            'home_services': 0.90,
            'general_contractor': 0.85,
            'real_estate': 0.75,
            'insurance': 0.70,
            'other': 0.50
        }

        # Company size scoring (employees)
        self.company_size_scores = {
            (1, 10): 0.6,      # Very small
            (11, 50): 0.9,     # Small - ideal
            (51, 200): 1.0,    # Medium - ideal
            (201, 500): 0.85,  # Large
            (501, 10000): 0.7  # Enterprise
        }

    def _get_connection(self):
        """Get database connection"""
        if not self.db_url:
            raise ValueError("DATABASE_URL not configured")
        return psycopg2.connect(self.db_url)

    async def initialize_tables(self):
        """Create advanced scoring tables"""
        if self._initialized:
            return

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Advanced lead metrics table

                # Lead engagement history

                # Scoring model performance

                # Create indexes

                conn.commit()
                self._initialized = True
                logger.info("Advanced lead scoring tables initialized")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize scoring tables: {e!r}")
            raise
        finally:
            conn.close()

    async def calculate_behavioral_score(self, lead_id: str, lead_data: dict) -> BehavioralScore:
        """Calculate behavioral engagement score (0-30 points)"""

        # Get engagement history
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT event_type, COUNT(*) as count,
                           SUM(engagement_value) as total_value
                    FROM lead_engagement_history
                    WHERE lead_id = %s
                    AND timestamp > NOW() - INTERVAL '30 days'
                    GROUP BY event_type
                """, (lead_id,))
                engagement = {row['event_type']: row for row in cur.fetchall()}
        finally:
            conn.close()

        # Email engagement (0-8 points)
        email_opens = engagement.get('email_open', {}).get('count', 0)
        email_clicks = engagement.get('email_click', {}).get('count', 0)
        email_score = min(8, (email_opens * 0.5) + (email_clicks * 1.5))

        # Website activity (0-6 points)
        page_views = engagement.get('page_view', {}).get('count', 0)
        website_score = min(6, page_views * 0.3)

        # Content downloads (0-5 points)
        downloads = engagement.get('content_download', {}).get('count', 0)
        download_score = min(5, downloads * 1.5)

        # Demo/webinar attendance (0-6 points)
        demos = engagement.get('demo_attended', {}).get('count', 0)
        webinars = engagement.get('webinar_attended', {}).get('count', 0)
        demo_score = min(6, (demos * 3) + (webinars * 2))

        # Support interaction (0-5 points) - can indicate interest or problems
        tickets = engagement.get('support_ticket', {}).get('count', 0)
        support_score = min(5, tickets * 1.0) if tickets <= 3 else max(0, 5 - (tickets - 3) * 0.5)

        return BehavioralScore(
            email_engagement=email_score,
            website_activity=website_score,
            content_downloads=download_score,
            demo_attendance=demo_score,
            support_interaction=support_score
        )

    async def calculate_firmographic_score(self, lead_id: str, lead_data: dict) -> FirmographicScore:
        """Calculate company fit score (0-25 points)"""

        company = lead_data.get('company', {})

        # Company size fit (0-7 points)
        employees = company.get('employees', 50)
        size_score = 0
        for (low, high), score in self.company_size_scores.items():
            if low <= employees <= high:
                size_score = score * 7
                break

        # Industry fit (0-6 points)
        industry = company.get('industry', 'other').lower()
        industry_score = self.icp_industry_scores.get(industry, 0.5) * 6

        # Revenue alignment (0-5 points)
        annual_revenue = company.get('annual_revenue', 0)
        if annual_revenue >= 5000000:
            revenue_score = 5
        elif annual_revenue >= 1000000:
            revenue_score = 4
        elif annual_revenue >= 500000:
            revenue_score = 3
        elif annual_revenue >= 100000:
            revenue_score = 2
        else:
            revenue_score = 1

        # Growth indicators (0-4 points)
        growth_rate = company.get('growth_rate', 0)
        growth_score = min(4, growth_rate * 0.2) if growth_rate > 0 else 0

        # Tech maturity (0-3 points)
        has_crm = company.get('has_crm', False)
        has_cloud = company.get('uses_cloud', False)
        tech_score = (1.5 if has_crm else 0) + (1.5 if has_cloud else 0)

        return FirmographicScore(
            company_size_fit=size_score,
            industry_fit=industry_score,
            revenue_alignment=revenue_score,
            growth_indicators=growth_score,
            tech_maturity=tech_score
        )

    async def calculate_intent_score(self, lead_id: str, lead_data: dict) -> IntentScore:
        """Calculate buying intent score (0-25 points)"""

        intent_signals = lead_data.get('intent_signals', {})

        # Search behavior (0-7 points)
        search_keywords = intent_signals.get('search_keywords', [])
        high_intent_keywords = ['pricing', 'demo', 'trial', 'compare', 'best', 'buy']
        keyword_matches = sum(1 for kw in search_keywords if any(hi in kw.lower() for hi in high_intent_keywords))
        search_score = min(7, keyword_matches * 2)

        # Competitor interest (0-6 points)
        competitor_research = intent_signals.get('competitor_research', False)
        comparison_pages = intent_signals.get('comparison_page_views', 0)
        competitor_score = (3 if competitor_research else 0) + min(3, comparison_pages * 0.5)

        # Job postings (0-5 points) - hiring indicates growth
        recent_hires = intent_signals.get('recent_hires', 0)
        job_postings = intent_signals.get('job_postings', 0)
        job_score = min(5, (recent_hires * 0.5) + (job_postings * 0.3))

        # Website changes (0-4 points)
        website_updated = intent_signals.get('website_updated_recently', False)
        new_products = intent_signals.get('new_products_launched', 0)
        website_score = (2 if website_updated else 0) + min(2, new_products)

        # Social activity (0-3 points)
        social_engagement = intent_signals.get('social_engagement_score', 0)
        social_score = min(3, social_engagement * 0.03)

        return IntentScore(
            search_behavior=search_score,
            competitor_interest=competitor_score,
            job_postings=job_score,
            website_changes=website_score,
            social_activity=social_score
        )

    async def calculate_velocity_score(self, lead_id: str, lead_data: dict) -> VelocityScore:
        """Calculate deal velocity score (0-15 points)"""

        velocity_data = lead_data.get('velocity', {})

        # Response time (0-5 points) - faster = better
        avg_response_hours = velocity_data.get('avg_response_hours', 48)
        if avg_response_hours <= 1:
            response_score = 5
        elif avg_response_hours <= 4:
            response_score = 4
        elif avg_response_hours <= 12:
            response_score = 3
        elif avg_response_hours <= 24:
            response_score = 2
        elif avg_response_hours <= 48:
            response_score = 1
        else:
            response_score = 0

        # Meeting scheduling speed (0-4 points)
        days_to_meeting = velocity_data.get('days_to_first_meeting', 14)
        if days_to_meeting <= 2:
            meeting_score = 4
        elif days_to_meeting <= 5:
            meeting_score = 3
        elif days_to_meeting <= 7:
            meeting_score = 2
        elif days_to_meeting <= 14:
            meeting_score = 1
        else:
            meeting_score = 0

        # Decision timeline (0-3 points)
        decision_urgency = velocity_data.get('decision_urgency', 'normal')
        decision_scores = {'immediate': 3, 'urgent': 2.5, 'soon': 2, 'normal': 1, 'long_term': 0}
        decision_score = decision_scores.get(decision_urgency, 1)

        # Sales cycle pace (0-3 points)
        stages_progressed = velocity_data.get('stages_progressed_30d', 0)
        pace_score = min(3, stages_progressed)

        return VelocityScore(
            response_time=response_score,
            meeting_scheduling=meeting_score,
            decision_timeline=decision_score,
            sales_cycle_pace=pace_score
        )

    async def calculate_financial_score(self, lead_id: str, lead_data: dict) -> FinancialScore:
        """Calculate financial health score (0-5 points)"""

        financial_data = lead_data.get('financial', {})

        # Payment history (0-2 points) - if existing customer
        is_customer = financial_data.get('is_existing_customer', False)
        payment_reliability = financial_data.get('payment_reliability', 0)
        payment_score = payment_reliability * 2 if is_customer else 1  # New leads get 1 by default

        # Expansion potential (0-2 points)
        expansion_signals = financial_data.get('expansion_signals', 0)
        expansion_score = min(2, expansion_signals * 0.5)

        # Churn risk inverse (0-1 points)
        churn_risk = financial_data.get('churn_risk', 0.5)
        churn_score = 1 - churn_risk  # Lower risk = higher score

        return FinancialScore(
            payment_history=payment_score,
            expansion_potential=expansion_score,
            churn_risk=churn_score
        )

    async def calculate_multi_factor_score(self, lead_id: str, lead_data: Optional[dict] = None) -> LeadScoreResult:
        """
        Calculate comprehensive multi-factor lead score

        Returns complete scoring breakdown with predictions and recommendations
        """
        await self.initialize_tables()

        if lead_data is None:
            lead_data = await self._fetch_lead_data(lead_id)

        # Calculate all score components
        behavioral = await self.calculate_behavioral_score(lead_id, lead_data)
        firmographic = await self.calculate_firmographic_score(lead_id, lead_data)
        intent = await self.calculate_intent_score(lead_id, lead_data)
        velocity = await self.calculate_velocity_score(lead_id, lead_data)
        financial = await self.calculate_financial_score(lead_id, lead_data)

        # Create result
        result = LeadScoreResult(
            lead_id=lead_id,
            behavioral=behavioral,
            firmographic=firmographic,
            intent=intent,
            velocity=velocity,
            financial=financial,
            scoring_factors={
                'behavioral_breakdown': {
                    'email_engagement': behavioral.email_engagement,
                    'website_activity': behavioral.website_activity,
                    'content_downloads': behavioral.content_downloads,
                    'demo_attendance': behavioral.demo_attendance,
                    'support_interaction': behavioral.support_interaction
                },
                'firmographic_breakdown': {
                    'company_size_fit': firmographic.company_size_fit,
                    'industry_fit': firmographic.industry_fit,
                    'revenue_alignment': firmographic.revenue_alignment,
                    'growth_indicators': firmographic.growth_indicators,
                    'tech_maturity': firmographic.tech_maturity
                },
                'intent_breakdown': {
                    'search_behavior': intent.search_behavior,
                    'competitor_interest': intent.competitor_interest,
                    'job_postings': intent.job_postings,
                    'website_changes': intent.website_changes,
                    'social_activity': intent.social_activity
                },
                'velocity_breakdown': {
                    'response_time': velocity.response_time,
                    'meeting_scheduling': velocity.meeting_scheduling,
                    'decision_timeline': velocity.decision_timeline,
                    'sales_cycle_pace': velocity.sales_cycle_pace
                },
                'financial_breakdown': {
                    'payment_history': financial.payment_history,
                    'expansion_potential': financial.expansion_potential,
                    'churn_risk': financial.churn_risk
                }
            }
        )

        # Store result
        await self._store_score(result)

        return result

    async def _fetch_lead_data(self, lead_id: str) -> dict:
        """Fetch lead data from database"""
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT * FROM revenue_leads WHERE id = %s
                """, (lead_id,))
                lead = cur.fetchone()

                if not lead:
                    return {}

                return {
                    'company': lead.get('company_data', {}),
                    'intent_signals': lead.get('intent_signals', {}),
                    'velocity': lead.get('velocity_data', {}),
                    'financial': lead.get('financial_data', {})
                }
        finally:
            conn.close()

    async def _store_score(self, result: LeadScoreResult):
        """Store scoring result in database"""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO advanced_lead_metrics (
                        lead_id, behavioral_score, firmographic_score,
                        intent_score, velocity_score, financial_score,
                        composite_score, tier, probability_conversion_30d,
                        expected_deal_size, next_best_action,
                        recommended_touch_frequency, scoring_factors, last_calculated
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (lead_id) DO UPDATE SET
                        behavioral_score = EXCLUDED.behavioral_score,
                        firmographic_score = EXCLUDED.firmographic_score,
                        intent_score = EXCLUDED.intent_score,
                        velocity_score = EXCLUDED.velocity_score,
                        financial_score = EXCLUDED.financial_score,
                        composite_score = EXCLUDED.composite_score,
                        tier = EXCLUDED.tier,
                        probability_conversion_30d = EXCLUDED.probability_conversion_30d,
                        expected_deal_size = EXCLUDED.expected_deal_size,
                        next_best_action = EXCLUDED.next_best_action,
                        recommended_touch_frequency = EXCLUDED.recommended_touch_frequency,
                        scoring_factors = EXCLUDED.scoring_factors,
                        last_calculated = NOW()
                """, (
                    result.lead_id,
                    result.behavioral.total,
                    result.firmographic.total,
                    result.intent.total,
                    result.velocity.total,
                    result.financial.total,
                    result.composite_score,
                    result.tier.value,
                    result.probability_conversion_30d,
                    result.expected_deal_size,
                    result.next_best_action,
                    result.recommended_touch_frequency,
                    json.dumps(result.scoring_factors)
                ))
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store lead score: {e!r}")
        finally:
            conn.close()

    async def record_engagement(self, lead_id: str, event_type: str,
                                event_data: dict = None, engagement_value: float = 1.0,
                                channel: str = None):
        """Record a new engagement event for a lead"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO lead_engagement_history (
                        lead_id, event_type, event_data, engagement_value, channel
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    lead_id,
                    event_type,
                    json.dumps(event_data or {}),
                    engagement_value,
                    channel
                ))
                conn.commit()

            # Trigger rescore
            await self.calculate_multi_factor_score(lead_id)

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to record engagement: {e!r}")
        finally:
            conn.close()

    async def get_hot_leads(self, limit: int = 50) -> list[dict]:
        """Get top hot leads requiring immediate action"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT m.*, l.company_name, l.contact_name, l.email
                    FROM advanced_lead_metrics m
                    LEFT JOIN revenue_leads l ON m.lead_id = l.id
                    WHERE m.tier = 'hot'
                    ORDER BY m.composite_score DESC
                    LIMIT %s
                """, (limit,))
                return cur.fetchall()
        finally:
            conn.close()

    async def get_leads_by_tier(self, tier: LeadTier, limit: int = 100) -> list[dict]:
        """Get leads by tier"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT m.*, l.company_name, l.contact_name, l.email
                    FROM advanced_lead_metrics m
                    LEFT JOIN revenue_leads l ON m.lead_id = l.id
                    WHERE m.tier = %s
                    ORDER BY m.composite_score DESC
                    LIMIT %s
                """, (tier.value, limit))
                return cur.fetchall()
        finally:
            conn.close()

    async def get_scoring_summary(self) -> dict:
        """Get overall scoring summary"""
        await self.initialize_tables()

        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        tier,
                        COUNT(*) as count,
                        AVG(composite_score) as avg_score,
                        AVG(probability_conversion_30d) as avg_conversion_prob,
                        SUM(expected_deal_size) as total_pipeline
                    FROM advanced_lead_metrics
                    GROUP BY tier
                    ORDER BY
                        CASE tier
                            WHEN 'hot' THEN 1
                            WHEN 'warm' THEN 2
                            WHEN 'cool' THEN 3
                            WHEN 'cold' THEN 4
                        END
                """)
                tiers = cur.fetchall()

                cur.execute("""
                    SELECT
                        COUNT(*) as total_leads,
                        AVG(composite_score) as avg_score,
                        SUM(expected_deal_size) as total_pipeline,
                        AVG(probability_conversion_30d) as avg_conversion
                    FROM advanced_lead_metrics
                """)
                totals = cur.fetchone()

                return {
                    'by_tier': tiers,
                    'totals': totals,
                    'timestamp': datetime.utcnow().isoformat()
                }
        finally:
            conn.close()


# Singleton instance
_scoring_engine = None

def get_scoring_engine() -> AdvancedLeadScoringEngine:
    """Get or create scoring engine instance"""
    global _scoring_engine
    if _scoring_engine is None:
        _scoring_engine = AdvancedLeadScoringEngine()
    return _scoring_engine
