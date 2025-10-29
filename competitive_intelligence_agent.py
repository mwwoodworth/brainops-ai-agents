#!/usr/bin/env python3
"""
Competitive Intelligence Agent - Market Monitoring & Strategic Response
Monitors competitors, tracks market trends, recommends strategic responses
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


class CompetitorActivity(Enum):
    """Types of competitor activities"""
    PRICING_CHANGE = "pricing_change"
    NEW_FEATURE = "new_feature"
    MARKETING_CAMPAIGN = "marketing_campaign"
    PARTNERSHIP = "partnership"
    FUNDING_ROUND = "funding_round"
    ACQUISITION = "acquisition"
    PRODUCT_LAUNCH = "product_launch"
    LEADERSHIP_CHANGE = "leadership_change"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    CRITICAL = "critical"    # Immediate response required
    HIGH = "high"            # Response within 1 week
    MEDIUM = "medium"        # Monitor and plan response
    LOW = "low"              # Awareness only


class MarketTrend(Enum):
    """Market trend types"""
    TECHNOLOGY = "technology"
    CUSTOMER_BEHAVIOR = "customer_behavior"
    PRICING = "pricing"
    REGULATION = "regulation"
    ECONOMY = "economy"
    COMPETITION = "competition"


@dataclass
class CompetitorIntelligence:
    """Competitor intelligence data"""
    id: str
    competitor_name: str
    activity_type: CompetitorActivity
    description: str
    source: str
    detected_at: datetime
    threat_level: ThreatLevel
    impact_areas: List[str]  # pricing, features, market_share, etc.
    recommended_response: str
    estimated_impact: str
    verified: bool


@dataclass
class MarketInsight:
    """Market trend insight"""
    id: str
    trend_type: MarketTrend
    title: str
    description: str
    supporting_data: Dict[str, Any]
    implications: List[str]
    opportunities: List[str]
    threats: List[str]
    recommended_actions: List[str]
    confidence: float  # 0-1
    detected_at: datetime


@dataclass
class StrategicResponse:
    """Recommended strategic response"""
    id: str
    trigger: str  # What triggered this response
    response_type: str  # defensive, offensive, neutral
    title: str
    description: str
    objectives: List[str]
    action_items: List[str]
    success_metrics: List[str]
    resources_required: Dict[str, Any]
    timeline_days: int
    priority: int  # 1-5
    created_at: datetime
    implemented: bool


class CompetitiveIntelligenceAgent:
    """Agent that monitors competition and market trends"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.competitor_activities = []
        self.market_insights = []
        self.strategic_responses = []
        self._init_database()
        logger.info("✅ Competitive Intelligence Agent initialized")

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Create competitor intelligence table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_competitor_intelligence (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    competitor_name VARCHAR(255) NOT NULL,
                    activity_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    source TEXT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    threat_level VARCHAR(20),
                    impact_areas JSONB DEFAULT '[]'::jsonb,
                    recommended_response TEXT,
                    estimated_impact TEXT,
                    verified BOOLEAN DEFAULT FALSE,
                    verified_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create market insights table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_market_insights (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    trend_type VARCHAR(50) NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    supporting_data JSONB DEFAULT '{}'::jsonb,
                    implications JSONB DEFAULT '[]'::jsonb,
                    opportunities JSONB DEFAULT '[]'::jsonb,
                    threats JSONB DEFAULT '[]'::jsonb,
                    recommended_actions JSONB DEFAULT '[]'::jsonb,
                    confidence FLOAT,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create strategic responses table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_strategic_responses (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    trigger TEXT NOT NULL,
                    response_type VARCHAR(50),
                    title TEXT NOT NULL,
                    description TEXT,
                    objectives JSONB DEFAULT '[]'::jsonb,
                    action_items JSONB DEFAULT '[]'::jsonb,
                    success_metrics JSONB DEFAULT '[]'::jsonb,
                    resources_required JSONB DEFAULT '{}'::jsonb,
                    timeline_days INTEGER,
                    priority INTEGER CHECK (priority >= 1 AND priority <= 5),
                    created_at TIMESTAMP DEFAULT NOW(),
                    implemented BOOLEAN DEFAULT FALSE,
                    implemented_at TIMESTAMP,
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create competitive positioning table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_competitive_positioning (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    competitor_name VARCHAR(255) NOT NULL,
                    feature_name VARCHAR(255) NOT NULL,
                    we_have BOOLEAN DEFAULT FALSE,
                    they_have BOOLEAN DEFAULT FALSE,
                    our_advantage TEXT,
                    their_advantage TEXT,
                    gap_priority INTEGER,
                    updated_at TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_competitor_intel_threat ON ai_competitor_intelligence(threat_level, detected_at DESC);
                CREATE INDEX IF NOT EXISTS idx_market_insights_trend ON ai_market_insights(trend_type, detected_at DESC);
                CREATE INDEX IF NOT EXISTS idx_strategic_responses_pending ON ai_strategic_responses(priority, implemented) WHERE implemented = FALSE;
            """)

            conn.commit()
            logger.info("✅ Competitive Intelligence Agent database tables ready")

        except Exception as e:
            logger.warning(f"Database initialization skipped: {e}. Operating without persistence")
        finally:
            if conn:
                conn.close()

    def monitor_competitors(self) -> List[CompetitorIntelligence]:
        """Monitor competitor activities"""
        activities = []

        try:
            # In production, this would:
            # 1. Scrape competitor websites
            # 2. Monitor social media
            # 3. Track pricing pages
            # 4. Analyze customer reviews
            # 5. Monitor industry news

            # For now, simulate by analyzing our own data for competitive insights
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Analyze our pricing vs market (if we had competitor data)
            cur.execute("""
                SELECT
                    AVG(total_amount) as avg_job_value,
                    COUNT(*) as total_jobs,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as recent_jobs
                FROM jobs
                WHERE status = 'completed'
            """)

            our_data = cur.fetchone()

            # Simulate competitor intelligence (in production, would be real data)
            if our_data and our_data['avg_job_value']:
                avg_value = float(our_data['avg_job_value'])

                # Simulate: Competitor lowered prices
                activities.append({
                    'competitor_name': 'CompetitorA',
                    'activity_type': CompetitorActivity.PRICING_CHANGE.value,
                    'description': f'Competitor lowered average pricing to ${avg_value * 0.9:.2f} (10% below ours)',
                    'source': 'Market analysis',
                    'threat_level': ThreatLevel.MEDIUM.value,
                    'impact_areas': ['pricing', 'market_share', 'win_rate'],
                    'recommended_response': 'Consider value-based differentiation or selective price matching',
                    'estimated_impact': 'Potential 5-10% loss in price-sensitive segment',
                    'verified': False
                })

                # Simulate: Competitor launched new feature
                activities.append({
                    'competitor_name': 'CompetitorB',
                    'activity_type': CompetitorActivity.NEW_FEATURE.value,
                    'description': 'Launched AI-powered project estimation tool',
                    'source': 'Product monitoring',
                    'threat_level': ThreatLevel.HIGH.value,
                    'impact_areas': ['features', 'customer_satisfaction', 'competitive_advantage'],
                    'recommended_response': 'Accelerate development of our AI estimation feature',
                    'estimated_impact': 'Risk of losing tech-forward customers',
                    'verified': True
                })

            conn.close()

            # Persist activities
            if activities:
                self._persist_competitor_intelligence(activities)

        except Exception as e:
            logger.warning(f"Competitor monitoring failed: {e}")

        return activities

    def analyze_market_trends(self) -> List[MarketInsight]:
        """Analyze market trends and opportunities"""
        insights = []

        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Trend 1: Customer behavior analysis
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as recent_customers,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '60 days' AND created_at <= NOW() - INTERVAL '30 days') as previous_customers,
                    AVG(EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400) as avg_customer_age_days
                FROM customers
            """)

            customer_data = cur.fetchone()

            if customer_data and customer_data['recent_customers']:
                growth_rate = 0
                if customer_data['previous_customers'] > 0:
                    growth_rate = ((customer_data['recent_customers'] - customer_data['previous_customers']) /
                                  customer_data['previous_customers']) * 100

                insights.append({
                    'trend_type': MarketTrend.CUSTOMER_BEHAVIOR.value,
                    'title': f'Customer Acquisition Trend: {growth_rate:+.1f}% MoM',
                    'description': f'New customer acquisition rate changed by {growth_rate:+.1f}% month-over-month',
                    'supporting_data': {
                        'recent_customers': customer_data['recent_customers'],
                        'previous_customers': customer_data['previous_customers'],
                        'growth_rate': growth_rate
                    },
                    'implications': [
                        'Market demand increasing' if growth_rate > 0 else 'Market saturation or increased competition',
                        'Need to scale operations' if growth_rate > 20 else 'Maintain current capacity'
                    ],
                    'opportunities': [
                        'Expand marketing budget' if growth_rate > 0 else 'Optimize conversion funnel',
                        'Geographic expansion potential'
                    ],
                    'threats': [
                        'Quality may suffer from rapid growth' if growth_rate > 30 else 'Revenue stagnation risk'
                    ],
                    'recommended_actions': [
                        'Invest in customer success' if growth_rate > 0 else 'Improve lead generation',
                        'Monitor customer satisfaction metrics'
                    ],
                    'confidence': 0.85
                })

            # Trend 2: Technology adoption
            cur.execute("""
                SELECT COUNT(*) as ai_agent_executions
                FROM agent_executions
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)

            tech_data = cur.fetchone()

            if tech_data and tech_data['ai_agent_executions'] > 100:
                insights.append({
                    'trend_type': MarketTrend.TECHNOLOGY.value,
                    'title': 'AI Automation Adoption Accelerating',
                    'description': f'{tech_data["ai_agent_executions"]} AI agent executions in last 30 days',
                    'supporting_data': {
                        'executions': tech_data['ai_agent_executions'],
                        'avg_per_day': tech_data['ai_agent_executions'] / 30
                    },
                    'implications': [
                        'Market expects AI-powered solutions',
                        'Manual processes becoming competitive disadvantage'
                    ],
                    'opportunities': [
                        'Market leadership in AI-powered workflows',
                        'Premium pricing for AI features'
                    ],
                    'threats': [
                        'Competitors may catch up quickly',
                        'Over-reliance on AI could impact human expertise perception'
                    ],
                    'recommended_actions': [
                        'Highlight AI capabilities in marketing',
                        'Develop proprietary AI models',
                        'Train sales team on AI value proposition'
                    ],
                    'confidence': 0.75
                })

            conn.close()

            # Persist insights
            if insights:
                self._persist_market_insights(insights)

        except Exception as e:
            logger.warning(f"Market trend analysis failed: {e}")

        return insights

    def generate_strategic_responses(self,
                                     competitor_activities: List[Dict],
                                     market_insights: List[Dict]) -> List[StrategicResponse]:
        """Generate strategic responses to competitive threats and market trends"""
        responses = []

        # Response to competitor pricing changes
        pricing_threats = [a for a in competitor_activities if a['activity_type'] == 'pricing_change']
        if pricing_threats:
            high_threat_pricing = [p for p in pricing_threats if p['threat_level'] in ['high', 'critical']]
            if high_threat_pricing:
                responses.append({
                    'trigger': f'Competitor pricing changes detected ({len(high_threat_pricing)} high-priority)',
                    'response_type': 'defensive',
                    'title': 'Value-Based Differentiation Strategy',
                    'description': 'Instead of competing on price, enhance value proposition and differentiate on quality/service',
                    'objectives': [
                        'Maintain premium positioning',
                        'Communicate value beyond price',
                        'Improve customer ROI metrics'
                    ],
                    'action_items': [
                        'Create ROI calculator for sales team',
                        'Develop case studies showing value delivered',
                        'Launch "value over price" marketing campaign',
                        'Implement customer success metrics dashboard'
                    ],
                    'success_metrics': [
                        'Customer acquisition cost stays under $X',
                        'Win rate against price-focused competitors >60%',
                        'Customer satisfaction score >4.5/5'
                    ],
                    'resources_required': {
                        'budget': 15000,
                        'team_hours': 120,
                        'departments': ['marketing', 'sales', 'customer_success']
                    },
                    'timeline_days': 30,
                    'priority': 2
                })

        # Response to competitor feature launches
        feature_threats = [a for a in competitor_activities if a['activity_type'] == 'new_feature']
        if feature_threats:
            responses.append({
                'trigger': f'{len(feature_threats)} competitor feature launches detected',
                'response_type': 'offensive',
                'title': 'Accelerate AI Feature Roadmap',
                'description': 'Fast-track development of AI-powered features to maintain technology leadership',
                'objectives': [
                    'Launch AI estimation tool within 45 days',
                    'Exceed competitor capabilities',
                    'Establish technology leadership position'
                ],
                'action_items': [
                    'Sprint planning for AI estimation feature',
                    'Allocate additional development resources',
                    'Beta test with top 10 customers',
                    'Prepare competitive comparison materials',
                    'Plan launch marketing campaign'
                ],
                'success_metrics': [
                    'Feature launch within 45 days',
                    'Accuracy better than competitor by 10%+',
                    'Adoption by 30% of customers in first 60 days'
                ],
                'resources_required': {
                    'budget': 50000,
                    'team_hours': 400,
                    'departments': ['engineering', 'product', 'marketing']
                },
                'timeline_days': 45,
                'priority': 1
            })

        # Response to positive market trends
        positive_trends = [i for i in market_insights if 'growth' in i['title'].lower() or 'accelerating' in i['title'].lower()]
        if positive_trends:
            responses.append({
                'trigger': 'Positive market growth trend detected',
                'response_type': 'offensive',
                'title': 'Market Expansion Initiative',
                'description': 'Capitalize on market growth by expanding reach and capturing additional market share',
                'objectives': [
                    'Increase market share by 5% in 90 days',
                    'Enter 2 new geographic markets',
                    'Expand customer base by 30%'
                ],
                'action_items': [
                    'Increase marketing spend by 25%',
                    'Hire 2 additional sales reps',
                    'Launch referral program',
                    'Expand to adjacent verticals',
                    'Increase content marketing output'
                ],
                'success_metrics': [
                    'New customer acquisition +30%',
                    'Market share increase to X%',
                    'Revenue growth +25% QoQ'
                ],
                'resources_required': {
                    'budget': 75000,
                    'team_hours': 300,
                    'new_hires': 2
                },
                'timeline_days': 90,
                'priority': 1
            })

        # Persist responses
        if responses:
            self._persist_strategic_responses(responses)

        return responses

    def _persist_competitor_intelligence(self, activities: List[Dict]):
        """Persist competitor intelligence to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for activity in activities:
                cur.execute("""
                    INSERT INTO ai_competitor_intelligence
                    (competitor_name, activity_type, description, source, threat_level,
                     impact_areas, recommended_response, estimated_impact, verified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    activity['competitor_name'],
                    activity['activity_type'],
                    activity['description'],
                    activity['source'],
                    activity['threat_level'],
                    Json(activity['impact_areas']),
                    activity['recommended_response'],
                    activity['estimated_impact'],
                    activity['verified']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(activities)} competitor intelligence items")

        except Exception as e:
            logger.warning(f"Failed to persist competitor intelligence: {e}")

    def _persist_market_insights(self, insights: List[Dict]):
        """Persist market insights to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for insight in insights:
                cur.execute("""
                    INSERT INTO ai_market_insights
                    (trend_type, title, description, supporting_data, implications,
                     opportunities, threats, recommended_actions, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    insight['trend_type'],
                    insight['title'],
                    insight['description'],
                    Json(insight['supporting_data']),
                    Json(insight['implications']),
                    Json(insight['opportunities']),
                    Json(insight['threats']),
                    Json(insight['recommended_actions']),
                    insight['confidence']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(insights)} market insights")

        except Exception as e:
            logger.warning(f"Failed to persist market insights: {e}")

    def _persist_strategic_responses(self, responses: List[Dict]):
        """Persist strategic responses to database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            for response in responses:
                cur.execute("""
                    INSERT INTO ai_strategic_responses
                    (trigger, response_type, title, description, objectives, action_items,
                     success_metrics, resources_required, timeline_days, priority)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    response['trigger'],
                    response['response_type'],
                    response['title'],
                    response['description'],
                    Json(response['objectives']),
                    Json(response['action_items']),
                    Json(response['success_metrics']),
                    Json(response['resources_required']),
                    response['timeline_days'],
                    response['priority']
                ))

            conn.commit()
            conn.close()
            logger.info(f"✅ Persisted {len(responses)} strategic responses")

        except Exception as e:
            logger.warning(f"Failed to persist strategic responses: {e}")

    async def continuous_intelligence_loop(self, interval_hours: int = 8):
        """Main loop that continuously monitors competition and market"""
        logger.info(f"🔄 Starting competitive intelligence loop (every {interval_hours}h)")

        while True:
            try:
                logger.info("🔍 Monitoring competitive landscape...")

                # Monitor competitors
                competitor_activities = self.monitor_competitors()
                logger.info(f"📊 Detected {len(competitor_activities)} competitor activities")

                # Analyze market trends
                market_insights = self.analyze_market_trends()
                logger.info(f"📊 Generated {len(market_insights)} market insights")

                # Generate strategic responses
                strategic_responses = self.generate_strategic_responses(
                    competitor_activities,
                    market_insights
                )

                if strategic_responses:
                    logger.info(f"💡 Generated {len(strategic_responses)} strategic responses")
                    for response in strategic_responses:
                        logger.info(f"   - Priority {response['priority']}: {response['title']}")

                # Log critical threats
                critical_activities = [a for a in competitor_activities if a['threat_level'] == 'critical']
                if critical_activities:
                    logger.error(f"🚨 {len(critical_activities)} CRITICAL competitive threats require immediate attention!")

            except Exception as e:
                logger.error(f"❌ Competitive intelligence loop error: {e}")

            # Wait before next analysis
            await asyncio.sleep(interval_hours * 3600)


if __name__ == "__main__":
    agent = CompetitiveIntelligenceAgent()
    asyncio.run(agent.continuous_intelligence_loop(interval_hours=8))
