#!/usr/bin/env python3
"""
Automated Lead Nurturing Sequences
Build multi-touch automated follow-up campaigns
"""

import os
import json
import logging
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from openai import OpenAI
import httpx
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# OpenAI configuration
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class NurtureSequenceType(Enum):
    """Types of nurture sequences"""
    WELCOME = "welcome"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    REENGAGEMENT = "reengagement"
    ONBOARDING = "onboarding"
    UPSELL = "upsell"
    RETENTION = "retention"
    WIN_BACK = "win_back"

class TouchPointType(Enum):
    """Types of touch points"""
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    SOCIAL = "social"
    DIRECT_MAIL = "direct_mail"
    WEBINAR = "webinar"
    MEETING = "meeting"
    CONTENT = "content"

class LeadSegment(Enum):
    """Lead segmentation categories"""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    CUSTOMER = "customer"
    CHURNED = "churned"
    PROSPECT = "prospect"
    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"

class CampaignStatus(Enum):
    """Campaign status"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class TouchStatus(Enum):
    """Touch point status"""
    SCHEDULED = "scheduled"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    CLICKED = "clicked"
    RESPONDED = "responded"
    BOUNCED = "bounced"
    FAILED = "failed"

class LeadNurturingSystem:
    """Main lead nurturing system class"""

    def __init__(self):
        """Initialize the lead nurturing system"""
        self.sequence_templates = self._load_default_templates()
        self.personalization_engine = PersonalizationEngine()
        self.delivery_manager = DeliveryManager()
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create nurture sequences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_nurture_sequences (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sequence_name VARCHAR(255),
                    sequence_type VARCHAR(50),
                    target_segment VARCHAR(50),
                    touchpoint_count INT DEFAULT 0,
                    days_duration INT,
                    success_criteria JSONB DEFAULT '{}',
                    configuration JSONB DEFAULT '{}',
                    effectiveness_score FLOAT DEFAULT 0.5,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create sequence touchpoints table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_sequence_touchpoints (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sequence_id UUID REFERENCES ai_nurture_sequences(id),
                    touchpoint_number INT,
                    touchpoint_type VARCHAR(50),
                    days_after_trigger INT,
                    time_of_day TIME,
                    subject_line TEXT,
                    content_template TEXT,
                    personalization_tokens JSONB DEFAULT '[]',
                    call_to_action VARCHAR(255),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create lead enrollments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_lead_enrollments (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    lead_id VARCHAR(255),
                    sequence_id UUID REFERENCES ai_nurture_sequences(id),
                    enrollment_date TIMESTAMPTZ DEFAULT NOW(),
                    current_touchpoint INT DEFAULT 0,
                    status VARCHAR(50) DEFAULT 'active',
                    completion_date TIMESTAMPTZ,
                    opt_out_date TIMESTAMPTZ,
                    engagement_score FLOAT DEFAULT 0.0,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create touchpoint executions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_touchpoint_executions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    enrollment_id UUID REFERENCES ai_lead_enrollments(id),
                    touchpoint_id UUID REFERENCES ai_sequence_touchpoints(id),
                    scheduled_for TIMESTAMPTZ,
                    executed_at TIMESTAMPTZ,
                    status VARCHAR(50) DEFAULT 'scheduled',
                    delivery_channel VARCHAR(50),
                    personalized_content TEXT,
                    response_data JSONB DEFAULT '{}',
                    engagement_metrics JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create engagement tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_nurture_engagement (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    execution_id UUID REFERENCES ai_touchpoint_executions(id),
                    engagement_type VARCHAR(50),
                    engagement_timestamp TIMESTAMPTZ DEFAULT NOW(),
                    engagement_data JSONB DEFAULT '{}',
                    lead_score_impact FLOAT DEFAULT 0.0
                )
            """)

            # Create A/B testing table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_nurture_ab_tests (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sequence_id UUID REFERENCES ai_nurture_sequences(id),
                    test_name VARCHAR(255),
                    variant_a JSONB,
                    variant_b JSONB,
                    test_metric VARCHAR(50),
                    sample_size INT,
                    variant_a_results JSONB DEFAULT '{}',
                    variant_b_results JSONB DEFAULT '{}',
                    winner VARCHAR(1),
                    confidence_level FLOAT,
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)

            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_nurture_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    sequence_id UUID REFERENCES ai_nurture_sequences(id),
                    metric_date DATE,
                    enrollments INT DEFAULT 0,
                    completions INT DEFAULT 0,
                    opt_outs INT DEFAULT 0,
                    total_touches INT DEFAULT 0,
                    opens INT DEFAULT 0,
                    clicks INT DEFAULT 0,
                    responses INT DEFAULT 0,
                    conversions INT DEFAULT 0,
                    revenue_generated DECIMAL(10,2) DEFAULT 0,
                    avg_engagement_score FLOAT DEFAULT 0.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(sequence_id, metric_date)
                )
            """)

            # Create content library table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_nurture_content (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content_name VARCHAR(255),
                    content_type VARCHAR(50),
                    category VARCHAR(100),
                    subject_line TEXT,
                    body_content TEXT,
                    html_content TEXT,
                    personalization_fields JSONB DEFAULT '[]',
                    performance_score FLOAT DEFAULT 0.5,
                    tags JSONB DEFAULT '[]',
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sequences_type ON ai_nurture_sequences(sequence_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_lead ON ai_lead_enrollments(lead_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enrollments_status ON ai_lead_enrollments(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_scheduled ON ai_touchpoint_executions(scheduled_for)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_executions_status ON ai_touchpoint_executions(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_date ON ai_nurture_metrics(metric_date)")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _load_default_templates(self) -> Dict:
        """Load default sequence templates"""
        return {
            NurtureSequenceType.WELCOME: {
                'touchpoints': [
                    {'day': 0, 'type': 'email', 'template': 'welcome_immediate'},
                    {'day': 1, 'type': 'email', 'template': 'welcome_value'},
                    {'day': 3, 'type': 'email', 'template': 'welcome_resources'},
                    {'day': 7, 'type': 'email', 'template': 'welcome_cta'}
                ]
            },
            NurtureSequenceType.EDUCATIONAL: {
                'touchpoints': [
                    {'day': 0, 'type': 'email', 'template': 'edu_intro'},
                    {'day': 3, 'type': 'content', 'template': 'edu_guide'},
                    {'day': 7, 'type': 'webinar', 'template': 'edu_webinar'},
                    {'day': 10, 'type': 'email', 'template': 'edu_case_study'},
                    {'day': 14, 'type': 'email', 'template': 'edu_consultation'}
                ]
            },
            NurtureSequenceType.REENGAGEMENT: {
                'touchpoints': [
                    {'day': 0, 'type': 'email', 'template': 're_missed_you'},
                    {'day': 7, 'type': 'email', 'template': 're_special_offer'},
                    {'day': 14, 'type': 'email', 'template': 're_success_story'},
                    {'day': 21, 'type': 'email', 'template': 're_last_chance'}
                ]
            }
        }

    async def create_nurture_sequence(
        self,
        name: str,
        sequence_type: NurtureSequenceType,
        target_segment: LeadSegment,
        touchpoints: List[Dict],
        success_criteria: Dict = None,
        configuration: Dict = None
    ) -> str:
        """Create a new nurture sequence"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            sequence_id = str(uuid.uuid4())

            # Calculate duration
            max_days = max([tp.get('days_after_trigger', 0) for tp in touchpoints])

            # Create sequence
            cursor.execute("""
                INSERT INTO ai_nurture_sequences
                (id, sequence_name, sequence_type, target_segment,
                 touchpoint_count, days_duration, success_criteria, configuration)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                sequence_id,
                name,
                sequence_type.value,
                target_segment.value,
                len(touchpoints),
                max_days,
                Json(success_criteria or {}),
                Json(configuration or {})
            ))

            # Create touchpoints
            for idx, touchpoint in enumerate(touchpoints, 1):
                cursor.execute("""
                    INSERT INTO ai_sequence_touchpoints
                    (sequence_id, touchpoint_number, touchpoint_type,
                     days_after_trigger, time_of_day, subject_line,
                     content_template, personalization_tokens,
                     call_to_action, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    sequence_id,
                    idx,
                    touchpoint.get('type', TouchPointType.EMAIL.value),
                    touchpoint.get('days_after_trigger', 0),
                    touchpoint.get('time_of_day', '09:00:00'),
                    touchpoint.get('subject_line', ''),
                    touchpoint.get('content_template', ''),
                    Json(touchpoint.get('personalization_tokens', [])),
                    touchpoint.get('call_to_action', ''),
                    Json(touchpoint.get('metadata', {}))
                ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Created nurture sequence: {sequence_id}")
            return sequence_id

        except Exception as e:
            logger.error(f"Failed to create nurture sequence: {e}")
            return None

    async def enroll_lead(
        self,
        lead_id: str,
        sequence_id: str,
        metadata: Dict = None
    ) -> str:
        """Enroll a lead in a nurture sequence"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Check if already enrolled
            cursor.execute("""
                SELECT id FROM ai_lead_enrollments
                WHERE lead_id = %s AND sequence_id = %s AND status = 'active'
            """, (lead_id, sequence_id))

            if cursor.fetchone():
                logger.warning(f"Lead {lead_id} already enrolled in sequence {sequence_id}")
                return None

            enrollment_id = str(uuid.uuid4())

            # Create enrollment
            cursor.execute("""
                INSERT INTO ai_lead_enrollments
                (id, lead_id, sequence_id, metadata)
                VALUES (%s, %s, %s, %s)
            """, (enrollment_id, lead_id, sequence_id, Json(metadata or {})))

            # Get sequence touchpoints
            cursor.execute("""
                SELECT * FROM ai_sequence_touchpoints
                WHERE sequence_id = %s
                ORDER BY touchpoint_number
            """, (sequence_id,))

            touchpoints = cursor.fetchall()

            # Schedule touchpoints
            for touchpoint in touchpoints:
                scheduled_time = datetime.now(timezone.utc) + timedelta(
                    days=touchpoint[3]  # days_after_trigger
                )

                # Adjust for time of day
                if touchpoint[4]:  # time_of_day
                    scheduled_time = scheduled_time.replace(
                        hour=touchpoint[4].hour,
                        minute=touchpoint[4].minute
                    )

                cursor.execute("""
                    INSERT INTO ai_touchpoint_executions
                    (enrollment_id, touchpoint_id, scheduled_for,
                     delivery_channel, status)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    enrollment_id,
                    touchpoint[0],  # touchpoint id
                    scheduled_time,
                    touchpoint[2],  # touchpoint_type
                    TouchStatus.SCHEDULED.value
                ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Enrolled lead {lead_id} in sequence {sequence_id}")
            return enrollment_id

        except Exception as e:
            logger.error(f"Failed to enroll lead: {e}")
            return None

    async def execute_scheduled_touches(self) -> int:
        """Execute all scheduled touchpoints that are due"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get due touchpoints
            cursor.execute("""
                SELECT
                    te.*,
                    tp.subject_line,
                    tp.content_template,
                    tp.personalization_tokens,
                    tp.call_to_action,
                    le.lead_id,
                    le.metadata as enrollment_metadata
                FROM ai_touchpoint_executions te
                JOIN ai_sequence_touchpoints tp ON te.touchpoint_id = tp.id
                JOIN ai_lead_enrollments le ON te.enrollment_id = le.id
                WHERE te.status = %s
                  AND te.scheduled_for <= NOW()
                  AND le.status = 'active'
                ORDER BY te.scheduled_for
                LIMIT 100
            """, (TouchStatus.SCHEDULED.value,))

            touchpoints = cursor.fetchall()
            executed_count = 0

            for touchpoint in touchpoints:
                try:
                    # Personalize content
                    personalized = await self.personalization_engine.personalize(
                        lead_id=touchpoint['lead_id'],
                        template=touchpoint['content_template'],
                        subject=touchpoint['subject_line'],
                        tokens=touchpoint['personalization_tokens']
                    )

                    # Execute delivery
                    result = await self.delivery_manager.deliver(
                        channel=touchpoint['delivery_channel'],
                        lead_id=touchpoint['lead_id'],
                        content=personalized
                    )

                    # Update execution status
                    cursor.execute("""
                        UPDATE ai_touchpoint_executions
                        SET status = %s,
                            executed_at = NOW(),
                            personalized_content = %s,
                            response_data = %s
                        WHERE id = %s
                    """, (
                        TouchStatus.SENT.value if result['success'] else TouchStatus.FAILED.value,
                        personalized['content'],
                        Json(result),
                        touchpoint['id']
                    ))

                    executed_count += 1

                    # Update current touchpoint in enrollment
                    cursor.execute("""
                        UPDATE ai_lead_enrollments
                        SET current_touchpoint = current_touchpoint + 1
                        WHERE id = %s
                    """, (touchpoint['enrollment_id'],))

                except Exception as e:
                    logger.error(f"Failed to execute touchpoint {touchpoint['id']}: {e}")
                    cursor.execute("""
                        UPDATE ai_touchpoint_executions
                        SET status = %s,
                            response_data = %s
                        WHERE id = %s
                    """, (TouchStatus.FAILED.value, Json({'error': str(e)}), touchpoint['id']))

            conn.commit()
            cursor.close()
            conn.close()

            return executed_count

        except Exception as e:
            logger.error(f"Failed to execute scheduled touches: {e}")
            return 0

    async def track_engagement(
        self,
        execution_id: str,
        engagement_type: str,
        engagement_data: Dict = None
    ) -> bool:
        """Track engagement for a touchpoint execution"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Record engagement
            cursor.execute("""
                INSERT INTO ai_nurture_engagement
                (execution_id, engagement_type, engagement_data)
                VALUES (%s, %s, %s)
            """, (execution_id, engagement_type, Json(engagement_data or {})))

            # Update execution status
            status_map = {
                'open': TouchStatus.OPENED,
                'click': TouchStatus.CLICKED,
                'response': TouchStatus.RESPONDED
            }

            if engagement_type in status_map:
                cursor.execute("""
                    UPDATE ai_touchpoint_executions
                    SET status = %s,
                        engagement_metrics = engagement_metrics || %s
                    WHERE id = %s
                """, (
                    status_map[engagement_type].value,
                    Json({engagement_type: datetime.now(timezone.utc).isoformat()}),
                    execution_id
                ))

            # Update lead engagement score
            cursor.execute("""
                UPDATE ai_lead_enrollments
                SET engagement_score = engagement_score + %s
                WHERE id = (
                    SELECT enrollment_id FROM ai_touchpoint_executions
                    WHERE id = %s
                )
            """, (0.1 if engagement_type == 'open' else 0.3, execution_id))

            conn.commit()
            cursor.close()
            conn.close()

            return True

        except Exception as e:
            logger.error(f"Failed to track engagement: {e}")
            return False

    async def optimize_sequence(self, sequence_id: str) -> Dict:
        """Optimize a nurture sequence based on performance"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get sequence performance
            cursor.execute("""
                SELECT
                    AVG(engagement_score) as avg_engagement,
                    COUNT(*) FILTER (WHERE status = 'completed') as completions,
                    COUNT(*) FILTER (WHERE opt_out_date IS NOT NULL) as opt_outs,
                    COUNT(*) as total_enrollments
                FROM ai_lead_enrollments
                WHERE sequence_id = %s
            """, (sequence_id,))

            performance = cursor.fetchone()

            # Get touchpoint performance
            cursor.execute("""
                SELECT
                    tp.touchpoint_number,
                    tp.content_template,
                    COUNT(*) FILTER (WHERE te.status = 'opened') as opens,
                    COUNT(*) FILTER (WHERE te.status = 'clicked') as clicks,
                    COUNT(*) FILTER (WHERE te.status = 'responded') as responses,
                    COUNT(*) as total_sent
                FROM ai_sequence_touchpoints tp
                LEFT JOIN ai_touchpoint_executions te ON tp.id = te.touchpoint_id
                WHERE tp.sequence_id = %s
                GROUP BY tp.touchpoint_number, tp.content_template
                ORDER BY tp.touchpoint_number
            """, (sequence_id,))

            touchpoint_performance = cursor.fetchall()

            # Generate optimization recommendations
            recommendations = []

            # Check engagement rate
            if performance['avg_engagement'] < 0.3:
                recommendations.append({
                    'type': 'engagement',
                    'issue': 'Low engagement rate',
                    'suggestion': 'Consider more personalized content or different timing'
                })

            # Check opt-out rate
            opt_out_rate = performance['opt_outs'] / performance['total_enrollments'] if performance['total_enrollments'] > 0 else 0
            if opt_out_rate > 0.2:
                recommendations.append({
                    'type': 'opt_out',
                    'issue': 'High opt-out rate',
                    'suggestion': 'Reduce frequency or improve content relevance'
                })

            # Check touchpoint performance
            for tp in touchpoint_performance:
                if tp['total_sent'] > 0:
                    open_rate = tp['opens'] / tp['total_sent']
                    if open_rate < 0.2:
                        recommendations.append({
                            'type': 'touchpoint',
                            'touchpoint': tp['touchpoint_number'],
                            'issue': 'Low open rate',
                            'suggestion': 'Improve subject line or timing'
                        })

            cursor.close()
            conn.close()

            return {
                'sequence_id': sequence_id,
                'performance': performance,
                'touchpoint_analysis': touchpoint_performance,
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Failed to optimize sequence: {e}")
            return {}

    async def run_ab_test(
        self,
        sequence_id: str,
        test_name: str,
        variant_a: Dict,
        variant_b: Dict,
        test_metric: str,
        sample_size: int = 100
    ) -> str:
        """Run an A/B test on a nurture sequence"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            test_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO ai_nurture_ab_tests
                (id, sequence_id, test_name, variant_a, variant_b,
                 test_metric, sample_size)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                test_id,
                sequence_id,
                test_name,
                Json(variant_a),
                Json(variant_b),
                test_metric,
                sample_size
            ))

            conn.commit()
            cursor.close()
            conn.close()

            # Start test execution
            asyncio.create_task(self._execute_ab_test(test_id))

            return test_id

        except Exception as e:
            logger.error(f"Failed to create A/B test: {e}")
            return None

    async def _execute_ab_test(self, test_id: str):
        """Execute an A/B test"""
        # Implementation would handle test execution
        pass

    async def get_sequence_metrics(
        self,
        sequence_id: str,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> Dict:
        """Get comprehensive metrics for a nurture sequence"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build date filter
            date_filter = ""
            params = [sequence_id]
            if start_date:
                date_filter += " AND metric_date >= %s"
                params.append(start_date.date())
            if end_date:
                date_filter += " AND metric_date <= %s"
                params.append(end_date.date())

            # Get metrics
            cursor.execute(f"""
                SELECT
                    SUM(enrollments) as total_enrollments,
                    SUM(completions) as total_completions,
                    SUM(opt_outs) as total_opt_outs,
                    SUM(opens) as total_opens,
                    SUM(clicks) as total_clicks,
                    SUM(responses) as total_responses,
                    SUM(conversions) as total_conversions,
                    SUM(revenue_generated) as total_revenue,
                    AVG(avg_engagement_score) as avg_engagement
                FROM ai_nurture_metrics
                WHERE sequence_id = %s {date_filter}
            """, params)

            metrics = cursor.fetchone()

            # Calculate rates
            if metrics['total_enrollments'] > 0:
                metrics['completion_rate'] = metrics['total_completions'] / metrics['total_enrollments']
                metrics['opt_out_rate'] = metrics['total_opt_outs'] / metrics['total_enrollments']
                metrics['conversion_rate'] = metrics['total_conversions'] / metrics['total_enrollments']

            if metrics['total_opens'] > 0:
                metrics['click_rate'] = metrics['total_clicks'] / metrics['total_opens']

            cursor.close()
            conn.close()

            return metrics

        except Exception as e:
            logger.error(f"Failed to get sequence metrics: {e}")
            return {}

class PersonalizationEngine:
    """Engine for personalizing nurture content"""

    async def personalize(
        self,
        lead_id: str,
        template: str,
        subject: str = None,
        tokens: List[str] = None
    ) -> Dict:
        """Personalize content for a specific lead"""
        try:
            # Get lead data
            lead_data = await self._get_lead_data(lead_id)

            # Replace tokens
            personalized_content = template
            personalized_subject = subject

            if tokens:
                for token in tokens:
                    value = lead_data.get(token, '')
                    personalized_content = personalized_content.replace(f'{{{token}}}', str(value))
                    if subject:
                        personalized_subject = personalized_subject.replace(f'{{{token}}}', str(value))

            # AI enhancement
            if lead_data.get('preferences'):
                personalized_content = await self._ai_enhance(
                    personalized_content,
                    lead_data['preferences']
                )

            return {
                'subject': personalized_subject,
                'content': personalized_content,
                'lead_data': lead_data
            }

        except Exception as e:
            logger.error(f"Failed to personalize content: {e}")
            return {'content': template, 'subject': subject}

    async def _get_lead_data(self, lead_id: str) -> Dict:
        """Get lead data for personalization"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get lead info (simplified - would join with actual lead table)
            cursor.execute("""
                SELECT * FROM ai_leads
                WHERE id = %s OR company_name = %s
                LIMIT 1
            """, (lead_id, lead_id))

            lead = cursor.fetchone()
            cursor.close()
            conn.close()

            return lead or {'lead_id': lead_id}

        except Exception as e:
            logger.error(f"Failed to get lead data: {e}")
            return {'lead_id': lead_id}

    async def _ai_enhance(self, content: str, preferences: Dict) -> str:
        """Use AI to enhance personalization"""
        try:
            prompt = f"""
            Enhance this email content based on lead preferences:
            Content: {content}
            Preferences: {json.dumps(preferences)}

            Make it more personalized while keeping the same structure and call-to-action.
            """

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Failed to AI enhance: {e}")
            return content

class DeliveryManager:
    """Manager for delivering nurture touchpoints"""

    async def deliver(
        self,
        channel: str,
        lead_id: str,
        content: Dict
    ) -> Dict:
        """Deliver content through specified channel"""
        try:
            if channel == TouchPointType.EMAIL.value:
                return await self._deliver_email(lead_id, content)
            elif channel == TouchPointType.SMS.value:
                return await self._deliver_sms(lead_id, content)
            else:
                # Default to email
                return await self._deliver_email(lead_id, content)

        except Exception as e:
            logger.error(f"Failed to deliver: {e}")
            return {'success': False, 'error': str(e)}

    async def _deliver_email(self, lead_id: str, content: Dict) -> Dict:
        """Deliver email (simplified - would integrate with email service)"""
        try:
            # Store in email queue
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_email_queue
                (recipient, subject, body, status, metadata)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                lead_id,
                content.get('subject', 'Update'),
                content.get('content', ''),
                'queued',
                Json({'source': 'nurture_system'})
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return {'success': True, 'channel': 'email', 'status': 'queued'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def _deliver_sms(self, lead_id: str, content: Dict) -> Dict:
        """Deliver SMS (simplified - would integrate with SMS service)"""
        return {'success': True, 'channel': 'sms', 'status': 'sent'}

# Singleton instance
_nurturing_system = None

def get_lead_nurturing_system():
    """Get or create the nurturing system instance"""
    global _nurturing_system
    if _nurturing_system is None:
        _nurturing_system = LeadNurturingSystem()
    return _nurturing_system