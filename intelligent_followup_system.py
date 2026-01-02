#!/usr/bin/env python3
"""
Intelligent Follow-up System - Task 17
Automated follow-up orchestration with AI-powered timing, personalization, and multi-channel delivery
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
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

class FollowUpType(Enum):
    LEAD_INQUIRY = "lead_inquiry"
    QUOTE_SENT = "quote_sent"
    PROPOSAL_REVIEW = "proposal_review"
    CONTRACT_NEGOTIATION = "contract_negotiation"
    PAYMENT_REMINDER = "payment_reminder"
    SERVICE_COMPLETION = "service_completion"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    RENEWAL_NOTICE = "renewal_notice"
    UPSELL_OPPORTUNITY = "upsell_opportunity"
    WIN_BACK = "win_back"
    REFERRAL_REQUEST = "referral_request"
    FEEDBACK_REQUEST = "feedback_request"
    APPOINTMENT_REMINDER = "appointment_reminder"
    DOCUMENT_REQUEST = "document_request"
    TASK_UPDATE = "task_update"

class FollowUpPriority(Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SCHEDULED = "scheduled"

class FollowUpStatus(Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    SENT = "sent"
    DELIVERED = "delivered"
    OPENED = "opened"
    RESPONDED = "responded"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class DeliveryChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PHONE = "phone"
    IN_APP = "in_app"
    PUSH = "push"
    WEBHOOK = "webhook"
    SOCIAL = "social"
    CHAT = "chat"

class ResponseType(Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    NO_RESPONSE = "no_response"
    BOUNCE = "bounce"
    UNSUBSCRIBE = "unsubscribe"

class IntelligentFollowUpSystem:
    """Main intelligent follow-up system with AI-powered orchestration"""

    def __init__(self):
        self.conn = None
        self.ai_model = "gpt-4"
        self.timing_optimizer = TimingOptimizer()
        self.content_generator = ContentGenerator()
        self.channel_selector = ChannelSelector()
        self.response_analyzer = ResponseAnalyzer()
        self.performance_tracker = PerformanceTracker()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def create_followup_sequence(
        self,
        followup_type: FollowUpType,
        entity_id: str,
        entity_type: str,
        context: dict[str, Any],
        priority: FollowUpPriority = FollowUpPriority.MEDIUM,
        custom_rules: Optional[dict] = None
    ) -> str:
        """Create an intelligent follow-up sequence"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            sequence_id = str(uuid.uuid4())

            # Analyze context to determine optimal follow-up strategy
            strategy = await self._analyze_followup_strategy(
                followup_type, context, priority
            )

            # Generate follow-up touchpoints
            touchpoints = await self._generate_touchpoints(
                followup_type, strategy, context
            )

            # Store follow-up sequence
            cursor.execute("""
                INSERT INTO ai_followup_sequences (
                    id, followup_type, entity_id, entity_type,
                    priority, context, strategy, touchpoints,
                    custom_rules, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                sequence_id, followup_type.value, entity_id, entity_type,
                priority.value, json.dumps(context), json.dumps(strategy),
                json.dumps(touchpoints), json.dumps(custom_rules or {}),
                FollowUpStatus.PENDING.value
            ))

            # Schedule initial touchpoints
            for touchpoint in touchpoints[:3]:  # Schedule first 3
                await self._schedule_touchpoint(
                    sequence_id, touchpoint, cursor
                )

            conn.commit()

            logger.info(f"Created follow-up sequence {sequence_id} for {entity_type} {entity_id}")
            return sequence_id

        except Exception as e:
            logger.error(f"Error creating follow-up sequence: {e}")
            if conn:
                conn.rollback()
            raise

    async def _analyze_followup_strategy(
        self,
        followup_type: FollowUpType,
        context: dict,
        priority: FollowUpPriority
    ) -> dict:
        """Analyze context to determine optimal follow-up strategy"""
        strategy = {
            "approach": "multi_touch",
            "urgency_level": priority.value,
            "personalization_level": "high",
            "channels": [],
            "timing": {},
            "escalation_rules": {}
        }

        # Determine channels based on type and context
        if followup_type in [FollowUpType.URGENT, FollowUpType.PAYMENT_REMINDER]:
            strategy["channels"] = ["email", "sms", "phone"]
            strategy["timing"]["initial_delay"] = 0
            strategy["timing"]["retry_interval"] = 24  # hours
        elif followup_type in [FollowUpType.LEAD_INQUIRY, FollowUpType.QUOTE_SENT]:
            strategy["channels"] = ["email", "sms"]
            strategy["timing"]["initial_delay"] = 1  # hour
            strategy["timing"]["retry_interval"] = 48  # hours
        else:
            strategy["channels"] = ["email"]
            strategy["timing"]["initial_delay"] = 24  # hours
            strategy["timing"]["retry_interval"] = 72  # hours

        # Set escalation rules
        strategy["escalation_rules"] = {
            "max_attempts": 5,
            "escalate_after": 3,
            "escalation_channels": ["phone", "in_app"],
            "final_action": "notify_manager"
        }

        return strategy

    async def _generate_touchpoints(
        self,
        followup_type: FollowUpType,
        strategy: dict,
        context: dict
    ) -> list[dict]:
        """Generate intelligent touchpoints for follow-up"""
        touchpoints = []

        # Define touchpoint templates based on type
        templates = self._get_touchpoint_templates(followup_type)

        for i, template in enumerate(templates):
            delay_hours = strategy["timing"]["initial_delay"] + (
                i * strategy["timing"]["retry_interval"]
            )

            touchpoint = {
                "step": i + 1,
                "delay_hours": delay_hours,
                "channel": strategy["channels"][min(i, len(strategy["channels"]) - 1)],
                "template": template,
                "personalization": {
                    "use_ai": True,
                    "context_aware": True,
                    "tone": self._determine_tone(followup_type, i)
                },
                "conditions": {
                    "skip_if_responded": i > 0,
                    "skip_if_converted": True
                }
            }

            touchpoints.append(touchpoint)

        return touchpoints

    def _get_touchpoint_templates(self, followup_type: FollowUpType) -> list[dict]:
        """Get touchpoint templates for follow-up type"""
        templates_map = {
            FollowUpType.LEAD_INQUIRY: [
                {"subject": "Thanks for your interest!", "tone": "friendly"},
                {"subject": "Quick question about your needs", "tone": "consultative"},
                {"subject": "Special offer for you", "tone": "value_focused"},
                {"subject": "Last chance - Limited time offer", "tone": "urgent"},
                {"subject": "We're here when you're ready", "tone": "patient"}
            ],
            FollowUpType.QUOTE_SENT: [
                {"subject": "Your quote is ready", "tone": "professional"},
                {"subject": "Questions about your quote?", "tone": "helpful"},
                {"subject": "Quote expires soon", "tone": "urgent"},
                {"subject": "Special discount available", "tone": "incentive"},
                {"subject": "Final notice on your quote", "tone": "final"}
            ],
            FollowUpType.PAYMENT_REMINDER: [
                {"subject": "Payment reminder", "tone": "friendly"},
                {"subject": "Payment due soon", "tone": "professional"},
                {"subject": "Urgent: Payment overdue", "tone": "urgent"},
                {"subject": "Action required: Payment", "tone": "firm"},
                {"subject": "Final notice", "tone": "final"}
            ]
        }

        return templates_map.get(followup_type, [
            {"subject": "Following up", "tone": "professional"}
        ])

    def _determine_tone(self, followup_type: FollowUpType, step: int) -> str:
        """Determine appropriate tone based on type and step"""
        if followup_type in [FollowUpType.PAYMENT_REMINDER, FollowUpType.CONTRACT_NEGOTIATION]:
            tones = ["friendly", "professional", "firm", "urgent", "final"]
        elif followup_type in [FollowUpType.LEAD_INQUIRY, FollowUpType.UPSELL_OPPORTUNITY]:
            tones = ["friendly", "consultative", "value_focused", "patient", "helpful"]
        else:
            tones = ["professional", "helpful", "friendly", "consultative", "patient"]

        return tones[min(step, len(tones) - 1)]

    async def _schedule_touchpoint(
        self,
        sequence_id: str,
        touchpoint: dict,
        cursor: Any
    ) -> str:
        """Schedule a follow-up touchpoint"""
        touchpoint_id = str(uuid.uuid4())
        scheduled_time = datetime.now(timezone.utc) + timedelta(
            hours=touchpoint["delay_hours"]
        )

        cursor.execute("""
            INSERT INTO ai_followup_touchpoints (
                id, sequence_id, step_number, scheduled_at,
                channel, template, personalization,
                conditions, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            touchpoint_id, sequence_id, touchpoint["step"],
            scheduled_time, touchpoint["channel"],
            json.dumps(touchpoint["template"]),
            json.dumps(touchpoint["personalization"]),
            json.dumps(touchpoint["conditions"]),
            FollowUpStatus.SCHEDULED.value
        ))

        return touchpoint_id

    async def execute_scheduled_followups(self) -> list[dict]:
        """Execute all scheduled follow-ups"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get due follow-ups
            cursor.execute("""
                SELECT
                    ft.*,
                    fs.entity_id,
                    fs.entity_type,
                    fs.context,
                    fs.followup_type
                FROM ai_followup_touchpoints ft
                JOIN ai_followup_sequences fs ON ft.sequence_id = fs.id
                WHERE ft.status = %s
                AND ft.scheduled_at <= NOW()
                ORDER BY ft.scheduled_at
                LIMIT 100
            """, (FollowUpStatus.SCHEDULED.value,))

            due_followups = cursor.fetchall()
            results = []

            for followup in due_followups:
                result = await self._execute_followup(followup, cursor)
                results.append(result)

            conn.commit()

            logger.info(f"Executed {len(results)} follow-ups")
            return results

        except Exception as e:
            logger.error(f"Error executing scheduled follow-ups: {e}")
            if conn:
                conn.rollback()
            return []

    async def _execute_followup(self, followup: dict, cursor: Any) -> dict:
        """Execute a single follow-up"""
        try:
            # Update status to in_progress
            cursor.execute("""
                UPDATE ai_followup_touchpoints
                SET status = %s, updated_at = NOW()
                WHERE id = %s
            """, (FollowUpStatus.IN_PROGRESS.value, followup['id']))

            # Generate personalized content
            content = await self.content_generator.generate_content(
                followup['template'],
                json.loads(followup['context']),
                json.loads(followup['personalization'])
            )

            # Select optimal channel
            channel = await self.channel_selector.select_channel(
                followup['entity_id'],
                followup['channel'],
                followup['followup_type']
            )

            # Send follow-up
            delivery_result = await self._send_followup(
                channel, followup['entity_id'], content
            )

            # Record execution
            execution_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_followup_executions (
                    id, touchpoint_id, sequence_id,
                    channel_used, content_sent, delivery_result,
                    status, executed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                execution_id, followup['id'], followup['sequence_id'],
                channel, json.dumps(content), json.dumps(delivery_result),
                delivery_result['status']
            ))

            # Update touchpoint status
            cursor.execute("""
                UPDATE ai_followup_touchpoints
                SET status = %s, executed_at = NOW()
                WHERE id = %s
            """, (delivery_result['status'], followup['id']))

            return {
                "followup_id": followup['id'],
                "execution_id": execution_id,
                "status": delivery_result['status'],
                "channel": channel
            }

        except Exception as e:
            logger.error(f"Error executing follow-up {followup['id']}: {e}")
            cursor.execute("""
                UPDATE ai_followup_touchpoints
                SET status = %s, error_message = %s
                WHERE id = %s
            """, (FollowUpStatus.FAILED.value, str(e), followup['id']))

            return {
                "followup_id": followup['id'],
                "status": "failed",
                "error": str(e)
            }

    async def _send_followup(
        self,
        channel: str,
        entity_id: str,
        content: dict
    ) -> dict:
        """Send follow-up through specified channel"""
        # Simulate sending for now - integrate with actual channels
        delivery_result = {
            "status": FollowUpStatus.SENT.value,
            "channel": channel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_id": str(uuid.uuid4())
        }

        # Here you would integrate with actual delivery services:
        # - Email: SendGrid, AWS SES, etc.
        # - SMS: Twilio, AWS SNS, etc.
        # - Phone: Twilio Voice, etc.
        # - In-app: Push notifications, etc.

        logger.info(f"Sent follow-up to {entity_id} via {channel}")
        return delivery_result

    async def track_response(
        self,
        execution_id: str,
        response_type: ResponseType,
        response_data: Optional[dict] = None
    ) -> dict:
        """Track response to follow-up"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Record response
            response_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_followup_responses (
                    id, execution_id, response_type,
                    response_data, received_at
                ) VALUES (%s, %s, %s, %s, NOW())
            """, (
                response_id, execution_id, response_type.value,
                json.dumps(response_data or {})
            ))

            # Analyze response
            analysis = await self.response_analyzer.analyze_response(
                response_type, response_data
            )

            # Update execution status
            cursor.execute("""
                UPDATE ai_followup_executions
                SET response_received = true,
                    response_type = %s,
                    response_analysis = %s
                WHERE id = %s
            """, (response_type.value, json.dumps(analysis), execution_id))

            # Check if sequence should be updated
            if response_type in [ResponseType.POSITIVE, ResponseType.UNSUBSCRIBE]:
                cursor.execute("""
                    UPDATE ai_followup_sequences fs
                    SET status = %s
                    FROM ai_followup_executions fe
                    WHERE fe.sequence_id = fs.id
                    AND fe.id = %s
                """, (FollowUpStatus.COMPLETED.value, execution_id))

            conn.commit()

            return {
                "response_id": response_id,
                "analysis": analysis,
                "sequence_updated": response_type in [ResponseType.POSITIVE, ResponseType.UNSUBSCRIBE]
            }

        except Exception as e:
            logger.error(f"Error tracking response: {e}")
            if conn:
                conn.rollback()
            raise

    async def get_followup_analytics(
        self,
        entity_id: Optional[str] = None,
        followup_type: Optional[FollowUpType] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> dict:
        """Get follow-up performance analytics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query conditions
            conditions = []
            params = []

            if entity_id:
                conditions.append("fs.entity_id = %s")
                params.append(entity_id)

            if followup_type:
                conditions.append("fs.followup_type = %s")
                params.append(followup_type.value)

            if date_from:
                conditions.append("fs.created_at >= %s")
                params.append(date_from)

            if date_to:
                conditions.append("fs.created_at <= %s")
                params.append(date_to)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get overall metrics
            cursor.execute(f"""
                SELECT
                    COUNT(DISTINCT fs.id) as total_sequences,
                    COUNT(DISTINCT ft.id) as total_touchpoints,
                    COUNT(DISTINCT fe.id) as total_executions,
                    COUNT(DISTINCT fr.id) as total_responses,
                    AVG(CASE
                        WHEN fr.response_type IN ('positive', 'responded')
                        THEN 1 ELSE 0
                    END) * 100 as response_rate,
                    AVG(EXTRACT(EPOCH FROM (fr.received_at - fe.executed_at))/3600) as avg_response_time_hours
                FROM ai_followup_sequences fs
                LEFT JOIN ai_followup_touchpoints ft ON ft.sequence_id = fs.id
                LEFT JOIN ai_followup_executions fe ON fe.touchpoint_id = ft.id
                LEFT JOIN ai_followup_responses fr ON fr.execution_id = fe.id
                WHERE {where_clause}
            """, params)

            overall_metrics = cursor.fetchone()

            # Get channel performance
            cursor.execute(f"""
                SELECT
                    fe.channel_used as channel,
                    COUNT(*) as sent_count,
                    COUNT(fr.id) as response_count,
                    AVG(CASE
                        WHEN fr.response_type IN ('positive', 'responded')
                        THEN 1 ELSE 0
                    END) * 100 as success_rate
                FROM ai_followup_sequences fs
                JOIN ai_followup_touchpoints ft ON ft.sequence_id = fs.id
                JOIN ai_followup_executions fe ON fe.touchpoint_id = ft.id
                LEFT JOIN ai_followup_responses fr ON fr.execution_id = fe.id
                WHERE {where_clause}
                GROUP BY fe.channel_used
            """, params)

            channel_performance = cursor.fetchall()

            return {
                "overall": overall_metrics,
                "by_channel": channel_performance,
                "period": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None
                }
            }

        except Exception as e:
            logger.error(f"Error getting follow-up analytics: {e}")
            return {}


class TimingOptimizer:
    """Optimize follow-up timing based on historical performance"""

    async def optimize_timing(
        self,
        entity_profile: dict,
        followup_type: FollowUpType,
        historical_data: list[dict]
    ) -> dict:
        """Determine optimal timing for follow-up"""
        # Analyze historical response patterns
        best_times = self._analyze_response_patterns(historical_data)

        # Consider entity timezone and preferences
        timezone_offset = entity_profile.get('timezone_offset', 0)
        preferred_contact_time = entity_profile.get('preferred_contact_time', 'business_hours')

        # Calculate optimal send time
        optimal_timing = {
            "best_day": best_times.get('day', 'weekday'),
            "best_hour": best_times.get('hour', 10) + timezone_offset,
            "avoid_times": ["weekend_nights", "early_morning"],
            "urgency_factor": self._calculate_urgency(followup_type)
        }

        return optimal_timing

    def _analyze_response_patterns(self, historical_data: list[dict]) -> dict:
        """Analyze historical response patterns"""
        if not historical_data:
            return {"day": "weekday", "hour": 10}

        # Analyze response times
        response_times = []
        for record in historical_data:
            if record.get('response_time'):
                response_times.append(record['response_time'])

        # Calculate best performing times
        # This would involve more sophisticated analysis in production
        return {
            "day": "tuesday",  # Example: Tuesday performs best
            "hour": 14  # Example: 2 PM performs best
        }

    def _calculate_urgency(self, followup_type: FollowUpType) -> float:
        """Calculate urgency factor for follow-up type"""
        urgency_map = {
            FollowUpType.PAYMENT_REMINDER: 0.9,
            FollowUpType.CONTRACT_NEGOTIATION: 0.8,
            FollowUpType.QUOTE_SENT: 0.7,
            FollowUpType.LEAD_INQUIRY: 0.6,
            FollowUpType.UPSELL_OPPORTUNITY: 0.4,
            FollowUpType.FEEDBACK_REQUEST: 0.3
        }
        return urgency_map.get(followup_type, 0.5)


class ContentGenerator:
    """Generate personalized follow-up content"""

    async def generate_content(
        self,
        template: dict,
        context: dict,
        personalization: dict
    ) -> dict:
        """Generate personalized content for follow-up"""
        # Extract template details
        subject = template.get('subject', 'Follow-up')
        tone = template.get('tone', 'professional')

        # Apply personalization
        if personalization.get('use_ai'):
            content = await self._generate_ai_content(subject, context, tone)
        else:
            content = self._generate_template_content(template, context)

        # Add personalization tokens
        content = self._apply_personalization(content, context)

        return content

    async def _generate_ai_content(
        self,
        subject: str,
        context: dict,
        tone: str
    ) -> dict:
        """Generate AI-powered content"""
        # This would integrate with OpenAI or similar
        # For now, return template-based content
        return {
            "subject": subject,
            "body": f"Hello {context.get('name', 'there')},\n\n" +
                   f"We wanted to follow up regarding {context.get('topic', 'your inquiry')}.\n\n" +
                   "Best regards,\nThe Team",
            "tone": tone,
            "personalized": True
        }

    def _generate_template_content(self, template: dict, context: dict) -> dict:
        """Generate template-based content"""
        return {
            "subject": template['subject'],
            "body": template.get('body', 'Follow-up message'),
            "tone": template.get('tone', 'professional')
        }

    def _apply_personalization(self, content: dict, context: dict) -> dict:
        """Apply personalization tokens to content"""
        # Replace tokens in content
        for key, value in context.items():
            if isinstance(value, str):
                content['subject'] = content['subject'].replace(f"{{{key}}}", value)
                content['body'] = content['body'].replace(f"{{{key}}}", value)

        return content


class ChannelSelector:
    """Select optimal delivery channel for follow-up"""

    async def select_channel(
        self,
        entity_id: str,
        preferred_channel: str,
        followup_type: str
    ) -> str:
        """Select the best channel for delivery"""
        # Check channel availability
        available_channels = await self._get_available_channels(entity_id)

        # If preferred channel is available, use it
        if preferred_channel in available_channels:
            return preferred_channel

        # Otherwise select based on priority
        channel_priority = self._get_channel_priority(followup_type)

        for channel in channel_priority:
            if channel in available_channels:
                return channel

        # Default to email
        return "email"

    async def _get_available_channels(self, entity_id: str) -> list[str]:
        """Get available channels for entity"""
        # This would check actual contact information
        # For now, return common channels
        return ["email", "sms", "in_app"]

    def _get_channel_priority(self, followup_type: str) -> list[str]:
        """Get channel priority for follow-up type"""
        priorities = {
            "payment_reminder": ["sms", "email", "phone"],
            "lead_inquiry": ["email", "sms", "chat"],
            "service_completion": ["email", "in_app", "sms"]
        }
        return priorities.get(followup_type, ["email", "sms", "in_app"])


class ResponseAnalyzer:
    """Analyze responses to follow-ups"""

    async def analyze_response(
        self,
        response_type: ResponseType,
        response_data: Optional[dict]
    ) -> dict:
        """Analyze follow-up response"""
        analysis = {
            "sentiment": self._determine_sentiment(response_type),
            "intent": self._determine_intent(response_type, response_data),
            "urgency": self._determine_urgency(response_data),
            "next_action": self._determine_next_action(response_type)
        }

        # Add AI analysis if response data contains text
        if response_data and response_data.get('text'):
            analysis['ai_insights'] = await self._analyze_text(response_data['text'])

        return analysis

    def _determine_sentiment(self, response_type: ResponseType) -> str:
        """Determine sentiment from response type"""
        sentiment_map = {
            ResponseType.POSITIVE: "positive",
            ResponseType.NEUTRAL: "neutral",
            ResponseType.NEGATIVE: "negative",
            ResponseType.NO_RESPONSE: "unknown",
            ResponseType.BOUNCE: "failed",
            ResponseType.UNSUBSCRIBE: "negative"
        }
        return sentiment_map.get(response_type, "unknown")

    def _determine_intent(
        self,
        response_type: ResponseType,
        response_data: Optional[dict]
    ) -> str:
        """Determine intent from response"""
        if response_type == ResponseType.POSITIVE:
            return "interested"
        elif response_type == ResponseType.NEGATIVE:
            return "not_interested"
        elif response_type == ResponseType.UNSUBSCRIBE:
            return "opt_out"
        elif response_data and response_data.get('action'):
            return response_data['action']
        else:
            return "unclear"

    def _determine_urgency(self, response_data: Optional[dict]) -> str:
        """Determine urgency from response"""
        if not response_data:
            return "normal"

        # Check for urgency indicators
        urgent_keywords = ["urgent", "asap", "immediately", "now", "today"]
        if response_data.get('text'):
            text_lower = response_data['text'].lower()
            if any(keyword in text_lower for keyword in urgent_keywords):
                return "high"

        return "normal"

    def _determine_next_action(self, response_type: ResponseType) -> str:
        """Determine next action based on response"""
        action_map = {
            ResponseType.POSITIVE: "proceed_with_next_step",
            ResponseType.NEUTRAL: "continue_nurturing",
            ResponseType.NEGATIVE: "pause_sequence",
            ResponseType.NO_RESPONSE: "send_next_touchpoint",
            ResponseType.BOUNCE: "update_contact_info",
            ResponseType.UNSUBSCRIBE: "remove_from_sequence"
        }
        return action_map.get(response_type, "review_manually")

    async def _analyze_text(self, text: str) -> dict:
        """Analyze text content using AI"""
        # This would use NLP/AI for deeper analysis
        # For now, return basic analysis
        return {
            "length": len(text),
            "questions_asked": text.count('?'),
            "exclamations": text.count('!'),
            "key_phrases": []  # Would extract key phrases
        }


class AutomatedCheckInScheduler:
    """Automated check-in scheduling based on customer health and activity"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def schedule_checkins(
        self,
        customer_id: str,
        check_in_strategy: str = "proactive"
    ) -> dict:
        """Schedule automated check-ins for a customer"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get customer activity and health data
            cursor.execute("""
                SELECT
                    c.id,
                    c.name,
                    c.created_at,
                    COUNT(DISTINCT j.id) as total_jobs,
                    MAX(j.created_at) as last_job_date,
                    EXTRACT(EPOCH FROM (NOW() - MAX(j.created_at)))/86400 as days_since_last_job
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                WHERE c.id::text = %s
                GROUP BY c.id, c.name, c.created_at
            """, (customer_id,))

            customer_data = cursor.fetchone()

            if not customer_data:
                return {"error": "Customer not found"}

            # Determine check-in schedule based on activity level
            schedule = self._determine_checkin_schedule(
                customer_data, check_in_strategy
            )

            # Create check-in sequence
            sequence_id = str(uuid.uuid4())
            for check_in in schedule['check_ins']:
                await self._schedule_single_checkin(
                    sequence_id, customer_id, check_in, cursor
                )

            conn.commit()

            return {
                "sequence_id": sequence_id,
                "customer_id": customer_id,
                "strategy": check_in_strategy,
                "check_ins_scheduled": len(schedule['check_ins']),
                "schedule": schedule,
                "scheduled_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error scheduling check-ins: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    def _determine_checkin_schedule(
        self,
        customer_data: dict,
        strategy: str
    ) -> dict:
        """Determine appropriate check-in schedule"""
        days_since_last_job = customer_data.get('days_since_last_job', 0) or 0
        total_jobs = customer_data.get('total_jobs', 0)

        schedule = {"check_ins": [], "frequency": "monthly"}

        if strategy == "proactive":
            # High-touch for at-risk customers
            if days_since_last_job > 60 or total_jobs == 0:
                schedule['frequency'] = "weekly"
                schedule['check_ins'] = [
                    {"type": "initial_outreach", "delay_days": 0, "priority": "high"},
                    {"type": "value_reminder", "delay_days": 7, "priority": "high"},
                    {"type": "offer_assistance", "delay_days": 14, "priority": "medium"},
                    {"type": "feedback_request", "delay_days": 21, "priority": "medium"}
                ]
            # Medium-touch for active customers
            elif days_since_last_job <= 30:
                schedule['frequency'] = "monthly"
                schedule['check_ins'] = [
                    {"type": "satisfaction_check", "delay_days": 30, "priority": "medium"},
                    {"type": "upsell_opportunity", "delay_days": 60, "priority": "low"},
                    {"type": "quarterly_review", "delay_days": 90, "priority": "medium"}
                ]
            # Standard touch for moderate activity
            else:
                schedule['frequency'] = "bi_weekly"
                schedule['check_ins'] = [
                    {"type": "engagement_prompt", "delay_days": 14, "priority": "medium"},
                    {"type": "value_share", "delay_days": 28, "priority": "low"},
                    {"type": "satisfaction_check", "delay_days": 42, "priority": "medium"}
                ]

        elif strategy == "milestone_based":
            # Check-ins based on customer milestones
            schedule['frequency'] = "milestone"
            schedule['check_ins'] = [
                {"type": "post_onboarding", "delay_days": 30, "priority": "high"},
                {"type": "first_renewal", "delay_days": 180, "priority": "high"},
                {"type": "anniversary", "delay_days": 365, "priority": "medium"}
            ]

        return schedule

    async def _schedule_single_checkin(
        self,
        sequence_id: str,
        customer_id: str,
        check_in: dict,
        cursor: Any
    ) -> str:
        """Schedule a single check-in"""
        check_in_id = str(uuid.uuid4())
        scheduled_date = datetime.now(timezone.utc) + timedelta(days=check_in['delay_days'])

        cursor.execute("""
            INSERT INTO ai_automated_checkins (
                id, sequence_id, customer_id,
                checkin_type, scheduled_date, priority,
                status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            check_in_id, sequence_id, customer_id,
            check_in['type'], scheduled_date,
            check_in['priority'], 'scheduled'
        ))

        return check_in_id

    async def execute_due_checkins(self) -> list[dict]:
        """Execute all due check-ins"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get due check-ins
            cursor.execute("""
                SELECT * FROM ai_automated_checkins
                WHERE status = 'scheduled'
                AND scheduled_date <= NOW()
                ORDER BY priority DESC, scheduled_date
                LIMIT 50
            """)

            due_checkins = cursor.fetchall()
            results = []

            for checkin in due_checkins:
                result = await self._execute_checkin(checkin, cursor)
                results.append(result)

            conn.commit()

            return results

        except Exception as e:
            logger.error(f"Error executing due check-ins: {e}")
            if conn:
                conn.rollback()
            return []

    async def _execute_checkin(self, checkin: dict, cursor: Any) -> dict:
        """Execute a single check-in"""
        try:
            # Generate personalized check-in content
            content = self._generate_checkin_content(checkin['checkin_type'])

            # Update status
            cursor.execute("""
                UPDATE ai_automated_checkins
                SET status = 'completed',
                    executed_at = NOW(),
                    content = %s
                WHERE id = %s
            """, (json.dumps(content), checkin['id']))

            return {
                "checkin_id": checkin['id'],
                "customer_id": checkin['customer_id'],
                "type": checkin['checkin_type'],
                "status": "completed",
                "content": content
            }

        except Exception as e:
            logger.error(f"Error executing check-in: {e}")
            cursor.execute("""
                UPDATE ai_automated_checkins
                SET status = 'failed', error_message = %s
                WHERE id = %s
            """, (str(e), checkin['id']))
            return {"checkin_id": checkin['id'], "status": "failed", "error": str(e)}

    def _generate_checkin_content(self, checkin_type: str) -> dict:
        """Generate content for check-in based on type"""
        content_templates = {
            "satisfaction_check": {
                "subject": "How are we doing?",
                "body": "We'd love to hear your feedback on our service...",
                "action": "Take 2-minute survey"
            },
            "engagement_prompt": {
                "subject": "We'd love to help",
                "body": "Have you had a chance to try our latest features?",
                "action": "Schedule a demo"
            },
            "value_reminder": {
                "subject": "Your success matters to us",
                "body": "Here's how we've helped you save time and money...",
                "action": "View your impact report"
            },
            "post_onboarding": {
                "subject": "How's your experience so far?",
                "body": "We want to make sure you're getting the most value...",
                "action": "Share feedback"
            }
        }

        return content_templates.get(checkin_type, {
            "subject": "Checking in",
            "body": "We wanted to reach out...",
            "action": "Reply to this message"
        })


class SupportEscalationManager:
    """Automated support escalation based on customer issues and urgency"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def analyze_and_escalate(
        self,
        customer_id: str,
        issue_data: dict,
        context: dict
    ) -> dict:
        """Analyze issue and determine if escalation is needed"""
        try:
            # Analyze issue severity
            severity_analysis = self._analyze_severity(issue_data, context)

            # Check escalation criteria
            should_escalate = self._should_escalate(severity_analysis, context)

            if should_escalate:
                escalation = await self._create_escalation(
                    customer_id, issue_data, severity_analysis
                )
                return escalation
            else:
                return {
                    "escalated": False,
                    "severity": severity_analysis['level'],
                    "recommended_action": "standard_support",
                    "reason": "Issue within normal support parameters"
                }

        except Exception as e:
            logger.error(f"Error analyzing escalation: {e}")
            return {"error": str(e)}

    def _analyze_severity(self, issue_data: dict, context: dict) -> dict:
        """Analyze issue severity"""
        severity_score = 0
        factors = []

        # Factor 1: Issue type
        critical_keywords = ["down", "broken", "urgent", "critical", "emergency", "loss"]
        issue_description = issue_data.get('description', '').lower()

        if any(keyword in issue_description for keyword in critical_keywords):
            severity_score += 30
            factors.append("Critical keywords detected")

        # Factor 2: Customer value/tier
        customer_tier = context.get('customer_tier', 'standard')
        if customer_tier == 'enterprise':
            severity_score += 20
            factors.append("Enterprise customer")
        elif customer_tier == 'premium':
            severity_score += 10
            factors.append("Premium customer")

        # Factor 3: Impact scope
        impact = issue_data.get('impact', 'individual')
        if impact == 'organization_wide':
            severity_score += 25
            factors.append("Organization-wide impact")
        elif impact == 'team':
            severity_score += 15
            factors.append("Team-level impact")

        # Factor 4: Business impact
        if issue_data.get('blocks_revenue', False):
            severity_score += 25
            factors.append("Blocking revenue")

        # Factor 5: Previous escalations
        previous_escalations = context.get('recent_escalations', 0)
        if previous_escalations >= 2:
            severity_score += 15
            factors.append("Multiple recent escalations")

        # Determine severity level
        if severity_score >= 70:
            level = "critical"
        elif severity_score >= 50:
            level = "high"
        elif severity_score >= 30:
            level = "medium"
        else:
            level = "low"

        return {
            "score": severity_score,
            "level": level,
            "factors": factors
        }

    def _should_escalate(self, severity: dict, context: dict) -> bool:
        """Determine if issue should be escalated"""
        # Auto-escalate critical issues
        if severity['level'] == 'critical':
            return True

        # Escalate high severity for premium/enterprise
        if severity['level'] == 'high' and context.get('customer_tier') in ['enterprise', 'premium']:
            return True

        # Escalate if SLA at risk
        if context.get('sla_at_risk', False):
            return True

        # Escalate if repeated issues
        if context.get('recent_escalations', 0) >= 2:
            return True

        return False

    async def _create_escalation(
        self,
        customer_id: str,
        issue_data: dict,
        severity: dict
    ) -> dict:
        """Create an escalation record"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            escalation_id = str(uuid.uuid4())

            # Determine escalation tier
            tier = self._get_escalation_tier(severity['level'])

            # Assign to appropriate team/person
            assignment = self._get_assignment(severity['level'], issue_data)

            # Create escalation
            cursor.execute("""
                INSERT INTO ai_support_escalations (
                    id, customer_id, issue_data,
                    severity_level, severity_score,
                    escalation_tier, assigned_to,
                    assigned_team, status, priority,
                    created_at, sla_deadline
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s)
            """, (
                escalation_id, customer_id,
                json.dumps(issue_data),
                severity['level'], severity['score'],
                tier, assignment['person'],
                assignment['team'], 'escalated',
                severity['level'],
                datetime.now(timezone.utc) + timedelta(hours=tier['sla_hours'])
            ))

            # Send escalation notifications
            await self._send_escalation_notifications(
                escalation_id, customer_id, severity, assignment, cursor
            )

            conn.commit()

            return {
                "escalated": True,
                "escalation_id": escalation_id,
                "severity": severity['level'],
                "tier": tier['name'],
                "assigned_to": assignment['person'],
                "assigned_team": assignment['team'],
                "sla_hours": tier['sla_hours'],
                "factors": severity['factors'],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating escalation: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    def _get_escalation_tier(self, severity_level: str) -> dict:
        """Get escalation tier based on severity"""
        tiers = {
            "critical": {"name": "Tier 3", "sla_hours": 2, "level": 3},
            "high": {"name": "Tier 2", "sla_hours": 8, "level": 2},
            "medium": {"name": "Tier 1", "sla_hours": 24, "level": 1},
            "low": {"name": "Standard", "sla_hours": 48, "level": 0}
        }
        return tiers.get(severity_level, tiers["low"])

    def _get_assignment(self, severity_level: str, issue_data: dict) -> dict:
        """Determine assignment for escalation"""
        assignments = {
            "critical": {
                "person": "VP Customer Success",
                "team": "Executive Support"
            },
            "high": {
                "person": "Senior Support Engineer",
                "team": "Advanced Support"
            },
            "medium": {
                "person": "Support Team Lead",
                "team": "Support Team"
            },
            "low": {
                "person": "Support Representative",
                "team": "Support Team"
            }
        }

        return assignments.get(severity_level, assignments["low"])

    async def _send_escalation_notifications(
        self,
        escalation_id: str,
        customer_id: str,
        severity: dict,
        assignment: dict,
        cursor: Any
    ) -> None:
        """Send notifications for escalation"""
        # Record notification
        cursor.execute("""
            INSERT INTO ai_escalation_notifications (
                id, escalation_id, recipient,
                notification_type, sent_at
            ) VALUES (%s, %s, %s, %s, NOW())
        """, (
            str(uuid.uuid4()), escalation_id,
            assignment['person'], 'email'
        ))

        logger.info(f"Escalation {escalation_id} assigned to {assignment['person']} ({assignment['team']})")


class PerformanceTracker:
    """Track and optimize follow-up performance"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def track_performance(
        self,
        sequence_id: str,
        metrics: dict
    ) -> None:
        """Track performance metrics for follow-up sequence"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_followup_metrics (
                    id, sequence_id, metric_date,
                    sent_count, delivered_count, opened_count,
                    responded_count, conversion_count,
                    response_rate, conversion_rate,
                    avg_response_time, created_at
                ) VALUES (%s, %s, CURRENT_DATE, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (sequence_id, metric_date)
                DO UPDATE SET
                    sent_count = EXCLUDED.sent_count,
                    delivered_count = EXCLUDED.delivered_count,
                    opened_count = EXCLUDED.opened_count,
                    responded_count = EXCLUDED.responded_count,
                    conversion_count = EXCLUDED.conversion_count,
                    response_rate = EXCLUDED.response_rate,
                    conversion_rate = EXCLUDED.conversion_rate,
                    avg_response_time = EXCLUDED.avg_response_time,
                    updated_at = NOW()
            """, (
                str(uuid.uuid4()), sequence_id,
                metrics.get('sent_count', 0),
                metrics.get('delivered_count', 0),
                metrics.get('opened_count', 0),
                metrics.get('responded_count', 0),
                metrics.get('conversion_count', 0),
                metrics.get('response_rate', 0),
                metrics.get('conversion_rate', 0),
                metrics.get('avg_response_time', 0)
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Error tracking performance: {e}")
            if conn:
                conn.rollback()

    async def get_performance_insights(
        self,
        sequence_id: Optional[str] = None,
        followup_type: Optional[str] = None
    ) -> dict:
        """Get performance insights for follow-ups"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            if sequence_id:
                cursor.execute("""
                    SELECT * FROM ai_followup_metrics
                    WHERE sequence_id = %s
                    ORDER BY metric_date DESC
                    LIMIT 30
                """, (sequence_id,))
            else:
                cursor.execute("""
                    SELECT
                        AVG(response_rate) as avg_response_rate,
                        AVG(conversion_rate) as avg_conversion_rate,
                        AVG(avg_response_time) as avg_response_time,
                        SUM(sent_count) as total_sent,
                        SUM(responded_count) as total_responded
                    FROM ai_followup_metrics
                    WHERE metric_date >= CURRENT_DATE - INTERVAL '30 days'
                """)

            results = cursor.fetchall()

            return {
                "performance": results,
                "insights": self._generate_insights(results)
            }

        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
            return {}

    def _generate_insights(self, performance_data: list[dict]) -> list[str]:
        """Generate actionable insights from performance data"""
        insights = []

        if not performance_data:
            return ["No data available for analysis"]

        # Analyze trends
        if len(performance_data) > 1:
            if performance_data[0].get('response_rate', 0) > performance_data[-1].get('response_rate', 0):
                insights.append("Response rates are improving over time")
            else:
                insights.append("Consider A/B testing different content approaches")

        # Check performance thresholds
        avg_response_rate = sum(d.get('response_rate', 0) for d in performance_data) / len(performance_data)
        if avg_response_rate < 20:
            insights.append("Response rates are below industry average - review targeting and content")
        elif avg_response_rate > 40:
            insights.append("Excellent response rates - maintain current strategy")

        return insights


# Singleton instance
_followup_system = None

def get_intelligent_followup_system():
    """Get singleton instance of intelligent follow-up system"""
    global _followup_system
    if _followup_system is None:
        _followup_system = IntelligentFollowUpSystem()
    return _followup_system


# Export main components
__all__ = [
    'IntelligentFollowUpSystem',
    'FollowUpType',
    'FollowUpPriority',
    'FollowUpStatus',
    'DeliveryChannel',
    'ResponseType',
    'TimingOptimizer',
    'ContentGenerator',
    'ChannelSelector',
    'ResponseAnalyzer',
    'PerformanceTracker',
    'AutomatedCheckInScheduler',
    'SupportEscalationManager',
    'get_intelligent_followup_system'
]
