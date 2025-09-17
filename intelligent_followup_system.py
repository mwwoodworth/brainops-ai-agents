#!/usr/bin/env python3
"""
Intelligent Follow-up System - Task 17
Automated follow-up orchestration with AI-powered timing, personalization, and multi-channel delivery
"""

import os
import json
import logging
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import aiohttp
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": os.getenv("DB_PORT", "5432")
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
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    async def create_followup_sequence(
        self,
        followup_type: FollowUpType,
        entity_id: str,
        entity_type: str,
        context: Dict[str, Any],
        priority: FollowUpPriority = FollowUpPriority.MEDIUM,
        custom_rules: Optional[Dict] = None
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
        context: Dict,
        priority: FollowUpPriority
    ) -> Dict:
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
        strategy: Dict,
        context: Dict
    ) -> List[Dict]:
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

    def _get_touchpoint_templates(self, followup_type: FollowUpType) -> List[Dict]:
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
        touchpoint: Dict,
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

    async def execute_scheduled_followups(self) -> List[Dict]:
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

    async def _execute_followup(self, followup: Dict, cursor: Any) -> Dict:
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
        content: Dict
    ) -> Dict:
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
        response_data: Optional[Dict] = None
    ) -> Dict:
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
    ) -> Dict:
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
        entity_profile: Dict,
        followup_type: FollowUpType,
        historical_data: List[Dict]
    ) -> Dict:
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

    def _analyze_response_patterns(self, historical_data: List[Dict]) -> Dict:
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
        template: Dict,
        context: Dict,
        personalization: Dict
    ) -> Dict:
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
        context: Dict,
        tone: str
    ) -> Dict:
        """Generate AI-powered content"""
        # This would integrate with OpenAI or similar
        # For now, return template-based content
        return {
            "subject": subject,
            "body": f"Hello {context.get('name', 'there')},\n\n" +
                   f"We wanted to follow up regarding {context.get('topic', 'your inquiry')}.\n\n" +
                   f"Best regards,\nThe Team",
            "tone": tone,
            "personalized": True
        }

    def _generate_template_content(self, template: Dict, context: Dict) -> Dict:
        """Generate template-based content"""
        return {
            "subject": template['subject'],
            "body": template.get('body', 'Follow-up message'),
            "tone": template.get('tone', 'professional')
        }

    def _apply_personalization(self, content: Dict, context: Dict) -> Dict:
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

    async def _get_available_channels(self, entity_id: str) -> List[str]:
        """Get available channels for entity"""
        # This would check actual contact information
        # For now, return common channels
        return ["email", "sms", "in_app"]

    def _get_channel_priority(self, followup_type: str) -> List[str]:
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
        response_data: Optional[Dict]
    ) -> Dict:
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
        response_data: Optional[Dict]
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

    def _determine_urgency(self, response_data: Optional[Dict]) -> str:
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

    async def _analyze_text(self, text: str) -> Dict:
        """Analyze text content using AI"""
        # This would use NLP/AI for deeper analysis
        # For now, return basic analysis
        return {
            "length": len(text),
            "questions_asked": text.count('?'),
            "exclamations": text.count('!'),
            "key_phrases": []  # Would extract key phrases
        }


class PerformanceTracker:
    """Track and optimize follow-up performance"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    async def track_performance(
        self,
        sequence_id: str,
        metrics: Dict
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
    ) -> Dict:
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

    def _generate_insights(self, performance_data: List[Dict]) -> List[str]:
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
    'get_intelligent_followup_system'
]