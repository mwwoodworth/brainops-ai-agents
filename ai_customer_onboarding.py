#!/usr/bin/env python3
"""
AI-Powered Customer Onboarding System - Task 18
Intelligent, personalized customer onboarding with automated workflows and progress tracking
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - NO hardcoded fallback credentials
def _get_db_config():
    """Get database configuration from environment variables."""
    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    missing = []
    if not db_host:
        missing.append("DB_HOST")
    if not db_name:
        missing.append("DB_NAME")
    if not db_user:
        missing.append("DB_USER")
    if not db_password:
        missing.append("DB_PASSWORD")

    if missing:
        raise RuntimeError(
            f"Required environment variables not set: {', '.join(missing)}. "
            "Set these variables before using AI customer onboarding."
        )

    return {
        "host": db_host,
        "database": db_name,
        "user": db_user,
        "password": db_password,
        "port": int(db_port)
    }


class OnboardingStage(Enum):
    REGISTRATION = "registration"
    VERIFICATION = "verification"
    PROFILE_SETUP = "profile_setup"
    PREFERENCES = "preferences"
    INTEGRATION = "integration"
    TRAINING = "training"
    ACTIVATION = "activation"
    FIRST_VALUE = "first_value"
    ENGAGEMENT = "engagement"
    RETENTION = "retention"

class OnboardingStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    STUCK = "stuck"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    REACTIVATED = "reactivated"

class CustomerSegment(Enum):
    ENTERPRISE = "enterprise"
    SMB = "smb"
    STARTUP = "startup"
    INDIVIDUAL = "individual"
    TRIAL = "trial"
    FREEMIUM = "freemium"
    PREMIUM = "premium"

class OnboardingAction(Enum):
    SEND_EMAIL = "send_email"
    SEND_SMS = "send_sms"
    SCHEDULE_CALL = "schedule_call"
    ASSIGN_CSM = "assign_csm"
    CREATE_TASK = "create_task"
    UNLOCK_FEATURE = "unlock_feature"
    PROVIDE_RESOURCE = "provide_resource"
    OFFER_TRAINING = "offer_training"
    REQUEST_FEEDBACK = "request_feedback"
    ESCALATE = "escalate"

class InterventionType(Enum):
    AUTOMATED = "automated"
    SEMI_AUTOMATED = "semi_automated"
    HUMAN_REQUIRED = "human_required"
    AI_COACHED = "ai_coached"
    PEER_ASSISTED = "peer_assisted"

class AICustomerOnboarding:
    """Main AI-powered customer onboarding system"""

    def __init__(self):
        self.conn = None
        self.ai_model = "gpt-4"
        self.journey_designer = JourneyDesigner()
        self.progress_tracker = ProgressTracker()
        self.personalization_engine = PersonalizationEngine()
        self.intervention_manager = InterventionManager()
        self.success_predictor = SuccessPredictor()
        self.content_generator = OnboardingContentGenerator()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def create_onboarding_journey(
        self,
        customer_id: str,
        customer_data: Dict[str, Any],
        segment: CustomerSegment,
        custom_requirements: Optional[Dict] = None
    ) -> str:
        """Create personalized onboarding journey for customer"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            journey_id = str(uuid.uuid4())

            # Analyze customer profile
            profile_analysis = await self._analyze_customer_profile(
                customer_data, segment
            )

            # Design personalized journey
            journey_plan = await self.journey_designer.design_journey(
                segment, profile_analysis, custom_requirements
            )

            # Create journey record
            cursor.execute("""
                INSERT INTO ai_onboarding_journeys (
                    id, customer_id, segment, profile_analysis,
                    journey_plan, custom_requirements, status,
                    expected_duration_days, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                journey_id, customer_id, segment.value,
                json.dumps(profile_analysis), json.dumps(journey_plan),
                json.dumps(custom_requirements or {}),
                OnboardingStatus.NOT_STARTED.value,
                journey_plan.get('expected_duration_days', 14)
            ))

            # Create stages for journey
            for stage_config in journey_plan['stages']:
                await self._create_journey_stage(
                    journey_id, stage_config, cursor
                )

            # Schedule initial actions
            await self._schedule_initial_actions(journey_id, cursor)

            conn.commit()

            logger.info(f"Created onboarding journey {journey_id} for customer {customer_id}")
            return journey_id

        except Exception as e:
            logger.error(f"Error creating onboarding journey: {e}")
            if conn:
                conn.rollback()
            raise

    async def _analyze_customer_profile(
        self,
        customer_data: Dict,
        segment: CustomerSegment
    ) -> Dict:
        """Analyze customer profile for personalization"""
        analysis = {
            "segment": segment.value,
            "industry": customer_data.get('industry', 'general'),
            "company_size": customer_data.get('company_size', 'unknown'),
            "technical_level": self._assess_technical_level(customer_data),
            "goals": customer_data.get('goals', []),
            "challenges": customer_data.get('challenges', []),
            "preferred_pace": self._determine_pace(segment),
            "communication_preferences": {
                "channel": customer_data.get('preferred_channel', 'email'),
                "frequency": customer_data.get('contact_frequency', 'moderate'),
                "timezone": customer_data.get('timezone', 'UTC')
            },
            "risk_factors": self._identify_risk_factors(customer_data),
            "success_indicators": self._define_success_indicators(segment)
        }

        return analysis

    def _assess_technical_level(self, customer_data: Dict) -> str:
        """Assess customer's technical sophistication"""
        indicators = {
            'has_api_experience': 'advanced',
            'has_integration_needs': 'intermediate',
            'is_first_time_user': 'beginner'
        }

        for key, level in indicators.items():
            if customer_data.get(key):
                return level

        return 'intermediate'

    def _determine_pace(self, segment: CustomerSegment) -> str:
        """Determine appropriate onboarding pace"""
        pace_map = {
            CustomerSegment.ENTERPRISE: 'measured',
            CustomerSegment.SMB: 'moderate',
            CustomerSegment.STARTUP: 'fast',
            CustomerSegment.INDIVIDUAL: 'flexible',
            CustomerSegment.TRIAL: 'accelerated',
            CustomerSegment.FREEMIUM: 'self_paced'
        }
        return pace_map.get(segment, 'moderate')

    def _identify_risk_factors(self, customer_data: Dict) -> List[str]:
        """Identify potential churn risk factors"""
        risk_factors = []

        if customer_data.get('previous_churn'):
            risk_factors.append('previous_churn_history')
        if customer_data.get('budget_constraints'):
            risk_factors.append('budget_sensitive')
        if customer_data.get('competitor_evaluation'):
            risk_factors.append('actively_comparing')
        if not customer_data.get('has_champion'):
            risk_factors.append('no_internal_champion')

        return risk_factors

    def _define_success_indicators(self, segment: CustomerSegment) -> List[str]:
        """Define success indicators for segment"""
        base_indicators = [
            'account_activated',
            'first_value_achieved',
            'regular_usage_pattern',
            'feature_adoption'
        ]

        segment_specific = {
            CustomerSegment.ENTERPRISE: ['team_onboarded', 'integration_complete'],
            CustomerSegment.SMB: ['workflow_established', 'roi_demonstrated'],
            CustomerSegment.INDIVIDUAL: ['habit_formed', 'goal_achieved']
        }

        return base_indicators + segment_specific.get(segment, [])

    async def _create_journey_stage(
        self,
        journey_id: str,
        stage_config: Dict,
        cursor: Any
    ) -> str:
        """Create a stage in the onboarding journey"""
        stage_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO ai_onboarding_stages (
                id, journey_id, stage_type, stage_order,
                name, description, required_actions,
                success_criteria, estimated_duration_hours,
                is_mandatory, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            stage_id, journey_id, stage_config['type'],
            stage_config['order'], stage_config['name'],
            stage_config['description'],
            json.dumps(stage_config['required_actions']),
            json.dumps(stage_config['success_criteria']),
            stage_config.get('duration_hours', 24),
            stage_config.get('is_mandatory', True)
        ))

        # Create tasks for stage
        for task_config in stage_config.get('tasks', []):
            await self._create_stage_task(stage_id, task_config, cursor)

        return stage_id

    async def _create_stage_task(
        self,
        stage_id: str,
        task_config: Dict,
        cursor: Any
    ) -> str:
        """Create a task within an onboarding stage"""
        task_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO ai_onboarding_tasks (
                id, stage_id, task_type, title,
                description, instructions, resources,
                is_automated, automation_config,
                estimated_minutes, points_value,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            task_id, stage_id, task_config['type'],
            task_config['title'], task_config['description'],
            json.dumps(task_config.get('instructions', [])),
            json.dumps(task_config.get('resources', [])),
            task_config.get('is_automated', False),
            json.dumps(task_config.get('automation_config', {})),
            task_config.get('estimated_minutes', 15),
            task_config.get('points_value', 10)
        ))

        return task_id

    async def _schedule_initial_actions(
        self,
        journey_id: str,
        cursor: Any
    ) -> None:
        """Schedule initial onboarding actions"""
        # Welcome message
        action_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO ai_onboarding_actions (
                id, journey_id, action_type,
                scheduled_at, content, status,
                created_at
            ) VALUES (%s, %s, %s, NOW(), %s, %s, NOW())
        """, (
            action_id, journey_id, OnboardingAction.SEND_EMAIL.value,
            json.dumps({
                "subject": "Welcome! Let's get you started",
                "template": "welcome_email"
            }),
            OnboardingStatus.IN_PROGRESS.value
        ))

    async def track_progress(
        self,
        journey_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> Dict:
        """Track progress in onboarding journey"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Record event
            event_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_onboarding_events (
                    id, journey_id, event_type,
                    event_data, created_at
                ) VALUES (%s, %s, %s, %s, NOW())
            """, (
                event_id, journey_id, event_type,
                json.dumps(event_data)
            ))

            # Update progress
            progress = await self.progress_tracker.update_progress(
                journey_id, event_type, event_data, cursor
            )

            # Check for interventions needed
            if progress['completion_rate'] < progress['expected_rate']:
                intervention = await self.intervention_manager.assess_intervention(
                    journey_id, progress, cursor
                )
                if intervention:
                    await self._execute_intervention(intervention, cursor)

            # Predict success
            success_probability = await self.success_predictor.predict_success(
                journey_id, progress
            )

            conn.commit()

            return {
                "event_id": event_id,
                "progress": progress,
                "success_probability": success_probability,
                "intervention_triggered": bool(intervention) if 'intervention' in locals() else False
            }

        except Exception as e:
            logger.error(f"Error tracking progress: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    async def _execute_intervention(
        self,
        intervention: Dict,
        cursor: Any
    ) -> None:
        """Execute an intervention to help customer progress"""
        intervention_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO ai_onboarding_interventions (
                id, journey_id, intervention_type,
                reason, actions, status,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
        """, (
            intervention_id, intervention['journey_id'],
            intervention['type'], intervention['reason'],
            json.dumps(intervention['actions']),
            'executed'
        ))

        # Execute intervention actions
        for action in intervention['actions']:
            await self._execute_action(action, cursor)

    async def _execute_action(
        self,
        action: Dict,
        cursor: Any
    ) -> None:
        """Execute a specific onboarding action"""
        # This would integrate with various systems
        # For now, record the action
        logger.info(f"Executing action: {action['type']}")

    async def get_journey_analytics(
        self,
        segment: Optional[CustomerSegment] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict:
        """Get onboarding analytics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query conditions
            conditions = []
            params = []

            if segment:
                conditions.append("segment = %s")
                params.append(segment.value)

            if date_from:
                conditions.append("created_at >= %s")
                params.append(date_from)

            if date_to:
                conditions.append("created_at <= %s")
                params.append(date_to)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # Get overall metrics
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_journeys,
                    AVG(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100 as completion_rate,
                    AVG(time_to_complete_days) as avg_completion_days,
                    AVG(progress_score) as avg_progress_score,
                    COUNT(DISTINCT segment) as segments_served
                FROM ai_onboarding_journeys
                WHERE {where_clause}
            """, params)

            overall = cursor.fetchone()

            # Get stage performance
            cursor.execute(f"""
                SELECT
                    os.stage_type,
                    COUNT(*) as total_stages,
                    AVG(CASE WHEN os.status = 'completed' THEN 1 ELSE 0 END) * 100 as completion_rate,
                    AVG(os.time_spent_hours) as avg_time_hours
                FROM ai_onboarding_journeys oj
                JOIN ai_onboarding_stages os ON os.journey_id = oj.id
                WHERE {where_clause}
                GROUP BY os.stage_type
                ORDER BY AVG(os.stage_order)
            """, params)

            stage_performance = cursor.fetchall()

            # Get intervention effectiveness
            cursor.execute(f"""
                SELECT
                    intervention_type,
                    COUNT(*) as intervention_count,
                    AVG(success_rate) * 100 as effectiveness_rate
                FROM ai_onboarding_interventions
                GROUP BY intervention_type
            """)

            interventions = cursor.fetchall()

            return {
                "overall": overall,
                "by_stage": stage_performance,
                "interventions": interventions,
                "period": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None
                }
            }

        except Exception as e:
            logger.error(f"Error getting journey analytics: {e}")
            return {}

    async def personalize_experience(
        self,
        journey_id: str,
        interaction_data: Dict
    ) -> Dict:
        """Personalize onboarding experience based on behavior"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get journey context
            cursor.execute("""
                SELECT * FROM ai_onboarding_journeys
                WHERE id = %s
            """, (journey_id,))

            journey = cursor.fetchone()

            if not journey:
                return {"error": "Journey not found"}

            # Generate personalization
            personalization = await self.personalization_engine.generate_personalization(
                journey, interaction_data
            )

            # Update journey with personalization
            cursor.execute("""
                UPDATE ai_onboarding_journeys
                SET personalization = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (json.dumps(personalization), journey_id))

            # Apply personalization to upcoming actions
            await self._apply_personalization(
                journey_id, personalization, cursor
            )

            conn.commit()

            return personalization

        except Exception as e:
            logger.error(f"Error personalizing experience: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    async def _apply_personalization(
        self,
        journey_id: str,
        personalization: Dict,
        cursor: Any
    ) -> None:
        """Apply personalization to journey actions"""
        # Update scheduled actions with personalized content
        if personalization.get('content_adjustments'):
            cursor.execute("""
                UPDATE ai_onboarding_actions
                SET content = content || %s
                WHERE journey_id = %s
                AND status = 'scheduled'
            """, (json.dumps(personalization['content_adjustments']), journey_id))

        # Adjust pacing if needed
        if personalization.get('pace_adjustment'):
            cursor.execute("""
                UPDATE ai_onboarding_actions
                SET scheduled_at = scheduled_at + INTERVAL '%s hours'
                WHERE journey_id = %s
                AND status = 'scheduled'
            """, (personalization['pace_adjustment'], journey_id))


class JourneyDesigner:
    """Design personalized onboarding journeys"""

    def __init__(self):
        self.stage_templates = self._load_stage_templates()

    def _load_stage_templates(self) -> Dict:
        """Load stage templates for different segments"""
        return {
            CustomerSegment.ENTERPRISE: [
                {"type": "registration", "order": 1, "name": "Account Setup", "duration_hours": 1},
                {"type": "verification", "order": 2, "name": "Enterprise Verification", "duration_hours": 48},
                {"type": "profile_setup", "order": 3, "name": "Organization Profile", "duration_hours": 24},
                {"type": "integration", "order": 4, "name": "System Integration", "duration_hours": 72},
                {"type": "training", "order": 5, "name": "Team Training", "duration_hours": 120},
                {"type": "activation", "order": 6, "name": "Full Activation", "duration_hours": 48}
            ],
            CustomerSegment.SMB: [
                {"type": "registration", "order": 1, "name": "Quick Start", "duration_hours": 0.5},
                {"type": "profile_setup", "order": 2, "name": "Business Profile", "duration_hours": 2},
                {"type": "preferences", "order": 3, "name": "Configure Preferences", "duration_hours": 1},
                {"type": "training", "order": 4, "name": "Getting Started Tutorial", "duration_hours": 4},
                {"type": "first_value", "order": 5, "name": "First Success", "duration_hours": 24}
            ],
            CustomerSegment.INDIVIDUAL: [
                {"type": "registration", "order": 1, "name": "Sign Up", "duration_hours": 0.25},
                {"type": "preferences", "order": 2, "name": "Personalize", "duration_hours": 0.5},
                {"type": "training", "order": 3, "name": "Quick Tour", "duration_hours": 1},
                {"type": "first_value", "order": 4, "name": "First Achievement", "duration_hours": 24}
            ]
        }

    async def design_journey(
        self,
        segment: CustomerSegment,
        profile_analysis: Dict,
        custom_requirements: Optional[Dict]
    ) -> Dict:
        """Design personalized journey based on segment and profile"""
        # Get base template
        base_stages = self.stage_templates.get(
            segment,
            self.stage_templates[CustomerSegment.SMB]
        )

        # Customize based on profile
        stages = []
        for stage_template in base_stages:
            stage = dict(stage_template)

            # Add required actions
            stage['required_actions'] = self._get_required_actions(
                stage['type'], profile_analysis
            )

            # Add success criteria
            stage['success_criteria'] = self._get_success_criteria(
                stage['type'], segment
            )

            # Add tasks
            stage['tasks'] = self._generate_tasks(
                stage['type'], profile_analysis
            )

            stages.append(stage)

        # Apply custom requirements
        if custom_requirements:
            stages = self._apply_custom_requirements(stages, custom_requirements)

        # Calculate expected duration
        total_hours = sum(s.get('duration_hours', 24) for s in stages)
        expected_days = max(7, int(total_hours / 24) + 3)  # Add buffer

        return {
            "stages": stages,
            "expected_duration_days": expected_days,
            "optimization_notes": self._generate_optimization_notes(profile_analysis),
            "risk_mitigation": self._generate_risk_mitigation(profile_analysis)
        }

    def _get_required_actions(self, stage_type: str, profile: Dict) -> List[str]:
        """Get required actions for stage"""
        base_actions = {
            "registration": ["create_account", "verify_email"],
            "verification": ["verify_identity", "confirm_organization"],
            "profile_setup": ["complete_profile", "add_logo", "set_preferences"],
            "integration": ["connect_systems", "import_data", "test_integration"],
            "training": ["complete_tutorial", "attend_webinar", "review_resources"],
            "activation": ["first_transaction", "invite_team", "configure_workflows"]
        }

        return base_actions.get(stage_type, [])

    def _get_success_criteria(
        self,
        stage_type: str,
        segment: CustomerSegment
    ) -> Dict:
        """Get success criteria for stage"""
        return {
            "completion_threshold": 0.8,
            "required_actions_complete": True,
            "time_limit_hours": 72,
            "engagement_minimum": self._get_engagement_minimum(segment)
        }

    def _get_engagement_minimum(self, segment: CustomerSegment) -> int:
        """Get minimum engagement actions for segment"""
        engagement_map = {
            CustomerSegment.ENTERPRISE: 10,
            CustomerSegment.SMB: 5,
            CustomerSegment.INDIVIDUAL: 3,
            CustomerSegment.TRIAL: 2
        }
        return engagement_map.get(segment, 5)

    def _generate_tasks(self, stage_type: str, profile: Dict) -> List[Dict]:
        """Generate tasks for stage based on profile"""
        tasks = []

        # Add standard tasks for stage type
        if stage_type == "registration":
            tasks.append({
                "type": "form_completion",
                "title": "Complete registration",
                "description": "Fill in your account details",
                "is_automated": False,
                "estimated_minutes": 5
            })

        elif stage_type == "training":
            # Customize training based on technical level
            if profile.get('technical_level') == 'beginner':
                tasks.append({
                    "type": "video_tutorial",
                    "title": "Watch getting started video",
                    "description": "Learn the basics in 10 minutes",
                    "resources": ["intro_video_url"],
                    "estimated_minutes": 10
                })
            else:
                tasks.append({
                    "type": "documentation",
                    "title": "Review quick start guide",
                    "description": "Technical setup documentation",
                    "resources": ["docs_url"],
                    "estimated_minutes": 15
                })

        return tasks

    def _apply_custom_requirements(
        self,
        stages: List[Dict],
        custom_requirements: Dict
    ) -> List[Dict]:
        """Apply custom requirements to journey stages"""
        # Add custom stages if needed
        if custom_requirements.get('additional_stages'):
            for custom_stage in custom_requirements['additional_stages']:
                stages.append(custom_stage)

        # Modify existing stages
        if custom_requirements.get('stage_modifications'):
            for stage in stages:
                if stage['type'] in custom_requirements['stage_modifications']:
                    stage.update(custom_requirements['stage_modifications'][stage['type']])

        return sorted(stages, key=lambda x: x['order'])

    def _generate_optimization_notes(self, profile: Dict) -> List[str]:
        """Generate optimization notes for journey"""
        notes = []

        if 'beginner' in profile.get('technical_level', ''):
            notes.append("Provide extra support and simplified documentation")

        if profile.get('risk_factors'):
            notes.append("Monitor closely for early churn signals")

        if profile.get('goals'):
            notes.append(f"Focus on achieving: {', '.join(profile['goals'][:3])}")

        return notes

    def _generate_risk_mitigation(self, profile: Dict) -> Dict:
        """Generate risk mitigation strategies"""
        strategies = {}

        for risk in profile.get('risk_factors', []):
            if risk == 'budget_sensitive':
                strategies['budget'] = "Emphasize ROI and value early"
            elif risk == 'actively_comparing':
                strategies['competition'] = "Highlight unique differentiators"
            elif risk == 'no_internal_champion':
                strategies['champion'] = "Identify and nurture potential champion"

        return strategies


class ProgressTracker:
    """Track and analyze onboarding progress"""

    async def update_progress(
        self,
        journey_id: str,
        event_type: str,
        event_data: Dict,
        cursor: Any
    ) -> Dict:
        """Update journey progress based on event"""
        # Get current progress
        cursor.execute("""
            SELECT * FROM ai_onboarding_journeys
            WHERE id = %s
        """, (journey_id,))

        journey = cursor.fetchone()

        # Calculate new progress
        progress = {
            "completion_rate": self._calculate_completion_rate(journey, event_type),
            "expected_rate": self._calculate_expected_rate(journey),
            "current_stage": self._determine_current_stage(journey, event_type),
            "time_spent_hours": self._calculate_time_spent(journey),
            "engagement_score": self._calculate_engagement_score(journey, event_data),
            "velocity": self._calculate_velocity(journey)
        }

        # Update journey progress
        cursor.execute("""
            UPDATE ai_onboarding_journeys
            SET progress_score = %s,
                current_stage = %s,
                last_activity = NOW(),
                updated_at = NOW()
            WHERE id = %s
        """, (progress['completion_rate'], progress['current_stage'], journey_id))

        return progress

    def _calculate_completion_rate(self, journey: Dict, event_type: str) -> float:
        """Calculate overall completion rate"""
        # This would involve checking completed stages/tasks
        # Simplified for demonstration
        base_rate = journey.get('progress_score', 0)

        event_values = {
            'task_completed': 0.05,
            'stage_completed': 0.15,
            'milestone_reached': 0.10
        }

        return min(100, base_rate + event_values.get(event_type, 0.01))

    def _calculate_expected_rate(self, journey: Dict) -> float:
        """Calculate expected progress rate"""
        created_at = journey.get('created_at')
        expected_duration = journey.get('expected_duration_days', 14)

        if created_at:
            days_elapsed = (datetime.now(timezone.utc) - created_at).days
            return min(100, (days_elapsed / expected_duration) * 100)

        return 0

    def _determine_current_stage(self, journey: Dict, event_type: str) -> str:
        """Determine current onboarding stage"""
        # This would check actual stage completion
        return journey.get('current_stage', 'registration')

    def _calculate_time_spent(self, journey: Dict) -> float:
        """Calculate total time spent in onboarding"""
        created_at = journey.get('created_at')
        if created_at:
            return (datetime.now(timezone.utc) - created_at).total_seconds() / 3600

        return 0

    def _calculate_engagement_score(
        self,
        journey: Dict,
        event_data: Dict
    ) -> float:
        """Calculate engagement score"""
        # Factor in various engagement metrics
        base_score = 50

        # Add points for different activities
        if event_data.get('page_views', 0) > 10:
            base_score += 10
        if event_data.get('features_used', 0) > 3:
            base_score += 15
        if event_data.get('team_members_invited', 0) > 0:
            base_score += 20
        if event_data.get('integrations_connected', 0) > 0:
            base_score += 15

        return min(100, base_score)

    def _calculate_velocity(self, journey: Dict) -> str:
        """Calculate progress velocity"""
        progress = journey.get('progress_score', 0)
        expected_rate = self._calculate_expected_rate(journey)

        if progress > expected_rate * 1.2:
            return 'fast'
        elif progress > expected_rate * 0.8:
            return 'normal'
        else:
            return 'slow'


class PersonalizationEngine:
    """Generate personalized onboarding experiences"""

    async def generate_personalization(
        self,
        journey: Dict,
        interaction_data: Dict
    ) -> Dict:
        """Generate personalization based on behavior"""
        personalization = {
            "content_adjustments": {},
            "pace_adjustment": 0,
            "channel_preferences": [],
            "recommended_resources": [],
            "coaching_style": "standard"
        }

        # Analyze interaction patterns
        patterns = self._analyze_patterns(interaction_data)

        # Adjust content based on patterns
        if patterns.get('prefers_visual'):
            personalization['content_adjustments']['format'] = 'video_heavy'
            personalization['recommended_resources'].append('video_library')

        if patterns.get('technical_user'):
            personalization['content_adjustments']['depth'] = 'technical'
            personalization['coaching_style'] = 'minimal'

        if patterns.get('needs_more_time'):
            personalization['pace_adjustment'] = 24  # Slow down by 24 hours
            personalization['coaching_style'] = 'supportive'

        # Set channel preferences
        if interaction_data.get('email_engagement_rate', 0) > 0.5:
            personalization['channel_preferences'].append('email')
        if interaction_data.get('in_app_time_minutes', 0) > 30:
            personalization['channel_preferences'].append('in_app')

        return personalization

    def _analyze_patterns(self, interaction_data: Dict) -> Dict:
        """Analyze user interaction patterns"""
        patterns = {}

        # Check for visual preference
        if interaction_data.get('video_views', 0) > interaction_data.get('doc_views', 0):
            patterns['prefers_visual'] = True

        # Check technical level
        if interaction_data.get('api_calls', 0) > 0:
            patterns['technical_user'] = True

        # Check pacing needs
        if interaction_data.get('session_duration_avg', 0) < 5:
            patterns['needs_more_time'] = True

        return patterns


class InterventionManager:
    """Manage interventions for stuck or at-risk customers"""

    async def assess_intervention(
        self,
        journey_id: str,
        progress: Dict,
        cursor: Any
    ) -> Optional[Dict]:
        """Assess if intervention is needed"""
        # Check intervention triggers
        if progress['completion_rate'] < progress['expected_rate'] - 20:
            return await self._create_intervention(
                journey_id,
                InterventionType.AUTOMATED,
                "Behind schedule",
                self._get_acceleration_actions()
            )

        if progress['engagement_score'] < 30:
            return await self._create_intervention(
                journey_id,
                InterventionType.SEMI_AUTOMATED,
                "Low engagement",
                self._get_engagement_actions()
            )

        if progress['velocity'] == 'slow' and progress.get('current_stage') == 'registration':
            return await self._create_intervention(
                journey_id,
                InterventionType.HUMAN_REQUIRED,
                "Stuck at registration",
                self._get_human_touch_actions()
            )

        return None

    async def _create_intervention(
        self,
        journey_id: str,
        intervention_type: InterventionType,
        reason: str,
        actions: List[Dict]
    ) -> Dict:
        """Create intervention plan"""
        return {
            "journey_id": journey_id,
            "type": intervention_type.value,
            "reason": reason,
            "actions": actions,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    def _get_acceleration_actions(self) -> List[Dict]:
        """Get actions to accelerate progress"""
        return [
            {"type": "send_email", "content": "quick_win_tips"},
            {"type": "unlock_feature", "feature": "fast_track_mode"},
            {"type": "schedule_call", "purpose": "remove_blockers"}
        ]

    def _get_engagement_actions(self) -> List[Dict]:
        """Get actions to increase engagement"""
        return [
            {"type": "send_sms", "content": "value_reminder"},
            {"type": "offer_training", "format": "1_on_1_demo"},
            {"type": "provide_resource", "resource": "success_stories"}
        ]

    def _get_human_touch_actions(self) -> List[Dict]:
        """Get human intervention actions"""
        return [
            {"type": "assign_csm", "priority": "high"},
            {"type": "schedule_call", "purpose": "onboarding_assistance"},
            {"type": "create_task", "for": "customer_success_team"}
        ]


class SuccessPredictor:
    """Predict onboarding success probability"""

    async def predict_success(
        self,
        journey_id: str,
        progress: Dict
    ) -> float:
        """Predict probability of successful onboarding"""
        # Base probability
        probability = 0.5

        # Adjust based on progress
        if progress['completion_rate'] > progress['expected_rate']:
            probability += 0.2
        elif progress['completion_rate'] < progress['expected_rate'] - 20:
            probability -= 0.2

        # Adjust based on engagement
        engagement_factor = progress['engagement_score'] / 100
        probability += engagement_factor * 0.3

        # Adjust based on velocity
        velocity_adjustments = {
            'fast': 0.1,
            'normal': 0,
            'slow': -0.1
        }
        probability += velocity_adjustments.get(progress['velocity'], 0)

        return max(0, min(1, probability))


class FeatureAdoptionTracker:
    """Track and analyze feature adoption during onboarding"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def track_feature_usage(
        self,
        journey_id: str,
        customer_id: str,
        feature_name: str,
        usage_context: Dict
    ) -> Dict:
        """Track when a customer uses a feature"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Record feature usage
            usage_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_feature_usage (
                    id, journey_id, customer_id, feature_name,
                    usage_context, first_used_at, last_used_at,
                    usage_count, created_at
                ) VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), 1, NOW())
                ON CONFLICT (customer_id, feature_name)
                DO UPDATE SET
                    last_used_at = NOW(),
                    usage_count = ai_feature_usage.usage_count + 1,
                    usage_context = EXCLUDED.usage_context
                RETURNING *
            """, (
                usage_id, journey_id, customer_id, feature_name,
                json.dumps(usage_context)
            ))

            usage_record = cursor.fetchone()
            conn.commit()

            # Check if this unlocks any achievements
            achievements = await self._check_achievements(
                customer_id, feature_name, cursor
            )

            # Update onboarding progress
            await self._update_adoption_progress(journey_id, customer_id, cursor)

            conn.commit()

            return {
                "usage_id": usage_record['id'],
                "feature": feature_name,
                "usage_count": usage_record['usage_count'],
                "achievements": achievements,
                "tracked_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error tracking feature usage: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    async def _check_achievements(
        self,
        customer_id: str,
        feature_name: str,
        cursor: Any
    ) -> List[Dict]:
        """Check if feature usage unlocks any achievements"""
        achievements = []

        # Get total features used
        cursor.execute("""
            SELECT COUNT(DISTINCT feature_name) as features_used
            FROM ai_feature_usage
            WHERE customer_id = %s
        """, (customer_id,))

        result = cursor.fetchone()
        features_used = result['features_used'] if result else 0

        # Achievement milestones
        if features_used == 1:
            achievements.append({
                "name": "First Steps",
                "description": "Used your first feature!",
                "points": 10
            })
        elif features_used == 5:
            achievements.append({
                "name": "Explorer",
                "description": "Tried 5 different features",
                "points": 50
            })
        elif features_used == 10:
            achievements.append({
                "name": "Power User",
                "description": "Mastered 10 features",
                "points": 100
            })

        # Record achievements
        for achievement in achievements:
            cursor.execute("""
                INSERT INTO ai_customer_achievements (
                    id, customer_id, achievement_name,
                    achievement_description, points_earned,
                    earned_at
                ) VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (customer_id, achievement_name) DO NOTHING
            """, (
                str(uuid.uuid4()), customer_id,
                achievement['name'], achievement['description'],
                achievement['points']
            ))

        return achievements

    async def _update_adoption_progress(
        self,
        journey_id: str,
        customer_id: str,
        cursor: Any
    ) -> None:
        """Update overall feature adoption progress"""
        # Get adoption metrics
        cursor.execute("""
            SELECT
                COUNT(DISTINCT feature_name) as unique_features,
                SUM(usage_count) as total_usages,
                AVG(usage_count) as avg_usage_per_feature
            FROM ai_feature_usage
            WHERE customer_id = %s
        """, (customer_id,))

        metrics = cursor.fetchone()

        # Update journey with adoption metrics
        cursor.execute("""
            UPDATE ai_onboarding_journeys
            SET adoption_metrics = %s,
                updated_at = NOW()
            WHERE id = %s
        """, (json.dumps(dict(metrics)), journey_id))

    async def get_adoption_analysis(
        self,
        customer_id: str,
        journey_id: Optional[str] = None
    ) -> Dict:
        """Get comprehensive feature adoption analysis"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get feature usage summary
            cursor.execute("""
                SELECT
                    feature_name,
                    usage_count,
                    first_used_at,
                    last_used_at,
                    EXTRACT(EPOCH FROM (last_used_at - first_used_at))/86400 as usage_span_days
                FROM ai_feature_usage
                WHERE customer_id = %s
                ORDER BY first_used_at
            """, (customer_id,))

            features = cursor.fetchall()

            # Calculate adoption metrics
            total_features = len(features)
            total_usages = sum(f['usage_count'] for f in features)
            avg_usage = total_usages / total_features if total_features > 0 else 0

            # Determine adoption level
            if total_features >= 10 and avg_usage >= 5:
                adoption_level = "advanced"
            elif total_features >= 5 and avg_usage >= 3:
                adoption_level = "intermediate"
            elif total_features >= 2:
                adoption_level = "beginner"
            else:
                adoption_level = "getting_started"

            # Get recommended next features
            recommended_features = await self._get_recommended_features(
                customer_id, features, cursor
            )

            return {
                "customer_id": customer_id,
                "adoption_level": adoption_level,
                "total_features_used": total_features,
                "total_feature_usages": total_usages,
                "average_usage_per_feature": round(avg_usage, 2),
                "features": [dict(f) for f in features],
                "recommended_next_features": recommended_features,
                "analyzed_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting adoption analysis: {e}")
            return {"error": str(e)}

    async def _get_recommended_features(
        self,
        customer_id: str,
        current_features: List[Dict],
        cursor: Any
    ) -> List[Dict]:
        """Get recommended features based on current usage"""
        used_features = [f['feature_name'] for f in current_features]

        # Define feature progression paths
        feature_paths = {
            "basic_profile": ["advanced_settings", "team_management"],
            "first_job": ["job_templates", "scheduling", "invoicing"],
            "invoicing": ["payment_processing", "financial_reports"],
            "team_management": ["role_permissions", "team_analytics"],
            "scheduling": ["calendar_integration", "automated_reminders"]
        }

        recommendations = []
        for used_feature in used_features:
            if used_feature in feature_paths:
                for next_feature in feature_paths[used_feature]:
                    if next_feature not in used_features:
                        recommendations.append({
                            "feature": next_feature,
                            "reason": f"Natural progression from {used_feature}",
                            "priority": "high"
                        })

        return recommendations[:5]  # Return top 5 recommendations


class PersonalizedOnboardingFlows:
    """Create and manage personalized onboarding flows"""

    def __init__(self):
        self.conn = None

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**_get_db_config())
        return self.conn

    async def create_personalized_flow(
        self,
        customer_id: str,
        customer_profile: Dict,
        business_goals: List[str],
        experience_level: str
    ) -> Dict:
        """Create a fully personalized onboarding flow"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            flow_id = str(uuid.uuid4())

            # Analyze customer needs
            flow_analysis = self._analyze_customer_needs(
                customer_profile, business_goals, experience_level
            )

            # Generate personalized milestones
            milestones = self._generate_personalized_milestones(
                business_goals, experience_level, flow_analysis
            )

            # Create custom content sequence
            content_sequence = self._create_content_sequence(
                customer_profile, flow_analysis
            )

            # Define success metrics
            success_metrics = self._define_success_metrics(
                business_goals, experience_level
            )

            # Store personalized flow
            cursor.execute("""
                INSERT INTO ai_personalized_flows (
                    id, customer_id, flow_analysis,
                    milestones, content_sequence,
                    success_metrics, experience_level,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                flow_id, customer_id,
                json.dumps(flow_analysis),
                json.dumps(milestones),
                json.dumps(content_sequence),
                json.dumps(success_metrics),
                experience_level
            ))

            conn.commit()

            return {
                "flow_id": flow_id,
                "customer_id": customer_id,
                "flow_type": flow_analysis['flow_type'],
                "milestones": milestones,
                "content_sequence": content_sequence,
                "success_metrics": success_metrics,
                "estimated_completion": flow_analysis['estimated_days'],
                "created_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error creating personalized flow: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}

    def _analyze_customer_needs(
        self,
        profile: Dict,
        goals: List[str],
        experience_level: str
    ) -> Dict:
        """Analyze customer needs to determine flow type"""
        analysis = {
            "flow_type": "standard",
            "pace": "moderate",
            "focus_areas": [],
            "estimated_days": 14
        }

        # Determine flow type based on profile
        if profile.get('company_size', 0) > 50:
            analysis['flow_type'] = "enterprise"
            analysis['estimated_days'] = 30
        elif profile.get('is_technical', False):
            analysis['flow_type'] = "technical"
            analysis['pace'] = "fast"
            analysis['estimated_days'] = 7
        elif experience_level == "beginner":
            analysis['flow_type'] = "beginner_friendly"
            analysis['pace'] = "slow"
            analysis['estimated_days'] = 21

        # Determine focus areas from goals
        goal_mapping = {
            "increase_efficiency": "automation",
            "improve_tracking": "analytics",
            "grow_business": "sales",
            "manage_team": "team_management",
            "financial_control": "accounting"
        }

        analysis['focus_areas'] = [
            goal_mapping.get(goal, goal) for goal in goals
        ]

        return analysis

    def _generate_personalized_milestones(
        self,
        goals: List[str],
        experience_level: str,
        analysis: Dict
    ) -> List[Dict]:
        """Generate milestones tailored to customer goals"""
        milestones = []

        # Universal first milestone
        milestones.append({
            "name": "Getting Started",
            "description": "Complete account setup and profile",
            "target_day": 1,
            "tasks": [
                "Complete profile information",
                "Set up company details",
                "Configure preferences"
            ],
            "success_criteria": "Profile 100% complete"
        })

        # Goal-specific milestones
        if "increase_efficiency" in goals:
            milestones.append({
                "name": "Automation Setup",
                "description": "Configure automated workflows",
                "target_day": 5,
                "tasks": [
                    "Set up job templates",
                    "Configure automated notifications",
                    "Enable smart scheduling"
                ],
                "success_criteria": "First automated workflow created"
            })

        if "grow_business" in goals:
            milestones.append({
                "name": "First Customer Win",
                "description": "Complete your first customer transaction",
                "target_day": 7,
                "tasks": [
                    "Add first customer",
                    "Create first job",
                    "Send first invoice"
                ],
                "success_criteria": "Complete end-to-end customer workflow"
            })

        if "manage_team" in goals:
            milestones.append({
                "name": "Team Onboarded",
                "description": "Get your team up and running",
                "target_day": 10,
                "tasks": [
                    "Invite team members",
                    "Set up roles and permissions",
                    "Assign first tasks"
                ],
                "success_criteria": "At least 3 team members active"
            })

        # Final milestone
        milestones.append({
            "name": "Full Adoption",
            "description": "Become a power user",
            "target_day": analysis['estimated_days'],
            "tasks": [
                "Use 10+ features",
                "Complete 5+ workflows",
                "Share feedback or referral"
            ],
            "success_criteria": "Advanced adoption level achieved"
        })

        return milestones

    def _create_content_sequence(
        self,
        profile: Dict,
        analysis: Dict
    ) -> List[Dict]:
        """Create personalized content delivery sequence"""
        sequence = []

        # Welcome content
        sequence.append({
            "day": 0,
            "type": "welcome",
            "title": f"Welcome to your {analysis['flow_type']} onboarding",
            "content_items": [
                "Welcome video",
                "Quick start guide",
                "Success checklist"
            ],
            "channel": "email"
        })

        # Focus area content
        for focus in analysis['focus_areas']:
            sequence.append({
                "day": len(sequence) + 2,
                "type": "educational",
                "title": f"Mastering {focus}",
                "content_items": [
                    f"{focus} tutorial",
                    f"{focus} best practices",
                    f"{focus} templates"
                ],
                "channel": "in_app"
            })

        # Check-in content
        sequence.append({
            "day": 7,
            "type": "check_in",
            "title": "How are you doing?",
            "content_items": [
                "Progress review",
                "Feedback request",
                "Support offer"
            ],
            "channel": "email"
        })

        return sequence

    def _define_success_metrics(
        self,
        goals: List[str],
        experience_level: str
    ) -> Dict:
        """Define success metrics for the personalized flow"""
        metrics = {
            "completion_threshold": 80 if experience_level == "beginner" else 90,
            "engagement_minimum": 5 if experience_level == "beginner" else 10,
            "time_to_value_days": 14 if experience_level == "beginner" else 7,
            "feature_adoption_target": 5 if experience_level == "beginner" else 10,
            "goal_specific_metrics": {}
        }

        # Add goal-specific metrics
        for goal in goals:
            if goal == "increase_efficiency":
                metrics['goal_specific_metrics'][goal] = {
                    "automated_workflows": 3,
                    "time_saved_hours": 5
                }
            elif goal == "grow_business":
                metrics['goal_specific_metrics'][goal] = {
                    "new_customers": 5,
                    "revenue_generated": 1000
                }

        return metrics


class OnboardingContentGenerator:
    """Generate dynamic onboarding content"""

    async def generate_content(
        self,
        content_type: str,
        context: Dict,
        personalization: Dict
    ) -> Dict:
        """Generate personalized onboarding content"""
        base_content = self._get_base_content(content_type)

        # Apply personalization
        if personalization.get('coaching_style') == 'minimal':
            content = self._simplify_content(base_content)
        elif personalization.get('coaching_style') == 'supportive':
            content = self._add_support_content(base_content)
        else:
            content = base_content

        # Add context-specific elements
        content = self._add_context(content, context)

        return content

    def _get_base_content(self, content_type: str) -> Dict:
        """Get base content template"""
        templates = {
            "welcome_email": {
                "subject": "Welcome to Your Journey!",
                "body": "We're excited to help you get started..."
            },
            "progress_update": {
                "subject": "You're making great progress!",
                "body": "You've completed {progress}% of your onboarding..."
            },
            "resource_recommendation": {
                "subject": "Resources to help you succeed",
                "body": "Based on your goals, we recommend..."
            }
        }
        return templates.get(content_type, {"subject": "Update", "body": "Content"})

    def _simplify_content(self, content: Dict) -> Dict:
        """Simplify content for technical users"""
        content['subject'] = content['subject'].replace('Journey', 'Setup')
        content['body'] = "Quick start: " + content['body']
        return content

    def _add_support_content(self, content: Dict) -> Dict:
        """Add supportive elements to content"""
        content['body'] += "\n\nNeed help? We're here for you 24/7."
        return content

    def _add_context(self, content: Dict, context: Dict) -> Dict:
        """Add context-specific information"""
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in content['body']:
                content['body'] = content['body'].replace(placeholder, str(value))
        return content


# Singleton instance
_onboarding_system = None

def get_ai_customer_onboarding():
    """Get singleton instance of AI customer onboarding system"""
    global _onboarding_system
    if _onboarding_system is None:
        _onboarding_system = AICustomerOnboarding()
    return _onboarding_system


# Export main components
__all__ = [
    'AICustomerOnboarding',
    'OnboardingStage',
    'OnboardingStatus',
    'CustomerSegment',
    'OnboardingAction',
    'InterventionType',
    'JourneyDesigner',
    'ProgressTracker',
    'PersonalizationEngine',
    'InterventionManager',
    'SuccessPredictor',
    'OnboardingContentGenerator',
    'FeatureAdoptionTracker',
    'PersonalizedOnboardingFlows',
    'get_ai_customer_onboarding'
]