#!/usr/bin/env python3
"""
AI Context Awareness using Auth
Build user-specific AI personalization
"""

import os
import json
import logging
import uuid
import hashlib
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import jwt
from openai import OpenAI
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# Auth configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
JWT_ALGORITHM = "HS256"

# OpenAI configuration
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UserRole(Enum):
    """User roles in the system"""
    ADMIN = "admin"
    MANAGER = "manager"
    USER = "user"
    AGENT = "agent"
    CUSTOMER = "customer"
    VENDOR = "vendor"
    PARTNER = "partner"

class ContextType(Enum):
    """Types of context"""
    PERSONAL = "personal"
    BEHAVIORAL = "behavioral"
    PREFERENCE = "preference"
    HISTORICAL = "historical"
    PERMISSION = "permission"
    INTERACTION = "interaction"
    BUSINESS = "business"
    TECHNICAL = "technical"

class PersonalizationType(Enum):
    """Types of personalization"""
    CONTENT = "content"
    INTERFACE = "interface"
    WORKFLOW = "workflow"
    RECOMMENDATION = "recommendation"
    COMMUNICATION = "communication"
    AUTOMATION = "automation"
    LEARNING = "learning"

class PrivacyLevel(Enum):
    """Privacy levels for data"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"

class AIContextAwareness:
    """Main AI context awareness class"""

    def __init__(self):
        """Initialize the context awareness system"""
        self.user_cache = {}
        self.context_cache = {}
        self.session_timeout = 3600  # 1 hour
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create user profile table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_profiles (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) UNIQUE NOT NULL,
                    email VARCHAR(255),
                    role VARCHAR(50),
                    organization VARCHAR(255),
                    department VARCHAR(255),
                    preferences JSONB DEFAULT '{}',
                    permissions JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create user context table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_context (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    context_type VARCHAR(50),
                    context_data JSONB,
                    privacy_level VARCHAR(50) DEFAULT 'internal',
                    confidence FLOAT DEFAULT 1.0,
                    expires_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create personalization table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_personalizations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    personalization_type VARCHAR(50),
                    configuration JSONB,
                    effectiveness_score FLOAT DEFAULT 0.5,
                    active BOOLEAN DEFAULT TRUE,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create user sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    session_token VARCHAR(500) UNIQUE,
                    context_snapshot JSONB,
                    started_at TIMESTAMPTZ DEFAULT NOW(),
                    last_activity TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{}'
                )
            """)

            # Create user interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_interactions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    interaction_type VARCHAR(50),
                    action VARCHAR(100),
                    entity_type VARCHAR(50),
                    entity_id VARCHAR(255),
                    context JSONB DEFAULT '{}',
                    result JSONB DEFAULT '{}',
                    duration_ms INT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create user preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_preferences (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    category VARCHAR(50),
                    preference_key VARCHAR(100),
                    preference_value JSONB,
                    learned BOOLEAN DEFAULT FALSE,
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(user_id, category, preference_key)
                )
            """)

            # Create permission policies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_permission_policies (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    policy_name VARCHAR(255) UNIQUE,
                    resource_type VARCHAR(50),
                    resource_pattern VARCHAR(255),
                    allowed_roles JSONB DEFAULT '[]',
                    allowed_users JSONB DEFAULT '[]',
                    conditions JSONB DEFAULT '{}',
                    actions JSONB DEFAULT '[]',
                    effect VARCHAR(20) DEFAULT 'allow',
                    priority INT DEFAULT 100,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create user embeddings table for similarity matching
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_user_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id VARCHAR(255) REFERENCES ai_user_profiles(user_id),
                    embedding_type VARCHAR(50),
                    embedding vector(1536),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_user ON ai_user_profiles(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_context_user ON ai_user_context(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_personalizations_user ON ai_personalizations(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON ai_user_sessions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON ai_user_sessions(session_token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON ai_user_interactions(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_preferences_user ON ai_user_preferences(user_id)")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def create_user_profile(
        self,
        user_id: str,
        email: str = None,
        role: UserRole = UserRole.USER,
        organization: str = None,
        department: str = None,
        metadata: Dict = None
    ) -> Dict:
        """Create or update user profile"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Check if profile exists
            cursor.execute("""
                SELECT id FROM ai_user_profiles WHERE user_id = %s
            """, (user_id,))

            existing = cursor.fetchone()

            if existing:
                # Update existing profile
                cursor.execute("""
                    UPDATE ai_user_profiles
                    SET email = COALESCE(%s, email),
                        role = COALESCE(%s, role),
                        organization = COALESCE(%s, organization),
                        department = COALESCE(%s, department),
                        metadata = COALESCE(%s, metadata),
                        updated_at = NOW()
                    WHERE user_id = %s
                    RETURNING id
                """, (email, role.value, organization, department, Json(metadata or {}), user_id))
            else:
                # Create new profile
                cursor.execute("""
                    INSERT INTO ai_user_profiles
                    (user_id, email, role, organization, department, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, email, role.value, organization, department, Json(metadata or {})))

            profile_id = cursor.fetchone()[0]

            # Initialize default context
            await self._initialize_user_context(user_id, role)

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'profile_id': profile_id,
                'user_id': user_id,
                'status': 'updated' if existing else 'created'
            }

        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            return {'error': str(e)}

    async def _initialize_user_context(self, user_id: str, role: UserRole):
        """Initialize default context for user"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Default preferences based on role
            default_preferences = self._get_default_preferences(role)

            for category, prefs in default_preferences.items():
                for key, value in prefs.items():
                    cursor.execute("""
                        INSERT INTO ai_user_preferences
                        (user_id, category, preference_key, preference_value)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, category, preference_key)
                        DO NOTHING
                    """, (user_id, category, key, Json(value)))

            # Initialize context
            cursor.execute("""
                INSERT INTO ai_user_context
                (user_id, context_type, context_data, privacy_level)
                VALUES (%s, %s, %s, %s)
            """, (
                user_id,
                ContextType.PERSONAL.value,
                Json({'role': role.value, 'initialized': True}),
                PrivacyLevel.PERSONAL.value
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize user context: {e}")

    def _get_default_preferences(self, role: UserRole) -> Dict:
        """Get default preferences based on role"""
        base_preferences = {
            'interface': {
                'theme': 'light',
                'language': 'en',
                'notifications': True
            },
            'communication': {
                'email_frequency': 'daily',
                'preferred_channel': 'email'
            }
        }

        role_specific = {
            UserRole.ADMIN: {
                'dashboard': {
                    'default_view': 'analytics',
                    'show_all_metrics': True
                },
                'automation': {
                    'approval_required': False,
                    'auto_escalate': True
                }
            },
            UserRole.MANAGER: {
                'dashboard': {
                    'default_view': 'team',
                    'show_team_metrics': True
                },
                'automation': {
                    'approval_required': True,
                    'auto_assign': True
                }
            },
            UserRole.USER: {
                'dashboard': {
                    'default_view': 'tasks',
                    'show_personal_metrics': True
                },
                'automation': {
                    'auto_suggestions': True
                }
            }
        }

        base_preferences.update(role_specific.get(role, {}))
        return base_preferences

    async def authenticate_user(
        self,
        credentials: Dict
    ) -> Dict:
        """Authenticate user and create session"""
        try:
            # Verify credentials (simplified - would integrate with real auth)
            user_id = credentials.get('user_id')
            if not user_id:
                return {'error': 'Invalid credentials'}

            # Create session token
            session_id = str(uuid.uuid4())
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.session_timeout)

            token_payload = {
                'user_id': user_id,
                'session_id': session_id,
                'exp': expires_at.timestamp()
            }

            token = jwt.encode(token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

            # Get user context
            context = await self.get_user_context(user_id)

            # Store session
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_user_sessions
                (user_id, session_token, context_snapshot, expires_at)
                VALUES (%s, %s, %s, %s)
            """, (user_id, token, Json(context), expires_at))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'token': token,
                'user_id': user_id,
                'session_id': session_id,
                'expires_at': expires_at.isoformat(),
                'context': context
            }

        except Exception as e:
            logger.error(f"Failed to authenticate user: {e}")
            return {'error': str(e)}

    async def verify_token(self, token: str) -> Dict:
        """Verify JWT token and get user context"""
        try:
            # Decode token
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            user_id = payload.get('user_id')
            session_id = payload.get('session_id')

            # Check session validity
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM ai_user_sessions
                WHERE session_token = %s
                  AND expires_at > NOW()
            """, (token,))

            session = cursor.fetchone()

            if not session:
                return {'valid': False, 'error': 'Invalid or expired session'}

            # Update last activity
            cursor.execute("""
                UPDATE ai_user_sessions
                SET last_activity = NOW()
                WHERE session_token = %s
            """, (token,))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'valid': True,
                'user_id': user_id,
                'session_id': session_id,
                'context': session['context_snapshot']
            }

        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return {'valid': False, 'error': str(e)}

    async def get_user_context(
        self,
        user_id: str,
        context_types: List[ContextType] = None
    ) -> Dict:
        """Get comprehensive user context"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get user profile
            cursor.execute("""
                SELECT * FROM ai_user_profiles WHERE user_id = %s
            """, (user_id,))

            profile = cursor.fetchone()
            if not profile:
                return {}

            # Get context data
            if context_types:
                context_filter = [ct.value for ct in context_types]
                cursor.execute("""
                    SELECT * FROM ai_user_context
                    WHERE user_id = %s
                      AND context_type = ANY(%s)
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY updated_at DESC
                """, (user_id, context_filter))
            else:
                cursor.execute("""
                    SELECT * FROM ai_user_context
                    WHERE user_id = %s
                      AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY updated_at DESC
                """, (user_id,))

            contexts = cursor.fetchall()

            # Get preferences
            cursor.execute("""
                SELECT category, preference_key, preference_value
                FROM ai_user_preferences
                WHERE user_id = %s
            """, (user_id,))

            preferences = {}
            for pref in cursor.fetchall():
                if pref['category'] not in preferences:
                    preferences[pref['category']] = {}
                preferences[pref['category']][pref['preference_key']] = pref['preference_value']

            # Get recent interactions
            cursor.execute("""
                SELECT * FROM ai_user_interactions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 10
            """, (user_id,))

            recent_interactions = cursor.fetchall()

            cursor.close()
            conn.close()

            # Build comprehensive context
            context = {
                'user_id': user_id,
                'profile': {
                    'email': profile['email'],
                    'role': profile['role'],
                    'organization': profile['organization'],
                    'department': profile['department']
                },
                'contexts': {c['context_type']: c['context_data'] for c in contexts},
                'preferences': preferences,
                'recent_activity': recent_interactions,
                'permissions': profile.get('permissions', {}),
                'metadata': profile.get('metadata', {})
            }

            # Cache context
            self.context_cache[user_id] = context

            return context

        except Exception as e:
            logger.error(f"Failed to get user context: {e}")
            return {}

    async def update_user_context(
        self,
        user_id: str,
        context_type: ContextType,
        context_data: Dict,
        privacy_level: PrivacyLevel = PrivacyLevel.INTERNAL
    ) -> bool:
        """Update user context"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_user_context
                (user_id, context_type, context_data, privacy_level)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT ON CONSTRAINT unique_user_context
                DO UPDATE SET
                    context_data = %s,
                    privacy_level = %s,
                    updated_at = NOW()
            """, (
                user_id,
                context_type.value,
                Json(context_data),
                privacy_level.value,
                Json(context_data),
                privacy_level.value
            ))

            conn.commit()
            cursor.close()
            conn.close()

            # Invalidate cache
            if user_id in self.context_cache:
                del self.context_cache[user_id]

            return True

        except Exception as e:
            logger.error(f"Failed to update context: {e}")
            return False

    async def track_interaction(
        self,
        user_id: str,
        interaction_type: str,
        action: str,
        entity_type: str = None,
        entity_id: str = None,
        context: Dict = None,
        result: Dict = None,
        duration_ms: int = None
    ) -> str:
        """Track user interaction"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            interaction_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO ai_user_interactions
                (id, user_id, interaction_type, action, entity_type,
                 entity_id, context, result, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                interaction_id,
                user_id,
                interaction_type,
                action,
                entity_type,
                entity_id,
                Json(context or {}),
                Json(result or {}),
                duration_ms
            ))

            conn.commit()
            cursor.close()
            conn.close()

            # Learn from interaction
            await self._learn_from_interaction(
                user_id,
                interaction_type,
                action,
                context,
                result
            )

            return interaction_id

        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
            return None

    async def _learn_from_interaction(
        self,
        user_id: str,
        interaction_type: str,
        action: str,
        context: Dict,
        result: Dict
    ):
        """Learn from user interaction to improve personalization"""
        try:
            # Analyze interaction patterns
            if interaction_type == 'navigation' and action == 'view':
                # Learn navigation preferences
                await self._update_preference(
                    user_id,
                    'navigation',
                    'frequently_visited',
                    {'page': context.get('page'), 'count': 1},
                    learned=True
                )

            elif interaction_type == 'action' and result and result.get('success'):
                # Learn successful action patterns
                await self._update_preference(
                    user_id,
                    'automation',
                    f'preferred_{action}',
                    context,
                    learned=True
                )

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")

    async def _update_preference(
        self,
        user_id: str,
        category: str,
        key: str,
        value: Any,
        learned: bool = False,
        confidence: float = 0.8
    ):
        """Update user preference"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO ai_user_preferences
                (user_id, category, preference_key, preference_value,
                 learned, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, category, preference_key)
                DO UPDATE SET
                    preference_value = %s,
                    learned = %s,
                    confidence = GREATEST(ai_user_preferences.confidence, %s),
                    updated_at = NOW()
            """, (
                user_id,
                category,
                key,
                Json(value),
                learned,
                confidence,
                Json(value),
                learned,
                confidence
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update preference: {e}")

    async def check_permission(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str
    ) -> bool:
        """Check if user has permission for action"""
        try:
            # Get user context
            context = await self.get_user_context(user_id)
            user_role = context.get('profile', {}).get('role')

            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Check permission policies
            cursor.execute("""
                SELECT * FROM ai_permission_policies
                WHERE resource_type = %s
                  AND %s = ANY(actions)
                  AND active = TRUE
                  AND (
                    %s = ANY(allowed_roles)
                    OR %s = ANY(allowed_users)
                  )
                ORDER BY priority DESC
            """, (resource_type, action, user_role, user_id))

            policies = cursor.fetchall()

            cursor.close()
            conn.close()

            # Evaluate policies
            for policy in policies:
                if policy['effect'] == 'deny':
                    return False

                # Check conditions
                conditions = policy.get('conditions', {})
                if self._evaluate_conditions(conditions, context, resource_id):
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to check permission: {e}")
            return False

    def _evaluate_conditions(
        self,
        conditions: Dict,
        context: Dict,
        resource_id: str
    ) -> bool:
        """Evaluate permission conditions"""
        if not conditions:
            return True

        # Simple condition evaluation
        for key, value in conditions.items():
            if key == 'organization':
                if context.get('profile', {}).get('organization') != value:
                    return False
            elif key == 'department':
                if context.get('profile', {}).get('department') != value:
                    return False
            elif key == 'resource_owner':
                # Would check resource ownership
                pass

        return True

    async def personalize_content(
        self,
        user_id: str,
        content_type: str,
        base_content: Any
    ) -> Any:
        """Personalize content for user"""
        try:
            # Get user context
            context = await self.get_user_context(user_id)
            preferences = context.get('preferences', {})

            # Apply personalizations
            if content_type == 'dashboard':
                return self._personalize_dashboard(base_content, preferences)
            elif content_type == 'recommendations':
                return await self._generate_recommendations(user_id, context)
            elif content_type == 'communication':
                return self._personalize_communication(base_content, preferences)
            else:
                return base_content

        except Exception as e:
            logger.error(f"Failed to personalize content: {e}")
            return base_content

    def _personalize_dashboard(self, dashboard: Dict, preferences: Dict) -> Dict:
        """Personalize dashboard layout"""
        dashboard_prefs = preferences.get('dashboard', {})

        personalized = dashboard.copy()
        personalized['theme'] = preferences.get('interface', {}).get('theme', 'light')
        personalized['default_view'] = dashboard_prefs.get('default_view', 'overview')
        personalized['widgets'] = dashboard_prefs.get('preferred_widgets', dashboard.get('widgets', []))

        return personalized

    async def _generate_recommendations(
        self,
        user_id: str,
        context: Dict
    ) -> List[Dict]:
        """Generate personalized recommendations"""
        try:
            # Get user embedding
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT embedding FROM ai_user_embeddings
                WHERE user_id = %s
                  AND embedding_type = 'preference'
                ORDER BY updated_at DESC
                LIMIT 1
            """, (user_id,))

            user_embedding = cursor.fetchone()

            recommendations = []

            if user_embedding:
                # Find similar content/users
                cursor.execute("""
                    SELECT
                        entity_id,
                        entity_type,
                        1 - (embedding <=> %s) as similarity
                    FROM ai_content_embeddings
                    WHERE entity_type IN ('content', 'product', 'service')
                    ORDER BY similarity DESC
                    LIMIT 10
                """, (user_embedding[0],))

                for rec in cursor.fetchall():
                    recommendations.append({
                        'id': rec[0],
                        'type': rec[1],
                        'score': rec[2]
                    })

            cursor.close()
            conn.close()

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def _personalize_communication(
        self,
        message: Dict,
        preferences: Dict
    ) -> Dict:
        """Personalize communication based on preferences"""
        comm_prefs = preferences.get('communication', {})

        personalized = message.copy()
        personalized['channel'] = comm_prefs.get('preferred_channel', 'email')
        personalized['tone'] = comm_prefs.get('preferred_tone', 'professional')
        personalized['frequency'] = comm_prefs.get('email_frequency', 'immediate')

        return personalized

    async def find_similar_users(
        self,
        user_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """Find users with similar preferences/behavior"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get user embedding
            cursor.execute("""
                SELECT embedding FROM ai_user_embeddings
                WHERE user_id = %s
                  AND embedding_type = 'behavior'
                ORDER BY updated_at DESC
                LIMIT 1
            """, (user_id,))

            user_embedding = cursor.fetchone()

            if not user_embedding:
                return []

            # Find similar users
            cursor.execute("""
                SELECT
                    ue.user_id,
                    up.role,
                    up.organization,
                    1 - (ue.embedding <=> %s) as similarity
                FROM ai_user_embeddings ue
                JOIN ai_user_profiles up ON ue.user_id = up.user_id
                WHERE ue.user_id != %s
                  AND ue.embedding_type = 'behavior'
                ORDER BY similarity DESC
                LIMIT %s
            """, (user_embedding['embedding'], user_id, limit))

            similar_users = cursor.fetchall()

            cursor.close()
            conn.close()

            return similar_users

        except Exception as e:
            logger.error(f"Failed to find similar users: {e}")
            return []

    async def update_user_embedding(
        self,
        user_id: str,
        embedding_type: str = 'behavior'
    ):
        """Update user embedding based on their activity"""
        try:
            # Get user interactions and preferences
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT * FROM ai_user_interactions
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT 100
            """, (user_id,))

            interactions = cursor.fetchall()

            # Create text representation for embedding
            text_data = []
            for interaction in interactions:
                text_data.append(f"{interaction['interaction_type']} {interaction['action']}")

            if text_data:
                # Generate embedding
                response = openai_client.embeddings.create(
                    input=" ".join(text_data),
                    model="text-embedding-ada-002"
                )

                embedding = response.data[0].embedding

                # Store embedding
                cursor.execute("""
                    INSERT INTO ai_user_embeddings
                    (user_id, embedding_type, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, embedding_type)
                    DO UPDATE SET
                        embedding = %s,
                        updated_at = NOW()
                """, (user_id, embedding_type, embedding, embedding))

                conn.commit()

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update user embedding: {e}")

# Singleton instance
_context_awareness = None

def get_context_awareness() -> AIContextAwareness:
    """Get or create the context awareness instance"""
    global _context_awareness
    if _context_awareness is None:
        _context_awareness = AIContextAwareness()
    return _context_awareness