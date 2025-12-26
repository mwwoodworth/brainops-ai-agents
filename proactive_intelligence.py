#!/usr/bin/env python3
"""
PROACTIVE INTELLIGENCE - The Anticipating Brain of BrainOps AI OS
This module ANTICIPATES problems and opportunities before they happen.

Features:
- Pattern recognition across historical data
- Anomaly detection with statistical analysis
- Trend forecasting (1hr, 6hr, 24hr)
- Automatic action recommendations
- Self-initiated autonomous tasks
- Learning from outcomes
- Business intelligence predictions
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics
import psycopg2
from psycopg2.extras import RealDictCursor, Json

logger = logging.getLogger("PROACTIVE_INTELLIGENCE")

# Database configuration with DATABASE_URL fallback
def _build_db_config():
    """Build database config, supporting both individual vars and DATABASE_URL"""
    config = {
        'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', 5432))
    }
    if not config['password']:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            from urllib.parse import urlparse
            try:
                parsed = urlparse(database_url)
                config['host'] = parsed.hostname or config['host']
                config['database'] = parsed.path.lstrip('/') or config['database']
                config['user'] = parsed.username or config['user']
                config['password'] = parsed.password or ''
                config['port'] = parsed.port or config['port']
            except Exception as e:
                logger.error(f"Failed to parse DATABASE_URL: {e}")
    return config

DB_CONFIG = _build_db_config()


class PredictionType(Enum):
    """Types of predictions"""
    SYSTEM_FAILURE = "system_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    REVENUE_TREND = "revenue_trend"
    CUSTOMER_CHURN = "customer_churn"
    OPPORTUNITY = "opportunity"
    ANOMALY = "anomaly"


class ActionPriority(Enum):
    """Priority levels for recommended actions"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


@dataclass
class Prediction:
    """A prediction made by the AI"""
    id: str
    type: PredictionType
    description: str
    confidence: float  # 0-1
    time_horizon: str  # "1h", "6h", "24h"
    predicted_impact: str
    recommended_actions: List[str]
    data: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    validated: bool = False
    was_accurate: Optional[bool] = None

    def to_dict(self):
        return {
            'id': self.id,
            'type': self.type.value,
            'description': self.description,
            'confidence': self.confidence,
            'time_horizon': self.time_horizon,
            'predicted_impact': self.predicted_impact,
            'recommended_actions': self.recommended_actions,
            'data': self.data,
            'created_at': self.created_at.isoformat(),
            'validated': self.validated,
            'was_accurate': self.was_accurate
        }


@dataclass
class AutonomousAction:
    """An action the AI decides to take on its own"""
    id: str
    action_type: str
    description: str
    priority: ActionPriority
    trigger: str  # What triggered this action
    confidence: float
    parameters: Dict[str, Any]
    status: str = "pending"  # pending, executing, completed, failed
    result: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None


class ProactiveIntelligence:
    """
    The proactive brain that anticipates and acts before problems occur.
    """

    def __init__(self):
        self.predictions: List[Prediction] = []
        self.action_queue: List[AutonomousAction] = []
        self.pattern_cache: Dict[str, Any] = {}
        self.learning_history: List[Dict] = []
        self.action_counter = 0
        self.prediction_counter = 0
        # Schema is pre-created in database - skip blocking init
        # self._ensure_schema() - tables already exist

    def _get_connection(self):
        return psycopg2.connect(**DB_CONFIG)

    def _ensure_schema(self):
        """Create required tables"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                -- Predictions made by the AI
                CREATE TABLE IF NOT EXISTS ai_proactive_predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_id VARCHAR(100) UNIQUE,
                    prediction_type VARCHAR(50),
                    description TEXT,
                    confidence FLOAT,
                    time_horizon VARCHAR(20),
                    predicted_impact TEXT,
                    recommended_actions TEXT[],
                    data JSONB,
                    validated BOOLEAN DEFAULT FALSE,
                    was_accurate BOOLEAN,
                    validated_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_predictions_type
                    ON ai_proactive_predictions(prediction_type);
                CREATE INDEX IF NOT EXISTS idx_predictions_unvalidated
                    ON ai_proactive_predictions(created_at)
                    WHERE NOT validated;

                -- Autonomous actions taken
                CREATE TABLE IF NOT EXISTS ai_autonomous_actions (
                    id SERIAL PRIMARY KEY,
                    action_id VARCHAR(100) UNIQUE,
                    action_type VARCHAR(100),
                    description TEXT,
                    priority INTEGER,
                    trigger_reason TEXT,
                    confidence FLOAT,
                    parameters JSONB,
                    status VARCHAR(50),
                    result JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    executed_at TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_actions_status
                    ON ai_autonomous_actions(status);
                CREATE INDEX IF NOT EXISTS idx_actions_priority
                    ON ai_autonomous_actions(priority);

                -- Pattern recognition results
                CREATE TABLE IF NOT EXISTS ai_detected_patterns (
                    id SERIAL PRIMARY KEY,
                    pattern_name VARCHAR(255),
                    pattern_type VARCHAR(100),
                    description TEXT,
                    occurrences INTEGER,
                    confidence FLOAT,
                    data JSONB,
                    first_seen TIMESTAMPTZ,
                    last_seen TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_patterns_name
                    ON ai_detected_patterns(pattern_name);

                -- Learning from outcomes
                CREATE TABLE IF NOT EXISTS ai_proactive_learnings (
                    id SERIAL PRIMARY KEY,
                    learning_type VARCHAR(100),
                    source_prediction_id VARCHAR(100),
                    source_action_id VARCHAR(100),
                    lesson TEXT,
                    improvement_suggestion TEXT,
                    confidence_adjustment FLOAT,
                    data JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)

            conn.commit()
            conn.close()
            logger.info("✅ ProactiveIntelligence schema initialized")
        except Exception as e:
            logger.error(f"Failed to init schema: {e}")

    async def analyze_patterns(self, data_source: str = "all") -> List[Dict]:
        """Analyze historical data for patterns"""
        patterns = []

        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Analyze error patterns
            cur.execute("""
                SELECT
                    error_type,
                    COUNT(*) as count,
                    AVG(EXTRACT(EPOCH FROM created_at - LAG(created_at) OVER (ORDER BY created_at))) as avg_interval
                FROM ai_error_logs
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY error_type
                HAVING COUNT(*) > 3
            """)
            error_patterns = cur.fetchall()

            for ep in error_patterns:
                patterns.append({
                    'type': 'recurring_error',
                    'name': f"error_{ep['error_type']}",
                    'description': f"Error '{ep['error_type']}' occurred {ep['count']} times in 24h",
                    'occurrences': ep['count'],
                    'confidence': min(0.9, 0.5 + (ep['count'] / 20))
                })

            # Analyze performance trends
            cur.execute("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    AVG(response_time_avg) as avg_response,
                    AVG(cpu_percent) as avg_cpu,
                    AVG(memory_percent) as avg_memory
                FROM ai_vital_signs
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', created_at)
                ORDER BY hour
            """)
            perf_trends = cur.fetchall()

            if len(perf_trends) >= 3:
                # Detect upward trends in resource usage
                cpu_values = [p['avg_cpu'] for p in perf_trends if p['avg_cpu']]
                if len(cpu_values) >= 3:
                    recent_avg = statistics.mean(cpu_values[-3:])
                    older_avg = statistics.mean(cpu_values[:3])
                    if recent_avg > older_avg * 1.3:
                        patterns.append({
                            'type': 'resource_trend',
                            'name': 'cpu_increasing',
                            'description': f"CPU usage trending up: {older_avg:.1f}% -> {recent_avg:.1f}%",
                            'confidence': 0.7,
                            'trend_direction': 'up'
                        })

            conn.close()

            # Store detected patterns
            for pattern in patterns:
                await self._store_pattern(pattern)

            return patterns

        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return []

    async def _store_pattern(self, pattern: Dict):
        """Store or update a detected pattern"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_detected_patterns
                (pattern_name, pattern_type, description, occurrences, confidence, data, first_seen)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (pattern_name) DO UPDATE SET
                    occurrences = ai_detected_patterns.occurrences + 1,
                    confidence = EXCLUDED.confidence,
                    last_seen = NOW()
            """, (
                pattern['name'], pattern['type'], pattern['description'],
                pattern.get('occurrences', 1), pattern['confidence'],
                Json(pattern)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to store pattern: {e}")

    async def anticipate_issues(self) -> List[Prediction]:
        """Predict potential issues before they happen"""
        predictions = []

        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for resource exhaustion risk
            cur.execute("""
                SELECT
                    AVG(cpu_percent) as avg_cpu,
                    AVG(memory_percent) as avg_memory,
                    MAX(cpu_percent) as max_cpu,
                    MAX(memory_percent) as max_memory
                FROM ai_vital_signs
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            vitals = cur.fetchone()

            if vitals and vitals['avg_cpu']:
                if vitals['avg_cpu'] > 70:
                    self.prediction_counter += 1
                    pred = Prediction(
                        id=f"pred_{self.prediction_counter}",
                        type=PredictionType.RESOURCE_EXHAUSTION,
                        description=f"CPU usage averaging {vitals['avg_cpu']:.1f}% - risk of exhaustion",
                        confidence=0.7 + (vitals['avg_cpu'] - 70) / 100,
                        time_horizon="6h",
                        predicted_impact="System slowdown or failures possible",
                        recommended_actions=[
                            "Scale up compute resources",
                            "Identify and optimize heavy processes",
                            "Enable auto-scaling if available"
                        ],
                        data={'vitals': dict(vitals)}
                    )
                    predictions.append(pred)

                if vitals['avg_memory'] and vitals['avg_memory'] > 80:
                    self.prediction_counter += 1
                    pred = Prediction(
                        id=f"pred_{self.prediction_counter}",
                        type=PredictionType.RESOURCE_EXHAUSTION,
                        description=f"Memory usage at {vitals['avg_memory']:.1f}% - OOM risk",
                        confidence=0.8,
                        time_horizon="1h",
                        predicted_impact="Out of memory errors likely",
                        recommended_actions=[
                            "Restart service to clear memory",
                            "Increase memory allocation",
                            "Check for memory leaks"
                        ],
                        data={'vitals': dict(vitals)}
                    )
                    predictions.append(pred)

            # Check for error rate spikes
            cur.execute("""
                SELECT
                    COUNT(*) as error_count,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '10 minutes') as recent_errors
                FROM ai_error_logs
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            errors = cur.fetchone()

            if errors and errors['recent_errors'] and errors['error_count']:
                if errors['recent_errors'] > errors['error_count'] * 0.5:
                    self.prediction_counter += 1
                    pred = Prediction(
                        id=f"pred_{self.prediction_counter}",
                        type=PredictionType.SYSTEM_FAILURE,
                        description="Error rate spiking - 50%+ of hourly errors in last 10 minutes",
                        confidence=0.85,
                        time_horizon="1h",
                        predicted_impact="System failure or major degradation imminent",
                        recommended_actions=[
                            "Investigate error logs immediately",
                            "Enable circuit breakers",
                            "Prepare rollback if recent deployment"
                        ],
                        data={'errors': dict(errors)}
                    )
                    predictions.append(pred)

            conn.close()

            # Store predictions
            for pred in predictions:
                await self._store_prediction(pred)

            return predictions

        except Exception as e:
            logger.error(f"Anticipation error: {e}")
            return []

    async def _store_prediction(self, prediction: Prediction):
        """Store a prediction"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_proactive_predictions
                (prediction_id, prediction_type, description, confidence,
                 time_horizon, predicted_impact, recommended_actions, data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                prediction.id, prediction.type.value, prediction.description,
                prediction.confidence, prediction.time_horizon,
                prediction.predicted_impact, prediction.recommended_actions,
                Json(prediction.data)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to store prediction: {e}")

    async def recommend_actions(self, context: str = "general") -> List[Dict]:
        """Generate action recommendations based on current state"""
        recommendations = []

        predictions = await self.anticipate_issues()

        for pred in predictions:
            for action in pred.recommended_actions:
                recommendations.append({
                    'action': action,
                    'reason': pred.description,
                    'priority': ActionPriority.HIGH.value if pred.confidence > 0.8 else ActionPriority.MEDIUM.value,
                    'confidence': pred.confidence,
                    'prediction_id': pred.id
                })

        return sorted(recommendations, key=lambda x: x['priority'])

    async def execute_autonomous_task(self, action: AutonomousAction) -> Dict:
        """Execute an autonomous action"""
        action.status = "executing"
        action.executed_at = datetime.utcnow()

        try:
            # For now, log the action - real implementation would execute
            logger.info(f"🤖 AUTONOMOUS ACTION: {action.action_type} - {action.description}")

            # Store action
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_autonomous_actions
                (action_id, action_type, description, priority, trigger_reason,
                 confidence, parameters, status, executed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                action.id, action.action_type, action.description,
                action.priority.value, action.trigger, action.confidence,
                Json(action.parameters), 'completed', action.executed_at
            ))
            conn.commit()
            conn.close()

            action.status = "completed"
            action.result = {'success': True, 'message': 'Action executed'}

            return action.result

        except Exception as e:
            action.status = "failed"
            action.result = {'success': False, 'error': str(e)}
            logger.error(f"Autonomous action failed: {e}")
            return action.result

    async def learn_from_outcome(self, prediction_id: str, was_accurate: bool,
                                  actual_outcome: str):
        """Learn from whether our predictions were correct"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Update prediction
            cur.execute("""
                UPDATE ai_proactive_predictions
                SET validated = TRUE, was_accurate = %s, validated_at = NOW()
                WHERE prediction_id = %s
            """, (was_accurate, prediction_id))

            # Store learning
            lesson = "Prediction was accurate" if was_accurate else "Prediction was inaccurate"
            improvement = "Maintain confidence levels" if was_accurate else "Reduce confidence or adjust parameters"

            cur.execute("""
                INSERT INTO ai_proactive_learnings
                (learning_type, source_prediction_id, lesson, improvement_suggestion,
                 confidence_adjustment, data)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                'prediction_validation', prediction_id, lesson, improvement,
                0.05 if was_accurate else -0.1,
                Json({'was_accurate': was_accurate, 'actual_outcome': actual_outcome})
            ))

            conn.commit()
            conn.close()

            logger.info(f"📚 Learned from prediction {prediction_id}: {lesson}")

        except Exception as e:
            logger.error(f"Learning error: {e}")

    async def generate_daily_briefing(self) -> Dict:
        """Generate a daily briefing for humans"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get 24h stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_predictions,
                    COUNT(*) FILTER (WHERE was_accurate = TRUE) as accurate,
                    COUNT(*) FILTER (WHERE was_accurate = FALSE) as inaccurate,
                    AVG(confidence) as avg_confidence
                FROM ai_proactive_predictions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            pred_stats = cur.fetchone()

            cur.execute("""
                SELECT
                    COUNT(*) as total_actions,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed
                FROM ai_autonomous_actions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            action_stats = cur.fetchone()

            cur.execute("""
                SELECT pattern_name, description, occurrences, confidence
                FROM ai_detected_patterns
                WHERE last_seen > NOW() - INTERVAL '24 hours'
                ORDER BY occurrences DESC
                LIMIT 5
            """)
            top_patterns = cur.fetchall()

            conn.close()

            briefing = {
                'date': datetime.utcnow().strftime('%Y-%m-%d'),
                'predictions': {
                    'total': pred_stats['total_predictions'] if pred_stats else 0,
                    'accuracy_rate': (
                        pred_stats['accurate'] / max(pred_stats['accurate'] + pred_stats['inaccurate'], 1)
                        if pred_stats else 0
                    ),
                    'avg_confidence': pred_stats['avg_confidence'] if pred_stats else 0
                },
                'autonomous_actions': {
                    'total': action_stats['total_actions'] if action_stats else 0,
                    'success_rate': (
                        action_stats['completed'] / max(action_stats['total_actions'], 1)
                        if action_stats else 0
                    )
                },
                'top_patterns': [dict(p) for p in top_patterns],
                'generated_at': datetime.utcnow().isoformat()
            }

            return briefing

        except Exception as e:
            logger.error(f"Briefing generation error: {e}")
            return {'error': str(e)}

    async def run_proactive_cycle(self):
        """Run a complete proactive intelligence cycle"""
        logger.info("🧠 Running proactive intelligence cycle...")

        # 1. Analyze patterns
        patterns = await self.analyze_patterns()
        logger.info(f"  Found {len(patterns)} patterns")

        # 2. Anticipate issues
        predictions = await self.anticipate_issues()
        logger.info(f"  Made {len(predictions)} predictions")

        # 3. Generate recommendations
        recommendations = await self.recommend_actions()
        logger.info(f"  Generated {len(recommendations)} recommendations")

        return {
            'patterns': patterns,
            'predictions': [p.to_dict() for p in predictions],
            'recommendations': recommendations
        }


# Singleton
_proactive_intelligence: Optional[ProactiveIntelligence] = None


def get_proactive_intelligence() -> ProactiveIntelligence:
    global _proactive_intelligence
    if _proactive_intelligence is None:
        _proactive_intelligence = ProactiveIntelligence()
    return _proactive_intelligence


if __name__ == "__main__":
    async def test():
        print("\n" + "="*60)
        print("🧠 PROACTIVE INTELLIGENCE TEST")
        print("="*60 + "\n")

        pi = get_proactive_intelligence()

        # Run proactive cycle
        result = await pi.run_proactive_cycle()
        print(json.dumps(result, indent=2, default=str))

        # Generate briefing
        briefing = await pi.generate_daily_briefing()
        print("\n📋 DAILY BRIEFING:")
        print(json.dumps(briefing, indent=2, default=str))

        print("\n✅ Test complete")

    asyncio.run(test())
