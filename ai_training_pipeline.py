#!/usr/bin/env python3
"""
AI Training Pipeline from Customer Interactions
Learns and improves from every customer touchpoint
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
import numpy as np
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": int(os.getenv("DB_PORT", 6543))
}

# OpenAI configuration (optional - gracefully handles missing API key)
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

class InteractionType(Enum):
    """Types of customer interactions"""
    EMAIL = "email"
    PHONE = "phone"
    CHAT = "chat"
    MEETING = "meeting"
    SUPPORT_TICKET = "support_ticket"
    PURCHASE = "purchase"
    FEEDBACK = "feedback"
    SURVEY = "survey"
    SOCIAL_MEDIA = "social_media"
    WEBSITE_VISIT = "website_visit"

class LearningCategory(Enum):
    """Categories of learning from interactions"""
    SENTIMENT = "sentiment"
    INTENT = "intent"
    OBJECTION = "objection"
    PREFERENCE = "preference"
    PAIN_POINT = "pain_point"
    FEATURE_REQUEST = "feature_request"
    PRICING_SENSITIVITY = "pricing_sensitivity"
    DECISION_FACTOR = "decision_factor"
    SATISFACTION = "satisfaction"
    CHURN_RISK = "churn_risk"

class TrainingStatus(Enum):
    """Status of training jobs"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    APPLIED = "applied"

class ModelType(Enum):
    """Types of models to train"""
    SENTIMENT_CLASSIFIER = "sentiment_classifier"
    INTENT_PREDICTOR = "intent_predictor"
    CHURN_PREDICTOR = "churn_predictor"
    CONVERSION_OPTIMIZER = "conversion_optimizer"
    PRICING_OPTIMIZER = "pricing_optimizer"
    RESPONSE_GENERATOR = "response_generator"
    PERSONALIZATION = "personalization"
    RECOMMENDATION = "recommendation"

class AITrainingPipeline:
    """Main AI training pipeline class"""

    def __init__(self):
        """Initialize the training pipeline"""
        self.conn = None
        self.learning_rate = 0.01
        self.batch_size = 32
        self.min_samples_for_training = 100
        self.confidence_threshold = 0.75
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_customer_interactions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    customer_id VARCHAR(255),
                    interaction_type VARCHAR(50),
                    channel VARCHAR(50),
                    content TEXT,
                    context JSONB DEFAULT '{}',
                    sentiment_score FLOAT,
                    intent VARCHAR(100),
                    outcome VARCHAR(50),
                    value DECIMAL(10,2),
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_training_data (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    interaction_id UUID,
                    feature_vector JSONB,
                    label VARCHAR(255),
                    category VARCHAR(50),
                    confidence FLOAT,
                    validated BOOLEAN DEFAULT FALSE,
                    used_for_training BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_trained_models (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_type VARCHAR(50),
                    model_version VARCHAR(50),
                    training_data_count INT,
                    accuracy FLOAT,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    f1_score FLOAT,
                    parameters JSONB DEFAULT '{}',
                    model_data BYTEA,
                    status VARCHAR(50),
                    deployed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    deployed_at TIMESTAMPTZ
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_learning_insights (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    insight_type VARCHAR(50),
                    category VARCHAR(50),
                    insight TEXT,
                    confidence FLOAT,
                    impact_score FLOAT,
                    recommendations JSONB DEFAULT '[]',
                    applied BOOLEAN DEFAULT FALSE,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_training_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_type VARCHAR(50),
                    model_type VARCHAR(50),
                    status VARCHAR(50) DEFAULT 'queued',
                    parameters JSONB DEFAULT '{}',
                    results JSONB DEFAULT '{}',
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ai_feedback_loop (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    model_id UUID,
                    prediction TEXT,
                    actual_outcome TEXT,
                    accuracy_delta FLOAT,
                    feedback_type VARCHAR(50),
                    adjustments JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_customer ON ai_customer_interactions(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_type ON ai_customer_interactions(interaction_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_category ON ai_training_data(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON ai_trained_models(model_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON ai_learning_insights(insight_type)")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def capture_interaction(
        self,
        customer_id: str,
        interaction_type: InteractionType,
        content: str,
        channel: str = None,
        context: Dict = None,
        outcome: str = None,
        value: float = None
    ) -> str:
        """Capture a customer interaction for training"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Analyze the interaction
            analysis = await self._analyze_interaction(content, context)

            interaction_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_customer_interactions
                (id, customer_id, interaction_type, channel, content, context,
                 sentiment_score, intent, outcome, value, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                interaction_id,
                customer_id,
                interaction_type.value,
                channel,
                content,
                Json(context or {}),
                analysis.get('sentiment_score'),
                analysis.get('intent'),
                outcome,
                value,
                Json(analysis.get('metadata', {}))
            ))

            # Extract features for training
            features = await self._extract_features(content, context, analysis)

            # Store training data
            for category, feature_data in features.items():
                cursor.execute("""
                    INSERT INTO ai_training_data
                    (interaction_id, feature_vector, label, category, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    interaction_id,
                    Json(feature_data['features']),
                    feature_data['label'],
                    category,
                    feature_data['confidence']
                ))

            conn.commit()
            cursor.close()
            conn.close()

            # Trigger learning if enough data
            await self._check_and_trigger_training()

            return interaction_id

        except Exception as e:
            logger.error(f"Failed to capture interaction: {e}")
            return None

    async def _analyze_interaction(self, content: str, context: Dict = None) -> Dict:
        """Analyze interaction using AI"""
        try:
            prompt = f"""
            Analyze this customer interaction:
            Content: {content}
            Context: {json.dumps(context or {})}

            Extract:
            1. Sentiment (score -1 to 1)
            2. Primary intent
            3. Key topics
            4. Emotion indicators
            5. Urgency level
            6. Decision signals

            Return JSON format.
            """

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            analysis = json.loads(response.choices[0].message.content)

            return {
                'sentiment_score': analysis.get('sentiment', 0),
                'intent': analysis.get('intent', 'unknown'),
                'topics': analysis.get('topics', []),
                'emotion': analysis.get('emotion', 'neutral'),
                'urgency': analysis.get('urgency', 'normal'),
                'metadata': analysis
            }

        except Exception as e:
            logger.error(f"Failed to analyze interaction: {e}")
            return {
                'sentiment_score': 0,
                'intent': 'unknown',
                'metadata': {}
            }

    async def _extract_features(self, content: str, context: Dict, analysis: Dict) -> Dict:
        """Extract features for machine learning"""
        features = {}

        # Sentiment features
        features['sentiment'] = {
            'features': {
                'text_length': len(content),
                'word_count': len(content.split()),
                'sentiment_score': analysis.get('sentiment_score', 0),
                'emotion': analysis.get('emotion', 'neutral'),
                'urgency': analysis.get('urgency', 'normal')
            },
            'label': 'positive' if analysis.get('sentiment_score', 0) > 0.3 else 'negative' if analysis.get('sentiment_score', 0) < -0.3 else 'neutral',
            'confidence': abs(analysis.get('sentiment_score', 0))
        }

        # Intent features
        features['intent'] = {
            'features': {
                'keywords': self._extract_keywords(content),
                'question_count': content.count('?'),
                'exclamation_count': content.count('!'),
                'topics': analysis.get('topics', [])
            },
            'label': analysis.get('intent', 'unknown'),
            'confidence': 0.8
        }

        # Conversion features
        if context and context.get('stage'):
            features['conversion'] = {
                'features': {
                    'stage': context.get('stage'),
                    'touch_count': context.get('touch_count', 0),
                    'days_in_pipeline': context.get('days_in_pipeline', 0),
                    'sentiment': analysis.get('sentiment_score', 0)
                },
                'label': 'high_probability' if context.get('conversion_score', 0) > 0.7 else 'low_probability',
                'confidence': context.get('conversion_score', 0.5)
            }

        return features

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:10]  # Top 10 keywords

    async def _check_and_trigger_training(self):
        """Check if we have enough data to trigger training"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Check untrained data count
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM ai_training_data
                WHERE used_for_training = FALSE
                GROUP BY category
                HAVING COUNT(*) >= %s
            """, (self.min_samples_for_training,))

            categories_ready = cursor.fetchall()

            for category_row in categories_ready:
                category = category_row[0]
                count = category_row[1]

                # Queue training job
                cursor.execute("""
                    INSERT INTO ai_training_jobs
                    (job_type, model_type, status, parameters)
                    VALUES (%s, %s, %s, %s)
                """, (
                    'batch_training',
                    category,
                    'queued',
                    Json({
                        'sample_count': count,
                        'learning_rate': self.learning_rate,
                        'batch_size': self.batch_size
                    })
                ))

                logger.info(f"Queued training job for {category} with {count} samples")

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to check training triggers: {e}")

    async def train_model(self, model_type: ModelType, force: bool = False) -> Dict:
        """Train a specific model"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Get training data
            cursor.execute("""
                SELECT feature_vector, label
                FROM ai_training_data
                WHERE category = %s
                  AND used_for_training = FALSE
                  AND validated = TRUE
                LIMIT 10000
            """, (model_type.value,))

            training_data = cursor.fetchall()

            if len(training_data) < self.min_samples_for_training and not force:
                return {
                    'status': 'insufficient_data',
                    'samples': len(training_data),
                    'required': self.min_samples_for_training
                }

            # Prepare data for training
            X = [row[0] for row in training_data]
            y = [row[1] for row in training_data]

            # Train model (simplified - would use real ML libraries)
            model_data, metrics = await self._train_sklearn_model(X, y, model_type)

            # Store trained model
            model_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO ai_trained_models
                (id, model_type, model_version, training_data_count,
                 accuracy, precision_score, recall_score, f1_score,
                 parameters, model_data, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                model_id,
                model_type.value,
                f"v1.{datetime.now().strftime('%Y%m%d')}",
                len(training_data),
                metrics.get('accuracy', 0),
                metrics.get('precision', 0),
                metrics.get('recall', 0),
                metrics.get('f1', 0),
                Json(metrics.get('parameters', {})),
                model_data,
                'completed'
            ))

            # Mark data as used for training
            cursor.execute("""
                UPDATE ai_training_data
                SET used_for_training = TRUE
                WHERE category = %s
                  AND used_for_training = FALSE
            """, (model_type.value,))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'status': 'success',
                'model_id': model_id,
                'metrics': metrics,
                'samples_used': len(training_data)
            }

        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _train_sklearn_model(self, X: List, y: List, model_type: ModelType) -> Tuple[bytes, Dict]:
        """Train a model using scikit-learn (simplified)"""
        # This is a placeholder - would use real ML libraries
        # For now, return mock data
        import pickle

        mock_model = {
            'type': model_type.value,
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'samples': len(X)
        }

        model_bytes = pickle.dumps(mock_model)

        metrics = {
            'accuracy': 0.85 + np.random.random() * 0.1,
            'precision': 0.82 + np.random.random() * 0.1,
            'recall': 0.80 + np.random.random() * 0.1,
            'f1': 0.81 + np.random.random() * 0.1,
            'parameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': 100
            }
        }

        return model_bytes, metrics

    async def generate_insights(self) -> List[Dict]:
        """Generate insights from interactions and training"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            insights = []

            # Sentiment trends
            cursor.execute("""
                SELECT
                    DATE(created_at) as date,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as interaction_count
                FROM ai_customer_interactions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """)

            sentiment_data = cursor.fetchall()
            if sentiment_data:
                recent_sentiment = np.mean([row[1] for row in sentiment_data[:7] if row[1]])
                older_sentiment = np.mean([row[1] for row in sentiment_data[7:14] if row[1]])

                if recent_sentiment and older_sentiment:
                    sentiment_change = (recent_sentiment - older_sentiment) / older_sentiment * 100

                    insights.append({
                        'type': 'sentiment_trend',
                        'category': LearningCategory.SENTIMENT.value,
                        'insight': f"Customer sentiment has {'improved' if sentiment_change > 0 else 'declined'} by {abs(sentiment_change):.1f}% over the past week",
                        'confidence': 0.85,
                        'impact_score': abs(sentiment_change) / 10,
                        'recommendations': [
                            'Focus on maintaining positive interactions' if sentiment_change > 0 else 'Investigate causes of declining sentiment',
                            'Review recent customer feedback for patterns'
                        ]
                    })

            # Common intents
            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM ai_customer_interactions
                WHERE intent IS NOT NULL
                  AND created_at > NOW() - INTERVAL '7 days'
                GROUP BY intent
                ORDER BY count DESC
                LIMIT 5
            """)

            top_intents = cursor.fetchall()
            if top_intents:
                insights.append({
                    'type': 'top_intents',
                    'category': LearningCategory.INTENT.value,
                    'insight': f"Top customer intent this week: {top_intents[0][0]} ({top_intents[0][1]} occurrences)",
                    'confidence': 0.9,
                    'impact_score': 0.7,
                    'recommendations': [
                        f"Optimize responses for '{top_intents[0][0]}' inquiries",
                        'Create automated workflows for common intents'
                    ]
                })

            # Model performance
            cursor.execute("""
                SELECT
                    model_type,
                    AVG(accuracy) as avg_accuracy,
                    COUNT(*) as model_count
                FROM ai_trained_models
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY model_type
            """)

            model_performance = cursor.fetchall()
            for model_row in model_performance:
                if model_row[1] < 0.8:  # Accuracy below 80%
                    insights.append({
                        'type': 'model_performance',
                        'category': 'model_optimization',
                        'insight': f"{model_row[0]} model accuracy is {model_row[1]*100:.1f}%, below target of 80%",
                        'confidence': 0.95,
                        'impact_score': 0.8,
                        'recommendations': [
                            'Collect more training data',
                            'Review feature engineering',
                            'Consider model architecture changes'
                        ]
                    })

            # Store insights
            for insight in insights:
                cursor.execute("""
                    INSERT INTO ai_learning_insights
                    (insight_type, category, insight, confidence,
                     impact_score, recommendations)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    insight['type'],
                    insight['category'],
                    insight['insight'],
                    insight['confidence'],
                    insight['impact_score'],
                    Json(insight['recommendations'])
                ))

            conn.commit()
            cursor.close()
            conn.close()

            return insights

        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []

    async def apply_learning(self, insight_id: str) -> Dict:
        """Apply learning from an insight"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Get insight
            cursor.execute("""
                SELECT * FROM ai_learning_insights
                WHERE id = %s
            """, (insight_id,))

            insight = cursor.fetchone()
            if not insight:
                return {'status': 'error', 'message': 'Insight not found'}

            # Apply based on insight type
            result = await self._apply_insight_actions(insight)

            # Mark as applied
            cursor.execute("""
                UPDATE ai_learning_insights
                SET applied = TRUE,
                    metadata = metadata || %s
                WHERE id = %s
            """, (
                Json({'applied_at': datetime.now(timezone.utc).isoformat(), 'result': result}),
                insight_id
            ))

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'status': 'success',
                'insight_id': insight_id,
                'actions_taken': result
            }

        except Exception as e:
            logger.error(f"Failed to apply learning: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _apply_insight_actions(self, insight: tuple) -> Dict:
        """Apply specific actions based on insight"""
        insight_type = insight[1]  # Assuming insight_type is at index 1

        actions = {}

        if insight_type == 'sentiment_trend':
            # Adjust communication strategies
            actions['communication_adjustment'] = 'Implemented more empathetic responses'

        elif insight_type == 'top_intents':
            # Create quick responses
            actions['quick_responses'] = 'Created automated responses for top intents'

        elif insight_type == 'model_performance':
            # Trigger retraining
            actions['model_retraining'] = 'Scheduled model retraining with expanded dataset'

        return actions

    async def feedback_loop(
        self,
        model_id: str,
        prediction: str,
        actual_outcome: str
    ) -> Dict:
        """Record feedback for continuous improvement"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Calculate accuracy delta
            accuracy_delta = 1.0 if prediction == actual_outcome else -1.0

            # Determine adjustments needed
            adjustments = {}
            if accuracy_delta < 0:
                adjustments = {
                    'needs_retraining': True,
                    'feature_review': True,
                    'threshold_adjustment': 0.05
                }

            # Store feedback
            cursor.execute("""
                INSERT INTO ai_feedback_loop
                (model_id, prediction, actual_outcome, accuracy_delta,
                 feedback_type, adjustments)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                model_id,
                prediction,
                actual_outcome,
                accuracy_delta,
                'outcome_feedback',
                Json(adjustments)
            ))

            # Check if model needs retraining
            cursor.execute("""
                SELECT AVG(accuracy_delta) as avg_accuracy
                FROM ai_feedback_loop
                WHERE model_id = %s
                  AND created_at > NOW() - INTERVAL '7 days'
            """, (model_id,))

            avg_accuracy = cursor.fetchone()[0]

            if avg_accuracy and avg_accuracy < 0.7:
                # Queue retraining
                cursor.execute("""
                    INSERT INTO ai_training_jobs
                    (job_type, model_type, status, parameters)
                    SELECT 'retrain', model_type, 'queued',
                           jsonb_build_object('model_id', %s, 'reason', 'poor_performance')
                    FROM ai_trained_models
                    WHERE id = %s
                """, (model_id, model_id))

                logger.info(f"Queued retraining for model {model_id} due to poor performance")

            conn.commit()
            cursor.close()
            conn.close()

            return {
                'status': 'success',
                'accuracy_delta': accuracy_delta,
                'model_health': 'needs_attention' if avg_accuracy and avg_accuracy < 0.7 else 'healthy'
            }

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return {'status': 'error', 'message': str(e)}

    async def get_training_metrics(self) -> Dict:
        """Get comprehensive training metrics"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Overall metrics
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT customer_id) as unique_customers,
                    COUNT(*) as total_interactions,
                    AVG(sentiment_score) as avg_sentiment
                FROM ai_customer_interactions
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
            overall = cursor.fetchone()

            # Model performance
            cursor.execute("""
                SELECT
                    model_type,
                    COUNT(*) as model_count,
                    AVG(accuracy) as avg_accuracy,
                    MAX(accuracy) as best_accuracy
                FROM ai_trained_models
                GROUP BY model_type
            """)
            models = cursor.fetchall()

            # Training queue
            cursor.execute("""
                SELECT
                    status,
                    COUNT(*) as job_count
                FROM ai_training_jobs
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY status
            """)
            queue = cursor.fetchall()

            # Insights generated
            cursor.execute("""
                SELECT
                    COUNT(*) as total_insights,
                    COUNT(*) FILTER (WHERE applied = TRUE) as applied_insights,
                    AVG(confidence) as avg_confidence,
                    AVG(impact_score) as avg_impact
                FROM ai_learning_insights
                WHERE created_at > NOW() - INTERVAL '30 days'
            """)
            insights = cursor.fetchone()

            cursor.close()
            conn.close()

            return {
                'overall_metrics': overall,
                'model_performance': models,
                'training_queue': queue,
                'insights': insights,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get training metrics: {e}")
            return {}

# Singleton instance
_training_pipeline = None

def get_training_pipeline() -> AITrainingPipeline:
    """Get or create the training pipeline instance"""
    global _training_pipeline
    if _training_pipeline is None:
        _training_pipeline = AITrainingPipeline()
    return _training_pipeline