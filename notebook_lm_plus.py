#!/usr/bin/env python3
"""
Notebook LM+ Style Learning System
Implements continuous learning from all interactions with knowledge synthesis
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
import psycopg2
from openai import OpenAI
from psycopg2.extras import RealDictCursor

# Initialize OpenAI client
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client singleton"""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
    return _openai_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - uses centralized config
from config import config as app_config


def get_db_config():
    """Get database config from centralized config module"""
    db = app_config.database
    return {
        "host": db.host,
        "database": db.database,
        "user": db.user,
        "password": db.password,
        "port": db.port
    }

# Lazy-loaded for backward compatibility
DB_CONFIG = None

def _get_db_config():
    global DB_CONFIG
    if DB_CONFIG is None:
        DB_CONFIG = get_db_config()
    return DB_CONFIG

# OpenAI configuration handled by get_openai_client() function above

class KnowledgeType(Enum):
    """Types of knowledge that can be learned"""
    CONCEPT = "concept"
    PROCEDURE = "procedure"
    FACT = "fact"
    PATTERN = "pattern"
    RELATIONSHIP = "relationship"
    INSIGHT = "insight"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    CONTEXT = "context"
    SYNTHESIS = "synthesis"

class LearningSource(Enum):
    """Sources of learning data"""
    USER_INTERACTION = "user_interaction"
    AGENT_EXECUTION = "agent_execution"
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_SEARCH = "web_search"
    DATABASE_QUERY = "database_query"
    ERROR_ANALYSIS = "error_analysis"
    PATTERN_DETECTION = "pattern_detection"
    CROSS_REFERENCE = "cross_reference"

@dataclass
class KnowledgeNode:
    """Represents a piece of learned knowledge"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    source: LearningSource
    confidence: float
    importance: float
    connections: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    accessed_count: int
    last_accessed: datetime
    synthesis_count: int
    embedding: Optional[list[float]] = None

@dataclass
class LearningSession:
    """Tracks a learning session"""
    session_id: str
    start_time: datetime
    topics: list[str]
    nodes_created: int
    connections_made: int
    insights_generated: int
    confidence_avg: float
    status: str

class NotebookLMPlus:
    """Advanced learning system inspired by Notebook LM with enhanced capabilities"""

    def __init__(self):
        """Initialize the learning system"""
        self.conn = None
        self._ensure_tables()
        self.active_session = None
        self.knowledge_graph = {}
        self.synthesis_threshold = 5  # Min nodes for synthesis
        self.pattern_recognition_enabled = True
        self.continuous_learning_enabled = True
        self.outcome_tracker = {}

    def _ensure_tables(self):
        """Create necessary database tables"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Knowledge nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_lm_knowledge (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    content TEXT NOT NULL,
                    knowledge_type VARCHAR(50) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    importance FLOAT DEFAULT 0.5,
                    connections JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    embedding vector(1536),
                    accessed_count INT DEFAULT 0,
                    last_accessed TIMESTAMPTZ DEFAULT NOW(),
                    synthesis_count INT DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Learning sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_lm_sessions (
                    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    start_time TIMESTAMPTZ DEFAULT NOW(),
                    end_time TIMESTAMPTZ,
                    topics JSONB DEFAULT '[]'::jsonb,
                    nodes_created INT DEFAULT 0,
                    connections_made INT DEFAULT 0,
                    insights_generated INT DEFAULT 0,
                    confidence_avg FLOAT DEFAULT 0.0,
                    status VARCHAR(20) DEFAULT 'active',
                    summary TEXT
                )
            """)

            # Synthesized insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_lm_insights (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    insight TEXT NOT NULL,
                    source_nodes JSONB NOT NULL,
                    confidence FLOAT DEFAULT 0.5,
                    impact_score FLOAT DEFAULT 0.5,
                    category VARCHAR(100),
                    applied_count INT DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    validated BOOLEAN DEFAULT FALSE
                )
            """)

            # Pattern recognition table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_lm_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_type VARCHAR(50) NOT NULL,
                    pattern_name VARCHAR(255) NOT NULL,
                    description TEXT,
                    occurrences INT DEFAULT 1,
                    confidence FLOAT DEFAULT 0.5,
                    supporting_nodes JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_seen TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Outcome tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_lm_outcomes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    knowledge_id UUID REFERENCES notebook_lm_knowledge(id),
                    action_taken TEXT,
                    expected_result TEXT,
                    actual_result TEXT,
                    success BOOLEAN,
                    feedback_score FLOAT,
                    learned_improvement TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_type
                ON notebook_lm_knowledge(knowledge_type);

                CREATE INDEX IF NOT EXISTS idx_knowledge_importance
                ON notebook_lm_knowledge(importance DESC);

                CREATE INDEX IF NOT EXISTS idx_knowledge_confidence
                ON notebook_lm_knowledge(confidence DESC);

                CREATE INDEX IF NOT EXISTS idx_insights_impact
                ON notebook_lm_insights(impact_score DESC);

                CREATE INDEX IF NOT EXISTS idx_patterns_type
                ON notebook_lm_patterns(pattern_type);

                CREATE INDEX IF NOT EXISTS idx_patterns_confidence
                ON notebook_lm_patterns(confidence DESC);

                CREATE INDEX IF NOT EXISTS idx_outcomes_success
                ON notebook_lm_outcomes(success);
            """)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("Notebook LM+ tables initialized")

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def start_learning_session(self, topics: list[str] = None) -> str:
        """Start a new learning session"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO notebook_lm_sessions (topics)
                VALUES (%s)
                RETURNING session_id
            """, (json.dumps(topics or []),))

            result = cursor.fetchone()
            self.active_session = result['session_id']

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Started learning session: {self.active_session}")
            return self.active_session

        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return None

    def learn_from_interaction(self,
                              content: str,
                              context: dict[str, Any],
                              source: LearningSource = LearningSource.USER_INTERACTION) -> Optional[str]:
        """Learn from an interaction and store knowledge"""
        try:
            # Analyze the content
            analysis = self._analyze_content(content, context)

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Find related knowledge
            related = self._find_related_knowledge(embedding, limit=5)

            # Store the knowledge
            knowledge_id = self._store_knowledge(
                content=analysis['processed_content'],
                knowledge_type=analysis['type'],
                source=source,
                confidence=analysis['confidence'],
                importance=analysis['importance'],
                connections=[r['id'] for r in related],
                metadata={**context, 'analysis': analysis},
                embedding=embedding
            )

            # Update connections
            self._update_connections(knowledge_id, related)

            # Check for synthesis opportunities
            if len(related) >= self.synthesis_threshold:
                self._attempt_synthesis([knowledge_id] + [r['id'] for r in related])

            # Update session stats
            if self.active_session:
                self._update_session_stats(nodes_created=1, connections_made=len(related))

            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
            return None

    def _analyze_content(self, content: str, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze content to determine knowledge type and importance"""
        try:
            # Use OpenAI to analyze the content
            client = get_openai_client()
            if not client:
                return {"importance": 0.5, "confidence": 0.5, "knowledge_type": "concept"}

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze this content and determine its knowledge type, importance, and confidence. Respond in JSON format."},
                    {"role": "user", "content": f"Content: {content}\nContext: {json.dumps(context)}"}
                ],
                temperature=0.3
            )

            analysis = json.loads(response.choices[0].message.content)

            # Map to our knowledge types
            knowledge_type = KnowledgeType.CONCEPT  # Default
            if 'procedure' in content.lower() or 'step' in content.lower():
                knowledge_type = KnowledgeType.PROCEDURE
            elif 'error' in content.lower() or 'fix' in content.lower():
                knowledge_type = KnowledgeType.CORRECTION
            elif 'pattern' in content.lower() or 'trend' in content.lower():
                knowledge_type = KnowledgeType.PATTERN

            return {
                'processed_content': content,
                'type': knowledge_type,
                'confidence': min(analysis.get('confidence', 0.5), 1.0),
                'importance': min(analysis.get('importance', 0.5), 1.0),
                'topics': analysis.get('topics', []),
                'entities': analysis.get('entities', [])
            }

        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                'processed_content': content,
                'type': KnowledgeType.FACT,
                'confidence': 0.3,
                'importance': 0.5,
                'topics': [],
                'entities': []
            }

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI"""
        try:
            client = get_openai_client()
            if not client:
                logger.warning("OpenAI client unavailable; returning zero embedding")
                return [0.0] * 1536
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 1536

    def _find_related_knowledge(self, embedding: list[float], limit: int = 5) -> list[dict]:
        """Find related knowledge using vector similarity"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Convert to string format for pgvector
            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            cursor.execute("""
                SELECT
                    id, content, knowledge_type, confidence, importance,
                    1 - (embedding <=> %s::vector) as similarity
                FROM notebook_lm_knowledge
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, limit))

            related = cursor.fetchall()
            cursor.close()
            conn.close()

            return [r for r in related if r['similarity'] > 0.7]

        except Exception as e:
            logger.error(f"Failed to find related knowledge: {e}")
            return []

    def _store_knowledge(self, **kwargs) -> str:
        """Store knowledge in the database"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            embedding_str = None
            if kwargs.get('embedding'):
                embedding_str = '[' + ','.join(map(str, kwargs['embedding'])) + ']'

            cursor.execute("""
                INSERT INTO notebook_lm_knowledge (
                    content, knowledge_type, source, confidence, importance,
                    connections, metadata, embedding
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                RETURNING id
            """, (
                kwargs['content'],
                kwargs['knowledge_type'].value,
                kwargs['source'].value,
                kwargs['confidence'],
                kwargs['importance'],
                json.dumps(kwargs.get('connections', []), default=str),
                json.dumps(kwargs.get('metadata', {}), default=str),
                embedding_str
            ))

            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()

            return str(result['id'])

        except Exception as e:
            logger.error(f"Failed to store knowledge: {e}")
            return None

    def _attempt_synthesis(self, node_ids: list[str]):
        """Attempt to synthesize insights from multiple knowledge nodes"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Get the nodes
            cursor.execute("""
                SELECT content, knowledge_type, confidence, importance
                FROM notebook_lm_knowledge
                WHERE id = ANY(%s)
            """, (node_ids,))

            nodes = cursor.fetchall()

            if len(nodes) < 3:
                return  # Not enough for synthesis

            # Use AI to synthesize
            combined_content = "\n".join([n['content'] for n in nodes])

            client = get_openai_client()
            if not client:
                return  # Skip synthesis without AI

            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "Synthesize insights from these related pieces of knowledge. Identify patterns, connections, and higher-level insights."},
                    {"role": "user", "content": combined_content}
                ],
                temperature=0.5
            )

            insight = response.choices[0].message.content

            # Calculate confidence and impact
            avg_confidence = np.mean([n['confidence'] for n in nodes])
            avg_importance = np.mean([n['importance'] for n in nodes])
            impact_score = avg_confidence * avg_importance

            # Store the insight
            cursor.execute("""
                INSERT INTO notebook_lm_insights (
                    insight, source_nodes, confidence, impact_score
                ) VALUES (%s, %s, %s, %s)
            """, (insight, json.dumps(node_ids), avg_confidence, impact_score))

            # Update synthesis count for nodes
            cursor.execute("""
                UPDATE notebook_lm_knowledge
                SET synthesis_count = synthesis_count + 1
                WHERE id = ANY(%s)
            """, (node_ids,))

            conn.commit()
            cursor.close()
            conn.close()

            # Update session stats
            if self.active_session:
                self._update_session_stats(insights_generated=1)

            logger.info(f"Synthesized insight from {len(nodes)} nodes")

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")

    def _update_connections(self, knowledge_id: str, related: list[dict]):
        """Update bidirectional connections between knowledge nodes"""
        try:
            if not related:
                return

            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            for r in related:
                # Update the related node to include this connection
                cursor.execute("""
                    UPDATE notebook_lm_knowledge
                    SET connections = connections || %s::jsonb
                    WHERE id = %s AND NOT connections @> %s::jsonb
                """, (
                    json.dumps([knowledge_id]),
                    r['id'],
                    json.dumps([knowledge_id])
                ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update connections: {e}")

    def _update_session_stats(self, **kwargs):
        """Update current session statistics"""
        if not self.active_session:
            return

        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            updates = []
            params = []

            # Whitelist of allowed columns to prevent SQL injection
            ALLOWED_STAT_COLUMNS = {'total_interactions', 'knowledge_items_added', 'insights_generated', 'patterns_discovered'}

            for field, value in kwargs.items():
                # Validate field against whitelist
                if field not in ALLOWED_STAT_COLUMNS:
                    logger.warning(f"Ignoring unknown stat column: {field}")
                    continue
                updates.append(f"{field} = {field} + %s")
                params.append(value)

            if updates:
                params.append(self.active_session)
                cursor.execute(f"""
                    UPDATE notebook_lm_sessions
                    SET {', '.join(updates)}
                    WHERE session_id = %s
                """, params)

                conn.commit()

            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to update session stats: {e}")

    def query_knowledge(self, query: str, limit: int = 10) -> list[dict]:
        """Query the knowledge base"""
        try:
            # Generate query embedding
            embedding = self._generate_embedding(query)

            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            embedding_str = '[' + ','.join(map(str, embedding)) + ']'

            cursor.execute("""
                SELECT
                    id, content, knowledge_type, source, confidence, importance,
                    created_at, accessed_count,
                    1 - (embedding <=> %s::vector) as relevance
                FROM notebook_lm_knowledge
                WHERE embedding IS NOT NULL
                ORDER BY (embedding <=> %s::vector) ASC
                LIMIT %s
            """, (embedding_str, embedding_str, limit))

            results = cursor.fetchall()

            # Update access counts
            if results:
                cursor.execute("""
                    UPDATE notebook_lm_knowledge
                    SET accessed_count = accessed_count + 1,
                        last_accessed = NOW()
                    WHERE id = ANY(%s)
                """, ([r['id'] for r in results],))

                conn.commit()

            cursor.close()
            conn.close()

            return results

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return []

    def get_insights(self, category: str = None, min_impact: float = 0.5) -> list[dict]:
        """Get synthesized insights"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            query = """
                SELECT insight, confidence, impact_score, created_at, validated
                FROM notebook_lm_insights
                WHERE impact_score >= %s
            """
            params = [min_impact]

            if category:
                query += " AND category = %s"
                params.append(category)

            query += " ORDER BY impact_score DESC LIMIT 20"

            cursor.execute(query, params)
            insights = cursor.fetchall()

            cursor.close()
            conn.close()

            return insights

        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            return []

    def recognize_patterns(self, timeframe_days: int = 7) -> list[dict]:
        """Recognize patterns across historical data"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Identify recurring knowledge types
            cursor.execute("""
                SELECT
                    knowledge_type,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    ARRAY_AGG(id) as node_ids
                FROM notebook_lm_knowledge
                WHERE created_at > NOW() - INTERVAL '%s days'
                GROUP BY knowledge_type
                HAVING COUNT(*) >= 3
                ORDER BY count DESC
            """, (timeframe_days,))

            type_patterns = cursor.fetchall()
            patterns = []

            for pattern in type_patterns:
                # Store pattern
                cursor.execute("""
                    INSERT INTO notebook_lm_patterns
                    (pattern_type, pattern_name, description, occurrences, confidence, supporting_nodes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pattern_type, pattern_name)
                    DO UPDATE SET
                        occurrences = EXCLUDED.occurrences,
                        confidence = EXCLUDED.confidence,
                        last_seen = NOW()
                    RETURNING id
                """, (
                    'knowledge_type_frequency',
                    f"frequent_{pattern['knowledge_type']}",
                    f"Frequent occurrence of {pattern['knowledge_type']} knowledge",
                    pattern['count'],
                    pattern['avg_confidence'],
                    json.dumps([str(nid) for nid in pattern['node_ids']])
                ))

                patterns.append({
                    "type": "frequency",
                    "pattern": pattern['knowledge_type'],
                    "count": pattern['count'],
                    "confidence": float(pattern['avg_confidence'])
                })

            # Identify temporal patterns
            cursor.execute("""
                SELECT
                    DATE_TRUNC('hour', created_at) as hour,
                    knowledge_type,
                    COUNT(*) as count
                FROM notebook_lm_knowledge
                WHERE created_at > NOW() - INTERVAL '%s days'
                GROUP BY DATE_TRUNC('hour', created_at), knowledge_type
                HAVING COUNT(*) >= 2
                ORDER BY hour DESC, count DESC
            """, (timeframe_days,))

            temporal_patterns = cursor.fetchall()
            for tpattern in temporal_patterns[:5]:  # Top 5
                patterns.append({
                    "type": "temporal",
                    "pattern": f"{tpattern['knowledge_type']}_at_{tpattern['hour'].hour}h",
                    "count": tpattern['count'],
                    "hour": tpattern['hour'].hour
                })

            conn.commit()
            cursor.close()
            conn.close()

            return patterns

        except Exception as e:
            logger.error(f"Pattern recognition failed: {e}")
            return []

    def track_outcome(self, knowledge_id: str, action: str, expected: str, actual: str, success: bool) -> None:
        """Track outcome of applied knowledge for continuous learning"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Calculate feedback score
            feedback_score = 1.0 if success else 0.0

            # Determine learned improvement
            learned_improvement = ""
            if not success:
                learned_improvement = f"Action '{action}' did not produce expected result '{expected}'. Actual: '{actual}'"

            cursor.execute("""
                INSERT INTO notebook_lm_outcomes
                (knowledge_id, action_taken, expected_result, actual_result, success, feedback_score, learned_improvement)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (knowledge_id, action, expected, actual, success, feedback_score, learned_improvement))

            # Update knowledge confidence based on outcome
            if success:
                cursor.execute("""
                    UPDATE notebook_lm_knowledge
                    SET confidence = LEAST(1.0, confidence + 0.05),
                        importance = LEAST(1.0, importance + 0.03)
                    WHERE id = %s
                """, (knowledge_id,))
            else:
                cursor.execute("""
                    UPDATE notebook_lm_knowledge
                    SET confidence = GREATEST(0.1, confidence - 0.1)
                    WHERE id = %s
                """, (knowledge_id,))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Tracked outcome for knowledge {knowledge_id}: {'success' if success else 'failure'}")

        except Exception as e:
            logger.error(f"Failed to track outcome: {e}")

    def generate_recommendations(self) -> list[dict]:
        """Generate actionable recommendations from learned knowledge"""
        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            recommendations = []

            # Recommend based on high-confidence knowledge
            cursor.execute("""
                SELECT id, content, knowledge_type, confidence, importance
                FROM notebook_lm_knowledge
                WHERE confidence > 0.8 AND importance > 0.7
                ORDER BY importance DESC, confidence DESC
                LIMIT 5
            """)

            high_value = cursor.fetchall()
            for item in high_value:
                recommendations.append({
                    "type": "apply_knowledge",
                    "priority": "high",
                    "recommendation": f"Apply high-confidence {item['knowledge_type']}: {item['content'][:100]}",
                    "confidence": float(item['confidence']),
                    "knowledge_id": str(item['id'])
                })

            # Recommend based on successful outcomes
            cursor.execute("""
                SELECT k.id, k.content, k.knowledge_type, COUNT(o.id) as success_count
                FROM notebook_lm_knowledge k
                JOIN notebook_lm_outcomes o ON o.knowledge_id = k.id
                WHERE o.success = TRUE
                GROUP BY k.id, k.content, k.knowledge_type
                HAVING COUNT(o.id) >= 2
                ORDER BY success_count DESC
                LIMIT 3
            """)

            proven_knowledge = cursor.fetchall()
            for item in proven_knowledge:
                recommendations.append({
                    "type": "proven_strategy",
                    "priority": "high",
                    "recommendation": f"Repeat successful {item['knowledge_type']}: {item['content'][:100]}",
                    "success_count": item['success_count'],
                    "knowledge_id": str(item['id'])
                })

            # Recommend pattern-based actions
            cursor.execute("""
                SELECT pattern_name, description, confidence, occurrences
                FROM notebook_lm_patterns
                WHERE confidence > 0.7
                ORDER BY occurrences DESC, confidence DESC
                LIMIT 3
            """)

            patterns = cursor.fetchall()
            for pattern in patterns:
                recommendations.append({
                    "type": "pattern_based",
                    "priority": "medium",
                    "recommendation": f"Leverage pattern: {pattern['description']}",
                    "confidence": float(pattern['confidence']),
                    "occurrences": pattern['occurrences']
                })

            cursor.close()
            conn.close()

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return []

    def end_learning_session(self) -> dict[str, Any]:
        """End the current learning session and generate summary with patterns"""
        if not self.active_session:
            return {"error": "No active session"}

        try:
            conn = psycopg2.connect(**_get_db_config(), cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Get session stats
            cursor.execute("""
                SELECT * FROM notebook_lm_sessions
                WHERE session_id = %s
            """, (self.active_session,))

            session = cursor.fetchone()

            # Recognize patterns from this session
            patterns = self.recognize_patterns(timeframe_days=1)

            # Generate recommendations
            recommendations = self.generate_recommendations()

            # Generate summary
            summary = f"Learning session completed. Created {session['nodes_created']} knowledge nodes, "
            summary += f"made {session['connections_made']} connections, "
            summary += f"and generated {session['insights_generated']} insights. "
            summary += f"Recognized {len(patterns)} patterns and generated {len(recommendations)} recommendations."

            # Update session
            cursor.execute("""
                UPDATE notebook_lm_sessions
                SET end_time = NOW(),
                    status = 'completed',
                    summary = %s
                WHERE session_id = %s
            """, (summary, self.active_session))

            conn.commit()
            cursor.close()
            conn.close()

            self.active_session = None

            return {
                **dict(session),
                "patterns_recognized": patterns,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return {"error": str(e)}

    async def record_execution(
        self,
        agent: str,
        task_type: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        success: bool
    ) -> Optional[str]:
        """
        Record an agent execution for learning.
        This is the async interface expected by unified_system_integration.py.

        Args:
            agent: Name of the agent that executed
            task_type: Type of task executed
            input_data: Input data provided to the agent
            output_data: Output data returned by the agent
            success: Whether the execution succeeded

        Returns:
            Knowledge ID if recorded successfully, None otherwise
        """
        try:
            # Build learning content from execution
            content = f"Agent '{agent}' executed '{task_type}' task"
            if success:
                content += " successfully"
            else:
                content += " with failure"

            # Add relevant details from output
            if isinstance(output_data, dict):
                if output_data.get('result'):
                    content += f". Result: {str(output_data.get('result'))[:200]}"
                if output_data.get('error'):
                    content += f". Error: {str(output_data.get('error'))[:200]}"

            # Build context
            context = {
                "agent": agent,
                "task_type": task_type,
                "success": success,
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "input_summary": str(input_data)[:500] if input_data else None,
                "output_summary": str(output_data)[:500] if output_data else None
            }

            # Use the existing learn_from_interaction method
            knowledge_id = self.learn_from_interaction(
                content=content,
                context=context,
                source=LearningSource.AGENT_EXECUTION
            )

            logger.info(f"Recorded execution for {agent}/{task_type}: success={success}")
            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to record execution: {e}")
            return None

    async def record_error(
        self,
        agent: str,
        task_type: str,
        error: str,
        context: dict[str, Any]
    ) -> Optional[str]:
        """
        Record an agent error for learning.
        This is the async interface expected by unified_system_integration.py.

        Args:
            agent: Name of the agent that failed
            task_type: Type of task that failed
            error: Error message
            context: Additional context about the error

        Returns:
            Knowledge ID if recorded successfully, None otherwise
        """
        try:
            # Build error learning content
            content = f"Agent '{agent}' encountered error during '{task_type}': {error[:300]}"

            # Build learning context
            error_context = {
                "agent": agent,
                "task_type": task_type,
                "error_type": "execution_failure",
                "error_message": error,
                "error_timestamp": datetime.now(timezone.utc).isoformat(),
                **context
            }

            # Use the existing learn_from_interaction method
            knowledge_id = self.learn_from_interaction(
                content=content,
                context=error_context,
                source=LearningSource.ERROR_ANALYSIS
            )

            logger.warning(f"Recorded error for {agent}/{task_type}: {error[:100]}")
            return knowledge_id

        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            return None

# Global instance - create lazily
notebook_lm = None

def get_notebook_lm():
    """Get or create notebook LM instance"""
    global notebook_lm
    if notebook_lm is None:
        notebook_lm = NotebookLMPlus()
    return notebook_lm

# Example usage
if __name__ == "__main__":
    nlm = get_notebook_lm()

    # Start a learning session
    session_id = nlm.start_learning_session(["AI", "agents", "production"])

    # Learn from some interactions
    nlm.learn_from_interaction(
        "The agent execution endpoint requires a task_execution_id to be created first",
        {"source": "debugging", "importance": "high"}
    )

    nlm.learn_from_interaction(
        "Render deployments can take 5-10 minutes to complete after pushing to GitHub",
        {"source": "experience", "category": "deployment"}
    )

    # Query knowledge
    results = nlm.query_knowledge("How long do deployments take?")
    for r in results:
        print(f"- {r['content']} (relevance: {r['relevance']:.2f})")

    # End session
    summary = nlm.end_learning_session()
    print(f"Session summary: {summary}")
