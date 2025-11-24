#!/usr/bin/env python3
"""
Notebook LM+ Style Learning System
Implements continuous learning from all interactions with knowledge synthesis
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor
import openai
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    connections: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    accessed_count: int
    last_accessed: datetime
    synthesis_count: int
    embedding: Optional[List[float]] = None

@dataclass
class LearningSession:
    """Tracks a learning session"""
    session_id: str
    start_time: datetime
    topics: List[str]
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

    def _ensure_tables(self):
        """Create necessary database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
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
            """)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("Notebook LM+ tables initialized")

        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    def start_learning_session(self, topics: List[str] = None) -> str:
        """Start a new learning session"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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
                              context: Dict[str, Any],
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

    def _analyze_content(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content to determine knowledge type and importance"""
        try:
            # Use OpenAI to analyze the content
            response = openai.ChatCompletion.create(
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

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * 1536

    def _find_related_knowledge(self, embedding: List[float], limit: int = 5) -> List[Dict]:
        """Find related knowledge using vector similarity"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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
                json.dumps(kwargs.get('connections', [])),
                json.dumps(kwargs.get('metadata', {})),
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

    def _attempt_synthesis(self, node_ids: List[str]):
        """Attempt to synthesize insights from multiple knowledge nodes"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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

            response = openai.ChatCompletion.create(
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

    def _update_connections(self, knowledge_id: str, related: List[Dict]):
        """Update bidirectional connections between knowledge nodes"""
        try:
            if not related:
                return

            conn = psycopg2.connect(**DB_CONFIG)
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
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            updates = []
            params = []

            for field, value in kwargs.items():
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

    def query_knowledge(self, query: str, limit: int = 10) -> List[Dict]:
        """Query the knowledge base"""
        try:
            # Generate query embedding
            embedding = self._generate_embedding(query)

            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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

    def get_insights(self, category: str = None, min_impact: float = 0.5) -> List[Dict]:
        """Get synthesized insights"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
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

    def end_learning_session(self) -> Dict[str, Any]:
        """End the current learning session and generate summary"""
        if not self.active_session:
            return {"error": "No active session"}

        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Get session stats
            cursor.execute("""
                SELECT * FROM notebook_lm_sessions
                WHERE session_id = %s
            """, (self.active_session,))

            session = cursor.fetchone()

            # Generate summary
            summary = f"Learning session completed. Created {session['nodes_created']} knowledge nodes, "
            summary += f"made {session['connections_made']} connections, "
            summary += f"and generated {session['insights_generated']} insights."

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

            return session

        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return {"error": str(e)}

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
