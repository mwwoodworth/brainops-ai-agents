#!/usr/bin/env python3
"""
Vector-Based Persistent Memory System
Implements semantic memory with embeddings for true AI memory persistence
"""

import os
import json
import logging
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

import openai
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import register_adapter, AsIs

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register numpy float adapter for PostgreSQL
def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
register_adapter(np.float64, addapt_numpy_float64)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Memory:
    """Represents a memory item with embedding"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    importance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int

class VectorMemorySystem:
    """Advanced vector-based memory system with semantic search"""

    def __init__(self):
        """Initialize the vector memory system"""
        self.embedding_model = "text-embedding-3-small"
        self.embedding_dimension = 1536
        self.conn = None
        self._ensure_tables()
        logger.info("Vector Memory System initialized")

    def _ensure_tables(self):
        """Ensure all required tables exist"""
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Create comprehensive memory tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                memory_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536),
                importance_score FLOAT DEFAULT 0.5,
                confidence_score FLOAT DEFAULT 1.0,
                decay_rate FLOAT DEFAULT 0.01,
                last_accessed TIMESTAMPTZ DEFAULT NOW(),
                access_count INT DEFAULT 0,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ
            );

            CREATE TABLE IF NOT EXISTS memory_associations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                memory_id_1 UUID REFERENCES vector_memories(id) ON DELETE CASCADE,
                memory_id_2 UUID REFERENCES vector_memories(id) ON DELETE CASCADE,
                association_strength FLOAT DEFAULT 0.5,
                association_type VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(memory_id_1, memory_id_2)
            );

            CREATE TABLE IF NOT EXISTS memory_consolidation (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source_memories UUID[],
                consolidated_content TEXT,
                consolidation_type VARCHAR(50),
                quality_score FLOAT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_vector_memories_embedding
                ON vector_memories USING ivfflat (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_vector_memories_importance
                ON vector_memories(importance_score DESC);
            CREATE INDEX IF NOT EXISTS idx_vector_memories_type
                ON vector_memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memory_associations_strength
                ON memory_associations(association_strength DESC);
        """)

        conn.commit()
        cursor.close()
        conn.close()

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = openai.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dimension

    def store_memory(
        self,
        content: str,
        memory_type: str = "general",
        metadata: Dict = None,
        importance: float = 0.5
    ) -> str:
        """Store a new memory with embedding"""
        try:
            # Generate embedding
            embedding = self._get_embedding(content)

            # Store in database
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            memory_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO vector_memories
                (id, memory_type, content, embedding, importance_score, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                memory_id,
                memory_type,
                content,
                embedding,
                importance,
                json.dumps(metadata or {})
            ))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Stored memory {memory_id} with importance {importance}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return None

    def recall_memories(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        threshold: float = 0.7
    ) -> List[Dict]:
        """Recall relevant memories using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)

            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Semantic search with cosine similarity
            if memory_type:
                cursor.execute("""
                    SELECT
                        id, memory_type, content, importance_score,
                        confidence_score, metadata, created_at,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM vector_memories
                    WHERE memory_type = %s
                        AND 1 - (embedding <=> %s::vector) > %s
                    ORDER BY
                        similarity DESC,
                        importance_score DESC
                    LIMIT %s
                """, (query_embedding, memory_type, query_embedding, threshold, limit))
            else:
                cursor.execute("""
                    SELECT
                        id, memory_type, content, importance_score,
                        confidence_score, metadata, created_at,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM vector_memories
                    WHERE 1 - (embedding <=> %s::vector) > %s
                    ORDER BY
                        similarity DESC,
                        importance_score DESC
                    LIMIT %s
                """, (query_embedding, query_embedding, threshold, limit))

            memories = cursor.fetchall()

            # Update access counts
            if memories:
                memory_ids = [m['id'] for m in memories]
                cursor.execute("""
                    UPDATE vector_memories
                    SET access_count = access_count + 1,
                        last_accessed = NOW()
                    WHERE id = ANY(%s)
                """, (memory_ids,))
                conn.commit()

            cursor.close()
            conn.close()

            logger.info(f"Recalled {len(memories)} memories for query")
            return memories

        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []

    def associate_memories(
        self,
        memory_id_1: str,
        memory_id_2: str,
        strength: float = 0.5,
        association_type: str = "related"
    ):
        """Create association between two memories"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO memory_associations
                (memory_id_1, memory_id_2, association_strength, association_type)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (memory_id_1, memory_id_2)
                DO UPDATE SET
                    association_strength = EXCLUDED.association_strength,
                    association_type = EXCLUDED.association_type
            """, (memory_id_1, memory_id_2, strength, association_type))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Associated memories {memory_id_1} <-> {memory_id_2}")

        except Exception as e:
            logger.error(f"Failed to associate memories: {e}")

    def consolidate_memories(
        self,
        memory_ids: List[str],
        consolidation_type: str = "summary"
    ) -> Optional[str]:
        """Consolidate multiple memories into a single memory"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            # Retrieve memories to consolidate
            cursor.execute("""
                SELECT id, content, importance_score, metadata
                FROM vector_memories
                WHERE id = ANY(%s)
            """, (memory_ids,))

            memories = cursor.fetchall()

            if not memories:
                return None

            # Combine content for consolidation
            combined_content = "\n\n".join([m['content'] for m in memories])
            avg_importance = sum(m['importance_score'] for m in memories) / len(memories)

            # Generate consolidated memory
            if consolidation_type == "summary":
                prompt = f"Summarize these related memories into a single cohesive memory:\n{combined_content}"
            else:
                prompt = f"Combine these memories into a comprehensive understanding:\n{combined_content}"

            # Use OpenAI to consolidate
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a memory consolidation system."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )

            consolidated_content = response.choices[0].message.content

            # Store consolidated memory
            new_memory_id = self.store_memory(
                consolidated_content,
                memory_type="consolidated",
                metadata={
                    "source_memories": memory_ids,
                    "consolidation_type": consolidation_type,
                    "original_count": len(memories)
                },
                importance=min(avg_importance * 1.2, 1.0)  # Boost importance
            )

            # Record consolidation
            cursor.execute("""
                INSERT INTO memory_consolidation
                (source_memories, consolidated_content, consolidation_type, quality_score)
                VALUES (%s, %s, %s, %s)
            """, (memory_ids, consolidated_content, consolidation_type, avg_importance))

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Consolidated {len(memories)} memories into {new_memory_id}")
            return new_memory_id

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            return None

    def decay_memories(self):
        """Apply decay to memories based on access patterns"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Apply time-based decay
            cursor.execute("""
                UPDATE vector_memories
                SET importance_score = GREATEST(
                    importance_score - (decay_rate * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400),
                    0.1
                )
                WHERE last_accessed < NOW() - INTERVAL '7 days'
            """)

            # Boost frequently accessed memories
            cursor.execute("""
                UPDATE vector_memories
                SET importance_score = LEAST(
                    importance_score + (0.01 * LOG(access_count + 1)),
                    1.0
                )
                WHERE access_count > 10
            """)

            conn.commit()
            cursor.close()
            conn.close()

            logger.info("Applied memory decay and reinforcement")

        except Exception as e:
            logger.error(f"Failed to decay memories: {e}")

    def get_memory_statistics(self) -> Dict:
        """Get statistics about the memory system"""
        try:
            conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_memories,
                    COUNT(DISTINCT memory_type) as memory_types,
                    AVG(importance_score) as avg_importance,
                    AVG(access_count) as avg_access_count,
                    MAX(created_at) as latest_memory,
                    MIN(created_at) as oldest_memory
                FROM vector_memories
            """)

            stats = cursor.fetchone()

            cursor.execute("""
                SELECT memory_type, COUNT(*) as count
                FROM vector_memories
                GROUP BY memory_type
                ORDER BY count DESC
            """)

            type_distribution = cursor.fetchall()

            cursor.close()
            conn.close()

            return {
                "statistics": stats,
                "type_distribution": type_distribution
            }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}

    def prune_old_memories(self, days: int = 90):
        """Remove old, unimportant memories"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM vector_memories
                WHERE created_at < NOW() - INTERVAL '%s days'
                    AND importance_score < 0.3
                    AND access_count < 5
            """, (days,))

            deleted = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            logger.info(f"Pruned {deleted} old memories")
            return deleted

        except Exception as e:
            logger.error(f"Failed to prune memories: {e}")
            return 0

# Global instance - create lazily to avoid DB connection on import
vector_memory = None

def get_vector_memory():
    """Get or create vector memory instance"""
    global vector_memory
    if vector_memory is None:
        vector_memory = VectorMemorySystem()
    return vector_memory

# For backward compatibility
def init_vector_memory():
    """Initialize vector memory system"""
    return get_vector_memory()