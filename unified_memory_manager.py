#!/usr/bin/env python3
"""
Unified Memory Manager - Enterprise Grade Memory System
Consolidates 53 chaotic memory tables into one intelligent system
"""

import os
import json
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_values
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder for datetime, Decimal, and Enum types
from decimal import Decimal

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': 6543
}


class MemoryType(Enum):
    """Types of memory in the unified system"""
    EPISODIC = "episodic"      # Specific events and experiences
    SEMANTIC = "semantic"      # Facts and knowledge
    PROCEDURAL = "procedural"  # How to do things
    WORKING = "working"        # Short-term active memory
    META = "meta"             # Memory about memories


@dataclass
class Memory:
    """Unified memory structure"""
    memory_type: MemoryType
    content: Dict[str, Any]
    source_system: str
    source_agent: str
    created_by: str
    importance_score: float = 0.5
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    context_id: Optional[str] = None
    parent_memory_id: Optional[str] = None
    related_memories: List[str] = None
    expires_at: Optional[datetime] = None
    tenant_id: str = "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"


class UnifiedMemoryManager:
    """Enterprise-grade unified memory management system"""

    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.embedding_cache = {}
        self.consolidation_threshold = 0.85  # Similarity threshold for consolidation
        self._connect()

    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Connected to unified memory system")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

    def _get_cursor(self):
        """Get database cursor with auto-reconnect"""
        try:
            self.conn.cursor().execute("SELECT 1")
        except:
            self._connect()
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def store(self, memory: Memory) -> str:
        """Store a memory with deduplication and linking"""
        try:
            # Check for duplicates
            existing = self._find_duplicate(memory)
            if existing:
                # Reinforce existing memory instead of creating duplicate
                return self._reinforce_memory(existing['id'], memory)

            # Generate embedding if we have content
            embedding = self._generate_embedding(memory.content)

            # Find related memories
            related = self._find_related_memories(memory.content, limit=5)

            # Store the memory
            with self._get_cursor() as cur:
                query = """
                INSERT INTO unified_ai_memory (
                    memory_type, content, source_system, source_agent,
                    created_by, importance_score, tags, metadata,
                    context_id, parent_memory_id, related_memories,
                    expires_at, tenant_id, embedding, search_text
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid[], %s, %s, %s, %s
                ) RETURNING id
                """

                search_text = self._generate_search_text(memory)

                # Serialize with custom encoder to handle datetime and Enum
                content_json = json.dumps(memory.content, cls=CustomJSONEncoder)
                metadata_json = json.dumps(memory.metadata or {}, cls=CustomJSONEncoder)

                cur.execute(query, (
                    memory.memory_type.value,
                    content_json,
                    memory.source_system,
                    memory.source_agent,
                    memory.created_by,
                    memory.importance_score,
                    memory.tags or [],
                    metadata_json,
                    memory.context_id,
                    memory.parent_memory_id,
                    [r['id'] for r in related] if related else None,
                    memory.expires_at,
                    memory.tenant_id,
                    embedding,
                    search_text
                ))

                memory_id = cur.fetchone()['id']
                self.conn.commit()

                logger.info(f"✅ Stored memory {memory_id} ({memory.memory_type.value})")
                return memory_id

        except Exception as e:
            logger.error(f"❌ Failed to store memory: {e}")
            self.conn.rollback()
            raise

    def recall(self, query: Union[str, Dict], context: Optional[str] = None,
               limit: int = 10, memory_type: Optional[MemoryType] = None) -> List[Dict]:
        """Recall relevant memories with semantic search"""
        try:
            # Generate query embedding
            if isinstance(query, str):
                query_content = {"query": query}
            else:
                query_content = query

            query_embedding = self._generate_embedding(query_content)

            with self._get_cursor() as cur:
                # Build the query
                base_query = """
                SELECT
                    id, memory_type, content, source_system, source_agent,
                    importance_score, access_count, created_at, tags, metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM unified_ai_memory
                WHERE tenant_id = %s
                """

                params = [query_embedding, self.db_config.get('tenant_id', '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457')]

                # Add filters
                filters = []
                if context:
                    filters.append("context_id = %s")
                    params.append(context)

                if memory_type:
                    filters.append("memory_type = %s")
                    params.append(memory_type.value)

                if filters:
                    base_query += " AND " + " AND ".join(filters)

                # Order by relevance (similarity + importance)
                base_query += """
                ORDER BY (1 - (embedding <=> %s::vector)) * importance_score DESC
                LIMIT %s
                """
                params.extend([query_embedding, limit])

                cur.execute(base_query, params)
                memories = cur.fetchall()

                # Update access counts
                if memories:
                    memory_ids = [m['id'] for m in memories]
                    self._update_access_counts(memory_ids)

                logger.info(f"📚 Recalled {len(memories)} relevant memories")
                return [dict(m) for m in memories]

        except Exception as e:
            logger.error(f"❌ Failed to recall memories: {e}")
            return []

    def synthesize(self, time_window: timedelta = timedelta(hours=24)) -> List[Dict]:
        """Synthesize insights from recent memories"""
        try:
            with self._get_cursor() as cur:
                # Get recent high-importance memories
                query = """
                SELECT
                    memory_type, source_system, content, importance_score, tags
                FROM unified_ai_memory
                WHERE created_at > %s
                    AND importance_score > 0.7
                    AND tenant_id = %s
                ORDER BY importance_score DESC
                LIMIT 100
                """

                cutoff_time = datetime.now() - time_window
                cur.execute(query, (cutoff_time, self.db_config.get('tenant_id', '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457')))
                memories = cur.fetchall()

                insights = []

                # Group by type and look for patterns
                patterns = self._identify_patterns(memories)

                for pattern in patterns:
                    insight = {
                        'type': 'pattern',
                        'confidence': pattern['confidence'],
                        'description': pattern['description'],
                        'supporting_memories': pattern['memory_count'],
                        'recommended_action': pattern['action'],
                        'impact': pattern['estimated_impact'],
                        'discovered_at': datetime.now().isoformat()
                    }
                    insights.append(insight)

                    # Store the insight as a meta memory
                    self.store(Memory(
                        memory_type=MemoryType.META,
                        content=insight,
                        source_system='memory_synthesis',
                        source_agent='synthesizer',
                        created_by='unified_memory_manager',
                        importance_score=pattern['confidence'],
                        tags=['insight', 'pattern', 'synthesis']
                    ))

                logger.info(f"🧠 Synthesized {len(insights)} insights from {len(memories)} memories")
                return insights

        except Exception as e:
            logger.error(f"❌ Failed to synthesize insights: {e}")
            return []

    def consolidate(self, aggressive: bool = False):
        """Consolidate similar memories to reduce redundancy"""
        try:
            threshold = 0.7 if aggressive else self.consolidation_threshold

            with self._get_cursor() as cur:
                # Find similar memories
                query = """
                WITH similarity_pairs AS (
                    SELECT
                        m1.id as id1, m2.id as id2,
                        m1.content as content1, m2.content as content2,
                        1 - (m1.embedding <=> m2.embedding) as similarity
                    FROM unified_ai_memory m1
                    JOIN unified_ai_memory m2 ON m1.id < m2.id
                    WHERE m1.memory_type = m2.memory_type
                        AND m1.source_system = m2.source_system
                        AND 1 - (m1.embedding <=> m2.embedding) > %s
                )
                SELECT * FROM similarity_pairs
                ORDER BY similarity DESC
                LIMIT 100
                """

                cur.execute(query, (threshold,))
                similar_pairs = cur.fetchall()

                consolidated_count = 0

                for pair in similar_pairs:
                    # Merge the memories
                    merged_content = self._merge_memories(
                        pair['content1'],
                        pair['content2']
                    )

                    # Update the first memory with merged content
                    update_query = """
                    UPDATE unified_ai_memory
                    SET content = %s,
                        importance_score = importance_score + 0.1,
                        reinforcement_count = reinforcement_count + 1
                    WHERE id = %s
                    """
                    cur.execute(update_query, (Json(merged_content), pair['id1']))

                    # Mark the second memory as consolidated
                    delete_query = """
                    UPDATE unified_ai_memory
                    SET expires_at = NOW(),
                        metadata = jsonb_set(
                            COALESCE(metadata, '{}'::jsonb),
                            '{consolidated_into}',
                            %s::jsonb
                        )
                    WHERE id = %s
                    """
                    cur.execute(delete_query, (
                        json.dumps(str(pair['id1'])),
                        pair['id2']
                    ))

                    consolidated_count += 1

                self.conn.commit()
                logger.info(f"♻️ Consolidated {consolidated_count} memory pairs")

        except Exception as e:
            logger.error(f"❌ Failed to consolidate memories: {e}")
            self.conn.rollback()

    def migrate_from_chaos(self, limit: int = 1000):
        """Migrate data from the 53 chaotic memory tables"""
        tables_to_migrate = [
            'ai_context_memory',
            'ai_persistent_memory',
            'agent_memory',
            'cross_ai_memory',
            'production_memory',
            'unified_memory',  # ironic that there was already one called this
            'system_memory'
        ]

        total_migrated = 0

        for table in tables_to_migrate:
            try:
                migrated = self._migrate_table(table, limit)
                total_migrated += migrated
                logger.info(f"📦 Migrated {migrated} memories from {table}")
            except Exception as e:
                logger.warning(f"⚠️ Could not migrate from {table}: {e}")

        logger.info(f"✅ Total migrated: {total_migrated} memories from chaos to order")

    def _migrate_table(self, table_name: str, limit: int) -> int:
        """Migrate a single table to unified memory"""
        try:
            with self._get_cursor() as cur:
                # First check if table exists
                check_query = """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
                """
                cur.execute(check_query, (table_name,))
                if not cur.fetchone()['exists']:
                    return 0

                # Get recent data from old table
                query = f"""
                SELECT * FROM {table_name}
                WHERE created_at > NOW() - INTERVAL '30 days'
                ORDER BY created_at DESC
                LIMIT %s
                """
                cur.execute(query, (limit,))
                old_memories = cur.fetchall()

                migrated = 0
                for old_mem in old_memories:
                    try:
                        # Convert to unified format
                        memory = Memory(
                            memory_type=MemoryType.SEMANTIC,
                            content=old_mem.get('content') or old_mem.get('memory_data') or dict(old_mem),
                            source_system=table_name,
                            source_agent='migrated',
                            created_by='migration',
                            importance_score=old_mem.get('importance', 0.5),
                            tags=['migrated', table_name],
                            metadata={'original_id': str(old_mem.get('id', ''))},
                            created_at=old_mem.get('created_at')
                        )
                        self.store(memory)
                        migrated += 1
                    except:
                        continue

                return migrated

        except Exception as e:
            logger.error(f"Migration error for {table_name}: {e}")
            return 0

    def _find_duplicate(self, memory: Memory) -> Optional[Dict]:
        """Find duplicate memory using content hash"""
        content_str = json.dumps(memory.content, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()

        with self._get_cursor() as cur:
            query = """
            SELECT id, importance_score, reinforcement_count
            FROM unified_ai_memory
            WHERE content_hash = %s
                AND memory_type = %s
                AND source_system = %s
            LIMIT 1
            """
            cur.execute(query, (content_hash, memory.memory_type.value, memory.source_system))
            return cur.fetchone()

    def _reinforce_memory(self, memory_id: str, new_memory: Memory) -> str:
        """Reinforce existing memory instead of duplicating"""
        with self._get_cursor() as cur:
            query = """
            UPDATE unified_ai_memory
            SET importance_score = LEAST(importance_score + 0.05, 1.0),
                reinforcement_count = reinforcement_count + 1,
                access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = %s
            RETURNING id
            """
            cur.execute(query, (memory_id,))
            self.conn.commit()
            logger.info(f"💪 Reinforced existing memory {memory_id}")
            return memory_id

    def _find_related_memories(self, content: Dict, limit: int = 5) -> List[Dict]:
        """Find memories related to the given content"""
        embedding = self._generate_embedding(content)

        with self._get_cursor() as cur:
            query = """
            SELECT id, memory_type, importance_score,
                   1 - (embedding <=> %s::vector) as similarity
            FROM unified_ai_memory
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            cur.execute(query, (embedding, embedding, limit))
            return cur.fetchall()

    def _generate_embedding(self, content: Dict) -> Optional[List[float]]:
        """Generate real embedding for content using OpenAI"""
        try:
            import openai
            
            # Extract text content
            text_content = json.dumps(content, sort_keys=True)
            
            # Call OpenAI Embedding API
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                input=text_content,
                model="text-embedding-3-small"
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            # Fallback to zero vector to prevent crash, but log clearly
            return [0.0] * 1536

    def _generate_search_text(self, memory: Memory) -> str:
        """Generate searchable text from memory"""
        parts = []

        # Add content values
        if isinstance(memory.content, dict):
            for key, value in memory.content.items():
                if isinstance(value, str):
                    parts.append(value)

        # Add tags
        if memory.tags:
            parts.extend(memory.tags)

        # Add metadata values
        if memory.metadata:
            for value in memory.metadata.values():
                if isinstance(value, str):
                    parts.append(value)

        return ' '.join(parts)

    def _update_access_counts(self, memory_ids: List[str]):
        """Update access counts for recalled memories"""
        with self._get_cursor() as cur:
            query = """
            UPDATE unified_ai_memory
            SET access_count = access_count + 1,
                last_accessed = NOW()
            WHERE id = ANY(%s)
            """
            cur.execute(query, (memory_ids,))
            self.conn.commit()

    def _identify_patterns(self, memories: List[Dict]) -> List[Dict]:
        """Identify patterns in memories (simplified version)"""
        patterns = []

        # Group by source system
        system_groups = {}
        for mem in memories:
            system = mem.get('source_system', 'unknown')
            if system not in system_groups:
                system_groups[system] = []
            system_groups[system].append(mem)

        # Look for patterns in each group
        for system, group_memories in system_groups.items():
            if len(group_memories) >= 3:
                # Pattern detected
                pattern = {
                    'confidence': min(0.9, len(group_memories) / 10),
                    'description': f"High activity in {system} system",
                    'memory_count': len(group_memories),
                    'action': f"Optimize {system} for increased load",
                    'estimated_impact': f"{len(group_memories) * 100} operations/day"
                }
                patterns.append(pattern)

        return patterns

    def _merge_memories(self, content1: Dict, content2: Dict) -> Dict:
        """Merge two similar memories"""
        merged = content1.copy()

        # Add unique keys from content2
        for key, value in content2.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, list) and isinstance(merged[key], list):
                # Merge lists
                merged[key] = list(set(merged[key] + value))
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Merge dicts recursively
                merged[key] = {**merged[key], **value}

        return merged

    def get_stats(self) -> Dict:
        """Get memory system statistics"""
        with self._get_cursor() as cur:
            query = """
            SELECT
                COUNT(*) as total_memories,
                COUNT(DISTINCT source_system) as unique_systems,
                COUNT(DISTINCT source_agent) as unique_agents,
                AVG(importance_score) as avg_importance,
                MAX(access_count) as max_access_count,
                COUNT(DISTINCT context_id) as unique_contexts
            FROM unified_ai_memory
            WHERE tenant_id = %s
            """
            cur.execute(query, (self.db_config.get('tenant_id', '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'),))
            return dict(cur.fetchone())


# Singleton instance
memory_manager = None

def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create the singleton memory manager"""
    global memory_manager
    if memory_manager is None:
        memory_manager = UnifiedMemoryManager()
    return memory_manager


if __name__ == "__main__":
    # Test the memory system
    manager = get_memory_manager()

    # Store a test memory
    test_memory = Memory(
        memory_type=MemoryType.SEMANTIC,
        content={"test": "Unified memory system operational", "timestamp": datetime.now().isoformat()},
        source_system="test",
        source_agent="initializer",
        created_by="system",
        importance_score=0.9,
        tags=["test", "initialization"]
    )

    memory_id = manager.store(test_memory)
    print(f"✅ Stored test memory: {memory_id}")

    # Get stats
    stats = manager.get_stats()
    print(f"📊 Memory Stats: {json.dumps(stats, indent=2)}")

    # Migrate from chaos
    print("🔄 Starting migration from chaotic tables...")
    manager.migrate_from_chaos(limit=100)

    print("✅ Unified Memory Manager operational!")