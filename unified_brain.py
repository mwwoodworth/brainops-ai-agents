#!/usr/bin/env python3
"""
UNIFIED BRAIN - The Single Source of Truth for All BrainOps Memory
Consolidates 98 fragmented memory tables into ONE coherent system
Designed for Claude Code + Codex integration

ENHANCED FEATURES:
- Semantic search using OpenAI embeddings + pgvector
- Automatic summarization of stored content
- Cross-referencing between related entries
- Expiration/TTL for temporary entries
- Importance scoring based on access patterns
- Vector similarity queries

ASYNC VERSION: Uses asyncpg for non-blocking database operations
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import re

# Use async database connection
from database.async_connection import get_pool, init_pool, PoolConfig

logger = logging.getLogger(__name__)

# Database configuration - NO hardcoded credentials
# All values MUST come from environment variables in production
DB_CONFIG = PoolConfig(
    host=os.getenv("DB_HOST"),  # Required - no default
    database=os.getenv("DB_NAME", "postgres"),
    user=os.getenv("DB_USER"),  # Required - no default
    password=os.getenv("DB_PASSWORD"),  # Required - no default
    port=int(os.getenv("DB_PORT", "5432")),
    min_size=2,
    max_size=10,  # Increased from 4 for production load
    ssl=os.getenv("DB_SSL", "true").lower() not in ("false", "0", "no"),
    ssl_verify=os.getenv("DB_SSL_VERIFY", "false").lower() not in ("false", "0", "no")
)

# Validate required config
if not all([DB_CONFIG.host, DB_CONFIG.user, DB_CONFIG.password]):
    logger.warning("⚠️ Database credentials not fully configured - set DB_HOST, DB_USER, DB_PASSWORD environment variables")

# OpenAI configuration for embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Try to import OpenAI for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
    if OPENAI_AVAILABLE:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai_client = None
        print("⚠️ OpenAI API key not found. Semantic search will be limited.")
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None
    print("⚠️ OpenAI not installed. Semantic search disabled.")


@dataclass
class BrainContext:
    """Unified context structure"""
    key: str
    value: Any
    category: str  # system, session, architecture, deployment, issue
    priority: str  # critical, high, medium, low
    last_updated: datetime
    source: str  # where this came from
    metadata: Dict[str, Any]


class UnifiedBrain:
    """
    The ONE unified memory system for ALL BrainOps operations
    Replaces 98 fragmented tables with a single coherent interface
    Now integrated with embedded memory for ultra-fast RAG search
    Uses async asyncpg for non-blocking database operations
    """

    def __init__(self, lazy_init: bool = True):
        self.embedded_memory = None
        self._initialized = False
        self._initializing = False  # Reentry guard
        self._table_checked = False  # Track table existence separately
        self._lazy_init = lazy_init
        self._pool_initialized = False

    async def _ensure_pool(self):
        """Ensure the database pool is initialized"""
        if not self._pool_initialized:
            try:
                await init_pool(DB_CONFIG)
                self._pool_initialized = True
                logger.info("✅ UnifiedBrain database pool initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize database pool: {e}")
                raise

    async def _ensure_table(self):
        """Ensure the unified_brain table exists"""
        if self._table_checked:
            return

        await self._ensure_pool()
        pool = get_pool()

        try:
            # First ensure vector extension exists
            await pool.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create main table with enhanced columns
            await pool.execute("""
                CREATE TABLE IF NOT EXISTS unified_brain (
                    id SERIAL PRIMARY KEY,
                    key TEXT UNIQUE NOT NULL,
                    value JSONB NOT NULL,
                    category TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    source TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    access_count INT DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),

                    -- NEW ENHANCED FEATURES
                    embedding vector(1536),  -- OpenAI embedding dimension
                    summary TEXT,  -- Auto-generated summary
                    importance_score FLOAT DEFAULT 0.5,  -- Calculated importance
                    expires_at TIMESTAMPTZ,  -- Optional TTL
                    last_accessed TIMESTAMPTZ DEFAULT NOW(),
                    access_frequency FLOAT DEFAULT 0.0,  -- Access per day
                    related_keys TEXT[],  -- Cross-references
                    tags TEXT[]  -- Searchable tags
                );
            """)

            # Create indexes
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_key ON unified_brain(key);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_category ON unified_brain(category);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_priority ON unified_brain(priority);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_updated ON unified_brain(last_updated DESC);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_importance ON unified_brain(importance_score DESC);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_expires ON unified_brain(expires_at) WHERE expires_at IS NOT NULL;")

            # Vector similarity index (IVFFlat for fast approximate search)
            try:
                await pool.execute("""
                    CREATE INDEX IF NOT EXISTS idx_unified_brain_embedding ON unified_brain
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except Exception as e:
                logger.warning(f"⚠️ Could not create vector index (may need more data): {e}")

            # GIN indexes for array searching
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_tags ON unified_brain USING GIN(tags);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_unified_brain_related ON unified_brain USING GIN(related_keys);")

            # Create cross-reference tracking table
            await pool.execute("""
                CREATE TABLE IF NOT EXISTS brain_references (
                    id SERIAL PRIMARY KEY,
                    from_key TEXT NOT NULL,
                    to_key TEXT NOT NULL,
                    reference_type TEXT NOT NULL,  -- 'related', 'superseded', 'depends_on', 'derived_from'
                    strength FLOAT DEFAULT 1.0,  -- How strong is this relationship
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(from_key, to_key, reference_type)
                );
            """)

            await pool.execute("CREATE INDEX IF NOT EXISTS idx_brain_ref_from ON brain_references(from_key);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_brain_ref_to ON brain_references(to_key);")
            await pool.execute("CREATE INDEX IF NOT EXISTS idx_brain_ref_type ON brain_references(reference_type);")

            self._table_checked = True
            logger.info("✅ UnifiedBrain enhanced table ensured with vector search")
        except Exception as e:
            logger.warning(f"⚠️ Table creation (may already exist): {e}")

    async def _init_embedded_memory(self):
        """Initialize embedded memory system for RAG search - LAZY"""
        if self.embedded_memory is not None:
            return self.embedded_memory
        try:
            from embedded_memory_system import get_embedded_memory
            self.embedded_memory = await get_embedded_memory()
            logger.info("✅ Embedded memory integrated with UnifiedBrain")
            return self.embedded_memory
        except Exception as e:
            logger.warning(f"⚠️ Embedded memory not available: {e}")
            self.embedded_memory = None
            return None

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text using OpenAI"""
        if not OPENAI_AVAILABLE or not openai_client:
            return None

        try:
            # Convert text to string if it's not
            if not isinstance(text, str):
                text = json.dumps(text)

            # Truncate if too long (OpenAI limit is ~8k tokens, ~32k chars)
            if len(text) > 30000:
                text = text[:30000] + "..."

            response = openai_client.embeddings.create(
                model="text-embedding-3-small",  # Cheaper and faster than ada-002
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate embedding: {e}")
            return None

    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate automatic summary of content"""
        if not isinstance(text, str):
            text = json.dumps(text)

        # Simple extractive summary - take first sentences up to max_length
        sentences = re.split(r'(?<=[.!?])\s+', text)
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) < max_length:
                summary += sentence + " "
            else:
                break

        if not summary:
            # Fallback: just truncate
            summary = text[:max_length] + ("..." if len(text) > max_length else "")

        return summary.strip()

    def _calculate_importance_score(self, priority: str, access_count: int,
                                   category: str, age_days: float) -> float:
        """
        Calculate importance score based on multiple factors
        Score range: 0.0 to 1.0
        """
        # Base score from priority
        priority_scores = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.5,
            'low': 0.3
        }
        score = priority_scores.get(priority.lower(), 0.5)

        # Boost for access patterns (logarithmic scaling)
        if access_count > 0:
            access_boost = min(0.15, 0.03 * (access_count ** 0.5) / 10)
            score += access_boost

        # Boost for critical categories
        critical_categories = {'system', 'architecture', 'deployment'}
        if category in critical_categories:
            score += 0.05

        # Decay for age (older = less important, unless frequently accessed)
        if age_days > 30 and access_count < 5:
            age_penalty = min(0.1, (age_days - 30) / 1000)
            score -= age_penalty

        # Clamp to 0.0 - 1.0
        return max(0.0, min(1.0, score))

    def _extract_tags(self, key: str, value: Any, category: str) -> List[str]:
        """Extract searchable tags from content"""
        tags = set()

        # Add category as tag
        tags.add(category)

        # Extract from key (split on underscore and dash)
        key_parts = re.split(r'[_\-]', key.lower())
        tags.update(p for p in key_parts if len(p) > 2)

        # Extract from value if it's a string
        if isinstance(value, str):
            # Find potential tags (capitalized words, tech terms)
            words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', value)
            tags.update(w.lower() for w in words if len(w) > 3)

        # Extract from value if it's a dict
        if isinstance(value, dict):
            for k in value.keys():
                if isinstance(k, str):
                    tags.add(k.lower())

        return list(tags)[:20]  # Limit to 20 tags

    async def _find_related_entries(self, key: str, embedding: Optional[List[float]],
                             limit: int = 5) -> List[str]:
        """Find related entries using vector similarity"""
        if not embedding:
            return []

        await self._ensure_pool()
        pool = get_pool()

        try:
            # Find similar entries using cosine similarity
            rows = await pool.fetch("""
                SELECT key, 1 - (embedding <=> $1::vector) as similarity
                FROM unified_brain
                WHERE key != $2 AND embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """, str(embedding), key, limit)

            related = [row['key'] for row in rows if row.get('similarity', 0) > 0.7]
            return related
        except Exception as e:
            logger.warning(f"⚠️ Failed to find related entries: {e}")
            return []

    async def store(self, key: str, value: Any, category: str = "general",
              priority: str = "medium", source: str = "manual",
              metadata: Optional[Dict] = None, ttl_hours: Optional[int] = None) -> str:
        """
        Store or update a piece of context with enhanced features

        Args:
            key: Unique identifier for this context
            value: The actual data (will be JSON serialized)
            category: system, session, architecture, deployment, issue
            priority: critical, high, medium, low
            source: Where this came from (claude_code, codex, api, manual)
            metadata: Additional context about this entry
            ttl_hours: Optional time-to-live in hours (for temporary data)
        """
        await self._ensure_table()
        pool = get_pool()

        # Generate enhanced features
        text_content = value if isinstance(value, str) else json.dumps(value)
        embedding = self._generate_embedding(text_content)
        summary = self._generate_summary(text_content)
        tags = self._extract_tags(key, value, category)

        # Calculate expiration if TTL specified
        expires_at = None
        if ttl_hours:
            expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

        # Initial importance score (will be recalculated on access)
        importance_score = self._calculate_importance_score(
            priority=priority,
            access_count=0,
            category=category,
            age_days=0
        )

        # Store with all enhanced features
        result = await pool.fetchrow("""
            INSERT INTO unified_brain (
                key, value, category, priority, source, metadata,
                embedding, summary, importance_score, expires_at, tags,
                last_updated, last_accessed
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                category = EXCLUDED.category,
                priority = EXCLUDED.priority,
                source = EXCLUDED.source,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                summary = EXCLUDED.summary,
                importance_score = EXCLUDED.importance_score,
                expires_at = EXCLUDED.expires_at,
                tags = EXCLUDED.tags,
                last_updated = NOW(),
                access_count = unified_brain.access_count + 1
            RETURNING id, created_at
        """, key, json.dumps(value), category, priority, source,
            json.dumps(metadata or {}), str(embedding) if embedding else None, summary,
            importance_score, expires_at, tags)

        entry_id = str(result['id'])
        related_keys = []

        # Find and store related entries using vector similarity
        if embedding:
            try:
                related_keys = await self._find_related_entries(key, embedding)
                if related_keys:
                    await pool.execute("""
                        UPDATE unified_brain
                        SET related_keys = $1
                        WHERE key = $2
                    """, related_keys, key)

                    # Also create bidirectional references
                    for related_key in related_keys:
                        await self._add_reference(key, related_key, 'related', 0.8)
            except Exception as e:
                logger.warning(f"⚠️ Failed to update related keys: {e}")

        # DUAL-WRITE: Also store in embedded memory for fast RAG search
        if self.embedded_memory:
            try:
                importance = 0.9 if priority == 'critical' else 0.7 if priority == 'high' else 0.5
                self.embedded_memory.store_memory(
                    content=text_content,
                    memory_type=category,
                    importance_score=importance,
                    metadata={
                        **({'source': source} if source else {}),
                        **(metadata or {}),
                        'brain_key': key,
                        'brain_id': entry_id
                    }
                )
                logger.info(f"✅ Dual-write: Stored '{key}' with semantic search enabled")
            except Exception as e:
                logger.warning(f"⚠️ Embedded memory store failed: {e} (data still in Postgres)")

        logger.info(f"✅ Stored '{key}' with embedding, summary, and {len(related_keys)} related entries")
        return entry_id

    async def _add_reference(self, from_key: str, to_key: str,
                      reference_type: str = 'related', strength: float = 1.0):
        """Add a cross-reference between two entries"""
        await self._ensure_pool()
        pool = get_pool()

        try:
            await pool.execute("""
                INSERT INTO brain_references (from_key, to_key, reference_type, strength)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (from_key, to_key, reference_type) DO UPDATE
                SET strength = EXCLUDED.strength
            """, from_key, to_key, reference_type, strength)
        except Exception as e:
            logger.warning(f"⚠️ Failed to add reference: {e}")

    async def store_learning(self, agent_id: str, task_id: str, mistake: str,
                       lesson: str, root_cause: str, impact: str = "medium") -> bool:
        """
        Store a specific learning/insight derived from a mistake or success.

        Args:
            agent_id: The agent that learned this
            task_id: The task where it happened
            mistake: Description of the mistake/situation
            lesson: The actionable lesson learned
            root_cause: Why it happened
            impact: critical, high, medium, low
        """
        # Ensure initialization
        if self.embedded_memory is None:
            await self._init_embedded_memory()

        if self.embedded_memory:
            try:
                import uuid
                learning_id = str(uuid.uuid4())
                return self.embedded_memory.store_learning(
                    learning_id=learning_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    mistake_description=mistake,
                    root_cause=root_cause,
                    lesson_learned=lesson,
                    impact_level=impact
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to store learning: {e}")
                return False
        return False

    async def get(self, key: str, include_related: bool = False) -> Optional[Dict]:
        """
        Retrieve a piece of context with enhanced tracking

        Args:
            key: The key to retrieve
            include_related: Whether to include related entries
        """
        await self._ensure_table()
        pool = get_pool()

        # First check if entry has expired
        expiry_check = await pool.fetchrow("""
            SELECT expires_at FROM unified_brain
            WHERE key = $1 AND expires_at IS NOT NULL
        """, key)

        if expiry_check and expiry_check['expires_at'] < datetime.now(timezone.utc):
            # Entry has expired, delete it
            await pool.execute("DELETE FROM unified_brain WHERE key = $1", key)
            return None

        # Get the entry and update access tracking
        result = await pool.fetchrow("""
            SELECT *,
                   EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days,
                   EXTRACT(EPOCH FROM (NOW() - last_accessed))/86400.0 as days_since_access
            FROM unified_brain
            WHERE key = $1
        """, key)

        if not result:
            return None

        # Convert Record to dict
        result = dict(result)

        # Recalculate importance score based on access patterns
        age_days = result['age_days'] or 0
        access_count = result['access_count'] + 1  # Including this access
        new_importance = self._calculate_importance_score(
            priority=result['priority'],
            access_count=access_count,
            category=result['category'],
            age_days=age_days
        )

        # Calculate access frequency (accesses per day)
        access_frequency = access_count / max(age_days, 0.1)

        # Update access tracking and importance
        await pool.execute("""
            UPDATE unified_brain
            SET access_count = access_count + 1,
                last_accessed = NOW(),
                importance_score = $1,
                access_frequency = $2
            WHERE key = $3
        """, new_importance, access_frequency, key)

        # Build response
        response = dict(result)

        # Include related entries if requested
        if include_related and result.get('related_keys'):
            related_data = []
            for rel_key in result['related_keys']:
                rel = await pool.fetchrow("""
                    SELECT key, summary, category, priority, importance_score
                    FROM unified_brain
                    WHERE key = $1
                """, rel_key)
                if rel:
                    related_data.append(dict(rel))
            response['related_entries'] = related_data

        return response

    async def get_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """Get all context in a category"""
        await self._ensure_table()
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT key, value, category, priority, last_updated, source, metadata, access_count
            FROM unified_brain
            WHERE category = $1
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                last_updated DESC
            LIMIT $2
        """, category, limit)

        return [dict(row) for row in rows]

    async def get_all_critical(self) -> List[Dict]:
        """Get ALL critical context across all categories"""
        await self._ensure_table()
        pool = get_pool()

        rows = await pool.fetch("""
            SELECT key, value, category, priority, last_updated, source, metadata
            FROM unified_brain
            WHERE priority = 'critical'
            ORDER BY category, last_updated DESC
        """)

        return [dict(row) for row in rows]

    async def get_full_context(self) -> Dict[str, Any]:
        """
        Get COMPLETE system context for Claude Code session initialization
        This is THE function that runs at session start
        """
        await self._ensure_table()
        pool = get_pool()

        # Get overview stats
        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE priority = 'critical') as critical_count,
                COUNT(*) FILTER (WHERE category = 'system') as system_count,
                COUNT(*) FILTER (WHERE category = 'session') as session_count,
                MAX(last_updated) as last_update
            FROM unified_brain
        """)

        # Get critical context organized by category
        rows = await pool.fetch("""
            SELECT category, key, value, priority, last_updated, source
            FROM unified_brain
            WHERE priority IN ('critical', 'high')
            ORDER BY
                CASE priority WHEN 'critical' THEN 1 ELSE 2 END,
                category,
                last_updated DESC
        """)

        context = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'stats': dict(stats) if stats else {},
            'critical_context': {},
            'by_category': {}
        }

        for row in rows:
            cat = row['category']
            if cat not in context['by_category']:
                context['by_category'][cat] = []

            context['by_category'][cat].append({
                'key': row['key'],
                'value': row['value'],
                'priority': row['priority'],
                'last_updated': row['last_updated'].isoformat() if row['last_updated'] else None,
                'source': row['source']
            })

            # Add critical items to top-level
            if row['priority'] == 'critical':
                context['critical_context'][row['key']] = row['value']

        return context

    async def search(self, query: str, limit: int = 20, use_semantic: bool = True) -> List[Dict]:
        """
        Enhanced search with multiple strategies:
        1. Semantic vector search (if embeddings available)
        2. Tag-based search
        3. PostgreSQL full-text search

        Args:
            query: Search query
            limit: Maximum results to return
            use_semantic: Whether to use semantic vector search
        """
        await self._ensure_table()
        pool = get_pool()
        all_results = []

        # Strategy 1: Semantic vector search
        if use_semantic and OPENAI_AVAILABLE:
            try:
                query_embedding = self._generate_embedding(query)
                if query_embedding:
                    semantic_results = await pool.fetch("""
                        SELECT *,
                               1 - (embedding <=> $1::vector) as similarity_score,
                               EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                        FROM unified_brain
                        WHERE embedding IS NOT NULL
                          AND (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY embedding <=> $1::vector
                        LIMIT $2
                    """, str(query_embedding), limit)

                    for row in semantic_results:
                        result = dict(row)
                        result['search_method'] = 'semantic'
                        result['relevance_score'] = (
                            result['similarity_score'] * 0.7 +  # Semantic similarity
                            result['importance_score'] * 0.3    # Importance boost
                        )
                        all_results.append(result)
                    logger.info(f"✅ Semantic search: found {len(semantic_results)} results")
            except Exception as e:
                logger.warning(f"⚠️ Semantic search failed: {e}")

        # Strategy 2: Tag-based search
        try:
            query_tags = query.lower().split()
            tag_results = await pool.fetch("""
                SELECT *,
                       EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                FROM unified_brain
                WHERE tags && $1
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY importance_score DESC
                LIMIT $2
            """, query_tags, limit)

            for row in tag_results:
                # Avoid duplicates from semantic search
                if not any(r['key'] == row['key'] for r in all_results):
                    result = dict(row)
                    result['search_method'] = 'tag'
                    # Count tag matches
                    tag_matches = len(set(query_tags) & set(result.get('tags', []) or []))
                    result['relevance_score'] = (
                        (tag_matches / len(query_tags)) * 0.6 +
                        result['importance_score'] * 0.4
                    )
                    all_results.append(result)
            logger.info(f"✅ Tag search: found {len(tag_results)} results")
        except Exception as e:
            logger.warning(f"⚠️ Tag search failed: {e}")

        # Strategy 3: Full-text search (fallback)
        try:
            search_pattern = f'%{query}%'
            text_results = await pool.fetch("""
                SELECT *,
                       EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                FROM unified_brain
                WHERE (expires_at IS NULL OR expires_at > NOW())
                  AND (
                    key ILIKE $1
                    OR summary ILIKE $1
                    OR (CASE
                          WHEN jsonb_typeof(value) = 'string' THEN value #>> '{}'
                          ELSE value::text
                        END) ILIKE $1
                    OR category ILIKE $1
                  )
                ORDER BY importance_score DESC, last_updated DESC
                LIMIT $2
            """, search_pattern, limit)

            for row in text_results:
                # Avoid duplicates
                if not any(r['key'] == row['key'] for r in all_results):
                    result = dict(row)
                    result['search_method'] = 'fulltext'
                    result['relevance_score'] = result['importance_score'] * 0.8
                    all_results.append(result)
            logger.info(f"✅ Full-text search: found {len(text_results)} results")
        except Exception as e:
            logger.warning(f"⚠️ Full-text search failed: {e}")

        # Try embedded memory as backup
        if not all_results and self.embedded_memory:
            try:
                results = self.embedded_memory.search_memories(
                    query=query,
                    limit=limit,
                    min_importance=0.0
                )
                if results:
                    for mem in results:
                        all_results.append({
                            'key': mem.get('memory_id', 'unknown'),
                            'value': mem.get('content', ''),
                            'category': mem.get('memory_type', 'general'),
                            'priority': 'high' if mem.get('importance_score', 0) > 0.8 else 'medium',
                            'last_updated': mem.get('created_at'),
                            'source': 'embedded_memory',
                            'metadata': mem.get('metadata', {}),
                            'search_method': 'embedded_rag',
                            'relevance_score': mem.get('combined_score', 0)
                        })
                    logger.info(f"✅ Embedded RAG fallback: found {len(results)} results")
            except Exception as e:
                logger.warning(f"⚠️ Embedded memory search failed: {e}")

        # Sort by relevance score and limit
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        final_results = all_results[:limit]

        logger.info(f"✅ Combined search: returning {len(final_results)} total results")
        return final_results

    async def consolidate_from_legacy_tables(self):
        """
        ONE-TIME MIGRATION: Pull critical data from the 98 legacy tables
        into unified_brain
        """
        await self._ensure_table()
        pool = get_pool()

        logger.info("🧠 Consolidating 98 legacy memory tables into unified brain...")

        # 1. Consolidate ai_context_memory
        rows = await pool.fetch("""
            SELECT context_key, context_value
            FROM ai_context_memory
            WHERE context_key IS NOT NULL
        """)

        for row in rows:
            await self.store(
                key=row['context_key'],
                value=row['context_value'],
                category='system',
                priority='high',
                source='legacy_ai_context_memory'
            )

        # 2. Consolidate brainops_knowledge
        rows = await pool.fetch("""
            SELECT title, content, category, importance
            FROM brainops_knowledge
            WHERE title IS NOT NULL
            LIMIT 100
        """)

        for row in rows:
            priority = 'critical' if row.get('importance', 0) > 0.8 else 'high'
            await self.store(
                key=f"knowledge_{row['title']}",
                value={'content': row['content'], 'category': row.get('category')},
                category='knowledge',
                priority=priority,
                source='legacy_brainops_knowledge'
            )

        # 3. Consolidate production_memory (sample - table is huge)
        rows = await pool.fetch("""
            SELECT key, value, importance
            FROM production_memory
            WHERE importance > 0.7
            ORDER BY created_at DESC
            LIMIT 500
        """)

        for row in rows:
            priority = 'critical' if row.get('importance', 0) > 0.9 else 'high'
            await self.store(
                key=row['key'],
                value=row['value'],
                category='production',
                priority=priority,
                source='legacy_production_memory'
            )

        logger.info("✅ Consolidation complete!")

    async def record_session_summary(self, session_id: str, summary: Dict[str, Any]):
        """Record a Claude Code session summary"""
        await self.store(
            key=f"session_{session_id}",
            value=summary,
            category='session',
            priority='high' if summary.get('tasks_completed', 0) > 0 else 'medium',
            source='claude_code',
            metadata={
                'session_id': session_id,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )

    async def update_system_state(self, component: str, state: Dict[str, Any]):
        """Update current system state"""
        await self.store(
            key=f"system_state_{component}",
            value=state,
            category='system',
            priority='critical',
            source='automated',
            metadata={'component': component}
        )

    async def record_deployment(self, service: str, version: str, status: str, metadata: Optional[Dict] = None):
        """Record a deployment"""
        await self.store(
            key=f"deployment_{service}_latest",
            value={
                'service': service,
                'version': version,
                'status': status,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            },
            category='deployment',
            priority='high',
            source='automated'
        )

    async def cleanup_expired(self) -> int:
        """Remove expired entries and return count of deleted items"""
        await self._ensure_pool()
        pool = get_pool()

        try:
            deleted = await pool.fetch("""
                DELETE FROM unified_brain
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                RETURNING key
            """)
            count = len(deleted)
            if count > 0:
                logger.info(f"✅ Cleaned up {count} expired entries")
            return count
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup expired entries: {e}")
            return 0

    async def get_related_entries(self, key: str, max_depth: int = 2) -> List[Dict]:
        """
        Get related entries recursively up to max_depth

        Args:
            key: Starting key
            max_depth: How many levels of relationships to traverse
        """
        await self._ensure_pool()
        pool = get_pool()
        visited = set()
        results = []

        async def _get_related_recursive(current_key: str, depth: int):
            if depth > max_depth or current_key in visited:
                return
            visited.add(current_key)

            # Get direct relationships from references table
            rows = await pool.fetch("""
                SELECT to_key, reference_type, strength
                FROM brain_references
                WHERE from_key = $1
                ORDER BY strength DESC
            """, current_key)

            for row in rows:
                to_key = row['to_key']
                if to_key not in visited:
                    # Get the actual entry
                    entry = await pool.fetchrow("""
                        SELECT key, summary, category, priority, importance_score
                        FROM unified_brain
                        WHERE key = $1
                    """, to_key)
                    if entry:
                        entry_dict = dict(entry)
                        entry_dict['relationship'] = {
                            'type': row['reference_type'],
                            'strength': row['strength'],
                            'depth': depth
                        }
                        results.append(entry_dict)
                        # Recurse
                        await _get_related_recursive(to_key, depth + 1)

        await _get_related_recursive(key, 1)
        return results

    async def find_similar(self, key: str, limit: int = 10) -> List[Dict]:
        """Find entries similar to the given key using vector similarity"""
        await self._ensure_pool()
        pool = get_pool()

        # Get the embedding of the source entry
        result = await pool.fetchrow("""
            SELECT embedding FROM unified_brain WHERE key = $1
        """, key)

        if not result or not result['embedding']:
            logger.warning(f"⚠️ No embedding found for key: {key}")
            return []

        embedding = result['embedding']

        # Find similar entries
        rows = await pool.fetch("""
            SELECT *,
                   1 - (embedding <=> $1::vector) as similarity_score
            FROM unified_brain
            WHERE key != $2
              AND embedding IS NOT NULL
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY embedding <=> $1::vector
            LIMIT $3
        """, str(embedding), key, limit)

        results = []
        for row in rows:
            result = dict(row)
            result['search_method'] = 'vector_similarity'
            results.append(result)

        logger.info(f"✅ Found {len(results)} similar entries to '{key}'")
        return results

    async def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the brain"""
        await self._ensure_pool()
        pool = get_pool()

        stats = {}

        # Overall counts
        overview = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE priority = 'critical') as critical_count,
                COUNT(*) FILTER (WHERE priority = 'high') as high_count,
                COUNT(*) FILTER (WHERE priority = 'medium') as medium_count,
                COUNT(*) FILTER (WHERE priority = 'low') as low_count,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL) as with_expiry,
                COUNT(*) FILTER (WHERE expires_at IS NOT NULL AND expires_at < NOW()) as expired,
                AVG(access_count) as avg_access_count,
                AVG(importance_score) as avg_importance,
                MAX(last_updated) as last_update
            FROM unified_brain
        """)
        stats['overview'] = dict(overview) if overview else {}

        # By category
        by_category = await pool.fetch("""
            SELECT category,
                   COUNT(*) as count,
                   AVG(importance_score) as avg_importance,
                   AVG(access_count) as avg_access_count
            FROM unified_brain
            GROUP BY category
            ORDER BY count DESC
        """)
        stats['by_category'] = [dict(row) for row in by_category]

        # Top accessed
        top_accessed = await pool.fetch("""
            SELECT key, category, access_count, importance_score, last_accessed
            FROM unified_brain
            ORDER BY access_count DESC
            LIMIT 10
        """)
        stats['top_accessed'] = [dict(row) for row in top_accessed]

        # Most important
        most_important = await pool.fetch("""
            SELECT key, category, importance_score, access_count, priority
            FROM unified_brain
            ORDER BY importance_score DESC
            LIMIT 10
        """)
        stats['most_important'] = [dict(row) for row in most_important]

        # References
        references = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_references,
                COUNT(DISTINCT from_key) as entries_with_refs,
                AVG(strength) as avg_strength
            FROM brain_references
        """)
        stats['references'] = dict(references) if references else {}

        # By reference type
        ref_types = await pool.fetch("""
            SELECT reference_type, COUNT(*) as count
            FROM brain_references
            GROUP BY reference_type
            ORDER BY count DESC
        """)
        stats['reference_types'] = [dict(row) for row in ref_types]

        return stats

    async def close(self):
        """Close database connection - now handled by pool manager"""
        # The pool is managed globally, so we don't close it here
        pass


# Global instance with lazy initialization (doesn't connect until first use)
brain = UnifiedBrain(lazy_init=True)


def get_brain() -> UnifiedBrain:
    """Get the global brain instance (lazy-initialized)"""
    return brain


async def initialize_brain_with_current_state():
    """Initialize brain with current production state using REAL data"""
    logger.info("🧠 Initializing Unified Brain with current state...")

    # Fetch REAL system scale from database
    try:
        await brain._ensure_pool()
        pool = get_pool()

        # Get counts
        customers = await pool.fetchval("SELECT COUNT(*) as count FROM customers")
        jobs = await pool.fetchval("SELECT COUNT(*) as count FROM jobs")
        invoices = await pool.fetchval("SELECT COUNT(*) as count FROM invoices")
        tenants = await pool.fetchval("SELECT COUNT(*) as count FROM tenants")
        agents = await pool.fetchval("SELECT COUNT(*) as count FROM ai_agents")
        tables = await pool.fetchval("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public'")

        # Store REAL scale
        await brain.store(
            'system_scale',
            {
                'customers': customers,
                'jobs': jobs,
                'invoices': invoices,
                'estimates': 0,  # Estimates table might not exist or be separate
                'tenants': tenants,
                'tables': tables,
                'ai_agents': agents,
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            category='system',
            priority='critical',
            source='initialization_realtime'
        )
        logger.info(f"✅ Stored REAL system scale: {customers} customers, {jobs} jobs")

    except Exception as e:
        logger.warning(f"⚠️ Failed to fetch real stats, using fallback: {e}")
        # Fallback to safe defaults if DB fails
        await brain.store(
            'system_scale',
            {
                'customers': 0,
                'jobs': 0,
                'invoices': 0,
                'status': 'db_connection_failed'
            },
            category='system',
            priority='high',
            source='initialization_fallback'
        )

    # Production services
    await brain.store(
        'production_services',
        {
            'myroofgenius': {'url': 'https://myroofgenius.com', 'status': 'healthy'},
            'weathercraft': {'url': 'https://weathercraft-erp.vercel.app', 'status': 'healthy'},
            'backend': {'url': 'https://brainops-backend-prod.onrender.com', 'status': 'healthy', 'version': '163.0.8'},
            'ai_agents': {'url': 'https://brainops-ai-agents.onrender.com', 'status': 'healthy', 'version': '6.0.0'},
            'database': {'host': 'aws-0-us-east-2.pooler.supabase.com', 'status': 'connected'}
        },
        category='system',
        priority='critical',
        source='initialization'
    )

    # Architecture principles
    await brain.store(
        'architecture_principles',
        {
            'build_dont_buy': 'Use free tiers and existing infrastructure',
            'cost_conscious': 'Stay at $25/month until revenue',
            'permanent_memory': 'Critical - never lose context between sessions',
            'autonomous_ai': 'AI should self-improve and self-heal'
        },
        category='architecture',
        priority='critical',
        source='initialization'
    )

    # Known issues
    await brain.store(
        'known_issues',
        {
            'memory_fragmentation': '98 memory tables causing context loss',
            'monitoring': 'No dashboards - only health checks',
            'test_coverage': 'Minimal automated testing',
            'github_actions': 'Recent runs failing'
        },
        category='issue',
        priority='high',
        source='initialization'
    )

    # Active capabilities
    await brain.store(
        'active_capabilities',
        {
            'aurea_orchestrator': True,
            'self_healing': True,
            'memory_manager': True,
            'training_pipeline': True,
            'learning_system': True,
            'agent_scheduler': True,
            'ai_core': True,
            'devops_optimization': True,
            'code_quality': True,
            'customer_success': True
        },
        category='system',
        priority='critical',
        source='initialization'
    )

    logger.info("✅ Brain initialized!")


if __name__ == "__main__":
    async def main():
        # Initialize with current state
        await initialize_brain_with_current_state()

        # Optional: Consolidate legacy tables
        # await brain.consolidate_from_legacy_tables()

        # Test retrieval
        context = await brain.get_full_context()
        print(json.dumps(context, indent=2, default=str))

        await brain.close()

    asyncio.run(main())
