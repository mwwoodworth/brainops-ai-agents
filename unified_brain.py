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
"""

import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor
import re

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": "postgres",
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

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
    Uses lazy initialization to avoid connection pool exhaustion
    """

    def __init__(self, lazy_init: bool = True):
        self.conn = None
        self.cursor = None
        self.embedded_memory = None
        self._initialized = False
        self._initializing = False  # Reentry guard
        self._table_checked = False  # Track table existence separately
        self._lazy_init = lazy_init
        # Only initialize immediately if not lazy
        if not lazy_init:
            self._do_init()

    def _do_init(self):
        """Perform actual initialization (called lazily on first use)"""
        # Reentry guard to prevent infinite recursion
        if self._initialized or self._initializing:
            return

        self._initializing = True
        try:
            # Note: _ensure_table and _init_embedded_memory will be called
            # on first actual use, not here, to avoid circular calls
            self._initialized = True
            print("✅ UnifiedBrain lazy init complete (will setup on first use)")
        except Exception as e:
            print(f"⚠️ UnifiedBrain init deferred: {e}")
        finally:
            self._initializing = False

    def _get_connection(self):
        """Get database connection with retry logic"""
        # Mark as initialized to prevent circular calls
        if not self._initialized:
            self._initialized = True

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.conn or self.conn.closed:
                    self.conn = psycopg2.connect(
                        **DB_CONFIG,
                        cursor_factory=RealDictCursor,
                        connect_timeout=10
                    )
                    self.cursor = self.conn.cursor()

                # Ensure table exists on first real connection
                if not self._table_checked:
                    self._table_checked = True
                    self._create_table_if_needed()

                return self.conn, self.cursor
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ DB connection attempt {attempt + 1} failed, retrying...")
                    import time
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    # Close any stale connections
                    try:
                        if self.conn:
                            self.conn.close()
                    except:
                        pass
                    self.conn = None
                    self.cursor = None
                else:
                    raise

    def _create_table_if_needed(self):
        """Create table without recursive connection calls"""
        try:
            # First ensure vector extension exists
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create main table with enhanced columns
            self.cursor.execute("""
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

                CREATE INDEX IF NOT EXISTS idx_unified_brain_key ON unified_brain(key);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_category ON unified_brain(category);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_priority ON unified_brain(priority);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_updated ON unified_brain(last_updated DESC);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_importance ON unified_brain(importance_score DESC);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_expires ON unified_brain(expires_at) WHERE expires_at IS NOT NULL;

                -- Vector similarity index (IVFFlat for fast approximate search)
                CREATE INDEX IF NOT EXISTS idx_unified_brain_embedding ON unified_brain
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);

                -- GIN index for array searching
                CREATE INDEX IF NOT EXISTS idx_unified_brain_tags ON unified_brain USING GIN(tags);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_related ON unified_brain USING GIN(related_keys);
            """)

            # Create cross-reference tracking table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS brain_references (
                    id SERIAL PRIMARY KEY,
                    from_key TEXT NOT NULL,
                    to_key TEXT NOT NULL,
                    reference_type TEXT NOT NULL,  -- 'related', 'superseded', 'depends_on', 'derived_from'
                    strength FLOAT DEFAULT 1.0,  -- How strong is this relationship
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(from_key, to_key, reference_type)
                );

                CREATE INDEX IF NOT EXISTS idx_brain_ref_from ON brain_references(from_key);
                CREATE INDEX IF NOT EXISTS idx_brain_ref_to ON brain_references(to_key);
                CREATE INDEX IF NOT EXISTS idx_brain_ref_type ON brain_references(reference_type);
            """)

            self.conn.commit()
            print("✅ UnifiedBrain enhanced table ensured with vector search")
        except Exception as e:
            print(f"⚠️ Table creation (may already exist): {e}")
            try:
                self.conn.rollback()
            except:
                pass

    def _ensure_table(self):
        """Legacy method - now handled in _get_connection via _create_table_if_needed"""
        # Just calls _get_connection which handles table creation
        self._get_connection()

    def _init_embedded_memory(self):
        """Initialize embedded memory system for RAG search - LAZY"""
        # This is now called lazily on first vector search, not at init
        if self.embedded_memory is not None:
            return self.embedded_memory
        try:
            from embedded_memory_system import get_embedded_memory
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self.embedded_memory = loop.run_until_complete(get_embedded_memory())
            print("✅ Embedded memory integrated with UnifiedBrain")
            return self.embedded_memory
        except Exception as e:
            print(f"⚠️ Embedded memory not available: {e}")
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
            print(f"⚠️ Failed to generate embedding: {e}")
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

    def _find_related_entries(self, key: str, embedding: Optional[List[float]],
                             limit: int = 5) -> List[str]:
        """Find related entries using vector similarity"""
        if not embedding:
            return []

        conn, cursor = self._get_connection()
        try:
            # Find similar entries using cosine similarity
            cursor.execute("""
                SELECT key, 1 - (embedding <=> %s::vector) as similarity
                FROM unified_brain
                WHERE key != %s AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding, key, embedding, limit))

            related = [row['key'] for row in cursor.fetchall() if row.get('similarity', 0) > 0.7]
            return related
        except Exception as e:
            print(f"⚠️ Failed to find related entries: {e}")
            return []

    def store(self, key: str, value: Any, category: str = "general",
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
        conn, cursor = self._get_connection()

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
        cursor.execute("""
            INSERT INTO unified_brain (
                key, value, category, priority, source, metadata,
                embedding, summary, importance_score, expires_at, tags,
                last_updated, last_accessed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
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
        """, (
            key, json.dumps(value), category, priority, source,
            json.dumps(metadata or {}), embedding, summary,
            importance_score, expires_at, tags
        ))

        result = cursor.fetchone()
        conn.commit()
        entry_id = str(result['id'])

        # Find and store related entries using vector similarity
        if embedding:
            try:
                related_keys = self._find_related_entries(key, embedding)
                if related_keys:
                    cursor.execute("""
                        UPDATE unified_brain
                        SET related_keys = %s
                        WHERE key = %s
                    """, (related_keys, key))
                    conn.commit()

                    # Also create bidirectional references
                    for related_key in related_keys:
                        self._add_reference(key, related_key, 'related', 0.8)
            except Exception as e:
                print(f"⚠️ Failed to update related keys: {e}")

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
                print(f"✅ Dual-write: Stored '{key}' with semantic search enabled")
            except Exception as e:
                print(f"⚠️ Embedded memory store failed: {e} (data still in Postgres)")

        print(f"✅ Stored '{key}' with embedding, summary, and {len(related_keys) if embedding else 0} related entries")
        return entry_id

    def _add_reference(self, from_key: str, to_key: str,
                      reference_type: str = 'related', strength: float = 1.0):
        """Add a cross-reference between two entries"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                INSERT INTO brain_references (from_key, to_key, reference_type, strength)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (from_key, to_key, reference_type) DO UPDATE
                SET strength = EXCLUDED.strength
            """, (from_key, to_key, reference_type, strength))
            conn.commit()
        except Exception as e:
            print(f"⚠️ Failed to add reference: {e}")
            try:
                conn.rollback()
            except:
                pass

    def store_learning(self, agent_id: str, task_id: str, mistake: str,
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
        if not self.embedded_memory:
            self._init_embedded_memory()
            
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
                print(f"⚠️ Failed to store learning: {e}")
                return False
        return False

    def get(self, key: str, include_related: bool = False) -> Optional[Dict]:
        """
        Retrieve a piece of context with enhanced tracking

        Args:
            key: The key to retrieve
            include_related: Whether to include related entries
        """
        conn, cursor = self._get_connection()

        # First check if entry has expired
        cursor.execute("""
            SELECT expires_at FROM unified_brain
            WHERE key = %s AND expires_at IS NOT NULL
        """, (key,))
        expiry_check = cursor.fetchone()
        if expiry_check and expiry_check['expires_at'] < datetime.now(timezone.utc):
            # Entry has expired, delete it
            cursor.execute("DELETE FROM unified_brain WHERE key = %s", (key,))
            conn.commit()
            return None

        # Get the entry and update access tracking
        cursor.execute("""
            SELECT *,
                   EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days,
                   EXTRACT(EPOCH FROM (NOW() - last_accessed))/86400.0 as days_since_access
            FROM unified_brain
            WHERE key = %s
        """, (key,))

        result = cursor.fetchone()
        if not result:
            return None

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
        cursor.execute("""
            UPDATE unified_brain
            SET access_count = access_count + 1,
                last_accessed = NOW(),
                importance_score = %s,
                access_frequency = %s
            WHERE key = %s
        """, (new_importance, access_frequency, key))
        conn.commit()

        # Build response
        response = dict(result)

        # Include related entries if requested
        if include_related and result.get('related_keys'):
            related_data = []
            for rel_key in result['related_keys']:
                cursor.execute("""
                    SELECT key, summary, category, priority, importance_score
                    FROM unified_brain
                    WHERE key = %s
                """, (rel_key,))
                rel = cursor.fetchone()
                if rel:
                    related_data.append(dict(rel))
            response['related_entries'] = related_data

        return response

    def get_by_category(self, category: str, limit: int = 100) -> List[Dict]:
        """Get all context in a category"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            SELECT key, value, category, priority, last_updated, source, metadata, access_count
            FROM unified_brain
            WHERE category = %s
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                last_updated DESC
            LIMIT %s
        """, (category, limit))

        return [dict(row) for row in cursor.fetchall()]

    def get_all_critical(self) -> List[Dict]:
        """Get ALL critical context across all categories"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            SELECT key, value, category, priority, last_updated, source, metadata
            FROM unified_brain
            WHERE priority = 'critical'
            ORDER BY category, last_updated DESC
        """)

        return [dict(row) for row in cursor.fetchall()]

    def get_full_context(self) -> Dict[str, Any]:
        """
        Get COMPLETE system context for Claude Code session initialization
        This is THE function that runs at session start
        """
        conn, cursor = self._get_connection()

        # Get overview stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE priority = 'critical') as critical_count,
                COUNT(*) FILTER (WHERE category = 'system') as system_count,
                COUNT(*) FILTER (WHERE category = 'session') as session_count,
                MAX(last_updated) as last_update
            FROM unified_brain
        """)
        stats = cursor.fetchone()

        # Get critical context organized by category
        cursor.execute("""
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

        for row in cursor.fetchall():
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

    def search(self, query: str, limit: int = 20, use_semantic: bool = True) -> List[Dict]:
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
        conn, cursor = self._get_connection()
        all_results = []

        # Strategy 1: Semantic vector search
        if use_semantic and OPENAI_AVAILABLE:
            try:
                query_embedding = self._generate_embedding(query)
                if query_embedding:
                    cursor.execute("""
                        SELECT *,
                               1 - (embedding <=> %s::vector) as similarity_score,
                               EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                        FROM unified_brain
                        WHERE embedding IS NOT NULL
                          AND (expires_at IS NULL OR expires_at > NOW())
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (query_embedding, query_embedding, limit))

                    semantic_results = cursor.fetchall()
                    for row in semantic_results:
                        result = dict(row)
                        result['search_method'] = 'semantic'
                        result['relevance_score'] = (
                            result['similarity_score'] * 0.7 +  # Semantic similarity
                            result['importance_score'] * 0.3    # Importance boost
                        )
                        all_results.append(result)
                    print(f"✅ Semantic search: found {len(semantic_results)} results")
            except Exception as e:
                print(f"⚠️ Semantic search failed: {e}")

        # Strategy 2: Tag-based search
        try:
            query_tags = query.lower().split()
            cursor.execute("""
                SELECT *,
                       EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                FROM unified_brain
                WHERE tags && %s
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY importance_score DESC
                LIMIT %s
            """, (query_tags, limit))

            tag_results = cursor.fetchall()
            for row in tag_results:
                # Avoid duplicates from semantic search
                if not any(r['key'] == row['key'] for r in all_results):
                    result = dict(row)
                    result['search_method'] = 'tag'
                    # Count tag matches
                    tag_matches = len(set(query_tags) & set(result.get('tags', [])))
                    result['relevance_score'] = (
                        (tag_matches / len(query_tags)) * 0.6 +
                        result['importance_score'] * 0.4
                    )
                    all_results.append(result)
            print(f"✅ Tag search: found {len(tag_results)} results")
        except Exception as e:
            print(f"⚠️ Tag search failed: {e}")

        # Strategy 3: Full-text search (fallback)
        try:
            cursor.execute("""
                SELECT *,
                       EXTRACT(EPOCH FROM (NOW() - created_at))/86400.0 as age_days
                FROM unified_brain
                WHERE (expires_at IS NULL OR expires_at > NOW())
                  AND (
                    key ILIKE %s
                    OR summary ILIKE %s
                    OR (CASE
                          WHEN jsonb_typeof(value) = 'string' THEN value #>> '{}'
                          ELSE value::text
                        END) ILIKE %s
                    OR category ILIKE %s
                  )
                ORDER BY importance_score DESC, last_updated DESC
                LIMIT %s
            """, (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%', limit))

            text_results = cursor.fetchall()
            for row in text_results:
                # Avoid duplicates
                if not any(r['key'] == row['key'] for r in all_results):
                    result = dict(row)
                    result['search_method'] = 'fulltext'
                    result['relevance_score'] = result['importance_score'] * 0.8
                    all_results.append(result)
            print(f"✅ Full-text search: found {len(text_results)} results")
        except Exception as e:
            print(f"⚠️ Full-text search failed: {e}")

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
                    print(f"✅ Embedded RAG fallback: found {len(results)} results")
            except Exception as e:
                print(f"⚠️ Embedded memory search failed: {e}")

        # Sort by relevance score and limit
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        final_results = all_results[:limit]

        print(f"✅ Combined search: returning {len(final_results)} total results")
        return final_results

    def consolidate_from_legacy_tables(self):
        """
        ONE-TIME MIGRATION: Pull critical data from the 98 legacy tables
        into unified_brain
        """
        conn, cursor = self._get_connection()

        print("🧠 Consolidating 98 legacy memory tables into unified brain...")

        # 1. Consolidate ai_context_memory
        cursor.execute("""
            SELECT context_key, context_value
            FROM ai_context_memory
            WHERE context_key IS NOT NULL
        """)

        for row in cursor.fetchall():
            self.store(
                key=row['context_key'],
                value=row['context_value'],
                category='system',
                priority='high',
                source='legacy_ai_context_memory'
            )

        # 2. Consolidate brainops_knowledge
        cursor.execute("""
            SELECT title, content, category, importance
            FROM brainops_knowledge
            WHERE title IS NOT NULL
            LIMIT 100
        """)

        for row in cursor.fetchall():
            priority = 'critical' if row.get('importance', 0) > 0.8 else 'high'
            self.store(
                key=f"knowledge_{row['title']}",
                value={'content': row['content'], 'category': row.get('category')},
                category='knowledge',
                priority=priority,
                source='legacy_brainops_knowledge'
            )

        # 3. Consolidate production_memory (sample - table is huge)
        cursor.execute("""
            SELECT key, value, importance
            FROM production_memory
            WHERE importance > 0.7
            ORDER BY created_at DESC
            LIMIT 500
        """)

        for row in cursor.fetchall():
            priority = 'critical' if row.get('importance', 0) > 0.9 else 'high'
            self.store(
                key=row['key'],
                value=row['value'],
                category='production',
                priority=priority,
                source='legacy_production_memory'
            )

        print("✅ Consolidation complete!")

    def record_session_summary(self, session_id: str, summary: Dict[str, Any]):
        """Record a Claude Code session summary"""
        self.store(
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

    def update_system_state(self, component: str, state: Dict[str, Any]):
        """Update current system state"""
        self.store(
            key=f"system_state_{component}",
            value=state,
            category='system',
            priority='critical',
            source='automated',
            metadata={'component': component}
        )

    def record_deployment(self, service: str, version: str, status: str, metadata: Optional[Dict] = None):
        """Record a deployment"""
        self.store(
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

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of deleted items"""
        conn, cursor = self._get_connection()
        try:
            cursor.execute("""
                DELETE FROM unified_brain
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                RETURNING key
            """)
            deleted = cursor.fetchall()
            conn.commit()
            count = len(deleted)
            if count > 0:
                print(f"✅ Cleaned up {count} expired entries")
            return count
        except Exception as e:
            print(f"⚠️ Failed to cleanup expired entries: {e}")
            try:
                conn.rollback()
            except:
                pass
            return 0

    def get_related_entries(self, key: str, max_depth: int = 2) -> List[Dict]:
        """
        Get related entries recursively up to max_depth

        Args:
            key: Starting key
            max_depth: How many levels of relationships to traverse
        """
        conn, cursor = self._get_connection()
        visited = set()
        results = []

        def _get_related_recursive(current_key: str, depth: int):
            if depth > max_depth or current_key in visited:
                return
            visited.add(current_key)

            # Get direct relationships from references table
            cursor.execute("""
                SELECT to_key, reference_type, strength
                FROM brain_references
                WHERE from_key = %s
                ORDER BY strength DESC
            """, (current_key,))

            for row in cursor.fetchall():
                to_key = row['to_key']
                if to_key not in visited:
                    # Get the actual entry
                    cursor.execute("""
                        SELECT key, summary, category, priority, importance_score
                        FROM unified_brain
                        WHERE key = %s
                    """, (to_key,))
                    entry = cursor.fetchone()
                    if entry:
                        entry_dict = dict(entry)
                        entry_dict['relationship'] = {
                            'type': row['reference_type'],
                            'strength': row['strength'],
                            'depth': depth
                        }
                        results.append(entry_dict)
                        # Recurse
                        _get_related_recursive(to_key, depth + 1)

        _get_related_recursive(key, 1)
        return results

    def find_similar(self, key: str, limit: int = 10) -> List[Dict]:
        """Find entries similar to the given key using vector similarity"""
        conn, cursor = self._get_connection()

        # Get the embedding of the source entry
        cursor.execute("""
            SELECT embedding FROM unified_brain WHERE key = %s
        """, (key,))
        result = cursor.fetchone()

        if not result or not result['embedding']:
            print(f"⚠️ No embedding found for key: {key}")
            return []

        embedding = result['embedding']

        # Find similar entries
        cursor.execute("""
            SELECT *,
                   1 - (embedding <=> %s::vector) as similarity_score
            FROM unified_brain
            WHERE key != %s
              AND embedding IS NOT NULL
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (embedding, key, embedding, limit))

        results = []
        for row in cursor.fetchall():
            result = dict(row)
            result['search_method'] = 'vector_similarity'
            results.append(result)

        print(f"✅ Found {len(results)} similar entries to '{key}'")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the brain"""
        conn, cursor = self._get_connection()

        stats = {}

        # Overall counts
        cursor.execute("""
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
        stats['overview'] = dict(cursor.fetchone())

        # By category
        cursor.execute("""
            SELECT category,
                   COUNT(*) as count,
                   AVG(importance_score) as avg_importance,
                   AVG(access_count) as avg_access_count
            FROM unified_brain
            GROUP BY category
            ORDER BY count DESC
        """)
        stats['by_category'] = [dict(row) for row in cursor.fetchall()]

        # Top accessed
        cursor.execute("""
            SELECT key, category, access_count, importance_score, last_accessed
            FROM unified_brain
            ORDER BY access_count DESC
            LIMIT 10
        """)
        stats['top_accessed'] = [dict(row) for row in cursor.fetchall()]

        # Most important
        cursor.execute("""
            SELECT key, category, importance_score, access_count, priority
            FROM unified_brain
            ORDER BY importance_score DESC
            LIMIT 10
        """)
        stats['most_important'] = [dict(row) for row in cursor.fetchall()]

        # References
        cursor.execute("""
            SELECT
                COUNT(*) as total_references,
                COUNT(DISTINCT from_key) as entries_with_refs,
                AVG(strength) as avg_strength
            FROM brain_references
        """)
        stats['references'] = dict(cursor.fetchone())

        # By reference type
        cursor.execute("""
            SELECT reference_type, COUNT(*) as count
            FROM brain_references
            GROUP BY reference_type
            ORDER BY count DESC
        """)
        stats['reference_types'] = [dict(row) for row in cursor.fetchall()]

        return stats

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()


# Global instance with lazy initialization (doesn't connect until first use)
brain = UnifiedBrain(lazy_init=True)


def get_brain() -> UnifiedBrain:
    """Get the global brain instance (lazy-initialized)"""
    return brain


def initialize_brain_with_current_state():
    """Initialize brain with current production state using REAL data"""
    print("🧠 Initializing Unified Brain with current state...")

    # Fetch REAL system scale from database
    try:
        conn, cursor = brain._get_connection()
        
        # Get counts
        cursor.execute("SELECT COUNT(*) as count FROM customers")
        customers = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM jobs")
        jobs = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM invoices")
        invoices = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM tenants")
        tenants = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM ai_agents")
        agents = cursor.fetchone()['count']
        
        cursor.execute("SELECT COUNT(*) as count FROM information_schema.tables WHERE table_schema = 'public'")
        tables = cursor.fetchone()['count']
        
        # Store REAL scale
        brain.store(
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
        print(f"✅ Stored REAL system scale: {customers} customers, {jobs} jobs")
        
    except Exception as e:
        print(f"⚠️ Failed to fetch real stats, using fallback: {e}")
        # Fallback to safe defaults if DB fails
        brain.store(
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
    brain.store(
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
    brain.store(
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
    brain.store(
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
    brain.store(
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

    print("✅ Brain initialized!")


if __name__ == "__main__":
    # Initialize with current state
    initialize_brain_with_current_state()

    # Optional: Consolidate legacy tables
    # brain.consolidate_from_legacy_tables()

    # Test retrieval
    context = brain.get_full_context()
    print(json.dumps(context, indent=2, default=str))

    brain.close()
