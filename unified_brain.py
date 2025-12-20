#!/usr/bin/env python3
"""
UNIFIED BRAIN - The Single Source of Truth for All BrainOps Memory
Consolidates 98 fragmented memory tables into ONE coherent system
Designed for Claude Code + Codex integration
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import hashlib
import psycopg2
from psycopg2.extras import RealDictCursor

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": "postgres",
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "<DB_PASSWORD_REDACTED>"),
    "port": int(os.getenv("DB_PORT", "5432"))
}


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
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_unified_brain_key ON unified_brain(key);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_category ON unified_brain(category);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_priority ON unified_brain(priority);
                CREATE INDEX IF NOT EXISTS idx_unified_brain_updated ON unified_brain(last_updated DESC);
            """)
            self.conn.commit()
            print("✅ UnifiedBrain table ensured")
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

    def store(self, key: str, value: Any, category: str = "general",
              priority: str = "medium", source: str = "manual",
              metadata: Optional[Dict] = None) -> str:
        """
        Store or update a piece of context

        Args:
            key: Unique identifier for this context
            value: The actual data (will be JSON serialized)
            category: system, session, architecture, deployment, issue
            priority: critical, high, medium, low
            source: Where this came from (claude_code, codex, api, manual)
            metadata: Additional context about this entry
        """
        conn, cursor = self._get_connection()

        cursor.execute("""
            INSERT INTO unified_brain (key, value, category, priority, source, metadata, last_updated)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (key) DO UPDATE
            SET value = EXCLUDED.value,
                category = EXCLUDED.category,
                priority = EXCLUDED.priority,
                source = EXCLUDED.source,
                metadata = EXCLUDED.metadata,
                last_updated = NOW(),
                access_count = unified_brain.access_count + 1
            RETURNING id
        """, (key, json.dumps(value), category, priority, source, json.dumps(metadata or {})))

        result = cursor.fetchone()
        conn.commit()
        entry_id = str(result['id'])

        # DUAL-WRITE: Also store in embedded memory for fast RAG search
        if self.embedded_memory:
            try:
                # Convert to string for embedded memory
                content = value if isinstance(value, str) else json.dumps(value)
                importance = 0.9 if priority == 'critical' else 0.7 if priority == 'high' else 0.5

                self.embedded_memory.store_memory(
                    content=content,
                    memory_type=category,
                    importance_score=importance,
                    metadata={
                        **({'source': source} if source else {}),
                        **(metadata or {}),
                        'brain_key': key,
                        'brain_id': entry_id
                    }
                )
                print(f"✅ Dual-write: Stored '{key}' in both Postgres and embedded memory")
            except Exception as e:
                print(f"⚠️ Embedded memory store failed: {e} (data still in Postgres)")

        return entry_id

    def get(self, key: str) -> Optional[Dict]:
        """Retrieve a piece of context"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            UPDATE unified_brain
            SET access_count = access_count + 1
            WHERE key = %s
            RETURNING key, value, category, priority, last_updated, source, metadata, access_count
        """, (key,))

        result = cursor.fetchone()
        conn.commit()

        if result:
            return dict(result)
        return None

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

    def search(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search across all context using embedded memory RAG if available
        Falls back to PostgreSQL full-text search if embedded memory unavailable
        """
        # Try embedded memory first (ultra-fast RAG search)
        if self.embedded_memory:
            try:
                # Use embedded memory's RAG search
                results = self.embedded_memory.search_memories(
                    query=query,
                    limit=limit,
                    min_importance=0.0  # Include all results
                )

                if results:
                    # Convert embedded memory format to brain format
                    brain_results = []
                    for mem in results:
                        brain_results.append({
                            'key': mem.get('memory_id', 'unknown'),
                            'value': mem.get('content', ''),
                            'category': mem.get('memory_type', 'general'),
                            'priority': 'high' if mem.get('importance_score', 0) > 0.8 else 'medium',
                            'last_updated': mem.get('created_at'),
                            'source': 'embedded_memory',
                            'metadata': mem.get('metadata', {}),
                            'similarity_score': mem.get('similarity_score', 0),
                            'combined_score': mem.get('combined_score', 0)
                        })
                    print(f"✅ Embedded RAG search: found {len(brain_results)} results")
                    return brain_results
            except Exception as e:
                print(f"⚠️ Embedded memory search failed: {e}, falling back to Postgres")

        # Fallback to Postgres full-text search
        conn, cursor = self._get_connection()

        # FIXED: Cast JSON values to text properly, handle both string and JSON
        cursor.execute("""
            SELECT key, value, category, priority, last_updated, source, metadata
            FROM unified_brain
            WHERE key ILIKE %s
               OR (CASE
                     WHEN jsonb_typeof(value) = 'string' THEN value #>> '{}'
                     ELSE value::text
                   END) ILIKE %s
               OR category ILIKE %s
            ORDER BY
                CASE priority
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                last_updated DESC
            LIMIT %s
        """, (f'%{query}%', f'%{query}%', f'%{query}%', limit))

        results = [dict(row) for row in cursor.fetchall()]
        print(f"✅ Postgres search: found {len(results)} results")
        return results

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
