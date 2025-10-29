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
    "password": os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
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
    """

    def __init__(self):
        self.conn = None
        self.cursor = None
        self._ensure_table()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
            self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def _ensure_table(self):
        """Create unified brain table if it doesn't exist"""
        conn, cursor = self._get_connection()

        cursor.execute("""
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

        conn.commit()

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
        return str(result['id'])

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
        """Search across all context"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            SELECT key, value, category, priority, last_updated, source, metadata
            FROM unified_brain
            WHERE key ILIKE %s
               OR value::text ILIKE %s
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

        return [dict(row) for row in cursor.fetchall()]

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


# Global instance
brain = UnifiedBrain()


def initialize_brain_with_current_state():
    """Initialize brain with current production state"""
    print("🧠 Initializing Unified Brain with current state...")

    # System scale
    brain.store(
        'system_scale',
        {
            'customers': 3724,
            'jobs': 12920,
            'invoices': 2037,
            'estimates': 51,
            'tenants': 34,
            'tables': 1322,
            'ai_agents': 59
        },
        category='system',
        priority='critical',
        source='initialization'
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
