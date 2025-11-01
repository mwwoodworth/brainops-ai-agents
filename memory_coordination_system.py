#!/usr/bin/env python3
"""
MEMORY COORDINATION SYSTEM - Perfect E2E Context Management
Ensures seamless coordination across all memory systems, sessions, and agents
"""

import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
from uuid import UUID
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MEMORY LAYER DEFINITIONS
# ============================================================================

class MemoryLayer(Enum):
    """Different layers of memory storage"""
    EPHEMERAL = "ephemeral"  # In-memory cache (seconds to minutes)
    SESSION = "session"  # Session-scoped (hours)
    SHORT_TERM = "short_term"  # Days to weeks
    LONG_TERM = "long_term"  # Weeks to months
    PERMANENT = "permanent"  # Forever


class ContextScope(Enum):
    """Scope of context visibility"""
    GLOBAL = "global"  # Visible to all systems
    TENANT = "tenant"  # Tenant-specific
    USER = "user"  # User-specific
    SESSION = "session"  # Current session only
    AGENT = "agent"  # Specific AI agent only


class SyncPriority(Enum):
    """Priority for cross-system synchronization"""
    IMMEDIATE = "immediate"  # Sync within 1 second
    HIGH = "high"  # Sync within 5 seconds
    NORMAL = "normal"  # Sync within 30 seconds
    LOW = "low"  # Sync within 5 minutes
    EVENTUAL = "eventual"  # Sync eventually (background)


# ============================================================================
# CONTEXT STRUCTURES
# ============================================================================

@dataclass
class ContextEntry:
    """Universal context entry structure"""
    key: str
    value: Any
    layer: MemoryLayer
    scope: ContextScope
    priority: str  # critical, high, medium, low
    category: str
    source: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    access_count: int = 0
    sync_version: int = 1


@dataclass
class SyncEvent:
    """Event for cross-system synchronization"""
    event_id: str
    event_type: str  # create, update, delete, merge
    context_key: str
    source_system: str
    target_systems: List[str]
    priority: SyncPriority
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed: bool = False


# ============================================================================
# UNIFIED MEMORY COORDINATOR
# ============================================================================

class UnifiedMemoryCoordinator:
    """
    Master coordinator that ensures perfect context synchronization
    across all memory systems and agents
    """

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.cursor = None

        # Memory system references
        self.unified_brain = None
        self.embedded_memory = None
        self.vector_memory = None
        self.conversation_memory = None

        # In-memory caches for performance
        self.ephemeral_cache: Dict[str, ContextEntry] = {}
        self.session_cache: Dict[str, Dict[str, ContextEntry]] = {}

        # Sync tracking
        self.pending_syncs: List[SyncEvent] = []
        self.sync_locks: Set[str] = set()

        self._ensure_tables()
        self._init_memory_systems()

    def _get_connection(self):
        """Get database connection"""
        if not self.conn or self.conn.closed:
            self.conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
            self.cursor = self.conn.cursor()
        return self.conn, self.cursor

    def _ensure_tables(self):
        """Create coordination tables"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            -- Master context registry
            CREATE TABLE IF NOT EXISTS memory_context_registry (
                id SERIAL PRIMARY KEY,
                key TEXT NOT NULL,
                layer TEXT NOT NULL,
                scope TEXT NOT NULL,
                priority TEXT NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                tenant_id TEXT,
                user_id TEXT,
                session_id TEXT,
                agent_id TEXT,
                value JSONB NOT NULL,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                expires_at TIMESTAMPTZ,
                access_count INT DEFAULT 0,
                sync_version INT DEFAULT 1,
                dedupe_key TEXT GENERATED ALWAYS AS (
                    scope || ':' ||
                    COALESCE(tenant_id, '') || ':' ||
                    COALESCE(user_id, '') || ':' ||
                    COALESCE(session_id, '') || ':' ||
                    COALESCE(agent_id, '') || ':' ||
                    key
                ) STORED,
                CONSTRAINT memory_context_registry_dedupe_key UNIQUE (dedupe_key)
            );

            -- Session context tracking
            CREATE TABLE IF NOT EXISTS memory_session_context (
                id SERIAL PRIMARY KEY,
                session_id TEXT UNIQUE NOT NULL,
                tenant_id TEXT,
                user_id TEXT,
                context_snapshot JSONB NOT NULL,
                active_agents TEXT[] DEFAULT '{}',
                start_time TIMESTAMPTZ DEFAULT NOW(),
                last_activity TIMESTAMPTZ DEFAULT NOW(),
                status TEXT DEFAULT 'active',
                metadata JSONB DEFAULT '{}'::jsonb
            );

            -- Cross-system sync events
            CREATE TABLE IF NOT EXISTS memory_sync_events (
                id SERIAL PRIMARY KEY,
                event_id TEXT UNIQUE NOT NULL,
                event_type TEXT NOT NULL,
                context_key TEXT NOT NULL,
                source_system TEXT NOT NULL,
                target_systems TEXT[] NOT NULL,
                priority TEXT NOT NULL,
                payload JSONB NOT NULL,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                processed BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMPTZ,
                error TEXT
            );

            -- Context access log for analytics
            CREATE TABLE IF NOT EXISTS memory_context_access_log (
                id SERIAL PRIMARY KEY,
                context_key TEXT NOT NULL,
                accessed_by TEXT NOT NULL,
                access_type TEXT NOT NULL,
                session_id TEXT,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                latency_ms INT,
                hit_cache BOOLEAN DEFAULT FALSE
            );

            -- Sync conflict resolution
            CREATE TABLE IF NOT EXISTS memory_sync_conflicts (
                id SERIAL PRIMARY KEY,
                context_key TEXT NOT NULL,
                conflict_type TEXT NOT NULL,
                system_a TEXT NOT NULL,
                value_a JSONB,
                version_a INT,
                system_b TEXT NOT NULL,
                value_b JSONB,
                version_b INT,
                resolution TEXT,
                resolved_value JSONB,
                resolved_at TIMESTAMPTZ,
                resolved_by TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_context_key ON memory_context_registry(key);
            CREATE INDEX IF NOT EXISTS idx_context_layer ON memory_context_registry(layer);
            CREATE INDEX IF NOT EXISTS idx_context_scope ON memory_context_registry(scope, tenant_id, user_id);
            CREATE INDEX IF NOT EXISTS idx_context_session ON memory_context_registry(session_id);
            CREATE INDEX IF NOT EXISTS idx_context_expires ON memory_context_registry(expires_at) WHERE expires_at IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_sync_events_pending ON memory_sync_events(processed, priority, timestamp);
            CREATE INDEX IF NOT EXISTS idx_session_active ON memory_session_context(status, last_activity);
        """)

        # Legacy >5.0 migrations: remove obsolete unique constraint and enforce scoped uniqueness
        cursor.execute("""
            ALTER TABLE memory_context_registry
            DROP CONSTRAINT IF EXISTS memory_context_unique_key;
        """)

        cursor.execute("""
            DELETE FROM memory_context_registry mc
            USING memory_context_registry dup
            WHERE mc.ctid < dup.ctid
              AND mc.key = dup.key
              AND mc.scope = dup.scope
              AND COALESCE(mc.tenant_id, '') = COALESCE(dup.tenant_id, '')
              AND COALESCE(mc.user_id, '') = COALESCE(dup.user_id, '')
              AND COALESCE(mc.session_id, '') = COALESCE(dup.session_id, '')
              AND COALESCE(mc.agent_id, '') = COALESCE(dup.agent_id, '');
        """)

        cursor.execute("""
            ALTER TABLE memory_context_registry
            ADD COLUMN IF NOT EXISTS dedupe_key TEXT GENERATED ALWAYS AS (
                scope || ':' ||
                COALESCE(tenant_id, '') || ':' ||
                COALESCE(user_id, '') || ':' ||
                COALESCE(session_id, '') || ':' ||
                COALESCE(agent_id, '') || ':' ||
                key
            ) STORED;
        """)

        cursor.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints
                    WHERE table_name = 'memory_context_registry'
                      AND constraint_type = 'UNIQUE'
                      AND constraint_name = 'memory_context_registry_dedupe_key'
                ) THEN
                    ALTER TABLE memory_context_registry
                    ADD CONSTRAINT memory_context_registry_dedupe_key
                    UNIQUE (dedupe_key);
                END IF;
            END $$;
        """)

        conn.commit()
        logger.info("✅ Memory coordination tables ready")

    def _init_memory_systems(self):
        """Initialize connections to all memory systems"""
        try:
            from unified_brain import brain
            self.unified_brain = brain
            logger.info("✅ Connected to UnifiedBrain")
        except Exception as e:
            logger.warning(f"⚠️ UnifiedBrain not available: {e}")

        try:
            from embedded_memory_system import get_embedded_memory
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.embedded_memory = loop.run_until_complete(get_embedded_memory())
            logger.info("✅ Connected to Embedded Memory")
        except Exception as e:
            logger.warning(f"⚠️ Embedded Memory not available: {e}")

        try:
            from vector_memory_system import VectorMemorySystem
            self.vector_memory = VectorMemorySystem()
            logger.info("✅ Connected to Vector Memory")
        except Exception as e:
            logger.warning(f"⚠️ Vector Memory not available: {e}")

        try:
            from conversation_memory import ConversationMemory
            self.conversation_memory = ConversationMemory()
            logger.info("✅ Connected to Conversation Memory")
        except Exception as e:
            logger.warning(f"⚠️ Conversation Memory not available: {e}")

    # ========================================================================
    # CORE CONTEXT OPERATIONS
    # ========================================================================

    async def store_context(self, entry: ContextEntry) -> str:
        """
        Store context entry across appropriate memory layers
        Ensures consistency and proper synchronization
        """
        start_time = datetime.now(timezone.utc)

        # 1. Store in appropriate layer
        if entry.layer == MemoryLayer.EPHEMERAL:
            self.ephemeral_cache[entry.key] = entry

        elif entry.layer == MemoryLayer.SESSION:
            if entry.session_id:
                if entry.session_id not in self.session_cache:
                    self.session_cache[entry.session_id] = {}
                self.session_cache[entry.session_id][entry.key] = entry

        # Always store in master registry for persistence
        conn, cursor = self._get_connection()

        cursor.execute("""
            INSERT INTO memory_context_registry
            (key, layer, scope, priority, category, source, tenant_id, user_id,
             session_id, agent_id, value, metadata, expires_at, sync_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dedupe_key)
            DO UPDATE SET
                layer = EXCLUDED.layer,
                scope = EXCLUDED.scope,
                priority = EXCLUDED.priority,
                category = EXCLUDED.category,
                source = EXCLUDED.source,
                tenant_id = EXCLUDED.tenant_id,
                user_id = EXCLUDED.user_id,
                session_id = EXCLUDED.session_id,
                agent_id = EXCLUDED.agent_id,
                value = EXCLUDED.value,
                metadata = EXCLUDED.metadata,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW(),
                sync_version = memory_context_registry.sync_version + 1,
                access_count = memory_context_registry.access_count + 1
            RETURNING id, sync_version
        """, (
            entry.key, entry.layer.value, entry.scope.value, entry.priority,
            entry.category, entry.source, entry.tenant_id, entry.user_id,
            entry.session_id, entry.agent_id, json.dumps(entry.value),
            json.dumps(entry.metadata), entry.expires_at, entry.sync_version
        ))

        result = cursor.fetchone()
        conn.commit()
        entry_id = str(result['id'])
        entry.sync_version = result['sync_version']

        # 2. Dual-write to specialized systems based on category
        await self._dual_write_to_systems(entry)

        # 3. Create sync event for cross-system coordination
        await self._create_sync_event(
            event_type="create",
            context_key=entry.key,
            source_system="coordinator",
            priority=self._determine_sync_priority(entry),
            payload=asdict(entry)
        )

        # 4. Log access
        latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        self._log_access(entry.key, "coordinator", "write", entry.session_id, latency_ms, False)

        logger.info(f"✅ Stored context: {entry.key} (layer={entry.layer.value}, scope={entry.scope.value})")
        return entry_id

    async def retrieve_context(
        self,
        key: str,
        scope: ContextScope,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None
    ) -> Optional[ContextEntry]:
        """
        Retrieve context with intelligent caching and fallback
        """
        start_time = datetime.now(timezone.utc)
        hit_cache = False

        # 1. Check ephemeral cache first
        if key in self.ephemeral_cache:
            entry = self.ephemeral_cache[key]
            if self._is_valid_entry(entry):
                hit_cache = True
                self._log_access(key, "coordinator", "read", session_id, 1, True)
                return entry

        # 2. Check session cache
        if session_id and session_id in self.session_cache:
            if key in self.session_cache[session_id]:
                entry = self.session_cache[session_id][key]
                if self._is_valid_entry(entry):
                    hit_cache = True
                    self._log_access(key, "coordinator", "read", session_id, 2, True)
                    return entry

        # 3. Query master registry
        conn, cursor = self._get_connection()

        cursor.execute("""
            SELECT * FROM memory_context_registry
            WHERE key = %s
              AND scope = %s
              AND (tenant_id = %s OR tenant_id IS NULL OR scope != 'tenant')
              AND (user_id = %s OR user_id IS NULL OR scope != 'user')
              AND (session_id = %s OR session_id IS NULL OR scope != 'session')
              AND (agent_id = %s OR agent_id IS NULL OR scope != 'agent')
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY sync_version DESC
            LIMIT 1
        """, (key, scope.value, tenant_id, user_id, session_id, agent_id))

        row = cursor.fetchone()

        if row:
            # Update access count
            cursor.execute("""
                UPDATE memory_context_registry
                SET access_count = access_count + 1
                WHERE id = %s
            """, (row['id'],))
            conn.commit()

            # Convert to ContextEntry
            entry = ContextEntry(
                key=row['key'],
                value=row['value'],
                layer=MemoryLayer(row['layer']),
                scope=ContextScope(row['scope']),
                priority=row['priority'],
                category=row['category'],
                source=row['source'],
                tenant_id=row['tenant_id'],
                user_id=row['user_id'],
                session_id=row['session_id'],
                agent_id=row['agent_id'],
                metadata=row['metadata'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                access_count=row['access_count'],
                sync_version=row['sync_version']
            )

            # Cache for future access
            if entry.layer == MemoryLayer.EPHEMERAL:
                self.ephemeral_cache[key] = entry
            elif entry.layer == MemoryLayer.SESSION and session_id:
                if session_id not in self.session_cache:
                    self.session_cache[session_id] = {}
                self.session_cache[session_id][key] = entry

            latency_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            self._log_access(key, "coordinator", "read", session_id, latency_ms, hit_cache)

            return entry

        logger.warning(f"⚠️ Context not found: {key}")
        return None

    async def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete context for a session
        Returns all relevant context entries organized by category
        """
        conn, cursor = self._get_connection()

        # Get or create session record
        cursor.execute("""
            SELECT * FROM memory_session_context
            WHERE session_id = %s
        """, (session_id,))

        session = cursor.fetchone()

        if not session:
            # Create new session
            cursor.execute("""
                INSERT INTO memory_session_context (session_id, context_snapshot)
                VALUES (%s, '{}'::jsonb)
                RETURNING *
            """, (session_id,))
            session = cursor.fetchone()
            conn.commit()

        # Get all context entries for this session
        cursor.execute("""
            SELECT * FROM memory_context_registry
            WHERE session_id = %s
               OR scope = 'global'
               OR (scope = 'tenant' AND tenant_id = %s)
            ORDER BY priority DESC, updated_at DESC
        """, (session_id, session.get('tenant_id')))

        entries = cursor.fetchall()

        # Organize by category
        context = {
            'session_id': session_id,
            'tenant_id': session.get('tenant_id'),
            'user_id': session.get('user_id'),
            'start_time': session['start_time'],
            'last_activity': session['last_activity'],
            'active_agents': session['active_agents'] or [],
            'by_category': {},
            'by_layer': {},
            'critical_context': {}
        }

        for row in entries:
            cat = row['category']
            layer = row['layer']

            if cat not in context['by_category']:
                context['by_category'][cat] = []
            if layer not in context['by_layer']:
                context['by_layer'][layer] = []

            entry_data = {
                'key': row['key'],
                'value': row['value'],
                'priority': row['priority'],
                'source': row['source'],
                'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None
            }

            context['by_category'][cat].append(entry_data)
            context['by_layer'][layer].append(entry_data)

            if row['priority'] == 'critical':
                context['critical_context'][row['key']] = row['value']

        # Update last activity
        cursor.execute("""
            UPDATE memory_session_context
            SET last_activity = NOW()
            WHERE session_id = %s
        """, (session_id,))
        conn.commit()

        return context

    async def search_context(
        self,
        query: str,
        scope: Optional[ContextScope] = None,
        layer: Optional[MemoryLayer] = None,
        category: Optional[str] = None,
        tenant_id: Optional[str] = None,
        limit: int = 20
    ) -> List[ContextEntry]:
        """
        Search across all context with filtering
        """
        conn, cursor = self._get_connection()

        # Build dynamic query
        conditions = ["(expires_at IS NULL OR expires_at > NOW())"]
        params = []

        if scope:
            conditions.append("scope = %s")
            params.append(scope.value)

        if layer:
            conditions.append("layer = %s")
            params.append(layer.value)

        if category:
            conditions.append("category = %s")
            params.append(category)

        if tenant_id:
            conditions.append("(tenant_id = %s OR scope = 'global')")
            params.append(tenant_id)

        # Add search condition
        conditions.append("""
            (key ILIKE %s OR
             (CASE WHEN jsonb_typeof(value) = 'string' THEN value #>> '{}'
                   ELSE value::text END) ILIKE %s)
        """)
        params.extend([f'%{query}%', f'%{query}%'])

        where_clause = " AND ".join(conditions)
        params.append(limit)

        cursor.execute(f"""
            SELECT * FROM memory_context_registry
            WHERE {where_clause}
            ORDER BY priority DESC, updated_at DESC
            LIMIT %s
        """, params)

        results = []
        for row in cursor.fetchall():
            results.append(ContextEntry(
                key=row['key'],
                value=row['value'],
                layer=MemoryLayer(row['layer']),
                scope=ContextScope(row['scope']),
                priority=row['priority'],
                category=row['category'],
                source=row['source'],
                tenant_id=row['tenant_id'],
                user_id=row['user_id'],
                session_id=row['session_id'],
                agent_id=row['agent_id'],
                metadata=row['metadata'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                expires_at=row['expires_at'],
                access_count=row['access_count'],
                sync_version=row['sync_version']
            ))

        logger.info(f"✅ Found {len(results)} context entries for query: {query}")
        return results

    # ========================================================================
    # SYNCHRONIZATION
    # ========================================================================

    async def _dual_write_to_systems(self, entry: ContextEntry):
        """
        Write to appropriate specialized memory systems
        """
        # UnifiedBrain for critical/persistent data
        if self.unified_brain and entry.layer in [MemoryLayer.LONG_TERM, MemoryLayer.PERMANENT]:
            try:
                self.unified_brain.store(
                    key=entry.key,
                    value=entry.value,
                    category=entry.category,
                    priority=entry.priority,
                    source=entry.source,
                    metadata=entry.metadata
                )
            except Exception as e:
                logger.error(f"❌ UnifiedBrain write failed: {e}")

        # Embedded memory for fast retrieval
        if self.embedded_memory:
            try:
                content = entry.value if isinstance(entry.value, str) else json.dumps(entry.value)
                importance = 0.9 if entry.priority == 'critical' else 0.7 if entry.priority == 'high' else 0.5

                self.embedded_memory.store_memory(
                    content=content,
                    memory_type=entry.category,
                    importance_score=importance,
                    metadata={
                        **entry.metadata,
                        'context_key': entry.key,
                        'layer': entry.layer.value,
                        'scope': entry.scope.value
                    }
                )
            except Exception as e:
                logger.error(f"❌ Embedded memory write failed: {e}")

        # Vector memory for semantic search
        if self.vector_memory and entry.category in ['knowledge', 'learning', 'insight']:
            try:
                await self.vector_memory.store_memory(
                    content=str(entry.value),
                    memory_type=entry.category,
                    importance_score=0.9 if entry.priority == 'critical' else 0.7,
                    metadata=entry.metadata
                )
            except Exception as e:
                logger.error(f"❌ Vector memory write failed: {e}")

    async def _create_sync_event(
        self,
        event_type: str,
        context_key: str,
        source_system: str,
        priority: SyncPriority,
        payload: Dict[str, Any]
    ):
        """Create synchronization event for cross-system coordination"""
        event_id = hashlib.sha256(
            f"{context_key}:{event_type}:{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        target_systems = ["unified_brain", "embedded_memory", "vector_memory"]

        conn, cursor = self._get_connection()
        cursor.execute("""
            INSERT INTO memory_sync_events
            (event_id, event_type, context_key, source_system, target_systems, priority, payload)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            event_id, event_type, context_key, source_system,
            target_systems, priority.value, json.dumps(payload, default=self._json_serializer)
        ))
        conn.commit()

    def _determine_sync_priority(self, entry: ContextEntry) -> SyncPriority:
        """Determine sync priority based on entry characteristics"""
        if entry.priority == 'critical':
            return SyncPriority.IMMEDIATE
        elif entry.scope == ContextScope.SESSION:
            return SyncPriority.HIGH
        elif entry.layer == MemoryLayer.EPHEMERAL:
            return SyncPriority.LOW
        else:
            return SyncPriority.NORMAL

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _is_valid_entry(self, entry: ContextEntry) -> bool:
        """Check if entry is still valid"""
        if entry.expires_at and entry.expires_at < datetime.now(timezone.utc):
            return False
        return True

    @staticmethod
    def _json_serializer(obj: Any):
        """Serialize objects not handled by default json encoder"""
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (datetime, timedelta)):
            return obj.isoformat() if isinstance(obj, datetime) else obj.total_seconds()
        if isinstance(obj, (set, tuple)):
            return list(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if hasattr(obj, 'isoformat'):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
        return str(obj)

    def _log_access(
        self,
        context_key: str,
        accessed_by: str,
        access_type: str,
        session_id: Optional[str],
        latency_ms: int,
        hit_cache: bool
    ):
        """Log context access for analytics"""
        try:
            conn, cursor = self._get_connection()
            cursor.execute("""
                INSERT INTO memory_context_access_log
                (context_key, accessed_by, access_type, session_id, latency_ms, hit_cache)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (context_key, accessed_by, access_type, session_id, latency_ms, hit_cache))
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to log access: {e}")

    async def cleanup_expired_context(self):
        """Remove expired context entries"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            DELETE FROM memory_context_registry
            WHERE expires_at IS NOT NULL AND expires_at < NOW()
        """)

        deleted = cursor.rowcount
        conn.commit()

        logger.info(f"✅ Cleaned up {deleted} expired context entries")
        return deleted

    async def get_stats(self) -> Dict[str, Any]:
        """Get memory coordination statistics"""
        conn, cursor = self._get_connection()

        cursor.execute("""
            SELECT
                COUNT(*) as total_entries,
                COUNT(*) FILTER (WHERE layer = 'ephemeral') as ephemeral_count,
                COUNT(*) FILTER (WHERE layer = 'session') as session_count,
                COUNT(*) FILTER (WHERE layer = 'short_term') as short_term_count,
                COUNT(*) FILTER (WHERE layer = 'long_term') as long_term_count,
                COUNT(*) FILTER (WHERE layer = 'permanent') as permanent_count,
                COUNT(*) FILTER (WHERE priority = 'critical') as critical_count,
                COUNT(DISTINCT session_id) as active_sessions,
                COUNT(DISTINCT tenant_id) as active_tenants,
                SUM(access_count) as total_accesses
            FROM memory_context_registry
            WHERE expires_at IS NULL OR expires_at > NOW()
        """)

        stats = dict(cursor.fetchone())

        # Cache stats
        stats['cache_size'] = {
            'ephemeral': len(self.ephemeral_cache),
            'session': sum(len(s) for s in self.session_cache.values())
        }

        # Pending syncs
        cursor.execute("""
            SELECT COUNT(*) as pending_syncs
            FROM memory_sync_events
            WHERE NOT processed
        """)
        stats['pending_syncs'] = cursor.fetchone()['pending_syncs']

        return stats


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": "postgres",
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", "5432"))
}

_coordinator_instance = None

def get_memory_coordinator() -> UnifiedMemoryCoordinator:
    """Get singleton memory coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = UnifiedMemoryCoordinator(DB_CONFIG)
    return _coordinator_instance


if __name__ == "__main__":
    # Test the coordinator
    coordinator = get_memory_coordinator()

    # Test storing context
    async def test():
        entry = ContextEntry(
            key="test_coordination",
            value={"message": "Perfect E2E context coordination"},
            layer=MemoryLayer.SESSION,
            scope=ContextScope.GLOBAL,
            priority="high",
            category="test",
            source="test_script",
            session_id="test_session_123"
        )

        entry_id = await coordinator.store_context(entry)
        print(f"✅ Stored: {entry_id}")

        # Retrieve it
        retrieved = await coordinator.retrieve_context(
            key="test_coordination",
            scope=ContextScope.GLOBAL,
            session_id="test_session_123"
        )

        if retrieved:
            print(f"✅ Retrieved: {retrieved.value}")

        # Get session context
        session_ctx = await coordinator.get_session_context("test_session_123")
        print(f"✅ Session context: {len(session_ctx['by_category'])} categories")

        # Stats
        stats = await coordinator.get_stats()
        print(f"✅ Stats: {stats}")

    asyncio.run(test())
