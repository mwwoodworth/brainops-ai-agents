#!/usr/bin/env python3
"""
AUREA - Autonomous Universal Resource & Execution Assistant
The Master Orchestration Brain for BrainOps AI OS
Coordinates all 59 agents to work as one unified intelligence
"""

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

# ============================================================================
# SHARED CONNECTION POOL - CRITICAL for preventing MaxClientsInSessionMode
# ============================================================================
try:
    from database.sync_pool import get_sync_pool
    _POOL_AVAILABLE = True
except ImportError:
    _POOL_AVAILABLE = False


@contextmanager
def _get_pooled_connection():
    """Get connection from shared pool - ALWAYS use this instead of psycopg2.connect()"""
    if _POOL_AVAILABLE:
        with get_sync_pool().get_connection() as conn:
            yield conn
    else:
        conn = psycopg2.connect(**DB_CONFIG)
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()


import warnings

import aiohttp

from agent_activation_system import BusinessEventType, get_activation_system
from ai_board_governance import get_ai_board
from ai_core import RealAICore
from ai_knowledge_graph import AIKnowledgeGraph, get_knowledge_graph
from ai_self_awareness import SelfAwareAI, get_self_aware_ai
from mcp_integration import MCPClient, MCPServer, MCPToolResult, get_mcp_client
from revenue_generation_system import AutonomousRevenueSystem, get_revenue_system
from unified_brain import UnifiedBrain, get_brain
from unified_memory_manager import Memory, MemoryType, get_memory_manager

warnings.filterwarnings('ignore')


def json_safe_serialize(obj: Any) -> Any:
    """Recursively convert datetime/Decimal/Enum/bytes objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, bytes):
        # Convert bytes to base64 string to avoid bytea interpretation errors
        import base64
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif hasattr(obj, '__dataclass_fields__'):
        # Handle dataclasses
        return {k: json_safe_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {str(k): json_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(json_safe_serialize(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return json_safe_serialize(obj.__dict__)
    else:
        # Fallback: convert to string
        return str(obj)


# MCP Bridge Configuration
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = os.getenv("MCP_API_KEY") or os.getenv("BRAINOPS_API_KEY") or "brainops_mcp_2025"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AUREA')

# Database configuration - supports both individual env vars and DATABASE_URL
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    from urllib.parse import urlparse

    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME", "postgres")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_port = os.getenv("DB_PORT", "5432")

    # Fallback to DATABASE_URL if individual vars not set
    if not all([db_host, db_user, db_password]):
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            db_host = parsed.hostname or ''
            db_name = parsed.path.lstrip('/') if parsed.path else 'postgres'
            db_user = parsed.username or ''
            db_password = parsed.password or ''
            db_port = str(parsed.port) if parsed.port else '5432'

    if not all([db_host, db_user, db_password]):
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        'host': db_host,
        'database': db_name,
        'user': db_user,
        'password': db_password,
        'port': int(db_port)
    }

DB_CONFIG = _get_db_config()


class DecisionType(Enum):
    """Types of decisions AUREA can make"""
    STRATEGIC = "strategic"          # Long-term business decisions
    OPERATIONAL = "operational"      # Day-to-day operations
    TACTICAL = "tactical"            # Short-term optimizations
    EMERGENCY = "emergency"          # Crisis response
    FINANCIAL = "financial"          # Money-related decisions
    CUSTOMER = "customer"            # Customer-facing decisions
    TECHNICAL = "technical"          # System and infrastructure
    LEARNING = "learning"            # Self-improvement decisions


class AutonomyLevel(Enum):
    """Levels of autonomous operation"""
    MANUAL = 0        # Human operates, AI observes
    ASSISTED = 25     # AI suggests, human decides
    SUPERVISED = 50   # AI acts, human can veto
    AUTONOMOUS = 75   # AI acts, human monitors
    FULL_AUTO = 100   # AI decides everything autonomously
    # Backwards-compatible aliases
    SEMI_AUTO = 50
    MOSTLY_AUTO = 75


@dataclass
class Decision:
    """A decision made by AUREA"""
    id: str
    type: DecisionType
    description: str
    confidence: float
    impact_assessment: str
    recommended_action: str
    alternatives: list[str]
    requires_human_approval: bool
    deadline: Optional[datetime]
    context: dict[str, Any]
    db_id: Optional[str] = None  # Database UUID after logging


@dataclass
class SystemHealth:
    """Overall system health metrics"""
    overall_score: float  # 0-100
    component_health: dict[str, float]
    active_agents: int
    memory_utilization: float
    decision_backlog: int
    error_rate: float
    performance_score: float
    alerts: list[str]


@dataclass
class CycleMetrics:
    """Metrics for a single OODA loop cycle"""
    cycle_number: int
    timestamp: datetime
    observations_count: int
    decisions_count: int
    actions_executed: int
    actions_successful: int
    actions_failed: int
    cycle_duration_seconds: float
    learning_insights_generated: int
    health_score: float
    autonomy_level: int
    patterns_detected: list[str]
    goals_achieved: int
    goals_set: int


@dataclass
class AutonomousGoal:
    """A goal set by AUREA autonomously based on system state"""
    id: str
    goal_type: str  # performance, efficiency, revenue, quality, learning
    description: str
    target_metric: str
    current_value: float
    target_value: float
    deadline: datetime
    priority: int  # 1-10
    created_at: datetime
    status: str  # active, achieved, failed, abandoned
    progress: float  # 0-100


class AUREA:
    """
    The Master AI Orchestrator - The Brain of BrainOps
    """

    def __init__(
        self,
        autonomy_level: AutonomyLevel = AutonomyLevel.SUPERVISED,
        tenant_id: Optional[str] = None,
        db_pool: Optional[Any] = None  # Async database pool injection
    ):
        if not tenant_id:
            raise ValueError("tenant_id is required for AUREA")

        self.tenant_id = tenant_id
        self.autonomy_level = autonomy_level
        self._db_pool = db_pool  # Injected async pool
        self.memory = get_memory_manager()
        self.activation_system = get_activation_system(tenant_id)
        self.ai = RealAICore()
        self.board_ref: Optional[Any] = None
        self.safety_ref: Optional[SelfAwareAI] = None
        self.revenue_ref: Optional[AutonomousRevenueSystem] = None
        self.knowledge_ref: Optional[AIKnowledgeGraph] = None
        self.brain: UnifiedBrain = get_brain()
        self.running = False
        self.cycle_count = 0
        self.decisions_made = 0
        self.last_health_check = datetime.now()
        self.system_health = None
        self.decision_queue = asyncio.Queue()
        self.learning_insights = []
        self.cycle_metrics_history: list[CycleMetrics] = []
        self.autonomous_goals: list[AutonomousGoal] = []
        self.pattern_history: list[dict[str, Any]] = []
        self.confidence_thresholds = self._default_confidence_thresholds()
        self._last_observation_bundle: dict[str, Any] = {}
        self._last_orientation_bundle: dict[str, Any] = {}
        self._decision_success_rate_history: list[float] = []
        self._performance_trends: dict[str, list[float]] = {}
        self._init_database()

        logger.info(f"🧠 AUREA initialized for tenant {tenant_id} at autonomy level: {autonomy_level.name}")

    @property
    def db_pool(self) -> Optional[Any]:
        """Get async database pool - try injected first, then global"""
        if self._db_pool is not None:
            return self._db_pool
        try:
            from database.async_connection import get_pool
            return get_pool()
        except Exception as exc:
            logger.debug("Shared pool unavailable: %s", exc, exc_info=True)
            return None

    @property
    def mcp(self) -> MCPClient:
        """Get singleton MCPClient - the centralized MCP Bridge integration"""
        if not hasattr(self, '_mcp_client') or self._mcp_client is None:
            self._mcp_client = get_mcp_client()
        return self._mcp_client

    async def _async_fetch(self, query: str, *args) -> list[dict]:
        """Execute query and return results using async pool"""
        pool = self.db_pool
        if pool is None:
            # Fallback to sync psycopg2
            return self._sync_fetch(query, *args)
        try:
            rows = await pool.fetch(query, *args)
            return [dict(r) for r in rows] if rows else []
        except Exception as e:
            logger.warning(f"Async fetch failed, falling back to sync: {e}")
            return self._sync_fetch(query, *args)

    async def _async_fetchrow(self, query: str, *args) -> Optional[dict]:
        """Execute query and return single row using async pool"""
        pool = self.db_pool
        if pool is None:
            return self._sync_fetchrow(query, *args)
        try:
            row = await pool.fetchrow(query, *args)
            return dict(row) if row else None
        except Exception as e:
            logger.warning(f"Async fetchrow failed, falling back to sync: {e}")
            return self._sync_fetchrow(query, *args)

    async def _async_execute(self, query: str, *args) -> bool:
        """Execute command using async pool"""
        pool = self.db_pool
        if pool is None:
            return self._sync_execute(query, *args)
        try:
            await pool.execute(query, *args)
            return True
        except Exception as e:
            logger.warning(f"Async execute failed, falling back to sync: {e}")
            return self._sync_execute(query, *args)

    def _sync_fetch(self, query: str, *args) -> list[dict]:
        """Sync fallback for fetch - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, args if args else None)
                rows = cur.fetchall()
                cur.close()
                return [dict(r) for r in rows] if rows else []
        except Exception as e:
            logger.error(f"Sync fetch failed: {e}")
            return []

    def _sync_fetchrow(self, query: str, *args) -> Optional[dict]:
        """Sync fallback for fetchrow - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, args if args else None)
                row = cur.fetchone()
                cur.close()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Sync fetchrow failed: {e}")
            return None

    def _sync_execute(self, query: str, *args) -> bool:
        """Sync fallback for execute - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()
                cur.execute(query, args if args else None)
                conn.commit()
                cur.close()
                return True
        except Exception as e:
            logger.error(f"Sync execute failed: {e}")
            return False

    async def _ensure_integrations(self):
        """Lazy-init optional integrations (board, safety, revenue, knowledge)."""
        if self.board_ref is None:
            try:
                self.board_ref = get_ai_board()
            except Exception as e:
                logger.warning(f"AI Board unavailable: {e}")
                self.board_ref = None

        if self.safety_ref is None:
            try:
                self.safety_ref = await get_self_aware_ai()
            except Exception as e:
                logger.warning(f"Self-Awareness unavailable: {e}")
                self.safety_ref = None

        if self.revenue_ref is None:
            try:
                self.revenue_ref = get_revenue_system()
            except Exception as e:
                logger.warning(f"Revenue Engine unavailable: {e}")
                self.revenue_ref = None

        if self.knowledge_ref is None:
            try:
                self.knowledge_ref = get_knowledge_graph()
            except Exception as e:
                logger.warning(f"Knowledge Graph unavailable: {e}")
                self.knowledge_ref = None

    def _default_confidence_thresholds(self) -> dict[int, float]:
        """Default confidence thresholds (0-100) for autonomous execution."""
        # MANUAL/ASSISTED are handled by policy (always require approval).
        return {
            AutonomyLevel.SUPERVISED.value: 85.0,
            AutonomyLevel.AUTONOMOUS.value: 75.0,
            AutonomyLevel.FULL_AUTO.value: 65.0,
        }

    def _safe_json(self, text: Any) -> Any:
        if isinstance(text, (dict, list)):
            return text
        if text is None:
            return {}
        s = str(text).strip()
        s = re.sub(r"^```(?:json)?\\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\\s*```$", "", s)
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse JSON payload: %s", exc)
            match = re.search(r"(\\{.*\\}|\\[.*\\])", s, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.debug("Failed to parse extracted JSON payload: %s", exc)
                    return {}
        return {}

    def _db_connect(self):
        """Get connection from shared pool"""
        return _get_pooled_connection()

    def _db_fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        try:
            with self._db_connect() as conn:
                if not conn:
                    return []
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, params)
                rows = cur.fetchall()
                cur.close()
                return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"DB fetchall failed: {e}")
            return []

    def _db_fetchone(self, query: str, params: tuple[Any, ...] = ()) -> Optional[dict[str, Any]]:
        try:
            with self._db_connect() as conn:
                if not conn:
                    return None
                cur = conn.cursor(cursor_factory=RealDictCursor)
                cur.execute(query, params)
                row = cur.fetchone()
                cur.close()
                return dict(row) if row else None
        except Exception as e:
            logger.debug(f"DB fetchone failed: {e}")
            return None

    def _db_execute(self, query: str, params: tuple[Any, ...] = ()) -> bool:
        try:
            with self._db_connect() as conn:
                if not conn:
                    return False
                cur = conn.cursor()
                cur.execute(query, params)
                conn.commit()
                cur.close()
                return True
        except Exception as e:
            logger.warning(f"DB execute failed: {e}")
            return False

    def _truncate_text(self, value: Any, max_len: int = 800) -> str:
        s = "" if value is None else str(value)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    def _compact_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        max_rows: int = 30,
        max_field_len: int = 600
    ) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for row in (rows or [])[:max_rows]:
            cleaned: dict[str, Any] = {}
            for k, v in dict(row).items():
                if isinstance(v, (datetime,)):
                    cleaned[k] = v.isoformat()
                elif isinstance(v, Decimal):
                    cleaned[k] = float(v)
                else:
                    cleaned[k] = self._truncate_text(v, max_len=max_field_len)
            compacted.append(cleaned)
        return compacted

    def _coerce_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError) as exc:
            logger.debug("Failed to coerce float from %s: %s", value, exc)
            return None

    def _extract_keywords(self, signals: list[dict[str, Any]]) -> list[str]:
        keywords: list[str] = []
        for s in signals or []:
            for key in ("affected_components", "components", "component", "metric_name", "agent_name"):
                v = s.get(key)
                if isinstance(v, list):
                    keywords.extend([str(x) for x in v if x])
                elif v:
                    keywords.append(str(v))
            for key in ("title", "summary"):
                v = s.get(key)
                if v:
                    words = re.findall(r"[A-Za-z0-9_\\-\\.]{3,}", str(v))
                    keywords.extend(words[:12])
        # de-dupe while preserving order
        seen = set()
        out: list[str] = []
        for k in keywords:
            kl = k.strip()
            if not kl:
                continue
            norm = kl.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out.append(kl)
        return out[:25]

    def _store_state_snapshot(self, state_type: str, state_data: dict[str, Any]):
        """Persist OODA snapshots for audit/debug."""
        try:
            # Sanitize state_data to ensure JSON-serializable
            sanitized = json_safe_serialize(state_data)
            success = self._db_execute(
                """
                INSERT INTO aurea_state (state_type, state_data, cycle_number, tenant_id)
                VALUES (%s, %s, %s, %s)
                """,
                (state_type, Json(sanitized), self.cycle_count, self.tenant_id),
            )
            if not success:
                logger.warning(f"Failed to store AUREA state snapshot: {state_type}")
        except Exception as e:
            logger.error(f"Error storing AUREA state snapshot {state_type}: {e}")

    def _init_database(self):
        """Initialize AUREA's database tables - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()

                # Create AUREA decision log
                cur.execute("""
                CREATE TABLE IF NOT EXISTS aurea_decisions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    decision_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    impact_assessment TEXT,
                    recommended_action TEXT,
                    alternatives JSONB,
                    requires_human_approval BOOLEAN DEFAULT FALSE,
                    human_approved BOOLEAN,
                    human_feedback TEXT,
                    execution_status TEXT CHECK (execution_status IN (
                        'pending', 'approved', 'rejected', 'executing', 'completed', 'failed'
                    )),
                    execution_result JSONB,
                    context JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    executed_at TIMESTAMP,
                    tenant_id UUID
                );

                CREATE INDEX IF NOT EXISTS idx_aurea_decisions_type ON aurea_decisions(decision_type);
                CREATE INDEX IF NOT EXISTS idx_aurea_decisions_status ON aurea_decisions(execution_status);
                CREATE INDEX IF NOT EXISTS idx_aurea_decisions_created ON aurea_decisions(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_aurea_decisions_tenant ON aurea_decisions(tenant_id);

                -- Create AUREA system state
                CREATE TABLE IF NOT EXISTS aurea_state (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    state_type TEXT NOT NULL,
                    state_data JSONB NOT NULL,
                    cycle_number INTEGER,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID
                );

                -- Create AUREA learning log
                CREATE TABLE IF NOT EXISTS aurea_learning (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    insight_type TEXT NOT NULL,
                    insight TEXT NOT NULL,
                    confidence FLOAT,
                    source_data JSONB,
                    applied BOOLEAN DEFAULT FALSE,
                    impact_measured FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID
                );

                -- Create AUREA cycle metrics
                CREATE TABLE IF NOT EXISTS aurea_cycle_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    cycle_number INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    observations_count INTEGER,
                    decisions_count INTEGER,
                    actions_executed INTEGER,
                    actions_successful INTEGER,
                    actions_failed INTEGER,
                    cycle_duration_seconds FLOAT,
                    learning_insights_generated INTEGER,
                    health_score FLOAT,
                    autonomy_level INTEGER,
                    patterns_detected JSONB,
                    goals_achieved INTEGER,
                    goals_set INTEGER,
                    tenant_id UUID
                );

                CREATE INDEX IF NOT EXISTS idx_aurea_cycle_metrics_cycle ON aurea_cycle_metrics(cycle_number);
                CREATE INDEX IF NOT EXISTS idx_aurea_cycle_metrics_timestamp ON aurea_cycle_metrics(timestamp DESC);

                -- Create AUREA autonomous goals
                CREATE TABLE IF NOT EXISTS aurea_autonomous_goals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    goal_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    target_metric TEXT NOT NULL,
                    current_value FLOAT,
                    target_value FLOAT,
                    deadline TIMESTAMP,
                    priority INTEGER,
                    status TEXT CHECK (status IN ('active', 'achieved', 'failed', 'abandoned')),
                    progress FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    achieved_at TIMESTAMP,
                    tenant_id UUID
                );

                CREATE INDEX IF NOT EXISTS idx_aurea_goals_status ON aurea_autonomous_goals(status);
                CREATE INDEX IF NOT EXISTS idx_aurea_goals_priority ON aurea_autonomous_goals(priority DESC);

                -- Create AUREA pattern detection log
                CREATE TABLE IF NOT EXISTS aurea_patterns (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    pattern_type TEXT NOT NULL,
                    pattern_description TEXT NOT NULL,
                    confidence FLOAT,
                    frequency INTEGER,
                    impact_score FLOAT,
                    pattern_data JSONB,
                    first_detected TIMESTAMP DEFAULT NOW(),
                    last_detected TIMESTAMP DEFAULT NOW(),
                    tenant_id UUID
                );

                CREATE INDEX IF NOT EXISTS idx_aurea_patterns_type ON aurea_patterns(pattern_type);
                CREATE INDEX IF NOT EXISTS idx_aurea_patterns_confidence ON aurea_patterns(confidence DESC);
                """)

                conn.commit()
                cur.close()

            logger.info("✅ AUREA database initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize AUREA database: {e}")

    async def orchestrate(self):
        """Main orchestration loop - the heartbeat of AUREA"""
        self.running = True
        logger.info("🚀 AUREA orchestration started")
        await self._ensure_integrations()

        while self.running:
            try:
                self.cycle_count += 1
                cycle_start = datetime.now()

                # Phase 1: Observe - Check all triggers and gather data
                observations = await self._observe()
                self._store_state_snapshot("observe", {
                    "cycle": self.cycle_count,
                    "observations_count": len(observations),
                    "observations": observations[:10]  # Limit for storage
                })

                # Phase 2: Orient - Analyze situation and context
                context = await self._orient(observations)
                self._store_state_snapshot("orient", {
                    "cycle": self.cycle_count,
                    "priorities_count": len(context.get("priorities", [])),
                    "risks_count": len(context.get("risks", [])),
                    "opportunities_count": len(context.get("opportunities", []))
                })

                # Phase 3: Decide - Make decisions based on context
                decisions = await self._decide(context)
                self._store_state_snapshot("decide", {
                    "cycle": self.cycle_count,
                    "decisions_count": len(decisions),
                    "decision_types": [d.type.value for d in decisions]
                })

                # Phase 4: Act - Execute decisions through agents
                results = await self._act(decisions)
                self._store_state_snapshot("act", {
                    "cycle": self.cycle_count,
                    "results_count": len(results),
                    "success_count": len([r for r in results if r.get("status") != "failed"]),
                    "failed_count": len([r for r in results if r.get("status") == "failed"])
                })

                # Phase 5: Learn - Analyze results and improve
                await self._learn(results)

                # Phase 6: Heal - Fix any issues detected
                await self._self_heal()

                # Calculate cycle time
                cycle_time = (datetime.now() - cycle_start).total_seconds()

                # Count patterns detected this cycle
                patterns_count = len([obs for obs in observations if obs.get("type") == "patterns_detected"])

                # Count goals achieved and set this cycle
                goals_achieved = len([g for g in self.autonomous_goals if g.status == "achieved" and
                                      (datetime.now() - g.created_at).total_seconds() < cycle_time])
                goals_set = len([obs for obs in observations if obs.get("type") == "goal_progress"])

                # Create comprehensive cycle metrics
                cycle_metrics = CycleMetrics(
                    cycle_number=self.cycle_count,
                    timestamp=datetime.now(),
                    observations_count=len(observations),
                    decisions_count=len(decisions),
                    actions_executed=len(results),
                    actions_successful=len([r for r in results if r.get("status") != "failed"]),
                    actions_failed=len([r for r in results if r.get("status") == "failed"]),
                    cycle_duration_seconds=cycle_time,
                    learning_insights_generated=len(self.learning_insights) if self.learning_insights else 0,
                    health_score=self.system_health.overall_score if self.system_health else 0,
                    autonomy_level=self.autonomy_level.value,
                    patterns_detected=[obs.get("type", "") for obs in observations if obs.get("type") == "patterns_detected"],
                    goals_achieved=goals_achieved,
                    goals_set=goals_set
                )

                # Store cycle metrics
                self.cycle_metrics_history.append(cycle_metrics)
                if len(self.cycle_metrics_history) > 1000:
                    self.cycle_metrics_history.pop(0)

                await self._store_cycle_metrics(cycle_metrics)

                # Store final cycle state
                self._store_state_snapshot("cycle_complete", {
                    "cycle": self.cycle_count,
                    "cycle_time_seconds": cycle_time,
                    "observations": len(observations),
                    "decisions": len(decisions),
                    "results": len(results),
                    "patterns_detected": patterns_count,
                    "goals_achieved": goals_achieved,
                    "goals_set": goals_set
                })

                # Store cycle in memory
                self.memory.store(Memory(
                    memory_type=MemoryType.PROCEDURAL,
                    content={
                        "cycle": self.cycle_count,
                        "signals_observed": len(observations),
                        "decisions": len(decisions),
                        "actions_executed": len(results),
                        "cycle_time_seconds": cycle_time,
                        "autonomy_level": self.autonomy_level.value,
                        "patterns_detected": patterns_count,
                        "success_rate": cycle_metrics.actions_successful / max(cycle_metrics.actions_executed, 1)
                    },
                    source_system="aurea",
                    source_agent="orchestrator",
                    created_by="aurea",
                    importance_score=0.3,
                    tags=["orchestration", "cycle"],
                    tenant_id=self.tenant_id
                ))

                # Log cycle completion
                if self.cycle_count % 10 == 0:
                    logger.info(f"🔄 AUREA Cycle {self.cycle_count} completed in {cycle_time:.2f}s")

                # Adaptive sleep based on activity
                sleep_time = self._calculate_sleep_time(len(observations), len(decisions))
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"❌ AUREA cycle error: {e}")
                await self._handle_orchestration_error(e)
                await asyncio.sleep(30)  # Longer sleep on error

    async def _observe(self) -> list[dict[str, Any]]:
        """
        Observe the environment and gather all relevant data (ASYNC)
        Enhanced with comprehensive metrics, pattern detection, and trend analysis
        """
        observations = []
        patterns_detected = []

        try:
            # Track observation start time
            observe_start = datetime.now()

            # Detect patterns from historical data
            patterns_detected = await self._detect_patterns()
            if patterns_detected:
                observations.append({
                    "type": "patterns_detected",
                    "patterns": patterns_detected,
                    "count": len(patterns_detected),
                    "trigger": BusinessEventType.SYSTEM_HEALTH_CHECK
                })
            # Check for new customers (ASYNC)
            new_cust = await self._async_fetchrow("""
            SELECT COUNT(*) as count FROM customers
            WHERE created_at > NOW() - INTERVAL '5 minutes'
            AND tenant_id = $1
            """, self.tenant_id)
            if new_cust and new_cust.get('count', 0) > 0:
                observations.append({
                    "type": "new_customers",
                    "count": new_cust['count'],
                    "trigger": BusinessEventType.NEW_CUSTOMER
                })

            # Check for pending estimates (ASYNC)
            pending_est = await self._async_fetchrow("""
            SELECT COUNT(*) as count, MIN(created_at) as oldest
            FROM estimates WHERE status = 'pending'
            AND tenant_id = $1
            """, self.tenant_id)
            if pending_est and pending_est.get('count', 0) > 0:
                observations.append({
                    "type": "pending_estimates",
                    "count": pending_est['count'],
                    "oldest": pending_est.get('oldest'),
                    "trigger": BusinessEventType.ESTIMATE_REQUESTED
                })

            # Check for overdue invoices (ASYNC)
            overdue = await self._async_fetchrow("""
            SELECT COUNT(*) as count,
                   SUM(COALESCE(balance_cents::numeric/100, 0)) as total_due
            FROM invoices
            WHERE due_date < NOW() AND status != 'paid'
            AND tenant_id = $1
            """, self.tenant_id)
            if overdue and overdue.get('count', 0) > 0:
                observations.append({
                    "type": "overdue_invoices",
                    "count": overdue['count'],
                    "total_due": float(overdue.get('total_due') or 0),
                    "trigger": BusinessEventType.INVOICE_OVERDUE
                })

            # Check for scheduling conflicts (ASYNC)
            conflicts = await self._async_fetchrow("""
            SELECT COUNT(*) as conflicts FROM jobs j1
            JOIN jobs j2 ON j1.crew_id = j2.crew_id
            WHERE j1.id != j2.id
              AND j1.scheduled_start < j2.scheduled_end
              AND j1.scheduled_end > j2.scheduled_start
              AND j1.status = 'scheduled' AND j2.status = 'scheduled'
              AND j1.tenant_id = $1 AND j2.tenant_id = $1
            """, self.tenant_id)
            if conflicts and conflicts.get('conflicts', 0) > 0:
                observations.append({
                    "type": "scheduling_conflicts",
                    "count": conflicts['conflicts'],
                    "trigger": BusinessEventType.SCHEDULING_CONFLICT
                })

            # Check system health
            if (datetime.now() - self.last_health_check).total_seconds() > 300:
                health = await self._check_system_health()
                observations.append({
                    "type": "system_health",
                    "health": health,
                    "trigger": BusinessEventType.SYSTEM_HEALTH_CHECK
                })

            # Check for customer churn risks (ASYNC)
            churn = await self._async_fetchrow("""
            SELECT COUNT(*) as at_risk FROM customers c
            WHERE EXISTS (SELECT 1 FROM jobs j WHERE j.customer_id = c.id)
              AND COALESCE(c.last_job_date,
                          (SELECT MAX(scheduled_start) FROM jobs WHERE customer_id = c.id),
                          c.created_at) < NOW() - INTERVAL '90 days'
              AND tenant_id = $1
            """, self.tenant_id)
            if churn and churn.get('at_risk', 0) > 0:
                observations.append({
                    "type": "churn_risk",
                    "count": churn['at_risk'],
                    "trigger": BusinessEventType.CUSTOMER_CHURN_RISK
                })

        except Exception as e:
            logger.error(f"Observation error: {e}")

        # Check UI/Frontend health (async HTTP check)
        try:
            async with aiohttp.ClientSession() as session:
                # Quick health check of frontends
                frontend_health = {"status": "checking", "apps": {}}
                for app_name, app_url in [
                    ("weathercraft-erp", "https://weathercraft-erp.vercel.app"),
                    ("myroofgenius", "https://myroofgenius.com"),
                    ("command-center", "https://brainops-command-center.vercel.app")
                ]:
                    try:
                        async with session.get(app_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            frontend_health["apps"][app_name] = {
                                "status": "healthy" if resp.status == 200 else "degraded",
                                "status_code": resp.status
                            }
                    except Exception as e:
                        frontend_health["apps"][app_name] = {"status": "down", "error": str(e)}

                # Check if any frontends are down
                unhealthy_apps = [app for app, data in frontend_health["apps"].items() if data["status"] != "healthy"]
                if unhealthy_apps:
                    observations.append({
                        "type": "frontend_health_issue",
                        "unhealthy_apps": unhealthy_apps,
                        "details": frontend_health["apps"],
                        "trigger": BusinessEventType.SYSTEM_HEALTH_CHECK
                    })
                else:
                    frontend_health["status"] = "all_healthy"
        except Exception as e:
            logger.warning(f"Frontend health check failed: {e}")

        # Check autonomous goals progress
        goal_updates = await self._check_goal_progress()
        if goal_updates:
            observations.append({
                "type": "goal_progress",
                "goals_updated": goal_updates,
                "trigger": BusinessEventType.SYSTEM_HEALTH_CHECK
            })

        # Analyze performance trends
        trends = await self._analyze_performance_trends()
        if trends.get("anomalies"):
            observations.append({
                "type": "performance_anomaly",
                "trends": trends,
                "trigger": BusinessEventType.SYSTEM_HEALTH_CHECK
            })

        # Store observation metrics in unified brain
        try:
            self.brain.store(
                key=f"aurea_observations_cycle_{self.cycle_count}",
                value={
                    "cycle": self.cycle_count,
                    "observation_count": len(observations),
                    "patterns_detected": len(patterns_detected),
                    "timestamp": datetime.now().isoformat()
                },
                category="aurea_metrics",
                priority="medium",
                source="aurea_observe"
            )
        except Exception as e:
            logger.warning(f"Failed to store observations in brain: {e}")

        return observations

    async def _orient(self, observations: list[dict]) -> dict[str, Any]:
        """
        Analyze observations and build context
        Enhanced with autonomous goal setting and multi-criteria analysis
        """
        context = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "autonomy_level": self.autonomy_level.value,
            "observations": observations,
            "priorities": [],
            "risks": [],
            "opportunities": [],
            "patterns": [],
            "autonomous_goals": []
        }

        # Set autonomous goals based on system state
        new_goals = await self._set_autonomous_goals(observations)
        context["autonomous_goals"] = new_goals

        # Analyze each observation for priority
        for obs in observations:
            if obs["type"] == "overdue_invoices" and obs["total_due"] > 10000:
                context["priorities"].append({
                    "type": "financial",
                    "description": f"Collect ${obs['total_due']:.2f} in overdue invoices",
                    "urgency": "high"
                })

            elif obs["type"] == "scheduling_conflicts":
                context["priorities"].append({
                    "type": "operational",
                    "description": f"Resolve {obs['count']} scheduling conflicts",
                    "urgency": "high"
                })

            elif obs["type"] == "churn_risk":
                context["risks"].append({
                    "type": "customer",
                    "description": f"{obs['count']} customers at churn risk",
                    "impact": "revenue_loss"
                })

            elif obs["type"] == "pending_estimates":
                context["opportunities"].append({
                    "type": "revenue",
                    "description": f"Convert {obs['count']} pending estimates",
                    "potential_value": obs['count'] * 5000  # Assume $5k average
                })

            elif obs["type"] == "frontend_health_issue":
                context["priorities"].append({
                    "type": "technical",
                    "description": f"Frontend apps unhealthy: {', '.join(obs['unhealthy_apps'])}",
                    "urgency": "critical",
                    "details": obs.get("details", {})
                })

        # Recall relevant memories for context
        relevant_memories = self.memory.recall(
            {"context": "decision_making", "observations": observations},
            tenant_id=self.tenant_id,
            limit=5
        )
        context["historical_context"] = relevant_memories

        return context

    async def _decide(self, context: dict[str, Any]) -> list[Decision]:
        """Make decisions based on context and autonomy level"""
        decisions = []

        # Financial decisions
        for priority in context.get("priorities", []):
            if priority["type"] == "financial" and priority["urgency"] == "high":
                decision = Decision(
                    id=f"dec-{self.cycle_count}-{len(decisions)}",
                    type=DecisionType.FINANCIAL,
                    description="Initiate collection campaign for overdue invoices",
                    confidence=0.85,
                    impact_assessment="Recover outstanding revenue, improve cash flow",
                    recommended_action="activate_collection_agents",
                    alternatives=["send_reminders", "offer_payment_plans", "escalate_to_legal"],
                    requires_human_approval=self.autonomy_level.value < 75,
                    deadline=datetime.now() + timedelta(hours=24),
                    context={"priority": priority}
                )
                decisions.append(decision)

        # Operational decisions
        for priority in context.get("priorities", []):
            if priority["type"] == "operational":
                decision = Decision(
                    id=f"dec-{self.cycle_count}-{len(decisions)}",
                    type=DecisionType.OPERATIONAL,
                    description="Optimize schedule to resolve conflicts",
                    confidence=0.90,
                    impact_assessment="Eliminate scheduling conflicts, improve efficiency",
                    recommended_action="activate_scheduling_optimization",
                    alternatives=["manual_rescheduling", "crew_reallocation"],
                    requires_human_approval=self.autonomy_level.value < 50,
                    deadline=datetime.now() + timedelta(hours=2),
                    context={"priority": priority}
                )
                decisions.append(decision)

        # Technical/Frontend decisions
        for priority in context.get("priorities", []):
            if priority["type"] == "technical" and priority.get("urgency") == "critical":
                decision = Decision(
                    id=f"dec-{self.cycle_count}-{len(decisions)}",
                    type=DecisionType.TECHNICAL,
                    description=f"Investigate frontend issue: {priority['description']}",
                    confidence=0.95,
                    impact_assessment="User-facing applications affected - immediate investigation needed",
                    recommended_action="trigger_frontend_investigation",
                    alternatives=["check_vercel_logs", "redeploy_frontend", "notify_devops"],
                    requires_human_approval=False,  # Auto-investigate technical issues
                    deadline=datetime.now() + timedelta(minutes=30),
                    context={"priority": priority}
                )
                decisions.append(decision)

        # Customer retention decisions
        for risk in context.get("risks", []):
            if risk["type"] == "customer":
                decision = Decision(
                    id=f"dec-{self.cycle_count}-{len(decisions)}",
                    type=DecisionType.CUSTOMER,
                    description="Launch retention campaign for at-risk customers",
                    confidence=0.75,
                    impact_assessment="Prevent customer churn, maintain revenue",
                    recommended_action="activate_retention_campaign",
                    alternatives=["personal_outreach", "special_offers", "loyalty_program"],
                    requires_human_approval=self.autonomy_level.value < 50,
                    deadline=datetime.now() + timedelta(days=7),
                    context={"risk": risk}
                )
                decisions.append(decision)

        # Opportunity decisions
        for opportunity in context.get("opportunities", []):
            if opportunity["type"] == "revenue" and opportunity["potential_value"] > 10000:
                decision = Decision(
                    id=f"dec-{self.cycle_count}-{len(decisions)}",
                    type=DecisionType.STRATEGIC,
                    description="Accelerate estimate conversion process",
                    confidence=0.80,
                    impact_assessment=f"Potential revenue: ${opportunity['potential_value']}",
                    recommended_action="activate_sales_acceleration",
                    alternatives=["follow_up_calls", "discount_offers", "urgency_campaign"],
                    requires_human_approval=self.autonomy_level.value < 75,
                    deadline=datetime.now() + timedelta(days=3),
                    context={"opportunity": opportunity}
                )
                decisions.append(decision)

        # Log decisions and capture their database IDs
        for decision in decisions:
            db_id = self._log_decision(decision)
            if db_id:
                decision.db_id = db_id
                logger.info(f"📝 Decision {decision.id} logged with db_id={db_id}")
            else:
                logger.error(f"❌ Failed to log decision {decision.id} - execution may fail")

        return decisions

    async def _act(self, decisions: list[Decision]) -> list[dict[str, Any]]:
        """Execute decisions through agent activation"""
        results = []

        for decision in decisions:
            try:
                # Check if human approval needed
                if decision.requires_human_approval:
                    approval = await self._request_human_approval(decision)
                    if not approval:
                        results.append({
                            "decision_id": decision.id,
                            "status": "rejected",
                            "reason": "Human approval denied"
                        })
                        continue

                # Check if decision was logged to database
                if not decision.db_id:
                    logger.warning(f"⚠️ Decision {decision.id} has no db_id - skipping execution")
                    results.append({
                        "decision_id": decision.id,
                        "status": "skipped",
                        "reason": "Decision not logged to database"
                    })
                    continue

                # Execute the decision (ACTIVE mode for internal operations)
                # Outreach to seeded contacts remains protected via dry_run flag
                result = await self._execute_decision(decision)
                result["execution_mode"] = "active"  # System is now operational
                result["outreach_protected"] = True  # Seeded contacts protected
                results.append(result)

                # Update decision status in database using the actual database UUID
                await self._update_decision_status(decision.db_id, "completed", result)
                logger.info(f"✅ Decision {decision.db_id} executed and marked completed")

                # Store execution in memory
                self.memory.store(Memory(
                    memory_type=MemoryType.EPISODIC,
                    content={
                        "decision": json_safe_serialize(asdict(decision)),
                        "execution_result": json_safe_serialize(result),
                        "timestamp": datetime.now().isoformat()
                    },
                    source_system="aurea",
                    source_agent="executor",
                    created_by="aurea",
                    importance_score=0.7,
                    tags=["decision", "execution", decision.type.value],
                    tenant_id=self.tenant_id
                ))

            except Exception as e:
                logger.error(f"Failed to execute decision {decision.id}: {e}")
                # Only update database if we have a valid db_id
                if decision.db_id:
                    await self._update_decision_status(decision.db_id, "failed", {"error": str(e)})
                results.append({
                    "decision_id": decision.id,
                    "db_id": decision.db_id,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    async def _update_decision_status(self, decision_id: str, status: str, result: dict = None):
        """Update decision execution status in database (ASYNC)"""
        try:
            # Serialize result to ensure all datetime/Decimal objects are converted
            safe_result = json_safe_serialize(result) if result else None
            safe_json = json.dumps(safe_result) if safe_result else None

            # Try async first, fallback to sync
            pool = self.db_pool
            if pool is not None:
                await pool.execute("""
                UPDATE aurea_decisions
                SET execution_status = $1::varchar,
                    execution_result = $2::jsonb,
                    outcome = $2::jsonb,
                    status = $1::varchar,
                    success = CASE WHEN $1 = 'completed' THEN true ELSE false END,
                    executed_at = NOW(),
                    updated_at = NOW()
                WHERE id::text = $3::text
                """, status, safe_json, str(decision_id))
                logger.info(f"✅ Updated decision {decision_id} status to {status} with outcome")
            else:
                # Sync fallback - uses shared pool
                with _get_pooled_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                    UPDATE aurea_decisions
                    SET execution_status = %s,
                        execution_result = %s,
                        outcome = %s,
                        status = %s,
                        success = %s,
                        executed_at = NOW(),
                        updated_at = NOW()
                    WHERE id::text = %s
                    """, (
                        status,
                        Json(safe_result) if safe_result else None,
                        Json(safe_result) if safe_result else None,
                        status,
                        status == 'completed',
                        decision_id
                    ))
                    rows_updated = cur.rowcount
                    conn.commit()
                    cur.close()
                    if rows_updated > 0:
                        logger.info(f"✅ Updated decision {decision_id} status to {status} with outcome")
                    else:
                        logger.warning(f"⚠️ No decision found with id {decision_id} to update")

        except Exception as e:
            logger.error(f"Failed to update decision status: {e}")

    async def _execute_decision(self, decision: Decision) -> dict[str, Any]:
        """Execute a specific decision - now with MCP integration"""
        action_map = {
            "activate_collection_agents": self._activate_collection_agents,
            "activate_scheduling_optimization": self._activate_scheduling_optimization,
            "activate_retention_campaign": self._activate_retention_campaign,
            "activate_sales_acceleration": self._activate_sales_acceleration,
            "trigger_frontend_investigation": self._trigger_frontend_investigation,
            "trigger_deploy": self._trigger_deploy_via_mcp,
            "restart_service": self._restart_service_via_mcp,
        }

        action_func = action_map.get(decision.recommended_action)
        if action_func:
            return await action_func(decision.context)
        else:
            # Default: activate relevant agents based on decision type
            return await self._activate_agents_for_decision(decision)

    async def _trigger_frontend_investigation(self, context: dict) -> dict:
        """Investigate frontend issues using MCP tools (Vercel integration)"""
        try:
            mcp = self.mcp
            # Check Vercel deployments for issues
            result = await mcp.execute_tool(MCPServer.VERCEL, "vercel_list_deployments", {})
            logger.info(f"🔍 Frontend investigation via MCP: {result}")
            return {"action": "frontend_investigation", "mcp_result": result, "mode": "active"}
        except Exception as e:
            logger.error(f"MCP frontend investigation failed: {e}")
            return {"action": "frontend_investigation", "error": str(e), "mode": "fallback"}

    async def _trigger_deploy_via_mcp(self, context: dict) -> dict:
        """Trigger deployment using MCP Render integration"""
        try:
            mcp = self.mcp
            service_id = context.get("service_id", "brainops-ai-agents")
            result = await mcp.execute_tool(MCPServer.RENDER, "render_trigger_deploy", {"service_id": service_id})
            logger.info(f"🚀 Triggered deploy via MCP: {result}")
            return {"action": "trigger_deploy", "mcp_result": result, "mode": "active"}
        except Exception as e:
            logger.error(f"MCP deploy trigger failed: {e}")
            return {"action": "trigger_deploy", "error": str(e), "mode": "fallback"}

    async def _restart_service_via_mcp(self, context: dict) -> dict:
        """Restart service using MCP Render integration"""
        try:
            mcp = self.mcp
            service_id = context.get("service_id", "brainops-ai-agents")
            result = await mcp.execute_tool(MCPServer.RENDER, "render_restart_service", {"service_id": service_id})
            logger.info(f"🔄 Restarted service via MCP: {result}")
            return {"action": "restart_service", "mcp_result": result, "mode": "active"}
        except Exception as e:
            logger.error(f"MCP service restart failed: {e}")
            return {"action": "restart_service", "error": str(e), "mode": "fallback"}

    async def _activate_collection_agents(self, context: dict) -> dict:
        """Activate agents for collections (Active, outreach protected for seeded data)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.INVOICE_OVERDUE,
            {"context": context, "aurea_initiated": True, "execution_active": True, "dry_run_outreach": True}
        )
        return {"action": "collection_agents", "result": result, "mode": "active", "outreach_protected": True}

    async def _activate_scheduling_optimization(self, context: dict) -> dict:
        """Activate scheduling optimization (Fully active - internal operation)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.SCHEDULING_CONFLICT,
            {"context": context, "aurea_initiated": True, "execution_active": True}
        )
        return {"action": "scheduling_optimization", "result": result, "mode": "active"}

    async def _activate_retention_campaign(self, context: dict) -> dict:
        """Activate customer retention campaign (Active, outreach protected for seeded data)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.CUSTOMER_CHURN_RISK,
            {"context": context, "aurea_initiated": True, "execution_active": True, "dry_run_outreach": True}
        )
        return {"action": "retention_campaign", "result": result, "mode": "active", "outreach_protected": True}

    async def _activate_sales_acceleration(self, context: dict) -> dict:
        """Activate sales acceleration (Active, outreach protected for seeded data)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.QUOTE_REQUESTED,
            {"context": context, "aurea_initiated": True, "execution_active": True, "dry_run_outreach": True}
        )
        return {"action": "sales_acceleration", "result": result, "mode": "active", "outreach_protected": True}

    async def _activate_agents_for_decision(self, decision: Decision) -> dict:
        """Generic agent activation based on decision type (ACTIVE mode)"""
        event_type_map = {
            DecisionType.FINANCIAL: BusinessEventType.PAYMENT_RECEIVED,
            DecisionType.OPERATIONAL: BusinessEventType.JOB_SCHEDULED,
            DecisionType.CUSTOMER: BusinessEventType.NEW_CUSTOMER,
            DecisionType.STRATEGIC: BusinessEventType.SYSTEM_HEALTH_CHECK
        }

        event_type = event_type_map.get(decision.type, BusinessEventType.SYSTEM_HEALTH_CHECK)
        # Serialize decision to ensure datetime/Decimal objects are JSON-safe
        decision_dict = json_safe_serialize(asdict(decision))

        # Outreach to seeded contacts protected, internal operations fully active
        requires_outreach = decision.type in [DecisionType.CUSTOMER]
        result = await self.activation_system.handle_business_event(
            event_type,
            {"decision": decision_dict, "aurea_initiated": True, "execution_active": True,
             "dry_run_outreach": requires_outreach}
        )

        return {"action": "generic_activation", "event_type": event_type.value, "result": result,
                "mode": "active", "outreach_protected": requires_outreach}

    async def _learn(self, results: list[dict[str, Any]]):
        """
        Learn from ALL execution results and continuously improve
        Enhanced with pattern recognition, self-improvement, and unified brain integration
        """
        successful = [r for r in results if r.get("status") != "failed"]
        failed = [r for r in results if r.get("status") == "failed"]

        # Calculate success rate
        success_rate = len(successful) / len(results) if results else 0

        # Track success rate history for trend analysis
        self._decision_success_rate_history.append(success_rate)
        if len(self._decision_success_rate_history) > 100:
            self._decision_success_rate_history.pop(0)

        # ALWAYS learn from every cycle - not just failures
        insight = {
            "type": "execution_cycle",
            "insight": f"Cycle {self.cycle_count} completed with {success_rate*100:.0f}% success",
            "data": {
                "cycle": self.cycle_count,
                "success_rate": success_rate,
                "total_decisions": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "timestamp": datetime.now().isoformat()
            }
        }
        self.learning_insights.append(insight)
        await self._apply_learning(insight)

        # Store in unified brain for long-term learning
        try:
            self.brain.store_learning(
                agent_id="aurea_orchestrator",
                task_id=f"cycle_{self.cycle_count}",
                mistake=f"Cycle execution with {success_rate*100:.0f}% success",
                lesson=f"Decision patterns and outcomes from cycle {self.cycle_count}",
                root_cause="autonomous_operation",
                impact="high" if success_rate < 0.7 else "medium"
            )
        except Exception as e:
            logger.warning(f"Failed to store learning in unified brain: {e}")

        # Generate performance insight if low success
        if success_rate < 0.5 and results:
            performance_insight = {
                "type": "performance",
                "insight": "Low success rate in decision execution",
                "recommendation": "Review decision confidence thresholds",
                "data": {"success_rate": success_rate, "failures": failed}
            }
            self.learning_insights.append(performance_insight)
            await self._apply_learning(performance_insight)
            await self._self_improve_from_failures(failed)

        # Analyze decision patterns and adjust confidence thresholds
        if self.cycle_count % 5 == 0:
            await self._analyze_decision_patterns()

        # Store learning in memory
        self.memory.store(Memory(
            memory_type=MemoryType.META,
            content={
                "cycle": self.cycle_count,
                "results_analyzed": len(results),
                "success_rate": success_rate,
                "insights": self.learning_insights[-5:] if self.learning_insights else [],
                "success_rate_trend": self._decision_success_rate_history[-10:] if self._decision_success_rate_history else []
            },
            source_system="aurea",
            source_agent="learner",
            created_by="aurea",
            importance_score=0.6,
            tags=["learning", "meta"],
            tenant_id=self.tenant_id
        ))

        # Synthesize broader patterns every 10 cycles
        if self.cycle_count % 10 == 0:
            patterns = self.memory.synthesize(self.tenant_id, time_window=timedelta(hours=1))
            for pattern in patterns:
                logger.info(f"🧠 Pattern discovered: {pattern['description']}")
                # Store pattern in unified brain
                try:
                    self.brain.store(
                        key=f"pattern_{pattern.get('id', uuid.uuid4())}",
                        value=pattern,
                        category="pattern",
                        priority="high",
                        source="aurea_learning"
                    )
                except Exception as e:
                    logger.warning(f"Failed to store pattern in brain: {e}")

    async def _apply_learning(self, insight: dict):
        """Apply learning insights to improve performance - ALWAYS stores learning data"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()

                # ALWAYS store the learning insight to the database
                adjustment = "none"
                if insight["type"] == "performance" and insight["data"].get("success_rate", 1.0) < 0.5:
                    adjustment = "reduced_confidence_threshold"
                elif insight["type"] == "execution_cycle":
                    adjustment = "cycle_recorded"

                cur.execute("""
                    INSERT INTO ai_learning_insights
                    (tenant_id, insight_type, category, insight, confidence, impact_score, recommendations, applied, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    self.tenant_id,
                    insight["type"],
                    "aurea_ooda",  # category
                    insight.get("insight", f"Cycle insight: {insight['type']}"),
                    insight.get("data", {}).get("success_rate", 0.8),  # confidence
                    0.5,  # impact_score
                    json.dumps([adjustment]),  # recommendations as array
                    adjustment != "none",  # applied boolean
                    json.dumps(insight.get("data", {}))  # metadata
                ))
                conn.commit()
                logger.info(f"📚 Learning recorded: {insight['type']} - {adjustment}")

                if insight["type"] == "performance" and insight["data"].get("success_rate", 1.0) < 0.5:
                    # Adjust agent confidence thresholds for poor performers
                    if "agent_name" in insight["data"]:
                        cur.execute("""
                            UPDATE ai_agents
                            SET config = config || '{"confidence_threshold": 0.7}'::jsonb,
                                updated_at = NOW()
                            WHERE name = %s AND tenant_id = %s
                        """, (insight["data"]["agent_name"], self.tenant_id))

                    logger.info("📚 Learning applied: Adjusted confidence thresholds for low-performing agents")

                elif insight["type"] == "pattern" and insight.get("pattern_confidence", 0) > 0.8:
                    # Store high-confidence patterns for future decision making
                    cur.execute("""
                        INSERT INTO ai_decision_patterns (tenant_id, pattern_type, pattern_data, confidence, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT DO NOTHING
                    """, (
                        self.tenant_id,
                        insight.get("pattern_type", "general"),
                        json.dumps(insight["data"]),
                        insight.get("pattern_confidence", 0.8)
                    ))
                    logger.info(f"📚 Learning applied: Stored new decision pattern with {insight.get('pattern_confidence', 0.8):.0%} confidence")

                conn.commit()
                cur.close()

        except Exception as e:
            logger.warning(f"Could not apply learning insight: {e}")

    async def _self_heal(self):
        """
        Detect and fix system issues with MCP-powered auto-remediation.

        This is a KEY force multiplier - the system uses its own infrastructure
        tools (MCP Bridge) to heal itself, creating autonomous operations.

        FULL POWER MODE: Self-healing is now ACTIVE and will execute remediation.
        """
        # Force health check if not yet populated
        if not self.system_health:
            self.system_health = await self._check_system_health()
        if not self.system_health:
            return

        healing_actions = []
        mcp_actions = []  # Actions that require MCP Bridge

        # =========================================================================
        # RULE 1: Low overall health (<70) - Internal fixes first
        # =========================================================================
        if self.system_health.overall_score < 70:
            logger.warning(f"⚕️ System health low: {self.system_health.overall_score}")

            if self.system_health.error_rate > 0.1:
                healing_actions.append("restart_failed_agents")

            if self.system_health.memory_utilization > 0.9:
                healing_actions.append("consolidate_memory")

            if self.system_health.decision_backlog > 10:
                healing_actions.append("clear_decision_backlog")

        # =========================================================================
        # RULE 2: Critical error rate (>20%) - Trigger service restart via MCP
        # =========================================================================
        if self.system_health.error_rate > 0.2:
            logger.critical(f"🚨 Critical error rate: {self.system_health.error_rate*100:.1f}%")
            mcp_actions.append({
                "action": "mcp:render:restart_service",
                "reason": f"Critical error rate {self.system_health.error_rate*100:.1f}%",
                "params": {"service_id": "srv-d0ulv1idbo4c73apd4t0"}  # AI Agents service
            })

        # =========================================================================
        # RULE 3: Very low health (<50) - Scale up services
        # =========================================================================
        if self.system_health.overall_score < 50:
            logger.critical(f"🚨 Very low health: {self.system_health.overall_score}")
            mcp_actions.append({
                "action": "mcp:render:scale_service",
                "reason": f"Health score critical: {self.system_health.overall_score}",
                "params": {"num_instances": 2}
            })

        # =========================================================================
        # RULE 4: Performance score very low (<40) - Consider rollback
        # =========================================================================
        if self.system_health.performance_score < 40:
            logger.warning(f"⚠️ Performance degraded: {self.system_health.performance_score}")
            # Check if this started recently (after a deployment)
            # For now, log it for human review
            mcp_actions.append({
                "action": "mcp:supabase:execute_sql",
                "reason": "Log performance alert",
                "params": {
                    "query": f"""
                        INSERT INTO ai_system_alerts (alert_type, severity, message, component, created_at)
                        VALUES ('performance_degraded', 'high',
                                'Performance score: {self.system_health.performance_score}',
                                'aurea_orchestrator', NOW())
                    """
                }
            })

        # =========================================================================
        # RULE 5: Memory pressure - Notify and potentially scale
        # =========================================================================
        if self.system_health.memory_utilization > 0.85:
            logger.warning(f"⚠️ High memory: {self.system_health.memory_utilization*100:.1f}%")
            # First try internal consolidation
            healing_actions.append("consolidate_memory")
            # If still high, might need infrastructure action
            if self.system_health.memory_utilization > 0.95:
                mcp_actions.append({
                    "action": "mcp:render:restart_service",
                    "reason": f"Memory exhaustion: {self.system_health.memory_utilization*100:.1f}%",
                    "params": {"service_id": "srv-d0ulv1idbo4c73apd4t0"}
                })

        # =========================================================================
        # EXECUTE: Internal healing actions first, then MCP actions
        # =========================================================================
        for action in healing_actions:
            await self._execute_healing_action(action)

        for mcp_action in mcp_actions:
            logger.info(f"🔧 Executing MCP auto-remediation: {mcp_action['action']} - {mcp_action['reason']}")
            result = await self._execute_mcp_action(mcp_action["action"])
            # Log the remediation attempt - uses shared pool
            try:
                with _get_pooled_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO remediation_history
                        (incident_type, component, action_taken, success, recovery_time_seconds)
                        VALUES (%s, %s, %s, %s, 0)
                    """, (
                        mcp_action["reason"][:100],
                        "aurea_orchestrator",
                        mcp_action["action"],
                        result.get("success", False)
                    ))
                    conn.commit()
                    cur.close()
            except Exception as e:
                logger.error(f"Failed to log remediation: {e}")

    async def _execute_healing_action(self, action: str):
        """Execute a specific healing action"""
        logger.info(f"⚕️ Executing healing action: {action}")

        if action == "consolidate_memory":
            self.memory.consolidate(aggressive=True)
        elif action == "restart_failed_agents":
            # Restart failed agents by resetting their status - uses shared pool
            try:
                with _get_pooled_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE ai_agents SET status = 'active', updated_at = NOW()
                        WHERE status = 'failed' AND tenant_id = %s
                    """, (self.tenant_id,))
                    restarted = cur.rowcount
                    conn.commit()
                    cur.close()
                    logger.info(f"♻️ Restarted {restarted} failed agents")
            except Exception as e:
                logger.error(f"Failed to restart agents: {e}")
        elif action == "clear_decision_backlog":
            # Process backlogged decisions by marking stale ones as expired - uses shared pool
            try:
                with _get_pooled_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE aurea_decisions
                        SET execution_status = 'expired', updated_at = NOW()
                        WHERE execution_status = 'pending' AND created_at < NOW() - INTERVAL '24 hours'
                    """)
                    cleared = cur.rowcount
                    conn.commit()
                    cur.close()
                    logger.info(f"🧹 Cleared {cleared} stale decisions from backlog")
            except Exception as e:
                logger.error(f"Failed to clear decision backlog: {e}")
        elif action.startswith("mcp:"):
            # Execute MCP tool action (e.g., "mcp:render:restart_service")
            await self._execute_mcp_action(action)

    async def _execute_mcp_action(self, action: str) -> dict[str, Any]:
        """Execute an action via MCP Bridge - uses centralized MCPClient (345 tools)"""
        try:
            # Parse action: "mcp:server:tool" or "mcp:tool"
            parts = action.replace("mcp:", "").split(":")
            if len(parts) == 2:
                server_str, tool = parts
            else:
                tool = parts[0]
                server_str = self._infer_mcp_server(tool)

            logger.info(f"🔗 Executing MCP action via MCPClient: {server_str}/{tool}")

            # Convert server string to MCPServer enum
            server = self._get_mcp_server_enum(server_str)

            # Use centralized MCPClient with connection pooling and retry logic
            result: MCPToolResult = await self.mcp.execute_tool(
                server,
                tool,
                {"triggered_by": "aurea", "tenant_id": self.tenant_id}
            )

            if result.success:
                logger.info(f"✅ MCP action successful: {tool} ({result.duration_ms:.0f}ms)")
                return {"success": True, "result": result.result, "duration_ms": result.duration_ms}
            else:
                logger.warning(f"❌ MCP action failed: {result.error}")
                return {"success": False, "error": result.error}

        except Exception as e:
            logger.error(f"MCP execution error: {e}")
            return {"success": False, "error": str(e)}

    def _get_mcp_server_enum(self, server_str: str) -> MCPServer:
        """Convert server string to MCPServer enum"""
        server_map = {
            "render": MCPServer.RENDER,
            "render-mcp": MCPServer.RENDER,
            "vercel": MCPServer.VERCEL,
            "vercel-mcp": MCPServer.VERCEL,
            "supabase": MCPServer.SUPABASE,
            "supabase-mcp": MCPServer.SUPABASE,
            "github": MCPServer.GITHUB,
            "github-mcp": MCPServer.GITHUB,
            "docker": MCPServer.DOCKER,
            "docker-mcp": MCPServer.DOCKER,
            "stripe": MCPServer.STRIPE,
            "stripe-mcp": MCPServer.STRIPE,
            "openai": MCPServer.OPENAI,
            "anthropic": MCPServer.ANTHROPIC,
            "playwright": MCPServer.PLAYWRIGHT,
            "python": MCPServer.PYTHON,
            "python-executor": MCPServer.PYTHON,
        }
        return server_map.get(server_str.lower(), MCPServer.RENDER)

    def _infer_mcp_server(self, tool: str) -> str:
        """Infer which MCP server to use based on tool name"""
        tool_lower = tool.lower()
        if any(k in tool_lower for k in ["render", "service", "deploy", "restart"]):
            return "render"
        elif any(k in tool_lower for k in ["vercel", "preview", "domain", "deployment"]):
            return "vercel"
        elif any(k in tool_lower for k in ["supabase", "sql", "database", "query", "insert"]):
            return "supabase"
        elif any(k in tool_lower for k in ["github", "repo", "branch", "pr", "issue", "workflow"]):
            return "github"
        elif any(k in tool_lower for k in ["docker", "container", "image", "kubernetes"]):
            return "docker"
        elif any(k in tool_lower for k in ["stripe", "payment", "invoice", "subscription", "customer"]):
            return "stripe"
        elif any(k in tool_lower for k in ["playwright", "browser", "screenshot"]):
            return "playwright"
        elif any(k in tool_lower for k in ["openai", "gpt", "embedding"]):
            return "openai"
        elif any(k in tool_lower for k in ["anthropic", "claude"]):
            return "anthropic"
        else:
            return "render"  # Default to Render for infrastructure

    async def execute_external_tool(self, server: str, tool: str, params: dict[str, Any] = None) -> dict[str, Any]:
        """Public method for AUREA to execute any MCP tool - uses centralized MCPClient"""
        try:
            server_enum = self._get_mcp_server_enum(server)
            result: MCPToolResult = await self.mcp.execute_tool(server_enum, tool, params or {})

            return {
                "success": result.success,
                "result": result.result,
                "error": result.error,
                "duration_ms": result.duration_ms,
                "execution_id": result.execution_id
            }
        except Exception as e:
            logger.error(f"External tool execution error: {e}")
            return {"success": False, "error": str(e)}

    async def _check_system_health(self) -> SystemHealth:
        """Check overall system health - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get agent stats
                agent_stats = self.activation_system.get_agent_stats()

                # Get memory stats
                memory_stats = self.memory.get_stats(self.tenant_id)

                # Get error rate from logs
                cur.execute("""
                SELECT
                    COUNT(CASE WHEN status = 'failed' THEN 1 END)::float /
                    NULLIF(COUNT(*), 0) as error_rate
                FROM agent_activation_log
                WHERE created_at > NOW() - INTERVAL '1 hour'
                AND tenant_id = %s
                """, (self.tenant_id,))
                error_rate = cur.fetchone()['error_rate'] or 0

                # Get decision backlog
                cur.execute("""
                SELECT COUNT(*) as backlog
                FROM aurea_decisions
                WHERE execution_status = 'pending'
                  AND created_at > NOW() - INTERVAL '24 hours'
                  AND tenant_id = %s
                """, (self.tenant_id,))
                decision_backlog = cur.fetchone()['backlog']
                cur.close()

            # Calculate performance score based on error rate and backlog
            # Performance = 100 - (Error Rate % * 0.5) - (Backlog Penalty)
            performance_score = max(0, 100 - (error_rate * 100 * 0.5) - (min(decision_backlog, 50) * 0.5))

            # Build health_components from gathered metrics
            health_components = {
                "agents": min(100, (agent_stats.get("enabled_agents", 0) / max(agent_stats.get("total_agents", 1), 1)) * 100),
                "memory": min(100, 100 - (memory_stats.get("total_memories", 0) / 1000000) * 100) if memory_stats.get("total_memories", 0) < 1000000 else 10,
                "errors": max(0, 100 - (error_rate * 100)),
                "backlog": max(0, 100 - (min(decision_backlog, 100) * 1)),
                "performance": performance_score
            }

            overall_score = sum(health_components.values()) / len(health_components) if health_components else 0

            self.system_health = SystemHealth(
                overall_score=overall_score,
                component_health=health_components,
                active_agents=agent_stats.get("enabled_agents", 0),
                memory_utilization=memory_stats.get("total_memories", 0) / 1000000,
                decision_backlog=decision_backlog,
                error_rate=error_rate,
                performance_score=performance_score,
                alerts=[]
            )

            self.last_health_check = datetime.now()
            return self.system_health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                overall_score=0,
                component_health={},
                active_agents=0,
                memory_utilization=0,
                decision_backlog=0,
                error_rate=1.0,
                performance_score=0,
                alerts=[f"Health check failed: {str(e)}"]
            )

    async def _request_human_approval(self, decision: Decision) -> bool:
        """Request human approval for a decision - uses shared pool"""
        logger.info(f"🤔 Human approval requested for: {decision.description}")

        # Check for test mode override
        if os.getenv("AUREA_TEST_MODE", "false").lower() == "true":
            logger.info("⚠️ Auto-approving decision in TEST MODE")
            return True

        # Store pending decision - uses shared pool
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()
                try:
                    # Update decision status
                    cur.execute("""
                    UPDATE aurea_decisions
                    SET execution_status = 'pending'
                    WHERE id = %s
                    """, (decision.id,))

                    # Create notification if table exists
                    try:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS task_notifications (
                                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                title TEXT NOT NULL,
                                message TEXT NOT NULL,
                                type TEXT NOT NULL,
                                status TEXT DEFAULT 'unread',
                                created_at TIMESTAMP DEFAULT NOW(),
                                tenant_id UUID
                            )
                        """)

                        cur.execute("""
                            INSERT INTO task_notifications (title, message, type, tenant_id)
                            VALUES (%s, %s, %s, %s)
                        """, (
                            f"Approval Required: {decision.type.value}",
                            f"Decision requires approval: {decision.description}",
                            "approval_request",
                            self.tenant_id
                        ))
                    except Exception as notify_err:
                        logger.warning(f"Could not create notification: {notify_err}")

                    conn.commit()
                    logger.info(f"Decision {decision.id} queued for approval")
                finally:
                    cur.close()

        except Exception as e:
            logger.error(f"Failed to queue decision for approval: {e}")
            return False

        return False

    def _log_decision(self, decision: Decision) -> Optional[str]:
        """Log decision to database and return the database UUID - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                INSERT INTO aurea_decisions
                (decision_type, description, confidence, impact_assessment,
                 recommended_action, alternatives, requires_human_approval,
                 execution_status, context, tenant_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """, (
                    decision.type.value,
                    decision.description,
                    decision.confidence,
                    decision.impact_assessment,
                    decision.recommended_action,
                    Json(decision.alternatives),
                    decision.requires_human_approval,
                    "pending",
                    Json(decision.context),
                    self.tenant_id
                ))

                result = cur.fetchone()
                db_id = str(result[0]) if result else None

                # Store the database ID on the decision for later use
                decision.db_id = db_id

                # ALSO persist to ai_decisions table for visibility
                # This table is the central AI decisions tracking table
                self._persist_to_ai_decisions(decision, conn, cur)

                # Log to unified brain
                self.memory.log_to_brain("aurea_orchestrator", "decision_made", {
                    "decision_id": decision.id,
                    "db_id": db_id,
                    "type": decision.type.value,
                    "description": decision.description,
                    "confidence": decision.confidence
                })

                conn.commit()
                cur.close()

                return db_id

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return None

    def _persist_to_ai_decisions(self, decision: Decision, conn, cur) -> None:
        """
        Persist decision to the ai_decisions table for centralized visibility.
        This ensures all AUREA decisions are tracked in the main AI decisions table.

        Schema: id, agent_id, decision_type, input_data, output_data, confidence, timestamp
        """
        try:
            # Build comprehensive input_data with full context
            input_data = {
                "decision_id": decision.id,
                "context": json_safe_serialize(decision.context),
                "cycle_count": self.cycle_count,
                "autonomy_level": self.autonomy_level.value,
                "tenant_id": str(self.tenant_id) if self.tenant_id else None
            }

            # Build output_data with reasoning (WHY the decision was made)
            output_data = {
                "description": decision.description,
                "reasoning": self._generate_decision_reasoning(decision),
                "recommended_action": decision.recommended_action,
                "alternatives": decision.alternatives,
                "impact_assessment": decision.impact_assessment,
                "requires_human_approval": decision.requires_human_approval,
                "deadline": decision.deadline.isoformat() if decision.deadline else None,
                "db_id": decision.db_id
            }

            cur.execute("""
                INSERT INTO ai_decisions
                (agent_id, decision_type, input_data, output_data, confidence, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                "AUREA",  # agent_id
                decision.type.value,  # decision_type
                Json(input_data),  # input_data
                Json(output_data),  # output_data
                decision.confidence  # confidence
            ))

            logger.debug(f"📊 Decision {decision.id} also logged to ai_decisions table")

        except Exception as e:
            # Don't fail the main decision logging if ai_decisions insert fails
            logger.warning(f"Failed to persist to ai_decisions: {e}")

    def _generate_decision_reasoning(self, decision: Decision) -> str:
        """
        Generate human-readable reasoning explaining WHY this decision was made.
        This provides transparency into AUREA's decision-making process.
        """
        reasoning_parts = []

        # Base reasoning from decision type
        type_reasoning = {
            DecisionType.FINANCIAL: "Financial priority detected requiring immediate attention to revenue/cash flow",
            DecisionType.OPERATIONAL: "Operational efficiency issue identified that impacts daily business operations",
            DecisionType.TACTICAL: "Short-term optimization opportunity identified for immediate improvement",
            DecisionType.EMERGENCY: "Critical situation requiring immediate emergency response",
            DecisionType.CUSTOMER: "Customer-related risk or opportunity detected requiring action",
            DecisionType.TECHNICAL: "Technical issue affecting system performance or user experience",
            DecisionType.STRATEGIC: "Strategic opportunity identified with significant business impact",
            DecisionType.LEARNING: "Self-improvement opportunity identified for system enhancement"
        }
        reasoning_parts.append(type_reasoning.get(decision.type, "Decision criteria met based on system analysis"))

        # Add context-specific reasoning
        context = decision.context or {}

        # Priority-based reasoning
        if "priority" in context:
            priority = context["priority"]
            if priority.get("urgency") == "high":
                reasoning_parts.append(f"HIGH URGENCY: {priority.get('description', 'Immediate action required')}")
            elif priority.get("urgency") == "critical":
                reasoning_parts.append(f"CRITICAL: {priority.get('description', 'System stability at risk')}")

        # Risk-based reasoning
        if "risk" in context:
            risk = context["risk"]
            reasoning_parts.append(f"Risk identified: {risk.get('description', 'Potential negative outcome detected')}")

        # Opportunity-based reasoning
        if "opportunity" in context:
            opportunity = context["opportunity"]
            if "potential_value" in opportunity:
                reasoning_parts.append(f"Revenue opportunity: ${opportunity['potential_value']:,.2f} potential value")

        # Confidence level reasoning
        if decision.confidence >= 0.90:
            reasoning_parts.append("High confidence based on strong historical patterns and clear indicators")
        elif decision.confidence >= 0.75:
            reasoning_parts.append("Good confidence based on consistent data signals")
        else:
            reasoning_parts.append("Moderate confidence - monitoring recommended")

        # Impact reasoning
        if decision.impact_assessment:
            reasoning_parts.append(f"Expected impact: {decision.impact_assessment}")

        return " | ".join(reasoning_parts)

    async def _handle_orchestration_error(self, error: Exception):
        """Handle errors in orchestration loop"""
        logger.error(f"Orchestration error: {error}")

        # Store error in memory
        self.memory.store(Memory(
            memory_type=MemoryType.META,
            content={
                "error": str(error),
                "cycle": self.cycle_count,
                "timestamp": datetime.now().isoformat(),
                "recovery_action": "restart_cycle"
            },
            source_system="aurea",
            source_agent="error_handler",
            created_by="aurea",
            importance_score=0.9,
            tags=["error", "orchestration"],
            tenant_id=self.tenant_id
        ))

    def _calculate_sleep_time(self, observations: int, decisions: int) -> int:
        """Calculate adaptive sleep time based on activity - OPTIMIZED for stability"""
        # Minimum 60s between cycles to prevent resource exhaustion
        base_interval = int(os.getenv("AUREA_CYCLE_INTERVAL", "60"))
        if observations == 0 and decisions == 0:
            return base_interval * 2  # Low activity: 120s
        elif observations > 5 or decisions > 3:
            return base_interval  # High activity: 60s
        else:
            return int(base_interval * 1.5)  # Normal activity: 90s

    async def _detect_patterns(self) -> list[dict[str, Any]]:
        """Detect patterns from historical data and system behavior"""
        patterns = []

        try:
            # Analyze recent cycle metrics for patterns
            if len(self.cycle_metrics_history) >= 5:
                recent_metrics = self.cycle_metrics_history[-10:]

                # Pattern: Declining success rate
                success_rates = [m.actions_successful / max(m.actions_executed, 1) for m in recent_metrics if m.actions_executed > 0]
                if len(success_rates) >= 3:
                    if success_rates[-1] < success_rates[0] * 0.7:
                        patterns.append({
                            "type": "declining_success",
                            "description": "Decision execution success rate declining",
                            "confidence": 0.8,
                            "data": {"success_rates": success_rates}
                        })

                # Pattern: Consistent high performance
                if all(sr > 0.9 for sr in success_rates[-5:] if sr):
                    patterns.append({
                        "type": "high_performance",
                        "description": "Consistently high execution success rate",
                        "confidence": 0.9,
                        "data": {"success_rates": success_rates}
                    })

                # Pattern: Increasing cycle time
                cycle_times = [m.cycle_duration_seconds for m in recent_metrics]
                if cycle_times[-1] > cycle_times[0] * 1.5:
                    patterns.append({
                        "type": "increasing_latency",
                        "description": "OODA cycle duration increasing",
                        "confidence": 0.75,
                        "data": {"cycle_times": cycle_times}
                    })

            # Store detected patterns - uses shared pool
            for pattern in patterns:
                try:
                    with _get_pooled_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("""
                            INSERT INTO aurea_patterns (pattern_type, pattern_description, confidence,
                                                         frequency, impact_score, pattern_data, tenant_id)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            pattern["type"],
                            pattern["description"],
                            pattern["confidence"],
                            1,
                            0.7,
                            Json(pattern.get("data", {})),
                            self.tenant_id
                        ))
                        conn.commit()
                        cur.close()
                except Exception as e:
                    logger.warning(f"Failed to store pattern: {e}")

        except Exception as e:
            logger.error(f"Pattern detection error: {e}")

        return patterns

    async def _check_goal_progress(self) -> list[dict[str, Any]]:
        """Check progress on autonomous goals and update status"""
        goal_updates = []

        for goal in self.autonomous_goals:
            if goal.status != "active":
                continue

            try:
                # Recalculate current value based on goal type
                current_value = await self._measure_goal_metric(goal.target_metric)
                old_progress = goal.progress

                # Update progress
                if goal.target_value != goal.current_value:
                    goal.progress = min(100, max(0,
                        ((current_value - goal.current_value) / (goal.target_value - goal.current_value)) * 100
                    ))

                goal.current_value = current_value

                # Check if goal achieved
                if goal.progress >= 100:
                    goal.status = "achieved"
                    goal_updates.append({
                        "goal_id": goal.id,
                        "status": "achieved",
                        "progress": goal.progress
                    })
                    logger.info(f"🎯 Goal achieved: {goal.description}")

                # Check if deadline passed
                elif datetime.now() > goal.deadline:
                    goal.status = "failed"
                    goal_updates.append({
                        "goal_id": goal.id,
                        "status": "failed",
                        "progress": goal.progress
                    })

                # Update database
                await self._update_goal_in_db(goal)

            except Exception as e:
                logger.error(f"Goal progress check error: {e}")

        return goal_updates

    async def _measure_goal_metric(self, metric_name: str) -> float:
        """Measure a specific metric value"""
        try:
            if metric_name == "error_rate":
                if self.system_health:
                    return self.system_health.error_rate
            elif metric_name == "success_rate":
                if self._decision_success_rate_history:
                    return sum(self._decision_success_rate_history[-10:]) / len(self._decision_success_rate_history[-10:])
            elif metric_name == "cycle_time":
                if self.cycle_metrics_history:
                    return self.cycle_metrics_history[-1].cycle_duration_seconds
            elif metric_name == "health_score":
                if self.system_health:
                    return self.system_health.overall_score
        except Exception as e:
            logger.error(f"Metric measurement error: {e}")

        return 0.0

    async def _update_goal_in_db(self, goal: AutonomousGoal):
        """Update goal in database - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE aurea_autonomous_goals
                    SET current_value = %s, progress = %s, status = %s, updated_at = NOW(),
                        achieved_at = CASE WHEN %s = 'achieved' THEN NOW() ELSE achieved_at END
                    WHERE id = %s
                """, (goal.current_value, goal.progress, goal.status, goal.status, goal.id))
                conn.commit()
                cur.close()
        except Exception as e:
            logger.error(f"Goal update error: {e}")

    async def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends and detect anomalies"""
        trends = {
            "anomalies": [],
            "improvements": [],
            "degradations": []
        }

        try:
            # Analyze success rate trend
            if len(self._decision_success_rate_history) >= 10:
                recent = self._decision_success_rate_history[-5:]
                older = self._decision_success_rate_history[-10:-5]

                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)

                if recent_avg > older_avg * 1.2:
                    trends["improvements"].append("Decision success rate improving")
                elif recent_avg < older_avg * 0.8:
                    trends["degradations"].append("Decision success rate declining")
                    trends["anomalies"].append({
                        "type": "performance_degradation",
                        "metric": "success_rate",
                        "severity": "high"
                    })

            # Analyze cycle time trends
            if len(self.cycle_metrics_history) >= 10:
                recent_times = [m.cycle_duration_seconds for m in self.cycle_metrics_history[-5:]]
                older_times = [m.cycle_duration_seconds for m in self.cycle_metrics_history[-10:-5]]

                if sum(recent_times) / len(recent_times) > sum(older_times) / len(older_times) * 1.3:
                    trends["anomalies"].append({
                        "type": "latency_increase",
                        "metric": "cycle_time",
                        "severity": "medium"
                    })

        except Exception as e:
            logger.error(f"Trend analysis error: {e}")

        return trends

    async def _set_autonomous_goals(self, observations: list[dict]) -> list[AutonomousGoal]:
        """Set autonomous goals based on system state and observations"""
        new_goals = []

        try:
            # Goal: Improve success rate if below threshold
            if self._decision_success_rate_history:
                current_rate = sum(self._decision_success_rate_history[-5:]) / len(self._decision_success_rate_history[-5:])
                if current_rate < 0.85:
                    goal = AutonomousGoal(
                        id=str(uuid.uuid4()),
                        goal_type="performance",
                        description="Improve decision execution success rate to 90%",
                        target_metric="success_rate",
                        current_value=current_rate,
                        target_value=0.90,
                        deadline=datetime.now() + timedelta(hours=24),
                        priority=8,
                        created_at=datetime.now(),
                        status="active",
                        progress=0.0
                    )
                    new_goals.append(goal)
                    self.autonomous_goals.append(goal)
                    await self._store_goal_in_db(goal)

            # Goal: Reduce error rate if high
            if self.system_health and self.system_health.error_rate > 0.1:
                goal = AutonomousGoal(
                    id=str(uuid.uuid4()),
                    goal_type="quality",
                    description="Reduce system error rate to below 5%",
                    target_metric="error_rate",
                    current_value=self.system_health.error_rate,
                    target_value=0.05,
                    deadline=datetime.now() + timedelta(hours=12),
                    priority=9,
                    created_at=datetime.now(),
                    status="active",
                    progress=0.0
                )
                new_goals.append(goal)
                self.autonomous_goals.append(goal)
                await self._store_goal_in_db(goal)

            # Goal: Improve response time if slow
            if self.cycle_metrics_history and self.cycle_metrics_history[-1].cycle_duration_seconds > 30:
                goal = AutonomousGoal(
                    id=str(uuid.uuid4()),
                    goal_type="efficiency",
                    description="Reduce OODA cycle time to under 20 seconds",
                    target_metric="cycle_time",
                    current_value=self.cycle_metrics_history[-1].cycle_duration_seconds,
                    target_value=20.0,
                    deadline=datetime.now() + timedelta(hours=6),
                    priority=7,
                    created_at=datetime.now(),
                    status="active",
                    progress=0.0
                )
                new_goals.append(goal)
                self.autonomous_goals.append(goal)
                await self._store_goal_in_db(goal)

        except Exception as e:
            logger.error(f"Goal setting error: {e}")

        return new_goals

    async def _store_goal_in_db(self, goal: AutonomousGoal):
        """Store new goal in database - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO aurea_autonomous_goals
                    (id, goal_type, description, target_metric, current_value, target_value,
                     deadline, priority, status, progress, tenant_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    goal.id, goal.goal_type, goal.description, goal.target_metric,
                    goal.current_value, goal.target_value, goal.deadline, goal.priority,
                    goal.status, goal.progress, self.tenant_id
                ))
                conn.commit()
                cur.close()
            logger.info(f"🎯 New autonomous goal set: {goal.description}")
        except Exception as e:
            logger.error(f"Goal storage error: {e}")

    async def _self_improve_from_failures(self, failures: list[dict[str, Any]]):
        """Analyze failures and adjust system parameters to improve"""
        try:
            for failure in failures:
                error = failure.get("error", "")
                decision_id = failure.get("decision_id", "")

                # Analyze failure patterns
                if "timeout" in error.lower():
                    logger.info("🔧 Self-improvement: Detected timeout pattern, adjusting timeouts")
                    # Could adjust timeout parameters
                elif "permission" in error.lower() or "denied" in error.lower():
                    logger.info("🔧 Self-improvement: Detected permission issue, reviewing access controls")
                elif "database" in error.lower() or "connection" in error.lower():
                    logger.info("🔧 Self-improvement: Detected DB issue, triggering connection pool refresh")

                # Store self-improvement action in brain
                self.brain.store(
                    key=f"self_improvement_{uuid.uuid4()}",
                    value={
                        "failure": error,
                        "decision_id": decision_id,
                        "improvement_action": "parameter_adjustment",
                        "timestamp": datetime.now().isoformat()
                    },
                    category="self_improvement",
                    priority="high",
                    source="aurea_learning"
                )

        except Exception as e:
            logger.error(f"Self-improvement error: {e}")

    async def _analyze_decision_patterns(self):
        """Analyze decision patterns and adjust confidence thresholds - uses shared pool"""
        try:
            # Query recent decision outcomes
            with _get_pooled_connection() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                cur.execute("""
                    SELECT decision_type, confidence, execution_status,
                           COUNT(*) as count,
                           AVG(confidence) as avg_confidence
                    FROM aurea_decisions
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                      AND tenant_id = %s
                    GROUP BY decision_type, confidence, execution_status
                    ORDER BY decision_type, confidence DESC
                """, (self.tenant_id,))

                patterns = cur.fetchall()
                cur.close()

            # Analyze patterns and adjust thresholds
            for pattern in patterns:
                decision_type = pattern['decision_type']
                status = pattern['execution_status']
                avg_conf = pattern['avg_confidence']

                # If high confidence decisions are failing, raise threshold
                if status == 'failed' and avg_conf > 0.7:
                    logger.info(f"🔧 Adjusting confidence threshold for {decision_type} decisions")
                    # Store adjustment in brain
                    self.brain.store(
                        key=f"threshold_adjustment_{decision_type}",
                        value={
                            "decision_type": decision_type,
                            "old_threshold": avg_conf,
                            "adjustment": "increase",
                            "reason": "high_failure_rate"
                        },
                        category="self_improvement",
                        priority="high",
                        source="aurea_pattern_analysis"
                    )

        except Exception as e:
            logger.error(f"Decision pattern analysis error: {e}")

    async def _store_cycle_metrics(self, metrics: CycleMetrics):
        """Store cycle metrics in database - uses shared pool"""
        try:
            with _get_pooled_connection() as conn:
                cur = conn.cursor()

                cur.execute("""
                    INSERT INTO aurea_cycle_metrics
                    (cycle_number, timestamp, observations_count, decisions_count,
                     actions_executed, actions_successful, actions_failed, cycle_duration_seconds,
                     learning_insights_generated, health_score, autonomy_level,
                     patterns_detected, goals_achieved, goals_set, tenant_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    metrics.cycle_number, metrics.timestamp, metrics.observations_count,
                    metrics.decisions_count, metrics.actions_executed, metrics.actions_successful,
                    metrics.actions_failed, metrics.cycle_duration_seconds,
                    metrics.learning_insights_generated, metrics.health_score,
                    metrics.autonomy_level, Json(metrics.patterns_detected),
                    metrics.goals_achieved, metrics.goals_set, self.tenant_id
                ))

                conn.commit()
                cur.close()

            # Also store in unified brain (convert datetime to string for JSON)
            metrics_dict = asdict(metrics)
            if 'timestamp' in metrics_dict and hasattr(metrics_dict['timestamp'], 'isoformat'):
                metrics_dict['timestamp'] = metrics_dict['timestamp'].isoformat()
            self.brain.store(
                key=f"cycle_metrics_{metrics.cycle_number}",
                value=metrics_dict,
                category="aurea_metrics",
                priority="medium",
                source="aurea_orchestrator"
            )

        except Exception as e:
            logger.error(f"Cycle metrics storage error: {e}")

    def set_autonomy_level(self, level: AutonomyLevel):
        """Adjust autonomy level"""
        self.autonomy_level = level
        logger.info(f"🎚️ Autonomy level set to: {level.name} ({level.value}%)")

    def stop(self):
        """Stop orchestration"""
        self.running = False
        logger.info("🛑 AUREA orchestration stopped")

    def get_status(self) -> dict:
        """Get current AUREA status with comprehensive metrics"""
        # Calculate current averages
        recent_success_rate = 0.0
        if self._decision_success_rate_history:
            recent_success_rate = sum(self._decision_success_rate_history[-10:]) / len(self._decision_success_rate_history[-10:])

        recent_cycle_time = 0.0
        if self.cycle_metrics_history:
            recent_cycle_time = sum(m.cycle_duration_seconds for m in self.cycle_metrics_history[-10:]) / len(self.cycle_metrics_history[-10:])

        active_goals = [g for g in self.autonomous_goals if g.status == "active"]
        achieved_goals = [g for g in self.autonomous_goals if g.status == "achieved"]

        return {
            "running": self.running,
            "autonomy_level": self.autonomy_level.name,
            "autonomy_value": self.autonomy_level.value,
            "cycles_completed": self.cycle_count,
            "decisions_made": self.decisions_made,
            "system_health": asdict(self.system_health) if self.system_health else None,
            "learning_insights": len(self.learning_insights),
            "last_health_check": self.last_health_check.isoformat(),
            "performance_metrics": {
                "recent_success_rate": recent_success_rate,
                "recent_cycle_time_avg": recent_cycle_time,
                "total_cycle_metrics": len(self.cycle_metrics_history),
                "success_rate_trend": self._decision_success_rate_history[-10:] if self._decision_success_rate_history else []
            },
            "autonomous_goals": {
                "active": len(active_goals),
                "achieved": len(achieved_goals),
                "total": len(self.autonomous_goals),
                "active_goals_list": [
                    {
                        "description": g.description,
                        "progress": g.progress,
                        "target_metric": g.target_metric,
                        "priority": g.priority
                    }
                    for g in active_goals[:5]
                ]
            },
            "learning": {
                "total_insights": len(self.learning_insights),
                "recent_patterns": len(self.pattern_history),
                "self_improvement_actions": len([i for i in self.learning_insights if i.get("type") == "self_improvement"])
            }
        }


# Global AUREA instance
aurea_instance = None

def get_aurea(tenant_id: Optional[str] = None) -> AUREA:
    """Get or create the singleton AUREA instance"""
    global aurea_instance
    if aurea_instance is None:
        if not tenant_id:
             raise ValueError("Tenant ID required for initial AUREA instantiation")
        aurea_instance = AUREA(tenant_id=tenant_id)
    return aurea_instance


async def run_aurea(tenant_id: str):
    """Run AUREA orchestration"""
    aurea = get_aurea(tenant_id)
    await aurea.orchestrate()


if __name__ == "__main__":
    # Start AUREA
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     🧠 AUREA - Autonomous Universal Resource            ║
    ║              & Execution Assistant                      ║
    ║                                                          ║
    ║     The Master AI Orchestrator for BrainOps             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝

    Starting autonomous orchestration...
    """)

    # Test Tenant ID for standalone run
    TEST_TENANT_ID = "test-tenant-id"

    # Run AUREA
    try:
        asyncio.run(run_aurea(TEST_TENANT_ID))
    except KeyboardInterrupt:
        print("\n🛑 AUREA shutdown requested")
        aurea = get_aurea(TEST_TENANT_ID)
        aurea.stop()
        print("✅ AUREA stopped gracefully")
