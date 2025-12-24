#!/usr/bin/env python3
"""
AUREA - Autonomous Universal Resource & Execution Assistant
The Master Orchestration Brain for BrainOps AI OS
Coordinates all 59 agents to work as one unified intelligence
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import re
from decimal import Decimal
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from unified_memory_manager import get_memory_manager, Memory, MemoryType
from agent_activation_system import (
    get_activation_system, BusinessEventType, AgentActivationSystem
)
from ai_core import RealAICore
from ai_board_governance import get_ai_board, Proposal, ProposalType
from ai_self_awareness import get_self_aware_ai, SelfAwareAI
from revenue_generation_system import get_revenue_system, AutonomousRevenueSystem
from ai_knowledge_graph import get_knowledge_graph, AIKnowledgeGraph
from mcp_integration import get_mcp_client, MCPClient, MCPServer, MCPToolResult
import aiohttp
import warnings
warnings.filterwarnings('ignore')


def json_safe_serialize(obj: Any) -> Any:
    """Recursively convert datetime/Decimal objects to JSON-serializable types"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: json_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(json_safe_serialize(item) for item in obj)
    return obj


# MCP Bridge Configuration
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = os.getenv("MCP_API_KEY") or os.getenv("BRAINOPS_API_KEY") or "brainops_mcp_2025"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AUREA')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', '<DB_PASSWORD_REDACTED>'),
    'port': int(os.getenv('DB_PORT', 5432))
}


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
    alternatives: List[str]
    requires_human_approval: bool
    deadline: Optional[datetime]
    context: Dict[str, Any]


@dataclass
class SystemHealth:
    """Overall system health metrics"""
    overall_score: float  # 0-100
    component_health: Dict[str, float]
    active_agents: int
    memory_utilization: float
    decision_backlog: int
    error_rate: float
    performance_score: float
    alerts: List[str]


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
        self.running = False
        self.cycle_count = 0
        self.decisions_made = 0
        self.last_health_check = datetime.now()
        self.system_health = None
        self.decision_queue = asyncio.Queue()
        self.learning_insights = []
        self.confidence_thresholds = self._default_confidence_thresholds()
        self._last_observation_bundle: Dict[str, Any] = {}
        self._last_orientation_bundle: Dict[str, Any] = {}
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
        except Exception:
            return None

    @property
    def mcp(self) -> MCPClient:
        """Get singleton MCPClient - the centralized MCP Bridge integration"""
        if not hasattr(self, '_mcp_client') or self._mcp_client is None:
            self._mcp_client = get_mcp_client()
        return self._mcp_client

    async def _async_fetch(self, query: str, *args) -> List[Dict]:
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

    async def _async_fetchrow(self, query: str, *args) -> Optional[Dict]:
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

    def _sync_fetch(self, query: str, *args) -> List[Dict]:
        """Sync fallback for fetch"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, args if args else None)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [dict(r) for r in rows] if rows else []
        except Exception as e:
            logger.error(f"Sync fetch failed: {e}")
            return []

    def _sync_fetchrow(self, query: str, *args) -> Optional[Dict]:
        """Sync fallback for fetchrow"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, args if args else None)
            row = cur.fetchone()
            cur.close()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Sync fetchrow failed: {e}")
            return None

    def _sync_execute(self, query: str, *args) -> bool:
        """Sync fallback for execute"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute(query, args if args else None)
            conn.commit()
            cur.close()
            conn.close()
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

    def _default_confidence_thresholds(self) -> Dict[int, float]:
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
        except Exception:
            match = re.search(r"(\\{.*\\}|\\[.*\\])", s, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    return {}
        return {}

    def _db_connect(self):
        return psycopg2.connect(**DB_CONFIG)

    def _db_fetchall(self, query: str, params: Tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
        try:
            conn = self._db_connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"DB fetchall failed: {e}")
            return []

    def _db_fetchone(self, query: str, params: Tuple[Any, ...] = ()) -> Optional[Dict[str, Any]]:
        try:
            conn = self._db_connect()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)
            row = cur.fetchone()
            cur.close()
            conn.close()
            return dict(row) if row else None
        except Exception as e:
            logger.debug(f"DB fetchone failed: {e}")
            return None

    def _db_execute(self, query: str, params: Tuple[Any, ...] = ()) -> bool:
        try:
            conn = self._db_connect()
            cur = conn.cursor()
            cur.execute(query, params)
            conn.commit()
            cur.close()
            conn.close()
            return True
        except Exception as e:
            logger.debug(f"DB execute failed: {e}")
            return False

    def _truncate_text(self, value: Any, max_len: int = 800) -> str:
        s = "" if value is None else str(value)
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + "..."

    def _compact_rows(
        self,
        rows: List[Dict[str, Any]],
        *,
        max_rows: int = 30,
        max_field_len: int = 600
    ) -> List[Dict[str, Any]]:
        compacted: List[Dict[str, Any]] = []
        for row in (rows or [])[:max_rows]:
            cleaned: Dict[str, Any] = {}
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
        except Exception:
            return None

    def _extract_keywords(self, signals: List[Dict[str, Any]]) -> List[str]:
        keywords: List[str] = []
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
        out: List[str] = []
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

    def _store_state_snapshot(self, state_type: str, state_data: Dict[str, Any]):
        """Persist OODA snapshots for audit/debug (best-effort)."""
        try:
            self._db_execute(
                """
                INSERT INTO aurea_state (state_type, state_data, cycle_number, tenant_id)
                VALUES (%s, %s, %s, %s)
                """,
                (state_type, Json(state_data), self.cycle_count, self.tenant_id),
            )
        except Exception:
            # best-effort only
            return

    def _init_database(self):
        """Initialize AUREA's database tables"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
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
            """)

            conn.commit()
            cur.close()
            conn.close()

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

                # Phase 2: Orient - Analyze situation and context
                context = await self._orient(observations)

                # Phase 3: Decide - Make decisions based on context
                decisions = await self._decide(context)

                # Phase 4: Act - Execute decisions through agents
                results = await self._act(decisions)

                # Phase 5: Learn - Analyze results and improve
                await self._learn(results)

                # Phase 6: Heal - Fix any issues detected
                await self._self_heal()

                # Calculate cycle time
                cycle_time = (datetime.now() - cycle_start).total_seconds()

                # Store cycle in memory
                self.memory.store(Memory(
                    memory_type=MemoryType.PROCEDURAL,
                    content={
                        "cycle": self.cycle_count,
                        "signals_observed": len(observations),
                        "decisions": len(decisions),
                        "actions_executed": len(results),
                        "cycle_time_seconds": cycle_time,
                        "autonomy_level": self.autonomy_level.value
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

    async def _observe(self) -> List[Dict[str, Any]]:
        """Observe the environment and gather all relevant data (ASYNC)"""
        observations = []

        try:
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

        return observations

    async def _orient(self, observations: List[Dict]) -> Dict[str, Any]:
        """Analyze observations and build context"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self.cycle_count,
            "autonomy_level": self.autonomy_level.value,
            "observations": observations,
            "priorities": [],
            "risks": [],
            "opportunities": []
        }

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

    async def _decide(self, context: Dict[str, Any]) -> List[Decision]:
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

        # Log decisions
        for decision in decisions:
            self._log_decision(decision)

        return decisions

    async def _act(self, decisions: List[Decision]) -> List[Dict[str, Any]]:
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

                # Execute the decision (in verification mode - no real outreach)
                result = await self._execute_decision(decision)
                result["verification_mode"] = True  # Flag that this was verification only
                results.append(result)

                # Update decision status in database (use db_id if available, else fallback)
                decision_db_id = getattr(decision, 'db_id', None) or decision.id
                await self._update_decision_status(decision_db_id, "completed", result)

                # Store execution in memory
                self.memory.store(Memory(
                    memory_type=MemoryType.EPISODIC,
                    content={
                        "decision": asdict(decision),
                        "execution_result": result,
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
                decision_db_id = getattr(decision, 'db_id', None) or decision.id
                await self._update_decision_status(decision_db_id, "failed", {"error": str(e)})
                results.append({
                    "decision_id": decision.id,
                    "status": "failed",
                    "error": str(e)
                })

        return results

    async def _update_decision_status(self, decision_id: str, status: str, result: Dict = None):
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
                SET execution_status = $1,
                    execution_result = $2::jsonb,
                    executed_at = NOW()
                WHERE id::text = $3
                """, status, safe_json, decision_id)
                logger.info(f"✅ Updated decision {decision_id} status to {status}")
            else:
                # Sync fallback
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                cur.execute("""
                UPDATE aurea_decisions
                SET execution_status = %s,
                    execution_result = %s,
                    executed_at = NOW()
                WHERE id::text = %s
                """, (status, Json(safe_result) if safe_result else None, decision_id))
                rows_updated = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                if rows_updated > 0:
                    logger.info(f"✅ Updated decision {decision_id} status to {status}")
                else:
                    logger.warning(f"⚠️ No decision found with id {decision_id} to update")

        except Exception as e:
            logger.error(f"Failed to update decision status: {e}")

    async def _execute_decision(self, decision: Decision) -> Dict[str, Any]:
        """Execute a specific decision"""
        action_map = {
            "activate_collection_agents": self._activate_collection_agents,
            "activate_scheduling_optimization": self._activate_scheduling_optimization,
            "activate_retention_campaign": self._activate_retention_campaign,
            "activate_sales_acceleration": self._activate_sales_acceleration
        }

        action_func = action_map.get(decision.recommended_action)
        if action_func:
            return await action_func(decision.context)
        else:
            # Default: activate relevant agents based on decision type
            return await self._activate_agents_for_decision(decision)

    async def _activate_collection_agents(self, context: Dict) -> Dict:
        """Activate agents for collections (VERIFICATION MODE - no real outreach)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.INVOICE_OVERDUE,
            {"context": context, "aurea_initiated": True, "verification_mode": True, "dry_run": True}
        )
        return {"action": "collection_agents", "result": result, "mode": "verification_only"}

    async def _activate_scheduling_optimization(self, context: Dict) -> Dict:
        """Activate scheduling optimization (VERIFICATION MODE)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.SCHEDULING_CONFLICT,
            {"context": context, "aurea_initiated": True, "verification_mode": True, "dry_run": True}
        )
        return {"action": "scheduling_optimization", "result": result, "mode": "verification_only"}

    async def _activate_retention_campaign(self, context: Dict) -> Dict:
        """Activate customer retention campaign (VERIFICATION MODE - no real outreach)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.CUSTOMER_CHURN_RISK,
            {"context": context, "aurea_initiated": True, "verification_mode": True, "dry_run": True}
        )
        return {"action": "retention_campaign", "result": result, "mode": "verification_only"}

    async def _activate_sales_acceleration(self, context: Dict) -> Dict:
        """Activate sales acceleration (VERIFICATION MODE - no real outreach)"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.QUOTE_REQUESTED,
            {"context": context, "aurea_initiated": True, "verification_mode": True, "dry_run": True}
        )
        return {"action": "sales_acceleration", "result": result, "mode": "verification_only"}

    async def _activate_agents_for_decision(self, decision: Decision) -> Dict:
        """Generic agent activation based on decision type (VERIFICATION MODE)"""
        event_type_map = {
            DecisionType.FINANCIAL: BusinessEventType.PAYMENT_RECEIVED,
            DecisionType.OPERATIONAL: BusinessEventType.JOB_SCHEDULED,
            DecisionType.CUSTOMER: BusinessEventType.NEW_CUSTOMER,
            DecisionType.STRATEGIC: BusinessEventType.SYSTEM_HEALTH_CHECK
        }

        event_type = event_type_map.get(decision.type, BusinessEventType.SYSTEM_HEALTH_CHECK)
        result = await self.activation_system.handle_business_event(
            event_type,
            {"decision": asdict(decision), "aurea_initiated": True, "verification_mode": True, "dry_run": True}
        )

        return {"action": "generic_activation", "event_type": event_type.value, "result": result, "mode": "verification_only"}

    async def _learn(self, results: List[Dict[str, Any]]):
        """Learn from execution results and improve"""
        successful = [r for r in results if r.get("status") != "failed"]
        failed = [r for r in results if r.get("status") == "failed"]

        # Calculate success rate
        success_rate = len(successful) / len(results) if results else 0

        # Generate insights
        if success_rate < 0.5:
            insight = {
                "type": "performance",
                "insight": "Low success rate in decision execution",
                "recommendation": "Review decision confidence thresholds",
                "data": {"success_rate": success_rate, "failures": failed}
            }
            self.learning_insights.append(insight)
            await self._apply_learning(insight)

        # Store learning in memory
        self.memory.store(Memory(
            memory_type=MemoryType.META,
            content={
                "cycle": self.cycle_count,
                "results_analyzed": len(results),
                "success_rate": success_rate,
                "insights": self.learning_insights[-5:] if self.learning_insights else []
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

    async def _apply_learning(self, insight: Dict):
        """Apply learning insights to improve performance"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            if insight["type"] == "performance" and insight["data"].get("success_rate", 1.0) < 0.5:
                # Record the learning insight to the database
                cur.execute("""
                    INSERT INTO ai_learning_insights (tenant_id, insight_type, insight_data, applied_at, adjustment_made)
                    VALUES (%s, %s, %s, NOW(), %s)
                    ON CONFLICT DO NOTHING
                """, (
                    self.tenant_id,
                    insight["type"],
                    json.dumps(insight["data"]),
                    "reduced_confidence_threshold"
                ))

                # Adjust agent confidence thresholds for poor performers
                if "agent_name" in insight["data"]:
                    cur.execute("""
                        UPDATE ai_agents
                        SET config = config || '{"confidence_threshold": 0.7}'::jsonb,
                            updated_at = NOW()
                        WHERE name = %s AND tenant_id = %s
                    """, (insight["data"]["agent_name"], self.tenant_id))

                logger.info(f"📚 Learning applied: Adjusted confidence thresholds for low-performing agents")

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
            conn.close()

        except Exception as e:
            logger.warning(f"Could not apply learning insight: {e}")

    async def _self_heal(self):
        """
        Detect and fix system issues with MCP-powered auto-remediation.

        This is a KEY force multiplier - the system uses its own infrastructure
        tools (MCP Bridge) to heal itself, creating autonomous operations.
        """
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
            # Log the remediation attempt
            try:
                conn = psycopg2.connect(**DB_CONFIG)
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
                conn.close()
            except Exception as e:
                logger.error(f"Failed to log remediation: {e}")

    async def _execute_healing_action(self, action: str):
        """Execute a specific healing action"""
        logger.info(f"⚕️ Executing healing action: {action}")

        if action == "consolidate_memory":
            self.memory.consolidate(aggressive=True)
        elif action == "restart_failed_agents":
            # Restart failed agents by resetting their status
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                cur.execute("""
                    UPDATE ai_agents SET status = 'active', updated_at = NOW()
                    WHERE status = 'failed' AND tenant_id = %s
                """, (self.tenant_id,))
                restarted = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                logger.info(f"♻️ Restarted {restarted} failed agents")
            except Exception as e:
                logger.error(f"Failed to restart agents: {e}")
        elif action == "clear_decision_backlog":
            # Process backlogged decisions by marking stale ones as expired
            try:
                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()
                cur.execute("""
                    UPDATE ai_decision_history
                    SET status = 'expired', updated_at = NOW()
                    WHERE status = 'pending' AND created_at < NOW() - INTERVAL '24 hours'
                """)
                cleared = cur.rowcount
                conn.commit()
                cur.close()
                conn.close()
                logger.info(f"🧹 Cleared {cleared} stale decisions from backlog")
            except Exception as e:
                logger.error(f"Failed to clear decision backlog: {e}")
        elif action.startswith("mcp:"):
            # Execute MCP tool action (e.g., "mcp:render:restart_service")
            await self._execute_mcp_action(action)

    async def _execute_mcp_action(self, action: str) -> Dict[str, Any]:
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

    async def execute_external_tool(self, server: str, tool: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
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
        """Check overall system health"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
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
        """Request human approval for a decision"""
        logger.info(f"🤔 Human approval requested for: {decision.description}")

        # Check for test mode override
        if os.getenv("AUREA_TEST_MODE", "false").lower() == "true":
            logger.info("⚠️ Auto-approving decision in TEST MODE")
            return True

        # Store pending decision
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        try:
            # Update decision status
            cur.execute("""
            UPDATE aurea_decisions
            SET execution_status = 'pending'
            WHERE id = %s
            """, (decision.id,))
            
            # Create notification if table exists
            # We attempt to insert into task_notifications if it exists, otherwise rely on decision status
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
            
        except Exception as e:
            logger.error(f"Failed to queue decision for approval: {e}")
            return False
        finally:
            cur.close()
            conn.close()

        return False

    def _log_decision(self, decision: Decision) -> Optional[str]:
        """Log decision to database and return the database UUID"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
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

            conn.commit()
            cur.close()
            conn.close()

            return db_id

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")
            return None

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
        """Calculate adaptive sleep time based on activity"""
        if observations == 0 and decisions == 0:
            return 30  # Low activity, sleep longer
        elif observations > 5 or decisions > 3:
            return 5   # High activity, check frequently
        else:
            return 10  # Normal activity

    def set_autonomy_level(self, level: AutonomyLevel):
        """Adjust autonomy level"""
        self.autonomy_level = level
        logger.info(f"🎚️ Autonomy level set to: {level.name} ({level.value}%)")

    def stop(self):
        """Stop orchestration"""
        self.running = False
        logger.info("🛑 AUREA orchestration stopped")

    def get_status(self) -> Dict:
        """Get current AUREA status"""
        return {
            "running": self.running,
            "autonomy_level": self.autonomy_level.name,
            "cycles_completed": self.cycle_count,
            "decisions_made": self.decisions_made,
            "system_health": asdict(self.system_health) if self.system_health else None,
            "learning_insights": len(self.learning_insights),
            "last_health_check": self.last_health_check.isoformat()
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
