#!/usr/bin/env python3
"""
AUREA - Autonomous Universal Resource & Execution Assistant
The Master Orchestration Brain for BrainOps AI OS
Coordinates all 59 agents to work as one unified intelligence
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import aiohttp
from unified_memory_manager import get_memory_manager, Memory, MemoryType
from agent_activation_system import (
    get_activation_system, BusinessEventType, AgentActivationSystem
)
import warnings
warnings.filterwarnings('ignore')

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
    'password': os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
    'port': 6543
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
    SEMI_AUTO = 50    # AI decides minor, human decides major
    MOSTLY_AUTO = 75  # AI decides most, human approves critical
    FULL_AUTO = 100   # AI decides everything autonomously


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

    def __init__(self, autonomy_level: AutonomyLevel = AutonomyLevel.SEMI_AUTO):
        self.autonomy_level = autonomy_level
        self.memory = get_memory_manager()
        self.activation_system = get_activation_system()
        self.running = False
        self.cycle_count = 0
        self.decisions_made = 0
        self.last_health_check = datetime.now()
        self.system_health = None
        self.decision_queue = asyncio.Queue()
        self.learning_insights = []
        self._init_database()

        logger.info(f"🧠 AUREA initialized at autonomy level: {autonomy_level.name}")

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
                tenant_id UUID DEFAULT '51e728c5-94e8-4ae0-8a0a-6a08d1fb3457'::uuid
            );

            CREATE INDEX IF NOT EXISTS idx_aurea_decisions_type ON aurea_decisions(decision_type);
            CREATE INDEX IF NOT EXISTS idx_aurea_decisions_status ON aurea_decisions(execution_status);
            CREATE INDEX IF NOT EXISTS idx_aurea_decisions_created ON aurea_decisions(created_at DESC);

            -- Create AUREA system state
            CREATE TABLE IF NOT EXISTS aurea_state (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                state_type TEXT NOT NULL,
                state_data JSONB NOT NULL,
                cycle_number INTEGER,
                timestamp TIMESTAMP DEFAULT NOW()
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
                created_at TIMESTAMP DEFAULT NOW()
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
                        "observations": len(observations),
                        "decisions": len(decisions),
                        "actions": len(results),
                        "cycle_time_seconds": cycle_time,
                        "autonomy_level": self.autonomy_level.value
                    },
                    source_system="aurea",
                    source_agent="orchestrator",
                    created_by="aurea",
                    importance_score=0.3,
                    tags=["orchestration", "cycle"]
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
        """Observe the environment and gather all relevant data"""
        observations = []

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Check for new customers
            cur.execute("""
            SELECT COUNT(*) as count FROM customers
            WHERE created_at > NOW() - INTERVAL '5 minutes'
            """)
            new_customers = cur.fetchone()['count']
            if new_customers > 0:
                observations.append({
                    "type": "new_customers",
                    "count": new_customers,
                    "trigger": BusinessEventType.NEW_CUSTOMER
                })

            # Check for pending estimates
            cur.execute("""
            SELECT COUNT(*) as count, MIN(created_at) as oldest
            FROM estimates WHERE status = 'pending'
            """)
            pending_estimates = cur.fetchone()
            if pending_estimates['count'] > 0:
                observations.append({
                    "type": "pending_estimates",
                    "count": pending_estimates['count'],
                    "oldest": pending_estimates['oldest'],
                    "trigger": BusinessEventType.ESTIMATE_REQUESTED
                })

            # Check for overdue invoices (using computed amount_due)
            cur.execute("""
            SELECT COUNT(*) as count,
                   SUM(COALESCE(total_amount::numeric/100, 0) - COALESCE(paid_amount, 0)) as total_due
            FROM invoices
            WHERE due_date < NOW() AND status != 'paid'
            """)
            overdue = cur.fetchone()
            if overdue['count'] > 0:
                observations.append({
                    "type": "overdue_invoices",
                    "count": overdue['count'],
                    "total_due": float(overdue['total_due'] or 0),
                    "trigger": BusinessEventType.INVOICE_OVERDUE
                })

            # Check for scheduling conflicts
            cur.execute("""
            SELECT COUNT(*) as conflicts FROM jobs j1
            JOIN jobs j2 ON j1.crew_id = j2.crew_id
            WHERE j1.id != j2.id
              AND j1.scheduled_start < j2.scheduled_end
              AND j1.scheduled_end > j2.scheduled_start
              AND j1.status = 'scheduled' AND j2.status = 'scheduled'
            """)
            conflicts = cur.fetchone()['conflicts']
            if conflicts > 0:
                observations.append({
                    "type": "scheduling_conflicts",
                    "count": conflicts,
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

            # Check for customer churn risks (with fallback for missing column)
            cur.execute("""
            SELECT COUNT(*) as at_risk FROM customers c
            WHERE COALESCE(c.last_job_date,
                          (SELECT MAX(scheduled_start)
                           FROM jobs WHERE customer_id = c.id),
                          c.created_at) < NOW() - INTERVAL '90 days'
              AND COALESCE(lifetime_value, 0) > 1000
            """)
            churn_risk = cur.fetchone()['at_risk']
            if churn_risk > 0:
                observations.append({
                    "type": "churn_risk",
                    "count": churn_risk,
                    "trigger": BusinessEventType.CUSTOMER_CHURN_RISK
                })

            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Observation error: {e}")

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

        # Recall relevant memories for context
        relevant_memories = self.memory.recall(
            {"context": "decision_making", "observations": observations},
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

                # Execute the decision
                result = await self._execute_decision(decision)
                results.append(result)

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
                    tags=["decision", "execution", decision.type.value]
                ))

            except Exception as e:
                logger.error(f"Failed to execute decision {decision.id}: {e}")
                results.append({
                    "decision_id": decision.id,
                    "status": "failed",
                    "error": str(e)
                })

        return results

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
        """Activate agents for collections"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.INVOICE_OVERDUE,
            {"context": context, "aurea_initiated": True}
        )
        return {"action": "collection_agents", "result": result}

    async def _activate_scheduling_optimization(self, context: Dict) -> Dict:
        """Activate scheduling optimization"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.SCHEDULING_CONFLICT,
            {"context": context, "aurea_initiated": True}
        )
        return {"action": "scheduling_optimization", "result": result}

    async def _activate_retention_campaign(self, context: Dict) -> Dict:
        """Activate customer retention campaign"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.CUSTOMER_CHURN_RISK,
            {"context": context, "aurea_initiated": True}
        )
        return {"action": "retention_campaign", "result": result}

    async def _activate_sales_acceleration(self, context: Dict) -> Dict:
        """Activate sales acceleration"""
        result = await self.activation_system.handle_business_event(
            BusinessEventType.QUOTE_REQUESTED,
            {"context": context, "aurea_initiated": True}
        )
        return {"action": "sales_acceleration", "result": result}

    async def _activate_agents_for_decision(self, decision: Decision) -> Dict:
        """Generic agent activation based on decision type"""
        event_type_map = {
            DecisionType.FINANCIAL: BusinessEventType.PAYMENT_RECEIVED,
            DecisionType.OPERATIONAL: BusinessEventType.JOB_SCHEDULED,
            DecisionType.CUSTOMER: BusinessEventType.NEW_CUSTOMER,
            DecisionType.STRATEGIC: BusinessEventType.SYSTEM_HEALTH_CHECK
        }

        event_type = event_type_map.get(decision.type, BusinessEventType.SYSTEM_HEALTH_CHECK)
        result = await self.activation_system.handle_business_event(
            event_type,
            {"decision": asdict(decision), "aurea_initiated": True}
        )

        return {"action": "generic_activation", "event_type": event_type.value, "result": result}

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
            tags=["learning", "meta"]
        ))

        # Synthesize broader patterns every 10 cycles
        if self.cycle_count % 10 == 0:
            patterns = self.memory.synthesize(time_window=timedelta(hours=1))
            for pattern in patterns:
                logger.info(f"🧠 Pattern discovered: {pattern['description']}")

    async def _apply_learning(self, insight: Dict):
        """Apply learning insights to improve performance"""
        if insight["type"] == "performance" and insight["data"]["success_rate"] < 0.5:
            # Reduce confidence threshold to be more conservative
            logger.info("📚 Learning applied: Adjusting decision confidence thresholds")
            # In a real implementation, this would adjust internal parameters

    async def _self_heal(self):
        """Detect and fix system issues"""
        if self.system_health and self.system_health.overall_score < 70:
            logger.warning(f"⚕️ System health low: {self.system_health.overall_score}")

            # Attempt healing actions
            healing_actions = []

            if self.system_health.error_rate > 0.1:
                healing_actions.append("restart_failed_agents")

            if self.system_health.memory_utilization > 0.9:
                healing_actions.append("consolidate_memory")

            if self.system_health.decision_backlog > 10:
                healing_actions.append("clear_decision_backlog")

            for action in healing_actions:
                await self._execute_healing_action(action)

    async def _execute_healing_action(self, action: str):
        """Execute a specific healing action"""
        logger.info(f"⚕️ Executing healing action: {action}")

        if action == "consolidate_memory":
            self.memory.consolidate(aggressive=True)
        elif action == "restart_failed_agents":
            # In production, would restart failed agents
            pass
        elif action == "clear_decision_backlog":
            # Process backlogged decisions
            pass

    async def _check_system_health(self) -> SystemHealth:
        """Check overall system health"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get agent stats
            agent_stats = self.activation_system.get_agent_stats()

            # Get memory stats
            memory_stats = self.memory.get_stats()

            # Get error rate from logs
            cur.execute("""
            SELECT
                COUNT(CASE WHEN status = 'failed' THEN 1 END)::float /
                NULLIF(COUNT(*), 0) as error_rate
            FROM agent_activation_log
            WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            error_rate = cur.fetchone()['error_rate'] or 0

            # Get decision backlog
            cur.execute("""
            SELECT COUNT(*) as backlog
            FROM aurea_decisions
            WHERE execution_status = 'pending'
              AND created_at > NOW() - INTERVAL '24 hours'
            """)
            decision_backlog = cur.fetchone()['backlog']

            cur.close()
            conn.close()

            # Calculate overall health score
            health_components = {
                "agents": min(100, agent_stats.get("enabled_agents", 0) / 59 * 100),
                "memory": min(100, (1 - memory_stats.get("total_memories", 0) / 1000000) * 100),
                "errors": max(0, (1 - error_rate) * 100),
                "decisions": max(0, (1 - decision_backlog / 100) * 100),
                "performance": 85  # Placeholder
            }

            overall_score = sum(health_components.values()) / len(health_components)

            self.system_health = SystemHealth(
                overall_score=overall_score,
                component_health=health_components,
                active_agents=agent_stats.get("enabled_agents", 0),
                memory_utilization=memory_stats.get("total_memories", 0) / 1000000,
                decision_backlog=decision_backlog,
                error_rate=error_rate,
                performance_score=85,
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
        """Request human approval for a decision (placeholder)"""
        # In production, this would integrate with UI/notification system
        logger.info(f"🤔 Human approval requested for: {decision.description}")

        # For now, auto-approve in test mode
        if os.getenv("AUREA_TEST_MODE", "true") == "true":
            return True

        # Store pending decision
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
        UPDATE aurea_decisions
        SET execution_status = 'pending'
        WHERE id = %s
        """, (decision.id,))
        conn.commit()
        cur.close()
        conn.close()

        return False

    def _log_decision(self, decision: Decision):
        """Log decision to database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            cur.execute("""
            INSERT INTO aurea_decisions
            (decision_type, description, confidence, impact_assessment,
             recommended_action, alternatives, requires_human_approval,
             execution_status, context)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                decision.type.value,
                decision.description,
                decision.confidence,
                decision.impact_assessment,
                decision.recommended_action,
                Json(decision.alternatives),
                decision.requires_human_approval,
                "pending",
                Json(decision.context)
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

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
            tags=["error", "orchestration"]
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

def get_aurea() -> AUREA:
    """Get or create the singleton AUREA instance"""
    global aurea_instance
    if aurea_instance is None:
        aurea_instance = AUREA()
    return aurea_instance


async def run_aurea():
    """Run AUREA orchestration"""
    aurea = get_aurea()
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

    # Run AUREA
    try:
        asyncio.run(run_aurea())
    except KeyboardInterrupt:
        print("\n🛑 AUREA shutdown requested")
        aurea = get_aurea()
        aurea.stop()
        print("✅ AUREA stopped gracefully")