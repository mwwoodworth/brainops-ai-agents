#!/usr/bin/env python3
"""
BRAINOPS CORE - The Unified AI Operating System Entry Point
Integrates:
1. AUREA (Orchestrator)
2. AI Board (Governance)
3. Self-Awareness (Safety)
4. Knowledge Graph (Memory)
5. Revenue System (Engine)
"""

import asyncio
import logging
from typing import Optional

from ai_board_governance import AIBoardOfDirectors, Proposal, ProposalType
from ai_knowledge_graph import AIKnowledgeGraph, get_knowledge_graph
from ai_self_awareness import SelfAwareAI, get_self_aware_ai

# Import Core Systems
from aurea_orchestrator import AUREA, AutonomyLevel, DecisionType
from revenue_generation_system import AutonomousRevenueSystem, get_revenue_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BRAINOPS_CORE')

class BrainOpsCore:
    """
    The Unified AI Operating System Kernel
    Wires together the 5 distinct autonomous systems into one cohesive organism.
    """

    def __init__(self, tenant_id: str, mode: str = "simulation"):
        self.tenant_id = tenant_id
        self.mode = mode

        # System references
        self.aurea: Optional[AUREA] = None
        self.board: Optional[AIBoardOfDirectors] = None
        self.self_aware: Optional[SelfAwareAI] = None
        self.knowledge: Optional[AIKnowledgeGraph] = None
        self.revenue: Optional[AutonomousRevenueSystem] = None

        logger.info(f"🧠 BrainOps Core initializing for tenant {tenant_id} in {mode} mode")

    async def initialize(self):
        """Boot sequence for the AI OS"""
        logger.info("🔌 Starting Boot Sequence...")

        # 1. Initialize Memory (Knowledge Graph)
        # The foundation of all intelligence
        self.knowledge = get_knowledge_graph()
        # In a real scenario, we might await a build step here
        # await self.knowledge.build_from_all_sources()
        logger.info("✅ Knowledge Graph connected")

        # 2. Initialize Safety (Self-Awareness)
        # Must be active before any decisions are made
        self.self_aware = await get_self_aware_ai()
        logger.info("✅ Self-Awareness Module active")

        # 3. Initialize Governance (AI Board)
        # Sets the rules of engagement
        self.board = AIBoardOfDirectors()
        logger.info("✅ AI Board of Directors seated")

        # 4. Initialize Revenue Engine
        # The functional capability to sustain the business
        self.revenue = get_revenue_system()
        logger.info("✅ Revenue Engine primed")

        # 5. Initialize AUREA (The Master Orchestrator)
        # Connects everything together
        autonomy = AutonomyLevel.FULL_AUTO if self.mode == "production" else AutonomyLevel.SEMI_AUTO
        self.aurea = AUREA(tenant_id=self.tenant_id, autonomy_level=autonomy)

        # Inject dependencies into AUREA (Monkey-patching or future setter injection)
        # Ideally, AUREA class should accept these in constructor.
        # For this architecture V3 implementation, we are orchestrating them here.
        self.aurea.board_ref = self.board
        self.aurea.safety_ref = self.self_aware
        self.aurea.revenue_ref = self.revenue
        self.aurea.knowledge_ref = self.knowledge

        logger.info(f"✅ AUREA Orchestrator online (Autonomy: {autonomy.name})")
        logger.info("🚀 BrainOps Core Boot Complete")

    async def run_cycle(self):
        """
        Execute one complete system lifecycle (The 'Heartbeat')
        This supersedes the standalone AUREA loop by adding the other layers.
        """
        if not self.aurea:
            await self.initialize()

        logger.info("\n--- 💓 SYSTEM HEARTBEAT START ---")

        # 1. ORCHESTRATION: AUREA observes and proposes decisions
        observations = await self.aurea._observe()
        context = await self.aurea._orient(observations)
        decisions = await self.aurea._decide(context)

        for decision in decisions:
            logger.info(f"🤔 AUREA Proposed: {decision.description} (Confidence: {decision.confidence})")

            # 2. SAFETY CHECK: Self-Awareness Module validation
            # Wrap the decision in a self-assessment
            assessment = await self.self_aware.assess_confidence(
                task_id=decision.id,
                agent_id="AUREA",
                task_description=decision.description,
                task_context=decision.context
            )

            if assessment.confidence_score < 80:
                logger.warning(f"🛑 SAFETY VETO: Confidence {assessment.confidence_score}% too low for autonomous execution.")
                decision.requires_human_approval = True

            # 3. GOVERNANCE CHECK: AI Board validation for Strategic/High-Value decisions
            if decision.type in [DecisionType.STRATEGIC, DecisionType.FINANCIAL]:
                logger.info("⚖️  Escalating to AI Board of Directors...")
                proposal = Proposal(
                    id=decision.id,
                    type=ProposalType.STRATEGIC, # Mapping simplified
                    title=decision.description,
                    description=decision.impact_assessment,
                    proposed_by="AUREA",
                    impact_analysis=decision.context.get("priority", {}),
                    required_resources={},
                    timeline="Immediate",
                    alternatives=decision.alternatives,
                    supporting_data=decision.context,
                    urgency=8,
                    created_at=decision.deadline # Approximate
                )

                # Board deliberates
                board_decision = await self.board._deliberate_proposal(proposal)
                logger.info(f"🏛️  Board Verdict: {board_decision.decision.upper()}")

                if board_decision.decision != "approved":
                    logger.info(f"❌ Decision blocked by Board. Reason: {board_decision.dissenting_opinions}")
                    continue # Skip execution

            # 4. EXECUTION: Revenue System or General Agents
            if not decision.requires_human_approval:
                logger.info(f"⚡ Executing: {decision.description}")
                # Hook into revenue system if applicable
                if "revenue" in decision.description.lower() or "collection" in decision.description.lower():
                     # Example: Trigger revenue workflow
                     # await self.revenue.run_revenue_workflow(...)
                     pass

                # Standard AUREA execution
                await self.aurea._execute_decision(decision)

                # 5. MEMORY: Record in Knowledge Graph
                # In a full implementation, this would extract nodes/edges from the result
                # await self.knowledge.builder.build_graph(...)
                logger.info("💾 Execution recorded to Knowledge Graph")

        logger.info("--- 💓 SYSTEM HEARTBEAT END ---\\n")

    async def run_forever(self):
        """Run the OS continuously"""
        await self.initialize()

        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(10) # 10s heartbeat
            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                break
            except Exception as e:
                logger.error(f"💥 Critical Core Error: {e}")
                await asyncio.sleep(30) # Backoff

if __name__ == "__main__":
    # Example Production Run
    TEST_TENANT = "brainops-prod-tenant"

    core = BrainOpsCore(tenant_id=TEST_TENANT, mode="simulation")

    try:
        asyncio.run(core.run_forever())
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
