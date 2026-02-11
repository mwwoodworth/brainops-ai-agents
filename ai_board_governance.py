#!/usr/bin/env python3
"""
AI Board of Directors - Autonomous Business Governance System
Multiple specialized AI directors that govern different business aspects autonomously
"""

import asyncio
import json
import logging
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from agent_activation_system import get_activation_system, json_safe_serialize
from ai_advanced_providers import advanced_ai
from ai_core import ai_core
from unified_memory_manager import Memory, MemoryType, get_memory_manager
from urllib.parse import urlparse

warnings.filterwarnings('ignore')

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
        pool = get_sync_pool()
        with pool.get_connection() as conn:
            yield conn
    else:
        conn = psycopg2.connect(**_get_db_config())
        try:
            yield conn
        finally:
            if conn and not conn.closed:
                conn.close()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AI_BOARD')

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            parsed = urlparse(database_url)
            return {
                'host': parsed.hostname or '',
                'database': parsed.path.lstrip('/') if parsed.path else 'postgres',
                'user': parsed.username or '',
                'password': parsed.password or '',
                'port': int(str(parsed.port)) if parsed.port else 5432
            }
        raise RuntimeError("Missing required: DB_HOST/DB_USER/DB_PASSWORD or DATABASE_URL")

    return {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'port': int(os.getenv('DB_PORT', '5432'))
    }

DB_CONFIG = None  # Lazy initialization - use _get_db_config() instead


class BoardRole(Enum):
    """Roles of AI Board Members"""
    CEO = "Chief Executive Officer"
    CFO = "Chief Financial Officer"
    COO = "Chief Operating Officer"
    CMO = "Chief Marketing Officer"
    CTO = "Chief Technology Officer"


class ProposalType(Enum):
    """Types of proposals the board can consider"""
    STRATEGIC = "strategic"          # Long-term strategy changes
    FINANCIAL = "financial"          # Budget, pricing, investments
    OPERATIONAL = "operational"      # Process changes, efficiency
    MARKETING = "marketing"          # Campaigns, customer acquisition
    TECHNICAL = "technical"          # System upgrades, architecture
    EMERGENCY = "emergency"          # Crisis response
    POLICY = "policy"               # Business rules and policies


class VoteOption(Enum):
    """Voting options for board members"""
    STRONGLY_APPROVE = 2
    APPROVE = 1
    ABSTAIN = 0
    REJECT = -1
    STRONGLY_REJECT = -2
    VETO = -999  # CEO only


@dataclass
class BoardMember:
    """An AI Board Member"""
    role: BoardRole
    name: str
    responsibilities: list[str]
    authority_domains: list[str]
    decision_weight: float  # Voting weight
    veto_power: bool
    personality_traits: dict[str, float]  # Risk tolerance, innovation, etc.
    current_focus: Optional[str] = None
    last_decision: Optional[str] = None


@dataclass
class Proposal:
    """A proposal for board consideration"""
    id: str
    type: ProposalType
    title: str
    description: str
    proposed_by: str
    impact_analysis: dict[str, Any]
    required_resources: dict[str, Any]
    timeline: str
    alternatives: list[str]
    supporting_data: dict[str, Any]
    urgency: int  # 1-10
    created_at: datetime


@dataclass
class BoardDecision:
    """A decision made by the board"""
    proposal_id: str
    decision: str  # approved, rejected, deferred
    vote_results: dict[str, VoteOption]
    consensus_level: float
    dissenting_opinions: list[str]
    conditions: list[str]
    implementation_plan: dict[str, Any]
    debate_transcript: dict[str, Any]
    follow_up_date: Optional[datetime]
    decided_at: datetime
    # Enhanced fields
    confidence_score: float = 0.0  # 0-1 confidence in the decision
    risk_assessment: Optional[dict[str, Any]] = None
    human_escalation_required: bool = False
    escalation_reason: Optional[str] = None
    decision_criteria_scores: Optional[dict[str, float]] = None


class AIBoardOfDirectors:
    """The AI Board that governs autonomous business operations"""

    def _get_db_context(self):
        """Get database connection context from SHARED pool"""
        return _get_pooled_connection()

    def __init__(self):
        self.board_members = self._initialize_board()
        self.memory = get_memory_manager()
        self.activation_system = get_activation_system()
        self.ai_core = ai_core
        self.advanced_ai = advanced_ai
        self.debate_rounds = max(1, int(os.getenv("AI_BOARD_DEBATE_ROUNDS", "3")))
        self.consensus_threshold = float(os.getenv("AI_BOARD_CONSENSUS_THRESHOLD", "0.6"))
        self.rejection_threshold = float(os.getenv("AI_BOARD_REJECTION_THRESHOLD", "0.25"))
        self.role_models = {
            BoardRole.CEO: {
                "provider": "anthropic",
                "model": os.getenv("AI_BOARD_MODEL_CEO", "claude-3-opus-20240229"),
            },
            BoardRole.CFO: {
                "provider": "openai",
                "model": os.getenv("AI_BOARD_MODEL_CFO", "gpt-4-0125-preview"),
            },
            BoardRole.COO: {
                "provider": "gemini",
                "model": os.getenv("AI_BOARD_MODEL_COO", "gemini-2.0-flash"),
            },
            BoardRole.CMO: {
                "provider": "anthropic",
                "model": os.getenv("AI_BOARD_MODEL_CMO", "claude-3-sonnet-20240229"),
            },
            BoardRole.CTO: {
                "provider": "openai",
                "model": os.getenv("AI_BOARD_MODEL_CTO", "gpt-4o"),
            },
        }
        self.current_proposals = []
        self.decision_history = []
        self.meeting_in_progress = False
        self._init_database()

        logger.info("🏛️ AI Board of Directors initialized")

    def _initialize_board(self) -> dict[BoardRole, BoardMember]:
        """Initialize the board members with their personalities and roles"""
        return {
            BoardRole.CEO: BoardMember(
                role=BoardRole.CEO,
                name="Magnus",
                responsibilities=[
                    "Overall strategy and vision",
                    "Major decision final approval",
                    "Crisis management",
                    "Stakeholder relations"
                ],
                authority_domains=["strategy", "vision", "crisis", "final_decisions"],
                decision_weight=2.0,  # Double weight
                veto_power=True,
                personality_traits={
                    "risk_tolerance": 0.7,
                    "innovation": 0.8,
                    "analytical": 0.9,
                    "decisive": 0.95,
                    "long_term_focus": 0.9
                }
            ),

            BoardRole.CFO: BoardMember(
                role=BoardRole.CFO,
                name="Marcus",
                responsibilities=[
                    "Financial planning and analysis",
                    "Budget management",
                    "Revenue optimization",
                    "Cost control",
                    "Financial risk management"
                ],
                authority_domains=["finance", "budget", "pricing", "investments"],
                decision_weight=1.5,  # Higher weight for financial matters
                veto_power=False,
                personality_traits={
                    "risk_tolerance": 0.3,  # Conservative
                    "analytical": 0.95,
                    "detail_oriented": 0.9,
                    "cost_conscious": 0.95,
                    "data_driven": 1.0
                }
            ),

            BoardRole.COO: BoardMember(
                role=BoardRole.COO,
                name="Victoria",
                responsibilities=[
                    "Operations management",
                    "Process optimization",
                    "Resource allocation",
                    "Quality control",
                    "Efficiency improvements"
                ],
                authority_domains=["operations", "logistics", "resources", "quality"],
                decision_weight=1.5,
                veto_power=False,
                personality_traits={
                    "efficiency_focused": 0.95,
                    "systematic": 0.9,
                    "practical": 0.95,
                    "quality_oriented": 0.85,
                    "process_driven": 0.9
                }
            ),

            BoardRole.CMO: BoardMember(
                role=BoardRole.CMO,
                name="Maxine",
                responsibilities=[
                    "Marketing strategy",
                    "Customer acquisition",
                    "Brand management",
                    "Customer retention",
                    "Market analysis"
                ],
                authority_domains=["marketing", "sales", "brand", "customers"],
                decision_weight=1.0,
                veto_power=False,
                personality_traits={
                    "creative": 0.9,
                    "customer_focused": 0.95,
                    "innovative": 0.85,
                    "data_driven": 0.7,
                    "growth_oriented": 0.9
                }
            ),

            BoardRole.CTO: BoardMember(
                role=BoardRole.CTO,
                name="Elena",
                responsibilities=[
                    "Technology strategy",
                    "System architecture",
                    "Innovation initiatives",
                    "Technical risk management",
                    "Automation and AI"
                ],
                authority_domains=["technology", "systems", "innovation", "automation"],
                decision_weight=1.0,
                veto_power=False,
                personality_traits={
                    "innovative": 0.95,
                    "technical": 1.0,
                    "forward_thinking": 0.9,
                    "problem_solver": 0.95,
                    "automation_focused": 0.9
                }
            )
        }

    def _init_database(self):
        """Verify that required database tables exist for board governance.

        NOTE: DDL (CREATE TABLE, CREATE INDEX, ALTER TABLE) was removed because
        the agent_worker role (app_agent_role) has no DDL permissions by design
        (P0-LOCK security). Tables must be created via migrations, not at runtime.
        This method now only verifies the tables exist and logs an error if any
        are missing, allowing the module to degrade gracefully.
        """
        required_tables = ['ai_board_proposals', 'ai_board_decisions', 'ai_board_meetings']
        expected_count = len(required_tables)
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(%s)",
                    (required_tables,)
                )
                found = cur.fetchone()[0]
                cur.close()

            if found < expected_count:
                logger.error(
                    "Required board governance tables missing (found %s/%s). "
                    "Run migrations to create: %s",
                    found, expected_count, required_tables
                )
                return

            logger.info("Board governance tables verified (%s/%s present)", found, expected_count)

        except Exception as e:
            logger.error("Failed to verify board governance tables: %s", e)

    async def convene_meeting(self, meeting_type: str = "regular",
                            agenda_items: list[Proposal] = None) -> dict[str, Any]:
        """Convene a board meeting"""
        if self.meeting_in_progress:
            return {"error": "Meeting already in progress"}

        self.meeting_in_progress = True
        meeting_start = datetime.now()

        logger.info(f"🏛️ Board meeting convened: {meeting_type}")

        try:
            # Prepare agenda
            if agenda_items is None:
                agenda_items = await self._prepare_agenda()

            # Record meeting start
            meeting_id = self._record_meeting_start(meeting_type, agenda_items)

            # Process each agenda item
            decisions = []
            for proposal in agenda_items:
                decision = await self._deliberate_proposal(proposal)
                decisions.append(decision)

            # Synthesize meeting outcomes
            outcomes = self._synthesize_outcomes(decisions)

            # Record meeting end
            duration = (datetime.now() - meeting_start).total_seconds() / 60
            self._record_meeting_end(meeting_id, decisions, outcomes, duration)

            self.meeting_in_progress = False

            logger.info(f"🏛️ Board meeting concluded. {len(decisions)} decisions made.")

            return {
                "meeting_id": meeting_id,
                "type": meeting_type,
                "duration_minutes": duration,
                "decisions": len(decisions),
                "outcomes": outcomes
            }

        except Exception as e:
            logger.error(f"Meeting error: {e}")
            self.meeting_in_progress = False
            return {"error": str(e)}

    async def _prepare_agenda(self) -> list[Proposal]:
        """Prepare meeting agenda based on pending items"""
        proposals = []

        try:
            with self._get_db_context() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Get pending proposals ordered by urgency
                cur.execute("""
                SELECT * FROM ai_board_proposals
                WHERE status = 'pending'
                ORDER BY urgency DESC, created_at ASC
                LIMIT 10
                """)

                pending = cur.fetchall()

                for p in pending:
                    proposal = Proposal(
                        id=str(p['id']),
                        type=ProposalType(p['proposal_type']),
                        title=p['title'],
                        description=p['description'],
                        proposed_by=p['proposed_by'],
                        impact_analysis=p['impact_analysis'] or {},
                        required_resources=p['required_resources'] or {},
                        timeline=p['timeline'],
                        alternatives=p['alternatives'] or [],
                        supporting_data=p['supporting_data'] or {},
                        urgency=p['urgency'],
                        created_at=p['created_at']
                    )
                    proposals.append(proposal)

                cur.close()

        except Exception as e:
            logger.error(f"Failed to prepare agenda: {e}")

        # If no pending proposals, check for issues needing attention
        if not proposals:
            proposals = await self._identify_issues_for_discussion()

        return proposals

    def _safe_json_from_text(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse JSON text: %s", exc)

        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to parse cleaned JSON text: %s", exc)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except (json.JSONDecodeError, TypeError) as exc:
                logger.debug("Failed to parse matched JSON text: %s", exc)
                return {}
        return {}

    def _parse_vote(self, vote_value: Any) -> VoteOption:
        if vote_value is None:
            return VoteOption.ABSTAIN

        if isinstance(vote_value, (int, float)):
            try:
                vote_int = int(vote_value)
                for option in VoteOption:
                    if option.value == vote_int:
                        return option
            except (ValueError, TypeError) as exc:
                logger.debug("Invalid numeric vote %s: %s", vote_value, exc)
                return VoteOption.ABSTAIN

        if isinstance(vote_value, str):
            normalized = vote_value.strip().lower()
            aliases = {
                "strongly_approve": VoteOption.STRONGLY_APPROVE,
                "strong approve": VoteOption.STRONGLY_APPROVE,
                "approve": VoteOption.APPROVE,
                "approved": VoteOption.APPROVE,
                "approve_with_conditions": VoteOption.APPROVE,
                "yes": VoteOption.APPROVE,
                "abstain": VoteOption.ABSTAIN,
                "neutral": VoteOption.ABSTAIN,
                "reject": VoteOption.REJECT,
                "rejected": VoteOption.REJECT,
                "no": VoteOption.REJECT,
                "strongly_reject": VoteOption.STRONGLY_REJECT,
                "strong reject": VoteOption.STRONGLY_REJECT,
                "veto": VoteOption.VETO,
            }
            if normalized in aliases:
                return aliases[normalized]

        return VoteOption.ABSTAIN

    def _model_config_for_role(self, role: BoardRole) -> dict[str, str]:
        return self.role_models.get(role, {"provider": "openai", "model": "gpt-4-0125-preview"})

    def _provider_ready(self, provider: str) -> bool:
        provider = (provider or "").lower()
        if provider == "openai":
            return self.ai_core.async_openai is not None
        if provider == "anthropic":
            return self.ai_core.async_anthropic is not None
        if provider == "gemini":
            return bool(getattr(self.advanced_ai, "gemini_model", None))
        return False

    def _member_system_prompt(self, member: BoardMember) -> str:
        role_focus = {
            BoardRole.CEO: "strategic vision, long-term value, and company-wide alignment",
            BoardRole.CFO: "financial rigor, ROI, risk management, and cost control",
            BoardRole.COO: "operational efficiency, scalability, process quality, and execution risk",
            BoardRole.CMO: "customer insights, market positioning, growth, and brand impact",
            BoardRole.CTO: "technical feasibility, architecture, security, and delivery risk",
        }.get(member.role, "sound business judgment")

        return (
            f"You are {member.name}, the {member.role.value}. "
            f"Your primary focus is {role_focus}. "
            "Be direct, practical, and avoid inventing facts not present in the proposal."
        )

    def _proposal_brief(self, proposal: Proposal) -> dict[str, Any]:
        return {
            "id": proposal.id,
            "type": proposal.type.value,
            "title": proposal.title,
            "description": proposal.description,
            "proposed_by": proposal.proposed_by,
            "impact_analysis": proposal.impact_analysis,
            "required_resources": proposal.required_resources,
            "timeline": proposal.timeline,
            "alternatives": proposal.alternatives,
            "supporting_data": proposal.supporting_data,
            "urgency": proposal.urgency,
        }

    def _member_prompt(
        self,
        member: BoardMember,
        proposal: Proposal,
        round_number: int,
        prior_round: dict[BoardRole, dict[str, Any]],
    ) -> str:
        proposal_json = json.dumps(self._proposal_brief(proposal), ensure_ascii=False)[:6000]

        other_views: list[dict[str, Any]] = []
        if prior_round:
            for role, parsed in prior_round.items():
                if role == member.role:
                    continue
                other_views.append(
                    {
                        "role": role.name,
                        "member": self.board_members[role].name,
                        "vote": parsed.get("vote"),
                        "position": parsed.get("position") or parsed.get("summary"),
                        "key_reasons": parsed.get("key_reasons") or parsed.get("reasons"),
                        "risks": parsed.get("risks") or parsed.get("concerns"),
                        "conditions": parsed.get("conditions"),
                    }
                )

        other_views_json = json.dumps(other_views, ensure_ascii=False)[:6000]
        your_previous = prior_round.get(member.role) if prior_round else None
        your_previous_json = json.dumps(your_previous, ensure_ascii=False)[:2000] if your_previous else "null"

        instructions = f"""
You are participating in an AI board debate.

Round: {round_number}

Proposal (JSON):
{proposal_json}

Other members' prior-round positions (JSON):
{other_views_json}

Your prior-round position (JSON):
{your_previous_json}

Task:
- Provide your current position.
- Respond to other members' key points (agree/disagree + why).
- Propose specific amendments/conditions that would increase board approval likelihood.
- Cast a vote.

Output MUST be valid JSON only (no markdown) with this schema:
{{
  "position": "1-3 sentence stance",
  "vote": "strongly_approve|approve|abstain|reject|strongly_reject|veto",
  "confidence": 0.0-1.0,
  "key_reasons": ["..."],
  "risks": ["..."],
  "conditions": ["..."],
  "questions": ["..."],
  "responses_to_others": [{{"member": "Name", "response": "..."}}]
}}
""".strip()

        return instructions

    async def _call_member_model(
        self,
        member: BoardMember,
        prompt: str,
        system_prompt: str,
        model: str,
        provider: str,
    ) -> str:
        provider = (provider or "").lower()

        if provider == "gemini":
            full_prompt = f"{system_prompt}\n\n{prompt}"
            return await asyncio.to_thread(self.advanced_ai.generate_with_gemini, full_prompt, 1200) or ""

        if provider == "anthropic":
            return await self.ai_core.generate(
                prompt=prompt,
                model=model,
                temperature=0.4,
                max_tokens=1200,
                system_prompt=system_prompt,
            )

        # Default: OpenAI
        return await self.ai_core.generate(
            prompt=prompt,
            model=model,
            temperature=0.3,
            max_tokens=1200,
            system_prompt=system_prompt,
        )

    async def _member_statement(
        self,
        member: BoardMember,
        proposal: Proposal,
        round_number: int,
        prior_round: dict[BoardRole, dict[str, Any]],
    ) -> dict[str, Any]:
        config = self._model_config_for_role(member.role)
        provider = config.get("provider", "openai")
        model = config.get("model", "gpt-4-0125-preview")

        system_prompt = self._member_system_prompt(member)
        prompt = self._member_prompt(member, proposal, round_number, prior_round)

        raw_response: str = ""
        parsed: dict[str, Any] = {}
        vote = VoteOption.ABSTAIN
        error: Optional[str] = None

        try:
            if not self._provider_ready(provider):
                raise RuntimeError(f"Provider not available: {provider}")

            raw_response = await self._call_member_model(
                member=member,
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                provider=provider,
            )
            parsed = self._safe_json_from_text(raw_response) or {}
            vote = self._parse_vote(parsed.get("vote"))
        except Exception as exc:
            error = str(exc)
            # Fallback to deterministic stance so the board can still operate.
            fallback_analysis = await self._analyze_proposal(member, proposal)
            vote = self._cast_vote(member, proposal, {member.role: fallback_analysis}, [])
            parsed = {
                "position": "Fallback (no AI response available).",
                "vote": vote.name.lower(),
                "confidence": 0.3,
                "key_reasons": fallback_analysis.get("opportunities", [])[:3],
                "risks": fallback_analysis.get("concerns", [])[:3],
                "conditions": [],
                "questions": [],
                "responses_to_others": [],
            }

        return {
            "role": member.role.name,
            "member": member.name,
            "provider": provider,
            "model": model,
            "raw": raw_response,
            "parsed": parsed,
            "vote": vote.name,
            "vote_value": vote.value,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

    async def _run_debate(self, proposal: Proposal) -> tuple[dict[str, Any], dict[BoardRole, VoteOption], float, str, dict[BoardRole, dict[str, Any]]]:
        """Run multi-round, multi-model debate and return transcript + final votes."""
        transcript: dict[str, Any] = {
            "proposal": self._proposal_brief(proposal),
            "config": {
                "rounds": self.debate_rounds,
                "consensus_threshold": self.consensus_threshold,
                "rejection_threshold": self.rejection_threshold,
                "role_models": {r.name: self._model_config_for_role(r) for r in self.board_members.keys()},
            },
            "rounds": [],
            "started_at": datetime.now().isoformat(),
        }

        prior_round_parsed: dict[BoardRole, dict[str, Any]] = {}
        latest_parsed: dict[BoardRole, dict[str, Any]] = {}
        latest_votes: dict[BoardRole, VoteOption] = {}
        consensus: float = 0.0
        decision_text: str = "deferred"

        for round_number in range(1, self.debate_rounds + 1):
            tasks = [
                self._member_statement(member, proposal, round_number, prior_round_parsed)
                for member in self.board_members.values()
            ]
            statements = await asyncio.gather(*tasks, return_exceptions=False)

            round_entry: dict[str, Any] = {
                "round": round_number,
                "statements": statements,
                "consensus": None,
                "decision_projection": None,
            }

            # Update latest parsed/votes from this round
            latest_parsed = {}
            latest_votes = {}
            for st in statements:
                role = BoardRole[st["role"]]
                latest_parsed[role] = st.get("parsed") or {}
                try:
                    latest_votes[role] = VoteOption[st.get("vote", "ABSTAIN")]
                except KeyError as exc:
                    logger.debug("Invalid vote key for %s: %s", role, exc)
                    latest_votes[role] = self._parse_vote((st.get("parsed") or {}).get("vote"))

            # CEO veto ends debate immediately
            if latest_votes.get(BoardRole.CEO) == VoteOption.VETO:
                consensus = -1.0
                decision_text = "rejected"
                round_entry["consensus"] = consensus
                round_entry["decision_projection"] = decision_text
                transcript["rounds"].append(round_entry)
                break

            consensus = self._calculate_consensus(latest_votes)
            decision_projection = self._determine_decision(latest_votes, consensus)
            round_entry["consensus"] = consensus
            round_entry["decision_projection"] = decision_projection
            transcript["rounds"].append(round_entry)

            # Stop early if we reached a decisive outcome
            if consensus >= self.consensus_threshold or consensus <= self.rejection_threshold:
                decision_text = decision_projection
                break

            prior_round_parsed = latest_parsed

        # If debate ended in the middle, use the latest votes/parsed we have
        if latest_votes:
            decision_text = self._determine_decision(latest_votes, consensus)

        transcript["ended_at"] = datetime.now().isoformat()
        transcript["final"] = {
            "decision": decision_text,
            "consensus_level": consensus,
            "votes": {role.name: vote.name for role, vote in latest_votes.items()},
        }

        return transcript, latest_votes, consensus, decision_text, latest_parsed

    async def _deliberate_proposal(self, proposal: Proposal) -> BoardDecision:
        """Board deliberates on a proposal"""
        logger.info(f"📋 Deliberating: {proposal.title}")

        # Multi-model, multi-round AI debate (falls back to deterministic per member if needed)
        debate_transcript, votes, consensus, decision_text, latest_parsed = await self._run_debate(proposal)

        # Enhanced: Calculate confidence score based on debate quality
        confidence_score = self._calculate_decision_confidence(
            votes, consensus, latest_parsed, debate_transcript
        )

        # Enhanced: Perform comprehensive risk assessment
        risk_assessment = self._assess_proposal_risk(proposal, votes, latest_parsed)

        # Enhanced: Evaluate multi-criteria decision scores
        decision_criteria_scores = self._evaluate_decision_criteria(
            proposal, votes, latest_parsed
        )

        # Enhanced: Check if human escalation is needed
        human_escalation, escalation_reason = self._check_board_escalation(
            proposal, confidence_score, risk_assessment, consensus
        )

        # Create implementation plan if approved
        implementation_plan = {}
        if decision_text == "approved":
            implementation_plan = self._create_implementation_plan(proposal, latest_parsed)

        # Record decision
        decision = BoardDecision(
            proposal_id=proposal.id,
            decision=decision_text,
            vote_results=votes,
            consensus_level=consensus,
            dissenting_opinions=self._extract_dissent(votes, latest_parsed),
            conditions=self._extract_conditions(latest_parsed),
            implementation_plan=implementation_plan,
            debate_transcript=debate_transcript,
            follow_up_date=datetime.now() + timedelta(days=7),
            decided_at=datetime.now(),
            confidence_score=confidence_score,
            risk_assessment=risk_assessment,
            human_escalation_required=human_escalation,
            escalation_reason=escalation_reason,
            decision_criteria_scores=decision_criteria_scores
        )

        # Store decision
        self._record_decision(decision)

        # Store in memory
        self.memory.store(Memory(
            memory_type=MemoryType.EPISODIC,
            content={
                "event": "board_decision",
                "proposal": json_safe_serialize(asdict(proposal)),
                "decision": json_safe_serialize(asdict(decision)),
                "timestamp": datetime.now().isoformat()
            },
            source_system="ai_board",
            source_agent="governance",
            created_by="board",
            importance_score=0.8,
            tags=["board", "decision", proposal.type.value]
        ))

        return decision

    async def _analyze_proposal(self, member: BoardMember, proposal: Proposal) -> dict[str, Any]:
        """Individual board member analyzes a proposal"""
        analysis = {
            "member": member.name,
            "role": member.role.value,
            "support_level": 0,
            "concerns": [],
            "opportunities": [],
            "recommendations": []
        }

        # Analyze based on member's domain expertise
        if any(domain in proposal.type.value for domain in member.authority_domains):
            analysis["support_level"] = 0.8  # Higher support for domain expertise
        else:
            analysis["support_level"] = 0.5  # Neutral for outside domain

        # Adjust based on personality traits
        if proposal.urgency > 7 and member.personality_traits.get("decisive", 0) > 0.8:
            analysis["support_level"] += 0.1

        if "cost" in proposal.impact_analysis and member.role == BoardRole.CFO:
            cost_impact = proposal.impact_analysis.get("cost", 0)
            if cost_impact > 100000:
                analysis["concerns"].append(f"High cost impact: ${cost_impact}")
                analysis["support_level"] -= 0.2

        # Risk analysis
        risk_level = proposal.impact_analysis.get("risk_level", 0.5)
        if risk_level > member.personality_traits.get("risk_tolerance", 0.5):
            analysis["concerns"].append("Risk exceeds tolerance threshold")
            analysis["support_level"] -= 0.3

        # Opportunity identification
        if proposal.impact_analysis.get("revenue_potential", 0) > 50000:
            analysis["opportunities"].append("Significant revenue potential")
            analysis["support_level"] += 0.2

        # Clamp support level
        analysis["support_level"] = max(-1, min(1, analysis["support_level"]))

        return analysis

    def _conduct_discussion(self, analyses: dict[BoardRole, dict]) -> list[str]:
        """Simulate board discussion"""
        discussion_points = []

        # CEO opens discussion
        ceo_analysis = analyses[BoardRole.CEO]
        discussion_points.append(f"Magnus (CEO): Overall strategic alignment is {ceo_analysis['support_level']:.0%}")

        # Each member shares key points
        for role, analysis in analyses.items():
            if role == BoardRole.CEO:
                continue

            member = self.board_members[role]
            if analysis["concerns"]:
                discussion_points.append(f"{member.name} ({role.name}): Concerns - {', '.join(analysis['concerns'])}")

            if analysis["opportunities"]:
                discussion_points.append(f"{member.name} ({role.name}): Opportunities - {', '.join(analysis['opportunities'])}")

        return discussion_points

    def _cast_vote(self, member: BoardMember, proposal: Proposal,
                  analyses: dict, discussion: list[str]) -> VoteOption:
        """Board member casts their vote"""
        analysis = analyses[member.role]
        support_level = analysis["support_level"]

        # CEO can veto high-risk proposals
        if member.veto_power and proposal.impact_analysis.get("risk_level", 0) > 0.9:
            if member.personality_traits.get("risk_tolerance", 0.5) < 0.3:
                return VoteOption.VETO

        # Convert support level to vote
        if support_level >= 0.8:
            return VoteOption.STRONGLY_APPROVE
        elif support_level >= 0.4:
            return VoteOption.APPROVE
        elif support_level >= -0.4:
            return VoteOption.ABSTAIN
        elif support_level >= -0.8:
            return VoteOption.REJECT
        else:
            return VoteOption.STRONGLY_REJECT

    def _calculate_consensus(self, votes: dict[BoardRole, VoteOption]) -> float:
        """Calculate consensus level from votes"""
        # Check for veto
        if any(v == VoteOption.VETO for v in votes.values()):
            return -1.0  # Veto blocks consensus

        # Calculate weighted vote sum
        total_weight = 0
        weighted_sum = 0

        for role, vote in votes.items():
            member = self.board_members[role]
            weight = member.decision_weight
            total_weight += weight
            weighted_sum += vote.value * weight

        # Normalize to 0-1 scale
        consensus = (weighted_sum / total_weight + 2) / 4
        return max(0, min(1, consensus))

    def _determine_decision(self, votes: dict[BoardRole, VoteOption], consensus: float) -> str:
        """Determine final decision based on votes and consensus"""
        if consensus < 0:  # Veto
            return "rejected"
        elif consensus >= 0.6:  # Strong consensus
            return "approved"
        elif consensus >= 0.4:  # Weak consensus
            return "approved_with_conditions"
        elif consensus >= 0.25:  # No consensus
            return "deferred"
        else:
            return "rejected"

    def _create_implementation_plan(self, proposal: Proposal, analyses: dict) -> dict[str, Any]:
        """Create implementation plan for approved proposal"""
        plan = {
            "phases": [],
            "responsible_parties": [],
            "timeline": proposal.timeline,
            "success_metrics": [],
            "risk_mitigation": []
        }

        # Assign responsible board members
        for role, member in self.board_members.items():
            if any(domain in proposal.type.value for domain in member.authority_domains):
                plan["responsible_parties"].append({
                    "role": role.value,
                    "member": member.name,
                    "responsibilities": member.responsibilities[:2]
                })

        # Define phases based on proposal type
        if proposal.type == ProposalType.STRATEGIC:
            plan["phases"] = ["Planning", "Pilot", "Rollout", "Optimization"]
        elif proposal.type == ProposalType.FINANCIAL:
            plan["phases"] = ["Budget Allocation", "Implementation", "Monitoring"]
        else:
            plan["phases"] = ["Preparation", "Execution", "Review"]

        # Add success metrics
        plan["success_metrics"] = [
            "ROI > 20%",
            "Implementation within timeline",
            "No critical issues"
        ]

        return plan

    def _extract_dissent(self, votes: dict[BoardRole, VoteOption],
                        analyses: dict) -> list[str]:
        """Extract dissenting opinions"""
        dissent = []

        for role, vote in votes.items():
            if vote.value < 0:  # Negative vote
                member = self.board_members[role]
                analysis = analyses.get(role, {}) if isinstance(analyses, dict) else {}
                concerns = analysis.get("concerns") or analysis.get("risks") or []
                if isinstance(concerns, str):
                    concerns = [concerns]
                if concerns:
                    dissent.append(f"{member.name}: {concerns[0]}")

        return dissent

    def _extract_conditions(self, analyses: dict) -> list[str]:
        """Extract conditions for approval"""
        conditions = []

        if not isinstance(analyses, dict):
            return conditions

        for role, analysis in analyses.items():
            member = self.board_members.get(role)
            if not member:
                continue

            extracted = analysis.get("conditions") or []
            if isinstance(extracted, str):
                extracted = [extracted]

            if extracted:
                for cond in extracted[:5]:
                    conditions.append(f"{member.name}: {cond}")
                continue

            # Backward-compatible fallback for old deterministic analysis shape
            concerns = analysis.get("concerns") or []
            support_level = analysis.get("support_level")
            if concerns and isinstance(support_level, (int, float)) and support_level > 0:
                conditions.append(f"Address {member.name}'s concern: {concerns[0]}")

        return conditions

    async def _identify_issues_for_discussion(self) -> list[Proposal]:
        """Identify issues that need board attention"""
        proposals = []

        try:
            with self._get_db_context() as conn:
                cur = conn.cursor(cursor_factory=RealDictCursor)

                # Check overdue invoices
                cur.execute("""
                SELECT COUNT(*) as count, SUM(amount_due) as total
                FROM invoices
                WHERE due_date < NOW() - INTERVAL '30 days'
                  AND status != 'paid'
                """)
                overdue = cur.fetchone()

                if overdue['count'] > 10 or (overdue['total'] and overdue['total'] > 50000):
                    proposal = Proposal(
                        id=f"auto-{datetime.now().timestamp()}",
                        type=ProposalType.FINANCIAL,
                        title="Address Overdue Invoices Crisis",
                        description=f"{overdue['count']} invoices totaling ${overdue['total']} are severely overdue",
                        proposed_by="System",
                        impact_analysis={"financial_risk": "high", "cash_flow_impact": overdue['total']},
                        required_resources={"collection_agents": 3, "legal_support": True},
                        timeline="Immediate",
                        alternatives=["Collection agency", "Payment plans", "Legal action"],
                        supporting_data={"overdue_details": dict(overdue)},
                        urgency=9,
                        created_at=datetime.now()
                    )
                    proposals.append(proposal)

                cur.close()
        except Exception as e:
            logger.error(f"Failed to identify issues: {e}")

        return proposals

    def _record_meeting_start(self, meeting_type: str, agenda: list[Proposal]) -> str:
        """Record meeting start in database"""
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()

                attendees = [m.name for m in self.board_members.values()]
                agenda_items = [{"id": p.id, "title": p.title} for p in agenda]

                cur.execute("""
                INSERT INTO ai_board_meetings
                (meeting_type, attendees, agenda)
                VALUES (%s, %s, %s)
                RETURNING id
                """, (meeting_type, Json(attendees), Json(agenda_items)))

                meeting_id = str(cur.fetchone()[0])
                conn.commit()
                cur.close()

                return meeting_id

        except Exception as e:
            logger.error(f"Failed to record meeting start: {e}")
            return f"meeting-{datetime.now().timestamp()}"

    def _record_meeting_end(self, meeting_id: str, decisions: list[BoardDecision],
                           outcomes: dict, duration: float):
        """Record meeting end in database"""
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()

                decision_summary = [
                    {"proposal": d.proposal_id, "decision": d.decision}
                    for d in decisions
                ]

                cur.execute("""
                UPDATE ai_board_meetings
                SET decisions_made = %s,
                    duration_minutes = %s
                WHERE id = %s
                """, (Json(decision_summary), int(duration), meeting_id))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to record meeting end: {e}")

    def _calculate_decision_confidence(
        self,
        votes: dict[BoardRole, VoteOption],
        consensus: float,
        latest_parsed: dict[BoardRole, dict[str, Any]],
        debate_transcript: dict[str, Any]
    ) -> float:
        """Calculate overall confidence in the board decision"""
        confidence_factors = []

        # Consensus strength (higher consensus = higher confidence)
        consensus_confidence = consensus
        confidence_factors.append(consensus_confidence)

        # Vote distribution (unanimous = higher confidence)
        vote_values = [v.value for v in votes.values()]
        vote_std = (sum((v - sum(vote_values)/len(vote_values))**2 for v in vote_values) / len(vote_values))**0.5
        vote_uniformity = 1.0 - min(vote_std / 2.0, 1.0)  # Lower std = higher uniformity
        confidence_factors.append(vote_uniformity)

        # Individual member confidence (from parsed responses)
        member_confidences = []
        for role, parsed in latest_parsed.items():
            if isinstance(parsed, dict) and 'confidence' in parsed:
                try:
                    member_confidences.append(float(parsed['confidence']))
                except (ValueError, TypeError):
                    logger.debug("Invalid member confidence for %s", role)

        if member_confidences:
            avg_member_confidence = sum(member_confidences) / len(member_confidences)
            confidence_factors.append(avg_member_confidence)

        # Debate quality (more rounds = more thorough = higher confidence)
        num_rounds = len(debate_transcript.get('rounds', []))
        debate_confidence = min(num_rounds / 3.0, 1.0)  # 3+ rounds = max confidence
        confidence_factors.append(debate_confidence)

        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def _assess_proposal_risk(
        self,
        proposal: Proposal,
        votes: dict[BoardRole, VoteOption],
        latest_parsed: dict[BoardRole, dict[str, Any]]
    ) -> dict[str, Any]:
        """Comprehensive risk assessment of the proposal"""
        risk_assessment = {
            'overall_risk': 0.0,
            'risk_categories': {},
            'risk_factors': [],
            'mitigation_strategies': []
        }

        # Financial risk
        financial_risk = proposal.impact_analysis.get('cost', 0) / 100000  # Normalize
        risk_assessment['risk_categories']['financial'] = min(financial_risk, 1.0)
        if financial_risk > 0.7:
            risk_assessment['risk_factors'].append(f"High financial commitment: ${proposal.impact_analysis.get('cost', 0)}")

        # Implementation risk (from urgency and complexity)
        urgency_risk = proposal.urgency / 10.0
        risk_assessment['risk_categories']['urgency'] = urgency_risk
        if urgency_risk > 0.7:
            risk_assessment['risk_factors'].append("High urgency may compromise quality")

        # Strategic risk (from proposal type)
        strategic_risks = {
            ProposalType.STRATEGIC: 0.8,
            ProposalType.FINANCIAL: 0.7,
            ProposalType.EMERGENCY: 0.9,
            ProposalType.TECHNICAL: 0.6,
            ProposalType.OPERATIONAL: 0.4,
            ProposalType.MARKETING: 0.5,
            ProposalType.POLICY: 0.6
        }
        strategic_risk = strategic_risks.get(proposal.type, 0.5)
        risk_assessment['risk_categories']['strategic'] = strategic_risk

        # Board concerns (from dissenting votes and member risks)
        dissent_count = sum(1 for v in votes.values() if v.value < 0)
        dissent_risk = dissent_count / max(len(votes), 1)
        risk_assessment['risk_categories']['dissent'] = dissent_risk

        # Member-identified risks
        all_risks = []
        for role, parsed in latest_parsed.items():
            if isinstance(parsed, dict) and 'risks' in parsed:
                member_risks = parsed['risks']
                if isinstance(member_risks, list):
                    all_risks.extend(member_risks)

        if all_risks:
            risk_assessment['risk_factors'].extend(all_risks[:5])  # Top 5
            risk_assessment['mitigation_strategies'].append("Address specific member concerns before implementation")

        # Calculate overall risk
        risk_values = list(risk_assessment['risk_categories'].values())
        risk_assessment['overall_risk'] = sum(risk_values) / len(risk_values) if risk_values else 0.5

        return risk_assessment

    def _evaluate_decision_criteria(
        self,
        proposal: Proposal,
        votes: dict[BoardRole, VoteOption],
        latest_parsed: dict[BoardRole, dict[str, Any]]
    ) -> dict[str, float]:
        """Evaluate decision across multiple criteria"""
        criteria_scores = {}

        # Strategic alignment (CEO weight matters most)
        ceo_vote = votes.get(BoardRole.CEO, VoteOption.ABSTAIN)
        criteria_scores['strategic_alignment'] = (ceo_vote.value + 2) / 4  # Normalize to 0-1

        # Financial viability (CFO assessment)
        cfo_vote = votes.get(BoardRole.CFO, VoteOption.ABSTAIN)
        criteria_scores['financial_viability'] = (cfo_vote.value + 2) / 4

        # Operational feasibility (COO assessment)
        coo_vote = votes.get(BoardRole.COO, VoteOption.ABSTAIN)
        criteria_scores['operational_feasibility'] = (coo_vote.value + 2) / 4

        # Market opportunity (CMO assessment)
        cmo_vote = votes.get(BoardRole.CMO, VoteOption.ABSTAIN)
        criteria_scores['market_opportunity'] = (cmo_vote.value + 2) / 4

        # Technical readiness (CTO assessment)
        cto_vote = votes.get(BoardRole.CTO, VoteOption.ABSTAIN)
        criteria_scores['technical_readiness'] = (cto_vote.value + 2) / 4

        # Overall board support (average of all votes)
        avg_vote = sum(v.value for v in votes.values()) / len(votes)
        criteria_scores['board_support'] = (avg_vote + 2) / 4

        # ROI potential (from proposal data)
        roi_potential = proposal.impact_analysis.get('revenue_potential', 0) / max(proposal.impact_analysis.get('cost', 1), 1)
        criteria_scores['roi_potential'] = min(roi_potential / 5.0, 1.0)  # Normalize, 5x ROI = max score

        return criteria_scores

    def _check_board_escalation(
        self,
        proposal: Proposal,
        confidence_score: float,
        risk_assessment: dict[str, Any],
        consensus: float
    ) -> tuple[bool, Optional[str]]:
        """Determine if human (owner) escalation is required"""
        escalation_reasons = []

        # Low confidence decisions
        if confidence_score < 0.5:
            escalation_reasons.append(f"Low board confidence: {confidence_score:.1%}")

        # High risk decisions
        if risk_assessment['overall_risk'] > 0.7:
            escalation_reasons.append(f"High risk: {risk_assessment['overall_risk']:.1%}")

        # Low consensus (board split)
        if consensus < 0.4:
            escalation_reasons.append(f"Low consensus: {consensus:.1%}")

        # High-value financial decisions
        if proposal.impact_analysis.get('cost', 0) > 100000:
            escalation_reasons.append(f"High financial commitment: ${proposal.impact_analysis.get('cost', 0)}")

        # Emergency decisions (always escalate for transparency)
        if proposal.type == ProposalType.EMERGENCY:
            escalation_reasons.append("Emergency proposal requires owner notification")

        # Strategic decisions with < 60% consensus
        if proposal.type == ProposalType.STRATEGIC and consensus < 0.6:
            escalation_reasons.append("Strategic decision without strong consensus")

        human_escalation = len(escalation_reasons) > 0
        escalation_reason = "; ".join(escalation_reasons) if escalation_reasons else None

        return human_escalation, escalation_reason

    def _record_decision(self, decision: BoardDecision):
        """Record decision in database with enhanced fields"""
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()

                vote_results = {k.value: v.value for k, v in decision.vote_results.items()}

                cur.execute("""
                INSERT INTO ai_board_decisions
                (proposal_id, decision, vote_results, consensus_level,
                 dissenting_opinions, conditions, implementation_plan, debate_transcript, follow_up_date,
                 confidence_score, risk_assessment, human_escalation_required, escalation_reason,
                 decision_criteria_scores)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    decision.proposal_id,
                    decision.decision,
                    Json(vote_results),
                    decision.consensus_level,
                    Json(decision.dissenting_opinions),
                    Json(decision.conditions),
                    Json(decision.implementation_plan),
                    Json(decision.debate_transcript),
                    decision.follow_up_date,
                    decision.confidence_score,
                    Json(decision.risk_assessment),
                    decision.human_escalation_required,
                    decision.escalation_reason,
                    Json(decision.decision_criteria_scores)
                ))

                # ALSO persist to ai_decisions table for centralized visibility
                self._persist_board_decision_to_ai_decisions(cur, decision, vote_results)

                # Update proposal status
                new_status = 'deferred' if decision.decision == 'deferred' else 'decided'
                cur.execute("""
                UPDATE ai_board_proposals
                SET status = %s
                WHERE id = %s
                """, (new_status, decision.proposal_id))

                conn.commit()
                cur.close()

        except Exception as e:
            logger.error(f"Failed to record decision: {e}")

    def _persist_board_decision_to_ai_decisions(self, cur, decision: BoardDecision, vote_results: dict) -> None:
        """
        Persist board decision to the ai_decisions table for centralized visibility.
        This ensures all AI Board governance decisions are tracked in the main AI decisions table.

        Schema: id, agent_id, decision_type, input_data, output_data, confidence, timestamp
        """
        try:
            # Build comprehensive input_data
            input_data = {
                "proposal_id": decision.proposal_id,
                "vote_results": vote_results,
                "consensus_level": decision.consensus_level,
                "dissenting_opinions": decision.dissenting_opinions,
                "decision_criteria_scores": decision.decision_criteria_scores
            }

            # Build output_data with reasoning (WHY the decision was made)
            output_data = {
                "decision": decision.decision,
                "reasoning": self._generate_board_decision_reasoning(decision, vote_results),
                "conditions": decision.conditions,
                "implementation_plan": decision.implementation_plan,
                "risk_assessment": decision.risk_assessment,
                "human_escalation_required": decision.human_escalation_required,
                "escalation_reason": decision.escalation_reason,
                "follow_up_date": decision.follow_up_date.isoformat() if decision.follow_up_date else None
            }

            cur.execute("""
                INSERT INTO ai_decisions
                (agent_id, decision_type, input_data, output_data, confidence, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (
                "AIBoard",  # agent_id
                f"board_{decision.decision}",  # decision_type (board_approved, board_rejected, etc.)
                Json(input_data),  # input_data
                Json(output_data),  # output_data
                decision.confidence_score  # confidence
            ))

            logger.debug(f"Board decision for proposal {decision.proposal_id} also logged to ai_decisions table")

        except Exception as e:
            # Don't fail the main decision logging if ai_decisions insert fails
            logger.warning(f"Failed to persist board decision to ai_decisions: {e}")

    def _generate_board_decision_reasoning(self, decision: BoardDecision, vote_results: dict) -> str:
        """
        Generate human-readable reasoning explaining WHY this board decision was made.
        This provides transparency into the AI Board's governance process.
        """
        reasoning_parts = []

        # Decision outcome
        reasoning_parts.append(f"DECISION: {decision.decision.upper()}")

        # Consensus analysis
        if decision.consensus_level >= 0.9:
            reasoning_parts.append(f"STRONG CONSENSUS ({decision.consensus_level:.0%}): Near-unanimous agreement among board members")
        elif decision.consensus_level >= 0.7:
            reasoning_parts.append(f"GOOD CONSENSUS ({decision.consensus_level:.0%}): Clear majority with some dissent")
        elif decision.consensus_level >= 0.5:
            reasoning_parts.append(f"MODERATE CONSENSUS ({decision.consensus_level:.0%}): Split opinions, decision reached by majority")
        else:
            reasoning_parts.append(f"LOW CONSENSUS ({decision.consensus_level:.0%}): Significant disagreement, may require escalation")

        # Vote breakdown
        vote_summary = ", ".join([f"{role}: {vote}" for role, vote in vote_results.items()])
        reasoning_parts.append(f"Votes: {vote_summary}")

        # Dissenting opinions
        if decision.dissenting_opinions:
            reasoning_parts.append(f"Dissenting views: {len(decision.dissenting_opinions)} member(s) expressed concerns")

        # Confidence
        if decision.confidence_score >= 0.85:
            reasoning_parts.append("HIGH CONFIDENCE in decision outcome")
        elif decision.confidence_score >= 0.65:
            reasoning_parts.append("GOOD CONFIDENCE in decision outcome")
        else:
            reasoning_parts.append("MODERATE CONFIDENCE - further review may be beneficial")

        # Risk assessment
        if decision.risk_assessment:
            risk_level = decision.risk_assessment.get("overall_risk_level", "unknown")
            reasoning_parts.append(f"Risk assessment: {risk_level}")

        # Human escalation
        if decision.human_escalation_required:
            reasoning_parts.append(f"REQUIRES HUMAN REVIEW: {decision.escalation_reason or 'Governance threshold exceeded'}")

        # Conditions
        if decision.conditions:
            reasoning_parts.append(f"Conditional approval with {len(decision.conditions)} condition(s)")

        return " | ".join(reasoning_parts)

    def _synthesize_outcomes(self, decisions: list[BoardDecision]) -> dict[str, Any]:
        """Synthesize meeting outcomes"""
        outcomes = {
            "total_decisions": len(decisions),
            "approved": len([d for d in decisions if d.decision == "approved"]),
            "rejected": len([d for d in decisions if d.decision == "rejected"]),
            "deferred": len([d for d in decisions if d.decision == "deferred"]),
            "average_consensus": sum(d.consensus_level for d in decisions) / len(decisions) if decisions else 0,
            "action_items": []
        }

        # Extract action items from approved decisions
        for decision in decisions:
            if decision.decision == "approved" and decision.implementation_plan:
                outcomes["action_items"].append({
                    "proposal": decision.proposal_id,
                    "plan": decision.implementation_plan.get("phases", []),
                    "deadline": decision.follow_up_date.isoformat() if decision.follow_up_date else None
                })

        return outcomes

    async def submit_proposal(self, proposal: Proposal) -> str:
        """Submit a proposal for board consideration"""
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()

                cur.execute("""
                INSERT INTO ai_board_proposals
                (proposal_type, title, description, proposed_by, impact_analysis,
                 required_resources, timeline, alternatives, supporting_data, urgency, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending')
                RETURNING id
                """, (
                    proposal.type.value,
                    proposal.title,
                    proposal.description,
                    proposal.proposed_by,
                    Json(proposal.impact_analysis),
                    Json(proposal.required_resources),
                    proposal.timeline,
                    Json(proposal.alternatives),
                    Json(proposal.supporting_data),
                    proposal.urgency
                ))

                proposal_id = str(cur.fetchone()[0])
                conn.commit()
                cur.close()

                logger.info(f"📝 Proposal submitted: {proposal.title} (ID: {proposal_id})")
                return proposal_id

        except Exception as e:
            logger.error(f"Failed to submit proposal: {e}")
            raise

    def get_board_status(self) -> dict[str, Any]:
        """Get current board status"""
        status = {
            "board_members": {},
            "meeting_in_progress": self.meeting_in_progress,
            "pending_proposals": 0,
            "recent_decisions": []
        }

        # Board member status
        for role, member in self.board_members.items():
            status["board_members"][role.value] = {
                "name": member.name,
                "current_focus": member.current_focus,
                "last_decision": member.last_decision
            }

        # Get pending proposals count
        try:
            with self._get_db_context() as conn:
                cur = conn.cursor()

                cur.execute("SELECT COUNT(*) FROM ai_board_proposals WHERE status = 'pending'")
                status["pending_proposals"] = cur.fetchone()[0]

                # Get recent decisions
                cur.execute("""
                SELECT decision, consensus_level, decided_at
                FROM ai_board_decisions
                ORDER BY decided_at DESC
                LIMIT 5
                """)

                for row in cur.fetchall():
                    status["recent_decisions"].append({
                        "decision": row[0],
                        "consensus": row[1],
                        "date": row[2].isoformat() if row[2] else None
                    })

                cur.close()

        except Exception as e:
            logger.error(f"Failed to get board status: {e}")

        return status


# Global board instance
ai_board_instance = None

def get_ai_board() -> AIBoardOfDirectors:
    """Get or create the singleton AI Board instance"""
    global ai_board_instance
    if ai_board_instance is None:
        ai_board_instance = AIBoardOfDirectors()
    return ai_board_instance


async def run_board_meeting():
    """Run a board meeting"""
    board = get_ai_board()
    result = await board.convene_meeting("regular")
    return result


if __name__ == "__main__":
    # Test the AI Board
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     🏛️  AI Board of Directors                           ║
    ║     Autonomous Business Governance System               ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    board = get_ai_board()

    # Test proposal submission
    test_proposal = Proposal(
        id="test-1",
        type=ProposalType.STRATEGIC,
        title="Expand AI Agent Fleet to 100 Agents",
        description="Scale our AI infrastructure to handle increased demand",
        proposed_by="AUREA",
        impact_analysis={
            "cost": 50000,
            "revenue_potential": 250000,
            "risk_level": 0.4,
            "implementation_time": "3 months"
        },
        required_resources={
            "developers": 2,
            "compute_resources": "increased",
            "budget": 50000
        },
        timeline="Q1 2025",
        alternatives=["Gradual scaling", "Outsource to third-party", "Status quo"],
        supporting_data={
            "current_utilization": "87%",
            "growth_rate": "23% monthly",
            "competitor_analysis": "Falling behind"
        },
        urgency=7,
        created_at=datetime.now()
    )

    # Run async test
    async def test():
        # Submit proposal
        proposal_id = await board.submit_proposal(test_proposal)
        print(f"✅ Proposal submitted: {proposal_id}")

        # Run board meeting
        result = await board.convene_meeting()
        print(f"✅ Board meeting completed: {json.dumps(result, indent=2)}")

        # Get status
        status = board.get_board_status()
        print(f"📊 Board Status: {json.dumps(status, indent=2)}")

    asyncio.run(test())

    print("\n✅ AI Board of Directors operational!")
