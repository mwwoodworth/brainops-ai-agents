"""
BrainOps Agent Memory SDK
==========================
Standard client library for all agents to interact with memory.
Enforces read-before-act and write-back-after patterns.

Usage:
    from agent_memory_sdk import AgentMemoryClient

    async with AgentMemoryClient("my-agent-id") as client:
        # Mandatory: Retrieve context before acting
        context = await client.retrieve_context("customer inquiry about pricing")

        # Perform agent work...
        result = await do_agent_work(context)

        # Mandatory: Write back decision/action
        await client.write_decision(
            title="Pricing recommendation provided",
            content=result,
            object_type="decision",
            proof_type="log"
        )

Part of BrainOps OS Total Completion Protocol.
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from memory_enforcement import (
    EnforcementContext,
    EnforcementResult,
    EvidenceLevel,
    MemoryContract,
    MemoryEnforcementEngine,
    MemoryObjectType,
    MemoryVerification,
    OperationType,
    VerificationProof,
    VerificationState,
    get_enforcement_engine,
)

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")


class AgentMemoryClient:
    """
    Standard memory client for BrainOps agents.

    Enforces:
    - Read-before-act (RBA): Context retrieval required before actions
    - Write-back-after (WBA): Decision/action persistence required after completion

    Blocks:
    - Direct database writes without mediation
    - Operations without correlation tracking
    - Unverified high-impact decisions
    """

    def __init__(
        self,
        agent_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        enforce_rba: bool = True,
        enforce_wba: bool = True,
        allow_bypass: bool = False
    ):
        self.agent_id = agent_id
        self.tenant_id = tenant_id
        self.enforce_rba = enforce_rba
        self.enforce_wba = enforce_wba
        self.allow_bypass = allow_bypass

        self.correlation_id = str(uuid.uuid4())
        self.engine: Optional[MemoryEnforcementEngine] = None

        # Tracking
        self._rba_completed = False
        self._wba_completed = False
        self._retrieved_memory_ids: list[str] = []
        self._written_memory_ids: list[str] = []
        self._context: EnforcementContext = None

    async def __aenter__(self):
        """Initialize the client and enforcement engine"""
        self.engine = get_enforcement_engine()
        await self.engine.initialize()

        self._context = EnforcementContext(
            correlation_id=self.correlation_id,
            agent_id=self.agent_id,
            tenant_id=self.tenant_id,
            operation=OperationType.READ,
            rba_required=self.enforce_rba,
            wba_required=self.enforce_wba
        )

        logger.info(f"[{self.agent_id}] Memory client initialized | correlation_id={self.correlation_id}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Verify enforcement was completed before exiting"""
        if exc_type is not None:
            # Exception occurred - log but don't block
            logger.warning(
                f"[{self.agent_id}] Agent exiting with exception | "
                f"rba_completed={self._rba_completed} | wba_completed={self._wba_completed}"
            )
            return False

        # Verify RBA was completed
        if self.enforce_rba and not self._rba_completed:
            if not self.allow_bypass:
                logger.error(f"[{self.agent_id}] RBA enforcement violation - context not retrieved")
                raise EnforcementViolationError(
                    f"Agent '{self.agent_id}' did not retrieve context before acting. "
                    f"Call client.retrieve_context() before performing actions."
                )
            else:
                logger.warning(f"[{self.agent_id}] RBA not completed but bypass allowed")

        # Verify WBA was completed
        if self.enforce_wba and not self._wba_completed:
            if not self.allow_bypass:
                logger.error(f"[{self.agent_id}] WBA enforcement violation - decision not written")
                raise EnforcementViolationError(
                    f"Agent '{self.agent_id}' did not write back decision/action. "
                    f"Call client.write_decision() after completing actions."
                )
            else:
                logger.warning(f"[{self.agent_id}] WBA not completed but bypass allowed")

        logger.info(
            f"[{self.agent_id}] Memory client closed | "
            f"retrieved={len(self._retrieved_memory_ids)} | written={len(self._written_memory_ids)}"
        )
        return False

    # =========================================================================
    # RETRIEVE METHODS (RBA)
    # =========================================================================

    async def retrieve_context(
        self,
        query: str,
        filters: Optional[dict] = None,
        include_verified_only: bool = False
    ) -> list[dict]:
        """
        Retrieve relevant memory context before acting.

        This method MUST be called before any state-changing operation.
        Failure to call this method will result in enforcement violation.

        Args:
            query: Semantic query describing what context is needed
            filters: Optional filters (memory_type, category, etc.)
            include_verified_only: Only return VERIFIED memories

        Returns:
            List of relevant memories with verification status
        """
        self._context.operation = OperationType.READ

        result = await self.engine.enforce_read_before_act(
            context=self._context,
            query=query,
            filters=filters
        )

        if result.blocked:
            raise EnforcementBlockedError(result.block_reason)

        self._rba_completed = True
        self._retrieved_memory_ids.extend(result.memory_ids_retrieved)

        # Log warnings
        for warning in result.warnings:
            logger.warning(f"[{self.agent_id}] RBA warning: {warning}")

        # Filter by verification if requested
        if include_verified_only:
            # Re-query with verification filter
            memories = await self._get_memories_by_ids(result.memory_ids_retrieved)
            return [m for m in memories if m.get("verification_state") == "VERIFIED"]

        return await self._get_memories_by_ids(result.memory_ids_retrieved)

    async def retrieve_by_type(
        self,
        object_type: MemoryObjectType,
        limit: int = 20
    ) -> list[dict]:
        """
        Retrieve memories by object type.

        Args:
            object_type: The type of memory objects to retrieve
            limit: Maximum number of results

        Returns:
            List of memories of the specified type
        """
        try:
            rows = await self.engine.pool.fetch("""
                SELECT
                    id::text,
                    memory_type,
                    object_type,
                    content,
                    verification_state,
                    confidence_score,
                    owner,
                    project,
                    created_at
                FROM unified_ai_memory
                WHERE object_type = $1::memory_object_type
                    AND (tenant_id = $2::uuid OR tenant_id IS NULL)
                    AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY importance_score DESC, created_at DESC
                LIMIT $3
            """, object_type.value, self.tenant_id, limit)

            self._rba_completed = True
            return [dict(r) for r in rows]

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to retrieve by type: {e}")
            return []

    async def retrieve_related(
        self,
        memory_id: str,
        relationship: str = "all"
    ) -> list[dict]:
        """
        Retrieve memories related to a specific memory.

        Args:
            memory_id: The memory to find relationships for
            relationship: Type of relationship (supersedes, parent, related)

        Returns:
            List of related memories
        """
        try:
            if relationship == "supersedes":
                rows = await self.engine.pool.fetch("""
                    SELECT id::text, memory_type, content, verification_state
                    FROM unified_ai_memory
                    WHERE $1::uuid = ANY(supersedes)
                    LIMIT 100
                """, memory_id)
            elif relationship == "parent":
                rows = await self.engine.pool.fetch("""
                    SELECT id::text, memory_type, content, verification_state
                    FROM unified_ai_memory
                    WHERE id = (
                        SELECT parent_memory_id FROM unified_ai_memory WHERE id = $1::uuid
                    )
                    LIMIT 1
                """, memory_id)
            else:
                rows = await self.engine.pool.fetch("""
                    SELECT id::text, memory_type, content, verification_state
                    FROM unified_ai_memory
                    WHERE $1::uuid = ANY(related_memories)
                       OR id = ANY(
                           SELECT UNNEST(related_memories) FROM unified_ai_memory WHERE id = $1::uuid
                       )
                    LIMIT 100
                """, memory_id)

            self._rba_completed = True
            return [dict(r) for r in rows]

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to retrieve related: {e}")
            return []

    # =========================================================================
    # WRITE METHODS (WBA)
    # =========================================================================

    async def write_decision(
        self,
        title: str,
        content: Any,
        object_type: MemoryObjectType = MemoryObjectType.DECISION,
        proof_type: str = "log",
        proof_content: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        owner: Optional[str] = None,
        project: Optional[str] = None,
        supersedes: Optional[list[str]] = None,
        evidence_level: EvidenceLevel = EvidenceLevel.E1_RECORDED
    ) -> str:
        """
        Write a decision/action back to memory.

        This method MUST be called after completing any significant action.
        Failure to call this method will result in enforcement violation.

        Args:
            title: Short title for the decision
            content: The decision content (can be dict, str, or any JSON-serializable)
            object_type: Classification of the memory object
            proof_type: Type of proof artifact (log, test_result, screenshot, etc.)
            proof_content: Inline proof content
            tags: Tags for categorization
            owner: Responsible party
            project: Project association
            supersedes: IDs of memories this supersedes
            evidence_level: The evidence level for verification

        Returns:
            The ID of the created memory
        """
        # Create verification with appropriate state
        verification = MemoryVerification(
            state=VerificationState.VERIFIED if evidence_level.value >= EvidenceLevel.E2_TESTED.value else VerificationState.UNVERIFIED,
            evidence_level=evidence_level,
            last_verified_at=datetime.utcnow(),
            verified_by=self.agent_id,
            confidence_score=self._evidence_to_confidence(evidence_level)
        )

        # Create memory contract
        contract = MemoryContract(
            type=object_type,
            title=title,
            content=content,
            source="agent",
            project=project,
            owner=owner or self.agent_id,
            tags=tags or [],
            supersedes=supersedes or [],
            verification=verification,
            metadata={
                "agent_id": self.agent_id,
                "correlation_id": self.correlation_id,
                "retrieved_memories": self._retrieved_memory_ids
            }
        )

        # Create proof if provided
        proof = None
        if proof_content:
            proof = VerificationProof(
                artifact_type=proof_type,
                artifact_content=proof_content,
                evidence_level=evidence_level,
                created_by=self.agent_id
            )

        # Enforce WBA
        self._context.operation = OperationType.WRITE
        result = await self.engine.enforce_write_back_after(
            context=self._context,
            memory_contract=contract,
            proof=proof
        )

        if result.blocked:
            raise EnforcementBlockedError(result.block_reason)

        self._wba_completed = True
        self._written_memory_ids.extend(result.memory_ids_written)

        return result.memory_ids_written[0] if result.memory_ids_written else None

    async def write_sop(
        self,
        title: str,
        procedure: dict,
        owner: str,
        project: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> str:
        """
        Write a standard operating procedure to memory.

        Args:
            title: SOP title
            procedure: The procedure content (steps, conditions, etc.)
            owner: Responsible owner
            project: Project association
            tags: Tags for categorization

        Returns:
            The ID of the created SOP memory
        """
        return await self.write_decision(
            title=title,
            content=procedure,
            object_type=MemoryObjectType.SOP,
            proof_type="documentation",
            tags=tags or ["sop"],
            owner=owner,
            project=project,
            evidence_level=EvidenceLevel.E1_RECORDED
        )

    async def write_incident(
        self,
        title: str,
        incident_details: dict,
        severity: str = "medium",
        owner: Optional[str] = None
    ) -> str:
        """
        Write an incident report to memory.

        Args:
            title: Incident title
            incident_details: The incident details
            severity: Incident severity (low, medium, high, critical)
            owner: Responsible owner

        Returns:
            The ID of the created incident memory
        """
        return await self.write_decision(
            title=title,
            content={**incident_details, "severity": severity},
            object_type=MemoryObjectType.INCIDENT,
            proof_type="incident_log",
            tags=["incident", severity],
            owner=owner,
            evidence_level=EvidenceLevel.E1_RECORDED
        )

    async def verify_memory(
        self,
        memory_id: str,
        proof_type: str,
        proof_content: dict,
        evidence_level: EvidenceLevel = EvidenceLevel.E2_TESTED
    ) -> bool:
        """
        Verify an existing memory with proof.

        Args:
            memory_id: ID of memory to verify
            proof_type: Type of proof artifact
            proof_content: The proof content
            evidence_level: Level of evidence provided

        Returns:
            True if verification succeeded
        """
        proof = VerificationProof(
            artifact_type=proof_type,
            artifact_content=proof_content,
            evidence_level=evidence_level,
            created_by=self.agent_id
        )

        result = await self.engine.verify_memory(
            memory_id=memory_id,
            proof=proof,
            verified_by=self.agent_id,
            context=self._context
        )

        return result.success

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def get_truth_backlog(self, limit: int = 50) -> list[dict]:
        """
        Get memories needing verification.

        Returns:
            List of memories in the truth backlog
        """
        return await self.engine.get_truth_backlog(
            tenant_id=self.tenant_id,
            limit=limit
        )

    async def update_memory(
        self,
        memory_id: str,
        content: Optional[Any] = None,
        tags: Optional[list[str]] = None,
        owner: Optional[str] = None,
        project: Optional[str] = None
    ) -> bool:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update
            content: New content (optional)
            tags: New tags (optional)
            owner: New owner (optional)
            project: New project (optional)

        Returns:
            True if update succeeded
        """
        try:
            updates = []
            params = [memory_id]
            param_idx = 2

            if content is not None:
                content_json = json.dumps(content) if isinstance(content, dict) else json.dumps({"text": str(content)})
                updates.append(f"content = ${param_idx}::jsonb")
                params.append(content_json)
                param_idx += 1

            if tags is not None:
                updates.append(f"tags = ${param_idx}")
                params.append(tags)
                param_idx += 1

            if owner is not None:
                updates.append(f"owner = ${param_idx}")
                params.append(owner)
                param_idx += 1

            if project is not None:
                updates.append(f"project = ${param_idx}")
                params.append(project)
                param_idx += 1

            if not updates:
                return True

            updates.append("updated_at = NOW()")

            await self.engine.pool.execute(f"""
                UPDATE unified_ai_memory
                SET {", ".join(updates)}
                WHERE id = $1::uuid
            """, *params)

            return True

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to update memory: {e}")
            return False

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    async def _get_memories_by_ids(self, memory_ids: list[str]) -> list[dict]:
        """Get full memory objects by IDs"""
        if not memory_ids:
            return []

        try:
            rows = await self.engine.pool.fetch("""
                SELECT
                    id::text,
                    memory_type,
                    object_type,
                    content,
                    importance_score,
                    verification_state,
                    evidence_level,
                    confidence_score,
                    owner,
                    project,
                    tags,
                    created_at,
                    last_verified_at
                FROM unified_ai_memory
                WHERE id = ANY($1::uuid[])
            """, memory_ids)

            return [dict(r) for r in rows]

        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to get memories by IDs: {e}")
            return []

    def _evidence_to_confidence(self, level: EvidenceLevel) -> float:
        """Convert evidence level to confidence score"""
        mapping = {
            EvidenceLevel.E0_UNVERIFIED: 0.1,
            EvidenceLevel.E1_RECORDED: 0.3,
            EvidenceLevel.E2_TESTED: 0.7,
            EvidenceLevel.E3_OBSERVED: 0.85,
            EvidenceLevel.E4_AUDITED: 1.0
        }
        return mapping.get(level, 0.0)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class EnforcementViolationError(Exception):
    """Raised when an agent violates enforcement rules"""
    pass


class EnforcementBlockedError(Exception):
    """Raised when an operation is blocked by enforcement"""
    pass


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@asynccontextmanager
async def agent_memory_context(
    agent_id: str,
    tenant_id: str = DEFAULT_TENANT_ID,
    enforce_rba: bool = True,
    enforce_wba: bool = True
):
    """
    Context manager for agent memory operations.

    Usage:
        async with agent_memory_context("my-agent") as client:
            context = await client.retrieve_context("query")
            # ... do work ...
            await client.write_decision("title", result)
    """
    client = AgentMemoryClient(
        agent_id=agent_id,
        tenant_id=tenant_id,
        enforce_rba=enforce_rba,
        enforce_wba=enforce_wba
    )
    async with client:
        yield client


async def quick_retrieve(
    query: str,
    agent_id: str = "system",
    tenant_id: str = DEFAULT_TENANT_ID
) -> list[dict]:
    """
    Quick context retrieval without full enforcement.
    Use for read-only operations that don't change state.
    """
    async with AgentMemoryClient(
        agent_id=agent_id,
        tenant_id=tenant_id,
        enforce_rba=True,
        enforce_wba=False,
        allow_bypass=True
    ) as client:
        return await client.retrieve_context(query)


async def quick_write(
    title: str,
    content: Any,
    agent_id: str = "system",
    tenant_id: str = DEFAULT_TENANT_ID,
    object_type: MemoryObjectType = MemoryObjectType.DECISION
) -> str:
    """
    Quick decision write without full enforcement.
    Use for simple logging that doesn't require context retrieval.
    """
    async with AgentMemoryClient(
        agent_id=agent_id,
        tenant_id=tenant_id,
        enforce_rba=False,
        enforce_wba=True,
        allow_bypass=True
    ) as client:
        return await client.write_decision(
            title=title,
            content=content,
            object_type=object_type
        )
