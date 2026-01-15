"""
BrainOps Memory Enforcement System
===================================
Implements forced read-before-act (RBA) and write-back-after (WBA) patterns.
No agent or tool can bypass memory rules.

Part of BrainOps OS Total Completion Protocol.
"""

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID", "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457")


# =============================================================================
# ENUMS
# =============================================================================

class VerificationState(str, Enum):
    """Memory verification states"""
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    DEGRADED = "DEGRADED"
    BROKEN = "BROKEN"


class EvidenceLevel(str, Enum):
    """Evidence levels for proof standards"""
    E0_UNVERIFIED = "E0_UNVERIFIED"  # No artifacts, just claims
    E1_RECORDED = "E1_RECORDED"       # Logs captured + linked
    E2_TESTED = "E2_TESTED"           # Automated test results + env + timestamp + commit
    E3_OBSERVED = "E3_OBSERVED"       # Synthetic monitor / real usage observation + trend
    E4_AUDITED = "E4_AUDITED"         # Cross-checked evidence + rollback + owner signoff


class MemoryObjectType(str, Enum):
    """BrainOps OS memory object types"""
    DECISION = "decision"
    SOP = "sop"
    PROOF = "proof"
    TASK = "task"
    INCIDENT = "incident"
    KPI = "kpi"
    ARCHITECTURE = "architecture"
    INTEGRATION = "integration"
    EXPERIMENT = "experiment"
    RUNBOOK = "runbook"


class OperationType(str, Enum):
    """Memory operation types for audit"""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    VERIFY = "verify"
    SEARCH = "search"


# =============================================================================
# MODELS
# =============================================================================

class VerificationProof(BaseModel):
    """Verification proof artifact"""
    artifact_type: str
    artifact_url: Optional[str] = None
    artifact_hash: Optional[str] = None
    artifact_content: Optional[dict] = None
    evidence_level: EvidenceLevel = EvidenceLevel.E1_RECORDED
    created_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)


class MemoryVerification(BaseModel):
    """Verification object attached to memory"""
    state: VerificationState = VerificationState.UNVERIFIED
    evidence_level: EvidenceLevel = EvidenceLevel.E0_UNVERIFIED
    last_verified_at: Optional[datetime] = None
    verified_by: Optional[str] = None
    artifact_refs: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None


class MemoryContract(BaseModel):
    """BrainOps OS Memory Contract - the canonical schema"""
    id: Optional[str] = None
    type: MemoryObjectType
    title: str
    content: Any
    source: str = "agent"  # agent, human, automation, api, external
    project: Optional[str] = None
    owner: Optional[str] = None
    status: str = "active"  # active, superseded, deprecated
    supersedes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    verification: MemoryVerification = Field(default_factory=MemoryVerification)
    retrieval: dict = Field(default_factory=lambda: {
        "embedding_model": "text-embedding-3-small",
        "chunking_strategy": "none",
        "embedding_dim": 1536,
        "retrieval_policy_version": "v1.0"
    })
    metadata: dict = Field(default_factory=dict)


class EnforcementContext(BaseModel):
    """Context for enforcement operations"""
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None
    tenant_id: str = DEFAULT_TENANT_ID
    operation: OperationType
    rba_required: bool = True
    wba_required: bool = True
    bypass_reason: Optional[str] = None  # Only for emergency bypass


class EnforcementResult(BaseModel):
    """Result of enforcement operation"""
    success: bool
    operation: OperationType
    correlation_id: str
    memory_ids_retrieved: list[str] = Field(default_factory=list)
    memory_ids_written: list[str] = Field(default_factory=list)
    rba_enforced: bool = False
    wba_enforced: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# ENFORCEMENT ENGINE
# =============================================================================

class MemoryEnforcementEngine:
    """
    Core enforcement engine for BrainOps memory operations.

    Rules:
    1. Before any state-changing operation: retrieve relevant memory (RBA)
    2. After any decision/action: write back to memory (WBA)
    3. All operations are audited
    4. Verification state must be maintained
    """

    def __init__(self, pool=None):
        self.pool = pool
        self._initialized = False

    async def initialize(self):
        """Lazy initialization with database pool"""
        if not self._initialized:
            if self.pool is None:
                from database.async_connection import get_pool
                self.pool = get_pool()
            self._initialized = True

    async def enforce_read_before_act(
        self,
        context: EnforcementContext,
        query: str,
        filters: Optional[dict] = None
    ) -> EnforcementResult:
        """
        Enforce read-before-act pattern.
        Agent MUST retrieve relevant memory before taking action.
        """
        await self.initialize()

        result = EnforcementResult(
            success=False,
            operation=OperationType.READ,
            correlation_id=context.correlation_id
        )

        try:
            # Check if RBA is required
            if not context.rba_required:
                if context.bypass_reason:
                    result.warnings.append(f"RBA bypassed: {context.bypass_reason}")
                    result.success = True
                    return result
                else:
                    result.blocked = True
                    result.block_reason = "RBA required but not configured. Provide bypass_reason for emergency bypass."
                    await self._audit_operation(context, result)
                    return result

            # Perform memory retrieval
            retrieved = await self._retrieve_memory(
                query=query,
                filters=filters,
                tenant_id=context.tenant_id,
                limit=20
            )

            result.memory_ids_retrieved = [m["id"] for m in retrieved]
            result.rba_enforced = True
            result.success = True

            # Warn if no relevant memory found
            if not retrieved:
                result.warnings.append(
                    "No relevant memory found. Consider creating a memory capture task."
                )

            # Audit the operation
            await self._audit_operation(context, result)

            return result

        except Exception as e:
            logger.error(f"RBA enforcement failed: {e}")
            result.blocked = True
            result.block_reason = f"RBA enforcement error: {str(e)}"
            await self._audit_operation(context, result)
            return result

    async def enforce_write_back_after(
        self,
        context: EnforcementContext,
        memory_contract: MemoryContract,
        proof: Optional[VerificationProof] = None
    ) -> EnforcementResult:
        """
        Enforce write-back-after pattern.
        Agent MUST persist decision/action to memory after completion.
        """
        await self.initialize()

        result = EnforcementResult(
            success=False,
            operation=OperationType.WRITE,
            correlation_id=context.correlation_id
        )

        try:
            # Check if WBA is required
            if not context.wba_required:
                if context.bypass_reason:
                    result.warnings.append(f"WBA bypassed: {context.bypass_reason}")
                    result.success = True
                    return result
                else:
                    result.blocked = True
                    result.block_reason = "WBA required but not configured. Provide bypass_reason for emergency bypass."
                    await self._audit_operation(context, result)
                    return result

            # Validate memory contract
            if not memory_contract.title or not memory_contract.content:
                result.blocked = True
                result.block_reason = "Invalid memory contract: title and content required"
                await self._audit_operation(context, result)
                return result

            # Store the memory
            memory_id = await self._store_memory(
                contract=memory_contract,
                agent_id=context.agent_id,
                tenant_id=context.tenant_id,
                proof=proof
            )

            result.memory_ids_written = [memory_id]
            result.wba_enforced = True
            result.success = True

            # Audit the operation
            await self._audit_operation(context, result)

            return result

        except Exception as e:
            logger.error(f"WBA enforcement failed: {e}")
            result.blocked = True
            result.block_reason = f"WBA enforcement error: {str(e)}"
            await self._audit_operation(context, result)
            return result

    async def verify_memory(
        self,
        memory_id: str,
        proof: VerificationProof,
        verified_by: str,
        context: Optional[EnforcementContext] = None
    ) -> EnforcementResult:
        """
        Verify a memory with proof artifacts.
        Updates verification state based on evidence level.
        """
        await self.initialize()

        if context is None:
            context = EnforcementContext(
                operation=OperationType.VERIFY,
                agent_id=verified_by
            )

        result = EnforcementResult(
            success=False,
            operation=OperationType.VERIFY,
            correlation_id=context.correlation_id
        )

        try:
            # Determine verification state based on evidence level
            new_state = VerificationState.UNVERIFIED
            confidence = 0.0

            if proof.evidence_level == EvidenceLevel.E0_UNVERIFIED:
                new_state = VerificationState.UNVERIFIED
                confidence = 0.1
            elif proof.evidence_level == EvidenceLevel.E1_RECORDED:
                new_state = VerificationState.UNVERIFIED
                confidence = 0.3
            elif proof.evidence_level == EvidenceLevel.E2_TESTED:
                new_state = VerificationState.VERIFIED
                confidence = 0.7
            elif proof.evidence_level == EvidenceLevel.E3_OBSERVED:
                new_state = VerificationState.VERIFIED
                confidence = 0.85
            elif proof.evidence_level == EvidenceLevel.E4_AUDITED:
                new_state = VerificationState.VERIFIED
                confidence = 1.0

            # Calculate expiration (higher evidence = longer validity)
            expiry_days = {
                EvidenceLevel.E0_UNVERIFIED: 1,
                EvidenceLevel.E1_RECORDED: 7,
                EvidenceLevel.E2_TESTED: 30,
                EvidenceLevel.E3_OBSERVED: 60,
                EvidenceLevel.E4_AUDITED: 90
            }
            expires_at = datetime.utcnow() + timedelta(days=expiry_days.get(proof.evidence_level, 7))

            # Store proof artifact
            artifact_id = await self._store_proof_artifact(memory_id, proof)

            # Update memory verification
            await self.pool.execute("""
                UPDATE unified_ai_memory
                SET verification_state = $1::verification_state,
                    evidence_level = $2::evidence_level,
                    last_verified_at = NOW(),
                    verified_by = $3,
                    artifact_refs = array_append(artifact_refs, $4),
                    confidence_score = $5,
                    verification_expires_at = $6,
                    updated_at = NOW()
                WHERE id = $7::uuid
            """,
                new_state.value,
                proof.evidence_level.value,
                verified_by,
                artifact_id,
                confidence,
                expires_at,
                memory_id
            )

            result.memory_ids_written = [memory_id]
            result.success = True

            # Audit the operation
            await self._audit_operation(context, result)

            return result

        except Exception as e:
            logger.error(f"Memory verification failed: {e}")
            result.blocked = True
            result.block_reason = f"Verification error: {str(e)}"
            await self._audit_operation(context, result)
            return result

    async def get_truth_backlog(
        self,
        tenant_id: Optional[str] = None,
        limit: int = 50
    ) -> list[dict]:
        """
        Get the truth backlog: memories needing verification.
        Returns stale, unverified, conflicting, and ownerless items.
        """
        await self.initialize()

        try:
            rows = await self.pool.fetch("""
                SELECT * FROM memory_truth_backlog
                ORDER BY priority, open_conflicts DESC
                LIMIT $1
            """, limit)

            return [dict(r) for r in rows]

        except Exception as e:
            logger.error(f"Failed to get truth backlog: {e}")
            return []

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    async def _retrieve_memory(
        self,
        query: str,
        filters: Optional[dict] = None,
        tenant_id: str = DEFAULT_TENANT_ID,
        limit: int = 20
    ) -> list[dict]:
        """Retrieve relevant memories for RBA"""
        try:
            # Try semantic search first
            from api.memory import generate_embedding
            embedding = await generate_embedding(query)

            if embedding:
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                rows = await self.pool.fetch("""
                    SELECT
                        id::text,
                        memory_type,
                        content,
                        importance_score,
                        verification_state,
                        confidence_score,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM unified_ai_memory
                    WHERE (tenant_id = $2::uuid OR tenant_id IS NULL)
                        AND embedding IS NOT NULL
                        AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY (1 - (embedding <=> $1::vector)) * importance_score DESC
                    LIMIT $3
                """, embedding_str, tenant_id, limit)
            else:
                # Fallback to text search
                rows = await self.pool.fetch("""
                    SELECT
                        id::text,
                        memory_type,
                        content,
                        importance_score,
                        verification_state,
                        confidence_score,
                        NULL::float as similarity
                    FROM unified_ai_memory
                    WHERE (tenant_id = $1::uuid OR tenant_id IS NULL)
                        AND (search_text ILIKE $2 OR content::text ILIKE $2)
                        AND (expires_at IS NULL OR expires_at > NOW())
                    ORDER BY importance_score DESC
                    LIMIT $3
                """, tenant_id, f"%{query}%", limit)

            return [dict(r) for r in rows]

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    async def _store_memory(
        self,
        contract: MemoryContract,
        agent_id: Optional[str] = None,
        tenant_id: str = DEFAULT_TENANT_ID,
        proof: Optional[VerificationProof] = None
    ) -> str:
        """Store a memory according to the contract"""
        try:
            # Prepare content
            content_json = json.dumps(contract.content) if isinstance(contract.content, dict) else json.dumps({"text": str(contract.content)})

            # Generate embedding
            from api.memory import generate_embedding
            content_text = str(contract.content)
            embedding = await generate_embedding(content_text)
            embedding_str = "[" + ",".join(map(str, embedding)) + "]" if embedding else None

            # Determine verification state from proof
            verification_state = contract.verification.state.value if contract.verification else "UNVERIFIED"
            evidence_level = contract.verification.evidence_level.value if contract.verification else "E0_UNVERIFIED"
            confidence = contract.verification.confidence_score if contract.verification else 0.0

            # Insert memory
            result = await self.pool.fetchrow("""
                INSERT INTO unified_ai_memory (
                    memory_type, content, importance_score, tags,
                    source_system, source_agent, created_by,
                    metadata, embedding, search_text, tenant_id,
                    object_type, owner, project,
                    verification_state, evidence_level, confidence_score,
                    supersedes
                ) VALUES (
                    'procedural', $1::jsonb, 0.7, $2,
                    $3, $4, $5,
                    $6::jsonb, $7::vector, $8, $9::uuid,
                    $10::memory_object_type, $11, $12,
                    $13::verification_state, $14::evidence_level, $15,
                    $16::uuid[]
                )
                RETURNING id::text
            """,
                content_json,
                contract.tags,
                contract.source,
                agent_id,
                agent_id or "enforcement_engine",
                json.dumps(contract.metadata),
                embedding_str,
                f"{contract.title} {content_text}",
                tenant_id,
                contract.type.value if contract.type else None,
                contract.owner,
                contract.project,
                verification_state,
                evidence_level,
                confidence,
                contract.supersedes
            )

            memory_id = result["id"]

            # Store proof artifact if provided
            if proof:
                await self._store_proof_artifact(memory_id, proof)

            return memory_id

        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            raise

    async def _store_proof_artifact(
        self,
        memory_id: str,
        proof: VerificationProof
    ) -> str:
        """Store a verification proof artifact"""
        try:
            result = await self.pool.fetchrow("""
                INSERT INTO memory_verification_artifacts (
                    memory_id, artifact_type, artifact_url, artifact_hash,
                    artifact_content, evidence_level, created_by, expires_at, metadata
                ) VALUES (
                    $1::uuid, $2, $3, $4, $5::jsonb, $6::evidence_level, $7, $8, $9::jsonb
                )
                RETURNING id::text
            """,
                memory_id,
                proof.artifact_type,
                proof.artifact_url,
                proof.artifact_hash,
                json.dumps(proof.artifact_content) if proof.artifact_content else None,
                proof.evidence_level.value,
                proof.created_by,
                proof.expires_at,
                json.dumps(proof.metadata)
            )

            return result["id"]

        except Exception as e:
            logger.error(f"Proof artifact storage failed: {e}")
            raise

    async def _audit_operation(
        self,
        context: EnforcementContext,
        result: EnforcementResult
    ):
        """Audit a memory operation"""
        try:
            await self.pool.execute("""
                INSERT INTO memory_operation_audit (
                    operation, memory_id, agent_id, tool_id,
                    correlation_id, tenant_id, operation_context,
                    operation_result, error_message, rba_enforced, wba_enforced
                ) VALUES (
                    $1, $2::uuid, $3, $4,
                    $5::uuid, $6::uuid, $7::jsonb,
                    $8, $9, $10, $11
                )
            """,
                context.operation.value,
                result.memory_ids_written[0] if result.memory_ids_written else (result.memory_ids_retrieved[0] if result.memory_ids_retrieved else None),
                context.agent_id,
                context.tool_id,
                context.correlation_id,
                context.tenant_id,
                json.dumps({"bypass_reason": context.bypass_reason}),
                "success" if result.success else ("blocked" if result.blocked else "failed"),
                result.block_reason,
                result.rba_enforced,
                result.wba_enforced
            )
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")


# =============================================================================
# ENFORCEMENT MIDDLEWARE
# =============================================================================

class EnforcementMiddleware:
    """
    FastAPI middleware for enforcing memory operations.
    Wraps API handlers to ensure RBA/WBA compliance.
    """

    def __init__(self, engine: MemoryEnforcementEngine):
        self.engine = engine

    def enforce_rba(self, query_extractor):
        """
        Decorator to enforce read-before-act on an endpoint.

        Usage:
            @enforcement.enforce_rba(lambda req: req.query_params.get("context"))
            async def my_endpoint(request):
                ...
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                request = kwargs.get("request") or (args[0] if args else None)

                # Extract query for RBA
                query = query_extractor(request) if query_extractor else "general context"

                # Create enforcement context
                context = EnforcementContext(
                    operation=OperationType.READ,
                    agent_id=request.headers.get("X-Agent-ID"),
                    tool_id=request.headers.get("X-Tool-ID"),
                    tenant_id=request.headers.get("X-Tenant-ID", DEFAULT_TENANT_ID)
                )

                # Enforce RBA
                rba_result = await self.engine.enforce_read_before_act(context, query)

                if rba_result.blocked:
                    from fastapi import HTTPException
                    raise HTTPException(status_code=403, detail=rba_result.block_reason)

                # Add enforcement result to request state
                request.state.enforcement_context = context
                request.state.rba_result = rba_result

                return await func(*args, **kwargs)

            return wrapper
        return decorator

    def enforce_wba(self, memory_extractor):
        """
        Decorator to enforce write-back-after on an endpoint.

        Usage:
            @enforcement.enforce_wba(lambda result: MemoryContract(...))
            async def my_endpoint(request):
                ...
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Execute the function
                result = await func(*args, **kwargs)

                request = kwargs.get("request") or (args[0] if args else None)

                # Extract memory contract from result
                memory_contract = memory_extractor(result) if memory_extractor else None

                if memory_contract:
                    # Get context from request state or create new
                    context = getattr(request.state, "enforcement_context", None) or EnforcementContext(
                        operation=OperationType.WRITE,
                        agent_id=request.headers.get("X-Agent-ID"),
                        tool_id=request.headers.get("X-Tool-ID"),
                        tenant_id=request.headers.get("X-Tenant-ID", DEFAULT_TENANT_ID)
                    )
                    context.operation = OperationType.WRITE

                    # Enforce WBA
                    wba_result = await self.engine.enforce_write_back_after(context, memory_contract)

                    if wba_result.blocked:
                        from fastapi import HTTPException
                        raise HTTPException(status_code=403, detail=wba_result.block_reason)

                return result

            return wrapper
        return decorator


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_enforcement_engine: Optional[MemoryEnforcementEngine] = None


def get_enforcement_engine() -> MemoryEnforcementEngine:
    """Get or create the singleton enforcement engine"""
    global _enforcement_engine
    if _enforcement_engine is None:
        _enforcement_engine = MemoryEnforcementEngine()
    return _enforcement_engine


def get_enforcement_middleware() -> EnforcementMiddleware:
    """Get enforcement middleware instance"""
    return EnforcementMiddleware(get_enforcement_engine())
