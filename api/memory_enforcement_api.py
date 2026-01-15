"""
Memory Enforcement API Router
===============================
API endpoints for memory enforcement, RBA/WBA, and verification.

Part of BrainOps OS Total Completion Protocol.
"""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from memory_enforcement import (
    EvidenceLevel,
    MemoryContract,
    MemoryObjectType,
    VerificationProof,
    VerificationState,
    get_enforcement_engine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enforcement", tags=["Memory Enforcement"])


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class VerifyMemoryRequest(BaseModel):
    memory_id: str
    evidence_level: str = "E1_RECORDED"
    artifact_refs: list[str] = []
    verified_by: str = "api"
    expiration_days: int = 30


class WriteDecisionRequest(BaseModel):
    decision_type: str
    decision_content: dict[str, Any]
    reasoning: str
    agent_id: str
    project: Optional[str] = None
    evidence_level: str = "E1_RECORDED"
    artifact_refs: list[str] = []


class WriteSOPRequest(BaseModel):
    sop_name: str
    sop_content: dict[str, Any]
    version: str = "1.0.0"
    agent_id: str
    project: Optional[str] = None
    evidence_level: str = "E1_RECORDED"


class AuditQueryRequest(BaseModel):
    agent_id: Optional[str] = None
    operation: Optional[str] = None
    memory_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100


# =============================================================================
# VERIFICATION ENDPOINTS
# =============================================================================

@router.post("/verify")
async def verify_memory(request: VerifyMemoryRequest) -> dict[str, Any]:
    """
    Verify a memory with proof artifacts.

    Upgrades verification_state from UNVERIFIED to VERIFIED
    when proper evidence is provided.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        evidence = EvidenceLevel[request.evidence_level]

        # Construct VerificationProof object
        proof = VerificationProof(
            artifact_type="api_verification",
            artifact_content={"refs": request.artifact_refs},
            evidence_level=evidence,
            created_by=request.verified_by,
            metadata={"expiration_days": request.expiration_days}
        )

        result = await engine.verify_memory(
            memory_id=request.memory_id,
            proof=proof,
            verified_by=request.verified_by
        )

        return {
            "success": result.success,
            "memory_id": request.memory_id,
            "verification_state": "VERIFIED" if result.success else "FAILED",
            "evidence_level": request.evidence_level,
            "message": "Memory verified successfully" if result.success else result.block_reason
        }

    except Exception as e:
        logger.error(f"Memory verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{memory_id}")
async def get_verification_status(memory_id: str) -> dict[str, Any]:
    """
    Get verification status of a specific memory.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        row = await engine.pool.fetchrow("""
            SELECT
                id, verification_state, evidence_level, confidence_score,
                last_verified_at, verified_by, verification_expires_at,
                artifact_refs, supersedes, object_type, owner, project
            FROM unified_ai_memory
            WHERE id = $1::uuid
        """, memory_id)

        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Check for conflicts
        conflicts = await engine.pool.fetch("""
            SELECT id, conflict_type, severity, detected_at
            FROM memory_conflicts
            WHERE (memory_id_a = $1::uuid OR memory_id_b = $1::uuid)
              AND resolution_status = 'open'
        """, memory_id)

        return {
            "memory_id": str(row["id"]),
            "verification_state": row["verification_state"],
            "evidence_level": row["evidence_level"],
            "confidence_score": float(row["confidence_score"] or 0),
            "last_verified_at": row["last_verified_at"].isoformat() if row["last_verified_at"] else None,
            "verified_by": row["verified_by"],
            "verification_expires_at": row["verification_expires_at"].isoformat() if row["verification_expires_at"] else None,
            "artifact_refs": row["artifact_refs"] or [],
            "supersedes": [str(s) for s in (row["supersedes"] or [])],
            "object_type": row["object_type"],
            "owner": row["owner"],
            "project": row["project"],
            "open_conflicts": [
                {
                    "id": str(c["id"]),
                    "type": c["conflict_type"],
                    "severity": c["severity"],
                    "detected_at": c["detected_at"].isoformat()
                }
                for c in conflicts
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get verification status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TRUTH BACKLOG ENDPOINTS
# =============================================================================

@router.get("/backlog")
async def get_truth_backlog(
    limit: int = Query(default=50, le=500),
    priority: Optional[str] = Query(default=None),
    object_type: Optional[str] = Query(default=None),
    owner: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    """
    Get memories requiring verification from the truth backlog.

    This is the queue of items that need attention to establish trust.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        backlog = await engine.get_truth_backlog(limit=limit)

        # Get aggregate stats
        stats = await engine.pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE priority = 'critical') as critical,
                COUNT(*) FILTER (WHERE priority = 'high') as high,
                COUNT(*) FILTER (WHERE priority = 'medium') as medium,
                COUNT(*) FILTER (WHERE priority = 'low') as low
            FROM memory_truth_backlog
        """)

        return {
            "total_backlog": len(backlog),
            "stats": {
                "critical": stats["critical"] or 0,
                "high": stats["high"] or 0,
                "medium": stats["medium"] or 0,
                "low": stats["low"] or 0
            },
            "items": backlog
        }

    except Exception as e:
        logger.error(f"Failed to get truth backlog: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONFLICT MANAGEMENT
# =============================================================================

@router.get("/conflicts")
async def get_open_conflicts(
    limit: int = Query(default=50, le=200),
    severity: Optional[str] = Query(default=None)
) -> dict[str, Any]:
    """
    Get open memory conflicts requiring resolution.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        query = """
            SELECT
                mc.id, mc.memory_id_a, mc.memory_id_b,
                mc.conflict_type, mc.severity, mc.detected_at, mc.detected_by,
                ma.content->>'summary' as memory_a_summary,
                mb.content->>'summary' as memory_b_summary
            FROM memory_conflicts mc
            JOIN unified_ai_memory ma ON mc.memory_id_a = ma.id
            JOIN unified_ai_memory mb ON mc.memory_id_b = mb.id
            WHERE mc.resolution_status = 'open'
        """
        params = []

        if severity:
            query += " AND mc.severity = $1"
            params.append(severity)

        query += f" ORDER BY mc.detected_at DESC LIMIT {limit}"

        conflicts = await engine.pool.fetch(query, *params)

        return {
            "open_conflicts": len(conflicts),
            "conflicts": [
                {
                    "id": str(c["id"]),
                    "memory_id_a": str(c["memory_id_a"]),
                    "memory_id_b": str(c["memory_id_b"]),
                    "conflict_type": c["conflict_type"],
                    "severity": c["severity"],
                    "detected_at": c["detected_at"].isoformat(),
                    "detected_by": c["detected_by"],
                    "memory_a_summary": c["memory_a_summary"],
                    "memory_b_summary": c["memory_b_summary"]
                }
                for c in conflicts
            ]
        }

    except Exception as e:
        logger.error(f"Failed to get conflicts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: str,
    resolution: str = Query(...),
    resolution_notes: Optional[str] = Query(default=None),
    resolved_by: str = Query(default="api")
) -> dict[str, Any]:
    """
    Resolve a memory conflict.

    Resolution options: 'keep_a', 'keep_b', 'merge', 'ignore'
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        await engine.pool.execute("""
            UPDATE memory_conflicts
            SET resolution_status = 'resolved',
                resolved_at = NOW(),
                resolved_by = $2,
                resolution_notes = $3,
                metadata = metadata || jsonb_build_object('resolution_action', $4)
            WHERE id = $1::uuid
        """, conflict_id, resolved_by, resolution_notes, resolution)

        return {
            "success": True,
            "conflict_id": conflict_id,
            "resolution": resolution,
            "message": "Conflict resolved"
        }

    except Exception as e:
        logger.error(f"Failed to resolve conflict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AUDIT ENDPOINTS
# =============================================================================

@router.get("/audit")
async def query_audit_log(
    agent_id: Optional[str] = Query(default=None),
    operation: Optional[str] = Query(default=None),
    memory_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000)
) -> dict[str, Any]:
    """
    Query the memory operation audit log.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        conditions = []
        params = []
        param_idx = 1

        if agent_id:
            conditions.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1

        if operation:
            conditions.append(f"operation = ${param_idx}")
            params.append(operation)
            param_idx += 1

        if memory_id:
            conditions.append(f"memory_id = ${param_idx}::uuid")
            params.append(memory_id)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        rows = await engine.pool.fetch(f"""
            SELECT
                id, operation, memory_id, agent_id, tool_id,
                correlation_id, operation_result, error_message,
                duration_ms, rba_enforced, wba_enforced, created_at
            FROM memory_operation_audit
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT {limit}
        """, *params)

        return {
            "total": len(rows),
            "audit_entries": [
                {
                    "id": str(r["id"]),
                    "operation": r["operation"],
                    "memory_id": str(r["memory_id"]) if r["memory_id"] else None,
                    "agent_id": r["agent_id"],
                    "tool_id": r["tool_id"],
                    "correlation_id": str(r["correlation_id"]) if r["correlation_id"] else None,
                    "result": r["operation_result"],
                    "error": r["error_message"],
                    "duration_ms": r["duration_ms"],
                    "rba_enforced": r["rba_enforced"],
                    "wba_enforced": r["wba_enforced"],
                    "timestamp": r["created_at"].isoformat()
                }
                for r in rows
            ]
        }

    except Exception as e:
        logger.error(f"Failed to query audit log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENFORCEMENT STATS
# =============================================================================

@router.get("/stats")
async def get_enforcement_stats() -> dict[str, Any]:
    """
    Get overall enforcement statistics.
    """
    try:
        engine = get_enforcement_engine()
        await engine.initialize()

        # Memory verification stats
        verification_stats = await engine.pool.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE verification_state = 'VERIFIED') as verified,
                COUNT(*) FILTER (WHERE verification_state = 'UNVERIFIED') as unverified,
                COUNT(*) FILTER (WHERE verification_state = 'DEGRADED') as degraded,
                COUNT(*) FILTER (WHERE verification_state = 'BROKEN') as broken,
                AVG(confidence_score) as avg_confidence,
                COUNT(*) FILTER (WHERE verification_expires_at < NOW()) as expired
            FROM unified_ai_memory
            WHERE expires_at IS NULL OR expires_at > NOW()
        """)

        # Audit stats (last 24 hours)
        audit_stats = await engine.pool.fetchrow("""
            SELECT
                COUNT(*) as total_ops,
                COUNT(*) FILTER (WHERE rba_enforced = true) as rba_enforced,
                COUNT(*) FILTER (WHERE wba_enforced = true) as wba_enforced,
                COUNT(*) FILTER (WHERE operation_result = 'blocked') as blocked,
                COUNT(*) FILTER (WHERE operation_result = 'success') as successful
            FROM memory_operation_audit
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)

        # Conflict stats
        conflict_stats = await engine.pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE resolution_status = 'open') as open,
                COUNT(*) FILTER (WHERE resolution_status = 'resolved') as resolved,
                COUNT(*) FILTER (WHERE severity = 'critical' AND resolution_status = 'open') as critical_open
            FROM memory_conflicts
        """)

        # Evidence level distribution
        evidence_dist = await engine.pool.fetch("""
            SELECT evidence_level, COUNT(*) as count
            FROM unified_ai_memory
            WHERE expires_at IS NULL OR expires_at > NOW()
            GROUP BY evidence_level
            ORDER BY evidence_level
        """)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "verification": {
                "total_memories": verification_stats["total"] or 0,
                "verified": verification_stats["verified"] or 0,
                "unverified": verification_stats["unverified"] or 0,
                "degraded": verification_stats["degraded"] or 0,
                "broken": verification_stats["broken"] or 0,
                "expired": verification_stats["expired"] or 0,
                "avg_confidence": round(float(verification_stats["avg_confidence"] or 0), 3),
                "verification_rate": round(
                    (verification_stats["verified"] or 0) / max(verification_stats["total"] or 1, 1) * 100, 1
                )
            },
            "enforcement_24h": {
                "total_operations": audit_stats["total_ops"] or 0,
                "rba_enforced": audit_stats["rba_enforced"] or 0,
                "wba_enforced": audit_stats["wba_enforced"] or 0,
                "blocked": audit_stats["blocked"] or 0,
                "successful": audit_stats["successful"] or 0
            },
            "conflicts": {
                "open": conflict_stats["open"] or 0,
                "resolved": conflict_stats["resolved"] or 0,
                "critical_open": conflict_stats["critical_open"] or 0
            },
            "evidence_distribution": {
                (e["evidence_level"] or "E0_UNVERIFIED"): e["count"]
                for e in evidence_dist
            }
        }

    except Exception as e:
        logger.error(f"Failed to get enforcement stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
