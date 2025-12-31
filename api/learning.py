"""
Learning Feedback Loop API Endpoints

Exposes the learning feedback loop functionality via REST API:
- GET /api/learning/pending-proposals - List proposals awaiting approval
- POST /api/learning/approve/{id} - Approve and apply a proposal
- POST /api/learning/reject/{id} - Reject a proposal
- POST /api/learning/run-cycle - Manually trigger a feedback loop cycle
- GET /api/learning/status - Get current feedback loop status
- GET /api/learning/patterns - Get recently detected patterns
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["Learning Feedback Loop"])


class ApprovalRequest(BaseModel):
    """Request body for approving a proposal"""
    approver: str = "human"
    notes: Optional[str] = None


class RejectionRequest(BaseModel):
    """Request body for rejecting a proposal"""
    reason: str


class RunCycleRequest(BaseModel):
    """Request body for running a feedback loop cycle"""
    analysis_window_hours: int = 24


# Lazy load the feedback loop to avoid circular imports
_feedback_loop = None


async def get_loop():
    """Get the feedback loop instance"""
    global _feedback_loop
    if _feedback_loop is None:
        try:
            from learning_feedback_loop import get_feedback_loop
            _feedback_loop = await get_feedback_loop()
        except Exception as e:
            logger.error(f"Failed to initialize feedback loop: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Learning feedback loop not available: {e}"
            )
    return _feedback_loop


@router.get("/pending-proposals")
async def get_pending_proposals() -> Dict[str, Any]:
    """
    Get all proposals awaiting human approval.

    Returns proposals that:
    - Have status 'proposed'
    - Are not auto-approvable (or have risk level > low)
    """
    try:
        loop = await get_loop()
        proposals = await loop.get_pending_proposals()

        # Convert UUID and datetime objects for JSON serialization
        serialized = []
        for p in proposals:
            item = dict(p)
            for key, value in item.items():
                if hasattr(value, 'isoformat'):
                    item[key] = value.isoformat()
                elif hasattr(value, 'hex'):  # UUID
                    item[key] = str(value)
            serialized.append(item)

        return {
            "count": len(serialized),
            "proposals": serialized,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pending proposals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve/{proposal_id}")
async def approve_proposal(
    proposal_id: str,
    request: ApprovalRequest = Body(default=ApprovalRequest())
) -> Dict[str, Any]:
    """
    Approve a proposal for implementation.

    The proposal will be queued for application in the next cycle,
    or applied immediately if auto_apply is enabled.
    """
    try:
        loop = await get_loop()
        success = await loop.approve_proposal(proposal_id, request.approver)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Proposal {proposal_id} not found or already processed"
            )

        return {
            "success": True,
            "proposal_id": proposal_id,
            "approved_by": request.approver,
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "message": "Proposal approved and queued for implementation"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to approve proposal {proposal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reject/{proposal_id}")
async def reject_proposal(
    proposal_id: str,
    request: RejectionRequest
) -> Dict[str, Any]:
    """
    Reject a proposal.

    The proposal will be marked as rejected with the provided reason.
    """
    try:
        loop = await get_loop()
        success = await loop.reject_proposal(proposal_id, request.reason)

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Proposal {proposal_id} not found or already processed"
            )

        return {
            "success": True,
            "proposal_id": proposal_id,
            "rejected_at": datetime.now(timezone.utc).isoformat(),
            "reason": request.reason
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reject proposal {proposal_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-cycle")
async def run_feedback_cycle(
    request: RunCycleRequest = Body(default=RunCycleRequest())
) -> Dict[str, Any]:
    """
    Manually trigger a feedback loop cycle.

    This will:
    1. Analyze recent insights from the specified window
    2. Identify actionable patterns
    3. Generate improvement proposals
    4. Auto-approve eligible low-risk proposals
    5. Apply approved improvements

    Normally runs every 2 hours automatically.
    """
    try:
        loop = await get_loop()

        # Temporarily override analysis window if specified
        original_window = loop.analysis_window_hours
        if request.analysis_window_hours != original_window:
            loop.analysis_window_hours = request.analysis_window_hours

        try:
            results = await loop.run_feedback_loop()
        finally:
            # Restore original window
            loop.analysis_window_hours = original_window

        return {
            "success": True,
            "cycle_results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run feedback cycle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """
    Get the current status of the learning feedback loop.

    Returns:
    - Configuration settings
    - Recent cycle statistics
    - Proposal counts by status
    """
    try:
        from database.async_connection import get_pool

        pool = get_pool()

        # Get proposal statistics
        stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_proposals,
                COUNT(*) FILTER (WHERE status = 'proposed') as pending,
                COUNT(*) FILTER (WHERE status = 'approved') as approved,
                COUNT(*) FILTER (WHERE status = 'implementing') as implementing,
                COUNT(*) FILTER (WHERE status = 'completed') as completed,
                COUNT(*) FILTER (WHERE status = 'rejected') as rejected,
                COUNT(*) FILTER (WHERE auto_approved = true) as auto_approved_count,
                MAX(created_at) as latest_proposal
            FROM ai_improvement_proposals
        """)

        # Get pattern statistics
        pattern_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_patterns,
                COUNT(DISTINCT pattern_type) as pattern_types,
                MAX(created_at) as latest_pattern
            FROM ai_learning_patterns
        """)

        # Get recent cycle info
        recent_cycle = await pool.fetchrow("""
            SELECT insight, metadata, created_at
            FROM ai_learning_insights
            WHERE insight_type = 'feedback_loop_cycle'
            ORDER BY created_at DESC
            LIMIT 1
        """)

        loop = await get_loop()

        return {
            "status": "active",
            "configuration": {
                "analysis_window_hours": loop.analysis_window_hours,
                "min_pattern_confidence": loop.min_pattern_confidence,
                "min_occurrence_count": loop.min_occurrence_count,
                "auto_approve_risk_levels": [r.value for r in loop.auto_approve_risk_levels],
                "auto_approve_types": [t.value for t in loop.auto_approve_types]
            },
            "proposals": {
                "total": stats['total_proposals'] if stats else 0,
                "pending": stats['pending'] if stats else 0,
                "approved": stats['approved'] if stats else 0,
                "implementing": stats['implementing'] if stats else 0,
                "completed": stats['completed'] if stats else 0,
                "rejected": stats['rejected'] if stats else 0,
                "auto_approved_count": stats['auto_approved_count'] if stats else 0,
                "latest_proposal": stats['latest_proposal'].isoformat() if stats and stats['latest_proposal'] else None
            },
            "patterns": {
                "total": pattern_stats['total_patterns'] if pattern_stats else 0,
                "types": pattern_stats['pattern_types'] if pattern_stats else 0,
                "latest": pattern_stats['latest_pattern'].isoformat() if pattern_stats and pattern_stats['latest_pattern'] else None
            },
            "last_cycle": {
                "summary": recent_cycle['insight'] if recent_cycle else None,
                "details": recent_cycle['metadata'] if recent_cycle else None,
                "timestamp": recent_cycle['created_at'].isoformat() if recent_cycle and recent_cycle['created_at'] else None
            },
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def get_patterns(
    hours: int = Query(default=24, description="Hours to look back for patterns"),
    limit: int = Query(default=20, description="Maximum patterns to return")
) -> Dict[str, Any]:
    """
    Get recently detected patterns.

    Returns patterns identified in the specified time window.
    """
    try:
        loop = await get_loop()

        # Set the analysis window
        original_window = loop.analysis_window_hours
        loop.analysis_window_hours = hours

        try:
            patterns = await loop.analyze_insights(hours=hours)
        finally:
            loop.analysis_window_hours = original_window

        # Convert to serializable format
        serialized = []
        for p in patterns[:limit]:
            serialized.append({
                "pattern_type": p.pattern_type,
                "agent_name": p.agent_name,
                "metric": p.metric,
                "current_value": p.current_value,
                "expected_value": p.expected_value,
                "deviation_percent": p.deviation_percent,
                "occurrence_count": p.occurrence_count,
                "confidence": p.confidence,
                "time_window": p.time_window,
                "evidence": p.evidence
            })

        return {
            "count": len(serialized),
            "time_window_hours": hours,
            "patterns": serialized,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights-summary")
async def get_insights_summary() -> Dict[str, Any]:
    """
    Get a summary of all learning insights in the system.

    This shows what data the system has accumulated but not acted on.
    """
    try:
        from database.async_connection import get_pool

        pool = get_pool()

        # Get overall statistics
        overall = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_insights,
                COUNT(*) FILTER (WHERE applied = true) as applied_count,
                COUNT(*) FILTER (WHERE applied = false) as unapplied_count,
                AVG(confidence) as avg_confidence,
                AVG(impact_score) as avg_impact,
                MIN(created_at) as earliest,
                MAX(created_at) as latest
            FROM ai_learning_insights
        """)

        # Get by type
        by_type = await pool.fetch("""
            SELECT
                insight_type,
                category,
                COUNT(*) as count,
                COUNT(*) FILTER (WHERE applied = false) as unapplied,
                AVG(confidence) as avg_confidence,
                AVG(impact_score) as avg_impact
            FROM ai_learning_insights
            GROUP BY insight_type, category
            ORDER BY count DESC
        """)

        return {
            "overall": {
                "total_insights": overall['total_insights'] if overall else 0,
                "applied": overall['applied_count'] if overall else 0,
                "unapplied": overall['unapplied_count'] if overall else 0,
                "avg_confidence": float(overall['avg_confidence'] or 0) if overall else 0,
                "avg_impact": float(overall['avg_impact'] or 0) if overall else 0,
                "date_range": {
                    "earliest": overall['earliest'].isoformat() if overall and overall['earliest'] else None,
                    "latest": overall['latest'].isoformat() if overall and overall['latest'] else None
                }
            },
            "by_type": [
                {
                    "insight_type": row['insight_type'],
                    "category": row['category'],
                    "count": row['count'],
                    "unapplied": row['unapplied'],
                    "avg_confidence": float(row['avg_confidence'] or 0),
                    "avg_impact": float(row['avg_impact'] or 0)
                }
                for row in by_type
            ],
            "message": f"System has {overall['unapplied_count'] if overall else 0} unapplied insights - the feedback loop will process these",
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get insights summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
