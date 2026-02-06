"""
Revenue Operator API
====================
AI-driven revenue operations endpoints.

All operations are approval-gated and respect controls.
Part of Revenue Perfection Session.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader

from config import config
from revenue_operator import get_revenue_operator

logger = logging.getLogger(__name__)

# API Key Security - centralized authentication
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

# All endpoints require API key authentication
router = APIRouter(
    prefix="/revenue-operator",
    tags=["Revenue Operator"],
    dependencies=[Depends(verify_api_key)]
)


@router.get("/next-best-actions")
async def get_next_best_actions(
    limit: int = Query(default=10, le=50)
) -> dict[str, Any]:
    """
    Generate next-best-action plan for top leads.

    Returns prioritized list of actions for each lead.
    Blocked if any kill switches are active.
    """
    operator = get_revenue_operator()
    return await operator.get_next_best_actions(limit)


@router.get("/alarms")
async def get_alarms() -> dict[str, Any]:
    """
    Get current revenue pipeline alarms.

    Alarms trigger for:
    - NO_OUTREACH: Leads exist but no outreach sent
    - NO_REVENUE: Outreach sent but $0 revenue
    - MESSAGE_MISMATCH: Outreach sent but no replies
    - STALE_LEADS: Leads inactive for 14+ days
    """
    operator = get_revenue_operator()
    return await operator.generate_alarms()


@router.post("/auto-draft-proposals")
async def auto_draft_proposals(
    limit: int = Query(default=5, le=20)
) -> dict[str, Any]:
    """
    Auto-draft proposals for qualified leads.

    Creates drafts that require human approval.
    Blocked if kill switches are active.
    """
    operator = get_revenue_operator()
    return await operator.auto_draft_proposals(limit)


@router.get("/status")
async def get_operator_status() -> dict[str, Any]:
    """
    Get revenue operator status including kill switch state.
    """
    operator = get_revenue_operator()
    killed, reason = await operator.check_kill_switches()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operator_active": not killed,
        "kill_switches": reason,
        "capabilities": {
            "next_best_actions": not killed,
            "auto_proposals": not killed,
            "alarms": True  # Always available
        }
    }


@router.post("/execute-plan")
async def execute_plan(
    plan_type: str = Query(..., enum=["enrich_all", "draft_outreach", "draft_proposals"]),
    limit: int = Query(default=10, le=50),
    approved_by: str = Query(default="api")
) -> dict[str, Any]:
    """
    Execute a pre-defined action plan.

    Plans:
    - enrich_all: Enrich all unenriched real leads
    - draft_outreach: Draft outreach for enriched leads
    - draft_proposals: Draft proposals for qualified leads

    All actions respect daily limits and kill switches.
    """
    operator = get_revenue_operator()

    # Check kill switches
    killed, reason = await operator.check_kill_switches()
    if killed:
        raise HTTPException(status_code=403, detail=f"Blocked: {reason}")

    if plan_type == "enrich_all":
        from outreach_engine import get_outreach_engine
        engine = get_outreach_engine()

        from database.async_connection import get_pool
        pool = get_pool()
        if not pool:
            raise HTTPException(status_code=503, detail="Database not available")

        leads = await pool.fetch("""
            SELECT id FROM revenue_leads
            WHERE email NOT ILIKE '%test%'
            AND email NOT ILIKE '%example%'
            AND (metadata->>'enrichment' IS NULL OR metadata->'enrichment' = 'null')
            LIMIT $1
        """, limit)

        enriched = []
        for lead in leads:
            success, _, _ = await engine.enrich_lead(str(lead["id"]))
            if success:
                enriched.append(str(lead["id"])[:8] + "...")

        return {
            "plan": plan_type,
            "executed_by": approved_by,
            "enriched_count": len(enriched),
            "leads": enriched
        }

    elif plan_type == "draft_outreach":
        from outreach_engine import get_outreach_engine
        engine = get_outreach_engine()

        from database.async_connection import get_pool
        pool = get_pool()
        if not pool:
            raise HTTPException(status_code=503, detail="Database not available")

        leads = await pool.fetch("""
            SELECT rl.id
            FROM revenue_leads rl
            LEFT JOIN revenue_actions ra ON rl.id = ra.lead_id AND ra.action_type = 'outreach_draft'
            WHERE rl.email NOT ILIKE '%test%'
            AND rl.email NOT ILIKE '%example%'
            AND rl.metadata->>'enrichment' IS NOT NULL
            AND ra.id IS NULL
            LIMIT $1
        """, limit)

        drafts = []
        for lead in leads:
            success, _, draft = await engine.generate_outreach_draft(str(lead["id"]), 1)
            if success and draft:
                drafts.append({
                    "lead": str(lead["id"])[:8] + "...",
                    "draft": draft.id[:8] + "..."
                })

        return {
            "plan": plan_type,
            "executed_by": approved_by,
            "drafts_created": len(drafts),
            "drafts": drafts
        }

    elif plan_type == "draft_proposals":
        return await operator.auto_draft_proposals(limit)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {plan_type}")
