"""
Outreach API
============
REST endpoints for lead enrichment and outreach management.

Respects daily limits:
- Enrichment: 100/day
- Outreach: 50/day

Part of Revenue Perfection Session.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from outreach_engine import get_outreach_engine
from outreach_executor import run_outreach_cycle
from daily_wc_report import generate_daily_wc_report


def _parse_metadata(metadata: Any) -> dict:
    """Parse metadata field which may be a string or dict."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/outreach", tags=["Outreach Engine"])


class ReplyLogRequest(BaseModel):
    """Request to log a reply."""
    summary: str


@router.get("/stats")
async def get_outreach_stats() -> dict[str, Any]:
    """
    Get outreach statistics including daily limits.
    """
    engine = get_outreach_engine()
    stats = await engine.get_outreach_stats()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **stats
    }


@router.get("/leads/real")
async def get_real_leads(
    limit: int = Query(default=23, le=100)
) -> dict[str, Any]:
    """
    Get all REAL leads (excluding test/demo).

    Returns leads sorted by score.
    """
    engine = get_outreach_engine()
    leads = await engine.get_real_leads(limit)

    # Mask emails for security
    masked_leads = []
    for lead in leads:
        masked = {
            "id": str(lead["id"]),
            "company_name": lead["company_name"],
            "stage": lead["stage"],
            "score": lead["score"],
            "value_estimate": float(lead["value_estimate"] or 0),
            "source": lead["source"],
            "industry": lead.get("industry"),
            "email_masked": f"{lead['email'][:3]}***@{lead['email'].split('@')[1]}" if lead.get("email") and "@" in lead["email"] else "N/A",
            "has_enrichment": bool(_parse_metadata(lead.get("metadata")).get("enrichment")),
            "created_at": lead["created_at"].isoformat() if lead.get("created_at") else None
        }
        masked_leads.append(masked)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_real_leads": len(masked_leads),
        "leads": masked_leads
    }


@router.post("/leads/{lead_id}/enrich")
async def enrich_lead(lead_id: str) -> dict[str, Any]:
    """
    Enrich a lead with additional data.

    Limited to 100 enrichments per day.
    """
    engine = get_outreach_engine()

    success, message, result = await engine.enrich_lead(lead_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "lead_id": lead_id,
        "enrichment": {
            "decision_maker": result.decision_maker,
            "company_size": result.company_size,
            "region": result.region,
            "pain_points": result.pain_points,
            "value_band": result.estimated_value_band
        },
        "next_step": f"POST /outreach/leads/{lead_id}/draft to create outreach"
    }


@router.post("/leads/{lead_id}/draft")
async def create_outreach_draft(
    lead_id: str,
    sequence_step: int = Query(default=1, ge=1, le=3)
) -> dict[str, Any]:
    """
    Create an outreach message draft for a lead.

    Args:
        lead_id: UUID of the lead
        sequence_step: 1=initial, 2=followup1, 3=followup2
    """
    engine = get_outreach_engine()

    success, message, draft = await engine.generate_outreach_draft(lead_id, sequence_step)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "draft_id": draft.id,
        "lead_id": draft.lead_id,
        "sequence_step": draft.sequence_step,
        "subject": draft.subject,
        "body_preview": draft.body[:200] + "..." if len(draft.body) > 200 else draft.body,
        "next_step": f"POST /outreach/drafts/{draft.id}/submit-approval to request approval"
    }


@router.post("/drafts/{draft_id}/submit-approval")
async def submit_for_approval(draft_id: str) -> dict[str, Any]:
    """
    Submit an outreach draft for human approval.
    """
    engine = get_outreach_engine()

    success, message = await engine.submit_outreach_for_approval(draft_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "draft_id": draft_id,
        "status": "pending_approval",
        "next_step": f"POST /outreach/drafts/{draft_id}/approve?approved_by=name to approve and send"
    }


@router.post("/drafts/{draft_id}/approve")
async def approve_and_send(
    draft_id: str,
    approved_by: str = Query(..., description="Name of person approving")
) -> dict[str, Any]:
    """
    Approve and send an outreach message.

    Limited to 50 emails per day.
    """
    engine = get_outreach_engine()

    success, message = await engine.approve_and_send_outreach(draft_id, approved_by)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "draft_id": draft_id,
        "approved_by": approved_by,
        "status": "queued_for_send"
    }


@router.post("/leads/{lead_id}/reply-log")
async def log_reply(
    lead_id: str,
    request: ReplyLogRequest
) -> dict[str, Any]:
    """
    Log a reply received from a lead (manual entry).

    Use this when a prospect responds via email.
    """
    engine = get_outreach_engine()

    success, message = await engine.log_reply(lead_id, request.summary)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "lead_id": lead_id,
        "new_state": "replied"
    }


@router.post("/batch/enrich-all")
async def batch_enrich_all(
    limit: int = Query(default=10, le=50, description="Max leads to enrich")
) -> dict[str, Any]:
    """
    Batch enrich all unenriched REAL leads.

    Respects daily limit of 100.
    """
    engine = get_outreach_engine()

    # Get unenriched leads
    from database.async_connection import get_pool
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    leads = await pool.fetch("""
        SELECT id FROM revenue_leads
        WHERE email NOT ILIKE '%test%'
        AND email NOT ILIKE '%example%'
        AND email NOT ILIKE '%demo%'
        AND (metadata->>'enrichment' IS NULL OR metadata->'enrichment' = 'null')
        LIMIT $1
    """, limit)

    enriched = []
    failed = []
    for lead in leads:
        success, msg, _ = await engine.enrich_lead(str(lead["id"]))
        if success:
            enriched.append(str(lead["id"])[:8] + "...")
        else:
            if "limit" in msg.lower():
                break  # Stop if we hit the limit
            failed.append({"id": str(lead["id"])[:8] + "...", "reason": msg})

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "enriched_count": len(enriched),
        "failed_count": len(failed),
        "enriched_leads": enriched,
        "failed": failed
    }


@router.post("/batch/draft-outreach")
async def batch_draft_outreach(
    limit: int = Query(default=10, le=50)
) -> dict[str, Any]:
    """
    Batch create outreach drafts for enriched leads.
    """
    engine = get_outreach_engine()

    from database.async_connection import get_pool
    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get enriched leads without outreach drafts
    leads = await pool.fetch("""
        SELECT rl.id
        FROM revenue_leads rl
        LEFT JOIN revenue_actions ra ON rl.id = ra.lead_id AND ra.action_type = 'outreach_draft'
        WHERE rl.email NOT ILIKE '%test%'
        AND rl.email NOT ILIKE '%example%'
        AND rl.email NOT ILIKE '%demo%'
        AND rl.metadata->>'enrichment' IS NOT NULL
        AND ra.id IS NULL
        LIMIT $1
    """, limit)

    drafts_created = []
    for lead in leads:
        success, msg, draft = await engine.generate_outreach_draft(str(lead["id"]), 1)
        if success and draft:
            drafts_created.append({
                "lead_id": str(lead["id"])[:8] + "...",
                "draft_id": draft.id[:8] + "..."
            })

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "drafts_created": len(drafts_created),
        "drafts": drafts_created
    }


@router.post("/batch/scrape-emails")
async def batch_scrape_emails(
    limit: int = Query(default=10, le=50, description="Max leads to process")
) -> dict[str, Any]:
    """
    Scrape contact emails from lead websites.

    Finds leads with websites but no emails, visits their sites,
    and extracts contact email addresses.
    """
    engine = get_outreach_engine()

    results = await engine.enrich_leads_with_emails(limit)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
        "leads_processed": results["leads_processed"],
        "emails_found": results["emails_found"],
        "enriched_leads": results.get("enriched_leads", []),
        "failed_leads_count": len(results.get("failed_leads", [])),
        "next_step": "POST /outreach/batch/enrich-all to enrich leads with found emails"
    }


@router.post("/ai-leads/enrich")
async def enrich_ai_research_leads(
    limit: int = Query(default=10, le=100)
) -> dict[str, Any]:
    """
    Promote AI research leads into revenue_leads ONLY when a real
    email is found from the company's website (no guessing).
    """
    engine = get_outreach_engine()
    results = await engine.enrich_ai_revenue_leads(limit=limit)

    if results.get("status") == "error":
        raise HTTPException(status_code=503, detail=results.get("error", "enrichment_failed"))

    return results


@router.post("/execute-cycle")
async def execute_outreach_cycle() -> dict[str, Any]:
    """
    Trigger a full outreach cycle immediately.

    Processes ai_scheduled_outreach entries and enrolls new leads in campaigns.
    Requires ENABLE_OUTREACH_EXECUTOR=true env var.
    """
    import os
    enabled = os.getenv("ENABLE_OUTREACH_EXECUTOR", "").lower() in ("1", "true", "yes")
    if not enabled:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
            "error": "Outreach executor disabled. Set ENABLE_OUTREACH_EXECUTOR=true to enable.",
        }
    results = await run_outreach_cycle()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": True,
        **results,
    }


@router.post("/daily-report")
async def trigger_daily_report() -> dict[str, Any]:
    """
    Trigger the daily Weathercraft intelligence report immediately.

    Generates and emails the prospect pipeline report to matthew@weathercraft.net.
    """
    result = await generate_daily_wc_report()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result,
    }
