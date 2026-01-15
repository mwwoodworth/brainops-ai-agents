"""
Proposals API
=============
REST endpoints for proposal management.

All proposals require human approval before sending.
Part of Revenue Perfection Session.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from proposal_engine import (
    OFFER_CATALOG,
    ProposalStatus,
    get_proposal_engine,
    ensure_proposals_table,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/proposals", tags=["Proposal Engine"])


class DraftProposalRequest(BaseModel):
    """Request to create a draft proposal."""
    lead_id: str
    offer_id: str
    custom_notes: Optional[str] = None
    discount_percent: Optional[float] = None


class ApproveProposalRequest(BaseModel):
    """Request to approve a proposal."""
    approved_by: str


@router.on_event("startup")
async def startup():
    """Ensure proposals table exists on startup."""
    await ensure_proposals_table()


@router.get("/offers")
async def get_offers() -> dict[str, Any]:
    """
    Get the offer catalog.

    Returns all available products/services that can be proposed.
    """
    # Convert Decimal to float for JSON serialization
    offers = {}
    for offer_id, offer in OFFER_CATALOG.items():
        offers[offer_id] = {
            **offer,
            "price": float(offer["price"])
        }

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_offers": len(offers),
        "offers": offers,
        "categories": {
            "MyRoofGenius": ["mrg_starter", "mrg_pro"],
            "BrainOps AI OS": ["brainops_automation", "brainops_starter"],
            "Weathercraft ERP": ["erp_implementation", "erp_subscription"]
        }
    }


@router.post("/draft")
async def draft_proposal(request: DraftProposalRequest) -> dict[str, Any]:
    """
    Create a draft proposal for a lead.

    The proposal will need to be approved before sending.
    """
    engine = get_proposal_engine()

    success, message, proposal = await engine.draft_proposal(
        lead_id=request.lead_id,
        offer_id=request.offer_id,
        custom_notes=request.custom_notes,
        discount_percent=request.discount_percent
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "proposal_id": proposal.id,
        "status": proposal.status.value,
        "offer_name": proposal.offer_name,
        "final_price": float(proposal.final_price) if proposal.final_price else None,
        "next_step": f"POST /proposals/{proposal.id}/submit-approval to request approval"
    }


@router.post("/{proposal_id}/submit-approval")
async def submit_for_approval(proposal_id: str) -> dict[str, Any]:
    """
    Submit a draft proposal for human approval.

    Creates an approval item in the approval queue.
    """
    engine = get_proposal_engine()

    success, message = await engine.submit_for_approval(proposal_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "proposal_id": proposal_id,
        "status": "pending_approval",
        "next_step": f"POST /proposals/{proposal_id}/approve to approve the proposal"
    }


@router.post("/{proposal_id}/approve")
async def approve_proposal(
    proposal_id: str,
    request: ApproveProposalRequest
) -> dict[str, Any]:
    """
    Approve a proposal (human-in-the-loop).

    Only approved proposals can be sent to clients.
    """
    engine = get_proposal_engine()

    success, message = await engine.approve_proposal(
        proposal_id=proposal_id,
        approved_by=request.approved_by
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "proposal_id": proposal_id,
        "status": "approved",
        "approved_by": request.approved_by,
        "next_step": f"POST /proposals/{proposal_id}/send to send to client"
    }


@router.post("/{proposal_id}/send")
async def send_proposal(proposal_id: str) -> dict[str, Any]:
    """
    Send an approved proposal to the client.

    Requires prior approval. Queues email and updates lead state.
    """
    engine = get_proposal_engine()

    success, message, public_link = await engine.send_proposal(proposal_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "proposal_id": proposal_id,
        "status": "sent",
        "public_link": public_link,
        "note": "Email queued for delivery. Lead state updated to PROPOSAL_SENT."
    }


@router.get("/{proposal_id}")
async def get_proposal(proposal_id: str) -> dict[str, Any]:
    """Get a proposal by ID."""
    engine = get_proposal_engine()

    proposal = await engine.get_proposal(proposal_id)

    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found")

    # Mask email for security
    if proposal.get("client_email"):
        email = proposal["client_email"]
        parts = email.split("@")
        proposal["client_email_masked"] = f"{parts[0][:3]}***@{parts[1]}" if len(parts) == 2 else "***"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "proposal": {
            "id": str(proposal["id"]),
            "lead_id": str(proposal["lead_id"]),
            "status": proposal["status"],
            "offer_name": proposal["offer_name"],
            "price": float(proposal["price"]) if proposal["price"] else None,
            "final_price": float(proposal["final_price"]) if proposal["final_price"] else None,
            "client_company": proposal["client_company"],
            "client_email_masked": proposal.get("client_email_masked"),
            "created_at": proposal["created_at"].isoformat() if proposal["created_at"] else None,
            "sent_at": proposal["sent_at"].isoformat() if proposal.get("sent_at") else None,
            "public_link": proposal.get("public_link")
        }
    }


@router.get("/lead/{lead_id}")
async def get_proposals_for_lead(lead_id: str) -> dict[str, Any]:
    """Get all proposals for a lead."""
    engine = get_proposal_engine()

    proposals = await engine.get_proposals_for_lead(lead_id)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lead_id": lead_id,
        "total_proposals": len(proposals),
        "proposals": [
            {
                "id": str(p["id"]),
                "status": p["status"],
                "offer_name": p["offer_name"],
                "final_price": float(p["final_price"]) if p["final_price"] else None,
                "created_at": p["created_at"].isoformat() if p["created_at"] else None
            }
            for p in proposals
        ]
    }


@router.get("/pending")
async def get_pending_proposals() -> dict[str, Any]:
    """Get all proposals pending approval."""
    from database.async_connection import get_pool

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    rows = await pool.fetch("""
        SELECT id, lead_id, offer_name, client_company, final_price, created_at
        FROM ai_proposals
        WHERE status = 'pending_approval'
        ORDER BY created_at ASC
    """)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_pending": len(rows),
        "proposals": [
            {
                "id": str(r["id"]),
                "lead_id": str(r["lead_id"]),
                "offer_name": r["offer_name"],
                "client_company": r["client_company"],
                "final_price": float(r["final_price"]) if r["final_price"] else None,
                "created_at": r["created_at"].isoformat() if r["created_at"] else None
            }
            for r in rows
        ]
    }
