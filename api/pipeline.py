"""
Pipeline State Machine API
===========================
REST endpoints for lead state management with full audit trail.

All operations are ledger-backed and auditable.
Part of Revenue Perfection Session.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from pipeline_state_machine import (
    PipelineState,
    PipelineStateMachine,
    VALID_TRANSITIONS,
    get_state_machine,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline State Machine"])


class TransitionRequest(BaseModel):
    """Request to transition a lead to a new state."""
    to_state: str
    trigger: str
    actor: str = "human"
    metadata: Optional[dict] = None
    force: bool = False


class TransitionResponse(BaseModel):
    """Response from a state transition."""
    success: bool
    message: str
    lead_id: str
    from_state: Optional[str]
    to_state: Optional[str]
    transition_id: Optional[str]
    timestamp: str


@router.get("/states")
async def get_valid_states() -> dict[str, Any]:
    """
    Get all valid pipeline states and transitions.

    Returns the state machine definition.
    """
    return {
        "states": [s.value for s in PipelineState],
        "transitions": {
            s.value: [t.value for t in targets]
            for s, targets in VALID_TRANSITIONS.items()
        },
        "terminal_states": ["paid", "lost"],
        "description": {
            "new_real": "Verified real lead, not test/demo",
            "enriched": "Has company info, decision maker, pain points",
            "contact_ready": "All info available for outreach",
            "outreach_pending_approval": "Draft needs human approval",
            "outreach_sent": "First outreach delivered",
            "replied": "Prospect responded",
            "meeting_booked": "Call/meeting scheduled",
            "proposal_drafted": "Proposal created",
            "proposal_approved": "Proposal approved by human",
            "proposal_sent": "Proposal delivered to prospect",
            "won_invoice_pending": "Deal won, invoice not sent",
            "invoiced": "Invoice sent",
            "paid": "Payment received - REAL REVENUE",
            "lost": "Deal lost"
        }
    }


@router.get("/lead/{lead_id}/state")
async def get_lead_state(lead_id: str) -> dict[str, Any]:
    """Get current state of a lead from ledger."""
    sm = get_state_machine()
    current_state = await sm.get_lead_state(lead_id)

    if not current_state:
        raise HTTPException(status_code=404, detail="Lead not found")

    # Get valid next states
    try:
        state_enum = PipelineState(current_state)
        valid_next = [s.value for s in VALID_TRANSITIONS.get(state_enum, [])]
    except ValueError:
        valid_next = ["new_real"]  # Legacy state, can migrate

    return {
        "lead_id": lead_id,
        "current_state": current_state,
        "valid_next_states": valid_next,
        "is_terminal": current_state in ["paid", "lost"],
        "ledger_backed": True
    }


@router.get("/lead/{lead_id}/timeline")
async def get_lead_timeline(lead_id: str) -> dict[str, Any]:
    """Get full state transition history for a lead."""
    sm = get_state_machine()
    timeline = await sm.get_lead_timeline(lead_id)

    return {
        "lead_id": lead_id,
        "total_events": len(timeline),
        "timeline": timeline,
        "ledger_backed": True
    }


@router.post("/lead/{lead_id}/transition")
async def transition_lead(
    lead_id: str,
    request: TransitionRequest
) -> TransitionResponse:
    """
    Transition a lead to a new state.

    Requires human approval for certain transitions.
    All transitions are recorded in append-only ledger.
    """
    sm = get_state_machine()

    # Validate state
    try:
        to_state = PipelineState(request.to_state)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state: {request.to_state}. Valid states: {[s.value for s in PipelineState]}"
        )

    # Execute transition
    success, message, transition = await sm.transition(
        lead_id=lead_id,
        to_state=to_state,
        trigger=request.trigger,
        actor=request.actor,
        metadata=request.metadata,
        force=request.force
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return TransitionResponse(
        success=True,
        message=message,
        lead_id=lead_id,
        from_state=transition.from_state if transition else None,
        to_state=transition.to_state if transition else None,
        transition_id=transition.id if transition else None,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.post("/lead/{lead_id}/enrich")
async def enrich_lead(
    lead_id: str,
    enrichment_data: dict
) -> TransitionResponse:
    """
    Enrich a lead with additional data and transition to ENRICHED state.

    Accepts:
    - decision_maker: str
    - company_size: str
    - region: str
    - pain_points: list[str]
    - estimated_value_band: str
    """
    sm = get_state_machine()

    # Transition to ENRICHED
    success, message, transition = await sm.transition(
        lead_id=lead_id,
        to_state=PipelineState.ENRICHED,
        trigger="enrichment_completed",
        actor="system:enrichment",
        metadata={"enrichment": enrichment_data}
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return TransitionResponse(
        success=True,
        message="Lead enriched and transitioned",
        lead_id=lead_id,
        from_state=transition.from_state if transition else None,
        to_state=transition.to_state if transition else None,
        transition_id=transition.id if transition else None,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/stats")
async def get_pipeline_stats(
    real_only: bool = Query(default=True, description="Only count real leads")
) -> dict[str, Any]:
    """
    Get pipeline statistics from ledger facts.

    Returns state distribution and revenue from PAID leads.
    """
    sm = get_state_machine()
    stats = await sm.get_pipeline_stats(real_only=real_only)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_mode": "REAL_ONLY" if real_only else "ALL_DATA",
        **stats
    }


@router.post("/migrate")
async def migrate_legacy_leads() -> dict[str, Any]:
    """
    Migrate leads from legacy stage to new state machine.

    This creates initial state transition records for leads
    that don't have any in the ledger yet.
    """
    sm = get_state_machine()
    result = await sm.migrate_legacy_leads()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **result
    }


@router.post("/batch/advance-to-contact-ready")
async def batch_advance_to_contact_ready(
    actor: str = Query(default="system:batch"),
    limit: int = Query(default=10, le=50)
) -> dict[str, Any]:
    """
    Batch advance ENRICHED leads to CONTACT_READY state.

    Used for preparing leads for outreach.
    """
    from database.async_connection import get_pool

    pool = get_pool()
    if not pool:
        raise HTTPException(status_code=503, detail="Database not available")

    sm = get_state_machine()

    # Get enriched leads
    leads = await pool.fetch("""
        SELECT DISTINCT ON (ra.lead_id) ra.lead_id
        FROM revenue_actions ra
        JOIN revenue_leads rl ON ra.lead_id = rl.id
        WHERE ra.action_type = 'state_transition'
        AND ra.action_data->>'to_state' = 'enriched'
        AND rl.email NOT ILIKE '%test%'
        AND rl.email NOT ILIKE '%example%'
        ORDER BY ra.lead_id, ra.created_at DESC
        LIMIT $1
    """, limit)

    advanced = []
    for lead in leads:
        success, msg, _ = await sm.transition(
            str(lead["lead_id"]),
            PipelineState.CONTACT_READY,
            trigger="batch_advancement",
            actor=actor
        )
        if success:
            advanced.append(str(lead["lead_id"])[:8] + "...")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "advanced_count": len(advanced),
        "leads_advanced": advanced
    }
