"""
Lead Engine Relay API
=====================
Bridge between MyRoofGenius and Weathercraft ERP.

Receives qualified leads from MRG's handoff endpoint and:
1. Enriches the lead with AI qualification
2. Syncs to ERP leads table (tenant-scoped)
3. Triggers notifications and nurture enrollment
4. Logs the handoff for attribution tracking

This is the CORE PIPELINE connecting the two systems.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/lead-engine", tags=["lead-engine"])

# Weathercraft tenant ID for lead ingestion
WEATHERCRAFT_TENANT_ID = os.getenv(
    "WEATHERCRAFT_TENANT_ID",
    "00000000-0000-0000-0000-000000000001"
)


class LeadData(BaseModel):
    mrg_lead_id: str
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    roof_type: Optional[str] = None
    square_footage: Optional[int] = None
    timeline: Optional[str] = None
    budget_range: Optional[str] = None
    ai_score: Optional[int] = None
    insurance_claim: bool = False
    primary_concern: Optional[str] = None
    roof_age_years: Optional[int] = None
    preferred_contact_method: str = "phone"
    preferred_time: Optional[str] = None
    additional_notes: Optional[str] = None


class RelayRequest(BaseModel):
    action: str = Field(..., description="Action type: 'handoff' or 'update'")
    lead: LeadData


class RelayResponse(BaseModel):
    success: bool
    erp_lead_id: Optional[str] = None
    message: str
    attribution: dict = {}


def get_db_pool():
    """Get database connection pool."""
    try:
        from database.async_connection import get_pool
        return get_pool()
    except Exception as e:
        logger.error(f"Failed to get database pool: {e}")
        return None


@router.post("/relay", response_model=RelayResponse)
async def relay_lead(request: Request, payload: RelayRequest):
    """
    Relay a qualified lead from MyRoofGenius to Weathercraft ERP.

    This is the central bridge in the Lead Engine pipeline.
    """
    if payload.action not in ("handoff", "update"):
        raise HTTPException(status_code=400, detail="Invalid action. Use 'handoff' or 'update'.")

    lead = payload.lead
    pool = get_db_pool()

    if not pool:
        logger.error("Database pool unavailable for lead relay")
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        async with pool.acquire() as conn:
            now = datetime.now(timezone.utc).isoformat()

            if payload.action == "handoff":
                # Insert into ERP leads table (Weathercraft tenant)
                # NOTE: 'city' and 'state' columns do not exist in 'leads' table; mapping to 'address' and 'metadata'
                erp_lead_id = await conn.fetchval("""
                    INSERT INTO public.leads (
                        tenant_id, name, email, phone, address,
                        roof_type, square_footage,
                        urgency, insurance_claim, budget_range,
                        source, status, score, ai_score,
                        description, notes, metadata,
                        created_at, updated_at, last_activity_at
                    ) VALUES (
                        $1, $2, $3, $4, $5,
                        $6, $7,
                        $8, $9, $10,
                        $11, $12, $13, $14,
                        $15, $16, $17::jsonb,
                        $18, $18, $18
                    )
                    ON CONFLICT (tenant_id, related_entity_type, related_entity_id)
                        WHERE related_entity_type IS NOT NULL AND related_entity_id IS NOT NULL
                    DO UPDATE SET
                        status = 'qualified',
                        updated_at = $18,
                        last_activity_at = $18
                    RETURNING id::text
                """,
                    WEATHERCRAFT_TENANT_ID,
                    lead.name,
                    lead.email.lower(),
                    lead.phone,
                    _build_address(lead),
                    lead.roof_type,
                    lead.square_footage,
                    _calc_urgency(lead),
                    lead.insurance_claim,
                    lead.budget_range,
                    "myroofgenius_advisory",  # source
                    "qualified",  # status - already opted in
                    lead.ai_score or 70,
                    lead.ai_score or 70,
                    _build_description(lead),
                    lead.additional_notes,
                    _build_metadata(lead, now),
                    now,
                )

                if not erp_lead_id:
                    raise HTTPException(status_code=500, detail="Failed to create ERP lead")

                # Log the handoff event
                try:
                    await conn.execute("""
                        INSERT INTO public.lead_activities (
                            id, lead_id, activity_type, subject,
                            description, created_at
                        ) VALUES (
                            gen_random_uuid(), $1::uuid, $2, $3, $4, $5
                        )
                    """,
                        erp_lead_id,
                        "handoff_received",
                        "Lead handed off from MyRoofGenius",
                        f"Homeowner opted in via Roof Health Advisory. "
                        f"Score: {lead.ai_score}, Concern: {lead.primary_concern}, "
                        f"Timeline: {lead.timeline}",
                        now,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log handoff activity: {e}")

                # Trigger lead qualification agent (non-blocking)
                try:
                    from lead_qualification_agent import qualify_lead_by_id
                    # Fire and forget - don't await
                    import asyncio
                    asyncio.create_task(_qualify_async(erp_lead_id))
                except Exception as e:
                    logger.warning(f"Lead qualification trigger failed: {e}")

                # Enroll in nurture sequence (non-blocking)
                try:
                    from commercial_roof_sequences import enroll_lead_in_commercial_sequence
                    from services.lead_segmentation_sequences import enroll_lead_in_segment_sequence
                    
                    # Extract first name from full name
                    first_name = lead.name.split()[0] if lead.name else "there"
                    lead_dict = lead.model_dump()
                    
                    # Determine segment
                    segment = None
                    concern_lower = (lead.primary_concern or "").lower()
                    
                    if "leak" in concern_lower or "repair" in concern_lower or "maintenance" in concern_lower:
                        segment = "service"
                    elif "replacement" in concern_lower or "age" in concern_lower or "insurance" in concern_lower or "storm" in concern_lower:
                        segment = "reroof"
                    
                    if segment:
                        enroll_lead_in_segment_sequence(
                            lead_id=erp_lead_id,
                            email=lead.email,
                            first_name=first_name,
                            segment=segment,
                            lead_data=lead_dict
                        )
                        logger.info(f"Lead {erp_lead_id} enrolled in {segment} sequence")
                    else:
                        # Fallback to commercial sequence
                        enroll_lead_in_commercial_sequence(
                            lead_id=erp_lead_id,
                            email=lead.email,
                            first_name=first_name,
                            roof_type=lead.roof_type,
                            square_footage=lead.square_footage,
                            roof_age_years=lead.roof_age_years,
                            primary_concern=lead.primary_concern,
                            preferred_contact_method=lead.preferred_contact_method,
                        )
                        logger.info(f"Lead {erp_lead_id} enrolled in commercial roof nurture sequence")
                except Exception as e:
                    logger.warning(f"Nurture enrollment failed (non-critical): {e}")

                logger.info(
                    f"Lead handoff complete: MRG {lead.mrg_lead_id} -> ERP {erp_lead_id} "
                    f"(score={lead.ai_score}, concern={lead.primary_concern})"
                )

                return RelayResponse(
                    success=True,
                    erp_lead_id=erp_lead_id,
                    message="Lead successfully relayed to Weathercraft ERP",
                    attribution={
                        "source": "myroofgenius_advisory",
                        "mrg_lead_id": lead.mrg_lead_id,
                        "handoff_at": now,
                        "ai_score": lead.ai_score,
                        "funnel": "roof_health_advisory",
                    },
                )

            elif payload.action == "update":
                # Update existing ERP lead status
                result = await conn.execute("""
                    UPDATE public.leads
                    SET status = 'contacted',
                        updated_at = $1,
                        last_activity_at = $1
                    WHERE metadata->>'mrg_lead_id' = $2
                      AND tenant_id = $3
                """, now, lead.mrg_lead_id, WEATHERCRAFT_TENANT_ID)

                return RelayResponse(
                    success=True,
                    message="Lead status updated in ERP",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lead relay error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Relay failed: {str(e)}")


@router.get("/stats")
async def lead_engine_stats(request: Request):
    """Get Lead Engine pipeline statistics."""
    pool = get_db_pool()
    if not pool:
        return {"error": "Database unavailable"}

    try:
        async with pool.acquire() as conn:
            # MRG leads by status
            mrg_stats = await conn.fetch("""
                SELECT status, count(*) as count
                FROM public.mrg_leads
                GROUP BY status
                ORDER BY count DESC
            """)

            # ERP leads from MRG source
            erp_mrg_leads = await conn.fetchrow("""
                SELECT
                    count(*) as total,
                    count(*) FILTER (WHERE status = 'qualified') as qualified,
                    count(*) FILTER (WHERE status = 'converted') as converted,
                    avg(score) as avg_score
                FROM public.leads
                WHERE source = 'myroofgenius_advisory'
            """)

            # Revenue leads from MRG
            revenue_mrg = await conn.fetchrow("""
                SELECT
                    count(*) as total,
                    sum(value_estimate) as total_value,
                    count(*) FILTER (WHERE stage = 'won') as won
                FROM public.revenue_leads
                WHERE source = 'mrg_roof_advisory'
                  AND NOT COALESCE(is_test, false)
            """)

            return {
                "mrg_pipeline": {row["status"]: row["count"] for row in mrg_stats},
                "erp_leads": {
                    "total": erp_mrg_leads["total"] if erp_mrg_leads else 0,
                    "qualified": erp_mrg_leads["qualified"] if erp_mrg_leads else 0,
                    "converted": erp_mrg_leads["converted"] if erp_mrg_leads else 0,
                    "avg_score": float(erp_mrg_leads["avg_score"] or 0) if erp_mrg_leads else 0,
                },
                "revenue": {
                    "total_leads": revenue_mrg["total"] if revenue_mrg else 0,
                    "pipeline_value": float(revenue_mrg["total_value"] or 0) if revenue_mrg else 0,
                    "won": revenue_mrg["won"] if revenue_mrg else 0,
                },
            }
    except Exception as e:
        logger.error(f"Lead engine stats error: {e}")
        return {"error": str(e)}


async def _qualify_async(lead_id: str):
    """Run lead qualification asynchronously."""
    try:
        from lead_qualification_agent import qualify_lead_by_id
        await qualify_lead_by_id(lead_id)
    except Exception as e:
        logger.warning(f"Async lead qualification failed for {lead_id}: {e}")


def _build_address(lead: LeadData) -> str | None:
    parts = [lead.address, lead.city, lead.state, lead.zip]
    filtered = [p for p in parts if p]
    return ", ".join(filtered) if filtered else None


def _calc_urgency(lead: LeadData) -> str:
    if lead.insurance_claim or lead.timeline == "immediate":
        return "high"
    if lead.timeline == "1_3_months":
        return "medium"
    return "normal"


def _build_description(lead: LeadData) -> str:
    parts = []
    if lead.primary_concern:
        parts.append(f"Primary concern: {lead.primary_concern}")
    if lead.roof_type:
        parts.append(f"Roof type: {lead.roof_type}")
    if lead.square_footage:
        parts.append(f"Size: {lead.square_footage} sqft")
    if lead.roof_age_years:
        parts.append(f"Roof age: {lead.roof_age_years} years")
    if lead.timeline:
        parts.append(f"Timeline: {lead.timeline}")
    if lead.insurance_claim:
        parts.append("Insurance claim: Yes")
    parts.append("Source: MyRoofGenius Roof Health Advisory")
    return ". ".join(parts)


def _build_metadata(lead: LeadData, timestamp: str) -> str:
    import json
    return json.dumps({
        "mrg_lead_id": lead.mrg_lead_id,
        "source_platform": "myroofgenius",
        "funnel": "roof_health_advisory",
        "handoff_at": timestamp,
        "city": lead.city,
        "state": lead.state,
        "zip": lead.zip,
        "roof_type": lead.roof_type,
        "roof_age_years": lead.roof_age_years,
        "square_footage": lead.square_footage,
        "primary_concern": lead.primary_concern,
        "insurance_claim": lead.insurance_claim,
        "timeline": lead.timeline,
        "budget_range": lead.budget_range,
        "preferred_contact_method": lead.preferred_contact_method,
        "preferred_time": lead.preferred_time,
        "ai_score_at_handoff": lead.ai_score,
    })


class FeedbackRequest(BaseModel):
    lead_id: str
    event_type: str  # opened, clicked, responded, scheduled, converted, opted_out
    event_data: Optional[dict] = None


@router.post("/feedback")
async def lead_feedback(request: Request, payload: FeedbackRequest):
    """
    Receive engagement feedback from ERP or email tracking.

    Events:
    - opened: Email opened
    - clicked: Link clicked in email
    - responded: Lead replied to email
    - scheduled: Assessment scheduled
    - converted: Lead converted to customer
    - opted_out: Lead unsubscribed
    """
    valid_events = {"opened", "clicked", "responded", "scheduled", "converted", "opted_out"}
    if payload.event_type not in valid_events:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type. Must be one of: {', '.join(valid_events)}"
        )

    try:
        from commercial_roof_sequences import update_lead_engagement
        success = update_lead_engagement(
            lead_id=payload.lead_id,
            event_type=payload.event_type,
            event_data=payload.event_data,
        )

        if success:
            logger.info(f"Feedback recorded: {payload.lead_id} - {payload.event_type}")
            return {"success": True, "message": f"Feedback recorded: {payload.event_type}"}
        else:
            return {"success": False, "message": "Failed to record feedback"}

    except Exception as e:
        logger.error(f"Feedback processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
