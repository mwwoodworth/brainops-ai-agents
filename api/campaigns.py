"""
Campaign API
============
REST endpoints for managing campaigns, prospects, and outreach enrollment.
"""

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from campaign_manager import (
    get_campaign,
    list_campaigns,
    campaign_to_dict,
    personalize_template,
    notify_lead_to_partner,
    CAMPAIGNS,
)
from prospect_discovery import get_discovery_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/campaigns", tags=["Campaign Management"])


# ---- Pydantic models ----

class ProspectInput(BaseModel):
    company_name: str
    contact_name: Optional[str] = None
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None
    building_type: Optional[str] = None
    city: Optional[str] = None
    state: str = "CO"
    estimated_sqft: Optional[int] = None
    roof_system: Optional[str] = None


class BatchProspectsInput(BaseModel):
    prospects: list[ProspectInput]


class WebsiteDiscoveryInput(BaseModel):
    websites: list[str]
    building_type: Optional[str] = None
    city: Optional[str] = None


# ---- Campaign info endpoints ----

@router.get("/")
async def list_all_campaigns(active_only: bool = True) -> dict[str, Any]:
    """List all campaigns."""
    campaigns = list_campaigns(active_only=active_only)
    return {
        "campaigns": [campaign_to_dict(c) for c in campaigns],
        "total": len(campaigns),
    }


@router.get("/{campaign_id}")
async def get_campaign_details(campaign_id: str) -> dict[str, Any]:
    """Get campaign details including templates and partner info."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    result = campaign_to_dict(campaign)
    result["templates"] = [
        {
            "step": t.step,
            "delay_days": t.delay_days,
            "subject": t.subject,
            "call_to_action": t.call_to_action,
        }
        for t in campaign.templates
    ]
    if campaign.handoff_partner:
        result["handoff_partner_details"] = {
            "name": campaign.handoff_partner.name,
            "location": campaign.handoff_partner.location,
            "capabilities": campaign.handoff_partner.capabilities,
            "certifications": campaign.handoff_partner.certifications,
            "experience": campaign.handoff_partner.experience,
        }
    return result


@router.get("/{campaign_id}/stats")
async def get_campaign_stats(campaign_id: str) -> dict[str, Any]:
    """Get campaign performance metrics."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    engine = get_discovery_engine()
    stats = await engine.get_campaign_stats(campaign_id)
    stats["campaign_name"] = campaign.name
    stats["is_active"] = campaign.is_active
    return stats


# ---- Lead management endpoints ----

@router.get("/{campaign_id}/leads")
async def get_campaign_leads(
    campaign_id: str,
    stage: Optional[str] = None,
    limit: int = Query(default=100, le=500),
) -> dict[str, Any]:
    """List leads for a campaign."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    engine = get_discovery_engine()
    leads = await engine.get_campaign_leads(campaign_id, stage=stage, limit=limit)

    # Mask emails for API safety
    for lead in leads:
        if lead.get("email"):
            parts = lead["email"].split("@")
            if len(parts) == 2:
                local = parts[0]
                masked = local[:2] + "***" if len(local) > 2 else "***"
                lead["email_masked"] = f"{masked}@{parts[1]}"

    return {"campaign_id": campaign_id, "leads": leads, "total": len(leads)}


@router.post("/{campaign_id}/prospects")
async def add_prospect(campaign_id: str, prospect: ProspectInput) -> dict[str, Any]:
    """Add a single prospect to a campaign."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    engine = get_discovery_engine()
    result = await engine.add_prospect(
        campaign_id=campaign_id,
        company_name=prospect.company_name,
        contact_name=prospect.contact_name,
        email=prospect.email,
        phone=prospect.phone,
        website=prospect.website,
        building_type=prospect.building_type,
        city=prospect.city,
        state=prospect.state,
        estimated_sqft=prospect.estimated_sqft,
        roof_system=prospect.roof_system,
        discovery_source="api",
    )
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to add prospect"))

    # Notify partner for high-score prospects (score >= 50)
    if result.get("score", 0) >= 50:
        await notify_lead_to_partner(
            {
                "id": result.get("lead_id"),
                "company_name": prospect.company_name,
                "contact_name": prospect.contact_name,
                "email": prospect.email,
                "phone": prospect.phone,
                "website": prospect.website,
                "score": result.get("score", 0),
                "location": f"{prospect.city}, {prospect.state}" if prospect.city else prospect.state,
                "metadata": {
                    "building_type": prospect.building_type,
                    "city": prospect.city,
                    "state": prospect.state,
                    "estimated_sqft": prospect.estimated_sqft,
                    "roof_system": prospect.roof_system,
                },
            },
            campaign,
            event_type="new_prospect",
        )

    return result


@router.post("/{campaign_id}/prospects/batch")
async def add_prospects_batch(campaign_id: str, payload: BatchProspectsInput) -> dict[str, Any]:
    """Add multiple prospects to a campaign."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    engine = get_discovery_engine()
    prospects_data = [p.model_dump() for p in payload.prospects]
    for p in prospects_data:
        p["discovery_source"] = "api_batch"

    result = await engine.add_prospects_batch(campaign_id, prospects_data)
    return result


@router.post("/{campaign_id}/discover")
async def discover_from_websites(campaign_id: str, payload: WebsiteDiscoveryInput) -> dict[str, Any]:
    """Discover prospects by scraping emails from company websites."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    engine = get_discovery_engine()
    result = await engine.discover_from_website_list(
        campaign_id=campaign_id,
        websites=payload.websites,
        building_type=payload.building_type,
        city=payload.city,
    )
    return result


# ---- Outreach enrollment ----

@router.post("/{campaign_id}/outreach/enroll/{lead_id}")
async def enroll_lead_in_outreach(campaign_id: str, lead_id: str) -> dict[str, Any]:
    """Enroll a single lead in the campaign's outreach sequence."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    try:
        from database.async_connection import get_pool
        pool = get_pool()
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")

    # Fetch lead
    lead_row = await pool.fetchrow(
        "SELECT * FROM revenue_leads WHERE id = $1", uuid.UUID(lead_id)
    )
    if not lead_row:
        raise HTTPException(status_code=404, detail=f"Lead '{lead_id}' not found")

    lead_data = dict(lead_row)
    meta = lead_data.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    lead_data["metadata"] = meta or {}
    email = lead_data.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Lead has no email")

    # Schedule all campaign emails
    now = datetime.now(timezone.utc)
    queued = 0
    for template in campaign.templates:
        subject, body_html = personalize_template(template, lead_data, campaign)
        optimizer_meta: dict[str, Any] = {}
        try:
            from optimization.revenue_prompt_optimizer import get_revenue_prompt_optimizer

            optimizer = get_revenue_prompt_optimizer()
            leads_context = json.dumps(
                {
                    "lead_id": lead_id,
                    "company_name": lead_data.get("company_name"),
                    "contact_name": lead_data.get("contact_name"),
                    "email": lead_data.get("email"),
                    "industry": lead_data.get("industry"),
                    "source": lead_data.get("source"),
                    "metadata": lead_data.get("metadata") or {},
                },
                default=str,
            )
            revenue_metrics = json.dumps(
                {
                    "pipeline_stage": lead_data.get("stage"),
                    "campaign_id": campaign_id,
                    "template_step": template.step,
                    "delay_days": template.delay_days,
                    "source": "campaign_outreach",
                },
                default=str,
            )
            opt = await optimizer.optimize(
                leads=leads_context,
                revenue_metrics=revenue_metrics,
                subject=subject,
                body=body_html,
                pool=pool,
            )
            subject = opt.subject
            body_html = opt.body
            optimizer_meta = {
                "used_optimizer": opt.used_optimizer,
                "compiled": opt.compiled,
                "compiled_at": opt.compiled_at,
            }
        except Exception as exc:
            logger.debug("DSPy revenue optimizer unavailable: %s", exc)

        # Schedule at 9 AM Mountain Time (UTC-7)
        send_date = now + timedelta(days=template.delay_days)
        send_at = send_date.replace(hour=16, minute=0, second=0, microsecond=0)  # 16:00 UTC = 9 AM MT

        email_metadata = json.dumps({
            "campaign_id": campaign_id,
            "template_step": template.step,
            "lead_id": lead_id,
            "source": "campaign_outreach",
            "prompt_optimizer": optimizer_meta,
        })

        await pool.execute("""
            INSERT INTO ai_email_queue (
                id, recipient, subject, body, scheduled_for, status, metadata
            ) VALUES ($1, $2, $3, $4, $5, 'queued', $6)
        """,
            uuid.uuid4(), email, subject, body_html, send_at, email_metadata,
        )
        queued += 1

    # Update lead stage
    await pool.execute(
        "UPDATE revenue_leads SET stage = 'contacted', status = 'outreach_enrolled', updated_at = $1 WHERE id = $2",
        now, uuid.UUID(lead_id),
    )

    # Notify partner (matthew@weathercraft.net) about the enrolled lead
    await notify_lead_to_partner(lead_data, campaign, event_type="outreach_enrolled")

    logger.info(f"Lead {lead_id[:8]} enrolled in {campaign_id} outreach ({queued} emails queued)")
    return {
        "ok": True,
        "lead_id": lead_id,
        "campaign_id": campaign_id,
        "emails_queued": queued,
    }


@router.post("/{campaign_id}/outreach/batch")
async def batch_enroll_outreach(
    campaign_id: str,
    limit: int = Query(default=50, le=200),
) -> dict[str, Any]:
    """Enroll all un-contacted leads in outreach (up to limit)."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    try:
        from database.async_connection import get_pool
        pool = get_pool()
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")

    # Get leads that haven't been enrolled yet
    leads = await pool.fetch("""
        SELECT id FROM revenue_leads
        WHERE metadata->>'campaign_id' = $1
          AND stage = 'new'
          AND is_test = FALSE AND is_demo = FALSE
          AND email IS NOT NULL
        ORDER BY score DESC
        LIMIT $2
    """, campaign_id, limit)

    results = {"enrolled": 0, "errors": 0, "items": []}
    for lead in leads:
        try:
            lead_id = str(lead["id"])
            # Reuse the single enrollment logic inline
            lead_row = await pool.fetchrow(
                "SELECT * FROM revenue_leads WHERE id = $1", lead["id"]
            )
            if not lead_row:
                results["errors"] += 1
                continue

            lead_data = dict(lead_row)
            meta = lead_data.get("metadata")
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            lead_data["metadata"] = meta or {}
            email = lead_data.get("email")
            if not email:
                results["errors"] += 1
                continue

            now = datetime.now(timezone.utc)
            for template in campaign.templates:
                subject, body_html = personalize_template(template, lead_data, campaign)
                optimizer_meta: dict[str, Any] = {}
                try:
                    from optimization.revenue_prompt_optimizer import get_revenue_prompt_optimizer

                    optimizer = get_revenue_prompt_optimizer()
                    leads_context = json.dumps(
                        {
                            "lead_id": lead_id,
                            "company_name": lead_data.get("company_name"),
                            "contact_name": lead_data.get("contact_name"),
                            "email": lead_data.get("email"),
                            "industry": lead_data.get("industry"),
                            "source": lead_data.get("source"),
                            "metadata": lead_data.get("metadata") or {},
                        },
                        default=str,
                    )
                    revenue_metrics = json.dumps(
                        {
                            "pipeline_stage": lead_data.get("stage"),
                            "campaign_id": campaign_id,
                            "template_step": template.step,
                            "delay_days": template.delay_days,
                            "source": "campaign_outreach",
                        },
                        default=str,
                    )
                    opt = await optimizer.optimize(
                        leads=leads_context,
                        revenue_metrics=revenue_metrics,
                        subject=subject,
                        body=body_html,
                        pool=pool,
                    )
                    subject = opt.subject
                    body_html = opt.body
                    optimizer_meta = {
                        "used_optimizer": opt.used_optimizer,
                        "compiled": opt.compiled,
                        "compiled_at": opt.compiled_at,
                    }
                except Exception as exc:
                    logger.debug("DSPy revenue optimizer unavailable: %s", exc)

                send_date = now + timedelta(days=template.delay_days)
                send_at = send_date.replace(hour=16, minute=0, second=0, microsecond=0)

                email_metadata = json.dumps({
                    "campaign_id": campaign_id,
                    "template_step": template.step,
                    "lead_id": lead_id,
                    "source": "campaign_outreach",
                    "prompt_optimizer": optimizer_meta,
                })

                await pool.execute("""
                    INSERT INTO ai_email_queue (
                        id, recipient, subject, body, scheduled_for, status, metadata
                    ) VALUES ($1, $2, $3, $4, $5, 'queued', $6)
                """,
                    uuid.uuid4(), email, subject, body_html, send_at, email_metadata,
                )

            await pool.execute(
                "UPDATE revenue_leads SET stage = 'contacted', status = 'outreach_enrolled', updated_at = $1 WHERE id = $2",
                now, lead["id"],
            )
            results["enrolled"] += 1
            results["items"].append({"lead_id": lead_id, "status": "enrolled"})

        except Exception as e:
            results["errors"] += 1
            results["items"].append({"lead_id": str(lead["id"]), "status": f"error: {e}"})

    return {
        "campaign_id": campaign_id,
        "total_eligible": len(leads),
        **results,
    }


# ---- Lead reply / interest notification ----

class LeadReplyInput(BaseModel):
    summary: str
    reply_text: Optional[str] = None


@router.post("/{campaign_id}/leads/{lead_id}/reply")
async def log_lead_reply(campaign_id: str, lead_id: str, payload: LeadReplyInput) -> dict[str, Any]:
    """Log a lead reply and immediately notify partner via email."""
    campaign = get_campaign(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign '{campaign_id}' not found")

    try:
        from database.async_connection import get_pool
        pool = get_pool()
    except Exception:
        raise HTTPException(status_code=503, detail="Database not available")

    lead_row = await pool.fetchrow("SELECT * FROM revenue_leads WHERE id = $1", uuid.UUID(lead_id))
    if not lead_row:
        raise HTTPException(status_code=404, detail=f"Lead '{lead_id}' not found")

    lead_data = dict(lead_row)
    meta = lead_data.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    lead_data["metadata"] = meta or {}

    now = datetime.now(timezone.utc)

    # Update lead state via ledger-backed state machine (best-effort).
    transitioned = False
    try:
        from pipeline_state_machine import get_state_machine, PipelineState

        sm = get_state_machine()
        ok, _msg, _transition = await sm.transition(
            lead_id,
            PipelineState.REPLIED,
            trigger="reply_received",
            actor="human:campaigns_api",
            metadata={
                "campaign_id": campaign_id,
                "summary": payload.summary,
                "reply_text": payload.reply_text,
            },
            force=True,  # reply implies outreach happened even if earlier states weren't logged
        )
        transitioned = bool(ok)
    except Exception as exc:
        logger.debug("State machine transition skipped for reply_received: %s", exc)

    # Keep status denormalized for backwards compatibility. If transition failed, fall back to stage update.
    if transitioned:
        await pool.execute(
            "UPDATE revenue_leads SET status = 'replied', updated_at = $1 WHERE id = $2",
            now,
            uuid.UUID(lead_id),
        )
    else:
        await pool.execute(
            "UPDATE revenue_leads SET stage = 'replied', status = 'replied', updated_at = $1 WHERE id = $2",
            now,
            uuid.UUID(lead_id),
        )

    # Log engagement
    try:
        await pool.execute(
            """
            INSERT INTO lead_activities (
                id, lead_id, activity_type, subject, description,
                outcome, completed_at, created_at, created_by, event_type
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7, $8, $9)
            """,
            uuid.uuid4(),
            uuid.UUID(lead_id),
            "email",
            "Reply received",
            json.dumps({"summary": payload.summary, "reply_text": payload.reply_text}, default=str),
            "received",
            now,
            "human:campaigns_api",
            "reply_received",
        )
    except Exception:
        # Fallback: engagement history table may not exist in all DBs
        try:
            await pool.execute(
                """
                INSERT INTO lead_engagement_history (lead_id, event_type, event_data, channel, timestamp)
                VALUES ($1, 'reply_received', $2, 'email', $3)
                """,
                uuid.UUID(lead_id),
                json.dumps({"summary": payload.summary, "reply_text": payload.reply_text}),
                now,
            )
        except Exception:
            pass

    # Immediate priority notification to partner
    await notify_lead_to_partner(lead_data, campaign, event_type="reply_received")

    logger.info(f"Lead {lead_id[:8]} reply logged and partner notified for campaign {campaign_id}")
    return {
        "ok": True,
        "lead_id": lead_id,
        "stage": "replied",
        "partner_notified": True,
    }
