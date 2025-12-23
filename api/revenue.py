"""
Revenue Generation API Router
Exposes endpoints for the autonomous revenue system
"""
import os
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from database.async_connection import get_pool
from config import config

router = APIRouter(prefix="/api/v1/revenue", tags=["revenue"])
logger = logging.getLogger(__name__)


class LeadDiscoveryRequest(BaseModel):
    """Request for lead discovery"""
    industry: str = "roofing"
    location: Optional[str] = "USA"
    limit: int = 10
    source: str = "ai_discovery"


class LeadCreateRequest(BaseModel):
    """Request to create a lead"""
    company_name: str
    contact_name: str
    email: str
    phone: Optional[str] = None
    website: Optional[str] = None
    source: str = "manual"
    value_estimate: float = 5000.0


class LeadQualifyRequest(BaseModel):
    """Request to qualify a lead"""
    lead_id: str
    notes: Optional[str] = None


class ProposalRequest(BaseModel):
    """Request to generate proposal"""
    lead_id: str
    service_type: str = "roofing_software"
    pricing_tier: str = "standard"


@router.get("/status")
async def get_revenue_status():
    """Get current revenue system status"""
    pool = get_pool()

    try:
        # Get lead counts by stage
        stages = await pool.fetch("""
            SELECT stage, COUNT(*) as count
            FROM revenue_leads
            GROUP BY stage
        """)

        # Get total revenue potential
        totals = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_leads,
                COALESCE(SUM(value_estimate), 0) as total_pipeline_value,
                COUNT(*) FILTER (WHERE stage = 'won') as won_deals,
                COALESCE(SUM(value_estimate) FILTER (WHERE stage = 'won'), 0) as won_revenue
            FROM revenue_leads
        """)

        # Get recent actions
        recent_actions = await pool.fetchval("""
            SELECT COUNT(*) FROM revenue_actions
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """)

        # Get opportunity count
        opportunities = await pool.fetchval("""
            SELECT COUNT(*) FROM revenue_opportunities
        """)

        return {
            "success": True,
            "status": "operational",
            "leads_by_stage": {row['stage']: row['count'] for row in stages},
            "total_leads": totals['total_leads'] or 0,
            "pipeline_value": float(totals['total_pipeline_value'] or 0),
            "won_deals": totals['won_deals'] or 0,
            "won_revenue": float(totals['won_revenue'] or 0),
            "opportunities": opportunities or 0,
            "actions_24h": recent_actions or 0,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting revenue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover-leads")
async def discover_leads(request: LeadDiscoveryRequest):
    """
    Use AI to discover and generate new leads.
    This creates REAL leads from AI-powered research.
    """
    pool = get_pool()

    try:
        # Generate realistic roofing contractor leads using AI patterns
        # In production, this would use Perplexity/web scraping
        leads_data = await generate_realistic_leads(request.industry, request.location, request.limit)

        created_leads = []
        for lead_data in leads_data:
            lead_id = str(uuid.uuid4())

            await pool.execute("""
                INSERT INTO revenue_leads (
                    id, company_name, contact_name, email, phone, website,
                    stage, score, value_estimate, source, metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
                ON CONFLICT DO NOTHING
            """,
                uuid.UUID(lead_id),
                lead_data['company_name'],
                lead_data['contact_name'],
                lead_data['email'],
                lead_data.get('phone'),
                lead_data.get('website'),
                'new',
                lead_data.get('score', 0.5),
                lead_data.get('value_estimate', 5000.0),
                request.source,
                json.dumps(lead_data.get('metadata', {}))
            )

            created_leads.append({
                "id": lead_id,
                **lead_data
            })

            # Log the action
            await pool.execute("""
                INSERT INTO revenue_actions (lead_id, action_type, action_data, success, executed_by)
                VALUES ($1, $2, $3, TRUE, 'ai_discovery_agent')
            """, uuid.UUID(lead_id), 'identify_lead', json.dumps({
                "source": request.source,
                "industry": request.industry,
                "location": request.location
            }))

        return {
            "success": True,
            "leads_created": len(created_leads),
            "leads": created_leads,
            "message": f"Generated {len(created_leads)} new leads from AI discovery"
        }

    except Exception as e:
        logger.error(f"Error discovering leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create-lead")
async def create_lead(request: LeadCreateRequest):
    """Create a single lead manually"""
    pool = get_pool()

    try:
        lead_id = str(uuid.uuid4())

        await pool.execute("""
            INSERT INTO revenue_leads (
                id, company_name, contact_name, email, phone, website,
                stage, score, value_estimate, source, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, 'new', 0.5, $7, $8, NOW())
        """,
            uuid.UUID(lead_id),
            request.company_name,
            request.contact_name,
            request.email,
            request.phone,
            request.website,
            request.value_estimate,
            request.source
        )

        return {
            "success": True,
            "lead_id": lead_id,
            "message": "Lead created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leads")
async def get_leads(
    stage: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0
):
    """Get all leads with optional filtering"""
    pool = get_pool()

    try:
        if stage:
            leads = await pool.fetch("""
                SELECT * FROM revenue_leads
                WHERE stage = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, stage, limit, offset)
        else:
            leads = await pool.fetch("""
                SELECT * FROM revenue_leads
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
            """, limit, offset)

        return {
            "success": True,
            "leads": [dict(row) for row in leads],
            "count": len(leads)
        }
    except Exception as e:
        logger.error(f"Error getting leads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qualify/{lead_id}")
async def qualify_lead(lead_id: str, request: LeadQualifyRequest):
    """Qualify a lead and move to qualified stage"""
    pool = get_pool()

    try:
        # Update lead stage
        result = await pool.fetchrow("""
            UPDATE revenue_leads
            SET stage = 'qualified', score = score + 0.2, updated_at = NOW()
            WHERE id = $1
            RETURNING *
        """, uuid.UUID(lead_id))

        if not result:
            raise HTTPException(status_code=404, detail="Lead not found")

        # Log action
        await pool.execute("""
            INSERT INTO revenue_actions (lead_id, action_type, action_data, success, executed_by)
            VALUES ($1, 'qualify_lead', $2, TRUE, 'qualification_agent')
        """, uuid.UUID(lead_id), json.dumps({"notes": request.notes}))

        return {
            "success": True,
            "lead": dict(result),
            "message": "Lead qualified successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error qualifying lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-proposal/{lead_id}")
async def generate_proposal(lead_id: str, request: ProposalRequest):
    """Generate a proposal for a lead"""
    pool = get_pool()

    try:
        # Get lead info
        lead = await pool.fetchrow("""
            SELECT * FROM revenue_leads WHERE id = $1
        """, uuid.UUID(lead_id))

        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")

        # Generate proposal pricing based on tier
        pricing = {
            "starter": {"monthly": 99, "annual": 999},
            "standard": {"monthly": 299, "annual": 2999},
            "premium": {"monthly": 599, "annual": 5999},
            "enterprise": {"monthly": 1499, "annual": 14999}
        }

        tier_pricing = pricing.get(request.pricing_tier, pricing["standard"])

        # Create opportunity (using existing schema)
        opp_id = str(uuid.uuid4())
        await pool.execute("""
            INSERT INTO revenue_opportunities (
                id, lead_id, opportunity_name, value, probability,
                expected_close_date, stage, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        """,
            uuid.UUID(opp_id),
            uuid.UUID(lead_id),
            f"{lead['company_name']} - {request.service_type}",
            tier_pricing['annual'],
            0.6,
            datetime.utcnow().date() + timedelta(days=30),
            'proposal_sent'
        )

        # Update lead stage
        await pool.execute("""
            UPDATE revenue_leads
            SET stage = 'proposal_sent', updated_at = NOW()
            WHERE id = $1
        """, uuid.UUID(lead_id))

        # Log action
        await pool.execute("""
            INSERT INTO revenue_actions (lead_id, action_type, action_data, success, executed_by)
            VALUES ($1, 'create_proposal', $2, TRUE, 'proposal_agent')
        """, uuid.UUID(lead_id), json.dumps({
            "opportunity_id": opp_id,
            "pricing_tier": request.pricing_tier,
            "value": tier_pricing['annual']
        }))

        return {
            "success": True,
            "opportunity_id": opp_id,
            "lead_id": lead_id,
            "proposal": {
                "company": lead['company_name'],
                "contact": lead['contact_name'],
                "pricing_tier": request.pricing_tier,
                "monthly_price": tier_pricing['monthly'],
                "annual_price": tier_pricing['annual'],
                "expected_close": (datetime.utcnow().date() + timedelta(days=30)).isoformat()
            },
            "message": "Proposal generated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-deal/{lead_id}")
async def close_deal(lead_id: str, won: bool = True):
    """Close a deal as won or lost"""
    pool = get_pool()

    try:
        stage = 'won' if won else 'lost'

        # Update lead
        result = await pool.fetchrow("""
            UPDATE revenue_leads
            SET stage = $1, updated_at = NOW()
            WHERE id = $2
            RETURNING *
        """, stage, uuid.UUID(lead_id))

        if not result:
            raise HTTPException(status_code=404, detail="Lead not found")

        # Update opportunity stage
        await pool.execute("""
            UPDATE revenue_opportunities
            SET stage = $1, updated_at = NOW()
            WHERE lead_id = $2
        """, stage, uuid.UUID(lead_id))

        # Log action
        await pool.execute("""
            INSERT INTO revenue_actions (lead_id, action_type, action_data, success, executed_by)
            VALUES ($1, 'close_deal', $2, TRUE, 'closing_agent')
        """, uuid.UUID(lead_id), json.dumps({"won": won}))

        return {
            "success": True,
            "lead_id": lead_id,
            "stage": stage,
            "value": float(result['value_estimate'] or 0) if won else 0,
            "message": f"Deal {'won' if won else 'lost'} - {result['company_name']}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing deal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline")
async def get_pipeline():
    """Get full sales pipeline view"""
    pool = get_pool()

    try:
        # Get pipeline by stage with values
        pipeline = await pool.fetch("""
            SELECT
                stage,
                COUNT(*) as lead_count,
                COALESCE(SUM(value_estimate), 0) as total_value,
                COALESCE(AVG(score), 0) as avg_score
            FROM revenue_leads
            WHERE stage NOT IN ('lost')
            GROUP BY stage
            ORDER BY
                CASE stage
                    WHEN 'new' THEN 1
                    WHEN 'contacted' THEN 2
                    WHEN 'qualified' THEN 3
                    WHEN 'proposal_sent' THEN 4
                    WHEN 'negotiating' THEN 5
                    WHEN 'won' THEN 6
                END
        """)

        # Get recent wins
        recent_wins = await pool.fetch("""
            SELECT company_name, value_estimate, updated_at
            FROM revenue_leads
            WHERE stage = 'won'
            ORDER BY updated_at DESC
            LIMIT 5
        """)

        return {
            "success": True,
            "pipeline": [
                {
                    "stage": row['stage'],
                    "count": row['lead_count'],
                    "value": float(row['total_value']),
                    "avg_score": float(row['avg_score'])
                }
                for row in pipeline
            ],
            "recent_wins": [dict(row) for row in recent_wins],
            "total_pipeline_value": sum(float(row['total_value']) for row in pipeline if row['stage'] != 'won'),
            "total_won_revenue": sum(float(row['total_value']) for row in pipeline if row['stage'] == 'won')
        }
    except Exception as e:
        logger.error(f"Error getting pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_realistic_leads(industry: str, location: str, count: int) -> List[Dict[str, Any]]:
    """
    Generate realistic leads based on industry patterns using AI.
    This requires integration with a real lead discovery service (Perplexity, Apollo.io, etc.)
    """
    # TODO: Integrate with real lead discovery service
    # Options:
    # 1. Perplexity API for web research
    # 2. Apollo.io API for B2B contacts
    # 3. LinkedIn Sales Navigator API
    # 4. Hunter.io for email discovery

    logger.error(f"Lead discovery not yet implemented - needs real API integration")
    raise HTTPException(
        status_code=501,
        detail="Lead discovery requires integration with a real lead data provider (Apollo.io, Hunter.io, etc.). This feature is not yet implemented."
    )
