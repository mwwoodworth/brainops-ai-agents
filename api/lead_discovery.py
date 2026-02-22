"""
Lead Discovery API Router
========================
Exposes endpoints for the autonomous lead discovery system.

IMPORTANT: This API does NOT perform any outbound marketing or outreach.
It only discovers, qualifies, and syncs leads for human review and action.
"""

import logging
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from config import config
from lead_discovery_engine import (
    DiscoveredLead,
    LeadDiscoveryEngine,
    LeadQualificationCriteria,
    LeadQualificationStatus,
    LeadSource,
    LeadTier,
    get_discovery_engine,
)

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/api/v1/leads",
    tags=["lead-discovery"],
    dependencies=[Depends(verify_api_key)]
)


# ====================
# Request/Response Models
# ====================

class DiscoverLeadsRequest(BaseModel):
    """Request to discover leads from specified sources"""
    sources: Optional[list[str]] = Field(
        default=None,
        description="List of source types to query. None = all enabled sources."
    )
    limit: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum leads to discover per source"
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Minimum score threshold for qualification"
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant ID for multi-tenant isolation"
    )


class QualifyLeadRequest(BaseModel):
    """Request to qualify a single lead"""
    company_name: str
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    industry: str = "roofing"
    source: str = "manual_entry"
    source_detail: Optional[str] = None
    estimated_value: float = 5000.0
    signals: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SyncLeadRequest(BaseModel):
    """Request to sync a lead to ERP or revenue_leads"""
    lead_id: str
    sync_to_erp: bool = Field(
        default=False,
        description="Sync to ERP leads table (requires tenant_id)"
    )
    sync_to_revenue: bool = Field(
        default=True,
        description="Sync to revenue_leads table"
    )
    tenant_id: Optional[str] = None


class QualificationCriteriaRequest(BaseModel):
    """Request to update qualification criteria"""
    min_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    require_email: Optional[bool] = None
    require_phone: Optional[bool] = None
    require_company: Optional[bool] = None
    require_location: Optional[bool] = None
    excluded_domains: Optional[list[str]] = None
    min_estimated_value: Optional[float] = Field(default=None, ge=0.0)
    industries: Optional[list[str]] = None


class LeadResponse(BaseModel):
    """Response model for a discovered lead"""
    id: str
    company_name: str
    contact_name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    website: Optional[str]
    location: Optional[str]
    industry: str
    source: str
    source_detail: str
    score: float
    tier: str
    qualification_status: str
    estimated_value: float
    signals: list[str]
    metadata: dict[str, Any]
    discovered_at: str


class DiscoveryResponse(BaseModel):
    """Response model for lead discovery"""
    success: bool
    leads_found: int
    leads_qualified: int
    leads: list[LeadResponse]
    discovery_time_ms: int
    message: str


# ====================
# Discovery Endpoints
# ====================

@router.post("/discover", response_model=DiscoveryResponse)
async def discover_leads(request: DiscoverLeadsRequest):
    """
    Discover leads from configured sources.

    This endpoint:
    1. Queries multiple lead sources (ERP data, web search, social signals)
    2. Qualifies discovered leads against configurable criteria
    3. Scores and prioritizes leads
    4. Returns qualified leads for human review

    NOTE: This does NOT perform any outbound marketing or outreach.
    """
    start_time = datetime.utcnow()

    try:
        engine = get_discovery_engine(request.tenant_id)

        # Run discovery
        qualified_leads = await engine.discover_leads(
            sources=request.sources,
            limit=request.limit,
            min_score=request.min_score
        )

        # Calculate timing
        elapsed_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        # Format response
        lead_responses = [
            LeadResponse(
                id=lead.id,
                company_name=lead.company_name,
                contact_name=lead.contact_name,
                email=lead.email,
                phone=lead.phone,
                website=lead.website,
                location=lead.location,
                industry=lead.industry,
                source=lead.source.value,
                source_detail=lead.source_detail,
                score=lead.score,
                tier=lead.tier.value,
                qualification_status=lead.qualification_status.value,
                estimated_value=lead.estimated_value,
                signals=lead.signals,
                metadata=lead.metadata,
                discovered_at=lead.discovered_at.isoformat()
            )
            for lead in qualified_leads
        ]

        return DiscoveryResponse(
            success=True,
            leads_found=len(qualified_leads),
            leads_qualified=len(qualified_leads),
            leads=lead_responses,
            discovery_time_ms=elapsed_ms,
            message=f"Discovered {len(qualified_leads)} qualified leads from {len(request.sources or ['all'])} sources"
        )

    except Exception as e:
        logger.error("Lead discovery failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/discover/erp-only")
async def discover_erp_leads(
    tenant_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    include_reactivation: bool = True,
    include_upsell: bool = True,
    include_referral: bool = True
):
    """
    Discover leads from ERP data only.

    This is a focused endpoint for finding leads within existing customer data:
    - Re-engagement: Past customers who haven't had jobs in 12+ months
    - Upsell: High-value customers for premium services
    - Referral: Active satisfied customers who could refer others

    Requires tenant_id for proper data isolation.
    """
    try:
        engine = get_discovery_engine(tenant_id)

        sources = []
        if include_reactivation:
            sources.append(LeadSource.ERP_REACTIVATION.value)
        if include_upsell:
            sources.append(LeadSource.ERP_UPSELL.value)
        if include_referral:
            sources.append(LeadSource.ERP_REFERRAL.value)

        if not sources:
            return {
                "success": False,
                "error": "At least one source type must be enabled"
            }

        qualified_leads = await engine.discover_leads(
            sources=sources,
            limit=limit
        )

        return {
            "success": True,
            "tenant_id": tenant_id,
            "leads_found": len(qualified_leads),
            "by_source": {
                "reactivation": len([l for l in qualified_leads if l.source == LeadSource.ERP_REACTIVATION]),
                "upsell": len([l for l in qualified_leads if l.source == LeadSource.ERP_UPSELL]),
                "referral": len([l for l in qualified_leads if l.source == LeadSource.ERP_REFERRAL])
            },
            "leads": [lead.to_dict() for lead in qualified_leads]
        }

    except Exception as e:
        logger.error("ERP lead discovery failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/discover/web-search")
async def discover_web_leads(
    limit: int = Query(default=20, ge=1, le=50)
):
    """
    Discover leads through AI-powered web search.

    Uses Perplexity AI to search for businesses showing buying signals.
    Returns leads that need further qualification and contact enrichment.
    """
    try:
        engine = get_discovery_engine()

        qualified_leads = await engine.discover_leads(
            sources=[LeadSource.WEB_SEARCH.value, LeadSource.SOCIAL_SIGNAL.value],
            limit=limit
        )

        return {
            "success": True,
            "leads_found": len(qualified_leads),
            "by_source": {
                "web_search": len([l for l in qualified_leads if l.source == LeadSource.WEB_SEARCH]),
                "social_signal": len([l for l in qualified_leads if l.source == LeadSource.SOCIAL_SIGNAL])
            },
            "leads": [lead.to_dict() for lead in qualified_leads],
            "note": "These leads may require contact enrichment before outreach"
        }

    except Exception as e:
        logger.error("Web lead discovery failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# ====================
# Qualification Endpoints
# ====================

@router.post("/qualify")
async def qualify_lead(request: QualifyLeadRequest):
    """
    Qualify a manually entered lead.

    Scores and qualifies the lead against configurable criteria.
    Returns qualification status, score, and tier.
    """
    try:
        engine = get_discovery_engine()

        # Create lead object
        source_enum = LeadSource.MANUAL_ENTRY
        try:
            source_enum = LeadSource(request.source)
        except ValueError:
            pass

        lead = DiscoveredLead(
            company_name=request.company_name,
            contact_name=request.contact_name,
            email=request.email,
            phone=request.phone,
            website=request.website,
            location=request.location,
            industry=request.industry,
            source=source_enum,
            source_detail=request.source_detail or "",
            estimated_value=request.estimated_value,
            signals=request.signals,
            metadata=request.metadata
        )

        # Qualify the lead
        qualified_lead = await engine.qualify_lead(lead)

        return {
            "success": True,
            "lead": qualified_lead.to_dict(),
            "qualification": {
                "status": qualified_lead.qualification_status.value,
                "score": qualified_lead.score,
                "tier": qualified_lead.tier.value,
                "reasons": qualified_lead.metadata.get("disqualification_reasons", [])
            }
        }

    except Exception as e:
        logger.error("Lead qualification failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.put("/qualify/criteria")
async def update_qualification_criteria(request: QualificationCriteriaRequest):
    """
    Update the qualification criteria for the discovery engine.

    These criteria determine which leads pass qualification.
    """
    try:
        engine = get_discovery_engine()

        # Update criteria
        if request.min_score is not None:
            engine.criteria.min_score = request.min_score
        if request.require_email is not None:
            engine.criteria.require_email = request.require_email
        if request.require_phone is not None:
            engine.criteria.require_phone = request.require_phone
        if request.require_company is not None:
            engine.criteria.require_company = request.require_company
        if request.require_location is not None:
            engine.criteria.require_location = request.require_location
        if request.excluded_domains is not None:
            engine.criteria.excluded_domains = request.excluded_domains
        if request.min_estimated_value is not None:
            engine.criteria.min_estimated_value = request.min_estimated_value
        if request.industries is not None:
            engine.criteria.industries = request.industries

        return {
            "success": True,
            "criteria": {
                "min_score": engine.criteria.min_score,
                "require_email": engine.criteria.require_email,
                "require_phone": engine.criteria.require_phone,
                "require_company": engine.criteria.require_company,
                "require_location": engine.criteria.require_location,
                "excluded_domains": engine.criteria.excluded_domains,
                "min_estimated_value": engine.criteria.min_estimated_value,
                "industries": engine.criteria.industries
            },
            "message": "Qualification criteria updated"
        }

    except Exception as e:
        logger.error("Criteria update failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/qualify/criteria")
async def get_qualification_criteria():
    """Get current qualification criteria"""
    try:
        engine = get_discovery_engine()

        return {
            "success": True,
            "criteria": {
                "min_score": engine.criteria.min_score,
                "require_email": engine.criteria.require_email,
                "require_phone": engine.criteria.require_phone,
                "require_company": engine.criteria.require_company,
                "require_location": engine.criteria.require_location,
                "excluded_domains": engine.criteria.excluded_domains,
                "min_estimated_value": engine.criteria.min_estimated_value,
                "industries": engine.criteria.industries
            }
        }

    except Exception as e:
        logger.error("Failed to get criteria: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# ====================
# Sync Endpoints
# ====================

@router.post("/sync/erp")
async def sync_lead_to_erp(
    lead_id: str,
    tenant_id: str,
    company_name: str,
    contact_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    location: Optional[str] = None,
    score: float = 50.0,
    source: str = "lead_discovery"
):
    """
    Sync a qualified lead to the ERP leads table.

    Creates or updates a lead in the ERP system for human follow-up.
    """
    try:
        engine = get_discovery_engine(tenant_id)

        lead = DiscoveredLead(
            id=lead_id,
            company_name=company_name,
            contact_name=contact_name,
            email=email,
            phone=phone,
            location=location,
            source=LeadSource.MANUAL_ENTRY,
            source_detail=source,
            score=score,
            qualification_status=LeadQualificationStatus.QUALIFIED
        )

        result = await engine.sync_to_erp(lead)

        return {
            "success": result.get("success", False),
            "action": result.get("action"),
            "erp_lead_id": result.get("erp_lead_id"),
            "tenant_id": result.get("tenant_id"),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error("ERP sync failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/sync/revenue")
async def sync_lead_to_revenue(
    company_name: str,
    email: str,
    contact_name: Optional[str] = None,
    phone: Optional[str] = None,
    website: Optional[str] = None,
    location: Optional[str] = None,
    score: float = 50.0,
    estimated_value: float = 5000.0,
    source: str = "lead_discovery",
    signals: list[str] = Query(default=[])
):
    """
    Sync a lead to the revenue_leads table for nurturing.

    Creates or updates a lead in the revenue pipeline.
    """
    try:
        engine = get_discovery_engine()

        lead = DiscoveredLead(
            company_name=company_name,
            contact_name=contact_name,
            email=email,
            phone=phone,
            website=website,
            location=location,
            source=LeadSource.MANUAL_ENTRY,
            source_detail=source,
            score=score,
            estimated_value=estimated_value,
            signals=signals,
            qualification_status=LeadQualificationStatus.QUALIFIED
        )

        result = await engine.sync_to_revenue_leads(lead)

        return {
            "success": result.get("success", False),
            "action": result.get("action"),
            "revenue_lead_id": result.get("revenue_lead_id"),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error("Revenue sync failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/sync/batch")
async def sync_discovered_leads(tenant_id: Optional[str] = None, limit: int = 50):
    """
    Run discovery and automatically sync qualified leads to both systems.

    This is a convenience endpoint that:
    1. Discovers leads from all sources
    2. Qualifies them
    3. Syncs to revenue_leads
    4. Optionally syncs to ERP (if tenant_id provided)

    Returns summary of the operation.
    """
    try:
        engine = get_discovery_engine(tenant_id)

        # Run discovery
        qualified_leads = await engine.discover_leads(limit=limit)

        synced_revenue = 0
        synced_erp = 0
        errors = []

        for lead in qualified_leads:
            # Sync to revenue_leads
            revenue_result = await engine.sync_to_revenue_leads(lead)
            if revenue_result.get("success"):
                synced_revenue += 1
            else:
                errors.append({
                    "lead_id": lead.id,
                    "target": "revenue_leads",
                    "error": revenue_result.get("error")
                })

            # Sync to ERP if tenant_id provided
            if tenant_id:
                erp_result = await engine.sync_to_erp(lead)
                if erp_result.get("success"):
                    synced_erp += 1
                else:
                    errors.append({
                        "lead_id": lead.id,
                        "target": "erp",
                        "error": erp_result.get("error")
                    })

        return {
            "success": True,
            "leads_discovered": len(qualified_leads),
            "synced_to_revenue": synced_revenue,
            "synced_to_erp": synced_erp if tenant_id else "skipped (no tenant_id)",
            "errors": errors[:10] if errors else [],
            "total_errors": len(errors)
        }

    except Exception as e:
        logger.error("Batch sync failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


# ====================
# Stats Endpoints
# ====================

@router.get("/stats")
async def get_discovery_stats():
    """
    Get lead discovery statistics.

    Returns:
    - Source performance metrics
    - Recent discovery runs
    - Overall totals
    """
    try:
        engine = get_discovery_engine()
        stats = await engine.get_discovery_stats()

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error("Failed to get stats: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/sources")
async def get_lead_sources():
    """Get list of available lead sources and their status"""
    return {
        "success": True,
        "sources": [
            {
                "value": source.value,
                "name": source.name,
                "description": {
                    LeadSource.ERP_REACTIVATION: "Past customers from ERP - re-engagement opportunities",
                    LeadSource.ERP_UPSELL: "High-value customers - premium service opportunities",
                    LeadSource.ERP_REFERRAL: "Active customers - referral opportunities",
                    LeadSource.WEB_FORM: "Website form submissions",
                    LeadSource.WEB_SEARCH: "AI-powered web research",
                    LeadSource.SOCIAL_SIGNAL: "Social media buying signals",
                    LeadSource.STORM_TRACKER: "Weather event leads",
                    LeadSource.MANUAL_ENTRY: "Manually added leads",
                    LeadSource.PARTNER_REFERRAL: "Partner/affiliate referrals",
                    LeadSource.INBOUND_CALL: "Phone inquiries"
                }.get(source, "")
            }
            for source in LeadSource
        ]
    }


@router.get("/tiers")
async def get_lead_tiers():
    """Get list of lead tiers and their meanings"""
    return {
        "success": True,
        "tiers": [
            {
                "value": "hot",
                "name": "Hot",
                "score_range": "80-100",
                "priority": "High",
                "description": "High priority - ready for immediate contact"
            },
            {
                "value": "warm",
                "name": "Warm",
                "score_range": "60-79",
                "priority": "Medium",
                "description": "Medium priority - active interest shown"
            },
            {
                "value": "cool",
                "name": "Cool",
                "score_range": "40-59",
                "priority": "Low",
                "description": "Low priority - potential future opportunity"
            },
            {
                "value": "cold",
                "name": "Cold",
                "score_range": "0-39",
                "priority": "Nurture",
                "description": "Nurture only - long-term engagement"
            }
        ]
    }
