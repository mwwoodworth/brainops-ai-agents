"""
Revenue Automation API Router
==============================
REAL revenue generation endpoints. No placeholders.
Enhanced with proper validation and error handling.
"""

import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# API Key Security
try:
    from config import config
    VALID_API_KEYS = config.security.valid_api_keys
except (ImportError, AttributeError):
    import os
    fallback_key = os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY") or os.getenv("API_KEY")
    VALID_API_KEYS = {fallback_key} if fallback_key else set()

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

router = APIRouter(prefix="/revenue", tags=["Revenue Automation"])

_engine = None
_initialized = False


async def _get_engine():
    global _engine, _initialized
    if _engine is None:
        from revenue_automation_engine import RevenueAutomationEngine
        _engine = RevenueAutomationEngine()

    if not _initialized:
        await _engine.initialize()
        _initialized = True

    return _engine


class LeadCaptureRequest(BaseModel):
    """Request model for capturing leads with validation"""
    email: str = Field(..., min_length=5, max_length=255, description="Valid email address")
    name: str = Field(..., min_length=1, max_length=200, description="Lead name")
    industry: str = Field(..., min_length=2, max_length=50, description="Industry type")
    source: str = Field(..., min_length=2, max_length=50, description="Lead source")
    phone: Optional[str] = Field(None, max_length=30, description="Phone number")
    company: Optional[str] = Field(None, max_length=200, description="Company name")
    custom_fields: Optional[dict[str, Any]] = Field(default_factory=dict)

    @validator('email')
    def validate_email(cls, v):
        if not EMAIL_REGEX.match(v):
            raise ValueError('Invalid email format')
        return v.lower().strip()

    @validator('name')
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()

    @validator('industry')
    def validate_industry(cls, v):
        valid_industries = {
            'roofing', 'solar', 'hvac', 'plumbing', 'electrical',
            'landscaping', 'construction', 'home_services', 'saas',
            'ecommerce', 'consulting', 'real_estate', 'insurance',
            'automotive', 'healthcare', 'generic'
        }
        v_lower = v.lower().strip()
        if v_lower not in valid_industries:
            # Allow but log unknown industries
            logger.warning(f"Unknown industry: {v_lower}, using 'generic'")
            return 'generic'
        return v_lower

    @validator('source')
    def validate_source(cls, v):
        valid_sources = {
            'website', 'referral', 'google_ads', 'facebook', 'instagram',
            'linkedin', 'cold_outreach', 'partnership', 'organic_search',
            'direct', 'api', 'manual'
        }
        v_lower = v.lower().strip()
        if v_lower not in valid_sources:
            return 'direct'  # Default to direct
        return v_lower


class QualifyLeadRequest(BaseModel):
    """Request model for lead qualification"""
    qualification_data: dict[str, Any] = Field(..., description="Qualification criteria")

    @validator('qualification_data')
    def validate_qualification_data(cls, v):
        if not v:
            raise ValueError('Qualification data cannot be empty')
        return v


class PaymentLinkRequest(BaseModel):
    """Request model for creating payment links with validation"""
    amount: float = Field(..., gt=0, le=1000000, description="Payment amount in USD")
    product_service: str = Field(..., min_length=2, max_length=200, description="Product or service name")
    description: Optional[str] = Field(None, max_length=1000, description="Payment description")

    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be greater than 0')
        if v > 1000000:
            raise ValueError('Amount exceeds maximum limit')
        # Round to 2 decimal places
        return round(v, 2)


@router.get("/status")
async def get_revenue_status():
    """Get revenue automation system status"""
    try:
        engine = await _get_engine()
        metrics = engine.get_revenue_metrics()
        return {
            "system": "revenue_automation",
            "status": "operational",
            "initialized": _initialized,
            "total_revenue": metrics["total_revenue"],
            "pipeline_value": metrics["pipeline_value"],
            "total_leads": metrics["total_leads"],
            "conversion_rate": metrics["conversion_rate"],
            "capabilities": [
                "lead_capture",
                "auto_qualification",
                "outreach_automation",
                "payment_processing",
                "revenue_tracking",
                "multi_industry"
            ],
            "supported_industries": [
                "roofing", "solar", "hvac", "saas", "ecommerce",
                "consulting", "construction", "home_services"
            ]
        }
    except Exception as e:
        logger.error(f"Revenue status error: {e}")
        return {"system": "revenue_automation", "status": "error", "error": str(e)}


@router.post("/leads/capture")
async def capture_lead(request: LeadCaptureRequest, api_key: str = Depends(verify_api_key)):
    """Capture a new lead and start automation. Requires API key authentication."""
    try:
        engine = await _get_engine()
        result = await engine.capture_lead(
            email=request.email,
            name=request.name,
            industry=request.industry,
            source=request.source,
            phone=request.phone,
            company=request.company,
            custom_fields=request.custom_fields
        )
        return result
    except Exception as e:
        logger.error(f"Lead capture error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/leads")
async def list_leads(
    status: Optional[str] = None,
    industry: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """List leads in the pipeline"""
    try:
        engine = await _get_engine()

        leads = []
        for lead in list(engine.leads.values())[:limit]:
            if status and lead.status.value != status:
                continue
            if industry and lead.industry.value != industry:
                continue

            leads.append({
                "lead_id": lead.lead_id,
                "email": lead.email,
                "name": lead.name,
                "company": lead.company,
                "industry": lead.industry.value,
                "source": lead.source.value,
                "status": lead.status.value,
                "score": lead.score,
                "estimated_value": float(lead.estimated_value),
                "created_at": lead.created_at
            })

        return {"leads": leads, "total": len(leads)}
    except Exception as e:
        logger.error(f"List leads error: {e}")
        return {"leads": [], "error": str(e)}


@router.get("/leads/{lead_id}")
async def get_lead(lead_id: str):
    """Get lead details"""
    try:
        engine = await _get_engine()

        if lead_id not in engine.leads:
            raise HTTPException(status_code=404, detail="Lead not found")

        lead = engine.leads[lead_id]
        return {
            "lead_id": lead.lead_id,
            "email": lead.email,
            "phone": lead.phone,
            "name": lead.name,
            "company": lead.company,
            "industry": lead.industry.value,
            "source": lead.source.value,
            "status": lead.status.value,
            "score": lead.score,
            "estimated_value": float(lead.estimated_value),
            "created_at": lead.created_at,
            "updated_at": lead.updated_at,
            "contacted_at": lead.contacted_at,
            "converted_at": lead.converted_at,
            "custom_fields": lead.custom_fields,
            "automation_history": lead.automation_history[-10:],
            "notes": lead.notes
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get lead error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/leads/{lead_id}/qualify")
async def qualify_lead(lead_id: str, request: QualifyLeadRequest, api_key: str = Depends(verify_api_key)):
    """Qualify a lead with additional data. Requires API key authentication."""
    try:
        engine = await _get_engine()
        result = await engine.qualify_lead(lead_id, request.qualification_data)
        return result
    except Exception as e:
        logger.error(f"Qualify lead error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/leads/{lead_id}/payment-link")
async def create_payment_link(lead_id: str, request: PaymentLinkRequest, api_key: str = Depends(verify_api_key)):
    """Create a payment link for a lead. Requires API key authentication."""
    try:
        engine = await _get_engine()
        result = await engine.create_payment_link(
            lead_id=lead_id,
            amount=request.amount,
            product_service=request.product_service,
            description=request.description
        )
        return result
    except Exception as e:
        logger.error(f"Payment link error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/webhooks/stripe")
async def stripe_webhook(payload: dict[str, Any]):
    """Process Stripe payment webhook"""
    try:
        engine = await _get_engine()
        result = await engine.process_payment_webhook(payload)
        return result
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics")
async def get_metrics():
    """Get revenue metrics"""
    try:
        engine = await _get_engine()
        return engine.get_revenue_metrics()
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": str(e)}


@router.get("/pipeline")
async def get_pipeline():
    """Get pipeline dashboard"""
    try:
        engine = await _get_engine()
        return engine.get_pipeline_dashboard()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        return {"error": str(e)}


@router.get("/dashboard")
async def get_revenue_dashboard():
    """Complete revenue dashboard"""
    try:
        engine = await _get_engine()
        metrics = engine.get_revenue_metrics()
        pipeline = engine.get_pipeline_dashboard()

        return {
            "overview": {
                "total_revenue": metrics["total_revenue"],
                "monthly_revenue": metrics["monthly_revenue"],
                "pipeline_value": metrics["pipeline_value"],
                "conversion_rate": metrics["conversion_rate"]
            },
            "leads": {
                "total": metrics["total_leads"],
                "qualified": metrics["qualified_leads"],
                "won": metrics["won_leads"]
            },
            "pipeline": pipeline["leads_by_stage"],
            "value_by_stage": pipeline["value_by_stage"],
            "by_industry": metrics["revenue_by_industry"],
            "by_source": metrics["revenue_by_source"],
            "system_status": "operational"
        }
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return {"error": str(e), "system_status": "error"}
