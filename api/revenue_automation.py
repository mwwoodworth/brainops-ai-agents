"""
Revenue Automation API Router
==============================
REAL revenue generation endpoints. No placeholders.
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
    email: str
    name: str
    industry: str
    source: str
    phone: Optional[str] = None
    company: Optional[str] = None
    custom_fields: Optional[dict[str, Any]] = None


class QualifyLeadRequest(BaseModel):
    qualification_data: dict[str, Any]


class PaymentLinkRequest(BaseModel):
    amount: float
    product_service: str
    description: Optional[str] = None


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
async def capture_lead(request: LeadCaptureRequest):
    """Capture a new lead and start automation"""
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
async def qualify_lead(lead_id: str, request: QualifyLeadRequest):
    """Qualify a lead with additional data"""
    try:
        engine = await _get_engine()
        result = await engine.qualify_lead(lead_id, request.qualification_data)
        return result
    except Exception as e:
        logger.error(f"Qualify lead error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/leads/{lead_id}/payment-link")
async def create_payment_link(lead_id: str, request: PaymentLinkRequest):
    """Create a payment link for a lead"""
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
