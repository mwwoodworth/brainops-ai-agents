"""
Affiliate & Partnership API Router
Secure, authenticated endpoints for the Affiliate Partnership Pipeline
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import re

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = {"brainops_prod_key_2025", "brainops_dev_key_2025"}
ADMIN_API_KEYS = {"brainops_prod_key_2025"}  # Only admin keys can process payouts

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

async def verify_admin_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify admin API key for sensitive operations"""
    if not api_key or api_key not in ADMIN_API_KEYS:
        raise HTTPException(status_code=403, detail="Admin access required")
    return api_key

router = APIRouter(prefix="/affiliate", tags=["Affiliate & Partnerships"])

# Import the affiliate pipeline with fallback
try:
    from affiliate_partnership_pipeline import (
        PartnerType,
        get_affiliate_pipeline
    )
    AFFILIATE_PIPELINE_AVAILABLE = True
    logger.info("Affiliate Partnership Pipeline loaded")
except ImportError as e:
    AFFILIATE_PIPELINE_AVAILABLE = False
    logger.warning(f"Affiliate Partnership Pipeline not available: {e}")


# Pydantic models
class AffiliateRegistrationRequest(BaseModel):
    """Request to register as an affiliate"""
    name: str = Field(..., min_length=2, max_length=200)
    email: str = Field(..., max_length=255)
    company: Optional[str] = Field(None, max_length=200)
    website: Optional[str] = Field(None, max_length=500)
    partner_type: str = Field(default="affiliate")
    marketing_channels: List[str] = Field(default_factory=list, max_items=10)
    expected_monthly_referrals: int = Field(default=10, ge=1, le=100000)
    notes: Optional[str] = Field(None, max_length=2000)
    tenant_id: str = Field(default="default")

    @validator('email')
    def validate_email(cls, v):
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v):
            raise ValueError('Invalid email format')
        return v


class AffiliateResponse(BaseModel):
    """Response with affiliate details"""
    affiliate_id: str
    name: str
    email: str
    partner_type: str
    tier: str
    referral_code: str
    commission_rate: float
    status: str
    created_at: str


class ClickTrackRequest(BaseModel):
    """Request to track a referral click"""
    referral_code: str = Field(..., min_length=6, max_length=50)
    source_url: str = Field(..., max_length=2000)
    ip_address: Optional[str] = Field(None, max_length=45)
    user_agent: Optional[str] = Field(None, max_length=500)
    landing_page: Optional[str] = Field(None, max_length=500)


class ConversionRequest(BaseModel):
    """Request to track a conversion"""
    referral_code: str = Field(..., min_length=6, max_length=50)
    order_id: str = Field(..., max_length=100)
    amount: float = Field(..., gt=0, le=1000000)
    product_id: Optional[str] = Field(None, max_length=100)
    customer_email: Optional[str] = Field(None, max_length=255)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
async def affiliate_health():
    """Check affiliate system health"""
    return {
        "status": "available" if AFFILIATE_PIPELINE_AVAILABLE else "unavailable",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": {
            "affiliate_registration": AFFILIATE_PIPELINE_AVAILABLE,
            "click_tracking": AFFILIATE_PIPELINE_AVAILABLE,
            "conversion_tracking": AFFILIATE_PIPELINE_AVAILABLE,
            "commission_calculation": AFFILIATE_PIPELINE_AVAILABLE,
            "payout_processing": AFFILIATE_PIPELINE_AVAILABLE,
            "fraud_detection": AFFILIATE_PIPELINE_AVAILABLE,
            "content_generation": AFFILIATE_PIPELINE_AVAILABLE
        }
    }


@router.post("/register", response_model=AffiliateResponse)
async def register_affiliate(
    request: AffiliateRegistrationRequest,
    api_key: str = Depends(verify_api_key)
):
    """Register a new affiliate partner"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()

        # Map partner type
        try:
            partner_type = PartnerType(request.partner_type)
        except ValueError:
            partner_type = PartnerType.AFFILIATE

        affiliate = await pipeline.register_affiliate(
            name=request.name,
            email=request.email,
            company=request.company,
            website=request.website,
            partner_type=partner_type,
            marketing_channels=request.marketing_channels,
            expected_monthly_referrals=request.expected_monthly_referrals,
            notes=request.notes,
            tenant_id=request.tenant_id
        )

        return AffiliateResponse(
            affiliate_id=affiliate["id"],
            name=affiliate["name"],
            email=affiliate["email"],
            partner_type=affiliate["partner_type"],
            tier=affiliate["tier"],
            referral_code=affiliate["referral_code"],
            commission_rate=affiliate["commission_rate"],
            status=affiliate["status"],
            created_at=affiliate["created_at"]
        )

    except Exception as e:
        logger.error(f"Affiliate registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/click")
async def track_click(
    request: ClickTrackRequest,
    api_key: str = Depends(verify_api_key)
):
    """Track a referral click"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()

        result = await pipeline.track_click(
            referral_code=request.referral_code,
            source_url=request.source_url,
            ip_address=request.ip_address,
            user_agent=request.user_agent,
            landing_page=request.landing_page
        )

        return {
            "status": "tracked",
            "click_id": result.get("click_id"),
            "fraud_signals": result.get("fraud_signals", []),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Click tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/track/conversion")
async def track_conversion(
    request: ConversionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Track a conversion and calculate commission"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()

        result = await pipeline.track_conversion(
            referral_code=request.referral_code,
            order_id=request.order_id,
            amount=request.amount,
            product_id=request.product_id,
            customer_email=request.customer_email,
            metadata=request.metadata
        )

        return {
            "status": "conversion_recorded",
            "conversion_id": result.get("conversion_id"),
            "commission": {
                "amount": result.get("commission_amount"),
                "rate": result.get("commission_rate"),
                "tier": result.get("affiliate_tier")
            },
            "fraud_check": result.get("fraud_status", "passed"),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Conversion tracking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/{affiliate_id}")
async def get_affiliate_dashboard(
    affiliate_id: str,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get affiliate dashboard data"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        dashboard = await pipeline.get_dashboard(affiliate_id, tenant_id)

        if not dashboard:
            raise HTTPException(status_code=404, detail="Affiliate not found")

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/{affiliate_id}")
async def get_affiliate_stats(
    affiliate_id: str,
    period: str = "30d",
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get affiliate performance statistics"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        stats = await pipeline.get_stats(affiliate_id, period, tenant_id)

        return stats

    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/{affiliate_id}/generate")
async def generate_affiliate_content(
    affiliate_id: str,
    content_type: str = "social_media",
    product_id: Optional[str] = None,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Generate marketing content for an affiliate"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        content = await pipeline.generate_content(
            affiliate_id=affiliate_id,
            content_type=content_type,
            product_id=product_id,
            tenant_id=tenant_id
        )

        return {
            "affiliate_id": affiliate_id,
            "content_type": content_type,
            "content": content,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Content generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commissions/pending")
async def get_pending_commissions(
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get pending commissions awaiting payout"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        commissions = await pipeline.get_pending_commissions(tenant_id)

        return {
            "pending_commissions": commissions,
            "total_pending": sum(c.get("amount", 0) for c in commissions),
            "count": len(commissions)
        }

    except Exception as e:
        logger.error(f"Pending commissions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/payouts/process")
async def process_payouts(
    affiliate_ids: Optional[List[str]] = None,
    tenant_id: str = "default",
    api_key: str = Depends(verify_admin_key)  # Admin only!
):
    """
    Process pending payouts for affiliates.

    **Admin access required** - This endpoint processes financial transactions.
    """
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        payouts = await pipeline.process_payouts(affiliate_ids, tenant_id)

        return {
            "status": "processed",
            "payouts": payouts,
            "total_paid": sum(p.get("amount", 0) for p in payouts),
            "processed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Payout processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/leaderboard")
async def get_affiliate_leaderboard(
    period: str = "30d",
    limit: int = 10,
    tenant_id: str = "default",
    api_key: str = Depends(verify_api_key)
):
    """Get top performing affiliates"""
    if not AFFILIATE_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Affiliate system not available")

    try:
        pipeline = get_affiliate_pipeline()
        leaderboard = await pipeline.get_leaderboard(period, limit, tenant_id)

        return {
            "period": period,
            "leaderboard": leaderboard,
            "generated_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
