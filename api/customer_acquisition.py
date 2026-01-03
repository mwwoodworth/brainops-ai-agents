"""
CUSTOMER ACQUISITION API
Expose autonomous acquisition agents via REST endpoints.
Automates lead discovery, outreach, and conversion.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import config

logger = logging.getLogger(__name__)

# API Key Security
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
VALID_API_KEYS = config.security.valid_api_keys


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify API key for authentication"""
    from fastapi import HTTPException
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


router = APIRouter(
    prefix="/acquire",
    tags=["customer-acquisition"],
    dependencies=[Depends(verify_api_key)]
)


class SearchRequest(BaseModel):
    """Request to discover new targets"""
    query: Optional[str] = "roofing contractors looking for software solutions"
    max_results: int = 10


class OutreachRequest(BaseModel):
    """Request to generate outreach for a target"""
    target_id: str
    sequence_type: str = "email"  # email, linkedin, combined


class CampaignRequest(BaseModel):
    """Request to run full acquisition campaign"""
    name: str
    target_industry: str = "roofing"
    target_geography: Optional[str] = None
    budget: float = 1000.0


@router.get("/status")
async def get_acquisition_status():
    """
    Get current status of acquisition system.
    """
    try:
        from customer_acquisition_agents import (
            WebSearchAgent,
            SocialMediaAgent,
            OutreachAgent,
            ConversionAgent
        )

        return {
            "status": "operational",
            "agents": {
                "web_search": "available",
                "social_media": "available",
                "outreach": "available",
                "conversion": "available"
            },
            "message": "Customer acquisition agents ready"
        }
    except ImportError as e:
        return {
            "status": "degraded",
            "error": str(e),
            "message": "Some agents not available"
        }


@router.post("/search")
async def discover_targets(request: SearchRequest):
    """
    Use AI to discover new acquisition targets.
    Searches web and social for companies matching criteria.
    """
    try:
        from customer_acquisition_agents import WebSearchAgent

        agent = WebSearchAgent()
        targets = await agent.discover_targets(
            query=request.query,
            max_results=request.max_results
        )

        return {
            "status": "success",
            "count": len(targets),
            "targets": targets
        }
    except Exception as e:
        logger.error(f"Target discovery failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/social-monitor")
async def monitor_social_signals(
    keywords: str = Query(..., description="Keywords to monitor"),
    platforms: str = Query("twitter,linkedin,reddit", description="Platforms to search")
):
    """
    Monitor social media for buying signals.
    Finds companies actively discussing needs we can solve.
    """
    try:
        from customer_acquisition_agents import SocialMediaAgent

        agent = SocialMediaAgent()
        signals = await agent.monitor_signals(
            keywords=keywords.split(","),
            platforms=platforms.split(",")
        )

        return {
            "status": "success",
            "count": len(signals),
            "signals": signals
        }
    except Exception as e:
        logger.error(f"Social monitoring failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/outreach")
async def generate_outreach(request: OutreachRequest):
    """
    Generate personalized outreach sequence for a target.
    Creates multi-touch campaign with AI-written messages.
    """
    try:
        from customer_acquisition_agents import OutreachAgent

        agent = OutreachAgent()
        sequence = await agent.create_outreach_sequence(
            target_id=request.target_id,
            sequence_type=request.sequence_type
        )

        return {
            "status": "success",
            "target_id": request.target_id,
            "sequence": sequence
        }
    except Exception as e:
        logger.error(f"Outreach generation failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/convert")
async def optimize_conversion(target_id: str = Query(...)):
    """
    Get AI-optimized conversion strategy for a target.
    Analyzes engagement and recommends next best action.
    """
    try:
        from customer_acquisition_agents import ConversionAgent

        agent = ConversionAgent()
        strategy = await agent.get_conversion_strategy(target_id=target_id)

        return {
            "status": "success",
            "target_id": target_id,
            "strategy": strategy
        }
    except Exception as e:
        logger.error(f"Conversion optimization failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.post("/campaign/run")
async def run_full_campaign(request: CampaignRequest):
    """
    Run a full acquisition campaign.
    Orchestrates all agents: discovery -> outreach -> conversion.
    """
    try:
        from customer_acquisition_agents import AcquisitionOrchestrator

        orchestrator = AcquisitionOrchestrator(
            campaign_name=request.name,
            target_industry=request.target_industry
        )

        results = await orchestrator.run_campaign()

        return {
            "status": "success",
            "campaign": request.name,
            "results": results
        }
    except Exception as e:
        logger.error(f"Campaign execution failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/targets")
async def list_targets(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    List acquisition targets in the database.
    """
    try:
        from database.async_connection import get_pool, using_fallback

        if using_fallback():
            return {"status": "error", "error": "Database unavailable"}

        pool = get_pool()

        query = """
            SELECT id, company_name, industry, location, intent_score, status, created_at
            FROM acquisition_targets
        """
        params = []

        if status:
            query += " WHERE status = $1"
            params.append(status)

        query += " ORDER BY intent_score DESC, created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)

        rows = await pool.fetch(query, *params)

        return {
            "status": "success",
            "count": len(rows),
            "targets": [dict(row) for row in rows]
        }
    except Exception as e:
        logger.error(f"Failed to list targets: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@router.get("/metrics")
async def get_acquisition_metrics():
    """
    Get acquisition pipeline metrics.
    """
    try:
        from database.async_connection import get_pool, using_fallback

        if using_fallback():
            return {"status": "error", "error": "Database unavailable"}

        pool = get_pool()

        # Get metrics
        total = await pool.fetchval("SELECT COUNT(*) FROM acquisition_targets") or 0
        contacted = await pool.fetchval(
            "SELECT COUNT(*) FROM acquisition_targets WHERE status = 'contacted'"
        ) or 0
        qualified = await pool.fetchval(
            "SELECT COUNT(*) FROM acquisition_targets WHERE status = 'qualified'"
        ) or 0
        converted = await pool.fetchval(
            "SELECT COUNT(*) FROM acquisition_targets WHERE status = 'converted'"
        ) or 0

        avg_intent = await pool.fetchval(
            "SELECT AVG(intent_score) FROM acquisition_targets"
        ) or 0

        return {
            "status": "success",
            "metrics": {
                "total_targets": total,
                "contacted": contacted,
                "qualified": qualified,
                "converted": converted,
                "conversion_rate": (converted / total * 100) if total > 0 else 0,
                "avg_intent_score": round(float(avg_intent), 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
