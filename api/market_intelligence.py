"""
Predictive Market Intelligence API Router
==========================================
API endpoints for market intelligence, trend analysis, and autonomous content optimization.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market-intelligence", tags=["Market Intelligence"])

# Lazy initialization
_engine = None


def _get_engine():
    """Lazy load the Market Intelligence Engine"""
    global _engine
    if _engine is None:
        try:
            from predictive_market_intelligence import PredictiveMarketIntelligence
            _engine = PredictiveMarketIntelligence()
        except Exception as e:
            logger.error(f"Failed to initialize Market Intelligence: {e}")
            raise HTTPException(status_code=503, detail="Market Intelligence not available")
    return _engine


class MarketSignalRequest(BaseModel):
    signal_type: str  # TREND, COMPETITOR, PRICING, DEMAND, SENTIMENT, NEWS, REGULATORY, TECHNOLOGY
    source: str
    data: Dict[str, Any]
    confidence: float = 0.8


class ContentOptimizationRequest(BaseModel):
    campaign_id: str
    target_audience: str
    current_metrics: Dict[str, float]


class CompetitorAnalysisRequest(BaseModel):
    competitor_name: str
    analysis_type: str = "comprehensive"  # comprehensive, pricing, features, marketing


@router.get("/status")
async def get_market_status():
    """Get Market Intelligence system status"""
    engine = _get_engine()
    return {
        "system": "market_intelligence",
        "status": "operational",
        "initialized": engine._initialized if hasattr(engine, '_initialized') else True,
        "capabilities": [
            "trend_analysis",
            "competitor_intelligence",
            "demand_forecasting",
            "content_optimization",
            "pricing_intelligence",
            "sentiment_analysis"
        ]
    }


@router.post("/signals")
async def ingest_signal(request: MarketSignalRequest):
    """Ingest a new market signal for analysis"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    result = await engine.ingest_signal(
        signal_type=request.signal_type,
        source=request.source,
        data=request.data,
        confidence=request.confidence
    )
    return result


@router.get("/signals")
async def list_signals(
    signal_type: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    days: int = Query(7, ge=1, le=90)
):
    """List recent market signals"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    signals = await engine.get_recent_signals(
        signal_type=signal_type,
        limit=limit,
        days=days
    )
    return {"signals": signals, "total": len(signals)}


@router.get("/insights")
async def get_market_insights(
    category: Optional[str] = None,
    priority: Optional[str] = None
):
    """Get actionable market insights"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    insights = await engine.generate_insights(
        category=category,
        priority=priority
    )
    return {"insights": insights}


@router.get("/predictions")
async def get_market_predictions(
    prediction_type: Optional[str] = None,
    timeframe: str = Query("7d", description="Prediction timeframe: 1d, 7d, 30d, 90d")
):
    """Get market predictions"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    predictions = await engine.get_predictions(
        prediction_type=prediction_type,
        timeframe=timeframe
    )
    return {"predictions": predictions, "timeframe": timeframe}


@router.post("/content/optimize")
async def optimize_content(request: ContentOptimizationRequest):
    """Generate content optimization recommendations"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    recommendations = await engine.optimize_content(
        campaign_id=request.campaign_id,
        target_audience=request.target_audience,
        current_metrics=request.current_metrics
    )
    return recommendations


@router.get("/content/optimizations")
async def list_content_optimizations(
    status: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """List content optimization history"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    optimizations = await engine.get_optimization_history(
        status=status,
        limit=limit
    )
    return {"optimizations": optimizations}


@router.post("/competitors/analyze")
async def analyze_competitor(request: CompetitorAnalysisRequest):
    """Analyze a competitor"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    analysis = await engine.analyze_competitor(
        competitor_name=request.competitor_name,
        analysis_type=request.analysis_type
    )
    return analysis


@router.get("/competitors")
async def list_competitors():
    """List tracked competitors with their intelligence"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    competitors = await engine.get_competitor_intelligence()
    return {"competitors": competitors}


@router.get("/trends")
async def get_market_trends(
    category: Optional[str] = None,
    emerging_only: bool = False
):
    """Get current market trends"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    trends = await engine.get_trends(
        category=category,
        emerging_only=emerging_only
    )
    return {"trends": trends}


@router.get("/dashboard")
async def get_market_dashboard():
    """Get a comprehensive market intelligence dashboard"""
    engine = _get_engine()
    if hasattr(engine, 'initialize') and not getattr(engine, '_initialized', True):
        await engine.initialize()

    # Aggregate all market data
    signals_summary = await engine.get_signals_summary() if hasattr(engine, 'get_signals_summary') else {}
    top_insights = await engine.generate_insights(limit=5) if hasattr(engine, 'generate_insights') else []
    active_predictions = await engine.get_predictions(timeframe="7d") if hasattr(engine, 'get_predictions') else []

    return {
        "overview": {
            "total_signals_24h": signals_summary.get("count_24h", 0),
            "active_trends": signals_summary.get("active_trends", 0),
            "competitors_tracked": signals_summary.get("competitors", 0),
            "optimizations_active": signals_summary.get("active_optimizations", 0)
        },
        "top_insights": top_insights[:5] if isinstance(top_insights, list) else [],
        "active_predictions": active_predictions[:5] if isinstance(active_predictions, list) else [],
        "market_sentiment": signals_summary.get("overall_sentiment", "neutral"),
        "opportunity_score": signals_summary.get("opportunity_score", 0)
    }
