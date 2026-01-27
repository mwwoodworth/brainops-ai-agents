"""
TRUE AI INTELLIGENCE API
========================

Endpoints for real AI-powered analysis.
NO MORE HARDCODED PATTERNS.

Created: 2026-01-27
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intelligence", tags=["ai-intelligence"])

# Import AI Intelligence
try:
    from ai_intelligence import (
        get_ai_intelligence,
        analyze_with_ai,
        AnalysisDepth
    )
    AI_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    AI_INTELLIGENCE_AVAILABLE = False
    logger.warning(f"AI Intelligence not available: {e}")


class AnalyzeRequest(BaseModel):
    issue: str
    context: Optional[dict] = None
    depth: str = "standard"  # quick, standard, deep


class FixPlanRequest(BaseModel):
    issues: list[str]


@router.get("/")
async def root():
    """
    🧠 TRUE AI INTELLIGENCE ROOT

    Real AI-powered analysis, not hardcoded patterns.
    """
    return {
        "service": "BrainOps True AI Intelligence",
        "description": "Real AI analysis using LLMs, not pattern matching",
        "available": AI_INTELLIGENCE_AVAILABLE,
        "models": ["anthropic", "openai", "gemini"],
        "analysis_depths": {
            "quick": "Fast heuristic + quick AI check",
            "standard": "Full single-model AI analysis",
            "deep": "Multi-model consensus with reasoning"
        },
        "endpoints": {
            "POST /intelligence/analyze": "Analyze an issue with real AI",
            "POST /intelligence/fix-plan": "Generate AI fix plan for issues",
            "GET /intelligence/health": "AI-powered system health analysis",
            "GET /intelligence/learning": "View AI learning statistics"
        }
    }


@router.post("/analyze")
async def analyze_issue(request: AnalyzeRequest):
    """
    🔬 ANALYZE ISSUE WITH REAL AI

    Uses actual LLM calls to understand the issue.
    Not pattern matching - real intelligence.

    Depths:
    - quick: Fast analysis
    - standard: Full AI analysis
    - deep: Multi-model consensus
    """
    if not AI_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Intelligence not available")

    # Validate depth
    valid_depths = ["quick", "standard", "deep"]
    if request.depth not in valid_depths:
        request.depth = "standard"

    result = await analyze_with_ai(
        issue=request.issue,
        context=request.context,
        depth=request.depth
    )

    return {
        "analysis_complete": True,
        "ai_powered": True,
        "result": result
    }


@router.post("/fix-plan")
async def generate_fix_plan(request: FixPlanRequest):
    """
    📋 GENERATE AI FIX PLAN

    AI creates an ordered, prioritized fix plan
    with rollback steps and risk assessment.
    """
    if not AI_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Intelligence not available")

    ai = get_ai_intelligence()
    plan = await ai.generate_fix_plan(request.issues)

    return {
        "plan_generated": True,
        "ai_powered": True,
        "issues_count": len(request.issues),
        "plan": plan
    }


@router.get("/health")
async def ai_health_analysis():
    """
    🏥 AI-POWERED SYSTEM HEALTH ANALYSIS

    Uses AI to analyze overall system health,
    identify risks, and make recommendations.
    """
    if not AI_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Intelligence not available")

    # Get current system state
    try:
        from neural_core import get_neural_core
        core = get_neural_core()
        systems = core.get_system_details()
    except Exception as e:
        return {"error": f"Could not get system state: {e}"}

    ai = get_ai_intelligence()
    analysis = await ai.analyze_system_health(systems)

    return {
        "analysis_complete": True,
        "ai_powered": True,
        "health_analysis": analysis,
        "systems_analyzed": len(systems)
    }


@router.get("/learning")
async def get_learning_stats():
    """
    📚 AI LEARNING STATISTICS

    View what the AI has learned from past fixes.
    """
    if not AI_INTELLIGENCE_AVAILABLE:
        return {"message": "AI Intelligence not available", "learning_active": False}

    ai = get_ai_intelligence()
    stats = ai.get_learning_stats()

    return {
        "learning_active": True,
        "stats": stats
    }


@router.post("/record-outcome")
async def record_fix_outcome(
    issue: str,
    fix_applied: str,
    success: bool,
    details: str = ""
):
    """
    📝 RECORD FIX OUTCOME FOR LEARNING

    Tell the AI whether a fix worked so it can learn.
    """
    if not AI_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Intelligence not available")

    ai = get_ai_intelligence()
    ai.record_fix_outcome(issue, fix_applied, success, details)

    return {
        "recorded": True,
        "message": f"Learning recorded: {'SUCCESS' if success else 'FAILED'}"
    }


@router.get("/capabilities")
async def get_ai_capabilities():
    """
    🚀 AI INTELLIGENCE CAPABILITIES

    What can the AI Intelligence system do?
    """
    return {
        "capabilities": {
            "issue_analysis": {
                "description": "Deep AI analysis of any system issue",
                "depths": ["quick", "standard", "deep"],
                "models_used": ["Claude", "GPT-4", "Gemini"],
                "features": [
                    "Root cause identification",
                    "Severity assessment",
                    "Fix strategy generation",
                    "Auto-fixability determination"
                ]
            },
            "fix_planning": {
                "description": "AI-generated fix plans",
                "features": [
                    "Ordered execution steps",
                    "Risk assessment",
                    "Rollback procedures",
                    "Dependency tracking"
                ]
            },
            "health_analysis": {
                "description": "AI-powered system health assessment",
                "features": [
                    "Overall health scoring",
                    "Risk prediction",
                    "Proactive recommendations",
                    "Critical concern identification"
                ]
            },
            "learning": {
                "description": "Learning from past fixes",
                "features": [
                    "Success rate tracking",
                    "Pattern recognition",
                    "Improved recommendations over time"
                ]
            }
        },
        "vs_pattern_matching": {
            "pattern_matching": "Hardcoded regex rules, no understanding",
            "true_ai": "Real LLM analysis, actual reasoning, learning"
        }
    }
