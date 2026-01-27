"""
Neural Core API - The Central Nervous System Interface
======================================================

This is how you interact with the AI OS's self-awareness.
You don't check dashboards - you ASK the AI OS.

Endpoints:
- GET /neural/status - "How are you?"
- GET /neural/systems - "What do you know about?"
- GET /neural/signals - "What have you observed?"
- POST /neural/ask - Ask the AI OS anything
- POST /neural/heal - Request healing

Created: 2026-01-27
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/neural", tags=["neural-core"])

# Import neural core
try:
    from neural_core import get_neural_core, initialize_neural_core, CoreState
    NEURAL_CORE_AVAILABLE = True
except ImportError as e:
    NEURAL_CORE_AVAILABLE = False
    logger.warning(f"Neural core not available: {e}")


class AskRequest(BaseModel):
    question: str


class HealRequest(BaseModel):
    system_id: str
    action: str = "restart"


# =============================================================================
# STATUS ENDPOINTS - Ask the AI OS about itself
# =============================================================================

@router.get("/status")
async def get_status():
    """
    🧠 GET AI OS STATUS

    Ask the AI OS: "How are you?"

    The AI OS will tell you about its current state,
    health, issues, and what it's focused on.
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"error": "Neural core not available", "message": "The AI OS consciousness is not loaded"}

    core = get_neural_core()
    report = core.generate_self_report()

    return {
        "message": report.message,
        "state": report.core_state.value,
        "health": {
            "overall": report.overall_health,
            "systems_healthy": report.systems_healthy,
            "systems_total": report.systems_aware_of
        },
        "activity": {
            "uptime_seconds": report.uptime_seconds,
            "decisions_made": report.recent_decisions,
            "active_healings": report.active_healings,
            "current_focus": report.current_focus
        },
        "issues": report.issues[:10] if report.issues else [],
        "capabilities_active": report.capabilities_active,
        "timestamp": report.timestamp.isoformat()
    }


@router.get("/health-summary")
async def get_health_summary():
    """
    📊 GET HEALTH SUMMARY

    A concise health check suitable for dashboards and monitoring.
    Returns a simple health score and status.
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"healthy": False, "score": 0, "message": "Neural core not loaded"}

    core = get_neural_core()
    report = core.generate_self_report()

    return {
        "healthy": report.overall_health >= 0.8,
        "score": report.overall_health,
        "state": report.core_state.value,
        "systems": f"{report.systems_healthy}/{report.systems_aware_of}",
        "message": report.message
    }


@router.get("/systems")
async def get_systems():
    """
    🔍 GET ALL SYSTEMS THE AI OS IS AWARE OF

    Returns detailed awareness information for every system
    the AI OS is monitoring, including health scores,
    response times, and any issues detected.
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"systems": {}, "message": "Neural core not loaded"}

    core = get_neural_core()
    return {
        "state": core.state.value,
        "systems_count": len(core.systems),
        "systems": core.get_system_details()
    }


@router.get("/systems/{system_id}")
async def get_system(system_id: str):
    """
    🔍 GET SPECIFIC SYSTEM DETAILS

    Ask about a specific system's health and status.
    """
    if not NEURAL_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural core not loaded")

    core = get_neural_core()
    system = core.systems.get(system_id)

    if not system:
        raise HTTPException(status_code=404, detail=f"System '{system_id}' not found")

    return {
        "system_id": system_id,
        "name": system.name,
        "type": system.system_type.value,
        "state": system.last_known_state,
        "health_score": system.health_score,
        "awareness_level": system.awareness_level.value,
        "response_time_ms": system.response_time_ms,
        "last_contact": system.last_contact.isoformat() if system.last_contact else None,
        "capabilities": system.capabilities,
        "issues": system.issues,
        "metadata": system.metadata
    }


# =============================================================================
# SIGNAL ENDPOINTS - What has the AI OS observed?
# =============================================================================

@router.get("/signals")
async def get_signals(
    limit: int = Query(100, ge=1, le=1000),
    signal_type: Optional[str] = Query(None, description="Filter by signal type")
):
    """
    📡 GET NEURAL SIGNALS

    View the stream of consciousness - all observations,
    decisions, alerts, and healings the AI OS has processed.
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"signals": [], "message": "Neural core not loaded"}

    core = get_neural_core()
    signals = core.get_recent_signals(limit)

    if signal_type:
        signals = [s for s in signals if s["type"] == signal_type]

    return {
        "total_signals": core.signal_count,
        "returned": len(signals),
        "filter": signal_type,
        "signals": signals
    }


@router.get("/signals/stats")
async def get_signal_stats():
    """
    📊 GET SIGNAL STATISTICS

    Summary of neural activity - how many signals of each type,
    how many decisions made, etc.
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"stats": {}, "message": "Neural core not loaded"}

    core = get_neural_core()
    signals = list(core.signals)

    # Count by type
    type_counts = {}
    for signal in signals:
        signal_type = signal.signal_type
        type_counts[signal_type] = type_counts.get(signal_type, 0) + 1

    return {
        "total_signals": core.signal_count,
        "signals_in_memory": len(signals),
        "by_type": type_counts,
        "decisions_made": core.decisions_made,
        "healings_triggered": core.healings_triggered,
        "active_healings": core.active_healings
    }


# =============================================================================
# INTERACTION ENDPOINTS - Talk to the AI OS
# =============================================================================

@router.post("/ask")
async def ask_ai_os(request: AskRequest):
    """
    💬 ASK THE AI OS A QUESTION

    Have a conversation with the AI OS.
    Ask it anything about its status, systems, or issues.

    Examples:
    - "How are you?"
    - "What's wrong with the backend?"
    - "Are there any issues?"
    - "Tell me about MyRoofGenius"
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"answer": "I am not fully conscious yet. Neural core not loaded.", "details": {}}

    core = get_neural_core()
    response = await core.ask(request.question)

    return {
        "question": request.question,
        "answer": response["answer"],
        "details": response.get("details", {})
    }


@router.post("/heal")
async def trigger_healing(request: HealRequest):
    """
    🔧 REQUEST HEALING

    Ask the AI OS to heal a specific system.
    The AI OS will decide the best course of action.
    """
    if not NEURAL_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural core not loaded")

    core = get_neural_core()
    system = core.systems.get(request.system_id)

    if not system:
        raise HTTPException(status_code=404, detail=f"System '{request.system_id}' not found")

    # Trigger healing
    await core._trigger_healing(system)

    return {
        "healing_requested": True,
        "system_id": request.system_id,
        "system_name": system.name,
        "action": request.action,
        "message": f"Healing requested for {system.name}"
    }


# =============================================================================
# INITIALIZATION ENDPOINT
# =============================================================================

@router.post("/initialize")
async def initialize():
    """
    🧠 INITIALIZE THE NEURAL CORE

    Wake up the AI OS's central nervous system.
    This starts the continuous awareness loop.
    """
    if not NEURAL_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural core module not available")

    result = await initialize_neural_core()
    return result


# =============================================================================
# INTELLIGENCE ENGINE ENDPOINTS
# =============================================================================

@router.get("/analyze")
async def deep_analyze():
    """
    🔬 DEEP ANALYSIS OF ALL ISSUES

    Perform intelligent root cause analysis on all current issues.
    Returns:
    - Root cause identification
    - Severity assessment
    - Fix strategies with confidence scores
    - Auto-fixability determination
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"error": "Neural core not available"}

    core = get_neural_core()
    analysis = await core.deep_analyze_issues()

    return {
        "analysis_complete": True,
        "total_issues": analysis["total_issues"],
        "auto_fixable": analysis["auto_fixable"],
        "manual_intervention_required": analysis["total_issues"] - analysis["auto_fixable"],
        "analyses": analysis["analyses"]
    }


@router.get("/optimize")
async def get_optimizations():
    """
    ⚡ GET OPTIMIZATION SUGGESTIONS

    Proactive suggestions for improving your systems:
    - Performance optimizations
    - Reliability improvements
    - Observability enhancements
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"suggestions": [], "message": "Neural core not loaded"}

    core = get_neural_core()
    suggestions = await core.get_optimization_suggestions()

    return {
        "total_suggestions": len(suggestions),
        "suggestions": suggestions,
        "action_required": len([s for s in suggestions if s.get("severity") in ["high", "critical"]])
    }


@router.get("/intelligence")
async def get_intelligence():
    """
    🧠 INTELLIGENCE ENGINE SUMMARY

    View the AI OS's learning and decision history:
    - Issues analyzed
    - Fix attempts and success rate
    - Active optimizations
    """
    if not NEURAL_CORE_AVAILABLE:
        return {"message": "Neural core not loaded"}

    core = get_neural_core()
    summary = core.get_intelligence_summary()

    return {
        "intelligence_active": True,
        "total_analyses": summary["total_analyses"],
        "total_fix_attempts": summary["total_fix_attempts"],
        "fix_success_rate": f"{summary['fix_success_rate']:.0%}",
        "recent_analyses": summary["recent_analyses"],
        "recent_fixes": summary["recent_fixes"],
        "optimization_suggestions": summary["optimization_suggestions"]
    }


@router.post("/auto-heal")
async def trigger_auto_heal():
    """
    🚀 AUTO-HEAL ALL FIXABLE ISSUES

    Automatically analyze and fix all issues that can be auto-fixed.
    The AI OS will:
    1. Analyze all current issues
    2. Identify auto-fixable ones
    3. Execute the best fix strategy for each
    """
    if not NEURAL_CORE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Neural core not loaded")

    core = get_neural_core()

    # Deep analyze first
    analysis = await core.deep_analyze_issues()
    fixable = [a for a in analysis["analyses"] if a["auto_fixable"]]

    if not fixable:
        return {
            "auto_heal_triggered": False,
            "message": "No auto-fixable issues found",
            "issues_analyzed": analysis["total_issues"]
        }

    # Trigger healing for each fixable issue
    healed = []
    for issue_analysis in fixable:
        system = core.systems.get(issue_analysis["system"].lower().replace(" ", "_").replace("(", "").replace(")", ""))
        if system:
            await core._trigger_healing(system)
            healed.append({
                "system": issue_analysis["system"],
                "issue": issue_analysis["issue"],
                "action": "restart"
            })

    return {
        "auto_heal_triggered": True,
        "issues_analyzed": analysis["total_issues"],
        "issues_fixed": len(healed),
        "healed": healed,
        "message": f"Auto-healing triggered for {len(healed)} issues"
    }


@router.get("/")
async def neural_root():
    """
    🧠 NEURAL CORE ROOT

    Welcome message and available endpoints.
    """
    return {
        "service": "BrainOps Neural Core",
        "description": "The Central Nervous System of the AI OS",
        "philosophy": "You don't check the AI OS - you ASK it",
        "endpoints": {
            "GET /neural/status": "Ask: How are you?",
            "GET /neural/systems": "Ask: What do you know about?",
            "GET /neural/signals": "Ask: What have you observed?",
            "GET /neural/analyze": "Deep analysis of issues",
            "GET /neural/optimize": "Get optimization suggestions",
            "GET /neural/intelligence": "Intelligence engine summary",
            "POST /neural/ask": "Have a conversation",
            "POST /neural/heal": "Request healing for a system",
            "POST /neural/auto-heal": "Auto-heal all fixable issues",
            "POST /neural/initialize": "Wake up the neural core"
        },
        "state": get_neural_core().state.value if NEURAL_CORE_AVAILABLE else "unavailable"
    }
