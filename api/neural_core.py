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
            "POST /neural/ask": "Have a conversation",
            "POST /neural/heal": "Request healing",
            "POST /neural/initialize": "Wake up the neural core"
        },
        "state": get_neural_core().state.value if NEURAL_CORE_AVAILABLE else "unavailable"
    }
