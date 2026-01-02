#!/usr/bin/env python3
"""
Always-Know Brain API
======================
API endpoints for the comprehensive observability brain.
Provides instant access to system state without querying.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/always-know", tags=["Always-Know Observability"])

# Lazy import
_brain = None


def get_brain():
    global _brain
    if _brain is None:
        try:
            from always_know_brain import get_always_know_brain
            _brain = get_always_know_brain()
        except Exception as e:
            logger.error(f"Failed to get Always-Know Brain: {e}")
    return _brain


@router.get("/state")
async def get_current_state() -> dict[str, Any]:
    """
    Get the current system state.
    This is cached and updated every 30 seconds - no need to query services.
    """
    brain = get_brain()
    if not brain:
        raise HTTPException(status_code=503, detail="Always-Know Brain not available")

    return brain.get_current_state()


@router.get("/summary", response_class=PlainTextResponse)
async def get_state_summary() -> str:
    """
    Get a human-readable summary of system state.
    Perfect for terminal display.
    """
    brain = get_brain()
    if not brain:
        return "Always-Know Brain not available"

    return brain.get_state_summary()


@router.get("/alerts")
async def get_active_alerts() -> list[dict[str, Any]]:
    """
    Get all active (unresolved) alerts.
    """
    brain = get_brain()
    if not brain:
        return []

    return brain.get_active_alerts()


@router.get("/health")
async def get_quick_health() -> dict[str, Any]:
    """
    Get a quick health check based on cached state.
    Much faster than hitting all services.
    """
    brain = get_brain()
    if not brain:
        return {"status": "unknown", "brain_available": False}

    state = brain.current_state

    # Calculate overall health
    services = {
        "ai_agents": state.ai_agents_healthy,
        "backend": state.backend_healthy,
        "mcp_bridge": state.mcp_bridge_healthy,
        "database": state.database_connected,
        "aurea": state.aurea_operational,
        "mrg_frontend": state.mrg_healthy,
        "erp_frontend": state.erp_healthy
    }

    healthy_count = sum(services.values())
    total_count = len(services)

    if healthy_count == total_count:
        status = "healthy"
    elif healthy_count >= total_count * 0.7:
        status = "degraded"
    else:
        status = "unhealthy"

    return {
        "status": status,
        "healthy_services": healthy_count,
        "total_services": total_count,
        "health_percentage": (healthy_count / total_count) * 100,
        "services": services,
        "errors_last_hour": state.errors_last_hour,
        "response_time_ms": state.response_time_ms,
        "active_alerts": len(brain.get_active_alerts()),
        "last_update": state.timestamp
    }


@router.post("/test-ui")
async def trigger_ui_tests() -> dict[str, Any]:
    """
    Trigger manual UI tests for all frontends.
    """
    brain = get_brain()
    if not brain:
        raise HTTPException(status_code=503, detail="Always-Know Brain not available")

    try:
        import asyncio
        asyncio.create_task(brain._run_ui_tests())
        return {
            "status": "triggered",
            "message": "UI tests started in background. Check /always-know/state for results."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> str:
    """
    Get metrics in Prometheus format.
    Ready for scraping by Prometheus.
    """
    brain = get_brain()
    if not brain:
        return "# Always-Know Brain not available\n"

    state = brain.current_state

    lines = [
        "# HELP brainops_service_healthy Service health status (1=healthy, 0=unhealthy)",
        "# TYPE brainops_service_healthy gauge",
        f'brainops_service_healthy{{service="ai_agents"}} {1 if state.ai_agents_healthy else 0}',
        f'brainops_service_healthy{{service="backend"}} {1 if state.backend_healthy else 0}',
        f'brainops_service_healthy{{service="mcp_bridge"}} {1 if state.mcp_bridge_healthy else 0}',
        f'brainops_service_healthy{{service="database"}} {1 if state.database_connected else 0}',
        f'brainops_service_healthy{{service="aurea"}} {1 if state.aurea_operational else 0}',
        f'brainops_service_healthy{{service="mrg_frontend"}} {1 if state.mrg_healthy else 0}',
        f'brainops_service_healthy{{service="erp_frontend"}} {1 if state.erp_healthy else 0}',
        "",
        "# HELP brainops_errors_total Total errors in time window",
        "# TYPE brainops_errors_total counter",
        f'brainops_errors_total{{window="1h"}} {state.errors_last_hour}',
        f'brainops_errors_total{{window="24h"}} {state.errors_last_24h}',
        "",
        "# HELP brainops_response_time_ms Response time in milliseconds",
        "# TYPE brainops_response_time_ms gauge",
        f"brainops_response_time_ms {state.response_time_ms}",
        "",
        "# HELP brainops_aurea_ooda_cycles AUREA OODA cycles in last 5 minutes",
        "# TYPE brainops_aurea_ooda_cycles gauge",
        f"brainops_aurea_ooda_cycles {state.aurea_ooda_cycles}",
        "",
        "# HELP brainops_aurea_decisions AUREA decisions in last hour",
        "# TYPE brainops_aurea_decisions gauge",
        f"brainops_aurea_decisions {state.aurea_decisions}",
        "",
        "# HELP brainops_aurea_active_agents Number of active agents",
        "# TYPE brainops_aurea_active_agents gauge",
        f"brainops_aurea_active_agents {state.aurea_active_agents}",
        "",
        "# HELP brainops_embedded_memories Total embedded memories",
        "# TYPE brainops_embedded_memories gauge",
        f"brainops_embedded_memories {state.embedded_memories}",
        "",
        "# HELP brainops_customers_total Total customers",
        "# TYPE brainops_customers_total gauge",
        f"brainops_customers_total {state.customers_total}",
        "",
        "# HELP brainops_jobs_total Total jobs",
        "# TYPE brainops_jobs_total gauge",
        f"brainops_jobs_total {state.jobs_total}",
        "",
        "# HELP brainops_active_alerts Number of active alerts",
        "# TYPE brainops_active_alerts gauge",
        f"brainops_active_alerts {len(brain.get_active_alerts())}",
    ]

    return "\n".join(lines) + "\n"


@router.get("/history")
async def get_state_history(limit: int = 100) -> list[dict[str, Any]]:
    """
    Get historical state snapshots.
    """
    brain = get_brain()
    if not brain:
        return []

    from dataclasses import asdict
    return [asdict(s) for s in brain.state_history[-limit:]]


@router.post("/chatgpt-agent-test")
async def run_chatgpt_agent_test(full: bool = False) -> dict[str, Any]:
    """
    Run ChatGPT-Agent-Level UI tests.
    These are real human-like tests that login, navigate, fill forms, etc.

    Args:
        full: If True, runs full test suite. If False, runs quick health check.
    """
    try:
        from chatgpt_agent_tester import run_chatgpt_agent_tests, run_quick_health_test

        if full:
            return await run_chatgpt_agent_tests()
        else:
            return await run_quick_health_test()
    except ImportError:
        raise HTTPException(status_code=503, detail="ChatGPT Agent Tester not available (Playwright not installed)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
