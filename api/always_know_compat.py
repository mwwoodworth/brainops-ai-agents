#!/usr/bin/env python3
"""
Always-Know Brain Compatibility API
===================================
Compatibility endpoints for legacy /always-know routes.
"""

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api import always_know as always_know_api

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/always-know", tags=["Always-Know Observability (Compat)"])


@router.get("/state")
async def get_current_state() -> dict[str, Any]:
    return await always_know_api.get_current_state()


@router.get("/summary", response_class=PlainTextResponse)
async def get_state_summary() -> str:
    return await always_know_api.get_state_summary()


@router.get("/alerts")
async def get_active_alerts() -> list[dict[str, Any]]:
    return await always_know_api.get_active_alerts()


@router.get("/health")
async def get_quick_health() -> dict[str, Any]:
    return await always_know_api.get_quick_health()


@router.post("/test-ui")
async def trigger_ui_tests() -> dict[str, Any]:
    return await always_know_api.trigger_ui_tests()


@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics() -> str:
    return await always_know_api.get_prometheus_metrics()


@router.get("/history")
async def get_state_history(limit: int = 100) -> list[dict[str, Any]]:
    return await always_know_api.get_state_history(limit)


@router.post("/chatgpt-agent-test")
async def run_chatgpt_agent_test(
    full: bool = False,
    skip_erp: bool = False,
    blocking: bool = False,
) -> dict[str, Any]:
    return await always_know_api.run_chatgpt_agent_test(
        full=full,
        skip_erp=skip_erp,
        blocking=blocking,
    )


@router.get("/chatgpt-agent-test/{run_id}")
async def get_chatgpt_agent_test_result(run_id: str) -> dict[str, Any]:
    return await always_know_api.get_chatgpt_agent_test_result(run_id)
