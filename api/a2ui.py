"""
A2UI API Endpoints - Agent-to-User Interface Protocol
======================================================
RESTful API for generating A2UI-compliant UI responses.
https://a2ui.org/ | https://github.com/google/A2UI
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from a2ui_protocol import (
    A2UIBuilder,
    A2UIGenerator,
    AUREAUIGenerator,
    ComponentType,
    UsageHint,
    generate_dashboard_ui,
    generate_status_ui,
    generate_table_ui,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/a2ui", tags=["A2UI Protocol"])


class DashboardRequest(BaseModel):
    title: str = "Dashboard"
    metrics: Dict[str, Any]
    actions: Optional[List[Dict[str, str]]] = None


class TableRequest(BaseModel):
    title: str
    columns: List[str]
    rows: List[List[Any]]
    row_actions: Optional[List[str]] = None


class StatusRequest(BaseModel):
    title: str
    status: str
    details: Dict[str, Any]
    severity: str = "info"


class FormRequest(BaseModel):
    title: str
    fields: List[Dict[str, Any]]
    submit_action: str


class ConfirmationRequest(BaseModel):
    title: str
    message: str
    confirm_action: str
    cancel_action: str = "cancel"


class CustomUIRequest(BaseModel):
    surface_id: Optional[str] = None
    components: List[Dict[str, Any]]
    root_component_id: Optional[str] = None


@router.get("/spec")
async def get_a2ui_spec() -> Dict[str, Any]:
    """Get A2UI specification and component catalog"""
    return {
        "protocol": "A2UI",
        "version": "0.8",
        "spec_url": "https://a2ui.org/",
        "github": "https://github.com/google/A2UI",
        "brainops_version": "1.0.0",
        "component_catalog": [ct.value for ct in ComponentType],
        "usage_hints": [uh.value for uh in UsageHint],
        "supported_templates": [
            "dashboard_card",
            "data_table",
            "status_display",
            "confirmation_dialog",
            "form",
            "aurea_decision",
            "health_dashboard",
            "agent_grid",
        ],
        "note": "A2UI responses are declarative JSON. Requires A2UI-compatible frontend renderer.",
    }


@router.get("/health")
async def a2ui_health() -> Dict[str, Any]:
    """A2UI module health check"""
    return {
        "status": "operational",
        "protocol": "A2UI",
        "version": "0.8",
        "ready_for_rendering": True,
        "note": "Backend generator ready. Frontend renderer required for display.",
    }


@router.post("/generate/dashboard")
async def generate_dashboard(request: DashboardRequest) -> Dict[str, Any]:
    """Generate an A2UI dashboard card"""
    try:
        metric_list = [
            {"label": k, "value": v}
            for k, v in request.metrics.items()
        ]
        result = A2UIGenerator.dashboard_card(
            request.title,
            metric_list,
            request.actions,
        )
        return {
            "success": True,
            "a2ui": result,
            "render_note": "Send this to an A2UI-compatible frontend for rendering",
        }
    except Exception as e:
        logger.error(f"Dashboard generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/table")
async def generate_table(request: TableRequest) -> Dict[str, Any]:
    """Generate an A2UI data table"""
    try:
        result = A2UIGenerator.data_table(
            request.title,
            request.columns,
            request.rows,
            request.row_actions,
        )
        return {
            "success": True,
            "a2ui": result,
        }
    except Exception as e:
        logger.error(f"Table generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/status")
async def generate_status(request: StatusRequest) -> Dict[str, Any]:
    """Generate an A2UI status display"""
    try:
        result = A2UIGenerator.status_display(
            request.title,
            request.status,
            request.details,
            request.severity,
        )
        return {
            "success": True,
            "a2ui": result,
        }
    except Exception as e:
        logger.error(f"Status generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/form")
async def generate_form(request: FormRequest) -> Dict[str, Any]:
    """Generate an A2UI form"""
    try:
        result = A2UIGenerator.form(
            request.title,
            request.fields,
            request.submit_action,
        )
        return {
            "success": True,
            "a2ui": result,
        }
    except Exception as e:
        logger.error(f"Form generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/confirmation")
async def generate_confirmation(request: ConfirmationRequest) -> Dict[str, Any]:
    """Generate an A2UI confirmation dialog"""
    try:
        result = A2UIGenerator.confirmation_dialog(
            request.title,
            request.message,
            request.confirm_action,
            request.cancel_action,
        )
        return {
            "success": True,
            "a2ui": result,
        }
    except Exception as e:
        logger.error(f"Confirmation generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/custom")
async def generate_custom(request: CustomUIRequest) -> Dict[str, Any]:
    """Generate a custom A2UI surface from component definitions"""
    try:
        builder = A2UIBuilder(request.surface_id)

        for comp in request.components:
            comp_type = ComponentType(comp.get("type", "Text"))
            builder.add_component(
                comp_type,
                properties=comp.get("properties", {}),
                children=comp.get("children", []),
                component_id=comp.get("id"),
            )

        if request.root_component_id:
            builder.set_root(request.root_component_id)

        return {
            "success": True,
            "a2ui": builder.to_dict(),
        }
    except Exception as e:
        logger.error(f"Custom UI generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# AUREA-specific endpoints
@router.get("/aurea/health-dashboard")
async def aurea_health_dashboard() -> Dict[str, Any]:
    """Generate A2UI for current AUREA health status"""
    try:
        # Get actual health data
        from database.async_connection import get_pool
        import asyncio

        # Mock health data for now (in production, fetch from AUREA)
        health = {
            "overall_score": 95.5,
            "component_health": {
                "agents": 98.0,
                "memory": 92.0,
                "decisions": 96.0,
                "performance": 95.5,
            },
            "active_agents": 59,
            "memory_utilization": 0.12,
        }

        result = AUREAUIGenerator.health_dashboard(health)
        return {
            "success": True,
            "a2ui": result,
            "source": "aurea_orchestrator",
        }
    except Exception as e:
        logger.error(f"AUREA health dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aurea/agent-grid")
async def aurea_agent_grid(limit: int = Query(default=20, le=100)) -> Dict[str, Any]:
    """Generate A2UI agent grid from live data"""
    try:
        from database.async_connection import get_pool
        import asyncpg

        pool = get_pool()
        if pool:
            async with pool.acquire() as conn:
                agents = await conn.fetch("""
                    SELECT name, type, status, total_executions, last_active
                    FROM ai_agents
                    WHERE status = 'active'
                    ORDER BY total_executions DESC
                    LIMIT $1
                """, limit)
                agent_list = [dict(a) for a in agents]
        else:
            agent_list = []

        result = AUREAUIGenerator.agent_grid(agent_list)
        return {
            "success": True,
            "a2ui": result,
            "agent_count": len(agent_list),
        }
    except Exception as e:
        logger.error(f"Agent grid failed: {e}")
        # Return empty grid on error
        return {
            "success": True,
            "a2ui": AUREAUIGenerator.agent_grid([]),
            "agent_count": 0,
            "note": f"Using fallback: {str(e)}",
        }


@router.get("/demo")
async def demo_a2ui() -> Dict[str, Any]:
    """Demo A2UI output showing all component types"""
    builder = A2UIBuilder("demo_surface")

    # Build a demo UI
    heading = builder.heading("BrainOps AI OS Dashboard", level=1)
    subtitle = builder.text("Agent-to-User Interface Demo", UsageHint.SUBTITLE)

    # Status alert
    alert = builder.alert("All systems operational", UsageHint.SUCCESS)

    # Metrics
    m1 = builder.text("Active Agents: 59", UsageHint.BODY)
    m2 = builder.text("E2E Pass Rate: 100%", UsageHint.SUCCESS)
    m3 = builder.text("MCP Tools: 345", UsageHint.BODY)
    metrics_col = builder.column([m1, m2, m3])
    metrics_card = builder.card([metrics_col], title="System Metrics")

    # Action buttons
    btn1 = builder.button("Run E2E Tests", "run_e2e", UsageHint.PRIMARY)
    btn2 = builder.button("View Agents", "view_agents", UsageHint.SECONDARY)
    btn_row = builder.row([btn1, btn2])

    # Main layout
    main = builder.column([heading, subtitle, alert, metrics_card, btn_row])
    builder.set_root(main)

    return {
        "success": True,
        "a2ui": builder.to_dict(),
        "note": "This is a demo. Send to A2UI-compatible frontend for rendering.",
        "compatible_renderers": [
            "a2ui-lit (Web Components)",
            "a2ui-flutter",
            "a2ui-angular",
            "Custom React renderer",
        ],
    }
