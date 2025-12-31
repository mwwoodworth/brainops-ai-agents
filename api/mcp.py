"""
MCP Integration API Router
===========================
Exposes the 345-tool MCP Bridge to the REST API.
Enables external systems and the UI to invoke MCP tools.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Bridge Integration"])

# Lazy imports to avoid circular dependencies
_client = None
_aurea = None
_self_healing = None
_revenue = None
_digital_twin = None


def _get_client():
    global _client
    if _client is None:
        from mcp_integration import get_mcp_client
        _client = get_mcp_client()
    return _client


def _get_aurea():
    global _aurea
    if _aurea is None:
        from mcp_integration import get_aurea_executor
        _aurea = get_aurea_executor()
    return _aurea


def _get_self_healing():
    global _self_healing
    if _self_healing is None:
        from mcp_integration import get_self_healing_integration
        _self_healing = get_self_healing_integration()
    return _self_healing


def _get_revenue():
    global _revenue
    if _revenue is None:
        from mcp_integration import get_revenue_integration
        _revenue = get_revenue_integration()
    return _revenue


def _get_digital_twin():
    global _digital_twin
    if _digital_twin is None:
        from mcp_integration import get_digital_twin_integration
        _digital_twin = get_digital_twin_integration()
    return _digital_twin


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ExecuteToolRequest(BaseModel):
    server: str  # render, vercel, supabase, github, stripe, etc.
    tool: str    # The tool name
    params: Optional[Dict[str, Any]] = None


class AUREADecisionRequest(BaseModel):
    decision_type: str  # DEPLOY, HEAL, REVENUE, DATA, CODE
    params: Dict[str, Any]


class SelfHealRequest(BaseModel):
    service_name: str
    action: Optional[str] = "auto"  # auto, restart, scale, diagnose


class RevenueActionRequest(BaseModel):
    action: str  # new_customer, payment, metrics
    email: Optional[str] = None
    name: Optional[str] = None
    plan: Optional[str] = "pro"
    customer_id: Optional[str] = None
    amount: Optional[int] = None


class TwinSyncRequest(BaseModel):
    twin_id: str
    action: Optional[str] = "sync"  # sync, simulate


# =============================================================================
# CORE MCP ENDPOINTS
# =============================================================================

@router.get("/status")
async def mcp_status():
    """Get MCP integration status and available tools"""
    return {
        "status": "operational",
        "bridge_url": "https://brainops-mcp-bridge.onrender.com",
        "total_tools": 345,
        "servers": {
            "render": {"tools": 39, "status": "available"},
            "vercel": {"tools": 34, "status": "available"},
            "supabase": {"tools": 40, "status": "available"},
            "github": {"tools": 50, "status": "available"},
            "docker": {"tools": 53, "status": "available"},
            "stripe": {"tools": 55, "status": "available"},
            "openai": {"tools": 7, "status": "available"},
            "anthropic": {"tools": 3, "status": "available"},
            "playwright": {"tools": 60, "status": "available"},
            "python-executor": {"tools": 8, "status": "available"}
        },
        "integrations": {
            "aurea": "connected",
            "self_healing": "connected",
            "revenue": "connected",
            "digital_twin": "connected"
        }
    }


@router.post("/execute")
async def execute_tool(request: ExecuteToolRequest):
    """
    Execute any MCP tool directly

    Example:
    {
        "server": "render",
        "tool": "listServices",
        "params": {}
    }
    """
    try:
        client = _get_client()
        result = await client.execute_tool(
            request.server,
            request.tool,
            request.params
        )

        return {
            "success": result.success,
            "server": result.server,
            "tool": result.tool,
            "result": result.result,
            "duration_ms": result.duration_ms,
            "error": result.error
        }
    except Exception as e:
        logger.error(f"MCP execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RENDER SHORTCUTS
# =============================================================================

@router.get("/render/services")
async def list_render_services():
    """List all Render services"""
    client = _get_client()
    result = await client.render_list_services()
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error)
    return {"services": result.result}


@router.post("/render/deploy/{service_id}")
async def trigger_render_deploy(service_id: str):
    """Trigger a Render deployment"""
    client = _get_client()
    result = await client.render_trigger_deploy(service_id)
    return {
        "success": result.success,
        "service_id": service_id,
        "result": result.result,
        "error": result.error
    }


@router.post("/render/restart/{service_id}")
async def restart_render_service(service_id: str):
    """Restart a Render service"""
    client = _get_client()
    result = await client.render_restart_service(service_id)
    return {
        "success": result.success,
        "service_id": service_id,
        "action": "restart",
        "result": result.result,
        "error": result.error
    }


@router.get("/render/logs/{service_id}")
async def get_render_logs(service_id: str, lines: int = 100):
    """Get Render service logs"""
    client = _get_client()
    result = await client.render_get_logs(service_id, lines)
    return {
        "success": result.success,
        "service_id": service_id,
        "logs": result.result,
        "error": result.error
    }


# =============================================================================
# SUPABASE SHORTCUTS
# =============================================================================

@router.post("/supabase/query")
async def supabase_query(query: str, params: List[Any] = None):
    """Execute a Supabase SQL query - RESTRICTED TO SELECT ONLY"""
    # SECURITY: Only allow SELECT statements to prevent SQL injection/data modification
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        raise HTTPException(
            status_code=403,
            detail="Only SELECT queries are allowed. Use other MCP endpoints for data modification."
        )

    # Additional safety: block dangerous patterns
    dangerous_patterns = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "EXEC", "EXECUTE", ";"]
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            raise HTTPException(
                status_code=403,
                detail=f"Query contains forbidden keyword: {pattern}"
            )

    client = _get_client()
    result = await client.supabase_query(query, params)
    return {
        "success": result.success,
        "result": result.result,
        "duration_ms": result.duration_ms,
        "error": result.error
    }


@router.get("/supabase/tables")
async def list_supabase_tables():
    """List all Supabase tables"""
    client = _get_client()
    result = await client.supabase_get_tables()
    return {"tables": result.result if result.success else [], "error": result.error}


# =============================================================================
# GITHUB SHORTCUTS
# =============================================================================

@router.get("/github/repos")
async def list_github_repos():
    """List GitHub repositories"""
    client = _get_client()
    result = await client.github_list_repos()
    return {"repos": result.result if result.success else [], "error": result.error}


@router.post("/github/pr")
async def create_github_pr(
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str = "main"
):
    """Create a GitHub pull request"""
    client = _get_client()
    result = await client.github_create_pr(repo, title, body, head, base)
    return {
        "success": result.success,
        "pr": result.result,
        "error": result.error
    }


# =============================================================================
# STRIPE SHORTCUTS
# =============================================================================

@router.get("/stripe/balance")
async def get_stripe_balance():
    """Get Stripe account balance"""
    client = _get_client()
    result = await client.stripe_get_balance()
    return {
        "success": result.success,
        "balance": result.result,
        "error": result.error
    }


@router.get("/stripe/invoices")
async def list_stripe_invoices(customer_id: str = None):
    """List Stripe invoices"""
    client = _get_client()
    result = await client.stripe_list_invoices(customer_id)
    return {
        "success": result.success,
        "invoices": result.result,
        "error": result.error
    }


# =============================================================================
# AUREA INTEGRATION
# =============================================================================

@router.post("/aurea/execute")
async def aurea_execute_decision(request: AUREADecisionRequest):
    """
    Execute an AUREA decision via MCP tools

    Decision types:
    - DEPLOY: Deploy to Render/Vercel
    - HEAL: Restart or scale services
    - REVENUE: Stripe operations
    - DATA: Supabase queries
    - CODE: GitHub operations
    """
    try:
        executor = _get_aurea()
        result = await executor.execute_decision(
            request.decision_type,
            request.params
        )
        return result
    except Exception as e:
        logger.error(f"AUREA execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SELF-HEALING INTEGRATION
# =============================================================================

@router.post("/heal")
async def self_heal_service(request: SelfHealRequest):
    """
    Trigger self-healing for a service

    Actions:
    - auto: Automatic remediation (restart, then scale)
    - restart: Force restart
    - scale: Scale up instances
    - diagnose: Get logs and status only
    """
    try:
        healer = _get_self_healing()

        if request.action == "diagnose":
            result = await healer.get_diagnostic_info(request.service_name)
            return {"action": "diagnose", "result": result}

        result = await healer.handle_unhealthy_service(request.service_name)
        return result
    except Exception as e:
        logger.error(f"Self-heal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# REVENUE INTEGRATION
# =============================================================================

@router.post("/revenue/action")
async def revenue_action(request: RevenueActionRequest):
    """
    Execute revenue operations via MCP

    Actions:
    - new_customer: Create customer + subscription
    - payment: Process one-time payment
    - metrics: Get revenue metrics
    """
    try:
        revenue = _get_revenue()

        if request.action == "new_customer":
            if not request.email:
                raise HTTPException(status_code=400, detail="email required")
            result = await revenue.process_new_customer(
                request.email,
                request.name or "",
                request.plan or "pro"
            )
            return result

        elif request.action == "payment":
            if not request.customer_id or not request.amount:
                raise HTTPException(status_code=400, detail="customer_id and amount required")
            result = await revenue.process_payment(
                request.customer_id,
                request.amount
            )
            return result

        elif request.action == "metrics":
            result = await revenue.get_revenue_metrics()
            return result

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revenue action error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DIGITAL TWIN INTEGRATION
# =============================================================================

@router.post("/twin/sync")
async def sync_digital_twin(request: TwinSyncRequest):
    """
    Sync a digital twin with real infrastructure

    Actions:
    - sync: Sync twin state with reality
    - simulate: Run simulation against real infra
    """
    try:
        twin = _get_digital_twin()

        if request.action == "sync":
            result = await twin.sync_twin_with_reality(request.twin_id)
            return result

        elif request.action == "simulate":
            result = await twin.run_simulation_on_real_infra(
                request.twin_id,
                "default_scenario"
            )
            return result

        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

    except Exception as e:
        logger.error(f"Twin sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# BULK OPERATIONS
# =============================================================================

@router.post("/bulk/execute")
async def bulk_execute(tools: List[ExecuteToolRequest]):
    """
    Execute multiple MCP tools in parallel

    Enables complex workflows like:
    1. Query Supabase for data
    2. Create GitHub PR
    3. Deploy to Render
    4. Send Stripe invoice
    All in one API call
    """
    import asyncio

    client = _get_client()

    async def execute_one(req: ExecuteToolRequest):
        return await client.execute_tool(req.server, req.tool, req.params)

    results = await asyncio.gather(*[execute_one(t) for t in tools])

    return {
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
        "results": [
            {
                "server": r.server,
                "tool": r.tool,
                "success": r.success,
                "result": r.result,
                "error": r.error
            }
            for r in results
        ]
    }
