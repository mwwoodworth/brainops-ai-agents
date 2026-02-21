"""Content, Revenue, Inventory, and Observability router — Wave 2C/2D extraction from app.py.

Routes: /content/*, /inventory/*, /revenue/*, /logs/recent, /observability/full,
/debug/all-errors, /system/unified-status
"""

import logging
import os
from collections import deque
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request
from pydantic import BaseModel

from config import config
from database.async_connection import get_pool
from services.tenant_helpers import fetchval_with_tenant_context, resolve_tenant_uuid_from_request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["content_revenue"])


# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------


def _get_app():
    import app as _app

    return _app.app


def _version():
    import app as _app

    return getattr(_app, "VERSION", "unknown")


# ---------------------------------------------------------------------------
# Module-level conditional imports
# ---------------------------------------------------------------------------

try:
    from multi_ai_content_orchestrator import MultiAIContentOrchestrator, ContentType

    CONTENT_ORCHESTRATOR_AVAILABLE = True
    logger.info("Multi-AI Content Orchestrator loaded")
except ImportError as e:
    CONTENT_ORCHESTRATOR_AVAILABLE = False
    logger.warning(f"Multi-AI Content Orchestrator not available: {e}")

try:
    from revenue_intelligence_system import (
        RevenueIntelligenceSystem,
        get_revenue_system,
        get_business_state,
        get_revenue,
        sync_to_brain,
    )

    REVENUE_INTEL_AVAILABLE = True
    logger.info("Revenue Intelligence System loaded")
except ImportError as e:
    REVENUE_INTEL_AVAILABLE = False
    logger.warning(f"Revenue Intelligence System not available: {e}")


# ---------------------------------------------------------------------------
# Observability infrastructure
# ---------------------------------------------------------------------------

LOG_BUFFER: deque = deque(maxlen=500)


class LogCapture(logging.Handler):
    """Capture logs to buffer for API access"""

    def emit(self, record):
        try:
            LOG_BUFFER.append(
                {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "line": record.lineno,
                }
            )
        except Exception:
            self.handleError(record)


# Add log capture handler
_log_capture = LogCapture()
_log_capture.setLevel(logging.INFO)
logging.getLogger().addHandler(_log_capture)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ContentGenerationRequest(BaseModel):
    content_type: str = "blog_post"
    topic: str
    brand: str = "BrainOps"
    target_audience: str = "tech professionals"
    chapters: int = 5
    module_number: int = 1
    include_image: bool = True


# ---------------------------------------------------------------------------
# Content Endpoints
# ---------------------------------------------------------------------------


async def _run_content_generation(orchestrator, task, job_id):
    """Background content generation with status tracking."""
    try:
        result = await orchestrator.execute(task)
        logger.info(f"Content job {job_id} completed: {result.get('status')}")
    except Exception as e:
        logger.error(f"Content job {job_id} failed: {e}")


@router.post("/content/generate", tags=["Content"])
async def generate_content(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate content using Multi-AI Orchestration.
    """
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()

    task = {
        "content_type": request.content_type,
        "topic": request.topic,
        "brand": request.brand,
        "target_audience": request.target_audience,
        "chapters": request.chapters,
        "module_number": request.module_number,
        "include_image": request.include_image,
    }

    if request.content_type in ["ebook", "ebook_full"]:
        import uuid

        job_id = str(uuid.uuid4())
        background_tasks.add_task(_run_content_generation, orchestrator, task, job_id)
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Ebook generation started for: {request.topic}",
            "check_status": f"/content/status/{job_id}",
        }

    result = await orchestrator.execute(task)
    return result


@router.post("/content/newsletter", tags=["Content"])
async def generate_newsletter(
    topic: str = Body(..., embed=True),
    brand: str = Body("BrainOps", embed=True),
):
    """Generate a complete newsletter with HTML template."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_newsletter({"topic": topic, "brand": brand})
    return result


@router.post("/content/ebook", tags=["Content"])
async def generate_ebook(
    topic: str = Body(..., embed=True),
    chapters: int = Body(5, embed=True),
    author: str = Body("BrainOps AI", embed=True),
    background_tasks: BackgroundTasks = None,
):
    """Generate a complete ebook with multiple chapters."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_ebook(
        {"topic": topic, "chapters": chapters, "author": author}
    )
    return result


@router.post("/content/training", tags=["Content"])
async def generate_training_doc(
    topic: str = Body(..., embed=True),
    module_number: int = Body(1, embed=True),
    skill_level: str = Body("beginner", embed=True),
):
    """Generate training documentation with exercises and quizzes."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_training_doc(
        {"topic": topic, "module_number": module_number, "skill_level": skill_level}
    )
    return result


@router.get("/content/types", tags=["Content"])
async def get_content_types():
    """Get available content generation types (requires auth)."""
    return {
        "types": [
            {
                "id": "blog_post",
                "name": "Blog Post",
                "description": "SEO-optimized blog articles with research",
            },
            {
                "id": "newsletter",
                "name": "Newsletter",
                "description": "Complete email newsletters with HTML templates",
            },
            {
                "id": "ebook",
                "name": "Ebook",
                "description": "Full ebooks with multiple chapters and TOC",
            },
            {
                "id": "training",
                "name": "Training Doc",
                "description": "Training modules with exercises and quizzes",
            },
        ],
        "models_used": [
            "perplexity:sonar-pro",
            "anthropic:claude-3-sonnet",
            "google:gemini-2.0-flash",
            "openai:gpt-4-turbo-preview",
            "openai:dall-e-3",
        ],
    }


# ---------------------------------------------------------------------------
# Inventory Endpoints
# ---------------------------------------------------------------------------


@router.get("/inventory/products", tags=["Inventory"])
async def get_product_inventory():
    """
    Get complete product inventory across all platforms.
    """
    inventory = {
        "last_updated": datetime.utcnow().isoformat(),
        "platforms": {
            "gumroad": {
                "store_url": "https://woodworthia.gumroad.com",
                "products": [
                    {
                        "code": "HJHMSM",
                        "name": "MCP Server Starter Kit",
                        "price": 97,
                        "type": "code_kit",
                        "url": "https://woodworthia.gumroad.com/l/hjhmsm",
                        "status": "active",
                        "description": "Build AI tool integrations fast with MCP Server patterns",
                    },
                    {
                        "code": "GSAAVB",
                        "name": "AI Orchestration Framework",
                        "price": 147,
                        "type": "code_kit",
                        "url": "https://woodworthia.gumroad.com/l/gsaavb",
                        "status": "active",
                        "description": "Multi-LLM smart routing and orchestration system",
                    },
                    {
                        "code": "VJXCEW",
                        "name": "SaaS Automation Scripts",
                        "price": 67,
                        "type": "code_kit",
                        "url": "https://woodworthia.gumroad.com/l/vjxcew",
                        "status": "active",
                    },
                    {
                        "code": "UPSYKR",
                        "name": "Command Center UI Kit",
                        "price": 149,
                        "type": "code_kit",
                        "url": "https://woodworthia.gumroad.com/l/upsykr",
                        "status": "active",
                    },
                    {
                        "code": "XGFKP",
                        "name": "AI Prompt Engineering Pack",
                        "price": 47,
                        "type": "prompt_pack",
                        "url": "https://woodworthia.gumroad.com/l/xgfkp",
                        "status": "active",
                    },
                    {
                        "code": "CAWVO",
                        "name": "Business Automation Toolkit",
                        "price": 49,
                        "type": "prompt_pack",
                        "url": "https://woodworthia.gumroad.com/l/cawvo",
                        "status": "active",
                    },
                    {
                        "code": "GR-ERP-START",
                        "name": "SaaS ERP Starter Kit",
                        "price": 197,
                        "type": "code_kit",
                        "url": "https://woodworthia.gumroad.com/l/gr-erp-start",
                        "status": "active",
                        "description": "Multi-tenant SaaS foundation with auth, CRM, jobs, invoicing",
                    },
                    {
                        "code": "GR-CONTENT",
                        "name": "AI Content Production Pipeline",
                        "price": 347,
                        "type": "automation",
                        "url": "https://woodworthia.gumroad.com/l/gr-content",
                        "status": "active",
                        "description": "Scale content 10x with multi-stage AI pipeline",
                    },
                    {
                        "code": "GR-ONBOARD",
                        "name": "Intelligent Client Onboarding",
                        "price": 297,
                        "type": "automation",
                        "url": "https://woodworthia.gumroad.com/l/gr-onboard",
                        "status": "active",
                    },
                    {
                        "code": "GR-PMCMD",
                        "name": "AI Project Command Center (BrainOps)",
                        "price": 197,
                        "type": "template",
                        "url": "https://woodworthia.gumroad.com/l/gr-pmcmd",
                        "status": "active",
                    },
                ],
                "pricing_model": "one_time",
                "total_products": 10,
            },
            "myroofgenius": {
                "url": "https://myroofgenius.com",
                "products": [
                    {"name": "Starter", "price_monthly": 49, "price_annual": 588, "type": "subscription", "status": "ready", "features": ["1-3 users", "2-10 jobs/month", "Basic analysis"]},
                    {"name": "Professional", "price_monthly": 99, "price_annual": 1188, "type": "subscription", "status": "ready", "features": ["Up to 10 users", "10-30 jobs/month", "Advanced analytics"]},
                    {"name": "Enterprise", "price_monthly": 199, "price_annual": 2388, "type": "subscription", "status": "ready", "features": ["Unlimited users", "30+ jobs/month", "Full features"]},
                ],
                "pricing_model": "subscription",
                "payment_processor": "stripe",
                "current_mrr": 0,
                "active_subscribers": 0,
            },
            "brainstack_studio": {
                "url": "https://brainstackstudio.com",
                "products": [
                    {"name": "AI Playground", "price": 0, "type": "free", "status": "active", "features": ["Claude, GPT, Gemini access", "Local storage", "Basic highlighting"]},
                ],
                "pricing_model": "freemium",
                "monetization_status": "needs_setup",
                "notes": "Currently free - needs subscription model or upsell to Gumroad products",
            },
        },
        "revenue_summary": {
            "gumroad_lifetime": 0.0,
            "gumroad_real_sales": 0,
            "gumroad_test_sales": 0,
            "gumroad_test_revenue": 0.0,
            "mrg_mrr": 0.0,
            "mrg_active_subscribers_default_tenant": 0,
            "total_real_revenue": 0.0,
            "note": "Owner revenue only (Gumroad + MRG). Excludes Weathercraft ERP client operations and ERP invoice ledger.",
        },
    }

    try:
        pool = get_pool()
        mrg_default_tenant = os.getenv(
            "MRG_DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001"
        )

        gumroad = (
            await pool.fetchrow(
                """
                SELECT
                  COUNT(*) FILTER (WHERE NOT COALESCE(is_test, FALSE)) AS real_count,
                  COALESCE(SUM(price::numeric) FILTER (WHERE NOT COALESCE(is_test, FALSE)), 0) AS real_revenue,
                  COUNT(*) FILTER (WHERE COALESCE(is_test, FALSE)) AS test_count,
                  COALESCE(SUM(price::numeric) FILTER (WHERE COALESCE(is_test, FALSE)), 0) AS test_revenue
                FROM gumroad_sales
                WHERE lower(coalesce(metadata->>'refunded', 'false')) NOT IN ('true', '1')
                """
            )
            or {}
        )

        mrg = (
            await pool.fetchrow(
                """
                SELECT
                  COUNT(*) AS active_subscriptions,
                  COALESCE(SUM(
                    CASE
                      WHEN billing_cycle IN ('monthly', 'month') THEN amount
                      WHEN billing_cycle IN ('annual', 'yearly', 'year') THEN amount / 12
                      ELSE 0
                    END
                  ), 0) AS mrr
                FROM mrg_subscriptions
                WHERE tenant_id = $1
                  AND status = 'active'
                """,
                mrg_default_tenant,
            )
            or {}
        )

        gumroad_lifetime = float(gumroad.get("real_revenue") or 0)
        mrg_mrr = float(mrg.get("mrr") or 0)
        inventory["revenue_summary"].update(
            {
                "gumroad_lifetime": gumroad_lifetime,
                "gumroad_real_sales": int(gumroad.get("real_count") or 0),
                "gumroad_test_sales": int(gumroad.get("test_count") or 0),
                "gumroad_test_revenue": float(gumroad.get("test_revenue") or 0),
                "mrg_mrr": mrg_mrr,
                "mrg_active_subscribers_default_tenant": int(mrg.get("active_subscriptions") or 0),
                "total_real_revenue": gumroad_lifetime,
            }
        )
    except Exception as e:
        logger.warning(f"Failed computing live revenue summary for inventory/products: {e}")

    return inventory


@router.get("/inventory/revenue", tags=["Inventory"])
async def get_revenue_status():
    """
    Get real revenue status across all platforms.
    """
    try:
        pool = get_pool()
        mrg_default_tenant = os.getenv(
            "MRG_DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001"
        )

        gumroad = await pool.fetchrow(
            """
                SELECT
                    COUNT(*) as total_sales,
                    COALESCE(SUM(price::numeric), 0) as total_revenue,
                    MAX(sale_timestamp) as last_sale
                FROM gumroad_sales
                WHERE is_test = false OR is_test IS NULL
                """
        ) or {"total_sales": 0, "total_revenue": 0}

        mrg = await pool.fetchrow(
            """
                SELECT
                    COUNT(*) as active_subscriptions,
                    COALESCE(SUM(
                        CASE
                            WHEN billing_cycle = 'monthly' THEN amount
                            WHEN billing_cycle = 'annual' THEN amount / 12
                            ELSE 0
                        END
                    ), 0) as mrr
                FROM mrg_subscriptions
                WHERE tenant_id = $1
                  AND status = 'active'
                """,
            mrg_default_tenant,
        ) or {"active_subscriptions": 0, "mrr": 0}

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "real_revenue": {
                "gumroad": {
                    "total_sales": gumroad.get("total_sales", 0),
                    "total_revenue": float(gumroad.get("total_revenue", 0)),
                    "last_sale": str(gumroad.get("last_sale"))
                    if gumroad.get("last_sale")
                    else None,
                },
                "myroofgenius": {
                    "active_subscriptions": mrg.get("active_subscriptions", 0),
                    "mrr": float(mrg.get("mrr", 0)),
                },
                "total_lifetime_revenue": float(gumroad.get("total_revenue", 0)),
                "total_mrr": float(mrg.get("mrr", 0)),
            },
            "warning": "Weathercraft ERP customer/job/invoice data is client operations, not owner revenue.",
        }

    except Exception as e:
        logger.warning(f"Revenue status degraded: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "real_revenue": {
                "gumroad": {"total_sales": 0, "total_revenue": 0},
                "myroofgenius": {"active_subscriptions": 0, "mrr": 0},
            },
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Revenue Intelligence Endpoints
# ---------------------------------------------------------------------------


@router.get("/revenue/state", tags=["Revenue Intelligence"])
async def get_complete_business_state():
    """Get COMPLETE business state snapshot."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    state = await get_business_state()
    return state


@router.get("/revenue/live", tags=["Revenue Intelligence"])
async def get_live_revenue_data():
    """Get live revenue data across all platforms."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    return await get_revenue()


@router.get("/revenue/products", tags=["Revenue Intelligence"])
async def get_all_products_inventory():
    """Get complete product inventory across all platforms."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    system = get_revenue_system()
    return {
        "products": system.get_all_products(),
        "social": system.get_social_presence(),
        "websites": system.get_websites(),
    }


@router.get("/revenue/automations", tags=["Revenue Intelligence"])
async def get_automation_health():
    """Get status of all revenue-related automations."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    system = get_revenue_system()
    return await system.get_automation_status()


@router.post("/revenue/sync-brain", tags=["Revenue Intelligence"])
async def sync_business_state_to_brain():
    """Sync complete business state to AI brain."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    return await sync_to_brain()


@router.post("/revenue/event", tags=["Revenue Intelligence"])
async def record_revenue_event(
    event_type: str = Body(..., embed=True),
    platform: str = Body(..., embed=True),
    amount: float = Body(0, embed=True),
    metadata: dict = Body(None, embed=True),
):
    """Record a revenue event for tracking."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")
    system = get_revenue_system()
    event_id = await system.record_revenue_event(event_type, platform, amount, metadata)
    return {"status": "recorded", "event_id": event_id}


# ---------------------------------------------------------------------------
# Observability Endpoints
# ---------------------------------------------------------------------------


@router.get("/logs/recent")
async def get_recent_logs(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    contains: Optional[str] = None,
):
    """Get recent logs with filtering"""
    logs = list(LOG_BUFFER)

    if level:
        logs = [l for l in logs if l["level"] == level.upper()]

    if logger_name:
        logs = [l for l in logs if logger_name.lower() in l["logger"].lower()]

    if contains:
        logs = [l for l in logs if contains.lower() in l["message"].lower()]

    return {
        "logs": logs[-limit:],
        "total_in_buffer": len(LOG_BUFFER),
        "returned": min(limit, len(logs)),
        "filters": {"level": level, "logger": logger_name, "contains": contains},
    }


@router.get("/observability/full")
async def get_full_observability():
    """Complete unified observability across ALL systems"""
    import httpx

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "services": {},
        "database": {},
        "recent_errors": [],
        "system_metrics": {},
    }

    services = {
        "ai_agents": "https://brainops-ai-agents.onrender.com/health",
        "backend": "https://brainops-backend-prod.onrender.com/health",
        "mcp_bridge": "https://brainops-mcp-bridge.onrender.com/health",
        "myroofgenius": "https://myroofgenius.com",
        "weathercraft_erp": "https://weathercraft-erp.vercel.app",
        "brainstackstudio": "https://brainstackstudio.com",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services.items():
            try:
                resp = await client.get(url)
                results["services"][name] = {
                    "status": "healthy" if resp.status_code == 200 else "degraded",
                    "code": resp.status_code,
                    "data": resp.json()
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else None,
                }
            except Exception as e:
                results["services"][name] = {"status": "error", "error": str(e)}

    try:
        pool = get_pool()
        db_stats = await pool.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM customers) as customers,
                (SELECT COUNT(*) FROM jobs) as jobs,
                (SELECT COUNT(*) FROM ai_agents) as agents,
                (SELECT COUNT(*) FROM ai_agent_executions) as executions,
                (SELECT COUNT(*) FROM ai_agent_executions WHERE created_at > NOW() - INTERVAL '1 hour') as recent_executions,
                (SELECT COUNT(*) FROM ai_agent_executions WHERE status = 'failed' AND created_at > NOW() - INTERVAL '1 hour') as recent_failures
        """
        )
        results["database"] = dict(db_stats) if db_stats else {}
    except Exception as e:
        results["database"] = {"error": str(e)}

    results["recent_errors"] = [
        l for l in list(LOG_BUFFER)[-100:] if l["level"] in ["ERROR", "CRITICAL"]
    ][-20:]

    try:
        import psutil

        results["system_metrics"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
        }
    except ImportError:
        results["system_metrics"] = {"note": "psutil not available"}

    return results


@router.get("/debug/all-errors")
async def get_all_errors():
    """Get all recent errors across systems for debugging"""
    errors = [l for l in list(LOG_BUFFER) if l["level"] in ["ERROR", "CRITICAL", "WARNING"]]

    categorized = {"database": [], "connection": [], "schema": [], "api": [], "other": []}

    for err in errors:
        msg = err["message"].lower()
        if "database" in msg or "sql" in msg or "column" in msg or "relation" in msg:
            categorized["database"].append(err)
        elif "connection" in msg or "pool" in msg or "timeout" in msg:
            categorized["connection"].append(err)
        elif "does not exist" in msg or "schema" in msg:
            categorized["schema"].append(err)
        elif "api" in msg or "http" in msg or "request" in msg:
            categorized["api"].append(err)
        else:
            categorized["other"].append(err)

    return {
        "total_errors": len(errors),
        "categorized": {k: len(v) for k, v in categorized.items()},
        "recent_errors": errors[-50:],
        "by_category": categorized,
    }


@router.get("/system/unified-status")
async def get_unified_system_status(request: Request):
    """Get unified status of the entire AI OS"""
    pool = get_pool()
    tenant_uuid = resolve_tenant_uuid_from_request(request)

    status = {
        "version": _version(),
        "timestamp": datetime.utcnow().isoformat(),
        "overall_health": "healthy",
        "tenant_id": tenant_uuid,
        "components": {},
    }

    components = [
        ("database", "SELECT 1"),
        ("agents", "SELECT COUNT(*) FROM ai_agents"),
        (
            "executions",
            "SELECT COUNT(*) FROM ai_agent_executions WHERE created_at > NOW() - INTERVAL '24 hours'",
        ),
        ("memory", "SELECT COUNT(*) FROM unified_brain"),
        ("revenue", "SELECT COUNT(*) FROM revenue_leads"),
    ]

    issues = []
    for name, query in components:
        try:
            result = await fetchval_with_tenant_context(
                pool,
                query,
                tenant_uuid=tenant_uuid,
            )
            status["components"][name] = {"status": "ok", "value": result}
        except Exception as e:
            status["components"][name] = {"status": "error", "error": str(e)}
            issues.append(f"{name}: {str(e)}")

    if issues:
        status["overall_health"] = "degraded"
        status["issues"] = issues

    return status
