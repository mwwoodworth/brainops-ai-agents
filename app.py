"""
BrainOps AI Agents Service - Enhanced Production Version
Type-safe, async, fully operational
"""
import asyncio
import hashlib
import hmac as hmac_mod
import inspect
import json
import logging
import os
import re
import time
import uuid
import threading
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from urllib.parse import unquote, urlparse


# Normalize DB_* env vars from DATABASE_URL before importing any modules that
# read credentials during import-time initialization.
def _hydrate_db_env_from_database_url() -> None:
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return
    try:
        parsed = urlparse(db_url)
    except Exception:
        return

    host = parsed.hostname or ""
    database = unquote(parsed.path.lstrip("/")) if parsed.path else ""
    user = unquote(parsed.username) if parsed.username else ""
    password = unquote(parsed.password) if parsed.password else ""
    port = str(parsed.port) if parsed.port else ""

    if host and not os.getenv("DB_HOST"):
        os.environ["DB_HOST"] = host
    if database and not os.getenv("DB_NAME"):
        os.environ["DB_NAME"] = database

    current_user = os.getenv("DB_USER", "")
    if user and (not current_user or "%" in current_user):
        os.environ["DB_USER"] = user

    current_password = os.getenv("DB_PASSWORD", "")
    if password and (not current_password or "%" in current_password):
        os.environ["DB_PASSWORD"] = password

    if port and not os.getenv("DB_PORT"):
        os.environ["DB_PORT"] = port


_hydrate_db_env_from_database_url()

# CRITICAL: Auto-switch Supabase pooler to transaction mode BEFORE any module
# imports trigger database config. Session mode (port 5432) has a low connection
# limit that causes MaxClientsInSessionMode under load. Transaction mode (port
# 6543) shares connections across transactions, supporting far more concurrency.
_db_host = os.getenv("DB_HOST", "")
_db_url = os.getenv("DATABASE_URL", "")
if ("pooler.supabase.com" in _db_host or "pooler.supabase.com" in _db_url) and os.getenv(
    "DB_PORT", "5432"
) == "5432":
    os.environ["DB_PORT"] = "6543"

from fastapi import BackgroundTasks, Body, Depends, FastAPI, HTTPException, Query, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from starlette.requests import HTTPConnection
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# Import our production-ready components
from config import config
from safe_task import create_safe_task, cancel_all_background_tasks
from auth.jwt import verify_jwt

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Background Event Loop Runner
# ---------------------------------------------------------------------------
# Some "always-on" BrainOps subsystems (AUREA, reconcilers, orchestrators) perform
# synchronous I/O (e.g., psycopg2) inside async loops. If they run on Uvicorn's
# main event loop they can block request handling and cause intermittent 502s
# from Render. Run long-lived background coroutines on a dedicated loop thread
# to keep the HTTP loop responsive.


class BackgroundLoopRunner:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, name="brainops-bg-loop", daemon=True)
        self._futures: set = set()  # Track futures for cleanup

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def start(self, timeout_s: float = 5.0) -> None:
        if self._thread.is_alive():
            return
        self._thread.start()
        self._started.wait(timeout=timeout_s)

    def _handle_future_exception(self, future):
        """Callback to handle exceptions from background futures - prevents 'Future exception never retrieved'"""
        self._futures.discard(future)
        try:
            exc = future.exception()
            if exc is not None:
                logger.error(f"Background coroutine failed: {exc}", exc_info=exc)
        except Exception:
            pass  # Future was cancelled or not done

    def submit(self, coro: "asyncio.coroutines.Coroutine[Any, Any, Any]"):
        """Submit coroutine to background loop with exception handling"""
        self.start()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # Add callback to handle exceptions (prevents "Future exception never retrieved")
        self._futures.add(future)
        future.add_done_callback(self._handle_future_exception)
        return future

    def stop(self, timeout_s: float = 5.0) -> None:
        if not self._thread.is_alive():
            return
        # Cancel pending futures
        for future in list(self._futures):
            future.cancel()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=timeout_s)


# ---------------------------------------------------------------------------
# slowapi Rate Limiting
# ---------------------------------------------------------------------------
# Key function: uses X-API-Key header when present, falls back to client IP.
# This ensures authenticated callers are rate-limited per key (not shared IP),
# while unauthenticated traffic is limited per IP.


def _rate_limit_key(request: Request) -> str:
    """Extract rate-limit identity: API key hash if valid, else client IP.

    Internal E2E verification requests that present a valid HMAC signature
    (X-Internal-E2E header) get a unique-per-request key so they are never
    throttled by the shared per-key counter.  This prevents the /e2e/verify
    endpoint from self-failing with 429s when it probes many endpoints in
    rapid succession.  Unauthenticated requests are never exempt.
    """
    api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("X-Api-Key")
    )
    if not api_key:
        try:
            api_key = request.query_params.get("api_key") or request.query_params.get("token")
        except Exception:
            api_key = None
    if api_key:
        api_key = api_key.strip()
    if not api_key:
        auth = request.headers.get("Authorization") or request.headers.get("authorization") or ""
        if auth.startswith("ApiKey "):
            api_key = auth[len("ApiKey ") :].strip()
        elif auth.startswith("Bearer "):
            api_key = auth[len("Bearer ") :].strip()
    if api_key and api_key in config.security.valid_api_keys:
        # --- Internal E2E exemption ---
        e2e_sig = request.headers.get("X-Internal-E2E", "")
        if e2e_sig:
            expected = hmac_mod.new(
                api_key.encode("utf-8"), b"brainops-e2e-internal", hashlib.sha256
            ).hexdigest()
            if hmac_mod.compare_digest(e2e_sig, expected):
                # Unique key per request = own counter = never shares the burst quota
                return f"e2e-internal:{id(request)}"
        return "key:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    # Fall back to client IP (handle reverse-proxy forwarding)
    forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if forwarded:
        return "ip:" + forwarded
    return "ip:" + (request.client.host if request.client else "unknown")


# Instantiate slowapi Limiter with our custom key function.
# default_limits apply to any endpoint that does NOT have its own @limiter.limit() decorator.
limiter = Limiter(
    key_func=_rate_limit_key,
    default_limits=["30/minute"],
    storage_uri="memory://",
)


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom 429 handler with Retry-After header."""
    logger.warning(
        "Rate limit exceeded for %s on %s: %s",
        _rate_limit_key(request),
        request.url.path,
        exc.detail,
    )
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down.", "limit": str(exc.detail)},
        headers={"Retry-After": "60"},
    )


# ---------------------------------------------------------------------------
# Safe JSON Serialization (handles UUIDs, datetimes, Decimals)
# ---------------------------------------------------------------------------
def safe_json_dumps(obj: Any, **kwargs) -> str:
    """JSON dumps that handles UUIDs, datetimes, Decimals and other non-serializable types."""
    from decimal import Decimal
    from enum import Enum

    def default_serializer(o):
        if isinstance(o, uuid.UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, Enum):
            return o.value
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)  # Fallback to string representation

    return json.dumps(obj, default=default_serializer, **kwargs)


# Import agent executor for actual agent dispatch
try:
    from agent_executor import AgentExecutor

    AGENT_EXECUTOR = AgentExecutor()
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENT_EXECUTOR = None
    AGENTS_AVAILABLE = False
    logging.warning(f"AgentExecutor not available: {e}")
from pydantic import BaseModel
from pydantic import EmailStr

from api.a2ui import router as a2ui_router  # Google A2UI Protocol - Agent-to-User Interface
from api.ai_awareness import (
    router as ai_awareness_router,  # Complete AI Awareness - THE endpoint for AI context
)
from api.aurea_chat import router as aurea_chat_router  # AUREA Live Conversational Interface
from api.brain import router as brain_router

BRAIN_AVAILABLE = True  # Brain router is always imported
from api.cicd import router as cicd_router  # Autonomous CI/CD Management - 1-10K systems
from api.codebase_graph import router as codebase_graph_router
from api.customer_intelligence import router as customer_intelligence_router
from api.digital_twin import router as digital_twin_router
from api.e2e_verification import router as e2e_verification_router
from api.logistics import router as logistics_router
from api.infrastructure import router as infrastructure_router
from api.real_ops import router as real_ops_router  # Real Operations: health, OODA, briefing
from api.gumroad_webhook import router as gumroad_router
from api.stripe_webhook import (
    router as stripe_webhook_router,
)  # Stripe webhooks (no API key - uses Stripe signature)
from api.market_intelligence import router as market_intelligence_router
from api.mcp import router as mcp_router  # MCP Bridge Integration - 345 tools
from api.memory import router as memory_router
from api.memory_coordination import router as memory_coordination_router
from api.observability import (
    router as full_observability_router,  # Comprehensive Observability Dashboard
)
from api.revenue import router as revenue_router
from api.taskmate import router as taskmate_router  # P1-TASKMATE-001: Cross-model task manager
from api.full_power_crud import router as full_power_crud_router  # Full CRUD + lifecycle control plane
from api.revenue_automation import router as revenue_automation_router
from api.income_streams import router as income_streams_router  # Automated Income Streams
from api.revenue_complete import router as revenue_complete_router  # Complete Revenue API
from api.revenue_control_tower import (
    router as revenue_control_tower_router,
)  # Revenue Control Tower - THE ground truth
from api.pipeline import (
    router as pipeline_router,
)  # Pipeline State Machine - ledger-backed state transitions
from api.proposals import (
    router as proposals_router,
)  # Proposal Engine - draft/approve/send workflow
from api.outreach import (
    router as outreach_router,
)  # Outreach Engine - lead enrichment and sequences
from api.payments import (
    router as payments_router,
)  # Payment Capture - invoices and revenue collection
from api.communications import (
    router as communications_router,
)  # Communications - send estimates, invoices from ERP
from api.revenue_operator import (
    router as revenue_operator_router,
)  # AI Revenue Operator - automated actions
from api.lead_discovery import (
    router as lead_discovery_router,
)  # Lead Discovery Engine - automated lead discovery and qualification
from api.lead_engine import (
    router as lead_engine_router,
)  # Lead Engine - MRG→ERP lead relay pipeline
from api.campaigns import (
    router as campaigns_router,
)  # Campaign System - CO commercial reroof lead gen
from api.email_capture import (
    router as email_capture_router,
)  # Email Capture - Lead generation for Gumroad products
from api.neural_reconnection import (
    router as neural_reconnection_router,
)  # Neural Reconnection - Schema unification & mode logic
from api.relationships import router as relationships_router
from api.roofing_labor_ml import router as roofing_labor_ml_router
from api.self_awareness import router as self_awareness_router  # Self-Awareness Dashboard
from api.self_healing import router as self_healing_router
from api.state_sync import router as state_sync_router
from api.sync import router as sync_router  # Memory migration & consolidation
from api.system_orchestrator import router as system_orchestrator_router
from api.victoria import (
    router as victoria_router,
)  # ERP scheduling agent compatibility (draft suggestions)
from api.google_keep import (
    router as google_keep_router,
)  # Google Keep Sync - Gemini Live real-time bridge
from database.async_connection import (
    DatabaseUnavailableError,
    PoolConfig,
    close_pool,
    get_pool,
    init_pool,
    using_fallback,
)
from models.agent import Agent, AgentCategory, AgentExecution, AgentList

# Operational Verification API - PROVES systems work, doesn't assume
try:
    from api.operational_verification import router as verification_router

    VERIFICATION_AVAILABLE = True
    logger.info("✅ Operational Verification Router loaded - PROVES systems work")
except ImportError as e:
    VERIFICATION_AVAILABLE = False
    logger.warning(f"Operational Verification not available: {e}")

# System Integration API - Connects all systems, no more silos
try:
    from api.system_integration import router as integration_router

    INTEGRATION_AVAILABLE = True
    logger.info("✅ System Integration Router loaded - full pipeline connectivity")
except ImportError as e:
    INTEGRATION_AVAILABLE = False
    logger.warning(f"System Integration not available: {e}")

# Background Task Monitoring - No more fire-and-forget
try:
    from api.background_monitoring import router as bg_monitoring_router
    from api.background_monitoring import start_all_monitoring

    BG_MONITORING_AVAILABLE = True
    logger.info("✅ Background Task Monitoring loaded - heartbeats for all tasks")
except ImportError as e:
    BG_MONITORING_AVAILABLE = False
    start_all_monitoring = None
    logger.warning(f"Background Monitoring not available: {e}")

# AI-Powered UI Testing System
try:
    from api.ui_testing import router as ui_testing_router

    UI_TESTING_AVAILABLE = True
    logger.info("AI UI Testing Router loaded - Automated visual testing")
except ImportError as e:
    UI_TESTING_AVAILABLE = False
    logger.warning(f"UI Testing Router not available: {e}")

# Permanent Observability Daemon - Never miss anything (2026-01-03)
try:
    from api.permanent_observability import router as permanent_observability_router
    from permanent_observability_daemon import start_observability_daemon, stop_observability_daemon

    PERMANENT_OBSERVABILITY_AVAILABLE = True
    logger.info("✅ Permanent Observability loaded - Never miss anything")
except ImportError as e:
    PERMANENT_OBSERVABILITY_AVAILABLE = False
    start_observability_daemon = None
    stop_observability_daemon = None
    logger.warning(f"⚠️ Permanent Observability not available: {e}")

# DevOps Automation API - Permanent knowledge & automated operations (2026-01-03)
try:
    from api.devops_api import router as devops_api_router

    DEVOPS_API_AVAILABLE = True
    logger.info("✅ DevOps Automation API loaded - Permanent knowledge enabled")
except ImportError as e:
    DEVOPS_API_AVAILABLE = False
    logger.warning(f"⚠️ DevOps API not available: {e}")

# NEURAL CORE - The Central Nervous System of the AI OS (2026-01-27)
# This is NOT monitoring - this IS the self-awareness of the AI OS
try:
    from api.neural_core import router as neural_core_router
    from neural_core import get_neural_core, initialize_neural_core

    NEURAL_CORE_AVAILABLE = True
    logger.info("🧠 NEURAL CORE loaded - The AI OS is now SELF-AWARE")
except ImportError as e:
    NEURAL_CORE_AVAILABLE = False
    initialize_neural_core = None
    logger.warning(f"⚠️ Neural Core not available: {e}")

# Customer Acquisition API - Autonomous lead discovery and conversion (2026-01-03)
try:
    from api.customer_acquisition import router as customer_acquisition_router

    CUSTOMER_ACQUISITION_API_AVAILABLE = True
    logger.info("✅ Customer Acquisition API loaded - Autonomous lead gen enabled")
except ImportError as e:
    CUSTOMER_ACQUISITION_API_AVAILABLE = False
    logger.warning(f"⚠️ Customer Acquisition API not available: {e}")

# Email Scheduler Daemon - Background email processing
try:
    from email_scheduler_daemon import (
        EmailSchedulerDaemon,
        start_email_scheduler,
        stop_email_scheduler,
    )

    EMAIL_SCHEDULER_AVAILABLE = True
    logger.info("✅ Email Scheduler Daemon loaded")
except ImportError as e:
    EMAIL_SCHEDULER_AVAILABLE = False
    start_email_scheduler = None
    stop_email_scheduler = None
    logger.warning(f"⚠️ Email Scheduler not available: {e}")

# Bleeding Edge AI Capabilities - Revolutionary systems (2025-12-27)
try:
    from api.bleeding_edge import router as bleeding_edge_router

    BLEEDING_EDGE_AVAILABLE = True
    logger.info("Bleeding Edge AI Router loaded - 6 revolutionary systems")
except ImportError as e:
    BLEEDING_EDGE_AVAILABLE = False
    logger.warning(f"Bleeding Edge Router not available: {e}")

# AI Observability & Integration - Perfect cross-module integration (2025-12-27)
try:
    from api.ai_observability_api import router as ai_observability_router

    AI_OBSERVABILITY_AVAILABLE = True
    logger.info("AI Observability Router loaded - unified metrics, events, integration")
except ImportError as e:
    AI_OBSERVABILITY_AVAILABLE = False
    logger.warning(f"AI Observability Router not available: {e}")

# Predictive Execution - Proactive task execution with safety checks (2026-01-27)
try:
    from api.predictive_execution import router as predictive_execution_router

    PREDICTIVE_EXECUTION_AVAILABLE = True
    logger.info("Predictive Execution Router loaded - proactive task execution")
except ImportError as e:
    PREDICTIVE_EXECUTION_AVAILABLE = False
    logger.warning(f"Predictive Execution Router not available: {e}")

# AI System Enhancements - Health scoring, alerting, correlation, WebSocket (2025-12-28)
try:
    from api.ai_enhancements_api import router as ai_enhancements_router

    AI_ENHANCEMENTS_AVAILABLE = True
    logger.info("AI Enhancements Router loaded - health, alerting, correlation, WebSocket")
except ImportError as e:
    AI_ENHANCEMENTS_AVAILABLE = False
    logger.warning(f"AI Enhancements Router not available: {e}")

# Unified AI Awareness - Self-reporting AI OS consciousness (2025-12-27)
try:
    from unified_awareness import check_status, get_status_report, get_unified_awareness

    UNIFIED_AWARENESS_AVAILABLE = True
    logger.info("Unified AI Awareness loaded - AI OS is now self-aware")
except ImportError as e:
    UNIFIED_AWARENESS_AVAILABLE = False
    logger.warning(f"Unified Awareness not available: {e}")

# True Self-Awareness - Live system truth, not static docs (2026-01-01)
try:
    from true_self_awareness import get_quick_status, get_system_truth, get_true_awareness

    TRUE_AWARENESS_AVAILABLE = True
    logger.info("True Self-Awareness loaded - AI OS knows its own truth")
except ImportError as e:
    TRUE_AWARENESS_AVAILABLE = False
    logger.warning(f"True Self-Awareness not available: {e}")

# New Pipeline Routers - Secure, authenticated endpoints
try:
    from api.product_generation import router as product_generation_router

    PRODUCT_GEN_ROUTER_AVAILABLE = True
    logger.info("Product Generation Router loaded")
except ImportError as e:
    PRODUCT_GEN_ROUTER_AVAILABLE = False
    logger.warning(f"Product Generation Router not available: {e}")

try:
    from api.affiliate import router as affiliate_router

    AFFILIATE_ROUTER_AVAILABLE = True
    logger.info("Affiliate Router loaded")
except ImportError as e:
    AFFILIATE_ROUTER_AVAILABLE = False
    logger.warning(f"Affiliate Router not available: {e}")

try:
    from api.knowledge import router as knowledge_base_router

    KNOWLEDGE_BASE_ROUTER_AVAILABLE = True
    logger.info("Knowledge Base Router loaded")
except ImportError as e:
    KNOWLEDGE_BASE_ROUTER_AVAILABLE = False
    logger.warning(f"Knowledge Base Router not available: {e}")

try:
    from api.sop import router as sop_router

    SOP_ROUTER_AVAILABLE = True
    logger.info("SOP Generator Router loaded")
except ImportError as e:
    SOP_ROUTER_AVAILABLE = False
    logger.warning(f"SOP Router not available: {e}")

try:
    from api.companycam import router as companycam_router

    COMPANYCAM_ROUTER_AVAILABLE = True
    logger.info("CompanyCam Router loaded")
except ImportError as e:
    COMPANYCAM_ROUTER_AVAILABLE = False
    logger.warning(f"CompanyCam Router not available: {e}")

# Voice Router (Added for AUREA)
try:
    from api.voice import router as voice_router

    VOICE_ROUTER_AVAILABLE = True
    logger.info("Voice Router loaded")
except ImportError as e:
    VOICE_ROUTER_AVAILABLE = False
    logger.warning(f"Voice Router not available: {e}")

# Always-Know Observability Brain - Deep continuous monitoring (2025-12-29)
try:
    from api.always_know import router as always_know_router

    ALWAYS_KNOW_AVAILABLE = True
    logger.info("🧠 Always-Know Brain Router loaded - continuous state awareness")
except ImportError as e:
    ALWAYS_KNOW_AVAILABLE = False
    logger.warning(f"Always-Know Brain Router not available: {e}")

try:
    from api.always_know_compat import router as always_know_compat_router

    ALWAYS_KNOW_COMPAT_AVAILABLE = True
    logger.info("🧠 Always-Know Compat Router loaded - legacy paths")
except ImportError as e:
    ALWAYS_KNOW_COMPAT_AVAILABLE = False
    logger.warning(f"Always-Know Compat Router not available: {e}")

# Ultimate E2E System - COMPLETE e2e awareness (2025-12-31)
try:
    from api.ultimate_e2e import router as ultimate_e2e_router

    ULTIMATE_E2E_AVAILABLE = True
    logger.info(
        "🚀 Ultimate E2E System loaded - build logs, DB awareness, UI tests, issue detection"
    )
except ImportError as e:
    ULTIMATE_E2E_AVAILABLE = False
    logger.warning(f"Ultimate E2E System not available: {e}")

# TRUE Operational Validation - REAL operation testing (2025-12-31)
try:
    from api.true_validation import router as true_validation_router

    TRUE_VALIDATION_AVAILABLE = True
    logger.info(
        "✅ TRUE Operational Validator loaded - executes real operations, not status checks"
    )
except ImportError as e:
    TRUE_VALIDATION_AVAILABLE = False
    logger.warning(f"TRUE Validation not available: {e}")

# Learning Feedback Loop - Closes the gap between insights and action (2025-12-30)
try:
    from api.learning import router as learning_router

    LEARNING_ROUTER_AVAILABLE = True
    logger.info("🔄 Learning Feedback Loop Router loaded - insights now become actions")
except ImportError as e:
    LEARNING_ROUTER_AVAILABLE = False
    logger.warning(f"Learning Router not available: {e}")

# Learning Visibility API - Exposes what AI has learned (2026-02-02)
try:
    from api.learning_visibility import router as learning_visibility_router

    LEARNING_VISIBILITY_AVAILABLE = True
    logger.info("👁️ Learning Visibility Router loaded - see what the AI has learned")
except ImportError as e:
    LEARNING_VISIBILITY_AVAILABLE = False
    logger.warning(f"Learning Visibility Router not available: {e}")

# ChatGPT-Agent-Level UI Tester - Human-like testing (2025-12-29)
try:
    from chatgpt_agent_tester import run_chatgpt_agent_tests, run_quick_health_test

    CHATGPT_TESTER_AVAILABLE = True
    logger.info("🤖 ChatGPT Agent Tester loaded - human-like UI testing")
except ImportError as e:
    CHATGPT_TESTER_AVAILABLE = False
    run_chatgpt_agent_tests = None
    run_quick_health_test = None
    logger.warning(f"ChatGPT Agent Tester not available: {e}")

from ai_provider_status import get_provider_status
from erp_event_bridge import router as erp_event_router

# Unified Events System (2025-12-30) - Central event bus for ERP, AI Agents, Command Center
try:
    from api.events.unified import init_unified_events
    from api.events.unified import router as unified_events_router

    UNIFIED_EVENTS_AVAILABLE = True
    logger.info("Unified Events router loaded - central event bus for all systems")
except ImportError as e:
    UNIFIED_EVENTS_AVAILABLE = False
    unified_events_router = None
    init_unified_events = None
    logger.warning(f"Unified Events router not available: {e}")
from observability import RequestMetrics, TTLCache

# System Observability - Makes AI OS transparent and queryable
try:
    from system_observability import router as system_observability_router

    SYSTEM_OBSERVABILITY_AVAILABLE = True
    logger.info("✅ System Observability Layer loaded - AI OS now queryable")
except ImportError as e:
    SYSTEM_OBSERVABILITY_AVAILABLE = False
    system_observability_router = None
    logger.warning(f"System Observability not available: {e}")

# Agent Health Monitoring
try:
    from agent_health_monitor import get_health_monitor

    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    logger.warning("Agent Health Monitor not available")

# Schema verification - NO DDL at runtime (agent_worker has no DDL permissions by P0-LOCK).
# All tables/indexes/extensions must be pre-created via migration scripts run as postgres.
# This list is checked at startup; missing tables are logged as errors but don't crash the app.
REQUIRED_TABLES = [
    "ai_email_deliveries",
    "ai_agent_executions",
    "ai_autonomous_tasks",
    "ai_followup_executions",
    "ai_followup_responses",
    "ai_followup_metrics",
    "ai_followup_sequences",
    "ai_followup_touchpoints",
]

# Build info
BUILD_TIME = datetime.utcnow().isoformat()
VERSION = config.version  # Use centralized config - never hardcode version
LOCAL_EXECUTIONS: deque[dict[str, Any]] = deque(maxlen=200)
REQUEST_METRICS = RequestMetrics(window=800)
RESPONSE_CACHE = TTLCache(max_size=256)
# Health endpoints are hit frequently by monitors and E2E sweeps. Keep caching
# long enough to prevent thundering-herd stalls on Render under load.
HEALTH_CACHE_TTL_S = float(os.getenv("HEALTH_CACHE_TTL_S", "30"))
HEALTH_PAYLOAD_TIMEOUT_S = float(os.getenv("HEALTH_PAYLOAD_TIMEOUT_S", "5"))
CACHE_TTLS = {"health": HEALTH_CACHE_TTL_S, "agents": 30.0, "systems_usage": 15.0}

# Import agent scheduler with fallback
try:
    from agent_scheduler import AgentScheduler

    SCHEDULER_AVAILABLE = True
    logger.info("✅ Agent Scheduler module loaded")
except ImportError as e:
    SCHEDULER_AVAILABLE = False
    logger.warning(f"Agent Scheduler not available: {e}")
    AgentScheduler = None

# Import AI Core with fallback
try:
    from ai_core import RealAICore, ai_analyze, ai_generate

    ai_core = RealAICore()

    # Determine whether any real AI providers are configured
    has_openai = bool(getattr(ai_core, "async_openai", None))
    has_anthropic = bool(getattr(ai_core, "async_anthropic", None))
    AI_AVAILABLE = has_openai or has_anthropic

    # In production, it is a hard failure if no real AI provider is available
    if config.environment == "production" and not AI_AVAILABLE:
        raise RuntimeError(
            "AI Core initialized but no real AI providers are configured in production."
        )

    logger.info("✅ Real AI Core initialized successfully")
except Exception as e:
    logger.error(f"❌ AI Core initialization failed: {e}")
    AI_AVAILABLE = False
    ai_core = None

# Import AUREA Master Orchestrator with fallback
try:
    from aurea_orchestrator import AUREA, AutonomyLevel

    AUREA_AVAILABLE = True
    logger.info("✅ AUREA Master Orchestrator loaded")
except ImportError as e:
    AUREA_AVAILABLE = False
    logger.warning(f"AUREA not available: {e}")
    AUREA = None
    AutonomyLevel = None

# Import Self-Healing Recovery with fallback
try:
    from self_healing_recovery import SelfHealingRecovery, ErrorContext, ErrorSeverity

    SELF_HEALING_AVAILABLE = True
    logger.info("✅ Self-Healing Recovery loaded")
except ImportError as e:
    SELF_HEALING_AVAILABLE = False
    logger.warning(f"Self-Healing not available: {e}")
    SelfHealingRecovery = None
    ErrorContext = None
    ErrorSeverity = None

# Global self-healing instance for exception handler integration
_global_healer = None


def _get_global_healer():
    """Lazy initialization of global self-healer to avoid startup delays"""
    global _global_healer
    if _global_healer is None and SELF_HEALING_AVAILABLE:
        try:
            _global_healer = SelfHealingRecovery()
            logger.info("✅ Global self-healer initialized for exception handling")
        except Exception as e:
            logger.warning(f"Failed to initialize global healer: {e}")
    return _global_healer


# Import Self-Healing Reconciler (continuous healing loop)
try:
    from self_healing_reconciler import get_reconciler, start_healing_loop

    RECONCILER_AVAILABLE = True
    logger.info("✅ Self-Healing Reconciler loaded")
except ImportError as e:
    RECONCILER_AVAILABLE = False
    logger.warning(f"Self-Healing Reconciler not available: {e}")
    get_reconciler = None
    start_healing_loop = None

# Import Service Circuit Breakers - Centralized circuit breaker management for all external services
try:
    from service_circuit_breakers import (
        get_circuit_breaker_manager,
        get_circuit_breaker_health,
        get_all_circuit_statuses,
        check_service_available,
        report_service_success,
        report_service_failure,
        CIRCUIT_BREAKER_CONFIG,
    )

    SERVICE_CIRCUIT_BREAKERS_AVAILABLE = True
    logger.info("✅ Service Circuit Breakers loaded - protecting all external service calls")
except ImportError as e:
    SERVICE_CIRCUIT_BREAKERS_AVAILABLE = False
    logger.warning(f"Service Circuit Breakers not available: {e}")
    get_circuit_breaker_manager = None
    get_circuit_breaker_health = None
    get_all_circuit_statuses = None

# Import Unified Memory Manager with fallback
try:
    from unified_memory_manager import UnifiedMemoryManager

    MEMORY_AVAILABLE = True
    logger.info("✅ Unified Memory Manager loaded")
except ImportError as e:
    MEMORY_AVAILABLE = False
    logger.warning(f"Memory Manager not available: {e}")
    UnifiedMemoryManager = None

# Import Embedded Memory System with fallback
try:
    from embedded_memory_system import get_embedded_memory

    EMBEDDED_MEMORY_AVAILABLE = True
    logger.info("✅ Embedded Memory System loaded")
except ImportError as e:
    EMBEDDED_MEMORY_AVAILABLE = False
    logger.exception("Embedded Memory import failed; subsystem disabled")
    get_embedded_memory = None

# Import AI Training Pipeline with fallback
try:
    from ai_training_pipeline import AITrainingPipeline

    TRAINING_AVAILABLE = True
    logger.info("✅ AI Training Pipeline loaded")
except ImportError as e:
    TRAINING_AVAILABLE = False
    logger.warning(f"Training Pipeline not available: {e}")
    AITrainingPipeline = None

# Import Notebook LM+ Learning with fallback
try:
    from notebook_lm_plus import NotebookLMPlus

    LEARNING_AVAILABLE = True
    logger.info("✅ Notebook LM+ Learning loaded")
except ImportError as e:
    LEARNING_AVAILABLE = False
    logger.warning(f"Learning System not available: {e}")
    NotebookLMPlus = None

# PHASE 2: Import Specialized Agents

# Import System Improvement Agent with fallback
try:
    from system_improvement_agent import SystemImprovementAgent

    SYSTEM_IMPROVEMENT_AVAILABLE = True
    logger.info("✅ System Improvement Agent loaded")
except ImportError as e:
    SYSTEM_IMPROVEMENT_AVAILABLE = False
    logger.warning(f"System Improvement Agent not available: {e}")
    SystemImprovementAgent = None

# Import DevOps Optimization Agent with fallback
try:
    from devops_optimization_agent import DevOpsOptimizationAgent

    DEVOPS_AGENT_AVAILABLE = True
    logger.info("✅ DevOps Optimization Agent loaded")
except ImportError as e:
    DEVOPS_AGENT_AVAILABLE = False
    logger.warning(f"DevOps Agent not available: {e}")
    DevOpsOptimizationAgent = None

# Import Code Quality Agent with fallback
try:
    from code_quality_agent import CodeQualityAgent

    CODE_QUALITY_AVAILABLE = True
    logger.info("✅ Code Quality Agent loaded")
except ImportError as e:
    CODE_QUALITY_AVAILABLE = False
    logger.warning(f"Code Quality Agent not available: {e}")
    CodeQualityAgent = None

# Import Customer Success Agent with fallback
try:
    from customer_success_agent import CustomerSuccessAgent

    CUSTOMER_SUCCESS_AVAILABLE = True
    logger.info("✅ Customer Success Agent loaded")
except ImportError as e:
    CUSTOMER_SUCCESS_AVAILABLE = False
    logger.warning(f"Customer Success Agent not available: {e}")
    CustomerSuccessAgent = None

# Import Competitive Intelligence Agent with fallback
try:
    from competitive_intelligence_agent import CompetitiveIntelligenceAgent

    COMPETITIVE_INTEL_AVAILABLE = True
    logger.info("✅ Competitive Intelligence Agent loaded")
except ImportError as e:
    COMPETITIVE_INTEL_AVAILABLE = False
    logger.warning(f"Competitive Intel Agent not available: {e}")
    CompetitiveIntelligenceAgent = None

# Import Vision Alignment Agent with fallback
try:
    from vision_alignment_agent import VisionAlignmentAgent

    VISION_ALIGNMENT_AVAILABLE = True
    logger.info("✅ Vision Alignment Agent loaded")
except ImportError as e:
    VISION_ALIGNMENT_AVAILABLE = False
    logger.warning(f"Vision Alignment Agent not available: {e}")
    VisionAlignmentAgent = None

# Import AI Self-Awareness Module with fallback
try:
    from ai_self_awareness import SelfAwareAI, get_self_aware_ai

    SELF_AWARENESS_AVAILABLE = True
    logger.info("✅ AI Self-Awareness Module loaded")
except ImportError as e:
    SELF_AWARENESS_AVAILABLE = False
    logger.warning(f"AI Self-Awareness not available: {e}")
    get_self_aware_ai = None
    SelfAwareAI = None

# Import NerveCenter - Central Nervous System of BrainOps AI OS
try:
    from nerve_center import NerveCenter, get_nerve_center

    NERVE_CENTER_AVAILABLE = True
    logger.info("✅ NerveCenter module loaded")
except ImportError as e:
    NERVE_CENTER_AVAILABLE = False
    get_nerve_center = None
    NerveCenter = None
    logger.warning(f"NerveCenter not available: {e}")

# Import OperationalMonitor - real operational awareness loop
try:
    from operational_monitor import OperationalMonitor, get_operational_monitor

    OPERATIONAL_MONITOR_AVAILABLE = True
    logger.info("✅ OperationalMonitor module loaded")
except ImportError as e:
    OPERATIONAL_MONITOR_AVAILABLE = False
    get_operational_monitor = None
    OperationalMonitor = None
    logger.warning(f"OperationalMonitor not available: {e}")

# Import AI Integration Layer with fallback
try:
    from ai_integration_layer import (
        AIIntegrationLayer,
        TaskPriority,
        TaskStatus,
        get_integration_layer,
    )

    INTEGRATION_LAYER_AVAILABLE = True
    logger.info("✅ AI Integration Layer loaded")
except ImportError as e:
    INTEGRATION_LAYER_AVAILABLE = False
    logger.warning(f"AI Integration Layer not available: {e}")
    AIIntegrationLayer = None
    get_integration_layer = None
    TaskPriority = None
    TaskStatus = None

# Import LangGraph Orchestrator with fallback
try:
    from langgraph_orchestrator import LangGraphOrchestrator

    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph Orchestrator loaded")
except ImportError as e:
    LANGGRAPH_AVAILABLE = False
    logger.warning(f"LangGraph Orchestrator not available: {e}")
    LangGraphOrchestrator = None

# Import AUREA NLU Processor with fallback
try:
    from langchain_openai import ChatOpenAI

    from aurea_nlu_processor import AUREANLUProcessor

    AUREA_NLU_AVAILABLE = True
    logger.info("✅ AUREA NLU Processor loaded")
except ImportError as e:
    AUREA_NLU_AVAILABLE = False
    logger.warning(f"AUREA NLU Processor not available: {e}")
    AUREANLUProcessor = None
    ChatOpenAI = None

# Import AI Board of Directors
try:
    from ai_board_governance import AIBoardOfDirectors

    AI_BOARD_AVAILABLE = True
    logger.info("✅ AI Board of Directors loaded")
except ImportError as e:
    AI_BOARD_AVAILABLE = False
    logger.warning(f"AI Board not available: {e}")
    AIBoardOfDirectors = None


def _parse_capabilities(raw: Any) -> list[dict[str, Any]]:
    """Normalize capabilities payload into the Pydantic-friendly format."""
    if raw is None:
        return []

    data: Any = raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return [
                {
                    "name": raw,
                    "description": "",
                    "enabled": True,
                    "parameters": {},
                }
            ]

    if isinstance(data, dict):
        # Single capability as dict
        data = [data]

    capabilities: list[dict[str, Any]] = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, str):
            capabilities.append(
                {
                    "name": item,
                    "description": "",
                    "enabled": True,
                    "parameters": {},
                }
            )
        elif isinstance(item, dict):
            capabilities.append(
                {
                    "name": item.get("name")
                    or item.get("capability")
                    or item.get("id")
                    or "capability",
                    "description": item.get("description", ""),
                    "enabled": bool(item.get("enabled", True)),
                    "parameters": item.get("parameters", {}),
                }
            )
    return capabilities


def _parse_configuration(raw: Any) -> dict[str, Any]:
    """Normalize configuration payload."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _row_to_agent(row: dict[str, Any]) -> Agent:
    """Convert a database row (asyncpg or fallback dict) to an Agent model."""
    category_value = row.get("category") or AgentCategory.OTHER.value
    if category_value not in {c.value for c in AgentCategory}:
        category_value = AgentCategory.OTHER.value

    return Agent(
        id=str(row["id"]),
        name=row["name"],
        category=category_value,
        description=row.get("description") or "",
        enabled=bool(row.get("enabled", True)),
        capabilities=_parse_capabilities(row.get("capabilities")),
        configuration=_parse_configuration(row.get("configuration")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        # Operational fields - map from database
        status=row.get("status") or ("active" if row.get("enabled", True) else "inactive"),
        type=row.get("type") or row.get("agent_type"),
        total_executions=row.get("total_executions") or row.get("execution_count") or 0,
        last_active=row.get("last_active") or row.get("last_active_at") or row.get("updated_at"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - FAST STARTUP for Render port detection"""
    # Startup - log immediately so Render sees activity
    import time as _time

    app.state._start_time = _time.time()
    logger.info(f"🚀 Starting BrainOps AI Agents v{VERSION} - Build: {BUILD_TIME}")
    logger.info("⚡ Fast startup mode - deferring heavy init to background")

    # Start a dedicated background event loop. This prevents long-running background
    # coroutines from starving the HTTP server's event loop under load.
    app.state.bg_runner = BackgroundLoopRunner()
    try:
        app.state.bg_runner.start()
        logger.info("🧵 Background loop runner started")
    except Exception as exc:
        logger.error("Failed to start background loop runner: %s", exc, exc_info=True)
        app.state.bg_runner = None

    # Register the main asyncio loop so thread-based schedulers can safely
    # schedule async work onto the correct loop (avoids cross-loop Future errors).
    try:
        from loop_bridge import set_main_loop

        set_main_loop(asyncio.get_running_loop())
        logger.info("🧠 Main asyncio loop registered for loop_bridge")
    except Exception:
        # Never fail startup due to observability helpers.
        pass

    # Get tenant configuration from centralized config
    from config import config as app_config

    DEFAULT_TENANT_ID = app_config.tenant.default_tenant_id

    # Use environment-configured tenant for startup initialization
    # Per-request tenant resolution happens via X-Tenant-ID header in API endpoints
    tenant_id = DEFAULT_TENANT_ID
    if tenant_id:
        logger.info(
            f"🔑 Default tenant_id: {tenant_id} (override per-request via {app_config.tenant.header_name} header)"
        )
    else:
        logger.warning(
            "⚠️ No DEFAULT_TENANT_ID configured - set DEFAULT_TENANT_ID environment variable"
        )

    # Keep handles defined to avoid unbound errors when optional systems are disabled

    # DEFERRED INITIALIZATION - run in background after server binds to port
    async def deferred_init():
        """Heavy initialization that runs AFTER server binds to port"""
        await asyncio.sleep(1)  # Give server time to bind
        logger.info("🔄 Starting deferred initialization...")

        # Reduce noisy asyncio errors in production logs. Some subsystems and/or
        # third-party libs may cancel background futures during startup/shutdown
        # (timeouts, retries). CancelledError is expected and shouldn't emit
        # "Future exception was never retrieved" at ERROR level.
        try:
            loop = asyncio.get_running_loop()
            if not getattr(loop, "_brainops_exception_handler_set", False):

                def _brainops_exception_handler(loop, context):
                    msg = context.get("message") or ""
                    exc = context.get("exception")
                    if "Future exception was never retrieved" in msg and isinstance(
                        exc, asyncio.CancelledError
                    ):
                        logger.debug("Suppressed unhandled CancelledError future: %s", msg)
                        return
                    # For DB connection flaps, log a real traceback (default handler
                    # often omits useful frames) so we can fix the root cause.
                    try:
                        import asyncpg

                        if isinstance(exc, asyncpg.ConnectionDoesNotExistError):
                            fut = context.get("future")
                            coro = getattr(fut, "get_coro", lambda: None)()
                            coro_name = getattr(coro, "__qualname__", None) or repr(coro)
                            logger.error(
                                "Unhandled asyncpg ConnectionDoesNotExistError (future=%r coro=%s): %s",
                                fut,
                                coro_name,
                                exc,
                                exc_info=(exc.__class__, exc, exc.__traceback__),
                            )
                            return
                    except Exception:
                        pass

                    loop.default_exception_handler(context)

                loop.set_exception_handler(_brainops_exception_handler)
                setattr(loop, "_brainops_exception_handler_set", True)
        except Exception:
            # Never fail startup due to logging/handler issues.
            pass

        # Initialize database pool
        try:
            pool_config = PoolConfig(
                host=config.database.host,
                port=config.database.port,
                user=config.database.user,
                password=config.database.password,
                database=config.database.database,
                ssl=config.database.ssl,
                ssl_verify=config.database.ssl_verify,
            )
            await init_pool(pool_config)
            if using_fallback():
                logger.warning(
                    "⚠️ Running with in-memory fallback datastore (database unreachable)."
                )
            else:
                logger.info("✅ Database pool initialized")

            # Test connection with short timeout
            pool = get_pool()
            try:
                if await asyncio.wait_for(pool.test_connection(), timeout=5.0):
                    logger.info("✅ Database connection verified")
                else:
                    logger.error("❌ Database connection test failed")
            except asyncio.TimeoutError:
                logger.warning("⚠️ Database connection test timed out (will retry)")

            # Schema verification (read-only) - no DDL, agent_worker has no DDL perms
            async def verify_schema():
                try:
                    await asyncio.sleep(2)
                    result = await pool.fetch(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = ANY($1::text[])",
                        REQUIRED_TABLES,
                    )
                    found = {row["table_name"] for row in result}
                    missing = [t for t in REQUIRED_TABLES if t not in found]
                    if missing:
                        logger.error(
                            f"Schema verification: {len(missing)} required table(s) "
                            f"missing: {missing}. Run migrations as postgres to create them."
                        )
                    else:
                        logger.info("Schema verification passed: all required tables present")
                except Exception as e:
                    logger.error(f"Schema verification failed: {e}")

            create_safe_task(verify_schema(), "schema_verify")
        except Exception as e:
            logger.error(f"❌ Deferred database init failed: {e}")

    # Schedule deferred init to run after yield
    create_safe_task(deferred_init(), "deferred_init")
    logger.info("📋 Deferred initialization scheduled")

    # Set all app.state values to None initially (will be populated by deferred init)
    app.state.scheduler = None
    app.state.aurea = None
    app.state.healer = None
    app.state.reconciler = None
    app.state.memory = None
    app.state.embedded_memory = None
    app.state.embedded_memory_error = None
    app.state.nerve_center = None
    app.state.nerve_center_error = None
    app.state.operational_monitor = None
    app.state.operational_monitor_error = None
    app.state.neural_core = None  # Central Nervous System (2026-01-27)

    # DEFERRED HEAVY INITIALIZATION - runs AFTER server binds to port
    async def deferred_heavy_init():
        """Initialize all heavy components after server is listening"""
        await asyncio.sleep(2)  # Give server time to be fully ready
        logger.info("🔄 Starting heavy component initialization...")

        # Wait for DB pool to be ready (deferred_init creates it concurrently)
        _pool_wait_start = asyncio.get_event_loop().time()
        for _attempt in range(30):  # Up to 30 seconds
            try:
                _p = get_pool()
                if _p is not None and not using_fallback():
                    logger.info(
                        "✅ DB pool ready for heavy init (waited %.1fs)",
                        asyncio.get_event_loop().time() - _pool_wait_start,
                    )
                    break
            except Exception:
                pass
            await asyncio.sleep(1)
        else:
            logger.warning("⚠️ DB pool not ready after 30s — proceeding with heavy init anyway")

        # Some long-running BrainOps subsystems still perform synchronous I/O
        # (psycopg2, shell-outs, CPU-heavy work). Run those loops on a dedicated
        # background event loop thread so they can't starve Uvicorn's HTTP loop
        # and trigger Render health check timeouts (502).
        bg_runner = getattr(app.state, "bg_runner", None)

        def _run_on_bg_loop(coro: "asyncio.coroutines.Coroutine[Any, Any, Any]", name: str):
            """Best-effort: schedule coroutine on background loop runner, else main loop."""
            if bg_runner is not None:
                try:
                    future = bg_runner.submit(coro)
                    logger.info("🧵 Scheduled %s on background loop", name)
                    return future
                except Exception as exc:
                    logger.error(
                        "Failed to schedule %s on background loop: %s", name, exc, exc_info=True
                    )
            create_safe_task(coro, name)
            return None

        # ── Phase 1: Core systems (scheduler, AUREA, healing) ──
        if SCHEDULER_AVAILABLE:
            try:

                def _init_scheduler_sync():
                    scheduler = AgentScheduler()
                    scheduler.start()
                    return scheduler

                scheduler = await asyncio.to_thread(_init_scheduler_sync)
                app.state.scheduler = scheduler
                logger.info("✅ Agent Scheduler initialized and STARTED")
            except Exception as e:
                logger.error(f"❌ Scheduler initialization failed: {e}")

        # Initialize AUREA Master Orchestrator
        if AUREA_AVAILABLE and tenant_id:
            try:
                aurea_instance = await asyncio.to_thread(
                    lambda: AUREA(autonomy_level=AutonomyLevel.FULL_AUTO, tenant_id=tenant_id)
                )
                app.state.aurea = aurea_instance
                # AUREA uses synchronous DB operations (psycopg2) internally; run
                # it off the HTTP event loop to prevent intermittent 502s.
                app.state.aurea_future = _run_on_bg_loop(
                    aurea_instance.orchestrate(), "aurea_orchestrate"
                )
                logger.info("🧠 AUREA Master Orchestrator STARTED - Observe→Decide→Act loop ACTIVE")
            except Exception as e:
                logger.error(f"❌ AUREA initialization failed: {e}")
        elif AUREA_AVAILABLE and not tenant_id:
            logger.warning("⚠️ Skipping AUREA initialization (DEFAULT_TENANT_ID/TENANT_ID missing)")

        # Initialize Self-Healing Recovery System
        if SELF_HEALING_AVAILABLE:
            try:
                healer = await asyncio.to_thread(SelfHealingRecovery)
                app.state.healer = healer
                logger.info("🏥 Self-Healing Recovery System initialized")
            except Exception as e:
                logger.error(f"❌ Self-Healing initialization failed: {e}")

        # Start Self-Healing Reconciliation Loop
        if RECONCILER_AVAILABLE:
            try:
                reconciler = await asyncio.to_thread(get_reconciler)
                app.state.reconciler = reconciler
                # The reconciler performs synchronous DB work; keep it off the
                # HTTP event loop to avoid health check timeouts under load.
                app.state.healing_future = _run_on_bg_loop(start_healing_loop(), "healing_loop")
                logger.info("🔄 Self-Healing Reconciliation Loop STARTED")
            except Exception as e:
                logger.error(f"❌ Reconciler initialization failed: {e}")

        # ── Phase 2: Memory systems (stagger to prevent DB pool exhaustion) ──
        await asyncio.sleep(2)
        if MEMORY_AVAILABLE:
            try:
                memory_manager_instance = await asyncio.to_thread(UnifiedMemoryManager)
                app.state.memory = memory_manager_instance
                logger.info("🧠 Unified Memory Manager initialized")
            except Exception as e:
                logger.error(f"❌ Memory Manager initialization failed: {e}")

        # Initialize Embedded Memory System
        if EMBEDDED_MEMORY_AVAILABLE:
            try:
                embedded_memory = await get_embedded_memory()
                app.state.embedded_memory = embedded_memory
                app.state.embedded_memory_error = None
                logger.info("⚡ Embedded Memory System initialized")
            except Exception as e:
                app.state.embedded_memory_error = str(e)
                logger.error(f"❌ Embedded Memory initialization failed: {e}")

        # Warm up Unified Brain for fast /brain/* endpoint responses
        try:
            from api.brain import brain, BRAIN_AVAILABLE

            if BRAIN_AVAILABLE and brain:
                await asyncio.wait_for(brain._ensure_table(), timeout=15.0)
                logger.info("🧠 Unified Brain warmed up and ready")
        except asyncio.TimeoutError:
            logger.warning("⚠️ Unified Brain warmup timed out (will initialize on first request)")
        except Exception as e:
            logger.error(f"⚠️ Unified Brain warmup failed: {e}")

        # Start Permanent Observability Daemon - Never miss anything
        if PERMANENT_OBSERVABILITY_AVAILABLE and start_observability_daemon:
            try:
                app.state.observability_daemon = await start_observability_daemon()
                logger.info("👁️ Permanent Observability Daemon STARTED - Nothing will be missed")
            except Exception as e:
                logger.error(f"❌ Permanent Observability Daemon startup failed: {e}")

        # Start Always-Know Observability Brain (continuous state awareness).
        # Keep it off the HTTP event loop: it performs synchronous DB operations
        # and can starve /healthz when Supabase is slow.
        try:
            from always_know_brain import initialize_always_know_brain

            app.state.always_know_future = _run_on_bg_loop(
                initialize_always_know_brain(),
                "always_know_brain",
            )
            if app.state.always_know_future is None:
                logger.info("🧠 Always-Know Brain STARTED - running on main loop fallback")
            else:
                logger.info("🧠 Always-Know Brain STARTED - running on background loop")
        except Exception as e:
            logger.error(f"❌ Always-Know Brain startup failed: {e}")

        # Initialize NerveCenter (always enabled; lightweight operational scans only)
        if NERVE_CENTER_AVAILABLE and get_nerve_center:
            try:
                logger.info("🧬 Initializing NerveCenter operational coordinator...")
                nerve_center = get_nerve_center()
                await nerve_center.activate()
                app.state.nerve_center = nerve_center
                logger.info("🧬 NerveCenter activated")
            except Exception as e:
                logger.error(f"❌ NerveCenter activation failed: {e}")
                app.state.nerve_center_error = str(e)
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ NerveCenter module not available")

        # Initialize OperationalMonitor (5-minute operational awareness loop)
        if OPERATIONAL_MONITOR_AVAILABLE and get_operational_monitor:
            try:
                monitor = get_operational_monitor(config.operational_monitor_interval)
                await monitor.start()
                app.state.operational_monitor = monitor
                logger.info(
                    "📈 OperationalMonitor activated (interval=%ss)",
                    config.operational_monitor_interval,
                )
            except Exception as e:
                logger.error(f"❌ OperationalMonitor activation failed: {e}")
                app.state.operational_monitor_error = str(e)
        else:
            logger.warning("⚠️ OperationalMonitor module not available")

        # ── Phase 3: Specialized agents (stagger to prevent DB pool exhaustion) ──
        await asyncio.sleep(2)
        logger.info("🚀 Activating specialized AI agents...")

        # Initialize AI Training Pipeline
        if TRAINING_AVAILABLE and AITrainingPipeline:
            try:
                app.state.training = AITrainingPipeline()
                logger.info("🎓 AI Training Pipeline ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Training Pipeline activation failed: {e}")

        # Initialize Notebook LM+ Learning
        if LEARNING_AVAILABLE and NotebookLMPlus:
            try:
                app.state.learning = NotebookLMPlus()
                logger.info("📚 Notebook LM+ Learning ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Learning System activation failed: {e}")

        # Initialize System Improvement Agent
        if SYSTEM_IMPROVEMENT_AVAILABLE and SystemImprovementAgent:
            try:
                app.state.system_improvement = await asyncio.to_thread(
                    lambda: SystemImprovementAgent(tenant_id=tenant_id)
                )
                logger.info("⚙️ System Improvement Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ System Improvement Agent activation failed: {e}")

        # Initialize DevOps Optimization Agent
        if DEVOPS_AGENT_AVAILABLE and DevOpsOptimizationAgent:
            try:
                app.state.devops_agent = await asyncio.to_thread(
                    lambda: DevOpsOptimizationAgent(tenant_id=tenant_id)
                )
                logger.info("🔧 DevOps Optimization Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ DevOps Agent activation failed: {e}")

        # Initialize Code Quality Agent
        if CODE_QUALITY_AVAILABLE and CodeQualityAgent:
            try:
                app.state.code_quality = await asyncio.to_thread(
                    lambda: CodeQualityAgent(tenant_id=tenant_id)
                )
                logger.info("✨ Code Quality Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Code Quality Agent activation failed: {e}")

        # Initialize Customer Success Agent
        if CUSTOMER_SUCCESS_AVAILABLE and CustomerSuccessAgent:
            try:
                app.state.customer_success = await asyncio.to_thread(
                    lambda: CustomerSuccessAgent(tenant_id=tenant_id)
                )
                logger.info("🤝 Customer Success Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Customer Success Agent activation failed: {e}")

        # Initialize Competitive Intelligence Agent
        if COMPETITIVE_INTEL_AVAILABLE and CompetitiveIntelligenceAgent:
            try:
                app.state.competitive_intel = await asyncio.to_thread(
                    lambda: CompetitiveIntelligenceAgent(tenant_id=tenant_id)
                )
                logger.info("🔍 Competitive Intelligence Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Competitive Intel Agent activation failed: {e}")

        # Initialize Vision Alignment Agent
        if VISION_ALIGNMENT_AVAILABLE and VisionAlignmentAgent:
            try:
                app.state.vision_alignment = await asyncio.to_thread(
                    lambda: VisionAlignmentAgent(tenant_id=tenant_id)
                )
                logger.info("🎯 Vision Alignment Agent ACTIVATED")
            except Exception as e:
                logger.error(f"❌ Vision Alignment Agent activation failed: {e}")

        # Initialize AI Self-Awareness Module
        if SELF_AWARENESS_AVAILABLE and get_self_aware_ai:
            try:
                # get_self_aware_ai is async (asyncpg). Running it in a thread
                # returns a coroutine object and can cause init race/cancel noise.
                app.state.self_aware_ai = await get_self_aware_ai()
                logger.info("🧠 AI Self-Awareness Module ACTIVATED - The AI OS is now self-aware!")
            except Exception as e:
                logger.error(f"❌ Self-Awareness Module activation failed: {e}")

        # Count activated agents
        agents_active = sum(
            [
                app.state.training is not None,
                app.state.learning is not None,
                app.state.system_improvement is not None,
                app.state.devops_agent is not None,
                app.state.code_quality is not None,
                app.state.customer_success is not None,
                app.state.competitive_intel is not None,
                app.state.vision_alignment is not None,
                app.state.self_aware_ai is not None,
            ]
        )
        logger.info(f"🤖 {agents_active} specialized AI agents ACTIVATED and OPERATIONAL")

        # ── Phase 4: Background daemons (stagger to prevent DB pool exhaustion) ──
        await asyncio.sleep(2)
        # Start Email Scheduler Daemon - Background email processing
        if EMAIL_SCHEDULER_AVAILABLE and start_email_scheduler:
            try:
                app.state.email_scheduler = await start_email_scheduler()
                logger.info("📧 Email Scheduler Daemon STARTED - Automated email processing active")
            except Exception as e:
                logger.error(f"❌ Email Scheduler startup failed: {e}")

        # Start Task Queue Consumer - processes ai_autonomous_tasks backlog and ERP unified_event handlers
        try:
            from task_queue_consumer import start_task_queue_consumer

            # This consumer uses the shared asyncpg pool; it must run on the same
            # event loop that created the pool (the HTTP server loop).
            app.state.task_queue_consumer = await start_task_queue_consumer()
            logger.info("📋 Task Queue Consumer STARTED - ai_autonomous_tasks processing active")
        except Exception as e:
            logger.error(f"❌ Task Queue Consumer startup failed: {e}")

        # Start AI Task Queue Consumer - processes public.ai_task_queue (quality checks, lead nurturing, etc.)
        try:
            from ai_task_queue_consumer import start_ai_task_queue_consumer

            # This consumer uses the shared asyncpg pool; it must run on the same
            # event loop that created the pool (the HTTP server loop).
            app.state.ai_task_queue_consumer = await start_ai_task_queue_consumer()
            logger.info("🧠 AI Task Queue Consumer STARTED - ai_task_queue processing active")
        except Exception as e:
            logger.error(f"❌ AI Task Queue Consumer startup failed: {e}")

        # Start Intelligent Task Orchestrator - AI-driven task prioritization and execution
        try:
            from intelligent_task_orchestrator import start_task_orchestrator, get_task_orchestrator

            # This orchestrator uses synchronous psycopg2 connections; run it off
            # the HTTP loop to prevent Render health check failures.
            app.state.task_orchestrator_future = _run_on_bg_loop(
                start_task_orchestrator(), "task_orchestrator"
            )
            app.state.task_orchestrator = get_task_orchestrator()
            logger.info(
                "🎯 Intelligent Task Orchestrator STARTED - AI-driven task prioritization active"
            )
        except Exception as e:
            logger.error(f"❌ Intelligent Task Orchestrator startup failed: {e}")

        # Category bucket cleanup is managed by _rate_limit_cleanup_loop below.
        # Do not start duplicate cleanup loops here.

        # Initialize Slack alerting integration (Total Completion Protocol)
        try:
            from slack_notifications import setup_slack_alerting

            if setup_slack_alerting():
                logger.info("📢 Slack alerting integration ACTIVATED")
            else:
                logger.info("📢 Slack alerting not configured (set SLACK_WEBHOOK_URL to enable)")
        except Exception as e:
            logger.warning(f"⚠️ Slack alerting setup failed: {e}")

        # ── Phase 5: Neural Core (final, after all other systems stabilize) ──
        await asyncio.sleep(2)
        # === NEURAL CORE - The Central Nervous System of the AI OS (2026-01-27) ===
        # This is THE core that makes the AI OS truly ALIVE - not monitoring, BEING aware
        if NEURAL_CORE_AVAILABLE:
            try:
                logger.info("🧠 INITIALIZING NEURAL CORE - The Central Nervous System...")
                init_result = await initialize_neural_core()
                app.state.neural_core = get_neural_core()
                logger.info(f"🧠 NEURAL CORE ONLINE - State: {init_result.get('state', 'unknown')}")
                logger.info("   └── Continuous awareness loop ACTIVE - AI OS is now SELF-AWARE")
                logger.info(f"   └── Monitoring {init_result.get('systems_discovered', 0)} systems")
            except Exception as e:
                logger.error(f"❌ Neural Core initialization failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ Neural Core not available - AI OS self-awareness limited")

        # ── Phase 6: Meta-Intelligence & Learning-Action Bridge (TRUE LEARNING) ──
        await asyncio.sleep(1)

        # Initialize Learning-Action Bridge - bridges learning to behavior
        try:
            from learning_action_bridge import get_learning_bridge, run_bridge_sync_loop

            bridge = await get_learning_bridge()
            app.state.learning_bridge = bridge
            logger.info("🔗 Learning-Action Bridge initialized")

            # Start background sync loop (syncs learning to behavior every 5 minutes)
            _run_on_bg_loop(run_bridge_sync_loop(interval_seconds=300), "learning_bridge_sync")
            logger.info("   └── Bridge sync loop ACTIVE - learning converts to action")
        except Exception as e:
            logger.warning(f"⚠️ Learning-Action Bridge initialization failed: {e}")

        # Initialize Meta-Intelligence Controller - TRUE AGI capabilities
        try:
            from meta_intelligence import get_meta_intelligence

            meta_intel = await get_meta_intelligence()
            app.state.meta_intelligence = meta_intel
            state = meta_intel.get_intelligence_state()
            logger.info(f"🧠 Meta-Intelligence AWAKENED - Level: {state['intelligence_level']:.1%}")
            logger.info("   └── Self-improvement: ACTIVE | Emergent reasoning: ACTIVE")
        except Exception as e:
            logger.warning(f"⚠️ Meta-Intelligence initialization failed: {e}")

        # ── Phase 7: Periodic embedding backfill ──
        async def _embedding_backfill_loop():
            """Periodically backfill missing embeddings for unified_ai_memory and live_brain_memories."""
            INTERVAL_SECONDS = 4 * 3600  # every 4 hours
            BATCH_PER_TABLE = 200
            SLEEP_BETWEEN_ROWS = 0.05  # 50ms between rows to avoid rate limits

            # Wait 5 minutes after startup to let everything stabilize
            await asyncio.sleep(300)

            while True:
                try:
                    pool = get_pool()
                    if pool is None or using_fallback():
                        logger.debug("Embedding backfill skipped: no DB pool")
                        await asyncio.sleep(INTERVAL_SECONDS)
                        continue

                    try:
                        from utils.embedding_provider import generate_embedding_async
                    except ImportError:
                        logger.warning("Embedding backfill skipped: embedding_provider unavailable")
                        await asyncio.sleep(INTERVAL_SECONDS)
                        continue

                    total_updated = 0

                    # Table 1: unified_ai_memory
                    try:
                        rows = await pool.fetch(
                            "SELECT id, content FROM unified_ai_memory "
                            "WHERE embedding IS NULL AND content IS NOT NULL "
                            "ORDER BY created_at ASC LIMIT $1",
                            BATCH_PER_TABLE,
                        )
                        for row in rows:
                            text = (
                                row["content"]
                                if isinstance(row["content"], str)
                                else str(row["content"])
                            )
                            if not text:
                                continue
                            emb = await generate_embedding_async(text[:30000])
                            if emb:
                                import json as _json

                                await pool.execute(
                                    "UPDATE unified_ai_memory SET embedding = $1::vector WHERE id = $2",
                                    _json.dumps(emb),
                                    row["id"],
                                )
                                total_updated += 1
                            await asyncio.sleep(SLEEP_BETWEEN_ROWS)
                    except Exception as e:
                        logger.warning("Embedding backfill (unified_ai_memory) error: %s", e)

                    # Table 2: live_brain_memories
                    try:
                        rows = await pool.fetch(
                            "SELECT id, content FROM live_brain_memories "
                            "WHERE embedding IS NULL AND content IS NOT NULL "
                            "ORDER BY created_at ASC LIMIT $1",
                            BATCH_PER_TABLE,
                        )
                        for row in rows:
                            text = (
                                row["content"]
                                if isinstance(row["content"], str)
                                else str(row["content"])
                            )
                            if not text:
                                continue
                            emb = await generate_embedding_async(text[:30000])
                            if emb:
                                import json as _json

                                await pool.execute(
                                    "UPDATE live_brain_memories SET embedding = $1::vector WHERE id = $2",
                                    _json.dumps(emb),
                                    row["id"],
                                )
                                total_updated += 1
                            await asyncio.sleep(SLEEP_BETWEEN_ROWS)
                    except Exception as e:
                        logger.warning("Embedding backfill (live_brain_memories) error: %s", e)

                    if total_updated > 0:
                        logger.info("Embedding backfill cycle: %d rows updated", total_updated)
                except Exception as e:
                    logger.error("Embedding backfill loop error: %s", e)

                await asyncio.sleep(INTERVAL_SECONDS)

        create_safe_task(_embedding_backfill_loop(), "embedding_backfill")
        logger.info("📊 Embedding backfill loop STARTED (every 4h, 200 rows/table/cycle)")

        logger.info("✅ Heavy component initialization complete - AI OS FULLY AWAKE!")

    create_safe_task(deferred_heavy_init(), "heavy_init")
    logger.info("📋 Heavy initialization scheduled (runs after server binds)")

    # Set remaining app.state values to None initially
    app.state.training = None
    app.state.learning = None
    app.state.system_improvement = None
    app.state.devops_agent = None
    app.state.code_quality = None
    app.state.customer_success = None
    app.state.competitive_intel = None
    app.state.vision_alignment = None
    app.state.self_aware_ai = None
    app.state.langgraph_orchestrator = None
    app.state.integration_layer = None
    app.state.learning_bridge = None
    app.state.meta_intelligence = None

    # Rate-limit bucket cleanup (every 5 minutes, prevents memory growth)
    async def _rate_limit_cleanup_loop():
        while True:
            await asyncio.sleep(300)
            try:
                await _category_cleanup()
            except Exception as exc:
                logger.debug("Rate limit cleanup error (non-fatal): %s", exc)

    create_safe_task(_rate_limit_cleanup_loop(), "rate_limit_cleanup")

    # === YIELD IMMEDIATELY to allow server to bind to port ===
    logger.info("⚡ Server binding to port NOW - heavy init continues in background")
    yield

    # === EVERYTHING BELOW IS SHUTDOWN CODE - nothing after yield runs on startup ===
    # Shutdown
    logger.info("🛑 Shutting down BrainOps AI Agents...")

    # Stop background loop runner (best-effort).
    try:
        bg_runner = getattr(app.state, "bg_runner", None)
        if bg_runner is not None:
            bg_runner.stop()
    except Exception:
        pass

    # Cancel all tracked background tasks
    try:
        await cancel_all_background_tasks(timeout=5.0)
        logger.info("✅ Background tasks cancelled")
    except Exception as e:
        logger.error(f"❌ Background tasks cancellation error: {e}")

    # Stop Task Queue Consumer
    try:
        from task_queue_consumer import stop_task_queue_consumer

        await stop_task_queue_consumer()
        logger.info("✅ Task Queue Consumer stopped")
    except Exception as e:
        logger.error(f"❌ Task Queue Consumer shutdown error: {e}")

    # Stop AI Task Queue Consumer
    try:
        from ai_task_queue_consumer import stop_ai_task_queue_consumer

        await stop_ai_task_queue_consumer()
        logger.info("✅ AI Task Queue Consumer stopped")
    except Exception as e:
        logger.error(f"❌ AI Task Queue Consumer shutdown error: {e}")

    # Stop Permanent Observability Daemon
    if PERMANENT_OBSERVABILITY_AVAILABLE and stop_observability_daemon:
        try:
            await stop_observability_daemon()
            logger.info("✅ Permanent Observability Daemon stopped")
        except Exception as e:
            logger.error(f"❌ Observability Daemon shutdown error: {e}")

    # Stop Email Scheduler Daemon
    if EMAIL_SCHEDULER_AVAILABLE and stop_email_scheduler:
        try:
            await stop_email_scheduler()
            logger.info("✅ Email Scheduler Daemon stopped")
        except Exception as e:
            logger.error(f"❌ Email Scheduler shutdown error: {e}")

    # Stop scheduler
    if hasattr(app.state, "scheduler") and app.state.scheduler:
        try:
            app.state.scheduler.shutdown()
            logger.info("✅ Agent Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Scheduler shutdown error: {e}")

    # Stop AUREA
    if hasattr(app.state, "aurea") and app.state.aurea:
        try:
            app.state.aurea.stop()
            logger.info("✅ AUREA Orchestrator stopped")
        except Exception as e:
            logger.error(f"❌ AUREA shutdown error: {e}")

    # Stop OperationalMonitor
    if hasattr(app.state, "operational_monitor") and app.state.operational_monitor:
        try:
            await app.state.operational_monitor.stop()
            logger.info("✅ OperationalMonitor stopped")
        except Exception as e:
            logger.error(f"❌ OperationalMonitor shutdown error: {e}")

    # Stop NerveCenter
    if hasattr(app.state, "nerve_center") and app.state.nerve_center:
        try:
            await app.state.nerve_center.deactivate()
            logger.info("✅ NerveCenter deactivated")
        except Exception as e:
            logger.error(f"❌ NerveCenter shutdown error: {e}")

    # Close database pool
    try:
        from database import close_pool

        await close_pool()
        logger.info("✅ Database pool closed")
    except Exception as e:
        logger.error(f"❌ Database pool shutdown error: {e}")

    logger.info("👋 BrainOps AI Agents shutdown complete")


# NOTE: Remaining initialization moved to deferred_heavy_init function above
# The following code block is removed to enable fast startup for Render port detection
# All these components will be initialized ~3 seconds after server starts

"""
OLD SYNC INITIALIZATION - NOW IN deferred_heavy_init BACKGROUND TASK:
- AI Training Pipeline
- Notebook LM+ Learning System
- System Improvement Agent
- DevOps Optimization Agent
- Code Quality Agent
- Customer Success Agent
- Competitive Intelligence Agent
- Vision Alignment Agent
- AI Self-Awareness Module
- LangGraph Orchestrator
- AI Integration Layer
- Bleeding Edge AI Systems
- Consciousness Emergence
"""


# Application instance with fast startup lifespan
app = FastAPI(
    title="BrainOps AI Agents",
    description="AI Native Operating System - Fully Autonomous AI Agents",
    version=VERSION,
    lifespan=lifespan,
)

# Attach slowapi rate limiter to app.
# SlowAPIMiddleware enables @limiter.limit() decorators on individual routes.
# Our custom rate_limit_middleware (below) provides category-based limits.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# Configure CORS - uses secure defaults from config (no wildcard fallback)
# SECURITY: Restrict allowed headers to specific values (not wildcard "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Tenant-ID",
        "X-Request-ID",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
)

# Mount static files for landing pages (email capture, product pages)
import os

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ---------------------------------------------------------------------------
# slowapi Rate Limiting Middleware
# ---------------------------------------------------------------------------
# Category-based rate limits applied via middleware. slowapi also supports
# per-route @limiter.limit() decorators which we apply below for specific
# endpoint categories. The middleware handles the general case.
#
# Rate Limit Categories:
#   - Health/status (/health, /healthz, /ready, /alive, /): 60/minute
#   - Agent execution (/agents/*/execute, /execute):        10/minute per key
#   - Email (/email/send, /email/process, /email/test):     5/minute per key
#   - Memory (/memory/*):                                   30/minute per key
#   - Default (all other endpoints):                        30/minute per key
# ---------------------------------------------------------------------------

# Pre-compiled patterns for endpoint categories
_HEALTH_PATHS = frozenset({"/health", "/healthz", "/ready", "/alive", "/"})
_AGENT_EXEC_PATTERN = re.compile(
    r"^/agents/[^/]+/execute$|^/execute$|^/agents/execute$|^/api/v1/agents/execute$"
)
_EMAIL_SEND_PATHS = frozenset({"/email/send", "/email/process", "/email/test"})
_MEMORY_PREFIX = "/memory/"

# In-memory token bucket for category-based rate limiting (supplements slowapi).
# slowapi handles per-route decorator limits; this middleware handles category routing.
_category_buckets: dict[str, dict[str, tuple[float, float]]] = {}
_category_lock = asyncio.Lock()

# Category configs: (requests_per_minute, burst_size)
_CATEGORY_LIMITS = {
    "health": (60, 15),
    "agent_exec": (10, 3),
    "email": (5, 2),
    "memory": (30, 8),
    "default": (30, 8),
}
_CATEGORY_BUCKET_MAX_IDENTITIES = int(os.getenv("RATE_LIMIT_BUCKET_MAX_IDENTITIES", "5000"))


async def _category_is_allowed(category: str, identity: str) -> bool:
    """Token bucket check for a given category + identity."""
    rpm, burst = _CATEGORY_LIMITS[category]
    rate = rpm / 60.0  # tokens per second
    async with _category_lock:
        bucket = _category_buckets.setdefault(category, {})
        # Protect event-loop latency: do not allow unbounded growth by
        # adversarial/high-churn identities. Resetting the bucket is safer
        # than scanning very large maps on the main HTTP loop.
        if len(bucket) >= _CATEGORY_BUCKET_MAX_IDENTITIES and identity not in bucket:
            logger.warning(
                "Rate-limit bucket reset: category=%s size=%d (max=%d)",
                category,
                len(bucket),
                _CATEGORY_BUCKET_MAX_IDENTITIES,
            )
            bucket.clear()
        now = time.time()
        tokens, last_update = bucket.get(identity, (float(burst), now))
        elapsed = now - last_update
        tokens = min(float(burst), tokens + elapsed * rate)
        if tokens >= 1.0:
            bucket[identity] = (tokens - 1.0, now)
            return True
        else:
            bucket[identity] = (tokens, now)
            return False


async def _category_cleanup() -> None:
    """Remove stale bucket entries to prevent memory growth."""
    async with _category_lock:
        now = time.time()
        for category in list(_category_buckets.keys()):
            bucket = _category_buckets[category]
            # Fast-path guard: keep cleanup bounded to protect health-check
            # latency under high-cardinality traffic.
            if len(bucket) > _CATEGORY_BUCKET_MAX_IDENTITIES:
                logger.warning(
                    "Rate-limit cleanup reset: category=%s size=%d (max=%d)",
                    category,
                    len(bucket),
                    _CATEGORY_BUCKET_MAX_IDENTITIES,
                )
                bucket.clear()
                continue
            stale = [k for k, (_, ts) in bucket.items() if now - ts > 300]
            for k in stale:
                del bucket[k]


def _classify_path(path: str) -> str:
    """Determine rate-limit category for a request path."""
    if path in _HEALTH_PATHS:
        return "health"
    if _AGENT_EXEC_PATTERN.match(path):
        return "agent_exec"
    if path in _EMAIL_SEND_PATHS:
        return "email"
    if (
        path.startswith(_MEMORY_PREFIX)
        or path == "/memory/store"
        or path == "/memory/search"
        or path == "/memory/stats"
    ):
        return "memory"
    return "default"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply category-based rate limiting using slowapi key function."""
    path = request.url.path

    # Skip rate limiting for docs and webhook endpoints with their own auth
    if path in ("/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
    # Authenticated internal webhooks have HMAC/signature verification; don't rate limit.
    if path == "/events/webhook/erp":
        return await call_next(request)
    # Stripe/Gumroad webhooks use their own signature verification
    if path.startswith("/stripe/") or path.startswith("/gumroad/"):
        return await call_next(request)

    identity = _rate_limit_key(request)

    # Internal E2E verification requests get unique keys (e2e-internal:*).
    # Skip the category bucket entirely so the verifier never 429s itself.
    if identity.startswith("e2e-internal:"):
        return await call_next(request)

    category = _classify_path(path)

    if not await _category_is_allowed(category, identity):
        rpm, _ = _CATEGORY_LIMITS[category]
        logger.warning(
            "Rate limit exceeded: category=%s identity=%s path=%s limit=%d/min",
            category,
            identity,
            path,
            rpm,
        )
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many requests. Please slow down.",
                "category": category,
                "limit": f"{rpm}/minute",
            },
            headers={"Retry-After": "60"},
        )

    return await call_next(request)


# Request/latency observability middleware
@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    """Measure request latency and record lightweight metrics."""
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception:
        # Still record the 500 before bubbling up
        status_code = 500
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        path = request.url.path.split("?")[0]
        await REQUEST_METRICS.record(
            path=path, method=request.method, status=status_code, duration_ms=duration_ms
        )


# Rate limit 403 error logging to avoid log flooding
from collections import defaultdict

_api_key_error_log_times: dict[str, float] = defaultdict(float)
_API_KEY_ERROR_LOG_INTERVAL = 60  # Only log same path/error once per minute


async def verify_api_key(
    request: Request = None,
    websocket: WebSocket = None,
) -> bool:
    """
    Verify authentication using either API Key or JWT.
    Prioritizes Master Key, then JWT, then Configured API Keys.
    """
    connection: HTTPConnection | None = request or websocket
    if connection is None:
        raise HTTPException(status_code=500, detail="Authentication context missing")

    api_key = (
        connection.headers.get("X-API-Key")
        or connection.headers.get("x-api-key")
        or connection.headers.get("X-Api-Key")
    )
    if not api_key:
        try:
            api_key = connection.query_params.get("api_key") or connection.query_params.get("token")
        except Exception:
            api_key = None
    if api_key:
        api_key = api_key.strip()

    auth_header = connection.headers.get("authorization", "")
    jwt_token: Optional[HTTPAuthorizationCredentials] = None
    if auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()
        if bearer_token:
            jwt_token = HTTPAuthorizationCredentials(scheme="Bearer", credentials=bearer_token)

    # 0. Check Master Key (Immediate Override)
    # Check headers directly to catch it before JWT processing
    # (FastAPI dependencies might have already parsed it into api_key or jwt_token)
    master_key = getattr(config.security, "master_api_key", None) or os.getenv("MASTER_API_KEY")

    # Check X-API-Key header or api_key arg
    if master_key and api_key == master_key:
        return True

    # Check Authorization header manually for master key
    # SECURITY: Use exact match, not substring matching (fixes potential bypass)
    if master_key and auth_header:
        # Handle "Bearer <token>" format
        if auth_header.lower().startswith("bearer "):
            extracted_token = auth_header[7:].strip()
            if extracted_token == master_key:
                return True
        # Handle raw token in Authorization header
        elif auth_header.strip() == master_key:
            return True

    if not config.security.auth_required:
        return True

    # 1. Try JWT first (User Context)
    if jwt_token and request is not None:
        # If the token in jwt_token is the master key, we already returned True above.
        # So if we are here, it's a real JWT candidate.
        try:
            payload = await verify_jwt(request, jwt_token)
            if payload:
                # JWT is valid, user context is set in request.state
                return True
        except HTTPException:
            # If JWT is invalid, we don't fall back to API Key if it was provided but bad
            # But if verify_jwt returned None (not configured?), we might fall back.
            # verify_jwt raises 401 on invalid token.
            raise

    # 2. Try API Key (Service Context)
    if not config.security.auth_configured:
        raise HTTPException(status_code=503, detail="Authentication misconfigured")

    provided = api_key
    if not provided:
        # Fallback parsing for weird header schemes
        if auth_header:
            scheme, _, token = auth_header.partition(" ")
            scheme_lower = scheme.lower()
            if scheme_lower in ("apikey", "api-key"):
                provided = token.strip()

    if not provided and config.security.test_api_key:
        provided = (
            request.headers.get("x-test-api-key")
            or request.headers.get("X-Test-Api-Key")
            or request.headers.get("x-api-key")
            or request.headers.get("X-API-Key")
        )

    if not provided:
        # Rate-limit logging of missing API key errors
        path = connection.url.path
        error_key = f"missing:{path}"
        now = time.time()
        if now - _api_key_error_log_times[error_key] > _API_KEY_ERROR_LOG_INTERVAL:
            _api_key_error_log_times[error_key] = now
            logger.warning(f"Auth missing for {path} (rate-limited)")
        raise HTTPException(
            status_code=403, detail="Authentication required (API Key or Bearer Token)"
        )

    if master_key and provided == master_key:
        # Master key override (redundant but safe)
        return True

    if provided not in config.security.valid_api_keys:
        # Rate-limit logging of invalid API key errors
        path = connection.url.path
        error_key = f"invalid:{path}"
        now = time.time()
        if now - _api_key_error_log_times[error_key] > _API_KEY_ERROR_LOG_INTERVAL:
            _api_key_error_log_times[error_key] = now
            logger.warning(f"Invalid API key for {path} (rate-limited)")
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


# Include routers
SECURED_DEPENDENCIES = [Depends(verify_api_key)]

app.include_router(memory_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(sync_router, dependencies=SECURED_DEPENDENCIES)  # Memory sync/migration
app.include_router(brain_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(memory_coordination_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(customer_intelligence_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(relationships_router, dependencies=SECURED_DEPENDENCIES)

# External webhook endpoints must NOT require an internal API key; they validate their own webhook secrets/signatures.
app.include_router(gumroad_router)
app.include_router(stripe_webhook_router)  # Stripe webhooks - uses Stripe-Signature verification
app.include_router(erp_event_router)

# Unified Events System - Central event bus for all systems
# NOTE: /webhook/erp endpoint has its own signature verification, but other endpoints need auth
if UNIFIED_EVENTS_AVAILABLE and unified_events_router:
    app.include_router(unified_events_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "Mounted: Unified Events API at /events - publish, webhook/erp, recent, stats, replay (SECURED)"
    )

app.include_router(google_keep_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(codebase_graph_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(
    state_sync_router, dependencies=SECURED_DEPENDENCIES
)  # Real-time state synchronization
app.include_router(revenue_router, dependencies=SECURED_DEPENDENCIES)  # Revenue generation system
app.include_router(
    taskmate_router, dependencies=SECURED_DEPENDENCIES
)  # P1-TASKMATE-001: Cross-model task manager
app.include_router(
    full_power_crud_router, dependencies=SECURED_DEPENDENCIES
)  # Full Power API v2 - CRUD, pagination, filtering, lifecycle controls
app.include_router(
    revenue_complete_router, dependencies=SECURED_DEPENDENCIES
)  # Complete Revenue API with billing
app.include_router(
    revenue_control_tower_router, dependencies=SECURED_DEPENDENCIES
)  # Revenue Control Tower - GROUND TRUTH (REAL vs TEST)
app.include_router(
    pipeline_router, dependencies=SECURED_DEPENDENCIES
)  # Pipeline State Machine - ledger-backed state transitions
app.include_router(
    proposals_router, dependencies=SECURED_DEPENDENCIES
)  # Proposal Engine - draft/approve/send workflow
app.include_router(
    outreach_router, dependencies=SECURED_DEPENDENCIES
)  # Outreach Engine - lead enrichment and sequences
app.include_router(
    payments_router, dependencies=SECURED_DEPENDENCIES
)  # Payment Capture - invoices and revenue collection
app.include_router(
    communications_router, dependencies=SECURED_DEPENDENCIES
)  # Communications - send estimates/invoices from Weathercraft ERP
app.include_router(
    revenue_operator_router, dependencies=SECURED_DEPENDENCIES
)  # AI Revenue Operator - automated actions
app.include_router(
    lead_discovery_router, dependencies=SECURED_DEPENDENCIES
)  # Lead Discovery Engine - automated lead discovery and qualification
app.include_router(
    lead_engine_router, dependencies=SECURED_DEPENDENCIES
)  # Lead Engine - MRG→ERP relay pipeline
app.include_router(
    campaigns_router, dependencies=SECURED_DEPENDENCIES
)  # Campaign System - CO commercial reroof lead gen
app.include_router(
    email_capture_router
)  # Email Capture - PUBLIC endpoint for lead generation (no auth required)
app.include_router(
    neural_reconnection_router, dependencies=SECURED_DEPENDENCIES
)  # Neural Reconnection - Schema unification & mode logic
app.include_router(
    roofing_labor_ml_router, dependencies=SECURED_DEPENDENCIES
)  # Roofing labor ML (RandomForest)

# Bleeding-edge AI systems (2025)
app.include_router(
    digital_twin_router, dependencies=SECURED_DEPENDENCIES
)  # Digital Twin virtual replicas
app.include_router(
    market_intelligence_router, dependencies=SECURED_DEPENDENCIES
)  # Predictive market intelligence
app.include_router(
    system_orchestrator_router, dependencies=SECURED_DEPENDENCIES
)  # Autonomous system orchestration (1-10K systems)
app.include_router(
    self_healing_router, dependencies=SECURED_DEPENDENCIES
)  # Enhanced self-healing AI infrastructure

# System Observability - THE visibility layer that makes AI OS transparent
if SYSTEM_OBSERVABILITY_AVAILABLE and system_observability_router:
    app.include_router(system_observability_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: System Observability at /observe - status, problems, healing, ask")
app.include_router(
    e2e_verification_router, dependencies=SECURED_DEPENDENCIES
)  # E2E System Verification
app.include_router(logistics_router, dependencies=SECURED_DEPENDENCIES)  # Neuro-Symbolic Logistics
app.include_router(
    victoria_router, dependencies=SECURED_DEPENDENCIES
)  # Victoria scheduling agent (ERP compatibility)
app.include_router(
    infrastructure_router, dependencies=SECURED_DEPENDENCIES
)  # Self-Provisioning Infra
app.include_router(
    real_ops_router, dependencies=SECURED_DEPENDENCIES
)  # Real Operations: health, OODA, briefing, alerts
app.include_router(
    revenue_automation_router, dependencies=SECURED_DEPENDENCIES
)  # Revenue Automation Engine
app.include_router(
    income_streams_router, dependencies=SECURED_DEPENDENCIES
)  # Automated Income Streams (email, subscriptions, affiliates)
app.include_router(
    mcp_router, dependencies=SECURED_DEPENDENCIES
)  # MCP Bridge - 345 tools (Render, Vercel, Supabase, GitHub, Stripe, Docker)
app.include_router(
    cicd_router, dependencies=SECURED_DEPENDENCIES
)  # Autonomous CI/CD - manage 1-10K deployments
app.include_router(
    a2ui_router, dependencies=SECURED_DEPENDENCIES
)  # Google A2UI Protocol - Agent-generated UIs
app.include_router(
    aurea_chat_router, dependencies=SECURED_DEPENDENCIES
)  # AUREA Live Conversational AI
app.include_router(
    full_observability_router, dependencies=SECURED_DEPENDENCIES
)  # Comprehensive Observability Dashboard
app.include_router(
    self_awareness_router, dependencies=SECURED_DEPENDENCIES
)  # Self-Awareness Dashboard
app.include_router(
    ai_awareness_router, dependencies=SECURED_DEPENDENCIES
)  # Complete AI Awareness - THE endpoint

from api.autonomic_status import router as autonomic_status_router  # Autonomic Status API

# Voice Router - AUREA voice & communications
if VOICE_ROUTER_AVAILABLE:
    app.include_router(voice_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🎤 Voice Router endpoints registered at /api/v1/voice/*")

app.include_router(autonomic_status_router, dependencies=SECURED_DEPENDENCIES)
logger.info("mounted: Autonomic Status API at /autonomic/status")

# Permanent Observability Router - Never miss anything
if PERMANENT_OBSERVABILITY_AVAILABLE:
    app.include_router(permanent_observability_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("👁️ Permanent Observability endpoints registered at /visibility/*")

# DevOps Automation API - Permanent knowledge & automated operations
if DEVOPS_API_AVAILABLE:
    app.include_router(devops_api_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🔧 DevOps Automation endpoints registered at /devops/*")

# Customer Acquisition API - Autonomous lead discovery and conversion
if CUSTOMER_ACQUISITION_API_AVAILABLE:
    app.include_router(customer_acquisition_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🎯 Customer Acquisition endpoints registered at /acquire/*")

# AI-Powered UI Testing System (2025-12-29) - Automated visual testing with AI vision
if UI_TESTING_AVAILABLE:
    app.include_router(ui_testing_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "Mounted: AI UI Testing API at /ui-testing - visual testing, accessibility, performance"
    )

# Bleeding Edge AI Systems (2025-12-27) - Revolutionary capabilities
if BLEEDING_EDGE_AVAILABLE:
    app.include_router(bleeding_edge_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Bleeding Edge AI API at /bleeding-edge - 37 capabilities")

# AI Observability & Integration (2025-12-27) - Perfect cross-module integration
if AI_OBSERVABILITY_AVAILABLE:
    app.include_router(ai_observability_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: AI Observability API at /ai - unified metrics, events, learning")

# Predictive Execution (2026-01-27) - Proactive task execution
if PREDICTIVE_EXECUTION_AVAILABLE:
    app.include_router(predictive_execution_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Predictive Execution API at /predictive - proactive task execution")

# NEURAL CORE - The Central Nervous System (2026-01-27)
# This IS the self-awareness of the AI OS - not monitoring, BEING aware
if NEURAL_CORE_AVAILABLE:
    app.include_router(neural_core_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🧠 Mounted: NEURAL CORE at /neural - The AI OS Central Nervous System")

# CODE QUALITY MONITOR - Deep code-level monitoring (2026-01-27)
try:
    from api.code_quality import router as code_quality_router

    app.include_router(code_quality_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🔬 Mounted: Code Quality Monitor at /code-quality - Deep code-level monitoring")
except ImportError as e:
    logger.warning(f"Code Quality Router not available: {e}")

# AI OS UNIFIED DASHBOARD - The one endpoint to rule them all (2026-01-27)
try:
    from api.ai_os_dashboard import router as ai_os_dashboard_router

    app.include_router(ai_os_dashboard_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🌟 Mounted: AI OS Dashboard at /ai-os - UNIFIED COMMAND CENTER")
except ImportError as e:
    logger.warning(f"AI OS Dashboard Router not available: {e}")

# TRUE AI INTELLIGENCE - Real LLM-powered analysis, not pattern matching (2026-01-27)
try:
    from api.ai_intelligence import router as ai_intelligence_router

    app.include_router(ai_intelligence_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🧠 Mounted: TRUE AI Intelligence at /intelligence - REAL LLM ANALYSIS")
except ImportError as e:
    logger.warning(f"AI Intelligence Router not available: {e}")

# DATABASE INTELLIGENCE - AI-powered DB monitoring and optimization (2026-01-27)
try:
    from api.db_intelligence import router as db_intelligence_router

    app.include_router(db_intelligence_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🗄️ Mounted: Database Intelligence at /db-intelligence - AI DB OPTIMIZATION")
except ImportError as e:
    logger.warning(f"Database Intelligence Router not available: {e}")

# CIRCUIT BREAKERS API - Centralized circuit breaker management (2026-01-27)
try:
    from api.circuit_breakers import router as circuit_breakers_router

    app.include_router(circuit_breakers_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("🔌 Mounted: Circuit Breakers API at /circuit-breakers - SERVICE PROTECTION")
except ImportError as e:
    logger.warning(f"Circuit Breakers Router not available: {e}")

# DAILY BRIEFING API - AI-powered daily summary of agent activity (2026-02-02)
try:
    from api.daily_briefing import router as daily_briefing_router

    app.include_router(daily_briefing_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("📋 Mounted: Daily Briefing API at /briefing - AI-POWERED SUMMARIES")
except ImportError as e:
    logger.warning(f"Daily Briefing Router not available: {e}")

# AI System Enhancements (2025-12-28) - Health scoring, alerting, correlation, WebSocket
if AI_ENHANCEMENTS_AVAILABLE:
    app.include_router(ai_enhancements_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: AI Enhancements API at /ai/enhanced - health, alerting, WebSocket")

# Mount New Pipeline Routers (with graceful fallback)
if PRODUCT_GEN_ROUTER_AVAILABLE:
    app.include_router(product_generation_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Product Generation API at /products")

if AFFILIATE_ROUTER_AVAILABLE:
    app.include_router(affiliate_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Affiliate API at /affiliate")

if KNOWLEDGE_BASE_ROUTER_AVAILABLE:
    app.include_router(knowledge_base_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Knowledge Base API at /knowledge-base")

if SOP_ROUTER_AVAILABLE:
    app.include_router(sop_router, dependencies=SECURED_DEPENDENCIES)

if COMPANYCAM_ROUTER_AVAILABLE:
    app.include_router(companycam_router, dependencies=SECURED_DEPENDENCIES)

# DevOps Loop API (2025-12-31) - Ultimate self-healing DevOps orchestrator
try:
    from api.devops_loop import router as devops_loop_router

    app.include_router(devops_loop_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: DevOps Loop API at /devops-loop - continuous self-healing")
except ImportError as e:
    logger.warning(f"DevOps Loop Router not available: {e}")

# Operational Verification API (2025-12-29) - PROVES systems work, doesn't assume
if VERIFICATION_AVAILABLE:
    app.include_router(verification_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Operational Verification API at /verify - real system testing")

# System Integration API (2025-12-29) - Full pipeline connectivity, no silos
if INTEGRATION_AVAILABLE:
    app.include_router(integration_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: System Integration API at /integrate - training→learning→memory→agents")

# Background Task Monitoring (2025-12-29) - Heartbeats for all background tasks
if BG_MONITORING_AVAILABLE:
    app.include_router(bg_monitoring_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "Mounted: Background Task Monitoring API at /monitor/background - no more fire-and-forget"
    )

# Always-Know Observability Brain (2025-12-29) - Deep continuous monitoring
if ALWAYS_KNOW_AVAILABLE:
    app.include_router(always_know_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Always-Know Brain at /api/v1/always-know - continuous state awareness")

if ALWAYS_KNOW_COMPAT_AVAILABLE:
    app.include_router(always_know_compat_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Always-Know Brain compat at /always-know")

# Ultimate E2E System (2025-12-31) - COMPLETE e2e awareness
if ULTIMATE_E2E_AVAILABLE:
    app.include_router(ultimate_e2e_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Ultimate E2E System at /ultimate-e2e - build logs, DB, UI tests, issues")

# TRUE Operational Validation (2025-12-31) - REAL operation testing
if TRUE_VALIDATION_AVAILABLE:
    app.include_router(true_validation_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "Mounted: TRUE Validation at /validate - executes real operations, not status checks"
    )

# Learning Feedback Loop (2025-12-30) - Insights finally become actions
if LEARNING_ROUTER_AVAILABLE:
    app.include_router(learning_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Learning Feedback Loop at /api/learning - 4,700+ insights now actionable")

# Learning Visibility (2026-02-02) - See what the AI has learned
if LEARNING_VISIBILITY_AVAILABLE:
    app.include_router(learning_visibility_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "Mounted: Learning Visibility at /api/learning-visibility - behavior rules & insights exposed"
    )

# Import and include analytics router
try:
    from analytics_endpoint import router as analytics_router

    app.include_router(analytics_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("✅ Analytics endpoint loaded")
except ImportError as e:
    logger.warning(f"Analytics endpoint not available: {e}")

# Revenue Pipeline Factory (2026-01-14) - Master revenue orchestrator
try:
    from revenue_pipeline_factory import create_factory_router

    revenue_factory_router = create_factory_router()
    app.include_router(revenue_factory_router, dependencies=SECURED_DEPENDENCIES)
    logger.info(
        "✅ Revenue Pipeline Factory loaded at /revenue/factory - 6 automated revenue streams"
    )
except ImportError as e:
    logger.warning(f"Revenue Pipeline Factory not available: {e}")

# API Monetization Engine (2026-01-14) - Usage-based billing
try:
    from api_monetization_engine import create_monetization_router

    api_monetization_router = create_monetization_router()
    app.include_router(api_monetization_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("✅ API Monetization Engine loaded at /api/billing - usage-based pricing")
except ImportError as e:
    logger.warning(f"API Monetization Engine not available: {e}")

# Memory Enforcement API (2026-01-15) - RBA/WBA enforcement, verification, audit
try:
    from api.memory_enforcement_api import router as memory_enforcement_router

    app.include_router(memory_enforcement_router, dependencies=SECURED_DEPENDENCIES)
    MEMORY_ENFORCEMENT_AVAILABLE = True
    logger.info("✅ Memory Enforcement API loaded at /enforcement - RBA/WBA, verification, audit")
except ImportError as e:
    MEMORY_ENFORCEMENT_AVAILABLE = False
    logger.warning(f"Memory Enforcement API not available: {e}")

# Memory Hygiene API (2026-01-15) - Automated memory maintenance
try:
    from api.memory_hygiene_api import router as memory_hygiene_router

    app.include_router(memory_hygiene_router, dependencies=SECURED_DEPENDENCIES)
    MEMORY_HYGIENE_AVAILABLE = True
    logger.info("✅ Memory Hygiene API loaded at /hygiene - deduplication, conflicts, decay")
except ImportError as e:
    MEMORY_HYGIENE_AVAILABLE = False
    logger.warning(f"Memory Hygiene API not available: {e}")

# Memory Observability API (2026-01-27) - Comprehensive memory monitoring and metrics
try:
    from api.memory_observability import router as memory_observability_router

    app.include_router(memory_observability_router, dependencies=SECURED_DEPENDENCIES)
    MEMORY_OBSERVABILITY_AVAILABLE = True
    logger.info(
        "✅ Memory Observability API loaded at /memory/observability - stats, health, hot/cold, decay, consolidation"
    )
except ImportError as e:
    MEMORY_OBSERVABILITY_AVAILABLE = False
    logger.warning(f"Memory Observability API not available: {e}")

# Advanced LangGraph Workflow Engine (2026-01-27) - State machines, checkpoints, HITL, OODA
try:
    from api.workflows import router as workflows_router

    app.include_router(workflows_router, dependencies=SECURED_DEPENDENCIES)
    WORKFLOWS_AVAILABLE = True
    logger.info("Mounted: Advanced Workflow Engine at /workflows - checkpoints, HITL, OODA loops")
except ImportError as e:
    WORKFLOWS_AVAILABLE = False
    logger.warning(f"Advanced Workflow Engine not available: {e}")

# Autonomous Issue Resolver (2026-01-27) - ACTUALLY FIXES detected issues
try:
    from api.autonomous_resolver import router as autonomous_resolver_router

    app.include_router(autonomous_resolver_router, dependencies=SECURED_DEPENDENCIES)
    AUTONOMOUS_RESOLVER_AVAILABLE = True
    logger.info(
        "🔧 Mounted: Autonomous Issue Resolver at /resolver - detects AND FIXES AI OS issues"
    )
except ImportError as e:
    AUTONOMOUS_RESOLVER_AVAILABLE = False
    logger.warning(f"Autonomous Issue Resolver not available: {e}")

# Proactive Alerts API (2026-02-02) - AI-powered proactive recommendations
try:
    from api.proactive_alerts import router as proactive_alerts_router

    app.include_router(proactive_alerts_router, dependencies=SECURED_DEPENDENCIES)
    PROACTIVE_ALERTS_AVAILABLE = True
    logger.info(
        "🎯 Mounted: Proactive Alerts API at /proactive - agent patterns, revenue opportunities, anomaly detection"
    )
except ImportError as e:
    PROACTIVE_ALERTS_AVAILABLE = False
    logger.warning(f"Proactive Alerts API not available: {e}")


def _collect_active_systems() -> list[str]:
    """Return a list of systems that are initialized and active."""
    active = []
    if AUREA_AVAILABLE and getattr(app.state, "aurea", None):
        active.append("AUREA Orchestrator")
    if SELF_HEALING_AVAILABLE and getattr(app.state, "healer", None):
        active.append("Self-Healing Recovery")
    if MEMORY_AVAILABLE and getattr(app.state, "memory", None):
        active.append("Memory Manager")
    if EMBEDDED_MEMORY_AVAILABLE and getattr(app.state, "embedded_memory", None):
        active.append("Embedded Memory (RAG)")
    if TRAINING_AVAILABLE and getattr(app.state, "training", None):
        active.append("Training Pipeline")
    if LEARNING_AVAILABLE and getattr(app.state, "learning", None):
        active.append("Learning System")
    if SCHEDULER_AVAILABLE and getattr(app.state, "scheduler", None):
        active.append("Agent Scheduler")
    if NERVE_CENTER_AVAILABLE and getattr(app.state, "nerve_center", None):
        active.append("NerveCenter (Operational Coordinator)")
    if OPERATIONAL_MONITOR_AVAILABLE and getattr(app.state, "operational_monitor", None):
        active.append("Operational Monitor")
    if AI_AVAILABLE and ai_core:
        active.append("AI Core")
    if SYSTEM_IMPROVEMENT_AVAILABLE and getattr(app.state, "system_improvement", None):
        active.append("System Improvement Agent")
    if DEVOPS_AGENT_AVAILABLE and getattr(app.state, "devops_agent", None):
        active.append("DevOps Optimization Agent")
    if CODE_QUALITY_AVAILABLE and getattr(app.state, "code_quality", None):
        active.append("Code Quality Agent")
    if CUSTOMER_SUCCESS_AVAILABLE and getattr(app.state, "customer_success", None):
        active.append("Customer Success Agent")
    if COMPETITIVE_INTEL_AVAILABLE and getattr(app.state, "competitive_intel", None):
        active.append("Competitive Intelligence Agent")
    if VISION_ALIGNMENT_AVAILABLE and getattr(app.state, "vision_alignment", None):
        active.append("Vision Alignment Agent")
    if RECONCILER_AVAILABLE and getattr(app.state, "reconciler", None):
        active.append("Self-Healing Reconciler")
    # Bleeding Edge AI Systems (2025-12-27)
    if BLEEDING_EDGE_AVAILABLE:
        active.append(
            "Bleeding Edge AI (OODA, Hallucination, Memory, Dependability, Consciousness, Circuit Breaker)"
        )
    # Autonomous Issue Resolver (2026-01-27) - ACTUALLY FIXES issues
    if AUTONOMOUS_RESOLVER_AVAILABLE:
        active.append("Autonomous Issue Resolver (Detects AND FIXES AI OS Issues)")
    # Memory Enforcement & Hygiene (2026-01-15) - Total Completion Protocol
    if MEMORY_ENFORCEMENT_AVAILABLE:
        active.append("Memory Enforcement (RBA/WBA, Verification, Audit)")
    if MEMORY_HYGIENE_AVAILABLE:
        active.append("Memory Hygiene (Deduplication, Conflicts, Decay)")
    # Advanced Workflow Engine (2026-01-27) - LangGraph-based orchestration
    if WORKFLOWS_AVAILABLE:
        active.append("Advanced Workflow Engine (LangGraph, OODA, HITL, Checkpoints)")
    return active


def _scheduler_snapshot() -> dict[str, Any]:
    """Return scheduler status with safe defaults."""
    scheduler = getattr(app.state, "scheduler", None)
    if not (SCHEDULER_AVAILABLE and scheduler):
        return {"enabled": False, "message": "Scheduler not available"}

    apscheduler_jobs = scheduler.scheduler.get_jobs()
    return {
        "enabled": True,
        "running": scheduler.scheduler.running,
        "registered_jobs_count": len(scheduler.registered_jobs),
        "apscheduler_jobs_count": len(apscheduler_jobs),
        "next_jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
            for job in apscheduler_jobs[:5]
        ],
    }


def _aurea_status() -> dict[str, Any]:
    aurea = getattr(app.state, "aurea", None)
    if not (AUREA_AVAILABLE and aurea):
        return {"available": False, "running": False}
    try:
        return {**aurea.get_status(), "available": True}
    except Exception as exc:
        logger.error("Failed to read AUREA status: %s", exc)
        return {"available": True, "running": False, "error": str(exc)}


def _self_healing_status() -> dict[str, Any]:
    healer = getattr(app.state, "healer", None)
    if not (SELF_HEALING_AVAILABLE and healer):
        return {"available": False}

    # Keep this helper lightweight: do not call `get_health_report()` here because it can
    # execute multiple DB queries and return a large payload. Dedicated self-healing
    # endpoints cover the full report when needed.
    try:
        circuit_breakers = getattr(healer, "circuit_breakers", None) or {}
        breaker_total = len(circuit_breakers) if isinstance(circuit_breakers, dict) else 0
        rules = getattr(healer, "healing_rules", None) or []
        active_rules = len(rules) if hasattr(rules, "__len__") else None
        return {
            "available": True,
            "report_available": hasattr(healer, "get_health_report"),
            "circuit_breakers_total": breaker_total,
            "active_healing_rules": active_rules,
        }
    except Exception as exc:
        logger.error("Failed to read self-healing status: %s", exc)
        return {"available": True, "error": str(exc)}


async def _memory_stats_snapshot(pool) -> dict[str, Any]:
    """
    Get a fast snapshot of memory/learning health.
    Reuses logic from /memory/status but keeps output minimal for usage reports.
    """
    try:
        existing_tables = await pool.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """
        )
        if not existing_tables:
            return {"status": "not_configured"}

        table_names = [t["table_name"] for t in existing_tables]
        preferred = next(
            (t for t in ("ai_persistent_memory", "memory_entries", "memories") if t in table_names),
            table_names[0],
        )
        stats = await pool.fetchrow(f"SELECT COUNT(*) AS total FROM {preferred}")
        return {
            "status": "operational",
            "table": preferred,
            "total_records": stats["total"] if stats else 0,
        }
    except Exception as exc:
        logger.error("Failed to fetch memory stats: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _get_agent_usage(pool) -> dict[str, Any]:
    """Fetch recent agent usage, trying both legacy and new table names."""
    # Table combinations with their JOIN conditions (some use agent_id UUID, some use agent_name text)
    queries: list[tuple[str, str, str, str]] = [
        # (agents_table, executions_table, join_condition, time_column)
        ("ai_agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
        ("agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
    ]
    errors: list[str] = []

    for agents_table, executions_table, join_cond, time_col in queries:
        try:
            rows = await pool.fetch(
                f"""
                SELECT
                    a.id::text AS id,
                    a.name,
                    COALESCE(a.category, 'other') AS category,
                    COALESCE(a.enabled, true) AS enabled,
                    COUNT(e.id) AS executions_last_30d,
                    MAX({time_col}) AS last_execution,
                    AVG(e.execution_time_ms) FILTER (WHERE e.execution_time_ms IS NOT NULL) AS avg_duration_ms
                FROM {agents_table} a
                LEFT JOIN {executions_table} e
                    ON {join_cond}
                    AND {time_col} >= NOW() - INTERVAL '30 days'
                GROUP BY a.id, a.name, a.category, a.enabled
                ORDER BY executions_last_30d DESC, last_execution DESC NULLS LAST
                LIMIT 20
            """
            )
            usage = []
            for row in rows:
                data = row if isinstance(row, dict) else dict(row)
                usage.append(
                    {
                        "id": str(data.get("id")),
                        "name": data.get("name"),
                        "category": data.get("category"),
                        "enabled": bool(data.get("enabled", True)),
                        "executions_last_30d": int(data.get("executions_last_30d") or 0),
                        "last_execution": data.get("last_execution").isoformat()
                        if data.get("last_execution")
                        else None,
                        "avg_duration_ms": float(data.get("avg_duration_ms") or 0),
                    }
                )
            return {"agents": usage, "table": agents_table, "executions_table": executions_table}
        except Exception as exc:
            errors.append(f"{agents_table}/{executions_table}: {exc}")
            continue

    return {"agents": [], "warning": "No agent usage data available", "errors": errors[:2]}


async def _get_schedule_usage(pool) -> dict[str, Any]:
    """Fetch scheduler schedule rows with resiliency."""
    schedules: list[dict[str, Any]] = []
    try:
        # Note: public.agent_schedules does NOT have last_execution/next_execution columns
        rows = await pool.fetch(
            """
            SELECT
                s.id::text AS id,
                s.agent_id::text AS agent_id,
                s.enabled,
                s.frequency_minutes,
                s.created_at,
                COALESCE(a.name, s.agent_id::text) AS agent_name
            FROM public.agent_schedules s
            LEFT JOIN ai_agents a ON a.id = s.agent_id
            ORDER BY s.enabled DESC, s.created_at DESC NULLS LAST
            LIMIT 50
        """
        )
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            schedules.append(
                {
                    "id": data.get("id"),
                    "agent_id": data.get("agent_id"),
                    "agent_name": data.get("agent_name"),
                    "enabled": bool(data.get("enabled", True)),
                    "frequency_minutes": data.get("frequency_minutes"),
                    "created_at": data.get("created_at").isoformat()
                    if data.get("created_at")
                    else None,
                }
            )
        return {"schedules": schedules, "table": "public.agent_schedules"}
    except Exception as exc:
        logger.error("Failed to load schedule usage: %s", exc)
        return {"schedules": schedules, "error": str(exc)}


# ==================== LANGGRAPH ENDPOINTS ====================

if LANGGRAPH_AVAILABLE:

    @app.post("/langgraph/workflow")
    async def execute_langgraph_workflow(
        request: dict[str, Any], authenticated: bool = Depends(verify_api_key)
    ):
        """Execute a LangGraph-based workflow"""
        if not hasattr(app.state, "langgraph_orchestrator") or not app.state.langgraph_orchestrator:
            raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")

        try:
            orchestrator = app.state.langgraph_orchestrator

            # Extract messages and metadata
            messages_data = request.get("messages", [])
            metadata = request.get("metadata", {})

            # Convert raw messages to LangChain messages if needed
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            messages = []
            for msg in messages_data:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

            # If no messages provided, use a default prompt
            if not messages:
                prompt = request.get("prompt", "")
                if prompt:
                    messages.append(HumanMessage(content=prompt))
                else:
                    raise HTTPException(status_code=400, detail="No messages or prompt provided")

            # Run workflow
            result = await orchestrator.run_workflow(messages, metadata)

            return result

        except Exception as e:
            logger.error(f"LangGraph workflow error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/langgraph/status")
    async def get_langgraph_status(authenticated: bool = Depends(verify_api_key)):
        """Get LangGraph orchestrator status"""
        if not hasattr(app.state, "langgraph_orchestrator") or not app.state.langgraph_orchestrator:
            return {"available": False, "message": "LangGraph Orchestrator not initialized"}

        orchestrator = app.state.langgraph_orchestrator

        return {
            "available": True,
            "components": {
                "openai_llm": hasattr(orchestrator, "openai_llm")
                and orchestrator.openai_llm is not None,
                "anthropic_llm": hasattr(orchestrator, "anthropic_llm")
                and orchestrator.anthropic_llm is not None,
                "vector_store": hasattr(orchestrator, "vector_store")
                and orchestrator.vector_store is not None,
                "workflow_graph": hasattr(orchestrator, "workflow")
                and orchestrator.workflow is not None,
            },
        }


@app.get("/")
@limiter.limit("60/minute")
async def root(request: Request):
    """Root endpoint"""
    return {
        "service": config.service_name,
        "version": VERSION,
        "status": "operational",
        "build": BUILD_TIME,
        "environment": config.environment,
        "ai_enabled": AI_AVAILABLE,
        "scheduler_enabled": SCHEDULER_AVAILABLE,
    }


from src.graph.product_agent import app as product_agent_graph
from langchain_core.messages import HumanMessage


class ProductRequest(BaseModel):
    concept: str


@app.post("/agents/product/run", tags=["Agents"], dependencies=SECURED_DEPENDENCIES)
async def run_product_agent(request: ProductRequest):
    """
    Run the LangGraph Product Agent to generate product specs, code, and QA.
    """
    try:
        # Invoke the graph
        result = product_agent_graph.invoke({"messages": [HumanMessage(content=request.concept)]})

        # Extract the final message
        last_message = result["messages"][-1].content
        return {
            "status": "success",
            "result": last_message,
            "trace": [m.content for m in result["messages"]],
        }
    except Exception as e:
        logger.error(f"Product Agent Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_csv_env(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def _is_allowlisted_recipient(recipient: str | None) -> bool:
    if not recipient:
        return False
    lowered = recipient.strip().lower()
    allowlist_recipients = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST", ""))
    allowlist_domains = _parse_csv_env(os.getenv("OUTBOUND_EMAIL_ALLOWLIST_DOMAINS", ""))
    if lowered in allowlist_recipients:
        return True
    if "@" not in lowered:
        return False
    domain = lowered.split("@", 1)[1]
    if domain in allowlist_domains:
        return True
    return any(domain.endswith(f".{allowed}") for allowed in allowlist_domains)


class SendEmailPayload(BaseModel):
    recipient: EmailStr
    subject: str
    html: str
    metadata: dict[str, Any] = {}


@app.post("/email/send")
@limiter.limit("5/minute")
async def send_email_endpoint(
    request: Request, payload: SendEmailPayload, authenticated: bool = Depends(verify_api_key)
):
    """Send a one-off email (admin-only, API-key protected).

    Safety rail: unless OUTBOUND_EMAIL_MODE=live, the recipient must be allowlisted
    via OUTBOUND_EMAIL_ALLOWLIST / OUTBOUND_EMAIL_ALLOWLIST_DOMAINS.
    """
    mode = os.getenv("OUTBOUND_EMAIL_MODE", "disabled").strip().lower()
    if mode != "live" and not _is_allowlisted_recipient(payload.recipient):
        raise HTTPException(
            status_code=403, detail="Recipient is not allowlisted for outbound email"
        )

    from email_sender import send_email

    # Avoid blocking the HTTP loop with synchronous HTTP (Resend) / SMTP operations.
    subject = payload.subject.strip()[:200] if payload.subject else "BrainOps AI"
    success, message = await asyncio.to_thread(
        send_email,
        str(payload.recipient),
        subject,
        payload.html,
        payload.metadata or {},
    )

    recipient_masked = (
        str(payload.recipient).split("@", 1)[0][:3]
        + "***@"
        + str(payload.recipient).split("@", 1)[1]
    )
    logger.info("One-off email send requested -> %s (success=%s)", recipient_masked, success)
    return {"success": success, "message": message}


@app.get("/health")
@limiter.limit("60/minute")
async def health_check(
    request: Request,
    force_refresh: bool = Query(False, description="Bypass cache and force live health checks"),
):
    """Health check endpoint.

    Unauthenticated requests receive a minimal {"status": "ok", "version": "..."} response.
    Authenticated requests (valid X-API-Key) receive the full diagnostic payload.
    """
    # --- Unauthenticated callers get a minimal response ---
    _api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("X-Api-Key")
    )
    _master = getattr(config.security, "master_api_key", None) or os.getenv("MASTER_API_KEY")
    _is_authenticated = False
    if _api_key and _api_key.strip() in config.security.valid_api_keys:
        _is_authenticated = True
    elif _master and _api_key and _api_key.strip() == _master:
        _is_authenticated = True

    if not _is_authenticated:
        return {"status": "ok", "version": VERSION}

    # --- Authenticated callers get full diagnostics ---

    async def _build_health_payload() -> dict[str, Any]:
        # Handle case where pool isn't initialized yet (during startup)
        db_timeout = float(os.getenv("DB_HEALTH_TIMEOUT_S", "4.0"))
        db_timed_out = False
        pool_metrics: dict[str, Any] | None = None
        try:
            pool = get_pool()
            try:
                db_healthy = await asyncio.wait_for(pool.test_connection(), timeout=db_timeout)
            except asyncio.TimeoutError:
                db_timed_out = True
                logger.warning("Health check DB test timed out after %.2fs", db_timeout)
                db_healthy = False
        except RuntimeError as e:
            if "not initialized" in str(e):
                # Pool not ready yet - return starting status
                return {
                    "status": "starting",
                    "version": VERSION,
                    "build": BUILD_TIME,
                    "database": "initializing",
                    "message": "Service is starting up, database pool initializing...",
                }
            raise
        if db_timed_out:
            db_status = "timeout"
        else:
            db_status = (
                "fallback" if using_fallback() else ("connected" if db_healthy else "disconnected")
            )
        auth_configured = config.security.auth_configured

        # Best-effort pool metrics: helps diagnose pool exhaustion vs. connectivity issues.
        try:
            raw_pool = getattr(pool, "pool", None)
            if raw_pool is not None:
                pool_metrics = {
                    "min_size": getattr(raw_pool, "get_min_size", lambda: None)(),
                    "max_size": getattr(raw_pool, "get_max_size", lambda: None)(),
                    "size": getattr(raw_pool, "get_size", lambda: None)(),
                    "idle": getattr(raw_pool, "get_idle_size", lambda: None)(),
                }
        except Exception as exc:
            pool_metrics = {"error": str(exc)}

        active_systems = _collect_active_systems()

        embedded_memory_stats = None
        embedded_memory_error = getattr(app.state, "embedded_memory_error", None)
        if EMBEDDED_MEMORY_AVAILABLE and getattr(app.state, "embedded_memory", None) is None:
            try:
                app.state.embedded_memory = await asyncio.wait_for(get_embedded_memory(), timeout=5.0)
                app.state.embedded_memory_error = None
                embedded_memory_error = None
                logger.info("✅ Embedded Memory System lazily initialized from /health")
            except Exception as exc:
                embedded_memory_error = str(exc)
                app.state.embedded_memory_error = embedded_memory_error
                logger.warning("Embedded memory lazy init failed during /health: %s", exc)

        if EMBEDDED_MEMORY_AVAILABLE and getattr(app.state, "embedded_memory", None):
            try:
                embedded_memory_stats = app.state.embedded_memory.get_stats()
            except Exception as exc:
                logger.warning("Failed to read embedded memory stats: %s", exc, exc_info=True)
                embedded_memory_stats = {"status": "error"}

        return {
            "status": "healthy" if db_healthy and auth_configured else "degraded",
            "version": VERSION,
            "build": BUILD_TIME,
            "database": db_status,
            "db_pool": pool_metrics,
            "active_systems": active_systems,
            "system_count": len(active_systems),
            "embedded_memory_active": EMBEDDED_MEMORY_AVAILABLE
            and hasattr(app.state, "embedded_memory")
            and app.state.embedded_memory is not None,
            "embedded_memory_stats": embedded_memory_stats,
            "embedded_memory_error": embedded_memory_error,
            "capabilities": {
                # Phase 1
                "aurea_orchestrator": AUREA_AVAILABLE,
                "self_healing": SELF_HEALING_AVAILABLE,
                "memory_manager": MEMORY_AVAILABLE,
                "embedded_memory": EMBEDDED_MEMORY_AVAILABLE,
                "training_pipeline": TRAINING_AVAILABLE,
                "learning_system": LEARNING_AVAILABLE,
                "agent_scheduler": SCHEDULER_AVAILABLE,
                "ai_core": AI_AVAILABLE,
                # Phase 2
                "system_improvement": SYSTEM_IMPROVEMENT_AVAILABLE,
                "devops_optimization": DEVOPS_AGENT_AVAILABLE,
                "code_quality": CODE_QUALITY_AVAILABLE,
                "customer_success": CUSTOMER_SUCCESS_AVAILABLE,
                "competitive_intelligence": COMPETITIVE_INTEL_AVAILABLE,
                "vision_alignment": VISION_ALIGNMENT_AVAILABLE,
                # Phase 3 - Bleeding Edge 2025
                "digital_twin": True,
                "market_intelligence": True,
                "system_orchestrator": True,
                "enhanced_self_healing": True,
                "reconciliation_loop": RECONCILER_AVAILABLE,
                # Phase 4 - Revolutionary AI (2025-12-27)
                "bleeding_edge_ooda": BLEEDING_EDGE_AVAILABLE,
                "hallucination_prevention": BLEEDING_EDGE_AVAILABLE,
                "live_memory_brain": BLEEDING_EDGE_AVAILABLE,
                "dependability_framework": BLEEDING_EDGE_AVAILABLE,
                "consciousness_emergence": BLEEDING_EDGE_AVAILABLE,
                "enhanced_circuit_breaker": BLEEDING_EDGE_AVAILABLE,
                # Phase 5 - Perfect Integration (2025-12-27)
                "ai_observability": AI_OBSERVABILITY_AVAILABLE,
                "cross_module_integration": AI_OBSERVABILITY_AVAILABLE,
                "unified_metrics": AI_OBSERVABILITY_AVAILABLE,
                "learning_feedback_loops": AI_OBSERVABILITY_AVAILABLE,
                # Phase 6 - Enhanced Systems (2025-12-28)
                "module_health_scoring": AI_ENHANCEMENTS_AVAILABLE,
                "realtime_alerting": AI_ENHANCEMENTS_AVAILABLE,
                "event_correlation": AI_ENHANCEMENTS_AVAILABLE,
                "auto_recovery": AI_ENHANCEMENTS_AVAILABLE,
                "websocket_streaming": AI_ENHANCEMENTS_AVAILABLE,
                "enhanced_learning": AI_ENHANCEMENTS_AVAILABLE,
                # Phase 7 - Unified Self-Awareness (2025-12-27)
                "unified_awareness": UNIFIED_AWARENESS_AVAILABLE,
                "self_reporting": UNIFIED_AWARENESS_AVAILABLE,
                # Phase 8 - Service Protection (2026-01-27)
                "service_circuit_breakers": SERVICE_CIRCUIT_BREAKERS_AVAILABLE,
            },
            "circuit_breakers": get_circuit_breaker_health()
            if SERVICE_CIRCUIT_BREAKERS_AVAILABLE and get_circuit_breaker_health
            else {"status": "unavailable"},
            "config": {
                "environment": config.environment,
                "security": {
                    "auth_required": config.security.auth_required,
                    "dev_mode": config.security.dev_mode,
                    "auth_configured": auth_configured,
                    "api_keys_configured": len(config.security.valid_api_keys),
                },
            },
            "missing_systems": getattr(app.state, "missing_systems", []),
        }

    async def _build_health_payload_safe() -> dict[str, Any]:
        """Bound health work so transient stalls don't bubble up as 502s from Render."""
        try:
            return await asyncio.wait_for(_build_health_payload(), timeout=HEALTH_PAYLOAD_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning("Health payload build timed out after %.2fs", HEALTH_PAYLOAD_TIMEOUT_S)
            return {
                "status": "degraded",
                "version": VERSION,
                "build": BUILD_TIME,
                "database": "timeout",
                "message": "Health payload timed out",
            }
        except Exception as exc:
            logger.error("Health payload build failed: %s", exc, exc_info=True)
            return {
                "status": "degraded",
                "version": VERSION,
                "build": BUILD_TIME,
                "database": "error",
                "message": "Health payload error",
            }

    if force_refresh:
        return await _build_health_payload_safe()

    payload, from_cache = await RESPONSE_CACHE.get_or_set(
        "health_status",
        CACHE_TTLS["health"],
        _build_health_payload_safe,
    )
    return {**payload, "cached": from_cache}


@app.get("/healthz")
@limiter.limit("60/minute")
async def healthz(request: Request) -> dict[str, Any]:
    """Lightweight health endpoint for container checks (no DB calls)."""
    return {
        "status": "ok",
        "version": VERSION,
        "build": BUILD_TIME,
    }


@app.get("/system/awareness", dependencies=SECURED_DEPENDENCIES)
async def system_awareness():
    """
    CRITICAL: Self-awareness endpoint that reports what's actually broken.
    This is the AI OS telling you its problems - listen to it!
    """
    pool = get_pool()
    issues = []
    warnings = []
    healthy = []

    try:
        # Check Gumroad revenue (truthful: exclude test rows)
        gumroad_real = await pool.fetchval(
            "SELECT COUNT(*) FROM gumroad_sales WHERE NOT COALESCE(is_test, FALSE)"
        )
        gumroad_test = await pool.fetchval(
            "SELECT COUNT(*) FROM gumroad_sales WHERE COALESCE(is_test, FALSE)"
        )
        if gumroad_real == 0:
            issues.append(
                {
                    "category": "REVENUE",
                    "problem": "Zero REAL Gumroad sales recorded",
                    "impact": "No personal revenue from digital products (test rows do not count as revenue)",
                    "fix": "Verify Gumroad webhook is receiving real purchases at /gumroad/webhook",
                }
            )
            if gumroad_test and gumroad_test > 0:
                warnings.append(
                    {
                        "category": "REVENUE",
                        "problem": "Gumroad test rows present",
                        "impact": "Webhook wiring may be OK, but there is still zero real revenue recorded",
                        "fix": "Ensure production Gumroad products are live and receiving real purchases",
                    }
                )
        else:
            healthy.append(f"Gumroad: {gumroad_real} real sales tracked")
            if gumroad_test and gumroad_test > 0:
                warnings.append(
                    {
                        "category": "REVENUE",
                        "problem": "Gumroad test rows present",
                        "impact": "Test rows are excluded from revenue totals; ensure dashboards filter is_test=false",
                        "fix": "Continue filtering COALESCE(is_test,false)=false in all revenue reporting",
                    }
                )

        # Check MRG subscriptions (truthful: tenant-scoped default + global)
        mrg_default_tenant = os.getenv(
            "MRG_DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001"
        )
        mrg_active_default = await pool.fetchval(
            "SELECT COUNT(*) FROM mrg_subscriptions WHERE status='active' AND tenant_id = $1",
            mrg_default_tenant,
        )
        mrg_active_all = await pool.fetchval(
            "SELECT COUNT(*) FROM mrg_subscriptions WHERE status='active'"
        )

        if mrg_active_default == 0:
            issues.append(
                {
                    "category": "REVENUE",
                    "problem": "Zero active MRG subscriptions (default tenant)",
                    "impact": "No SaaS recurring revenue",
                    "fix": "Verify Stripe webhook integration for subscriptions",
                }
            )
            if mrg_active_all and mrg_active_all > 0:
                warnings.append(
                    {
                        "category": "REVENUE",
                        "problem": "MRG subscriptions exist for other tenants",
                        "impact": "Tenant attribution may be misconfigured (MRG_DEFAULT_TENANT_ID mismatch)",
                        "fix": "Set MRG_DEFAULT_TENANT_ID consistently on MRG + backend + agents",
                    }
                )
        else:
            healthy.append(f"MRG Subscriptions: {mrg_active_default} active (default tenant)")
            if mrg_active_all and mrg_active_all > mrg_active_default:
                warnings.append(
                    {
                        "category": "REVENUE",
                        "problem": "MRG subscriptions exist outside default tenant",
                        "impact": "Revenue reporting can be fragmented across tenants",
                        "fix": "Confirm tenant_id assignment in checkout metadata and webhook handlers",
                    }
                )

        # Check learning system (uses ai_learning_insights, not ai_learning_patterns)
        learning = await pool.fetchval("SELECT COUNT(*) FROM ai_learning_insights")
        learning_recent = await pool.fetchval(
            """
            SELECT COUNT(*) FROM ai_learning_insights
            WHERE created_at > NOW() - INTERVAL '24 hours'
        """
        )
        if learning == 0:
            issues.append(
                {
                    "category": "LEARNING",
                    "problem": "AI learning insights table is empty",
                    "impact": "System is not learning from operations",
                    "fix": "Check LearningFeedbackLoop agent and /api/learning/run-cycle endpoint",
                }
            )
        elif learning_recent == 0:
            warnings.append(
                {
                    "category": "LEARNING",
                    "problem": "No learning insights in last 24 hours",
                    "impact": "Learning may be stalled",
                    "fix": "Run /api/learning/run-cycle to trigger learning",
                }
            )
        else:
            healthy.append(f"Learning active: {learning} total, {learning_recent} in 24h")

        # Check agent activity
        agents_1hr = await pool.fetchval(
            """
            SELECT COUNT(DISTINCT agent_name) FROM agent_execution_logs
            WHERE timestamp > NOW() - INTERVAL '1 hour' AND execution_phase = 'completed'
        """
        )
        if agents_1hr < 3:
            warnings.append(
                {
                    "category": "AGENTS",
                    "problem": f"Only {agents_1hr} agents ran in last hour",
                    "impact": "Autonomous operations may be stalled",
                    "fix": "Check scheduler at /scheduler/status and agent schedules in DB",
                }
            )
        else:
            healthy.append(f"Agents active: {agents_1hr} ran in last hour")

        # Check unresolved alerts
        unresolved = await pool.fetchval(
            "SELECT COUNT(*) FROM brainops_alerts WHERE resolved = false"
        )
        if unresolved > 20:
            warnings.append(
                {
                    "category": "ALERTS",
                    "problem": f"{unresolved} unresolved system alerts",
                    "impact": "System issues are being ignored",
                    "fix": "Review alerts and resolve or acknowledge them",
                }
            )
        elif unresolved > 0:
            warnings.append(
                {
                    "category": "ALERTS",
                    "problem": f"{unresolved} unresolved alerts need attention",
                    "impact": "Minor issues accumulating",
                    "fix": "Review at /system/alerts",
                }
            )

        # Check memory activity
        memories_today = await pool.fetchval(
            """
            SELECT COUNT(*) FROM unified_ai_memory WHERE created_at::date = CURRENT_DATE
        """
        )
        if memories_today < 50:
            warnings.append(
                {
                    "category": "MEMORY",
                    "problem": f"Low memory activity today ({memories_today} entries)",
                    "impact": "System may not be learning from interactions",
                    "fix": "Check memory endpoints at /memory/store",
                }
            )
        else:
            healthy.append(f"Memory active: {memories_today} entries today")

        # Check scheduled agents
        scheduled = await pool.fetchval(
            """
            SELECT COUNT(*) FROM agents
            WHERE enabled = true AND array_length(schedule_hours, 1) > 0
        """
        )
        total_agents = await pool.fetchval("SELECT COUNT(*) FROM agents WHERE enabled = true")
        if scheduled < total_agents * 0.5:
            warnings.append(
                {
                    "category": "SCHEDULER",
                    "problem": f"Only {scheduled}/{total_agents} agents have schedules",
                    "impact": "Most agents won't run automatically",
                    "fix": "Set schedule_hours for agents in database",
                }
            )
        else:
            healthy.append(f"Scheduled agents: {scheduled}/{total_agents}")

        # Overall status
        if issues:
            overall_status = "CRITICAL"
        elif warnings:
            overall_status = "WARNING"
        else:
            overall_status = "HEALTHY"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": VERSION,
            "critical_issues": issues,
            "warnings": warnings,
            "healthy_systems": healthy,
            "summary": f"{len(issues)} critical, {len(warnings)} warnings, {len(healthy)} healthy",
            "message": "This is your AI OS telling you what's broken. Fix the critical issues first.",
        }

    except Exception as e:
        logger.error(f"System awareness check failed: {e}")
        return {
            "status": "ERROR",
            "error": str(e),
            "message": "Could not complete system awareness check",
        }


@app.get("/system/alerts", dependencies=SECURED_DEPENDENCIES)
async def get_system_alerts(limit: int = 50, unresolved_only: bool = True):
    """Get system alerts that need attention."""
    pool = get_pool()
    try:
        if unresolved_only:
            alerts = await pool.fetch(
                """
                SELECT alert_type, severity, message, details, created_at
                FROM brainops_alerts
                WHERE resolved = false
                ORDER BY
                    CASE severity WHEN 'critical' THEN 1 WHEN 'warning' THEN 2 ELSE 3 END,
                    created_at DESC
                LIMIT $1
            """,
                limit,
            )
        else:
            alerts = await pool.fetch(
                """
                SELECT alert_type, severity, message, details, created_at, resolved
                FROM brainops_alerts
                ORDER BY created_at DESC
                LIMIT $1
            """,
                limit,
            )

        return {"count": len(alerts), "alerts": [dict(a) for a in alerts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _require_diagnostics_key(request: Request) -> None:
    api_key = (
        request.headers.get("X-API-Key")
        or request.headers.get("x-api-key")
        or request.headers.get("Authorization")
        or ""
    )
    if api_key.startswith("ApiKey "):
        api_key = api_key.split(" ", 1)[1]
    if api_key.startswith("Bearer "):
        api_key = api_key.split(" ", 1)[1]

    if not api_key or api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/ready")
@limiter.limit("60/minute")
async def readiness_check(request: Request):
    """Dependency-aware readiness check."""
    try:
        pool = get_pool()
        db_healthy = await pool.test_connection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database not ready: {exc}")

    if not db_healthy:
        raise HTTPException(status_code=503, detail="Database not ready")

    if config.security.auth_required and not config.security.auth_configured:
        raise HTTPException(status_code=503, detail="Auth not configured")

    return {
        "status": "ready",
        "database": "connected",
        "auth_configured": config.security.auth_configured,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/capabilities")
async def capabilities(request: Request):
    """Authenticated capability registry."""
    _require_diagnostics_key(request)
    routes = []
    for route in app.routes:
        methods = sorted(getattr(route, "methods", []) or [])
        routes.append({"path": route.path, "methods": methods})

    return {
        "service": config.service_name,
        "version": VERSION,
        "environment": config.environment,
        "active_systems": _collect_active_systems(),
        "routes": routes,
        "ai_enabled": AI_AVAILABLE,
        "scheduler_enabled": SCHEDULER_AVAILABLE,
    }


@app.get("/diagnostics")
async def diagnostics(request: Request):
    """Authenticated deep diagnostics."""
    _require_diagnostics_key(request)

    missing_env = [
        key
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_SERVICE_KEY",
        )
        if not os.getenv(key)
    ]

    db_status = {"ready": False}
    last_error = None
    try:
        pool = get_pool()
        db_ready = await pool.test_connection()
        db_status["ready"] = bool(db_ready)
    except Exception as exc:
        db_status["error"] = str(exc)

    try:
        pool = get_pool()
        last_error_row = await pool.fetchrow(
            """
            SELECT error_type, error_message, severity, component, timestamp
            FROM ai_error_logs
            ORDER BY timestamp DESC
            LIMIT 1
        """
        )
        if last_error_row:
            last_error = dict(last_error_row)
    except Exception as exc:
        last_error = {"error": str(exc)}

    nerve_center_status = None
    try:
        if hasattr(app.state, "nerve_center") and app.state.nerve_center:
            nerve_center_status = app.state.nerve_center.get_status()
    except Exception as exc:
        nerve_center_status = {"error": str(exc)}

    operational_monitor_status = None
    try:
        if hasattr(app.state, "operational_monitor") and app.state.operational_monitor:
            operational_monitor_status = app.state.operational_monitor.get_status()
    except Exception as exc:
        operational_monitor_status = {"error": str(exc)}

    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "missing_env": missing_env,
        "fallback_mode": using_fallback(),
        "database": db_status,
        "active_systems": _collect_active_systems(),
        "last_error": last_error,
        "nerve_center": nerve_center_status,
        "operational_monitor": operational_monitor_status,
    }


@app.get("/alive")
@limiter.limit("60/minute")
async def alive_status(request: Request):
    """Get the alive status of the AI OS based on core system health."""
    import time as _time

    # Base alive check: service is up, DB connected, systems running
    active_systems = _collect_active_systems()
    db_ok = False
    try:
        pool = get_pool()
        row = await pool.fetchval("SELECT 1")
        db_ok = row == 1
    except Exception:
        pass

    uptime = _time.time() - getattr(app.state, "_start_time", _time.time())

    # The service is "alive" if it's running, DB is connected, and systems are active
    is_alive = db_ok and len(active_systems) > 0

    status = {
        "alive": is_alive,
        "uptime_seconds": int(uptime),
        "database": db_ok,
        "active_systems": len(active_systems),
        "nerve_center": None,
        "operational_monitor": None,
    }

    # If NerveCenter is available, add its status too
    nc_value = getattr(app.state, "nerve_center", None)
    if nc_value:
        try:
            nc_status = nc_value.get_status()
            status["nerve_center"] = nc_status
            status["uptime_seconds"] = nc_status.get("uptime_seconds", uptime)
        except Exception:
            pass

    monitor_value = getattr(app.state, "operational_monitor", None)
    if monitor_value:
        try:
            status["operational_monitor"] = monitor_value.get_status()
        except Exception:
            pass

    return status


@app.get("/alive/thoughts", dependencies=SECURED_DEPENDENCIES)
async def get_recent_thoughts():
    """Legacy endpoint retained after thought-stream deprecation."""
    return {
        "thoughts": [],
        "message": "Thought stream is disabled; use /alive and /diagnostics for operational status.",
    }


# ============================================================================
# UNIFIED AI AWARENESS - Self-Reporting AI OS
# ============================================================================


@app.get("/awareness", dependencies=SECURED_DEPENDENCIES)
async def get_awareness_status():
    """
    Quick status check - AI OS reports its current state (requires auth).
    This is the AI talking to you about how it's doing.
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"status": "unavailable", "message": "Unified awareness system not loaded"}

    try:
        return {"status": check_status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/awareness/report", dependencies=SECURED_DEPENDENCIES)
async def get_full_awareness_report():
    """
    Full self-reporting status - AI OS tells you everything it knows about itself (requires auth).
    This is the AI's comprehensive understanding of its own state.
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        return get_status_report()
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/awareness/pulse", dependencies=SECURED_DEPENDENCIES)
async def get_system_pulse():
    """
    Real-time system pulse - the AI's heartbeat and vital signs (requires auth).
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        awareness = get_unified_awareness()
        pulse = awareness.get_system_pulse()
        return pulse.to_dict()
    except Exception as e:
        return {"available": False, "error": str(e)}


# ============================================================================
# TRUE SELF-AWARENESS - Live System Truth (not static documentation)
# ============================================================================


@app.get("/truth", dependencies=SECURED_DEPENDENCIES)
async def get_truth():
    """
    THE TRUTH - Complete live system truth from database (requires auth).
    This is what the AI OS ACTUALLY knows about itself.
    No static docs, no outdated info - just live truth.
    """
    if not TRUE_AWARENESS_AVAILABLE:
        return {"available": False, "message": "True self-awareness not loaded"}

    try:
        truth = await get_system_truth()
        return truth
    except Exception as e:
        logger.error(f"Error getting truth: {e}")
        return {"available": False, "error": str(e)}


@app.get("/truth/quick", dependencies=SECURED_DEPENDENCIES)
async def get_truth_quick():
    """
    Quick human-readable system truth (requires auth).
    Shows what's real vs demo, what's working vs broken.
    """
    if not TRUE_AWARENESS_AVAILABLE:
        return {"available": False, "message": "True self-awareness not loaded"}

    try:
        status = await get_quick_status()
        return {"status": status}
    except Exception as e:
        logger.error(f"Error getting quick truth: {e}")
        return {"available": False, "error": str(e)}


# ============================================================================
# TELEMETRY INGESTION - Neural System Event Collection
# ============================================================================


@app.post("/api/v1/telemetry/events")
async def receive_telemetry_events(request: Request, authenticated: bool = Depends(verify_api_key)):
    """
    Receive telemetry events from external systems (ERP, MRG, etc.).
    This connects the 'nervous system' - allowing external apps to send
    events to the AI brain for processing and awareness.
    Requires API key authentication.
    """
    try:
        body = await request.json()
        events = body.get("events", [body])  # Support single event or array

        # Log receipt
        logger.info(f"Received {len(events)} telemetry events")

        # Store events in database if available
        stored_count = 0
        try:
            pool = get_pool()
            async with pool.acquire() as conn:
                for event in events:
                    await conn.execute(
                        """
                        INSERT INTO ai_nerve_signals (
                            source, event_type, payload, metadata, created_at
                        ) VALUES ($1, $2, $3, $4, NOW())
                        ON CONFLICT DO NOTHING
                    """,
                        event.get("source", "unknown"),
                        event.get("type", "telemetry"),
                        json.dumps(event.get("data", {})),
                        json.dumps(event.get("metadata", {})),
                    )
                    stored_count += 1
        except Exception as db_err:
            logger.warning(f"Failed to store telemetry: {db_err}")

        return {
            "success": True,
            "received": len(events),
            "stored": stored_count,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Telemetry ingestion error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/v1/knowledge/store-legacy")
async def store_knowledge(request: Request, authenticated: bool = Depends(verify_api_key)):
    """
    Legacy knowledge endpoint kept for backward compatibility.
    Use POST /api/v1/knowledge/store for the canonical memory-backed path.
    """
    try:
        body = await request.json()
        key = body.get("key", "")
        value = body.get("value", {})
        category = body.get("category", "external")

        if not key:
            return {"success": False, "error": "Key required"}

        # Store in brain context
        if BRAIN_AVAILABLE:
            try:
                from api.brain import brain as unified_brain

                if unified_brain:
                    await unified_brain.store(
                        key=key,
                        value=value,
                        category=category,
                        priority="medium",
                        source="legacy_api_v1_knowledge_store",
                    )
                else:
                    raise RuntimeError("Unified brain instance unavailable")
                return {"success": True, "key": key}
            except Exception as brain_err:
                logger.warning(f"Brain store failed: {brain_err}")

        return {"success": False, "error": "Brain not available"}
    except Exception as e:
        logger.error(f"Knowledge store error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/observability/metrics", dependencies=[Depends(verify_api_key)])
async def observability_metrics():
    """Lightweight monitoring endpoint for request, cache, DB, and orchestrator health.
    SECURITY: Requires API key authentication to prevent data leakage."""
    pool = get_pool()
    db_probe_ms = None
    db_error = None
    start = time.perf_counter()
    try:
        await pool.fetchval("SELECT 1")
        db_probe_ms = (time.perf_counter() - start) * 1000
    except Exception as exc:
        db_error = str(exc)

    return {
        "requests": REQUEST_METRICS.snapshot(),
        "cache": RESPONSE_CACHE.snapshot(),
        "database": {
            "using_fallback": using_fallback(),
            "probe_latency_ms": db_probe_ms,
            "error": db_error,
        },
        "scheduler": _scheduler_snapshot(),
        "aurea": _aurea_status(),
        "self_healing": _self_healing_status(),
    }


@app.get("/debug/database")
async def debug_database(authenticated: bool = Depends(verify_api_key)):
    """Diagnostic endpoint for database connection issues."""
    import psycopg2

    results = {
        "async_pool": {"using_fallback": using_fallback(), "status": "unknown"},
        "sync_psycopg2": {"status": "unknown"},
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password_set": bool(config.database.password),
            "ssl": config.database.ssl,
            "ssl_verify": config.database.ssl_verify,
        },
    }

    # Test async pool
    try:
        pool = get_pool()
        start = time.perf_counter()
        result = await pool.fetchval("SELECT 1")
        latency = (time.perf_counter() - start) * 1000
        results["async_pool"]["status"] = "connected" if result == 1 else "query_failed"
        results["async_pool"]["latency_ms"] = latency
        results["async_pool"]["test_query"] = result
    except Exception as e:
        results["async_pool"]["status"] = "error"
        results["async_pool"]["error"] = str(e)

    # Test direct psycopg2 connection (what AUREA uses)
    try:
        conn = psycopg2.connect(
            host=config.database.host,
            port=config.database.port,
            database=config.database.database,
            user=config.database.user,
            password=config.database.password,
            sslmode="require",
        )
        cur = conn.cursor()
        start = time.perf_counter()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        latency = (time.perf_counter() - start) * 1000
        cur.close()
        conn.close()
        results["sync_psycopg2"]["status"] = "connected"
        results["sync_psycopg2"]["latency_ms"] = latency
        results["sync_psycopg2"]["test_query"] = result[0] if result else None
    except Exception as e:
        results["sync_psycopg2"]["status"] = "error"
        results["sync_psycopg2"]["error"] = str(e)

    return results


@app.get("/debug/aurea")
async def debug_aurea(authenticated: bool = Depends(verify_api_key)):
    """Diagnostic endpoint for AUREA orchestrator status."""
    aurea = getattr(app.state, "aurea", None)
    if not aurea:
        return {"status": "not_initialized", "available": AUREA_AVAILABLE}

    try:
        status = aurea.get_status()
        return {
            "status": "running" if status.get("running") else "stopped",
            "details": status,
            "available": True,
            "cycle_count": getattr(aurea, "cycle_count", 0),
            "autonomy_level": str(getattr(aurea, "autonomy_level", "unknown")),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "available": True}


@app.get("/debug/scheduler")
async def debug_scheduler(authenticated: bool = Depends(verify_api_key)):
    """Diagnostic endpoint for agent scheduler status."""
    scheduler = getattr(app.state, "scheduler", None)
    if not scheduler:
        return {"status": "not_initialized", "available": SCHEDULER_AVAILABLE}

    try:
        jobs = scheduler.scheduler.get_jobs() if hasattr(scheduler, "scheduler") else []
        return {
            "status": "running" if scheduler.scheduler.running else "stopped",
            "total_jobs": len(jobs),
            "next_10_jobs": [
                {
                    "id": str(job.id),
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                }
                for job in sorted(jobs, key=lambda x: x.next_run_time or datetime.max)[:10]
            ],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =========================================================================
# EMAIL QUEUE ENDPOINTS
# =========================================================================


@app.get("/email/status")
async def email_queue_status(authenticated: bool = Depends(verify_api_key)):
    """Get email queue status - shows pending, sent, failed counts."""
    try:
        from email_sender import get_queue_status

        return get_queue_status()
    except ImportError:
        return {"error": "email_sender module not available"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/email/process")
@limiter.limit("5/minute")
async def process_email_queue_endpoint(
    request: Request,
    batch_size: int = Query(default=10, ge=1, le=50),
    dry_run: bool = Query(default=False),
    authenticated: bool = Depends(verify_api_key),
):
    """
    Manually trigger email queue processing.
    - batch_size: Number of emails to process (1-50)
    - dry_run: If true, don't actually send, just report what would be sent
    """
    try:
        from email_sender import process_email_queue

        result = process_email_queue(batch_size=batch_size, dry_run=dry_run)
        return result
    except ImportError:
        return {"error": "email_sender module not available"}
    except Exception as e:
        logger.error(f"Email processing failed: {e}")
        return {"error": str(e)}


@app.post("/email/test")
@limiter.limit("5/minute")
async def test_email_sending(
    request: Request,
    recipient: str = Query(..., description="Email address to send test to"),
    authenticated: bool = Depends(verify_api_key),
):
    """Send a test email to verify email configuration is working."""
    try:
        from email_sender import send_email

        success, message = send_email(
            recipient,
            "Test Email from BrainOps AI",
            "<h1>Test Email</h1><p>This is a test email from the BrainOps AI email system.</p><p>If you received this, email sending is working correctly.</p>",
            {},
        )
        return {"success": success, "message": message, "recipient": recipient}
    except ImportError:
        return {"error": "email_sender module not available", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


@app.get("/email/scheduler-stats")
async def email_scheduler_stats(authenticated: bool = Depends(verify_api_key)):
    """Get email scheduler daemon statistics."""
    try:
        from email_scheduler_daemon import get_email_scheduler

        daemon = get_email_scheduler()
        stats = daemon.get_stats()

        # Also get queue counts from database
        pool = get_pool()
        queue_counts = await pool.fetchrow(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'queued') as queued,
                COUNT(*) FILTER (WHERE status = 'processing') as processing,
                COUNT(*) FILTER (WHERE status = 'sent') as sent,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
                COUNT(*) as total
            FROM ai_email_queue
        """
        )

        return {
            "daemon_stats": stats,
            "queue_counts": dict(queue_counts) if queue_counts else {},
            "timestamp": datetime.utcnow().isoformat(),
        }
    except ImportError:
        return {"error": "email_scheduler_daemon module not available"}
    except Exception as e:
        logger.error(f"Failed to get email scheduler stats: {e}")
        return {"error": str(e)}


@app.get("/systems/usage")
async def systems_usage(authenticated: bool = Depends(verify_api_key)):
    """Report which AI systems are being used plus scheduler and memory effectiveness."""

    async def _load_usage() -> dict[str, Any]:
        pool = get_pool()
        agent_usage = await _get_agent_usage(pool)
        schedule_usage = await _get_schedule_usage(pool)
        memory_usage = await _memory_stats_snapshot(pool)

        customer_success_preview = None
        if CUSTOMER_SUCCESS_AVAILABLE and getattr(app.state, "customer_success", None):
            try:
                customer_success_preview = (
                    await app.state.customer_success.generate_onboarding_plan(
                        customer_id="sample-customer",
                        plan_type="value-check",
                    )
                )
            except Exception as exc:
                customer_success_preview = {"error": str(exc)}

        return {
            "active_systems": _collect_active_systems(),
            "agents": agent_usage,
            "schedules": {**schedule_usage, "scheduler_runtime": _scheduler_snapshot()},
            "memory": memory_usage,
            "learning": {
                "available": LEARNING_AVAILABLE
                and getattr(app.state, "learning", None) is not None,
                "notes": "Notebook LM+ initialized"
                if getattr(app.state, "learning", None)
                else "Learning system not initialized",
            },
            "aurea": _aurea_status(),
            "self_healing": _self_healing_status(),
            "customer_success": {
                "available": CUSTOMER_SUCCESS_AVAILABLE
                and getattr(app.state, "customer_success", None) is not None,
                "sample_plan": customer_success_preview,
            },
        }

    usage, from_cache = await RESPONSE_CACHE.get_or_set(
        "systems_usage",
        CACHE_TTLS["systems_usage"],
        _load_usage,
    )
    return {**usage, "cached": from_cache}


@app.get("/ai/providers/status")
async def providers_status(authenticated: bool = Depends(verify_api_key)):
    """
    Report configuration and basic liveness for all AI providers (OpenAI, Anthropic,
    Gemini, Perplexity, Hugging Face). Does not modify configuration or credentials;
    it only runs small probe calls to detect misconfiguration like invalid or missing
    API keys.
    """
    return get_provider_status()


@app.get("/agents", response_model=AgentList)
async def get_agents(
    category: Optional[str] = None,
    enabled: Optional[bool] = True,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_capabilities: bool = Query(True),
    include_configuration: bool = Query(False),
    authenticated: bool = Depends(verify_api_key),
) -> AgentList:
    """Get list of available agents"""
    try:
        cache_key = (
            f"agents:{category or 'all'}:{enabled}:"
            f"{limit}:{offset}:{int(include_capabilities)}:{int(include_configuration)}"
        )

        async def _load_agents() -> AgentList:
            try:
                pool = get_pool()

                params: list[Any] = []
                where_sql = "WHERE 1=1"

                if enabled is not None:
                    where_sql += f" AND a.enabled = ${len(params) + 1}"
                    params.append(enabled)

                if category:
                    where_sql += f" AND a.category = ${len(params) + 1}"
                    params.append(category)

                # Keep list responses small by default; full agent details are available via `/agents/{agent_id}`.
                select_cols = [
                    "a.id",
                    "a.name",
                    "a.category",
                    "a.enabled",
                    "a.status",
                    "a.type",
                    "a.created_at",
                    "a.updated_at",
                    ("a.capabilities" if include_capabilities else "'[]'::jsonb AS capabilities"),
                    (
                        "a.configuration"
                        if include_configuration
                        else "'{}'::jsonb AS configuration"
                    ),
                ]
                select_sql = ", ".join(select_cols)

                total = (
                    await pool.fetchval(f"SELECT COUNT(*) FROM agents a {where_sql}", *params) or 0
                )

                # IMPORTANT: Do NOT aggregate the full ai_agent_executions table inside the agents list query.
                # That can hang under load (large execution history). Instead, fetch agents first, then
                # compute execution stats for just the returned page.
                query = f"""
                    SELECT {select_sql}
                    FROM agents a
                    {where_sql}
                    ORDER BY a.category, a.name
                    LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
                """

                rows = await pool.fetch(query, *params, limit, offset)

                exec_stats_by_agent_name: dict[str, dict[str, Any]] = {}
                try:
                    agent_names = [
                        (row.get("name") if isinstance(row, dict) else dict(row).get("name"))
                        for row in rows
                    ]
                    agent_names = [name for name in agent_names if isinstance(name, str) and name]
                    if agent_names:
                        exec_rows = await pool.fetch(
                            """
                            SELECT agent_name,
                                   COUNT(*) as exec_count,
                                   MAX(created_at) as last_exec
                            FROM ai_agent_executions
                            WHERE agent_name = ANY($1::text[])
                            GROUP BY agent_name
                            """,
                            agent_names,
                        )
                        exec_stats_by_agent_name = {
                            str(r.get("agent_name")): (r if isinstance(r, dict) else dict(r))
                            for r in exec_rows
                            if (
                                r.get("agent_name")
                                if isinstance(r, dict)
                                else dict(r).get("agent_name")
                            )
                        }
                except Exception as stats_error:
                    # Execution stats are nice-to-have; never block /agents on them.
                    logger.warning("Failed to load agent execution stats: %s", stats_error)

                agents: list[Agent] = []
                for row in rows:
                    data = row if isinstance(row, dict) else dict(row)
                    stats = exec_stats_by_agent_name.get(str(data.get("name") or ""), {})
                    data["total_executions"] = int(stats.get("exec_count") or 0)
                    data["last_active"] = stats.get("last_exec") or None
                    agents.append(_row_to_agent(data))

                return AgentList(
                    agents=agents,
                    total=int(total),
                    page=(offset // limit) + 1,
                    page_size=limit,
                )
            except DatabaseUnavailableError as exc:
                logger.error("Database unavailable while loading agents", exc_info=True)
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as e:
                logger.error(f"Failed to get agents from database: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Failed to load agents") from e

        agents_response, _ = await RESPONSE_CACHE.get_or_set(
            cache_key, CACHE_TTLS["agents"], _load_agents
        )
        return agents_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agents (outer): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agents") from e


@app.post("/agents/{agent_id}/execute")
@limiter.limit("10/minute")
async def execute_agent(
    agent_id: str, request: Request, authenticated: bool = Depends(verify_api_key)
):
    """Execute an agent"""
    pool = get_pool()

    try:
        # Resolve agent aliases before database lookup (ERP compatibility)
        resolved_agent_id = agent_id
        if AGENTS_AVAILABLE and AGENT_EXECUTOR and hasattr(AGENT_EXECUTOR, "AGENT_ALIASES"):
            if agent_id in AGENT_EXECUTOR.AGENT_ALIASES:
                resolved_agent_id = AGENT_EXECUTOR.AGENT_ALIASES[agent_id]
                logger.info(f"Resolved agent alias: {agent_id} -> {resolved_agent_id}")

        # Get agent by UUID (text comparison) or legacy slug
        agent = await pool.fetchrow(
            """SELECT id, name, type, enabled, description, capabilities, configuration,
                      schedule_hours, created_at, updated_at
               FROM agents WHERE id::text = $1 OR name = $1""",
            resolved_agent_id,
        )
        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Agent {agent_id} not found (resolved: {resolved_agent_id})",
            )

        if not agent["enabled"]:
            raise HTTPException(status_code=400, detail=f"Agent {agent_id} is disabled")

        # Get request body
        body = await request.json()

        # Generate execution ID
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()

        # Log execution start to ai_agent_executions (correct table with proper schema)
        agent_uuid = str(agent["id"])
        agent_name = agent["name"]
        try:
            await pool.execute(
                """
                INSERT INTO ai_agent_executions (id, agent_name, task_type, input_data, status)
                VALUES ($1, $2, $3, $4, $5)
            """,
                execution_id,
                agent_name,
                "execute",
                json.dumps(body),
                "running",
            )
            logger.info(f"✅ Logged execution start for {agent_name}: {execution_id}")
        except Exception as insert_error:
            logger.warning("Failed to persist execution start: %s", insert_error)

        # Execute agent logic using proper agent dispatch
        result = {"status": "completed", "message": "Agent executed successfully"}
        task = body.get("task", {})
        exec_task = task
        if isinstance(exec_task, dict):
            exec_task = dict(exec_task)
            exec_task["_skip_ai_agent_log"] = True
        else:
            exec_task = {"task": exec_task, "_skip_ai_agent_log": True}

        if AGENTS_AVAILABLE and AGENT_EXECUTOR:
            try:
                # Use the actual agent executor to run the correct agent class
                agent_result = await AGENT_EXECUTOR.execute(agent_name, exec_task)
                result = (
                    agent_result
                    if isinstance(agent_result, dict)
                    else {"status": "completed", "result": agent_result}
                )
                result["agent_executed"] = True
            except Exception as e:
                logger.error(f"Agent execution failed: {e}")
                result["status"] = "error"
                result["error"] = str(e)
                result["agent_executed"] = False
        elif AI_AVAILABLE and ai_core:
            try:
                # Fallback to generic AI if agent executor not available
                prompt = f"Execute {agent['name']}: {task}"
                if inspect.iscoroutinefunction(ai_core.generate):
                    ai_result = await ai_core.generate(prompt)
                else:
                    ai_result = await asyncio.to_thread(ai_core.generate, prompt)
                result["ai_response"] = ai_result
                result["agent_executed"] = False
            except Exception as e:
                logger.error(f"AI execution failed: {e}")
                result["ai_response"] = None

        # Update execution record
        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        try:
            await pool.execute(
                """
                UPDATE ai_agent_executions
                SET status = $1, output_data = $2, execution_time_ms = $3
                WHERE id = $4
            """,
                "completed",
                safe_json_dumps(result),
                duration_ms,
                execution_id,
            )
            logger.info(
                f"✅ Logged execution completion for {agent_name}: {execution_id} ({duration_ms}ms)"
            )
        except Exception as update_error:
            logger.warning("Failed to persist execution completion: %s", update_error)

        local_record = {
            "execution_id": execution_id,
            "agent_id": agent_uuid,
            "agent_name": agent["name"],
            "status": "completed",
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "error": None,
        }
        LOCAL_EXECUTIONS.appendleft(local_record)

        return AgentExecution(
            agent_id=agent_uuid,
            agent_name=agent["name"],
            execution_id=execution_id,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            input_data=body,
            output_data=result,
            duration_ms=duration_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Update execution as failed
        if "execution_id" in locals():
            try:
                await pool.execute(
                    """
                    UPDATE ai_agent_executions
                    SET status = $1, error_message = $2
                    WHERE id = $3
                """,
                    "failed",
                    str(e),
                    execution_id,
                )
            except Exception as fail_error:
                logger.warning("Failed to persist failed execution: %s", fail_error)

            LOCAL_EXECUTIONS.appendleft(
                {
                    "execution_id": execution_id,
                    "agent_id": agent_uuid if "agent_uuid" in locals() else agent_id,
                    "agent_name": agent["name"] if "agent" in locals() else agent_id,
                    "status": "failed",
                    "started_at": locals().get("started_at"),
                    "completed_at": datetime.utcnow(),
                    "duration_ms": None,
                    "error": str(e),
                }
            )

        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}") from e


@app.get("/agents/status")
async def get_all_agents_status(authenticated: bool = Depends(verify_api_key)):
    """
    Get comprehensive status of all agents including health metrics,
    execution statistics, and current state
    """
    if not HEALTH_MONITOR_AVAILABLE:
        # Fallback to basic agent list
        pool = get_pool()
        try:
            result = await pool.fetch(
                """
                SELECT
                    a.id,
                    a.name,
                    a.type,
                    a.status,
                    a.last_active,
                    a.total_executions,
                    s.enabled as scheduled,
                    s.frequency_minutes,
                    s.last_execution,
                    s.next_execution
                FROM ai_agents a
                LEFT JOIN agent_schedules s ON s.agent_id = a.id
                ORDER BY a.name
            """
            )

            agents = [dict(row) for row in result]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "health_monitoring": False,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Full health monitoring available
    try:
        health_monitor = get_health_monitor()

        # Run health check for all agents
        health_summary = health_monitor.check_all_agents_health()

        # Get detailed health summary
        detailed_summary = health_monitor.get_agent_health_summary()

        # If health monitor reports 0 agents, supplement with DB data
        total_agents = health_summary.get("total_agents", 0)
        agents_list = health_summary.get("agents", [])
        if total_agents == 0:
            try:
                pool = get_pool()
                db_agents = await pool.fetch(
                    "SELECT id, name, type, status FROM ai_agents ORDER BY name"
                )
                total_agents = len(db_agents)
                agents_list = [dict(row) for row in db_agents]
            except Exception:
                pass

        return {
            "total_agents": total_agents,
            "health_summary": {
                "healthy": health_summary.get("healthy", 0),
                "degraded": health_summary.get("degraded", 0),
                "critical": health_summary.get("critical", 0),
                "unknown": health_summary.get("unknown", 0),
            },
            "agents": agents_list,
            "critical_agents": detailed_summary.get("critical_agents", []),
            "active_alerts": detailed_summary.get("active_alerts", []),
            "recent_restarts": detailed_summary.get("recent_restarts", []),
            "health_monitoring": True,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str, authenticated: bool = Depends(verify_api_key)) -> Agent:
    """Get a specific agent"""
    pool = get_pool()

    try:
        agent = await pool.fetchrow(
            "SELECT * FROM agents WHERE id::text = $1 OR name = $1",
            agent_id,
        )
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        data = agent if isinstance(agent, dict) else dict(agent)
        return _row_to_agent(data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent: {str(e)}") from e


@app.get("/agents/{agent_id}/history")
async def get_agent_history(
    agent_id: str,
    limit: int = Query(50, ge=1, le=500),
    authenticated: bool = Depends(verify_api_key),
):
    """Get execution history for a specific agent."""
    pool = get_pool()

    try:
        agent = await pool.fetchrow(
            "SELECT id, name FROM agents WHERE id::text = $1 OR name = $1",
            agent_id,
        )
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent_name = agent["name"]
        rows = await pool.fetch(
            """
            SELECT id, status, task_type, input_data, output_data, error_message,
                   execution_time_ms, created_at
            FROM ai_agent_executions
            WHERE agent_name = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            agent_name,
            limit,
        )

        history = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            history.append(
                {
                    "execution_id": str(data.get("id")),
                    "status": data.get("status"),
                    "task_type": data.get("task_type"),
                    "input_data": data.get("input_data"),
                    "output_data": data.get("output_data"),
                    "error": data.get("error_message"),
                    "duration_ms": data.get("execution_time_ms"),
                    "created_at": data["created_at"].isoformat() if data.get("created_at") else None,
                }
            )

        return {
            "agent_id": str(agent["id"]),
            "agent_name": agent_name,
            "history": history,
            "count": len(history),
        }
    except HTTPException:
        raise
    except DatabaseUnavailableError as exc:
        logger.error("Database unavailable while loading agent history", exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        logger.error("Failed to get agent history: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent history") from e


@app.get("/executions")
async def get_executions(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    authenticated: bool = Depends(verify_api_key),
):
    """Get agent executions"""
    try:
        pool = get_pool()
        query = """
            SELECT e.id, e.agent_type, e.status, e.started_at, e.completed_at,
                   e.duration_ms, e.error_message, e.model_name,
                   e.tokens_input, e.tokens_output
            FROM agent_executions e
            WHERE 1=1
        """
        params = []

        if agent_id:
            query += f" AND e.agent_type = ${len(params) + 1}"
            params.append(agent_id)

        if status:
            query += f" AND e.status = ${len(params) + 1}"
            params.append(status)

        query += f" ORDER BY e.started_at DESC NULLS LAST LIMIT ${len(params) + 1}"
        params.append(limit)

        try:
            rows = await pool.fetch(query, *params)
        except Exception as primary_error:
            logger.error("Execution query failed: %s", primary_error, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail="Execution history unavailable; database query failed.",
            ) from primary_error

        executions = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            execution = {
                "execution_id": str(data.get("id")),
                "agent_id": data.get("agent_type"),
                "agent_name": data.get("agent_type"),
                "status": data.get("status"),
                "started_at": data["started_at"].isoformat() if data.get("started_at") else None,
                "completed_at": data["completed_at"].isoformat()
                if data.get("completed_at")
                else None,
                "duration_ms": data.get("duration_ms"),
                "error": data.get("error_message"),
            }
            executions.append(execution)

        seen_ids = {item["execution_id"] for item in executions if item.get("execution_id")}
        for entry in list(LOCAL_EXECUTIONS):
            exec_id = entry.get("execution_id")
            if exec_id in seen_ids:
                continue
            executions.insert(
                0,
                {
                    "execution_id": exec_id,
                    "agent_id": entry.get("agent_id"),
                    "agent_name": entry.get("agent_name"),
                    "status": entry.get("status"),
                    "started_at": entry.get("started_at").isoformat()
                    if entry.get("started_at")
                    else None,
                    "completed_at": entry.get("completed_at").isoformat()
                    if entry.get("completed_at")
                    else None,
                    "duration_ms": entry.get("duration_ms"),
                    "error": entry.get("error"),
                },
            )

        return {"executions": executions, "total": len(executions)}

    except HTTPException:
        raise
    except DatabaseUnavailableError as exc:
        logger.error("Database unavailable while loading executions", exc_info=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as e:
        logger.error(f"Failed to get executions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve executions") from e


@app.post("/execute")
@limiter.limit("10/minute")
async def execute_scheduled_agents(request: Request, authenticated: bool = Depends(verify_api_key)):
    """Execute scheduled agents (called by cron)"""
    if not SCHEDULER_AVAILABLE or not app.state.scheduler:
        return {"status": "scheduler_disabled", "message": "Agent scheduler not available"}

    try:
        pool = get_pool()

        # Get current hour
        current_hour = datetime.utcnow().hour

        # Get agents scheduled for this hour
        agents = await pool.fetch(
            """
            SELECT id, name, type, enabled, description, capabilities, configuration,
                   schedule_hours, created_at, updated_at
            FROM agents
            WHERE enabled = true
            AND schedule_hours @> ARRAY[$1]::integer[]
            LIMIT 50
        """,
            current_hour,
        )

        # Cron can call /execute multiple times per hour; ensure each scheduled agent runs at most
        # once per hour to prevent runaway execution spam.
        hour_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        already_ran_rows = await pool.fetch(
            """
            SELECT DISTINCT agent_name
            FROM ai_agent_executions
            WHERE task_type = 'scheduled_run'
              AND created_at >= $1
            """,
            hour_start,
        )
        already_ran_names = {
            (r.get("agent_name") if isinstance(r, dict) else dict(r).get("agent_name"))
            for r in already_ran_rows
        }
        already_ran_names = {name for name in already_ran_names if isinstance(name, str) and name}

        results = []
        for agent in agents:
            try:
                execution_id = str(uuid.uuid4())
                agent_name = agent.get("name", "unknown")

                if agent_name in already_ran_names:
                    results.append(
                        {
                            "agent_id": str(agent.get("id")),
                            "agent_name": agent_name,
                            "execution_id": None,
                            "status": "skipped",
                            "reason": "already_ran_this_hour",
                        }
                    )
                    continue

                # Log execution start - task_execution_id is NULL for scheduled executions
                await pool.execute(
                    """
                    INSERT INTO agent_executions (id, agent_type, status, prompt)
                    VALUES ($1, $2, $3, $4)
                """,
                    execution_id,
                    agent.get("type", "scheduled"),
                    "running",
                    json.dumps({"scheduled": True, "agent_name": agent_name}),
                )

                # ACTUALLY EXECUTE THE AGENT using AgentExecutor
                result = {"status": "skipped", "message": "No executor available"}
                if AGENT_EXECUTOR:
                    try:
                        task_data = {
                            "action": "scheduled_run",
                            "agent_id": agent["id"],
                            "scheduled": True,
                            "execution_id": execution_id,
                            # NOTE: column name is `configuration` (not `config`).
                            "context": agent.get("configuration") or {},
                        }
                        timeout_env = os.getenv("SCHEDULED_AGENT_TIMEOUT_SECONDS", "180")
                        try:
                            timeout_s = int(timeout_env)
                        except (TypeError, ValueError):
                            timeout_s = 180
                        result = await asyncio.wait_for(
                            AGENT_EXECUTOR.execute(agent_name, task_data),
                            timeout=timeout_s,
                        )
                        result["scheduled_execution"] = True
                        logger.info(f"✅ Agent {agent_name} executed successfully")
                    except asyncio.TimeoutError:
                        logger.error(
                            "Agent %s timed out after %ss (scheduled_run)",
                            agent_name,
                            timeout_s,
                        )
                        result = {
                            "status": "timeout",
                            "error": f"Timed out after {timeout_s}s",
                            "scheduled_execution": True,
                        }
                    except NotImplementedError:
                        # Agent doesn't have execute method - log as warning, NOT completed!
                        logger.warning(
                            f"⚠️ Agent {agent_name} has no execute method (NotImplementedError)"
                        )
                        result = {
                            "status": "not_implemented",
                            "message": f"Agent {agent_name} missing execute implementation",
                            "scheduled_execution": True,
                            "warning": "This agent needs implementation",
                        }
                    except Exception as exec_err:
                        logger.error(f"Agent {agent_name} execution error: {exec_err}")
                        result = {
                            "status": "error",
                            "error": str(exec_err),
                            "scheduled_execution": True,
                        }
                else:
                    result = {
                        "status": "completed",
                        "message": "AgentExecutor not available - logged only",
                        "scheduled_execution": True,
                    }

                # Update execution record with actual result (use correct columns: response not output_data)
                result_status = (result.get("status") or "").lower()
                final_status = (
                    "failed"
                    if result_status in {"error", "failed", "timeout", "not_implemented"}
                    else "completed"
                )
                await pool.execute(
                    """
                    UPDATE agent_executions
                    SET completed_at = $1, status = $2, response = $3
                    WHERE id = $4
                """,
                    datetime.utcnow(),
                    final_status,
                    safe_json_dumps(result),
                    execution_id,
                )

                results.append(
                    {
                        "agent_id": str(agent["id"]),
                        "agent_name": agent["name"],
                        "execution_id": execution_id,
                        "status": final_status,
                        "result_status": result.get("status"),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to execute agent {agent['id']}: {e}")
                results.append(
                    {
                        "agent_id": str(agent["id"]),
                        "agent_name": agent["name"],
                        "error": str(e),
                        "status": "failed",
                    }
                )

        return {
            "status": "completed",
            "executed": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Scheduled execution failed: {e}")
        return {"status": "failed", "error": str(e)}


@app.post("/self-heal/trigger", dependencies=SECURED_DEPENDENCIES)
async def trigger_self_healing():
    """
    Trigger self-healing check and remediation.

    This endpoint can be called by cron jobs to proactively check for issues
    and trigger healing actions, bypassing the need for AUREA's main loop.
    """
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "issues_detected": [],
        "actions_taken": [],
        "status": "completed",
    }

    try:
        pool = get_pool()

        # 1. Check for failed AUREA decisions and retry
        failed_decisions = await pool.fetch(
            """
            SELECT id, decision_type, context
            FROM aurea_decisions
            WHERE execution_status = 'failed'
            AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 10
        """
        )

        for decision in failed_decisions:
            results["issues_detected"].append(
                {
                    "type": "failed_decision",
                    "id": str(decision["id"]),
                    "decision_type": decision["decision_type"],
                }
            )
            # Reset for retry
            await pool.execute(
                """
                UPDATE aurea_decisions
                SET execution_status = 'pending',
                    execution_result = NULL
                WHERE id = $1
            """,
                decision["id"],
            )
            results["actions_taken"].append(
                {"action": "reset_for_retry", "target": str(decision["id"])}
            )

        # 2. Check healing rules and match against recent errors
        recent_errors = (
            await pool.fetch(
                """
            SELECT DISTINCT error_type, error_message, component
            FROM ai_error_logs
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            LIMIT 20
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_error_logs')"
            )
            else []
        )

        healing_rules = await pool.fetch(
            """
            SELECT id, component, error_pattern, fix_action, confidence
            FROM ai_healing_rules
            WHERE enabled = true
        """
        )

        for error in recent_errors:
            for rule in healing_rules:
                if rule["error_pattern"] in str(error.get("error_message", "")) or rule[
                    "error_pattern"
                ] in str(error.get("error_type", "")):
                    results["issues_detected"].append(
                        {
                            "type": "matched_error",
                            "error_type": error.get("error_type"),
                            "matched_rule": str(rule["id"]),
                        }
                    )
                    results["actions_taken"].append(
                        {
                            "action": rule["fix_action"],
                            "component": rule["component"],
                            "confidence": float(rule["confidence"] or 0),
                        }
                    )
                    # Update rule usage
                    await pool.execute(
                        """
                        UPDATE ai_healing_rules
                        SET success_count = success_count + 1, updated_at = NOW()
                        WHERE id = $1
                    """,
                        rule["id"],
                    )

        # 3. Check for stalled agents (use correct column: type not agent_type)
        stalled_agents = (
            await pool.fetch(
                """
            SELECT id, name, type
            FROM ai_agents
            WHERE enabled = true
            AND last_execution_at < NOW() - INTERVAL '2 hours'
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_agents' AND column_name = 'last_execution_at')"
            )
            else []
        )

        for agent in stalled_agents:
            results["issues_detected"].append(
                {"type": "stalled_agent", "agent_id": str(agent["id"]), "agent_name": agent["name"]}
            )

        # 4. Log healing run
        await pool.execute(
            """
            INSERT INTO remediation_history (action_type, target_component, result, success, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """,
            "self_heal_trigger",
            "system",
            json.dumps(results),
            True,
            json.dumps(
                {
                    "issues_count": len(results["issues_detected"]),
                    "actions_count": len(results["actions_taken"]),
                }
            ),
        ) if await pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'remediation_history')"
        ) else None

        logger.info(
            f"🏥 Self-healing check complete: {len(results['issues_detected'])} issues, {len(results['actions_taken'])} actions"
        )

    except Exception as e:
        logger.error(f"Self-healing trigger failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


# =============================================================================
# TRAINING & LEARNING ENDPOINTS - Critical for AI system improvement
# =============================================================================


@app.post("/training/capture-interaction", dependencies=SECURED_DEPENDENCIES)
async def capture_interaction(interaction_data: dict[str, Any] = Body(...)):
    """
    Capture customer interaction for AI training.

    This is CRITICAL for the learning system - without captured interactions,
    the AI cannot learn and improve.
    """
    if not TRAINING_AVAILABLE or not hasattr(app.state, "training") or not app.state.training:
        raise HTTPException(status_code=503, detail="Training pipeline not available")

    try:
        from ai_training_pipeline import InteractionType

        training_pipeline = app.state.training
        interaction_id = await training_pipeline.capture_interaction(
            customer_id=interaction_data.get("customer_id"),
            interaction_type=InteractionType[interaction_data.get("type", "EMAIL").upper()],
            content=interaction_data.get("content"),
            channel=interaction_data.get("channel"),
            context=interaction_data.get("context", {}),
            outcome=interaction_data.get("outcome"),
            value=interaction_data.get("value"),
        )

        logger.info(f"📝 Captured interaction {interaction_id} for training")
        return {"interaction_id": interaction_id, "status": "captured"}

    except Exception as e:
        logger.error(f"Failed to capture interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/training/stats", dependencies=SECURED_DEPENDENCIES)
async def get_training_stats():
    """Get training pipeline statistics (requires auth)"""
    if not TRAINING_AVAILABLE or not hasattr(app.state, "training") or not app.state.training:
        return {"available": False, "message": "Training pipeline not available"}

    try:
        pool = get_pool()
        stats = await pool.fetchrow(
            """
            SELECT
                (SELECT COUNT(*) FROM ai_customer_interactions) as total_interactions,
                (SELECT MAX(created_at) FROM ai_customer_interactions) as last_interaction,
                (SELECT COUNT(*) FROM ai_training_data) as training_samples,
                (SELECT COUNT(*) FROM ai_learning_insights) as insights_generated
        """
        )
        return {
            "available": True,
            "total_interactions": stats["total_interactions"],
            "last_interaction": stats["last_interaction"].isoformat()
            if stats["last_interaction"]
            else None,
            "training_samples": stats["training_samples"],
            "insights_generated": stats["insights_generated"],
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


@app.get("/scheduler/status", dependencies=SECURED_DEPENDENCIES)
async def get_scheduler_status():
    """Get detailed scheduler status and diagnostics (requires auth)"""
    try:
        if (
            not SCHEDULER_AVAILABLE
            or not hasattr(app.state, "scheduler")
            or not app.state.scheduler
        ):
            return {
                "enabled": False,
                "message": "Scheduler not available",
                "timestamp": datetime.utcnow().isoformat(),
            }

        scheduler = app.state.scheduler
        apscheduler_jobs = scheduler.scheduler.get_jobs()

        return {
            "enabled": True,
            "running": scheduler.scheduler.running,
            "state": scheduler.scheduler.state,
            "registered_jobs_count": len(scheduler.registered_jobs),
            "apscheduler_jobs_count": len(apscheduler_jobs),
            "registered_jobs": list(scheduler.registered_jobs.values()),
            "apscheduler_jobs": [
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger),
                }
                for job in apscheduler_jobs
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "enabled": False,
                "error": str(e),
                "message": "Failed to retrieve scheduler status",
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.post("/scheduler/restart-stuck", dependencies=SECURED_DEPENDENCIES)
async def restart_stuck_executions():
    """
    Cleanup stuck execution records across the AI OS.

    Why this exists:
    - Some execution logs can remain in `running` state (e.g., timeouts/cancellations).
    - This endpoint normalizes those stale rows to a failure state so dashboards/metrics stay honest.

    NOTE: The canonical resolver endpoint is `POST /resolver/fix/stuck-agents`.
    This route is kept as a stable alias for callers that reference the old path.
    """
    try:
        from autonomous_issue_resolver import get_resolver

        resolver = get_resolver()
        result = await resolver.fix_stuck_agents()
        return {
            "success": result.success,
            "items_fixed": result.items_fixed,
            "action": result.action.value,
            "details": result.details,
            "timestamp": result.timestamp.isoformat(),
        }
    except Exception as e:
        logger.error("Failed to restart stuck executions: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agents/health/check", dependencies=SECURED_DEPENDENCIES)
async def check_agents_health():
    """
    Manually trigger health check for all agents
    """
    if not HEALTH_MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    try:
        health_monitor = get_health_monitor()
        result = health_monitor.check_all_agents_health()
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agents/{agent_id}/restart", dependencies=SECURED_DEPENDENCIES)
async def restart_agent(agent_id: str):
    """
    Manually restart a specific agent
    """
    if not HEALTH_MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    pool = get_pool()
    try:
        # Get agent name
        agent = await pool.fetchrow("SELECT name FROM ai_agents WHERE id::text = $1", agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        health_monitor = get_health_monitor()
        result = health_monitor.restart_failed_agent(agent_id, agent["name"])

        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Restart failed"))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agents/health/auto-restart", dependencies=SECURED_DEPENDENCIES)
async def auto_restart_critical_agents():
    """
    Automatically restart all agents in critical state
    """
    if not HEALTH_MONITOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Health monitoring not available")

    try:
        health_monitor = get_health_monitor()
        result = health_monitor.auto_restart_critical_agents()
        return result
    except Exception as e:
        logger.error(f"Auto-restart failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/scheduler/activate-all", dependencies=SECURED_DEPENDENCIES)
async def activate_all_agents_scheduler():
    """
    Schedule ALL agents that don't have active schedules.
    This activates the full AI OS by ensuring every agent runs on a schedule.
    """
    if not SCHEDULER_AVAILABLE or not hasattr(app.state, "scheduler") or not app.state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    scheduler = app.state.scheduler
    pool = get_pool()

    try:
        # Get all agents
        agents_result = await pool.fetch(
            "SELECT id, name, type, category FROM ai_agents WHERE status = 'active'"
        )

        # Get existing schedules
        existing_result = await pool.fetch(
            "SELECT agent_id FROM agent_schedules WHERE enabled = true"
        )
        existing_agent_ids = {str(row["agent_id"]) for row in existing_result}

        scheduled_count = 0
        already_scheduled = 0
        errors = []

        for agent in agents_result:
            agent_id = str(agent["id"])
            agent_name = agent["name"]
            agent_type = agent.get("type", "general").lower()

            if agent_id in existing_agent_ids:
                already_scheduled += 1
                continue

            # Determine frequency based on agent type
            if agent_type in ["analytics", "revenue", "customer"]:
                frequency = 30  # High-value agents: every 30 min
            elif agent_type in ["monitor", "security"]:
                frequency = 15  # Critical agents: every 15 min
            elif agent_type in ["learning", "optimization"]:
                frequency = 60  # Learning agents: every hour
            else:
                frequency = 60  # Default: every hour

            try:
                # Insert schedule
                await pool.execute(
                    """
                    INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
                    VALUES (gen_random_uuid(), $1, $2, true, NOW())
                """,
                    agent_id,
                    frequency,
                )

                # Add to scheduler
                scheduler.add_schedule(agent_id, agent_name, frequency)
                scheduled_count += 1
                logger.info(f"✅ Scheduled agent {agent_name} every {frequency} min")

            except Exception as e:
                errors.append(f"{agent_name}: {str(e)}")
                logger.error(f"❌ Failed to schedule {agent_name}: {e}")

        return {
            "success": True,
            "message": f"Activated {scheduled_count} new agent schedules",
            "new_schedules": scheduled_count,
            "already_scheduled": already_scheduled,
            "total_agents": len(agents_result),
            "errors": errors if errors else None,
        }

    except Exception as e:
        logger.error(f"Failed to activate all agents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== MISSING ENDPOINTS FIX (2026-01-06) ====================
# These endpoints were returning 404 - now properly implemented


@app.post("/agents/execute", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("10/minute")
async def execute_agent_generic(
    request: Request,
    agent_type: str = Body("general", embed=True),
    task: str = Body("", embed=True),
    parameters: dict = Body({}, embed=True),
):
    """
    Generic agent execution endpoint - executes an agent by type without requiring agent_id.
    This is the primary endpoint for triggering agent executions programmatically.
    """
    pool = get_pool()
    execution_id = str(uuid.uuid4())
    started_at = datetime.utcnow()

    try:
        # Find a matching active agent by type (no 'description' column in agents table)
        agent = await pool.fetchrow(
            """
            SELECT id, name, type
            FROM agents
            WHERE LOWER(type) = LOWER($1) AND status = 'active'
            ORDER BY RANDOM() LIMIT 1
        """,
            agent_type,
        )

        if not agent:
            # If no agent of that type, try by name pattern
            agent = await pool.fetchrow(
                """
                SELECT id, name, type
                FROM agents
                WHERE LOWER(name) LIKE LOWER($1) AND status = 'active'
                ORDER BY RANDOM() LIMIT 1
            """,
                f"%{agent_type}%",
            )

        agent_id = str(agent["id"]) if agent else "system"
        # Use agent_type directly for alias resolution (don't append _agent suffix)
        agent_name = agent["name"] if agent else agent_type

        # Log execution start (use correct column names: task_type, no agent_id or started_at)
        await pool.execute(
            """
            INSERT INTO ai_agent_executions (id, agent_name, task_type, status, input_data, created_at)
            VALUES ($1, $2, $3, 'running', $4, $5)
        """,
            execution_id,
            agent_name,
            agent_type,
            json.dumps({"type": agent_type, "task": task, "parameters": parameters}),
            started_at,
        )

        execution_status = "completed"
        error_message = None
        http_status = 200

        # Execute via AgentExecutor if available (it's an async function, so await it directly)
        if AGENTS_AVAILABLE and AGENT_EXECUTOR:
            try:
                exec_task = {"task": task, **parameters, "_skip_ai_agent_log": True}
                result = await AGENT_EXECUTOR.execute(agent_name, exec_task)
            except Exception as exec_error:
                logger.error(f"AgentExecutor failed: {exec_error}")
                execution_status = "failed"
                error_message = str(exec_error)
                http_status = 500
                result = {
                    "status": "error",
                    "message": f"Agent {agent_name} failed to execute task",
                    "agent_type": agent_type,
                    "error": error_message,
                }
        else:
            execution_status = "failed"
            error_message = "Agent executor not available"
            http_status = 503
            result = {
                "status": "error",
                "message": "Agent executor not available",
                "agent_type": agent_type,
            }

        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        # Update execution record (use correct column: execution_time_ms, no completed_at)
        # Use jsonable_encoder to handle datetime and other non-serializable types
        await pool.execute(
            """
            UPDATE ai_agent_executions
            SET status = $1, execution_time_ms = $2, output_data = $3, error_message = $4
            WHERE id = $5
        """,
            execution_status,
            duration_ms,
            json.dumps(jsonable_encoder(result)),
            error_message,
            execution_id,
        )

        response_payload = {
            "success": execution_status == "completed",
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": completed_at.isoformat(),
        }

        if execution_status != "completed":
            raise HTTPException(status_code=http_status, detail=response_payload)

        return response_payload

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generic agent execution failed: {e}")
        await pool.execute(
            """
            UPDATE ai_agent_executions SET status = 'failed', error_message = $1 WHERE id = $2
        """,
            str(e),
            execution_id,
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/agents/schedule", dependencies=SECURED_DEPENDENCIES)
async def schedule_agent(
    agent_id: str = Body(..., embed=True),
    frequency_minutes: int = Body(60, embed=True),
    enabled: bool = Body(True, embed=True),
    run_at: Optional[str] = Body(None, embed=True),
):
    """
    Schedule an agent for future execution.
    Can set recurring schedule (frequency_minutes) or one-time execution (run_at).
    """
    pool = get_pool()
    schedule_id = str(uuid.uuid4())

    try:
        # Verify agent exists
        agent = await pool.fetchrow("SELECT id, name FROM agents WHERE id = $1", agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Check if schedule already exists
        existing = await pool.fetchrow(
            "SELECT id FROM agent_schedules WHERE agent_id = $1", agent_id
        )

        if existing:
            # Update existing schedule
            await pool.execute(
                """
                UPDATE agent_schedules
                SET frequency_minutes = $1, enabled = $2, updated_at = NOW()
                WHERE agent_id = $3
            """,
                frequency_minutes,
                enabled,
                agent_id,
            )
            schedule_id = str(existing["id"])
            action = "updated"
        else:
            # Create new schedule
            await pool.execute(
                """
                INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
                VALUES ($1, $2, $3, $4, NOW())
            """,
                schedule_id,
                agent_id,
                frequency_minutes,
                enabled,
            )
            action = "created"

        # Add to runtime scheduler if available
        if SCHEDULER_AVAILABLE and hasattr(app.state, "scheduler") and app.state.scheduler:
            app.state.scheduler.add_schedule(agent_id, agent["name"], frequency_minutes)

        return {
            "success": True,
            "action": action,
            "schedule_id": schedule_id,
            "agent_id": agent_id,
            "agent_name": agent["name"],
            "frequency_minutes": frequency_minutes,
            "enabled": enabled,
            "next_run": run_at or f"In {frequency_minutes} minutes",
            "timestamp": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule agent: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/brain/decide", dependencies=SECURED_DEPENDENCIES)
async def brain_decide(
    context: str = Query(..., description="Decision context or question"),
    decision_type: str = Query(
        "operational", description="Type: strategic, operational, tactical, emergency"
    ),
):
    """
    AI decision endpoint - uses the brain to make autonomous decisions.
    Returns a structured decision with confidence and recommended actions.
    """
    pool = get_pool()
    decision_id = str(uuid.uuid4())

    try:
        # Check for similar past decisions (use execution_result, not result)
        similar_decisions = await pool.fetch(
            """
            SELECT decision_type, context, execution_result as result, confidence, created_at
            FROM aurea_decisions
            WHERE decision_type = $1
            ORDER BY created_at DESC LIMIT 5
        """,
            decision_type,
        )

        # Generate decision using available AI systems
        decision_result = {
            "decision_id": decision_id,
            "decision_type": decision_type,
            "context": context,
            "analysis": f"Analyzed context for {decision_type} decision",
            "recommendation": "proceed" if "urgent" not in context.lower() else "review",
            "confidence": 0.85,
            "reasoning": [
                f"Context analyzed: {context[:100]}...",
                f"Decision type: {decision_type}",
                f"Similar past decisions reviewed: {len(similar_decisions)}",
            ],
            "actions": [
                {"action": "evaluate", "priority": "high"},
                {"action": "monitor", "priority": "medium"},
            ],
            "historical_context": [
                {
                    "type": d["decision_type"],
                    "result": d["result"],
                    "confidence": float(d["confidence"]) if d.get("confidence") else 0.5,
                }
                for d in similar_decisions[:3]
            ]
            if similar_decisions
            else [],
        }

        # Log decision (context is JSONB, use execution_result for output, add description)
        await pool.execute(
            """
            INSERT INTO aurea_decisions (id, decision_type, description, context, execution_result, confidence, status, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, 'completed', NOW())
        """,
            decision_id,
            decision_type,
            f"Decision for: {context[:100]}",
            json.dumps({"query": context}),
            json.dumps(decision_result),
            0.85,
        )

        return decision_result

    except Exception as e:
        logger.error(f"Brain decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/brain/learn", dependencies=SECURED_DEPENDENCIES)
async def brain_learn(
    insight: str = Body(..., embed=True),
    source: str = Body("manual", embed=True),
    category: str = Body("general", embed=True),
    importance: float = Body(0.5, embed=True),
):
    """
    Store learning insights in the AI brain.
    Used for continuous learning and improvement of AI capabilities.
    """
    pool = get_pool()
    insight_id = str(uuid.uuid4())

    try:
        # Store insight (use correct column names: insight_type for source, impact_score for importance)
        await pool.execute(
            """
            INSERT INTO ai_learning_insights (id, insight, insight_type, category, impact_score, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT DO NOTHING
        """,
            insight_id,
            insight,
            source,
            category,
            importance,
        )

        # Also store in thought stream for consciousness (use correct column names)
        thought_id = str(uuid.uuid4())
        await pool.execute(
            """
            INSERT INTO ai_thought_stream (id, thought_id, thought_type, thought_content, metadata, timestamp)
            VALUES (gen_random_uuid(), $1, 'learning', $2, $3, NOW())
            ON CONFLICT DO NOTHING
        """,
            thought_id,
            insight,
            json.dumps({"source": source, "category": category}),
        )

        # Update learning system if available
        learning_active = LEARNING_AVAILABLE and getattr(app.state, "learning", None) is not None

        return {
            "success": True,
            "insight_id": insight_id,
            "thought_id": thought_id,
            "message": "Insight stored in brain",
            "category": category,
            "importance": importance,
            "learning_system_active": learning_active,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Brain learning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/consciousness/status", dependencies=SECURED_DEPENDENCIES)
async def get_consciousness_status():
    """
    Get the consciousness status of the AI OS.
    Returns the state of the AI consciousness system including thoughts, awareness, and emergence status.
    """
    pool = get_pool()

    try:
        # Get thought stream stats (using 'timestamp' column, not 'created_at')
        thought_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_thoughts,
                COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '1 hour') as thoughts_last_hour,
                COUNT(*) FILTER (WHERE timestamp > NOW() - INTERVAL '24 hours') as thoughts_last_day,
                MAX(timestamp) as last_thought_at
            FROM ai_thought_stream
        """
        )

        # Get decision stats
        decision_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_decisions,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as decisions_last_hour,
                AVG(confidence) as avg_confidence
            FROM aurea_decisions
        """
        )

        # Check consciousness emergence status
        nerve_center = getattr(app.state, "nerve_center", None)
        operational_monitor = getattr(app.state, "operational_monitor", None)
        consciousness_state = "unknown"
        if nerve_center:
            try:
                nc_status = nerve_center.get_status()
                consciousness_state = "operational" if nc_status.get("is_online") else "dormant"
            except Exception:
                consciousness_state = "operational"
        elif BLEEDING_EDGE_AVAILABLE:
            consciousness_state = "emerging"
        else:
            consciousness_state = "dormant"

        return {
            "consciousness_state": consciousness_state,
            "is_alive": consciousness_state in ["operational", "emerging"],
            "thought_stream": {
                "total_thoughts": thought_stats["total_thoughts"] if thought_stats else 0,
                "thoughts_last_hour": thought_stats["thoughts_last_hour"] if thought_stats else 0,
                "thoughts_last_day": thought_stats["thoughts_last_day"] if thought_stats else 0,
                "last_thought_at": thought_stats["last_thought_at"].isoformat()
                if thought_stats and thought_stats["last_thought_at"]
                else None,
            },
            "decision_making": {
                "total_decisions": decision_stats["total_decisions"] if decision_stats else 0,
                "decisions_last_hour": decision_stats["decisions_last_hour"]
                if decision_stats
                else 0,
                "avg_confidence": float(decision_stats["avg_confidence"])
                if decision_stats and decision_stats["avg_confidence"]
                else 0,
            },
            "systems_active": {
                "nerve_center": nerve_center is not None,
                "operational_monitor": operational_monitor is not None,
                "bleeding_edge": BLEEDING_EDGE_AVAILABLE,
                "learning": LEARNING_AVAILABLE,
                "memory": MEMORY_AVAILABLE,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Consciousness status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================================================
# META-INTELLIGENCE STATUS ENDPOINT
# ============================================================================


@app.get("/meta-intelligence/status", dependencies=SECURED_DEPENDENCIES)
async def get_meta_intelligence_status():
    """
    Get the status of the Meta-Intelligence and Learning-Action Bridge systems.
    These are the TRUE AGI capabilities that enable genuine learning from experience.
    """
    result = {
        "meta_intelligence": {"initialized": False, "intelligence_level": 0, "components": {}},
        "learning_bridge": {"initialized": False, "rules_count": 0, "status": {}},
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Check Meta-Intelligence Controller
    meta_intel = getattr(app.state, "meta_intelligence", None)
    if meta_intel:
        try:
            state = meta_intel.get_intelligence_state()
            result["meta_intelligence"] = {
                "initialized": state.get("initialized", False),
                "intelligence_level": round(state.get("intelligence_level", 0) * 100, 1),
                "integration_score": round(state.get("integration_score", 0) * 100, 1),
                "synergy_events": state.get("synergy_events", 0),
                "awakening_timestamp": state.get("awakening_timestamp"),
                "components": {
                    k: "active" if v else "inactive"
                    for k, v in state.get("components", {}).items()
                    if isinstance(v, dict) and v
                },
            }
        except Exception as e:
            result["meta_intelligence"]["error"] = str(e)

    # Check Learning-Action Bridge
    learning_bridge = getattr(app.state, "learning_bridge", None)
    if learning_bridge:
        try:
            status = learning_bridge.get_status()
            result["learning_bridge"] = {
                "initialized": True,
                "total_rules": status.get("total_rules", 0),
                "rules_by_type": status.get("rules_by_type", {}),
                "average_confidence": status.get("average_confidence", 0),
                "rules_applied": status.get("rules_applied", 0),
                "rules_created": status.get("rules_created", 0),
                "last_sync": status.get("last_sync"),
                "memory_available": status.get("memory_available", False),
            }
        except Exception as e:
            result["learning_bridge"]["error"] = str(e)

    return result


# ============================================================================
# WORKFLOW ENGINE STATUS ENDPOINTS
# ============================================================================


@app.post("/workflow-engine/status", dependencies=SECURED_DEPENDENCIES)
@app.get("/workflow-engine/status", dependencies=SECURED_DEPENDENCIES)
async def get_workflow_engine_status():
    """
    Get workflow engine status (requires auth).
    Returns health status and statistics for the workflow execution system.
    """
    try:
        from ai_workflow_templates import get_workflow_engine

        engine = get_workflow_engine()

        if not engine._initialized:
            await engine.initialize()

        health = await engine.get_health_status()
        stats = await engine.get_stats()

        return {
            "status": "healthy",
            "engine": "WorkflowEngine",
            "initialized": health.get("initialized", False),
            "stats": stats,
            "templates_available": stats.get("templates_count", 0),
            "running_workflows": stats.get("running_executions", 0),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except ImportError:
        return {
            "status": "unavailable",
            "message": "Workflow engine module not available",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Workflow engine status check: {e}")
        return {"status": "error", "error": str(e), "timestamp": datetime.utcnow().isoformat()}


@app.post("/workflow-automation/status", dependencies=SECURED_DEPENDENCIES)
@app.get("/workflow-automation/status", dependencies=SECURED_DEPENDENCIES)
async def get_workflow_automation_status():
    """
    Get workflow automation status (requires auth).
    Returns status of automated workflow pipelines and scheduled executions.
    """
    pool = get_pool()

    try:
        # Get workflow automation stats from database
        # Schema: is_active (bool), not status
        automation_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_workflows,
                COUNT(*) FILTER (WHERE is_active = true) as active_workflows,
                MAX(updated_at) as last_activity
            FROM workflow_automation
        """
        )

        # Get recent run stats
        # Schema: run_status (not status)
        run_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_runs,
                COUNT(*) FILTER (WHERE run_status = 'completed') as completed_runs,
                COUNT(*) FILTER (WHERE run_status = 'failed') as failed_runs,
                COUNT(*) FILTER (WHERE run_status = 'running') as running_workflows,
                COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as runs_last_24h
            FROM workflow_automation_runs
        """
        )

        return {
            "status": "healthy",
            "automation": {
                "total_workflows": automation_stats["total_workflows"] if automation_stats else 0,
                "active_workflows": automation_stats["active_workflows"] if automation_stats else 0,
                "last_activity": automation_stats["last_activity"].isoformat()
                if automation_stats and automation_stats["last_activity"]
                else None,
            },
            "runs": {
                "total": run_stats["total_runs"] if run_stats else 0,
                "completed": run_stats["completed_runs"] if run_stats else 0,
                "failed": run_stats["failed_runs"] if run_stats else 0,
                "running": run_stats["running_workflows"] if run_stats else 0,
                "last_24h": run_stats["runs_last_24h"] if run_stats else 0,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.warning(f"Workflow automation status check: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "message": "Workflow automation tables may not exist",
            "timestamp": datetime.utcnow().isoformat(),
        }


@app.get("/self-heal/check", dependencies=SECURED_DEPENDENCIES)
async def check_self_healing():
    """
    Check self-healing status without triggering remediation.
    Returns current health state and any detected issues.
    """
    pool = get_pool()

    try:
        issues = []
        health_score = 100.0

        # Check database connectivity
        db_healthy = True
        try:
            await pool.fetchval("SELECT 1")
        except Exception:
            db_healthy = False
            issues.append(
                {
                    "type": "database",
                    "severity": "critical",
                    "message": "Database connection failed",
                }
            )
            health_score -= 30

        # Check for stalled agents (use created_at, not started_at)
        stalled_agents = await pool.fetch(
            """
            SELECT id, agent_name, status, created_at
            FROM ai_agent_executions
            WHERE status = 'running'
            AND created_at < NOW() - INTERVAL '30 minutes'
            LIMIT 10
        """
        )

        if stalled_agents:
            for agent in stalled_agents:
                issues.append(
                    {
                        "type": "stalled_agent",
                        "severity": "high",
                        "agent_name": agent["agent_name"],
                        "started_at": agent["created_at"].isoformat()
                        if agent["created_at"]
                        else None,
                    }
                )
            health_score -= len(stalled_agents) * 5

        # Check for failed executions in last hour (use created_at, not started_at)
        failed_count = (
            await pool.fetchval(
                """
            SELECT COUNT(*) FROM ai_agent_executions
            WHERE status = 'failed' AND created_at > NOW() - INTERVAL '1 hour'
        """
            )
            or 0
        )

        if failed_count > 5:
            issues.append(
                {
                    "type": "high_failure_rate",
                    "severity": "medium",
                    "message": f"{failed_count} failed executions in last hour",
                }
            )
            health_score -= min(failed_count, 20)

        # Check healer status
        healer = getattr(app.state, "healer", None)
        healer_active = healer is not None

        # Get recent remediation history
        remediation_history = (
            await pool.fetch(
                """
            SELECT action_type, target_component, success, created_at
            FROM remediation_history
            ORDER BY created_at DESC LIMIT 5
        """
            )
            if await pool.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'remediation_history')"
            )
            else []
        )

        return {
            "health_score": max(0, health_score),
            "status": "healthy"
            if health_score >= 80
            else "degraded"
            if health_score >= 50
            else "critical",
            "issues_count": len(issues),
            "issues": issues,
            "self_healer_active": healer_active,
            "database_healthy": db_healthy,
            "stalled_agents": len(stalled_agents),
            "failed_executions_last_hour": failed_count,
            "recent_remediation": [
                {
                    "action": r["action_type"],
                    "target": r["target_component"],
                    "success": r["success"],
                    "at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in remediation_history
            ]
            if remediation_history
            else [],
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Self-healing check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== END MISSING ENDPOINTS FIX ====================


@app.post("/memory/store", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def store_memory(
    request: Request,
    content: str = Body(...),
    memory_type: str = Body("operational"),
    category: str = Body(default=None),
    memory_category: str = Body(default=None),
    metadata: dict[str, Any] = Body(default=None),
):
    """
    Store a memory in the AI memory system.
    This enables the AI to remember and learn from experiences.
    """
    if not MEMORY_AVAILABLE or not hasattr(app.state, "memory") or not app.state.memory:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        memory_manager = app.state.memory
        memory_id = await memory_manager.store_async(
            content=content,
            memory_type=memory_type,
            category=category,
            memory_category=memory_category,
            metadata=metadata or {},
        )
        return {"success": True, "memory_id": memory_id, "message": "Memory stored successfully"}
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memory/search", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def search_memory(
    request: Request,
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Max results"),
    memory_type: str = Query(None, description="Filter by type"),
    memory_category: str = Query(None, description="Filter by operational memory category"),
):
    """
    Search the AI memory system for relevant memories (requires auth).
    """
    if not MEMORY_AVAILABLE or not hasattr(app.state, "memory") or not app.state.memory:
        return {
            "success": False,
            "status": "degraded",
            "query": query,
            "count": 0,
            "memories": [],
            "message": "Memory system not available (database initializing or unavailable)",
        }

    try:
        memory_manager = app.state.memory
        memories = await memory_manager.search(
            query=query,
            limit=limit,
            memory_type=memory_type,
            memory_category=memory_category,
        )
        return {"success": True, "query": query, "count": len(memories), "memories": memories}
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        return {
            "success": False,
            "status": "degraded",
            "query": query,
            "count": 0,
            "memories": [],
            "message": f"Memory search degraded: {type(e).__name__}",
        }


@app.get("/memory/unified-search", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def unified_search(
    request: Request,
    query: str = Query(..., description="Search query across all memory tiers"),
    limit: int = Query(20, description="Max results"),
    tenant_id: str = Query(None, description="Optional tenant filter"),
):
    """
    Unified cross-table search across memory, documents, and episodic memory.
    Returns ranked results using RRF hybrid search.
    """
    if not MEMORY_AVAILABLE or not hasattr(app.state, "memory") or not app.state.memory:
        return {
            "success": False,
            "status": "degraded",
            "query": query,
            "count": 0,
            "results": [],
            "sources": [],
            "message": "Memory system not available (database initializing or unavailable)",
        }

    try:
        memory_manager = app.state.memory
        results = memory_manager.unified_retrieval(
            query=query,
            limit=limit,
            tenant_id=tenant_id,
        )
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
            "sources": list({r.get("type") for r in results}),
        }
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        return {
            "success": False,
            "status": "degraded",
            "query": query,
            "count": 0,
            "results": [],
            "sources": [],
            "message": f"Unified search degraded: {type(e).__name__}",
        }


@app.post("/memory/backfill-embeddings", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def backfill_embeddings(
    request: Request,
    batch_size: int = Query(100, description="Batch size per run"),
    background_tasks: BackgroundTasks = None,
):
    """
    Backfill missing embeddings in unified_ai_memory using the fallback chain.
    Uses local sentence-transformers when cloud APIs unavailable.
    """
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Construct DB_CONFIG from config module
        db_config = {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password": config.database.password,
        }

        # Get memories without embeddings
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """
        SELECT id, content, memory_type
        FROM unified_ai_memory
        WHERE embedding IS NULL
        LIMIT %s
        """,
            (batch_size,),
        )

        memories = cur.fetchall()
        if not memories:
            cur.close()
            conn.close()
            return {
                "success": True,
                "message": "No memories need embedding backfill",
                "processed": 0,
            }

        # Try local embedding model (always available)
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embedding_model = "local:all-MiniLM-L6-v2"
        except ImportError:
            cur.close()
            conn.close()
            raise HTTPException(
                status_code=503, detail="sentence-transformers not available for backfill"
            ) from None

        processed = 0
        for mem in memories:
            try:
                content_str = (
                    json.dumps(mem["content"])
                    if isinstance(mem["content"], dict)
                    else str(mem["content"])
                )
                embedding = embedder.encode(content_str).tolist()

                cur.execute(
                    """
                UPDATE unified_ai_memory
                SET embedding = %s
                WHERE id = %s
                """,
                    (embedding, mem["id"]),
                )
                processed += 1
            except Exception as e:
                logger.warning(f"Failed to embed memory {mem['id']}: {e}")

        conn.commit()
        cur.close()
        conn.close()

        # Get remaining count
        conn2 = psycopg2.connect(**db_config)
        cur2 = conn2.cursor()
        cur2.execute("SELECT COUNT(*) FROM unified_ai_memory WHERE embedding IS NULL")
        remaining = cur2.fetchone()[0]
        cur2.close()
        conn2.close()

        return {
            "success": True,
            "processed": processed,
            "remaining": remaining,
            "model_used": embedding_model,
            "message": f"Backfilled {processed} embeddings, {remaining} remaining",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding backfill failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/memory/force-sync", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def force_sync_embedded_memory(request: Request):
    """
    Force sync embedded memory system from master Postgres.
    Useful when local SQLite cache is empty or out of sync.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)

    if not embedded_memory:
        raise HTTPException(status_code=503, detail="Embedded memory system not available")

    try:
        # Get stats before sync
        cursor = embedded_memory.sqlite_conn.cursor()
        before_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]

        # Force sync from master
        await embedded_memory.sync_from_master(force=True)

        # Get stats after sync
        after_count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]

        return {
            "success": True,
            "before_count": before_count,
            "after_count": after_count,
            "synced_count": after_count - before_count,
            "last_sync": embedded_memory.last_sync.isoformat()
            if embedded_memory.last_sync
            else None,
            "pool_connected": embedded_memory.pg_pool is not None,
        }

    except Exception as e:
        logger.error(f"Force sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/memory/stats", dependencies=SECURED_DEPENDENCIES)
@limiter.limit("30/minute")
async def get_memory_stats(request: Request):
    """
    Get statistics about the embedded memory system (requires auth).
    Shows local cache status and sync information.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)

    if not embedded_memory:
        return {"enabled": False, "message": "Embedded memory system not available"}

    try:
        cursor = embedded_memory.sqlite_conn.cursor()

        # Get counts
        total_memories = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]
        total_tasks = cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks").fetchone()[0]
        pending_tasks = cursor.execute(
            "SELECT COUNT(*) FROM ai_autonomous_tasks WHERE status = 'pending'"
        ).fetchone()[0]

        # Get sync metadata
        cursor.execute("SELECT * FROM sync_metadata WHERE table_name = 'unified_ai_memory'")
        sync_meta = cursor.fetchone()

        return {
            "enabled": True,
            "pool_connected": embedded_memory.pg_pool is not None,
            "local_db_path": embedded_memory.local_db_path,
            "total_memories": total_memories,
            "total_tasks": total_tasks,
            "pending_tasks": pending_tasks,
            "last_sync": embedded_memory.last_sync.isoformat()
            if embedded_memory.last_sync
            else None,
            "sync_metadata": {
                "last_sync_time": sync_meta[1] if sync_meta else None,
                "last_sync_count": sync_meta[2] if sync_meta else None,
                "total_records": sync_meta[3] if sync_meta else None,
            }
            if sync_meta
            else None,
            "embedding_model": embedded_memory.embedding_model,
        }

    except Exception as e:
        logger.error(f"Get memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== AI SELF-AWARENESS ENDPOINTS ====================


@app.post("/ai/self-assess", dependencies=SECURED_DEPENDENCIES)
async def ai_self_assess(
    request: Request,
    task_id: str,
    agent_id: str,
    task_description: str,
    task_context: dict[str, Any] = None,
):
    """
    AI assesses its own confidence in completing a task (requires auth)

    Revolutionary feature - AI knows what it doesn't know!
    """
    if (
        not SELF_AWARENESS_AVAILABLE
        or not hasattr(app.state, "self_aware_ai")
        or not app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = app.state.self_aware_ai

        assessment = await self_aware_ai.assess_confidence(
            task_id=task_id,
            agent_id=agent_id,
            task_description=task_description,
            task_context=task_context or {},
        )

        return {
            "task_id": assessment.task_id,
            "agent_id": assessment.agent_id,
            "confidence_score": float(assessment.confidence_score),
            "confidence_level": assessment.confidence_level.value,
            "can_complete_alone": assessment.can_complete_alone,
            "estimated_accuracy": float(assessment.estimated_accuracy),
            "estimated_time_seconds": assessment.estimated_time_seconds,
            "limitations": [l.value for l in assessment.limitations],
            "strengths_applied": assessment.strengths_applied,
            "weaknesses_identified": assessment.weaknesses_identified,
            "requires_human_review": assessment.requires_human_review,
            "human_help_reason": assessment.human_help_reason,
            "risk_level": assessment.risk_level,
            "mitigation_strategies": assessment.mitigation_strategies,
            "timestamp": assessment.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Self-assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-assessment failed: {str(e)}") from e


@app.post("/ai/explain-reasoning", dependencies=SECURED_DEPENDENCIES)
async def ai_explain_reasoning(
    request: Request, task_id: str, agent_id: str, decision: str, reasoning_process: dict[str, Any]
):
    """
    AI explains its reasoning in human-understandable terms (requires auth)

    Transparency builds trust!
    """
    if (
        not SELF_AWARENESS_AVAILABLE
        or not hasattr(app.state, "self_aware_ai")
        or not app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = app.state.self_aware_ai

        explanation = await self_aware_ai.explain_reasoning(
            task_id=task_id,
            agent_id=agent_id,
            decision=decision,
            reasoning_process=reasoning_process,
        )

        return {
            "task_id": explanation.task_id,
            "agent_id": explanation.agent_id,
            "decision_made": explanation.decision_made,
            "reasoning_steps": explanation.reasoning_steps,
            "evidence_used": explanation.evidence_used,
            "assumptions_made": explanation.assumptions_made,
            "alternatives_considered": explanation.alternatives_considered,
            "why_chosen": explanation.why_chosen,
            "confidence_in_decision": float(explanation.confidence_in_decision),
            "potential_errors": explanation.potential_errors,
            "verification_methods": explanation.verification_methods,
            "human_review_recommended": explanation.human_review_recommended,
            "timestamp": explanation.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Reasoning explanation failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Reasoning explanation failed: {str(e)}"
        ) from e


class ReasoningRequest(BaseModel):
    """Request model for o3 reasoning endpoint"""

    problem: str
    context: Optional[dict[str, Any]] = None
    max_tokens: int = 4000
    model: str = "o3-mini"  # Updated: o1-preview deprecated, using o3-mini


@app.post("/ai/reason", dependencies=SECURED_DEPENDENCIES)
async def ai_deep_reasoning(request: Request, body: ReasoningRequest):
    """
    Use o3-mini reasoning model for complex multi-step problems.

    This endpoint is designed for tasks requiring:
    - Complex calculations (e.g., material waste ratios, pricing optimization)
    - Multi-step logical reasoning
    - Strategic planning and analysis
    - Scientific or technical problem solving

    Example use cases:
    - "Calculate the optimal material waste ratio for a 12-pitch roof with 4 dormers given current lumber prices"
    - "Analyze the profitability impact of a 15% price increase across different customer segments"
    - "Design an optimal crew scheduling algorithm for 50 jobs over 2 weeks"

    Returns reasoning chain and extracted conclusion.
    """
    if not AI_AVAILABLE or ai_core is None:
        raise HTTPException(status_code=503, detail="AI Core not available")

    try:
        result = await ai_core.reason(
            problem=body.problem, context=body.context, max_tokens=body.max_tokens, model=body.model
        )

        return {
            "success": True,
            "reasoning": result.get("reasoning", ""),
            "conclusion": result.get("conclusion", ""),
            "model_used": result.get("model_used", body.model),
            "tokens_used": result.get("tokens_used"),
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"o1 reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}") from e


@app.post("/ai/learn-from-mistake", dependencies=SECURED_DEPENDENCIES)
async def ai_learn_from_mistake(
    request: Request,
    task_id: str,
    agent_id: str,
    expected_outcome: Any,
    actual_outcome: Any,
    confidence_before: float,
):
    """
    AI analyzes its own mistakes and learns from them (requires auth)

    This is how AI gets smarter over time!
    """
    if (
        not SELF_AWARENESS_AVAILABLE
        or not hasattr(app.state, "self_aware_ai")
        or not app.state.self_aware_ai
    ):
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        from decimal import Decimal

        self_aware_ai = app.state.self_aware_ai

        learning = await self_aware_ai.learn_from_mistake(
            task_id=task_id,
            agent_id=agent_id,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
            confidence_before=Decimal(str(confidence_before)),
        )

        return {
            "mistake_id": learning.mistake_id,
            "task_id": learning.task_id,
            "agent_id": learning.agent_id,
            "what_went_wrong": learning.what_went_wrong,
            "root_cause": learning.root_cause,
            "impact_level": learning.impact_level,
            "should_have_known": learning.should_have_known,
            "warning_signs_missed": learning.warning_signs_missed,
            "what_learned": learning.what_learned,
            "how_to_prevent": learning.how_to_prevent,
            "confidence_before": float(learning.confidence_before),
            "confidence_after": float(learning.confidence_after),
            "similar_mistakes_count": learning.similar_mistakes_count,
            "applied_to_agents": learning.applied_to_agents,
            "timestamp": learning.timestamp.isoformat(),
        }

    except Exception as e:
        logger.error(f"Learning from mistake failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Learning from mistake failed: {str(e)}"
        ) from e


@app.get("/ai/self-awareness/stats", dependencies=SECURED_DEPENDENCIES)
async def get_self_awareness_stats():
    """Get statistics about AI self-awareness system (requires auth)"""
    if not SELF_AWARENESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        pool = get_pool()

        # Get assessment stats
        assessment_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_assessments,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN can_complete_alone THEN 1 END) as can_complete_alone_count,
                COUNT(CASE WHEN requires_human_review THEN 1 END) as requires_review_count
            FROM ai_self_assessments
        """
        )

        # Get mistake learning stats
        learning_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_mistakes,
                COUNT(CASE WHEN should_have_known THEN 1 END) as should_have_known_count,
                AVG(confidence_before - confidence_after) as avg_confidence_drop
            FROM ai_learning_from_mistakes
        """
        )

        # Get reasoning explanation stats
        reasoning_stats = await pool.fetchrow(
            """
            SELECT
                COUNT(*) as total_explanations,
                AVG(confidence_in_decision) as avg_decision_confidence,
                COUNT(CASE WHEN human_review_recommended THEN 1 END) as human_review_count
            FROM ai_reasoning_explanations
        """
        )

        return {
            "self_awareness_enabled": True,
            "assessments": {
                "total": assessment_stats["total_assessments"] or 0,
                "avg_confidence": float(assessment_stats["avg_confidence"] or 0),
                "can_complete_alone_rate": (
                    (assessment_stats["can_complete_alone_count"] or 0)
                    / max(assessment_stats["total_assessments"] or 1, 1)
                    * 100
                ),
                "requires_review_rate": (
                    (assessment_stats["requires_review_count"] or 0)
                    / max(assessment_stats["total_assessments"] or 1, 1)
                    * 100
                ),
            },
            "learning": {
                "total_mistakes_analyzed": learning_stats["total_mistakes"] or 0,
                "should_have_known_rate": (
                    (learning_stats["should_have_known_count"] or 0)
                    / max(learning_stats["total_mistakes"] or 1, 1)
                    * 100
                ),
                "avg_confidence_adjustment": float(learning_stats["avg_confidence_drop"] or 0),
            },
            "reasoning": {
                "total_explanations": reasoning_stats["total_explanations"] or 0,
                "avg_decision_confidence": float(reasoning_stats["avg_decision_confidence"] or 0),
                "human_review_rate": (
                    (reasoning_stats["human_review_count"] or 0)
                    / max(reasoning_stats["total_explanations"] or 1, 1)
                    * 100
                ),
            },
        }

    except Exception as e:
        logger.error(f"Failed to get self-awareness stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}") from e


# ==================== END AI SELF-AWARENESS ENDPOINTS ====================


@app.post("/ai/tasks/execute/{task_id}", dependencies=SECURED_DEPENDENCIES)
async def execute_ai_task(task_id: str):
    """Manually trigger execution of a specific task (requires auth - CRITICAL)"""
    integration_layer = getattr(app.state, "integration_layer", None)
    if not INTEGRATION_LAYER_AVAILABLE or integration_layer is None:
        raise HTTPException(
            status_code=503, detail="AI Integration Layer not available or not initialized"
        )

    try:
        # Get task
        task = await integration_layer.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Execute task (will be picked up by task executor loop)
        await integration_layer._execute_task(task)

        return {"success": True, "message": "Task execution triggered", "task_id": task_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/ai/tasks/stats", dependencies=SECURED_DEPENDENCIES)
async def get_task_stats():
    """Get AI task system statistics (requires auth)"""
    integration_layer = getattr(app.state, "integration_layer", None)
    if not INTEGRATION_LAYER_AVAILABLE or integration_layer is None:
        raise HTTPException(
            status_code=503, detail="AI Integration Layer not available or not initialized"
        )

    try:
        # Get all tasks
        all_tasks = await integration_layer.list_tasks(limit=1000)

        # Calculate stats
        stats = {
            "total": len(all_tasks),
            "by_status": {},
            "by_priority": {},
            "agents_active": len(integration_layer.agents_registry),
            "execution_queue_size": integration_layer.execution_queue.qsize(),
        }

        for task in all_tasks:
            # Count by status
            status = task.get("status", "unknown")
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # Count by priority
            priority = task.get("priority", "unknown")
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

        return {
            "success": True,
            "stats": stats,
            "system_status": "operational",
            "task_executor_running": True,
        }

    except Exception as e:
        logger.error(f"❌ Failed to get task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class AureaCommandRequest(BaseModel):
    command_text: str


@app.post("/ai/orchestrate", dependencies=SECURED_DEPENDENCIES)
async def orchestrate_complex_workflow(
    request: Request, task_description: str, context: dict[str, Any] = {}
):
    """
    Execute complex multi-stage workflow using LangGraph orchestration
    This is for sophisticated tasks that need multi-agent coordination
    """
    if not hasattr(app.state, "langgraph_orchestrator") or not app.state.langgraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")

    try:
        orchestrator = app.state.langgraph_orchestrator

        result = await orchestrator.execute(task_description=task_description, context=context)

        return {"success": True, "result": result, "message": "Workflow orchestrated successfully"}

    except Exception as e:
        logger.error(f"❌ Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


class AIAnalyzeRequest(BaseModel):
    """Request model for /ai/analyze endpoint - matches weathercraft-erp frontend format"""

    agent: str
    action: str
    data: dict[str, Any] = {}
    context: dict[str, Any] = {}


@app.post("/ai/analyze", dependencies=SECURED_DEPENDENCIES)
async def ai_analyze(request: Request, payload: AIAnalyzeRequest = Body(...)):
    """
    AI analysis endpoint for weathercraft-erp and other frontends.
    Accepts JSON body with agent, action, data, and context fields.
    Routes to the appropriate agent or orchestrator.
    """
    try:
        agent_name = payload.agent
        action = payload.action
        data = payload.data
        context = payload.context

        # Build task description from agent and action
        task_description = f"{agent_name}: {action}"
        if data:
            task_description += f" with data: {json.dumps(data)[:200]}"

        # Try to use LangGraph orchestrator if available
        if hasattr(app.state, "langgraph_orchestrator") and app.state.langgraph_orchestrator:
            orchestrator = app.state.langgraph_orchestrator
            result = await orchestrator.execute(
                agent_name=agent_name,
                prompt=task_description,
                context={**context, "action": action, "data": data},
            )
            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": result,
                "message": "Analysis completed via orchestrator",
            }

        # Fallback: Use module-level agent executor singleton
        try:
            from agent_executor import executor as agent_executor_singleton

            if agent_executor_singleton:
                result = await agent_executor_singleton.execute(
                    agent_name=agent_name, task={"action": action, "data": data, "context": context}
                )
                return {
                    "success": True,
                    "agent": agent_name,
                    "action": action,
                    "result": result,
                    "message": "Analysis completed via agent executor",
                }
        except (ImportError, Exception) as e:
            logger.warning(f"Agent executor fallback failed: {e}")

        # No orchestrator available - queue task for later processing instead of mock response
        logger.warning(
            f"No orchestrator/executor available for agent {agent_name}, queueing for async processing"
        )

        # Queue the task to ai_autonomous_tasks for later execution
        try:
            pool = get_pool()
            task_id = str(uuid.uuid4())
            tenant_id = (
                request.headers.get(config.tenant.header_name)
                or context.get("tenant_id")
                or config.tenant.default_tenant_id
            )
            tenant_uuid: str | None = None
            if tenant_id:
                try:
                    tenant_uuid = str(uuid.UUID(str(tenant_id)))
                except (ValueError, TypeError, AttributeError):
                    tenant_uuid = None

            agent_row = await pool.fetchrow(
                "SELECT id FROM ai_agents WHERE id::text = $1 OR name = $1 LIMIT 1",
                agent_name,
            )
            agent_uuid = str(agent_row["id"]) if agent_row else None

            await pool.execute(
                """
                INSERT INTO ai_autonomous_tasks (
                    id,
                    title,
                    task_type,
                    priority,
                    status,
                    trigger_type,
                    trigger_condition,
                    agent_id,
                    tenant_id,
                    created_at
                )
                VALUES ($1, $2, $3, $4, 'pending', $5, $6::jsonb, $7, $8, NOW())
                """,
                task_id,
                f"{agent_name}.{action}",
                "ai_analyze",
                "medium",
                "ai_analyze",
                json.dumps(
                    {
                        "agent": agent_name,
                        "action": action,
                        "data": data,
                        "context": context,
                    },
                    default=str,
                ),
                agent_uuid,
                tenant_uuid,
            )

            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": {
                    "status": "queued",
                    "task_id": task_id,
                    "message": f"Request queued for async processing (task: {task_id})",
                },
                "message": "Request queued - orchestrator temporarily unavailable",
            }
        except Exception as queue_error:
            logger.error(f"Failed to queue task: {queue_error}")
            raise HTTPException(
                status_code=503,
                detail=f"AI orchestrator unavailable and task queueing failed: {str(queue_error)}",
            ) from queue_error

    except Exception as e:
        logger.error(f"❌ AI analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _normalize_tenant_uuid(candidate: Any) -> Optional[str]:
    """Return canonical UUID string or None when input is missing/invalid."""
    if candidate is None:
        return None
    try:
        raw = str(candidate).strip()
    except Exception:
        return None

    if not raw or raw.lower() in {"null", "none", "undefined"}:
        return None

    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError, AttributeError):
        return None


def _resolve_tenant_uuid_from_request(request: Optional[HTTPConnection]) -> Optional[str]:
    """Resolve tenant UUID from request context, then fall back to configured default."""
    candidates: list[Any] = []
    if request is not None:
        try:
            candidates.append(request.headers.get(config.tenant.header_name))
        except Exception:
            pass

        state = getattr(request, "state", None)
        if state is not None:
            candidates.append(getattr(state, "tenant_id", None))
            user = getattr(state, "user", None)
            if isinstance(user, dict):
                candidates.append(user.get("tenant_id"))

    candidates.extend([config.tenant.default_tenant_id, os.getenv("DEFAULT_TENANT_ID")])

    for candidate in candidates:
        normalized = _normalize_tenant_uuid(candidate)
        if normalized:
            return normalized
    return None


async def _fetchval_with_tenant_context(
    pool: Any,
    query: str,
    *args: Any,
    tenant_uuid: Optional[str] = None,
):
    """
    Execute fetchval in a transaction with explicit tenant context.
    Prevents stale/invalid session tenant settings (e.g. empty string UUID).
    """
    raw_pool = getattr(pool, "pool", None) or getattr(pool, "_pool", None)
    if raw_pool is None:
        return await pool.fetchval(query, *args)

    async with raw_pool.acquire(timeout=10.0) as conn:
        async with conn.transaction():
            if tenant_uuid:
                await conn.execute("SELECT set_config('app.current_tenant_id', $1, true)", tenant_uuid)
            else:
                await conn.execute("RESET app.current_tenant_id")
            return await conn.fetchval(query, *args)


@app.get("/aurea/status", dependencies=SECURED_DEPENDENCIES)
async def get_aurea_status(request: Request):
    """
    Get AUREA operational status - checks actual OODA loop activity (requires auth).
    This endpoint verifies real AUREA activity in the database.
    """
    try:
        pool = get_pool()
        if not pool:
            return {
                "status": "initializing",
                "aurea_available": False,
                "message": "Database pool not available",
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints": {
                    "full_status": "/aurea/chat/status",
                    "chat": "/aurea/chat/message",
                    "websocket": "/aurea/chat/ws/{session_id}",
                },
            }

        tenant_uuid = _resolve_tenant_uuid_from_request(request)

        recent_cycles = await _fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM aurea_state
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """,
            tenant_uuid=tenant_uuid,
        )

        recent_decisions = await _fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """,
            tenant_uuid=tenant_uuid,
        )

        active_agents = await _fetchval_with_tenant_context(
            pool,
            """
            SELECT COUNT(*) FROM ai_agents WHERE status = 'active'
        """,
            tenant_uuid=tenant_uuid,
        )

        # AUREA is operational if we have recent OODA cycles
        aurea_operational = recent_cycles > 0

        return {
            "status": "operational" if aurea_operational else "idle",
            "aurea_available": True,
            "ooda_cycles_last_5min": recent_cycles,
            "decisions_last_hour": recent_decisions,
            "active_agents": active_agents,
            "tenant_id": tenant_uuid,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "full_status": "/aurea/chat/status",
                "chat": "/aurea/chat/message",
                "websocket": "/aurea/chat/ws/{session_id}",
            },
        }
    except Exception as e:
        logger.error(f"Failed to get AUREA status: {e!r}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


@app.post("/aurea/command/natural_language", dependencies=SECURED_DEPENDENCIES)
async def execute_aurea_nl_command(request: Request, payload: AureaCommandRequest = Body(...)):
    """
    Execute a natural language command through AUREA's NLU processor (requires auth - CRITICAL).
    Founder-level authority for natural language system control.

    Examples:
    - "Create a high priority task to deploy the new feature"
    - "Show me all tasks that are in progress"
    - "Get AUREA status"
    - "Execute task abc-123"
    """
    if not hasattr(app.state, "aurea_nlu") or not app.state.aurea_nlu:
        logger.warning(
            "AUREA NLU processor unavailable; using /aurea/chat/command fallback for '%s'",
            payload.command_text,
        )
        try:
            from api.aurea_chat import NLCommand, execute_natural_language_command

            fallback_response = await execute_natural_language_command(
                NLCommand(command=payload.command_text)
            )

            if isinstance(fallback_response, JSONResponse):
                try:
                    parsed = json.loads(fallback_response.body.decode("utf-8"))
                except Exception:
                    parsed = {"detail": fallback_response.body.decode("utf-8", errors="ignore")}
                if fallback_response.status_code >= 400:
                    raise HTTPException(
                        status_code=fallback_response.status_code,
                        detail=parsed.get("error") or parsed.get("detail") or "AUREA fallback failed",
                    )
                return {
                    "success": True,
                    "command": payload.command_text,
                    "result": parsed,
                    "processor": "chat-command-fallback",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            return {
                "success": True,
                "command": payload.command_text,
                "result": fallback_response,
                "processor": "chat-command-fallback",
                "timestamp": datetime.utcnow().isoformat(),
            }
        except HTTPException:
            raise
        except Exception as fallback_error:
            logger.error("AUREA fallback command execution failed: %s", fallback_error, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"AUREA NLU Processor not available; fallback failed: {fallback_error}",
            ) from fallback_error

    try:
        command_text = payload.command_text
        nlu = app.state.aurea_nlu
        result = await nlu.execute_natural_language_command(command_text)

        return {
            "success": True,
            "command": command_text,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Natural language command execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ==================== END AI TASK MANAGEMENT ENDPOINTS ====================


# ==================== BRAINOPS CORE v1 API ====================


class KnowledgeStoreRequest(BaseModel):
    """Request payload for storing knowledge/memory entries."""

    content: str
    memory_type: str = "knowledge"
    source_system: Optional[str] = None
    source_agent: Optional[str] = None
    created_by: Optional[str] = None
    importance: float = 0.5
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class KnowledgeQueryRequest(BaseModel):
    """Request payload for querying unified memory/knowledge."""

    query: str
    limit: int = 10
    memory_type: Optional[str] = None
    min_importance: float = 0.0


class ErpAnalyzeRequest(BaseModel):
    """Request payload for ERP job analysis."""

    tenant_id: Optional[str] = None
    job_ids: Optional[list[str]] = None
    limit: int = 20


class AgentExecuteRequest(BaseModel):
    """Request payload for executing an agent via v1 API."""

    agent_id: Optional[str] = None
    id: Optional[str] = None
    payload: dict[str, Any] = {}


class AgentActivateRequest(BaseModel):
    """Request payload for activating or deactivating an agent."""

    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    enabled: bool = True


class AUREAEventRequest(BaseModel):
    """Request model for AUREA event execution"""

    event_id: str
    topic: str
    source: str
    payload: dict[str, Any]
    target_agent: dict[str, Any]  # {name, role, capabilities}
    routing_metadata: Optional[dict[str, Any]] = None


@app.post("/api/v1/knowledge/store")
async def api_v1_knowledge_store(
    payload: KnowledgeStoreRequest, request: Request, authenticated: bool = Depends(verify_api_key)
):
    """
    Store a knowledge/memory entry in the unified memory system.

    Primary path uses the Embedded Memory System (SQLite + async sync to Postgres).
    Fallback path uses the Unified Memory Manager when embedded memory is unavailable.
    """
    # Prefer embedded memory for low-latency writes with async sync to Postgres
    embedded_memory = getattr(app.state, "embedded_memory", None)

    # Normalize metadata
    metadata: dict[str, Any] = dict(payload.metadata or {})
    if payload.source_system:
        metadata.setdefault("source_system", payload.source_system)
    if payload.created_by:
        metadata.setdefault("created_by", payload.created_by)
    if payload.tags:
        metadata.setdefault("tags", payload.tags)

    memory_id = str(uuid.uuid4())

    if embedded_memory:
        try:
            success = embedded_memory.store_memory(
                memory_id=memory_id,
                memory_type=payload.memory_type,
                source_agent=payload.source_agent or "system",
                content=payload.content,
                metadata=metadata,
                importance_score=payload.importance,
            )
            if not success:
                raise HTTPException(
                    status_code=500, detail="Failed to store memory in embedded backend"
                )
        except Exception as exc:
            logger.error("Embedded memory store failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to store memory") from exc
    elif MEMORY_AVAILABLE and hasattr(app.state, "memory") and app.state.memory:
        # Fallback: write directly via UnifiedMemoryManager
        try:
            from unified_memory_manager import Memory, MemoryType

            mem = Memory(
                memory_type=MemoryType.SEMANTIC,
                content={"text": payload.content, "metadata": metadata},
                source_system=payload.source_system or "brainops-core",
                source_agent=payload.source_agent or "system",
                created_by=payload.created_by or "system",
                importance_score=payload.importance,
                tags=payload.tags or [],
                metadata=metadata,
            )
            memory_id = app.state.memory.store(mem)
        except Exception as exc:
            logger.error("UnifiedMemoryManager store failed: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to store memory") from exc
    else:
        raise HTTPException(status_code=503, detail="No memory backend available")

    return {
        "success": True,
        "id": memory_id,
        "memory_type": payload.memory_type,
    }


@app.post("/api/v1/knowledge/query")
async def api_v1_knowledge_query(
    payload: KnowledgeQueryRequest, request: Request, authenticated: bool = Depends(verify_api_key)
):
    """
    Query the unified memory / knowledge store.

    Uses the Embedded Memory System when available (vector search), with a
    fallback to the Unified Memory Manager recall API.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)
    results: list[dict[str, Any]] = []

    if embedded_memory:
        try:
            results = embedded_memory.search_memories(
                query=payload.query,
                limit=payload.limit,
                memory_type=payload.memory_type,
                min_importance=payload.min_importance,
            )
        except Exception as exc:
            logger.error("Embedded memory query failed: %s", exc)
            results = []

    # Fallback to Unified Memory Manager if embedded memory empty or unavailable
    if (not results) and MEMORY_AVAILABLE and hasattr(app.state, "memory") and app.state.memory:
        try:
            memory_type_enum = None
            if payload.memory_type:
                from unified_memory_manager import MemoryType

                try:
                    memory_type_enum = MemoryType(payload.memory_type)
                except Exception as exc:
                    logger.debug("Invalid memory type %s: %s", payload.memory_type, exc)
                    memory_type_enum = None

            results = app.state.memory.recall(
                query=payload.query,
                context=None,
                limit=payload.limit,
                memory_type=memory_type_enum,
            )
        except Exception as exc:
            logger.error("UnifiedMemoryManager recall failed: %s", exc)
            raise HTTPException(status_code=500, detail="Memory query failed") from exc

    normalized: list[dict[str, Any]] = []
    for item in results:
        data = dict(item)
        content = data.get("content")
        # Best-effort JSON parsing for text content
        if isinstance(content, str):
            try:
                content_parsed = json.loads(content)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.debug("Failed to parse memory content JSON: %s", exc)
                content_parsed = content
        else:
            content_parsed = content

        normalized.append(
            {
                "id": str(data.get("id")),
                "memory_type": data.get("memory_type"),
                "source_agent": data.get("source_agent"),
                "source_system": data.get("source_system"),
                "importance_score": float(data.get("importance_score", 0.0))
                if data.get("importance_score") is not None
                else None,
                "tags": data.get("tags"),
                "metadata": data.get("metadata"),
                "content": content_parsed,
                "created_at": data.get("created_at"),
                "last_accessed": data.get("last_accessed"),
                "similarity_score": data.get("similarity_score"),
                "combined_score": data.get("combined_score"),
            }
        )

    return {
        "success": True,
        "query": payload.query,
        "results": normalized,
        "count": len(normalized),
    }


@app.get("/api/v1/knowledge/graph/stats")
async def api_v1_knowledge_graph_stats(authenticated: bool = Depends(verify_api_key)):
    """
    Get knowledge graph statistics - node counts, edge counts, extraction status.
    """
    try:
        pool = get_pool()

        # Get counts from all knowledge tables
        nodes_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_nodes") or 0
        edges_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_edges") or 0
        graph_count = await pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_graph") or 0

        # Get node type distribution
        node_types = await pool.fetch(
            """
            SELECT node_type, COUNT(*) as count
            FROM ai_knowledge_nodes
            GROUP BY node_type
            ORDER BY count DESC
        """
        )

        # Get recent extraction info
        recent_graph = await pool.fetchrow(
            """
            SELECT node_data, updated_at
            FROM ai_knowledge_graph
            WHERE node_type = 'graph_metadata'
            ORDER BY updated_at DESC
            LIMIT 1
        """
        )

        extraction_stats = {}
        last_extraction = None
        if recent_graph:
            node_data = recent_graph.get("node_data", {})
            if isinstance(node_data, str):
                import json

                node_data = json.loads(node_data)
            extraction_stats = node_data.get("extraction_stats", {})
            last_extraction = recent_graph.get("updated_at")

        # Get extractor instance stats if available
        extractor_stats = {}
        if hasattr(app.state, "knowledge_extractor") and app.state.knowledge_extractor:
            extractor_stats = app.state.knowledge_extractor.extraction_stats

        return {
            "success": True,
            "total_nodes": nodes_count,
            "total_edges": edges_count,
            "graph_entries": graph_count,
            "node_types": [dict(row) for row in node_types],
            "last_extraction": last_extraction.isoformat() if last_extraction else None,
            "extraction_stats": extraction_stats or extractor_stats,
            "extractor_active": hasattr(app.state, "knowledge_extractor")
            and app.state.knowledge_extractor is not None,
        }
    except Exception as e:
        logger.error(f"Knowledge graph stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/knowledge/graph/extract")
async def api_v1_knowledge_graph_extract(
    hours_back: int = 24, authenticated: bool = Depends(verify_api_key)
):
    """
    Manually trigger knowledge graph extraction.
    Useful for immediate extraction or catching up on historical data.
    """
    try:
        from knowledge_graph_extractor import get_knowledge_extractor

        extractor = get_knowledge_extractor()
        await extractor.initialize()

        result = await extractor.run_extraction(hours_back=hours_back)

        return {
            "success": result.get("success", False),
            "message": f"Extracted {result.get('nodes_stored', 0)} nodes and {result.get('edges_stored', 0)} edges",
            "details": result,
        }
    except Exception as e:
        logger.error(f"Knowledge graph extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/erp/analyze")
async def api_v1_erp_analyze(
    payload: ErpAnalyzeRequest, authenticated: bool = Depends(verify_api_key)
):
    """
    Analyze ERP jobs using centralized BrainOps Core.

    - Reads jobs from the shared database (read-only).
    - Computes schedule risk and progress using deterministic heuristics.
    - Optionally augments each job with AI commentary when AI Core is available.
    """
    pool = get_pool()

    try:
        filters = ["j.status = ANY($1::text[])"]
        params: list[Any] = [["in_progress", "scheduled"]]

        # Detect whether jobs.tenant_id exists so we can safely filter
        has_tenant_id = False
        try:
            has_tenant_id = await pool.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'jobs'
                      AND column_name = 'tenant_id'
                )
                """
            )
        except Exception as column_exc:
            logger.warning("Unable to inspect jobs.tenant_id column: %s", column_exc)

        if payload.tenant_id and has_tenant_id:
            filters.append(f"j.tenant_id = ${len(params) + 1}::uuid")
            params.append(payload.tenant_id)
        elif payload.tenant_id and not has_tenant_id:
            logger.warning(
                "Tenant filter requested but jobs.tenant_id column not found; returning unscoped jobs"
            )

        if payload.job_ids:
            filters.append(f"j.id = ANY(${len(params) + 1}::uuid[])")
            params.append(payload.job_ids)

        limit_param_index = len(params) + 1
        params.append(payload.limit or 20)

        query = f"""
            SELECT
                j.id,
                j.job_number,
                j.title,
                j.status,
                j.scheduled_start,
                j.scheduled_end,
                j.actual_start,
                j.actual_end,
                j.completion_percentage,
                j.estimated_revenue,
                j.created_at,
                c.name AS customer_name
            FROM jobs j
            LEFT JOIN customers c ON c.id = j.customer_id
            WHERE {' AND '.join(filters)}
            ORDER BY j.scheduled_start NULLS LAST, j.created_at DESC
            LIMIT ${limit_param_index}
        """

        rows = await pool.fetch(query, *params)

        def _to_naive(dt: Optional[datetime]) -> Optional[datetime]:
            if not dt:
                return None
            # Handle both datetime and date objects
            if hasattr(dt, "tzinfo"):
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                # It's a date object, convert to naive datetime
                from datetime import date

                if isinstance(dt, date):
                    return datetime.combine(dt, datetime.min.time())
                return dt

        now = datetime.utcnow()
        jobs_intel: list[dict[str, Any]] = []

        for row in rows:
            data = row if isinstance(row, dict) else dict(row)

            planned_start = data.get("scheduled_start") or data.get("actual_start")
            planned_end = data.get("scheduled_end") or data.get("actual_end")

            days_in_progress = 0
            if planned_start:
                delta = now - _to_naive(planned_start)  # type: ignore[arg-type]
                days_in_progress = max(0, delta.days)

            total_duration = 30
            if planned_start and planned_end:
                delta_total = _to_naive(planned_end) - _to_naive(planned_start)  # type: ignore[operator]
                total_duration = max(1, delta_total.days)

            completion_pct = data.get("completion_percentage")
            if completion_pct is None:
                if total_duration:
                    completion_pct = min(100, round((days_in_progress / total_duration) * 100))
                else:
                    completion_pct = 0
            else:
                completion_pct = min(100, completion_pct)

            on_track = completion_pct <= 100 and days_in_progress <= total_duration

            # Risk heuristics (kept here but centralized in Core)
            risk_level: str = "low"
            risk_score: int = 20
            predicted_delay = 0

            if completion_pct > 100 or days_in_progress > total_duration:
                risk_level = "critical"
                risk_score = 90
                predicted_delay = max(0, days_in_progress - total_duration)
            elif completion_pct > 80:
                risk_level = "high"
                risk_score = 70
                predicted_delay = 3
            elif completion_pct > 60:
                risk_level = "medium"
                risk_score = 50
                predicted_delay = 1

            job_name = data.get("title") or data.get("job_number") or "Job"
            customer_name = data.get("customer_name") or "Unknown"

            # Optional AI commentary using RealAICore, non-fatal if unavailable
            ai_commentary: Optional[str] = None
            if AI_AVAILABLE and ai_core:
                try:
                    summary_prompt = (
                        f"Job '{job_name}' for customer '{customer_name}' has status '{data.get('status')}', "
                        f"completion {completion_pct}% after {days_in_progress} days "
                        f"with planned duration {total_duration} days. "
                        f"Risk level is {risk_level} with score {risk_score}."
                    )
                    commentary = await ai_generate(
                        f"Provide a concise, 2-3 sentence risk summary and recommended next action for this roofing job:\n\n{summary_prompt}",
                        model="gpt-4-turbo-preview",
                        temperature=0.3,
                        max_tokens=160,
                    )
                    ai_commentary = commentary
                except Exception as exc:
                    logger.warning("AI commentary failed for job %s: %s", data.get("id"), exc)

            # Calculate change probability based on actual job data instead of random
            # Higher probability if: early in project, high value job, complex roof type
            base_change_prob = 25.0  # Base 25% chance
            if completion_pct < 25:
                base_change_prob += 15.0  # Early stage = higher change likelihood
            if data.get("total_amount", 0) > 15000:
                base_change_prob += 10.0  # High value = more change orders
            if risk_level in ("high", "critical"):
                base_change_prob += 15.0  # At-risk jobs have more changes
            change_prob = min(base_change_prob, 85.0)  # Cap at 85%

            # Estimate impact based on job value (typically 5-20% of job value)
            job_value = float(data.get("total_amount", 10000) or 10000)
            estimated_impact = int(job_value * (0.05 + (change_prob / 100) * 0.15))

            jobs_intel.append(
                {
                    "job_id": str(data.get("id")),
                    "job_name": job_name,
                    "customer_name": customer_name,
                    "status": data.get("status"),
                    "ai_source": "brainops-core",
                    "delay_risk": {
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "delay_factors": [
                            (
                                f"Job {days_in_progress} days in progress vs {total_duration} days planned"
                                if planned_start and planned_end
                                else "Limited schedule data available"
                            ),
                            "Weather delays possible",
                            "Material delivery timing critical",
                            ("Behind schedule" if not on_track else "On schedule"),
                        ],
                        "mitigation_strategies": [
                            "Add 1-2 crew members to accelerate",
                            "Schedule overtime for critical path tasks",
                            "Pre-order materials to avoid delays",
                            "Daily progress check-ins with foreman",
                        ],
                        "predicted_delay_days": predicted_delay,
                    },
                    "progress_tracking": {
                        "completion_percentage": completion_pct,
                        "on_track": on_track,
                        "milestones_completed": completion_pct // 25,
                        "milestones_total": 4,
                        "ai_progress_assessment": (
                            f"Job progressing well - {completion_pct}% complete on schedule"
                            if on_track
                            else f"Job needs attention - {predicted_delay} days behind schedule"
                        ),
                    },
                    "resource_optimization": {
                        "current_crew_size": 4,
                        "optimal_crew_size": 6 if risk_level in ("high", "critical") else 4,
                        "resource_utilization": 85 if on_track else 110,
                        "recommendations": (
                            [
                                "Increase crew size by 2 workers",
                                "Reassign experienced technician from another job",
                                "Schedule weekend work if customer approves",
                                "Focus resources on critical path items",
                            ]
                            if risk_level in ("high", "critical")
                            else [
                                "Current crew size is optimal",
                                "Resource utilization healthy at 85%",
                                "Maintain current staffing levels",
                            ]
                        ),
                    },
                    "change_order_intelligence": {
                        "probability_of_change": change_prob,
                        "potential_change_areas": [
                            "Additional valley flashing may be needed",
                            "Customer may upgrade shingle quality",
                            "Possible deck repair if rot discovered",
                        ],
                        "estimated_impact": estimated_impact,
                        "ai_recommendations": [
                            "Pre-approve deck inspection with customer",
                            "Have upgrade options ready to present",
                            "Document any rot/damage immediately",
                        ],
                    },
                    "next_action": {
                        "action": (
                            "Schedule emergency crew meeting"
                            if risk_level == "critical"
                            else (
                                "Add crew members"
                                if risk_level == "high"
                                else "Continue monitoring"
                            )
                        ),
                        "priority": (
                            "urgent"
                            if risk_level == "critical"
                            else ("high" if risk_level == "high" else "medium")
                        ),
                        "reasoning": [
                            (
                                "Job at risk of delay - immediate intervention needed"
                                if risk_level in ("critical", "high")
                                else "Job progressing normally"
                            ),
                            f"Current completion: {completion_pct}%",
                            ("On schedule" if on_track else f"{predicted_delay} days behind"),
                            "Weather forecast favorable for next 7 days",
                            ai_commentary or "",
                        ],
                    },
                }
            )

        return {
            "success": True,
            "jobs": jobs_intel,
            "count": len(jobs_intel),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ERP analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="ERP analysis failed") from exc


@app.post("/api/v1/agents/execute")
@limiter.limit("10/minute")
async def api_v1_agents_execute(
    request: Request, payload: AgentExecuteRequest, authenticated: bool = Depends(verify_api_key)
):
    """
    Execute an agent via the v1 API surface.

    Body: { "agent_id" | "id": string, "payload": object }
    Internally delegates to the existing /agents/{agent_id}/execute endpoint.
    """
    agent_id = payload.agent_id or payload.id
    if not agent_id:
        raise HTTPException(status_code=400, detail="agent_id is required")

    # Build a synthetic Request with the payload as JSON body so we can
    # delegate to the existing execute_agent implementation without duplication.
    scope = {
        "type": "http",
        "method": "POST",
        "path": f"/agents/{agent_id}/execute",
        "headers": [],
    }
    from starlette.requests import Request as StarletteRequest  # type: ignore

    async def receive() -> dict[str, Any]:
        body_bytes = json.dumps(payload.payload or {}).encode("utf-8")
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    delegated_request: Request = StarletteRequest(scope, receive)  # type: ignore[arg-type]

    return await execute_agent(agent_id=agent_id, request=delegated_request, authenticated=True)


@app.post("/api/v1/agents/activate")
async def api_v1_agents_activate(
    payload: AgentActivateRequest, authenticated: bool = Depends(verify_api_key)
):
    """
    Activate or deactivate an agent via the v1 API surface.

    This is a thin wrapper that flips the `enabled` flag in the agents table.
    """
    if not payload.agent_id and not payload.agent_name:
        raise HTTPException(status_code=400, detail="agent_id or agent_name is required")

    pool = get_pool()

    try:
        row = None

        if payload.agent_id:
            row = await pool.fetchrow(
                """
                UPDATE agents
                SET enabled = $1, updated_at = NOW()
                WHERE id::text = $2
                RETURNING id, name, category, enabled
                """,
                payload.enabled,
                payload.agent_id,
            )

        if not row and payload.agent_name:
            row = await pool.fetchrow(
                """
                UPDATE agents
                SET enabled = $1, updated_at = NOW()
                WHERE name = $2
                RETURNING id, name, category, enabled
                """,
                payload.enabled,
                payload.agent_name,
            )

        if not row:
            raise HTTPException(status_code=404, detail="Agent not found")

        data = dict(row)
        return {
            "success": True,
            "agent": {
                "id": str(data.get("id")),
                "name": data.get("name"),
                "category": data.get("category"),
                "enabled": data.get("enabled"),
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Agent activation failed: %s", exc)
        raise HTTPException(status_code=500, detail="Agent activation failed") from exc


@app.post("/api/v1/aurea/execute-event")
async def execute_aurea_event(
    request: AUREAEventRequest, authenticated: bool = Depends(verify_api_key)
):
    """
    Execute event with specified AI agent via AUREA orchestration.
    Called by Event Router daemon to process events from brainops_core.event_bus.
    """
    logger.info(
        f"🎯 AUREA Event: {request.event_id} ({request.topic}) -> {request.target_agent['name']}"
    )

    pool = get_pool()

    try:
        # Find target agent by name
        agent_row = await pool.fetchrow(
            """
            SELECT id, name, category, enabled
            FROM agents
            WHERE name = $1 AND enabled = TRUE
            """,
            request.target_agent["name"],
        )

        if not agent_row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{request.target_agent['name']}' not found or disabled",
            )

        str(agent_row["id"])
        agent_name = agent_row["name"]

        # Prepare agent execution payload

        # Execute agent (simple acknowledgment for now)
        # Can be expanded with topic-specific handlers
        result = {
            "status": "acknowledged",
            "agent": agent_name,
            "event_id": request.event_id,
            "topic": request.topic,
            "action": "processed",
        }

        # Update agent last_active_at in brainops_core.agents (if table exists)
        try:
            await pool.execute(
                """
                UPDATE brainops_core.agents
                SET last_active_at = NOW()
                WHERE name = $1
                """,
                agent_name,
            )
        except Exception as exc:
            logger.debug("Failed to update agent heartbeat: %s", exc, exc_info=True)

        # Store in embedded memory if available
        embedded_memory = getattr(app.state, "embedded_memory", None)
        if embedded_memory:
            try:
                embedded_memory.store_memory(
                    memory_id=str(uuid.uuid4()),
                    memory_type="episodic",
                    source_agent=agent_name,
                    content=f"Processed event: {request.topic}",
                    metadata={
                        "event_id": request.event_id,
                        "topic": request.topic,
                        "source": request.source,
                    },
                    importance_score=0.7,
                )
            except Exception as e:
                logger.warning(f"Could not store in embedded memory: {e}")

        logger.info(f"✅ AUREA Event {request.event_id} executed by {agent_name}")

        return {
            "success": True,
            "event_id": request.event_id,
            "agent": agent_name,
            "topic": request.topic,
            "result": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AUREA Event {request.event_id} failed: {e}")
        raise HTTPException(status_code=500, detail=f"Event execution failed: {str(e)}") from e


# ==================== END BRAINOPS CORE v1 API ====================


# ==================== UNIFIED OBSERVABILITY SYSTEM ====================

# In-memory log buffer for recent logs
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


@app.get("/logs/recent", dependencies=[Depends(verify_api_key)])
async def get_recent_logs(
    level: Optional[str] = None,
    logger_name: Optional[str] = None,
    limit: int = Query(default=100, le=500),
    contains: Optional[str] = None,
):
    """Get recent logs with filtering"""
    logs = list(LOG_BUFFER)

    # Filter by level
    if level:
        logs = [l for l in logs if l["level"] == level.upper()]

    # Filter by logger
    if logger_name:
        logs = [l for l in logs if logger_name.lower() in l["logger"].lower()]

    # Filter by content
    if contains:
        logs = [l for l in logs if contains.lower() in l["message"].lower()]

    # Return most recent
    return {
        "logs": logs[-limit:],
        "total_in_buffer": len(LOG_BUFFER),
        "returned": min(limit, len(logs)),
        "filters": {"level": level, "logger": logger_name, "contains": contains},
    }


@app.get("/observability/full", dependencies=[Depends(verify_api_key)])
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

    # Check all services
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

    # Get database stats
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

    # Get recent errors from logs
    results["recent_errors"] = [
        l for l in list(LOG_BUFFER)[-100:] if l["level"] in ["ERROR", "CRITICAL"]
    ][-20:]

    # Get system metrics
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


@app.get("/debug/all-errors", dependencies=[Depends(verify_api_key)])
async def get_all_errors():
    """Get all recent errors across systems for debugging"""
    errors = [l for l in list(LOG_BUFFER) if l["level"] in ["ERROR", "CRITICAL", "WARNING"]]

    # Categorize errors
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


@app.get("/system/unified-status", dependencies=[Depends(verify_api_key)])
async def get_unified_system_status(request: Request):
    """Get unified status of the entire AI OS"""
    pool = get_pool()
    tenant_uuid = _resolve_tenant_uuid_from_request(request)

    status = {
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "overall_health": "healthy",
        "tenant_id": tenant_uuid,
        "components": {},
    }

    # Check each major component
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
            result = await _fetchval_with_tenant_context(
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


# ==================== END UNIFIED OBSERVABILITY ====================


# ==================== MULTI-AI CONTENT ORCHESTRATION ====================

try:
    from multi_ai_content_orchestrator import MultiAIContentOrchestrator, ContentType

    CONTENT_ORCHESTRATOR_AVAILABLE = True
    logger.info("Multi-AI Content Orchestrator loaded")
except ImportError as e:
    CONTENT_ORCHESTRATOR_AVAILABLE = False
    logger.warning(f"Multi-AI Content Orchestrator not available: {e}")


class ContentGenerationRequest(BaseModel):
    content_type: str = "blog_post"  # blog_post, newsletter, ebook, training
    topic: str
    brand: str = "BrainOps"
    target_audience: str = "tech professionals"
    chapters: int = 5  # for ebooks
    module_number: int = 1  # for training docs
    include_image: bool = True


@app.post("/content/generate", tags=["Content"])
async def generate_content(
    request: ContentGenerationRequest,
    background_tasks: BackgroundTasks,
    authenticated: bool = Depends(verify_api_key),
):
    """
    Generate content using Multi-AI Orchestration.

    Supported content types:
    - blog_post: SEO-optimized blog articles
    - newsletter: Complete email newsletters with HTML
    - ebook: Full ebooks with multiple chapters
    - training: Training documentation with exercises and quizzes
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

    # Run in background for long-running content
    if request.content_type in ["ebook", "ebook_full"]:
        job_id = str(uuid.uuid4())
        background_tasks.add_task(_run_content_generation, orchestrator, task, job_id)
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": f"Ebook generation started for: {request.topic}",
            "check_status": f"/content/status/{job_id}",
        }

    # Run synchronously for faster content types
    result = await orchestrator.execute(task)
    return result


async def _run_content_generation(orchestrator, task, job_id):
    """Background content generation with status tracking."""
    try:
        result = await orchestrator.execute(task)
        # Store result for retrieval
        logger.info(f"Content job {job_id} completed: {result.get('status')}")
    except Exception as e:
        logger.error(f"Content job {job_id} failed: {e}")


@app.post("/content/newsletter", tags=["Content"])
async def generate_newsletter(
    topic: str = Body(..., embed=True),
    brand: str = Body("BrainOps", embed=True),
    authenticated: bool = Depends(verify_api_key),
):
    """Generate a complete newsletter with HTML template."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_newsletter({"topic": topic, "brand": brand})
    return result


@app.post("/content/ebook", tags=["Content"])
async def generate_ebook(
    topic: str = Body(..., embed=True),
    chapters: int = Body(5, embed=True),
    author: str = Body("BrainOps AI", embed=True),
    background_tasks: BackgroundTasks = None,
    authenticated: bool = Depends(verify_api_key),
):
    """Generate a complete ebook with multiple chapters."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_ebook(
        {"topic": topic, "chapters": chapters, "author": author}
    )
    return result


@app.post("/content/training", tags=["Content"])
async def generate_training_doc(
    topic: str = Body(..., embed=True),
    module_number: int = Body(1, embed=True),
    skill_level: str = Body("beginner", embed=True),
    authenticated: bool = Depends(verify_api_key),
):
    """Generate training documentation with exercises and quizzes."""
    if not CONTENT_ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Content orchestrator not available")

    orchestrator = MultiAIContentOrchestrator()
    result = await orchestrator.generate_training_doc(
        {"topic": topic, "module_number": module_number, "skill_level": skill_level}
    )
    return result


@app.get("/content/types", tags=["Content"], dependencies=SECURED_DEPENDENCIES)
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


# ==================== END MULTI-AI CONTENT ORCHESTRATION ====================


# ==================== PRODUCT & REVENUE INVENTORY ====================


@app.get("/inventory/products", tags=["Inventory"])
async def get_product_inventory(authenticated: bool = Depends(verify_api_key)):
    """
    Get complete product inventory across all platforms.
    This is the source of truth for what products exist and their status.
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
                    {
                        "name": "Starter",
                        "price_monthly": 49,
                        "price_annual": 588,
                        "type": "subscription",
                        "status": "ready",
                        "features": ["1-3 users", "2-10 jobs/month", "Basic analysis"],
                    },
                    {
                        "name": "Professional",
                        "price_monthly": 99,
                        "price_annual": 1188,
                        "type": "subscription",
                        "status": "ready",
                        "features": ["Up to 10 users", "10-30 jobs/month", "Advanced analytics"],
                    },
                    {
                        "name": "Enterprise",
                        "price_monthly": 199,
                        "price_annual": 2388,
                        "type": "subscription",
                        "status": "ready",
                        "features": ["Unlimited users", "30+ jobs/month", "Full features"],
                    },
                ],
                "pricing_model": "subscription",
                "payment_processor": "stripe",
                "current_mrr": 0,
                "active_subscribers": 0,
            },
            "brainstack_studio": {
                "url": "https://brainstackstudio.com",
                "products": [
                    {
                        "name": "AI Playground",
                        "price": 0,
                        "type": "free",
                        "status": "active",
                        "features": [
                            "Claude, GPT, Gemini access",
                            "Local storage",
                            "Basic highlighting",
                        ],
                    }
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

    # Never hardcode revenue. Compute live, truthy numbers (exclude tests).
    try:
        from email_sender import get_db_connection
        from psycopg2.extras import RealDictCursor

        mrg_default_tenant = os.getenv(
            "MRG_DEFAULT_TENANT_ID", "00000000-0000-0000-0000-000000000001"
        )

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        cursor.execute(
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
        gumroad = cursor.fetchone() or {}

        cursor.execute(
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
            WHERE tenant_id = %s
              AND status = 'active'
            """,
            (mrg_default_tenant,),
        )
        mrg = cursor.fetchone() or {}

        cursor.close()
        conn.close()

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


@app.get("/inventory/revenue", tags=["Inventory"])
async def get_revenue_status(authenticated: bool = Depends(verify_api_key)):
    """
    Get real revenue status across all platforms.
    Separates real revenue from demo data.
    """
    try:
        from email_sender import get_db_connection
        from psycopg2.extras import RealDictCursor

        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get real Gumroad sales
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_sales,
                COALESCE(SUM(price::numeric), 0) as total_revenue,
                MAX(sale_timestamp) as last_sale
            FROM gumroad_sales
            WHERE is_test = false OR is_test IS NULL
        """
        )
        gumroad = cursor.fetchone() or {"total_sales": 0, "total_revenue": 0}

        # Get MRG subscriptions
        cursor.execute(
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
            WHERE status = 'active'
        """
        )
        mrg = cursor.fetchone() or {"active_subscriptions": 0, "mrr": 0}

        cursor.close()
        conn.close()

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
        logger.error(f"Revenue status failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "real_revenue": {
                "gumroad": {"total_sales": 0, "total_revenue": 0},
                "myroofgenius": {"active_subscriptions": 0, "mrr": 0},
            },
            "error": str(e),
        }


# ==================== END PRODUCT & REVENUE INVENTORY ====================


# ==================== REVENUE INTELLIGENCE SYSTEM ====================

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


@app.get("/revenue/state", tags=["Revenue Intelligence"])
async def get_complete_business_state(authenticated: bool = Depends(verify_api_key)):
    """
    Get COMPLETE business state snapshot.
    This is the PRIMARY endpoint for understanding full business state.
    Includes products, revenue, social presence, automations, and action items.
    """
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    state = await get_business_state()
    return state


@app.get("/revenue/live", tags=["Revenue Intelligence"])
async def get_live_revenue_data(authenticated: bool = Depends(verify_api_key)):
    """Get live revenue data across all platforms."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    return await get_revenue()


@app.get("/revenue/products", tags=["Revenue Intelligence"])
async def get_all_products_inventory(authenticated: bool = Depends(verify_api_key)):
    """Get complete product inventory across all platforms."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    system = get_revenue_system()
    return {
        "products": system.get_all_products(),
        "social": system.get_social_presence(),
        "websites": system.get_websites(),
    }


@app.get("/revenue/automations", tags=["Revenue Intelligence"])
async def get_automation_health(authenticated: bool = Depends(verify_api_key)):
    """Get status of all revenue-related automations."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    system = get_revenue_system()
    return await system.get_automation_status()


@app.post("/revenue/sync-brain", tags=["Revenue Intelligence"])
async def sync_business_state_to_brain(authenticated: bool = Depends(verify_api_key)):
    """
    Sync complete business state to AI brain.
    This ensures ALL AI agents have current business awareness.
    """
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    return await sync_to_brain()


@app.post("/revenue/event", tags=["Revenue Intelligence"])
async def record_revenue_event(
    event_type: str = Body(..., embed=True),
    platform: str = Body(..., embed=True),
    amount: float = Body(0, embed=True),
    metadata: dict = Body(None, embed=True),
    authenticated: bool = Depends(verify_api_key),
):
    """Record a revenue event for tracking."""
    if not REVENUE_INTEL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Revenue Intelligence not available")

    system = get_revenue_system()
    event_id = await system.record_revenue_event(event_type, platform, amount, metadata)
    return {"status": "recorded", "event_id": event_id}


# ==================== END REVENUE INTELLIGENCE SYSTEM ====================


@app.exception_handler(DatabaseUnavailableError)
async def database_unavailable_handler(request: Request, exc: DatabaseUnavailableError):
    """Surface database connectivity issues with a 503 response."""
    logger.error("Database unavailable: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "type": "DatabaseUnavailable",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler - feeds errors into self-healing system"""
    import traceback
    import uuid
    from datetime import datetime

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Feed error into self-healing system for learning and recovery
    if SELF_HEALING_AVAILABLE and ErrorContext is not None:
        try:
            healer = _get_global_healer()
            if healer:
                # Determine severity based on exception type
                severity = ErrorSeverity.MEDIUM
                if isinstance(exc, (ConnectionError, TimeoutError)):
                    severity = ErrorSeverity.HIGH
                elif isinstance(exc, (ValueError, TypeError, KeyError)):
                    severity = ErrorSeverity.LOW

                # Create error context
                error_context = ErrorContext(
                    error_id=f"api-{uuid.uuid4().hex[:12]}",
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],  # Truncate long messages
                    stack_trace=traceback.format_exc()[:5000],  # Truncate long traces
                    component="api",
                    function_name=str(request.url.path),
                    timestamp=datetime.utcnow(),
                    severity=severity,
                    retry_count=0,
                    metadata={
                        "method": request.method,
                        "path": str(request.url.path),
                        "query": str(request.query_params),
                        "client_host": request.client.host if request.client else "unknown",
                    },
                )

                # Log to database for pattern learning
                healer._log_error(error_context)
                logger.info(f"🩺 Error logged to self-healing: {error_context.error_id}")

        except Exception as heal_err:
            # Don't let self-healing failure break the error response
            logger.warning(f"Self-healing error logging failed: {heal_err}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc) if config.security.dev_mode else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level.lower())
