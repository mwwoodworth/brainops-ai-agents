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

from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
)
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
from api.agents import router as agents_router  # WAVE 2B: extracted agent CRUD/execution/dispatch
from api.ai_operations import (
    router as ai_operations_router,
)  # WAVE 2C: extracted AI/LLM/consciousness
from api.operational import (
    router as operational_router,
)  # WAVE 2C: self-heal/AUREA/executions/training
from api.platform import router as platform_router  # WAVE 2C/2D: root/email/knowledge/erp/usage
from api.content_revenue import (
    router as content_revenue_router,
)  # WAVE 2C/2D: content/inventory/revenue/observability
from api.health import router as health_router  # WAVE 2A: extracted health/readiness/status
from api.scheduler import router as scheduler_router  # WAVE 2B: extracted scheduler/scheduling
from api.system_health import router as system_health_router  # WAVE 2A: extracted observability
from services.db_health import attempt_db_pool_init_once, pool_roundtrip_healthy
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
from api.full_power_crud import (
    router as full_power_crud_router,
)  # Full CRUD + lifecycle control plane
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
            db_ready = await attempt_db_pool_init_once(app.state, "deferred_init", timeout=5.0)
            if using_fallback():
                logger.warning(
                    "⚠️ Running with in-memory fallback datastore (database unreachable)."
                )
            elif db_ready:
                logger.info("✅ Database pool initialized")
            else:
                logger.warning(
                    "⚠️ Database pool initialized but health check failed during deferred_init"
                )

            # Schema verification (read-only) - no DDL, agent_worker has no DDL perms
            pool = get_pool()

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
            app.state.db_init_error = str(e)
            logger.error(f"❌ Deferred database init failed: {e}")

            async def retry_db_init():
                attempts = int(os.getenv("DB_INIT_RETRY_ATTEMPTS", "18"))
                interval_s = float(os.getenv("DB_INIT_RETRY_INTERVAL_S", "10"))
                for attempt in range(1, attempts + 1):
                    await asyncio.sleep(interval_s)
                    if await attempt_db_pool_init_once(
                        app.state, f"deferred_retry_{attempt}/{attempts}", timeout=5.0
                    ):
                        logger.info(
                            "✅ Deferred database init retry succeeded (%d/%d)", attempt, attempts
                        )
                        return
                    logger.warning("Deferred database init retry failed (%d/%d)", attempt, attempts)
                logger.error("❌ Deferred database init retries exhausted (%d attempts)", attempts)

            create_safe_task(retry_db_init(), "retry_db_init")

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
    app.state.db_init_error = None
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
        "X-Correlation-ID",
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


# ---------------------------------------------------------------------------
# Correlation ID Middleware — cross-service traceability
# ---------------------------------------------------------------------------
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Attach correlation ID to every request for cross-service tracing."""
    correlation_id = (
        request.headers.get("x-correlation-id")
        or request.headers.get("x-request-id")
        or str(uuid.uuid4())
    )
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Request-ID"] = correlation_id
    return response


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

app.include_router(
    health_router
)  # WAVE 2A: /health, /healthz, /ready, /alive, /capabilities, /diagnostics, /system/alerts
app.include_router(
    system_health_router
)  # WAVE 2A: /system/awareness, /awareness/*, /truth/*, /debug/*, etc.
app.include_router(memory_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(sync_router, dependencies=SECURED_DEPENDENCIES)  # Memory sync/migration
app.include_router(brain_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(
    agents_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2B: /agents/*, /execute, /api/v1/agents/*, /api/v1/aurea/execute-event
app.include_router(
    scheduler_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2B: /scheduler/*, /agents/schedule, /email/scheduler-stats
app.include_router(
    ai_operations_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2C: /langgraph/*, /ai/*, /consciousness/*, /meta-intelligence/*, /workflow-*/status
app.include_router(
    operational_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2C: /self-heal/*, /aurea/status, /aurea/command/*, /executions, /training/*
app.include_router(
    platform_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2C/2D: /, /email/*, /api/v1/knowledge/*, /api/v1/erp/analyze, /systems/usage
app.include_router(
    content_revenue_router, dependencies=SECURED_DEPENDENCIES
)  # WAVE 2C/2D: /content/*, /inventory/*, /revenue/*, /logs/*, /observability/*, /debug/*, /system/unified-status
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

# LangGraph Product Agent - optional import with graceful degradation
try:
    from src.graph.product_agent import app as product_agent_graph

    PRODUCT_AGENT_AVAILABLE = True
    logger.info("Product Agent loaded - LangGraph product generation enabled")
except (ImportError, Exception) as e:
    PRODUCT_AGENT_AVAILABLE = False
    product_agent_graph = None
    logger.warning(f"Product Agent not available: {e}")


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
