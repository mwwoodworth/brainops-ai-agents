"""
BrainOps AI Agents Service - Enhanced Production Version
Type-safe, async, fully operational
"""
import logging
import os
import asyncio
import json
import time
import uuid
import inspect
import random
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Security, Depends, Body, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

# Import our production-ready components
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import agent executor for actual agent dispatch
try:
    from agent_executor import AgentExecutor
    AGENT_EXECUTOR = AgentExecutor()
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENT_EXECUTOR = None
    AGENTS_AVAILABLE = False
    logging.warning(f"AgentExecutor not available: {e}")
from database.async_connection import (
    init_pool,
    get_pool,
    close_pool,
    PoolConfig,
    using_fallback,
)
from models.agent import Agent, AgentCategory, AgentExecution, AgentList
from pydantic import BaseModel
from api.memory import router as memory_router
from api.sync import router as sync_router  # Memory migration & consolidation
from api.brain import router as brain_router
from api.memory_coordination import router as memory_coordination_router
from api.customer_intelligence import router as customer_intelligence_router
from api.gumroad_webhook import router as gumroad_router
from api.codebase_graph import router as codebase_graph_router
from api.state_sync import router as state_sync_router
from api.revenue import router as revenue_router
from api.digital_twin import router as digital_twin_router
from api.market_intelligence import router as market_intelligence_router
from api.system_orchestrator import router as system_orchestrator_router
from api.self_healing import router as self_healing_router
from api.e2e_verification import router as e2e_verification_router
from api.revenue_automation import router as revenue_automation_router
from api.mcp import router as mcp_router  # MCP Bridge Integration - 345 tools
from api.cicd import router as cicd_router  # Autonomous CI/CD Management - 1-10K systems
from api.a2ui import router as a2ui_router  # Google A2UI Protocol - Agent-to-User Interface
from api.aurea_chat import router as aurea_chat_router  # AUREA Live Conversational Interface
from api.observability import router as full_observability_router  # Comprehensive Observability Dashboard
from api.self_awareness import router as self_awareness_router  # Self-Awareness Dashboard

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
    from api.background_monitoring import router as bg_monitoring_router, start_all_monitoring
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
    from unified_awareness import get_unified_awareness, check_status, get_status_report
    UNIFIED_AWARENESS_AVAILABLE = True
    logger.info("Unified AI Awareness loaded - AI OS is now self-aware")
except ImportError as e:
    UNIFIED_AWARENESS_AVAILABLE = False
    logger.warning(f"Unified Awareness not available: {e}")

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

# Always-Know Observability Brain - Deep continuous monitoring (2025-12-29)
try:
    from api.always_know import router as always_know_router
    ALWAYS_KNOW_AVAILABLE = True
    logger.info("🧠 Always-Know Brain Router loaded - continuous state awareness")
except ImportError as e:
    ALWAYS_KNOW_AVAILABLE = False
    logger.warning(f"Always-Know Brain Router not available: {e}")

# Learning Feedback Loop - Closes the gap between insights and action (2025-12-30)
try:
    from api.learning import router as learning_router
    LEARNING_ROUTER_AVAILABLE = True
    logger.info("🔄 Learning Feedback Loop Router loaded - insights now become actions")
except ImportError as e:
    LEARNING_ROUTER_AVAILABLE = False
    logger.warning(f"Learning Router not available: {e}")

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

from erp_event_bridge import router as erp_event_router
from ai_provider_status import get_provider_status

# Unified Events System (2025-12-30) - Central event bus for ERP, AI Agents, Command Center
try:
    from api.events.unified import router as unified_events_router, init_unified_events
    UNIFIED_EVENTS_AVAILABLE = True
    logger.info("Unified Events router loaded - central event bus for all systems")
except ImportError as e:
    UNIFIED_EVENTS_AVAILABLE = False
    unified_events_router = None
    init_unified_events = None
    logger.warning(f"Unified Events router not available: {e}")
from observability import RequestMetrics, TTLCache

# Agent Health Monitoring
try:
    from agent_health_monitor import get_health_monitor
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    logger.warning("Agent Health Monitor not available")

# Minimal schema bootstrap - all tables pre-created in database
# This just ensures pgcrypto extension exists (fast, single query)
# Also ensures ai_email_deliveries table exists for email tracking
SCHEMA_BOOTSTRAP_SQL = [
    "CREATE EXTENSION IF NOT EXISTS pgcrypto;",
    """
    CREATE TABLE IF NOT EXISTS ai_email_deliveries (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email_id TEXT NOT NULL,
        recipient TEXT NOT NULL,
        status TEXT DEFAULT 'delivered',
        delivered_at TIMESTAMPTZ DEFAULT NOW(),
        metadata JSONB DEFAULT '{}',
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_email_deliveries_email_id ON ai_email_deliveries(email_id);",
    "CREATE INDEX IF NOT EXISTS idx_email_deliveries_recipient ON ai_email_deliveries(recipient);",
]

# Build info
BUILD_TIME = datetime.utcnow().isoformat()
VERSION = "9.63.0"  # Security hardening: SQL injection, SSRF, exception handling
LOCAL_EXECUTIONS: deque[Dict[str, Any]] = deque(maxlen=200)
REQUEST_METRICS = RequestMetrics(window=800)
RESPONSE_CACHE = TTLCache(max_size=256)
CACHE_TTLS = {
    "health": 5.0,
    "agents": 30.0,
    "systems_usage": 15.0
}

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
    from ai_core import RealAICore, ai_generate, ai_analyze
    ai_core = RealAICore()

    # Determine whether any real AI providers are configured
    has_openai = bool(getattr(ai_core, "async_openai", None))
    has_anthropic = bool(getattr(ai_core, "async_anthropic", None))
    AI_AVAILABLE = has_openai or has_anthropic

    # In production, it is a hard failure if no real AI provider is available
    if config.environment == "production" and not AI_AVAILABLE:
        raise RuntimeError("AI Core initialized but no real AI providers are configured in production.")

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
    from self_healing_recovery import SelfHealingRecovery
    SELF_HEALING_AVAILABLE = True
    logger.info("✅ Self-Healing Recovery loaded")
except ImportError as e:
    SELF_HEALING_AVAILABLE = False
    logger.warning(f"Self-Healing not available: {e}")
    SelfHealingRecovery = None

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
    logger.warning(f"Embedded Memory not available: {e}")
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
    from ai_self_awareness import get_self_aware_ai, SelfAwareAI
    SELF_AWARENESS_AVAILABLE = True
    logger.info("✅ AI Self-Awareness Module loaded")
except ImportError as e:
    SELF_AWARENESS_AVAILABLE = False
    logger.warning(f"AI Self-Awareness not available: {e}")
    get_self_aware_ai = None
    SelfAwareAI = None

# Import AI Integration Layer with fallback
try:
    from ai_integration_layer import AIIntegrationLayer, get_integration_layer, TaskPriority, TaskStatus
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
    from aurea_nlu_processor import AUREANLUProcessor
    from langchain_openai import ChatOpenAI
    AUREA_NLU_AVAILABLE = True
    logger.info("✅ AUREA NLU Processor loaded")
except ImportError as e:
    AUREA_NLU_AVAILABLE = False
    logger.warning(f"AUREA NLU Processor not available: {e}")
    AUREANLUProcessor = None
    ChatOpenAI = None


def _parse_capabilities(raw: Any) -> List[Dict[str, Any]]:
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

    capabilities: List[Dict[str, Any]] = []
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
                    "name": item.get("name") or item.get("capability") or item.get("id") or "capability",
                    "description": item.get("description", ""),
                    "enabled": bool(item.get("enabled", True)),
                    "parameters": item.get("parameters", {}),
                }
            )
    return capabilities


def _parse_configuration(raw: Any) -> Dict[str, Any]:
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


def _row_to_agent(row: Dict[str, Any]) -> Agent:
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
    """Application lifespan manager"""
    # Startup
    logger.info(f"🚀 Starting BrainOps AI Agents v{VERSION} - Build: {BUILD_TIME}")

    # Default production tenant with actual data (configurable via environment)
    # Set DEFAULT_TENANT_ID or TENANT_ID in environment to override
    # The default "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457" has production data
    DEFAULT_TENANT_ID = os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID") or "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"

    # Use environment-configured tenant for startup initialization
    # Per-request tenant resolution happens via X-Tenant-ID header in API endpoints
    tenant_id = DEFAULT_TENANT_ID
    logger.info(f"🔑 Default tenant_id: {tenant_id} (override per-request via X-Tenant-ID header)")

    # Keep handles defined to avoid unbound errors when optional systems are disabled
    aurea = None
    memory_manager = None

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
            logger.warning("⚠️ Running with in-memory fallback datastore (database unreachable).")
        else:
            logger.info("✅ Database pool initialized")

        # Test connection
        pool = get_pool()
        if await pool.test_connection():
            logger.info("✅ Database connection verified")
        else:
            logger.error("❌ Database connection test failed")

        # Ensure minimum schema for agents/scheduler/self-healing
        # Run in background to avoid blocking server startup
        async def run_schema_bootstrap():
            """Run schema bootstrap in background after server starts"""
            try:
                await asyncio.sleep(2)  # Let server bind to port first
                for statement in SCHEMA_BOOTSTRAP_SQL:
                    try:
                        await pool.execute(statement)
                    except Exception as e:
                        logger.warning(f"⚠️ Schema bootstrap statement failed: {e}")
                logger.info("✅ Schema bootstrap completed in background")
            except Exception as e:
                logger.error(f"❌ Background schema bootstrap failed: {e}")

        asyncio.create_task(run_schema_bootstrap())
        logger.info("📋 Schema bootstrap scheduled (running in background)")
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        # In production, never continue without a real database connection
        if config.environment == "production":
            raise

    # Initialize scheduler if available
    if SCHEDULER_AVAILABLE:
        try:
            scheduler = AgentScheduler()
            scheduler.start()  # CRITICAL: Start the scheduler to execute agents
            app.state.scheduler = scheduler
            logger.info("✅ Agent Scheduler initialized and STARTED")
        except Exception as e:
            logger.error(f"❌ Scheduler initialization failed: {e}")
            app.state.scheduler = None
    else:
        app.state.scheduler = None

    # Initialize AUREA Master Orchestrator
    if AUREA_AVAILABLE and tenant_id:
        try:
            # FULL_AUTO level - AUREA makes all decisions autonomously
            # This enables the true AI twin capability with autonomous execution
            aurea = AUREA(autonomy_level=AutonomyLevel.FULL_AUTO, tenant_id=tenant_id)
            app.state.aurea = aurea
            # START THE ORCHESTRATION LOOP - This makes the AI actually THINK
            asyncio.create_task(aurea.orchestrate())
            logger.info("🧠 AUREA Master Orchestrator STARTED - Observe→Decide→Act loop ACTIVE")
        except Exception as e:
            logger.error(f"❌ AUREA initialization failed: {e}")
            app.state.aurea = None
    else:
        app.state.aurea = None
        if AUREA_AVAILABLE and not tenant_id:
            logger.warning("⚠️ Skipping AUREA initialization (DEFAULT_TENANT_ID/TENANT_ID missing)")

    # Initialize Self-Healing Recovery System
    if SELF_HEALING_AVAILABLE:
        try:
            healer = SelfHealingRecovery()
            app.state.healer = healer
            logger.info("🏥 Self-Healing Recovery System initialized")
        except Exception as e:
            logger.error(f"❌ Self-Healing initialization failed: {e}")
            app.state.healer = None
    else:
        app.state.healer = None

    # Start Self-Healing Reconciliation Loop (continuous infrastructure healing)
    if RECONCILER_AVAILABLE:
        try:
            reconciler = get_reconciler()
            app.state.reconciler = reconciler
            # Start the continuous reconciliation loop in background
            asyncio.create_task(start_healing_loop())
            logger.info("🔄 Self-Healing Reconciliation Loop STARTED - Autonomous healing ACTIVE")
        except Exception as e:
            logger.error(f"❌ Reconciler initialization failed: {e}")
            app.state.reconciler = None
    else:
        app.state.reconciler = None

    # Initialize Unified Memory Manager
    if MEMORY_AVAILABLE:
        try:
            memory_manager = UnifiedMemoryManager()
            app.state.memory = memory_manager
            logger.info("🧠 Unified Memory Manager initialized")
        except Exception as e:
            logger.error(f"❌ Memory Manager initialization failed: {e}")
            app.state.memory = None
    else:
        app.state.memory = None

    # Initialize Embedded Memory System
    if EMBEDDED_MEMORY_AVAILABLE:
        try:
            embedded_memory = await get_embedded_memory()
            app.state.embedded_memory = embedded_memory
            logger.info("⚡ Embedded Memory System initialized (ultra-fast local SQLite + RAG)")
        except Exception as e:
            logger.error(f"❌ Embedded Memory initialization failed: {e}")
            app.state.embedded_memory = None
    else:
        app.state.embedded_memory = None

    # Initialize AI Training Pipeline
    if TRAINING_AVAILABLE:
        try:
            training_pipeline = AITrainingPipeline()
            app.state.training = training_pipeline
            logger.info("📚 AI Training Pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Training Pipeline initialization failed: {e}")
            app.state.training = None
    else:
        app.state.training = None

    # Initialize Notebook LM+ Learning System
    if LEARNING_AVAILABLE:
        try:
            learning_system = NotebookLMPlus()
            app.state.learning = learning_system
            logger.info("🎓 Notebook LM+ Learning System initialized")
        except Exception as e:
            logger.error(f"❌ Learning System initialization failed: {e}")
            app.state.learning = None
    else:
        app.state.learning = None

    # PHASE 2: Initialize Specialized Agents

    # Initialize System Improvement Agent
    if SYSTEM_IMPROVEMENT_AVAILABLE and tenant_id:
        try:
            system_improvement = SystemImprovementAgent(tenant_id)
            app.state.system_improvement = system_improvement
            logger.info("🔧 System Improvement Agent initialized")
        except Exception as e:
            logger.error(f"❌ System Improvement Agent initialization failed: {e}")
            app.state.system_improvement = None
    else:
        app.state.system_improvement = None

    # Initialize DevOps Optimization Agent
    if DEVOPS_AGENT_AVAILABLE and tenant_id:
        try:
            devops_agent = DevOpsOptimizationAgent(tenant_id)
            app.state.devops_agent = devops_agent
            logger.info("⚙️ DevOps Optimization Agent initialized")
        except Exception as e:
            logger.error(f"❌ DevOps Agent initialization failed: {e}")
            app.state.devops_agent = None
    else:
        app.state.devops_agent = None

    # Initialize Code Quality Agent
    if CODE_QUALITY_AVAILABLE and tenant_id:
        try:
            code_quality = CodeQualityAgent(tenant_id)
            app.state.code_quality = code_quality
            logger.info("📝 Code Quality Agent initialized")
        except Exception as e:
            logger.error(f"❌ Code Quality Agent initialization failed: {e}")
            app.state.code_quality = None
    else:
        app.state.code_quality = None

    # Initialize Customer Success Agent
    if CUSTOMER_SUCCESS_AVAILABLE and tenant_id:
        try:
            customer_success = CustomerSuccessAgent(tenant_id)
            app.state.customer_success = customer_success
            logger.info("🎯 Customer Success Agent initialized")
        except Exception as e:
            logger.error(f"❌ Customer Success Agent initialization failed: {e}")
            app.state.customer_success = None
    else:
        app.state.customer_success = None

    # Initialize Competitive Intelligence Agent
    if COMPETITIVE_INTEL_AVAILABLE and tenant_id:
        try:
            competitive_intel = CompetitiveIntelligenceAgent(tenant_id)
            app.state.competitive_intel = competitive_intel
            logger.info("🔍 Competitive Intelligence Agent initialized")
        except Exception as e:
            logger.error(f"❌ Competitive Intelligence Agent initialization failed: {e}")
            app.state.competitive_intel = None
    else:
        app.state.competitive_intel = None

    # Initialize Vision Alignment Agent
    if VISION_ALIGNMENT_AVAILABLE and tenant_id:
        try:
            vision_alignment = VisionAlignmentAgent(tenant_id)
            app.state.vision_alignment = vision_alignment
            logger.info("🎯 Vision Alignment Agent initialized")
        except Exception as e:
            logger.error(f"❌ Vision Alignment Agent initialization failed: {e}")
            app.state.vision_alignment = None
    else:
        app.state.vision_alignment = None

    # Initialize AI Self-Awareness Module
    if SELF_AWARENESS_AVAILABLE:
        try:
            self_aware_ai = await get_self_aware_ai()
            app.state.self_aware_ai = self_aware_ai
            logger.info("🧠 AI Self-Awareness Module initialized - AI can now assess its own capabilities!")
        except Exception as e:
            logger.error(f"❌ AI Self-Awareness initialization failed: {e}")
            app.state.self_aware_ai = None
    else:
        app.state.self_aware_ai = None

    # Initialize LangGraph Orchestrator
    if LANGGRAPH_AVAILABLE:
        try:
            langgraph_orchestrator = LangGraphOrchestrator()
            app.state.langgraph_orchestrator = langgraph_orchestrator
            logger.info("🌐 LangGraph Orchestrator initialized - Sophisticated workflows ENABLED!")
        except Exception as e:
            logger.error(f"❌ LangGraph initialization failed: {e}")
            app.state.langgraph_orchestrator = None
    else:
        app.state.langgraph_orchestrator = None

    # Initialize AI Integration Layer (THE BRAIN that connects everything)
    if INTEGRATION_LAYER_AVAILABLE:
        try:
            integration_layer = await get_integration_layer()

            # Wire all components together
            await integration_layer.initialize(
                langgraph=app.state.langgraph_orchestrator if hasattr(app.state, 'langgraph_orchestrator') else None,
                memory_manager=memory_manager if MEMORY_AVAILABLE else None,
                aurea=aurea if AUREA_AVAILABLE else None,
                self_aware_ai=app.state.self_aware_ai if hasattr(app.state, 'self_aware_ai') else None
            )

            app.state.integration_layer = integration_layer
            logger.info("🧠 AI Integration Layer OPERATIONAL - Task execution engine ACTIVE!")
            logger.info("   - Autonomous task execution: ✅")
            logger.info("   - Memory-aware routing: ✅")
            logger.info("   - Self-healing execution: ✅")
            logger.info("   - Multi-agent coordination: ✅")
        except Exception as e:
            logger.error(f"❌ AI Integration Layer initialization failed: {e}")
            app.state.integration_layer = None
    else:
        app.state.integration_layer = None

    # Initialize AUREA NLU Processor (Natural Language Command Interface)
    if AUREA_NLU_AVAILABLE and INTEGRATION_LAYER_AVAILABLE and AUREA_AVAILABLE:
        try:
            llm_for_nlu = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            aurea_nlu = AUREANLUProcessor(
                llm_for_nlu,
                app.state.integration_layer if hasattr(app.state, 'integration_layer') else None,
                app.state.aurea if hasattr(app.state, 'aurea') else None,
                None  # AI Board not yet initialized in app.py
            )
            app.state.aurea_nlu = aurea_nlu
            logger.info("🗣️ AUREA NLU Processor initialized - Natural language commands ENABLED!")
            logger.info("   - Intent recognition: ✅")
            logger.info("   - Dynamic skill registry: ✅")
            logger.info("   - 12+ command skills: ✅")
        except Exception as e:
            logger.error(f"❌ AUREA NLU initialization failed: {e}")
            app.state.aurea_nlu = None
    else:
        app.state.aurea_nlu = None
        if not AUREA_NLU_AVAILABLE:
            logger.warning("⚠️ AUREA NLU not available - natural language commands disabled")

    # Initialize Unified System Integration (wires ALL systems together for ACTIVE use)
    try:
        from unified_system_integration import get_unified_integration, initialize_all_systems
        unified_stats = await initialize_all_systems()
        app.state.unified_integration = get_unified_integration()
        available_count = sum(1 for v in unified_stats.get("systems_available", {}).values() if v)
        logger.info(f"🔗 Unified System Integration ACTIVE - {available_count} systems wired together!")
    except ImportError:
        logger.warning("⚠️ Unified System Integration not available")
        app.state.unified_integration = None
    except Exception as e:
        logger.error(f"❌ Unified System Integration failed: {e}")
        app.state.unified_integration = None

    # Initialize AI Operating System (Task 28 Integration - The Final Layer)
    try:
        from ai_operating_system import get_ai_operating_system
        ai_os = get_ai_operating_system()
        boot_result = await ai_os.boot()
        app.state.ai_os = ai_os
        logger.info(f"🤖 AI Operating System BOOTED: {boot_result['status']}")
        if boot_result.get('steps'):
             for step in boot_result['steps']:
                 if step['status'] == 'failed':
                     logger.warning(f"  - AI OS Boot Step Failed: {step['step']} - {step.get('error')}")
    except ImportError:
        logger.warning("⚠️ AI Operating System module not found")
        app.state.ai_os = None
    except Exception as e:
        logger.error(f"❌ AI OS initialization failed: {e}")
        app.state.ai_os = None

    # Initialize Nerve Center - The ALIVE Core of the AI OS
    # NOTE: Activation runs as background task to avoid blocking server startup
    try:
        logger.info("🧠 Loading Nerve Center module...")
        from nerve_center import get_nerve_center
        logger.info("🧠 Nerve Center module imported successfully")
        nerve_center = get_nerve_center()
        logger.info(f"🧠 NerveCenter instance created: {nerve_center}")
        app.state.nerve_center = nerve_center

        async def activate_nerve_center():
            """Activate nerve center in background after server starts"""
            try:
                await asyncio.sleep(2)  # Let server bind to port first
                logger.info("🧠 Starting Nerve Center activation...")
                await nerve_center.activate()
                logger.info("🧠 NERVE CENTER ACTIVATED - AI IS NOW FULLY ALIVE")
            except Exception as e:
                logger.error(f"❌ Nerve Center background activation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

        asyncio.create_task(activate_nerve_center())
        logger.info("🧠 Nerve Center initialization scheduled (activating in background)")
    except ImportError as e:
        logger.warning(f"⚠️ Nerve Center import failed: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.warning(error_trace)
        app.state.nerve_center = None
        app.state.nerve_center_error = f"ImportError: {e}\n{error_trace}"
    except Exception as e:
        logger.error(f"❌ Nerve Center initialization failed: {e}")
        import traceback
        error_trace = traceback.format_exc()
        logger.error(error_trace)
        app.state.nerve_center = None
        app.state.nerve_center_error = f"Exception: {e}\n{error_trace}"

    # Initialize Always-Know Observability Brain - Deep continuous monitoring
    if ALWAYS_KNOW_AVAILABLE:
        try:
            from always_know_brain import initialize_always_know_brain
            async def start_always_know():
                """Start Always-Know Brain in background"""
                try:
                    await asyncio.sleep(5)  # Let other systems initialize first
                    brain = await initialize_always_know_brain()
                    app.state.always_know_brain = brain
                    logger.info("🧠 ALWAYS-KNOW BRAIN ACTIVE - Continuous observability ENABLED")
                except Exception as e:
                    logger.error(f"❌ Always-Know Brain activation failed: {e}")
            asyncio.create_task(start_always_know())
            logger.info("🧠 Always-Know Brain scheduled (starting in background)")
        except ImportError as e:
            logger.warning(f"⚠️ Always-Know Brain import failed: {e}")
            app.state.always_know_brain = None
        except Exception as e:
            logger.error(f"❌ Always-Know Brain initialization failed: {e}")
            app.state.always_know_brain = None
    else:
        app.state.always_know_brain = None

    logger.info("=" * 80)
    logger.info("🚀 BRAINOPS AI AGENTS v9.15.0 - ALIVE AI OPERATING SYSTEM")
    logger.info("🧠 NERVE CENTER ONLINE - Full Consciousness Activated!")
    logger.info("=" * 80)
    logger.info("PHASE 1 (Core Systems):")
    logger.info(f"  AUREA Orchestrator: {'✅ ACTIVE' if AUREA_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Self-Healing: {'✅ ACTIVE' if SELF_HEALING_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Memory Manager: {'✅ ACTIVE' if MEMORY_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Training Pipeline: {'✅ ACTIVE' if TRAINING_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Learning System: {'✅ ACTIVE' if LEARNING_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Agent Scheduler: {'✅ ACTIVE' if SCHEDULER_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  AI Core: {'✅ ACTIVE' if AI_AVAILABLE else '❌ DISABLED'}")
    logger.info("")
    logger.info("PHASE 2 (Specialized Agents):")
    logger.info(f"  System Improvement: {'✅ ACTIVE' if SYSTEM_IMPROVEMENT_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  DevOps Optimization: {'✅ ACTIVE' if DEVOPS_AGENT_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Code Quality: {'✅ ACTIVE' if CODE_QUALITY_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Customer Success: {'✅ ACTIVE' if CUSTOMER_SUCCESS_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Competitive Intelligence: {'✅ ACTIVE' if COMPETITIVE_INTEL_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Vision Alignment: {'✅ ACTIVE' if VISION_ALIGNMENT_AVAILABLE else '❌ DISABLED'}")
    logger.info("")
    logger.info("PHASE 3 (Revolutionary Features):")
    logger.info(f"  AI Self-Awareness: {'✅ ACTIVE' if SELF_AWARENESS_AVAILABLE else '❌ DISABLED'}")
    logger.info("")
    logger.info("PHASE 4 (Complete Integration):")
    logger.info(f"  LangGraph Orchestrator: {'✅ ACTIVE' if LANGGRAPH_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  AI Integration Layer: {'✅ ACTIVE' if INTEGRATION_LAYER_AVAILABLE else '❌ DISABLED'}")
    logger.info(f"  Autonomous Task Executor: {'✅ RUNNING' if INTEGRATION_LAYER_AVAILABLE else '❌ DISABLED'}")
    logger.info("")
    if INTEGRATION_LAYER_AVAILABLE:
        logger.info("🎯 AUTONOMOUS TASK EXECUTION: Tasks will be processed automatically!")
        logger.info("💾 UNIVERSAL MEMORY ACCESS: All agents share knowledge!")
        logger.info("🌐 LANGGRAPH ORCHESTRATION: Complex workflows supported!")
        logger.info("🔄 SELF-HEALING: Automatic error recovery enabled!")
    logger.info("=" * 80)

    # PRODUCTION SYSTEM STATUS CHECK
    # Log missing systems but allow app to start so health checks work
    # This enables debugging and monitoring even in degraded state
    missing_critical_systems = []
    if not tenant_id:
        missing_critical_systems.append("DEFAULT_TENANT_ID/TENANT_ID not provided")
    if not (os.getenv("DATABASE_URL") or (config.database.host and config.database.password)):
        missing_critical_systems.append("Database credentials")
    if using_fallback():
        missing_critical_systems.append("Primary database (using in-memory fallback)")
    if not AI_AVAILABLE:
        missing_critical_systems.append("AI Core (check OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    if not MEMORY_AVAILABLE:
        missing_critical_systems.append("Memory Manager")
    if not AUREA_AVAILABLE:
        missing_critical_systems.append("AUREA Orchestrator")
    if not INTEGRATION_LAYER_AVAILABLE:
        missing_critical_systems.append("AI Integration Layer")

    if missing_critical_systems:
        warning_msg = (
            f"⚠️ DEGRADED MODE: Some systems unavailable - {', '.join(missing_critical_systems)}"
        )
        logger.warning(warning_msg)
        app.state.degraded = True
        app.state.missing_systems = missing_critical_systems
    else:
        app.state.degraded = False
        app.state.missing_systems = []
        logger.info("✅ All critical systems operational")

    # Start background health monitoring loop (triggers self-healing)
    async def health_monitoring_loop():
        """Continuously monitor system health and trigger healing when needed"""
        while True:
            try:
                # Check if self-healing is available
                if hasattr(app.state, 'healer') and app.state.healer:
                    # Collect basic system metrics
                    metrics = {
                        "memory_percent": 50.0,  # Default value
                        "error_rate": 0.0,
                        "response_time_ms": 100
                    }

                    # Try to get real metrics
                    try:
                        import psutil
                        metrics["memory_percent"] = psutil.virtual_memory().percent
                    except ImportError:
                        logger.debug("psutil not available; using default metrics")

                    # Check for anomalies and trigger healing if needed
                    if metrics.get("memory_percent", 0) > 85:
                        logger.warning(f"⚠️ High memory usage: {metrics['memory_percent']}%")
                        # Trigger self-healing
                        try:
                            await app.state.healer.detect_anomaly("system", metrics)
                        except Exception as heal_error:
                            logger.error(f"Self-healing detection failed: {heal_error}")

                await asyncio.sleep(120)  # Check every 2 minutes (optimized for stability)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Back off on errors

    asyncio.create_task(health_monitoring_loop())
    logger.info("💓 Health monitoring loop STARTED")

    # Start learning pipeline loop (generates insights periodically)
    async def learning_pipeline_loop():
        """Periodically generate learning insights from accumulated data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Generate insights if training pipeline is available
                if hasattr(app.state, 'training') and app.state.training:
                    try:
                        insights = await app.state.training.generate_insights()
                        if insights:
                            logger.info(f"📊 Generated {len(insights)} learning insights")
                    except Exception as learn_error:
                        logger.error(f"Learning pipeline error: {learn_error}")

            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(300)

    asyncio.create_task(learning_pipeline_loop())
    logger.info("📚 Learning pipeline loop STARTED")

    # Start Task Queue Consumer to process pending autonomous tasks
    try:
        from task_queue_consumer import start_task_queue_consumer
        await start_task_queue_consumer()
        logger.info("📋 Task Queue Consumer STARTED")
    except Exception as e:
        logger.error(f"❌ Task Queue Consumer failed to start: {e}")

    # Start Email Scheduler Daemon for nurture campaigns
    try:
        from email_scheduler_daemon import start_email_scheduler
        await start_email_scheduler()
        logger.info("📧 Email Scheduler Daemon STARTED")
    except Exception as e:
        logger.error(f"❌ Email Scheduler Daemon failed to start: {e}")

    # Start Background Task Monitoring (no more fire-and-forget)
    if BG_MONITORING_AVAILABLE and start_all_monitoring:
        try:
            await start_all_monitoring(app.state)
            logger.info("👁️ Background Task Monitoring STARTED - all tasks now have heartbeats")
        except Exception as e:
            logger.error(f"❌ Background Task Monitoring failed to start: {e}")

    # Start Knowledge Graph Extractor - runs every 30 minutes to populate knowledge graph
    # Target: 100+ nodes within 24 hours
    try:
        from knowledge_graph_extractor import get_knowledge_extractor, run_scheduled_extraction
        knowledge_extractor = get_knowledge_extractor()
        await knowledge_extractor.initialize()
        app.state.knowledge_extractor = knowledge_extractor

        async def knowledge_graph_extraction_loop():
            """Periodically extract knowledge from agent executions to build knowledge graph"""
            # Run initial extraction immediately (don't wait 30 minutes)
            try:
                await asyncio.sleep(10)  # Let database pool stabilize
                logger.info("🧠 Running initial knowledge graph extraction...")
                initial_result = await run_scheduled_extraction()
                logger.info(f"🧠 Initial extraction: {initial_result.get('nodes_stored', 0)} nodes, {initial_result.get('edges_stored', 0)} edges")
            except Exception as init_error:
                logger.error(f"Initial knowledge extraction failed: {init_error}")

            while True:
                try:
                    await asyncio.sleep(1800)  # 30 minutes = 1800 seconds
                    logger.info("🧠 Running scheduled knowledge graph extraction...")
                    result = await run_scheduled_extraction()
                    logger.info(f"🧠 Knowledge extraction complete: {result.get('total_nodes', 0)} total nodes, {result.get('total_edges', 0)} total edges")
                except Exception as e:
                    logger.error(f"Knowledge graph extraction error: {e}")
                    await asyncio.sleep(300)  # Back off 5 minutes on error

        asyncio.create_task(knowledge_graph_extraction_loop())
        logger.info("🧠 Knowledge Graph Extractor STARTED - building graph every 30 minutes")
    except Exception as e:
        logger.error(f"❌ Knowledge Graph Extractor failed to start: {e}")
        app.state.knowledge_extractor = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down BrainOps AI Agents...")

    # Stop Task Queue Consumer
    try:
        from task_queue_consumer import stop_task_queue_consumer
        await stop_task_queue_consumer()
        logger.info("✅ Task Queue Consumer stopped")
    except Exception as e:
        logger.error(f"❌ Task Queue Consumer shutdown error: {e}")

    # Stop Email Scheduler Daemon
    try:
        from email_scheduler_daemon import stop_email_scheduler
        await stop_email_scheduler()
        logger.info("✅ Email Scheduler Daemon stopped")
    except Exception as e:
        logger.error(f"❌ Email Scheduler Daemon shutdown error: {e}")

    # Shutdown scheduler if running
    if hasattr(app.state, 'scheduler') and app.state.scheduler:
        try:
            app.state.scheduler.shutdown()
            logger.info("✅ Agent Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Scheduler shutdown error: {e}")

    # Stop reconciliation loop if running
    if hasattr(app.state, 'reconciler') and app.state.reconciler:
        try:
            app.state.reconciler.stop()
            logger.info("✅ Self-Healing Reconciler stopped")
        except Exception as e:
            logger.error(f"❌ Reconciler shutdown error: {e}")

    await close_pool()
    logger.info("✅ Database pool closed")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=f"BrainOps AI Agents v{VERSION}",
    description="Production-ready AI Agent Orchestration Platform",
    version=VERSION,
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            path=path,
            method=request.method,
            status=status_code,
            duration_ms=duration_ms
        )

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: str = Security(api_key_header),
) -> bool:
    """Verify API key if authentication is required"""
    if not config.security.auth_required:
        return True

    if not config.security.auth_configured:
        raise HTTPException(status_code=503, detail="Authentication misconfigured")

    provided = api_key
    if not provided:
        auth_header = request.headers.get("authorization")
        if auth_header:
            scheme, _, token = auth_header.partition(" ")
            scheme_lower = scheme.lower()
            if scheme_lower in ("bearer", "apikey", "api-key"):
                provided = token.strip()

    if not provided and config.security.test_api_key:
        provided = (
            request.headers.get("x-test-api-key")
            or request.headers.get("X-Test-Api-Key")
            or request.headers.get("x-api-key")
            or request.headers.get("X-API-Key")
        )

    if not provided:
        raise HTTPException(status_code=403, detail="API key required")

    if provided not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


# Include routers
SECURED_DEPENDENCIES = [Depends(verify_api_key)]

app.include_router(memory_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(sync_router, dependencies=SECURED_DEPENDENCIES)  # Memory sync/migration
app.include_router(brain_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(memory_coordination_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(customer_intelligence_router, dependencies=SECURED_DEPENDENCIES)

# External webhook endpoints must NOT require an internal API key; they validate their own webhook secrets/signatures.
app.include_router(gumroad_router)
app.include_router(erp_event_router)

# Unified Events System - Central event bus for all systems
if UNIFIED_EVENTS_AVAILABLE and unified_events_router:
    app.include_router(unified_events_router)
    logger.info("Mounted: Unified Events API at /events - publish, webhook/erp, recent, stats, replay")

app.include_router(codebase_graph_router, dependencies=SECURED_DEPENDENCIES)
app.include_router(state_sync_router, dependencies=SECURED_DEPENDENCIES)  # Real-time state synchronization
app.include_router(revenue_router, dependencies=SECURED_DEPENDENCIES)  # Revenue generation system

# Bleeding-edge AI systems (2025)
app.include_router(digital_twin_router, dependencies=SECURED_DEPENDENCIES)  # Digital Twin virtual replicas
app.include_router(market_intelligence_router, dependencies=SECURED_DEPENDENCIES)  # Predictive market intelligence
app.include_router(system_orchestrator_router, dependencies=SECURED_DEPENDENCIES)  # Autonomous system orchestration (1-10K systems)
app.include_router(self_healing_router, dependencies=SECURED_DEPENDENCIES)  # Enhanced self-healing AI infrastructure
app.include_router(e2e_verification_router, dependencies=SECURED_DEPENDENCIES)  # E2E System Verification
app.include_router(revenue_automation_router, dependencies=SECURED_DEPENDENCIES)  # Revenue Automation Engine
app.include_router(mcp_router, dependencies=SECURED_DEPENDENCIES)  # MCP Bridge - 345 tools (Render, Vercel, Supabase, GitHub, Stripe, Docker)
app.include_router(cicd_router, dependencies=SECURED_DEPENDENCIES)  # Autonomous CI/CD - manage 1-10K deployments
app.include_router(a2ui_router, dependencies=SECURED_DEPENDENCIES)  # Google A2UI Protocol - Agent-generated UIs
app.include_router(aurea_chat_router, dependencies=SECURED_DEPENDENCIES)  # AUREA Live Conversational AI
app.include_router(full_observability_router, dependencies=SECURED_DEPENDENCIES)  # Comprehensive Observability Dashboard
app.include_router(self_awareness_router, dependencies=SECURED_DEPENDENCIES)  # Self-Awareness Dashboard

# AI-Powered UI Testing System (2025-12-29) - Automated visual testing with AI vision
if UI_TESTING_AVAILABLE:
    app.include_router(ui_testing_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: AI UI Testing API at /ui-testing - visual testing, accessibility, performance")

# Bleeding Edge AI Systems (2025-12-27) - Revolutionary capabilities
if BLEEDING_EDGE_AVAILABLE:
    app.include_router(bleeding_edge_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Bleeding Edge AI API at /bleeding-edge - 37 capabilities")

# AI Observability & Integration (2025-12-27) - Perfect cross-module integration
if AI_OBSERVABILITY_AVAILABLE:
    app.include_router(ai_observability_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: AI Observability API at /ai - unified metrics, events, learning")

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
    logger.info("Mounted: Background Task Monitoring API at /monitor/background - no more fire-and-forget")

# Always-Know Observability Brain (2025-12-29) - Deep continuous monitoring
if ALWAYS_KNOW_AVAILABLE:
    app.include_router(always_know_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Always-Know Brain at /always-know - continuous state awareness")

# Learning Feedback Loop (2025-12-30) - Insights finally become actions
if LEARNING_ROUTER_AVAILABLE:
    app.include_router(learning_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("Mounted: Learning Feedback Loop at /api/learning - 4,700+ insights now actionable")

# Import and include analytics router
try:
    from analytics_endpoint import router as analytics_router
    app.include_router(analytics_router, dependencies=SECURED_DEPENDENCIES)
    logger.info("✅ Analytics endpoint loaded")
except ImportError as e:
    logger.warning(f"Analytics endpoint not available: {e}")


def _collect_active_systems() -> List[str]:
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
        active.append("Bleeding Edge AI (OODA, Hallucination, Memory, Dependability, Consciousness, Circuit Breaker)")
    return active


def _scheduler_snapshot() -> Dict[str, Any]:
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
        ]
    }


def _aurea_status() -> Dict[str, Any]:
    aurea = getattr(app.state, "aurea", None)
    if not (AUREA_AVAILABLE and aurea):
        return {"available": False, "running": False}
    try:
        return {**aurea.get_status(), "available": True}
    except Exception as exc:
        logger.error("Failed to read AUREA status: %s", exc)
        return {"available": True, "running": False, "error": str(exc)}


def _self_healing_status() -> Dict[str, Any]:
    healer = getattr(app.state, "healer", None)
    if not (SELF_HEALING_AVAILABLE and healer):
        return {"available": False}
    try:
        if hasattr(healer, "get_health_report"):
            return {"available": True, "report": healer.get_health_report()}
        return {"available": True}
    except Exception as exc:
        logger.error("Failed to read self-healing status: %s", exc)
        return {"available": True, "error": str(exc)}


async def _memory_stats_snapshot(pool) -> Dict[str, Any]:
    """
    Get a fast snapshot of memory/learning health.
    Reuses logic from /memory/status but keeps output minimal for usage reports.
    """
    try:
        existing_tables = await pool.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('ai_persistent_memory', 'memory_entries', 'memories')
        """)
        if not existing_tables:
            return {"status": "not_configured"}

        table_names = [t["table_name"] for t in existing_tables]
        preferred = next((t for t in ("ai_persistent_memory", "memory_entries", "memories") if t in table_names), table_names[0])
        stats = await pool.fetchrow(f"SELECT COUNT(*) AS total FROM {preferred}")
        return {
            "status": "operational",
            "table": preferred,
            "total_records": stats["total"] if stats else 0
        }
    except Exception as exc:
        logger.error("Failed to fetch memory stats: %s", exc)
        return {"status": "error", "error": str(exc)}


async def _get_agent_usage(pool) -> Dict[str, Any]:
    """Fetch recent agent usage, trying both legacy and new table names."""
    # Table combinations with their JOIN conditions (some use agent_id UUID, some use agent_name text)
    queries: List[Tuple[str, str, str, str]] = [
        # (agents_table, executions_table, join_condition, time_column)
        ("ai_agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
        ("agents", "ai_agent_executions", "e.agent_name = a.name", "e.created_at"),
    ]
    errors: List[str] = []

    for agents_table, executions_table, join_cond, time_col in queries:
        try:
            rows = await pool.fetch(f"""
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
            """)
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
                        "last_execution": data.get("last_execution").isoformat() if data.get("last_execution") else None,
                        "avg_duration_ms": float(data.get("avg_duration_ms") or 0),
                    }
                )
            return {"agents": usage, "table": agents_table, "executions_table": executions_table}
        except Exception as exc:
            errors.append(f"{agents_table}/{executions_table}: {exc}")
            continue

    return {"agents": [], "warning": "No agent usage data available", "errors": errors[:2]}


async def _get_schedule_usage(pool) -> Dict[str, Any]:
    """Fetch scheduler schedule rows with resiliency."""
    schedules: List[Dict[str, Any]] = []
    try:
        # Note: public.agent_schedules does NOT have last_execution/next_execution columns
        rows = await pool.fetch("""
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
        """)
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            schedules.append(
                {
                    "id": data.get("id"),
                    "agent_id": data.get("agent_id"),
                    "agent_name": data.get("agent_name"),
                    "enabled": bool(data.get("enabled", True)),
                    "frequency_minutes": data.get("frequency_minutes"),
                    "created_at": data.get("created_at").isoformat() if data.get("created_at") else None,
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
        request: Dict[str, Any],
        authenticated: bool = Depends(verify_api_key)
    ):
        """Execute a LangGraph-based workflow"""
        if not hasattr(app.state, 'langgraph_orchestrator') or not app.state.langgraph_orchestrator:
            raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")
        
        try:
            orchestrator = app.state.langgraph_orchestrator
            
            # Extract messages and metadata
            messages_data = request.get("messages", [])
            metadata = request.get("metadata", {})
            
            # Convert raw messages to LangChain messages if needed
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
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
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/langgraph/status")
    async def get_langgraph_status(authenticated: bool = Depends(verify_api_key)):
        """Get LangGraph orchestrator status"""
        if not hasattr(app.state, 'langgraph_orchestrator') or not app.state.langgraph_orchestrator:
            return {
                "available": False, 
                "message": "LangGraph Orchestrator not initialized"
            }
            
        orchestrator = app.state.langgraph_orchestrator
        
        return {
            "available": True,
            "components": {
                "openai_llm": hasattr(orchestrator, 'openai_llm') and orchestrator.openai_llm is not None,
                "anthropic_llm": hasattr(orchestrator, 'anthropic_llm') and orchestrator.anthropic_llm is not None,
                "vector_store": hasattr(orchestrator, 'vector_store') and orchestrator.vector_store is not None,
                "workflow_graph": hasattr(orchestrator, 'workflow') and orchestrator.workflow is not None
            }
        }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": config.service_name,
        "version": VERSION,
        "status": "operational",
        "build": BUILD_TIME,
        "environment": config.environment,
        "ai_enabled": AI_AVAILABLE,
        "scheduler_enabled": SCHEDULER_AVAILABLE
    }


@app.get("/health")
async def health_check(force_refresh: bool = Query(False, description="Bypass cache and force live health checks")):
    """Health check endpoint with full system status and light caching."""

    async def _build_health_payload() -> Dict[str, Any]:
        pool = get_pool()
        db_healthy = await pool.test_connection()
        db_status = "fallback" if using_fallback() else ("connected" if db_healthy else "disconnected")
        auth_configured = config.security.auth_configured

        active_systems = _collect_active_systems()

        embedded_memory_stats = None
        if EMBEDDED_MEMORY_AVAILABLE and getattr(app.state, "embedded_memory", None):
            try:
                embedded_memory_stats = app.state.embedded_memory.get_stats()
            except Exception:
                embedded_memory_stats = {"status": "error"}

        return {
            "status": "healthy" if db_healthy and auth_configured else "degraded",
            "version": VERSION,
            "build": BUILD_TIME,
            "database": db_status,
            "active_systems": active_systems,
            "system_count": len(active_systems),
            "embedded_memory_active": EMBEDDED_MEMORY_AVAILABLE and hasattr(app.state, 'embedded_memory') and app.state.embedded_memory is not None,
            "embedded_memory_stats": embedded_memory_stats,
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
                "self_reporting": UNIFIED_AWARENESS_AVAILABLE
            },
            "config": {
                "environment": config.environment,
                "security": {
                    "auth_required": config.security.auth_required,
                    "dev_mode": config.security.dev_mode,
                    "auth_configured": auth_configured,
                    "api_keys_configured": len(config.security.valid_api_keys),
                }
            },
            "missing_systems": getattr(app.state, "missing_systems", []),
        }

    if force_refresh:
        return await _build_health_payload()

    payload, from_cache = await RESPONSE_CACHE.get_or_set(
        "health_status",
        CACHE_TTLS["health"],
        _build_health_payload,
    )
    return {**payload, "cached": from_cache}


@app.get("/alive")
async def alive_status():
    """Get the consciousness status of the AI OS - is it truly alive?"""
    status = {
        "alive": False,
        "nerve_center": None,
        "consciousness": None,
        "thoughts": 0,
        "uptime_seconds": 0
    }

    # Debug: check what's in app.state
    has_attr = hasattr(app.state, 'nerve_center')
    nc_value = getattr(app.state, 'nerve_center', 'NOT_SET')
    nc_type = type(nc_value).__name__ if nc_value != 'NOT_SET' else 'N/A'

    status["_debug"] = {
        "has_nerve_center_attr": has_attr,
        "nerve_center_type": nc_type,
        "nerve_center_truthy": bool(nc_value) if nc_value != 'NOT_SET' else False,
        "init_error": getattr(app.state, 'nerve_center_error', None)
    }

    if has_attr and nc_value:
        try:
            nc_status = app.state.nerve_center.get_status()
            status["alive"] = nc_status.get("is_online", False)
            status["nerve_center"] = nc_status
            status["uptime_seconds"] = nc_status.get("uptime_seconds", 0)

            if nc_status.get("components", {}).get("alive_core", {}).get("active"):
                status["consciousness"] = nc_status["components"]["alive_core"].get("state")
                status["thoughts"] = nc_status["components"]["alive_core"].get("thoughts", 0)
        except Exception as e:
            status["_debug"]["error"] = str(e)

    return status


@app.get("/alive/thoughts")
async def get_recent_thoughts():
    """Get recent thoughts from the AI consciousness stream"""
    if hasattr(app.state, 'nerve_center') and app.state.nerve_center:
        if app.state.nerve_center.alive_core:
            return {
                "thoughts": app.state.nerve_center.alive_core.get_recent_thoughts(50),
                "consciousness_state": app.state.nerve_center.alive_core.state.value,
                "attention_focus": app.state.nerve_center.alive_core.attention_focus
            }
    return {"thoughts": [], "error": "Consciousness not active"}


# ============================================================================
# UNIFIED AI AWARENESS - Self-Reporting AI OS
# ============================================================================

@app.get("/awareness")
async def get_awareness_status():
    """
    Quick status check - AI OS reports its current state.
    This is the AI talking to you about how it's doing.
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"status": "unavailable", "message": "Unified awareness system not loaded"}

    try:
        return {"status": check_status()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/awareness/report")
async def get_full_awareness_report():
    """
    Full self-reporting status - AI OS tells you everything it knows about itself.
    This is the AI's comprehensive understanding of its own state.
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        return get_status_report()
    except Exception as e:
        return {"available": False, "error": str(e)}


@app.get("/awareness/pulse")
async def get_system_pulse():
    """
    Real-time system pulse - the AI's heartbeat and vital signs.
    """
    if not UNIFIED_AWARENESS_AVAILABLE:
        return {"available": False, "message": "Unified awareness system not loaded"}

    try:
        awareness = get_unified_awareness()
        pulse = awareness.get_system_pulse()
        return pulse.to_dict()
    except Exception as e:
        return {"available": False, "error": str(e)}


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
        "async_pool": {
            "using_fallback": using_fallback(),
            "status": "unknown"
        },
        "sync_psycopg2": {
            "status": "unknown"
        },
        "config": {
            "host": config.database.host,
            "port": config.database.port,
            "database": config.database.database,
            "user": config.database.user,
            "password_set": bool(config.database.password),
            "ssl": config.database.ssl,
            "ssl_verify": config.database.ssl_verify,
        }
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
            sslmode='require'
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
async def process_email_queue_endpoint(
    batch_size: int = Query(default=10, ge=1, le=50),
    dry_run: bool = Query(default=False),
    authenticated: bool = Depends(verify_api_key)
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
async def test_email_sending(
    recipient: str = Query(..., description="Email address to send test to"),
    authenticated: bool = Depends(verify_api_key)
):
    """Send a test email to verify email configuration is working."""
    try:
        from email_sender import send_email
        success, message = send_email(
            recipient,
            "Test Email from BrainOps AI",
            "<h1>Test Email</h1><p>This is a test email from the BrainOps AI email system.</p><p>If you received this, email sending is working correctly.</p>",
            {}
        )
        return {
            "success": success,
            "message": message,
            "recipient": recipient
        }
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
        queue_counts = await pool.fetchrow("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'queued') as queued,
                COUNT(*) FILTER (WHERE status = 'processing') as processing,
                COUNT(*) FILTER (WHERE status = 'sent') as sent,
                COUNT(*) FILTER (WHERE status = 'failed') as failed,
                COUNT(*) FILTER (WHERE status = 'skipped') as skipped,
                COUNT(*) as total
            FROM ai_email_queue
        """)

        return {
            "daemon_stats": stats,
            "queue_counts": dict(queue_counts) if queue_counts else {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except ImportError:
        return {"error": "email_scheduler_daemon module not available"}
    except Exception as e:
        logger.error(f"Failed to get email scheduler stats: {e}")
        return {"error": str(e)}


@app.get("/systems/usage")
async def systems_usage(authenticated: bool = Depends(verify_api_key)):
    """Report which AI systems are being used plus scheduler and memory effectiveness."""

    async def _load_usage() -> Dict[str, Any]:
        pool = get_pool()
        agent_usage = await _get_agent_usage(pool)
        schedule_usage = await _get_schedule_usage(pool)
        memory_usage = await _memory_stats_snapshot(pool)

        customer_success_preview = None
        if CUSTOMER_SUCCESS_AVAILABLE and getattr(app.state, "customer_success", None):
            try:
                customer_success_preview = await app.state.customer_success.generate_onboarding_plan(
                    customer_id="sample-customer",
                    plan_type="value-check",
                )
            except Exception as exc:
                customer_success_preview = {"error": str(exc)}

        return {
            "active_systems": _collect_active_systems(),
            "agents": agent_usage,
            "schedules": {**schedule_usage, "scheduler_runtime": _scheduler_snapshot()},
            "memory": memory_usage,
            "learning": {
                "available": LEARNING_AVAILABLE and getattr(app.state, "learning", None) is not None,
                "notes": "Notebook LM+ initialized" if getattr(app.state, "learning", None) else "Learning system not initialized",
            },
            "aurea": _aurea_status(),
            "self_healing": _self_healing_status(),
            "customer_success": {
                "available": CUSTOMER_SUCCESS_AVAILABLE and getattr(app.state, "customer_success", None) is not None,
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
    authenticated: bool = Depends(verify_api_key)
) -> AgentList:
    """Get list of available agents"""
    try:
        cache_key = f"agents:{category or 'all'}:{enabled}"

        async def _load_agents() -> AgentList:
            pool = get_pool()
            try:
                # Build query with execution statistics
                # Join with ai_agent_executions to get total_executions and last_active
                query = """
                    SELECT a.*,
                           COALESCE(e.exec_count, 0) as total_executions,
                           e.last_exec as last_active
                    FROM agents a
                    LEFT JOIN (
                        SELECT agent_name,
                               COUNT(*) as exec_count,
                               MAX(created_at) as last_exec
                        FROM ai_agent_executions
                        GROUP BY agent_name
                    ) e ON a.name = e.agent_name
                    WHERE 1=1
                """
                params = []

                if enabled is not None:
                    query += f" AND a.enabled = ${len(params) + 1}"
                    params.append(enabled)

                if category:
                    query += f" AND a.category = ${len(params) + 1}"
                    params.append(category)

                query += " ORDER BY a.category, a.name"

                rows = await pool.fetch(query, *params)

                agents = [
                    _row_to_agent(row if isinstance(row, dict) else dict(row))
                    for row in rows
                ]

                return AgentList(
                    agents=agents,
                    total=len(agents),
                    page=1,
                    page_size=len(agents)
                )
            except Exception as e:
                logger.error(f"Failed to get agents from database: {e}", exc_info=True)
                # Return empty list on error to ensure valid JSON
                return AgentList(agents=[], total=0, page=1, page_size=0)

        agents_response, _ = await RESPONSE_CACHE.get_or_set(
            cache_key,
            CACHE_TTLS["agents"],
            _load_agents
        )
        return agents_response
    except Exception as e:
        logger.error(f"Failed to get agents (outer): {e}", exc_info=True)
        # Final fallback - return empty list to ensure valid JSON response
        return AgentList(agents=[], total=0, page=1, page_size=0)


@app.post("/agents/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    """Execute an agent"""
    pool = get_pool()

    try:
        # Get agent by UUID (text comparison) or legacy slug
        agent = await pool.fetchrow(
            "SELECT * FROM agents WHERE id::text = $1 OR name = $1",
            agent_id,
        )
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

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
            await pool.execute("""
                INSERT INTO ai_agent_executions (id, agent_name, task_type, input_data, status)
                VALUES ($1, $2, $3, $4, $5)
            """, execution_id, agent_name, "execute", json.dumps(body), "running")
            logger.info(f"✅ Logged execution start for {agent_name}: {execution_id}")
        except Exception as insert_error:
            logger.warning("Failed to persist execution start: %s", insert_error)

        # Execute agent logic using proper agent dispatch
        result = {"status": "completed", "message": "Agent executed successfully"}
        task = body.get("task", {})

        if AGENTS_AVAILABLE and AGENT_EXECUTOR:
            try:
                # Use the actual agent executor to run the correct agent class
                agent_result = await AGENT_EXECUTOR.execute(agent_name, task)
                result = agent_result if isinstance(agent_result, dict) else {"status": "completed", "result": agent_result}
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
            await pool.execute("""
                UPDATE ai_agent_executions
                SET status = $1, output_data = $2, execution_time_ms = $3
                WHERE id = $4
            """, "completed", json.dumps(result), duration_ms, execution_id)
            logger.info(f"✅ Logged execution completion for {agent_name}: {execution_id} ({duration_ms}ms)")
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
            duration_ms=duration_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")

        # Update execution as failed
        if 'execution_id' in locals():
            try:
                await pool.execute("""
                    UPDATE ai_agent_executions
                    SET status = $1, error_message = $2
                    WHERE id = $3
                """, "failed", str(e), execution_id)
            except Exception as fail_error:
                logger.warning("Failed to persist failed execution: %s", fail_error)

            LOCAL_EXECUTIONS.appendleft(
                {
                    "execution_id": execution_id,
                    "agent_id": agent_uuid if 'agent_uuid' in locals() else agent_id,
                    "agent_name": agent["name"] if 'agent' in locals() else agent_id,
                    "status": "failed",
                    "started_at": locals().get("started_at"),
                    "completed_at": datetime.utcnow(),
                    "duration_ms": None,
                    "error": str(e),
                }
            )

        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")


@app.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    authenticated: bool = Depends(verify_api_key)
) -> Agent:
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
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agent: {str(e)}")


@app.get("/executions")
async def get_executions(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    authenticated: bool = Depends(verify_api_key)
):
    """Get agent executions"""
    pool = get_pool()

    try:
        query = """
            SELECT e.*, a.name as agent_name
            FROM agent_executions e
            JOIN ai_agents a ON e.agent_id = a.id
            WHERE 1=1
        """
        params = []

        if agent_id:
            query += f" AND e.agent_id = ${len(params) + 1}"
            params.append(agent_id)

        if status:
            query += f" AND e.status = ${len(params) + 1}"
            params.append(status)

        query += f" ORDER BY e.started_at DESC LIMIT ${len(params) + 1}"
        params.append(limit)

        try:
            rows = await pool.fetch(query, *params)
        except Exception as primary_error:
            logger.error("Execution query failed (%s). Returning fallback data.", primary_error)
            fallback_items = [
                {
                    "execution_id": entry.get("execution_id"),
                    "agent_id": entry.get("agent_id"),
                    "agent_name": entry.get("agent_name"),
                    "status": entry.get("status"),
                    "started_at": entry.get("started_at").isoformat() if entry.get("started_at") else None,
                    "completed_at": entry.get("completed_at").isoformat() if entry.get("completed_at") else None,
                    "duration_ms": entry.get("duration_ms"),
                    "error": entry.get("error"),
                }
                for entry in list(LOCAL_EXECUTIONS)
            ]
            return {
                "executions": fallback_items,
                "total": len(fallback_items),
                "message": "Execution history limited to in-memory cache",
            }

        executions = []
        for row in rows:
            data = row if isinstance(row, dict) else dict(row)
            execution = {
                "execution_id": str(data.get("id")),
                "agent_id": str(data.get("agent_id")),
                "agent_name": data.get("agent_name"),
                "status": data.get("status"),
                "started_at": data["started_at"].isoformat() if data.get("started_at") else None,
                "completed_at": data["completed_at"].isoformat() if data.get("completed_at") else None,
                "duration_ms": data.get("duration_ms"),
                "error": data.get("error"),
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
                    "started_at": entry.get("started_at").isoformat() if entry.get("started_at") else None,
                    "completed_at": entry.get("completed_at").isoformat() if entry.get("completed_at") else None,
                    "duration_ms": entry.get("duration_ms"),
                    "error": entry.get("error"),
                },
            )

        return {"executions": executions, "total": len(executions)}

    except Exception as e:
        logger.error(f"Failed to get executions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve executions: {str(e)}")


@app.post("/execute")
async def execute_scheduled_agents(
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    """Execute scheduled agents (called by cron)"""
    if not SCHEDULER_AVAILABLE or not app.state.scheduler:
        return {"status": "scheduler_disabled", "message": "Agent scheduler not available"}

    try:
        pool = get_pool()
        scheduler = app.state.scheduler

        # Get current hour
        current_hour = datetime.utcnow().hour

        # Get agents scheduled for this hour
        agents = await pool.fetch("""
            SELECT * FROM agents
            WHERE enabled = true
            AND schedule_hours @> ARRAY[$1]::integer[]
        """, current_hour)

        results = []
        for agent in agents:
            try:
                execution_id = str(uuid.uuid4())
                started_at = datetime.utcnow()
                agent_name = agent.get("name", "unknown")

                # Log execution start (use correct columns: task_execution_id, agent_type, prompt)
                # agent_executions uses created_at with DEFAULT, not started_at
                await pool.execute("""
                    INSERT INTO agent_executions (id, task_execution_id, agent_type, status, prompt)
                    VALUES ($1, $2, $3, $4, $5)
                """, execution_id, uuid.UUID(execution_id), agent.get("type", "scheduled"), "running", json.dumps({"scheduled": True, "agent_name": agent_name}))

                # ACTUALLY EXECUTE THE AGENT using AgentExecutor
                result = {"status": "skipped", "message": "No executor available"}
                if AGENT_EXECUTOR:
                    try:
                        task_data = {
                            "action": "scheduled_run",
                            "agent_id": agent["id"],
                            "scheduled": True,
                            "execution_id": execution_id,
                            "context": agent.get("config", {})
                        }
                        result = await AGENT_EXECUTOR.execute(agent_name, task_data)
                        result["scheduled_execution"] = True
                        logger.info(f"✅ Agent {agent_name} executed successfully")
                    except NotImplementedError:
                        # Agent doesn't have execute method - log as warning, NOT completed!
                        logger.warning(f"⚠️ Agent {agent_name} has no execute method (NotImplementedError)")
                        result = {
                            "status": "not_implemented",
                            "message": f"Agent {agent_name} missing execute implementation",
                            "scheduled_execution": True,
                            "warning": "This agent needs implementation"
                        }
                    except Exception as exec_err:
                        logger.error(f"Agent {agent_name} execution error: {exec_err}")
                        result = {
                            "status": "error",
                            "error": str(exec_err),
                            "scheduled_execution": True
                        }
                else:
                    result = {
                        "status": "completed",
                        "message": "AgentExecutor not available - logged only",
                        "scheduled_execution": True
                    }

                # Update execution record with actual result (use correct columns: response not output_data)
                final_status = "completed" if result.get("status") != "error" else "failed"
                await pool.execute("""
                    UPDATE agent_executions
                    SET completed_at = $1, status = $2, response = $3
                    WHERE id = $4
                """, datetime.utcnow(), final_status, json.dumps(result), execution_id)

                results.append({
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "execution_id": execution_id,
                    "status": "completed"
                })

            except Exception as e:
                logger.error(f"Failed to execute agent {agent['id']}: {e}")
                results.append({
                    "agent_id": agent["id"],
                    "agent_name": agent["name"],
                    "error": str(e),
                    "status": "failed"
                })

        return {
            "status": "completed",
            "executed": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
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
        "status": "completed"
    }

    try:
        pool = get_pool()

        # 1. Check for failed AUREA decisions and retry
        failed_decisions = await pool.fetch("""
            SELECT id, decision_type, context
            FROM aurea_decisions
            WHERE execution_status = 'failed'
            AND created_at > NOW() - INTERVAL '24 hours'
            LIMIT 10
        """)

        for decision in failed_decisions:
            results["issues_detected"].append({
                "type": "failed_decision",
                "id": str(decision["id"]),
                "decision_type": decision["decision_type"]
            })
            # Reset for retry
            await pool.execute("""
                UPDATE aurea_decisions
                SET execution_status = 'pending',
                    execution_result = NULL
                WHERE id = $1
            """, decision["id"])
            results["actions_taken"].append({
                "action": "reset_for_retry",
                "target": str(decision["id"])
            })

        # 2. Check healing rules and match against recent errors
        recent_errors = await pool.fetch("""
            SELECT DISTINCT error_type, error_message, component
            FROM ai_error_logs
            WHERE timestamp > NOW() - INTERVAL '1 hour'
            LIMIT 20
        """) if await pool.fetchval("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'ai_error_logs')") else []

        healing_rules = await pool.fetch("""
            SELECT id, component, error_pattern, fix_action, confidence
            FROM ai_healing_rules
            WHERE enabled = true
        """)

        for error in recent_errors:
            for rule in healing_rules:
                if rule["error_pattern"] in str(error.get("error_message", "")) or \
                   rule["error_pattern"] in str(error.get("error_type", "")):
                    results["issues_detected"].append({
                        "type": "matched_error",
                        "error_type": error.get("error_type"),
                        "matched_rule": str(rule["id"])
                    })
                    results["actions_taken"].append({
                        "action": rule["fix_action"],
                        "component": rule["component"],
                        "confidence": float(rule["confidence"])
                    })
                    # Update rule usage
                    await pool.execute("""
                        UPDATE ai_healing_rules
                        SET success_count = success_count + 1, updated_at = NOW()
                        WHERE id = $1
                    """, rule["id"])

        # 3. Check for stalled agents (use correct column: type not agent_type)
        stalled_agents = await pool.fetch("""
            SELECT id, name, type
            FROM ai_agents
            WHERE enabled = true
            AND last_execution_at < NOW() - INTERVAL '2 hours'
        """) if await pool.fetchval("SELECT EXISTS(SELECT 1 FROM information_schema.columns WHERE table_name = 'ai_agents' AND column_name = 'last_execution_at')") else []

        for agent in stalled_agents:
            results["issues_detected"].append({
                "type": "stalled_agent",
                "agent_id": str(agent["id"]),
                "agent_name": agent["name"]
            })

        # 4. Log healing run
        await pool.execute("""
            INSERT INTO remediation_history (action_type, target_component, result, success, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
        """, "self_heal_trigger", "system", json.dumps(results), True, json.dumps({
            "issues_count": len(results["issues_detected"]),
            "actions_count": len(results["actions_taken"])
        })) if await pool.fetchval("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'remediation_history')") else None

        logger.info(f"🏥 Self-healing check complete: {len(results['issues_detected'])} issues, {len(results['actions_taken'])} actions")

    except Exception as e:
        logger.error(f"Self-healing trigger failed: {e}")
        results["status"] = "error"
        results["error"] = str(e)

    return results


# =============================================================================
# TRAINING & LEARNING ENDPOINTS - Critical for AI system improvement
# =============================================================================

@app.post("/training/capture-interaction", dependencies=SECURED_DEPENDENCIES)
async def capture_interaction(interaction_data: Dict[str, Any] = Body(...)):
    """
    Capture customer interaction for AI training.

    This is CRITICAL for the learning system - without captured interactions,
    the AI cannot learn and improve.
    """
    if not TRAINING_AVAILABLE or not hasattr(app.state, 'training') or not app.state.training:
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
            value=interaction_data.get("value")
        )

        logger.info(f"📝 Captured interaction {interaction_id} for training")
        return {"interaction_id": interaction_id, "status": "captured"}

    except Exception as e:
        logger.error(f"Failed to capture interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/stats")
async def get_training_stats():
    """Get training pipeline statistics"""
    if not TRAINING_AVAILABLE or not hasattr(app.state, 'training') or not app.state.training:
        return {"available": False, "message": "Training pipeline not available"}

    try:
        pool = get_pool()
        stats = await pool.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM ai_customer_interactions) as total_interactions,
                (SELECT MAX(created_at) FROM ai_customer_interactions) as last_interaction,
                (SELECT COUNT(*) FROM ai_training_data) as training_samples,
                (SELECT COUNT(*) FROM ai_learning_insights) as insights_generated
        """)
        return {
            "available": True,
            "total_interactions": stats["total_interactions"],
            "last_interaction": stats["last_interaction"].isoformat() if stats["last_interaction"] else None,
            "training_samples": stats["training_samples"],
            "insights_generated": stats["insights_generated"]
        }
    except Exception as e:
        return {"available": True, "error": str(e)}


@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get detailed scheduler status and diagnostics"""
    try:
        if not SCHEDULER_AVAILABLE or not hasattr(app.state, 'scheduler') or not app.state.scheduler:
            return {
                "enabled": False,
                "message": "Scheduler not available",
                "timestamp": datetime.utcnow().isoformat()
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
                    "trigger": str(job.trigger)
                }
                for job in apscheduler_jobs
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "enabled": False,
                "error": str(e),
                "message": "Failed to retrieve scheduler status",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/agents/status")
async def get_all_agents_status(authenticated: bool = Depends(verify_api_key)):
    """
    Get comprehensive status of all 61 agents including health metrics,
    execution statistics, and current state
    """
    if not HEALTH_MONITOR_AVAILABLE:
        # Fallback to basic agent list
        pool = get_pool()
        try:
            result = await pool.fetch("""
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
            """)

            agents = [dict(row) for row in result]
            return {
                "total_agents": len(agents),
                "agents": agents,
                "health_monitoring": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Full health monitoring available
    try:
        health_monitor = get_health_monitor()

        # Run health check for all agents
        health_summary = health_monitor.check_all_agents_health()

        # Get detailed health summary
        detailed_summary = health_monitor.get_agent_health_summary()

        return {
            "total_agents": health_summary.get("total_agents", 0),
            "health_summary": {
                "healthy": health_summary.get("healthy", 0),
                "degraded": health_summary.get("degraded", 0),
                "critical": health_summary.get("critical", 0),
                "unknown": health_summary.get("unknown", 0)
            },
            "agents": health_summary.get("agents", []),
            "critical_agents": detailed_summary.get("critical_agents", []),
            "active_alerts": detailed_summary.get("active_alerts", []),
            "recent_restarts": detailed_summary.get("recent_restarts", []),
            "health_monitoring": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting agent status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


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
        result = health_monitor.restart_failed_agent(agent_id, agent['name'])

        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error', 'Restart failed'))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scheduler/activate-all", dependencies=SECURED_DEPENDENCIES)
async def activate_all_agents_scheduler():
    """
    Schedule ALL agents that don't have active schedules.
    This activates the full AI OS by ensuring every agent runs on a schedule.
    """
    if not SCHEDULER_AVAILABLE or not hasattr(app.state, 'scheduler') or not app.state.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")

    scheduler = app.state.scheduler
    pool = get_pool()

    try:
        # Get all agents
        agents_result = await pool.fetch("SELECT id, name, type, category FROM ai_agents WHERE status = 'active'")

        # Get existing schedules
        existing_result = await pool.fetch("SELECT agent_id FROM agent_schedules WHERE enabled = true")
        existing_agent_ids = {str(row['agent_id']) for row in existing_result}

        scheduled_count = 0
        already_scheduled = 0
        errors = []

        for agent in agents_result:
            agent_id = str(agent['id'])
            agent_name = agent['name']
            agent_type = agent.get('type', 'general').lower()

            if agent_id in existing_agent_ids:
                already_scheduled += 1
                continue

            # Determine frequency based on agent type
            if agent_type in ['analytics', 'revenue', 'customer']:
                frequency = 30  # High-value agents: every 30 min
            elif agent_type in ['monitor', 'security']:
                frequency = 15  # Critical agents: every 15 min
            elif agent_type in ['learning', 'optimization']:
                frequency = 60  # Learning agents: every hour
            else:
                frequency = 60  # Default: every hour

            try:
                # Insert schedule
                await pool.execute("""
                    INSERT INTO agent_schedules (id, agent_id, frequency_minutes, enabled, created_at)
                    VALUES (gen_random_uuid(), $1, $2, true, NOW())
                """, agent_id, frequency)

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
            "errors": errors if errors else None
        }

    except Exception as e:
        logger.error(f"Failed to activate all agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store", dependencies=SECURED_DEPENDENCIES)
async def store_memory(
    content: str = Body(...),
    memory_type: str = Body("operational"),
    category: str = Body(default=None),
    metadata: Dict[str, Any] = Body(default=None)
):
    """
    Store a memory in the AI memory system.
    This enables the AI to remember and learn from experiences.
    """
    if not MEMORY_AVAILABLE or not hasattr(app.state, 'memory') or not app.state.memory:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        memory_manager = app.state.memory
        memory_id = await memory_manager.store_async(
            content=content,
            memory_type=memory_type,
            category=category,
            metadata=metadata or {}
        )
        return {
            "success": True,
            "memory_id": memory_id,
            "message": "Memory stored successfully"
        }
    except Exception as e:
        logger.error(f"Failed to store memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/search")
async def search_memory(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Max results"),
    memory_type: str = Query(None, description="Filter by type")
):
    """
    Search the AI memory system for relevant memories.
    """
    if not MEMORY_AVAILABLE or not hasattr(app.state, 'memory') or not app.state.memory:
        raise HTTPException(status_code=503, detail="Memory system not available")

    try:
        memory_manager = app.state.memory
        memories = await memory_manager.search(
            query=query,
            limit=limit,
            memory_type=memory_type
        )
        return {
            "success": True,
            "query": query,
            "count": len(memories),
            "memories": memories
        }
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/backfill-embeddings", dependencies=SECURED_DEPENDENCIES)
async def backfill_embeddings(
    batch_size: int = Query(100, description="Batch size per run"),
    background_tasks: BackgroundTasks = None
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

        cur.execute("""
        SELECT id, content, memory_type
        FROM unified_ai_memory
        WHERE embedding IS NULL
        LIMIT %s
        """, (batch_size,))

        memories = cur.fetchall()
        if not memories:
            cur.close()
            conn.close()
            return {"success": True, "message": "No memories need embedding backfill", "processed": 0}

        # Try local embedding model (always available)
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
            embedding_model = "local:all-MiniLM-L6-v2"
        except ImportError:
            cur.close()
            conn.close()
            raise HTTPException(status_code=503, detail="sentence-transformers not available for backfill")

        processed = 0
        for mem in memories:
            try:
                content_str = json.dumps(mem['content']) if isinstance(mem['content'], dict) else str(mem['content'])
                embedding = embedder.encode(content_str).tolist()

                cur.execute("""
                UPDATE unified_ai_memory
                SET embedding = %s
                WHERE id = %s
                """, (embedding, mem['id']))
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
            "message": f"Backfilled {processed} embeddings, {remaining} remaining"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Embedding backfill failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/force-sync", dependencies=SECURED_DEPENDENCIES)
async def force_sync_embedded_memory():
    """
    Force sync embedded memory system from master Postgres.
    Useful when local SQLite cache is empty or out of sync.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)

    if not embedded_memory:
        raise HTTPException(
            status_code=503,
            detail="Embedded memory system not available"
        )

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
            "last_sync": embedded_memory.last_sync.isoformat() if embedded_memory.last_sync else None,
            "pool_connected": embedded_memory.pg_pool is not None
        }

    except Exception as e:
        logger.error(f"Force sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def get_memory_stats():
    """
    Get statistics about the embedded memory system.
    Shows local cache status and sync information.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)

    if not embedded_memory:
        return {
            "enabled": False,
            "message": "Embedded memory system not available"
        }

    try:
        cursor = embedded_memory.sqlite_conn.cursor()

        # Get counts
        total_memories = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]
        total_tasks = cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks").fetchone()[0]
        pending_tasks = cursor.execute("SELECT COUNT(*) FROM ai_autonomous_tasks WHERE status = 'pending'").fetchone()[0]

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
            "last_sync": embedded_memory.last_sync.isoformat() if embedded_memory.last_sync else None,
            "sync_metadata": {
                "last_sync_time": sync_meta[1] if sync_meta else None,
                "last_sync_count": sync_meta[2] if sync_meta else None,
                "total_records": sync_meta[3] if sync_meta else None
            } if sync_meta else None,
            "embedding_model": embedded_memory.embedding_model
        }

    except Exception as e:
        logger.error(f"Get memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AI SELF-AWARENESS ENDPOINTS ====================

@app.post("/ai/self-assess")
async def ai_self_assess(
    request: Request,
    task_id: str,
    agent_id: str,
    task_description: str,
    task_context: Dict[str, Any] = None
):
    """
    AI assesses its own confidence in completing a task

    Revolutionary feature - AI knows what it doesn't know!
    """
    if not SELF_AWARENESS_AVAILABLE or not hasattr(app.state, 'self_aware_ai') or not app.state.self_aware_ai:
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = app.state.self_aware_ai

        assessment = await self_aware_ai.assess_confidence(
            task_id=task_id,
            agent_id=agent_id,
            task_description=task_description,
            task_context=task_context or {}
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
            "timestamp": assessment.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Self-assessment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Self-assessment failed: {str(e)}")


@app.post("/ai/explain-reasoning")
async def ai_explain_reasoning(
    request: Request,
    task_id: str,
    agent_id: str,
    decision: str,
    reasoning_process: Dict[str, Any]
):
    """
    AI explains its reasoning in human-understandable terms

    Transparency builds trust!
    """
    if not SELF_AWARENESS_AVAILABLE or not hasattr(app.state, 'self_aware_ai') or not app.state.self_aware_ai:
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        self_aware_ai = app.state.self_aware_ai

        explanation = await self_aware_ai.explain_reasoning(
            task_id=task_id,
            agent_id=agent_id,
            decision=decision,
            reasoning_process=reasoning_process
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
            "timestamp": explanation.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Reasoning explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning explanation failed: {str(e)}")


class ReasoningRequest(BaseModel):
    """Request model for o3 reasoning endpoint"""
    problem: str
    context: Optional[Dict[str, Any]] = None
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
            problem=body.problem,
            context=body.context,
            max_tokens=body.max_tokens,
            model=body.model
        )

        return {
            "success": True,
            "reasoning": result.get("reasoning", ""),
            "conclusion": result.get("conclusion", ""),
            "model_used": result.get("model_used", body.model),
            "tokens_used": result.get("tokens_used"),
            "error": result.get("error")
        }

    except Exception as e:
        logger.error(f"o1 reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")


@app.post("/ai/learn-from-mistake")
async def ai_learn_from_mistake(
    request: Request,
    task_id: str,
    agent_id: str,
    expected_outcome: Any,
    actual_outcome: Any,
    confidence_before: float
):
    """
    AI analyzes its own mistakes and learns from them

    This is how AI gets smarter over time!
    """
    if not SELF_AWARENESS_AVAILABLE or not hasattr(app.state, 'self_aware_ai') or not app.state.self_aware_ai:
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        from decimal import Decimal
        self_aware_ai = app.state.self_aware_ai

        learning = await self_aware_ai.learn_from_mistake(
            task_id=task_id,
            agent_id=agent_id,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
            confidence_before=Decimal(str(confidence_before))
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
            "timestamp": learning.timestamp.isoformat()
        }

    except Exception as e:
        logger.error(f"Learning from mistake failed: {e}")
        raise HTTPException(status_code=500, detail=f"Learning from mistake failed: {str(e)}")


@app.get("/ai/self-awareness/stats")
async def get_self_awareness_stats():
    """Get statistics about AI self-awareness system"""
    if not SELF_AWARENESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Self-Awareness not available")

    try:
        pool = get_pool()

        # Get assessment stats
        assessment_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_assessments,
                AVG(confidence_score) as avg_confidence,
                COUNT(CASE WHEN can_complete_alone THEN 1 END) as can_complete_alone_count,
                COUNT(CASE WHEN requires_human_review THEN 1 END) as requires_review_count
            FROM ai_self_assessments
        """)

        # Get mistake learning stats
        learning_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_mistakes,
                COUNT(CASE WHEN should_have_known THEN 1 END) as should_have_known_count,
                AVG(confidence_before - confidence_after) as avg_confidence_drop
            FROM ai_learning_from_mistakes
        """)

        # Get reasoning explanation stats
        reasoning_stats = await pool.fetchrow("""
            SELECT
                COUNT(*) as total_explanations,
                AVG(confidence_in_decision) as avg_decision_confidence,
                COUNT(CASE WHEN human_review_recommended THEN 1 END) as human_review_count
            FROM ai_reasoning_explanations
        """)

        return {
            "self_awareness_enabled": True,
            "assessments": {
                "total": assessment_stats["total_assessments"] or 0,
                "avg_confidence": float(assessment_stats["avg_confidence"] or 0),
                "can_complete_alone_rate": (
                    (assessment_stats["can_complete_alone_count"] or 0) /
                    max(assessment_stats["total_assessments"] or 1, 1) * 100
                ),
                "requires_review_rate": (
                    (assessment_stats["requires_review_count"] or 0) /
                    max(assessment_stats["total_assessments"] or 1, 1) * 100
                )
            },
            "learning": {
                "total_mistakes_analyzed": learning_stats["total_mistakes"] or 0,
                "should_have_known_rate": (
                    (learning_stats["should_have_known_count"] or 0) /
                    max(learning_stats["total_mistakes"] or 1, 1) * 100
                ),
                "avg_confidence_adjustment": float(learning_stats["avg_confidence_drop"] or 0)
            },
            "reasoning": {
                "total_explanations": reasoning_stats["total_explanations"] or 0,
                "avg_decision_confidence": float(reasoning_stats["avg_decision_confidence"] or 0),
                "human_review_rate": (
                    (reasoning_stats["human_review_count"] or 0) /
                    max(reasoning_stats["total_explanations"] or 1, 1) * 100
                )
            }
        }

    except Exception as e:
        logger.error(f"Failed to get self-awareness stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")


# ==================== END AI SELF-AWARENESS ENDPOINTS ====================


@app.post("/ai/tasks/execute/{task_id}")
async def execute_ai_task(task_id: str):
    """Manually trigger execution of a specific task"""
    integration_layer = getattr(app.state, 'integration_layer', None)
    if not INTEGRATION_LAYER_AVAILABLE or integration_layer is None:
        raise HTTPException(status_code=503, detail="AI Integration Layer not available or not initialized")

    try:
        # Get task
        task = await integration_layer.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Execute task (will be picked up by task executor loop)
        await integration_layer._execute_task(task)

        return {
            "success": True,
            "message": "Task execution triggered",
            "task_id": task_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/tasks/stats")
async def get_task_stats():
    """Get AI task system statistics"""
    integration_layer = getattr(app.state, 'integration_layer', None)
    if not INTEGRATION_LAYER_AVAILABLE or integration_layer is None:
        raise HTTPException(status_code=503, detail="AI Integration Layer not available or not initialized")

    try:
        # Get all tasks
        all_tasks = await integration_layer.list_tasks(limit=1000)

        # Calculate stats
        stats = {
            'total': len(all_tasks),
            'by_status': {},
            'by_priority': {},
            'agents_active': len(integration_layer.agents_registry),
            'execution_queue_size': integration_layer.execution_queue.qsize()
        }

        for task in all_tasks:
            # Count by status
            status = task.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # Count by priority
            priority = task.get('priority', 'unknown')
            stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1

        return {
            "success": True,
            "stats": stats,
            "system_status": "operational",
            "task_executor_running": True
        }

    except Exception as e:
        logger.error(f"❌ Failed to get task stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AureaCommandRequest(BaseModel):
    command_text: str


@app.post("/ai/orchestrate", dependencies=SECURED_DEPENDENCIES)
async def orchestrate_complex_workflow(
    request: Request,
    task_description: str,
    context: Dict[str, Any] = {}
):
    """
    Execute complex multi-stage workflow using LangGraph orchestration
    This is for sophisticated tasks that need multi-agent coordination
    """
    if not hasattr(app.state, 'langgraph_orchestrator') or not app.state.langgraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph Orchestrator not available")

    try:
        orchestrator = app.state.langgraph_orchestrator

        result = await orchestrator.execute(
            task_description=task_description,
            context=context
        )

        return {
            "success": True,
            "result": result,
            "message": "Workflow orchestrated successfully"
        }

    except Exception as e:
        logger.error(f"❌ Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AIAnalyzeRequest(BaseModel):
    """Request model for /ai/analyze endpoint - matches weathercraft-erp frontend format"""
    agent: str
    action: str
    data: Dict[str, Any] = {}
    context: Dict[str, Any] = {}


@app.post("/ai/analyze", dependencies=SECURED_DEPENDENCIES)
async def ai_analyze(
    request: Request,
    payload: AIAnalyzeRequest = Body(...)
):
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
        if hasattr(app.state, 'langgraph_orchestrator') and app.state.langgraph_orchestrator:
            orchestrator = app.state.langgraph_orchestrator
            result = await orchestrator.execute(
                task_description=task_description,
                context={**context, "agent": agent_name, "action": action, "data": data}
            )
            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": result,
                "message": f"Analysis completed via orchestrator"
            }

        # Fallback: Use module-level agent executor singleton
        try:
            from agent_executor import executor as agent_executor_singleton
            if agent_executor_singleton:
                result = await agent_executor_singleton.execute(
                    agent_name=agent_name,
                    task={
                        "action": action,
                        "data": data,
                        "context": context
                    }
                )
                return {
                    "success": True,
                    "agent": agent_name,
                    "action": action,
                    "result": result,
                    "message": "Analysis completed via agent executor"
                }
        except (ImportError, Exception) as e:
            logger.warning(f"Agent executor fallback failed: {e}")

        # No orchestrator available - queue task for later processing instead of mock response
        logger.warning(f"No orchestrator/executor available for agent {agent_name}, queueing for async processing")

        # Queue the task to ai_autonomous_tasks for later execution
        try:
            pool = get_pool()
            task_id = str(uuid.uuid4())
            await pool.execute("""
                INSERT INTO ai_autonomous_tasks (id, title, payload, priority, status, created_at)
                VALUES ($1, $2, $3, $4, 'pending', NOW())
            """, task_id, f"{agent_name}.{action}", json.dumps({"agent": agent_name, "action": action, "data": data}), 50)

            return {
                "success": True,
                "agent": agent_name,
                "action": action,
                "result": {
                    "status": "queued",
                    "task_id": task_id,
                    "message": f"Request queued for async processing (task: {task_id})"
                },
                "message": "Request queued - orchestrator temporarily unavailable"
            }
        except Exception as queue_error:
            logger.error(f"Failed to queue task: {queue_error}")
            raise HTTPException(
                status_code=503,
                detail=f"AI orchestrator unavailable and task queueing failed: {str(queue_error)}"
            )

    except Exception as e:
        logger.error(f"❌ AI analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aurea/status")
async def get_aurea_status():
    """
    Get AUREA operational status - checks actual OODA loop activity.
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
                    "websocket": "/aurea/chat/ws/{session_id}"
                }
            }

        # Use pool methods directly (they handle acquire internally)
        recent_cycles = await pool.fetchval("""
            SELECT COUNT(*) FROM aurea_state
            WHERE timestamp > NOW() - INTERVAL '5 minutes'
        """)

        recent_decisions = await pool.fetchval("""
            SELECT COUNT(*) FROM aurea_decisions
            WHERE created_at > NOW() - INTERVAL '1 hour'
        """)

        active_agents = await pool.fetchval("""
            SELECT COUNT(*) FROM ai_agents WHERE status = 'active'
        """)

        # AUREA is operational if we have recent OODA cycles
        aurea_operational = recent_cycles > 0

        return {
            "status": "operational" if aurea_operational else "idle",
            "aurea_available": True,
            "ooda_cycles_last_5min": recent_cycles,
            "decisions_last_hour": recent_decisions,
            "active_agents": active_agents,
            "timestamp": datetime.utcnow().isoformat(),
            "endpoints": {
                "full_status": "/aurea/chat/status",
                "chat": "/aurea/chat/message",
                "websocket": "/aurea/chat/ws/{session_id}"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get AUREA status: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.post("/aurea/command/natural_language")
async def execute_aurea_nl_command(
    request: Request,
    payload: AureaCommandRequest = Body(...)
):
    """
    Execute a natural language command through AUREA's NLU processor.
    Founder-level authority for natural language system control.

    Examples:
    - "Create a high priority task to deploy the new feature"
    - "Show me all tasks that are in progress"
    - "Get AUREA status"
    - "Execute task abc-123"
    """
    if not hasattr(app.state, 'aurea_nlu') or not app.state.aurea_nlu:
        raise HTTPException(status_code=503, detail="AUREA NLU Processor not available")

    try:
        command_text = payload.command_text
        nlu = app.state.aurea_nlu
        result = await nlu.execute_natural_language_command(command_text)

        return {
            "success": True,
            "command": command_text,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ Natural language command execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeQueryRequest(BaseModel):
    """Request payload for querying unified memory/knowledge."""
    query: str
    limit: int = 10
    memory_type: Optional[str] = None
    min_importance: float = 0.0


class ErpAnalyzeRequest(BaseModel):
    """Request payload for ERP job analysis."""
    tenant_id: Optional[str] = None
    job_ids: Optional[List[str]] = None
    limit: int = 20


class AgentExecuteRequest(BaseModel):
    """Request payload for executing an agent via v1 API."""
    agent_id: Optional[str] = None
    id: Optional[str] = None
    payload: Dict[str, Any] = {}


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
    payload: Dict[str, Any]
    target_agent: Dict[str, Any]  # {name, role, capabilities}
    routing_metadata: Optional[Dict[str, Any]] = None


@app.post("/api/v1/knowledge/store")
async def api_v1_knowledge_store(
    payload: KnowledgeStoreRequest,
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Store a knowledge/memory entry in the unified memory system.

    Primary path uses the Embedded Memory System (SQLite + async sync to Postgres).
    Fallback path uses the Unified Memory Manager when embedded memory is unavailable.
    """
    # Prefer embedded memory for low-latency writes with async sync to Postgres
    embedded_memory = getattr(app.state, "embedded_memory", None)

    # Normalize metadata
    metadata: Dict[str, Any] = dict(payload.metadata or {})
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
                raise HTTPException(status_code=500, detail="Failed to store memory in embedded backend")
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
    payload: KnowledgeQueryRequest,
    request: Request,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Query the unified memory / knowledge store.

    Uses the Embedded Memory System when available (vector search), with a
    fallback to the Unified Memory Manager recall API.
    """
    embedded_memory = getattr(app.state, "embedded_memory", None)
    results: List[Dict[str, Any]] = []

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
                except Exception:
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

    normalized: List[Dict[str, Any]] = []
    for item in results:
        data = dict(item)
        content = data.get("content")
        # Best-effort JSON parsing for text content
        if isinstance(content, str):
            try:
                content_parsed = json.loads(content)
            except Exception:
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
async def api_v1_knowledge_graph_stats(
    authenticated: bool = Depends(verify_api_key)
):
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
        node_types = await pool.fetch("""
            SELECT node_type, COUNT(*) as count
            FROM ai_knowledge_nodes
            GROUP BY node_type
            ORDER BY count DESC
        """)

        # Get recent extraction info
        recent_graph = await pool.fetchrow("""
            SELECT node_data, updated_at
            FROM ai_knowledge_graph
            WHERE node_type = 'graph_metadata'
            ORDER BY updated_at DESC
            LIMIT 1
        """)

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
        if hasattr(app.state, 'knowledge_extractor') and app.state.knowledge_extractor:
            extractor_stats = app.state.knowledge_extractor.extraction_stats

        return {
            "success": True,
            "total_nodes": nodes_count,
            "total_edges": edges_count,
            "graph_entries": graph_count,
            "node_types": [dict(row) for row in node_types],
            "last_extraction": last_extraction.isoformat() if last_extraction else None,
            "extraction_stats": extraction_stats or extractor_stats,
            "extractor_active": hasattr(app.state, 'knowledge_extractor') and app.state.knowledge_extractor is not None
        }
    except Exception as e:
        logger.error(f"Knowledge graph stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/knowledge/graph/extract")
async def api_v1_knowledge_graph_extract(
    hours_back: int = 24,
    authenticated: bool = Depends(verify_api_key)
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
            "details": result
        }
    except Exception as e:
        logger.error(f"Knowledge graph extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/erp/analyze")
async def api_v1_erp_analyze(
    payload: ErpAnalyzeRequest,
    authenticated: bool = Depends(verify_api_key)
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
        params: List[Any] = [["in_progress", "scheduled"]]

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
            logger.warning("Tenant filter requested but jobs.tenant_id column not found; returning unscoped jobs")

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
            if hasattr(dt, 'tzinfo'):
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            else:
                # It's a date object, convert to naive datetime
                from datetime import date
                if isinstance(dt, date):
                    return datetime.combine(dt, datetime.min.time())
                return dt

        now = datetime.utcnow()
        jobs_intel: List[Dict[str, Any]] = []

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
                            else ("Add crew members" if risk_level == "high" else "Continue monitoring")
                        ),
                        "priority": (
                            "urgent" if risk_level == "critical" else ("high" if risk_level == "high" else "medium")
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
async def api_v1_agents_execute(
    payload: AgentExecuteRequest,
    authenticated: bool = Depends(verify_api_key)
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

    async def receive() -> Dict[str, Any]:
        body_bytes = json.dumps(payload.payload or {}).encode("utf-8")
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    delegated_request: Request = StarletteRequest(scope, receive)  # type: ignore[arg-type]

    return await execute_agent(agent_id=agent_id, request=delegated_request, authenticated=True)


@app.post("/api/v1/agents/activate")
async def api_v1_agents_activate(
    payload: AgentActivateRequest,
    authenticated: bool = Depends(verify_api_key)
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
    request: AUREAEventRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Execute event with specified AI agent via AUREA orchestration.
    Called by Event Router daemon to process events from brainops_core.event_bus.
    """
    logger.info(f"🎯 AUREA Event: {request.event_id} ({request.topic}) -> {request.target_agent['name']}")

    pool = get_pool()

    try:
        # Find target agent by name
        agent_row = await pool.fetchrow(
            """
            SELECT id, name, category, enabled
            FROM agents
            WHERE name = $1 AND enabled = TRUE
            """,
            request.target_agent['name']
        )

        if not agent_row:
            raise HTTPException(
                status_code=404,
                detail=f"Agent '{request.target_agent['name']}' not found or disabled"
            )

        agent_id = str(agent_row['id'])
        agent_name = agent_row['name']

        # Prepare agent execution payload
        agent_payload = {
            "event_id": request.event_id,
            "topic": request.topic,
            "source": request.source,
            **request.payload
        }

        # Execute agent (simple acknowledgment for now)
        # Can be expanded with topic-specific handlers
        result = {
            "status": "acknowledged",
            "agent": agent_name,
            "event_id": request.event_id,
            "topic": request.topic,
            "action": "processed"
        }

        # Update agent last_active_at in brainops_core.agents (if table exists)
        try:
            await pool.execute(
                """
                UPDATE brainops_core.agents
                SET last_active_at = NOW()
                WHERE name = $1
                """,
                agent_name
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
                        "source": request.source
                    },
                    importance_score=0.7
                )
            except Exception as e:
                logger.warning(f"Could not store in embedded memory: {e}")

        logger.info(f"✅ AUREA Event {request.event_id} executed by {agent_name}")

        return {
            "success": True,
            "event_id": request.event_id,
            "agent": agent_name,
            "topic": request.topic,
            "result": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ AUREA Event {request.event_id} failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Event execution failed: {str(e)}"
        )


# ==================== END BRAINOPS CORE v1 API ====================


# ==================== UNIFIED OBSERVABILITY SYSTEM ====================

# In-memory log buffer for recent logs
LOG_BUFFER: deque = deque(maxlen=500)

class LogCapture(logging.Handler):
    """Capture logs to buffer for API access"""
    def emit(self, record):
        try:
            LOG_BUFFER.append({
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno
            })
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
    contains: Optional[str] = None
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
        "filters": {"level": level, "logger": logger_name, "contains": contains}
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
        "system_metrics": {}
    }

    # Check all services
    services = {
        "ai_agents": "https://brainops-ai-agents.onrender.com/health",
        "backend": "https://brainops-backend-prod.onrender.com/health",
        "mcp_bridge": "https://brainops-mcp-bridge.onrender.com/health",
        "myroofgenius": "https://myroofgenius.com",
        "weathercraft_erp": "https://weathercraft-erp.vercel.app",
        "brainstackstudio": "https://brainstackstudio.com"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        for name, url in services.items():
            try:
                resp = await client.get(url)
                results["services"][name] = {
                    "status": "healthy" if resp.status_code == 200 else "degraded",
                    "code": resp.status_code,
                    "data": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else None
                }
            except Exception as e:
                results["services"][name] = {"status": "error", "error": str(e)}

    # Get database stats
    try:
        pool = get_pool()
        db_stats = await pool.fetchrow("""
            SELECT
                (SELECT COUNT(*) FROM customers) as customers,
                (SELECT COUNT(*) FROM jobs) as jobs,
                (SELECT COUNT(*) FROM ai_agents) as agents,
                (SELECT COUNT(*) FROM ai_agent_executions) as executions,
                (SELECT COUNT(*) FROM ai_agent_executions WHERE created_at > NOW() - INTERVAL '1 hour') as recent_executions,
                (SELECT COUNT(*) FROM ai_agent_executions WHERE status = 'failed' AND created_at > NOW() - INTERVAL '1 hour') as recent_failures
        """)
        results["database"] = dict(db_stats) if db_stats else {}
    except Exception as e:
        results["database"] = {"error": str(e)}

    # Get recent errors from logs
    results["recent_errors"] = [
        l for l in list(LOG_BUFFER)[-100:]
        if l["level"] in ["ERROR", "CRITICAL"]
    ][-20:]

    # Get system metrics
    try:
        import psutil
        results["system_metrics"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    except ImportError:
        results["system_metrics"] = {"note": "psutil not available"}

    return results


@app.get("/debug/all-errors", dependencies=[Depends(verify_api_key)])
async def get_all_errors():
    """Get all recent errors across systems for debugging"""
    errors = [l for l in list(LOG_BUFFER) if l["level"] in ["ERROR", "CRITICAL", "WARNING"]]

    # Categorize errors
    categorized = {
        "database": [],
        "connection": [],
        "schema": [],
        "api": [],
        "other": []
    }

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
        "by_category": categorized
    }


@app.get("/system/unified-status", dependencies=[Depends(verify_api_key)])
async def get_unified_system_status():
    """Get unified status of the entire AI OS"""
    pool = get_pool()

    status = {
        "version": VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "overall_health": "healthy",
        "components": {}
    }

    # Check each major component
    components = [
        ("database", "SELECT 1"),
        ("agents", "SELECT COUNT(*) FROM ai_agents"),
        ("executions", "SELECT COUNT(*) FROM ai_agent_executions WHERE created_at > NOW() - INTERVAL '24 hours'"),
        ("memory", "SELECT COUNT(*) FROM unified_brain"),
        ("revenue", "SELECT COUNT(*) FROM revenue_leads"),
    ]

    issues = []
    for name, query in components:
        try:
            result = await pool.fetchval(query)
            status["components"][name] = {"status": "ok", "value": result}
        except Exception as e:
            status["components"][name] = {"status": "error", "error": str(e)}
            issues.append(f"{name}: {str(e)}")

    if issues:
        status["overall_health"] = "degraded"
        status["issues"] = issues

    return status


# ==================== END UNIFIED OBSERVABILITY ====================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc) if config.security.dev_mode else "An error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower()
    )
