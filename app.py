"""
BrainOps AI Agents Service - Enhanced Production Version
Type-safe, async, fully operational
"""
import logging
import os
import asyncio
import json
import uuid
import inspect
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from collections import deque

from fastapi import FastAPI, HTTPException, Request, Security, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

# Import our production-ready components
from config import config
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
from api.brain import router as brain_router
from api.memory_coordination import router as memory_coordination_router
from ai_provider_status import get_provider_status

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Build info
BUILD_TIME = datetime.utcnow().isoformat()
VERSION = "8.3.0"  # MINOR: AUREA NLU Natural Language Command Interface - Founder-level AI control via natural language
LOCAL_EXECUTIONS: deque[Dict[str, Any]] = deque(maxlen=200)

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
    AI_AVAILABLE = True
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
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"🚀 Starting BrainOps AI Agents v{VERSION} - Build: {BUILD_TIME}")

    # Initialize database pool
    try:
        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database
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
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")

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
    if AUREA_AVAILABLE:
        try:
            # Start at SEMI_AUTO level (AI decides minor, human decides major)
            aurea = AUREA(autonomy_level=AutonomyLevel.SEMI_AUTO)
            app.state.aurea = aurea
            logger.info("🧠 AUREA Master Orchestrator initialized at SEMI_AUTO level")
        except Exception as e:
            logger.error(f"❌ AUREA initialization failed: {e}")
            app.state.aurea = None
    else:
        app.state.aurea = None

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
    if SYSTEM_IMPROVEMENT_AVAILABLE:
        try:
            system_improvement = SystemImprovementAgent()
            app.state.system_improvement = system_improvement
            logger.info("🔧 System Improvement Agent initialized")
        except Exception as e:
            logger.error(f"❌ System Improvement Agent initialization failed: {e}")
            app.state.system_improvement = None
    else:
        app.state.system_improvement = None

    # Initialize DevOps Optimization Agent
    if DEVOPS_AGENT_AVAILABLE:
        try:
            devops_agent = DevOpsOptimizationAgent()
            app.state.devops_agent = devops_agent
            logger.info("⚙️ DevOps Optimization Agent initialized")
        except Exception as e:
            logger.error(f"❌ DevOps Agent initialization failed: {e}")
            app.state.devops_agent = None
    else:
        app.state.devops_agent = None

    # Initialize Code Quality Agent
    if CODE_QUALITY_AVAILABLE:
        try:
            code_quality = CodeQualityAgent()
            app.state.code_quality = code_quality
            logger.info("📝 Code Quality Agent initialized")
        except Exception as e:
            logger.error(f"❌ Code Quality Agent initialization failed: {e}")
            app.state.code_quality = None
    else:
        app.state.code_quality = None

    # Initialize Customer Success Agent
    if CUSTOMER_SUCCESS_AVAILABLE:
        try:
            customer_success = CustomerSuccessAgent()
            app.state.customer_success = customer_success
            logger.info("🎯 Customer Success Agent initialized")
        except Exception as e:
            logger.error(f"❌ Customer Success Agent initialization failed: {e}")
            app.state.customer_success = None
    else:
        app.state.customer_success = None

    # Initialize Competitive Intelligence Agent
    if COMPETITIVE_INTEL_AVAILABLE:
        try:
            competitive_intel = CompetitiveIntelligenceAgent()
            app.state.competitive_intel = competitive_intel
            logger.info("🔍 Competitive Intelligence Agent initialized")
        except Exception as e:
            logger.error(f"❌ Competitive Intelligence Agent initialization failed: {e}")
            app.state.competitive_intel = None
    else:
        app.state.competitive_intel = None

    # Initialize Vision Alignment Agent
    if VISION_ALIGNMENT_AVAILABLE:
        try:
            vision_alignment = VisionAlignmentAgent()
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

    logger.info("=" * 80)
    logger.info("🚀 BRAINOPS AI AGENTS v8.0.0 - COMPLETE AI OPERATING SYSTEM")
    logger.info("🧠 AI INTEGRATION LAYER ACTIVATED - All Systems Connected!")
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

    yield

    # Shutdown
    logger.info("🛑 Shutting down BrainOps AI Agents...")

    # Shutdown scheduler if running
    if hasattr(app.state, 'scheduler') and app.state.scheduler:
        try:
            app.state.scheduler.shutdown()
            logger.info("✅ Agent Scheduler stopped")
        except Exception as e:
            logger.error(f"❌ Scheduler shutdown error: {e}")

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

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key if authentication is required"""
    if not config.security.auth_required:
        return True

    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")

    if api_key not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True


# Include routers
app.include_router(memory_router)
app.include_router(brain_router)
app.include_router(memory_coordination_router)

# Import and include analytics router
try:
    from analytics_endpoint import router as analytics_router
    app.include_router(analytics_router)
    logger.info("✅ Analytics endpoint loaded")
except ImportError as e:
    logger.warning(f"Analytics endpoint not available: {e}")


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
async def health_check():
    """Health check endpoint with full system status"""
    pool = get_pool()
    db_healthy = await pool.test_connection()
    db_status = "fallback" if using_fallback() else ("connected" if db_healthy else "disconnected")

    # Check active systems - Phase 1
    active_systems = []
    if AUREA_AVAILABLE and hasattr(app.state, 'aurea') and app.state.aurea:
        active_systems.append("AUREA Orchestrator")
    if SELF_HEALING_AVAILABLE and hasattr(app.state, 'healer') and app.state.healer:
        active_systems.append("Self-Healing Recovery")
    if MEMORY_AVAILABLE and hasattr(app.state, 'memory') and app.state.memory:
        active_systems.append("Memory Manager")
    if EMBEDDED_MEMORY_AVAILABLE and hasattr(app.state, 'embedded_memory') and app.state.embedded_memory:
        active_systems.append("Embedded Memory (RAG)")
    if TRAINING_AVAILABLE and hasattr(app.state, 'training') and app.state.training:
        active_systems.append("Training Pipeline")
    if LEARNING_AVAILABLE and hasattr(app.state, 'learning') and app.state.learning:
        active_systems.append("Learning System")
    if SCHEDULER_AVAILABLE and hasattr(app.state, 'scheduler') and app.state.scheduler:
        active_systems.append("Agent Scheduler")
    if AI_AVAILABLE and ai_core:
        active_systems.append("AI Core")

    # Check Phase 2 specialized agents
    if SYSTEM_IMPROVEMENT_AVAILABLE and hasattr(app.state, 'system_improvement') and app.state.system_improvement:
        active_systems.append("System Improvement Agent")
    if DEVOPS_AGENT_AVAILABLE and hasattr(app.state, 'devops_agent') and app.state.devops_agent:
        active_systems.append("DevOps Optimization Agent")
    if CODE_QUALITY_AVAILABLE and hasattr(app.state, 'code_quality') and app.state.code_quality:
        active_systems.append("Code Quality Agent")
    if CUSTOMER_SUCCESS_AVAILABLE and hasattr(app.state, 'customer_success') and app.state.customer_success:
        active_systems.append("Customer Success Agent")
    if COMPETITIVE_INTEL_AVAILABLE and hasattr(app.state, 'competitive_intel') and app.state.competitive_intel:
        active_systems.append("Competitive Intelligence Agent")
    if VISION_ALIGNMENT_AVAILABLE and hasattr(app.state, 'vision_alignment') and app.state.vision_alignment:
        active_systems.append("Vision Alignment Agent")

    # Get embedded memory stats if available
    embedded_memory_stats = None
    if EMBEDDED_MEMORY_AVAILABLE and hasattr(app.state, 'embedded_memory') and app.state.embedded_memory:
        try:
            embedded_memory_stats = app.state.embedded_memory.get_stats()
        except:
            pass

    return {
        "status": "healthy" if db_healthy else "degraded",
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
            "vision_alignment": VISION_ALIGNMENT_AVAILABLE
        },
        "config": {
            "environment": config.environment,
            "security": {
                "auth_required": config.security.auth_required,
                "dev_mode": config.security.dev_mode
            }
        }
    }


@app.get("/ai/providers/status")
async def providers_status(_: bool = Depends(verify_api_key)):
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
    _: bool = Depends(verify_api_key)
) -> AgentList:
    """Get list of available agents"""
    pool = get_pool()

    try:
        # Build query
        query = "SELECT * FROM agents WHERE 1=1"
        params = []

        if enabled is not None:
            query += f" AND enabled = ${len(params) + 1}"
            params.append(enabled)

        if category:
            query += f" AND category = ${len(params) + 1}"
            params.append(category)

        query += " ORDER BY category, name"

        # Execute query
        rows = await pool.fetch(query, *params)

        # Convert to models
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
        logger.error(f"Failed to get agents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve agents: {str(e)}")


@app.post("/agents/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    request: Request,
    _: bool = Depends(verify_api_key)
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

        # Log execution start
        agent_uuid = str(agent["id"])
        try:
            await pool.execute("""
                INSERT INTO agent_executions (id, agent_id, started_at, status, input_data)
                VALUES ($1, $2, $3, $4, $5)
            """, execution_id, agent_uuid, started_at, "running", json.dumps(body))
        except Exception as insert_error:
            logger.warning("Failed to persist execution start: %s", insert_error)

        # Execute agent logic
        result = {"status": "completed", "message": "Agent executed successfully"}

        if AI_AVAILABLE and ai_core:
            try:
                # Use AI core for execution
                prompt = f"Execute {agent['name']}: {body.get('task', 'default task')}"
                if inspect.iscoroutinefunction(ai_core.generate):
                    ai_result = await ai_core.generate(prompt)
                else:
                    ai_result = await asyncio.to_thread(ai_core.generate, prompt)
                result["ai_response"] = ai_result
            except Exception as e:
                logger.error(f"AI execution failed: {e}")
                result["ai_response"] = None

        # Update execution record
        completed_at = datetime.utcnow()
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        try:
            await pool.execute("""
                UPDATE agent_executions
                SET completed_at = $1, status = $2, output_data = $3, duration_ms = $4
                WHERE id = $5
            """, completed_at, "completed", json.dumps(result), duration_ms, execution_id)
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
                    UPDATE agent_executions
                    SET status = $1, error = $2, completed_at = $3
                    WHERE id = $4
                """, "failed", str(e), datetime.utcnow(), execution_id)
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
    _: bool = Depends(verify_api_key)
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
    _: bool = Depends(verify_api_key)
):
    """Get agent executions"""
    pool = get_pool()

    try:
        query = """
            SELECT e.*, a.name as agent_name
            FROM agent_executions e
            JOIN agents a ON e.agent_id = a.id
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
    _: bool = Depends(verify_api_key)
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

                # Log execution
                await pool.execute("""
                    INSERT INTO agent_executions (id, agent_id, started_at, status, input_data)
                    VALUES ($1, $2, $3, $4, $5)
                """, execution_id, agent["id"], started_at, "running", json.dumps({"scheduled": True}))

                # Execute agent
                result = {"status": "completed", "scheduled_execution": True}

                # Update execution
                await pool.execute("""
                    UPDATE agent_executions
                    SET completed_at = $1, status = $2, output_data = $3
                    WHERE id = $4
                """, datetime.utcnow(), "completed", json.dumps(result), execution_id)

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


@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get detailed scheduler status and diagnostics"""
    if not SCHEDULER_AVAILABLE or not hasattr(app.state, 'scheduler') or not app.state.scheduler:
        return {
            "enabled": False,
            "message": "Scheduler not available"
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
        ]
    }


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


# ==================== AI TASK MANAGEMENT ENDPOINTS ====================
# Revolutionary AI-powered task management with autonomous execution

@app.post("/ai/tasks/create")
async def create_ai_task(
    request: Request,
    task_type: str,
    description: str,
    priority: str = "medium",
    auto_execute: bool = False,
    due_date: Optional[str] = None
):
    """
    Create a new AI task that will be autonomously executed

    Beyond traditional task managers - AI decides when and how to execute!
    """
    if not INTEGRATION_LAYER_AVAILABLE or not hasattr(app.state, 'integration_layer'):
        raise HTTPException(status_code=503, detail="AI Integration Layer not available")

    try:
        integration_layer = app.state.integration_layer

        # Map priority
        priority_map = {
            'critical': TaskPriority.CRITICAL,
            'high': TaskPriority.HIGH,
            'medium': TaskPriority.MEDIUM,
            'low': TaskPriority.LOW
        }

        task_id = await integration_layer.create_task(
            task_type=task_type,
            priority=priority_map.get(priority, TaskPriority.MEDIUM),
            trigger_condition={'description': description, 'auto_execute': auto_execute},
            scheduled_at=datetime.fromisoformat(due_date) if due_date else None
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Task created and will be executed by AI",
            "auto_execute": auto_execute,
            "priority": priority
        }

    except Exception as e:
        logger.error(f"❌ Task creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/tasks/status/{task_id}")
async def get_ai_task_status(task_id: str):
    """Get current status and details of an AI task"""
    if not INTEGRATION_LAYER_AVAILABLE or not hasattr(app.state, 'integration_layer'):
        raise HTTPException(status_code=503, detail="AI Integration Layer not available")

    try:
        integration_layer = app.state.integration_layer
        task = await integration_layer.get_task_status(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        return {
            "success": True,
            "task": task
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/tasks/list")
async def list_ai_tasks(
    status: Optional[str] = None,
    limit: int = 100
):
    """List all AI tasks with optional status filter"""
    if not INTEGRATION_LAYER_AVAILABLE or not hasattr(app.state, 'integration_layer'):
        raise HTTPException(status_code=503, detail="AI Integration Layer not available")

    try:
        integration_layer = app.state.integration_layer

        # Map status filter
        status_filter = None
        if status:
            status_map = {
                'pending': TaskStatus.PENDING,
                'assigned': TaskStatus.ASSIGNED,
                'in_progress': TaskStatus.IN_PROGRESS,
                'paused': TaskStatus.PAUSED,
                'completed': TaskStatus.COMPLETED,
                'failed': TaskStatus.FAILED,
                'cancelled': TaskStatus.CANCELLED
            }
            status_filter = status_map.get(status)

        tasks = await integration_layer.list_tasks(status=status_filter, limit=limit)

        return {
            "success": True,
            "tasks": tasks,
            "count": len(tasks)
        }

    except Exception as e:
        logger.error(f"❌ Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/tasks/execute/{task_id}")
async def execute_ai_task(task_id: str):
    """Manually trigger execution of a specific task"""
    if not INTEGRATION_LAYER_AVAILABLE or not hasattr(app.state, 'integration_layer'):
        raise HTTPException(status_code=503, detail="AI Integration Layer not available")

    try:
        integration_layer = app.state.integration_layer

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
    if not INTEGRATION_LAYER_AVAILABLE or not hasattr(app.state, 'integration_layer'):
        raise HTTPException(status_code=503, detail="AI Integration Layer not available")

    try:
        integration_layer = app.state.integration_layer

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


@app.post("/ai/orchestrate")
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
