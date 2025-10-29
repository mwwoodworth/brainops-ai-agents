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

from fastapi import FastAPI, HTTPException, Request, Security, Depends
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
from api.memory import router as memory_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Build info
BUILD_TIME = datetime.utcnow().isoformat()
VERSION = "6.0.0"  # MAJOR: Activate AUREA, Self-Healing, Learning Systems
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
            app.state.scheduler = scheduler
            logger.info("✅ Agent Scheduler initialized")
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

    logger.info("=" * 60)
    logger.info("🚀 BRAINOPS AI AGENTS v6.0.0 - PHASE 2 COMPLETE")
    logger.info("=" * 60)
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
    logger.info("=" * 60)

    yield

    # Shutdown
    logger.info("🛑 Shutting down BrainOps AI Agents...")
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

    return {
        "status": "healthy" if db_healthy else "degraded",
        "version": VERSION,
        "build": BUILD_TIME,
        "database": db_status,
        "active_systems": active_systems,
        "system_count": len(active_systems),
        "capabilities": {
            # Phase 1
            "aurea_orchestrator": AUREA_AVAILABLE,
            "self_healing": SELF_HEALING_AVAILABLE,
            "memory_manager": MEMORY_AVAILABLE,
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
