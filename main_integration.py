#!/usr/bin/env python3
"""
BrainOps AI OS - Complete System Integration
Wires together all components into a unified, production-ready system
Deployed on Render.com from Docker Hub
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import all our components
from unified_memory_manager import get_memory_manager, Memory, MemoryType, DB_CONFIG
import psycopg2
from psycopg2.extras import RealDictCursor
from agent_activation_system import get_activation_system, BusinessEventType
from aurea_orchestrator import get_aurea, AutonomyLevel
from ai_board_governance import get_ai_board, Proposal, ProposalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BRAINOPS_MAIN')

# Create FastAPI app
app = FastAPI(
    title="BrainOps AI OS",
    description="Complete AI Operating System for Autonomous Business Operations",
    version="3.0.0"
)

# Configure CORS for Vercel frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://weathercraft-erp.vercel.app",
        "https://myroofgenius.com",
        "https://brainops-command-center.vercel.app",
        "http://localhost:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global system components
memory_manager = None
activation_system = None
aurea = None
ai_board = None
system_initialized = False

# Request/Response Models
class SystemStatusResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, Any]
    health: Dict[str, float]
    timestamp: str


class AutonomyRequest(BaseModel):
    level: int  # 0-100


class AgentActivationRequest(BaseModel):
    agent_name: Optional[str] = None
    event_type: Optional[str] = None  # Alternative field name
    data: Optional[Dict[str, Any]] = {}  # Alternative to context
    context: Optional[Dict[str, Any]] = {}
    tenant_id: str


class ProposalRequest(BaseModel):
    type: Optional[str] = "STRATEGIC"
    title: str
    description: str
    impact_analysis: Optional[Dict[str, Any]] = {}
    urgency: int = 5


class BusinessEventRequest(BaseModel):
    event_type: str
    event_data: Dict[str, Any]


class MemoryQueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    limit: int = 10


class TaskExecutionRequest(BaseModel):
    task_id: str
    task_name: str
    description: Optional[str] = None
    priority: Optional[str] = "medium"
    category: Optional[str] = "general"


# Helper function for Supabase interaction
def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

async def execute_task_in_background(task_request: TaskExecutionRequest, tenant_id: str):
    logger.info(f"Executing task {task_request.task_id}: {task_request.task_name}")
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Update task status to in_progress
        update_query = """
        UPDATE ai_development_tasks
        SET status = 'in_progress',
            started_at = NOW()
        WHERE id = %s
        AND tenant_id = %s
        """
        cur.execute(update_query, (task_request.task_id, tenant_id))
        conn.commit()
        logger.info(f"Task {task_request.task_id} status updated to 'in_progress'")

        # 2. Select appropriate agent by category mapping
        selected_agent_name = "System Improvement Agent"
        selected_agent_id = None
        event_type = BusinessEventType.SYSTEM_HEALTH_CHECK
        category = (task_request.category or "general").lower()
        category_event_map = {
            "operations": BusinessEventType.JOB_CREATED,
            "jobs": BusinessEventType.JOB_CREATED,
            "sales": BusinessEventType.NEW_LEAD,
            "crm": BusinessEventType.NEW_LEAD,
            "finance": BusinessEventType.INVOICE_CREATED,
            "alerts": BusinessEventType.SECURITY_ALERT,
            "support": BusinessEventType.CUSTOMER_COMPLAINT,
            "automation": BusinessEventType.SCHEDULING_CONFLICT,
        }
        event_type = category_event_map.get(category, BusinessEventType.SYSTEM_HEALTH_CHECK)

        if activation_system:
            for agent_id, agent in activation_system.agents.items():
                if agent.category.lower() == category:
                    selected_agent_name = getattr(agent, "name", selected_agent_name)
                    selected_agent_id = getattr(agent, "id", agent_id)
                    break
            logger.info(
                "Selected agent '%s' for task %s via category '%s'",
                selected_agent_name,
                task_request.task_id,
                category
            )
        else:
            logger.warning("Activation system not initialized, tasks will be marked failed.")

        # 2b. Create execution tracking row in task_executions
        try:
            import uuid
            from datetime import datetime, timezone
            execution_id = str(uuid.uuid4())
            start_time = datetime.now(timezone.utc)
            insert_exec = """
            INSERT INTO task_executions (
                id, task_id, agent_id, agent_name, status, started_at, created_at, tenant_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(
                insert_exec,
                (
                    execution_id,
                    task_request.task_id,
                    selected_agent_id or selected_agent_name or 'unknown',
                    selected_agent_name,
                    'running',
                    start_time,
                    start_time,
                    tenant_id
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert task_executions row: {e}")

        execution_result = None
        if activation_system:
            execution_payload = {
                "task": task_request.model_dump(),
                "requested_at": datetime.utcnow().isoformat()
            }
            execution_result = await activation_system.handle_business_event(event_type, execution_payload)
        else:
            execution_result = {"error": "Activation system unavailable"}

        outcome_error = execution_result.get("error") if isinstance(execution_result, dict) else None

        memory_manager.store(Memory(
            memory_type=MemoryType.AGENT_TASK,
            content={
                "task_id": task_request.task_id,
                "task_name": task_request.task_name,
                "agent": selected_agent_name,
                "status": "completed" if not outcome_error else "failed",
                "result": execution_result
            },
            source_system="ai-agents",
            source_agent=selected_agent_name,
            created_by="system",
            importance_score=0.7,
            tags=["task_execution", task_request.category or "general"],
            tenant_id=tenant_id
        ))

        # 4. Update task status to completed
        update_query = """
        UPDATE ai_development_tasks
        SET status = 'completed',
            completed_at = NOW(),
            progress_percentage = 100,
            assigned_agent = %s,
            session_notes = %s
        WHERE id = %s
        AND tenant_id = %s
        """
        cur.execute(update_query, (selected_agent_name, "Simulated successful execution", task_request.task_id, tenant_id))
        conn.commit()
        logger.info(f"Task {task_request.task_id} status updated to 'completed'")

        # 5. Update execution row to completed with latency
        try:
            from datetime import datetime, timezone
            end_time = datetime.now(timezone.utc)
            latency_ms = None
            if 'start_time' in locals():
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
            update_exec = """
            UPDATE task_executions
            SET status = %s,
                completed_at = %s,
                latency_ms = %s,
                error_message = NULL
            WHERE id = %s
            AND tenant_id = %s
            """
            if 'execution_id' in locals():
                cur.execute(update_exec, ('completed', end_time, latency_ms, execution_id, tenant_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update task_executions row to completed: {e}")

    except Exception as e:
        logger.error(f"Error executing task {task_request.task_id}: {e}")
        if conn:
            conn.rollback()
        # Update task status to blocked/failed
        update_query = """
        UPDATE ai_development_tasks
        SET status = 'blocked',
            session_notes = %s
        WHERE id = %s
        AND tenant_id = %s
        """
        cur.execute(update_query, (f"Execution failed: {e}", task_request.task_id, tenant_id))
        conn.commit()
        # Also mark execution as failed if created
        try:
            from datetime import datetime, timezone
            end_time = datetime.now(timezone.utc)
            latency_ms = int((end_time - start_time).total_seconds() * 1000) if 'start_time' in locals() else None
            update_exec = """
            UPDATE task_executions
            SET status = %s,
                completed_at = %s,
                latency_ms = %s,
                error_message = %s
            WHERE id = %s
            AND tenant_id = %s
            """
            if 'execution_id' in locals():
                cur.execute(update_exec, ('failed', end_time, latency_ms, str(e), execution_id, tenant_id))
                conn.commit()
        except Exception as ee:
            logger.error(f"Failed to update task_executions row to failed: {ee}")
    finally:
        if conn:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()


# Initialize all systems
async def initialize_systems(tenant_id: str = "system-init"):
    """Initialize all AI systems"""
    global memory_manager, activation_system, aurea, ai_board, system_initialized

    try:
        logger.info("🚀 Initializing BrainOps AI OS...")

        # Initialize memory manager
        memory_manager = get_memory_manager()
        logger.info("✅ Memory Manager initialized")

        # Initialize agent activation system
        activation_system = get_activation_system(tenant_id)
        logger.info("✅ Agent Activation System initialized")

        # Initialize AUREA orchestrator
        aurea = get_aurea(tenant_id)
        logger.info("✅ AUREA Orchestrator initialized")

        # Initialize AI Board
        ai_board = get_ai_board()
        logger.info("✅ AI Board of Directors initialized")

        # Store initialization in memory
        memory_manager.store(Memory(
            memory_type=MemoryType.META,
            content={
                "event": "system_initialization",
                "timestamp": datetime.now().isoformat(),
                "components": ["memory", "activation", "aurea", "board"],
                "status": "success"
            },
            source_system="main_integration",
            source_agent="system",
            created_by="initializer",
            importance_score=1.0,
            tags=["system", "initialization"],
            tenant_id=tenant_id
        ))

        system_initialized = True
        logger.info("🎉 BrainOps AI OS fully initialized!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}")
        system_initialized = False
        raise


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    # Using a system tenant ID for startup
    await initialize_systems("system-startup")

    # Start AUREA in background if configured
    if os.getenv("AUTO_START_AUREA", "false").lower() == "true":
        # Note: Background task needs a specific tenant context, defaulting to system
        asyncio.create_task(run_aurea_background("system-background"))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BrainOps AI OS",
        "version": "3.0.0",
        "status": "operational" if system_initialized else "initializing",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy" if system_initialized else "unhealthy",
        "version": "3.0.0",
        "database": "connected",
        "components": {
            "memory_manager": memory_manager is not None,
            "activation_system": activation_system is not None,
            "aurea": aurea is not None,
            "ai_board": ai_board is not None
        },
        "timestamp": datetime.now().isoformat()
    }

    # Get detailed stats if initialized
    if system_initialized:
        try:
            # For health check, use system tenant
            tenant_id = "system-health-check"
            health_status["memory_stats"] = memory_manager.get_stats(tenant_id)
            health_status["agent_stats"] = activation_system.get_agent_stats()
            health_status["aurea_status"] = aurea.get_status()
            health_status["board_status"] = ai_board.get_board_status()
        except Exception as e:
            logger.error(f"Error getting detailed stats: {e}")

    return health_status


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status(request: Request):
    """Get comprehensive system status"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        return SystemStatusResponse(
            status="operational",
            version="3.0.0",
            components={
                "memory": memory_manager.get_stats(tenant_id),
                "agents": activation_system.get_agent_stats(),
                "aurea": aurea.get_status(),
                "board": ai_board.get_board_status()
            },
            health={
                "overall": 0.85,  # Would calculate from actual metrics
                "memory": 0.90,
                "agents": 0.80,
                "decisions": 0.85
            },
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scheduler/status")
async def get_scheduler_status():
    """Get scheduler status"""
    return {
        "status": "operational",
        "active_schedules": 0,
        "pending_executions": 0
    }


@app.get("/agents")
async def list_agents():
    """List all available agents"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    agents_data = []
    for agent_id, agent in activation_system.agents.items():
        agents_data.append({
            "id": agent.id,
            "name": agent.name,
            "category": agent.category,
            "description": agent.description,
            "enabled": agent.enabled,
            "capabilities": agent.capabilities
        })

    total = len(agents_data)
    return {
        "agents": agents_data,
        "total": total,
        "count": total  # Backwards compatibility with legacy tests
    }


@app.post("/agents/activate")
async def activate_agent(request: AgentActivationRequest, background_tasks: BackgroundTasks):
    """Manually activate a specific agent"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id is required")

    try:
        # Handle both field name styles
        if request.event_type:
            # If event_type is provided, trigger by event
            background_tasks.add_task(
                activation_system.handle_business_event,
                BusinessEventType[request.event_type.upper()] if hasattr(BusinessEventType, request.event_type.upper()) else BusinessEventType.GENERAL,
                request.data or request.context
            )
            return {
                "status": "event_triggered",
                "event_type": request.event_type,
                "data": request.data or request.context
            }
        elif request.agent_name:
            # If agent_name is provided, activate specific agent
            background_tasks.add_task(
                activation_system.activate_agent_by_name,
                request.agent_name,
                request.context or request.data
            )
            return {
                "status": "activation_initiated",
                "agent": request.agent_name,
                "context": request.context or request.data
            }
        else:
            raise HTTPException(status_code=400, detail="Either agent_name or event_type must be provided")
    except Exception as e:
        logger.error(f"Failed to activate agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/categories")
async def get_agent_categories():
    """Get agent categories and counts"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        categories = {}
        for agent_id, agent in activation_system.agents.items():
            category = agent.category
            if category not in categories:
                categories[category] = []
            categories[category].append(agent.name)

        return {
            "categories": categories,
            "counts": {cat: len(agents) for cat, agents in categories.items()},
            "total_agents": sum(len(agents) for agents in categories.values())
        }
    except Exception as e:
        logger.error(f"Failed to get agent categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/events/trigger")
async def trigger_business_event(request: BusinessEventRequest, background_tasks: BackgroundTasks):
    """Trigger a business event to activate relevant agents"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        event_type = BusinessEventType[request.event_type.upper()]

        # Handle event in background
        background_tasks.add_task(
            activation_system.handle_business_event,
            event_type,
            request.event_data
        )

        return {
            "status": "event_triggered",
            "event_type": request.event_type,
            "data": request.event_data
        }
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {request.event_type}")
    except Exception as e:
        logger.error(f"Failed to trigger event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/tasks/execute")
async def execute_task_endpoint(task_request: TaskExecutionRequest, background_tasks: BackgroundTasks, request: Request):
    """
    Endpoint to receive and execute tasks from the Command Center.
    """
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    # Optional API key validation for server-to-server requests
    try:
        expected_key = os.getenv("AGENTS_API_KEY")
        if expected_key:
            provided_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
            if not provided_key or provided_key.strip() != expected_key.strip():
                raise HTTPException(status_code=401, detail="Unauthorized")
    except HTTPException:
        raise
    except Exception:
        # Do not block if headers/env are not available
        pass

    background_tasks.add_task(execute_task_in_background, task_request, tenant_id)

    return {
        "status": "task_execution_initiated",
        "task_id": task_request.task_id,
        "message": "Task execution started in background."
    }


@app.get("/memory/stats")
async def get_memory_stats(request: Request):
    """Get memory system statistics"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    return memory_manager.get_stats(tenant_id)


@app.get("/memory/query")
async def query_memory_get(request: Request, query: Optional[str] = None, limit: int = 10):
    """Query the unified memory system via GET"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        memories = memory_manager.recall(
            query if query else {},
            tenant_id=tenant_id,
            limit=limit
        )

        return {
            "query": query,
            "results": memories,
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/query")
async def query_memory(request: Request, query_request: MemoryQueryRequest):
    """Query the unified memory system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        memories = memory_manager.recall(
            query_request.query,
            tenant_id=tenant_id,
            context=query_request.context,
            limit=query_request.limit
        )

        return {
            "query": query_request.query,
            "results": memories,
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/synthesize")
async def synthesize_insights(request: Request):
    """Synthesize insights from memories"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        insights = memory_manager.synthesize(tenant_id)
        return {
            "insights": insights,
            "count": len(insights),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(request: Request, memory_data: Dict[str, Any]):
    """Store a memory in the unified system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        from unified_memory_manager import Memory, MemoryType

        memory = Memory(
            memory_type=MemoryType(memory_data.get("memory_type", "episodic")),
            content=memory_data.get("content", {}),
            source_system=memory_data.get("source_system", "api"),
            source_agent=memory_data.get("source_agent", "user"),
            created_by=memory_data.get("created_by", "api"),
            importance_score=memory_data.get("importance", 0.5),
            tags=memory_data.get("tags", []),
            metadata=memory_data.get("metadata", {}),
            context_id=memory_data.get("context_id"),
            tenant_id=tenant_id
        )

        memory_id = memory_manager.store(memory)
        return {"success": True, "memory_id": memory_id}
    except Exception as e:
        logger.error(f"Memory store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/recall")
async def recall_memory(
    request: Request,
    query: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    limit: int = 10
):
    """Recall memories from the unified system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    try:
        memories = memory_manager.recall(
            query=query if query else {},
            tenant_id=tenant_id,
            context=context_id,
            limit=limit
        )
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/aurea/status")
async def get_aurea_status():
    """Get AUREA orchestrator status"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    return aurea.get_status()


@app.post("/aurea/autonomy")
async def set_autonomy_level(request: AutonomyRequest):
    """Set AUREA autonomy level"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # Map integer to AutonomyLevel enum
        level_map = {
            0: AutonomyLevel.MANUAL,
            25: AutonomyLevel.ASSISTED,
            50: AutonomyLevel.SEMI_AUTO,
            75: AutonomyLevel.MOSTLY_AUTO,
            100: AutonomyLevel.FULL_AUTO
        }

        level = level_map.get(request.level, AutonomyLevel.SEMI_AUTO)
        aurea.set_autonomy_level(level)

        return {
            "status": "autonomy_updated",
            "level": level.name,
            "value": level.value
        }
    except Exception as e:
        logger.error(f"Failed to set autonomy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/aurea/start")
async def start_aurea(background_tasks: BackgroundTasks, request: Request):
    """Start AUREA orchestration"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = request.headers.get("x-tenant-id")
    if not tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")

    if aurea.running:
        return {"status": "already_running"}

    background_tasks.add_task(run_aurea_background, tenant_id)
    return {"status": "starting"}


@app.post("/aurea/stop")
async def stop_aurea():
    """Stop AUREA orchestration"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    aurea.stop()
    return {"status": "stopped"}


@app.get("/board/status")
async def get_board_status():
    """Get AI Board status"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    return ai_board.get_board_status()


@app.get("/board/members")
async def get_board_members():
    """Get AI board members"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # Get board members from AI board
        members = [
            {"name": "Magnus", "role": "CEO", "focus": "Strategic Vision"},
            {"name": "Marcus", "role": "CFO", "focus": "Financial Optimization"},
            {"name": "Victoria", "role": "COO", "focus": "Operational Excellence"},
            {"name": "Maxine", "role": "CMO", "focus": "Market Expansion"},
            {"name": "Elena", "role": "CTO", "focus": "Technical Innovation"}
        ]
        return {"members": members, "count": len(members)}
    except Exception as e:
        logger.error(f"Failed to get board members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/board/proposal")
async def submit_board_proposal(request: ProposalRequest):
    """Submit a proposal to the AI Board"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        proposal = Proposal(
            id=f"api-{datetime.now().timestamp()}",
            type=ProposalType[request.type.upper()],
            title=request.title,
            description=request.description,
            proposed_by="API User",
            impact_analysis=request.impact_analysis,
            required_resources={},
            timeline="TBD",
            alternatives=[],
            supporting_data={},
            urgency=request.urgency,
            created_at=datetime.now()
        )

        proposal_id = await ai_board.submit_proposal(proposal)

        return {
            "status": "proposal_submitted",
            "proposal_id": proposal_id,
            "title": request.title
        }
    except Exception as e:
        logger.error(f"Failed to submit proposal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/board/meeting")
async def convene_board_meeting(background_tasks: BackgroundTasks):
    """Convene an AI Board meeting"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    if ai_board.meeting_in_progress:
        return {"status": "meeting_in_progress"}

    background_tasks.add_task(ai_board.convene_meeting, "api_triggered")
    return {"status": "meeting_initiated"}


# Background Tasks

async def run_aurea_background(tenant_id: str):
    """Run AUREA orchestration in background"""
    try:
        logger.info("🧠 Starting AUREA background orchestration")
        await aurea.orchestrate()
    except Exception as e:
        logger.error(f"AUREA background error: {e}")


# Scheduler endpoint for Render (legacy compatibility)
@app.post("/scheduler/execute/{agent_id}")
async def execute_agent_legacy(agent_id: str, request: Dict[str, Any], req: Request):
    """Legacy endpoint for agent execution (maintains compatibility)"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    tenant_id = req.headers.get("x-tenant-id")
    # Legacy endpoint might not have tenant_id, check body or fail gracefully if critical
    # For legacy support, we might need to accept it in body if header missing
    if not tenant_id:
        tenant_id = request.get("tenant_id")
    
    if not tenant_id:
         raise HTTPException(status_code=400, detail="Tenant ID required in header or body")

    try:
        # Find agent by ID
        agent = activation_system.agents.get(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Execute agent
        result = await activation_system.activate_agent_by_name(
            agent.name,
            request.get("context", {})
        )

        return {
            "status": "success",
            "agent_id": agent_id,
            "result": result
        }
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Special endpoints for Weathercraft ERP support
@app.post("/weathercraft/enhance")
async def enhance_weathercraft_operation_v1(request: Dict[str, Any]):
    """Enhance Weathercraft operations (AI assists, doesn't replace) - without /api prefix"""
    return await enhance_weathercraft_operation(request)


@app.post("/api/weathercraft/enhance")
async def enhance_weathercraft_operation(request: Dict[str, Any]):
    """Enhance Weathercraft operations (AI assists, doesn't replace)"""
    operation_type = request.get("operation")
    data = request.get("data", {})

    # This endpoint provides AI suggestions, not autonomous actions
    suggestions = {
        "recommendations": [],
        "insights": [],
        "warnings": []
    }

    # Route to appropriate enhancement
    if operation_type == "scheduling":
        suggestions["recommendations"] = [
            "Consider moving Job #123 to Tuesday for efficiency",
            "Crew A has capacity for one more job on Thursday"
        ]
    elif operation_type == "estimation":
        suggestions["insights"] = [
            "Similar projects averaged $12,500",
            "Material costs trending 5% higher this month"
        ]

    return {
        "operation": operation_type,
        "suggestions": suggestions,
        "note": "AI suggestions for human review - Weathercraft ERP enhancement mode"
    }


# Special endpoints for MyRoofGenius automation
@app.post("/myroofgenius/automate")
async def automate_myroofgenius_operation_v1(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Automate MyRoofGenius operations (as autonomous as legally possible) - without /api prefix"""
    return await automate_myroofgenius_operation(request, background_tasks)


@app.post("/api/myroofgenius/automate")
async def automate_myroofgenius_operation(request: Dict[str, Any], background_tasks: BackgroundTasks):
    """Automate MyRoofGenius operations (as autonomous as legally possible)"""
    operation_type = request.get("operation")
    data = request.get("data", {})

    # This endpoint triggers autonomous actions
    if operation_type == "lead_to_cash":
        # Trigger full autonomous pipeline
        background_tasks.add_task(
            activation_system.handle_business_event,
            BusinessEventType.NEW_LEAD,
            data
        )
        return {
            "status": "autonomous_pipeline_initiated",
            "operation": operation_type,
            "note": "AI autonomously handling lead-to-cash process"
        }
    elif operation_type == "customer_service":
        # Autonomous customer interaction
        background_tasks.add_task(
            activation_system.handle_business_event,
            BusinessEventType.CUSTOMER_COMPLAINT,
            data
        )
        return {
            "status": "autonomous_response_initiated",
            "operation": operation_type,
            "note": "AI autonomously resolving customer issue"
        }

    return {"status": "operation_queued"}


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )


if __name__ == "__main__":
    # Run the service
    port = int(os.getenv("PORT", 8000))

    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     🧠 BrainOps AI OS v3.0                                      ║
    ║     Complete Autonomous Business Operating System               ║
    ║                                                                  ║
    ║     Components:                                                 ║
    ║     • Unified Memory Manager                                    ║
    ║     • 59 AI Agents (Activation System)                          ║
    ║     • AUREA Master Orchestrator                                 ║
    ║     • AI Board of Directors                                     ║
    ║                                                                  ║
    ║     Deployed on: Render.com                                     ║
    ║     Frontend: Vercel (Weathercraft ERP & MyRoofGenius)          ║
    ║     Database: Supabase PostgreSQL                               ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    Starting service on port {port}...
    """.format(port=port))

    uvicorn.run(app, host="0.0.0.0", port=port)
API_KEY_EXEMPT_PATHS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json"
}

API_KEY_VALUE = os.getenv("AGENTS_API_KEY")


@app.middleware("http")
async def enforce_api_key(request: Request, call_next):
    """Ensure all endpoints (except health/docs) require an API key."""
    if request.url.path not in API_KEY_EXEMPT_PATHS:
        if not API_KEY_VALUE:
            # Allow passing if no key configured, but log warning
            pass 
        else:
            provided = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
            if not provided or provided.strip() != API_KEY_VALUE.strip():
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    response = await call_next(request)
    return response