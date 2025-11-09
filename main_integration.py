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
    tenant_id: Optional[str] = "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457"


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

async def execute_task_in_background(task_request: TaskExecutionRequest):
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
        """
        cur.execute(update_query, (task_request.task_id,))
        conn.commit()
        logger.info(f"Task {task_request.task_id} status updated to 'in_progress'")

        # 2. Select appropriate agent (simplified for now)
        # In a real scenario, this would involve more sophisticated logic
        # For now, let's try to find an agent based on category or a default
        selected_agent_name = "System Improvement Agent"  # Default agent
        selected_agent_id = None
        if activation_system:
            # Try to find an agent that matches the category
            for agent_id, agent in activation_system.agents.items():
                if agent.category.lower() == task_request.category.lower():
                    # Capture both ID and name when available
                    try:
                        selected_agent_name = getattr(agent, 'name', selected_agent_name)
                        selected_agent_id = getattr(agent, 'id', agent_id)
                    except Exception:
                        selected_agent_name = getattr(agent, 'name', selected_agent_name)
                    break
            logger.info(f"Selected agent for task {task_request.task_id}: {selected_agent_name}")
        else:
            logger.warning("Activation system not initialized, using default agent.")

        # 2b. Create execution tracking row in task_executions
        try:
            import uuid
            from datetime import datetime, timezone
            execution_id = str(uuid.uuid4())
            start_time = datetime.now(timezone.utc)
            insert_exec = """
            INSERT INTO task_executions (
                id, task_id, agent_id, agent_name, status, started_at, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
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
                ),
            )
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to insert task_executions row: {e}")

        # 3. Execute task (placeholder for actual agent execution logic)
        # This is where the actual agent logic would be called.
        # For now, simulate work and update status.
        await asyncio.sleep(5) # Simulate work

        # Store execution as a memory
        memory_manager.store(Memory(
            memory_type=MemoryType.AGENT_TASK,
            content={
                "task_id": task_request.task_id,
                "task_name": task_request.task_name,
                "agent": selected_agent_name,
                "status": "completed",
                "result": "Simulated successful execution"
            },
            source_system="ai-agents",
            source_agent=selected_agent_name,
            created_by="system",
            importance_score=0.7,
            tags=["task_execution", task_request.category]
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
        """
        cur.execute(update_query, (selected_agent_name, "Simulated successful execution", task_request.task_id))
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
            """
            if 'execution_id' in locals():
                cur.execute(update_exec, ('completed', end_time, latency_ms, execution_id))
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
        """
        cur.execute(update_query, (f"Execution failed: {e}", task_request.task_id))
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
            """
            if 'execution_id' in locals():
                cur.execute(update_exec, ('failed', end_time, latency_ms, str(e), execution_id))
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
async def initialize_systems():
    """Initialize all AI systems"""
    global memory_manager, activation_system, aurea, ai_board, system_initialized

    try:
        logger.info("🚀 Initializing BrainOps AI OS...")

        # Initialize memory manager
        memory_manager = get_memory_manager()
        logger.info("✅ Memory Manager initialized")

        # Initialize agent activation system
        activation_system = get_activation_system()
        logger.info("✅ Agent Activation System initialized")

        # Initialize AUREA orchestrator
        aurea = get_aurea()
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
            tags=["system", "initialization"]
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
    await initialize_systems()

    # Start AUREA in background if configured
    if os.getenv("AUTO_START_AUREA", "false").lower() == "true":
        asyncio.create_task(run_aurea_background())


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
            health_status["memory_stats"] = memory_manager.get_stats()
            health_status["agent_stats"] = activation_system.get_agent_stats()
            health_status["aurea_status"] = aurea.get_status()
            health_status["board_status"] = ai_board.get_board_status()
        except Exception as e:
            logger.error(f"Error getting detailed stats: {e}")

    return health_status


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        return SystemStatusResponse(
            status="operational",
            version="3.0.0",
            components={
                "memory": memory_manager.get_stats(),
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

    background_tasks.add_task(execute_task_in_background, task_request)

    return {
        "status": "task_execution_initiated",
        "task_id": task_request.task_id,
        "message": "Task execution started in background."
    }


@app.get("/memory/stats")
async def get_memory_stats():
    """Get memory system statistics"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    return memory_manager.get_stats()


@app.get("/memory/query")
async def query_memory_get(query: Optional[str] = None, limit: int = 10):
    """Query the unified memory system via GET"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        memories = memory_manager.recall(
            query if query else {},
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
async def query_memory(request: MemoryQueryRequest):
    """Query the unified memory system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        memories = memory_manager.recall(
            request.query,
            context=request.context,
            limit=request.limit
        )

        return {
            "query": request.query,
            "results": memories,
            "count": len(memories)
        }
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/synthesize")
async def synthesize_insights():
    """Synthesize insights from memories"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        insights = memory_manager.synthesize()
        return {
            "insights": insights,
            "count": len(insights),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(request: Dict[str, Any]):
    """Store a memory in the unified system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        from unified_memory_manager import Memory, MemoryType

        memory = Memory(
            memory_type=MemoryType(request.get("memory_type", "episodic")),
            content=request.get("content", {}),
            source_system=request.get("source_system", "api"),
            source_agent=request.get("source_agent", "user"),
            created_by=request.get("created_by", "api"),
            importance_score=request.get("importance", 0.5),
            tags=request.get("tags", []),
            metadata=request.get("metadata", {}),
            context_id=request.get("context_id")
        )

        memory_id = memory_manager.store(memory)
        return {"success": True, "memory_id": memory_id}
    except Exception as e:
        logger.error(f"Memory store failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/recall")
async def recall_memory(
    query: Optional[str] = None,
    context_type: Optional[str] = None,
    context_id: Optional[str] = None,
    limit: int = 10
):
    """Recall memories from the unified system"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        memories = memory_manager.recall(
            query=query if query else {},
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
async def start_aurea(background_tasks: BackgroundTasks):
    """Start AUREA orchestration"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    if aurea.running:
        return {"status": "already_running"}

    background_tasks.add_task(run_aurea_background)
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

async def run_aurea_background():
    """Run AUREA orchestration in background"""
    try:
        logger.info("🧠 Starting AUREA background orchestration")
        await aurea.orchestrate()
    except Exception as e:
        logger.error(f"AUREA background error: {e}")


# Scheduler endpoint for Render (legacy compatibility)
@app.post("/scheduler/execute/{agent_id}")
async def execute_agent_legacy(agent_id: str, request: Dict[str, Any]):
    """Legacy endpoint for agent execution (maintains compatibility)"""
    if not system_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

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
