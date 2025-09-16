#!/usr/bin/env python3
"""
BrainOps AI Agents Web Service
Production HTTP API for AI agent orchestration
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the main orchestrator
from main import SimpleOrchestrator as AIOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BrainOps AI Agents Service",
    description="Production AI agent orchestration system",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": os.getenv("DB_PORT", "5432")
}

# Global orchestrator instance
orchestrator = None

class AgentRequest(BaseModel):
    agent_name: str
    action: str
    data: Optional[Dict[str, Any]] = {}

class EstimateRequest(BaseModel):
    customer_id: str
    property_address: str
    roof_type: Optional[str] = "Asphalt Shingle"
    square_footage: Optional[float] = 2000

class JobRequest(BaseModel):
    customer_id: str
    estimate_id: Optional[str] = None
    description: str
    scheduled_date: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup"""
    global orchestrator
    try:
        orchestrator = AIOrchestrator()
        logger.info("✅ AI Orchestrator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        # Don't fail startup, allow health checks to work
        orchestrator = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BrainOps AI Agents",
        "version": "2.0.0",
        "status": "operational" if orchestrator else "initializing",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ai_agents")
        agent_count = cur.fetchone()[0]
        cur.close()
        conn.close()

        return {
            "status": "healthy",
            "orchestrator": "active" if orchestrator else "inactive",
            "database": "connected",
            "agents_registered": agent_count,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/agents")
async def list_agents():
    """List all registered AI agents"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, name, type, capabilities, status, config
            FROM ai_agents
            ORDER BY name
        """)
        agents = cur.fetchall()
        cur.close()
        conn.close()

        return {
            "agents": agents,
            "count": len(agents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/execute")
async def execute_agent(request: AgentRequest, background_tasks: BackgroundTasks):
    """Execute an AI agent action"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Find the agent
        agent = None
        for a in orchestrator.agents:
            if a.__class__.__name__.lower() == request.agent_name.lower():
                agent = a
                break

        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_name} not found")

        # Execute the action
        if hasattr(agent, request.action):
            method = getattr(agent, request.action)
            result = method(**request.data)

            return {
                "agent": request.agent_name,
                "action": request.action,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Agent {request.agent_name} doesn't support action {request.action}")

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/estimates/create")
async def create_estimate(request: EstimateRequest):
    """Create a new estimate using AI"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Find EstimationAgent
        estimation_agent = None
        for agent in orchestrator.agents:
            if agent.__class__.__name__ == "EstimationAgent":
                estimation_agent = agent
                break

        if not estimation_agent:
            raise HTTPException(status_code=503, detail="EstimationAgent not available")

        # Generate estimate
        result = estimation_agent.generate_estimate(
            customer_id=request.customer_id,
            property_address=request.property_address,
            roof_type=request.roof_type,
            square_footage=request.square_footage
        )

        return {
            "estimate": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Estimate creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/jobs/schedule")
async def schedule_job(request: JobRequest):
    """Schedule a new job"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    try:
        # Find IntelligentScheduler
        scheduler = None
        for agent in orchestrator.agents:
            if agent.__class__.__name__ == "IntelligentScheduler":
                scheduler = agent
                break

        if not scheduler:
            raise HTTPException(status_code=503, detail="IntelligentScheduler not available")

        # Schedule job
        result = scheduler.schedule_job(
            customer_id=request.customer_id,
            estimate_id=request.estimate_id,
            description=request.description,
            date=request.scheduled_date
        )

        return {
            "job": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Job scheduling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows")
async def list_workflows():
    """List active workflows"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    return {
        "workflows": [
            {
                "name": "Lead to Customer",
                "status": "active",
                "description": "Convert leads to paying customers"
            },
            {
                "name": "Estimate to Invoice",
                "status": "active",
                "description": "Generate estimates and convert to invoices"
            },
            {
                "name": "Job Scheduling",
                "status": "active",
                "description": "Intelligent job scheduling and dispatch"
            },
            {
                "name": "Revenue Optimization",
                "status": "active",
                "description": "Maximize revenue through AI pricing"
            }
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/orchestrate")
async def orchestrate_cycle(background_tasks: BackgroundTasks):
    """Trigger a full orchestration cycle"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")

    # Run in background
    background_tasks.add_task(orchestrator.run_cycle)

    return {
        "status": "orchestration cycle started",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        stats = {}

        # Get counts from various tables
        tables = ["customers", "jobs", "estimates", "invoices", "ai_agents"]
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cur.fetchone()[0]
            except:
                stats[table] = 0

        cur.close()
        conn.close()

        return {
            "stats": stats,
            "orchestrator_status": "active" if orchestrator else "inactive",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)