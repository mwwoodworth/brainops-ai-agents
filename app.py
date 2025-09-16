#!/usr/bin/env python3
"""
BrainOps AI Agent Service - Web API
Provides REST API for AI agent orchestration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BrainOps AI Agent Service",
    description="Orchestration service for AI agents",
    version="1.0.0"
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
    "password": os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "BrainOps AI Agent Service",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@app.get("/agents")
async def get_agents():
    """Get all AI agents"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id, name, type, status, capabilities,
                last_active, created_at
            FROM ai_agents
            WHERE status = 'active'
            ORDER BY last_active DESC
        """)

        agents = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "agents": agents,
            "total": len(agents),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent details"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id, name, type, status, capabilities,
                last_active, created_at, metadata
            FROM ai_agents
            WHERE id = %s
        """, (agent_id,))

        agent = cursor.fetchone()
        cursor.close()
        conn.close()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return agent
    except Exception as e:
        logger.error(f"Error fetching agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, task: Dict[str, Any]):
    """Execute a task with specific agent"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Update agent last_active
        cursor.execute("""
            UPDATE ai_agents
            SET last_active = NOW()
            WHERE id = %s
            RETURNING name, type, capabilities
        """, (agent_id,))

        agent = cursor.fetchone()
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Log execution
        cursor.execute("""
            INSERT INTO agent_executions (agent_id, task, status, started_at)
            VALUES (%s, %s, 'running', NOW())
            RETURNING id
        """, (agent_id, json.dumps(task)))

        execution_id = cursor.fetchone()['id']

        conn.commit()
        cursor.close()
        conn.close()

        # TODO: Implement actual agent execution logic
        # For now, return a simulated response
        return {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent['name'],
            "status": "accepted",
            "message": f"Task accepted by {agent['name']}",
            "estimated_completion": "2-5 seconds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Error executing agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/activity/recent")
async def get_recent_activity():
    """Get recent agent activity"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                a.id, a.name, a.type,
                a.last_active,
                EXTRACT(EPOCH FROM (NOW() - a.last_active))/60 as minutes_since_active,
                COUNT(ae.id) as recent_executions
            FROM ai_agents a
            LEFT JOIN agent_executions ae ON a.id = ae.agent_id
                AND ae.started_at > NOW() - INTERVAL '1 hour'
            WHERE a.status = 'active'
            GROUP BY a.id, a.name, a.type, a.last_active
            ORDER BY a.last_active DESC
            LIMIT 20
        """)

        activity = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "activity": activity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/orchestrate")
async def orchestrate_agents(workflow: Dict[str, Any]):
    """Orchestrate multiple agents for a workflow"""
    try:
        # TODO: Implement actual orchestration logic
        # For now, return a simulated response
        return {
            "workflow_id": "wf_" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "status": "initiated",
            "agents_involved": workflow.get("agents", []),
            "estimated_completion": "10-30 seconds",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error orchestrating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Agent Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)