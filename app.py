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
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
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
        # Import the real executor
        from agent_executor import executor

        conn = get_db_connection()
        cursor = conn.cursor()

        # Get agent by ID or name
        if agent_id.startswith('srv-') or len(agent_id) == 36:  # UUID format
            cursor.execute("""
                SELECT id, name, type, capabilities
                FROM ai_agents
                WHERE id = %s
            """, (agent_id,))
        else:
            # Try by name
            cursor.execute("""
                SELECT id, name, type, capabilities
                FROM ai_agents
                WHERE LOWER(name) = LOWER(%s)
            """, (agent_id,))

        agent = cursor.fetchone()
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Update agent last_active
        cursor.execute("""
            UPDATE ai_agents
            SET last_active = NOW()
            WHERE id = %s
        """, (agent['id'],))

        # Log execution start
        import uuid
        execution_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO agent_executions (id, task_execution_id, agent_type, prompt, status, created_at)
            VALUES (%s, %s, %s, %s, 'running', NOW())
            RETURNING id
        """, (execution_id, execution_id, agent['type'], json.dumps(task)))

        execution_id = cursor.fetchone()['id']

        conn.commit()
        cursor.close()
        conn.close()

        # Execute with real executor
        try:
            result = await executor.execute(agent['name'], task)

            # Update execution status
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE agent_executions
                SET status = %s, response = %s, completed_at = NOW()
                WHERE id = %s
            """, (result.get('status', 'completed'), json.dumps(result), execution_id))
            conn.commit()
            cursor.close()
            conn.close()

            return {
                "execution_id": execution_id,
                "agent_id": agent['id'],
                "agent_name": agent['name'],
                "status": result.get('status', 'completed'),
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as exec_error:
            logger.error(f"Agent execution error: {exec_error}")
            # Fallback to simulated response
            return {
                "execution_id": execution_id,
                "agent_id": agent['id'],
                "agent_name": agent['name'],
                "status": "error",
                "message": str(exec_error),
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
        from agent_executor import executor

        workflow_type = workflow.get('type', 'custom')
        workflow_id = "wf_" + datetime.now().strftime("%Y%m%d%H%M%S")

        # Execute with WorkflowEngine
        workflow_task = {
            'workflow_type': workflow_type,
            'workflow_id': workflow_id,
            **workflow
        }

        result = await executor.execute('WorkflowEngine', workflow_task)

        return {
            "workflow_id": workflow_id,
            "status": result.get('status', 'completed'),
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error orchestrating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Agent Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)