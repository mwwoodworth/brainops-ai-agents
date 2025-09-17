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
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
import uvicorn
from memory_system import memory_system
from orchestrator import orchestrator
from langgraph_orchestrator import langgraph_orchestrator
from vector_memory_system import vector_memory
from revenue_generation_system import revenue_system
from customer_acquisition_agents import acquisition_orchestrator
from ai_pricing_engine import pricing_engine, PricingFactors, CustomerSegment
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

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
        # Try to import the real executor
        try:
            from agent_executor import executor
            has_executor = True
        except Exception as e:
            logger.error(f"Failed to import executor: {e}")
            has_executor = False

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

        # First create task_execution entry
        import uuid
        task_execution_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO task_executions (id, task_id, status, created_at)
            VALUES (%s, %s, 'running', NOW())
        """, (task_execution_id, f"agent_{agent['type']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"))

        # Then create agent_execution entry
        execution_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO agent_executions (id, task_execution_id, agent_type, prompt, status, created_at)
            VALUES (%s, %s, %s, %s, 'running', NOW())
            RETURNING id
        """, (execution_id, task_execution_id, agent['type'], json.dumps(task)))

        execution_id = cursor.fetchone()['id']

        conn.commit()
        cursor.close()
        conn.close()

        # Execute with real executor if available
        if has_executor:
            try:
                result = await executor.execute(agent['name'], task)
            except Exception as exec_error:
                logger.error(f"Executor failed: {exec_error}")
                result = {"status": "error", "error": str(exec_error)}
        else:
            # Fallback execution
            result = {
                "status": "simulated",
                "agent": agent['name'],
                "task": task,
                "message": "Executor not available, using simulation"
            }

        # Update execution status
        conn = get_db_connection()
        cursor = conn.cursor()

        # Update agent_execution
        cursor.execute("""
            UPDATE agent_executions
            SET status = %s, response = %s, completed_at = NOW()
            WHERE id = %s
        """, (result.get('status', 'completed'), json.dumps(result), execution_id))

        # Update task_execution
        cursor.execute("""
            UPDATE task_executions
            SET status = %s, result = %s, completed_at = NOW()
            WHERE id = %s
        """, (result.get('status', 'completed'), json.dumps(result), task_execution_id))

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
        # Use the new orchestrator
        result = await orchestrator.execute_workflow(
            workflow.get('type', 'full_system_check'),
            workflow
        )

        # Store in memory
        memory_system.update_system_state('workflow', result['workflow_id'], result)

        return result
    except Exception as e:
        logger.error(f"Error orchestrating workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/context/{key}")
async def get_memory_context(key: str):
    """Get context from memory"""
    context = memory_system.get_context(key)
    if context:
        return {"key": key, "value": context}
    raise HTTPException(status_code=404, detail="Context not found")

@app.post("/memory/context")
async def store_memory_context(data: Dict[str, Any]):
    """Store context in memory"""
    try:
        context_id = memory_system.store_context(
            data.get('type', 'general'),
            data['key'],
            data['value'],
            data.get('critical', False)
        )
        return {"id": context_id, "status": "stored"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/critical")
async def get_critical_memory():
    """Get all critical context"""
    return memory_system.get_critical_context()

@app.get("/memory/overview")
async def get_system_overview():
    """Get complete system overview"""
    return memory_system.get_system_overview()

@app.post("/memory/search")
async def search_knowledge(query: Dict[str, Any]):
    """Search knowledge base"""
    results = memory_system.search_knowledge(
        query['query'],
        query.get('category'),
        query.get('limit', 10)
    )
    return {"results": results, "count": len(results)}

@app.post("/langgraph/workflow")
async def execute_langgraph_workflow(request: Dict[str, Any]):
    """Execute a LangGraph-based workflow with advanced orchestration"""
    try:
        # Convert request to LangChain messages
        messages = [
            SystemMessage(content=request.get('system_prompt', 'You are a helpful AI assistant.')),
            HumanMessage(content=request.get('prompt', ''))
        ]

        # Add metadata
        metadata = {
            "workflow_id": str(uuid.uuid4()),
            "user_id": request.get('user_id'),
            "context": request.get('context', {}),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Execute workflow
        result = await langgraph_orchestrator.run_workflow(messages, metadata)

        return {
            "workflow_id": metadata["workflow_id"],
            "success": result["success"],
            "response": result["response"],
            "agent_used": result["agent_used"],
            "errors": result["errors"],
            "execution_time": result["execution_time"]
        }

    except Exception as e:
        logger.error(f"LangGraph workflow error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/langgraph/analyze")
async def analyze_with_langgraph(data: Dict[str, Any]):
    """Analyze data using LangGraph orchestration"""
    try:
        messages = [
            SystemMessage(content="You are a data analysis expert using advanced AI orchestration."),
            HumanMessage(content=f"Analyze this data: {json.dumps(data.get('data', {}))}")
        ]

        result = await langgraph_orchestrator.run_workflow(messages, {"analysis_type": data.get('type', 'general')})

        return {
            "analysis": result["response"],
            "agent": result["agent_used"],
            "success": result["success"]
        }

    except Exception as e:
        logger.error(f"LangGraph analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/langgraph/status")
async def get_langgraph_status():
    """Get LangGraph orchestrator status"""
    return {
        "status": "operational",
        "components": {
            "openai_llm": langgraph_orchestrator.openai_llm is not None,
            "anthropic_llm": langgraph_orchestrator.anthropic_llm is not None,
            "vector_store": langgraph_orchestrator.vector_store is not None,
            "workflow_graph": langgraph_orchestrator.workflow is not None
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/vector-memory/store")
async def store_vector_memory(data: Dict[str, Any]):
    """Store a memory with vector embedding"""
    try:
        memory_id = vector_memory.store_memory(
            content=data['content'],
            memory_type=data.get('type', 'general'),
            metadata=data.get('metadata', {}),
            importance=data.get('importance', 0.5)
        )

        return {
            "memory_id": memory_id,
            "status": "stored" if memory_id else "failed"
        }

    except Exception as e:
        logger.error(f"Failed to store vector memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-memory/recall")
async def recall_vector_memories(query: Dict[str, Any]):
    """Recall memories using semantic search"""
    try:
        memories = vector_memory.recall_memories(
            query=query['query'],
            limit=query.get('limit', 10),
            memory_type=query.get('type'),
            threshold=query.get('threshold', 0.7)
        )

        return {
            "memories": memories,
            "count": len(memories)
        }

    except Exception as e:
        logger.error(f"Failed to recall memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-memory/associate")
async def associate_vector_memories(data: Dict[str, Any]):
    """Create association between memories"""
    try:
        vector_memory.associate_memories(
            memory_id_1=data['memory_id_1'],
            memory_id_2=data['memory_id_2'],
            strength=data.get('strength', 0.5),
            association_type=data.get('type', 'related')
        )

        return {"status": "associated"}

    except Exception as e:
        logger.error(f"Failed to associate memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-memory/consolidate")
async def consolidate_vector_memories(data: Dict[str, Any]):
    """Consolidate multiple memories"""
    try:
        new_memory_id = vector_memory.consolidate_memories(
            memory_ids=data['memory_ids'],
            consolidation_type=data.get('type', 'summary')
        )

        return {
            "consolidated_memory_id": new_memory_id,
            "status": "consolidated" if new_memory_id else "failed"
        }

    except Exception as e:
        logger.error(f"Failed to consolidate memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-memory/stats")
async def get_vector_memory_stats():
    """Get vector memory statistics"""
    try:
        stats = vector_memory.get_memory_statistics()
        return stats

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-memory/decay")
async def apply_memory_decay():
    """Apply decay to memories"""
    try:
        vector_memory.decay_memories()
        return {"status": "decay applied"}

    except Exception as e:
        logger.error(f"Failed to apply decay: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Revenue Generation Endpoints
@app.post("/revenue/identify-leads")
async def identify_revenue_leads(criteria: Dict[str, Any]):
    """Identify new revenue leads"""
    try:
        leads = await revenue_system.identify_new_leads(criteria)
        return {"leads_found": len(leads), "lead_ids": leads}
    except Exception as e:
        logger.error(f"Lead identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/revenue/qualify-lead/{lead_id}")
async def qualify_revenue_lead(lead_id: str):
    """Qualify a lead for revenue potential"""
    try:
        score, qualification = await revenue_system.qualify_lead(lead_id)
        return {"lead_id": lead_id, "score": score, "qualification": qualification}
    except Exception as e:
        logger.error(f"Lead qualification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/revenue/generate-proposal/{lead_id}")
async def generate_revenue_proposal(lead_id: str, requirements: Dict[str, Any]):
    """Generate AI-powered proposal"""
    try:
        proposal = await revenue_system.generate_proposal(lead_id, requirements)
        return proposal
    except Exception as e:
        logger.error(f"Proposal generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Customer Acquisition Endpoints
@app.post("/acquisition/run-pipeline")
async def run_acquisition_pipeline(criteria: Dict[str, Any]):
    """Run customer acquisition pipeline"""
    try:
        result = await acquisition_orchestrator.run_acquisition_pipeline(criteria)
        return result
    except Exception as e:
        logger.error(f"Acquisition pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/acquisition/metrics")
async def get_acquisition_metrics():
    """Get customer acquisition metrics"""
    try:
        metrics = await acquisition_orchestrator.get_acquisition_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get acquisition metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Pricing Engine Endpoints
@app.post("/pricing/generate-quote")
async def generate_price_quote(request: Dict[str, Any]):
    """Generate AI-optimized price quote"""
    try:
        # Convert request to PricingFactors
        factors = PricingFactors(
            customer_segment=CustomerSegment[request.get('segment', 'SMALL_BUSINESS')],
            company_size=request.get('company_size', 10),
            revenue_range=(request.get('revenue_min', 0), request.get('revenue_max', 1000000)),
            urgency_level=request.get('urgency', 0.5),
            competition_present=request.get('competition', False),
            feature_requirements=request.get('features', []),
            contract_length=request.get('contract_months', 12),
            payment_terms=request.get('payment_terms', 'Net 30'),
            market_conditions=request.get('market_conditions', {}),
            historical_data=request.get('historical_data', {})
        )

        quote = await pricing_engine.generate_quote(factors, request.get('lead_id'))

        if quote:
            return {
                "quote_id": quote.id,
                "base_price": quote.base_price,
                "final_price": quote.final_price,
                "discount": quote.discount_percentage,
                "strategy": quote.pricing_strategy.value,
                "win_probability": quote.win_probability,
                "components": quote.components,
                "terms": quote.terms,
                "expires_at": quote.expires_at.isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Quote generation failed")

    except Exception as e:
        logger.error(f"Quote generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pricing/ab-test")
async def run_pricing_ab_test(test_config: Dict[str, Any]):
    """Run A/B test on pricing strategies"""
    try:
        result = await pricing_engine.run_ab_test(
            test_config['name'],
            test_config['variant_a'],
            test_config['variant_b'],
            test_config.get('sample_size', 100)
        )
        return result
    except Exception as e:
        logger.error(f"A/B test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    from scheduled_executor import scheduler

    # Start scheduled executor (database-driven)
    asyncio.create_task(scheduler.run_scheduler())

    logger.info("Scheduled executor started")
    logger.info("LangGraph orchestrator initialized")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Agent Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)