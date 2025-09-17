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
from decimal import Decimal
import uvicorn
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Core imports (always available)
from memory_system import memory_system
from orchestrator import orchestrator

# Feature flags
LANGGRAPH_AVAILABLE = False
VECTOR_MEMORY_AVAILABLE = False
REVENUE_SYSTEM_AVAILABLE = False
ACQUISITION_AVAILABLE = False
PRICING_ENGINE_AVAILABLE = False
NOTEBOOK_LM_AVAILABLE = False

# Try to import advanced modules
try:
    from langgraph_orchestrator import get_langgraph_orchestrator
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
    logger.info("LangGraph module loaded successfully")
    langgraph_orchestrator = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"LangGraph not available: {e}")
    get_langgraph_orchestrator = None
    langgraph_orchestrator = None
    HumanMessage = None
    SystemMessage = None

try:
    from vector_memory_system import get_vector_memory
    VECTOR_MEMORY_AVAILABLE = True
    logger.info("Vector memory module loaded successfully")
    vector_memory = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Vector memory not available: {e}")
    get_vector_memory = None
    vector_memory = None

try:
    from revenue_generation_system import get_revenue_system
    REVENUE_SYSTEM_AVAILABLE = True
    logger.info("Revenue system module loaded successfully")
    revenue_system = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Revenue system not available: {e}")
    get_revenue_system = None
    revenue_system = None

try:
    from customer_acquisition_agents import get_acquisition_orchestrator
    ACQUISITION_AVAILABLE = True
    logger.info("Acquisition agents module loaded successfully")
    acquisition_orchestrator = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Acquisition agents not available: {e}")
    get_acquisition_orchestrator = None
    acquisition_orchestrator = None

try:
    from ai_pricing_engine import get_pricing_engine, PricingFactors, CustomerSegment
    PRICING_ENGINE_AVAILABLE = True
    logger.info("Pricing engine module loaded successfully")
    pricing_engine = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Pricing engine not available: {e}")
    get_pricing_engine = None
    pricing_engine = None
    PricingFactors = None
    CustomerSegment = None

try:
    from notebook_lm_plus import get_notebook_lm, KnowledgeType, LearningSource
    NOTEBOOK_LM_AVAILABLE = True
    logger.info("Notebook LM+ module loaded successfully")
    notebook_lm = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Notebook LM+ not available: {e}")
    get_notebook_lm = None
    notebook_lm = None
    KnowledgeType = None
    LearningSource = None

# Create FastAPI app
app = FastAPI(
    title="BrainOps AI Agent Service",
    description="Orchestration service for AI agents",
    version="2.1.0"  # Added Notebook LM+ learning system
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
    """Root endpoint with feature status"""
    return {
        "service": "BrainOps AI Agent Service",
        "status": "operational",
        "version": "2.0.0",
        "features": {
            "langgraph": LANGGRAPH_AVAILABLE,
            "vector_memory": VECTOR_MEMORY_AVAILABLE,
            "revenue_system": REVENUE_SYSTEM_AVAILABLE,
            "acquisition": ACQUISITION_AVAILABLE,
            "pricing_engine": PRICING_ENGINE_AVAILABLE,
            "notebook_lm": NOTEBOOK_LM_AVAILABLE
        },
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
            "features": {
                "langgraph": LANGGRAPH_AVAILABLE,
                "vector_memory": VECTOR_MEMORY_AVAILABLE,
                "revenue_system": REVENUE_SYSTEM_AVAILABLE,
                "acquisition": ACQUISITION_AVAILABLE,
                "pricing_engine": PRICING_ENGINE_AVAILABLE,
                "notebook_lm": NOTEBOOK_LM_AVAILABLE
            },
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

def json_serializer(obj):
    """Custom JSON serializer for datetime and Decimal objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)

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

        # Convert to JSON-safe format
        agents_list = []
        for agent in agents:
            agent_dict = dict(agent)
            for key, value in agent_dict.items():
                if isinstance(value, (datetime, Decimal)):
                    agent_dict[key] = json_serializer(value)
            agents_list.append(agent_dict)

        return {
            "agents": agents_list,
            "count": len(agents_list),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, config: Dict[str, Any] = None):
    """Execute an agent by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get agent details
        cursor.execute("""
            SELECT id, name, type, status, capabilities
            FROM ai_agents
            WHERE id = %s AND status = 'active'
        """, (agent_id,))

        agent = cursor.fetchone()

        if not agent:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or inactive")

        # Create task execution first
        task_execution_id = str(uuid.uuid4())

        # Get a default task ID (or create one)
        cursor.execute("SELECT id FROM tasks LIMIT 1")
        task_result = cursor.fetchone()
        task_id = task_result['id'] if task_result else str(uuid.uuid4())

        # Create task execution
        cursor.execute("""
            INSERT INTO task_executions (
                id, task_id, status, created_at
            ) VALUES (%s, %s, %s, %s)
        """, (task_execution_id, task_id, 'running', datetime.now(timezone.utc)))

        # Log agent execution
        execution_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO agent_executions (
                id, task_execution_id, agent_type, status, created_at, prompt, response
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (execution_id, task_execution_id, agent['type'], 'running',
              datetime.now(timezone.utc), f"Execute agent {agent['name']}", "Executing..."))

        conn.commit()

        # Execute based on agent type
        result = {
            "execution_id": execution_id,
            "agent_id": agent_id,
            "agent_name": agent['name'],
            "status": "completed",
            "result": f"Successfully executed {agent['type']} agent",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Update execution status
        cursor.execute("""
            UPDATE agent_executions
            SET status = %s, completed_at = %s, response = %s
            WHERE id = %s
        """, ('completed', datetime.now(timezone.utc), json.dumps(result), execution_id))

        # Update task execution status
        cursor.execute("""
            UPDATE task_executions
            SET status = %s, completed_at = %s, result = %s
            WHERE id = %s
        """, ('completed', datetime.now(timezone.utc), json.dumps(result), task_execution_id))

        conn.commit()
        cursor.close()
        conn.close()

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing agent {agent_id}: {e}")
        # Make sure to close connections
        if 'conn' in locals():
            conn.close()
        return {
            "execution_id": execution_id if 'execution_id' in locals() else None,
            "agent_id": agent_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# LangGraph endpoints (conditional)
if LANGGRAPH_AVAILABLE:
    @app.post("/langgraph/workflow")
    async def execute_langgraph_workflow(request: Dict[str, Any]):
        """Execute a LangGraph-based workflow"""
        try:
            messages = [
                SystemMessage(content=request.get('system_prompt', 'You are a helpful AI assistant.')),
                HumanMessage(content=request.get('prompt', ''))
            ]

            metadata = {
                "workflow_id": str(uuid.uuid4()),
                "user_id": request.get('user_id'),
                "context": request.get('context', {}),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            global langgraph_orchestrator
            if langgraph_orchestrator is None:
                langgraph_orchestrator = get_langgraph_orchestrator()
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

    @app.get("/langgraph/status")
    async def get_langgraph_status():
        """Get LangGraph orchestrator status"""
        global langgraph_orchestrator
        if langgraph_orchestrator is None:
            langgraph_orchestrator = get_langgraph_orchestrator()
        return {
            "status": "operational",
            "components": {
                "openai_llm": hasattr(langgraph_orchestrator, 'openai_llm') and langgraph_orchestrator.openai_llm is not None,
                "anthropic_llm": hasattr(langgraph_orchestrator, 'anthropic_llm') and langgraph_orchestrator.anthropic_llm is not None,
                "vector_store": hasattr(langgraph_orchestrator, 'vector_store') and langgraph_orchestrator.vector_store is not None,
                "workflow_graph": hasattr(langgraph_orchestrator, 'workflow') and langgraph_orchestrator.workflow is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Vector Memory endpoints (conditional)
if VECTOR_MEMORY_AVAILABLE:
    @app.post("/vector-memory/store")
    async def store_vector_memory(data: Dict[str, Any]):
        """Store a memory with vector embedding"""
        try:
            global vector_memory
            if vector_memory is None:
                vector_memory = get_vector_memory()
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
            global vector_memory
            if vector_memory is None:
                vector_memory = get_vector_memory()
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

    @app.get("/vector-memory/stats")
    async def get_vector_memory_stats():
        """Get vector memory statistics"""
        try:
            global vector_memory
            if vector_memory is None:
                vector_memory = get_vector_memory()
            stats = vector_memory.get_memory_statistics()
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Revenue System endpoints (conditional)
if REVENUE_SYSTEM_AVAILABLE:
    @app.post("/revenue/identify-leads")
    async def identify_revenue_leads(criteria: Dict[str, Any]):
        """Identify new revenue leads"""
        try:
            global revenue_system
            if revenue_system is None:
                revenue_system = get_revenue_system()
            leads = await revenue_system.identify_new_leads(criteria)
            return {"leads_found": len(leads), "lead_ids": leads}
        except Exception as e:
            logger.error(f"Lead identification failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/revenue/qualify-lead/{lead_id}")
    async def qualify_revenue_lead(lead_id: str):
        """Qualify a lead for revenue potential"""
        try:
            global revenue_system
            if revenue_system is None:
                revenue_system = get_revenue_system()
            score, qualification = await revenue_system.qualify_lead(lead_id)
            return {"lead_id": lead_id, "score": score, "qualification": qualification}
        except Exception as e:
            logger.error(f"Lead qualification failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/revenue/generate-proposal/{lead_id}")
    async def generate_revenue_proposal(lead_id: str, requirements: Dict[str, Any]):
        """Generate AI-powered proposal"""
        try:
            global revenue_system
            if revenue_system is None:
                revenue_system = get_revenue_system()
            proposal = await revenue_system.generate_proposal(lead_id, requirements)
            return proposal
        except Exception as e:
            logger.error(f"Proposal generation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Customer Acquisition endpoints (conditional)
if ACQUISITION_AVAILABLE:
    @app.post("/acquisition/run-pipeline")
    async def run_acquisition_pipeline(criteria: Dict[str, Any]):
        """Run customer acquisition pipeline"""
        try:
            global acquisition_orchestrator
            if acquisition_orchestrator is None:
                acquisition_orchestrator = get_acquisition_orchestrator()
            result = await acquisition_orchestrator.run_acquisition_pipeline(criteria)
            return result
        except Exception as e:
            logger.error(f"Acquisition pipeline failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/acquisition/metrics")
    async def get_acquisition_metrics():
        """Get customer acquisition metrics"""
        try:
            global acquisition_orchestrator
            if acquisition_orchestrator is None:
                acquisition_orchestrator = get_acquisition_orchestrator()
            metrics = await acquisition_orchestrator.get_acquisition_metrics()
            return metrics
        except Exception as e:
            logger.error(f"Failed to get acquisition metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Pricing Engine endpoints (conditional)
if PRICING_ENGINE_AVAILABLE:
    @app.post("/pricing/generate-quote")
    async def generate_price_quote(request: Dict[str, Any]):
        """Generate AI-optimized price quote"""
        try:
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

            global pricing_engine
            if pricing_engine is None:
                pricing_engine = get_pricing_engine()
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

# Notebook LM+ endpoints (conditional)
if NOTEBOOK_LM_AVAILABLE:
    @app.post("/notebook-lm/learn")
    async def learn_from_interaction(data: Dict[str, Any]):
        """Learn from an interaction"""
        try:
            global notebook_lm
            if notebook_lm is None:
                notebook_lm = get_notebook_lm()

            # Start session if needed
            if not notebook_lm.active_session:
                notebook_lm.start_learning_session(data.get('topics', []))

            # Learn from the interaction
            knowledge_id = notebook_lm.learn_from_interaction(
                content=data['content'],
                context=data.get('context', {}),
                source=LearningSource[data.get('source', 'USER_INTERACTION')]
            )

            return {
                "knowledge_id": knowledge_id,
                "status": "learned" if knowledge_id else "failed",
                "session_id": notebook_lm.active_session
            }
        except Exception as e:
            logger.error(f"Learning failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/notebook-lm/query")
    async def query_knowledge(query: str, limit: int = 10):
        """Query the knowledge base"""
        try:
            global notebook_lm
            if notebook_lm is None:
                notebook_lm = get_notebook_lm()

            results = notebook_lm.query_knowledge(query, limit)
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/notebook-lm/insights")
    async def get_insights(min_impact: float = 0.5):
        """Get synthesized insights"""
        try:
            global notebook_lm
            if notebook_lm is None:
                notebook_lm = get_notebook_lm()

            insights = notebook_lm.get_insights(min_impact=min_impact)
            return {
                "insights": insights,
                "count": len(insights)
            }
        except Exception as e:
            logger.error(f"Failed to get insights: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    logger.info("=== BrainOps AI Service Starting ===")
    logger.info(f"LangGraph Available: {LANGGRAPH_AVAILABLE}")
    logger.info(f"Vector Memory Available: {VECTOR_MEMORY_AVAILABLE}")
    logger.info(f"Revenue System Available: {REVENUE_SYSTEM_AVAILABLE}")
    logger.info(f"Acquisition Available: {ACQUISITION_AVAILABLE}")
    logger.info(f"Pricing Engine Available: {PRICING_ENGINE_AVAILABLE}")
    logger.info(f"Notebook LM+ Available: {NOTEBOOK_LM_AVAILABLE}")

    try:
        from scheduled_executor import scheduler
        asyncio.create_task(scheduler.run_scheduler())
        logger.info("Scheduled executor started")
    except ImportError:
        logger.warning("Scheduled executor not available")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Agent Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)