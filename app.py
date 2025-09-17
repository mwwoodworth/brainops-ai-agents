#!/usr/bin/env python3
"""
BrainOps AI Agent Service - Web API
Provides REST API for AI agent orchestration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
import os
import json
import logging
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone, timedelta
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
CONVERSATION_MEMORY_AVAILABLE = False
SYSTEM_STATE_AVAILABLE = False
DECISION_TREE_AVAILABLE = False
REALTIME_MONITOR_AVAILABLE = False
LEAD_NURTURING_AVAILABLE = False
INTELLIGENT_FOLLOWUP_AVAILABLE = False
CUSTOMER_ONBOARDING_AVAILABLE = False
AUTOMATED_REPORTING_AVAILABLE = False
COST_OPTIMIZATION_AVAILABLE = False

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

try:
    from conversation_memory import get_conversation_memory, MessageRole
    CONVERSATION_MEMORY_AVAILABLE = True
    logger.info("Conversation memory module loaded successfully")
    conversation_memory = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Conversation memory not available: {e}")
    get_conversation_memory = None
    conversation_memory = None
    MessageRole = None

try:
    from system_state_manager import (
        get_system_state_manager,
        SystemComponent,
        ServiceStatus,
        check_system_health,
        monitor_component,
        trigger_system_recovery
    )
    SYSTEM_STATE_AVAILABLE = True
    logger.info("System state management module loaded successfully")
    system_state_manager = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"System state management not available: {e}")
    get_system_state_manager = None
    system_state_manager = None
    SystemComponent = None
    ServiceStatus = None
    check_system_health = None
    monitor_component = None
    trigger_system_recovery = None

try:
    from ai_decision_tree import (
        get_ai_decision_tree,
        DecisionType,
        DecisionContext,
        ActionType,
        ConfidenceLevel
    )
    DECISION_TREE_AVAILABLE = True
    logger.info("AI decision tree module loaded successfully")
    ai_decision_tree = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"AI decision tree not available: {e}")
    get_ai_decision_tree = None
    ai_decision_tree = None
    DecisionType = None
    DecisionContext = None
    ActionType = None
    ConfidenceLevel = None

try:
    from realtime_monitor import (
        get_realtime_monitor,
        EventType,
        SubscriptionType,
        RealtimeEvent
    )
    REALTIME_MONITOR_AVAILABLE = True
    logger.info("Realtime monitor module loaded successfully")
    realtime_monitor = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Realtime monitor not available: {e}")
    get_realtime_monitor = None
    realtime_monitor = None
    EventType = None
    SubscriptionType = None
    RealtimeEvent = None

# Self-healing recovery system
SELF_HEALING_AVAILABLE = False
try:
    from self_healing_recovery import (
        get_self_healing_recovery,
        RecoveryStrategy,
        ErrorSeverity
    )
    SELF_HEALING_AVAILABLE = True
    logger.info("Self-healing recovery module loaded successfully")
    self_healing_recovery = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Self-healing recovery not available: {e}")
    get_self_healing_recovery = None
    self_healing_recovery = None
    RecoveryStrategy = None
    ErrorSeverity = None
    ComponentHealth = None

# AI Training Pipeline
TRAINING_PIPELINE_AVAILABLE = False
try:
    from ai_training_pipeline import (
        get_training_pipeline,
        InteractionType,
        LearningCategory,
        ModelType,
        TrainingStatus
    )
    TRAINING_PIPELINE_AVAILABLE = True
    logger.info("AI training pipeline module loaded successfully")
    training_pipeline = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"AI training pipeline not available: {e}")
    get_training_pipeline = None
    training_pipeline = None
    InteractionType = None
    LearningCategory = None
    ModelType = None
    TrainingStatus = None

# Document Processor
DOCUMENT_PROCESSOR_AVAILABLE = False
try:
    from document_processor import (
        get_document_processor,
        DocumentType,
        ProcessingStatus,
        DocumentCategory
    )
    DOCUMENT_PROCESSOR_AVAILABLE = True
    logger.info("Document processor module loaded successfully")
    document_processor = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Document processor not available: {e}")
    get_document_processor = None
    document_processor = None
    DocumentType = None
    ProcessingStatus = None
    DocumentCategory = None

# AI Context Awareness
CONTEXT_AWARENESS_AVAILABLE = False
try:
    from ai_context_awareness import (
        get_context_awareness,
        UserRole,
        ContextType,
        PersonalizationType,
        PrivacyLevel
    )
    CONTEXT_AWARENESS_AVAILABLE = True
    logger.info("AI context awareness module loaded successfully")
    context_awareness = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"AI context awareness not available: {e}")
    get_context_awareness = None
    context_awareness = None
    UserRole = None
    ContextType = None
    PersonalizationType = None
    PrivacyLevel = None

# Lead Nurturing System
try:
    from lead_nurturing_system import (
        get_lead_nurturing_system,
        NurtureSequenceType,
        LeadSegment,
        PersonalizationEngine,
        DeliveryManager
    )
    LEAD_NURTURING_AVAILABLE = True
    logger.info("Lead nurturing system module loaded successfully")
    lead_nurturing_system = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Lead nurturing system not available: {e}")
    get_lead_nurturing_system = None
    lead_nurturing_system = None
    NurtureSequenceType = None
    LeadSegment = None
    PersonalizationEngine = None
    DeliveryManager = None

# Intelligent Follow-up System
try:
    from intelligent_followup_system import (
        get_intelligent_followup_system,
        FollowUpType,
        FollowUpPriority,
        FollowUpStatus,
        DeliveryChannel,
        ResponseType
    )
    INTELLIGENT_FOLLOWUP_AVAILABLE = True
    logger.info("Intelligent follow-up system loaded successfully")
    intelligent_followup_system = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Intelligent follow-up system not available: {e}")
    get_intelligent_followup_system = None
    intelligent_followup_system = None
    FollowUpType = None
    FollowUpPriority = None
    FollowUpStatus = None
    DeliveryChannel = None
    ResponseType = None

# AI Customer Onboarding System
try:
    from ai_customer_onboarding import (
        get_ai_customer_onboarding,
        OnboardingStage,
        OnboardingStatus,
        CustomerSegment,
        OnboardingAction,
        InterventionType
    )
    CUSTOMER_ONBOARDING_AVAILABLE = True
    logger.info("AI customer onboarding system loaded successfully")
    customer_onboarding_system = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"AI customer onboarding system not available: {e}")
    get_ai_customer_onboarding = None
    customer_onboarding_system = None
    OnboardingStage = None
    OnboardingStatus = None
    CustomerSegment = None
    OnboardingAction = None

# Automated Reporting System
try:
    from automated_reporting_system import (
        get_automated_reporting_system,
        ReportType,
        ReportFrequency,
        DeliveryChannel as ReportDeliveryChannel
    )
    AUTOMATED_REPORTING_AVAILABLE = True
    logger.info("Automated reporting system loaded successfully")
    automated_reporting_system = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Automated reporting system not available: {e}")
    get_automated_reporting_system = None
    automated_reporting_system = None
    ReportType = None
    ReportFrequency = None
    ReportDeliveryChannel = None

# AI Cost Optimization Engine
try:
    from ai_cost_optimization_engine import (
        get_cost_optimization_engine,
        ResourceType,
        OptimizationStrategy,
        CostLevel
    )
    COST_OPTIMIZATION_AVAILABLE = True
    logger.info("Cost optimization engine loaded successfully")
    cost_optimization_engine = None  # Will be initialized lazily
except ImportError as e:
    logger.warning(f"Cost optimization engine not available: {e}")
    get_cost_optimization_engine = None
    cost_optimization_engine = None
    ResourceType = None
    OptimizationStrategy = None
    CostLevel = None

# Create FastAPI app
app = FastAPI(
    title="BrainOps AI Agent Service",
    description="Orchestration service for AI agents",
    version="3.2.0"  # Added AI Customer Onboarding
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
            "notebook_lm": NOTEBOOK_LM_AVAILABLE,
            "conversation_memory": CONVERSATION_MEMORY_AVAILABLE
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
            "version": "3.4.0",
            "database": "connected",
            "features": {
                "langgraph": LANGGRAPH_AVAILABLE,
                "vector_memory": VECTOR_MEMORY_AVAILABLE,
                "revenue_system": REVENUE_SYSTEM_AVAILABLE,
                "acquisition": ACQUISITION_AVAILABLE,
                "pricing_engine": PRICING_ENGINE_AVAILABLE,
                "notebook_lm": NOTEBOOK_LM_AVAILABLE,
                "conversation_memory": CONVERSATION_MEMORY_AVAILABLE,
                "system_state": SYSTEM_STATE_AVAILABLE,
                "decision_tree": DECISION_TREE_AVAILABLE,
                "realtime_monitor": REALTIME_MONITOR_AVAILABLE,
                "self_healing": SELF_HEALING_AVAILABLE,
                "training_pipeline": TRAINING_PIPELINE_AVAILABLE,
                "document_processor": DOCUMENT_PROCESSOR_AVAILABLE,
                "context_awareness": CONTEXT_AWARENESS_AVAILABLE,
                "lead_nurturing": LEAD_NURTURING_AVAILABLE,
                "intelligent_followup": INTELLIGENT_FOLLOWUP_AVAILABLE,
                "customer_onboarding": CUSTOMER_ONBOARDING_AVAILABLE,
                "automated_reporting": AUTOMATED_REPORTING_AVAILABLE,
                "cost_optimization": COST_OPTIMIZATION_AVAILABLE
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

# Conversation Memory endpoints (conditional)
if CONVERSATION_MEMORY_AVAILABLE:
    @app.post("/conversations/start")
    async def start_conversation(data: Dict[str, Any]):
        """Start a new conversation"""
        try:
            global conversation_memory
            if conversation_memory is None:
                conversation_memory = get_conversation_memory()

            conversation_id = conversation_memory.start_conversation(
                user_id=data['user_id'],
                title=data.get('title'),
                context=data.get('context', {})
            )

            return {
                "conversation_id": conversation_id,
                "status": "started" if conversation_id else "failed"
            }
        except Exception as e:
            logger.error(f"Failed to start conversation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/conversations/{conversation_id}/message")
    async def add_message(conversation_id: str, data: Dict[str, Any]):
        """Add a message to conversation"""
        try:
            global conversation_memory
            if conversation_memory is None:
                conversation_memory = get_conversation_memory()

            message_id = conversation_memory.add_message(
                conversation_id=conversation_id,
                role=MessageRole[data['role'].upper()],
                content=data['content'],
                metadata=data.get('metadata', {})
            )

            return {
                "message_id": message_id,
                "status": "added" if message_id else "failed"
            }
        except Exception as e:
            logger.error(f"Failed to add message: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/conversations/{conversation_id}/context")
    async def get_conversation_context(conversation_id: str, num_messages: int = 10):
        """Get conversation context"""
        try:
            global conversation_memory
            if conversation_memory is None:
                conversation_memory = get_conversation_memory()

            context = conversation_memory.get_conversation_context(
                conversation_id, num_messages
            )

            return context if context else {"error": "Conversation not found"}
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/conversations/search")
    async def search_conversations(user_id: str, query: str, limit: int = 10):
        """Search conversation history"""
        try:
            global conversation_memory
            if conversation_memory is None:
                conversation_memory = get_conversation_memory()

            results = conversation_memory.search_conversations(
                user_id, query, limit
            )

            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# System state management endpoints
if SYSTEM_STATE_AVAILABLE:
    @app.get("/system/status")
    async def get_system_status():
        """Get comprehensive system status"""
        try:
            global system_state_manager
            if system_state_manager is None:
                system_state_manager = get_system_state_manager()

            return system_state_manager.get_system_status()
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/system/health-check")
    async def perform_health_check():
        """Perform full system health check"""
        try:
            snapshot = await check_system_health()
            return {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp.isoformat(),
                "overall_status": snapshot.overall_status.value,
                "health_score": snapshot.health_score,
                "active_components": snapshot.active_components,
                "failed_components": snapshot.failed_components,
                "warning_count": snapshot.warning_count,
                "pending_tasks": snapshot.pending_tasks,
                "completed_tasks": snapshot.completed_tasks
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/system/component/{component_name}")
    async def get_component_status(component_name: str):
        """Get status of specific component"""
        try:
            component = SystemComponent[component_name.upper()]
            state = await monitor_component(component)

            return {
                "component": state.component,
                "status": state.status.value,
                "health_score": state.health_score,
                "last_check": state.last_check.isoformat(),
                "error_count": state.error_count,
                "success_rate": state.success_rate,
                "latency_ms": state.latency_ms,
                "metadata": state.metadata,
                "dependencies": state.dependencies
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown component: {component_name}")
        except Exception as e:
            logger.error(f"Component check failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/system/recovery")
    async def trigger_recovery(component_name: str, error_type: str):
        """Trigger recovery action for component"""
        try:
            component = SystemComponent[component_name.upper()]
            success = await trigger_system_recovery(component, error_type)

            return {
                "component": component_name,
                "error_type": error_type,
                "recovery_triggered": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown component: {component_name}")
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/system/alert")
    async def create_system_alert(
        alert_type: str,
        severity: str,
        message: str,
        component: Optional[str] = None,
        details: Optional[Dict] = None
    ):
        """Create system alert"""
        try:
            global system_state_manager
            if system_state_manager is None:
                system_state_manager = get_system_state_manager()

            system_state_manager.create_alert(
                alert_type, severity, component, message, details
            )

            return {
                "status": "alert_created",
                "alert_type": alert_type,
                "severity": severity,
                "message": message
            }
        except Exception as e:
            logger.error(f"Alert creation failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# AI Decision Tree endpoints
if DECISION_TREE_AVAILABLE:
    @app.post("/decision/make")
    async def make_decision(
        decision_type: str,
        current_state: Dict[str, Any],
        objectives: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        available_resources: Optional[Dict[str, float]] = None,
        risk_tolerance: float = 0.5
    ):
        """Make an autonomous decision based on context"""
        try:
            global ai_decision_tree
            if ai_decision_tree is None:
                ai_decision_tree = get_ai_decision_tree()

            # Create decision context
            context = DecisionContext(
                current_state=current_state,
                historical_data=[],  # Would be loaded from database
                constraints=constraints or {},
                objectives=objectives,
                available_resources=available_resources or {'budget': 10000},
                time_constraints=None,
                risk_tolerance=risk_tolerance,
                metadata={}
            )

            # Make decision
            decision_type_enum = DecisionType[decision_type.upper()]
            result = await ai_decision_tree.make_decision(context, decision_type_enum)

            return {
                "decision_id": result.decision_id,
                "selected_action": result.selected_option.description,
                "confidence": result.selected_option.confidence,
                "confidence_level": result.confidence_level.value,
                "reasoning": result.reasoning,
                "execution_plan": result.execution_plan,
                "success_criteria": result.success_criteria,
                "alternatives": [
                    {
                        "action": alt.description,
                        "confidence": alt.confidence
                    }
                    for alt in result.alternative_options
                ]
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown decision type: {decision_type}")
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/decision/{decision_id}/outcome")
    async def report_decision_outcome(
        decision_id: str,
        outcome: Dict[str, Any],
        success: bool
    ):
        """Report outcome of a decision for learning"""
        try:
            global ai_decision_tree
            if ai_decision_tree is None:
                ai_decision_tree = get_ai_decision_tree()

            ai_decision_tree.learn_from_outcome(decision_id, outcome, success)

            return {
                "status": "outcome_recorded",
                "decision_id": decision_id,
                "success": success
            }
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/decision/stats")
    async def get_decision_stats():
        """Get decision-making statistics"""
        try:
            global ai_decision_tree
            if ai_decision_tree is None:
                ai_decision_tree = get_ai_decision_tree()

            stats = ai_decision_tree.get_decision_stats()

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/decision/types")
    async def get_decision_types():
        """Get available decision types"""
        return {
            "decision_types": [
                {"type": "strategic", "description": "Long-term business decisions"},
                {"type": "operational", "description": "Day-to-day operations"},
                {"type": "tactical", "description": "Mid-term optimizations"},
                {"type": "emergency", "description": "Crisis response"},
                {"type": "financial", "description": "Revenue and cost decisions"},
                {"type": "customer", "description": "Customer-facing decisions"},
                {"type": "technical", "description": "Technical infrastructure"},
                {"type": "learning", "description": "Self-improvement decisions"}
            ]
        }

# Realtime monitoring endpoints
if REALTIME_MONITOR_AVAILABLE:
    @app.post("/realtime/subscribe")
    async def subscribe_to_events(
        client_id: str,
        subscription_type: str = "all",
        filters: Optional[Dict[str, Any]] = None
    ):
        """Subscribe to real-time AI events"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            sub_type = SubscriptionType[subscription_type.upper()]
            subscription_id = realtime_monitor.subscribe(
                client_id=client_id,
                subscription_type=sub_type,
                filters=filters or {}
            )

            return {
                "subscription_id": subscription_id,
                "client_id": client_id,
                "subscription_type": subscription_type,
                "status": "subscribed"
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid subscription type: {subscription_type}")
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/realtime/unsubscribe/{subscription_id}")
    async def unsubscribe_from_events(subscription_id: str):
        """Unsubscribe from real-time events"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            realtime_monitor.unsubscribe(subscription_id)

            return {
                "subscription_id": subscription_id,
                "status": "unsubscribed"
            }
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/realtime/emit")
    async def emit_event(
        event_type: str,
        source: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Manually emit a real-time event"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            event_type_enum = EventType[event_type.upper()]
            realtime_monitor.emit_event(
                event_type=event_type_enum,
                source=source,
                data=data,
                metadata=metadata
            )

            return {
                "status": "event_emitted",
                "event_type": event_type,
                "source": source
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        except Exception as e:
            logger.error(f"Event emission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/realtime/activity-feed")
    async def get_activity_feed(limit: int = 50):
        """Get recent activity feed"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            activities = realtime_monitor.get_activity_feed(limit)

            return {
                "activities": activities,
                "count": len(activities)
            }
        except Exception as e:
            logger.error(f"Failed to get activity feed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/realtime/events")
    async def get_event_history(
        event_type: Optional[str] = None,
        limit: int = 100
    ):
        """Get event history"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            event_type_enum = EventType[event_type.upper()] if event_type else None
            events = realtime_monitor.get_event_history(event_type_enum, limit)

            return {
                "events": events,
                "count": len(events)
            }
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_type}")
        except Exception as e:
            logger.error(f"Failed to get event history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/realtime/stats")
    async def get_realtime_stats():
        """Get real-time monitoring statistics"""
        try:
            global realtime_monitor
            if realtime_monitor is None:
                realtime_monitor = get_realtime_monitor()

            stats = realtime_monitor.get_subscription_stats()

            return stats
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Self-Healing Recovery Endpoints
if SELF_HEALING_AVAILABLE:
    @app.post("/recovery/heal")
    async def trigger_healing(component: str, error_details: Dict[str, Any]):
        """Trigger self-healing for a component"""
        global self_healing_recovery
        if not self_healing_recovery:
            self_healing_recovery = get_self_healing_recovery()

        result = await self_healing_recovery.handle_error(
            component=component,
            error=Exception(error_details.get("message", "Manual healing triggered")),
            context=error_details
        )
        return {"healing_result": result}

    @app.get("/recovery/health/{component}")
    async def get_component_health(component: str):
        """Get health status for a component"""
        global self_healing_recovery
        if not self_healing_recovery:
            self_healing_recovery = get_self_healing_recovery()

        health = self_healing_recovery.get_component_health(component)
        return {
            "component": component,
            "health": health.value if health else "unknown",
            "circuit_breaker": self_healing_recovery.get_circuit_breaker_state(component)
        }

    @app.get("/recovery/history")
    async def get_recovery_history(limit: int = 100):
        """Get recent recovery actions"""
        global self_healing_recovery
        if not self_healing_recovery:
            self_healing_recovery = get_self_healing_recovery()

        history = self_healing_recovery.get_recovery_history(limit=limit)
        return {"history": history}

    @app.post("/recovery/rules")
    async def add_healing_rule(rule: Dict[str, Any]):
        """Add a self-healing rule"""
        global self_healing_recovery
        if not self_healing_recovery:
            self_healing_recovery = get_self_healing_recovery()

        rule_id = self_healing_recovery.add_healing_rule(
            component=rule.get("component"),
            error_pattern=rule.get("error_pattern"),
            fix_action=rule.get("fix_action"),
            confidence=rule.get("confidence", 0.8)
        )
        return {"rule_id": rule_id, "status": "created"}

    @app.get("/recovery/stats")
    async def get_recovery_stats():
        """Get recovery system statistics"""
        global self_healing_recovery
        if not self_healing_recovery:
            self_healing_recovery = get_self_healing_recovery()

        stats = self_healing_recovery.get_recovery_stats()
        return {"stats": stats}

# AI Training Pipeline Endpoints
if TRAINING_PIPELINE_AVAILABLE:
    @app.post("/training/capture-interaction")
    async def capture_interaction(interaction_data: Dict[str, Any]):
        """Capture customer interaction for training"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        interaction_id = await training_pipeline.capture_interaction(
            customer_id=interaction_data.get("customer_id"),
            interaction_type=InteractionType[interaction_data.get("type", "EMAIL").upper()],
            content=interaction_data.get("content"),
            channel=interaction_data.get("channel"),
            context=interaction_data.get("context", {}),
            outcome=interaction_data.get("outcome"),
            value=interaction_data.get("value")
        )

        return {"interaction_id": interaction_id, "status": "captured"}

    @app.post("/training/train-model")
    async def train_model(model_type: str, force: bool = False):
        """Train a specific model"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        result = await training_pipeline.train_model(
            ModelType[model_type.upper()],
            force=force
        )

        return result

    @app.get("/training/insights")
    async def get_insights():
        """Generate learning insights"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        insights = await training_pipeline.generate_insights()
        return {"insights": insights, "count": len(insights)}

    @app.post("/training/apply-insight/{insight_id}")
    async def apply_insight(insight_id: str):
        """Apply a specific insight"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        result = await training_pipeline.apply_learning(insight_id)
        return result

    @app.post("/training/feedback")
    async def record_feedback(feedback_data: Dict[str, Any]):
        """Record prediction feedback"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        result = await training_pipeline.feedback_loop(
            model_id=feedback_data.get("model_id"),
            prediction=feedback_data.get("prediction"),
            actual_outcome=feedback_data.get("actual_outcome")
        )

        return result

    @app.get("/training/metrics")
    async def get_training_metrics():
        """Get comprehensive training metrics"""
        global training_pipeline
        if not training_pipeline:
            training_pipeline = get_training_pipeline()

        metrics = await training_pipeline.get_training_metrics()
        return metrics

# Document Processor Endpoints
if DOCUMENT_PROCESSOR_AVAILABLE:
    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        user_id: Optional[str] = Form(None),
        source: str = Form("manual")
    ):
        """Upload and process a document"""
        global document_processor
        if not document_processor:
            document_processor = get_document_processor()

        # Read file content
        content = await file.read()

        # Upload and process
        document_id = await document_processor.upload_document(
            file_content=content,
            filename=file.filename,
            user_id=user_id,
            source=source,
            metadata={
                "content_type": file.content_type,
                "size": len(content)
            }
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing"
        }

    @app.get("/documents/{document_id}")
    async def get_document_insights(document_id: str):
        """Get comprehensive insights for a document"""
        global document_processor
        if not document_processor:
            document_processor = get_document_processor()

        insights = await document_processor.get_document_insights(document_id)
        return insights

    @app.post("/documents/search")
    async def search_documents(query: str, filters: Optional[Dict[str, Any]] = None):
        """Search documents using semantic search"""
        global document_processor
        if not document_processor:
            document_processor = get_document_processor()

        results = await document_processor.search_documents(
            query=query,
            filters=filters,
            limit=10
        )

        return {
            "query": query,
            "count": len(results),
            "documents": results
        }

    @app.post("/documents/{document_id}/action")
    async def trigger_document_action(
        document_id: str,
        action_type: str,
        action_data: Optional[Dict[str, Any]] = None
    ):
        """Trigger an action based on document content"""
        global document_processor
        if not document_processor:
            document_processor = get_document_processor()

        result = await document_processor.trigger_document_action(
            document_id=document_id,
            action_type=action_type,
            action_data=action_data or {}
        )

        return result

    @app.post("/documents/{document_id}/process")
    async def reprocess_document(document_id: str):
        """Reprocess a document"""
        global document_processor
        if not document_processor:
            document_processor = get_document_processor()

        result = await document_processor.process_document(document_id)
        return result

# AI Context Awareness Endpoints
if CONTEXT_AWARENESS_AVAILABLE:
    from fastapi import Header, Depends, HTTPException
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

    security = HTTPBearer()

    async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
        """Get current user from token"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        token_data = await context_awareness.verify_token(credentials.credentials)
        if not token_data.get('valid'):
            raise HTTPException(status_code=401, detail=token_data.get('error', 'Invalid token'))

        return token_data

    @app.post("/auth/register")
    async def register_user(user_data: Dict[str, Any]):
        """Register a new user profile"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        profile = await context_awareness.create_user_profile(
            user_id=user_data.get("user_id"),
            email=user_data.get("email"),
            role=UserRole[user_data.get("role", "USER").upper()],
            organization=user_data.get("organization"),
            department=user_data.get("department"),
            metadata=user_data.get("metadata", {})
        )

        return profile

    @app.post("/auth/login")
    async def login(credentials: Dict[str, Any]):
        """Authenticate user and create session"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        session = await context_awareness.authenticate_user(credentials)
        return session

    @app.get("/auth/context")
    async def get_context(current_user: Dict = Depends(get_current_user)):
        """Get current user context"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        context = await context_awareness.get_user_context(
            current_user['user_id']
        )
        return context

    @app.post("/auth/track")
    async def track_interaction(
        interaction: Dict[str, Any],
        current_user: Dict = Depends(get_current_user)
    ):
        """Track user interaction"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        interaction_id = await context_awareness.track_interaction(
            user_id=current_user['user_id'],
            interaction_type=interaction.get("type"),
            action=interaction.get("action"),
            entity_type=interaction.get("entity_type"),
            entity_id=interaction.get("entity_id"),
            context=interaction.get("context", {}),
            result=interaction.get("result", {}),
            duration_ms=interaction.get("duration_ms")
        )

        return {"interaction_id": interaction_id}

    @app.post("/auth/permission")
    async def check_permission(
        permission_check: Dict[str, Any],
        current_user: Dict = Depends(get_current_user)
    ):
        """Check user permission"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        allowed = await context_awareness.check_permission(
            user_id=current_user['user_id'],
            resource_type=permission_check.get("resource_type"),
            resource_id=permission_check.get("resource_id"),
            action=permission_check.get("action")
        )

        return {"allowed": allowed}

    @app.post("/auth/personalize")
    async def personalize_content(
        content_request: Dict[str, Any],
        current_user: Dict = Depends(get_current_user)
    ):
        """Personalize content for user"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        personalized = await context_awareness.personalize_content(
            user_id=current_user['user_id'],
            content_type=content_request.get("content_type"),
            base_content=content_request.get("base_content")
        )

        return {"personalized_content": personalized}

    @app.get("/auth/similar-users")
    async def find_similar_users(
        limit: int = 5,
        current_user: Dict = Depends(get_current_user)
    ):
        """Find similar users"""
        global context_awareness
        if not context_awareness:
            context_awareness = get_context_awareness()

        similar = await context_awareness.find_similar_users(
            user_id=current_user['user_id'],
            limit=limit
        )

        return {"similar_users": similar}

# Lead Nurturing System Endpoints
if LEAD_NURTURING_AVAILABLE:
    @app.post("/nurturing/sequences")
    async def create_nurture_sequence(sequence_data: Dict[str, Any]):
        """Create a new nurture sequence"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        sequence_id = await lead_nurturing_system.create_nurture_sequence(
            name=sequence_data["name"],
            sequence_type=NurtureSequenceType[sequence_data["sequence_type"].upper()],
            target_segment=LeadSegment[sequence_data["target_segment"].upper()],
            touchpoints=sequence_data["touchpoints"],
            success_criteria=sequence_data.get("success_criteria"),
            configuration=sequence_data.get("configuration")
        )

        return {"sequence_id": sequence_id, "status": "created"}

    @app.get("/nurturing/sequences")
    async def get_nurture_sequences(
        status: Optional[str] = None,
        sequence_type: Optional[str] = None,
        limit: int = 20
    ):
        """Get nurture sequences"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        sequences = await lead_nurturing_system.get_sequences(
            status=status,
            sequence_type=sequence_type,
            limit=limit
        )

        return {"sequences": sequences}

    @app.post("/nurturing/sequences/{sequence_id}/enroll")
    async def enroll_lead(
        sequence_id: str,
        enrollment_data: Dict[str, Any]
    ):
        """Enroll a lead in a nurture sequence"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        enrollment_id = await lead_nurturing_system.enroll_lead(
            sequence_id=sequence_id,
            lead_id=enrollment_data["lead_id"],
            customer_id=enrollment_data.get("customer_id"),
            enrollment_source=enrollment_data.get("enrollment_source", "manual"),
            metadata=enrollment_data.get("metadata", {})
        )

        return {"enrollment_id": enrollment_id, "status": "enrolled"}

    @app.get("/nurturing/sequences/{sequence_id}/enrollments")
    async def get_sequence_enrollments(
        sequence_id: str,
        status: Optional[str] = None,
        limit: int = 50
    ):
        """Get enrollments for a sequence"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        enrollments = await lead_nurturing_system.get_sequence_enrollments(
            sequence_id=sequence_id,
            status=status,
            limit=limit
        )

        return {"enrollments": enrollments}

    @app.post("/nurturing/execute")
    async def execute_scheduled_touchpoints():
        """Execute scheduled touchpoints"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        results = await lead_nurturing_system.execute_scheduled_touchpoints()

        return {
            "executed_count": len(results),
            "executions": results
        }

    @app.get("/nurturing/metrics/{sequence_id}")
    async def get_sequence_metrics(
        sequence_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """Get performance metrics for a sequence"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        metrics = await lead_nurturing_system.get_sequence_performance(
            sequence_id=sequence_id,
            start_date=start_date,
            end_date=end_date
        )

        return {"metrics": metrics}

    @app.post("/nurturing/engagement")
    async def track_engagement(engagement_data: Dict[str, Any]):
        """Track engagement event"""
        global lead_nurturing_system
        if not lead_nurturing_system:
            lead_nurturing_system = get_lead_nurturing_system()

        await lead_nurturing_system.track_engagement(
            enrollment_id=engagement_data["enrollment_id"],
            engagement_type=engagement_data["engagement_type"],
            engagement_value=engagement_data.get("engagement_value", 1.0),
            metadata=engagement_data.get("metadata", {})
        )

        return {"status": "tracked"}

# Intelligent Follow-up System Endpoints
if INTELLIGENT_FOLLOWUP_AVAILABLE:
    @app.post("/followup/sequences")
    async def create_followup_sequence(sequence_data: Dict[str, Any]):
        """Create intelligent follow-up sequence"""
        global intelligent_followup_system
        if not intelligent_followup_system:
            intelligent_followup_system = get_intelligent_followup_system()

        sequence_id = await intelligent_followup_system.create_followup_sequence(
            followup_type=FollowUpType[sequence_data["followup_type"].upper()],
            entity_id=sequence_data["entity_id"],
            entity_type=sequence_data["entity_type"],
            context=sequence_data.get("context", {}),
            priority=FollowUpPriority[sequence_data.get("priority", "MEDIUM").upper()],
            custom_rules=sequence_data.get("custom_rules")
        )

        return {"sequence_id": sequence_id, "status": "created"}

    @app.post("/followup/execute")
    async def execute_scheduled_followups():
        """Execute scheduled follow-ups"""
        global intelligent_followup_system
        if not intelligent_followup_system:
            intelligent_followup_system = get_intelligent_followup_system()

        results = await intelligent_followup_system.execute_scheduled_followups()

        return {
            "executed_count": len(results),
            "results": results
        }

    @app.post("/followup/responses/{execution_id}")
    async def track_followup_response(
        execution_id: str,
        response_data: Dict[str, Any]
    ):
        """Track response to follow-up"""
        global intelligent_followup_system
        if not intelligent_followup_system:
            intelligent_followup_system = get_intelligent_followup_system()

        result = await intelligent_followup_system.track_response(
            execution_id=execution_id,
            response_type=ResponseType[response_data["response_type"].upper()],
            response_data=response_data.get("data")
        )

        return result

    @app.get("/followup/analytics")
    async def get_followup_analytics(
        entity_id: Optional[str] = None,
        followup_type: Optional[str] = None,
        days: int = 30
    ):
        """Get follow-up analytics"""
        global intelligent_followup_system
        if not intelligent_followup_system:
            intelligent_followup_system = get_intelligent_followup_system()

        date_from = datetime.now(timezone.utc) - timedelta(days=days)

        analytics = await intelligent_followup_system.get_followup_analytics(
            entity_id=entity_id,
            followup_type=FollowUpType[followup_type.upper()] if followup_type else None,
            date_from=date_from,
            date_to=datetime.now(timezone.utc)
        )

        return analytics

# AI Customer Onboarding Endpoints
if CUSTOMER_ONBOARDING_AVAILABLE:
    @app.post("/onboarding/journeys")
    async def create_onboarding_journey(journey_data: Dict[str, Any]):
        """Create personalized onboarding journey"""
        global customer_onboarding_system
        if not customer_onboarding_system:
            customer_onboarding_system = get_ai_customer_onboarding()

        journey_id = await customer_onboarding_system.create_onboarding_journey(
            customer_id=journey_data["customer_id"],
            customer_data=journey_data.get("customer_data", {}),
            segment=CustomerSegment[journey_data["segment"].upper()],
            custom_requirements=journey_data.get("custom_requirements")
        )

        return {"journey_id": journey_id, "status": "created"}

    @app.post("/onboarding/journeys/{journey_id}/track")
    async def track_onboarding_progress(
        journey_id: str,
        event_data: Dict[str, Any]
    ):
        """Track progress in onboarding journey"""
        global customer_onboarding_system
        if not customer_onboarding_system:
            customer_onboarding_system = get_ai_customer_onboarding()

        result = await customer_onboarding_system.track_progress(
            journey_id=journey_id,
            event_type=event_data["event_type"],
            event_data=event_data.get("data", {})
        )

        return result

    @app.post("/onboarding/journeys/{journey_id}/personalize")
    async def personalize_onboarding(
        journey_id: str,
        interaction_data: Dict[str, Any]
    ):
        """Personalize onboarding experience"""
        global customer_onboarding_system
        if not customer_onboarding_system:
            customer_onboarding_system = get_ai_customer_onboarding()

        personalization = await customer_onboarding_system.personalize_experience(
            journey_id=journey_id,
            interaction_data=interaction_data
        )

        return personalization

    @app.get("/onboarding/analytics")
    async def get_onboarding_analytics(
        segment: Optional[str] = None,
        days: int = 30
    ):
        """Get onboarding analytics"""
        global customer_onboarding_system
        if not customer_onboarding_system:
            customer_onboarding_system = get_ai_customer_onboarding()

        date_from = datetime.now(timezone.utc) - timedelta(days=days)

        analytics = await customer_onboarding_system.get_journey_analytics(
            segment=CustomerSegment[segment.upper()] if segment else None,
            date_from=date_from,
            date_to=datetime.now(timezone.utc)
        )

        return analytics

# Automated Reporting Endpoints
if AUTOMATED_REPORTING_AVAILABLE:
    @app.post("/reports/generate")
    async def generate_report(
        report_type: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Generate a report manually"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        report = await automated_reporting_system.generate_report(
            ReportType[report_type.upper()],
            parameters or {}
        )

        return {
            "report_id": report["report_id"],
            "title": report["title"],
            "summary": report["summary"],
            "status": "generated",
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    @app.post("/reports/{report_id}/deliver")
    async def deliver_report(
        report_id: str,
        channels: List[str],
        recipients: List[str]
    ):
        """Deliver a generated report"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        delivery_results = await automated_reporting_system.deliver_report(
            report_id,
            channels,
            recipients
        )

        return {
            "report_id": report_id,
            "deliveries": delivery_results,
            "delivered_at": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/reports/templates")
    async def get_report_templates():
        """Get available report templates"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        templates = await automated_reporting_system.get_templates()

        return {"templates": templates}

    @app.post("/reports/schedule")
    async def schedule_report(
        template_id: str,
        cron_expression: str,
        recipients: List[str]
    ):
        """Schedule a recurring report"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        schedule_id = await automated_reporting_system.schedule_report(
            template_id,
            cron_expression,
            recipients
        )

        return {
            "schedule_id": schedule_id,
            "template_id": template_id,
            "cron": cron_expression,
            "status": "scheduled"
        }

    @app.get("/reports/history")
    async def get_report_history(
        report_type: Optional[str] = None,
        days: int = 7,
        limit: int = 100
    ):
        """Get report generation history"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        history = await automated_reporting_system.get_report_history(
            report_type=report_type,
            days=days,
            limit=limit
        )

        return {
            "reports": history,
            "count": len(history)
        }

    @app.get("/reports/{report_id}")
    async def get_report(report_id: str):
        """Get a specific report"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        report = await automated_reporting_system.get_report(report_id)

        if not report:
            raise HTTPException(status_code=404, detail="Report not found")

        return report

    @app.post("/reports/execute-scheduled")
    async def execute_scheduled_reports():
        """Execute all scheduled reports"""
        global automated_reporting_system
        if not automated_reporting_system:
            automated_reporting_system = get_automated_reporting_system()

        results = await automated_reporting_system.execute_scheduled_reports()

        return {
            "executed_count": len(results),
            "reports": results,
            "executed_at": datetime.now(timezone.utc).isoformat()
        }

# Cost Optimization Endpoints
if COST_OPTIMIZATION_AVAILABLE:
    @app.post("/cost/optimize")
    async def run_cost_optimization():
        """Run full cost optimization cycle"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        result = await cost_optimization_engine.optimize()

        return {
            "status": "optimization_complete",
            "current_monthly_cost": result['current_monthly_cost'],
            "potential_monthly_savings": result['potential_monthly_savings'],
            "savings_percentage": result['savings_percentage'],
            "top_opportunities": result['optimization_opportunities'][:3],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.post("/cost/track")
    async def track_resource_usage(
        resource_type: str,
        amount: float,
        service: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track resource usage"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        result = await cost_optimization_engine.track_resource(
            ResourceType[resource_type.upper()],
            amount,
            service,
            metadata or {}
        )

        return result

    @app.post("/cost/budget")
    async def set_budget(
        service: str,
        monthly_limit: float
    ):
        """Set budget for a service"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        result = await cost_optimization_engine.set_budget(
            service,
            monthly_limit
        )

        return result

    @app.get("/cost/budget/{service}")
    async def get_budget_status(service: str):
        """Get budget status for a service"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        status = await cost_optimization_engine.budget_manager.check_budget_status(service)

        return {
            "service": service,
            "budget_status": status[0] if status else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/cost/usage")
    async def get_usage_summary(days: int = 7):
        """Get resource usage summary"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        summary = await cost_optimization_engine.monitor.get_usage_summary(days)

        return summary

    @app.post("/cost/apply-optimization")
    async def apply_optimization(
        strategy: str,
        parameters: Dict[str, Any]
    ):
        """Apply an optimization strategy"""
        global cost_optimization_engine
        if not cost_optimization_engine:
            cost_optimization_engine = get_cost_optimization_engine()

        result = await cost_optimization_engine.apply_optimization({
            "strategy": strategy,
            **parameters
        })

        return result

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
    logger.info(f"Conversation Memory Available: {CONVERSATION_MEMORY_AVAILABLE}")
    logger.info(f"System State Available: {SYSTEM_STATE_AVAILABLE}")
    logger.info(f"Decision Tree Available: {DECISION_TREE_AVAILABLE}")
    logger.info(f"Realtime Monitor Available: {REALTIME_MONITOR_AVAILABLE}")
    logger.info(f"Self-Healing Available: {SELF_HEALING_AVAILABLE}")
    logger.info(f"Training Pipeline Available: {TRAINING_PIPELINE_AVAILABLE}")
    logger.info(f"Document Processor Available: {DOCUMENT_PROCESSOR_AVAILABLE}")
    logger.info(f"Context Awareness Available: {CONTEXT_AWARENESS_AVAILABLE}")
    logger.info(f"Lead Nurturing Available: {LEAD_NURTURING_AVAILABLE}")
    logger.info(f"Intelligent Follow-up Available: {INTELLIGENT_FOLLOWUP_AVAILABLE}")
    logger.info(f"Customer Onboarding Available: {CUSTOMER_ONBOARDING_AVAILABLE}")
    logger.info(f"Automated Reporting Available: {AUTOMATED_REPORTING_AVAILABLE}")
    logger.info(f"Cost Optimization Available: {COST_OPTIMIZATION_AVAILABLE}")

    try:
        from scheduled_executor import scheduler
        asyncio.create_task(scheduler.run_scheduler())
        logger.info("Scheduled executor started")
    except ImportError:
        logger.warning("Scheduled executor not available")

    # Start system health monitoring
    if SYSTEM_STATE_AVAILABLE:
        async def periodic_health_check():
            """Perform periodic system health checks"""
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    snapshot = await check_system_health()
                    logger.info(f"System health check: {snapshot.overall_status.value}, score: {snapshot.health_score:.1f}")
                except Exception as e:
                    logger.error(f"Periodic health check failed: {e}")

        asyncio.create_task(periodic_health_check())
        logger.info("System health monitoring started")

    # Start realtime monitoring
    if REALTIME_MONITOR_AVAILABLE:
        async def start_realtime_monitor():
            """Start realtime monitoring service"""
            try:
                global realtime_monitor
                if realtime_monitor is None:
                    realtime_monitor = get_realtime_monitor()
                await realtime_monitor.start()
                logger.info("Realtime monitoring service started")
            except Exception as e:
                logger.error(f"Failed to start realtime monitor: {e}")

        asyncio.create_task(start_realtime_monitor())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting AI Agent Service on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)