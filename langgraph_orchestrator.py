#!/usr/bin/env python3
"""
LangChain/LangGraph Orchestration Layer
Implements graph-based AI agent workflow orchestration
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Annotated, Sequence
from datetime import datetime, timezone
from enum import Enum
import operator

# LangChain/LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph, END

import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),
    "port": int(os.getenv("DB_PORT", 5432))
}

class AgentState(TypedDict):
    """State for AI agent workflows"""
    messages: Sequence[BaseMessage]
    context: Dict[str, Any]
    current_agent: str
    workflow_stage: str
    memory_context: List[Dict]
    decisions: List[Dict]
    outputs: List[Any]
    errors: List[str]
    metadata: Dict[str, Any]

class WorkflowStage(Enum):
    """Workflow stages for agent orchestration"""
    INITIALIZATION = "initialization"
    MEMORY_RETRIEVAL = "memory_retrieval"
    CONTEXT_BUILDING = "context_building"
    AGENT_SELECTION = "agent_selection"
    TASK_EXECUTION = "task_execution"
    MEMORY_STORAGE = "memory_storage"
    RESPONSE_GENERATION = "response_generation"
    COMPLETION = "completion"

class LangGraphOrchestrator:
    """Main orchestrator using LangGraph for complex AI workflows"""

    def __init__(self):
        """Initialize the orchestrator with LLMs and memory systems"""
        # Initialize LLMs
        self.openai_llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.anthropic_llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            temperature=0.7,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize vector store for memory
        self.vector_store = self._init_vector_store()

        # Initialize workflow graph
        self.workflow = self._build_workflow_graph()

        # Database connection
        self.db_conn = None

        logger.info("LangGraph Orchestrator initialized successfully")

    def _init_vector_store(self):
        """Initialize Supabase vector store for semantic memory"""
        try:
            # Try to import supabase client for vector store
            from supabase import create_client
            import os

            supabase_url = os.getenv('SUPABASE_URL', '')
            supabase_key = os.getenv('SUPABASE_KEY', '')

            if supabase_url and supabase_key:
                client = create_client(supabase_url, supabase_key)
                vector_store = SupabaseVectorStore(
                    client=client,
                    table_name="ai_memory_vectors",
                    embedding=self.embeddings,
                    query_name="match_documents"
                )
                return vector_store
            else:
                logger.warning("Supabase URL/Key not configured, vector store disabled")
                return None
        except ImportError:
            logger.warning("Supabase client not installed, vector store disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            return None

    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow for agent orchestration"""
        workflow = StateGraph(AgentState)

        # Add nodes for each workflow stage
        workflow.add_node("initialize", self.initialize_workflow)
        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("build_context", self.build_context)
        workflow.add_node("select_agent", self.select_agent)
        workflow.add_node("execute_task", self.execute_task)
        workflow.add_node("store_memory", self.store_memory)
        workflow.add_node("generate_response", self.generate_response)

        # Define edges (workflow transitions)
        workflow.add_edge("initialize", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "build_context")
        workflow.add_edge("build_context", "select_agent")
        workflow.add_edge("select_agent", "execute_task")
        workflow.add_edge("execute_task", "store_memory")
        workflow.add_edge("store_memory", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set entry point
        workflow.set_entry_point("initialize")

        return workflow.compile()

    async def initialize_workflow(self, state: AgentState) -> AgentState:
        """Initialize the workflow with initial state"""
        logger.info("Initializing workflow")

        state["workflow_stage"] = WorkflowStage.INITIALIZATION.value
        state["metadata"]["start_time"] = datetime.now(timezone.utc).isoformat()
        state["errors"] = []
        state["outputs"] = []

        return state

    async def retrieve_memory(self, state: AgentState) -> AgentState:
        """Retrieve relevant memories from vector store"""
        logger.info("Retrieving relevant memories")

        state["workflow_stage"] = WorkflowStage.MEMORY_RETRIEVAL.value

        try:
            if self.vector_store and state["messages"]:
                # Get the last message for context
                last_message = state["messages"][-1].content

                # Retrieve similar memories
                similar_docs = self.vector_store.similarity_search(
                    last_message,
                    k=5
                )

                state["memory_context"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in similar_docs
                ]
                logger.info(f"Retrieved {len(similar_docs)} relevant memories")
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            state["errors"].append(str(e))

        return state

    async def build_context(self, state: AgentState) -> AgentState:
        """Build comprehensive context from memories and current state"""
        logger.info("Building context from memories and state")

        state["workflow_stage"] = WorkflowStage.CONTEXT_BUILDING.value

        # Combine memories, messages, and metadata into context
        context = {
            "conversation_history": [msg.content for msg in state["messages"][-10:]],
            "relevant_memories": state["memory_context"],
            "current_timestamp": datetime.now(timezone.utc).isoformat(),
            "session_metadata": state["metadata"]
        }

        state["context"] = context

        return state

    async def select_agent(self, state: AgentState) -> AgentState:
        """Select the appropriate agent based on context"""
        logger.info("Selecting appropriate agent for task")

        state["workflow_stage"] = WorkflowStage.AGENT_SELECTION.value

        # Agent selection prompt
        selection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an agent selector. Based on the context and user request,
                         select the most appropriate agent type:
                         - analyzer: For data analysis and insights
                         - workflow: For multi-step processes
                         - revenue: For sales and revenue generation
                         - customer: For customer service and support
                         - technical: For technical implementation

                         Respond with just the agent type."""),
            HumanMessage(content=f"Context: {json.dumps(state['context'])}\n\nSelect agent:")
        ])

        response = await self.openai_llm.ainvoke(selection_prompt.format_messages())
        selected_agent = response.content.strip().lower()

        state["current_agent"] = selected_agent
        logger.info(f"Selected agent: {selected_agent}")

        return state

    async def execute_task(self, state: AgentState) -> AgentState:
        """Execute the task with the selected agent"""
        logger.info(f"Executing task with agent: {state['current_agent']}")

        state["workflow_stage"] = WorkflowStage.TASK_EXECUTION.value

        try:
            # Build agent-specific prompt
            agent_prompt = self._get_agent_prompt(state["current_agent"])

            # Execute with appropriate LLM
            if state["current_agent"] in ["revenue", "customer"]:
                # Use Claude for customer-facing tasks
                response = await self.anthropic_llm.ainvoke(
                    agent_prompt.format_messages(
                        context=json.dumps(state["context"]),
                        messages=state["messages"]
                    )
                )
            else:
                # Use GPT-4 for analytical tasks
                response = await self.openai_llm.ainvoke(
                    agent_prompt.format_messages(
                        context=json.dumps(state["context"]),
                        messages=state["messages"]
                    )
                )

            state["outputs"].append({
                "agent": state["current_agent"],
                "response": response.content,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Store execution in database
            await self._store_execution(state)

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            state["errors"].append(str(e))

        return state

    async def store_memory(self, state: AgentState) -> AgentState:
        """Store important information in vector memory"""
        logger.info("Storing execution results in memory")

        state["workflow_stage"] = WorkflowStage.MEMORY_STORAGE.value

        try:
            if self.vector_store and state["outputs"]:
                # Create memory document
                last_output = state["outputs"][-1]
                memory_doc = Document(
                    page_content=f"Agent: {last_output['agent']}\nResponse: {last_output['response']}",
                    metadata={
                        "agent": last_output["agent"],
                        "timestamp": last_output["timestamp"],
                        "workflow_id": state["metadata"].get("workflow_id")
                    }
                )

                # Store in vector database
                self.vector_store.add_documents([memory_doc])
                logger.info("Memory stored successfully")

        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            state["errors"].append(str(e))

        return state

    async def generate_response(self, state: AgentState) -> AgentState:
        """Generate final response from workflow outputs"""
        logger.info("Generating final response")

        state["workflow_stage"] = WorkflowStage.RESPONSE_GENERATION.value

        # Compile all outputs into final response
        if state["outputs"]:
            state["metadata"]["final_response"] = state["outputs"][-1]["response"]
        else:
            state["metadata"]["final_response"] = "Workflow completed but no output generated"

        state["metadata"]["end_time"] = datetime.now(timezone.utc).isoformat()
        state["workflow_stage"] = WorkflowStage.COMPLETION.value

        return state

    def _get_agent_prompt(self, agent_type: str) -> ChatPromptTemplate:
        """Get specialized prompt for each agent type"""
        prompts = {
            "analyzer": ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a data analysis expert. Analyze the provided context and data to generate insights."),
                HumanMessage(content="Context: {context}\n\nMessages: {messages}")
            ]),
            "workflow": ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a workflow orchestration expert. Design and execute multi-step processes."),
                HumanMessage(content="Context: {context}\n\nMessages: {messages}")
            ]),
            "revenue": ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a revenue generation expert. Focus on identifying and closing sales opportunities."),
                HumanMessage(content="Context: {context}\n\nMessages: {messages}")
            ]),
            "customer": ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a customer service expert. Provide helpful and empathetic support."),
                HumanMessage(content="Context: {context}\n\nMessages: {messages}")
            ]),
            "technical": ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a technical implementation expert. Provide detailed technical solutions."),
                HumanMessage(content="Context: {context}\n\nMessages: {messages}")
            ])
        }

        return prompts.get(agent_type, prompts["workflow"])

    async def _store_execution(self, state: AgentState):
        """Store agent execution in database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO agent_executions
                (agent_type, prompt, response, status, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                state["current_agent"],
                json.dumps(state["context"]),
                state["outputs"][-1]["response"] if state["outputs"] else "",
                "completed",
                datetime.now(timezone.utc)
            ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store execution: {e}")

    async def run_workflow(self, messages: List[BaseMessage], metadata: Dict = None) -> Dict:
        """Run the complete workflow"""
        initial_state = AgentState(
            messages=messages,
            context={},
            current_agent="",
            workflow_stage=WorkflowStage.INITIALIZATION.value,
            memory_context=[],
            decisions=[],
            outputs=[],
            errors=[],
            metadata=metadata or {}
        )

        # Run the workflow
        final_state = await self.workflow.ainvoke(initial_state)

        return {
            "success": len(final_state["errors"]) == 0,
            "response": final_state["metadata"].get("final_response"),
            "agent_used": final_state["current_agent"],
            "errors": final_state["errors"],
            "execution_time": final_state["metadata"].get("end_time")
        }

    async def execute(self, agent_name: str, prompt: str, tenant_id: str = None, context: Dict = None) -> Dict:
        """
        Execute an agent task - main entry point called by external systems.

        This method provides a simple interface for invoking the orchestrator
        with an agent name and prompt, handling all the internal workflow details.

        Args:
            agent_name: Name of the agent to invoke (e.g., 'workflow-orchestrator', 'crew-allocator')
            prompt: The task/prompt to execute
            tenant_id: Optional tenant ID for multi-tenancy
            context: Optional additional context

        Returns:
            Dict with success, response, agent_used, errors, execution_time
        """
        try:
            # Convert agent_name to internal agent type
            agent_type_map = {
                'workflow-orchestrator': 'workflow',
                'crew-allocator': 'workflow',
                'weather-monitor': 'analyzer',
                'customer-success': 'customer',
                'revenue-optimizer': 'revenue',
                'technical-support': 'technical',
                'data-analyst': 'analyzer',
            }

            # Build messages from prompt
            messages = [HumanMessage(content=prompt)]

            # Build metadata
            metadata = {
                'workflow_id': f"{agent_name}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                'requested_agent': agent_name,
                'tenant_id': tenant_id,
                'context': context or {}
            }

            # Force agent type if known
            mapped_type = agent_type_map.get(agent_name, 'workflow')

            # Run the workflow
            result = await self.run_workflow(messages, metadata)

            # If agent mapping caused a different agent, note it
            if result.get('agent_used') != mapped_type:
                result['requested_agent'] = agent_name

            logger.info(f"Executed agent {agent_name}: success={result.get('success')}")
            return result

        except Exception as e:
            logger.error(f"Execute failed for agent {agent_name}: {e}")
            return {
                'success': False,
                'response': None,
                'agent_used': agent_name,
                'errors': [str(e)],
                'execution_time': datetime.now(timezone.utc).isoformat()
            }

# Global orchestrator instance - create lazily
langgraph_orchestrator = None

def get_langgraph_orchestrator():
    """Get or create langgraph orchestrator instance"""
    global langgraph_orchestrator
    if langgraph_orchestrator is None:
        langgraph_orchestrator = LangGraphOrchestrator()
    return langgraph_orchestrator