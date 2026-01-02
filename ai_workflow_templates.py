#!/usr/bin/env python3
"""
AI Workflow Templates
Reusable workflow templates for common AI operations
"""

import asyncio
import json
import logging
import os
from urllib.parse import urlparse as _urlparse
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class WorkflowStatus(Enum):
    """Status of a workflow execution"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(Enum):
    """Type of workflow step"""
    ACTION = "action"           # Execute an action
    DECISION = "decision"       # Make a decision
    PARALLEL = "parallel"       # Run steps in parallel
    LOOP = "loop"               # Iterate over items
    CONDITION = "condition"     # Conditional execution
    WAIT = "wait"               # Wait for condition/time
    SUBPROCESS = "subprocess"   # Run a sub-workflow
    HUMAN_INPUT = "human_input" # Require human input


class WorkflowCategory(Enum):
    """Categories of workflows"""
    DATA_PROCESSING = "data_processing"
    CUSTOMER_SUPPORT = "customer_support"
    SALES_AUTOMATION = "sales_automation"
    CONTENT_GENERATION = "content_generation"
    SYSTEM_MAINTENANCE = "system_maintenance"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    INTEGRATION = "integration"
    CUSTOM = "custom"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class WorkflowStep:
    """A single step in a workflow"""
    step_id: str
    name: str
    step_type: StepType
    handler: Optional[str] = None  # Handler function name
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)  # Input mappings
    outputs: dict[str, str] = field(default_factory=dict)  # Output mappings
    conditions: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    on_failure: str = "fail"  # fail, skip, retry, continue
    next_steps: list[str] = field(default_factory=list)


@dataclass
class WorkflowTemplate:
    """A workflow template definition"""
    template_id: str
    name: str
    description: str
    category: WorkflowCategory
    version: str = "1.0.0"
    steps: list[WorkflowStep] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WorkflowExecution:
    """An execution instance of a workflow"""
    execution_id: str
    template_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    current_step: Optional[str] = None
    step_results: dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepExecution:
    """Execution state of a single step"""
    step_id: str
    execution_id: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0


# ============================================================================
# STEP HANDLERS
# ============================================================================

class StepHandler(ABC):
    """Base class for step handlers"""

    @abstractmethod
    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the step and return outputs"""
        pass


class ActionHandler(StepHandler):
    """Handler for action steps"""

    def __init__(self, action_registry: dict[str, Callable] = None):
        self.actions = action_registry or {}

    def register_action(self, name: str, func: Callable):
        """Register an action function"""
        self.actions[name] = func

    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        if step.handler not in self.actions:
            raise ValueError(f"Unknown action: {step.handler}")

        action = self.actions[step.handler]

        if asyncio.iscoroutinefunction(action):
            result = await action(inputs, context, step.config)
        else:
            result = action(inputs, context, step.config)

        return result if isinstance(result, dict) else {"result": result}


class DecisionHandler(StepHandler):
    """Handler for decision steps"""

    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        # Evaluate conditions
        for condition in step.conditions:
            field = condition.get("field")
            operator = condition.get("operator", "eq")
            value = condition.get("value")
            target = condition.get("target")

            input_value = inputs.get(field) or context.get(field)

            match = False
            if operator == "eq":
                match = input_value == value
            elif operator == "ne":
                match = input_value != value
            elif operator == "gt":
                match = input_value > value
            elif operator == "lt":
                match = input_value < value
            elif operator == "contains":
                match = value in str(input_value)
            elif operator == "exists":
                match = input_value is not None

            if match:
                return {"decision": target, "matched_condition": condition}

        return {"decision": step.config.get("default", "continue")}


class ParallelHandler(StepHandler):
    """Handler for parallel execution steps"""

    def __init__(self, workflow_engine):
        self.engine = workflow_engine

    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        parallel_steps = step.config.get("steps", [])

        tasks = []
        for sub_step_id in parallel_steps:
            sub_step = self.engine._get_step(step.config.get("template_id"), sub_step_id)
            if sub_step:
                tasks.append(
                    self.engine._execute_step(sub_step, context, inputs)
                )

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return {
                "parallel_results": results,
                "success": all(not isinstance(r, Exception) for r in results)
            }

        return {"parallel_results": [], "success": True}


class LoopHandler(StepHandler):
    """Handler for loop steps"""

    def __init__(self, workflow_engine):
        self.engine = workflow_engine

    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        items = inputs.get(step.config.get("items_field", "items"), [])
        loop_step_id = step.config.get("loop_step")
        results = []

        for i, item in enumerate(items):
            loop_context = {**context, "loop_index": i, "loop_item": item}
            loop_step = self.engine._get_step(step.config.get("template_id"), loop_step_id)

            if loop_step:
                result = await self.engine._execute_step(loop_step, loop_context, {"item": item})
                results.append(result)

        return {"loop_results": results, "count": len(results)}


class WaitHandler(StepHandler):
    """Handler for wait steps"""

    async def execute(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        wait_type = step.config.get("wait_type", "duration")

        if wait_type == "duration":
            seconds = step.config.get("seconds", 0)
            await asyncio.sleep(seconds)
            return {"waited_seconds": seconds}

        elif wait_type == "condition":
            # Would poll for condition
            max_wait = step.config.get("max_wait_seconds", 300)
            return {"condition_met": True, "waited_seconds": 0}

        elif wait_type == "event":
            # Would wait for event
            return {"event_received": True}

        return {}


# ============================================================================
# WORKFLOW ENGINE
# ============================================================================

class WorkflowEngine:
    """
    Engine for executing workflow templates
    """

    def __init__(self):
        self._initialized = False
        self._db_config = None

        # Template storage
        self._templates: dict[str, WorkflowTemplate] = {}

        # Execution tracking
        self._executions: dict[str, WorkflowExecution] = {}
        self._step_executions: dict[str, dict[str, StepExecution]] = {}

        # Handlers
        self._action_handler = ActionHandler()
        self._decision_handler = DecisionHandler()
        self._handlers: dict[StepType, StepHandler] = {
            StepType.ACTION: self._action_handler,
            StepType.DECISION: self._decision_handler,
        }

        # Callbacks
        self._on_step_complete: list[Callable] = []
        self._on_workflow_complete: list[Callable] = []

        self._lock = asyncio.Lock()

    def _get_db_config(self) -> dict[str, Any]:
        """Get database configuration lazily with validation"""
        if not self._db_config:
            required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
            missing = [var for var in required_vars if not os.getenv(var)]
            if missing:
                
        # DATABASE_URL fallback
        _db_url = os.getenv('DATABASE_URL', '')
        if _db_url:
            try:
                _p = _urlparse(_db_url)
                globals().update({'_DB_HOST': _p.hostname, '_DB_NAME': _p.path.lstrip('/'), '_DB_USER': _p.username, '_DB_PASSWORD': _p.password, '_DB_PORT': str(_p.port or 5432)})
            except: pass
        missing = [v for v in required_vars if not os.getenv(v) and not globals().get('_' + v)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

            self._db_config = {
                'host': os.getenv('DB_HOST'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', '5432'))
            }
        return self._db_config

    async def initialize(self):
        """Initialize the workflow engine"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                await self._initialize_database()
                await self._load_default_templates()
                self._setup_handlers()
                self._initialized = True
                logger.info("Workflow engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize workflow engine: {e}")

    async def _initialize_database(self):
        """Initialize database tables"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            # Workflow templates table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_workflow_templates (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    template_id VARCHAR(255) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    category VARCHAR(100) NOT NULL,
                    version VARCHAR(50) DEFAULT '1.0.0',
                    steps JSONB NOT NULL,
                    input_schema JSONB DEFAULT '{}'::jsonb,
                    output_schema JSONB DEFAULT '{}'::jsonb,
                    variables JSONB DEFAULT '{}'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Workflow executions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_workflow_executions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    execution_id VARCHAR(255) UNIQUE NOT NULL,
                    template_id VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    inputs JSONB DEFAULT '{}'::jsonb,
                    outputs JSONB DEFAULT '{}'::jsonb,
                    context JSONB DEFAULT '{}'::jsonb,
                    current_step VARCHAR(255),
                    step_results JSONB DEFAULT '{}'::jsonb,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error TEXT,
                    correlation_id VARCHAR(255),
                    tenant_id VARCHAR(255),
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Step executions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_workflow_step_executions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    step_id VARCHAR(255) NOT NULL,
                    execution_id VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    inputs JSONB DEFAULT '{}'::jsonb,
                    outputs JSONB DEFAULT '{}'::jsonb,
                    error TEXT,
                    retry_count INT DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(step_id, execution_id)
                )
            """)

            # Create indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_executions_status
                ON ai_workflow_executions(status)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_workflow_templates_category
                ON ai_workflow_templates(category)
            """)

            conn.commit()
            conn.close()
            logger.info("Workflow database tables initialized")

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def _setup_handlers(self):
        """Set up step handlers"""
        self._handlers[StepType.PARALLEL] = ParallelHandler(self)
        self._handlers[StepType.LOOP] = LoopHandler(self)
        self._handlers[StepType.WAIT] = WaitHandler()

        # Register default actions
        self._register_default_actions()

    def _register_default_actions(self):
        """Register default action handlers"""
        # Log action
        async def log_action(inputs, context, config):
            message = config.get("message", "").format(**inputs, **context)
            logger.info(f"Workflow log: {message}")
            return {"logged": True, "message": message}

        self._action_handler.register_action("log", log_action)

        # Transform action
        async def transform_action(inputs, context, config):
            mapping = config.get("mapping", {})
            result = {}
            for target, source in mapping.items():
                if source in inputs:
                    result[target] = inputs[source]
                elif source in context:
                    result[target] = context[source]
            return result

        self._action_handler.register_action("transform", transform_action)

        # HTTP request action
        async def http_action(inputs, context, config):
            import aiohttp
            url = config.get("url", "").format(**inputs, **context)
            method = config.get("method", "GET")
            headers = config.get("headers", {})
            body = config.get("body", {})

            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, json=body) as response:
                    return {
                        "status_code": response.status,
                        "body": await response.json() if response.status == 200 else None
                    }

        self._action_handler.register_action("http_request", http_action)

        # AI generation action
        async def ai_generate_action(inputs, context, config):
            prompt = config.get("prompt", "").format(**inputs, **context)
            # Would call AI integration
            return {"generated": True, "prompt": prompt, "response": f"AI response for: {prompt[:50]}..."}

        self._action_handler.register_action("ai_generate", ai_generate_action)

        # Database query action
        async def db_query_action(inputs, context, config):
            import psycopg2
            from psycopg2.extras import RealDictCursor

            query = config.get("query", "")
            params = config.get("params", [])

            # Substitute parameters
            for i, param in enumerate(params):
                if param.startswith("$"):
                    key = param[1:]
                    params[i] = inputs.get(key) or context.get(key)

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query, params)

            if query.strip().upper().startswith("SELECT"):
                result = cur.fetchall()
            else:
                result = {"rows_affected": cur.rowcount}
                conn.commit()

            conn.close()
            return {"result": result}

        self._action_handler.register_action("db_query", db_query_action)

        # Send notification action
        async def notify_action(inputs, context, config):
            channel = config.get("channel", "log")
            message = config.get("message", "").format(**inputs, **context)

            if channel == "log":
                logger.info(f"Notification: {message}")
            # Would handle other channels (email, slack, etc.)

            return {"notified": True, "channel": channel}

        self._action_handler.register_action("notify", notify_action)

    async def _load_default_templates(self):
        """Load default workflow templates"""
        default_templates = [
            self._create_customer_onboarding_template(),
            self._create_data_processing_template(),
            self._create_content_generation_template(),
            self._create_notification_template(),
            self._create_lead_qualification_template(),
            self._create_system_health_check_template(),
        ]

        for template in default_templates:
            self._templates[template.template_id] = template

    # ========================================================================
    # DEFAULT TEMPLATE DEFINITIONS
    # ========================================================================

    def _create_customer_onboarding_template(self) -> WorkflowTemplate:
        """Create customer onboarding workflow template"""
        return WorkflowTemplate(
            template_id="customer_onboarding",
            name="Customer Onboarding",
            description="Automated customer onboarding workflow",
            category=WorkflowCategory.CUSTOMER_SUPPORT,
            steps=[
                WorkflowStep(
                    step_id="validate_customer",
                    name="Validate Customer Data",
                    step_type=StepType.ACTION,
                    handler="transform",
                    config={"mapping": {"customer_id": "customer_id", "email": "email"}},
                    next_steps=["create_account"]
                ),
                WorkflowStep(
                    step_id="create_account",
                    name="Create Account",
                    step_type=StepType.ACTION,
                    handler="db_query",
                    config={
                        "query": "INSERT INTO customers (email) VALUES ($1) RETURNING id",
                        "params": ["$email"]
                    },
                    next_steps=["send_welcome"]
                ),
                WorkflowStep(
                    step_id="send_welcome",
                    name="Send Welcome Email",
                    step_type=StepType.ACTION,
                    handler="notify",
                    config={
                        "channel": "email",
                        "message": "Welcome {email} to our platform!"
                    },
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="complete",
                    name="Complete Onboarding",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Customer onboarding completed for {email}"}
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "email": {"type": "string", "format": "email"},
                    "name": {"type": "string"}
                },
                "required": ["email"]
            }
        )

    def _create_data_processing_template(self) -> WorkflowTemplate:
        """Create data processing workflow template"""
        return WorkflowTemplate(
            template_id="data_processing",
            name="Data Processing Pipeline",
            description="Process and transform data",
            category=WorkflowCategory.DATA_PROCESSING,
            steps=[
                WorkflowStep(
                    step_id="fetch_data",
                    name="Fetch Data",
                    step_type=StepType.ACTION,
                    handler="http_request",
                    config={"url": "{data_url}", "method": "GET"},
                    next_steps=["transform_data"]
                ),
                WorkflowStep(
                    step_id="transform_data",
                    name="Transform Data",
                    step_type=StepType.ACTION,
                    handler="transform",
                    config={"mapping": {"processed": "body"}},
                    next_steps=["store_data"]
                ),
                WorkflowStep(
                    step_id="store_data",
                    name="Store Processed Data",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Data processed and stored"}
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "data_url": {"type": "string", "format": "uri"}
                },
                "required": ["data_url"]
            }
        )

    def _create_content_generation_template(self) -> WorkflowTemplate:
        """Create content generation workflow template"""
        return WorkflowTemplate(
            template_id="content_generation",
            name="AI Content Generation",
            description="Generate content using AI",
            category=WorkflowCategory.CONTENT_GENERATION,
            steps=[
                WorkflowStep(
                    step_id="analyze_topic",
                    name="Analyze Topic",
                    step_type=StepType.ACTION,
                    handler="ai_generate",
                    config={"prompt": "Analyze the topic: {topic}"},
                    next_steps=["generate_outline"]
                ),
                WorkflowStep(
                    step_id="generate_outline",
                    name="Generate Outline",
                    step_type=StepType.ACTION,
                    handler="ai_generate",
                    config={"prompt": "Create an outline for: {topic}"},
                    next_steps=["generate_content"]
                ),
                WorkflowStep(
                    step_id="generate_content",
                    name="Generate Content",
                    step_type=StepType.ACTION,
                    handler="ai_generate",
                    config={"prompt": "Write detailed content about: {topic}"},
                    next_steps=["review"]
                ),
                WorkflowStep(
                    step_id="review",
                    name="Review and Finalize",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Content generation completed for: {topic}"}
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "style": {"type": "string"},
                    "length": {"type": "integer"}
                },
                "required": ["topic"]
            }
        )

    def _create_notification_template(self) -> WorkflowTemplate:
        """Create notification workflow template"""
        return WorkflowTemplate(
            template_id="notification",
            name="Multi-Channel Notification",
            description="Send notifications across channels",
            category=WorkflowCategory.NOTIFICATION,
            steps=[
                WorkflowStep(
                    step_id="prepare_message",
                    name="Prepare Message",
                    step_type=StepType.ACTION,
                    handler="transform",
                    config={"mapping": {"formatted_message": "message"}},
                    next_steps=["send_notifications"]
                ),
                WorkflowStep(
                    step_id="send_notifications",
                    name="Send Notifications",
                    step_type=StepType.PARALLEL,
                    config={"steps": ["notify_email", "notify_slack", "notify_log"]},
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="notify_email",
                    name="Send Email",
                    step_type=StepType.ACTION,
                    handler="notify",
                    config={"channel": "email", "message": "{formatted_message}"}
                ),
                WorkflowStep(
                    step_id="notify_slack",
                    name="Send Slack",
                    step_type=StepType.ACTION,
                    handler="notify",
                    config={"channel": "slack", "message": "{formatted_message}"}
                ),
                WorkflowStep(
                    step_id="notify_log",
                    name="Log Notification",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Notification sent: {formatted_message}"}
                ),
                WorkflowStep(
                    step_id="complete",
                    name="Complete",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "All notifications sent"}
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "channels": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["message"]
            }
        )

    def _create_lead_qualification_template(self) -> WorkflowTemplate:
        """Create lead qualification workflow template"""
        return WorkflowTemplate(
            template_id="lead_qualification",
            name="Lead Qualification",
            description="Qualify and score leads",
            category=WorkflowCategory.SALES_AUTOMATION,
            steps=[
                WorkflowStep(
                    step_id="fetch_lead",
                    name="Fetch Lead Data",
                    step_type=StepType.ACTION,
                    handler="db_query",
                    config={
                        "query": "SELECT * FROM leads WHERE id = $1",
                        "params": ["$lead_id"]
                    },
                    next_steps=["score_lead"]
                ),
                WorkflowStep(
                    step_id="score_lead",
                    name="Score Lead",
                    step_type=StepType.ACTION,
                    handler="ai_generate",
                    config={"prompt": "Score this lead: {result}"},
                    next_steps=["check_score"]
                ),
                WorkflowStep(
                    step_id="check_score",
                    name="Check Score",
                    step_type=StepType.DECISION,
                    conditions=[
                        {"field": "score", "operator": "gt", "value": 80, "target": "high_priority"},
                        {"field": "score", "operator": "gt", "value": 50, "target": "medium_priority"}
                    ],
                    config={"default": "low_priority"},
                    next_steps=["high_priority", "medium_priority", "low_priority"]
                ),
                WorkflowStep(
                    step_id="high_priority",
                    name="High Priority Lead",
                    step_type=StepType.ACTION,
                    handler="notify",
                    config={"channel": "slack", "message": "High priority lead: {lead_id}"},
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="medium_priority",
                    name="Medium Priority Lead",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Medium priority lead: {lead_id}"},
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="low_priority",
                    name="Low Priority Lead",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Low priority lead: {lead_id}"},
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="complete",
                    name="Complete",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "Lead qualification completed"}
                )
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "lead_id": {"type": "string"}
                },
                "required": ["lead_id"]
            }
        )

    def _create_system_health_check_template(self) -> WorkflowTemplate:
        """Create system health check workflow template"""
        return WorkflowTemplate(
            template_id="system_health_check",
            name="System Health Check",
            description="Check system health and alert on issues",
            category=WorkflowCategory.SYSTEM_MAINTENANCE,
            steps=[
                WorkflowStep(
                    step_id="check_services",
                    name="Check Services",
                    step_type=StepType.PARALLEL,
                    config={
                        "template_id": "system_health_check",
                        "steps": ["check_api", "check_db", "check_agents"]
                    },
                    next_steps=["evaluate_health"]
                ),
                WorkflowStep(
                    step_id="check_api",
                    name="Check API",
                    step_type=StepType.ACTION,
                    handler="http_request",
                    config={"url": "https://brainops-ai-agents.onrender.com/health", "method": "GET"}
                ),
                WorkflowStep(
                    step_id="check_db",
                    name="Check Database",
                    step_type=StepType.ACTION,
                    handler="db_query",
                    config={"query": "SELECT 1", "params": []}
                ),
                WorkflowStep(
                    step_id="check_agents",
                    name="Check Agents",
                    step_type=StepType.ACTION,
                    handler="db_query",
                    config={
                        "query": "SELECT COUNT(*) as count FROM ai_agents WHERE status = 'active'",
                        "params": []
                    }
                ),
                WorkflowStep(
                    step_id="evaluate_health",
                    name="Evaluate Health",
                    step_type=StepType.DECISION,
                    conditions=[
                        {"field": "success", "operator": "eq", "value": False, "target": "alert"}
                    ],
                    config={"default": "complete"},
                    next_steps=["alert", "complete"]
                ),
                WorkflowStep(
                    step_id="alert",
                    name="Send Alert",
                    step_type=StepType.ACTION,
                    handler="notify",
                    config={"channel": "slack", "message": "System health issue detected!"},
                    next_steps=["complete"]
                ),
                WorkflowStep(
                    step_id="complete",
                    name="Complete",
                    step_type=StepType.ACTION,
                    handler="log",
                    config={"message": "System health check completed"}
                )
            ]
        )

    # ========================================================================
    # TEMPLATE MANAGEMENT
    # ========================================================================

    async def register_template(self, template: WorkflowTemplate) -> str:
        """Register a workflow template"""
        await self.initialize()

        self._templates[template.template_id] = template
        await self._persist_template(template)

        logger.info(f"Registered workflow template: {template.template_id}")
        return template.template_id

    async def get_template(self, template_id: str) -> Optional[WorkflowTemplate]:
        """Get a workflow template"""
        return self._templates.get(template_id)

    async def list_templates(
        self,
        category: Optional[WorkflowCategory] = None
    ) -> list[WorkflowTemplate]:
        """List workflow templates"""
        templates = list(self._templates.values())
        if category:
            templates = [t for t in templates if t.category == category]
        return templates

    async def _persist_template(self, template: WorkflowTemplate):
        """Persist template to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            steps_json = json.dumps([asdict(s) for s in template.steps], default=str)

            cur.execute("""
                INSERT INTO ai_workflow_templates (
                    template_id, name, description, category, version,
                    steps, input_schema, output_schema, variables, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (template_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    steps = EXCLUDED.steps,
                    input_schema = EXCLUDED.input_schema,
                    output_schema = EXCLUDED.output_schema,
                    variables = EXCLUDED.variables,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, (
                template.template_id,
                template.name,
                template.description,
                template.category.value,
                template.version,
                steps_json,
                json.dumps(template.input_schema),
                json.dumps(template.output_schema),
                json.dumps(template.variables),
                json.dumps(template.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist template: {e}")

    # ========================================================================
    # WORKFLOW EXECUTION
    # ========================================================================

    async def execute_workflow(
        self,
        template_id: str,
        inputs: dict[str, Any],
        correlation_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Execute a workflow"""
        await self.initialize()

        template = self._templates.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        execution_id = f"exec_{uuid.uuid4().hex[:12]}"

        execution = WorkflowExecution(
            execution_id=execution_id,
            template_id=template_id,
            status=WorkflowStatus.RUNNING,
            inputs=inputs,
            context={**inputs},
            started_at=datetime.now(timezone.utc),
            correlation_id=correlation_id,
            tenant_id=tenant_id,
            metadata=metadata or {}
        )

        self._executions[execution_id] = execution
        self._step_executions[execution_id] = {}

        await self._persist_execution(execution)

        # Execute workflow
        try:
            await self._run_workflow(execution, template)
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            logger.error(f"Workflow execution failed: {e}")

        await self._persist_execution(execution)

        # Notify completion
        for callback in self._on_workflow_complete:
            try:
                await callback(execution)
            except Exception as e:
                logger.error(f"Workflow completion callback error: {e}")

        return execution

    async def _run_workflow(self, execution: WorkflowExecution, template: WorkflowTemplate):
        """Run workflow steps"""
        if not template.steps:
            return

        # Start with first step
        current_step = template.steps[0]
        execution.current_step = current_step.step_id

        while current_step:
            result = await self._execute_step(current_step, execution.context, execution.inputs)

            # Store result
            execution.step_results[current_step.step_id] = result
            execution.context.update(result)

            # Get next step
            next_step = None
            if current_step.next_steps:
                # Handle decision-based routing
                if current_step.step_type == StepType.DECISION:
                    decision = result.get("decision")
                    if decision in current_step.next_steps:
                        next_step_id = decision
                    else:
                        next_step_id = current_step.next_steps[0]
                else:
                    next_step_id = current_step.next_steps[0]

                next_step = self._get_step_by_id(template, next_step_id)

            current_step = next_step
            if current_step:
                execution.current_step = current_step.step_id

        execution.outputs = execution.context

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a single step"""
        handler = self._handlers.get(step.step_type)
        if not handler:
            raise ValueError(f"No handler for step type: {step.step_type}")

        # Prepare inputs
        step_inputs = {**inputs}
        for output_key, input_key in step.inputs.items():
            if input_key in context:
                step_inputs[output_key] = context[input_key]

        retry_count = 0
        last_error = None

        while retry_count <= step.max_retries:
            try:
                result = await asyncio.wait_for(
                    handler.execute(step, context, step_inputs),
                    timeout=step.timeout_seconds
                )

                # Apply output mappings
                mapped_result = {}
                for result_key, output_key in step.outputs.items():
                    if result_key in result:
                        mapped_result[output_key] = result[result_key]
                mapped_result.update(result)

                # Notify step completion
                for callback in self._on_step_complete:
                    try:
                        await callback(step, mapped_result)
                    except Exception as exc:
                        logger.warning("Step completion callback failed: %s", exc, exc_info=True)

                return mapped_result

            except asyncio.TimeoutError:
                last_error = f"Step timed out after {step.timeout_seconds}s"
                retry_count += 1

            except Exception as e:
                last_error = str(e)
                retry_count += 1

                if step.on_failure == "skip":
                    return {"skipped": True, "error": last_error}
                elif step.on_failure == "continue":
                    return {"error": last_error}

        raise Exception(f"Step failed after {retry_count} retries: {last_error}")

    def _get_step_by_id(
        self,
        template: WorkflowTemplate,
        step_id: str
    ) -> Optional[WorkflowStep]:
        """Get step by ID"""
        for step in template.steps:
            if step.step_id == step_id:
                return step
        return None

    def _get_step(self, template_id: str, step_id: str) -> Optional[WorkflowStep]:
        """Get step from template"""
        template = self._templates.get(template_id)
        if template:
            return self._get_step_by_id(template, step_id)
        return None

    async def _persist_execution(self, execution: WorkflowExecution):
        """Persist execution to database"""
        try:
            import psycopg2

            conn = psycopg2.connect(**self._get_db_config())
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ai_workflow_executions (
                    execution_id, template_id, status, inputs, outputs,
                    context, current_step, step_results, started_at,
                    completed_at, error, correlation_id, tenant_id, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (execution_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    outputs = EXCLUDED.outputs,
                    context = EXCLUDED.context,
                    current_step = EXCLUDED.current_step,
                    step_results = EXCLUDED.step_results,
                    completed_at = EXCLUDED.completed_at,
                    error = EXCLUDED.error
            """, (
                execution.execution_id,
                execution.template_id,
                execution.status.value,
                json.dumps(execution.inputs, default=str),
                json.dumps(execution.outputs, default=str),
                json.dumps(execution.context, default=str),
                execution.current_step,
                json.dumps(execution.step_results, default=str),
                execution.started_at,
                execution.completed_at,
                execution.error,
                execution.correlation_id,
                execution.tenant_id,
                json.dumps(execution.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to persist execution: {e}")

    # ========================================================================
    # EXECUTION MANAGEMENT
    # ========================================================================

    async def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID"""
        return self._executions.get(execution_id)

    async def list_executions(
        self,
        template_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 100
    ) -> list[WorkflowExecution]:
        """List executions"""
        executions = list(self._executions.values())

        if template_id:
            executions = [e for e in executions if e.template_id == template_id]
        if status:
            executions = [e for e in executions if e.status == status]

        return sorted(executions, key=lambda e: e.started_at or datetime.min, reverse=True)[:limit]

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id not in self._executions:
            return False

        execution = self._executions[execution_id]
        if execution.status == WorkflowStatus.RUNNING:
            execution.status = WorkflowStatus.CANCELLED
            execution.completed_at = datetime.now(timezone.utc)
            await self._persist_execution(execution)
            return True

        return False

    # ========================================================================
    # ACTION REGISTRATION
    # ========================================================================

    def register_action(self, name: str, handler: Callable):
        """Register a custom action handler"""
        self._action_handler.register_action(name, handler)
        logger.info(f"Registered custom action: {name}")

    def on_step_complete(self, callback: Callable):
        """Register step completion callback"""
        self._on_step_complete.append(callback)

    def on_workflow_complete(self, callback: Callable):
        """Register workflow completion callback"""
        self._on_workflow_complete.append(callback)

    # ========================================================================
    # HEALTH & STATS
    # ========================================================================

    async def get_stats(self) -> dict[str, Any]:
        """Get workflow engine statistics"""
        return {
            "templates_count": len(self._templates),
            "executions_count": len(self._executions),
            "running_executions": len([e for e in self._executions.values() if e.status == WorkflowStatus.RUNNING]),
            "completed_executions": len([e for e in self._executions.values() if e.status == WorkflowStatus.COMPLETED]),
            "failed_executions": len([e for e in self._executions.values() if e.status == WorkflowStatus.FAILED]),
            "registered_actions": len(self._action_handler.actions)
        }

    async def get_health_status(self) -> dict[str, Any]:
        """Get health status"""
        stats = await self.get_stats()

        return {
            "status": "healthy",
            "initialized": self._initialized,
            "stats": stats
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_workflow_engine_instance: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get or create the workflow engine instance"""
    global _workflow_engine_instance
    if _workflow_engine_instance is None:
        _workflow_engine_instance = WorkflowEngine()
    return _workflow_engine_instance


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def execute_workflow(
    template_id: str,
    inputs: dict[str, Any]
) -> WorkflowExecution:
    """Execute a workflow"""
    engine = get_workflow_engine()
    return await engine.execute_workflow(template_id, inputs)


async def get_template(template_id: str) -> Optional[WorkflowTemplate]:
    """Get a workflow template"""
    engine = get_workflow_engine()
    return await engine.get_template(template_id)


async def list_templates() -> list[WorkflowTemplate]:
    """List all templates"""
    engine = get_workflow_engine()
    return await engine.list_templates()


async def get_workflow_stats() -> dict[str, Any]:
    """Get workflow statistics"""
    engine = get_workflow_engine()
    return await engine.get_stats()


if __name__ == "__main__":
    async def test():
        engine = get_workflow_engine()
        await engine.initialize()

        # List templates
        templates = await engine.list_templates()
        print(f"Available templates: {[t.template_id for t in templates]}")

        # Execute notification workflow
        execution = await engine.execute_workflow(
            template_id="notification",
            inputs={"message": "Test notification message"}
        )
        print(f"Execution: {execution.execution_id} - {execution.status.value}")

        # Execute content generation workflow
        execution = await engine.execute_workflow(
            template_id="content_generation",
            inputs={"topic": "AI in modern business"}
        )
        print(f"Execution: {execution.execution_id} - {execution.status.value}")

        # Get stats
        stats = await engine.get_stats()
        print(f"Stats: {json.dumps(stats, indent=2)}")

        # Get health
        health = await engine.get_health_status()
        print(f"Health: {json.dumps(health, indent=2)}")

    asyncio.run(test())
