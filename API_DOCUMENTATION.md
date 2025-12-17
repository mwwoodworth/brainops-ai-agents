# BrainOps AI Agents API Documentation

**Title:** BrainOps AI Agents v9.0.0
**Version:** 9.0.0
**Description:** Production-ready AI Agent Orchestration Platform

## Endpoints

### GET `/memory/status`
**Summary:** Get Memory Status
**Description:** Get memory system status with proper error handling
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/memory/search`
**Summary:** Search Memories
**Description:** Search memory entries across available memory tables
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| query | query | True | Search query |
| context_key | query | False | Filter by context key |
| limit | query | False | Maximum results |
| importance_threshold | query | False | Minimum importance |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/brain/context`
**Summary:** Get Full Context
**Description:** Get COMPLETE system context for Claude Code session initialization
This is THE endpoint that runs at session start
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/brain/critical`
**Summary:** Get Critical Context
**Description:** Get ALL critical context across all categories
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/brain/category/{category}`
**Summary:** Get By Category
**Description:** Get all context in a specific category
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| category | path | True | - |
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/brain/get/{key}`
**Summary:** Get Context
**Description:** Retrieve a specific piece of context
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| key | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/brain/store`
**Summary:** Store Context
**Description:** Store or update a piece of context
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `BrainEntry`
  - Properties:
    - `key` (string): Key
    - `value` (any): Value
    - `category` (string): Category
    - `priority` (string): Priority
    - `source` (string): Source
    - `metadata` (any): Metadata
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/brain/search`
**Summary:** Search Context
**Description:** Search across all context
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `BrainQuery`
  - Properties:
    - `query` (string): Query
    - `limit` (integer): Limit
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/brain/session`
**Summary:** Record Session
**Description:** Record a Claude Code session summary
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `Body_record_session_brain_session_post`
  - Properties:
    - `session_id` (string): Session Id
    - `summary` (object): Summary
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/brain/deployment`
**Summary:** Record Deployment
**Description:** Record a deployment
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `Body_record_deployment_brain_deployment_post`
  - Properties:
    - `service` (string): Service
    - `version` (string): Version
    - `status` (string): Status
    - `metadata` (any): Metadata
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/brain/system-state`
**Summary:** Update System State
**Description:** Update current system state
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `Body_update_system_state_brain_system_state_post`
  - Properties:
    - `component` (string): Component
    - `state` (object): State
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/brain/health`
**Summary:** Brain Health
**Description:** Check unified brain health
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### POST `/memory/context/store`
**Summary:** Store Context
**Description:** Store context entry with automatic synchronization

**Layers:**
- ephemeral: In-memory cache (seconds to minutes)
- session: Session-scoped (hours)
- short_term: Days to weeks
- long_term: Weeks to months
- permanent: Forever

**Scopes:**
- global: Visible to all systems
- tenant: Tenant-specific
- user: User-specific
- session: Current session only
- agent: Specific AI agent only
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `StoreContextRequest`
  - Properties:
    - `key` (string): Key
    - `value` (any): Value
    - `layer` (string): Layer
    - `scope` (string): Scope
    - `priority` (string): Priority
    - `category` (string): Category
    - `source` (string): Source
    - `tenant_id` (any): Tenant Id
    - `user_id` (any): User Id
    - `session_id` (any): Session Id
    - `agent_id` (any): Agent Id
    - `metadata` (object): Metadata
    - `expires_in_seconds` (any): Expires In Seconds
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/context/retrieve`
**Summary:** Retrieve Context
**Description:** Retrieve context with intelligent caching
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `RetrieveContextRequest`
  - Properties:
    - `key` (string): Key
    - `scope` (string): Scope
    - `tenant_id` (any): Tenant Id
    - `user_id` (any): User Id
    - `session_id` (any): Session Id
    - `agent_id` (any): Agent Id
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/context/search`
**Summary:** Search Context
**Description:** Search across all context with filtering
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `SearchContextRequest`
  - Properties:
    - `query` (string): Query
    - `scope` (any): Scope
    - `layer` (any): Layer
    - `category` (any): Category
    - `tenant_id` (any): Tenant Id
    - `limit` (integer): Limit
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/memory/context/stats`
**Summary:** Get Stats
**Description:** Get memory coordination statistics
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### POST `/memory/session/start`
**Summary:** Start Session
**Description:** Start a new session with full context initialization
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `StartSessionRequest`
  - Properties:
    - `session_id` (string): Session Id
    - `tenant_id` (any): Tenant Id
    - `user_id` (any): User Id
    - `initial_context` (object): Initial Context
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/session/resume/{session_id}`
**Summary:** Resume Session
**Description:** Resume an existing session with full context restoration
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| session_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/session/end/{session_id}`
**Summary:** End Session
**Description:** End a session with full context preservation
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| session_id | path | True | - |
| reason | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/session/message`
**Summary:** Add Message
**Description:** Add a message to conversation history
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AddMessageRequest`
  - Properties:
    - `session_id` (string): Session Id
    - `role` (string): Role
    - `content` (string): Content
    - `metadata` (object): Metadata
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/session/task`
**Summary:** Add Task
**Description:** Add a task to session tracking
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AddTaskRequest`
  - Properties:
    - `session_id` (string): Session Id
    - `task` (object): Task
    - `status` (string): Status
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/memory/session/context/{session_id}`
**Summary:** Get Session Context
**Description:** Get complete context for a session
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| session_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/memory/session/handoff`
**Summary:** Handoff To Agent
**Description:** Hand off session to another agent with perfect context transfer
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `HandoffRequest`
  - Properties:
    - `session_id` (string): Session Id
    - `to_agent` (string): To Agent
    - `handoff_reason` (string): Handoff Reason
    - `critical_info` (object): Critical Info
    - `continuation_instructions` (string): Continuation Instructions
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/memory/session/handoff/{session_id}`
**Summary:** Get Handoff Context
**Description:** Get the latest handoff context for an agent
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| session_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/memory/health`
**Summary:** Memory Health
**Description:** Health check for memory coordination system
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/v1/ai/customer-intelligence/{customer_id}`
**Summary:** Get Customer Intelligence
**Description:** Get AI-powered intelligence for a specific customer.
Calculates risk, LTV, and behavioral profile.
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| customer_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/ai/analyze-customer`
**Summary:** Trigger Customer Analysis
**Description:** Trigger an async analysis for a customer.
In a real system, this would push to a queue.
Here we just return success as the GET endpoint does real-time analysis.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `CustomerAnalysisRequest`
  - Properties:
    - `customer_id` (string): Customer Id
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/ai/batch-customer-intelligence`
**Summary:** Batch Customer Intelligence
**Description:** Get intelligence for multiple customers at once.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `BatchAnalysisRequest`
  - Properties:
    - `customer_ids` (array): Customer Ids
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/gumroad/webhook`
**Summary:** Handle Gumroad Webhook
**Description:** Main webhook handler for Gumroad sales
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### GET `/gumroad/analytics`
**Summary:** Get Sales Analytics
**Description:** Get sales analytics from database
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### POST `/gumroad/test`
**Summary:** Test Webhook
**Description:** Test endpoint for webhook
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### GET `/gumroad/products`
**Summary:** Get Products
**Description:** Get list of configured products
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### GET `/api/codebase-graph/database/stats`
**Summary:** Get Database Stats
**Description:** Get database schema statistics
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/codebase-graph/database/tables`
**Summary:** Get Database Tables
**Description:** Get all tables with their columns
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| schema | query | False | - |
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/api/codebase-graph/database/relationships`
**Summary:** Get Database Relationships
**Description:** Get all foreign key relationships for ERD visualization
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/codebase-graph/database/visualize`
**Summary:** Visualize Database Erd
**Description:** Interactive ERD visualization using vis.js
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/codebase-graph/stats`
**Summary:** Get Graph Stats
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/codebase-graph/data`
**Summary:** Get Graph Data
**Description:** Get graph data (nodes and edges) for visualization.
Limited to prevent browser crash on massive graphs.
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/api/codebase-graph/visualize`
**Summary:** Visualize Graph
**Description:** Simple D3-like visualization using vis.js (Network) for easy graph rendering.
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/health`
**Summary:** Get System Health
**Description:** Get overall system health summary
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/components`
**Summary:** Get All Components
**Description:** Get all tracked components
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| component_type | query | False | Filter by type: codebase, agent, database_table, api_endpoint |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/api/state-sync/components/{component_name}`
**Summary:** Get Component
**Description:** Get a specific component by name
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| component_name | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/api/state-sync/changes`
**Summary:** Get Change History
**Description:** Get recent change history
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/api/state-sync/changes/by-codebase/{codebase}`
**Summary:** Get Changes By Codebase
**Description:** Get changes for a specific codebase
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| codebase | path | True | - |
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/state-sync/scan`
**Summary:** Trigger Full Scan
**Description:** Trigger a full system scan
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### POST `/api/state-sync/propagate`
**Summary:** Trigger Change Propagation
**Description:** Trigger change detection and propagation
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/context`
**Summary:** Get Ai Context
**Description:** Get formatted context for AI sessions
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/context/raw`
**Summary:** Get Raw State
**Description:** Get raw system state JSON
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/git-status`
**Summary:** Get Git Status
**Description:** Get git status across all codebases
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/api/state-sync/visualize`
**Summary:** Visualize System State
**Description:** Interactive system state visualization
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### POST `/agents/analytics`
**Summary:** Agent Analytics
**Description:** AI Agent Analytics Endpoint

Actions:
- analyze: Analyze agent performance metrics
- report: Generate detailed reports
- predict: Predictive analytics based on historical data

Periods:
- current_month, last_month, current_quarter, current_year

Metrics:
- revenue: Revenue analysis
- performance: Agent performance metrics
- utilization: Agent utilization rates
- success_rate: Task success rates
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| action | query | False | - |
| period | query | False | - |
| metric | query | False | - |
| agent_id | query | False | - |
**Request Body (application/json):**
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/agents/analytics/summary`
**Summary:** Get Analytics Summary
**Description:** Get quick analytics summary
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/`
**Summary:** Root
**Description:** Root endpoint
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### GET `/health`
**Summary:** Health Check
**Description:** Health check endpoint with full system status and light caching.
**Auth Required:** No
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| force_refresh | query | False | Bypass cache and force live health checks |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/observability/metrics`
**Summary:** Observability Metrics
**Description:** Lightweight monitoring endpoint for request, cache, DB, and orchestrator health.
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### GET `/systems/usage`
**Summary:** Systems Usage
**Description:** Report which AI systems are being used plus scheduler and memory effectiveness.
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/ai/providers/status`
**Summary:** Providers Status
**Description:** Report configuration and basic liveness for all AI providers (OpenAI, Anthropic,
Gemini, Perplexity, Hugging Face). Does not modify configuration or credentials;
it only runs small probe calls to detect misconfiguration like invalid or missing
API keys.
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/agents`
**Summary:** Get Agents
**Description:** Get list of available agents
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| category | query | False | - |
| enabled | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/agents/{agent_id}/execute`
**Summary:** Execute Agent
**Description:** Execute an agent
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| agent_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/agents/{agent_id}`
**Summary:** Get Agent
**Description:** Get a specific agent
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| agent_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/executions`
**Summary:** Get Executions
**Description:** Get agent executions
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| agent_id | query | False | - |
| status | query | False | - |
| limit | query | False | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/execute`
**Summary:** Execute Scheduled Agents
**Description:** Execute scheduled agents (called by cron)
**Auth Required:** Yes (APIKeyHeader)
**Responses:**
- **200**: Successful Response

---

### GET `/scheduler/status`
**Summary:** Get Scheduler Status
**Description:** Get detailed scheduler status and diagnostics
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### POST `/ai/self-assess`
**Summary:** Ai Self Assess
**Description:** AI assesses its own confidence in completing a task

Revolutionary feature - AI knows what it doesn't know!
**Auth Required:** No
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| task_id | query | True | - |
| agent_id | query | True | - |
| task_description | query | True | - |
**Request Body (application/json):**
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/ai/explain-reasoning`
**Summary:** Ai Explain Reasoning
**Description:** AI explains its reasoning in human-understandable terms

Transparency builds trust!
**Auth Required:** No
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| task_id | query | True | - |
| agent_id | query | True | - |
| decision | query | True | - |
**Request Body (application/json):**
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/ai/learn-from-mistake`
**Summary:** Ai Learn From Mistake
**Description:** AI analyzes its own mistakes and learns from them

This is how AI gets smarter over time!
**Auth Required:** No
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| task_id | query | True | - |
| agent_id | query | True | - |
| expected_outcome | query | True | - |
| actual_outcome | query | True | - |
| confidence_before | query | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/ai/self-awareness/stats`
**Summary:** Get Self Awareness Stats
**Description:** Get statistics about AI self-awareness system
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### POST `/ai/tasks/execute/{task_id}`
**Summary:** Execute Ai Task
**Description:** Manually trigger execution of a specific task
**Auth Required:** No
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| task_id | path | True | - |
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### GET `/ai/tasks/stats`
**Summary:** Get Task Stats
**Description:** Get AI task system statistics
**Auth Required:** No
**Responses:**
- **200**: Successful Response

---

### POST `/ai/orchestrate`
**Summary:** Orchestrate Complex Workflow
**Description:** Execute complex multi-stage workflow using LangGraph orchestration
This is for sophisticated tasks that need multi-agent coordination
**Auth Required:** Yes (APIKeyHeader)
**Parameters:**
| Name | In | Required | Description |
|---|---|---|---|
| task_description | query | True | - |
**Request Body (application/json):**
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/ai/analyze`
**Summary:** Ai Analyze
**Description:** AI analysis endpoint for weathercraft-erp and other frontends.
Accepts JSON body with agent, action, data, and context fields.
Routes to the appropriate agent or orchestrator.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AIAnalyzeRequest`
  - Properties:
    - `agent` (string): Agent
    - `action` (string): Action
    - `data` (object): Data
    - `context` (object): Context
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/aurea/command/natural_language`
**Summary:** Execute Aurea Nl Command
**Description:** Execute a natural language command through AUREA's NLU processor.
Founder-level authority for natural language system control.

Examples:
- "Create a high priority task to deploy the new feature"
- "Show me all tasks that are in progress"
- "Get AUREA status"
- "Execute task abc-123"
**Auth Required:** No
**Request Body (application/json):**
Schema: `AureaCommandRequest`
  - Properties:
    - `command_text` (string): Command Text
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/knowledge/store`
**Summary:** Api V1 Knowledge Store
**Description:** Store a knowledge/memory entry in the unified memory system.

Primary path uses the Embedded Memory System (SQLite + async sync to Postgres).
Fallback path uses the Unified Memory Manager when embedded memory is unavailable.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `KnowledgeStoreRequest`
  - Properties:
    - `content` (string): Content
    - `memory_type` (string): Memory Type
    - `source_system` (any): Source System
    - `source_agent` (any): Source Agent
    - `created_by` (any): Created By
    - `importance` (number): Importance
    - `tags` (any): Tags
    - `metadata` (any): Metadata
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/knowledge/query`
**Summary:** Api V1 Knowledge Query
**Description:** Query the unified memory / knowledge store.

Uses the Embedded Memory System when available (vector search), with a
fallback to the Unified Memory Manager recall API.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `KnowledgeQueryRequest`
  - Properties:
    - `query` (string): Query
    - `limit` (integer): Limit
    - `memory_type` (any): Memory Type
    - `min_importance` (number): Min Importance
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/erp/analyze`
**Summary:** Api V1 Erp Analyze
**Description:** Analyze ERP jobs using centralized BrainOps Core.

- Reads jobs from the shared database (read-only).
- Computes schedule risk and progress using deterministic heuristics.
- Optionally augments each job with AI commentary when AI Core is available.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `ErpAnalyzeRequest`
  - Properties:
    - `tenant_id` (any): Tenant Id
    - `job_ids` (any): Job Ids
    - `limit` (integer): Limit
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/agents/execute`
**Summary:** Api V1 Agents Execute
**Description:** Execute an agent via the v1 API surface.

Body: { "agent_id" | "id": string, "payload": object }
Internally delegates to the existing /agents/{agent_id}/execute endpoint.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AgentExecuteRequest`
  - Properties:
    - `agent_id` (any): Agent Id
    - `id` (any): Id
    - `payload` (object): Payload
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/agents/activate`
**Summary:** Api V1 Agents Activate
**Description:** Activate or deactivate an agent via the v1 API surface.

This is a thin wrapper that flips the `enabled` flag in the agents table.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AgentActivateRequest`
  - Properties:
    - `agent_id` (any): Agent Id
    - `agent_name` (any): Agent Name
    - `enabled` (boolean): Enabled
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

### POST `/api/v1/aurea/execute-event`
**Summary:** Execute Aurea Event
**Description:** Execute event with specified AI agent via AUREA orchestration.
Called by Event Router daemon to process events from brainops_core.event_bus.
**Auth Required:** Yes (APIKeyHeader)
**Request Body (application/json):**
Schema: `AUREAEventRequest`
  - Properties:
    - `event_id` (string): Event Id
    - `topic` (string): Topic
    - `source` (string): Source
    - `payload` (object): Payload
    - `target_agent` (object): Target Agent
    - `routing_metadata` (any): Routing Metadata
**Responses:**
- **200**: Successful Response
- **422**: Validation Error

---

## Verification

### Public Endpoint Checks (Live)
- **GET /**: 200 OK (BrainOps AI OS 9.0.0)
- **GET /health**: 200 OK (Status: operational)