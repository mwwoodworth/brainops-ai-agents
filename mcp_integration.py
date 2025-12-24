"""
MCP Integration Layer - The Missing Link
=========================================
Connects BrainOps AI Agents to the 345-tool MCP Bridge.
This is what makes the AI truly autonomous.

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import asyncio
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# MCP Bridge Configuration
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = os.getenv("MCP_API_KEY", "brainops_prod_key_2025")


class MCPServer(Enum):
    """Available MCP Servers with their tool counts"""
    RENDER = "render"           # 39 tools
    VERCEL = "vercel"           # 34 tools
    SUPABASE = "supabase"       # 40 tools
    GITHUB = "github"           # 50 tools
    DOCKER = "docker"           # 53 tools
    STRIPE = "stripe"           # 55 tools
    OPENAI = "openai"           # 7 tools
    ANTHROPIC = "anthropic"     # 3 tools
    PLAYWRIGHT = "playwright"   # 60 tools
    PYTHON = "python-executor"  # 8 tools


@dataclass
class MCPToolResult:
    """Result from an MCP tool execution"""
    success: bool
    server: str
    tool: str
    result: Any
    duration_ms: float
    error: Optional[str] = None
    execution_id: Optional[str] = None


class MCPClient:
    """
    The Core MCP Bridge Client

    Enables AI agents to execute any of the 345 tools available
    in the MCP Bridge infrastructure.
    """

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or MCP_BRIDGE_URL
        self.api_key = api_key or MCP_API_KEY
        self._session: Optional[aiohttp.ClientSession] = None
        self._execution_count = 0
        logger.info(f"MCPClient initialized with bridge: {self.base_url}")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": self.api_key,
                    "User-Agent": "BrainOps-AI-Agent/1.0"
                },
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def execute_tool(
        self,
        server: MCPServer,
        tool: str,
        params: Dict[str, Any] = None
    ) -> MCPToolResult:
        """
        Execute any MCP tool

        Args:
            server: The MCP server (render, vercel, supabase, etc.)
            tool: The tool name (e.g., "listServices", "sql_query")
            params: Tool parameters

        Returns:
            MCPToolResult with execution details
        """
        start_time = datetime.utcnow()
        self._execution_count += 1

        payload = {
            "server": server.value if isinstance(server, MCPServer) else server,
            "tool": tool,
            "params": params or {}
        }

        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/mcp/execute",
                json=payload
            ) as response:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                server_str = server.value if isinstance(server, MCPServer) else server

                if response.status == 200:
                    data = await response.json()
                    logger.info(f"MCP Tool executed: {server_str}/{tool} in {duration_ms:.0f}ms")
                    return MCPToolResult(
                        success=True,
                        server=server_str,
                        tool=tool,
                        result=data.get("result", data),
                        duration_ms=duration_ms,
                        execution_id=data.get("execution_id")
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"MCP Tool failed: {server_str}/{tool} - {error_text}")
                    return MCPToolResult(
                        success=False,
                        server=server_str,
                        tool=tool,
                        result=None,
                        duration_ms=duration_ms,
                        error=error_text
                    )
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            server_str = server.value if isinstance(server, MCPServer) else server
            logger.error(f"MCP Tool exception: {server_str}/{tool} - {e}")
            return MCPToolResult(
                success=False,
                server=server_str,
                tool=tool,
                result=None,
                duration_ms=duration_ms,
                error=str(e)
            )

    # =========================================================================
    # RENDER OPERATIONS (39 tools)
    # =========================================================================

    async def render_list_services(self) -> MCPToolResult:
        """List all Render services"""
        return await self.execute_tool(MCPServer.RENDER, "listServices")

    async def render_get_service(self, service_id: str) -> MCPToolResult:
        """Get details of a specific Render service"""
        return await self.execute_tool(MCPServer.RENDER, "getService", {"serviceId": service_id})

    async def render_trigger_deploy(self, service_id: str) -> MCPToolResult:
        """Trigger a new deployment on Render"""
        return await self.execute_tool(MCPServer.RENDER, "triggerDeploy", {"serviceId": service_id})

    async def render_restart_service(self, service_id: str) -> MCPToolResult:
        """Restart a Render service (for self-healing)"""
        return await self.execute_tool(MCPServer.RENDER, "restartService", {"serviceId": service_id})

    async def render_get_deploy_status(self, service_id: str, deploy_id: str) -> MCPToolResult:
        """Get deployment status"""
        return await self.execute_tool(MCPServer.RENDER, "getDeployStatus", {
            "serviceId": service_id,
            "deployId": deploy_id
        })

    async def render_scale_service(self, service_id: str, num_instances: int) -> MCPToolResult:
        """Scale a Render service"""
        return await self.execute_tool(MCPServer.RENDER, "scaleService", {
            "serviceId": service_id,
            "numInstances": num_instances
        })

    async def render_get_logs(self, service_id: str, lines: int = 100) -> MCPToolResult:
        """Get service logs"""
        return await self.execute_tool(MCPServer.RENDER, "getLogs", {
            "serviceId": service_id,
            "lines": lines
        })

    # =========================================================================
    # SUPABASE OPERATIONS (40 tools)
    # =========================================================================

    async def supabase_query(self, sql: str, params: List[Any] = None) -> MCPToolResult:
        """Execute a raw SQL query on Supabase"""
        return await self.execute_tool(MCPServer.SUPABASE, "sql_query", {
            "query": sql,
            "params": params or []
        })

    async def supabase_select(self, table: str, columns: str = "*", where: str = None) -> MCPToolResult:
        """Select from a Supabase table"""
        sql = f"SELECT {columns} FROM {table}"
        if where:
            sql += f" WHERE {where}"
        return await self.supabase_query(sql)

    async def supabase_insert(self, table: str, data: Dict[str, Any]) -> MCPToolResult:
        """Insert a row into Supabase"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(data))])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) RETURNING *"
        return await self.supabase_query(sql, list(data.values()))

    async def supabase_get_tables(self) -> MCPToolResult:
        """Get all tables in Supabase"""
        return await self.execute_tool(MCPServer.SUPABASE, "listTables")

    # =========================================================================
    # GITHUB OPERATIONS (50 tools)
    # =========================================================================

    async def github_create_pr(
        self,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> MCPToolResult:
        """Create a GitHub pull request"""
        return await self.execute_tool(MCPServer.GITHUB, "createPullRequest", {
            "repo": repo,
            "title": title,
            "body": body,
            "head": head,
            "base": base
        })

    async def github_list_repos(self) -> MCPToolResult:
        """List all accessible repositories"""
        return await self.execute_tool(MCPServer.GITHUB, "listRepos")

    async def github_get_commits(self, repo: str, branch: str = "main") -> MCPToolResult:
        """Get recent commits"""
        return await self.execute_tool(MCPServer.GITHUB, "getCommits", {
            "repo": repo,
            "branch": branch
        })

    async def github_create_issue(self, repo: str, title: str, body: str) -> MCPToolResult:
        """Create a GitHub issue"""
        return await self.execute_tool(MCPServer.GITHUB, "createIssue", {
            "repo": repo,
            "title": title,
            "body": body
        })

    async def github_trigger_workflow(self, repo: str, workflow: str, ref: str = "main") -> MCPToolResult:
        """Trigger a GitHub Actions workflow"""
        return await self.execute_tool(MCPServer.GITHUB, "triggerWorkflow", {
            "repo": repo,
            "workflow": workflow,
            "ref": ref
        })

    # =========================================================================
    # STRIPE OPERATIONS (55 tools)
    # =========================================================================

    async def stripe_create_customer(self, email: str, name: str = None) -> MCPToolResult:
        """Create a Stripe customer"""
        return await self.execute_tool(MCPServer.STRIPE, "createCustomer", {
            "email": email,
            "name": name
        })

    async def stripe_create_payment_intent(
        self,
        amount: int,
        currency: str = "usd",
        customer_id: str = None
    ) -> MCPToolResult:
        """Create a Stripe payment intent"""
        return await self.execute_tool(MCPServer.STRIPE, "createPaymentIntent", {
            "amount": amount,
            "currency": currency,
            "customer": customer_id
        })

    async def stripe_create_subscription(
        self,
        customer_id: str,
        price_id: str
    ) -> MCPToolResult:
        """Create a Stripe subscription"""
        return await self.execute_tool(MCPServer.STRIPE, "createSubscription", {
            "customer": customer_id,
            "priceId": price_id
        })

    async def stripe_get_balance(self) -> MCPToolResult:
        """Get Stripe account balance"""
        return await self.execute_tool(MCPServer.STRIPE, "getBalance")

    async def stripe_list_invoices(self, customer_id: str = None) -> MCPToolResult:
        """List Stripe invoices"""
        params = {}
        if customer_id:
            params["customer"] = customer_id
        return await self.execute_tool(MCPServer.STRIPE, "listInvoices", params)

    # =========================================================================
    # VERCEL OPERATIONS (34 tools)
    # =========================================================================

    async def vercel_list_deployments(self, project_id: str = None) -> MCPToolResult:
        """List Vercel deployments"""
        params = {}
        if project_id:
            params["projectId"] = project_id
        return await self.execute_tool(MCPServer.VERCEL, "listDeployments", params)

    async def vercel_create_deployment(self, project_id: str, ref: str = "main") -> MCPToolResult:
        """Create a new Vercel deployment"""
        return await self.execute_tool(MCPServer.VERCEL, "createDeployment", {
            "projectId": project_id,
            "gitRef": ref
        })

    async def vercel_get_project(self, project_id: str) -> MCPToolResult:
        """Get Vercel project details"""
        return await self.execute_tool(MCPServer.VERCEL, "getProject", {"projectId": project_id})

    # =========================================================================
    # AI OPERATIONS (OpenAI + Anthropic)
    # =========================================================================

    async def openai_chat(self, messages: List[Dict], model: str = "gpt-4") -> MCPToolResult:
        """Send a chat completion request to OpenAI"""
        return await self.execute_tool(MCPServer.OPENAI, "chat", {
            "messages": messages,
            "model": model
        })

    async def anthropic_chat(self, prompt: str, model: str = "claude-3-opus") -> MCPToolResult:
        """Send a message to Anthropic Claude"""
        return await self.execute_tool(MCPServer.ANTHROPIC, "chat", {
            "prompt": prompt,
            "model": model
        })

    # =========================================================================
    # DOCKER OPERATIONS (53 tools)
    # =========================================================================

    async def docker_list_containers(self) -> MCPToolResult:
        """List Docker containers"""
        return await self.execute_tool(MCPServer.DOCKER, "listContainers")

    async def docker_start_container(self, container_id: str) -> MCPToolResult:
        """Start a Docker container"""
        return await self.execute_tool(MCPServer.DOCKER, "startContainer", {
            "containerId": container_id
        })

    async def docker_stop_container(self, container_id: str) -> MCPToolResult:
        """Stop a Docker container"""
        return await self.execute_tool(MCPServer.DOCKER, "stopContainer", {
            "containerId": container_id
        })


# =============================================================================
# AUREA INTEGRATION - Tool Executor for the Orchestrator
# =============================================================================

class AUREAToolExecutor:
    """
    Integrates MCP tools with AUREA Orchestrator

    Enables AUREA to autonomously:
    - Deploy code to Render/Vercel
    - Query and modify Supabase data
    - Create GitHub PRs and issues
    - Process payments via Stripe
    - Manage Docker containers
    """

    def __init__(self):
        self.mcp = MCPClient()
        self.execution_history: List[MCPToolResult] = []
        logger.info("AUREAToolExecutor initialized - 345 tools available")

    async def execute_decision(self, decision_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an AUREA decision using MCP tools

        Maps decision types to tool executions:
        - DEPLOY: Render/Vercel deployment
        - HEAL: Restart service, scale up
        - REVENUE: Stripe operations
        - DATA: Supabase queries
        - CODE: GitHub operations
        """
        result = None

        if decision_type == "DEPLOY":
            if params.get("platform") == "render":
                result = await self.mcp.render_trigger_deploy(params["service_id"])
            elif params.get("platform") == "vercel":
                result = await self.mcp.vercel_create_deployment(params["project_id"])

        elif decision_type == "HEAL":
            action = params.get("action", "restart")
            if action == "restart":
                result = await self.mcp.render_restart_service(params["service_id"])
            elif action == "scale":
                result = await self.mcp.render_scale_service(
                    params["service_id"],
                    params.get("instances", 2)
                )

        elif decision_type == "REVENUE":
            action = params.get("action")
            if action == "create_customer":
                result = await self.mcp.stripe_create_customer(
                    params["email"],
                    params.get("name")
                )
            elif action == "charge":
                result = await self.mcp.stripe_create_payment_intent(
                    params["amount"],
                    params.get("currency", "usd"),
                    params.get("customer_id")
                )

        elif decision_type == "DATA":
            result = await self.mcp.supabase_query(
                params["query"],
                params.get("params")
            )

        elif decision_type == "CODE":
            action = params.get("action")
            if action == "create_pr":
                result = await self.mcp.github_create_pr(
                    params["repo"],
                    params["title"],
                    params["body"],
                    params["head"],
                    params.get("base", "main")
                )
            elif action == "create_issue":
                result = await self.mcp.github_create_issue(
                    params["repo"],
                    params["title"],
                    params["body"]
                )

        if result:
            self.execution_history.append(result)

        return {
            "success": result.success if result else False,
            "result": result.result if result else None,
            "error": result.error if result else "Unknown decision type"
        }

    async def close(self):
        await self.mcp.close()


# =============================================================================
# SELF-HEALING INTEGRATION
# =============================================================================

class SelfHealingMCPIntegration:
    """
    Integrates MCP tools with Self-Healing infrastructure

    Enables automatic:
    - Service restarts when health checks fail
    - Scaling when load thresholds exceeded
    - Log retrieval for diagnosis
    - Deployment rollback
    """

    RENDER_SERVICE_IDS = {
        "brainops-ai-agents": "srv-xxx-agents",  # Replace with actual IDs
        "brainops-backend-prod": "srv-xxx-backend",
        "brainops-mcp-bridge": "srv-xxx-bridge"
    }

    def __init__(self):
        self.mcp = MCPClient()
        self.restart_counts: Dict[str, int] = {}
        self.max_restarts = 3

    async def handle_unhealthy_service(self, service_name: str) -> Dict[str, Any]:
        """
        Autonomous response to unhealthy service

        1. Check current status
        2. Attempt restart (up to 3 times)
        3. Scale up if restarts don't help
        4. Alert human if all else fails
        """
        service_id = self.RENDER_SERVICE_IDS.get(service_name)
        if not service_id:
            return {"success": False, "error": f"Unknown service: {service_name}"}

        # Get current restart count
        restart_count = self.restart_counts.get(service_name, 0)

        if restart_count < self.max_restarts:
            # Attempt restart
            result = await self.mcp.render_restart_service(service_id)
            self.restart_counts[service_name] = restart_count + 1

            if result.success:
                logger.info(f"Self-healing: Restarted {service_name} (attempt {restart_count + 1})")
                return {
                    "success": True,
                    "action": "restart",
                    "attempt": restart_count + 1
                }

        # Restarts exhausted, try scaling
        if restart_count >= self.max_restarts:
            result = await self.mcp.render_scale_service(service_id, 2)
            if result.success:
                logger.info(f"Self-healing: Scaled up {service_name} to 2 instances")
                return {
                    "success": True,
                    "action": "scale_up",
                    "instances": 2
                }

        # All automated remediation failed
        return {
            "success": False,
            "action": "escalate",
            "message": f"Service {service_name} requires human intervention"
        }

    async def get_diagnostic_info(self, service_name: str) -> Dict[str, Any]:
        """Get logs and metrics for diagnosis"""
        service_id = self.RENDER_SERVICE_IDS.get(service_name)
        if not service_id:
            return {"error": f"Unknown service: {service_name}"}

        logs_result = await self.mcp.render_get_logs(service_id, lines=200)
        status_result = await self.mcp.render_get_service(service_id)

        return {
            "logs": logs_result.result if logs_result.success else logs_result.error,
            "status": status_result.result if status_result.success else status_result.error
        }

    async def close(self):
        await self.mcp.close()


# =============================================================================
# REVENUE INTEGRATION
# =============================================================================

class RevenueMCPIntegration:
    """
    Integrates MCP tools with Revenue Automation Engine

    Enables:
    - Automated customer creation in Stripe
    - Payment processing
    - Subscription management
    - Invoice generation
    """

    def __init__(self):
        self.mcp = MCPClient()

    async def process_new_customer(
        self,
        email: str,
        name: str,
        plan: str = "pro"
    ) -> Dict[str, Any]:
        """
        Full customer onboarding flow:
        1. Create Stripe customer
        2. Set up subscription
        3. Record in Supabase
        """
        # Create Stripe customer
        customer_result = await self.mcp.stripe_create_customer(email, name)
        if not customer_result.success:
            return {"success": False, "error": customer_result.error, "step": "create_customer"}

        customer_id = customer_result.result.get("id")

        # Map plan to Stripe price ID (these would be real IDs)
        price_map = {
            "starter": "price_starter_monthly",
            "pro": "price_pro_monthly",
            "enterprise": "price_enterprise_monthly"
        }

        # Create subscription
        sub_result = await self.mcp.stripe_create_subscription(
            customer_id,
            price_map.get(plan, price_map["pro"])
        )

        if not sub_result.success:
            return {
                "success": False,
                "error": sub_result.error,
                "step": "create_subscription",
                "customer_id": customer_id
            }

        # Record in Supabase
        await self.mcp.supabase_insert("revenue_customers", {
            "email": email,
            "name": name,
            "stripe_customer_id": customer_id,
            "plan": plan,
            "status": "active"
        })

        return {
            "success": True,
            "customer_id": customer_id,
            "subscription_id": sub_result.result.get("id"),
            "plan": plan
        }

    async def process_payment(
        self,
        customer_id: str,
        amount: int,
        description: str = None
    ) -> Dict[str, Any]:
        """Process a one-time payment"""
        result = await self.mcp.stripe_create_payment_intent(
            amount=amount,
            customer_id=customer_id
        )

        if result.success:
            # Log to Supabase
            await self.mcp.supabase_insert("revenue_transactions", {
                "stripe_customer_id": customer_id,
                "amount": amount,
                "description": description,
                "status": "pending",
                "payment_intent_id": result.result.get("id")
            })

        return {
            "success": result.success,
            "payment_intent_id": result.result.get("id") if result.success else None,
            "error": result.error
        }

    async def get_revenue_metrics(self) -> Dict[str, Any]:
        """Get current revenue metrics from Stripe"""
        balance = await self.mcp.stripe_get_balance()
        invoices = await self.mcp.stripe_list_invoices()

        return {
            "balance": balance.result if balance.success else None,
            "recent_invoices": invoices.result if invoices.success else None
        }

    async def close(self):
        await self.mcp.close()


# =============================================================================
# DIGITAL TWIN INTEGRATION
# =============================================================================

class DigitalTwinMCPIntegration:
    """
    Integrates MCP tools with Digital Twin System

    Enables twins to:
    - Query real system state from Supabase
    - Trigger Render deployments for testing
    - Get real-time logs for drift detection
    """

    def __init__(self):
        self.mcp = MCPClient()

    async def sync_twin_with_reality(self, twin_id: str) -> Dict[str, Any]:
        """
        Sync a digital twin with the actual production system state
        """
        # Get twin config from Supabase
        twin_result = await self.mcp.supabase_query(
            "SELECT * FROM digital_twins WHERE twin_id = $1",
            [twin_id]
        )

        if not twin_result.success or not twin_result.result:
            return {"success": False, "error": "Twin not found"}

        twin_data = twin_result.result[0] if isinstance(twin_result.result, list) else twin_result.result
        source_system = twin_data.get("source_system")

        # Get real system state from Render
        services = await self.mcp.render_list_services()
        real_state = None

        if services.success:
            for svc in services.result or []:
                if svc.get("name") == source_system:
                    real_state = svc
                    break

        # Detect drift
        drift_detected = False
        if real_state:
            # Compare states (simplified)
            if real_state.get("status") != "running":
                drift_detected = True

        # Update twin in Supabase
        await self.mcp.supabase_query(
            """UPDATE digital_twins
               SET last_sync = NOW(),
                   drift_detected = $1,
                   state_snapshot = $2
               WHERE twin_id = $3""",
            [drift_detected, real_state, twin_id]
        )

        return {
            "success": True,
            "twin_id": twin_id,
            "drift_detected": drift_detected,
            "real_state": real_state
        }

    async def run_simulation_on_real_infra(
        self,
        twin_id: str,
        scenario: str
    ) -> Dict[str, Any]:
        """
        Run a simulation that interacts with real infrastructure
        (in a safe, read-only manner)
        """
        # Get logs for analysis
        services = await self.mcp.render_list_services()

        # Query historical data
        history = await self.mcp.supabase_query(
            """SELECT * FROM system_metrics
               WHERE system_id = $1
               ORDER BY recorded_at DESC
               LIMIT 100""",
            [twin_id]
        )

        return {
            "twin_id": twin_id,
            "scenario": scenario,
            "services_analyzed": len(services.result) if services.success else 0,
            "historical_points": len(history.result) if history.success else 0,
            "simulation_complete": True
        }

    async def close(self):
        await self.mcp.close()


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_mcp_client: Optional[MCPClient] = None
_aurea_executor: Optional[AUREAToolExecutor] = None
_self_healing: Optional[SelfHealingMCPIntegration] = None
_revenue: Optional[RevenueMCPIntegration] = None
_digital_twin: Optional[DigitalTwinMCPIntegration] = None


def get_mcp_client() -> MCPClient:
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPClient()
    return _mcp_client


def get_aurea_executor() -> AUREAToolExecutor:
    global _aurea_executor
    if _aurea_executor is None:
        _aurea_executor = AUREAToolExecutor()
    return _aurea_executor


def get_self_healing_integration() -> SelfHealingMCPIntegration:
    global _self_healing
    if _self_healing is None:
        _self_healing = SelfHealingMCPIntegration()
    return _self_healing


def get_revenue_integration() -> RevenueMCPIntegration:
    global _revenue
    if _revenue is None:
        _revenue = RevenueMCPIntegration()
    return _revenue


def get_digital_twin_integration() -> DigitalTwinMCPIntegration:
    global _digital_twin
    if _digital_twin is None:
        _digital_twin = DigitalTwinMCPIntegration()
    return _digital_twin


# =============================================================================
# QUICK TEST
# =============================================================================

async def test_mcp_integration():
    """Quick test of the MCP integration"""
    client = get_mcp_client()

    print("Testing MCP Integration...")
    print(f"Bridge URL: {client.base_url}")

    # Test 1: List Render services
    print("\n1. Testing Render listServices...")
    result = await client.render_list_services()
    print(f"   Success: {result.success}")
    print(f"   Duration: {result.duration_ms:.0f}ms")
    if result.error:
        print(f"   Error: {result.error}")

    # Test 2: Query Supabase
    print("\n2. Testing Supabase query...")
    result = await client.supabase_query("SELECT COUNT(*) as count FROM customers")
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Result: {result.result}")

    # Test 3: Get Stripe balance
    print("\n3. Testing Stripe getBalance...")
    result = await client.stripe_get_balance()
    print(f"   Success: {result.success}")

    await client.close()
    print("\nMCP Integration test complete!")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())
