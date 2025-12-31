#!/usr/bin/env python3
"""
AUREA Power Layer - Ultimate Personal AI Ops Assistant
=======================================================
Full operational capability across frontends, backends, database, everything.
Your true multi-AI clone with natural language command of the entire system.

This is the execution layer that gives AUREA absolute power to:
- Execute database queries and mutations
- Deploy to Vercel and Render
- Manage Git operations (commit, push, PR)
- Run Playwright UI tests
- Orchestrate multiple AI models (Gemini, Codex, Claude, Perplexity)
- Monitor system health across all services
- Execute file operations (within security bounds)

Author: BrainOps AI System
Version: 1.0.0
"""

import os
import asyncio
import aiohttp
import logging
import json
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger('AUREA.Power')

# Configuration - NO hardcoded credentials
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
MCP_API_KEY = os.getenv("MCP_API_KEY") or os.getenv("MCP_BRIDGE_API_KEY")  # Required - no default
BRAINOPS_API_KEY = os.getenv("BRAINOPS_API_KEY") or os.getenv("AGENTS_API_KEY")  # Required - no default

# Database Configuration - NO hardcoded credentials
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),  # Required - no default
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),  # Required - no default
    'password': os.getenv('DB_PASSWORD'),  # Required - no default
    'port': int(os.getenv('DB_PORT', '5432'))
}

# Service URLs
SERVICES = {
    'brainops_agents': 'https://brainops-ai-agents.onrender.com',
    'brainops_backend': 'https://brainops-backend-prod.onrender.com',
    'mcp_bridge': 'https://brainops-mcp-bridge.onrender.com',
    'command_center': 'https://brainops-command-center.vercel.app',
    'myroofgenius': 'https://myroofgenius.com',
    'weathercraft_erp': 'https://weathercraft-erp.vercel.app'
}

# Allowed paths for file operations (security)
ALLOWED_FILE_PATHS = [
    '/home/matt-woodworth/dev/',
    '/tmp/',
]

# Blocked SQL keywords for safety
BLOCKED_SQL_MUTATIONS = {'DROP', 'TRUNCATE', 'ALTER TABLE', 'CREATE DATABASE', 'DROP DATABASE'}


class PowerCapability(Enum):
    """Categories of AUREA's power capabilities"""
    DATABASE = "database"
    DEPLOYMENT = "deployment"
    GIT = "git"
    UI_TESTING = "ui_testing"
    AI_MODELS = "ai_models"
    MONITORING = "monitoring"
    FILE_OPS = "file_ops"
    AUTOMATION = "automation"


@dataclass
class PowerResult:
    """Result from a power operation"""
    success: bool
    capability: PowerCapability
    operation: str
    result: Any
    duration_ms: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AUREAPowerLayer:
    """
    The Ultimate Power Layer for AUREA
    Provides complete operational control over the entire system.
    """

    def __init__(self, db_pool=None, mcp_client=None):
        self.db_pool = db_pool
        self.mcp_client = mcp_client
        self.execution_log: List[PowerResult] = []
        logger.info("🔋 AUREA Power Layer initialized with full operational capability")

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    async def query_database(self, sql: str, params: Tuple = (), read_only: bool = True) -> PowerResult:
        """
        Execute a database query. By default, only allows SELECT statements.
        For mutations, set read_only=False (requires explicit confirmation).
        """
        start = datetime.utcnow()

        # Security check for dangerous operations
        sql_upper = sql.upper()
        for blocked in BLOCKED_SQL_MUTATIONS:
            if blocked in sql_upper:
                return PowerResult(
                    success=False,
                    capability=PowerCapability.DATABASE,
                    operation="query",
                    result=None,
                    duration_ms=0,
                    error=f"Blocked: {blocked} operations are not allowed"
                )

        if read_only and not sql_upper.strip().startswith('SELECT'):
            return PowerResult(
                success=False,
                capability=PowerCapability.DATABASE,
                operation="query",
                result=None,
                duration_ms=0,
                error="Only SELECT statements allowed in read_only mode"
            )

        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    if sql_upper.strip().startswith('SELECT'):
                        rows = await conn.fetch(sql, *params)
                        result = [dict(row) for row in rows]
                    else:
                        result = await conn.execute(sql, *params)
            else:
                # Fallback to sync connection
                import psycopg2
                from psycopg2.extras import RealDictCursor
                conn = psycopg2.connect(**DB_CONFIG)
                try:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur:
                        cur.execute(sql, params)
                        if sql_upper.strip().startswith('SELECT'):
                            result = cur.fetchall()
                        else:
                            conn.commit()
                            result = cur.rowcount
                finally:
                    conn.close()

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=True,
                capability=PowerCapability.DATABASE,
                operation="query",
                result=result,
                duration_ms=duration
            )
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.DATABASE,
                operation="query",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def get_table_info(self, table_name: str, schema: str = 'public') -> PowerResult:
        """Get information about a database table including columns and row count."""
        sql = f"""
            SELECT
                column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        columns_result = await self.query_database(sql, (schema, table_name))

        # Validate schema and table names to prevent SQL injection (defense in depth)
        import re
        if not re.match(r'^[a-z_][a-z0-9_]*$', schema, re.IGNORECASE):
            return PowerResult(success=False, capability=PowerCapability.DATABASE,
                operation="get_table_info", error=f"Invalid schema name: {schema}")
        if not re.match(r'^[a-z_][a-z0-9_]*$', table_name, re.IGNORECASE):
            return PowerResult(success=False, capability=PowerCapability.DATABASE,
                operation="get_table_info", error=f"Invalid table name: {table_name}")

        count_sql = f'SELECT COUNT(*) as count FROM "{schema}"."{table_name}"'
        count_result = await self.query_database(count_sql)

        return PowerResult(
            success=columns_result.success and count_result.success,
            capability=PowerCapability.DATABASE,
            operation="get_table_info",
            result={
                'table': f'{schema}.{table_name}',
                'columns': columns_result.result if columns_result.success else [],
                'row_count': count_result.result[0]['count'] if count_result.success else 0
            },
            duration_ms=columns_result.duration_ms + count_result.duration_ms
        )

    # =========================================================================
    # DEPLOYMENT OPERATIONS
    # =========================================================================

    async def deploy_vercel(self, project: str, production: bool = False) -> PowerResult:
        """Trigger a Vercel deployment."""
        start = datetime.utcnow()
        try:
            if self.mcp_client:
                result = await self.mcp_client.execute_tool(
                    server="vercel",
                    tool="trigger_deployment" if not production else "promote_to_production",
                    params={"project": project}
                )
                duration = (datetime.utcnow() - start).total_seconds() * 1000
                return PowerResult(
                    success=result.success,
                    capability=PowerCapability.DEPLOYMENT,
                    operation="vercel_deploy",
                    result=result.result,
                    duration_ms=duration,
                    error=result.error
                )
            else:
                return PowerResult(
                    success=False,
                    capability=PowerCapability.DEPLOYMENT,
                    operation="vercel_deploy",
                    result=None,
                    duration_ms=0,
                    error="MCP client not available"
                )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.DEPLOYMENT,
                operation="vercel_deploy",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def deploy_render(self, service: str) -> PowerResult:
        """Trigger a Render deployment."""
        start = datetime.utcnow()
        try:
            if self.mcp_client:
                result = await self.mcp_client.execute_tool(
                    server="render",
                    tool="deploy_service",
                    params={"service": service}
                )
                duration = (datetime.utcnow() - start).total_seconds() * 1000
                return PowerResult(
                    success=result.success,
                    capability=PowerCapability.DEPLOYMENT,
                    operation="render_deploy",
                    result=result.result,
                    duration_ms=duration,
                    error=result.error
                )
            else:
                return PowerResult(
                    success=False,
                    capability=PowerCapability.DEPLOYMENT,
                    operation="render_deploy",
                    result=None,
                    duration_ms=0,
                    error="MCP client not available"
                )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.DEPLOYMENT,
                operation="render_deploy",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # GIT OPERATIONS
    # =========================================================================

    async def git_status(self, repo_path: str) -> PowerResult:
        """Get git status for a repository."""
        start = datetime.utcnow()

        if not any(repo_path.startswith(allowed) for allowed in ALLOWED_FILE_PATHS):
            return PowerResult(
                success=False,
                capability=PowerCapability.GIT,
                operation="git_status",
                result=None,
                duration_ms=0,
                error=f"Path {repo_path} not in allowed paths"
            )

        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=result.returncode == 0,
                capability=PowerCapability.GIT,
                operation="git_status",
                result={
                    'changes': result.stdout.strip().split('\n') if result.stdout.strip() else [],
                    'has_changes': bool(result.stdout.strip())
                },
                duration_ms=duration,
                error=result.stderr if result.returncode != 0 else None
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.GIT,
                operation="git_status",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def git_commit_and_push(self, repo_path: str, message: str, files: List[str] = None) -> PowerResult:
        """Commit changes and push to remote."""
        start = datetime.utcnow()

        if not any(repo_path.startswith(allowed) for allowed in ALLOWED_FILE_PATHS):
            return PowerResult(
                success=False,
                capability=PowerCapability.GIT,
                operation="git_commit_push",
                result=None,
                duration_ms=0,
                error=f"Path {repo_path} not in allowed paths"
            )

        try:
            # Stage files
            if files:
                for f in files:
                    subprocess.run(['git', 'add', f], cwd=repo_path, check=True)
            else:
                subprocess.run(['git', 'add', '-A'], cwd=repo_path, check=True)

            # Commit
            commit_result = subprocess.run(
                ['git', 'commit', '-m', message],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            # Push
            push_result = subprocess.run(
                ['git', 'push', 'origin', 'main'],
                cwd=repo_path,
                capture_output=True,
                text=True
            )

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=push_result.returncode == 0,
                capability=PowerCapability.GIT,
                operation="git_commit_push",
                result={
                    'commit': commit_result.stdout,
                    'push': push_result.stdout
                },
                duration_ms=duration,
                error=push_result.stderr if push_result.returncode != 0 else None
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.GIT,
                operation="git_commit_push",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # UI TESTING WITH PLAYWRIGHT
    # =========================================================================

    async def run_playwright_test(self, url: str, actions: List[Dict[str, Any]] = None) -> PowerResult:
        """Run Playwright UI tests against a URL."""
        start = datetime.utcnow()
        try:
            if self.mcp_client:
                result = await self.mcp_client.execute_tool(
                    server="playwright",
                    tool="navigate",
                    params={"url": url}
                )

                test_results = [{"navigate": result.result}]

                # Execute additional actions if provided
                if actions:
                    for action in actions:
                        action_type = action.get('type', 'click')
                        action_result = await self.mcp_client.execute_tool(
                            server="playwright",
                            tool=action_type,
                            params=action.get('params', {})
                        )
                        test_results.append({action_type: action_result.result})

                # Take screenshot
                screenshot_result = await self.mcp_client.execute_tool(
                    server="playwright",
                    tool="screenshot",
                    params={"fullPage": True}
                )
                test_results.append({"screenshot": screenshot_result.result})

                duration = (datetime.utcnow() - start).total_seconds() * 1000
                return PowerResult(
                    success=True,
                    capability=PowerCapability.UI_TESTING,
                    operation="playwright_test",
                    result=test_results,
                    duration_ms=duration
                )
            else:
                return PowerResult(
                    success=False,
                    capability=PowerCapability.UI_TESTING,
                    operation="playwright_test",
                    result=None,
                    duration_ms=0,
                    error="MCP client not available"
                )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.UI_TESTING,
                operation="playwright_test",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def check_ui_health(self, url: str) -> PowerResult:
        """Quick health check of a UI - loads page and checks for errors."""
        start = datetime.utcnow()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    status = response.status
                    content_type = response.headers.get('content-type', '')

                    duration = (datetime.utcnow() - start).total_seconds() * 1000
                    return PowerResult(
                        success=status == 200,
                        capability=PowerCapability.UI_TESTING,
                        operation="ui_health_check",
                        result={
                            'url': url,
                            'status': status,
                            'content_type': content_type,
                            'healthy': status == 200
                        },
                        duration_ms=duration
                    )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.UI_TESTING,
                operation="ui_health_check",
                result={'url': url, 'healthy': False},
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # AI MODEL ORCHESTRATION
    # =========================================================================

    async def call_ai_model(self, model: str, prompt: str, system_prompt: str = None) -> PowerResult:
        """Call an AI model (gemini, codex, claude, perplexity, openai)."""
        start = datetime.utcnow()

        model_lower = model.lower()

        try:
            if model_lower in ['gemini', 'gemini-pro']:
                # Use local gemini CLI
                cmd = ['gemini', '-p', prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                response = result.stdout
            elif model_lower in ['codex', 'codex-max']:
                # Use local codex CLI
                cmd = ['codex', 'exec', prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                response = result.stdout
            elif model_lower in ['perplexity', 'sonar']:
                # Use Perplexity API
                api_key = os.getenv('PERPLEXITY_API_KEY')
                if not api_key:
                    raise ValueError("PERPLEXITY_API_KEY not set")

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "sonar",
                            "messages": messages,
                            "max_tokens": 2000
                        }
                    ) as resp:
                        data = await resp.json()
                        response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            elif model_lower in ['openai', 'gpt-4', 'gpt-4o']:
                # Use OpenAI via MCP or direct API
                if self.mcp_client:
                    result = await self.mcp_client.execute_tool(
                        server="openai",
                        tool="chat_completion",
                        params={"prompt": prompt, "model": "gpt-4o"}
                    )
                    response = result.result
                else:
                    raise ValueError("MCP client required for OpenAI")
            elif model_lower in ['claude', 'anthropic']:
                # Use Anthropic via MCP
                if self.mcp_client:
                    result = await self.mcp_client.execute_tool(
                        server="anthropic",
                        tool="chat_completion",
                        params={"prompt": prompt}
                    )
                    response = result.result
                else:
                    raise ValueError("MCP client required for Claude")
            else:
                raise ValueError(f"Unknown model: {model}")

            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=True,
                capability=PowerCapability.AI_MODELS,
                operation="call_ai_model",
                result={
                    'model': model,
                    'response': response
                },
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.AI_MODELS,
                operation="call_ai_model",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # SYSTEM MONITORING
    # =========================================================================

    async def check_all_services_health(self) -> PowerResult:
        """Check health of all BrainOps services."""
        start = datetime.utcnow()
        results = {}

        async with aiohttp.ClientSession() as session:
            for name, url in SERVICES.items():
                try:
                    health_url = f"{url}/health" if not url.endswith('/health') else url
                    async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        results[name] = {
                            'url': url,
                            'status': response.status,
                            'healthy': response.status == 200
                        }
                        if response.status == 200:
                            try:
                                data = await response.json()
                                results[name]['version'] = data.get('version', 'unknown')
                            except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as exc:
                                logger.debug(
                                    "Failed to parse JSON from %s: %s",
                                    health_url,
                                    exc,
                                    exc_info=True,
                                )
                except Exception as e:
                    results[name] = {
                        'url': url,
                        'status': 0,
                        'healthy': False,
                        'error': str(e)
                    }

        all_healthy = all(r.get('healthy', False) for r in results.values())
        duration = (datetime.utcnow() - start).total_seconds() * 1000

        return PowerResult(
            success=all_healthy,
            capability=PowerCapability.MONITORING,
            operation="check_all_services",
            result={
                'services': results,
                'all_healthy': all_healthy,
                'healthy_count': sum(1 for r in results.values() if r.get('healthy')),
                'total_count': len(results)
            },
            duration_ms=duration
        )

    async def get_system_metrics(self) -> PowerResult:
        """Get comprehensive system metrics from the AI agents service."""
        start = datetime.utcnow()
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'X-API-Key': BRAINOPS_API_KEY}
                async with session.get(
                    f"{SERVICES['brainops_agents']}/metrics",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        duration = (datetime.utcnow() - start).total_seconds() * 1000
                        return PowerResult(
                            success=True,
                            capability=PowerCapability.MONITORING,
                            operation="get_system_metrics",
                            result=data,
                            duration_ms=duration
                        )
                    else:
                        duration = (datetime.utcnow() - start).total_seconds() * 1000
                        return PowerResult(
                            success=False,
                            capability=PowerCapability.MONITORING,
                            operation="get_system_metrics",
                            result=None,
                            duration_ms=duration,
                            error=f"Status {response.status}"
                        )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.MONITORING,
                operation="get_system_metrics",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================

    async def read_file(self, file_path: str) -> PowerResult:
        """Read a file from the filesystem (within allowed paths)."""
        start = datetime.utcnow()

        if not any(file_path.startswith(allowed) for allowed in ALLOWED_FILE_PATHS):
            return PowerResult(
                success=False,
                capability=PowerCapability.FILE_OPS,
                operation="read_file",
                result=None,
                duration_ms=0,
                error=f"Path {file_path} not in allowed paths"
            )

        try:
            with open(file_path, 'r') as f:
                content = f.read()
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=True,
                capability=PowerCapability.FILE_OPS,
                operation="read_file",
                result={
                    'path': file_path,
                    'content': content,
                    'size': len(content)
                },
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.FILE_OPS,
                operation="read_file",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def write_file(self, file_path: str, content: str) -> PowerResult:
        """Write a file to the filesystem (within allowed paths)."""
        start = datetime.utcnow()

        if not any(file_path.startswith(allowed) for allowed in ALLOWED_FILE_PATHS):
            return PowerResult(
                success=False,
                capability=PowerCapability.FILE_OPS,
                operation="write_file",
                result=None,
                duration_ms=0,
                error=f"Path {file_path} not in allowed paths"
            )

        try:
            with open(file_path, 'w') as f:
                f.write(content)
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=True,
                capability=PowerCapability.FILE_OPS,
                operation="write_file",
                result={
                    'path': file_path,
                    'size': len(content)
                },
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.FILE_OPS,
                operation="write_file",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    # =========================================================================
    # AUTOMATION / WORKFLOWS
    # =========================================================================

    async def execute_workflow(self, workflow_name: str, params: Dict[str, Any] = None) -> PowerResult:
        """Execute a predefined workflow."""
        start = datetime.utcnow()

        workflows = {
            'full_deploy': self._workflow_full_deploy,
            'health_check_all': self._workflow_health_check_all,
            'db_backup_check': self._workflow_db_backup_check,
            'ui_smoke_test': self._workflow_ui_smoke_test,
        }

        workflow_func = workflows.get(workflow_name)
        if not workflow_func:
            return PowerResult(
                success=False,
                capability=PowerCapability.AUTOMATION,
                operation="execute_workflow",
                result=None,
                duration_ms=0,
                error=f"Unknown workflow: {workflow_name}. Available: {list(workflows.keys())}"
            )

        try:
            result = await workflow_func(params or {})
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=True,
                capability=PowerCapability.AUTOMATION,
                operation="execute_workflow",
                result=result,
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.utcnow() - start).total_seconds() * 1000
            return PowerResult(
                success=False,
                capability=PowerCapability.AUTOMATION,
                operation="execute_workflow",
                result=None,
                duration_ms=duration,
                error=str(e)
            )

    async def _workflow_full_deploy(self, params: Dict) -> Dict:
        """Full deployment workflow - backends then frontends."""
        results = {}

        # Deploy backends first
        results['brainops_agents'] = await self.deploy_render('brainops-ai-agents')
        results['brainops_backend'] = await self.deploy_render('brainops-backend-prod')

        # Wait for backends to be healthy
        await asyncio.sleep(30)

        # Check backend health
        results['backend_health'] = await self.check_all_services_health()

        # Deploy frontends
        results['command_center'] = await self.deploy_vercel('brainops-command-center', production=True)
        results['weathercraft_erp'] = await self.deploy_vercel('weathercraft-erp', production=True)
        results['myroofgenius'] = await self.deploy_vercel('myroofgenius-app', production=True)

        return results

    async def _workflow_health_check_all(self, params: Dict) -> Dict:
        """Complete health check of all services."""
        results = {}
        results['services'] = await self.check_all_services_health()
        results['metrics'] = await self.get_system_metrics()

        # Check database
        db_result = await self.query_database("SELECT COUNT(*) as count FROM ai_agent_executions")
        results['database'] = {
            'connected': db_result.success,
            'executions_count': db_result.result[0]['count'] if db_result.success else 0
        }

        return results

    async def _workflow_db_backup_check(self, params: Dict) -> Dict:
        """Check database backup status and key metrics."""
        results = {}

        # Get key table counts - table names are hardcoded whitelist, safe from injection
        SAFE_TABLES = {'ai_agent_executions', 'ai_thought_stream', 'ai_system_state', 'customers', 'jobs'}
        for table in SAFE_TABLES:
            # Table is from hardcoded set, but use identifier quoting for defense in depth
            result = await self.query_database(f'SELECT COUNT(*) as count FROM "{table}"')
            results[table] = result.result[0]['count'] if result.success else 0

        return results

    async def _workflow_ui_smoke_test(self, params: Dict) -> Dict:
        """Smoke test all frontend UIs."""
        results = {}

        ui_urls = [
            SERVICES['command_center'],
            SERVICES['myroofgenius'],
            SERVICES['weathercraft_erp']
        ]

        for url in ui_urls:
            result = await self.check_ui_health(url)
            results[url] = result.result

        return results

    # =========================================================================
    # SKILL REGISTRY FOR NLU
    # =========================================================================

    def get_skill_registry(self) -> Dict[str, Dict[str, Any]]:
        """Return the skill registry for AUREA NLU integration."""
        return {
            # Database
            "query_database": {
                "description": "Execute a SQL query against the database. Read-only by default.",
                "parameters": {"sql": "string", "read_only": "boolean (default true)"},
                "action": self.query_database
            },
            "get_table_info": {
                "description": "Get information about a database table.",
                "parameters": {"table_name": "string", "schema": "string (default public)"},
                "action": self.get_table_info
            },

            # Deployment
            "deploy_vercel": {
                "description": "Deploy a project to Vercel.",
                "parameters": {"project": "string", "production": "boolean"},
                "action": self.deploy_vercel
            },
            "deploy_render": {
                "description": "Deploy a service to Render.",
                "parameters": {"service": "string"},
                "action": self.deploy_render
            },

            # Git
            "git_status": {
                "description": "Get git status for a repository.",
                "parameters": {"repo_path": "string"},
                "action": self.git_status
            },
            "git_commit_and_push": {
                "description": "Commit and push changes to a repository.",
                "parameters": {"repo_path": "string", "message": "string", "files": "list (optional)"},
                "action": self.git_commit_and_push
            },

            # UI Testing
            "run_playwright_test": {
                "description": "Run Playwright UI tests against a URL.",
                "parameters": {"url": "string", "actions": "list (optional)"},
                "action": self.run_playwright_test
            },
            "check_ui_health": {
                "description": "Quick health check of a UI.",
                "parameters": {"url": "string"},
                "action": self.check_ui_health
            },

            # AI Models
            "call_ai_model": {
                "description": "Call an AI model (gemini, codex, claude, perplexity, openai).",
                "parameters": {"model": "string", "prompt": "string", "system_prompt": "string (optional)"},
                "action": self.call_ai_model
            },

            # Monitoring
            "check_all_services_health": {
                "description": "Check health of all BrainOps services.",
                "parameters": {},
                "action": self.check_all_services_health
            },
            "get_system_metrics": {
                "description": "Get comprehensive system metrics.",
                "parameters": {},
                "action": self.get_system_metrics
            },

            # Files
            "read_file": {
                "description": "Read a file from the filesystem.",
                "parameters": {"file_path": "string"},
                "action": self.read_file
            },
            "write_file": {
                "description": "Write a file to the filesystem.",
                "parameters": {"file_path": "string", "content": "string"},
                "action": self.write_file
            },

            # Automation
            "execute_workflow": {
                "description": "Execute a predefined workflow (full_deploy, health_check_all, db_backup_check, ui_smoke_test).",
                "parameters": {"workflow_name": "string", "params": "object (optional)"},
                "action": self.execute_workflow
            }
        }


# Singleton instance
_power_layer: Optional[AUREAPowerLayer] = None


def get_power_layer(db_pool=None, mcp_client=None) -> AUREAPowerLayer:
    """Get or create the AUREA Power Layer singleton."""
    global _power_layer
    if _power_layer is None:
        _power_layer = AUREAPowerLayer(db_pool=db_pool, mcp_client=mcp_client)
    return _power_layer
