#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for BrainOps AI Agents
Provides complete transparency and control for AI systems

Version: 2.0.0

Enhancements:
- Tool discovery and registration endpoints
- Performance metrics tracking
- Execution history with detailed logging
- Tool chaining for complex workflows
- Cache management endpoints
- Enhanced monitoring and diagnostics
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import subprocess
import httpx
from pathlib import Path
from config import config
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(
    request: Request,
    api_key: str = Security(api_key_header),
) -> bool:
    """Verify API key if authentication is required"""
    if not config.security.auth_required:
        return True

    provided = api_key
    if not provided:
        auth_header = request.headers.get("authorization")
        if auth_header:
            scheme, _, token = auth_header.partition(" ")
            scheme_lower = scheme.lower()
            if scheme_lower in ("bearer", "apikey", "api-key"):
                provided = token.strip()

    if not provided and config.security.test_api_key:
        provided = (
            request.headers.get("x-test-api-key")
            or request.headers.get("X-Test-Api-Key")
            or request.headers.get("x-api-key")
            or request.headers.get("X-API-Key")
        )

    if not provided:
        raise HTTPException(status_code=403, detail="API key required")

    if provided not in config.security.valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return True

# Initialize FastAPI app
app = FastAPI(
    title="BrainOps MCP Server",
    description="Model Context Protocol server for complete AI system transparency with enhanced features",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.security.allowed_origins if config.security.allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': 'postgres',
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': 5432
}

# Service endpoints
SERVICES = {
    'ai_agents': 'https://brainops-ai-agents.onrender.com',
    'erp_backend': 'https://myroofgenius.com/api',
    'erp_frontend': 'https://myroofgenius.com'
}

class MCPServer:
    """Enhanced MCP Server for AI system control and monitoring"""

    def __init__(self):
        self.connections = []
        self.system_state = {}
        self.active_tools = []
        self.execution_history = deque(maxlen=1000)  # Keep last 1000 executions

        # Enhanced features
        self._registered_tools: Dict[str, Dict] = {}
        self._tool_metrics: Dict[str, Dict] = defaultdict(lambda: {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration_ms": 0.0,
            "avg_duration_ms": 0.0,
            "last_execution": None
        })
        self._chain_executions: List[Dict] = []
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'timestamp': datetime.utcnow().isoformat(),
            'services': {},
            'database': {},
            'ai_agents': {},
            'errors': [],
            'metrics': {}
        }

        # Check services
        async with httpx.AsyncClient(timeout=5.0) as client:
            for name, url in SERVICES.items():
                try:
                    response = await client.get(f"{url}/health")
                    status['services'][name] = {
                        'status': 'online',
                        'code': response.status_code,
                        'version': response.json().get('version', 'unknown')
                    }
                except Exception as e:
                    status['services'][name] = {
                        'status': 'offline',
                        'error': str(e)
                    }

        # Check database
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get database metrics
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM ai_agents WHERE status = 'active') as active_agents,
                    (SELECT COUNT(*) FROM agent_executions WHERE created_at > NOW() - INTERVAL '1 hour') as recent_executions,
                    (SELECT COUNT(*) FROM ai_master_context) as memory_entries,
                    (SELECT pg_database_size('postgres')) as db_size
            """)
            metrics = cursor.fetchone()
            status['database'] = {
                'connected': True,
                'metrics': metrics
            }

            cursor.close()
            conn.close()
        except Exception as e:
            status['database'] = {
                'connected': False,
                'error': str(e)
            }

        return status

    async def execute_tool(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool with full transparency"""
        execution_id = datetime.utcnow().isoformat()

        result = {
            'tool': tool,
            'params': params,
            'execution_id': execution_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'pending',
            'result': None,
            'error': None
        }

        try:
            if tool == 'bash':
                # Execute bash command
                cmd = params.get('command', '')
                timeout = params.get('timeout', 30)

                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )

                result['result'] = {
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'returncode': proc.returncode
                }
                result['status'] = 'success' if proc.returncode == 0 else 'failed'

            elif tool == 'read_file':
                # Read file contents
                file_path = params.get('path', '')
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    result['result'] = {'content': content}
                    result['status'] = 'success'
                else:
                    result['error'] = f"File not found: {file_path}"
                    result['status'] = 'failed'

            elif tool == 'write_file':
                # Write file contents
                file_path = params.get('path', '')
                content = params.get('content', '')

                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    f.write(content)

                result['result'] = {'written': len(content)}
                result['status'] = 'success'

            elif tool == 'database_query':
                # Execute database query
                query = params.get('query', '')
                conn = psycopg2.connect(**DB_CONFIG)
                cursor = conn.cursor(cursor_factory=RealDictCursor)

                cursor.execute(query)

                if query.strip().upper().startswith('SELECT'):
                    rows = cursor.fetchall()
                    result['result'] = {'rows': rows, 'count': len(rows)}
                else:
                    conn.commit()
                    result['result'] = {'affected': cursor.rowcount}

                cursor.close()
                conn.close()
                result['status'] = 'success'

            elif tool == 'http_request':
                # Make HTTP request
                method = params.get('method', 'GET')
                url = params.get('url', '')
                headers = params.get('headers', {})
                body = params.get('body', None)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.request(
                        method=method,
                        url=url,
                        headers=headers,
                        json=body if body else None
                    )

                    result['result'] = {
                        'status_code': response.status_code,
                        'headers': dict(response.headers),
                        'body': response.text[:10000]  # Limit response size
                    }
                    result['status'] = 'success'

            else:
                result['error'] = f"Unknown tool: {tool}"
                result['status'] = 'failed'

        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'failed'

        # Store execution history
        self.execution_history.append(result)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]

        return result

    async def get_file_tree(self, path: str, max_depth: int = 3) -> Dict[str, Any]:
        """Get file tree structure for transparency"""
        def build_tree(dir_path: Path, current_depth: int = 0):
            tree = {
                'name': dir_path.name,
                'path': str(dir_path),
                'type': 'directory',
                'children': []
            }

            if current_depth >= max_depth:
                return tree

            try:
                for item in dir_path.iterdir():
                    if item.is_dir() and not item.name.startswith('.'):
                        tree['children'].append(
                            build_tree(item, current_depth + 1)
                        )
                    elif item.is_file():
                        tree['children'].append({
                            'name': item.name,
                            'path': str(item),
                            'type': 'file',
                            'size': item.stat().st_size
                        })
            except PermissionError:
                pass

            return tree

        root_path = Path(path)
        if not root_path.exists():
            return {'error': f"Path not found: {path}"}

        return build_tree(root_path)

    async def monitor_logs(self, service: str, lines: int = 100) -> List[str]:
        """Monitor service logs in real-time"""
        if service == 'ai_agents':
            # Get Render logs via API
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{SERVICES['ai_agents']}/logs",
                        params={'lines': lines}
                    )
                    if response.status_code == 200:
                        return response.json().get('logs', [])
            except:
                pass

        # Fallback to local logs if available
        log_files = {
            'ai_agents': '/var/log/ai_agents.log',
            'erp_frontend': '/var/log/erp_frontend.log',
            'database': '/var/log/postgresql/postgresql.log'
        }

        log_file = log_files.get(service)
        if log_file and os.path.exists(log_file):
            try:
                # SECURITY FIX: Sanitize lines parameter to prevent command injection
                safe_lines = max(1, min(int(lines), 10000))  # Clamp to reasonable range
                # Use subprocess with args list instead of shell=True to prevent injection
                proc = await asyncio.create_subprocess_exec(
                    'tail', '-n', str(safe_lines), log_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                return stdout.decode().split('\n')
            except Exception as e:
                logger.warning(f"Failed to read log file {log_file}: {e}")
                pass

        return []

mcp_server = MCPServer()

@app.get("/")
async def root():
    """Root endpoint with MCP information"""
    return {
        "name": "BrainOps MCP Server",
        "version": "1.0.0",
        "protocol": "MCP",
        "capabilities": [
            "system_status",
            "tool_execution",
            "file_operations",
            "database_access",
            "http_requests",
            "log_monitoring",
            "real_time_updates"
        ],
        "tools": [
            "bash",
            "read_file",
            "write_file",
            "database_query",
            "http_request"
        ]
    }

@app.get("/status")
async def get_status():
    """Get comprehensive system status"""
    return await mcp_server.get_system_status()

@app.post("/execute", dependencies=[Depends(verify_api_key)])
async def execute_tool(request: Dict[str, Any]):
    """Execute an MCP tool"""
    tool = request.get('tool')
    params = request.get('params', {})

    if not tool:
        raise HTTPException(status_code=400, detail="Tool name required")

    result = await mcp_server.execute_tool(tool, params)
    return result

@app.get("/files", dependencies=[Depends(verify_api_key)])
async def get_files(path: str = "/home/matt-woodworth", depth: int = 2):
    """Get file tree structure"""
    return await mcp_server.get_file_tree(path, depth)

@app.get("/logs/{service}", dependencies=[Depends(verify_api_key)])
async def get_logs(service: str, lines: int = 100):
    """Get service logs"""
    logs = await mcp_server.monitor_logs(service, lines)
    return {"service": service, "logs": logs}

@app.get("/history", dependencies=[Depends(verify_api_key)])
async def get_history(limit: int = 100):
    """Get execution history"""
    return {
        "history": mcp_server.execution_history[-limit:],
        "total": len(mcp_server.execution_history)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    mcp_server.connections.append(websocket)

    try:
        while True:
            # Send system status every 5 seconds
            status = await mcp_server.get_system_status()
            await websocket.send_json({
                'type': 'status',
                'data': status
            })
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        mcp_server.connections.remove(websocket)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": len(mcp_server.connections)
    }

# Vercel MCP adapter endpoints
@app.get("/mcp/tools")
async def get_mcp_tools():
    """Get available MCP tools for Vercel"""
    return {
        "tools": [
            {
                "name": "bash",
                "description": "Execute bash commands",
                "parameters": {
                    "command": "string",
                    "timeout": "number"
                }
            },
            {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "path": "string"
                }
            },
            {
                "name": "write_file",
                "description": "Write file contents",
                "parameters": {
                    "path": "string",
                    "content": "string"
                }
            },
            {
                "name": "database_query",
                "description": "Execute database query",
                "parameters": {
                    "query": "string"
                }
            },
            {
                "name": "http_request",
                "description": "Make HTTP request",
                "parameters": {
                    "method": "string",
                    "url": "string",
                    "headers": "object",
                    "body": "object"
                }
            }
        ]
    }

@app.post("/mcp/execute", dependencies=[Depends(verify_api_key)])
async def mcp_execute(request: Dict[str, Any]):
    """Execute MCP tool via Vercel adapter"""
    return await execute_tool(request)

# =============================================================================
# ENHANCED ENDPOINTS - Tool Discovery, Metrics, and Chaining
# =============================================================================

@app.get("/mcp/discover")
async def mcp_discover_tools():
    """
    Discover all available MCP tools from connected servers

    Returns a comprehensive catalog of all available tools with their
    descriptions, parameters, and server information.
    """
    try:
        # Import here to avoid circular dependency
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        discovered = await client.discover_tools(force_refresh=True)

        tools_by_server = {}
        total_tools = 0

        for server, tools in discovered.items():
            tools_by_server[server] = {
                "count": len(tools),
                "tools": [
                    {
                        "name": tool.tool_name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "use_count": tool.use_count,
                        "enabled": tool.enabled
                    }
                    for tool in tools
                ]
            }
            total_tools += len(tools)

        return {
            "total_tools": total_tools,
            "total_servers": len(discovered),
            "servers": tools_by_server,
            "discovery_timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Tool discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/metrics")
async def mcp_get_metrics(server: str = None, tool: str = None):
    """
    Get performance metrics for MCP tools

    Args:
        server: Filter by specific server (optional)
        tool: Filter by specific tool (optional)

    Returns comprehensive performance metrics including:
    - Call counts (total, successful, failed)
    - Duration statistics (avg, min, max)
    - Cache hit rates
    - Retry and fallback statistics
    """
    try:
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        metrics = client.get_metrics(server=server, tool=tool)

        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/history")
async def mcp_get_execution_history(limit: int = 100):
    """
    Get execution history for MCP tools

    Args:
        limit: Maximum number of recent executions to return (default: 100)

    Returns detailed execution history including:
    - Execution IDs and timestamps
    - Success/failure status
    - Duration and performance
    - Cache hits and retries
    """
    try:
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        history = client.get_execution_history(limit=limit)

        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/chain", dependencies=[Depends(verify_api_key)])
async def mcp_execute_chain(request: Dict[str, Any]):
    """
    Execute a chain of MCP tools in sequence

    Request body:
    {
        "chain_name": "deployment_workflow",
        "steps": [
            {
                "server": "github",
                "tool": "getCommits",
                "params": {"repo": "myrepo", "branch": "main"}
            },
            {
                "server": "render",
                "tool": "render_trigger_deploy",
                "params": {"serviceId": "srv-123"}
            }
        ],
        "fail_fast": true
    }

    Returns:
    - Chain execution results
    - Individual step results
    - Total duration and success status
    """
    try:
        from mcp_integration import get_mcp_client, MCPServer

        client = get_mcp_client()

        chain_name = request.get("chain_name", "unnamed_chain")
        steps_data = request.get("steps", [])
        fail_fast = request.get("fail_fast", True)

        # Convert steps to proper format
        steps = []
        for step in steps_data:
            server_str = step.get("server")
            tool = step.get("tool")
            params = step.get("params", {})

            # Try to convert to MCPServer enum
            try:
                server = MCPServer(server_str)
            except ValueError:
                server = server_str  # Use string if not in enum

            steps.append((server, tool, params))

        result = await client.execute_chain(chain_name, steps, fail_fast)

        # Convert MCPToolResult objects to dicts for JSON serialization
        if "results" in result:
            result["results"] = [
                {
                    "success": r.success,
                    "server": r.server,
                    "tool": r.tool,
                    "duration_ms": r.duration_ms,
                    "cached": r.cached,
                    "retry_count": r.retry_count,
                    "fallback_used": r.fallback_used,
                    "error": r.error
                }
                for r in result["results"]
            ]

        return result

    except Exception as e:
        logger.error(f"Chain execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/chains")
async def mcp_get_chain_history(limit: int = 10):
    """
    Get history of chain executions

    Args:
        limit: Maximum number of chain executions to return (default: 10)

    Returns recent chain execution history with success rates
    """
    try:
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        chains = client.get_chain_history(limit=limit)

        return {
            "chains": chains,
            "count": len(chains),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Chain history retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/cache/clear", dependencies=[Depends(verify_api_key)])
async def mcp_clear_cache():
    """Clear all cached tool results"""
    try:
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        cache_size = len(client._cache)
        client.clear_cache()

        return {
            "cleared": cache_size,
            "message": f"Cleared {cache_size} cache entries",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/cache/toggle", dependencies=[Depends(verify_api_key)])
async def mcp_toggle_cache(enabled: bool):
    """
    Enable or disable caching

    Args:
        enabled: True to enable caching, False to disable
    """
    try:
        from mcp_integration import get_mcp_client

        client = get_mcp_client()
        client.set_cache_enabled(enabled)

        return {
            "cache_enabled": enabled,
            "message": f"Caching {'enabled' if enabled else 'disabled'}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache toggle error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/workflows")
async def mcp_get_workflows():
    """
    Get available workflow templates for AUREA integration

    Returns a list of pre-configured workflow templates that can be
    executed for common operations like deployments, onboarding, etc.
    """
    try:
        from mcp_integration import get_aurea_executor

        executor = get_aurea_executor()
        workflows = executor.get_available_workflows()

        return {
            "workflows": workflows,
            "count": len(workflows),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Workflows retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/workflow/execute", dependencies=[Depends(verify_api_key)])
async def mcp_execute_workflow(request: Dict[str, Any]):
    """
    Execute a pre-defined workflow template

    Request body:
    {
        "workflow_name": "full_deployment",
        "params": {
            "repo": "myrepo",
            "branch": "main",
            "service_id": "srv-123"
        },
        "fail_fast": true
    }
    """
    try:
        from mcp_integration import get_aurea_executor

        executor = get_aurea_executor()

        workflow_name = request.get("workflow_name")
        params = request.get("params", {})
        fail_fast = request.get("fail_fast", True)

        if not workflow_name:
            raise HTTPException(status_code=400, detail="workflow_name is required")

        result = await executor.execute_workflow(workflow_name, params, fail_fast)

        # Convert MCPToolResult objects for JSON serialization
        if "results" in result:
            result["results"] = [
                {
                    "success": r.success,
                    "server": r.server,
                    "tool": r.tool,
                    "duration_ms": r.duration_ms,
                    "cached": r.cached,
                    "retry_count": r.retry_count,
                    "fallback_used": r.fallback_used,
                    "error": r.error
                }
                for r in result["results"]
            ]

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/performance")
async def mcp_get_performance():
    """
    Get comprehensive performance metrics for the MCP system

    Returns:
    - Overall system performance
    - Tool-specific metrics
    - Cache statistics
    - Chain execution stats
    """
    try:
        from mcp_integration import get_aurea_executor

        executor = get_aurea_executor()
        metrics = await executor.get_performance_metrics()

        return {
            "performance": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)