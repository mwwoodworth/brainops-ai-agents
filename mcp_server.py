#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for BrainOps AI Agents
Provides complete transparency and control for AI systems
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import subprocess
import httpx
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BrainOps MCP Server",
    description="Model Context Protocol server for complete AI system transparency",
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
    """MCP Server for AI system control and monitoring"""

    def __init__(self):
        self.connections = []
        self.system_state = {}
        self.active_tools = []
        self.execution_history = []

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
                proc = await asyncio.create_subprocess_shell(
                    f"tail -n {lines} {log_file}",
                    stdout=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                return stdout.decode().split('\n')
            except:
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

@app.post("/execute")
async def execute_tool(request: Dict[str, Any]):
    """Execute an MCP tool"""
    tool = request.get('tool')
    params = request.get('params', {})

    if not tool:
        raise HTTPException(status_code=400, detail="Tool name required")

    result = await mcp_server.execute_tool(tool, params)
    return result

@app.get("/files")
async def get_files(path: str = "/home/matt-woodworth", depth: int = 2):
    """Get file tree structure"""
    return await mcp_server.get_file_tree(path, depth)

@app.get("/logs/{service}")
async def get_logs(service: str, lines: int = 100):
    """Get service logs"""
    logs = await mcp_server.monitor_logs(service, lines)
    return {"service": service, "logs": logs}

@app.get("/history")
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

@app.post("/mcp/execute")
async def mcp_execute(request: Dict[str, Any]):
    """Execute MCP tool via Vercel adapter"""
    return await execute_tool(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MCP_PORT", 8100))
    uvicorn.run(app, host="0.0.0.0", port=port)