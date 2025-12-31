"""
State Sync API Endpoints
========================
Exposes the Real-Time State Sync system via REST API.

Provides endpoints for:
- System health overview
- Component status
- Change history
- Full system scan
- AI context export

Author: Claude Opus 4.5 + BrainOps AI Team
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
from datetime import datetime, timezone
import json
import os
from pathlib import Path

router = APIRouter(prefix="/api/state-sync", tags=["state-sync"])

# Environment-aware paths (same pattern as realtime_state_sync.py)
_IS_PRODUCTION = os.getenv("RENDER") is not None or not Path("/home/matt-woodworth/dev").exists()

if _IS_PRODUCTION:
    _CHANGE_LOG_PATH = Path("/tmp/AI_CHANGE_LOG.json")
    _STATE_PATH = Path("/tmp/AI_SYSTEM_STATE.json")
else:
    _CHANGE_LOG_PATH = Path("/home/matt-woodworth/dev/AI_CHANGE_LOG.json")
    _STATE_PATH = Path("/home/matt-woodworth/dev/AI_SYSTEM_STATE.json")


# ============== HEALTH & STATUS ==============

@router.get("/health")
async def get_system_health():
    """Get overall system health summary"""
    try:
        from realtime_state_sync import get_state_sync
        sync = get_state_sync()
        return sync.compute_health_summary()
    except Exception as e:
        return {
            "status": "unknown",
            "error": str(e),
            "message": "State sync not initialized"
        }


@router.get("/components")
async def get_all_components(
    component_type: Optional[str] = Query(None, description="Filter by type: codebase, agent, database_table, api_endpoint")
):
    """Get all tracked components"""
    try:
        from realtime_state_sync import get_state_sync
        sync = get_state_sync()
        components = sync.get_all_components(component_type)
        return {
            "count": len(components),
            "components": [
                {
                    "name": c.name,
                    "type": c.component_type,
                    "status": c.status,
                    "path": c.path,
                    "last_updated": c.last_updated,
                    "version": c.version,
                    "metadata": c.metadata
                }
                for c in components
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components/{component_name}")
async def get_component(component_name: str):
    """Get a specific component by name"""
    try:
        from realtime_state_sync import get_state_sync
        sync = get_state_sync()
        component = sync.get_component(component_name)

        if not component:
            raise HTTPException(status_code=404, detail=f"Component not found: {component_name}")

        return {
            "name": component.name,
            "type": component.component_type,
            "status": component.status,
            "path": component.path,
            "last_updated": component.last_updated,
            "version": component.version,
            "dependencies": component.dependencies,
            "metadata": component.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== CHANGE HISTORY ==============

@router.get("/changes")
async def get_change_history(limit: int = 50):
    """Get recent change history"""
    if not _CHANGE_LOG_PATH.exists():
        return {"changes": [], "message": "No change log found"}

    try:
        with open(_CHANGE_LOG_PATH) as f:
            changes = json.load(f)

        # Return most recent first
        return {
            "total": len(changes),
            "showing": min(limit, len(changes)),
            "changes": changes[-limit:][::-1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/changes/by-codebase/{codebase}")
async def get_changes_by_codebase(codebase: str, limit: int = 20):
    """Get changes for a specific codebase"""
    if not _CHANGE_LOG_PATH.exists():
        return {"changes": [], "message": "No change log found"}

    try:
        with open(_CHANGE_LOG_PATH) as f:
            all_changes = json.load(f)

        filtered = [c for c in all_changes if c.get("codebase") == codebase]

        return {
            "codebase": codebase,
            "total": len(filtered),
            "showing": min(limit, len(filtered)),
            "changes": filtered[-limit:][::-1]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== FULL SCAN ==============

@router.post("/scan")
async def trigger_full_scan():
    """Trigger a full system scan"""
    try:
        from realtime_state_sync import get_state_sync
        sync = get_state_sync()
        result = await sync.full_system_scan()
        return {
            "status": "complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/propagate")
async def trigger_change_propagation():
    """Trigger change detection and propagation"""
    try:
        from change_propagation_daemon import ChangePropagator
        propagator = ChangePropagator()
        result = await propagator.run_propagation_cycle()
        return {
            "status": "complete",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "changes_detected": result["changes"],
            "successful": result.get("successful", 0),
            "failed": result.get("failed", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== AI CONTEXT ==============

@router.get("/context")
async def get_ai_context():
    """Get formatted context for AI sessions"""
    try:
        from realtime_state_sync import get_state_sync
        sync = get_state_sync()
        context = sync.export_for_ai_context()
        return {
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/context/raw", response_class=JSONResponse)
async def get_raw_state():
    """Get raw system state JSON"""
    if not _STATE_PATH.exists():
        raise HTTPException(status_code=404, detail="State file not found")

    try:
        with open(_STATE_PATH) as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== GIT STATUS ==============

@router.get("/git-status")
async def get_git_status():
    """Get git status across all codebases"""
    try:
        from change_propagation_daemon import detect_git_changes
        changes = await detect_git_changes()

        # Group by codebase
        by_codebase = {}
        for change in changes:
            codebase = change["codebase"]
            if codebase not in by_codebase:
                by_codebase[codebase] = []
            by_codebase[codebase].append(change)

        return {
            "total_uncommitted": len(changes),
            "by_codebase": {
                codebase: {
                    "count": len(files),
                    "files": files
                }
                for codebase, files in by_codebase.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============== VISUALIZATION ==============

@router.get("/visualize", response_class=HTMLResponse)
async def visualize_system_state():
    """Interactive system state visualization"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>BrainOps AI OS - System State</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'SF Mono', 'Monaco', monospace; background: #0d1117; color: #c9d1d9; }
            .header { background: #161b22; padding: 20px; border-bottom: 1px solid #30363d; }
            .header h1 { color: #58a6ff; font-size: 1.5em; }
            .header .status { margin-top: 10px; }
            .status-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 0.9em; }
            .status-healthy { background: #238636; color: #fff; }
            .status-degraded { background: #9e6a03; color: #fff; }
            .status-error { background: #da3633; color: #fff; }
            .container { display: flex; height: calc(100vh - 120px); }
            .sidebar { width: 300px; background: #161b22; border-right: 1px solid #30363d; overflow-y: auto; }
            .sidebar h3 { padding: 15px; color: #8b949e; font-size: 0.85em; text-transform: uppercase; }
            .component-list { list-style: none; }
            .component-item { padding: 10px 15px; border-bottom: 1px solid #21262d; cursor: pointer; }
            .component-item:hover { background: #21262d; }
            .component-item .name { color: #c9d1d9; }
            .component-item .type { color: #8b949e; font-size: 0.85em; }
            .component-item .status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 8px; }
            .dot-healthy { background: #238636; }
            .dot-degraded { background: #9e6a03; }
            .dot-error { background: #da3633; }
            .dot-unknown { background: #484f58; }
            .main { flex: 1; padding: 20px; overflow-y: auto; }
            #network { width: 100%; height: 100%; background: #0d1117; border-radius: 8px; }
            .stats { display: flex; gap: 20px; margin-bottom: 20px; }
            .stat-card { background: #21262d; padding: 15px 20px; border-radius: 8px; }
            .stat-value { font-size: 2em; color: #58a6ff; }
            .stat-label { color: #8b949e; font-size: 0.9em; }
            .changes-panel { margin-top: 20px; }
            .change-item { background: #21262d; padding: 10px; border-radius: 4px; margin-bottom: 8px; }
            .change-type { padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
            .change-created { background: #238636; color: #fff; }
            .change-modified { background: #9e6a03; color: #fff; }
            .change-deleted { background: #da3633; color: #fff; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 BrainOps AI OS - Real-Time State</h1>
            <div class="status">
                <span id="status-badge" class="status-badge">Loading...</span>
                <span id="component-count" style="margin-left: 20px; color: #8b949e;"></span>
            </div>
        </div>

        <div class="container">
            <div class="sidebar">
                <h3>Components</h3>
                <ul id="component-list" class="component-list"></ul>
            </div>

            <div class="main">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value" id="total-components">-</div>
                        <div class="stat-label">Total Components</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="healthy-count">-</div>
                        <div class="stat-label">Healthy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="changes-today">-</div>
                        <div class="stat-label">Changes Today</div>
                    </div>
                </div>

                <div id="network"></div>

                <div class="changes-panel">
                    <h3 style="color: #8b949e; margin-bottom: 10px;">Recent Changes</h3>
                    <div id="changes-list"></div>
                </div>
            </div>
        </div>

        <script>
        async function loadData() {
            try {
                // Load health
                const healthRes = await fetch('/api/state-sync/health');
                const health = await healthRes.json();

                const badge = document.getElementById('status-badge');
                badge.textContent = health.status.toUpperCase();
                badge.className = 'status-badge status-' + health.status;

                document.getElementById('total-components').textContent = health.total_components || 0;
                document.getElementById('healthy-count').textContent = health.by_status?.healthy || 0;
                document.getElementById('component-count').textContent =
                    `${health.total_components || 0} components tracked`;

                // Load components
                const compRes = await fetch('/api/state-sync/components');
                const compData = await compRes.json();

                renderComponentList(compData.components);
                renderNetwork(compData.components);

                // Load changes
                const changesRes = await fetch('/api/state-sync/changes?limit=10');
                const changesData = await changesRes.json();

                document.getElementById('changes-today').textContent = changesData.total || 0;
                renderChanges(changesData.changes || []);

            } catch (err) {
                console.error('Failed to load data:', err);
            }
        }

        function renderComponentList(components) {
            const list = document.getElementById('component-list');
            list.innerHTML = '';

            // Group by type
            const byType = {};
            components.forEach(c => {
                if (!byType[c.type]) byType[c.type] = [];
                byType[c.type].push(c);
            });

            for (const [type, items] of Object.entries(byType)) {
                items.slice(0, 10).forEach(c => {
                    const li = document.createElement('li');
                    li.className = 'component-item';
                    li.innerHTML = `
                        <span class="status-dot dot-${c.status}"></span>
                        <span class="name">${c.name}</span>
                        <div class="type">${c.type}</div>
                    `;
                    list.appendChild(li);
                });
            }
        }

        function renderNetwork(components) {
            const container = document.getElementById('network');

            const colors = {
                'codebase': '#58a6ff',
                'agent': '#a371f7',
                'database_table': '#3fb950',
                'api_endpoint': '#f0883e'
            };

            const nodes = new vis.DataSet(
                components.slice(0, 100).map((c, i) => ({
                    id: i,
                    label: c.name.length > 20 ? c.name.substring(0, 20) + '...' : c.name,
                    color: colors[c.type] || '#8b949e',
                    title: `${c.type}: ${c.name}\\nStatus: ${c.status}`
                }))
            );

            const edges = new vis.DataSet([]);

            const options = {
                nodes: { shape: 'dot', size: 16, font: { color: '#c9d1d9', size: 12 } },
                physics: {
                    forceAtlas2Based: {
                        gravitationalConstant: -50,
                        centralGravity: 0.01,
                        springLength: 100
                    },
                    solver: 'forceAtlas2Based'
                }
            };

            new vis.Network(container, { nodes, edges }, options);
        }

        function renderChanges(changes) {
            const list = document.getElementById('changes-list');
            list.innerHTML = '';

            changes.forEach(c => {
                const div = document.createElement('div');
                div.className = 'change-item';
                const typeClass = 'change-' + (c.change_type || 'modified');
                div.innerHTML = `
                    <span class="${typeClass} change-type">${c.change_type || 'change'}</span>
                    <span style="margin-left: 10px;">${c.path || c.component || 'Unknown'}</span>
                `;
                list.appendChild(div);
            });
        }

        // Initial load
        loadData();

        // Refresh every 30 seconds
        setInterval(loadData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
