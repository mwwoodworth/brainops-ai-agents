from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from database.async_connection import get_pool
from database.schema_cache import (
    get_cached_constraint_count,
    get_cached_foreign_keys,
    get_schema_cache,
)

router = APIRouter(prefix="/api/codebase-graph", tags=["codebase-graph"])


# ============== DATABASE SCHEMA ENDPOINTS ==============

@router.get("/database/stats")
async def get_database_stats():
    """Get database schema statistics (with caching for slow queries)"""
    pool = get_pool()

    # Count tables - fast pg_catalog query, no caching needed
    tables = await pool.fetchval("""
        SELECT COUNT(*) FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND c.relkind = 'r'
    """)

    # Count foreign keys - use cached query (was 72+ seconds)
    fks = await get_cached_constraint_count(pool, "FOREIGN KEY")

    # Count columns - use pg_catalog for speed
    columns = await pool.fetchval("""
        SELECT COUNT(*) FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND c.relkind = 'r'
        AND a.attnum > 0
        AND NOT a.attisdropped
    """)

    # Get largest tables - already using pg_catalog, fast
    largest = await pool.fetch("""
        SELECT c.relname as tablename, c.reltuples::bigint as row_count
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND c.relkind = 'r'
        ORDER BY c.reltuples DESC
        LIMIT 10
    """)

    return {
        "tables": tables,
        "columns": columns,
        "foreign_keys": fks,
        "largest_tables": [{"name": r["tablename"], "rows": r["row_count"]} for r in largest]
    }


@router.get("/database/tables")
async def get_database_tables(schema: str = "public", limit: int = 100):
    """Get all tables with their columns"""
    pool = get_pool()

    tables = await pool.fetch("""
        SELECT
            t.table_name,
            t.table_schema,
            array_agg(json_build_object(
                'name', c.column_name,
                'type', c.data_type,
                'nullable', c.is_nullable = 'YES'
            ) ORDER BY c.ordinal_position) as columns
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_name = c.table_name AND t.table_schema = c.table_schema
        WHERE t.table_schema = $1
        AND t.table_type = 'BASE TABLE'
        GROUP BY t.table_name, t.table_schema
        ORDER BY t.table_name
        LIMIT $2
    """, schema, limit)

    return [{"name": r["table_name"], "schema": r["table_schema"], "columns": r["columns"]} for r in tables]


@router.get("/database/relationships")
async def get_database_relationships():
    """Get all foreign key relationships for ERD visualization (cached)"""
    pool = get_pool()

    # Use cached foreign key query - was taking 72+ seconds
    fks = await get_cached_foreign_keys(pool, schema="public")

    return [{
        "source": r["source_table"],
        "source_column": r["source_column"],
        "target": r["target_table"],
        "target_column": r["target_column"],
        "constraint": r["constraint_name"]
    } for r in fks]


@router.get("/database/visualize", response_class=HTMLResponse)
async def visualize_database_erd():
    """Interactive ERD visualization using vis.js"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Database ERD Visualization</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            #mynetwork {
                width: 100%;
                height: 85vh;
                border: 1px solid lightgray;
            }
            body { font-family: sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: #eee; }
            .controls { margin-bottom: 10px; }
            h2 { color: #7CD8FF; }
            #stats { color: #7BE3C8; margin-bottom: 10px; }
            select, input { padding: 5px; margin-right: 10px; background: #2a2a4e; color: #fff; border: 1px solid #444; }
        </style>
    </head>
    <body>
        <div class="controls">
            <h2>BrainOps Database ERD</h2>
            <div>
                <input type="text" id="search" placeholder="Filter tables..." onkeyup="filterGraph()">
                <select id="schema" onchange="loadERD()">
                    <option value="public">public</option>
                    <option value="all">All Schemas</option>
                </select>
            </div>
            <div id="stats">Loading...</div>
        </div>
        <div id="mynetwork"></div>

        <script type="text/javascript">
        let allNodes = [];
        let allEdges = [];
        let network = null;

        async function loadERD() {
            try {
                const [tablesRes, relsRes, statsRes] = await Promise.all([
                    fetch('/api/codebase-graph/database/tables?limit=200'),
                    fetch('/api/codebase-graph/database/relationships'),
                    fetch('/api/codebase-graph/database/stats')
                ]);

                const tables = await tablesRes.json();
                const relationships = await relsRes.json();
                const stats = await statsRes.json();

                document.getElementById('stats').innerHTML =
                    `Tables: ${stats.tables} | Columns: ${stats.columns} | Foreign Keys: ${stats.foreign_keys}`;

                // Create nodes for each table
                allNodes = tables.map(t => ({
                    id: t.name,
                    label: t.name + '\\n(' + t.columns.length + ' cols)',
                    shape: 'box',
                    color: {
                        background: '#2a2a4e',
                        border: '#7CD8FF',
                        highlight: { background: '#3a3a6e', border: '#7BE3C8' }
                    },
                    font: { color: '#fff', size: 12 },
                    title: t.columns.map(c => c.name + ': ' + c.type).join('\\n')
                }));

                // Create edges for relationships
                allEdges = relationships.map(r => ({
                    from: r.source,
                    to: r.target,
                    label: r.source_column,
                    arrows: 'to',
                    color: { color: '#7BE3C8', opacity: 0.7 },
                    font: { color: '#888', size: 10 }
                }));

                renderGraph(allNodes, allEdges);

            } catch (err) {
                console.error(err);
                document.getElementById('stats').innerText = "Error loading ERD data.";
            }
        }

        function filterGraph() {
            const filter = document.getElementById('search').value.toLowerCase();
            if (!filter) {
                renderGraph(allNodes, allEdges);
                return;
            }

            const filteredNodes = allNodes.filter(n => n.id.toLowerCase().includes(filter));
            const nodeIds = new Set(filteredNodes.map(n => n.id));
            const filteredEdges = allEdges.filter(e => nodeIds.has(e.from) || nodeIds.has(e.to));

            renderGraph(filteredNodes, filteredEdges);
        }

        function renderGraph(nodes, edges) {
            const container = document.getElementById('mynetwork');
            const networkData = {
                nodes: new vis.DataSet(nodes),
                edges: new vis.DataSet(edges)
            };
            const options = {
                layout: {
                    improvedLayout: true,
                    hierarchical: false
                },
                physics: {
                    forceAtlas2Based: {
                        gravitationalConstant: -50,
                        centralGravity: 0.01,
                        springLength: 150,
                        springConstant: 0.08
                    },
                    maxVelocity: 50,
                    solver: 'forceAtlas2Based',
                    stabilization: { iterations: 100 }
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 100
                }
            };

            if (network) {
                network.setData(networkData);
            } else {
                network = new vis.Network(container, networkData, options);
            }
        }

        window.onload = loadERD;
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@router.get("/stats")
async def get_graph_stats():
    pool = get_pool()
    nodes = await pool.fetchval("SELECT COUNT(*) FROM codebase_nodes")
    edges = await pool.fetchval("SELECT COUNT(*) FROM codebase_edges")
    return {"nodes": nodes, "edges": edges}


@router.get("/cache/stats")
async def get_cache_stats():
    """Get schema cache statistics"""
    cache = get_schema_cache()
    return await cache.get_stats()


@router.post("/cache/clear")
async def clear_cache():
    """Clear the schema cache (use after DDL changes)"""
    cache = get_schema_cache()
    await cache.invalidate_all()
    return {"status": "cleared", "message": "Schema cache has been cleared"}

@router.get("/data")
async def get_graph_data(limit: int = 2000):
    """
    Get graph data (nodes and edges) for visualization.
    Limited to prevent browser crash on massive graphs.
    """
    pool = get_pool()

    # Fetch nodes - using correct column names per actual schema
    nodes_rows = await pool.fetch(
        "SELECT node_id, name, node_type, codebase, filepath, metadata FROM codebase_nodes LIMIT $1",
        limit
    )

    node_ids = [row['node_id'] for row in nodes_rows]
    if not node_ids:
        return {"nodes": [], "edges": []}

    # Fetch edges connecting these nodes - using correct column names
    edges_rows = await pool.fetch(
        """
        SELECT source_node_id, target_node_id, edge_type
        FROM codebase_edges
        WHERE source_node_id = ANY($1) AND target_node_id = ANY($1)
        """,
        node_ids
    )

    nodes = []
    for row in nodes_rows:
        nodes.append({
            "id": row['node_id'],
            "label": row['name'],
            "group": row['node_type'],  # file, class, function
            "repo": row['codebase'],
            "path": row['filepath']
        })

    edges = []
    for row in edges_rows:
        edges.append({
            "from": row['source_node_id'],
            "to": row['target_node_id'],
            "type": row['edge_type']
        })

    return {"nodes": nodes, "edges": edges}

@router.get("/visualize", response_class=HTMLResponse)
async def visualize_graph():
    """
    Simple D3-like visualization using vis.js (Network) for easy graph rendering.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Codebase Graph Visualization</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            #mynetwork {
                width: 100%;
                height: 90vh;
                border: 1px solid lightgray;
            }
            body { font-family: sans-serif; margin: 0; padding: 20px; }
            .controls { margin-bottom: 10px; }
        </style>
    </head>
    <body>
        <div class="controls">
            <h2>BrainOps Codebase Graph</h2>
            <div id="stats">Loading...</div>
        </div>
        <div id="mynetwork"></div>

        <script type="text/javascript">
        async function loadGraph() {
            try {
                const response = await fetch('/api/codebase-graph/data?limit=1000');
                const data = await response.json();

                document.getElementById('stats').innerText =
                    `Nodes: ${data.nodes.length}, Edges: ${data.edges.length}`;

                // Color mapping
                const colors = {
                    'file': '#97C2FC',
                    'class': '#FFFF00',
                    'function': '#FB7E81',
                    'endpoint': '#7BE141'
                };

                const nodes = new vis.DataSet(
                    data.nodes.map(n => ({
                        id: n.id,
                        label: n.label,
                        group: n.group,
                        color: colors[n.group] || '#DDDDDD',
                        title: `${n.group}: ${n.path}`
                    }))
                );

                const edges = new vis.DataSet(
                    data.edges.map(e => ({
                        from: e.from,
                        to: e.to,
                        label: e.type,
                        arrows: 'to',
                        color: { color: '#848484', opacity: 0.5 }
                    }))
                );

                const container = document.getElementById('mynetwork');
                const networkData = { nodes: nodes, edges: edges };
                const options = {
                    nodes: {
                        shape: 'dot',
                        size: 16
                    },
                    physics: {
                        forceAtlas2Based: {
                            gravitationalConstant: -26,
                            centralGravity: 0.005,
                            springLength: 230,
                            springConstant: 0.18
                        },
                        maxVelocity: 146,
                        solver: 'forceAtlas2Based',
                        timestep: 0.35,
                        stabilization: { iterations: 150 }
                    }
                };
                const network = new vis.Network(container, networkData, options);

            } catch (err) {
                console.error(err);
                document.getElementById('stats').innerText = "Error loading graph data.";
            }
        }

        window.onload = loadGraph;
    </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
