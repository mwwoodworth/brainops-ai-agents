-- BrainOps DB verification queries (run with psql)
-- Example:
--   psql "$DATABASE_URL" -X -f scripts/db_verify.sql

SELECT now() AS connected_at;

-- Requested counts
SELECT COUNT(*) AS ai_agents_count FROM ai_agents;
SELECT COUNT(*) AS ai_autonomous_tasks_count FROM ai_autonomous_tasks;
SELECT COUNT(*) AS unified_ai_memory_count FROM unified_ai_memory;
SELECT COUNT(*) AS ai_knowledge_graph_count FROM ai_knowledge_graph;

-- Helpful breakdowns
SELECT memory_type, COUNT(*) AS count
FROM unified_ai_memory
GROUP BY memory_type
ORDER BY COUNT(*) DESC;

SELECT node_type, COUNT(*) AS count
FROM ai_knowledge_graph
GROUP BY node_type
ORDER BY COUNT(*) DESC;

-- Optional (may not exist in all environments)
SELECT COUNT(*) AS ai_knowledge_nodes_count FROM ai_knowledge_nodes;
SELECT COUNT(*) AS ai_knowledge_edges_count FROM ai_knowledge_edges;

