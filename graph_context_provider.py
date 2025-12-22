#!/usr/bin/env python3
"""
Graph-Enhanced Context Provider
Provides intelligent codebase context to AI agents using the codebase graph.

Phase 2 Enhancement: Agents can now query relevant files, functions, and
relationships from the codebase graph to inform their responses.
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

from config import config
from database.async_connection import get_pool, init_pool, PoolConfig

logger = logging.getLogger(__name__)


@dataclass
class CodeContext:
    """Structured context from codebase graph"""
    files: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    relevance_score: float = 0.0
    query_time_ms: float = 0.0

    def to_prompt_context(self) -> str:
        """Format context for LLM prompt injection"""
        parts = []

        if self.files:
            parts.append("## Relevant Files")
            for f in self.files[:10]:
                parts.append(f"- {f['repo_name']}/{f['file_path']} (line {f.get('line_number', 0)})")

        if self.functions:
            parts.append("\n## Key Functions")
            for fn in self.functions[:15]:
                parts.append(f"- `{fn['name']}` in {fn['file_path']}:{fn.get('line_number', 0)}")

        if self.classes:
            parts.append("\n## Relevant Classes")
            for cls in self.classes[:10]:
                parts.append(f"- `{cls['name']}` in {cls['file_path']}:{cls.get('line_number', 0)}")

        if self.endpoints:
            parts.append("\n## API Endpoints")
            for ep in self.endpoints[:10]:
                method = ep.get('metadata', {}).get('http_method', 'GET').upper()
                path = ep.get('metadata', {}).get('path', ep['name'])
                parts.append(f"- [{method}] `{path}` in {ep['file_path']}")

        if self.relationships:
            parts.append("\n## Code Relationships")
            for rel in self.relationships[:10]:
                parts.append(f"- {rel['source_name']} --{rel['type']}--> {rel['target_name']}")

        if self.summary:
            parts.append(f"\n## Summary\n{self.summary}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphContextProvider:
    """
    Provides intelligent codebase context using the graph database.

    Usage:
        provider = GraphContextProvider()
        context = await provider.get_context_for_task(task_description)
        enhanced_prompt = f"{original_prompt}\n\n# Codebase Context\n{context.to_prompt_context()}"
    """

    def __init__(self, max_results: int = 50):
        self.max_results = max_results
        self._pool_initialized = False

    async def _ensure_pool(self):
        """Ensure database pool is initialized"""
        if self._pool_initialized:
            return

        try:
            pool = get_pool()
            if pool:
                self._pool_initialized = True
                return
        except Exception:
            pass

        db_config = config.database
        pool_config = PoolConfig(
            host=db_config.host,
            port=db_config.port,
            user=db_config.user,
            password=db_config.password,
            database=db_config.database,
            ssl=db_config.ssl,
            ssl_verify=db_config.ssl_verify
        )
        await init_pool(pool_config)
        self._pool_initialized = True

    async def get_context_for_task(
        self,
        task_description: str,
        repos: Optional[List[str]] = None,
        include_relationships: bool = True,
        focus_types: Optional[List[str]] = None
    ) -> CodeContext:
        """
        Get relevant codebase context for a task description.

        Args:
            task_description: Natural language description of the task
            repos: Optional list of repos to search (None = all)
            include_relationships: Whether to include code relationships
            focus_types: Node types to focus on ('function', 'class', 'endpoint', 'file')

        Returns:
            CodeContext with relevant code elements
        """
        start_time = datetime.now()
        await self._ensure_pool()

        # Extract keywords from task
        keywords = self._extract_keywords(task_description)

        context = CodeContext()

        try:
            pool = get_pool()

            # Search for matching nodes
            nodes = await self._search_nodes(pool, keywords, repos, focus_types)

            # Categorize nodes
            for node in nodes:
                node_dict = dict(node)
                node_type = node_dict.get('type', '')

                if node_type == 'file':
                    context.files.append(node_dict)
                elif node_type == 'function':
                    context.functions.append(node_dict)
                elif node_type == 'class':
                    context.classes.append(node_dict)
                elif node_type == 'endpoint':
                    context.endpoints.append(node_dict)

            # Get relationships if requested
            if include_relationships and nodes:
                node_ids = [n['id'] for n in nodes[:20]]  # Limit for performance
                context.relationships = await self._get_relationships(pool, node_ids)

            # Generate summary
            context.summary = self._generate_summary(context, keywords)
            context.relevance_score = self._calculate_relevance(context, keywords)

        except Exception as e:
            logger.error(f"Graph context query failed: {e}")
            context.summary = f"Context retrieval failed: {str(e)}"

        context.query_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return context

    async def get_context_for_agent(
        self,
        agent_name: str,
        task_data: Optional[Dict[str, Any]] = None
    ) -> CodeContext:
        """
        Get relevant codebase context for an AI agent's current task.

        Args:
            agent_name: Name of the agent requesting context
            task_data: Optional task data containing action, data, etc.

        Returns:
            CodeContext with relevant code elements for the agent's task
        """
        # Build task description from agent name and task data
        task_parts = [agent_name]

        if task_data:
            if "action" in task_data:
                task_parts.append(task_data["action"])
            if "description" in task_data:
                task_parts.append(task_data["description"])
            if "data" in task_data and isinstance(task_data["data"], dict):
                # Extract key fields from data
                for key in ["type", "target", "query", "keyword"]:
                    if key in task_data["data"]:
                        task_parts.append(str(task_data["data"][key]))

        task_description = " ".join(task_parts)

        # Determine repos to search based on agent type
        repos = None
        agent_lower = agent_name.lower()
        if "erp" in agent_lower or "weathercraft" in agent_lower:
            repos = ["weathercraft-erp"]
        elif "roof" in agent_lower or "mrg" in agent_lower:
            repos = ["myroofgenius-app"]
        elif "backend" in agent_lower or "api" in agent_lower:
            repos = ["brainops-ai-agents", "brainops-backend"]

        return await self.get_context_for_task(task_description, repos=repos)

    async def get_context_for_file(
        self,
        file_path: str,
        repo_name: Optional[str] = None
    ) -> CodeContext:
        """Get context for a specific file including its contents and relationships"""
        await self._ensure_pool()
        pool = get_pool()

        context = CodeContext()

        try:
            # Find the file node
            query = """
                SELECT * FROM codebase_nodes
                WHERE file_path ILIKE $1
            """
            params = [f"%{file_path}%"]

            if repo_name:
                query += " AND repo_name = $2"
                params.append(repo_name)

            query += " LIMIT 50"

            nodes = await pool.fetch(query, *params)

            for node in nodes:
                node_dict = dict(node)
                node_type = node_dict.get('type', '')

                if node_type == 'file':
                    context.files.append(node_dict)
                elif node_type == 'function':
                    context.functions.append(node_dict)
                elif node_type == 'class':
                    context.classes.append(node_dict)
                elif node_type == 'endpoint':
                    context.endpoints.append(node_dict)

            # Get relationships
            if nodes:
                node_ids = [n['id'] for n in nodes]
                context.relationships = await self._get_relationships(pool, node_ids)

            context.summary = f"File context for {file_path}: {len(context.functions)} functions, {len(context.classes)} classes"

        except Exception as e:
            logger.error(f"File context query failed: {e}")

        return context

    async def search_by_name(
        self,
        name: str,
        node_type: Optional[str] = None,
        repos: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for code elements by name"""
        await self._ensure_pool()
        pool = get_pool()

        query = """
            SELECT * FROM codebase_nodes
            WHERE name ILIKE $1
        """
        params = [f"%{name}%"]
        param_idx = 2

        if node_type:
            query += f" AND type = ${param_idx}"
            params.append(node_type)
            param_idx += 1

        if repos:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(repos)))
            query += f" AND repo_name IN ({placeholders})"
            params.extend(repos)

        query += f" ORDER BY name LIMIT {self.max_results}"

        try:
            results = await pool.fetch(query, *params)
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Name search failed: {e}")
            return []

    async def get_api_endpoints(
        self,
        repos: Optional[List[str]] = None,
        method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all API endpoints from the codebase"""
        await self._ensure_pool()
        pool = get_pool()

        query = "SELECT * FROM codebase_nodes WHERE type = 'endpoint'"
        params = []
        param_idx = 1

        if repos:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(repos)))
            query += f" AND repo_name IN ({placeholders})"
            params.extend(repos)
            param_idx += len(repos)

        if method:
            query += f" AND metadata->>'http_method' ILIKE ${param_idx}"
            params.append(method)

        query += f" LIMIT {self.max_results}"

        try:
            results = await pool.fetch(query, *params)
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Endpoint query failed: {e}")
            return []

    async def find_callers(self, function_name: str, repo_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find all callers of a function"""
        await self._ensure_pool()
        pool = get_pool()

        query = """
            SELECT DISTINCT source.*
            FROM codebase_edges e
            JOIN codebase_nodes target ON e.target_id = target.id
            JOIN codebase_nodes source ON e.source_id = source.id
            WHERE target.name = $1 AND e.type = 'calls'
        """
        params = [function_name]

        if repo_name:
            query += " AND target.repo_name = $2"
            params.append(repo_name)

        query += f" LIMIT {self.max_results}"

        try:
            results = await pool.fetch(query, *params)
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Caller search failed: {e}")
            return []

    async def get_class_hierarchy(self, class_name: str, repo_name: Optional[str] = None) -> Dict[str, Any]:
        """Get inheritance hierarchy for a class"""
        await self._ensure_pool()
        pool = get_pool()

        hierarchy = {
            "class": class_name,
            "parents": [],
            "children": []
        }

        try:
            # Find parent classes
            parent_query = """
                SELECT target.*
                FROM codebase_edges e
                JOIN codebase_nodes source ON e.source_id = source.id
                JOIN codebase_nodes target ON e.target_id = target.id
                WHERE source.name = $1 AND e.type = 'inherits'
            """
            params = [class_name]
            if repo_name:
                parent_query += " AND source.repo_name = $2"
                params.append(repo_name)

            parents = await pool.fetch(parent_query, *params)
            hierarchy["parents"] = [dict(p) for p in parents]

            # Find child classes
            child_query = """
                SELECT source.*
                FROM codebase_edges e
                JOIN codebase_nodes source ON e.source_id = source.id
                JOIN codebase_nodes target ON e.target_id = target.id
                WHERE target.name = $1 AND e.type = 'inherits'
            """
            children = await pool.fetch(child_query, *params)
            hierarchy["children"] = [dict(c) for c in children]

        except Exception as e:
            logger.error(f"Hierarchy query failed: {e}")

        return hierarchy

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the codebase graph"""
        await self._ensure_pool()
        pool = get_pool()

        try:
            stats = {}

            # Node counts by type
            type_counts = await pool.fetch("""
                SELECT type, COUNT(*) as count
                FROM codebase_nodes
                GROUP BY type
            """)
            stats["node_counts"] = {r["type"]: r["count"] for r in type_counts}

            # Node counts by repo
            repo_counts = await pool.fetch("""
                SELECT repo_name, COUNT(*) as count
                FROM codebase_nodes
                GROUP BY repo_name
            """)
            stats["repo_counts"] = {r["repo_name"]: r["count"] for r in repo_counts}

            # Edge counts
            edge_counts = await pool.fetch("""
                SELECT type, COUNT(*) as count
                FROM codebase_edges
                GROUP BY type
            """)
            stats["edge_counts"] = {r["type"]: r["count"] for r in edge_counts}

            # Total counts
            total_nodes = await pool.fetchval("SELECT COUNT(*) FROM codebase_nodes")
            total_edges = await pool.fetchval("SELECT COUNT(*) FROM codebase_edges")
            stats["total_nodes"] = total_nodes
            stats["total_edges"] = total_edges

            return stats

        except Exception as e:
            logger.error(f"Stats query failed: {e}")
            return {"error": str(e)}

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract search keywords from natural language text"""
        # Common words to filter out
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'if', 'or', 'because', 'until', 'while', 'this',
            'that', 'these', 'those', 'what', 'which', 'who', 'whom',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
            'she', 'her', 'it', 'its', 'they', 'them', 'their'
        }

        # Technical terms to preserve
        tech_terms = {
            'api', 'endpoint', 'function', 'class', 'method', 'route',
            'database', 'query', 'model', 'schema', 'table', 'column',
            'agent', 'workflow', 'task', 'job', 'customer', 'invoice',
            'proposal', 'contract', 'deploy', 'build', 'test'
        }

        # Extract words
        import re
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())

        keywords = []
        for word in words:
            if word in tech_terms:
                keywords.append(word)
            elif word not in stopwords and len(word) > 2:
                keywords.append(word)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:20]  # Limit keywords

    async def _search_nodes(
        self,
        pool,
        keywords: List[str],
        repos: Optional[List[str]],
        focus_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Search for nodes matching keywords"""
        if not keywords:
            return []

        # Build search query with pattern matching
        conditions = []
        params = []
        param_idx = 1

        # Name matching
        name_patterns = []
        for kw in keywords[:5]:  # Limit to top 5 keywords
            name_patterns.append(f"name ILIKE ${param_idx}")
            params.append(f"%{kw}%")
            param_idx += 1

        if name_patterns:
            conditions.append(f"({' OR '.join(name_patterns)})")

        # Repo filter
        if repos:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(repos)))
            conditions.append(f"repo_name IN ({placeholders})")
            params.extend(repos)
            param_idx += len(repos)

        # Type filter
        if focus_types:
            placeholders = ", ".join(f"${i}" for i in range(param_idx, param_idx + len(focus_types)))
            conditions.append(f"type IN ({placeholders})")
            params.extend(focus_types)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
            SELECT * FROM codebase_nodes
            WHERE {where_clause}
            ORDER BY
                CASE WHEN type = 'endpoint' THEN 1
                     WHEN type = 'function' THEN 2
                     WHEN type = 'class' THEN 3
                     ELSE 4 END,
                name
            LIMIT {self.max_results}
        """

        try:
            results = await pool.fetch(query, *params)
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Node search failed: {e}")
            return []

    async def _get_relationships(self, pool, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Get relationships for given node IDs"""
        if not node_ids:
            return []

        # Convert UUIDs to strings for parameterized query
        id_placeholders = ", ".join(f"${i+1}" for i in range(len(node_ids)))

        query = f"""
            SELECT
                e.type,
                source.name as source_name,
                source.type as source_type,
                target.name as target_name,
                target.type as target_type
            FROM codebase_edges e
            JOIN codebase_nodes source ON e.source_id = source.id
            JOIN codebase_nodes target ON e.target_id = target.id
            WHERE e.source_id IN ({id_placeholders})
               OR e.target_id IN ({id_placeholders})
            LIMIT 50
        """

        # Duplicate params for both IN clauses
        params = node_ids + node_ids

        try:
            results = await pool.fetch(query, *params)
            return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Relationship query failed: {e}")
            return []

    def _generate_summary(self, context: CodeContext, keywords: List[str]) -> str:
        """Generate a summary of the retrieved context"""
        parts = []

        total_items = len(context.files) + len(context.functions) + len(context.classes) + len(context.endpoints)

        if total_items == 0:
            return f"No code elements found matching keywords: {', '.join(keywords[:5])}"

        parts.append(f"Found {total_items} relevant code elements")

        if context.endpoints:
            parts.append(f"({len(context.endpoints)} API endpoints)")
        if context.functions:
            parts.append(f"({len(context.functions)} functions)")
        if context.classes:
            parts.append(f"({len(context.classes)} classes)")
        if context.files:
            parts.append(f"({len(context.files)} files)")

        repos = set()
        for item in context.files + context.functions + context.classes + context.endpoints:
            repos.add(item.get('repo_name', 'unknown'))

        if repos:
            parts.append(f"across repos: {', '.join(repos)}")

        return " ".join(parts)

    def _calculate_relevance(self, context: CodeContext, keywords: List[str]) -> float:
        """Calculate relevance score for the context"""
        if not keywords:
            return 0.0

        total_score = 0.0
        items_checked = 0

        all_items = context.files + context.functions + context.classes + context.endpoints

        for item in all_items:
            name = item.get('name', '').lower()
            file_path = item.get('file_path', '').lower()

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in name:
                    total_score += 1.0
                elif kw_lower in file_path:
                    total_score += 0.5

            items_checked += 1

        if items_checked == 0:
            return 0.0

        # Normalize to 0-1 range
        max_possible = len(keywords) * items_checked
        return min(1.0, total_score / max_possible) if max_possible > 0 else 0.0


# Convenience functions for agent integration
_provider: Optional[GraphContextProvider] = None


def get_graph_context_provider() -> GraphContextProvider:
    """Get or create singleton provider instance"""
    global _provider
    if _provider is None:
        _provider = GraphContextProvider()
    return _provider


async def get_context_for_task(task_description: str, **kwargs) -> CodeContext:
    """Quick function to get context for a task"""
    provider = get_graph_context_provider()
    return await provider.get_context_for_task(task_description, **kwargs)


async def enrich_prompt_with_context(
    prompt: str,
    task_description: Optional[str] = None,
    repos: Optional[List[str]] = None
) -> str:
    """Enrich a prompt with relevant codebase context"""
    provider = get_graph_context_provider()

    # Use task description or extract from prompt
    search_text = task_description or prompt[:500]

    context = await provider.get_context_for_task(search_text, repos=repos)

    if context.relevance_score > 0.1:
        return f"""{prompt}

# Codebase Context (auto-retrieved)
{context.to_prompt_context()}
"""

    return prompt


if __name__ == "__main__":
    async def test():
        provider = GraphContextProvider()

        print("Testing Graph Context Provider...")

        # Test 1: Get stats
        stats = await provider.get_graph_stats()
        print(f"\nGraph Stats: {json.dumps(stats, indent=2)}")

        # Test 2: Search for task
        context = await provider.get_context_for_task(
            "Find the customer invoice generation workflow"
        )
        print(f"\nTask Context:\n{context.to_prompt_context()}")
        print(f"Relevance: {context.relevance_score:.2f}")
        print(f"Query Time: {context.query_time_ms:.2f}ms")

        # Test 3: Get API endpoints
        endpoints = await provider.get_api_endpoints(repos=["brainops-ai-agents"])
        print(f"\nAPI Endpoints: {len(endpoints)} found")
        for ep in endpoints[:5]:
            print(f"  - {ep['name']} in {ep['file_path']}")

    asyncio.run(test())
