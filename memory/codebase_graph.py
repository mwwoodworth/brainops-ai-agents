"""
Codebase Graph (GraphRAG)
=========================
Builds a lightweight dependency graph across the ecosystem.
Node Types: Table, API_Route, Frontend_Page, Agent
Edge Types: reads_from, writes_to, calls, imports
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)

DEFAULT_ROOTS = [
    os.getenv("BRAINOPS_CODEBASE_ROOT", "/home/matt-woodworth/dev/brainops-ai-agents"),
    "/home/matt-woodworth/dev/myroofgenius-backend",
    "/home/matt-woodworth/dev/weathercraft-erp",
    "/home/matt-woodworth/dev/brainops-command-center",
]

FILE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}

ROUTE_RE = re.compile(r'@router\.(get|post|put|delete|patch)\(\s*[\'"]([^\'"]+)', re.IGNORECASE)
SQL_READ_RE = re.compile(r"\bFROM\s+([a-zA-Z0-9_\.]+)", re.IGNORECASE)
SQL_WRITE_RE = re.compile(
    r"\bINSERT\s+INTO\s+([a-zA-Z0-9_\.]+) | \bUPDATE\s+([a-zA-Z0-9_\.]+) | \bDELETE\s+FROM\s+([a-zA-Z0-9_\.]+)",
    re.IGNORECASE,
)
AGENT_RE = re.compile(r"class\s+([A-Za-z0-9_]+)\s*\(.*BaseAgent.*\)")
FETCH_RE = re.compile(r'fetch\(\s*[\'"](/api/[^\'"]+)')


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str


class CodebaseGraph:
    def __init__(self, roots: Optional[Iterable[str]] = None, max_files: int = 6000) -> None:
        self.roots = [Path(root) for root in (roots or DEFAULT_ROOTS)]
        self.graph = nx.MultiDiGraph()
        self.max_files = max_files

    def _add_node(self, node_type: str, label: str, metadata: Optional[Dict] = None) -> str:
        node_id = f"{node_type}:{label}"
        self.graph.add_node(node_id, type=node_type, label=label, metadata=metadata or {})
        return node_id

    def _add_edge(self, source: str, target: str, relation: str) -> None:
        self.graph.add_edge(source, target, relation=relation)

    def _iter_files(self) -> Iterable[Path]:
        count = 0
        for root in self.roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if count >= self.max_files:
                    return
                if not path.is_file():
                    continue
                if path.suffix not in FILE_EXTENSIONS:
                    continue
                if any(part in {"node_modules", ".git", ".next", "dist", "build", "__pycache__"} for part in path.parts):
                    continue
                yield path
                count += 1

    def _extract_tables(self, text: str) -> Tuple[List[str], List[str]]:
        reads = [match.group(1) for match in SQL_READ_RE.finditer(text)]
        writes = []
        for match in SQL_WRITE_RE.finditer(text):
            for idx in range(1, 4):
                if match.group(idx):
                    writes.append(match.group(idx))
                    break
        return reads, writes

    def _extract_routes(self, text: str) -> List[str]:
        return [f"{method.upper()} {path}" for method, path in ROUTE_RE.findall(text)]

    def _extract_frontend_calls(self, text: str) -> List[str]:
        return [path for path in FETCH_RE.findall(text)]

    def _extract_agents(self, text: str) -> List[str]:
        return [match.group(1) for match in AGENT_RE.finditer(text)]

    def build(self) -> nx.MultiDiGraph:
        self.graph.clear()
        for path in self._iter_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.debug("Failed reading %s: %s", path, exc)
                continue

            routes = self._extract_routes(text)
            read_tables, write_tables = self._extract_tables(text)
            agents = self._extract_agents(text)
            api_calls = self._extract_frontend_calls(text)

            route_nodes = []
            for route in routes:
                route_nodes.append(self._add_node("API_Route", route, {"path": str(path)}))

            for table in read_tables:
                table_node = self._add_node("Table", table)
                for route_node in route_nodes:
                    self._add_edge(route_node, table_node, "reads_from")

            for table in write_tables:
                table_node = self._add_node("Table", table)
                for route_node in route_nodes:
                    self._add_edge(route_node, table_node, "writes_to")

            for agent in agents:
                agent_node = self._add_node("Agent", agent, {"path": str(path)})
                for table in read_tables:
                    table_node = self._add_node("Table", table)
                    self._add_edge(agent_node, table_node, "reads_from")
                for table in write_tables:
                    table_node = self._add_node("Table", table)
                    self._add_edge(agent_node, table_node, "writes_to")

            if path.name in {"page.tsx", "page.jsx", "page.js", "page.ts"}:
                page_label = str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else str(path)
                page_node = self._add_node("Frontend_Page", page_label, {"path": str(path)})
                for api_path in api_calls:
                    api_node = self._add_node("API_Route", f"FETCH {api_path}")
                    self._add_edge(page_node, api_node, "calls")

        logger.info("Graph built with %s nodes and %s edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        return self.graph

    def get_dependents(self, table: str) -> List[str]:
        node_id = f"Table:{table}"
        if node_id not in self.graph:
            return []
        dependents = {edge[0] for edge in self.graph.in_edges(node_id)}
        return sorted(dependents)

    def to_json(self) -> Dict[str, List[Dict[str, str]]]:
        nodes = [
            {"id": node_id, **attrs}
            for node_id, attrs in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": src, "target": dst, **attrs}
            for src, dst, attrs in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}

    def save_json(self, path: str) -> None:
        payload = self.to_json()
        Path(path).write_text(json.dumps(payload, indent=2))
