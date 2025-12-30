#!/usr/bin/env python3
"""
Knowledge Graph Extractor - Automated Entity and Relationship Extraction
Extracts entities from ai_agent_executions and populates knowledge graph tables

Scheduled to run every 30 minutes to continuously build the knowledge graph.
Target: 100+ nodes within 24 hours of deployment.
"""

import os
import json
import logging
import asyncio
import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of knowledge nodes"""
    AGENT = "agent"
    CUSTOMER = "customer"
    JOB = "job"
    DECISION = "decision"
    ACTION = "action"
    METRIC = "metric"
    ERROR = "error"
    PATTERN = "pattern"
    INSIGHT = "insight"
    WORKFLOW = "workflow"
    MONEY = "money"
    TIMESTAMP = "timestamp"
    ENTITY = "entity"


class EdgeType(Enum):
    """Types of relationships between nodes"""
    EXECUTED = "executed"
    PRODUCED = "produced"
    FAILED_WITH = "failed_with"
    AFFECTED = "affected"
    TRIGGERED = "triggered"
    RELATED_TO = "related_to"
    MEASURED = "measured"
    CREATED = "created"
    DECIDED = "decided"
    PROCESSED = "processed"


@dataclass
class ExtractedNode:
    """Represents an extracted knowledge node"""
    node_id: str
    node_type: str
    name: str
    properties: Dict[str, Any]
    importance_score: float = 0.5
    source_execution_id: Optional[str] = None


@dataclass
class ExtractedEdge:
    """Represents an extracted relationship"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = None


class KnowledgeGraphExtractor:
    """
    Extracts entities and relationships from ai_agent_executions
    and populates ai_knowledge_nodes and ai_knowledge_edges tables.
    """

    def __init__(self):
        self.pool = None
        self.last_extraction_time = None
        self.extraction_stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "executions_processed": 0,
            "errors": 0
        }

        # Patterns for entity extraction
        self.money_pattern = re.compile(r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:USD|dollars?)')
        self.customer_id_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
        self.metric_pattern = re.compile(r'\b(\d+(?:\.\d+)?)\s*(%|percent|GB|MB|KB|ms|seconds?|minutes?|hours?|days?)\b')

        # High-value agents that should have higher importance scores
        self.high_value_agents = {
            "CustomerIntelligence", "RevenueOptimizer", "HealthMonitor",
            "LeadQualificationAgent", "BudgetingAgent", "InvoicingAgent"
        }

    async def initialize(self):
        """Initialize database connection pool"""
        from database.async_connection import get_pool
        self.pool = get_pool()
        logger.info("KnowledgeGraphExtractor initialized with database pool")

    def _generate_node_id(self, node_type: str, name: str, extra: str = "") -> str:
        """Generate deterministic node ID from type and name"""
        unique_str = f"{node_type}:{name}:{extra}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: str) -> str:
        """Generate deterministic edge ID"""
        unique_str = f"{source_id}:{target_id}:{edge_type}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def _calculate_importance(self, agent_name: str, status: str, output_data: Dict) -> float:
        """Calculate importance score for a node"""
        base_score = 0.5

        # High-value agents get higher scores
        if agent_name in self.high_value_agents:
            base_score += 0.2

        # Successful executions get higher scores
        if status == "completed":
            base_score += 0.1
        elif status == "failed":
            base_score += 0.15  # Errors are important for learning

        # Check for significant actions in output
        if isinstance(output_data, dict):
            actions = output_data.get("actions_taken", [])
            if len(actions) > 5:
                base_score += 0.1

            # Revenue-related data is important
            if any(k in output_data for k in ["total_revenue", "revenue", "potential_revenue"]):
                base_score += 0.1

            # Customer interventions are important
            if output_data.get("interventions_created", 0) > 0:
                base_score += 0.1

        return min(base_score, 1.0)  # Cap at 1.0

    def _extract_entities_from_json(self, data: Any, prefix: str = "") -> List[ExtractedNode]:
        """Extract entities from JSON output data"""
        nodes = []

        if not data:
            return nodes

        if isinstance(data, dict):
            # Extract customer IDs
            for key in ["customer_id", "customer_ids", "customers"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, str) and self.customer_id_pattern.match(val):
                        nodes.append(ExtractedNode(
                            node_id=self._generate_node_id("customer", val),
                            node_type=NodeType.CUSTOMER.value,
                            name=f"customer_{val[:8]}",
                            properties={"customer_id": val, "source": prefix}
                        ))
                    elif isinstance(val, list):
                        for cid in val[:10]:  # Limit to 10 customers
                            if isinstance(cid, str) and self.customer_id_pattern.match(cid):
                                nodes.append(ExtractedNode(
                                    node_id=self._generate_node_id("customer", cid),
                                    node_type=NodeType.CUSTOMER.value,
                                    name=f"customer_{cid[:8]}",
                                    properties={"customer_id": cid}
                                ))

            # Extract revenue/money values
            for key in ["total_revenue", "revenue", "amount", "ltv", "value_at_risk", "potential_revenue"]:
                if key in data and data[key]:
                    try:
                        value = float(data[key])
                        if value > 0:
                            nodes.append(ExtractedNode(
                                node_id=self._generate_node_id("money", f"{key}_{value}"),
                                node_type=NodeType.MONEY.value,
                                name=f"{key}: ${value:,.2f}",
                                properties={"metric_type": key, "value": value},
                                importance_score=0.6 if value > 10000 else 0.4
                            ))
                    except (ValueError, TypeError):
                        pass

            # Extract metrics
            for key in ["total_jobs", "total_customers", "total_invoices", "jobs_analyzed",
                       "active_jobs", "pending_jobs", "interventions_created", "at_risk_customers_found"]:
                if key in data and data[key]:
                    try:
                        value = int(data[key])
                        if value > 0:
                            nodes.append(ExtractedNode(
                                node_id=self._generate_node_id("metric", f"{key}_{value}"),
                                node_type=NodeType.METRIC.value,
                                name=f"{key}: {value:,}",
                                properties={"metric_type": key, "value": value}
                            ))
                    except (ValueError, TypeError):
                        pass

            # Extract actions taken
            if "actions_taken" in data and isinstance(data["actions_taken"], list):
                for i, action in enumerate(data["actions_taken"][:20]):  # Limit to 20 actions
                    if isinstance(action, dict):
                        action_type = action.get("action", "unknown_action")
                        nodes.append(ExtractedNode(
                            node_id=self._generate_node_id("action", f"{action_type}_{i}"),
                            node_type=NodeType.ACTION.value,
                            name=action_type,
                            properties=action
                        ))

            # Extract errors
            if "error" in data and data["error"]:
                error_msg = str(data["error"])[:200]
                nodes.append(ExtractedNode(
                    node_id=self._generate_node_id("error", error_msg),
                    node_type=NodeType.ERROR.value,
                    name=f"error: {error_msg[:50]}...",
                    properties={"error_message": error_msg},
                    importance_score=0.7  # Errors are important for learning
                ))

            # Extract health summary
            if "health_summary" in data and isinstance(data["health_summary"], dict):
                summary = data["health_summary"]
                if "error" in summary:
                    nodes.append(ExtractedNode(
                        node_id=self._generate_node_id("error", str(summary["error"])[:100]),
                        node_type=NodeType.ERROR.value,
                        name="health_check_error",
                        properties={"error": str(summary["error"])[:500]},
                        importance_score=0.8
                    ))

            # Extract decisions/insights
            for key in ["decision", "insight", "recommendation"]:
                if key in data and data[key]:
                    val = str(data[key])[:200]
                    nodes.append(ExtractedNode(
                        node_id=self._generate_node_id("decision", val),
                        node_type=NodeType.DECISION.value if key == "decision" else NodeType.INSIGHT.value,
                        name=f"{key}: {val[:50]}...",
                        properties={key: val}
                    ))

        return nodes

    async def extract_from_executions(
        self,
        hours_back: int = 24,
        limit: int = 500
    ) -> Tuple[List[ExtractedNode], List[ExtractedEdge]]:
        """
        Extract knowledge from recent agent executions.

        Args:
            hours_back: How many hours back to look for executions
            limit: Maximum number of executions to process

        Returns:
            Tuple of (nodes, edges) extracted
        """
        if not self.pool:
            await self.initialize()

        nodes: List[ExtractedNode] = []
        edges: List[ExtractedEdge] = []
        seen_node_ids: Set[str] = set()

        try:
            # Get recent executions
            # Note: asyncpg uses $1, $2 style parameters
            query = """
                SELECT
                    id, agent_name, task_type, input_data, output_data,
                    status, error_message, execution_time_ms, created_at
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour' * $1
                ORDER BY created_at DESC
                LIMIT $2
            """

            rows = await self.pool.fetch(query, hours_back, limit)
            logger.info(f"Processing {len(rows)} agent executions from last {hours_back} hours")

            for row in rows:
                try:
                    execution_id = str(row["id"])
                    agent_name = row["agent_name"] or "unknown_agent"
                    status = row["status"] or "unknown"
                    output_data = row["output_data"] or {}
                    input_data = row["input_data"] or {}
                    created_at = row["created_at"]

                    # Create agent node
                    agent_node_id = self._generate_node_id("agent", agent_name)
                    if agent_node_id not in seen_node_ids:
                        importance = self._calculate_importance(agent_name, status, output_data)
                        nodes.append(ExtractedNode(
                            node_id=agent_node_id,
                            node_type=NodeType.AGENT.value,
                            name=agent_name,
                            properties={
                                "agent_name": agent_name,
                                "last_execution": created_at.isoformat() if created_at else None,
                                "status": status
                            },
                            importance_score=importance,
                            source_execution_id=execution_id
                        ))
                        seen_node_ids.add(agent_node_id)

                    # Create execution/workflow node
                    workflow_node_id = self._generate_node_id("workflow", f"{agent_name}_{execution_id[:8]}")
                    if workflow_node_id not in seen_node_ids:
                        nodes.append(ExtractedNode(
                            node_id=workflow_node_id,
                            node_type=NodeType.WORKFLOW.value,
                            name=f"{agent_name}_execution",
                            properties={
                                "execution_id": execution_id,
                                "status": status,
                                "execution_time_ms": row.get("execution_time_ms"),
                                "timestamp": created_at.isoformat() if created_at else None
                            },
                            source_execution_id=execution_id
                        ))
                        seen_node_ids.add(workflow_node_id)

                        # Create edge: agent -> executed -> workflow
                        edges.append(ExtractedEdge(
                            edge_id=self._generate_edge_id(agent_node_id, workflow_node_id, "executed"),
                            source_id=agent_node_id,
                            target_id=workflow_node_id,
                            edge_type=EdgeType.EXECUTED.value,
                            weight=1.0,
                            properties={"timestamp": created_at.isoformat() if created_at else None}
                        ))

                    # Extract entities from output data
                    if output_data:
                        extracted = self._extract_entities_from_json(output_data, prefix=agent_name)
                        for entity_node in extracted:
                            if entity_node.node_id not in seen_node_ids:
                                entity_node.source_execution_id = execution_id
                                nodes.append(entity_node)
                                seen_node_ids.add(entity_node.node_id)

                                # Create edge: workflow -> produced -> entity
                                edge_type = EdgeType.FAILED_WITH if entity_node.node_type == NodeType.ERROR.value else EdgeType.PRODUCED
                                edges.append(ExtractedEdge(
                                    edge_id=self._generate_edge_id(workflow_node_id, entity_node.node_id, edge_type.value),
                                    source_id=workflow_node_id,
                                    target_id=entity_node.node_id,
                                    edge_type=edge_type.value,
                                    weight=0.8
                                ))

                    # Handle errors
                    if status == "failed" and row.get("error_message"):
                        error_msg = str(row["error_message"])[:200]
                        error_node_id = self._generate_node_id("error", error_msg)
                        if error_node_id not in seen_node_ids:
                            nodes.append(ExtractedNode(
                                node_id=error_node_id,
                                node_type=NodeType.ERROR.value,
                                name=f"error: {error_msg[:50]}...",
                                properties={"error_message": error_msg, "agent": agent_name},
                                importance_score=0.7,
                                source_execution_id=execution_id
                            ))
                            seen_node_ids.add(error_node_id)

                            # Create edge: workflow -> failed_with -> error
                            edges.append(ExtractedEdge(
                                edge_id=self._generate_edge_id(workflow_node_id, error_node_id, "failed_with"),
                                source_id=workflow_node_id,
                                target_id=error_node_id,
                                edge_type=EdgeType.FAILED_WITH.value,
                                weight=1.0
                            ))

                    self.extraction_stats["executions_processed"] += 1

                except Exception as e:
                    logger.error(f"Error processing execution {row.get('id')}: {e}")
                    self.extraction_stats["errors"] += 1

            logger.info(f"Extracted {len(nodes)} nodes and {len(edges)} edges from {len(rows)} executions")
            return nodes, edges

        except Exception as e:
            logger.error(f"Error extracting from executions: {e}")
            raise

    async def store_nodes(self, nodes: List[ExtractedNode]) -> int:
        """Store extracted nodes in ai_knowledge_nodes table"""
        if not self.pool:
            await self.initialize()

        if not nodes:
            return 0

        stored = 0

        try:
            for node in nodes:
                # Convert properties to JSON string if needed
                props = node.properties if isinstance(node.properties, str) else json.dumps(node.properties)

                await self.pool.execute("""
                    INSERT INTO ai_knowledge_nodes (
                        id, node_type, name, properties, importance_score,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, $4::jsonb, $5, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        properties = EXCLUDED.properties,
                        importance_score = GREATEST(ai_knowledge_nodes.importance_score, EXCLUDED.importance_score),
                        updated_at = NOW()
                """, node.node_id, node.node_type, node.name, props, node.importance_score)
                stored += 1

            self.extraction_stats["nodes_created"] += stored
            logger.info(f"Stored {stored} nodes in ai_knowledge_nodes")
            return stored

        except Exception as e:
            logger.error(f"Error storing nodes: {e}")
            raise

    async def store_edges(self, edges: List[ExtractedEdge]) -> int:
        """Store extracted edges in ai_knowledge_edges table"""
        if not self.pool:
            await self.initialize()

        if not edges:
            return 0

        stored = 0

        try:
            for edge in edges:
                props = json.dumps(edge.properties or {})

                await self.pool.execute("""
                    INSERT INTO ai_knowledge_edges (
                        id, source_id, target_id, edge_type, weight, properties, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        weight = GREATEST(ai_knowledge_edges.weight, EXCLUDED.weight),
                        properties = EXCLUDED.properties
                """, edge.edge_id, edge.source_id, edge.target_id, edge.edge_type, edge.weight, props)
                stored += 1

            self.extraction_stats["edges_created"] += stored
            logger.info(f"Stored {stored} edges in ai_knowledge_edges")
            return stored

        except Exception as e:
            logger.error(f"Error storing edges: {e}")
            raise

    async def update_graph_metadata(self, nodes_count: int, edges_count: int) -> None:
        """Update ai_knowledge_graph with extraction metadata"""
        if not self.pool:
            await self.initialize()

        try:
            # Insert or update graph metadata
            graph_id = self._generate_node_id("graph", "main_knowledge_graph")

            await self.pool.execute("""
                INSERT INTO ai_knowledge_graph (
                    id, node_type, node_id, node_data, edges, importance_score,
                    created_at, updated_at
                ) VALUES ($1, 'graph_metadata', $2, $3::jsonb, $4::jsonb, 0.9, NOW(), NOW())
                ON CONFLICT (id) DO UPDATE SET
                    node_data = $3::jsonb,
                    edges = $4::jsonb,
                    updated_at = NOW()
            """,
                graph_id,
                "main_graph",
                json.dumps({
                    "total_nodes": nodes_count,
                    "total_edges": edges_count,
                    "last_extraction": datetime.now(timezone.utc).isoformat(),
                    "extraction_stats": self.extraction_stats
                }),
                json.dumps({"edge_count": edges_count})
            )

            logger.info(f"Updated graph metadata: {nodes_count} nodes, {edges_count} edges")

        except Exception as e:
            logger.error(f"Error updating graph metadata: {e}")

    async def run_extraction(self, hours_back: int = 24) -> Dict[str, Any]:
        """
        Run full extraction pipeline.

        Args:
            hours_back: How many hours of executions to process

        Returns:
            Dictionary with extraction results
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Extract from executions
            nodes, edges = await self.extract_from_executions(hours_back=hours_back)

            # Store nodes
            nodes_stored = await self.store_nodes(nodes)

            # Store edges
            edges_stored = await self.store_edges(edges)

            # Get total counts
            total_nodes = await self.pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_nodes")
            total_edges = await self.pool.fetchval("SELECT COUNT(*) FROM ai_knowledge_edges")

            # Update metadata
            await self.update_graph_metadata(total_nodes, total_edges)

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = {
                "success": True,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "nodes_extracted": len(nodes),
                "nodes_stored": nodes_stored,
                "edges_extracted": len(edges),
                "edges_stored": edges_stored,
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "executions_processed": self.extraction_stats["executions_processed"],
                "errors": self.extraction_stats["errors"]
            }

            logger.info(f"Extraction completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": start_time.isoformat()
            }


# Singleton instance
_extractor_instance: Optional[KnowledgeGraphExtractor] = None


def get_knowledge_extractor() -> KnowledgeGraphExtractor:
    """Get or create the knowledge extractor instance"""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = KnowledgeGraphExtractor()
    return _extractor_instance


async def run_scheduled_extraction() -> Dict[str, Any]:
    """
    Scheduled task to extract knowledge graph data.
    Called every 30 minutes by the scheduler.
    """
    extractor = get_knowledge_extractor()
    await extractor.initialize()

    # Look back 2 hours to catch any missed executions
    result = await extractor.run_extraction(hours_back=2)

    return result


# CLI entry point for testing
if __name__ == "__main__":
    async def main():
        extractor = KnowledgeGraphExtractor()
        await extractor.initialize()
        result = await extractor.run_extraction(hours_back=24)
        print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())
