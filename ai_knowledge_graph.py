#!/usr/bin/env python3
"""
AI Knowledge Graph - Task 22
Build comprehensive knowledge graph from all system interactions
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import networkx as nx
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration - validate required environment variables
def _get_db_config():
    """Get database configuration with validation for required env vars."""
    required_vars = ["DB_HOST", "DB_USER", "DB_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return {
        "host": os.getenv("DB_HOST"),
        "database": os.getenv("DB_NAME", "postgres"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "port": int(os.getenv("DB_PORT", "5432"))
    }

class NodeType(Enum):
    """Types of knowledge nodes"""
    CUSTOMER = "customer"
    JOB = "job"
    INVOICE = "invoice"
    AGENT = "agent"
    WORKFLOW = "workflow"
    DECISION = "decision"
    CONCEPT = "concept"
    SKILL = "skill"
    PATTERN = "pattern"
    INSIGHT = "insight"
    PROBLEM = "problem"
    SOLUTION = "solution"

class EdgeType(Enum):
    """Types of relationships between nodes"""
    OWNS = "owns"                    # Customer owns Job
    CREATED = "created"              # Agent created Invoice
    EXECUTED = "executed"            # Agent executed Workflow
    DECIDED = "decided"              # Decision decided Outcome
    LEARNED = "learned"              # System learned Pattern
    APPLIES_TO = "applies_to"        # Solution applies to Problem
    SIMILAR_TO = "similar_to"        # Pattern similar to Pattern
    DEPENDS_ON = "depends_on"        # Workflow depends on Skill
    TRIGGERED_BY = "triggered_by"    # Action triggered by Event
    RESULTED_IN = "resulted_in"      # Job resulted in Invoice
    REFERENCES = "references"        # Concept references Concept
    PERFORMED_BY = "performed_by"    # Job performed by Agent

class EntityType(Enum):
    """Types of entities that can be extracted"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    PRODUCT = "product"
    SERVICE = "service"
    TECHNOLOGY = "technology"
    METRIC = "metric"
    EVENT = "event"
    TIMESTAMP = "timestamp"
    MONEY = "money"

class KnowledgeExtractor:
    """Extract knowledge from system interactions"""

    def __init__(self):
        self.extraction_patterns = {}
        self.entity_cache = {}
        self.relationship_patterns = {
            "customer_owns_job": r"customer.*(?:owns|has|purchased).*job",
            "agent_executed": r"agent.*(?:executed|performed|completed).*(?:task|workflow)",
            "caused_by": r"(?:caused by|due to|because of)",
            "resulted_in": r"(?:resulted in|led to|produced)",
            "depends_on": r"(?:depends on|requires|needs)",
        }

    async def extract_from_executions(
        self,
        days: int = 30
    ) -> list[dict]:
        """Extract knowledge from agent executions"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get recent executions
            cursor.execute("""
                SELECT
                    id,
                    agent_type,
                    prompt,
                    response,
                    status,
                    created_at
                FROM agent_executions
                WHERE created_at > NOW() - INTERVAL '%s days'
                  AND status = 'completed'
                ORDER BY created_at DESC
                LIMIT 1000
            """, (days,))

            executions = cursor.fetchall()
            knowledge_items = []

            for execution in executions:
                # Extract entities and relationships
                items = await self._extract_from_execution(execution)
                knowledge_items.extend(items)

            cursor.close()
            conn.close()

            return knowledge_items

        except Exception as e:
            logger.error(f"Error extracting from executions: {e}")
            raise

    async def _extract_from_execution(
        self,
        execution: dict
    ) -> list[dict]:
        """Extract knowledge from single execution"""
        items = []

        # Extract agent as entity
        agent_node = {
            "type": NodeType.AGENT.value,
            "name": execution['agent_type'],
            "properties": {
                "execution_id": execution['id'],
                "status": execution['status'],
                "timestamp": execution['created_at'].isoformat()
            }
        }
        items.append(agent_node)

        # Extract entities from prompt
        if execution.get('prompt'):
            # Extract concepts
            concepts = self._extract_concepts(execution['prompt'])
            for concept in concepts:
                items.append({
                    "type": NodeType.CONCEPT.value,
                    "name": concept,
                    "properties": {
                        "source": "prompt",
                        "execution_id": execution['id']
                    }
                })

            # Extract entities with NER
            entities = self.extract_entities(execution['prompt'])
            for entity in entities:
                items.append({
                    "type": NodeType.CONCEPT.value,
                    "name": f"{entity['type']}:{entity['value']}",
                    "properties": {
                        "entity_type": entity['type'],
                        "entity_value": entity['value'],
                        "source": "entity_extraction",
                        "execution_id": execution['id']
                    }
                })

            # Extract relationships
            relationships = self.extract_relationships(execution['prompt'], entities)
            for rel in relationships:
                items.append({
                    "type": NodeType.PATTERN.value,
                    "name": f"relationship:{rel['type']}",
                    "properties": {
                        "relationship_type": rel['type'],
                        "context": rel.get('context', ''),
                        "confidence": rel.get('confidence', 0.5)
                    }
                })

        # Extract patterns from results
        if execution.get('response'):
            patterns = self._extract_patterns(execution['response'])
            for pattern in patterns:
                items.append({
                    "type": NodeType.PATTERN.value,
                    "name": pattern['name'],
                    "properties": pattern['properties']
                })

        return items

    def extract_entities(self, text: str) -> list[dict]:
        """Extract entities from text with NER-like functionality"""
        entities = []
        import re

        # Extract monetary values
        money_pattern = r'\$[\d,]+(?:\.\d{2})?'
        for match in re.finditer(money_pattern, text):
            entities.append({
                "type": EntityType.MONEY.value,
                "value": match.group(),
                "position": match.span()
            })

        # Extract dates and timestamps
        date_pattern = r'\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "type": EntityType.TIMESTAMP.value,
                "value": match.group(),
                "position": match.span()
            })

        # Extract metrics (percentages, numbers with units)
        metric_pattern = r'\d+(?:\.\d+)?(?:%|GB|MB|KB|hours?|days?|minutes?)'
        for match in re.finditer(metric_pattern, text):
            entities.append({
                "type": EntityType.METRIC.value,
                "value": match.group(),
                "position": match.span()
            })

        # Extract technology terms
        tech_terms = ['AI', 'API', 'database', 'PostgreSQL', 'Python', 'FastAPI', 'LangChain', 'OpenAI']
        for term in tech_terms:
            if term in text:
                entities.append({
                    "type": EntityType.TECHNOLOGY.value,
                    "value": term,
                    "context": text
                })

        # Extract business entities
        business_patterns = {
            EntityType.PRODUCT.value: r'(?:product|service|package):\s*([A-Z][a-zA-Z\s]+)',
            EntityType.EVENT.value: r'(?:event|meeting|call):\s*([A-Z][a-zA-Z\s]+)',
        }

        for entity_type, pattern in business_patterns.items():
            for match in re.finditer(pattern, text):
                entities.append({
                    "type": entity_type,
                    "value": match.group(1).strip(),
                    "position": match.span()
                })

        return entities

    def extract_relationships(self, text: str, entities: list[dict]) -> list[dict]:
        """Extract relationships between entities"""
        relationships = []
        import re

        # Pattern-based relationship extraction
        for rel_type, pattern in self.relationship_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                relationships.append({
                    "type": rel_type,
                    "context": match.group(),
                    "position": match.span(),
                    "confidence": 0.7
                })

        # Entity proximity-based relationships
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if 'position' in entity1 and 'position' in entity2:
                    distance = abs(entity1['position'][0] - entity2['position'][0])
                    if distance < 50:  # Within 50 characters
                        relationships.append({
                            "type": "related_to",
                            "source": entity1,
                            "target": entity2,
                            "distance": distance,
                            "confidence": max(0.3, 1.0 - (distance / 100))
                        })

        return relationships

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract concepts from text"""
        concepts = []

        # Look for key business terms
        business_terms = [
            'customer', 'job', 'invoice', 'payment', 'estimate',
            'lead', 'revenue', 'cost', 'profit', 'workflow',
            'automation', 'optimization', 'analysis', 'report'
        ]

        text_lower = text.lower()
        for term in business_terms:
            if term in text_lower:
                concepts.append(term)

        return concepts

    def _extract_patterns(self, result: Any) -> list[dict]:
        """Extract patterns from execution results"""
        patterns = []

        if isinstance(result, dict):
            # Look for success patterns
            if result.get('status') == 'success':
                patterns.append({
                    "name": "successful_execution",
                    "properties": {
                        "action": result.get('action', 'unknown'),
                        "outcome": result.get('outcome', 'completed')
                    }
                })

            # Look for data patterns
            if 'data' in result:
                data = result['data']
                if isinstance(data, list) and len(data) > 10:
                    patterns.append({
                        "name": "bulk_operation",
                        "properties": {
                            "size": len(data),
                            "type": "batch_processing"
                        }
                    })

        return patterns

    async def extract_from_conversations(self) -> list[dict]:
        """Extract knowledge from conversations"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get conversations with messages
            cursor.execute("""
                SELECT
                    c.id,
                    c.user_id,
                    c.context,
                    COUNT(m.id) as message_count,
                    MAX(m.timestamp) as last_message
                FROM ai_conversations c
                LEFT JOIN ai_messages m ON m.conversation_id = c.id
                GROUP BY c.id, c.user_id, c.context
                HAVING COUNT(m.id) > 0
                ORDER BY MAX(m.timestamp) DESC
                LIMIT 100
            """)

            conversations = cursor.fetchall()
            knowledge_items = []

            for conv in conversations:
                # Extract topics and insights
                if conv.get('context'):
                    items = self._extract_from_context(conv['context'])
                    knowledge_items.extend(items)

            cursor.close()
            conn.close()

            return knowledge_items

        except Exception as e:
            logger.error(f"Error extracting from conversations: {e}")
            return []

    def _extract_from_context(self, context: Any) -> list[dict]:
        """Extract knowledge from conversation context"""
        items = []

        if isinstance(context, dict):
            # Extract topics
            if 'topics' in context:
                for topic in context['topics']:
                    items.append({
                        "type": NodeType.CONCEPT.value,
                        "name": topic,
                        "properties": {"source": "conversation"}
                    })

            # Extract insights
            if 'insights' in context:
                for insight in context['insights']:
                    items.append({
                        "type": NodeType.INSIGHT.value,
                        "name": insight.get('title', 'insight'),
                        "properties": insight
                    })

        return items

    async def extract_from_business_data(self) -> list[dict]:
        """Extract knowledge from business data"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            knowledge_items = []

            # Extract customer relationships
            cursor.execute("""
                SELECT
                    c.id as customer_id,
                    c.name as customer_name,
                    COUNT(DISTINCT j.id) as job_count,
                    COUNT(DISTINCT i.id) as invoice_count,
                    SUM(COALESCE(i.amount, 0)) as total_revenue
                FROM customers c
                LEFT JOIN jobs j ON j.customer_id = c.id
                LEFT JOIN invoices i ON i.customer_id = c.id
                GROUP BY c.id, c.name
                HAVING COUNT(j.id) > 0
                LIMIT 100
            """)

            customers = cursor.fetchall()

            for customer in customers:
                # Create customer node
                knowledge_items.append({
                    "type": NodeType.CUSTOMER.value,
                    "name": customer['customer_name'] or f"Customer_{customer['customer_id']}",
                    "properties": {
                        "id": customer['customer_id'],
                        "job_count": customer['job_count'],
                        "invoice_count": customer['invoice_count'],
                        "total_revenue": float(customer['total_revenue'] or 0)
                    }
                })

                # Create value pattern if high revenue
                if customer['total_revenue'] and customer['total_revenue'] > 1000:
                    knowledge_items.append({
                        "type": NodeType.PATTERN.value,
                        "name": "high_value_customer",
                        "properties": {
                            "customer_id": customer['customer_id'],
                            "revenue": float(customer['total_revenue'])
                        }
                    })

            cursor.close()
            conn.close()

            return knowledge_items

        except Exception as e:
            logger.error(f"Error extracting from business data: {e}")
            return []

class KnowledgeGraphBuilder:
    """Build and maintain knowledge graph"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_index = {}
        self.edge_index = defaultdict(list)

    async def build_graph(
        self,
        knowledge_items: list[dict]
    ) -> dict:
        """Build graph from knowledge items"""
        try:
            # Add nodes
            for item in knowledge_items:
                node_id = self._create_node_id(item)
                self.graph.add_node(
                    node_id,
                    type=item['type'],
                    name=item['name'],
                    **item.get('properties', {})
                )
                self.node_index[node_id] = item

            # Infer relationships
            await self._infer_relationships()

            # Calculate graph metrics
            metrics = self._calculate_metrics()

            # Store in database
            await self._store_graph()

            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "metrics": metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise

    def _create_node_id(self, item: dict) -> str:
        """Create unique node ID"""
        unique_string = f"{item['type']}_{item['name']}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    async def _infer_relationships(self):
        """Infer relationships between nodes"""
        nodes = list(self.graph.nodes(data=True))

        for i, (node1_id, node1_data) in enumerate(nodes):
            for node2_id, node2_data in nodes[i+1:]:
                # Check for relationships
                relationship = self._check_relationship(
                    node1_data,
                    node2_data
                )

                if relationship:
                    self.graph.add_edge(
                        node1_id,
                        node2_id,
                        type=relationship,
                        weight=1.0
                    )

    def _check_relationship(
        self,
        node1: dict,
        node2: dict
    ) -> Optional[str]:
        """Check if two nodes are related"""
        # Customer owns Job
        if node1['type'] == NodeType.CUSTOMER.value and node2['type'] == NodeType.JOB.value:
            if node1.get('id') == node2.get('customer_id'):
                return EdgeType.OWNS.value

        # Agent executed Workflow
        if node1['type'] == NodeType.AGENT.value and node2['type'] == NodeType.WORKFLOW.value:
            if node1.get('name') in node2.get('name', ''):
                return EdgeType.EXECUTED.value

        # Pattern similar to Pattern
        if node1['type'] == NodeType.PATTERN.value and node2['type'] == NodeType.PATTERN.value:
            if node1.get('name') == node2.get('name'):
                return EdgeType.SIMILAR_TO.value

        # Concept references Concept
        if node1['type'] == NodeType.CONCEPT.value and node2['type'] == NodeType.CONCEPT.value:
            # Check semantic similarity
            if self._semantic_similarity(node1['name'], node2['name']) > 0.7:
                return EdgeType.REFERENCES.value

        return None

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        # Simple overlap coefficient
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        return len(intersection) / min(len(words1), len(words2))

    def _calculate_metrics(self) -> dict:
        """Calculate graph metrics"""
        metrics = {}

        # Basic metrics
        metrics['density'] = nx.density(self.graph)
        metrics['is_connected'] = nx.is_weakly_connected(self.graph)

        # Centrality metrics
        if self.graph.number_of_nodes() > 0:
            try:
                metrics['avg_degree'] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()

                # PageRank for importance
                pagerank = nx.pagerank(self.graph, max_iter=100)
                top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics['top_nodes'] = [
                    {
                        "id": node_id,
                        "name": self.graph.nodes[node_id].get('name'),
                        "type": self.graph.nodes[node_id].get('type'),
                        "score": score
                    }
                    for node_id, score in top_nodes
                ]
            except (nx.NetworkXException, ZeroDivisionError) as exc:
                logger.warning("Failed to compute centrality metrics: %s", exc, exc_info=True)

        # Component analysis
        if self.graph.number_of_nodes() > 0:
            components = list(nx.weakly_connected_components(self.graph))
            metrics['num_components'] = len(components)
            metrics['largest_component_size'] = max(len(c) for c in components)

        return metrics

    async def _store_graph(self):
        """Store graph in database"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor()

            # Store nodes
            for node_id, data in self.graph.nodes(data=True):
                cursor.execute("""
                    INSERT INTO ai_knowledge_nodes (
                        id, node_type, name, properties,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, NOW(), NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        properties = EXCLUDED.properties,
                        updated_at = NOW()
                """, (
                    node_id,
                    data['type'],
                    data['name'],
                    json.dumps({k: v for k, v in data.items() if k not in ['type', 'name']})
                ))

            # Store edges
            for source, target, edge_data in self.graph.edges(data=True):
                edge_id = hashlib.md5(f"{source}_{target}".encode()).hexdigest()

                cursor.execute("""
                    INSERT INTO ai_knowledge_edges (
                        id, source_id, target_id, edge_type,
                        weight, properties, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        properties = EXCLUDED.properties
                """, (
                    edge_id,
                    source,
                    target,
                    edge_data.get('type', 'related'),
                    edge_data.get('weight', 1.0),
                    json.dumps({k: v for k, v in edge_data.items() if k not in ['type', 'weight']})
                ))

            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing graph: {e}")
            raise

class KnowledgeQueryEngine:
    """Query and traverse knowledge graph with vector embeddings support"""

    def __init__(self):
        self.graph = None
        self.embedding_cache = {}
        self.openai_client = None

        # Initialize OpenAI for semantic queries
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.warning(f"OpenAI not available for semantic queries: {e}")

    async def load_graph(self) -> nx.DiGraph:
        """Load graph from database"""
        try:
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            graph = nx.DiGraph()

            # Load nodes
            cursor.execute("""
                SELECT id, node_type, name, properties
                FROM ai_knowledge_nodes
            """)

            nodes = cursor.fetchall()
            for node in nodes:
                props = node['properties']
                if isinstance(props, dict):
                    properties = props
                else:
                    properties = json.loads(props or '{}')

                graph.add_node(
                    node['id'],
                    type=node['node_type'],
                    name=node['name'],
                    **properties
                )

            # Load edges
            cursor.execute("""
                SELECT source_id, target_id, edge_type, weight, properties
                FROM ai_knowledge_edges
            """)

            edges = cursor.fetchall()
            for edge in edges:
                props = edge['properties']
                if isinstance(props, dict):
                    properties = props
                else:
                    properties = json.loads(props or '{}')

                graph.add_edge(
                    edge['source_id'],
                    edge['target_id'],
                    type=edge['edge_type'],
                    weight=float(edge['weight']),
                    **properties
                )

            cursor.close()
            conn.close()

            self.graph = graph
            return graph

        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            raise

    async def find_path(
        self,
        source_name: str,
        target_name: str
    ) -> Optional[list[str]]:
        """Find path between two nodes"""
        if not self.graph:
            await self.load_graph()

        # Find nodes by name
        source_node = None
        target_node = None

        for node_id, data in self.graph.nodes(data=True):
            if data.get('name') == source_name:
                source_node = node_id
            if data.get('name') == target_name:
                target_node = node_id

        if not source_node or not target_node:
            return None

        try:
            path = nx.shortest_path(self.graph, source_node, target_node)
            return [self.graph.nodes[node_id]['name'] for node_id in path]
        except nx.NetworkXNoPath:
            return None

    async def get_related_nodes(
        self,
        node_name: str,
        max_distance: int = 2
    ) -> list[dict]:
        """Get nodes related to given node"""
        if not self.graph:
            await self.load_graph()

        # Find node by name
        target_node = None
        for node_id, data in self.graph.nodes(data=True):
            if data.get('name') == node_name:
                target_node = node_id
                break

        if not target_node:
            return []

        # Find related nodes within distance
        related = []
        distances = nx.single_source_shortest_path_length(
            self.graph,
            target_node,
            cutoff=max_distance
        )

        for node_id, distance in distances.items():
            if node_id != target_node:
                node_data = self.graph.nodes[node_id]
                related.append({
                    "name": node_data['name'],
                    "type": node_data['type'],
                    "distance": distance,
                    "properties": {k: v for k, v in node_data.items() if k not in ['name', 'type']}
                })

        # Sort by distance
        related.sort(key=lambda x: x['distance'])

        return related

    async def semantic_query(self, query_text: str, limit: int = 10) -> list[dict]:
        """Perform semantic search using vector embeddings"""
        if not self.openai_client:
            logger.warning("OpenAI not available, falling back to keyword search")
            return await self.keyword_search(query_text, limit)

        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query_text
            )
            query_embedding = response.data[0].embedding

            # Load graph with embeddings
            if not self.graph:
                await self.load_graph()

            # Search in database for vector similarity
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

            cursor.execute("""
                SELECT
                    id, node_type, name, properties,
                    1 - (embedding <=> %s::vector) as similarity
                FROM ai_knowledge_nodes
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, limit))

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            return [{
                "id": r['id'],
                "type": r['node_type'],
                "name": r['name'],
                "properties": r['properties'],
                "similarity": float(r['similarity'])
            } for r in results]

        except Exception as e:
            logger.error(f"Semantic query failed: {e}")
            return []

    async def keyword_search(self, query_text: str, limit: int = 10) -> list[dict]:
        """Fallback keyword-based search"""
        if not self.graph:
            await self.load_graph()

        results = []
        query_lower = query_text.lower()

        for node_id, data in self.graph.nodes(data=True):
            score = 0
            if query_lower in data.get('name', '').lower():
                score += 1.0

            # Check properties
            props = {k: v for k, v in data.items() if k not in ['name', 'type']}
            for value in props.values():
                if isinstance(value, str) and query_lower in value.lower():
                    score += 0.5

            if score > 0:
                results.append({
                    "id": node_id,
                    "type": data.get('type'),
                    "name": data.get('name'),
                    "properties": props,
                    "score": score
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]

    async def get_insights(self) -> list[dict]:
        """Extract insights from graph structure with pattern recognition"""
        if not self.graph:
            await self.load_graph()

        insights = []

        # Find hub nodes (high degree centrality)
        if self.graph.number_of_nodes() > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            top_hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]

            for node_id, centrality in top_hubs:
                node_data = self.graph.nodes[node_id]
                insights.append({
                    "type": "hub_node",
                    "title": f"Key {node_data['type']}: {node_data['name']}",
                    "description": f"Central to {int(centrality * 100)}% of relationships",
                    "importance": centrality,
                    "recommendation": f"Focus on maintaining and expanding connections for {node_data['name']}"
                })

        # Find clusters and communities
        if self.graph.number_of_nodes() > 10:
            components = list(nx.weakly_connected_components(self.graph))
            if len(components) > 1:
                insights.append({
                    "type": "clustering",
                    "title": f"Knowledge organized in {len(components)} clusters",
                    "description": f"Largest cluster has {max(len(c) for c in components)} nodes",
                    "importance": 0.7,
                    "recommendation": "Consider building bridges between isolated knowledge clusters"
                })

        # Find patterns and trends
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data['type']] += 1

        dominant_type = max(node_types.items(), key=lambda x: x[1])
        insights.append({
            "type": "pattern",
            "title": f"Knowledge focus: {dominant_type[0]}",
            "description": f"{dominant_type[1]} nodes of this type ({int(dominant_type[1]/self.graph.number_of_nodes()*100)}%)",
            "importance": 0.5,
            "recommendation": "Diversify knowledge base by adding more varied node types"
        })

        # Detect knowledge gaps
        edge_types = defaultdict(int)
        for _, _, edge_data in self.graph.edges(data=True):
            edge_types[edge_data.get('type', 'unknown')] += 1

        if len(edge_types) < 3:
            insights.append({
                "type": "knowledge_gap",
                "title": "Limited relationship diversity",
                "description": f"Only {len(edge_types)} relationship types found",
                "importance": 0.6,
                "recommendation": "Capture more diverse interactions to enrich knowledge graph"
            })

        # Identify trending patterns
        try:
            # Get temporal patterns
            conn = psycopg2.connect(**_get_db_config())
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    node_type,
                    COUNT(*) as count,
                    DATE_TRUNC('day', created_at) as day
                FROM ai_knowledge_nodes
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY node_type, DATE_TRUNC('day', created_at)
                ORDER BY day DESC, count DESC
            """)

            temporal_data = cursor.fetchall()
            if temporal_data:
                recent_type = temporal_data[0]['node_type']
                recent_count = temporal_data[0]['count']

                insights.append({
                    "type": "trend",
                    "title": f"Trending knowledge type: {recent_type}",
                    "description": f"{recent_count} new nodes in the last day",
                    "importance": 0.8,
                    "recommendation": f"Monitor {recent_type} growth and ensure quality"
                })

            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error detecting trends: {e}")

        return insights

class AIKnowledgeGraph:
    """Main knowledge graph system"""

    def __init__(self):
        self.extractor = KnowledgeExtractor()
        self.builder = KnowledgeGraphBuilder()
        self.query_engine = KnowledgeQueryEngine()

    async def build_from_all_sources(self) -> dict:
        """Build knowledge graph from all available sources"""
        try:
            knowledge_items = []

            # Extract from executions
            execution_items = await self.extractor.extract_from_executions()
            knowledge_items.extend(execution_items)

            # Extract from conversations
            conversation_items = await self.extractor.extract_from_conversations()
            knowledge_items.extend(conversation_items)

            # Extract from business data
            business_items = await self.extractor.extract_from_business_data()
            knowledge_items.extend(business_items)

            # Build graph
            result = await self.builder.build_graph(knowledge_items)

            # Generate insights
            insights = await self.query_engine.get_insights()
            result['insights'] = insights

            return result

        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise

    async def query(
        self,
        query_type: str,
        parameters: dict
    ) -> Any:
        """Query knowledge graph"""
        if query_type == "find_path":
            return await self.query_engine.find_path(
                parameters['source'],
                parameters['target']
            )
        elif query_type == "related_nodes":
            return await self.query_engine.get_related_nodes(
                parameters['node'],
                parameters.get('max_distance', 2)
            )
        elif query_type == "insights":
            return await self.query_engine.get_insights()
        else:
            raise ValueError(f"Unknown query type: {query_type}")

# Singleton instance
_graph_instance = None

def get_knowledge_graph():
    """Get or create knowledge graph instance"""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = AIKnowledgeGraph()
    return _graph_instance

# Export main components
__all__ = [
    'AIKnowledgeGraph',
    'get_knowledge_graph',
    'NodeType',
    'EdgeType'
]
