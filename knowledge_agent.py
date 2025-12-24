"""
BrainOps Knowledge Agent - Permanent Memory & Context Manager
Provides permanent knowledge storage with vector embeddings for Claude Code sessions
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
import os

# AI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """Permanent knowledge and context manager for Claude Code sessions with AUREA integration"""

    def __init__(self, db_config: Dict[str, Any], tenant_id: str = "default"):
        self.db_config = db_config
        self.tenant_id = tenant_id
        self.agent_type = "knowledge"
        self.openai_client = None
        self.gemini_model = None

        # AUREA Integration for decision recording and learning
        try:
            from aurea_integration import AUREAIntegration
            self.aurea = AUREAIntegration(tenant_id, self.agent_type)
        except ImportError:
            logger.warning("AUREA integration not available")
            self.aurea = None

        # Initialize AI clients
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ OpenAI client initialized for embeddings")

        if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-002')
            logger.info("✅ Gemini model initialized for Q&A")

    def get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        if not self.openai_client:
            logger.warning("OpenAI not available, returning zero vector")
            return [0.0] * 1536

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 1536

    async def store_knowledge(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Store knowledge entry with vector embedding"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database unavailable")

        try:
            # Generate embedding
            text = f"{entry.get('title', '')}\n\n{entry.get('content', '')}"
            embedding = await self.generate_embedding(text)

            cursor = conn.cursor(cursor_factory=RealDictCursor)
            knowledge_id = str(uuid.uuid4())

            cursor.execute("""
                INSERT INTO brainops_knowledge (
                    id, type, system, title, content, metadata, embedding, version, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                RETURNING id, type, system, title, created_at
            """, (
                knowledge_id,
                entry.get('type', 'general'),
                entry.get('system', 'brainops'),
                entry.get('title'),
                entry.get('content'),
                json.dumps(entry.get('metadata', {})),
                embedding,
                entry.get('version', '1.0.0')
            ))

            result = cursor.fetchone()
            conn.commit()
            cursor.close()
            conn.close()

            return dict(result)
        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            logger.error(f"Knowledge storage error: {e}")
            raise

    async def query_knowledge(self, query: str, filters: Optional[Dict] = None) -> List[Dict]:
        """Semantic search across knowledge base"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database unavailable")

        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Build query with filters
            where_clauses = []
            params = [query_embedding]

            if filters:
                if filters.get('system'):
                    where_clauses.append("system = %s")
                    params.append(filters['system'])
                if filters.get('type'):
                    where_clauses.append("type = %s")
                    params.append(filters['type'])

            where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
            limit = filters.get('limit', 10) if filters else 10
            params.append(limit)

            cursor.execute(f"""
                SELECT
                    id, type, system, title, content, metadata,
                    embedding, version, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM brainops_knowledge
                {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, params + [query_embedding])

            results = cursor.fetchall()
            cursor.close()
            conn.close()

            # Convert to JSON-serializable format
            for result in results:
                result['id'] = str(result['id'])
                result['created_at'] = result['created_at'].isoformat() if result['created_at'] else None
                # Convert embedding to list if needed
                if hasattr(result.get('embedding'), 'tolist'):
                    result['embedding'] = result['embedding'].tolist()

            return [dict(r) for r in results]
        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Knowledge query error: {e}")
            raise

    async def get_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive context summary for Claude sessions"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database unavailable")

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get all knowledge organized by type
            cursor.execute("""
                SELECT
                    id, type, system, title, content, metadata, created_at, updated_at
                FROM brainops_knowledge
                WHERE deleted_at IS NULL
                ORDER BY created_at DESC
                LIMIT 100
            """)

            all_entries = cursor.fetchall()

            # Organize by type and system
            summaries = {}
            systems = set()

            for entry in all_entries:
                entry_type = entry['type']
                entry_system = entry['system']

                systems.add(entry_system)

                if entry_type not in summaries:
                    summaries[entry_type] = []

                summaries[entry_type].append({
                    'id': str(entry['id']),
                    'system': entry_system,
                    'title': entry['title'],
                    'content': entry['content'][:500],  # First 500 chars
                    'metadata': entry['metadata'],
                    'created_at': entry['created_at'].isoformat() if entry['created_at'] else None
                })

            # Get latest updates
            latest = all_entries[:10] if all_entries else []

            # Get quick facts from database
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM customers) as weathercraft_customers,
                    (SELECT COUNT(*) FROM jobs) as weathercraft_jobs,
                    (SELECT COUNT(*) FROM ai_agents WHERE status = 'active') as active_agents
            """)
            quick_facts = cursor.fetchone() or {}

            cursor.close()
            conn.close()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'total_entries': len(all_entries),
                'systems': sorted(list(systems)),
                'summaries': summaries,
                'latest_updates': [{
                    'id': str(e['id']),
                    'type': e['type'],
                    'system': e['system'],
                    'title': e['title'],
                    'created_at': e['created_at'].isoformat() if e['created_at'] else None
                } for e in latest],
                'quick_facts': {
                    'weathercraft_customers': quick_facts.get('weathercraft_customers', 0),
                    'weathercraft_jobs': quick_facts.get('weathercraft_jobs', 0),
                    'active_agents': quick_facts.get('active_agents', 0),
                    'operational_status': 'ONLINE',
                    'ai_tier': 'gemini' if self.gemini_model else 'openai',
                    'cost_per_month': '$0'
                }
            }
        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Context summary error: {e}")
            raise

    async def generate_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from knowledge base patterns"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database unavailable")

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            insights = []

            # Knowledge growth trends
            cursor.execute("""
                SELECT
                    type,
                    DATE_TRUNC('day', created_at) as day,
                    COUNT(*) as count
                FROM brainops_knowledge
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY type, DATE_TRUNC('day', created_at)
                ORDER BY day DESC, count DESC
            """)

            growth_data = cursor.fetchall()
            if growth_data:
                insights.append({
                    "type": "knowledge_growth",
                    "title": "Knowledge Base Growth",
                    "description": f"Added {sum(r['count'] for r in growth_data[:7])} entries in last 7 days",
                    "priority": "medium",
                    "data": [dict(r) for r in growth_data[:7]]
                })

            # System coverage analysis
            cursor.execute("""
                SELECT
                    system,
                    COUNT(*) as entry_count,
                    COUNT(DISTINCT type) as type_diversity
                FROM brainops_knowledge
                WHERE deleted_at IS NULL
                GROUP BY system
                ORDER BY entry_count DESC
            """)

            coverage = cursor.fetchall()
            if coverage:
                top_system = coverage[0]
                insights.append({
                    "type": "system_coverage",
                    "title": f"Primary Focus: {top_system['system']}",
                    "description": f"{top_system['entry_count']} entries across {top_system['type_diversity']} types",
                    "priority": "high",
                    "recommendation": "Ensure balanced coverage across all systems"
                })

            # Knowledge freshness
            cursor.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '7 days') as recent_updates,
                    COUNT(*) FILTER (WHERE updated_at < NOW() - INTERVAL '30 days') as stale_entries,
                    COUNT(*) as total
                FROM brainops_knowledge
                WHERE deleted_at IS NULL
            """)

            freshness = cursor.fetchone()
            if freshness and freshness['stale_entries'] > 0:
                stale_pct = (freshness['stale_entries'] / freshness['total']) * 100
                insights.append({
                    "type": "knowledge_freshness",
                    "title": "Knowledge Maintenance Needed",
                    "description": f"{stale_pct:.1f}% of entries haven't been updated in 30+ days",
                    "priority": "medium" if stale_pct > 50 else "low",
                    "recommendation": "Review and update stale knowledge entries"
                })

            cursor.close()
            conn.close()

            return insights

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Insight generation error: {e}")
            raise

    async def generate_recommendations(self, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on knowledge base"""
        conn = self.get_db_connection()
        if not conn:
            raise Exception("Database unavailable")

        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            recommendations = []

            # Recommend knowledge gaps to fill
            cursor.execute("""
                SELECT DISTINCT type
                FROM brainops_knowledge
                WHERE deleted_at IS NULL
            """)
            existing_types = {r['type'] for r in cursor.fetchall()}

            expected_types = {'architecture', 'api', 'deployment', 'troubleshooting', 'configuration', 'security'}
            missing_types = expected_types - existing_types

            if missing_types:
                recommendations.append({
                    "type": "knowledge_gap",
                    "priority": "high",
                    "action": f"Add knowledge entries for: {', '.join(missing_types)}",
                    "impact": "Improves system coverage and reduces blind spots"
                })

            # Recommend consolidation of duplicate knowledge
            cursor.execute("""
                SELECT title, COUNT(*) as count
                FROM brainops_knowledge
                WHERE deleted_at IS NULL
                GROUP BY title
                HAVING COUNT(*) > 1
                ORDER BY count DESC
                LIMIT 5
            """)

            duplicates = cursor.fetchall()
            if duplicates:
                recommendations.append({
                    "type": "consolidation",
                    "priority": "medium",
                    "action": f"Review and consolidate {len(duplicates)} duplicate title(s)",
                    "impact": "Reduces confusion and improves knowledge quality"
                })

            # Recommend based on recent activity
            cursor.execute("""
                SELECT system, COUNT(*) as recent_activity
                FROM brainops_knowledge
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY system
                ORDER BY recent_activity DESC
                LIMIT 1
            """)

            active_system = cursor.fetchone()
            if active_system:
                recommendations.append({
                    "type": "expand_coverage",
                    "priority": "medium",
                    "action": f"Expand documentation for {active_system['system']} (high activity detected)",
                    "impact": "Capitalizes on current focus area"
                })

            cursor.close()
            conn.close()

            return recommendations

        except Exception as e:
            if conn:
                conn.close()
            logger.error(f"Recommendation generation error: {e}")
            raise

    async def ask(self, question: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """AI-powered question answering using knowledge base with insights"""
        # First, search knowledge base
        relevant_knowledge = await self.query_knowledge(question, {'limit': 5})

        # Build context from knowledge
        context_text = "\n\n".join([
            f"**{k['title']}** ({k['system']})\n{k['content'][:1000]}"
            for k in relevant_knowledge
        ])

        # Generate answer with AI
        if self.gemini_model:
            try:
                prompt = f"""Based on the following knowledge base entries, answer this question:

Question: {question}

Relevant Knowledge:
{context_text}

Provide a concise, accurate answer based only on the information provided.
Include actionable recommendations if applicable."""

                response = self.gemini_model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                answer = "AI generation failed. Here's what I found in the knowledge base:\n\n" + context_text[:500]
        else:
            answer = f"Knowledge base results:\n\n{context_text[:1000]}"

        # Generate contextual recommendations
        try:
            recommendations = await self.generate_recommendations({'question': question})
        except:
            recommendations = []

        return {
            'question': question,
            'answer': answer,
            'relevant_entries': len(relevant_knowledge),
            'sources': [{'title': k['title'], 'system': k['system']} for k in relevant_knowledge],
            'recommendations': recommendations[:3],  # Top 3
            'timestamp': datetime.utcnow().isoformat()
        }

# Global instance
_knowledge_agent = None

def get_knowledge_agent(db_config: Dict[str, Any]) -> KnowledgeAgent:
    """Get or create global knowledge agent instance"""
    global _knowledge_agent
    if _knowledge_agent is None:
        _knowledge_agent = KnowledgeAgent(db_config)
        logger.info("✅ Knowledge Agent initialized")
    return _knowledge_agent
