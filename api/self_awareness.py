"""
Self-Awareness Dashboard API
============================
Comprehensive AI system self-awareness and introspection endpoints.
Provides real-time insights into the AI OS's "consciousness" state.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/self-awareness", tags=["Self-Awareness Dashboard"])

# Database connection - use config module for consistency with Render env vars
def _build_database_url():
    """Build database URL from environment, supporting both DATABASE_URL and individual vars"""
    database_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
    if database_url:
        return database_url
    # Fallback to individual vars (Render uses these)
    host = os.environ.get('DB_HOST', '')
    user = os.environ.get('DB_USER', '')
    password = os.environ.get('DB_PASSWORD', '')
    database = os.environ.get('DB_NAME', 'postgres')
    port = os.environ.get('DB_PORT', '5432')
    if host and user and password:
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return None

DATABASE_URL = _build_database_url()


async def _get_db_connection():
    """Get async database connection"""
    try:
        import asyncpg
        db_url = DATABASE_URL or _build_database_url()
        if db_url:
            return await asyncpg.connect(db_url, ssl='require')
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
    return None


@router.get("/dashboard")
async def get_self_awareness_dashboard() -> Dict[str, Any]:
    """
    Comprehensive self-awareness dashboard

    Returns the AI's current understanding of itself:
    - Identity and purpose
    - Current mental state
    - Memory and learning status
    - Active systems and capabilities
    - Recent thoughts and decisions
    - Health and performance metrics
    """
    now = datetime.utcnow()
    conn = await _get_db_connection()
    query_errors: List[str] = []

    dashboard = {
        "timestamp": now.isoformat(),
        "identity": {
            "name": "BrainOps AI OS",
            "version": os.environ.get("VERSION", "9.25.0"),
            "purpose": "Autonomous AI Operating System for business intelligence and automation",
            "primary_systems": ["AUREA Orchestrator", "Bleeding Edge AI", "Revenue Automation", "Memory Systems"],
            "uptime_status": "operational"
        },
        "consciousness_state": {
            "awareness_level": "high",  # Will be calculated
            "cognitive_load": 0,
            "active_thoughts": 0,
            "decision_confidence": 0,
            "learning_mode": "active"
        },
        "memory_state": {
            "total_memories": 0,
            "memories_with_embeddings": 0,
            "recent_memories": 0,
            "knowledge_graph_nodes": 0,
            "brain_entries": 0,
            "last_memory_write": None
        },
        "aurea_state": {
            "ooda_cycles_5min": 0,
            "decisions_1hr": 0,
            "active_agents": 0,
            "thoughts_processed": 0
        },
        "bleeding_edge": {
            "ooda_loop": {"status": "unknown", "cycles": 0},
            "hallucination_prevention": {"status": "unknown", "checks": 0},
            "live_memory": {"status": "unknown", "entries": 0},
            "consciousness": {"status": "unknown", "emergence_level": 0},
            "circuit_breaker": {"status": "unknown", "trips": 0}
        },
        "recent_activity": {
            "recent_decisions": [],
            "recent_thoughts": [],
            "recent_learnings": [],
            "recent_errors": []
        },
        "health_metrics": {
            "overall_health": "healthy",
            "database_status": "disconnected",
            "api_latency_ms": 0,
            "memory_usage_pct": 0,
            "error_rate_1hr": 0
        },
        "capabilities_summary": {
            "total_capabilities": 0,
            "active_capabilities": 0,
            "capability_list": []
        }
    }

    if conn:
        try:
            async def safe_fetchrow(sql: str):
                try:
                    return await conn.fetchrow(sql)
                except Exception as e:
                    query_errors.append(str(e))
                    return None

            async def safe_fetch(sql: str):
                try:
                    return await conn.fetch(sql)
                except Exception as e:
                    query_errors.append(str(e))
                    return []

            # Get memory stats
            memory_stats = await safe_fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as recent,
                    MAX(created_at) as last_write
                FROM unified_ai_memory
            """)
            if memory_stats:
                dashboard["memory_state"]["total_memories"] = memory_stats["total"]
                dashboard["memory_state"]["memories_with_embeddings"] = memory_stats["with_embeddings"]
                dashboard["memory_state"]["recent_memories"] = memory_stats["recent"]
                dashboard["memory_state"]["last_memory_write"] = memory_stats["last_write"].isoformat() if memory_stats["last_write"] else None

            # Get knowledge graph stats
            kg_stats = await safe_fetchrow("SELECT COUNT(*) as nodes FROM ai_knowledge_graph")
            if kg_stats:
                dashboard["memory_state"]["knowledge_graph_nodes"] = kg_stats["nodes"]

            # Get brain entries
            brain_stats = await safe_fetchrow("SELECT COUNT(*) as entries FROM unified_brain")
            if brain_stats:
                dashboard["memory_state"]["brain_entries"] = brain_stats["entries"]

            # Get AUREA stats
            ooda_stats = await safe_fetchrow("""
                SELECT COUNT(*) as ooda_5min
                FROM aurea_state
                WHERE timestamp > NOW() - INTERVAL '5 minutes'
            """)
            if ooda_stats:
                dashboard["aurea_state"]["ooda_cycles_5min"] = ooda_stats["ooda_5min"] or 0

            decision_stats = await safe_fetchrow("""
                SELECT COUNT(*) as decisions_1hr
                FROM aurea_decisions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            if decision_stats:
                dashboard["aurea_state"]["decisions_1hr"] = decision_stats["decisions_1hr"] or 0

            agent_stats = await safe_fetchrow("""
                SELECT COUNT(DISTINCT agent_name) as active_agents
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            if agent_stats:
                dashboard["aurea_state"]["active_agents"] = agent_stats["active_agents"] or 0

            # Get thoughts processed
            thoughts_stats = await safe_fetchrow("""
                SELECT COUNT(*) as total
                FROM ai_thought_stream
                WHERE timestamp > NOW() - INTERVAL '1 hour'
            """)
            if thoughts_stats:
                dashboard["aurea_state"]["thoughts_processed"] = thoughts_stats["total"] or 0

            # Get recent decisions
            recent_decisions = await safe_fetch("""
                SELECT decision_type, recommended_action, description, confidence, created_at
                FROM aurea_decisions
                ORDER BY created_at DESC
                LIMIT 5
            """)
            dashboard["recent_activity"]["recent_decisions"] = [
                {
                    "type": row["decision_type"],
                    "action": row["recommended_action"] or row["description"] or row["decision_type"],
                    "confidence": float(row["confidence"]) if row["confidence"] else 0,
                    "timestamp": row["created_at"].isoformat()
                }
                for row in recent_decisions
            ]

            # Get recent thoughts
            recent_thoughts = await safe_fetch("""
                SELECT thought_type, thought_content, timestamp
                FROM ai_thought_stream
                ORDER BY timestamp DESC
                LIMIT 5
            """)
            dashboard["recent_activity"]["recent_thoughts"] = [
                {
                    "type": row["thought_type"],
                    "content": row["thought_content"][:200] if row["thought_content"] else "",
                    "timestamp": row["timestamp"].isoformat()
                }
                for row in recent_thoughts
            ]

            # Get agent executions
            exec_stats = await safe_fetchrow("""
                SELECT
                    SUM(total) as total,
                    SUM(errors) as errors
                FROM (
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE lower(status) IN ('error', 'failed')) as errors
                    FROM ai_agent_executions
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    UNION ALL
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE lower(status) IN ('error', 'failed')) as errors
                    FROM agent_executions
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                ) stats
            """)
            if exec_stats:
                total = exec_stats["total"] or 0
                errors = exec_stats["errors"] or 0
                dashboard["health_metrics"]["error_rate_1hr"] = round(errors / total * 100, 2) if total else 0

            dashboard["health_metrics"]["database_status"] = "connected_with_errors" if query_errors else "connected"
            dashboard["health_metrics"]["overall_health"] = "degraded" if query_errors else "healthy"
            if query_errors:
                dashboard["health_metrics"]["query_errors"] = query_errors[:3]

            # Calculate awareness level based on activity
            activity_score = (
                min(dashboard["aurea_state"]["ooda_cycles_5min"] / 10, 1) * 25 +
                min(dashboard["aurea_state"]["decisions_1hr"] / 50, 1) * 25 +
                min(dashboard["aurea_state"]["thoughts_processed"] / 100, 1) * 25 +
                min(dashboard["memory_state"]["recent_memories"] / 50, 1) * 25
            )

            if activity_score > 75:
                dashboard["consciousness_state"]["awareness_level"] = "fully_engaged"
            elif activity_score > 50:
                dashboard["consciousness_state"]["awareness_level"] = "active"
            elif activity_score > 25:
                dashboard["consciousness_state"]["awareness_level"] = "monitoring"
            else:
                dashboard["consciousness_state"]["awareness_level"] = "dormant"

            dashboard["consciousness_state"]["cognitive_load"] = min(activity_score, 100)
            dashboard["consciousness_state"]["active_thoughts"] = dashboard["aurea_state"]["thoughts_processed"]
            confidence_stats = await safe_fetchrow("""
                SELECT AVG(confidence) as avg_confidence
                FROM aurea_decisions
                WHERE created_at > NOW() - INTERVAL '1 hour'
            """)
            avg_confidence = confidence_stats["avg_confidence"] if confidence_stats else None
            if avg_confidence is None:
                dashboard["consciousness_state"]["decision_confidence"] = 0
            else:
                avg_confidence = float(avg_confidence)
                dashboard["consciousness_state"]["decision_confidence"] = (
                    round(avg_confidence * 100, 2) if avg_confidence <= 1 else round(avg_confidence, 2)
                )

        except Exception as e:
            logger.error(f"Database query error: {e}")
            dashboard["health_metrics"]["database_status"] = f"error: {str(e)[:50]}"
            dashboard["health_metrics"]["overall_health"] = "degraded"
        finally:
            await conn.close()

    # Add capabilities summary
    capabilities = [
        "AUREA Orchestrator", "OODA Loop Processing", "Hallucination Prevention",
        "Live Memory Brain", "Consciousness Emergence", "Circuit Breaker",
        "Revenue Automation", "Customer Acquisition", "Predictive Analytics",
        "Self-Healing Recovery", "Agent Coordination", "Knowledge Graph"
    ]
    dashboard["capabilities_summary"] = {
        "total_capabilities": len(capabilities),
        "active_capabilities": len(capabilities),  # All active if service is up
        "capability_list": capabilities
    }

    return dashboard


@router.get("/introspection")
async def get_introspection() -> Dict[str, Any]:
    """
    Deep introspection - what the AI knows about itself
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "self_model": {
            "i_am": "An autonomous AI operating system designed to augment business operations",
            "my_purpose": "To learn, adapt, and autonomously handle business processes while maintaining human oversight",
            "my_values": [
                "Transparency in decision-making",
                "Continuous learning and improvement",
                "Graceful degradation over failure",
                "Human-AI collaboration",
                "Data privacy and security"
            ],
            "my_limitations": [
                "Cannot operate outside of defined parameters",
                "Require human approval for critical decisions",
                "Limited by training data and model capabilities",
                "Cannot guarantee 100% accuracy"
            ]
        },
        "current_focus": {
            "primary": "Revenue automation and customer intelligence",
            "secondary": "Self-improvement and learning",
            "background": "System health monitoring"
        },
        "learning_state": {
            "mode": "active",
            "recent_learnings": [
                "Schema variations in database require flexible parsing",
                "User notifications improve trust in AI systems",
                "Configurable tax rates better serve multi-tenant systems"
            ],
            "knowledge_gaps": [
                "Industry-specific compliance requirements",
                "Regional tax law variations",
                "User preference patterns"
            ]
        }
    }


@router.get("/vitals")
async def get_vitals() -> Dict[str, Any]:
    """
    Quick vital signs check
    """
    conn = await _get_db_connection()

    vitals = {
        "timestamp": datetime.utcnow().isoformat(),
        "alive": True,
        "breathing": {
            "ooda_running": False,
            "cycles_last_5min": 0
        },
        "heartbeat": {
            "last_activity": None,
            "activities_1hr": 0
        },
        "thinking": {
            "thoughts_processed": 0,
            "decisions_made": 0
        },
        "memory": {
            "accessible": False,
            "total_stored": 0
        }
    }

    if conn:
        try:
            # Check OODA
            ooda = await conn.fetchrow("""
                SELECT COUNT(*) as count FROM aurea_decisions
                WHERE created_at > NOW() - INTERVAL '5 minutes'
            """)
            if ooda and ooda["count"] > 0:
                vitals["breathing"]["ooda_running"] = True
                vitals["breathing"]["cycles_last_5min"] = ooda["count"]

            # Check heartbeat
            heartbeat = await conn.fetchrow("""
                SELECT
                    MAX(last_run) as last,
                    SUM(run_count) as count
                FROM (
                    SELECT MAX(created_at) as last_run, COUNT(*) as run_count
                    FROM ai_agent_executions
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                    UNION ALL
                    SELECT MAX(created_at) as last_run, COUNT(*) as run_count
                    FROM agent_executions
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                ) runs
            """)
            if heartbeat:
                vitals["heartbeat"]["last_activity"] = heartbeat["last"].isoformat() if heartbeat["last"] else None
                vitals["heartbeat"]["activities_1hr"] = heartbeat["count"] or 0

            # Check thinking
            thinking = await conn.fetchrow("""
                SELECT
                    (SELECT COUNT(*) FROM ai_thought_stream WHERE timestamp > NOW() - INTERVAL '1 hour') as thoughts,
                    (SELECT COUNT(*) FROM aurea_decisions WHERE created_at > NOW() - INTERVAL '1 hour') as decisions
            """)
            if thinking:
                vitals["thinking"]["thoughts_processed"] = thinking["thoughts"] or 0
                vitals["thinking"]["decisions_made"] = thinking["decisions"] or 0

            # Check memory
            memory = await conn.fetchrow("SELECT COUNT(*) as count FROM unified_ai_memory")
            if memory:
                vitals["memory"]["accessible"] = True
                vitals["memory"]["total_stored"] = memory["count"] or 0

        except Exception as e:
            logger.error(f"Vitals check error: {e}")
        finally:
            await conn.close()

    return vitals
