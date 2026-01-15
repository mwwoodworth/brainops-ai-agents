"""
Self-Awareness Dashboard API
============================
Comprehensive AI system self-awareness and introspection endpoints.
Provides real-time insights into the AI OS's "consciousness" state.
"""

import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/self-awareness", tags=["Self-Awareness Dashboard"])

# Database connection - use config module for consistency with Render env vars
def _build_database_url():
    """Build database URL from environment, supporting both DATABASE_URL and individual vars"""
    database_url = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL')
    if database_url:
        return database_url
    # All credentials MUST come from environment variables - no hardcoded defaults
    host = os.environ.get('DB_HOST')
    user = os.environ.get('DB_USER')
    password = os.environ.get('DB_PASSWORD')
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
async def get_self_awareness_dashboard() -> dict[str, Any]:
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
    query_errors: list[str] = []
    tenant_id = os.getenv("DEFAULT_TENANT_ID") or os.getenv("TENANT_ID")

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
            "learning_mode": "active",
            "self_state": None,
            "self_state_recorded_at": None,
            "health_score": None,
            "mood": None,
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
            async def safe_fetchrow(sql: str, *params):
                try:
                    return await conn.fetchrow(sql, *params)
                except Exception as e:
                    query_errors.append(str(e))
                    return None

            async def safe_fetch(sql: str, *params):
                try:
                    return await conn.fetch(sql, *params)
                except Exception as e:
                    query_errors.append(str(e))
                    return []

            # Get memory stats
            memory_stats_sql = """
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as with_embeddings,
                    COUNT(CASE WHEN created_at > NOW() - INTERVAL '1 hour' THEN 1 END) as recent,
                    MAX(created_at) as last_write
                FROM unified_ai_memory
            """
            memory_params: list[Any] = []
            if tenant_id:
                memory_stats_sql += " WHERE tenant_id = $1"
                memory_params = [tenant_id]
            memory_stats = await safe_fetchrow(memory_stats_sql, *memory_params)
            if memory_stats:
                dashboard["memory_state"]["total_memories"] = memory_stats["total"]
                dashboard["memory_state"]["memories_with_embeddings"] = memory_stats["with_embeddings"]
                dashboard["memory_state"]["recent_memories"] = memory_stats["recent"]
                dashboard["memory_state"]["last_memory_write"] = memory_stats["last_write"].isoformat() if memory_stats["last_write"] else None

            # Get knowledge graph stats
            kg_stats = await safe_fetchrow("SELECT COUNT(*) as nodes FROM ai_knowledge_nodes")
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

            # Fetch latest self-state snapshot
            self_state_sql = """
                SELECT content, metadata, created_at
                FROM unified_ai_memory
                WHERE memory_type = 'meta'
                  AND tags @> ARRAY['self_state']::text[]
            """
            self_state_params: list[Any] = []
            if tenant_id:
                self_state_sql += " AND tenant_id = $1"
                self_state_params = [tenant_id]
            self_state_sql += " ORDER BY created_at DESC LIMIT 1"
            self_state_row = await safe_fetchrow(self_state_sql, *self_state_params)
            if self_state_row:
                content = self_state_row.get("content") or {}
                dashboard["consciousness_state"]["self_state"] = content
                dashboard["consciousness_state"]["self_state_recorded_at"] = (
                    self_state_row["created_at"].isoformat()
                    if self_state_row.get("created_at")
                    else None
                )
                dashboard["consciousness_state"]["health_score"] = content.get("health_score")
                dashboard["consciousness_state"]["mood"] = content.get("mood")

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
async def get_introspection() -> dict[str, Any]:
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
async def get_vitals() -> dict[str, Any]:
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


# =============================================================================
# VERIFICATION-AWARE ENDPOINTS (BrainOps Total Completion Protocol)
# =============================================================================


@router.get("/verification-status")
async def get_verification_status() -> dict[str, Any]:
    """
    Get overall verification status across the memory system.

    Returns what is verified vs assumed.
    Part of BrainOps OS Total Completion Protocol.
    """
    conn = await _get_db_connection()

    if not conn:
        return {
            "error": "Database unavailable",
            "timestamp": datetime.utcnow().isoformat()
        }

    try:
        # Get verification counts
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE verification_state = 'VERIFIED') as verified,
                COUNT(*) FILTER (WHERE verification_state = 'UNVERIFIED') as unverified,
                COUNT(*) FILTER (WHERE verification_state = 'DEGRADED') as degraded,
                COUNT(*) FILTER (WHERE verification_state = 'BROKEN') as broken,
                AVG(COALESCE(confidence_score, 0)) as avg_confidence
            FROM unified_ai_memory
            WHERE (expires_at IS NULL OR expires_at > NOW())
        """)

        if not stats:
            return {"error": "No data", "timestamp": datetime.utcnow().isoformat()}

        total = stats["total"] or 1
        verified_pct = round((stats["verified"] or 0) / total * 100, 2)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall": {
                "total": stats["total"] or 0,
                "verified": stats["verified"] or 0,
                "unverified": stats["unverified"] or 0,
                "degraded": stats["degraded"] or 0,
                "broken": stats["broken"] or 0,
                "avg_confidence": float(stats["avg_confidence"] or 0),
                "verification_percentage": verified_pct
            },
            "health": "good" if verified_pct > 50 else "needs_verification",
            "recommendations": [
                "Run memory hygiene job" if stats["degraded"] else None,
                "Verify critical memories" if verified_pct < 30 else None,
                "Clean broken entries" if stats["broken"] else None
            ]
        }

    except Exception as e:
        logger.error(f"Verification status error: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    finally:
        await conn.close()


@router.get("/truth/{topic}")
async def get_truth_for_topic(topic: str) -> dict[str, Any]:
    """
    Get the current single source of truth for a topic.

    Searches for the most authoritative memory about the topic.
    Part of BrainOps OS Total Completion Protocol.
    """
    conn = await _get_db_connection()

    if not conn:
        return {
            "topic": topic,
            "found": False,
            "error": "Database unavailable"
        }

    try:
        # Search for the topic
        rows = await conn.fetch("""
            SELECT
                id::text,
                memory_type,
                object_type,
                content,
                verification_state,
                evidence_level,
                confidence_score,
                last_verified_at,
                owner,
                project
            FROM unified_ai_memory
            WHERE (search_text ILIKE $1 OR content::text ILIKE $1)
              AND (expires_at IS NULL OR expires_at > NOW())
              AND verification_state != 'BROKEN'
            ORDER BY
                CASE verification_state
                    WHEN 'VERIFIED' THEN 0
                    WHEN 'UNVERIFIED' THEN 1
                    WHEN 'DEGRADED' THEN 2
                    ELSE 3
                END,
                importance_score DESC
            LIMIT 5
        """, f"%{topic}%")

        if not rows:
            return {
                "topic": topic,
                "found": False,
                "message": "No authoritative memory found for this topic"
            }

        primary = rows[0]
        content = primary["content"]
        if isinstance(content, dict):
            summary = content.get("text", content.get("summary", str(content)[:200]))
        else:
            summary = str(content)[:200]

        return {
            "topic": topic,
            "found": True,
            "primary_truth": {
                "memory_id": primary["id"],
                "object_type": primary["object_type"],
                "content_summary": summary,
                "verification_state": primary["verification_state"],
                "evidence_level": primary["evidence_level"],
                "confidence_score": float(primary["confidence_score"] or 0),
                "last_verified_at": primary["last_verified_at"].isoformat() if primary["last_verified_at"] else None,
                "owner": primary["owner"]
            },
            "alternatives_count": len(rows) - 1
        }

    except Exception as e:
        logger.error(f"Truth lookup error: {e}")
        return {"topic": topic, "found": False, "error": str(e)}
    finally:
        await conn.close()


@router.get("/truth-backlog")
async def get_truth_backlog() -> dict[str, Any]:
    """
    Get the truth backlog: memories needing verification.

    Part of BrainOps OS Total Completion Protocol.
    """
    conn = await _get_db_connection()

    if not conn:
        return {"error": "Database unavailable"}

    try:
        # Try to use the view if it exists, otherwise query directly
        try:
            rows = await conn.fetch("""
                SELECT * FROM memory_truth_backlog LIMIT 50
            """)
        except Exception:
            # Fallback query if view doesn't exist
            rows = await conn.fetch("""
                SELECT
                    id::text,
                    memory_type,
                    object_type,
                    verification_state,
                    confidence_score,
                    owner,
                    created_at
                FROM unified_ai_memory
                WHERE verification_state != 'VERIFIED'
                   OR verification_expires_at < NOW()
                   OR confidence_score < 0.5
                ORDER BY
                    CASE verification_state
                        WHEN 'BROKEN' THEN 1
                        WHEN 'DEGRADED' THEN 2
                        WHEN 'UNVERIFIED' THEN 3
                        ELSE 4
                    END,
                    importance_score DESC
                LIMIT 50
            """)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "backlog_count": len(rows),
            "items": [dict(r) for r in rows]
        }

    except Exception as e:
        logger.error(f"Truth backlog error: {e}")
        return {"error": str(e)}
    finally:
        await conn.close()


@router.get("/capabilities-matrix")
async def get_capabilities_matrix() -> dict[str, Any]:
    """
    Get the capability matrix: what the system can do, tool by tool.

    Part of BrainOps OS Total Completion Protocol.
    """
    # Core capabilities (hardcoded as these are architectural)
    capabilities = {
        "memory_systems": {
            "unified_ai_memory": {"status": "active", "verification_state": "VERIFIED"},
            "semantic_search": {"status": "active", "verification_state": "VERIFIED"},
            "memory_hygiene": {"status": "active", "verification_state": "UNVERIFIED"},
            "enforcement_engine": {"status": "active", "verification_state": "UNVERIFIED"}
        },
        "agent_systems": {
            "agent_scheduler": {"status": "active", "verification_state": "VERIFIED"},
            "agent_executor": {"status": "active", "verification_state": "VERIFIED"},
            "agent_memory_sdk": {"status": "active", "verification_state": "UNVERIFIED"}
        },
        "ai_systems": {
            "aurea_orchestrator": {"status": "active", "verification_state": "VERIFIED"},
            "ooda_loop": {"status": "active", "verification_state": "VERIFIED"},
            "consciousness_emergence": {"status": "active", "verification_state": "UNVERIFIED"},
            "hallucination_prevention": {"status": "active", "verification_state": "UNVERIFIED"}
        },
        "revenue_systems": {
            "gumroad_webhook": {"status": "active", "verification_state": "VERIFIED"},
            "stripe_webhook": {"status": "active", "verification_state": "VERIFIED"},
            "email_automation": {"status": "active", "verification_state": "VERIFIED"},
            "lead_nurturing": {"status": "active", "verification_state": "UNVERIFIED"}
        },
        "integration_systems": {
            "mcp_bridge": {"status": "active", "verification_state": "VERIFIED"},
            "erp_webhook": {"status": "active", "verification_state": "UNVERIFIED"},
            "api_authentication": {"status": "active", "verification_state": "VERIFIED"}
        }
    }

    # Count stats
    total = 0
    verified = 0
    active = 0

    for category, items in capabilities.items():
        for name, info in items.items():
            total += 1
            if info["verification_state"] == "VERIFIED":
                verified += 1
            if info["status"] == "active":
                active += 1

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": capabilities,
        "summary": {
            "total": total,
            "verified": verified,
            "active": active,
            "verification_percentage": round(verified / max(total, 1) * 100, 2)
        }
    }
