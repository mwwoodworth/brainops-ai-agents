"""
System Improvement Agent
AI agent for analyzing system performance and suggesting improvements.
Uses OpenAI for real analysis and persists results to database.
"""

import os
import json
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy OpenAI client initialization
_openai_client = None

def get_openai_client():
    """Get or create OpenAI client"""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                _openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    return _openai_client

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}

class SystemImprovementAgent:
    """AI-powered system improvement analysis agent"""

    def __init__(self, tenant_id: str):
        if not tenant_id:
            raise ValueError("tenant_id is required for SystemImprovementAgent")
        self.tenant_id = tenant_id
        self.agent_type = "system_improvement"

    def _get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def analyze_performance(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze system performance metrics using real data and AI"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "metrics_analyzed": len(metrics),
                "performance_data": {}
            }

            # Gather real system metrics from database
            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)

                    # Get agent execution performance
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_executions,
                            AVG(execution_time_ms) as avg_execution_ms,
                            MAX(execution_time_ms) as max_execution_ms,
                            COUNT(*) FILTER (WHERE status = 'failed') as failed_count
                        FROM ai_agent_executions
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                    """)
                    exec_stats = cur.fetchone()

                    # Get task processing stats
                    cur.execute("""
                        SELECT
                            COUNT(*) as total_tasks,
                            COUNT(*) FILTER (WHERE status = 'completed') as completed,
                            COUNT(*) FILTER (WHERE status = 'in_progress') as in_progress,
                            COUNT(*) FILTER (WHERE status = 'pending') as pending
                        FROM ai_autonomous_tasks
                    """)
                    task_stats = cur.fetchone()

                    results["performance_data"] = {
                        "executions_24h": exec_stats['total_executions'] if exec_stats else 0,
                        "avg_execution_ms": round(exec_stats['avg_execution_ms'] or 0, 2) if exec_stats else 0,
                        "max_execution_ms": exec_stats['max_execution_ms'] if exec_stats else 0,
                        "failed_executions": exec_stats['failed_count'] if exec_stats else 0,
                        "total_tasks": task_stats['total_tasks'] if task_stats else 0,
                        "completed_tasks": task_stats['completed'] if task_stats else 0,
                        "in_progress_tasks": task_stats['in_progress'] if task_stats else 0
                    }

                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch performance stats: {e}")
                    if conn:
                        conn.close()

            # Combine with provided metrics
            all_metrics = {**results["performance_data"], "provided_metrics": metrics}

            # Use AI for analysis
            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Analyze this system performance data:
{json.dumps(all_metrics, indent=2)}

Provide comprehensive performance analysis:
1. Current system health status
2. Performance bottlenecks
3. Optimization opportunities
4. Scalability assessment

Respond with JSON only:
{{
    "status": "healthy/degraded/critical",
    "health_score": 0-100,
    "bottlenecks": [
        {{"component": "name", "issue": "description", "severity": "high/medium/low", "impact": "description"}}
    ],
    "recommendations": [
        {{"area": "component", "action": "what to do", "priority": "high/medium/low", "effort": "low/medium/high"}}
    ],
    "quick_wins": ["quick improvement 1", "quick improvement 2"],
    "scalability_assessment": "good/fair/poor",
    "confidence_score": 0-100
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI analysis failed: {e}")
                    results["status"] = "healthy"
                    results["bottlenecks"] = []
                    results["recommendations"] = []

            await self._save_analysis("performance_analysis", results)
            return results

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {"error": str(e), "status": "error"}

    async def suggest_optimizations(self, component: str) -> Dict[str, Any]:
        """Suggest optimizations for a specific component using AI"""
        try:
            results = {
                "component": component,
                "analyzed_at": datetime.utcnow().isoformat(),
                "optimizations": []
            }

            client = get_openai_client()
            if client:
                try:
                    prompt = f"""Suggest optimizations for this system component: {component}

Consider:
1. Performance improvements
2. Code quality enhancements
3. Security hardening
4. Maintainability improvements
5. Cost optimization

Respond with JSON only:
{{
    "component": "{component}",
    "optimizations": [
        {{"type": "performance/security/quality/cost", "suggestion": "detailed suggestion", "impact": "high/medium/low", "implementation_effort": "low/medium/high"}}
    ],
    "best_practices": ["practice1", "practice2"],
    "anti_patterns_detected": ["antipattern1"],
    "modernization_suggestions": ["suggestion1"],
    "priority_ranking": ["highest priority action", "second priority"]
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI optimization analysis failed: {e}")
                    results["optimizations"] = [
                        {"type": "general", "suggestion": f"Review {component} for optimization opportunities"}
                    ]

            await self._save_analysis("component_optimization", results)
            return results

        except Exception as e:
            logger.error(f"Optimization suggestion failed: {e}")
            return {"error": str(e), "component": component}

    async def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns in the system"""
        try:
            results = {
                "analyzed_at": datetime.utcnow().isoformat(),
                "error_patterns": []
            }

            conn = self._get_db_connection()
            if conn:
                try:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    cur.execute("""
                        SELECT
                            error_message,
                            COUNT(*) as occurrence_count,
                            MAX(created_at) as last_occurrence
                        FROM ai_agent_executions
                        WHERE status = 'failed'
                        AND created_at > NOW() - INTERVAL '7 days'
                        GROUP BY error_message
                        ORDER BY occurrence_count DESC
                        LIMIT 10
                    """)
                    errors = cur.fetchall()
                    results["error_patterns"] = [dict(e) for e in errors] if errors else []
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Could not fetch error patterns: {e}")
                    if conn:
                        conn.close()

            # AI analysis of error patterns
            client = get_openai_client()
            if client and results["error_patterns"]:
                try:
                    prompt = f"""Analyze these system error patterns:
{json.dumps(results['error_patterns'], indent=2, default=str)}

Provide error analysis and remediation recommendations:
1. Root cause analysis
2. Pattern identification
3. Fix recommendations
4. Prevention strategies

Respond with JSON only:
{{
    "critical_errors": ["error requiring immediate attention"],
    "root_causes": [{{"error": "pattern", "likely_cause": "explanation", "fix": "solution"}}],
    "systemic_issues": ["issue1", "issue2"],
    "prevention_recommendations": ["rec1", "rec2"],
    "monitoring_suggestions": ["what to monitor"]
}}"""

                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=800
                    )
                    ai_analysis = json.loads(response.choices[0].message.content)
                    results.update(ai_analysis)
                except Exception as e:
                    logger.warning(f"AI error analysis failed: {e}")

            await self._save_analysis("error_pattern_analysis", results)
            return results

        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")
            return {"error": str(e)}

    async def _save_analysis(self, analysis_type: str, results: Dict[str, Any]):
        """Save analysis results to database"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO ai_system_analyses (tenant_id, analysis_type, results, analyzed_at)
                VALUES (%s, %s, %s, NOW())
            """, (self.tenant_id, analysis_type, json.dumps(results, default=str)))
            conn.commit()
            cur.close()
            conn.close()
            logger.info(f"Saved {analysis_type} analysis for tenant {self.tenant_id}")
        except Exception as e:
            logger.warning(f"Failed to save analysis (table may not exist): {e}")
            if conn:
                conn.close()

    # ==========================================================================
    # SELF-IMPROVEMENT LOOP - The system uses its own tools to improve itself
    # ==========================================================================

    async def execute_auto_improvements(self) -> Dict[str, Any]:
        """
        FORCE MULTIPLIER: The AI OS uses its own MCP Bridge to implement improvements.

        This creates a feedback loop where:
        1. System analyzes its own performance
        2. Identifies actionable improvements
        3. Executes those improvements via MCP Bridge
        4. Logs results for next analysis cycle
        """
        import aiohttp

        MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", "https://brainops-mcp-bridge.onrender.com")
        MCP_API_KEY = os.getenv("BRAINOPS_API_KEY", "brainops_prod_key_2025")

        results = {
            "executed_at": datetime.utcnow().isoformat(),
            "improvements_identified": 0,
            "improvements_executed": 0,
            "improvements_succeeded": 0,
            "actions": []
        }

        try:
            # Step 1: Run performance analysis
            logger.info("🔍 Step 1: Analyzing system performance...")
            analysis = await self.analyze_performance([])

            # Step 2: Get recommendations
            recommendations = analysis.get("recommendations", [])
            results["improvements_identified"] = len(recommendations)
            logger.info(f"📊 Found {len(recommendations)} improvement recommendations")

            # Step 3: Map recommendations to MCP actions
            actionable_improvements = self._map_recommendations_to_mcp_actions(recommendations)

            # Step 4: Execute actionable improvements via MCP Bridge
            async with aiohttp.ClientSession() as session:
                for improvement in actionable_improvements:
                    logger.info(f"🔧 Executing: {improvement['description']}")
                    results["improvements_executed"] += 1

                    try:
                        async with session.post(
                            f"{MCP_BRIDGE_URL}/mcp/execute",
                            headers={
                                "X-API-Key": MCP_API_KEY,
                                "Content-Type": "application/json"
                            },
                            json={
                                "server": improvement["mcp_server"],
                                "tool": improvement["mcp_tool"],
                                "params": improvement["params"]
                            },
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            if resp.status == 200:
                                mcp_result = await resp.json()
                                results["improvements_succeeded"] += 1
                                results["actions"].append({
                                    "action": improvement["description"],
                                    "status": "success",
                                    "result": mcp_result
                                })
                                logger.info(f"✅ Success: {improvement['description']}")
                            else:
                                error_text = await resp.text()
                                results["actions"].append({
                                    "action": improvement["description"],
                                    "status": "failed",
                                    "error": error_text
                                })
                                logger.warning(f"❌ Failed: {improvement['description']} - {error_text}")
                    except Exception as e:
                        results["actions"].append({
                            "action": improvement["description"],
                            "status": "error",
                            "error": str(e)
                        })
                        logger.error(f"Error executing {improvement['description']}: {e}")

            # Step 5: Save results to database
            await self._save_analysis("auto_improvement_execution", results)

            # Step 6: Log to remediation_history for audit trail
            self._log_improvement_to_db(results)

            logger.info(f"🎯 Self-improvement complete: {results['improvements_succeeded']}/{results['improvements_executed']} succeeded")
            return results

        except Exception as e:
            logger.error(f"Auto-improvement failed: {e}")
            return {"error": str(e), **results}

    def _map_recommendations_to_mcp_actions(self, recommendations: List[Dict]) -> List[Dict]:
        """
        Map AI-generated recommendations to concrete MCP Bridge actions.

        This is the intelligence layer that translates high-level
        recommendations into actionable infrastructure commands.
        """
        actionable = []

        # Mapping rules: recommendation patterns → MCP actions
        action_mappings = {
            # Performance recommendations
            "cache": {
                "mcp_server": "supabase",
                "mcp_tool": "execute_sql",
                "params_template": {"query": "DISCARD ALL; -- Clear session cache"}
            },
            "restart": {
                "mcp_server": "render",
                "mcp_tool": "restart_service",
                "params_template": {"service_id": "srv-d0ulv1idbo4c73apd4t0"}
            },
            "scale": {
                "mcp_server": "render",
                "mcp_tool": "scale_service",
                "params_template": {"num_instances": 2}
            },
            "database": {
                "mcp_server": "supabase",
                "mcp_tool": "execute_sql",
                "params_template": {"query": "ANALYZE;"}  # Update statistics
            },
            "index": {
                "mcp_server": "supabase",
                "mcp_tool": "execute_sql",
                "params_template": {"query": "REINDEX DATABASE postgres;"}
            },
            "vacuum": {
                "mcp_server": "supabase",
                "mcp_tool": "execute_sql",
                "params_template": {"query": "VACUUM ANALYZE;"}
            },
            "connection": {
                "mcp_server": "supabase",
                "mcp_tool": "execute_sql",
                "params_template": {
                    "query": "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND query_start < NOW() - INTERVAL '10 minutes'"
                }
            },
            # GitHub-related improvements
            "deploy": {
                "mcp_server": "github",
                "mcp_tool": "create_workflow_dispatch",
                "params_template": {"workflow_file": "deploy.yml"}
            }
        }

        for rec in recommendations:
            action = rec.get("action", "").lower()
            area = rec.get("area", "").lower()
            priority = rec.get("priority", "medium")

            # Only auto-execute high/medium priority, low-effort improvements
            if priority not in ["high", "medium"] or rec.get("effort") == "high":
                continue

            # Find matching action
            for keyword, mapping in action_mappings.items():
                if keyword in action or keyword in area:
                    actionable.append({
                        "description": f"{rec.get('area', 'system')}: {action[:50]}",
                        "mcp_server": mapping["mcp_server"],
                        "mcp_tool": mapping["mcp_tool"],
                        "params": mapping["params_template"].copy(),
                        "priority": priority,
                        "original_recommendation": rec
                    })
                    break  # Only one action per recommendation

        # Limit to 5 actions per cycle to avoid runaway automation
        return actionable[:5]

    def _log_improvement_to_db(self, results: Dict[str, Any]):
        """Log improvement execution to database for audit trail"""
        conn = self._get_db_connection()
        if not conn:
            return
        try:
            cur = conn.cursor()
            for action in results.get("actions", []):
                cur.execute("""
                    INSERT INTO remediation_history
                    (incident_type, component, action_taken, success, recovery_time_seconds)
                    VALUES (%s, %s, %s, %s, 0)
                """, (
                    "self_improvement",
                    "system_improvement_agent",
                    action.get("action", "unknown")[:200],
                    action.get("status") == "success"
                ))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not log improvement to DB: {e}")
            if conn:
                conn.close()
