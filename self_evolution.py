"""
Self-Evolution Module for BrainOps AI OS
The SELF-IMPROVING core that enables the AI to evolve itself.

Capabilities:
1. Self-Performance Analysis
2. Code Optimization Suggestions
3. Architecture Evolution
4. Capability Expansion
5. Knowledge Synthesis
6. Model Fine-tuning Recommendations
7. A/B Testing Framework
8. Configuration Version Control
9. Rollback Capabilities
10. Impact Measurement
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

# Internal imports
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    from ai_core import RealAICore
    from config import config
except (ImportError, RuntimeError, Exception) as e:
    logging.error(f"Failed to import dependencies or init config: {e}")
    # Partial functionality fallback
    config = None
    # Try to import RealAICore separately if config failed
    try:
        from ai_core import RealAICore
    except ImportError as exc:
        logging.warning("RealAICore import failed: %s", exc)
        RealAICore = None

logger = logging.getLogger(__name__)

class SelfEvolution:
    """
    The engine for continuous self-improvement of the AI OS.
    """

    def __init__(self, tenant_id: str = "00000000-0000-0000-0000-000000000001"):
        self.tenant_id = tenant_id
        self.ai_core = RealAICore() if RealAICore else None
        self.db_config = config.database.to_dict() if config else {}
        # Ensure password is correct if not using config object
        if not self.db_config and os.getenv('DB_PASSWORD'):
             self.db_config = {
                'host': os.getenv('DB_HOST'),
                'database': os.getenv('DB_NAME', 'postgres'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'port': int(os.getenv('DB_PORT', 5432))
            }

    def _get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def analyze_own_performance(self, lookback_hours: int = 24) -> dict[str, Any]:
        """
        Analyze the AI's own performance metrics over a time period.
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "unknown",
            "metrics": {},
            "analysis": ""
        }

        conn = self._get_db_connection()
        if not conn:
            return {"error": "No DB connection"}

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Query Execution Metrics
            query = """
                SELECT
                    COUNT(*) as total_requests,
                    AVG(execution_time_ms) as avg_latency,
                    MAX(execution_time_ms) as max_latency,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as error_count,
                    SUM(token_usage) as total_tokens
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '%s hours'
            """
            cur.execute(query, (lookback_hours,))
            metrics = cur.fetchone()

            # Query Slowest Agents
            cur.execute("""
                SELECT agent_id, AVG(execution_time_ms) as avg_time
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '%s hours'
                GROUP BY agent_id
                ORDER BY avg_time DESC
                LIMIT 3
            """, (lookback_hours,))
            slowest_agents = cur.fetchall()

            results["metrics"] = {
                "total_requests": metrics["total_requests"],
                "avg_latency_ms": float(metrics["avg_latency"] or 0),
                "error_rate": (metrics["error_count"] or 0) / (metrics["total_requests"] or 1),
                "slowest_agents": [dict(a) for a in slowest_agents]
            }

            # AI Self-Reflection
            if self.ai_core:
                analysis_prompt = f"""
                Analyze my own performance metrics:
                {json.dumps(results['metrics'], indent=2)}

                Identify 3 key areas for self-improvement.
                """
                # Use reason() for deep analysis
                reasoning_result = await self.ai_core.reason(
                    problem=analysis_prompt,
                    context=results['metrics']
                )
                results["analysis"] = reasoning_result.get("conclusion") or reasoning_result.get("reasoning")

            cur.close()
            conn.close()
            return results

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            if conn: conn.close()
            return {"error": str(e)}

    async def suggest_code_improvements(self, file_path: str) -> dict[str, Any]:
        """
        Read a code file and suggest specific improvements for performance, safety, or style.
        """
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        try:
            with open(file_path) as f:
                code_content = f.read()

            if not self.ai_core:
                return {"error": "AI Core not available"}

            prompt = f"""
            Review the following code file: {file_path}

            Code Content:
            ```python
            {code_content}
            ```

            Suggest 3 specific improvements focusing on:
            1. Performance bottlenecks (O(n) complexity, I/O blocking)
            2. Security vulnerabilities
            3. Idiomatic Python patterns

            Return JSON format: {{ "suggestions": [ {{ "line": int, "issue": str, "fix": str }} ] }}
            """

            response_text = await self.ai_core.generate(
                prompt=prompt,
                system_prompt="You are a senior Python architect optimizing code.",
                model="gpt-4-0125-preview",
                intent="review"
            )

            # Parse JSON safely
            try:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"suggestions": [], "raw_response": response_text}
            except (json.JSONDecodeError, TypeError, re.error) as exc:
                logger.debug("Failed to parse improvement JSON: %s", exc)
                return {"suggestions": [], "raw_response": response_text}

        except Exception as e:
            logger.error(f"Code improvement suggestion failed: {e}")
            return {"error": str(e)}

    async def suggest_architecture_evolution(self) -> dict[str, Any]:
        """
        Propose high-level architectural changes based on current system state.
        """
        # Use AI core reasoning with the current architecture description.
        if not self.ai_core:
            return {"error": "AI Core not available"}

        prompt = """
        Review the current BrainOps AI OS architecture (Modular Monolith with MCP integration).
        We have agents for DevOps, Reporting, Scheduling, etc.

        Propose the next evolutionary step for the architecture.
        Consider:
        1. Multi-agent coordination patterns
        2. Memory hierarchy (Hot/Cold/Vector)
        3. Autonomy levels
        """

        response_dict = await self.ai_core.reason(
            problem=prompt
        )

        return {
            "proposal": response_dict.get("conclusion", response_dict.get("reasoning")),
            "full_reasoning": response_dict.get("reasoning"),
            "timestamp": datetime.utcnow().isoformat()
        }

    def ab_test_decision(self, test_name: str, variants: list[str], distinct_id: str) -> str:
        """
        Deterministic A/B testing selection based on hashing.
        """
        if not variants:
            return None

        # Create a deterministic hash of the test name + distinct_id
        hash_input = f"{test_name}:{distinct_id}"
        hash_val = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

        # Select variant
        selected_index = hash_val % len(variants)
        selected_variant = variants[selected_index]

        # Log exposure (async fire-and-forget ideally, here synchronous DB insert for safety)
        self._log_ab_exposure(test_name, distinct_id, selected_variant)

        return selected_variant

    def _log_ab_exposure(self, test_name: str, distinct_id: str, variant: str):
        conn = self._get_db_connection()
        if not conn: return
        try:
            cur = conn.cursor()
            # Ensure table exists (idempotent)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_ab_exposures (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    test_name VARCHAR(255) NOT NULL,
                    distinct_id VARCHAR(255) NOT NULL,
                    variant VARCHAR(255) NOT NULL,
                    exposed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute(
                "INSERT INTO ai_ab_exposures (test_name, distinct_id, variant) VALUES (%s, %s, %s)",
                (test_name, distinct_id, variant)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log A/B exposure: {e}")
            if conn: conn.close()

    async def version_control_config(self, key: str, value: Any, author: str = "auto"):
        """
        Save a configuration change with version history.
        """
        conn = self._get_db_connection()
        if not conn: return
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ai_config_history (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    config_key VARCHAR(255) NOT NULL,
                    config_value JSONB NOT NULL,
                    author VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            cur.execute(
                "INSERT INTO ai_config_history (config_key, config_value, author) VALUES (%s, %s, %s)",
                (key, json.dumps(value), author)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            logger.error(f"Config VC failed: {e}")
            if conn: conn.close()

    async def rollback_config(self, key: str) -> Optional[Any]:
        """
        Revert a configuration to the previous version.
        """
        conn = self._get_db_connection()
        if not conn: return None
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            # Get the 2nd most recent entry
            cur.execute("""
                SELECT config_value FROM ai_config_history
                WHERE config_key = %s
                ORDER BY created_at DESC
                LIMIT 1 OFFSET 1
            """, (key,))
            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                previous_value = row['config_value']
                # Re-apply it as a new version to maintain history
                await self.version_control_config(key, previous_value, author="rollback_system")
                return previous_value
            return None
        except Exception as e:
            logger.error(f"Config rollback failed: {e}")
            if conn: conn.close()
            return None

    async def synthesize_knowledge(self) -> dict[str, Any]:
        """
        Synthesize disparate system events into new knowledge/insights.
        """
        if not self.ai_core:
            return {"error": "AI Core missing"}
        conn = self._get_db_connection()
        if not conn:
            return {"error": "No DB connection"}

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("""
                SELECT event_type, COUNT(*) as count
                FROM ai_events
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY event_type
                ORDER BY count DESC
                LIMIT 10
            """)
            ai_event_summary = cur.fetchall()

            cur.execute("""
                SELECT event_type, COUNT(*) as count
                FROM os_events
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY event_type
                ORDER BY count DESC
                LIMIT 10
            """)
            os_event_summary = cur.fetchall()

            cur.execute("""
                SELECT agent_name, COUNT(*) as failures
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                  AND status = 'failed'
                GROUP BY agent_name
                ORDER BY failures DESC
                LIMIT 5
            """)
            failure_summary = cur.fetchall()

            cur.execute("""
                SELECT agent_name, AVG(execution_time_ms) as avg_latency_ms
                FROM ai_agent_executions
                WHERE created_at > NOW() - INTERVAL '24 hours'
                GROUP BY agent_name
                ORDER BY avg_latency_ms DESC NULLS LAST
                LIMIT 5
            """)
            latency_summary = cur.fetchall()

            cur.close()
            conn.close()

            context = {
                "ai_events": [dict(row) for row in ai_event_summary],
                "os_events": [dict(row) for row in os_event_summary],
                "failures": [dict(row) for row in failure_summary],
                "latency": [dict(row) for row in latency_summary],
            }

            prompt = (
                "Synthesize the following system signals into a concise insight. "
                "Identify the dominant pattern, likely root cause, and the next best action.\n\n"
                f"{json.dumps(context, default=str)}"
            )

            response_dict = await self.ai_core.reason(problem=prompt, context=context)
            insight = response_dict.get("conclusion") or response_dict.get("reasoning")
            return {"insight": insight, "timestamp": datetime.utcnow().isoformat(), "signals": context}

        except Exception as e:
            logger.error("Knowledge synthesis failed: %s", e)
            if conn:
                conn.close()
            return {"error": str(e)}

    async def implement_safe_changes(self, change_proposal: dict[str, Any]) -> bool:
        """
        Execute a change with safety checks (Dry run, validation).
        """
        logger.info(f"Evaluating change proposal: {change_proposal}")

        # 1. Validation
        if change_proposal.get("risk_level") == "high":
            logger.warning("Change rejected: High risk detected.")
            return False

        # 2. Execution (SQL or MCP-driven)
        sql_changes = change_proposal.get("sql") or change_proposal.get("sql_statements")
        if sql_changes:
            statements = sql_changes if isinstance(sql_changes, list) else [sql_changes]
            conn = self._get_db_connection()
            if not conn:
                return False
            try:
                cur = conn.cursor()
                for stmt in statements:
                    cur.execute(stmt)
                if change_proposal.get("dry_run"):
                    conn.rollback()
                    logger.info("Dry run complete - changes rolled back.")
                else:
                    conn.commit()
                    logger.info("SQL changes committed.")
                cur.close()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"SQL change execution failed: {e}")
                conn.rollback()
                conn.close()
                return False

        mcp_request = change_proposal.get("mcp")
        if mcp_request:
            try:
                from mcp_integration import get_mcp_client
                mcp = get_mcp_client()
                server = mcp_request.get("server")
                tool = mcp_request.get("tool")
                params = mcp_request.get("params", {})
                if not server or not tool:
                    logger.error("MCP request missing server/tool.")
                    return False
                result = await mcp.execute_tool(server, tool, params)
                logger.info("MCP change executed: %s", result.success)
                return result.success
            except Exception as e:
                logger.error(f"MCP change execution failed: {e}")
                return False

        logger.error("No executable change proposal provided.")
        return False

    async def measure_impact(self, change_id: str, metric: str) -> dict[str, float]:
        """
        Compare a metric before and after a change.
        """
        conn = self._get_db_connection()
        if not conn:
            return {"error": "No DB connection"}

        metric_key = (metric or "").lower()
        if metric_key not in {"error_rate", "avg_latency_ms", "total_requests"}:
            return {"error": f"Unsupported metric: {metric}"}

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            change_time = None
            for table in ("ai_code_changes", "code_changes", "claude_code_changes"):
                cur.execute(f"SELECT created_at FROM {table} WHERE id = %s LIMIT 1", (change_id,))
                row = cur.fetchone()
                if row and row.get("created_at"):
                    change_time = row["created_at"]
                    break

            if not change_time:
                cur.execute("SELECT NOW() - INTERVAL '24 hours' AS change_time")
                change_time = cur.fetchone()["change_time"]

            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                    AVG(execution_time_ms) as avg_latency
                FROM ai_agent_executions
                WHERE created_at BETWEEN %s - INTERVAL '24 hours' AND %s
            """, (change_time, change_time))
            before = cur.fetchone()

            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failures,
                    AVG(execution_time_ms) as avg_latency
                FROM ai_agent_executions
                WHERE created_at BETWEEN %s AND %s + INTERVAL '24 hours'
            """, (change_time, change_time))
            after = cur.fetchone()

            cur.close()
            conn.close()

            def metric_value(row):
                if metric_key == "total_requests":
                    return float(row.get("total") or 0)
                if metric_key == "avg_latency_ms":
                    return float(row.get("avg_latency") or 0)
                failures = float(row.get("failures") or 0)
                total = float(row.get("total") or 0) or 1.0
                return failures / total

            return {
                "before": metric_value(before),
                "after": metric_value(after),
            }

        except Exception as e:
            logger.error(f"Impact measurement failed: {e}")
            if conn:
                conn.close()
            return {"error": str(e)}

if __name__ == "__main__":
    # Quick test if run directly
    import asyncio

    async def main():
        se = SelfEvolution()
        print("Analyzing performance...")
        try:
            perf = await se.analyze_own_performance()
            print(json.dumps(perf, indent=2, default=str))
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())
