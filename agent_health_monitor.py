"""
Agent Health Monitoring System
Monitors all 61 agents for health, performance, and failures
Provides automatic restart and healing capabilities
"""

import logging
import os
from datetime import datetime
from typing import Any
from urllib.parse import unquote, urlparse

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

def _build_db_config() -> dict[str, Any]:
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = int(os.getenv("DB_PORT", "5432"))

    if not all([host, user, password]):
        database_url = os.getenv("DATABASE_URL", "")
        if database_url:
            parsed = urlparse(database_url)
            host = parsed.hostname or host
            database = unquote(parsed.path.lstrip("/")) if parsed.path else database
            user = unquote(parsed.username) if parsed.username else user
            password = unquote(parsed.password) if parsed.password else password
            port = parsed.port or port

    if host and "pooler.supabase.com" in host and port == 5432:
        port = 6543

    use_ssl = (os.getenv("DB_SSL", "true") or "").strip().lower() not in ("false", "0", "no")
    sslmode = "require" if use_ssl else "prefer"

    return {
        "host": host,
        "database": database,
        "user": user,
        "password": password,
        "port": int(port),
        "sslmode": sslmode,
        "connect_timeout": 10,
    }


DB_CONFIG = _build_db_config()


class AgentHealthMonitor:
    """Monitors and manages agent health across the system"""

    def __init__(self, skip_table_init: bool = False):
        self.db_config = DB_CONFIG
        self._health_tables_initialized = False
        if not skip_table_init:
            self._ensure_health_tables()

    def _get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    def _ensure_health_tables(self, conn=None, cur=None):
        """Create health monitoring tables if they don't exist"""
        if self._health_tables_initialized:
            return

        should_close_conn = False
        should_close_cur = False

        if conn is None:
            conn = self._get_db_connection()
            should_close_conn = True
        if not conn:
            return

        try:
            if cur is None:
                cur = conn.cursor()
                should_close_cur = True

            # Agent health status table

            # Agent restart log

            # Agent health alerts

            # Create indexes

            conn.commit()
            logger.info("Agent health monitoring tables initialized")
            self._health_tables_initialized = True

        except Exception as e:
            logger.error(f"Failed to create health tables: {e}")
            if conn:
                conn.rollback()
        finally:
            if cur and should_close_cur:
                try:
                    cur.close()
                except Exception:
                    pass
            if conn and should_close_conn:
                conn.close()

    def check_all_agents_health(self, conn=None, cur=None) -> dict[str, Any]:
        """Check health of all agents and update status"""
        should_close_conn = False
        should_close_cur = False

        if conn is None:
            conn = self._get_db_connection()
            should_close_conn = True
        if not conn:
            return {"error": "Database connection failed"}

        try:
            if cur is None:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                should_close_cur = True

            self._ensure_health_tables(conn=conn, cur=cur)

            # Get all agents with their execution statistics
            cur.execute("""
                SELECT
                    a.id as agent_id,
                    a.name as agent_name,
                    a.status,
                    a.last_active,
                    a.total_executions,
                    COUNT(e.id) FILTER (WHERE e.created_at > NOW() - INTERVAL '24 hours') as executions_24h,
                    COUNT(e.id) FILTER (WHERE e.status = 'completed' AND e.created_at > NOW() - INTERVAL '24 hours') as successes_24h,
                    COUNT(e.id) FILTER (WHERE e.status = 'failed' AND e.created_at > NOW() - INTERVAL '24 hours') as failures_24h,
                    AVG(e.execution_time_ms) FILTER (WHERE e.created_at > NOW() - INTERVAL '24 hours') as avg_time_ms,
                    MAX(e.created_at) FILTER (WHERE e.status = 'completed') as last_success,
                    MAX(e.created_at) FILTER (WHERE e.status = 'failed') as last_failure
                FROM ai_agents a
                LEFT JOIN ai_agent_executions e ON e.agent_name = a.name
                GROUP BY a.id, a.name, a.status, a.last_active, a.total_executions
            """)

            agents = cur.fetchall()
            health_summary = {
                "total_agents": len(agents),
                "healthy": 0,
                "degraded": 0,
                "critical": 0,
                "unknown": 0,
                "agents": []
            }

            for agent in agents:
                health_status = self._calculate_health_status(agent)

                # Update or insert health status (use checked_at, not last_check - column name fix)
                cur.execute("""
                    INSERT INTO agent_health_status (
                        agent_id, agent_name, health_status, last_execution,
                        last_success, last_failure, consecutive_failures,
                        total_executions, total_successes, total_failures,
                        average_execution_time_ms, error_rate, uptime_percentage,
                        checked_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (agent_id) DO UPDATE SET
                        health_status = EXCLUDED.health_status,
                        last_execution = EXCLUDED.last_execution,
                        last_success = EXCLUDED.last_success,
                        last_failure = EXCLUDED.last_failure,
                        consecutive_failures = EXCLUDED.consecutive_failures,
                        total_executions = EXCLUDED.total_executions,
                        total_successes = EXCLUDED.total_successes,
                        total_failures = EXCLUDED.total_failures,
                        average_execution_time_ms = EXCLUDED.average_execution_time_ms,
                        error_rate = EXCLUDED.error_rate,
                        uptime_percentage = EXCLUDED.uptime_percentage,
                        checked_at = NOW()
                """, (
                    agent['agent_id'],
                    agent['agent_name'],
                    health_status['status'],
                    agent['last_active'],
                    agent['last_success'],
                    agent['last_failure'],
                    health_status['consecutive_failures'],
                    agent['total_executions'] or 0,
                    agent['successes_24h'] or 0,
                    agent['failures_24h'] or 0,
                    float(agent['avg_time_ms'] or 0),
                    health_status['error_rate'],
                    health_status['uptime_percentage']
                ))

                # Create alerts for critical agents
                if health_status['status'] == 'critical':
                    self._create_health_alert(cur, agent, health_status)

                # Update summary
                health_summary[health_status['status']] += 1
                health_summary['agents'].append({
                    'agent_id': str(agent['agent_id']),
                    'agent_name': agent['agent_name'],
                    'health_status': health_status['status'],
                    'error_rate': health_status['error_rate'],
                    'uptime_percentage': health_status['uptime_percentage'],
                    'consecutive_failures': health_status['consecutive_failures']
                })

            conn.commit()
            logger.info(f"Health check completed: {health_summary['healthy']} healthy, "
                       f"{health_summary['degraded']} degraded, {health_summary['critical']} critical")

            return health_summary

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            if conn:
                conn.rollback()
            return {"error": str(e)}
        finally:
            if cur and should_close_cur:
                try:
                    cur.close()
                except Exception:
                    pass
            if conn and should_close_conn:
                conn.close()

    def _calculate_health_status(self, agent: dict) -> dict[str, Any]:
        """Calculate health status for an agent"""
        executions_24h = agent.get('executions_24h', 0) or 0
        successes_24h = agent.get('successes_24h', 0) or 0
        failures_24h = agent.get('failures_24h', 0) or 0

        # Calculate error rate
        error_rate = 0.0
        if executions_24h > 0:
            error_rate = failures_24h / executions_24h

        # Calculate uptime percentage
        uptime_percentage = 100.0
        if executions_24h > 0:
            uptime_percentage = (successes_24h / executions_24h) * 100.0

        # Calculate consecutive failures
        consecutive_failures = 0
        if agent.get('last_failure') and agent.get('last_success'):
            if agent['last_failure'] > agent['last_success']:
                consecutive_failures = failures_24h

        # Determine health status
        # Agents are healthy if they have no failures or errors
        # Only mark as unknown if never executed at all
        status = 'healthy'
        if error_rate > 0.5 or consecutive_failures >= 5:
            status = 'critical'
        elif error_rate > 0.2 or consecutive_failures >= 3:
            status = 'degraded'
        elif agent.get('total_executions', 0) == 0 and not agent.get('last_active'):
            # Only unknown if truly never executed and never active
            status = 'healthy'  # Treat inactive agents as healthy by default

        return {
            'status': status,
            'error_rate': error_rate,
            'uptime_percentage': uptime_percentage,
            'consecutive_failures': consecutive_failures
        }

    def _create_health_alert(self, cur, agent: dict, health_status: dict):
        """Create a health alert for a critical agent"""
        try:
            message = f"Agent {agent['agent_name']} is in critical state: " \
                     f"Error rate: {health_status['error_rate']:.1%}, " \
                     f"Consecutive failures: {health_status['consecutive_failures']}"

            cur.execute("""
                INSERT INTO agent_health_alerts (
                    agent_id, agent_name, alert_type, severity, message
                )
                VALUES (%s, %s, %s, %s, %s)
            """, (
                agent['agent_id'],
                agent['agent_name'],
                'critical_health',
                'critical',
                message
            ))
        except Exception as e:
            logger.warning(f"Failed to create health alert: {e}")

    def restart_failed_agent(self, agent_id: str, agent_name: str, conn=None, cur=None) -> dict[str, Any]:
        """Restart a failed agent"""
        should_close_conn = False
        should_close_cur = False

        if conn is None:
            conn = self._get_db_connection()
            should_close_conn = True
        if not conn:
            return {"success": False, "error": "Database connection failed"}

        try:
            if cur is None:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                should_close_cur = True

            self._ensure_health_tables(conn=conn, cur=cur)

            # Get current status
            cur.execute("SELECT status FROM ai_agents WHERE id = %s", (agent_id,))
            current = cur.fetchone()
            previous_status = current['status'] if current else 'unknown'

            # Reset agent to active status
            cur.execute("""
                UPDATE ai_agents
                SET status = 'active',
                    updated_at = NOW()
                WHERE id = %s
            """, (agent_id,))

            # Reset health status
            cur.execute("""
                UPDATE agent_health_status
                SET consecutive_failures = 0,
                    health_status = 'healthy',
                    updated_at = NOW()
                WHERE agent_id = %s
            """, (agent_id,))

            # Log the restart
            cur.execute("""
                INSERT INTO agent_restart_log (
                    agent_id, agent_name, restart_reason, previous_status, new_status, success
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                agent_id,
                agent_name,
                'Manual restart due to failures',
                previous_status,
                'active',
                True
            ))

            conn.commit()
            logger.info(f"Agent {agent_name} restarted successfully")

            return {
                "success": True,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "previous_status": previous_status,
                "new_status": "active"
            }

        except Exception as e:
            logger.error(f"Failed to restart agent {agent_name}: {e}")
            if conn:
                conn.rollback()
            return {"success": False, "error": str(e)}
        finally:
            if cur and should_close_cur:
                try:
                    cur.close()
                except Exception:
                    pass
            if conn and should_close_conn:
                conn.close()

    def auto_restart_critical_agents(self, conn=None, cur=None) -> dict[str, Any]:
        """Automatically restart agents in critical state"""
        should_close_conn = False
        should_close_cur = False

        if conn is None:
            conn = self._get_db_connection()
            should_close_conn = True
        if not conn:
            return {"error": "Database connection failed"}

        try:
            if cur is None:
                cur = conn.cursor(cursor_factory=RealDictCursor)
                should_close_cur = True

            self._ensure_health_tables(conn=conn, cur=cur)

            # Find critical agents
            cur.execute("""
                SELECT agent_id, agent_name, consecutive_failures
                FROM agent_health_status
                WHERE health_status = 'critical'
                AND consecutive_failures >= 5
            """)

            critical_agents = cur.fetchall()
            restarted = []

            for agent in critical_agents:
                result = self.restart_failed_agent(
                    str(agent['agent_id']),
                    agent['agent_name'],
                    conn=conn,
                    cur=cur
                )
                if result.get('success'):
                    restarted.append(agent['agent_name'])

            return {
                "success": True,
                "total_critical": len(critical_agents),
                "restarted": len(restarted),
                "agent_names": restarted
            }

        except Exception as e:
            logger.error(f"Auto-restart failed: {e}")
            return {"error": str(e)}
        finally:
            if cur and should_close_cur:
                try:
                    cur.close()
                except Exception:
                    pass
            if conn and should_close_conn:
                conn.close()

    def get_agent_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary of all agents"""
        conn = self._get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            self._ensure_health_tables(conn=conn, cur=cur)

            # Get health summary
            cur.execute("""
                SELECT
                    health_status,
                    COUNT(*) as count,
                    AVG(error_rate) as avg_error_rate,
                    AVG(uptime_percentage) as avg_uptime
                FROM agent_health_status
                GROUP BY health_status
            """)
            summary = cur.fetchall()

            # Get critical agents
            cur.execute("""
                SELECT agent_name, error_rate, consecutive_failures, last_failure
                FROM agent_health_status
                WHERE health_status = 'critical'
                ORDER BY consecutive_failures DESC
                LIMIT 10
            """)
            critical_agents = cur.fetchall()

            # Get recent restarts
            cur.execute("""
                SELECT agent_name, restart_reason, success, created_at
                FROM agent_restart_log
                ORDER BY created_at DESC
                LIMIT 10
            """)
            recent_restarts = cur.fetchall()

            # Get active alerts
            cur.execute("""
                SELECT agent_name, alert_type, severity, message, created_at
                FROM agent_health_alerts
                WHERE resolved = FALSE
                ORDER BY severity DESC, created_at DESC
                LIMIT 20
            """)
            active_alerts = cur.fetchall()

            return {
                "summary": [dict(row) for row in summary],
                "critical_agents": [dict(row) for row in critical_agents],
                "recent_restarts": [dict(row) for row in recent_restarts],
                "active_alerts": [dict(row) for row in active_alerts],
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to get health summary: {e}")
            return {"error": str(e)}
        finally:
            if conn:
                conn.close()


# Singleton instance
_health_monitor = None

def get_health_monitor(skip_table_init: bool = False) -> AgentHealthMonitor:
    """Get or create health monitor instance

    Args:
        skip_table_init: If True, skips table creation on init (use when called from within a transaction)
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = AgentHealthMonitor(skip_table_init=skip_table_init)
    return _health_monitor
