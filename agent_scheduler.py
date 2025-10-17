"""
AI Agent Scheduler - Automatic Execution System
Schedules and executes AI agents based on configured intervals
"""

import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import json

logger = logging.getLogger(__name__)

class AgentScheduler:
    """Manages automatic execution of AI agents"""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.scheduler = AsyncIOScheduler()
        self.registered_jobs = {}

    def get_db_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None

    async def execute_agent(self, agent_id: str, agent_name: str):
        """Execute a scheduled agent"""
        try:
            logger.info(f"Executing scheduled agent: {agent_name} ({agent_id})")

            conn = self.get_db_connection()
            if not conn:
                logger.error("Database connection failed, cannot execute agent")
                return

            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Record execution start
            execution_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO ai_agent_executions
                (id, agent_name, status)
                VALUES (%s, %s, %s)
            """, (execution_id, agent_name, 'running'))
            conn.commit()

            # Get agent configuration
            cur.execute("SELECT * FROM ai_agents WHERE id = %s", (agent_id,))
            agent = cur.fetchone()

            if not agent:
                logger.error(f"Agent {agent_id} not found")
                cur.execute("""
                    UPDATE ai_agent_executions
                    SET status = %s, error_message = %s
                    WHERE id = %s
                """, ('failed', 'Agent not found', execution_id))
                conn.commit()
                cur.close()
                conn.close()
                return

            # Execute based on agent type
            start_time = datetime.utcnow()
            result = await self._execute_by_type(agent, cur, conn)
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Record execution completion
            cur.execute("""
                UPDATE ai_agent_executions
                SET status = %s, output_data = %s, execution_time_ms = %s
                WHERE id = %s
            """, ('completed', json.dumps(result), execution_time_ms, execution_id))

            # Update agent statistics
            cur.execute("""
                UPDATE ai_agents
                SET total_executions = total_executions + 1,
                    last_activation = %s,
                    last_active = %s
                WHERE id = %s
            """, (datetime.utcnow(), datetime.utcnow(), agent_id))

            # Update schedule next execution
            cur.execute("""
                UPDATE agent_schedules
                SET last_execution = %s,
                    next_execution = %s
                WHERE agent_id = %s AND enabled = true
            """, (datetime.utcnow(), datetime.utcnow() + timedelta(minutes=agent.get('frequency_minutes', 60)), agent_id))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"Agent {agent_name} executed successfully")

        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")

            # Record failure
            try:
                conn = self.get_db_connection()
                if conn:
                    cur = conn.cursor()
                    cur.execute("""
                        UPDATE ai_agent_executions
                        SET status = %s, error_message = %s
                        WHERE id = %s
                    """, ('failed', str(e), execution_id))
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as log_error:
                logger.error(f"Failed to log error: {log_error}")

    async def _execute_by_type(self, agent: Dict, cur, conn) -> Dict:
        """Execute agent based on its type"""
        agent_type = agent.get('type', '').lower()
        agent_name = agent.get('name', 'Unknown')

        logger.info(f"Executing {agent_type} agent: {agent_name}")

        # Revenue-generating agents
        if 'revenue' in agent_type or agent_name == 'RevenueOptimizer':
            return await self._execute_revenue_agent(agent, cur, conn)

        # Lead generation agents
        elif 'lead' in agent_type or agent_name in ['LeadGenerationAgent', 'LeadScorer']:
            return await self._execute_lead_agent(agent, cur, conn)

        # Customer intelligence agents
        elif 'customer' in agent_type or agent_name == 'CustomerIntelligence':
            return await self._execute_customer_agent(agent, cur, conn)

        # Analytics agents
        elif agent_type == 'analytics':
            return await self._execute_analytics_agent(agent, cur, conn)

        # Default: log and continue
        else:
            logger.info(f"No specific handler for agent type: {agent_type}")
            return {"status": "executed", "type": agent_type}

    async def _execute_revenue_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute revenue optimization agent"""
        logger.info(f"Running revenue optimization for agent: {agent['name']}")

        # Get revenue opportunities
        cur.execute("""
            SELECT COUNT(*) as total_jobs,
                   COUNT(*) FILTER (WHERE status = 'pending') as pending_jobs,
                   COUNT(*) FILTER (WHERE status = 'in_progress') as active_jobs
            FROM jobs
        """)
        stats = cur.fetchone()

        # Get pending estimates
        cur.execute("""
            SELECT COUNT(*) as pending_estimates,
                   SUM(total_amount) as potential_revenue
            FROM estimates
            WHERE status = 'pending'
        """)
        estimates = cur.fetchone()

        return {
            "agent": agent['name'],
            "jobs_analyzed": stats['total_jobs'],
            "pending_jobs": stats['pending_jobs'],
            "active_jobs": stats['active_jobs'],
            "pending_estimates": estimates['pending_estimates'],
            "potential_revenue": float(estimates['potential_revenue'] or 0),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _execute_lead_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute lead generation/scoring agent"""
        logger.info(f"Running lead analysis for agent: {agent['name']}")

        # Get lead statistics
        cur.execute("""
            SELECT COUNT(*) as total_customers,
                   COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '7 days') as new_this_week,
                   COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '30 days') as new_this_month
            FROM customers
        """)
        stats = cur.fetchone()

        return {
            "agent": agent['name'],
            "total_customers": stats['total_customers'],
            "new_this_week": stats['new_this_week'],
            "new_this_month": stats['new_this_month'],
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _execute_customer_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute customer intelligence agent"""
        logger.info(f"Running customer intelligence for agent: {agent['name']}")

        # Get customer insights
        cur.execute("""
            SELECT
                COUNT(DISTINCT c.id) as total_customers,
                COUNT(DISTINCT j.id) as total_jobs,
                COUNT(DISTINCT i.id) as total_invoices,
                SUM(i.total_amount) as total_revenue
            FROM customers c
            LEFT JOIN jobs j ON j.customer_id = c.id
            LEFT JOIN invoices i ON i.job_id = j.id
        """)
        stats = cur.fetchone()

        return {
            "agent": agent['name'],
            "total_customers": stats['total_customers'],
            "total_jobs": stats['total_jobs'],
            "total_invoices": stats['total_invoices'],
            "total_revenue": float(stats['total_revenue'] or 0),
            "timestamp": datetime.utcnow().isoformat()
        }

    async def _execute_analytics_agent(self, agent: Dict, cur, conn) -> Dict:
        """Execute analytics agent"""
        logger.info(f"Running analytics for agent: {agent['name']}")

        # Get general analytics
        cur.execute("""
            SELECT
                (SELECT COUNT(*) FROM customers) as total_customers,
                (SELECT COUNT(*) FROM jobs) as total_jobs,
                (SELECT COUNT(*) FROM invoices) as total_invoices,
                (SELECT COUNT(*) FROM estimates) as total_estimates
        """)
        stats = cur.fetchone()

        return {
            "agent": agent['name'],
            "analytics": {
                "customers": stats['total_customers'],
                "jobs": stats['total_jobs'],
                "invoices": stats['total_invoices'],
                "estimates": stats['total_estimates']
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    def load_schedules_from_db(self):
        """Load agent schedules from database"""
        try:
            conn = self.get_db_connection()
            if not conn:
                logger.error("Cannot load schedules: DB connection failed")
                return

            cur = conn.cursor(cursor_factory=RealDictCursor)

            # Get all enabled schedules
            cur.execute("""
                SELECT s.*, a.name as agent_name, a.type as agent_type
                FROM agent_schedules s
                JOIN ai_agents a ON a.id = s.agent_id
                WHERE s.enabled = true
            """)

            schedules = cur.fetchall()
            logger.info(f"Found {len(schedules)} enabled agent schedules")

            for schedule in schedules:
                self.add_schedule(
                    agent_id=schedule['agent_id'],
                    agent_name=schedule['agent_name'],
                    frequency_minutes=schedule['frequency_minutes'] or 60,
                    schedule_id=schedule['id']
                )

            cur.close()
            conn.close()

        except Exception as e:
            logger.error(f"Error loading schedules: {e}")

    def add_schedule(self, agent_id: str, agent_name: str, frequency_minutes: int, schedule_id: str = None):
        """Add an agent to the scheduler"""
        try:
            job_id = schedule_id or str(uuid.uuid4())

            # Add job to scheduler
            self.scheduler.add_job(
                func=self.execute_agent,
                trigger=IntervalTrigger(minutes=frequency_minutes),
                args=[agent_id, agent_name],
                id=job_id,
                name=f"Agent: {agent_name}",
                replace_existing=True
            )

            self.registered_jobs[job_id] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "frequency_minutes": frequency_minutes,
                "added_at": datetime.utcnow().isoformat()
            }

            logger.info(f"Scheduled agent {agent_name} to run every {frequency_minutes} minutes")

        except Exception as e:
            logger.error(f"Error adding schedule for {agent_name}: {e}")

    def remove_schedule(self, schedule_id: str):
        """Remove an agent from the scheduler"""
        try:
            self.scheduler.remove_job(schedule_id)
            if schedule_id in self.registered_jobs:
                del self.registered_jobs[schedule_id]
            logger.info(f"Removed schedule: {schedule_id}")
        except Exception as e:
            logger.error(f"Error removing schedule {schedule_id}: {e}")

    def start(self):
        """Start the scheduler"""
        try:
            # Load schedules from database
            self.load_schedules_from_db()

            # Start scheduler
            self.scheduler.start()
            logger.info(f"Agent scheduler started with {len(self.registered_jobs)} jobs")

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")

    def shutdown(self):
        """Shutdown the scheduler"""
        try:
            self.scheduler.shutdown()
            logger.info("Agent scheduler shutdown")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")

    def get_status(self) -> Dict:
        """Get scheduler status"""
        return {
            "running": self.scheduler.running,
            "total_jobs": len(self.registered_jobs),
            "jobs": [
                {
                    "id": job_id,
                    "agent": job_info["agent_name"],
                    "frequency_minutes": job_info["frequency_minutes"]
                }
                for job_id, job_info in self.registered_jobs.items()
            ]
        }


# Create execution tracking table if needed
def create_execution_table(db_config: Dict):
    """Create ai_agent_executions table if it doesn't exist"""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()

        # Table already exists with different schema - just ensure indexes
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_executions_agent ON ai_agent_executions(agent_name);
            CREATE INDEX IF NOT EXISTS idx_agent_executions_created ON ai_agent_executions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON ai_agent_executions(status);
        """)

        conn.commit()
        cur.close()
        conn.close()

        logger.info("Execution tracking table verified/created")

    except Exception as e:
        logger.error(f"Error creating execution table: {e}")


# Example usage
if __name__ == "__main__":
    # Configure database
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
        'password': os.getenv('DB_PASSWORD', 'Brain0ps2O2S'),
        'port': int(os.getenv('DB_PORT', 5432))
    }

    # Create execution table
    create_execution_table(DB_CONFIG)

    # Initialize scheduler
    scheduler = AgentScheduler(DB_CONFIG)

    # Start scheduler
    scheduler.start()

    # Keep running
    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
