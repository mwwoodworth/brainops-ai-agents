#!/usr/bin/env python3
"""
Database-Driven Scheduled Executor
Executes agents based on database schedules to prevent duplicates
"""

import asyncio
import httpx
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class ScheduledExecutor:
    def __init__(self):
        self.db_config = {
            "host": "aws-0-us-east-2.pooler.supabase.com",
            "database": "postgres",
            "user": "postgres.yomagoqdmxszqtdwuhab",
            "password": "Brain0ps2O2S",
            "port": 5432
        }
        self.agents_url = "https://brainops-ai-agents.onrender.com"

    def get_next_agent(self):
        """Get next scheduled agent from database"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM get_next_scheduled_agent()")
        result = cursor.fetchone()

        cursor.close()
        conn.close()
        return result

    def update_schedule(self, schedule_id):
        """Update schedule after execution"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        cursor.execute("SELECT update_agent_schedule(%s)", (schedule_id,))
        conn.commit()

        cursor.close()
        conn.close()

    async def execute_scheduled_agent(self):
        """Execute next scheduled agent"""
        agent_info = self.get_next_agent()

        if agent_info:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        f"{self.agents_url}/agents/{agent_info['agent_id']}/execute",
                        json=agent_info['task_config']
                    )

                    if response.status_code == 200:
                        # Update schedule on success
                        self.update_schedule(agent_info['schedule_id'])
                        logger.info(f"Executed agent {agent_info['agent_id']}")
                    else:
                        logger.error(f"Failed to execute agent: {response.status_code}")

            except Exception as e:
                logger.error(f"Execution error: {e}")

    async def run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Starting Scheduled Executor")

        while True:
            try:
                # Check for and execute scheduled agents
                await self.execute_scheduled_agent()

                # Short sleep to check frequently
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(30)

# Global scheduler
scheduler = ScheduledExecutor()

if __name__ == "__main__":
    asyncio.run(scheduler.run_scheduler())