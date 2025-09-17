#!/usr/bin/env python3
"""
Automatic Agent Executor
Ensures agents are continuously working
"""

import asyncio
import httpx
import json
from datetime import datetime, timezone
import random
import psycopg2
from psycopg2.extras import RealDictCursor

class AutoExecutor:
    def __init__(self):
        self.agents_url = "https://brainops-ai-agents.onrender.com"
        self.backend_url = "https://brainops-backend-prod.onrender.com"
        self.db_config = {
            "host": "aws-0-us-east-2.pooler.supabase.com",
            "database": "postgres",
            "user": "postgres.yomagoqdmxszqtdwuhab",
            "password": "Brain0ps2O2S",
            "port": 5432
        }

    async def execute_agent_cycle(self):
        """Execute a cycle of agent tasks"""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        # Get all active agents
        cursor.execute("""
            SELECT id, name, type, capabilities
            FROM ai_agents
            WHERE status = 'active'
        """)
        agents = cursor.fetchall()
        cursor.close()
        conn.close()

        async with httpx.AsyncClient(timeout=30) as client:
            # Execute monitoring agents
            monitoring_agents = [a for a in agents if 'monitoring' in str(a['capabilities']).lower()]
            for agent in monitoring_agents[:3]:  # Top 3 monitoring agents
                try:
                    await client.post(
                        f"{self.agents_url}/agents/{agent['id']}/execute",
                        json={"action": "automated_check", "source": "auto_executor"}
                    )
                    await asyncio.sleep(2)
                except:
                    pass

            # Execute analytics agents
            analytics_agents = [a for a in agents if 'analytics' in str(a['type']).lower()]
            for agent in analytics_agents[:2]:
                try:
                    await client.post(
                        f"{self.agents_url}/agents/{agent['id']}/execute",
                        json={"action": "analyze", "source": "auto_executor"}
                    )
                    await asyncio.sleep(2)
                except:
                    pass

            # Execute workflow agents
            workflow_agents = [a for a in agents if 'workflow' in str(a['type']).lower()]
            for agent in workflow_agents[:2]:
                try:
                    await client.post(
                        f"{self.agents_url}/agents/{agent['id']}/execute",
                        json={"type": "automation", "source": "auto_executor"}
                    )
                    await asyncio.sleep(2)
                except:
                    pass

    async def run_forever(self):
        """Run executor continuously"""
        while True:
            try:
                await self.execute_agent_cycle()
                await asyncio.sleep(180)  # Run every 3 minutes
            except Exception as e:
                print(f"Cycle error: {e}")
                await asyncio.sleep(60)

executor = AutoExecutor()

if __name__ == "__main__":
    asyncio.run(executor.run_forever())