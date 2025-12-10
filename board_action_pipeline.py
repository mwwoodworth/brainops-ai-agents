#!/usr/bin/env python3
"""
Board-to-Action Pipeline
Bridges AI Board of Directors decisions to LangGraph workflow execution.

This module:
1. Monitors approved board decisions
2. Converts decisions into executable LangGraph tasks
3. Triggers appropriate agent workflows
4. Reports execution results back to the board
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psycopg2
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BoardActionPipeline')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'aws-0-us-east-2.pooler.supabase.com'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER', 'postgres.yomagoqdmxszqtdwuhab'),
    'password': os.getenv('DB_PASSWORD', 'REDACTED_SUPABASE_DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}


class ActionStatus(Enum):
    """Status of board-initiated actions"""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class DecisionCategory(Enum):
    """Categories of board decisions that can trigger actions"""
    STRATEGIC = "strategic"
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    MARKETING = "marketing"
    TECHNICAL = "technical"
    EMERGENCY = "emergency"


@dataclass
class BoardAction:
    """An action derived from a board decision"""
    action_id: str
    decision_id: str
    category: DecisionCategory
    title: str
    description: str
    workflow_type: str
    workflow_params: Dict[str, Any]
    priority: int
    status: ActionStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    result: Optional[Dict] = None


class BoardActionPipeline:
    """
    Pipeline that converts board decisions into executable workflows.

    The Board of Directors makes strategic decisions, and this pipeline
    ensures those decisions are automatically executed through the
    appropriate LangGraph workflows.
    """

    def __init__(self):
        self.conn = None
        self._workflow_mapping = self._build_workflow_mapping()

    def _get_connection(self):
        """Get database connection"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(**DB_CONFIG)
        return self.conn

    def _build_workflow_mapping(self) -> Dict[str, Dict]:
        """
        Map decision categories to appropriate LangGraph workflows.
        This defines what workflow to run for each type of board decision.
        """
        return {
            "strategic": {
                "workflow_type": "strategic_planning",
                "agents": ["ceo_agent", "strategy_agent"],
                "requires_approval": True,
                "timeout_hours": 24
            },
            "financial": {
                "workflow_type": "financial_automation",
                "agents": ["cfo_agent", "financial_agent"],
                "requires_approval": True,
                "timeout_hours": 12
            },
            "operational": {
                "workflow_type": "operations_optimization",
                "agents": ["coo_agent", "operations_agent"],
                "requires_approval": False,
                "timeout_hours": 6
            },
            "marketing": {
                "workflow_type": "marketing_campaign",
                "agents": ["cmo_agent", "marketing_agent", "seo_agent"],
                "requires_approval": True,
                "timeout_hours": 48
            },
            "technical": {
                "workflow_type": "technical_implementation",
                "agents": ["cto_agent", "deployment_agent", "code_agent"],
                "requires_approval": True,
                "timeout_hours": 8
            },
            "emergency": {
                "workflow_type": "emergency_response",
                "agents": ["ceo_agent", "operations_agent", "deployment_agent"],
                "requires_approval": False,  # Emergency = immediate action
                "timeout_hours": 1
            }
        }

    async def initialize(self):
        """Initialize the pipeline and create necessary tables"""
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS board_action_queue (
                    id SERIAL PRIMARY KEY,
                    action_id UUID UNIQUE DEFAULT gen_random_uuid(),
                    decision_id VARCHAR(255) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    description TEXT,
                    workflow_type VARCHAR(100) NOT NULL,
                    workflow_params JSONB DEFAULT '{}'::jsonb,
                    priority INT DEFAULT 5,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    queued_at TIMESTAMP,
                    executed_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    result JSONB,
                    error_message TEXT,
                    retry_count INT DEFAULT 0,
                    max_retries INT DEFAULT 3
                );

                CREATE INDEX IF NOT EXISTS idx_board_actions_status
                ON board_action_queue(status);

                CREATE INDEX IF NOT EXISTS idx_board_actions_priority
                ON board_action_queue(priority DESC, created_at ASC);

                CREATE INDEX IF NOT EXISTS idx_board_actions_decision
                ON board_action_queue(decision_id);
            """)

            conn.commit()
            logger.info("Board Action Pipeline initialized successfully")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
        finally:
            cur.close()

    async def poll_approved_decisions(self) -> List[Dict]:
        """
        Poll for approved board decisions that haven't been converted to actions yet.
        """
        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Find approved decisions without corresponding actions
            cur.execute("""
                SELECT
                    bd.id as decision_id,
                    bd.decision,
                    bd.implementation_plan,
                    bd.consensus_level,
                    bd.decided_at,
                    bp.proposal_type,
                    bp.title,
                    bp.description,
                    bp.required_resources,
                    bp.urgency
                FROM ai_board_decisions bd
                LEFT JOIN ai_board_proposals bp ON bd.proposal_id = bp.id
                WHERE bd.decision = 'approved'
                AND bd.id NOT IN (
                    SELECT DISTINCT decision_id
                    FROM board_action_queue
                    WHERE decision_id IS NOT NULL
                )
                AND bd.decided_at > NOW() - INTERVAL '7 days'
                ORDER BY bp.urgency DESC, bd.decided_at ASC
                LIMIT 10
            """)

            decisions = cur.fetchall()
            return [dict(d) for d in decisions]

        except Exception as e:
            logger.error(f"Error polling decisions: {e}")
            return []
        finally:
            cur.close()

    async def convert_decision_to_action(self, decision: Dict) -> Optional[BoardAction]:
        """
        Convert an approved board decision into an executable action.
        """
        try:
            category = decision.get('proposal_type', 'operational')
            workflow_config = self._workflow_mapping.get(category, self._workflow_mapping['operational'])

            # Build workflow parameters from decision data
            workflow_params = {
                "decision_context": {
                    "title": decision.get('title', 'Board Action'),
                    "description": decision.get('description', ''),
                    "implementation_plan": decision.get('implementation_plan', {}),
                    "resources": decision.get('required_resources', {}),
                    "consensus_level": decision.get('consensus_level', 0),
                },
                "agents": workflow_config['agents'],
                "timeout_hours": workflow_config['timeout_hours'],
                "requires_human_approval": workflow_config['requires_approval']
            }

            # Create action record
            action = BoardAction(
                action_id="",  # Will be set by DB
                decision_id=str(decision['decision_id']),
                category=DecisionCategory(category) if category in DecisionCategory.__members__.values() else DecisionCategory.OPERATIONAL,
                title=decision.get('title', 'Untitled Action'),
                description=decision.get('description', ''),
                workflow_type=workflow_config['workflow_type'],
                workflow_params=workflow_params,
                priority=decision.get('urgency', 5),
                status=ActionStatus.PENDING,
                created_at=datetime.now(timezone.utc)
            )

            return action

        except Exception as e:
            logger.error(f"Error converting decision to action: {e}")
            return None

    async def queue_action(self, action: BoardAction) -> str:
        """
        Queue an action for execution.
        """
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO board_action_queue
                (decision_id, category, title, description, workflow_type,
                 workflow_params, priority, status, queued_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', NOW())
                RETURNING action_id
            """, (
                action.decision_id,
                action.category.value if hasattr(action.category, 'value') else action.category,
                action.title,
                action.description,
                action.workflow_type,
                Json(action.workflow_params),
                action.priority
            ))

            action_id = str(cur.fetchone()[0])
            conn.commit()

            logger.info(f"Queued action {action_id} from decision {action.decision_id}")
            return action_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Error queuing action: {e}")
            raise
        finally:
            cur.close()

    async def get_pending_actions(self, limit: int = 5) -> List[Dict]:
        """
        Get pending actions ready for execution.
        """
        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT *
                FROM board_action_queue
                WHERE status IN ('queued', 'pending')
                AND (retry_count < max_retries OR retry_count IS NULL)
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
            """, (limit,))

            return [dict(a) for a in cur.fetchall()]

        except Exception as e:
            logger.error(f"Error getting pending actions: {e}")
            return []
        finally:
            cur.close()

    async def execute_action(self, action: Dict) -> Dict:
        """
        Execute a board action by triggering the appropriate LangGraph workflow.
        """
        action_id = action['action_id']

        try:
            # Update status to in_progress
            await self._update_action_status(action_id, 'in_progress')

            # Import and execute the workflow
            workflow_type = action['workflow_type']
            workflow_params = action.get('workflow_params', {})

            # Dynamic workflow execution based on type
            result = await self._trigger_workflow(workflow_type, workflow_params, action)

            # Update with success
            await self._update_action_result(action_id, 'completed', result)

            logger.info(f"Action {action_id} completed successfully")
            return {"status": "completed", "result": result}

        except Exception as e:
            logger.error(f"Action {action_id} failed: {e}")
            await self._update_action_result(action_id, 'failed', None, str(e))
            return {"status": "failed", "error": str(e)}

    async def _trigger_workflow(self, workflow_type: str, params: Dict, action: Dict) -> Dict:
        """
        Trigger the appropriate LangGraph workflow based on type.
        """
        try:
            # Import agent executor dynamically to avoid circular imports
            from agent_executor import AgentExecutor

            executor = AgentExecutor()

            # Build task from workflow params
            task = {
                "id": str(action['action_id']),
                "task_type": workflow_type,
                "source": "board_decision",
                "decision_id": action['decision_id'],
                "title": action['title'],
                "description": action['description'],
                "context": params.get('decision_context', {}),
                "use_langgraph": True,  # Use enhanced LangGraph workflow
                "enable_review_loop": True,  # Enable review loops
                "quality_gate": True,  # Enable quality gates
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            # Execute through the agent executor
            result = await executor.execute(task)

            return result

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    async def _update_action_status(self, action_id: str, status: str):
        """Update action status"""
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                UPDATE board_action_queue
                SET status = %s, executed_at = CASE WHEN %s = 'in_progress' THEN NOW() ELSE executed_at END
                WHERE action_id = %s
            """, (status, status, action_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating action status: {e}")
        finally:
            cur.close()

    async def _update_action_result(self, action_id: str, status: str, result: Optional[Dict], error: str = None):
        """Update action with result"""
        conn = self._get_connection()
        cur = conn.cursor()

        try:
            cur.execute("""
                UPDATE board_action_queue
                SET status = %s,
                    result = %s,
                    error_message = %s,
                    completed_at = NOW(),
                    retry_count = CASE WHEN %s = 'failed' THEN retry_count + 1 ELSE retry_count END
                WHERE action_id = %s
            """, (status, Json(result) if result else None, error, status, action_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating action result: {e}")
        finally:
            cur.close()

    async def process_pipeline(self):
        """
        Main pipeline processing loop.
        1. Poll for approved decisions
        2. Convert to actions
        3. Queue for execution
        4. Execute pending actions
        """
        logger.info("Starting Board Action Pipeline processing...")

        # Step 1: Poll for new approved decisions
        decisions = await self.poll_approved_decisions()
        logger.info(f"Found {len(decisions)} approved decisions to process")

        # Step 2 & 3: Convert and queue
        for decision in decisions:
            action = await self.convert_decision_to_action(decision)
            if action:
                await self.queue_action(action)

        # Step 4: Execute pending actions
        pending_actions = await self.get_pending_actions()
        logger.info(f"Found {len(pending_actions)} pending actions to execute")

        results = []
        for action in pending_actions:
            result = await self.execute_action(action)
            results.append(result)

        return {
            "decisions_processed": len(decisions),
            "actions_executed": len(results),
            "results": results
        }

    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status"""
        conn = self._get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cur.execute("""
                SELECT
                    status,
                    COUNT(*) as count
                FROM board_action_queue
                WHERE created_at > NOW() - INTERVAL '7 days'
                GROUP BY status
            """)

            status_counts = {row['status']: row['count'] for row in cur.fetchall()}

            cur.execute("""
                SELECT
                    COUNT(*) as pending_decisions
                FROM ai_board_decisions
                WHERE decision = 'approved'
                AND id NOT IN (SELECT DISTINCT decision_id FROM board_action_queue WHERE decision_id IS NOT NULL)
                AND decided_at > NOW() - INTERVAL '7 days'
            """)

            pending_decisions = cur.fetchone()['pending_decisions']

            return {
                "action_counts": status_counts,
                "pending_decisions": pending_decisions,
                "pipeline_healthy": True
            }

        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {"error": str(e), "pipeline_healthy": False}
        finally:
            cur.close()


# Singleton instance
_pipeline_instance = None

async def get_board_action_pipeline() -> BoardActionPipeline:
    """Get or create the Board Action Pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = BoardActionPipeline()
        await _pipeline_instance.initialize()
    return _pipeline_instance


async def run_pipeline_once():
    """Run the pipeline once (for scheduled jobs)"""
    pipeline = await get_board_action_pipeline()
    return await pipeline.process_pipeline()


# CLI for testing
if __name__ == "__main__":
    import sys

    async def main():
        pipeline = await get_board_action_pipeline()

        if len(sys.argv) > 1 and sys.argv[1] == "status":
            status = pipeline.get_pipeline_status()
            print(json.dumps(status, indent=2))
        else:
            result = await pipeline.process_pipeline()
            print(json.dumps(result, indent=2, default=str))

    asyncio.run(main())
