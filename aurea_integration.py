"""
AUREA Integration Module
Provides a simple interface for all agents to record decisions to the AUREA memory system.

Usage:
    from aurea_integration import AUREAIntegration

    class MyAgent:
        def __init__(self, tenant_id: str):
            self.aurea = AUREAIntegration(tenant_id, "my_agent_type")

        async def do_something(self):
            result = await self._perform_action()
            await self.aurea.record_decision(
                decision_type="action_performed",
                context={"input": data},
                decision="performed action X",
                rationale="because Y",
                confidence=0.85,
                outcome=result
            )
"""

import json
import logging
import os
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME', 'postgres'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}


def json_safe_serialize(obj: Any) -> Any:
    """Recursively convert datetime/Decimal/UUID/Enum/bytes objects to JSON-serializable types"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, uuid.UUID):
        return str(obj)
    elif isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    elif hasattr(obj, '__dataclass_fields__'):
        return {k: json_safe_serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {str(k): json_safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_safe_serialize(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(json_safe_serialize(item) for item in obj)
    elif hasattr(obj, '__dict__'):
        return json_safe_serialize(obj.__dict__)
    else:
        return str(obj)


class AUREAIntegration:
    """
    AUREA Integration for AI Agents

    Provides a unified interface for agents to:
    - Record decisions to the AUREA memory system
    - Store observations and learning
    - Query historical decisions for context
    - Track decision outcomes for continuous learning
    """

    def __init__(self, tenant_id: str, agent_type: str):
        """
        Initialize AUREA integration for an agent.

        Args:
            tenant_id: The tenant ID for multi-tenancy
            agent_type: The type of agent (e.g., "customer_success", "code_quality")
        """
        if not tenant_id:
            raise ValueError("tenant_id is required for AUREA integration")
        if not agent_type:
            raise ValueError("agent_type is required for AUREA integration")

        self.tenant_id = tenant_id
        self.agent_type = agent_type
        self._conn = None

    def _get_db_connection(self):
        """Get database connection"""
        try:
            return psycopg2.connect(**DB_CONFIG)
        except Exception as e:
            logger.error(f"AUREA DB connection failed: {e}")
            return None

    async def record_decision(
        self,
        decision_type: str,
        context: dict[str, Any],
        decision: str,
        rationale: str,
        confidence: float = 0.5,
        outcome: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Record an agent decision to the AUREA memory system.

        Args:
            decision_type: Type of decision (e.g., "analysis", "recommendation", "action")
            context: The input/context that led to this decision
            decision: The actual decision made
            rationale: Why this decision was made
            confidence: Confidence score 0.0-1.0
            outcome: The result/outcome if already known
            metadata: Additional metadata

        Returns:
            The decision UUID if successful, None otherwise
        """
        conn = self._get_db_connection()
        if not conn:
            logger.warning("Cannot record decision - DB connection failed")
            return None

        decision_id = str(uuid.uuid4())

        try:
            cur = conn.cursor()

            # Serialize all data to be JSON-safe
            safe_context = json_safe_serialize(context)
            safe_outcome = json_safe_serialize(outcome) if outcome else None
            safe_metadata = json_safe_serialize(metadata) if metadata else {}

            # Add agent metadata
            safe_metadata["agent_type"] = self.agent_type
            safe_metadata["recorded_via"] = "aurea_integration"

            cur.execute("""
                INSERT INTO aurea_decisions (
                    id, tenant_id, decision_type, context, decision,
                    rationale, confidence, outcome, metadata,
                    status, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, NOW(), NOW()
                )
                RETURNING id
            """, (
                decision_id,
                self.tenant_id,
                f"{self.agent_type}:{decision_type}",
                json.dumps(safe_context),
                decision,
                rationale,
                confidence,
                json.dumps(safe_outcome) if safe_outcome else None,
                json.dumps(safe_metadata),
                "completed" if outcome else "pending"
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"AUREA: Recorded decision {decision_id[:8]} for {self.agent_type}")
            return decision_id

        except Exception as e:
            logger.warning(f"Failed to record AUREA decision: {e}")
            if conn:
                conn.close()
            return None

    async def update_decision_outcome(
        self,
        decision_id: str,
        outcome: dict[str, Any],
        success: bool = True
    ) -> bool:
        """
        Update the outcome of a previously recorded decision.
        Used for learning from decision results.

        Args:
            decision_id: The UUID of the decision to update
            outcome: The outcome/result of the decision
            success: Whether the decision was successful

        Returns:
            True if update succeeded, False otherwise
        """
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            cur = conn.cursor()
            safe_outcome = json_safe_serialize(outcome)

            cur.execute("""
                UPDATE aurea_decisions
                SET outcome = %s,
                    status = %s,
                    updated_at = NOW()
                WHERE id = %s AND tenant_id = %s
            """, (
                json.dumps(safe_outcome),
                "completed" if success else "failed",
                decision_id,
                self.tenant_id
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.info(f"AUREA: Updated decision {decision_id[:8]} outcome")
            return True

        except Exception as e:
            logger.warning(f"Failed to update AUREA decision outcome: {e}")
            if conn:
                conn.close()
            return False

    async def get_historical_decisions(
        self,
        decision_type: Optional[str] = None,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Query historical decisions for learning context.

        Args:
            decision_type: Optional filter by decision type
            limit: Maximum number of decisions to return

        Returns:
            List of historical decisions
        """
        conn = self._get_db_connection()
        if not conn:
            return []

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if decision_type:
                cur.execute("""
                    SELECT id, decision_type, decision, rationale, confidence,
                           outcome, status, created_at
                    FROM aurea_decisions
                    WHERE tenant_id = %s
                    AND decision_type LIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self.tenant_id, f"{self.agent_type}:{decision_type}%", limit))
            else:
                cur.execute("""
                    SELECT id, decision_type, decision, rationale, confidence,
                           outcome, status, created_at
                    FROM aurea_decisions
                    WHERE tenant_id = %s
                    AND decision_type LIKE %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self.tenant_id, f"{self.agent_type}:%", limit))

            decisions = cur.fetchall()
            cur.close()
            conn.close()

            return [json_safe_serialize(dict(d)) for d in decisions] if decisions else []

        except Exception as e:
            logger.warning(f"Failed to query AUREA decisions: {e}")
            if conn:
                conn.close()
            return []

    async def store_observation(
        self,
        observation_type: str,
        data: dict[str, Any],
        importance: float = 0.5
    ) -> bool:
        """
        Store an observation to AUREA memory for future learning.

        Args:
            observation_type: Type of observation (e.g., "performance", "anomaly", "pattern")
            data: The observation data
            importance: How important this observation is (0.0-1.0)

        Returns:
            True if stored successfully
        """
        conn = self._get_db_connection()
        if not conn:
            return False

        try:
            cur = conn.cursor()
            safe_data = json_safe_serialize(data)

            cur.execute("""
                INSERT INTO agent_memories (
                    id, tenant_id, agent_type, memory_type, content,
                    importance, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, NOW()
                )
            """, (
                str(uuid.uuid4()),
                self.tenant_id,
                self.agent_type,
                observation_type,
                json.dumps(safe_data),
                importance
            ))

            conn.commit()
            cur.close()
            conn.close()

            logger.debug(f"AUREA: Stored observation for {self.agent_type}")
            return True

        except Exception as e:
            logger.warning(f"Failed to store AUREA observation: {e}")
            if conn:
                conn.close()
            return False

    async def get_learning_context(
        self,
        observation_types: Optional[list[str]] = None,
        min_importance: float = 0.3,
        limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Get historical observations for learning context.

        Args:
            observation_types: Optional list of observation types to filter
            min_importance: Minimum importance threshold
            limit: Maximum observations to return

        Returns:
            List of relevant observations
        """
        conn = self._get_db_connection()
        if not conn:
            return []

        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)

            if observation_types:
                placeholders = ','.join(['%s'] * len(observation_types))
                cur.execute(f"""
                    SELECT memory_type, content, importance, created_at
                    FROM agent_memories
                    WHERE tenant_id = %s
                    AND agent_type = %s
                    AND memory_type IN ({placeholders})
                    AND importance >= %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self.tenant_id, self.agent_type, *observation_types, min_importance, limit))
            else:
                cur.execute("""
                    SELECT memory_type, content, importance, created_at
                    FROM agent_memories
                    WHERE tenant_id = %s
                    AND agent_type = %s
                    AND importance >= %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self.tenant_id, self.agent_type, min_importance, limit))

            memories = cur.fetchall()
            cur.close()
            conn.close()

            return [json_safe_serialize(dict(m)) for m in memories] if memories else []

        except Exception as e:
            logger.warning(f"Failed to get AUREA learning context: {e}")
            if conn:
                conn.close()
            return []
