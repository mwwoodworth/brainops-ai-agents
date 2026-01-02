#!/usr/bin/env python3
"""
BrainOps AI Memory System
Persistent memory and context management for all AI operations
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class AIMemorySystem:
    """Unified memory system for all AI operations"""

    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST"),
            "database": "postgres",
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "port": 5432
        }

    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)

    def store_context(self, context_type: str, key: str, value: Any, critical: bool = False) -> str:
        """Store context in master memory"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO ai_master_context (context_type, context_key, context_value, is_critical)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (context_key) DO UPDATE
                SET context_value = %s,
                    updated_at = NOW(),
                    access_count = ai_master_context.access_count + 1
                RETURNING id
            """, (context_type, key, json.dumps(value), critical, json.dumps(value)))

            result = cursor.fetchone()
            conn.commit()
            return str(result['id'])

        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def get_context(self, key: str) -> Optional[dict]:
        """Retrieve context from master memory"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                UPDATE ai_master_context
                SET last_accessed = NOW(),
                    access_count = access_count + 1
                WHERE context_key = %s
                RETURNING context_value
            """, (key,))

            result = cursor.fetchone()
            conn.commit()

            if result:
                return result['context_value']
            return None

        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            return None
        finally:
            cursor.close()
            conn.close()

    def record_conversation(self, session_id: str, role: str, content: str,
                          tools_used: list[str] = None, important_facts: dict = None):
        """Record conversation for learning"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO ai_conversation_memory
                (session_id, turn_number, role, content, tools_used, important_facts, created_at)
                VALUES (%s,
                    (SELECT COALESCE(MAX(turn_number), 0) + 1
                     FROM ai_conversation_memory WHERE session_id = %s),
                    %s, %s, %s, %s, NOW())
            """, (session_id, session_id, role, content,
                  json.dumps(tools_used) if tools_used else None,
                  json.dumps(important_facts) if important_facts else None))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to record conversation: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def update_system_state(self, component: str, state_key: str, value: Any, reason: str = None):
        """Update system state tracking"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            payload = {
                "value": value,
                "reason": reason,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            cursor.execute("""
                WITH target AS (
                    SELECT id FROM ai_system_state
                    ORDER BY last_updated DESC NULLS LAST
                    LIMIT 1
                ),
                updated AS (
                    UPDATE ai_system_state
                    SET state = jsonb_set(
                        COALESCE(state, '{}'::jsonb),
                        ARRAY[%s, %s],
                        %s::jsonb,
                        true
                    ),
                    last_updated = NOW()
                    WHERE id IN (SELECT id FROM target)
                    RETURNING id
                )
                INSERT INTO ai_system_state (state, last_updated)
                SELECT jsonb_build_object(%s, jsonb_build_object(%s, %s::jsonb)), NOW()
                WHERE NOT EXISTS (SELECT 1 FROM updated)
            """, (
                component,
                state_key,
                json.dumps(payload),
                component,
                state_key,
                json.dumps(payload)
            ))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to update system state: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def get_critical_context(self) -> dict:
        """Get all critical system context"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT context_key, context_value
                FROM ai_master_context
                WHERE is_critical = true
                ORDER BY importance DESC, updated_at DESC
            """)

            results = cursor.fetchall()
            context = {}

            for row in results:
                context[row['context_key']] = row['context_value']

            return context

        except Exception as e:
            logger.error(f"Failed to get critical context: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()

    def search_knowledge(self, query: str, category: str = None, limit: int = 10) -> list[dict]:
        """Search knowledge base"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            if category:
                cursor.execute("""
                    SELECT title, content, category, confidence_score, source
                    FROM ai_knowledge_base
                    WHERE category = %s
                      AND (title ILIKE %s OR content ILIKE %s)
                    ORDER BY confidence_score DESC, updated_at DESC
                    LIMIT %s
                """, (category, f'%{query}%', f'%{query}%', limit))
            else:
                cursor.execute("""
                    SELECT title, content, category, confidence_score, source
                    FROM ai_knowledge_base
                    WHERE title ILIKE %s OR content ILIKE %s
                    ORDER BY confidence_score DESC, updated_at DESC
                    LIMIT %s
                """, (f'%{query}%', f'%{query}%', limit))

            return cursor.fetchall()

        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []
        finally:
            cursor.close()
            conn.close()

    def get_system_overview(self) -> dict:
        """Get complete system overview"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            overview = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'services': {},
                'statistics': {},
                'recent_tasks': []
            }

            # Get service status
            cursor.execute("""
                SELECT service_name, environment, version, health_status, config
                FROM ai_operational_context
                WHERE environment = 'production'
            """)

            for service in cursor.fetchall():
                overview['services'][service['service_name']] = {
                    'version': service['version'],
                    'status': service['health_status'],
                    'config': service['config']
                }

            # Get statistics
            cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM customers) as customers,
                    (SELECT COUNT(*) FROM jobs) as jobs,
                    (SELECT COUNT(*) FROM invoices) as invoices,
                    (SELECT COUNT(*) FROM ai_agents) as agents,
                    (SELECT COUNT(*) FROM agent_executions WHERE created_at > NOW() - INTERVAL '24 hours') as recent_executions,
                    (SELECT COUNT(*) FROM ai_master_context) as context_entries,
                    (SELECT COUNT(*) FROM ai_knowledge_base) as knowledge_entries
            """)

            overview['statistics'] = cursor.fetchone()

            # Get recent tasks
            cursor.execute("""
                SELECT task_type, task_description, status, started_at
                FROM ai_task_history
                WHERE started_at > NOW() - INTERVAL '24 hours'
                ORDER BY started_at DESC
                LIMIT 10
            """)

            overview['recent_tasks'] = cursor.fetchall()

            return overview

        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {}
        finally:
            cursor.close()
            conn.close()

# Global instance
memory_system = AIMemorySystem()

def initialize_system():
    """Initialize the memory system with current context"""
    memory = memory_system

    # Store system information
    memory.store_context('system', 'initialization_time', datetime.now(timezone.utc).isoformat())

    # Store service endpoints
    endpoints = {
        'backend': 'https://brainops-backend-prod.onrender.com',
        'ai_agents': 'https://brainops-ai-agents.onrender.com',
        'myroofgenius': 'https://myroofgenius.com',
        'weathercraft': 'https://weathercraft-erp.vercel.app'
    }
    memory.store_context('system', 'service_endpoints', endpoints, critical=True)

    # Store database configuration
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': 'postgres',
        'user': os.getenv('DB_USER'),
        'port': 5432
    }
    memory.store_context('system', 'database_config', db_config, critical=True)

    logger.info("Memory system initialized")
    return memory

if __name__ == "__main__":
    # Test the memory system
    memory = initialize_system()

    # Get system overview
    overview = memory.get_system_overview()
    print(f"System Overview: {json.dumps(overview, indent=2, default=str)}")

    # Get critical context
    critical = memory.get_critical_context()
    print(f"\nCritical Context Keys: {list(critical.keys())}")
