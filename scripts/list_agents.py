"""
Utility script to list registered agents in the executor and in the ai_agents table.

Usage:
  python -m scripts.list_agents
"""

from typing import Any
import psycopg2
from psycopg2.extras import RealDictCursor

from config import config
from agent_executor import AgentExecutor


def list_registered_agents() -> list[str]:
    executor = AgentExecutor()
    executor._load_agent_implementations()
    return sorted(executor.agents.keys())


def list_ai_agents_table() -> list[dict[str, Any]]:
    conn = psycopg2.connect(config.database.connection_string, cursor_factory=RealDictCursor)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, name, is_active FROM ai_agents ORDER BY name")
        return list(cur.fetchall())
    finally:
        conn.close()


def main() -> None:
    print("Registered executor agents:")
    for name in list_registered_agents():
        print(f"  - {name}")

    try:
        print("\nDatabase ai_agents entries:")
        for row in list_ai_agents_table():
            status = "active" if row.get("is_active") else "inactive"
            print(f"  - {row.get('name')} ({row.get('id')}): {status}")
    except Exception as exc:  # pragma: no cover - defensive utility
        print("Unable to query ai_agents table:", exc)


if __name__ == "__main__":
    main()

