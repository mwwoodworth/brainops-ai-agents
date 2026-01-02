#!/usr/bin/env python3
"""
Minimal schema verifier for BrainOps AI Agents.
Checks presence of critical tables/columns in the live database.
Skips execution if DATABASE_URL is missing.
"""
import os
import sys

import psycopg2

REQUIRED_TABLES = {
    "ai_agents": ["id", "name", "category", "type", "capabilities", "status"],
    "agent_schedules": ["id", "agent_id", "frequency_minutes", "enabled"],
    "ai_autonomous_tasks": ["id", "title", "payload", "priority", "status"],
    "ai_error_logs": ["id", "error_id", "error_type", "timestamp"],
    "ai_recovery_actions_log": ["id", "error_id", "action_id", "executed_at"],
    "ai_healing_rules": ["id", "rule_name", "priority", "condition", "action"],
}


def get_conn():
    url = os.getenv("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set; skipping schema check.")
        sys.exit(0)
    return psycopg2.connect(url)


def fetch_columns(cur, table):
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
        """,
        (table,),
    )
    return {row[0] for row in cur.fetchall()}


def main():
    conn = get_conn()
    cur = conn.cursor()
    missing = []

    for table, cols in REQUIRED_TABLES.items():
        cur.execute(
            """
            SELECT to_regclass(%s)
            """,
            (table,),
        )
        exists = cur.fetchone()[0]
        if not exists:
            missing.append(f"table {table} is missing")
            continue
        present = fetch_columns(cur, table)
        for col in cols:
            if col not in present:
                missing.append(f"{table}.{col} is missing")

    cur.close()
    conn.close()

    if missing:
        print("❌ Schema check failed:")
        for item in missing:
            print(f" - {item}")
        sys.exit(1)
    else:
        print("✅ Schema check passed.")


if __name__ == "__main__":
    main()
