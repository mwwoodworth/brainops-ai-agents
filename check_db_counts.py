import os
import json
import psycopg2

# Use env vars from context
DB_HOST = os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab")
DB_PASSWORD = os.getenv("DB_PASSWORD")
if not DB_PASSWORD:
    raise RuntimeError("DB_PASSWORD environment variable is required")
DB_PORT = 5432

TABLES_TO_CHECK = [
    "ai_nerve_signals",
    "revenue_leads",
    "revenue_opportunities",
    "revenue_actions",
    "ai_email_queue",
    "ai_agent_executions",
    "ai_system_snapshot",
    "unified_brain_logs",
    "ai_usage_logs"
]

def check_tables():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        results = {}
        for table in TABLES_TO_CHECK:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                results[table] = count
            except Exception as e:
                # Table might not exist
                conn.rollback()
                results[table] = f"Error: {e}"
        
        print(json.dumps(results, indent=2))
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Global Error: {e}")

if __name__ == "__main__":
    check_tables()
