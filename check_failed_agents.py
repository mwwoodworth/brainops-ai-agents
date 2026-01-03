
import os
import json
import psycopg2

# Use env vars from context
DB_HOST = os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab")
DB_PASSWORD = os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD")
DB_PORT = 5432

def check_failed_agents():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        
        # Get count of failed executions
        cur.execute("SELECT COUNT(*) FROM ai_agent_executions WHERE status = 'failed'")
        failed_count = cur.fetchone()[0]
        
        # Get recent error messages
        cur.execute("""
            SELECT agent_name, error_message, created_at 
            FROM ai_agent_executions 
            WHERE status = 'failed' 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        errors = cur.fetchall()
        
        print(f"Total Failed Executions: {failed_count}")
        print("\nRecent Errors:")
        for error in errors:
            print(f"- {error[0]}: {error[1]} ({error[2]})")
            
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Global Error: {e}")

if __name__ == "__main__":
    check_failed_agents()
