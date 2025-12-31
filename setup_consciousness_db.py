import asyncio
import asyncpg

DB_URL = "postgresql://postgres.yomagoqdmxszqtdwuhab:REDACTED_SUPABASE_DB_PASSWORD@aws-0-us-east-2.pooler.supabase.com:5432/postgres"

async def setup_db():
    print(f"Connecting to {DB_URL.split('@')[1]}...")
    try:
        conn = await asyncpg.connect(DB_URL)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    try:
        with open("consciousness_schema.sql", "r") as f:
            schema_sql = f.read()
            
        print("Executing schema...")
        await conn.execute(schema_sql)
        print("Schema executed successfully.")
    except Exception as e:
        print(f"Schema execution failed: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_db())
