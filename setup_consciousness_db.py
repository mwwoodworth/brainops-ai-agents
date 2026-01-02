import asyncio
import os
from urllib.parse import urlparse as _urlparse

import asyncpg


# Build DB URL from environment variables - NO hardcoded credentials
def get_db_url() -> str:
    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv("DB_NAME", "postgres")

    if not all([host, user, password]):
        raise RuntimeError(
            "Database credentials not configured. "
            "Set DB_HOST, DB_USER, DB_PASSWORD environment variables."
        )
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"

async def setup_db():
    DB_URL = get_db_url()
    print(f"Connecting to {DB_URL.split('@')[1]}...")
    try:
        conn = await asyncpg.connect(DB_URL)
        print("Connected.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    try:
        with open("consciousness_schema.sql") as f:
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
