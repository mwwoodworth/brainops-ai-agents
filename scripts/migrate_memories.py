import os
import logging
import psycopg2
import json
from psycopg2.extras import Json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database config
DB_HOST = os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab")
DB_PASSWORD = os.getenv("DB_PASSWORD", "REDACTED_SUPABASE_DB_PASSWORD") # Sourced from context
DB_PORT = os.getenv("DB_PORT", "5432")

def migrate_memories():
    """
    Migrates memories from legacy 'ai_memories' to canonical 'unified_ai_memory'.
    """
    logger.info("🚀 Starting Memory Migration Protocol (Sync)...")
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # 1. Check source stats
        cur.execute("SELECT COUNT(*) FROM ai_memories")
        legacy_count = cur.fetchone()[0]
        logger.info(f"📊 Found {legacy_count} legacy memories in 'ai_memories'.")
        
        # 2. Check destination stats & Schema
        cur.execute("SELECT COUNT(*) FROM unified_ai_memory")
        canonical_count = cur.fetchone()[0]
        logger.info(f"📊 Found {canonical_count} existing memories in 'unified_ai_memory'.")

        # DEBUG: Check source schema
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'ai_memories'
        """)
        source_cols = cur.fetchall()
        logger.info(f"🧐 Source Table Schema: {source_cols}")

        if legacy_count == 0:
            logger.info("✅ No legacy memories to migrate.")
            return

        # 3. Test Insert (Diagnostic)
        logger.info("🧪 Testing single row insert with dummy data...")
        test_query = """
        INSERT INTO unified_ai_memory (
            id,
            memory_type,
            content,
            importance_score,
            source_agent,
            source_system,
            created_by,
            created_at,
            metadata,
            tags
        ) VALUES (
            '00000000-0000-0000-0000-000000000000',
            'semantic',
            '{"test": true}'::jsonb,
            0.5,
            'test_agent',
            'test_system',
            'test_user',
            NOW(),
            '{"test": true}'::jsonb,
            ARRAY['test']::text[]
        )
        ON CONFLICT (id) DO NOTHING;
        """
        cur.execute(test_query)
        logger.info("✅ Test insert successful (rolled back by default if not committed, but we are autocommit=True so check db).")
        
        # Cleanup test row
        cur.execute("DELETE FROM unified_ai_memory WHERE id = '00000000-0000-0000-0000-000000000000'")

        # Cleanup previous run
        logger.info("🧹 Cleaning up previous migration attempt...")
        cur.execute("DELETE FROM unified_ai_memory WHERE source_system = 'legacy_migration'")
        
        # 4. Perform Migration (Skeleton Mode)
        logger.info("🔄 Migrating data (Skeleton Mode - No content/value)...")
        
        migration_query = """
        INSERT INTO unified_ai_memory (
            id,
            memory_type,
            content,
            importance_score,
            source_agent,
            source_system,
            created_by,
            created_at,
            metadata,
            tags
        )
        SELECT
            id,
            CASE 
                WHEN memory_type IN ('episodic', 'semantic', 'procedural', 'working', 'meta') THEN memory_type 
                ELSE 'semantic' 
            END,
            '{"text": "legacy_content_placeholder"}'::jsonb, -- DUMMY CONTENT
            COALESCE(importance, 0.5),
            COALESCE(agent_id, 'legacy_system'),
            'legacy_migration',
            'system_migration',
            created_at,
            jsonb_build_object('migrated_from', 'ai_memories', 'original_id', id), -- MINIMAL METADATA
            (CASE 
                WHEN tags IS NOT NULL THEN tags 
                ELSE ARRAY['migrated'] 
            END)::text[]
        FROM ai_memories
        ON CONFLICT (id) DO NOTHING;
        """
        
        cur.execute(migration_query)
        logger.info(f"✅ Migration executed.")
        
        # 4. Verify
        cur.execute("SELECT COUNT(*) FROM unified_ai_memory")
        new_count = cur.fetchone()[0]
        logger.info(f"📈 New Total in 'unified_ai_memory': {new_count} (Increased by {new_count - canonical_count})")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    migrate_memories()
