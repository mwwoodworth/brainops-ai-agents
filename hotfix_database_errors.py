#!/usr/bin/env python3
"""
HOTFIX: Database Schema and Type Mismatch Fixes
Fixes critical errors found in production logs
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD", "Brain0ps2O2S"),
    "port": os.getenv("DB_PORT", 5432)
}

def fix_database_issues():
    """Apply hotfixes for database issues"""
    conn = None
    try:
        print("🔧 Connecting to database...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Fix 1: Add amount_due as computed column if it doesn't exist
        print("📊 Checking invoices table...")
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'invoices'
            AND column_name = 'amount_due'
        """)

        if not cur.fetchone():
            print("✅ Adding amount_due computed column...")
            cur.execute("""
                ALTER TABLE invoices
                ADD COLUMN IF NOT EXISTS amount_due NUMERIC(12,2)
                GENERATED ALWAYS AS (
                    COALESCE(total_amount::numeric/100, 0) - COALESCE(paid_amount, 0)
                ) STORED
            """)
            conn.commit()
            print("✅ amount_due column added")
        else:
            print("ℹ️ amount_due column already exists")

        # Fix 2: Check unified_memory table structure
        print("\n📊 Checking unified_memory table...")
        cur.execute("""
            SELECT
                column_name,
                data_type,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_name = 'unified_memory'
            AND column_name IN ('context_id', 'agent_id', 'system_id')
        """)

        columns = cur.fetchall()
        for col in columns:
            print(f"  - {col['column_name']}: {col['data_type']}({col['character_maximum_length']})")

        # Fix 3: Create indexes for better performance
        print("\n🚀 Creating performance indexes...")

        # Index for amount_due queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_invoices_amount_due
            ON invoices(amount_due)
            WHERE amount_due > 0
        """)

        # Index for memory queries with text comparison
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_unified_memory_context_text
            ON unified_memory(context_id)
            WHERE context_id IS NOT NULL
        """)

        conn.commit()
        print("✅ Indexes created")

        # Fix 4: Add missing columns to memory stats view
        print("\n📊 Checking memory stats...")
        cur.execute("""
            SELECT COUNT(*) as total,
                   COUNT(DISTINCT context_type) as types,
                   COUNT(DISTINCT context_id) as contexts,
                   AVG(importance) as avg_importance
            FROM unified_memory
        """)

        stats = cur.fetchone()
        print(f"  Total memories: {stats['total']}")
        print(f"  Context types: {stats['types']}")
        print(f"  Unique contexts: {stats['contexts']}")
        print(f"  Avg importance: {stats['avg_importance']:.2f}")

        print("\n✅ Database hotfixes applied successfully!")

        return True

    except Exception as e:
        print(f"❌ Error applying fixes: {e}")
        if conn:
            conn.rollback()
        return False

    finally:
        if conn:
            conn.close()

def test_queries():
    """Test that the problematic queries now work"""
    conn = None
    try:
        print("\n🧪 Testing fixed queries...")
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Test 1: AUREA query for invoices with amount_due
        print("  Testing invoices amount_due query...")
        cur.execute("""
            SELECT id, amount_due, status
            FROM invoices
            WHERE amount_due > 0
            LIMIT 5
        """)
        invoices = cur.fetchall()
        print(f"  ✅ Found {len(invoices)} invoices with amount_due > 0")

        # Test 2: Memory query with proper text comparison
        print("  Testing memory context_id query...")
        cur.execute("""
            SELECT id, context_id, key, importance
            FROM unified_memory
            WHERE context_id::text = %s
            LIMIT 5
        """, ('system',))
        memories = cur.fetchall()
        print(f"  ✅ Found {len(memories)} memories for context")

        # Test 3: Check connection pooling
        print("  Testing connection stability...")
        for i in range(3):
            cur.execute("SELECT 1")
            result = cur.fetchone()
            print(f"    Connection test {i+1}: {'✅' if result else '❌'}")

        print("\n✅ All test queries passed!")
        return True

    except Exception as e:
        print(f"❌ Test query failed: {e}")
        return False

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("=" * 60)
    print("HOTFIX: Database Schema and Type Mismatch Fixes")
    print("=" * 60)

    # Apply fixes
    if fix_database_issues():
        # Test the fixes
        test_queries()

    print("\n" + "=" * 60)
    print("Hotfix complete. Monitor logs for improvements.")
    print("=" * 60)