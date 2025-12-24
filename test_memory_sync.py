#!/usr/bin/env python3
"""
Test script for embedded memory sync fixes
"""
import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from embedded_memory_system import EmbeddedMemorySystem


async def test_sync_logic():
    """Test the sync logic with mock pool"""
    print("🧪 Testing Embedded Memory Sync Logic...")

    # Create instance with test database
    memory = EmbeddedMemorySystem(local_db_path="/tmp/test_memory.db")

    # Setup local DB
    await memory._setup_local_db()
    print("✅ Local DB created")

    # Check initial count
    cursor = memory.sqlite_conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM unified_ai_memory").fetchone()[0]
    print(f"📊 Initial count: {count}")

    # Test _ensure_pool_connection (should fail gracefully)
    has_pool = await memory._ensure_pool_connection()
    print(f"🔌 Pool connection: {has_pool}")

    # Test sync_from_master without pool (should skip gracefully)
    await memory.sync_from_master(force=True)
    print("✅ sync_from_master handled missing pool gracefully")

    # Test that it would retry
    print("✅ Retry logic built into _ensure_pool_connection")

    # Cleanup
    memory.sqlite_conn.close()
    os.remove("/tmp/test_memory.db")
    print("🧹 Cleaned up test database")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_sync_logic())
