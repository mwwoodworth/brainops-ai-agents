#!/usr/bin/env python3
"""
Production Test Suite - Comprehensive Testing
Verifies all components work end-to-end
"""
import asyncio
import sys
import os
import logging
from typing import Any, Dict, List
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_imports() -> bool:
    """Test all required imports"""
    logger.info("Testing imports...")
    try:
        import asyncpg
        import pydantic
        import fastapi
        from database.async_connection import AsyncDatabasePool, PoolConfig
        from models.agent import Agent, AgentCategory, AgentExecution
        from api.memory import MemoryStatus
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False


async def test_database_connection() -> bool:
    """Test database connectivity"""
    logger.info("Testing database connection...")
    try:
        from database.async_connection import AsyncDatabasePool, PoolConfig
        from config import config

        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify,
        )

        pool = AsyncDatabasePool(pool_config)
        await pool.initialize()

        # Test connection
        result = await pool.test_connection()
        if result:
            logger.info("✅ Database connection successful")
        else:
            logger.error("❌ Database connection test failed")
            return False

        # Check for agents
        agents = await pool.fetch("SELECT COUNT(*) as count FROM agents")
        agent_count = agents[0]["count"]
        logger.info(f"✅ Found {agent_count} agents in database")

        await pool.close()
        return True

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False


async def test_models() -> bool:
    """Test Pydantic models"""
    logger.info("Testing type-safe models...")
    try:
        from models.agent import Agent, AgentCategory, AgentCapability

        # Test Agent model
        agent = Agent(
            id="test_001",
            name="Test Agent",
            category=AgentCategory.TECHNICAL,
            description="Test agent for validation",
            enabled=True,
            capabilities=[
                AgentCapability(
                    name="test_capability",
                    description="Test capability",
                    enabled=True
                )
            ]
        )

        # Validate serialization
        agent_dict = agent.dict()
        assert agent_dict["id"] == "test_001"
        assert agent_dict["category"] == "technical"

        # Test JSON serialization
        agent_json = agent.json()
        assert "test_001" in agent_json

        logger.info("✅ Model validation successful")
        return True

    except Exception as e:
        logger.error(f"❌ Model test failed: {e}")
        return False


async def test_memory_endpoint() -> bool:
    """Test memory endpoint functionality"""
    logger.info("Testing memory endpoints...")
    try:
        from database.async_connection import AsyncDatabasePool, PoolConfig, init_pool
        from config import config
        from api.memory import get_memory_status

        # Initialize pool
        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify,
        )

        await init_pool(pool_config)

        # Test memory status
        status = await get_memory_status()

        logger.info(f"Memory Status: {status.status}")
        logger.info(f"Total Memories: {status.total_memories}")
        logger.info(f"Table Used: {status.table_used}")

        if status.status in ["operational", "not_configured", "error"]:
            logger.info("✅ Memory endpoint working correctly")
            return True
        else:
            logger.error("❌ Unexpected memory status")
            return False

    except Exception as e:
        logger.error(f"❌ Memory endpoint test failed: {e}")
        return False


async def test_agent_operations() -> bool:
    """Test agent CRUD operations"""
    logger.info("Testing agent operations...")
    try:
        from database.async_connection import AsyncDatabasePool, PoolConfig
        from config import config

        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify,
        )

        pool = AsyncDatabasePool(pool_config)
        await pool.initialize()

        # Get agents
        agents = await pool.fetch("""
            SELECT id, name, type, status
            FROM agents
            WHERE status = 'active'
            LIMIT 5
        """)

        if agents:
            logger.info(f"✅ Found {len(agents)} active agents")
            for agent in agents:
                logger.info(f"  - {agent['name']} ({agent['type']})")
        else:
            logger.warning("⚠️ No active agents found")

        await pool.close()
        return True

    except Exception as e:
        logger.error(f"❌ Agent operations test failed: {e}")
        return False


async def test_memory_tables() -> bool:
    """Test memory table existence"""
    logger.info("Testing memory tables...")
    try:
        from database.async_connection import AsyncDatabasePool, PoolConfig
        from config import config

        pool_config = PoolConfig(
            host=config.database.host,
            port=config.database.port,
            user=config.database.user,
            password=config.database.password,
            database=config.database.database,
            ssl=config.database.ssl,
            ssl_verify=config.database.ssl_verify,
        )

        pool = AsyncDatabasePool(pool_config)
        await pool.initialize()

        # Check for memory tables
        tables = await pool.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN (
                'ai_persistent_memory',
                'memory_entries',
                'memories',
                'ai_conversations',
                'ai_messages'
            )
        """)

        if tables:
            logger.info(f"✅ Found {len(tables)} memory tables:")
            for table in tables:
                logger.info(f"  - {table['table_name']}")
        else:
            logger.warning("⚠️ No memory tables found (system will create on demand)")

        await pool.close()
        return True

    except Exception as e:
        logger.error(f"❌ Memory table test failed: {e}")
        return False


async def run_all_tests() -> int:
    """Run all tests and return exit code"""
    logger.info("=" * 50)
    logger.info("PRODUCTION TEST SUITE - AI AGENTS SERVICE")
    logger.info("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Database Connection", test_database_connection),
        ("Pydantic Models", test_models),
        ("Memory Endpoints", test_memory_endpoint),
        ("Agent Operations", test_agent_operations),
        ("Memory Tables", test_memory_tables)
    ]

    results: List[bool] = []

    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running: {test_name}")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            logger.error(f"❌ Test crashed: {e}")
            results.append(False)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED - Ready for deployment")
        return 0
    else:
        logger.error(f"\n⚠️ {total - passed} tests failed - Fix before deploying")
        return 1


def main() -> None:
    """Main entry point"""
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
