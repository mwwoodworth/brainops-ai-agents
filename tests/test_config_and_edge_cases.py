"""
Tests for config.py, database edge cases, empty data scenarios,
concurrent handling, and MockTenantScopedPool behavior.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import config


# ---------------------------------------------------------------------------
# Config module
# ---------------------------------------------------------------------------


class TestConfig:
    def test_security_section_exists(self):
        assert hasattr(config, "security")
        assert hasattr(config.security, "valid_api_keys")
        assert hasattr(config.security, "auth_required")

    def test_valid_api_keys_is_set(self):
        assert isinstance(config.security.valid_api_keys, set)

    def test_log_level_is_string(self):
        assert isinstance(config.log_level, str)
        assert config.log_level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


# ---------------------------------------------------------------------------
# MockTenantScopedPool edge cases
# ---------------------------------------------------------------------------

from conftest import MockTenantScopedPool


class TestMockTenantScopedPool:
    @pytest.mark.asyncio
    async def test_default_fetch_returns_empty_list(self):
        pool = MockTenantScopedPool()
        result = await pool.fetch("SELECT 1")
        assert result == []

    @pytest.mark.asyncio
    async def test_default_fetchrow_returns_none(self):
        pool = MockTenantScopedPool()
        result = await pool.fetchrow("SELECT 1")
        assert result is None

    @pytest.mark.asyncio
    async def test_default_fetchval_returns_none(self):
        pool = MockTenantScopedPool()
        result = await pool.fetchval("SELECT 1")
        assert result is None

    @pytest.mark.asyncio
    async def test_default_execute_returns_ok(self):
        pool = MockTenantScopedPool()
        result = await pool.execute("INSERT INTO t VALUES (1)")
        assert result == "OK"

    @pytest.mark.asyncio
    async def test_custom_fetch_handler(self):
        pool = MockTenantScopedPool()
        pool.fetch_handler = lambda q, *a: [{"id": 1}]
        result = await pool.fetch("SELECT * FROM t")
        assert result == [{"id": 1}]

    @pytest.mark.asyncio
    async def test_custom_fetchrow_handler(self):
        pool = MockTenantScopedPool()
        pool.fetchrow_handler = lambda q, *a: {"name": "test"}
        result = await pool.fetchrow("SELECT * FROM t LIMIT 1")
        assert result == {"name": "test"}

    @pytest.mark.asyncio
    async def test_custom_fetchval_handler(self):
        pool = MockTenantScopedPool()
        pool.fetchval_handler = lambda q, *a: 42
        result = await pool.fetchval("SELECT count(*) FROM t")
        assert result == 42

    @pytest.mark.asyncio
    async def test_call_tracking_fetch(self):
        pool = MockTenantScopedPool()
        await pool.fetch("SELECT 1", "arg1")
        assert len(pool.calls["fetch"]) == 1
        assert pool.calls["fetch"][0] == ("SELECT 1", ("arg1",))

    @pytest.mark.asyncio
    async def test_call_tracking_fetchrow(self):
        pool = MockTenantScopedPool()
        await pool.fetchrow("SELECT 1")
        assert len(pool.calls["fetchrow"]) == 1

    @pytest.mark.asyncio
    async def test_call_tracking_execute(self):
        pool = MockTenantScopedPool()
        await pool.execute("DELETE FROM t")
        assert len(pool.calls["execute"]) == 1

    @pytest.mark.asyncio
    async def test_test_connection_default(self):
        pool = MockTenantScopedPool()
        result = await pool.test_connection()
        assert result is True
        assert pool.calls["test_connection"] == 1

    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        pool = MockTenantScopedPool()
        pool.test_connection_handler = lambda: False
        result = await pool.test_connection()
        assert result is False

    @pytest.mark.asyncio
    async def test_pool_attributes(self):
        pool = MockTenantScopedPool()
        assert pool.pool.get_min_size() == 1
        assert pool.pool.get_max_size() == 10
        assert pool.pool.get_size() == 1
        assert pool.pool.get_idle_size() == 1

    @pytest.mark.asyncio
    async def test_async_handler_support(self):
        """Handlers can be async functions."""
        pool = MockTenantScopedPool()

        async def async_handler(q, *a):
            return [{"async": True}]

        pool.fetch_handler = async_handler
        result = await pool.fetch("SELECT 1")
        assert result == [{"async": True}]


# ---------------------------------------------------------------------------
# Database connection failure edge cases
# ---------------------------------------------------------------------------


class TestDatabaseFailureHandling:
    @pytest.mark.asyncio
    async def test_pool_fetch_returns_empty_on_no_data(self):
        pool = MockTenantScopedPool()
        pool.fetch_handler = lambda q, *a: []
        result = await pool.fetch("SELECT * FROM nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_pool_fetchval_returns_none_for_missing_row(self):
        pool = MockTenantScopedPool()
        pool.fetchval_handler = lambda q, *a: None
        result = await pool.fetchval("SELECT val FROM t WHERE id = $1", "missing")
        assert result is None


# ---------------------------------------------------------------------------
# Concurrent request handling
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    @pytest.mark.asyncio
    async def test_concurrent_pool_operations(self):
        """Multiple concurrent operations on the mock pool should not interfere."""
        pool = MockTenantScopedPool()
        pool.fetch_handler = lambda q, *a: [{"id": 1}]
        pool.fetchval_handler = lambda q, *a: 42

        results = await asyncio.gather(
            pool.fetch("SELECT 1"),
            pool.fetchval("SELECT count(*)"),
            pool.fetchrow("SELECT 1"),
            pool.execute("INSERT INTO t VALUES (1)"),
            pool.test_connection(),
        )

        assert results[0] == [{"id": 1}]
        assert results[1] == 42
        assert results[2] is None
        assert results[3] == "OK"
        assert results[4] is True

    @pytest.mark.asyncio
    async def test_concurrent_fetch_call_tracking(self):
        pool = MockTenantScopedPool()
        await asyncio.gather(
            pool.fetch("Q1"),
            pool.fetch("Q2"),
            pool.fetch("Q3"),
        )
        assert len(pool.calls["fetch"]) == 3
