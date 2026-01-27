#!/usr/bin/env python3
"""
Circuit Breaker Unit Tests
==========================
Tests for the service_circuit_breakers module.

Run with: pytest tests/test_circuit_breakers.py -v
"""

import asyncio
import time
import pytest
import sys
sys.path.insert(0, '.')

from service_circuit_breakers import (
    get_circuit_breaker_manager,
    get_circuit_breaker_health,
    get_all_circuit_statuses,
    check_service_available,
    report_service_success,
    report_service_failure,
    reset_service_circuit,
    with_circuit_breaker,
    CircuitBreakerContext,
    CircuitOpenError,
    ServiceCircuitBreakerManager,
    ServiceCircuitConfig,
    ServiceType,
    CIRCUIT_BREAKER_CONFIG,
)


class TestServiceCircuitBreakers:
    """Tests for the ServiceCircuitBreakerManager"""

    def setup_method(self):
        """Reset circuits before each test"""
        manager = get_circuit_breaker_manager()
        for service_name in list(manager._circuits.keys()):
            manager.reset(service_name)

    def test_manager_singleton(self):
        """Test that manager is a singleton"""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()
        assert manager1 is manager2

    def test_default_circuits_configured(self):
        """Test that default circuits are configured"""
        manager = get_circuit_breaker_manager()
        assert len(manager._circuits) >= 15
        assert "openai" in manager._circuits
        assert "database" in manager._circuits
        assert "webhook_stripe" in manager._circuits

    def test_health_endpoint_data(self):
        """Test health endpoint data format"""
        health = get_circuit_breaker_health()
        assert "circuit_breakers" in health
        assert "total" in health["circuit_breakers"]
        assert "open" in health["circuit_breakers"]
        assert "closed" in health["circuit_breakers"]
        assert "overall_health" in health["circuit_breakers"]

    def test_record_success(self):
        """Test recording successful calls"""
        manager = get_circuit_breaker_manager()
        manager.record_success("openai", 100.0)
        status = manager.get_status("openai")
        assert status["success_count"] >= 1
        assert status["state"] == "closed"

    def test_record_failure_opens_circuit(self):
        """Test that failures open the circuit"""
        manager = get_circuit_breaker_manager()
        config = CIRCUIT_BREAKER_CONFIG["openai"]

        # Record enough failures to open circuit
        for i in range(config.failure_threshold):
            manager.record_failure("openai", 5000, f"Error {i}")

        status = manager.get_status("openai")
        assert status["state"] == "open"
        assert not manager.allows_request("openai")

    def test_reset_circuit(self):
        """Test resetting a circuit"""
        manager = get_circuit_breaker_manager()

        # Open the circuit
        for i in range(5):
            manager.record_failure("anthropic", 5000, "Error")

        assert manager.get_status("anthropic")["state"] == "open"

        # Reset it
        manager.reset("anthropic")
        status = manager.get_status("anthropic")
        assert status["state"] == "closed"
        assert manager.allows_request("anthropic")

    def test_allows_request_when_closed(self):
        """Test that closed circuits allow requests"""
        manager = get_circuit_breaker_manager()
        assert check_service_available("gemini")

    def test_blocks_request_when_open(self):
        """Test that open circuits block requests"""
        manager = get_circuit_breaker_manager()

        # Open the circuit
        for i in range(5):
            manager.record_failure("gemini", 5000, "Error")

        assert not check_service_available("gemini")

    def test_helper_functions(self):
        """Test helper functions"""
        # report_service_success
        report_service_success("resend_email", 50.0)
        manager = get_circuit_breaker_manager()
        assert manager.get_status("resend_email")["success_count"] >= 1

        # report_service_failure
        report_service_failure("resend_email", 5000, "Test error")
        assert manager.get_status("resend_email")["failure_count"] >= 1

        # reset_service_circuit
        reset_service_circuit("resend_email")
        assert manager.get_status("resend_email")["state"] == "closed"

    def test_auto_create_circuit(self):
        """Test that unknown services get auto-created"""
        manager = get_circuit_breaker_manager()
        status = manager.get_status("unknown_service")
        assert "state" in status  # Should not error


class TestCircuitBreakerDecorator:
    """Tests for the @with_circuit_breaker decorator"""

    def setup_method(self):
        """Reset circuits before each test"""
        manager = get_circuit_breaker_manager()
        manager.reset("render_api")

    def test_decorator_records_success(self):
        """Test decorator records successful calls"""
        @with_circuit_breaker("render_api")
        def success_func():
            return "success"

        result = success_func()
        assert result == "success"

        manager = get_circuit_breaker_manager()
        assert manager.get_status("render_api")["success_count"] >= 1

    def test_decorator_records_failure(self):
        """Test decorator records failed calls"""
        @with_circuit_breaker("render_api")
        def fail_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            fail_func()

        manager = get_circuit_breaker_manager()
        assert manager.get_status("render_api")["failure_count"] >= 1

    def test_decorator_blocks_when_open(self):
        """Test decorator raises error when circuit is open"""
        # Open the circuit first
        manager = get_circuit_breaker_manager()
        for i in range(5):
            manager.record_failure("render_api", 5000, "Error")

        @with_circuit_breaker("render_api")
        def blocked_func():
            return "should not run"

        with pytest.raises(CircuitOpenError):
            blocked_func()

    def test_decorator_with_fallback(self):
        """Test decorator uses fallback when circuit is open"""
        # Open the circuit
        manager = get_circuit_breaker_manager()
        for i in range(5):
            manager.record_failure("render_api", 5000, "Error")

        @with_circuit_breaker("render_api", fallback=lambda: "fallback_value")
        def func_with_fallback():
            return "should not run"

        result = func_with_fallback()
        assert result == "fallback_value"


class TestCircuitBreakerContext:
    """Tests for the CircuitBreakerContext context manager"""

    def setup_method(self):
        """Reset circuits before each test"""
        manager = get_circuit_breaker_manager()
        manager.reset("vercel_api")

    def test_context_manager_success(self):
        """Test context manager with successful operation"""
        with CircuitBreakerContext("vercel_api") as cb:
            time.sleep(0.01)  # Simulate work
            cb.mark_success()

        manager = get_circuit_breaker_manager()
        assert manager.get_status("vercel_api")["success_count"] >= 1

    def test_context_manager_failure(self):
        """Test context manager with failed operation"""
        with CircuitBreakerContext("vercel_api") as cb:
            cb.mark_failure("Test failure")

        manager = get_circuit_breaker_manager()
        assert manager.get_status("vercel_api")["failure_count"] >= 1

    def test_context_manager_exception(self):
        """Test context manager with exception"""
        try:
            with CircuitBreakerContext("vercel_api") as cb:
                raise ValueError("Test error")
        except ValueError:
            pass

        manager = get_circuit_breaker_manager()
        assert manager.get_status("vercel_api")["failure_count"] >= 1

    def test_context_manager_raises_when_open(self):
        """Test context manager raises when circuit is open"""
        # Open circuit
        manager = get_circuit_breaker_manager()
        for i in range(5):
            manager.record_failure("vercel_api", 5000, "Error")

        with pytest.raises(CircuitOpenError):
            with CircuitBreakerContext("vercel_api"):
                pass


class TestAsyncDecorator:
    """Tests for async circuit breaker decorator"""

    def setup_method(self):
        """Reset circuits before each test"""
        manager = get_circuit_breaker_manager()
        manager.reset("mcp_bridge")

    @pytest.mark.asyncio
    async def test_async_decorator_success(self):
        """Test async decorator records success"""
        @with_circuit_breaker("mcp_bridge")
        async def async_func():
            await asyncio.sleep(0.01)
            return "async_success"

        result = await async_func()
        assert result == "async_success"

        manager = get_circuit_breaker_manager()
        assert manager.get_status("mcp_bridge")["success_count"] >= 1

    @pytest.mark.asyncio
    async def test_async_decorator_failure(self):
        """Test async decorator records failure"""
        @with_circuit_breaker("mcp_bridge")
        async def async_fail_func():
            raise ValueError("Async error")

        with pytest.raises(ValueError):
            await async_fail_func()

        manager = get_circuit_breaker_manager()
        assert manager.get_status("mcp_bridge")["failure_count"] >= 1


class TestConfiguration:
    """Tests for circuit breaker configuration"""

    def test_all_services_configured(self):
        """Test that all expected services are configured"""
        expected_services = [
            "openai", "anthropic", "gemini", "huggingface",
            "database", "database_backup",
            "webhook_gumroad", "webhook_stripe", "webhook_github",
            "erp_api", "brainops_backend", "mcp_bridge",
            "resend_email", "render_api", "vercel_api"
        ]
        for service in expected_services:
            assert service in CIRCUIT_BREAKER_CONFIG, f"Missing config for {service}"

    def test_critical_services_marked(self):
        """Test that critical services are properly marked"""
        critical_services = ["openai", "anthropic", "database", "database_backup",
                           "webhook_stripe", "erp_api", "brainops_backend"]
        for service in critical_services:
            assert CIRCUIT_BREAKER_CONFIG[service].critical, f"{service} should be critical"

    def test_service_types_correct(self):
        """Test that service types are correctly assigned"""
        assert CIRCUIT_BREAKER_CONFIG["openai"].service_type == ServiceType.AI_PROVIDER
        assert CIRCUIT_BREAKER_CONFIG["database"].service_type == ServiceType.DATABASE
        assert CIRCUIT_BREAKER_CONFIG["webhook_stripe"].service_type == ServiceType.WEBHOOK
        assert CIRCUIT_BREAKER_CONFIG["erp_api"].service_type == ServiceType.API

    def test_database_has_faster_recovery(self):
        """Test that database has faster recovery than AI providers"""
        db_timeout = CIRCUIT_BREAKER_CONFIG["database"].recovery_timeout
        ai_timeout = CIRCUIT_BREAKER_CONFIG["openai"].recovery_timeout
        assert db_timeout < ai_timeout, "Database should have faster recovery"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
