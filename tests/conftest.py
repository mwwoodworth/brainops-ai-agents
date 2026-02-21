import inspect
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as ai_app  # noqa: E402
import api.brain as brain_api  # noqa: E402
import api.revenue as revenue_api  # noqa: E402
import api.revenue_automation as revenue_automation_api  # noqa: E402
import api.taskmate as taskmate_api  # noqa: E402
import database.async_connection as db_async  # noqa: E402
import api.health as health_api  # noqa: E402
import services.db_health as db_health_svc  # noqa: E402


def pytest_addoption(parser):
    """Provide a local fallback for --timeout when pytest-timeout isn't installed."""
    try:
        parser.addoption(
            "--timeout",
            action="store",
            default=None,
            help="Per-test timeout in seconds (noop fallback option).",
        )
    except ValueError:
        # Option already provided by pytest-timeout plugin.
        pass


class _LocalTestTimeoutError(TimeoutError):
    """Raised when local timeout fallback expires."""


def _local_timeout_seconds(config) -> float:
    timeout = config.getoption("timeout")
    try:
        return float(timeout) if timeout is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _use_local_timeout(config) -> bool:
    return (
        not config.pluginmanager.hasplugin("timeout")
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
        and _local_timeout_seconds(config) > 0
    )


class _LocalTimeoutGuard:
    def __init__(self, seconds: float, nodeid: str, phase: str):
        self.seconds = seconds
        self.nodeid = nodeid
        self.phase = phase
        self.previous_handler = None

    def __enter__(self):
        def _raise_timeout(_signum, _frame):
            raise _LocalTestTimeoutError(
                f"{self.nodeid} exceeded --timeout={self.seconds:g}s during {self.phase}"
            )

        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def _timeout_guard(item, phase: str):
    return _LocalTimeoutGuard(_local_timeout_seconds(item.config), item.nodeid, phase)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Enforce timeout during fixture setup when pytest-timeout is unavailable."""
    if not _use_local_timeout(item.config):
        yield
        return
    with _timeout_guard(item, "setup"):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Enforce timeout during test call when pytest-timeout is unavailable."""
    if not _use_local_timeout(item.config):
        yield
        return
    with _timeout_guard(item, "call"):
        yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item):
    """Enforce timeout during fixture teardown when pytest-timeout is unavailable."""
    if item.config.pluginmanager.hasplugin("timeout"):
        yield
        return
    if not _use_local_timeout(item.config):
        yield
        return
    with _timeout_guard(item, "teardown"):
        yield


class MockTenantScopedPool:
    """Configurable async pool stub for endpoint tests."""

    def __init__(self):
        self.fetch_handler = lambda _query, *_args: []
        self.fetchrow_handler = lambda _query, *_args: None
        self.fetchval_handler = lambda _query, *_args: None
        self.execute_handler = lambda _query, *_args: "OK"
        self.test_connection_handler = lambda: True
        self.calls = {
            "fetch": [],
            "fetchrow": [],
            "fetchval": [],
            "execute": [],
            "test_connection": 0,
        }
        self.pool = SimpleNamespace(
            get_min_size=lambda: 1,
            get_max_size=lambda: 10,
            get_size=lambda: 1,
            get_idle_size=lambda: 1,
        )

    async def _resolve(self, value):
        if inspect.isawaitable(value):
            return await value
        return value

    async def fetch(self, query, *args):
        self.calls["fetch"].append((query, args))
        return await self._resolve(self.fetch_handler(query, *args))

    async def fetchrow(self, query, *args):
        self.calls["fetchrow"].append((query, args))
        return await self._resolve(self.fetchrow_handler(query, *args))

    async def fetchval(self, query, *args):
        self.calls["fetchval"].append((query, args))
        return await self._resolve(self.fetchval_handler(query, *args))

    async def execute(self, query, *args):
        self.calls["execute"].append((query, args))
        return await self._resolve(self.execute_handler(query, *args))

    async def test_connection(self):
        self.calls["test_connection"] += 1
        return await self._resolve(self.test_connection_handler())


@pytest.fixture(scope="session")
def api_key() -> str:
    return "test-key"


@pytest.fixture(scope="session")
def configure_test_security(api_key):
    original_auth_required = ai_app.config.security.auth_required
    original_auth_configured = ai_app.config.security.auth_configured
    original_app_keys = set(ai_app.config.security.valid_api_keys)

    module_sets = [
        brain_api.VALID_API_KEYS,
        revenue_api.VALID_API_KEYS,
        revenue_automation_api.VALID_API_KEYS,
    ]
    original_module_keys = [set(s) for s in module_sets]

    ai_app.config.security.valid_api_keys.clear()
    ai_app.config.security.valid_api_keys.add(api_key)
    ai_app.config.security.auth_required = True
    ai_app.config.security.auth_configured = True

    for key_set in module_sets:
        key_set.clear()
        key_set.add(api_key)

    yield

    ai_app.config.security.valid_api_keys.clear()
    ai_app.config.security.valid_api_keys.update(original_app_keys)
    ai_app.config.security.auth_required = original_auth_required
    ai_app.config.security.auth_configured = original_auth_configured

    for key_set, original in zip(module_sets, original_module_keys):
        key_set.clear()
        key_set.update(original)


@pytest_asyncio.fixture
async def client(configure_test_security):
    class _NoopMetrics:
        async def record(self, **_kwargs):
            return None

        def snapshot(self):
            return {"sample_size": 0}

    class _NoopCache:
        async def get_or_set(self, _key, _ttl_seconds, loader):
            return await loader(), False

        def snapshot(self):
            return {"size": 0}

    original_metrics = ai_app.REQUEST_METRICS
    original_cache = ai_app.RESPONSE_CACHE
    original_category_is_allowed = ai_app._category_is_allowed

    async def _allow_all_rate_limits(*_args, **_kwargs):
        return True

    ai_app.REQUEST_METRICS = _NoopMetrics()
    ai_app.RESPONSE_CACHE = _NoopCache()
    ai_app._category_is_allowed = _allow_all_rate_limits

    try:
        transport = httpx.ASGITransport(app=ai_app.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
            yield async_client
    finally:
        ai_app.REQUEST_METRICS = original_metrics
        ai_app.RESPONSE_CACHE = original_cache
        ai_app._category_is_allowed = original_category_is_allowed


@pytest.fixture
def auth_headers(api_key):
    return {"X-API-Key": api_key}


@pytest.fixture
def tenant_ids():
    return {
        "tenant_a": "51e728c5-94e8-4ae0-8a0a-6a08d1fb3457",
        "tenant_b": "a17d1f59-7baf-4350-b0c1-1ea6ae2fbd2a",
    }


@pytest.fixture
def sample_brain_entry():
    return {
        "key": "test:context",
        "value": {"message": "hello"},
        "category": "system",
        "priority": "high",
        "source": "api",
        "metadata": {"scope": "test"},
    }


@pytest.fixture
def sample_task():
    return {
        "task_id": "P7-TRUTH-001",
        "title": "Validate production truth",
        "description": "Run audit checks",
        "priority": "P1",
        "status": "open",
        "owner": "ops",
    }


@pytest.fixture
def mock_tenant_pool():
    return MockTenantScopedPool()


@pytest.fixture
def patch_pool(monkeypatch, mock_tenant_pool):
    monkeypatch.setattr(ai_app, "get_pool", lambda: mock_tenant_pool)
    monkeypatch.setattr(taskmate_api, "get_pool", lambda: mock_tenant_pool)
    monkeypatch.setattr(revenue_api, "get_pool", lambda: mock_tenant_pool)
    monkeypatch.setattr(db_async, "get_pool", lambda: mock_tenant_pool)
    monkeypatch.setattr(health_api, "get_pool", lambda: mock_tenant_pool)
    monkeypatch.setattr(db_health_svc, "get_pool", lambda: mock_tenant_pool)
    return mock_tenant_pool
