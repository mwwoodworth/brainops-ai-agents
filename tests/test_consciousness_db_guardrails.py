import os

import pytest

# Prevent unrelated import-time DB config guards from aborting module import.
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "agent_worker")
os.environ.setdefault("DB_PASSWORD", "test-password")
os.environ.setdefault("DB_NAME", "postgres")

import consciousness_loop as cl


def _clear_db_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("DATABASE_URL", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        monkeypatch.delenv(key, raising=False)


def test_resolve_database_url_requires_explicit_config_in_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError, match="requires DATABASE_URL"):
        cl._resolve_database_url(None)


def test_resolve_database_url_allows_empty_in_non_production(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")

    assert cl._resolve_database_url(None) is None


def test_resolve_database_url_builds_from_explicit_components(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_db_env(monkeypatch)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("DB_HOST", "db.example.internal")
    monkeypatch.setenv("DB_PORT", "6543")
    monkeypatch.setenv("DB_NAME", "brainops")
    monkeypatch.setenv("DB_USER", "agent_worker")
    monkeypatch.setenv("DB_PASSWORD", "local-secret")

    resolved = cl._resolve_database_url(None)
    assert resolved == "postgresql://agent_worker:local-secret@db.example.internal:6543/brainops"
