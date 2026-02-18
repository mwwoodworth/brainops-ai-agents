#!/usr/bin/env python3
"""Compatibility shim.

The former ConsciousnessLoop was replaced by the operational monitor.
Use `operational_monitor.OperationalMonitor` for all new code.
"""

from __future__ import annotations

import os
from typing import Optional

from operational_monitor import OperationalMonitor as ConsciousnessLoop
from operational_monitor import get_operational_monitor


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _resolve_database_url(explicit_database_url: Optional[str]) -> Optional[str]:
    """Legacy helper kept for compatibility with existing tests/importers."""
    explicit = (explicit_database_url or "").strip()
    if explicit:
        return explicit

    db_url = _env("DATABASE_URL")
    if db_url:
        return db_url

    host = _env("DB_HOST")
    port = _env("DB_PORT") or "5432"
    name = _env("DB_NAME")
    user = _env("DB_USER")
    password = _env("DB_PASSWORD")
    if host and name and user and password:
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"

    env = (_env("ENVIRONMENT") or "production").lower()
    if env in {"production", "prod"}:
        raise RuntimeError("consciousness_loop requires DATABASE_URL in production")
    return None


__all__ = ["ConsciousnessLoop", "get_operational_monitor", "_resolve_database_url"]
