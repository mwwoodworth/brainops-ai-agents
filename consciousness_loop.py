#!/usr/bin/env python3
"""
Consciousness Loop (Guardrail-First Stub)
========================================
This module exists primarily to satisfy:
- `nerve_center.py` optional import of `ConsciousnessLoop`
- Unit tests that validate DB config guardrails via `_resolve_database_url`

The previous full implementation was archived/superseded. This version is:
- Import-safe (no DB work at import time)
- Strict in production (requires explicit `DATABASE_URL`)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _resolve_database_url(explicit_database_url: Optional[str]) -> Optional[str]:
    """
    Resolve database URL using explicit parameter or environment variables.

    Rules:
    - If `explicit_database_url` is provided, use it.
    - Else if `DATABASE_URL` is set, use it.
    - Else if individual DB_* components are present, build a URL.
    - Else:
      - In production: raise (must be explicit)
      - In non-production: return None
    """
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


@dataclass
class ConsciousnessLoop:
    """
    Minimal placeholder implementation.

    NerveCenter only needs the class to exist; the autonomic systems are disabled
    by default in config.
    """

    database_url: Optional[str] = None

    async def start(self) -> None:
        return

    async def stop(self) -> None:
        return

