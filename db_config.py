#!/usr/bin/env python3
"""
Centralized Database Configuration Module

This module provides database configuration with support for both:
1. Individual environment variables (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT)
2. DATABASE_URL fallback (for Render and other PaaS platforms)

Usage:
    from db_config import get_db_config, DB_CONFIG

    # Or use the function for lazy loading:
    config = get_db_config()
"""

import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_cached_config = None


def get_db_config() -> dict:
    """
    Get database configuration with validation.

    Supports both individual DB_* environment variables and DATABASE_URL fallback.
    Results are cached after first call.

    Returns:
        dict: Database configuration with host, database, user, password, port

    Raises:
        RuntimeError: If neither individual vars nor DATABASE_URL provide complete config
    """
    global _cached_config

    if _cached_config is not None:
        return _cached_config

    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME", "postgres")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "5432")

    # Fallback to DATABASE_URL if individual vars not set
    if not all([host, user, password]):
        database_url = os.getenv('DATABASE_URL', '')
        if database_url:
            try:
                parsed = urlparse(database_url)
                host = parsed.hostname or ''
                database = parsed.path.lstrip('/') if parsed.path else 'postgres'
                user = parsed.username or ''
                password = parsed.password or ''
                port = str(parsed.port or 5432)
                logger.info(f"Parsed DATABASE_URL: host={host}, db={database}")
            except Exception as e:
                logger.error(f"Failed to parse DATABASE_URL: {e}")

    if not all([host, user, password]):
        raise RuntimeError(
            "Database configuration is incomplete. "
            "Set DB_HOST, DB_USER, DB_PASSWORD environment variables, "
            "or provide DATABASE_URL."
        )

    # Auto-switch Supabase pooler to transaction mode (port 6543).
    # Session mode (5432) limits connections causing MaxClientsInSessionMode.
    if host and 'pooler.supabase.com' in host and int(port) == 5432:
        port = '6543'
        logger.info("Supabase pooler detected - using transaction mode (port 6543)")

    _cached_config = {
        "host": host,
        "database": database,
        "user": user,
        "password": password,
        "port": int(port)
    }

    return _cached_config


# Pre-load config for modules that import DB_CONFIG directly
# This will raise RuntimeError at import time if config is incomplete
try:
    DB_CONFIG = get_db_config()
except RuntimeError:
    # Allow import to succeed even without DB config (for testing/development)
    DB_CONFIG = None
    logger.warning("Database configuration not available - DB_CONFIG set to None")
