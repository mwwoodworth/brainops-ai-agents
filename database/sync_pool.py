"""
Unified Sync Connection Pool
============================
ALL modules that need sync DB connections MUST use this.
Do NOT create your own psycopg2 connections!
"""

import logging
import threading
import os
from typing import Optional, Any, Dict, List
from queue import Queue, Empty
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configuration - MUST come from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "aws-0-us-east-2.pooler.supabase.com"),
    "database": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres.yomagoqdmxszqtdwuhab"),
    "password": os.getenv("DB_PASSWORD"),  # No default - must be set
    "port": int(os.getenv("DB_PORT", "5432"))
}

# Validate password is set
if not DB_CONFIG["password"]:
    logger.warning("⚠️ DB_PASSWORD environment variable not set - sync pool may fail")

# Pool settings - CRITICAL: Keep these low to prevent MaxClientsInSessionMode
# Supabase session mode has limited connections - sync+async pools share the limit
POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 3
CONNECTION_TIMEOUT = 30
IDLE_TIMEOUT = 60


class SyncConnectionPool:
    """
    Thread-safe sync connection pool.
    Prevents connection exhaustion by reusing connections.
    """

    _instance: Optional['SyncConnectionPool'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._pool: Queue = Queue(maxsize=POOL_MAX_SIZE)
        self._size = 0
        self._pool_lock = threading.Lock()
        self._initialized = True
        self._last_error: Optional[str] = None

        # Pre-create minimum connections
        for _ in range(POOL_MIN_SIZE):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn)
                    self._size += 1
            except Exception as e:
                logger.warning(f"Failed to pre-create connection: {e}")

        logger.info(f"✅ Sync connection pool initialized (size: {self._size})")

    def _create_connection(self):
        """Create a new database connection."""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=DB_CONFIG["host"],
                database=DB_CONFIG["database"],
                user=DB_CONFIG["user"],
                password=DB_CONFIG["password"],
                port=DB_CONFIG["port"],
                connect_timeout=CONNECTION_TIMEOUT
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"Failed to create connection: {e}")
            return None

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        ALWAYS use with 'with' statement to ensure proper return.

        Usage:
            with sync_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")

        NOTE: Fixed "generator didn't stop after throw()" error by removing
        yield statements from except blocks. Context manager generators must
        not yield after catching exceptions - they should either re-raise
        or return normally.
        """
        conn = None
        acquired = False

        # Acquire connection before entering the context
        try:
            # Try to get from pool
            try:
                conn = self._pool.get_nowait()
                acquired = True
            except Empty:
                # Pool empty, try to create new if under limit
                with self._pool_lock:
                    if self._size < POOL_MAX_SIZE:
                        conn = self._create_connection()
                        if conn:
                            self._size += 1
                            acquired = True
                        else:
                            # Wait for one to become available
                            conn = self._pool.get(timeout=CONNECTION_TIMEOUT)
                            acquired = True
                    else:
                        # At limit, must wait
                        conn = self._pool.get(timeout=CONNECTION_TIMEOUT)
                        acquired = True

            # Verify connection is still valid
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.close()
                except Exception as exc:
                    logger.debug("Connection check failed; recreating connection: %s", exc, exc_info=True)
                    # Connection dead, create new
                    try:
                        conn.close()
                    except Exception:
                        logger.debug("Failed to close dead connection", exc_info=True)
                    conn = self._create_connection()
                    if not conn:
                        acquired = False

        except Empty:
            logger.error("Connection pool exhausted - all connections in use")
            conn = None
            acquired = False
        except Exception as e:
            logger.error(f"Connection pool error during acquisition: {e}")
            conn = None
            acquired = False

        # Now yield exactly once - this is the only yield in the generator
        try:
            yield conn
        finally:
            # Return connection to pool
            if conn and acquired:
                try:
                    if not conn.closed:
                        self._pool.put_nowait(conn)
                    else:
                        # Connection was closed, decrease size
                        with self._pool_lock:
                            self._size = max(0, self._size - 1)
                except Exception:
                    logger.debug("Failed to return connection to pool", exc_info=True)

    def execute(self, query: str, params: tuple = None) -> bool:
        """Execute a query without returning results."""
        with self.get_connection() as conn:
            if not conn:
                return False
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                cursor.close()
                return True
            except Exception as e:
                logger.error(f"Execute failed: {e}")
                return False

    def fetchone(self, query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
        """Fetch one row as a dictionary."""
        with self.get_connection() as conn:
            if not conn:
                return None
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                if row:
                    columns = [desc[0] for desc in cursor.description]
                    cursor.close()
                    return dict(zip(columns, row))
                cursor.close()
                return None
            except Exception as e:
                logger.error(f"Fetchone failed: {e}")
                return None

    def fetchall(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Fetch all rows as list of dictionaries."""
        with self.get_connection() as conn:
            if not conn:
                return []
            try:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                cursor.close()
                return [dict(zip(columns, row)) for row in rows]
            except Exception as e:
                logger.error(f"Fetchall failed: {e}")
                return []

    def get_status(self) -> Dict[str, Any]:
        """Get pool status."""
        return {
            "initialized": self._initialized,
            "pool_size": self._size,
            "available": self._pool.qsize(),
            "in_use": self._size - self._pool.qsize(),
            "max_size": POOL_MAX_SIZE,
            "last_error": self._last_error
        }

    def close_all(self):
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if conn:
                    conn.close()
            except Exception:
                logger.debug("Failed to close connection during pool shutdown", exc_info=True)
        self._size = 0
        logger.info("Sync connection pool closed")


# Singleton getter
def get_sync_pool() -> SyncConnectionPool:
    """Get the shared sync connection pool."""
    return SyncConnectionPool()


# Convenience functions for modules to use
def sync_execute(query: str, params: tuple = None) -> bool:
    """Execute a query using shared pool."""
    return get_sync_pool().execute(query, params)


def sync_fetchone(query: str, params: tuple = None) -> Optional[Dict[str, Any]]:
    """Fetch one row using shared pool."""
    return get_sync_pool().fetchone(query, params)


def sync_fetchall(query: str, params: tuple = None) -> List[Dict[str, Any]]:
    """Fetch all rows using shared pool."""
    return get_sync_pool().fetchall(query, params)
