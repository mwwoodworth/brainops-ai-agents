"""
Main Event Loop Bridge

BrainOps runs a mixture of:
- Uvicorn/FastAPI (main asyncio event loop)
- APScheduler jobs (threadpool / non-async contexts)
- Background threads (self-healing, revenue jobs, etc.)

Some legacy scheduler jobs created *new* event loops in worker threads and then
invoked async code that shares resources created on the main loop (e.g. asyncpg
pool, AgentExecutor). That can produce runtime errors like:
- "got Future attached to a different loop"
- asyncpg "another operation is in progress"
- ConnectionDoesNotExistError flaps during bursts

This module provides a safe bridge so thread-based jobs can schedule coroutines
onto the process' main asyncio loop (the one that created the shared async
resources) without creating new event loops.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine, Optional

logger = logging.getLogger(__name__)

_MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None
_MAIN_THREAD_ID: Optional[int] = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Register the main asyncio loop (usually Uvicorn's loop)."""
    global _MAIN_LOOP, _MAIN_THREAD_ID
    _MAIN_LOOP = loop
    _MAIN_THREAD_ID = threading.get_ident()


def has_main_loop() -> bool:
    loop = _MAIN_LOOP
    return loop is not None and loop.is_running()


def run_on_main_loop(coro: Coroutine[Any, Any, Any], *, timeout: Optional[float] = None) -> Any:
    """
    Run a coroutine on the registered main loop.

    - If called from a worker thread, it blocks waiting for the coroutine result.
    - If called from the main loop thread, it schedules and returns an asyncio.Task.
    - If no loop is registered, falls back to running the coroutine in a fresh loop
      (best-effort; avoids raising in local scripts).
    """
    loop = _MAIN_LOOP
    if loop is None or not loop.is_running():
        # Best-effort fallback for local scripts where the FastAPI app isn't running.
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Already in an event loop but no main loop registered; schedule locally.
            return asyncio.create_task(coro)

    if _MAIN_THREAD_ID == threading.get_ident():
        # We're on the main loop thread; never block here (would deadlock).
        return asyncio.create_task(coro)

    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return fut.result(timeout=timeout)
    except Exception:
        # Ensure the underlying task is cancelled on timeout/error to avoid leaks.
        try:
            fut.cancel()
        except Exception:
            pass
        raise

