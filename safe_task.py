"""
Safe Task Utility - Centralized fire-and-forget task handling
Prevents "Future exception was never retrieved" errors throughout the AI OS
"""
import asyncio
import logging
from typing import Coroutine, Optional, Callable, Any

logger = logging.getLogger(__name__)

# Track all background tasks for graceful shutdown
_background_tasks: set = set()


def create_safe_task(
    coro: Coroutine,
    name: str = "background_task",
    on_error: Optional[Callable[[Exception], Any]] = None
) -> asyncio.Task:
    """
    Create an asyncio task with exception handling to prevent
    'Future exception was never retrieved' errors.

    Args:
        coro: The coroutine to run
        name: Name for logging/debugging
        on_error: Optional callback when an error occurs

    Returns:
        The created task (can be ignored for fire-and-forget)
    """
    async def wrapped():
        try:
            return await coro
        except asyncio.CancelledError:
            logger.debug(f"Task '{name}' was cancelled")
            raise  # Re-raise CancelledError for proper cleanup
        except Exception as e:
            logger.error(f"Error in background task '{name}': {e}", exc_info=True)
            if on_error:
                try:
                    on_error(e)
                except Exception as callback_error:
                    logger.error(f"Error in on_error callback for '{name}': {callback_error}")
            return None

    task = asyncio.create_task(wrapped(), name=name)

    # Track task and auto-remove when done
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return task


def fire_and_forget(coro: Coroutine, name: str = "fire_and_forget") -> None:
    """
    Fire-and-forget a coroutine with proper exception handling.
    Use this when you don't need the task reference.

    Args:
        coro: The coroutine to run
        name: Name for logging
    """
    create_safe_task(coro, name=name)


async def cancel_all_background_tasks(timeout: float = 5.0) -> None:
    """
    Cancel all tracked background tasks gracefully.
    Call this during application shutdown.

    Args:
        timeout: Maximum time to wait for tasks to cancel
    """
    if not _background_tasks:
        return

    logger.info(f"Cancelling {len(_background_tasks)} background tasks...")

    for task in _background_tasks:
        task.cancel()

    # Wait for all tasks to complete cancellation
    if _background_tasks:
        done, pending = await asyncio.wait(
            _background_tasks,
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        if pending:
            logger.warning(f"{len(pending)} tasks did not cancel within {timeout}s")


def get_background_task_count() -> int:
    """Get count of active background tasks."""
    return len(_background_tasks)


def get_background_task_names() -> list:
    """Get names of all active background tasks."""
    return [t.get_name() for t in _background_tasks if not t.done()]
