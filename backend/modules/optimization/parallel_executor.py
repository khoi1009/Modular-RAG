"""
Parallel executor for concurrent task execution with resource limits.

Manages concurrent execution of tasks with configurable concurrency limits,
timeouts, and fallback mechanisms.
"""

import asyncio
from typing import Any, Callable, Dict, List, Tuple

from fastapi import HTTPException

from backend.logger import logger

try:
    import async_timeout

    ASYNC_TIMEOUT_AVAILABLE = True
except ImportError:
    ASYNC_TIMEOUT_AVAILABLE = False
    logger.warning("async_timeout not available, using asyncio.wait_for instead")


class ParallelExecutor:
    """
    Parallel task executor with concurrency control.

    Executes multiple async tasks in parallel while respecting
    concurrency limits and handling timeouts gracefully.
    """

    def __init__(self, max_concurrency: int = 10):
        """
        Initialize parallel executor.

        Args:
            max_concurrency: Maximum number of concurrent tasks
        """
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_concurrency = max_concurrency

        logger.info(f"ParallelExecutor initialized: max_concurrency={max_concurrency}")

    async def execute_parallel(
        self,
        tasks: List[Tuple[Callable, Dict]],
        timeout_sec: float = 30.0,
        return_exceptions: bool = True,
    ) -> List[Any]:
        """
        Execute tasks in parallel with concurrency limit.

        Args:
            tasks: List of (async_function, kwargs_dict) tuples
            timeout_sec: Timeout for the entire parallel execution
            return_exceptions: If True, exceptions are returned as results

        Returns:
            List of results (or exceptions if return_exceptions=True)
        """
        if not tasks:
            return []

        logger.debug(
            f"Executing {len(tasks)} tasks in parallel "
            f"(max_concurrency={self.max_concurrency}, timeout={timeout_sec}s)"
        )

        async def run_with_semaphore(fn: Callable, kwargs: Dict) -> Any:
            """Execute single task with semaphore."""
            async with self.semaphore:
                return await fn(**kwargs)

        try:
            # Execute all tasks with timeout
            if ASYNC_TIMEOUT_AVAILABLE:
                async with async_timeout.timeout(timeout_sec):
                    results = await asyncio.gather(
                        *[run_with_semaphore(fn, kwargs) for fn, kwargs in tasks],
                        return_exceptions=return_exceptions,
                    )
            else:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[run_with_semaphore(fn, kwargs) for fn, kwargs in tasks],
                        return_exceptions=return_exceptions,
                    ),
                    timeout=timeout_sec,
                )

            # Count successes and failures
            if return_exceptions:
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                error_count = len(results) - success_count
                logger.debug(
                    f"Parallel execution completed: {success_count} succeeded, "
                    f"{error_count} failed"
                )
            else:
                logger.debug(f"Parallel execution completed: {len(results)} tasks")

            return results

        except asyncio.TimeoutError:
            logger.error(f"Parallel execution timed out after {timeout_sec}s")
            raise HTTPException(
                status_code=504, detail="Parallel execution timed out"
            )

    async def execute_with_fallback(
        self,
        primary: Tuple[Callable, Dict],
        fallbacks: List[Tuple[Callable, Dict]],
        timeout_sec: float = 10.0,
    ) -> Any:
        """
        Execute primary task, fall back to alternatives on failure.

        Tries the primary task first, then tries each fallback in order
        until one succeeds or all fail.

        Args:
            primary: (async_function, kwargs_dict) for primary task
            fallbacks: List of (async_function, kwargs_dict) for fallback tasks
            timeout_sec: Timeout per task attempt

        Returns:
            Result from first successful task

        Raises:
            HTTPException: If all tasks fail
        """
        primary_fn, primary_kwargs = primary

        # Try primary task
        try:
            logger.debug("Executing primary task")
            if ASYNC_TIMEOUT_AVAILABLE:
                async with async_timeout.timeout(timeout_sec):
                    result = await primary_fn(**primary_kwargs)
            else:
                result = await asyncio.wait_for(
                    primary_fn(**primary_kwargs), timeout=timeout_sec
                )

            logger.debug("Primary task succeeded")
            return result

        except Exception as e:
            logger.warning(f"Primary task failed: {e}, trying fallbacks")

        # Try fallbacks
        for i, (fallback_fn, fallback_kwargs) in enumerate(fallbacks):
            try:
                logger.debug(f"Executing fallback {i + 1}/{len(fallbacks)}")
                if ASYNC_TIMEOUT_AVAILABLE:
                    async with async_timeout.timeout(timeout_sec):
                        result = await fallback_fn(**fallback_kwargs)
                else:
                    result = await asyncio.wait_for(
                        fallback_fn(**fallback_kwargs), timeout=timeout_sec
                    )

                logger.debug(f"Fallback {i + 1} succeeded")
                return result

            except Exception as e:
                logger.warning(f"Fallback {i + 1} failed: {e}")

        # All tasks failed
        logger.error("All execution paths failed (primary + all fallbacks)")
        raise HTTPException(
            status_code=500, detail="All execution paths failed"
        )

    async def execute_first_completed(
        self,
        tasks: List[Tuple[Callable, Dict]],
        timeout_sec: float = 10.0,
    ) -> Any:
        """
        Execute tasks in parallel, return first successful result.

        Useful for racing multiple approaches to find fastest result.

        Args:
            tasks: List of (async_function, kwargs_dict) tuples
            timeout_sec: Overall timeout

        Returns:
            First successful result

        Raises:
            HTTPException: If all tasks fail or timeout
        """
        if not tasks:
            raise ValueError("No tasks provided")

        logger.debug(f"Racing {len(tasks)} tasks, returning first success")

        async def run_task(fn: Callable, kwargs: Dict) -> Any:
            """Wrapper to run single task."""
            async with self.semaphore:
                return await fn(**kwargs)

        pending = [asyncio.create_task(run_task(fn, kw)) for fn, kw in tasks]

        try:
            if ASYNC_TIMEOUT_AVAILABLE:
                async with async_timeout.timeout(timeout_sec):
                    while pending:
                        done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )

                        for task in done:
                            try:
                                result = task.result()
                                # Cancel remaining tasks
                                for p in pending:
                                    p.cancel()
                                logger.debug("First task completed successfully")
                                return result
                            except Exception as e:
                                logger.warning(f"Task failed: {e}")
            else:
                deadline = asyncio.get_event_loop().time() + timeout_sec
                while pending:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    done, pending = await asyncio.wait(
                        pending,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=remaining,
                    )

                    for task in done:
                        try:
                            result = task.result()
                            for p in pending:
                                p.cancel()
                            logger.debug("First task completed successfully")
                            return result
                        except Exception as e:
                            logger.warning(f"Task failed: {e}")

            raise HTTPException(status_code=500, detail="All tasks failed")

        except asyncio.TimeoutError:
            for task in pending:
                task.cancel()
            logger.error(f"First-completed execution timed out after {timeout_sec}s")
            raise HTTPException(status_code=504, detail="Execution timed out")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get executor statistics.

        Returns:
            Dictionary with executor metrics
        """
        # Get current semaphore state (available permits)
        locked = self.semaphore.locked()
        return {
            "max_concurrency": self.max_concurrency,
            "semaphore_locked": locked,
        }
