"""
Cache invalidation strategies and utilities.

Provides manual and automatic cache invalidation mechanisms including
TTL-based expiration, event-driven invalidation, and scheduled cleanup.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from backend.logger import logger


class CacheInvalidationStrategy:
    """Base class for cache invalidation strategies."""

    async def should_invalidate(self, entry: Dict[str, Any]) -> bool:
        """
        Determine if cache entry should be invalidated.

        Args:
            entry: Cache entry with metadata

        Returns:
            True if entry should be invalidated
        """
        raise NotImplementedError


class TTLInvalidationStrategy(CacheInvalidationStrategy):
    """Time-to-live based invalidation."""

    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = ttl_seconds

    async def should_invalidate(self, entry: Dict[str, Any]) -> bool:
        """Check if entry has exceeded TTL."""
        if "timestamp" not in entry:
            return False

        age = datetime.now().timestamp() - entry["timestamp"]
        return age > self.ttl_seconds


class ManualInvalidationManager:
    """
    Manager for manual cache invalidation operations.

    Provides API for explicit cache invalidation by collection, query,
    or pattern matching.
    """

    def __init__(self, cache_manager: Any):
        """
        Initialize invalidation manager.

        Args:
            cache_manager: MultiLevelCache instance to manage
        """
        self.cache_manager = cache_manager
        self.invalidation_log: List[Dict[str, Any]] = []

    async def invalidate_by_collection(self, collection: str) -> Dict[str, Any]:
        """
        Invalidate all caches for a collection.

        Args:
            collection: Collection name

        Returns:
            Invalidation result summary
        """
        logger.info(f"Manual invalidation requested for collection: {collection}")

        await self.cache_manager.invalidate_collection(collection)

        result = {
            "collection": collection,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        self._log_invalidation(result)
        return result

    async def invalidate_by_query(
        self, query: str, collection: str, config_hash: str
    ) -> Dict[str, Any]:
        """
        Invalidate specific query from caches.

        Args:
            query: Query text
            collection: Collection name
            config_hash: Configuration hash

        Returns:
            Invalidation result summary
        """
        logger.info(f"Manual invalidation requested for query: {query[:50]}...")

        await self.cache_manager.invalidate_query(query, collection, config_hash)

        result = {
            "query": query[:100],
            "collection": collection,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        self._log_invalidation(result)
        return result

    async def invalidate_all(self) -> Dict[str, Any]:
        """
        Clear all caches.

        Returns:
            Invalidation result summary
        """
        logger.warning("Manual invalidation requested for ALL caches")

        await self.cache_manager.clear_all()

        result = {
            "scope": "all",
            "timestamp": datetime.now().isoformat(),
            "status": "success",
        }

        self._log_invalidation(result)
        return result

    def _log_invalidation(self, result: Dict[str, Any]) -> None:
        """
        Log invalidation event.

        Args:
            result: Invalidation result to log
        """
        self.invalidation_log.append(result)

        # Keep only last 1000 entries
        if len(self.invalidation_log) > 1000:
            self.invalidation_log = self.invalidation_log[-1000:]

    def get_invalidation_history(
        self, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent invalidation history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of invalidation events
        """
        return self.invalidation_log[-limit:]


class AutomaticCacheCleanup:
    """
    Automatic cache cleanup scheduler.

    Runs periodic cleanup tasks to remove expired entries and manage
    cache memory usage.
    """

    def __init__(
        self,
        cache_manager: Any,
        cleanup_interval_seconds: int = 3600,
        enable_auto_cleanup: bool = True,
    ):
        """
        Initialize automatic cleanup.

        Args:
            cache_manager: MultiLevelCache instance
            cleanup_interval_seconds: Interval between cleanup runs
            enable_auto_cleanup: Whether to enable automatic cleanup
        """
        self.cache_manager = cache_manager
        self.cleanup_interval = cleanup_interval_seconds
        self.enabled = enable_auto_cleanup
        self.cleanup_task: Optional[asyncio.Task] = None
        self.last_cleanup: Optional[datetime] = None
        self.cleanup_count = 0

        logger.info(
            f"AutomaticCacheCleanup initialized: interval={cleanup_interval_seconds}s, "
            f"enabled={enable_auto_cleanup}"
        )

    async def start(self) -> None:
        """Start automatic cleanup task."""
        if not self.enabled:
            logger.info("Automatic cache cleanup is disabled")
            return

        if self.cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return

        logger.info("Starting automatic cache cleanup task")
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop automatic cleanup task."""
        if self.cleanup_task is not None:
            logger.info("Stopping automatic cache cleanup task")
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Main cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._run_cleanup()
            except asyncio.CancelledError:
                logger.info("Cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def _run_cleanup(self) -> None:
        """Execute cleanup operations."""
        logger.info("Running automatic cache cleanup")

        try:
            # Cleanup expired entries in retrieval cache
            if self.cache_manager.retrieval_cache:
                removed = await self.cache_manager.retrieval_cache.cleanup_expired()
                if removed > 0:
                    logger.info(f"Removed {removed} expired retrieval cache entries")

            self.last_cleanup = datetime.now()
            self.cleanup_count += 1

            logger.info("Cache cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cleanup statistics.

        Returns:
            Dictionary with cleanup metrics
        """
        return {
            "enabled": self.enabled,
            "cleanup_interval_seconds": self.cleanup_interval,
            "last_cleanup": self.last_cleanup.isoformat() if self.last_cleanup else None,
            "cleanup_count": self.cleanup_count,
            "task_running": self.cleanup_task is not None and not self.cleanup_task.done(),
        }
