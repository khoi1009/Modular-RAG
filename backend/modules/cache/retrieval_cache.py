"""
Retrieval result cache with TTL-based invalidation.

Caches document retrieval results to avoid redundant vector database queries
for the same query and configuration.
"""

import hashlib
import time
from typing import Any, Dict, List, Optional

from backend.logger import logger


class RetrievalCache:
    """
    Cache for document retrieval results.

    Stores retrieved documents with TTL-based expiration to ensure
    freshness while avoiding redundant database queries.
    """

    def __init__(self, ttl_seconds: int = 1800, max_size: int = 5000):
        """
        Initialize retrieval cache.

        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
            max_size: Maximum number of cached entries
        """
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {result, timestamp}

        logger.info(
            f"RetrievalCache initialized: ttl={ttl_seconds}s, max_size={max_size}"
        )

    def _make_key(
        self, query: str, collection: str, retriever_config_hash: str
    ) -> str:
        """
        Generate cache key for retrieval request.

        Args:
            query: User query text
            collection: Collection name
            retriever_config_hash: Hash of retriever configuration

        Returns:
            SHA256-based cache key
        """
        content = f"{query}:{collection}:{retriever_config_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """
        Check if cached entry is expired.

        Args:
            timestamp: Entry creation timestamp

        Returns:
            True if expired, False otherwise
        """
        return (time.time() - timestamp) > self.ttl

    async def get(
        self, query: str, collection: str, retriever_config_hash: str
    ) -> Optional[List[Any]]:
        """
        Get cached retrieval results.

        Args:
            query: User query text
            collection: Collection name
            retriever_config_hash: Hash of retriever configuration

        Returns:
            Cached retrieval results or None if not found/expired
        """
        key = self._make_key(query, collection, retriever_config_hash)

        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry["timestamp"]):
                logger.debug(f"Retrieval cache hit: {key[:16]}...")
                return entry["result"]
            else:
                # Remove expired entry
                del self.cache[key]
                logger.debug(f"Retrieval cache expired: {key[:16]}...")

        logger.debug(f"Retrieval cache miss: {key[:16]}...")
        return None

    async def set(
        self, query: str, collection: str, retriever_config_hash: str, result: List[Any]
    ) -> None:
        """
        Store retrieval results in cache.

        Args:
            query: User query text
            collection: Collection name
            retriever_config_hash: Hash of retriever configuration
            result: Retrieval results to cache
        """
        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            # Find oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]
            logger.debug(f"Retrieval cache evicted: {oldest_key[:16]}...")

        key = self._make_key(query, collection, retriever_config_hash)
        self.cache[key] = {"result": result, "timestamp": time.time()}

        logger.debug(f"Retrieval cached: {key[:16]}...")

    async def invalidate_collection(self, collection: str) -> int:
        """
        Invalidate all cached entries for a collection.

        Args:
            collection: Collection name

        Returns:
            Number of entries invalidated
        """
        keys_to_remove = []
        for key in self.cache.keys():
            # Check if this key belongs to the collection
            # We need to store collection info in cache or iterate all
            # For now, we'll clear all which is safer
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache[key]

        count = len(keys_to_remove)
        logger.info(f"Retrieval cache invalidated for collection '{collection}': {count} entries")
        return count

    async def clear(self) -> None:
        """Clear all cached entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Retrieval cache cleared: {count} entries")

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        keys_to_remove = [
            key
            for key, entry in self.cache.items()
            if self._is_expired(entry["timestamp"])
        ]

        for key in keys_to_remove:
            del self.cache[key]

        if keys_to_remove:
            logger.debug(f"Retrieval cache cleanup: {len(keys_to_remove)} expired entries removed")

        return len(keys_to_remove)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        current_time = time.time()
        expired_count = sum(
            1
            for entry in self.cache.values()
            if self._is_expired(entry["timestamp"])
        )

        return {
            "total_entries": len(self.cache),
            "expired_entries": expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
        }
