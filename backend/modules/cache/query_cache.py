"""
Query cache for exact match caching with optional Redis backend.

Provides two-level caching:
1. Local in-memory LRU cache for fastest access
2. Optional Redis cache for distributed/persistent storage
"""

import hashlib
import json
from typing import Any, Dict, Optional

from cachetools import LRUCache

from backend.logger import logger


class QueryCache:
    """
    Exact match query cache with local + Redis support.

    Caches complete query results keyed by (query, collection, config_hash).
    Uses local LRU cache as L1, Redis as optional L2.
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        ttl_seconds: int = 3600,
        max_size: int = 10000,
    ):
        """
        Initialize query cache.

        Args:
            redis_client: Optional Redis client (aioredis or redis.asyncio)
            ttl_seconds: Time-to-live for cached entries in seconds
            max_size: Maximum number of entries in local LRU cache
        """
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.local_cache: LRUCache = LRUCache(maxsize=max_size)
        logger.info(
            f"QueryCache initialized: ttl={ttl_seconds}s, max_size={max_size}, "
            f"redis={'enabled' if redis_client else 'disabled'}"
        )

    def _make_key(self, query: str, collection: str, config_hash: str) -> str:
        """
        Generate cache key from query parameters.

        Args:
            query: User query text
            collection: Collection name
            config_hash: Hash of retrieval configuration

        Returns:
            SHA256-based cache key with prefix
        """
        content = f"{query}:{collection}:{config_hash}"
        hash_digest = hashlib.sha256(content.encode()).hexdigest()
        return f"qcache:{hash_digest}"

    async def get(
        self, query: str, collection: str, config_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.

        Args:
            query: User query text
            collection: Collection name
            config_hash: Hash of retrieval configuration

        Returns:
            Cached result dict or None if not found
        """
        key = self._make_key(query, collection, config_hash)

        # L1: Check local cache first
        if key in self.local_cache:
            logger.debug(f"Query cache hit (local): {key[:16]}...")
            return self.local_cache[key]

        # L2: Check Redis if available
        if self.redis:
            try:
                cached = await self.redis.get(key)
                if cached:
                    result = json.loads(cached)
                    # Populate local cache for future requests
                    self.local_cache[key] = result
                    logger.debug(f"Query cache hit (redis): {key[:16]}...")
                    return result
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        logger.debug(f"Query cache miss: {key[:16]}...")
        return None

    async def set(
        self, query: str, collection: str, config_hash: str, result: Dict[str, Any]
    ) -> None:
        """
        Store query result in cache.

        Args:
            query: User query text
            collection: Collection name
            config_hash: Hash of retrieval configuration
            result: Query result to cache
        """
        key = self._make_key(query, collection, config_hash)

        # Store in local cache
        self.local_cache[key] = result

        # Store in Redis if available
        if self.redis:
            try:
                await self.redis.setex(
                    key, self.ttl, json.dumps(result, default=str)
                )
                logger.debug(f"Query cached (local+redis): {key[:16]}...")
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        else:
            logger.debug(f"Query cached (local only): {key[:16]}...")

    async def invalidate(self, query: str, collection: str, config_hash: str) -> None:
        """
        Invalidate specific cache entry.

        Args:
            query: User query text
            collection: Collection name
            config_hash: Hash of retrieval configuration
        """
        key = self._make_key(query, collection, config_hash)

        # Remove from local cache
        if key in self.local_cache:
            del self.local_cache[key]

        # Remove from Redis
        if self.redis:
            try:
                await self.redis.delete(key)
                logger.debug(f"Query cache invalidated: {key[:16]}...")
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")

    async def clear(self) -> None:
        """Clear all cached entries."""
        self.local_cache.clear()
        if self.redis:
            try:
                # Delete all keys with qcache: prefix
                cursor = 0
                while True:
                    cursor, keys = await self.redis.scan(
                        cursor, match="qcache:*", count=100
                    )
                    if keys:
                        await self.redis.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Query cache cleared (local+redis)")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        else:
            logger.info("Query cache cleared (local only)")
