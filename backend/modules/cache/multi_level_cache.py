"""
Multi-level cache coordinator for Cognita RAG system.

Orchestrates query cache, semantic cache, embedding cache, and retrieval cache
to maximize cache hit rates and minimize redundant computations.
"""

from typing import Any, Callable, Dict, Optional

from backend.logger import logger
from backend.modules.cache.embedding_cache import EmbeddingCache
from backend.modules.cache.query_cache import QueryCache
from backend.modules.cache.retrieval_cache import RetrievalCache
from backend.modules.cache.semantic_cache import SemanticCache
from backend.types import ConfiguredBaseModel


class CacheHitInfo(ConfiguredBaseModel):
    """Information about cache hit/miss."""

    level: str  # "query", "semantic", "none"
    exact: bool = True  # False for semantic cache hits
    similarity: Optional[float] = None  # Only for semantic hits


class CacheMetrics:
    """Metrics tracking for cache performance."""

    def __init__(self):
        self.hits_by_level: Dict[str, int] = {
            "query": 0,
            "semantic": 0,
        }
        self.misses: int = 0
        self.total_requests: int = 0

    def record_hit(self, level: str) -> None:
        """Record cache hit at specific level."""
        self.hits_by_level[level] = self.hits_by_level.get(level, 0) + 1
        self.total_requests += 1

    def record_miss(self) -> None:
        """Record cache miss."""
        self.misses += 1
        self.total_requests += 1

    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        total_hits = sum(self.hits_by_level.values())
        return total_hits / self.total_requests

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        total_hits = sum(self.hits_by_level.values())
        return {
            "total_requests": self.total_requests,
            "total_hits": total_hits,
            "total_misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "hits_by_level": self.hits_by_level.copy(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits_by_level.clear()
        self.misses = 0
        self.total_requests = 0


class MultiLevelCache:
    """
    Multi-level cache coordinator.

    Manages query cache (L1), semantic cache (L2), and other specialized caches
    in a coordinated fashion to maximize performance.
    """

    def __init__(
        self,
        query_cache: Optional[QueryCache] = None,
        semantic_cache: Optional[SemanticCache] = None,
        retrieval_cache: Optional[RetrievalCache] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
    ):
        """
        Initialize multi-level cache.

        Args:
            query_cache: Query cache instance (L1 - exact match)
            semantic_cache: Semantic cache instance (L2 - similar queries)
            retrieval_cache: Retrieval result cache
            embedding_cache: Embedding cache
        """
        self.query_cache = query_cache or QueryCache()
        self.semantic_cache = semantic_cache
        self.retrieval_cache = retrieval_cache
        self.embedding_cache = embedding_cache
        self.metrics = CacheMetrics()

        logger.info(
            f"MultiLevelCache initialized: "
            f"query_cache={query_cache is not None}, "
            f"semantic_cache={semantic_cache is not None}, "
            f"retrieval_cache={retrieval_cache is not None}, "
            f"embedding_cache={embedding_cache is not None}"
        )

    async def get_or_compute(
        self,
        query: str,
        collection: str,
        config_hash: str,
        compute_fn: Callable[[], Any],
    ) -> tuple[Any, CacheHitInfo]:
        """
        Get cached result or compute using provided function.

        Tries caches in order:
        1. Query cache (exact match)
        2. Semantic cache (similar query)
        3. Compute and cache

        Args:
            query: User query text
            collection: Collection name
            config_hash: Hash of configuration
            compute_fn: Async function to compute result if not cached

        Returns:
            Tuple of (result, cache_hit_info)
        """
        # Level 1: Exact query cache
        cached = await self.query_cache.get(query, collection, config_hash)
        if cached:
            self.metrics.record_hit("query")
            logger.debug("Multi-level cache: L1 hit (exact query)")
            return cached, CacheHitInfo(level="query", exact=True)

        # Level 2: Semantic cache (if available)
        if self.semantic_cache:
            similar = await self.semantic_cache.get_similar(query, collection)
            if similar:
                cached_query, result, similarity = similar
                self.metrics.record_hit("semantic")
                logger.debug(
                    f"Multi-level cache: L2 hit (semantic, similarity={similarity:.3f})"
                )
                # Also cache in query cache for future exact matches
                await self.query_cache.set(query, collection, config_hash, result)
                return result, CacheHitInfo(
                    level="semantic", exact=False, similarity=similarity
                )

        # Cache miss - compute result
        self.metrics.record_miss()
        logger.debug("Multi-level cache: miss, computing...")

        result = await compute_fn()

        # Store in all available caches
        await self.query_cache.set(query, collection, config_hash, result)

        if self.semantic_cache:
            await self.semantic_cache.add(query, collection, result)

        logger.debug("Multi-level cache: result computed and cached")
        return result, CacheHitInfo(level="none", exact=True)

    async def invalidate_query(
        self, query: str, collection: str, config_hash: str
    ) -> None:
        """
        Invalidate specific query from all caches.

        Args:
            query: Query text
            collection: Collection name
            config_hash: Configuration hash
        """
        await self.query_cache.invalidate(query, collection, config_hash)
        logger.debug(f"Invalidated query cache for: {query[:50]}...")

    async def invalidate_collection(self, collection: str) -> None:
        """
        Invalidate all cached entries for a collection.

        Args:
            collection: Collection name
        """
        if self.retrieval_cache:
            count = await self.retrieval_cache.invalidate_collection(collection)
            logger.info(f"Invalidated {count} retrieval cache entries for collection: {collection}")

        # Query and semantic caches would need collection tracking to invalidate
        # For now, we'll clear them completely as a safe option
        await self.query_cache.clear()
        if self.semantic_cache:
            await self.semantic_cache.clear()

        logger.info(f"Invalidated all caches for collection: {collection}")

    async def clear_all(self) -> None:
        """Clear all caches."""
        await self.query_cache.clear()

        if self.semantic_cache:
            await self.semantic_cache.clear()

        if self.retrieval_cache:
            await self.retrieval_cache.clear()

        if self.embedding_cache:
            await self.embedding_cache.clear()

        self.metrics.reset()
        logger.info("All caches cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with statistics from all cache levels
        """
        stats = {
            "overall_metrics": self.metrics.get_stats(),
        }

        if self.semantic_cache:
            stats["semantic_cache"] = self.semantic_cache.get_stats()

        if self.retrieval_cache:
            stats["retrieval_cache"] = self.retrieval_cache.get_stats()

        if self.embedding_cache:
            stats["embedding_cache"] = self.embedding_cache.get_stats()

        return stats
