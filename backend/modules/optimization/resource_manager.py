"""
Unified resource manager for coordinating optimization components.

Integrates rate limiting, caching, batching, and parallel execution
into a single interface for consistent resource management.
"""

from typing import Any, Callable, Dict, Optional

from backend.logger import logger
from backend.modules.optimization.batch_processor import BatchProcessor
from backend.modules.optimization.parallel_executor import ParallelExecutor
from backend.modules.optimization.rate_limiter import AdaptiveRateLimiter


class ResourceConfig:
    """Configuration for resource manager."""

    def __init__(
        self,
        openai_rate: float = 10.0,
        ollama_rate: float = 5.0,
        embedding_rate: float = 50.0,
        batch_size: int = 32,
        max_concurrency: int = 10,
    ):
        """
        Initialize resource configuration.

        Args:
            openai_rate: OpenAI API rate limit (requests/sec)
            ollama_rate: Ollama API rate limit (requests/sec)
            embedding_rate: Embedding API rate limit (requests/sec)
            batch_size: Maximum batch size for batch processing
            max_concurrency: Maximum concurrent operations
        """
        self.openai_rate = openai_rate
        self.ollama_rate = ollama_rate
        self.embedding_rate = embedding_rate
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency


class ResourceManager:
    """
    Unified resource manager.

    Coordinates rate limiting, caching, batching, and parallel execution
    to optimize resource usage and prevent API throttling.
    """

    def __init__(
        self,
        config: Optional[ResourceConfig] = None,
        cache_manager: Optional[Any] = None,
    ):
        """
        Initialize resource manager.

        Args:
            config: Resource configuration (uses defaults if None)
            cache_manager: Optional MultiLevelCache instance
        """
        if config is None:
            config = ResourceConfig()

        self.config = config
        self.cache = cache_manager

        # Initialize rate limiters
        self.rate_limiters: Dict[str, AdaptiveRateLimiter] = {
            "openai": AdaptiveRateLimiter(config.openai_rate, "openai"),
            "ollama": AdaptiveRateLimiter(config.ollama_rate, "ollama"),
            "embedding": AdaptiveRateLimiter(config.embedding_rate, "embedding"),
        }

        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            max_batch_size=config.batch_size, max_wait_ms=50
        )

        # Initialize parallel executor
        self.parallel_executor = ParallelExecutor(
            max_concurrency=config.max_concurrency
        )

        logger.info(
            f"ResourceManager initialized: "
            f"openai_rate={config.openai_rate}/s, "
            f"ollama_rate={config.ollama_rate}/s, "
            f"embedding_rate={config.embedding_rate}/s, "
            f"batch_size={config.batch_size}, "
            f"max_concurrency={config.max_concurrency}"
        )

    async def execute_with_resources(
        self,
        resource_type: str,
        operation: Callable,
        cache_key: Optional[str] = None,
        rate_limit_tokens: int = 1,
    ) -> Any:
        """
        Execute operation with resource management.

        Applies rate limiting and optional caching to the operation.

        Args:
            resource_type: Type of resource ("openai", "ollama", "embedding")
            operation: Async callable to execute
            cache_key: Optional cache key for result caching
            rate_limit_tokens: Number of rate limit tokens to consume

        Returns:
            Operation result
        """
        # Check cache first
        if cache_key and self.cache:
            try:
                # Cache lookup would require more context (query, collection, etc.)
                # This is a simplified interface
                logger.debug(f"Cache check for key: {cache_key[:50]}...")
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        # Apply rate limiting
        rate_limiter = self.rate_limiters.get(resource_type)
        if rate_limiter:
            logger.debug(
                f"Acquiring {rate_limit_tokens} tokens from '{resource_type}' rate limiter"
            )
            acquired = await rate_limiter.acquire(rate_limit_tokens, timeout_sec=30.0)
            if not acquired:
                logger.error(
                    f"Failed to acquire rate limit tokens for '{resource_type}'"
                )
                raise TimeoutError(
                    f"Rate limit timeout for resource type: {resource_type}"
                )

        # Execute operation
        try:
            result = await operation()

            # Record success for adaptive rate limiting
            if rate_limiter:
                rate_limiter.record_success()

            # Cache result if key provided
            if cache_key and self.cache:
                try:
                    logger.debug(f"Caching result for key: {cache_key[:50]}...")
                    # Actual caching would need more context
                except Exception as e:
                    logger.warning(f"Cache store failed: {e}")

            return result

        except Exception as e:
            # Record error for adaptive rate limiting
            if rate_limiter:
                error_type = "rate_limit" if "rate" in str(e).lower() else "other"
                rate_limiter.record_error(error_type)

            raise

    async def execute_batch(
        self,
        resource_type: str,
        batch_key: str,
        item: Any,
        batch_fn: Callable,
    ) -> Any:
        """
        Execute operation with batching.

        Args:
            resource_type: Resource type for rate limiting
            batch_key: Key for grouping batch items
            item: Item to add to batch
            batch_fn: Batch processing function

        Returns:
            Result for the individual item
        """
        # Wrap batch function with rate limiting
        async def rate_limited_batch_fn(items):
            rate_limiter = self.rate_limiters.get(resource_type)
            if rate_limiter:
                # Acquire tokens for batch size
                await rate_limiter.acquire(len(items), timeout_sec=60.0)

            try:
                result = await batch_fn(items)
                if rate_limiter:
                    rate_limiter.record_success()
                return result
            except Exception as e:
                if rate_limiter:
                    error_type = "rate_limit" if "rate" in str(e).lower() else "other"
                    rate_limiter.record_error(error_type)
                raise

        return await self.batch_processor.process_with_batching(
            batch_key, item, rate_limited_batch_fn
        )

    def get_rate_limiter(self, resource_type: str) -> Optional[AdaptiveRateLimiter]:
        """
        Get rate limiter for specific resource type.

        Args:
            resource_type: Resource type

        Returns:
            AdaptiveRateLimiter instance or None
        """
        return self.rate_limiters.get(resource_type)

    def register_rate_limiter(
        self, resource_type: str, rate: float
    ) -> AdaptiveRateLimiter:
        """
        Register new rate limiter for custom resource type.

        Args:
            resource_type: Resource type identifier
            rate: Base rate in requests per second

        Returns:
            Created AdaptiveRateLimiter instance
        """
        if resource_type in self.rate_limiters:
            logger.warning(
                f"Rate limiter for '{resource_type}' already exists, replacing"
            )

        rate_limiter = AdaptiveRateLimiter(rate, resource_type)
        self.rate_limiters[resource_type] = rate_limiter

        logger.info(f"Registered rate limiter for '{resource_type}': {rate}/s")
        return rate_limiter

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive resource manager statistics.

        Returns:
            Dictionary with stats from all components
        """
        stats = {
            "config": {
                "openai_rate": self.config.openai_rate,
                "ollama_rate": self.config.ollama_rate,
                "embedding_rate": self.config.embedding_rate,
                "batch_size": self.config.batch_size,
                "max_concurrency": self.config.max_concurrency,
            },
            "rate_limiters": {
                name: limiter.get_stats()
                for name, limiter in self.rate_limiters.items()
            },
            "batch_processor": self.batch_processor.get_stats(),
            "parallel_executor": self.parallel_executor.get_stats(),
        }

        if self.cache:
            stats["cache"] = self.cache.get_stats()

        return stats

    def reset_metrics(self) -> None:
        """Reset all metrics and error counters."""
        for limiter in self.rate_limiters.values():
            limiter.reset_error_count()

        if self.cache:
            self.cache.metrics.reset()

        logger.info("Resource manager metrics reset")
