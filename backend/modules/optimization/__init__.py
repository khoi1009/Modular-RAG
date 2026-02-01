"""
Optimization module for performance enhancements in Cognita RAG system.

Provides:
- BatchProcessor: Batch processing for embeddings and LLM calls
- ParallelExecutor: Concurrent execution with limits
- RateLimiter: Token bucket and adaptive rate limiting
- ResourceManager: Unified resource management
"""

from backend.modules.optimization.batch_processor import (
    BatchProcessor,
    BatchedEmbedder,
)
from backend.modules.optimization.parallel_executor import ParallelExecutor
from backend.modules.optimization.rate_limiter import (
    AdaptiveRateLimiter,
    TokenBucketRateLimiter,
)
from backend.modules.optimization.resource_manager import ResourceManager

__all__ = [
    "BatchProcessor",
    "BatchedEmbedder",
    "ParallelExecutor",
    "TokenBucketRateLimiter",
    "AdaptiveRateLimiter",
    "ResourceManager",
]
