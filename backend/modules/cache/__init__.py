"""
Cache module for multi-level caching in Cognita RAG system.

Provides:
- QueryCache: Exact match caching with Redis support
- SemanticCache: Similar query detection using embeddings
- EmbeddingCache: Persistent embedding storage
- RetrievalCache: Retrieval result caching
- MultiLevelCache: Coordinated multi-level caching
"""

from backend.modules.cache.embedding_cache import EmbeddingCache
from backend.modules.cache.multi_level_cache import (
    CacheHitInfo,
    CacheMetrics,
    MultiLevelCache,
)
from backend.modules.cache.query_cache import QueryCache
from backend.modules.cache.retrieval_cache import RetrievalCache
from backend.modules.cache.semantic_cache import SemanticCache

__all__ = [
    "QueryCache",
    "SemanticCache",
    "EmbeddingCache",
    "RetrievalCache",
    "MultiLevelCache",
    "CacheHitInfo",
    "CacheMetrics",
]
