"""
Semantic cache for finding similar queries using embedding-based similarity.

Uses vector similarity search to find previously answered similar queries,
avoiding redundant computations for semantically equivalent questions.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain.embeddings.base import Embeddings

from backend.logger import logger

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "FAISS not available, semantic cache will use brute-force similarity search"
    )


class SemanticCache:
    """
    Semantic similarity-based query cache.

    Stores query embeddings and results, allowing retrieval of cached results
    for semantically similar queries even if not exact matches.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        similarity_threshold: float = 0.95,
        max_entries: int = 5000,
    ):
        """
        Initialize semantic cache.

        Args:
            embeddings: LangChain embeddings instance for query encoding
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
            max_entries: Maximum number of cached queries
        """
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.cache: Dict[str, Tuple[List[float], Dict[str, Any]]] = (
            {}
        )  # query -> (embedding, result)
        self.index: Optional[Any] = None  # FAISS index or None
        self.query_list: List[str] = []  # Ordered list of queries matching index

        logger.info(
            f"SemanticCache initialized: threshold={similarity_threshold}, "
            f"max_entries={max_entries}, faiss={'enabled' if FAISS_AVAILABLE else 'disabled'}"
        )

    async def get_similar(
        self, query: str, collection: str
    ) -> Optional[Tuple[str, Dict[str, Any], float]]:
        """
        Find semantically similar cached query.

        Args:
            query: User query text
            collection: Collection name (for namespacing)

        Returns:
            Tuple of (cached_query, cached_result, similarity_score) or None
        """
        if not self.cache:
            return None

        try:
            # Embed the query
            query_embedding = await self.embeddings.aembed_query(query)
            query_vec = np.array(query_embedding, dtype=np.float32)

            if FAISS_AVAILABLE and self.index is not None:
                # Use FAISS for fast search
                query_vec = query_vec.reshape(1, -1)
                distances, indices = self.index.search(query_vec, k=1)

                if len(indices[0]) > 0:
                    idx = indices[0][0]
                    # Convert L2 distance to cosine similarity approximation
                    # For normalized vectors: cosine_sim ≈ 1 - (l2_dist^2 / 2)
                    similarity = 1 - (distances[0][0] / 2)

                    if similarity >= self.threshold:
                        cached_query = self.query_list[idx]
                        cached_result = self.cache[cached_query][1]
                        logger.debug(
                            f"Semantic cache hit (FAISS): similarity={similarity:.3f}"
                        )
                        return cached_query, cached_result, similarity
            else:
                # Brute-force similarity search
                best_similarity = 0.0
                best_query = None
                best_result = None

                for cached_query, (cached_embedding, cached_result) in self.cache.items():
                    # Cosine similarity
                    similarity = float(
                        np.dot(query_vec, cached_embedding)
                        / (np.linalg.norm(query_vec) * np.linalg.norm(cached_embedding))
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_query = cached_query
                        best_result = cached_result

                if best_similarity >= self.threshold and best_query:
                    logger.debug(
                        f"Semantic cache hit (brute-force): similarity={best_similarity:.3f}"
                    )
                    return best_query, best_result, best_similarity

            logger.debug("Semantic cache miss")
            return None

        except Exception as e:
            logger.warning(f"Semantic cache lookup failed: {e}")
            return None

    async def add(self, query: str, collection: str, result: Dict[str, Any]) -> None:
        """
        Add query and result to semantic cache.

        Args:
            query: User query text
            collection: Collection name
            result: Query result to cache
        """
        try:
            # Check size limit
            if len(self.cache) >= self.max_entries:
                # Remove oldest entry (FIFO eviction)
                oldest_query = next(iter(self.cache))
                del self.cache[oldest_query]
                logger.debug(f"Semantic cache evicted oldest entry: {oldest_query[:50]}...")

            # Embed and store
            embedding = await self.embeddings.aembed_query(query)
            self.cache[query] = (embedding, result)

            # Rebuild index
            self._rebuild_index()

            logger.debug(f"Semantic cache entry added: {query[:50]}...")

        except Exception as e:
            logger.warning(f"Semantic cache add failed: {e}")

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index from current cache entries."""
        if not self.cache:
            self.index = None
            self.query_list = []
            return

        if not FAISS_AVAILABLE:
            return

        try:
            # Extract embeddings and maintain query order
            self.query_list = list(self.cache.keys())
            embeddings = np.array(
                [self.cache[q][0] for q in self.query_list], dtype=np.float32
            )

            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)

            # Create FAISS index (L2 on normalized vectors = cosine similarity)
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)

            logger.debug(f"FAISS index rebuilt: {len(self.query_list)} entries")

        except Exception as e:
            logger.warning(f"FAISS index rebuild failed: {e}")
            self.index = None

    async def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
        self.index = None
        self.query_list = []
        logger.info("Semantic cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "threshold": self.threshold,
            "faiss_enabled": FAISS_AVAILABLE and self.index is not None,
        }
