"""
Persistent embedding cache with disk and memory storage.

Caches computed embeddings to avoid redundant API calls for the same text.
Uses two-level storage: in-memory for speed, disk for persistence.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.logger import logger


class EmbeddingCache:
    """
    Two-level embedding cache (memory + disk).

    Stores embeddings keyed by text hash, with memory cache for fast access
    and disk cache for persistence across restarts.
    """

    def __init__(self, storage_path: str = "data/embedding_cache"):
        """
        Initialize embedding cache.

        Args:
            storage_path: Directory path for persistent storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, List[float]] = {}

        logger.info(f"EmbeddingCache initialized: storage={self.storage_path}")

    def _text_hash(self, text: str) -> str:
        """
        Generate hash for text content.

        Args:
            text: Input text

        Returns:
            16-character hex hash
        """
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None if not cached
        """
        text_hash = self._text_hash(text)

        # Check memory cache
        if text_hash in self.memory_cache:
            logger.debug(f"Embedding cache hit (memory): {text_hash}")
            return self.memory_cache[text_hash]

        # Check disk cache
        cache_file = self.storage_path / f"{text_hash}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(cache_file).tolist()
                # Populate memory cache
                self.memory_cache[text_hash] = embedding
                logger.debug(f"Embedding cache hit (disk): {text_hash}")
                return embedding
            except Exception as e:
                logger.warning(f"Failed to load embedding from disk: {e}")
                # Remove corrupted file
                cache_file.unlink(missing_ok=True)

        logger.debug(f"Embedding cache miss: {text_hash}")
        return None

    async def set(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        text_hash = self._text_hash(text)

        # Store in memory
        self.memory_cache[text_hash] = embedding

        # Store on disk
        try:
            cache_file = self.storage_path / f"{text_hash}.npy"
            np.save(cache_file, np.array(embedding))
            logger.debug(f"Embedding cached: {text_hash}")
        except Exception as e:
            logger.warning(f"Failed to save embedding to disk: {e}")

    async def get_batch(
        self, texts: List[str]
    ) -> Tuple[List[Optional[List[float]]], List[int]]:
        """
        Batch retrieve embeddings, return missing indices.

        Args:
            texts: List of input texts

        Returns:
            Tuple of (embeddings_list, missing_indices)
            - embeddings_list: List where cached embeddings are present, None for missing
            - missing_indices: Indices of texts that need embedding computation
        """
        embeddings: List[Optional[List[float]]] = []
        missing: List[int] = []

        for i, text in enumerate(texts):
            cached = await self.get(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                missing.append(i)

        logger.debug(
            f"Embedding batch: {len(texts)} total, {len(missing)} missing, "
            f"{len(texts) - len(missing)} cached"
        )
        return embeddings, missing

    async def set_batch(
        self, texts: List[str], embeddings: List[List[float]]
    ) -> None:
        """
        Batch store embeddings.

        Args:
            texts: List of input texts
            embeddings: List of embedding vectors
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        for text, embedding in zip(texts, embeddings):
            await self.set(text, embedding)

        logger.debug(f"Embedding batch cached: {len(texts)} entries")

    async def clear(self) -> None:
        """Clear all cached embeddings (memory and disk)."""
        # Clear memory
        self.memory_cache.clear()

        # Clear disk
        try:
            for cache_file in self.storage_path.glob("*.npy"):
                cache_file.unlink()
            logger.info(f"Embedding cache cleared: {self.storage_path}")
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")

    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        disk_entries = len(list(self.storage_path.glob("*.npy")))
        return {
            "memory_entries": len(self.memory_cache),
            "disk_entries": disk_entries,
        }
