"""
Batch processor for grouping API calls to improve throughput.

Automatically batches individual requests into groups for efficient
processing of embeddings, LLM calls, and other operations.
"""

import asyncio
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from langchain.embeddings.base import Embeddings

from backend.logger import logger


class BatchProcessor:
    """
    Intelligent batch processor for API calls.

    Collects individual requests and processes them in batches to improve
    throughput and reduce API overhead. Uses time-based and size-based
    triggering.
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: int = 50):
        """
        Initialize batch processor.

        Args:
            max_batch_size: Maximum items per batch before forcing execution
            max_wait_ms: Maximum wait time in milliseconds before processing batch
        """
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: Dict[str, List[Tuple[Any, asyncio.Future]]] = defaultdict(list)
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.processing: Dict[str, bool] = defaultdict(bool)

        logger.info(
            f"BatchProcessor initialized: max_batch_size={max_batch_size}, "
            f"max_wait_ms={max_wait_ms}"
        )

    async def process_with_batching(
        self,
        key: str,
        item: Any,
        batch_fn: Callable[[List[Any]], Awaitable[List[Any]]],
    ) -> Any:
        """
        Add item to batch queue and wait for result.

        Args:
            key: Batch key (groups items by this key)
            item: Item to process
            batch_fn: Async function that processes a batch of items

        Returns:
            Result for the individual item
        """
        future: asyncio.Future = asyncio.Future()

        async with self.locks[key]:
            self.pending[key].append((item, future))
            pending_count = len(self.pending[key])

            # Trigger immediate processing if batch is full
            if pending_count >= self.max_batch_size:
                logger.debug(f"Batch full ({pending_count} items), processing immediately")
                asyncio.create_task(self._process_batch(key, batch_fn))
            elif not self.processing[key]:
                # Schedule delayed processing
                logger.debug(f"Scheduled delayed batch processing for {self.max_wait_ms}ms")
                asyncio.create_task(self._delayed_process(key, batch_fn))

        # Wait for result
        return await future

    async def _delayed_process(self, key: str, batch_fn: Callable) -> None:
        """
        Process batch after delay if not already processing.

        Args:
            key: Batch key
            batch_fn: Batch processing function
        """
        await asyncio.sleep(self.max_wait_ms / 1000)

        async with self.locks[key]:
            if self.pending[key] and not self.processing[key]:
                await self._process_batch(key, batch_fn)

    async def _process_batch(self, key: str, batch_fn: Callable) -> None:
        """
        Process pending batch.

        Args:
            key: Batch key
            batch_fn: Batch processing function
        """
        # Mark as processing to prevent concurrent execution
        if self.processing[key]:
            return

        self.processing[key] = True

        try:
            # Extract pending items
            items_and_futures = self.pending[key]
            self.pending[key] = []

            if not items_and_futures:
                return

            items = [item for item, _ in items_and_futures]
            futures = [future for _, future in items_and_futures]

            logger.debug(f"Processing batch: {len(items)} items")

            try:
                # Execute batch function
                results = await batch_fn(items)

                if len(results) != len(futures):
                    raise ValueError(
                        f"Batch function returned {len(results)} results "
                        f"but expected {len(futures)}"
                    )

                # Set results for each future
                for future, result in zip(futures, results):
                    if not future.done():
                        future.set_result(result)

                logger.debug(f"Batch processed successfully: {len(items)} items")

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Set exception for all pending futures
                for future in futures:
                    if not future.done():
                        future.set_exception(e)

        finally:
            self.processing[key] = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get batch processor statistics.

        Returns:
            Dictionary with metrics
        """
        pending_counts = {k: len(v) for k, v in self.pending.items()}
        return {
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "pending_batches": pending_counts,
            "total_pending": sum(pending_counts.values()),
        }


class BatchedEmbedder:
    """
    Wrapper for LangChain Embeddings with automatic batching.

    Batches individual embed_query calls into efficient batch operations.
    """

    def __init__(self, embedder: Embeddings, batch_processor: BatchProcessor):
        """
        Initialize batched embedder.

        Args:
            embedder: LangChain Embeddings instance
            batch_processor: BatchProcessor instance
        """
        self.embedder = embedder
        self.processor = batch_processor

        logger.info("BatchedEmbedder initialized")

    async def embed_query(self, text: str) -> List[float]:
        """
        Embed single query with automatic batching.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self.processor.process_with_batching(
            "embed", text, self._batch_embed
        )

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (already batched).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Direct call since already batched
        return await self.embedder.aembed_documents(texts)

    async def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding function.

        Args:
            texts: Batch of texts to embed

        Returns:
            List of embedding vectors
        """
        logger.debug(f"Batch embedding {len(texts)} texts")
        return await self.embedder.aembed_documents(texts)
