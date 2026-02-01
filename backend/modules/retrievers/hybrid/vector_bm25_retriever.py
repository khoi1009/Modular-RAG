"""
Hybrid retriever combining vector similarity search with BM25 keyword search.
Uses fusion strategies to combine results from both methods.
"""

import asyncio
from typing import List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever
from langchain.schema.vectorstore import VectorStore
from rank_bm25 import BM25Okapi

from backend.logger import logger
from backend.modules.retrievers.hybrid.fusion_strategies import (
    FusionStrategy,
    RRFFusion,
    WeightedFusion,
)
from backend.modules.retrievers.hybrid.schemas import HybridRetrieverConfig


class SimpleBM25Index:
    """
    Simple in-memory BM25 index for keyword-based retrieval.
    Uses rank_bm25 library for BM25 scoring.
    """

    def __init__(
        self,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ):
        """
        Initialize BM25 index from documents.

        Args:
            documents: List of documents to index
            k1: BM25 k1 parameter (term saturation)
            b: BM25 b parameter (length normalization)
            epsilon: IDF floor value
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon

        # Tokenize documents (simple whitespace split, lowercase)
        self.tokenized_corpus = [
            doc.page_content.lower().split() for doc in documents
        ]

        # Initialize BM25
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

        logger.info(f"BM25 index created with {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Search documents using BM25 scoring.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of top-k documents sorted by BM25 score
        """
        if not self.documents:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        # Return documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            # Add BM25 score to metadata
            doc.metadata["bm25_score"] = float(scores[idx])
            doc.metadata["relevance_score"] = float(scores[idx])
            results.append(doc)

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    async def asearch(self, query: str, top_k: int = 10) -> List[Document]:
        """Async version of search."""
        # BM25 is fast enough to run synchronously
        return self.search(query, top_k)


class VectorBM25Retriever(BaseRetriever):
    """
    Hybrid retriever that combines vector similarity search with BM25.

    Performs parallel retrieval using both methods and fuses results
    using configurable fusion strategies (RRF or weighted).
    """

    vector_store: VectorStore
    bm25_index: SimpleBM25Index
    fusion_strategy: FusionStrategy
    config: HybridRetrieverConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: SimpleBM25Index,
        config: Optional[HybridRetrieverConfig] = None,
        fusion_strategy: Optional[FusionStrategy] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: Vector store for semantic search
            bm25_index: BM25 index for keyword search
            config: Hybrid retriever configuration
            fusion_strategy: Custom fusion strategy (overrides config)
        """
        config = config or HybridRetrieverConfig()

        # Initialize fusion strategy
        if fusion_strategy is None:
            if config.fusion_strategy == "rrf":
                fusion_strategy = RRFFusion(k=config.rrf_k)
            else:  # weighted
                fusion_strategy = WeightedFusion(normalize=True)

        super().__init__(
            vector_store=vector_store,
            bm25_index=bm25_index,
            fusion_strategy=fusion_strategy,
            config=config,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Synchronous retrieval (not recommended, use async version).

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Fused list of relevant documents
        """
        # Vector search
        vector_results = self.vector_store.similarity_search(
            query, k=self.config.vector_top_k
        )

        # BM25 search
        bm25_results = self.bm25_index.search(query, top_k=self.config.bm25_top_k)

        # Fuse results
        weights = self._get_fusion_weights()
        fused = self.fusion_strategy.fuse([vector_results, bm25_results], weights)

        # Return top-k
        return fused[: self.config.final_top_k]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async retrieval with parallel execution.

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Fused list of relevant documents
        """
        # Parallel retrieval
        vector_results, bm25_results = await asyncio.gather(
            self.vector_store.asimilarity_search(query, k=self.config.vector_top_k),
            self.bm25_index.asearch(query, top_k=self.config.bm25_top_k),
        )

        logger.debug(
            f"Hybrid retrieval: {len(vector_results)} vector + "
            f"{len(bm25_results)} BM25 results"
        )

        # Fuse results
        weights = self._get_fusion_weights()
        fused = self.fusion_strategy.fuse([vector_results, bm25_results], weights)

        # Return top-k
        return fused[: self.config.final_top_k]

    def _get_fusion_weights(self) -> List[float]:
        """
        Get weights for fusion strategy.

        Returns:
            [vector_weight, bm25_weight]
        """
        vector_weight = self.config.vector_weight
        bm25_weight = self.config.bm25_weight

        if bm25_weight is None:
            bm25_weight = 1.0 - vector_weight

        # Normalize to sum to 1.0
        total = vector_weight + bm25_weight
        return [vector_weight / total, bm25_weight / total]
