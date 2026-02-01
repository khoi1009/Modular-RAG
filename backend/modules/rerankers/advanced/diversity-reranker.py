"""
Diversity-aware reranker using Maximal Marginal Relevance (MMR).
Balances relevance with diversity to avoid redundant results.
"""

from typing import List, Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from backend.logger import logger
from backend.modules.rerankers.advanced.schemas import DiversityRerankerConfig


class DiversityReranker(BaseDocumentCompressor):
    """
    Reranker that promotes diversity in results using MMR algorithm.

    Maximal Marginal Relevance (MMR) balances:
    - Relevance to query
    - Diversity from already selected documents

    Formula: MMR = λ * relevance - (1-λ) * max_similarity_to_selected
    where λ is the diversity_weight parameter.
    """

    embeddings: Embeddings
    config: DiversityRerankerConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        embeddings: Embeddings,
        config: Optional[DiversityRerankerConfig] = None,
    ):
        """
        Initialize diversity reranker.

        Args:
            embeddings: Embedding model for similarity calculation
            config: Diversity reranker configuration
        """
        config = config or DiversityRerankerConfig()

        super().__init__(embeddings=embeddings, config=config)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Synchronous diversity reranking (not recommended).

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Diversified document list
        """
        raise NotImplementedError(
            "DiversityReranker requires async operation. Use acompress_documents."
        )

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Async diversity-aware reranking.

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Diversified and reranked documents
        """
        if not documents:
            return []

        if len(documents) <= self.config.top_k:
            return list(documents)

        logger.info(
            f"Diversity reranking {len(documents)} docs to top {self.config.top_k}"
        )

        if self.config.use_mmr:
            return await self._mmr_rerank(query, list(documents))
        else:
            return await self._simple_diversity_filter(list(documents))

    async def _mmr_rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Apply Maximal Marginal Relevance algorithm.

        Args:
            query: Search query
            documents: Documents to rerank

        Returns:
            MMR-selected diverse documents
        """
        # Get embeddings
        query_embedding = await self.embeddings.aembed_query(query)
        doc_embeddings = await self.embeddings.aembed_documents(
            [doc.page_content for doc in documents]
        )

        # Calculate relevance scores (cosine similarity to query)
        relevance_scores = [
            self._cosine_similarity(query_embedding, doc_emb)
            for doc_emb in doc_embeddings
        ]

        # MMR selection
        selected = []
        selected_embeddings = []
        remaining_indices = list(range(len(documents)))

        lambda_param = self.config.diversity_weight

        for _ in range(min(self.config.top_k, len(documents))):
            if not remaining_indices:
                break

            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                relevance = relevance_scores[idx]

                if selected_embeddings:
                    # Max similarity to already selected docs
                    max_sim = max(
                        self._cosine_similarity(doc_embeddings[idx], sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim = 0.0

                # MMR formula
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr_score))

            # Select document with highest MMR score
            best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
            selected.append(documents[best_idx])
            selected_embeddings.append(doc_embeddings[best_idx])
            remaining_indices.remove(best_idx)

            # Add MMR score to metadata
            documents[best_idx].metadata["mmr_score"] = best_score
            documents[best_idx].metadata["relevance_score"] = best_score

        logger.info(f"MMR selected {len(selected)} diverse documents")
        return selected

    async def _simple_diversity_filter(
        self, documents: List[Document]
    ) -> List[Document]:
        """
        Simple diversity filter based on similarity threshold.

        Args:
            documents: Documents to filter

        Returns:
            Filtered diverse documents
        """
        if not documents:
            return []

        # Get embeddings
        doc_embeddings = await self.embeddings.aembed_documents(
            [doc.page_content for doc in documents]
        )

        selected = [documents[0]]
        selected_embeddings = [doc_embeddings[0]]

        for doc, doc_emb in zip(documents[1:], doc_embeddings[1:]):
            # Check similarity to selected docs
            max_similarity = max(
                self._cosine_similarity(doc_emb, sel_emb)
                for sel_emb in selected_embeddings
            )

            # Add if sufficiently different
            if max_similarity < self.config.similarity_threshold:
                selected.append(doc)
                selected_embeddings.append(doc_emb)

                if len(selected) >= self.config.top_k:
                    break

        logger.info(
            f"Diversity filter: {len(selected)}/{len(documents)} docs selected"
        )
        return selected

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
