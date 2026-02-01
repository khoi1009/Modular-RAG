"""
Fusion strategies for combining results from multiple retrievers.
Supports Reciprocal Rank Fusion (RRF) and weighted score combination.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Optional

from langchain.docstore.document import Document

from backend.logger import logger


class FusionStrategy(ABC):
    """
    Base class for fusion strategies that combine retrieval results.
    """

    @abstractmethod
    def fuse(
        self,
        results: List[List[Document]],
        weights: Optional[List[float]] = None,
    ) -> List[Document]:
        """
        Fuse multiple retrieval result lists into a single ranked list.

        Args:
            results: List of document lists from different retrievers
            weights: Optional weights for each result list

        Returns:
            Fused and ranked list of documents
        """
        pass


class RRFFusion(FusionStrategy):
    """
    Reciprocal Rank Fusion (RRF) strategy.

    Combines rankings using: score = sum(1 / (k + rank))
    where k is a constant (default 60) to prevent division by zero
    and reduce impact of top-ranked items.

    Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: Constant to control rank fusion (default 60)
        """
        self.k = k

    def fuse(
        self,
        results: List[List[Document]],
        weights: Optional[List[float]] = None,
    ) -> List[Document]:
        """
        Apply RRF to combine multiple result lists.

        Args:
            results: List of document lists from different retrievers
            weights: Optional weights (applied as multipliers to RRF scores)

        Returns:
            Documents sorted by RRF score (descending)
        """
        if not results:
            return []

        # Use equal weights if not provided
        if weights is None:
            weights = [1.0] * len(results)

        scores = defaultdict(float)
        doc_map = {}

        for idx, result_list in enumerate(results):
            for rank, doc in enumerate(result_list):
                # Use _data_point_fqn as unique ID, fallback to object id
                doc_id = doc.metadata.get("_data_point_fqn", id(doc))

                # RRF formula with optional weighting
                rrf_score = weights[idx] / (self.k + rank + 1)
                scores[doc_id] += rrf_score

                # Keep reference to document
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Return documents with scores in metadata
        fused_docs = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            # Add fusion score to metadata
            doc.metadata["fusion_score"] = scores[doc_id]
            fused_docs.append(doc)

        logger.debug(f"RRF fusion: {len(results)} lists → {len(fused_docs)} docs")
        return fused_docs


class WeightedFusion(FusionStrategy):
    """
    Weighted score fusion strategy.

    Combines documents by weighted sum of their relevance scores.
    Requires documents to have 'relevance_score' in metadata.
    """

    def __init__(self, normalize: bool = True):
        """
        Initialize weighted fusion.

        Args:
            normalize: Whether to normalize scores to [0, 1] range
        """
        self.normalize = normalize

    def fuse(
        self,
        results: List[List[Document]],
        weights: Optional[List[float]] = None,
    ) -> List[Document]:
        """
        Apply weighted score fusion.

        Args:
            results: List of document lists from different retrievers
            weights: Weights for each retriever's scores (must sum to 1.0)

        Returns:
            Documents sorted by weighted score (descending)
        """
        if not results:
            return []

        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(results)] * len(results)

        # Validate weights sum to ~1.0
        if abs(sum(weights) - 1.0) > 0.01:
            logger.warning(f"Weights sum to {sum(weights)}, normalizing...")
            total = sum(weights)
            weights = [w / total for w in weights]

        scores = defaultdict(float)
        doc_map = {}

        for idx, result_list in enumerate(results):
            # Normalize scores if requested
            if self.normalize and result_list:
                result_scores = [
                    doc.metadata.get("relevance_score", 1.0 / (i + 1))
                    for i, doc in enumerate(result_list)
                ]
                max_score = max(result_scores) if result_scores else 1.0
                min_score = min(result_scores) if result_scores else 0.0
                score_range = max_score - min_score

                if score_range > 0:
                    normalized_scores = [
                        (s - min_score) / score_range for s in result_scores
                    ]
                else:
                    normalized_scores = [1.0] * len(result_scores)
            else:
                normalized_scores = [
                    doc.metadata.get("relevance_score", 1.0 / (i + 1))
                    for i, doc in enumerate(result_list)
                ]

            # Accumulate weighted scores
            for doc, norm_score in zip(result_list, normalized_scores):
                doc_id = doc.metadata.get("_data_point_fqn", id(doc))
                weighted_score = weights[idx] * norm_score
                scores[doc_id] += weighted_score

                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

        # Sort by weighted score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        fused_docs = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id]
            doc.metadata["fusion_score"] = scores[doc_id]
            fused_docs.append(doc)

        logger.debug(
            f"Weighted fusion: {len(results)} lists → {len(fused_docs)} docs"
        )
        return fused_docs
