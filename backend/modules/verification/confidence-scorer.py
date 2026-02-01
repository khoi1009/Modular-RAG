"""Score answer confidence based on multiple quality signals."""
from typing import Dict, List, Optional

from backend.logger import logger


class ConfidenceScorer:
    """Calculate confidence scores for generated answers"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        # Default weights for different signals
        self.weights = config.get(
            "weights",
            {
                "retrieval_quality": 0.3,
                "source_agreement": 0.25,
                "hallucination_score": 0.25,
                "consistency_score": 0.2,
            },
        )

    async def score(
        self,
        query: str,
        answer: str,
        sources: List,
        retrieval_scores: Optional[List[float]] = None,
        hallucination_score: float = 0.0,
        consistency_score: float = 1.0,
    ) -> float:
        """
        Calculate overall confidence score for an answer.

        Args:
            query: Original user query
            answer: Generated answer
            sources: Retrieved source documents
            retrieval_scores: Optional relevance scores from retrieval (0-1)
            hallucination_score: Score from hallucination detector (0=grounded, 1=hallucinated)
            consistency_score: Score from consistency checker (0-1)

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate individual components
            retrieval_quality = self._calculate_retrieval_quality(
                sources, retrieval_scores
            )
            source_agreement = self._calculate_source_agreement(sources, answer)
            grounding_score = 1.0 - hallucination_score  # Invert hallucination score

            # Weighted average
            confidence = (
                self.weights["retrieval_quality"] * retrieval_quality
                + self.weights["source_agreement"] * source_agreement
                + self.weights["hallucination_score"] * grounding_score
                + self.weights["consistency_score"] * consistency_score
            )

            # Ensure score is in [0, 1] range
            confidence = max(0.0, min(1.0, confidence))

            logger.debug(
                f"Confidence scoring - retrieval: {retrieval_quality:.3f}, "
                f"agreement: {source_agreement:.3f}, grounding: {grounding_score:.3f}, "
                f"consistency: {consistency_score:.3f}, final: {confidence:.3f}"
            )

            return confidence

        except Exception as e:
            logger.error(f"Confidence scoring error: {e}")
            return 0.5  # Return neutral score on error

    def _calculate_retrieval_quality(
        self, sources: List, retrieval_scores: Optional[List[float]]
    ) -> float:
        """Calculate quality score based on retrieval metrics"""
        if not sources:
            return 0.0

        # If retrieval scores provided, use them
        if retrieval_scores and len(retrieval_scores) > 0:
            # Average of top scores
            top_scores = sorted(retrieval_scores, reverse=True)[:3]
            return sum(top_scores) / len(top_scores)

        # Fallback: score based on number of sources (more sources = higher quality)
        num_sources = len(sources)
        if num_sources >= 5:
            return 1.0
        elif num_sources >= 3:
            return 0.8
        elif num_sources >= 1:
            return 0.6
        else:
            return 0.0

    def _calculate_source_agreement(self, sources: List, answer: str) -> float:
        """
        Calculate agreement between sources and answer.
        Simple implementation based on content overlap.
        """
        if not sources or not answer:
            return 0.0

        try:
            answer_words = set(answer.lower().split())
            if len(answer_words) == 0:
                return 0.0

            # Calculate overlap with each source
            overlaps = []
            for source in sources:
                # Handle Document objects or dicts
                content = (
                    source.page_content
                    if hasattr(source, "page_content")
                    else source.get("content", "")
                )
                source_words = set(content.lower().split())

                if len(source_words) > 0:
                    overlap = len(answer_words.intersection(source_words))
                    overlap_ratio = overlap / len(answer_words)
                    overlaps.append(overlap_ratio)

            if overlaps:
                # Return average overlap with sources
                return min(1.0, sum(overlaps) / len(overlaps))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Source agreement calculation error: {e}")
            return 0.5
