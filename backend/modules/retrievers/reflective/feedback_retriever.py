"""
Feedback-based retriever with iterative refinement.
Retrieves documents, evaluates quality, and refines query if needed.
"""

from typing import List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.docstore.document import Document
from langchain.schema.retriever import BaseRetriever

from backend.logger import logger
from backend.modules.retrievers.reflective.relevance_evaluators import (
    BaseRelevanceEvaluator,
)
from backend.modules.retrievers.reflective.schemas import FeedbackRetrieverConfig


class FeedbackRetriever(BaseRetriever):
    """
    Retriever with feedback loop for iterative query refinement.

    Process:
    1. Retrieve documents with base retriever
    2. Evaluate relevance with evaluator
    3. If quality below threshold, rewrite query and retry
    4. Repeat up to max_iterations or until quality acceptable
    """

    base_retriever: BaseRetriever
    evaluator: BaseRelevanceEvaluator
    query_rewriter: Optional[BaseRetriever]  # MultiQueryRetriever or custom
    config: FeedbackRetrieverConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        evaluator: BaseRelevanceEvaluator,
        query_rewriter: Optional[BaseRetriever] = None,
        config: Optional[FeedbackRetrieverConfig] = None,
    ):
        """
        Initialize feedback retriever.

        Args:
            base_retriever: Underlying retriever to use
            evaluator: Relevance evaluator for quality assessment
            query_rewriter: Optional query rewriter (uses base retriever if None)
            config: Feedback retriever configuration
        """
        config = config or FeedbackRetrieverConfig()

        super().__init__(
            base_retriever=base_retriever,
            evaluator=evaluator,
            query_rewriter=query_rewriter,
            config=config,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Synchronous retrieval with feedback (not recommended).

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Refined list of relevant documents
        """
        raise NotImplementedError(
            "FeedbackRetriever requires async operation. Use aget_relevant_documents."
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async retrieval with iterative feedback loop.

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Best documents after refinement
        """
        current_query = query
        best_docs = []
        best_avg_score = 0.0

        for iteration in range(self.config.max_iterations):
            logger.info(
                f"Feedback iteration {iteration + 1}/{self.config.max_iterations}: "
                f"query='{current_query[:50]}...'"
            )

            # Retrieve documents
            docs = await self.base_retriever.aget_relevant_documents(current_query)

            if not docs:
                logger.warning("No documents retrieved, stopping feedback loop")
                break

            # Evaluate relevance
            scores = await self.evaluator.evaluate(current_query, docs)
            avg_score = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                f"Iteration {iteration + 1} avg relevance: {avg_score:.3f} "
                f"(threshold: {self.config.quality_threshold})"
            )

            # Track best results
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_docs = docs

            # Check if quality acceptable
            if avg_score >= self.config.quality_threshold:
                logger.info(
                    f"Quality threshold met ({avg_score:.3f} >= "
                    f"{self.config.quality_threshold}), stopping refinement"
                )
                return docs

            # Check if quality too low to continue
            if avg_score < self.config.min_quality_threshold:
                logger.warning(
                    f"Quality below minimum ({avg_score:.3f} < "
                    f"{self.config.min_quality_threshold}), stopping refinement"
                )
                break

            # Last iteration - return best so far
            if iteration == self.config.max_iterations - 1:
                logger.info(
                    f"Max iterations reached, returning best results "
                    f"(score: {best_avg_score:.3f})"
                )
                break

            # Rewrite query for next iteration
            current_query = await self._rewrite_query(current_query, docs, scores)

        return best_docs if best_docs else []

    async def _rewrite_query(
        self, original_query: str, docs: List[Document], scores: List[float]
    ) -> str:
        """
        Rewrite query based on feedback.

        For now, uses simple heuristic:
        - If we have a query rewriter, use it
        - Otherwise, append context from low-scoring docs

        Args:
            original_query: Original query
            docs: Retrieved documents
            scores: Relevance scores

        Returns:
            Rewritten query
        """
        if self.query_rewriter:
            # Use custom query rewriter
            rewritten_docs = await self.query_rewriter.aget_relevant_documents(
                original_query
            )
            if rewritten_docs and rewritten_docs[0].page_content:
                rewritten = rewritten_docs[0].page_content
                logger.debug(f"Query rewritten to: '{rewritten[:50]}...'")
                return rewritten

        # Fallback: add context from documents
        # Find lowest-scoring docs - might indicate missing concepts
        if scores:
            low_score_idx = scores.index(min(scores))
            low_doc = docs[low_score_idx]

            # Extract key terms from low-scoring doc (simple approach)
            # In production, use more sophisticated query expansion
            key_terms = low_doc.page_content.split()[:5]
            expanded = f"{original_query} {' '.join(key_terms)}"

            logger.debug(f"Query expanded to: '{expanded[:50]}...'")
            return expanded

        return original_query
