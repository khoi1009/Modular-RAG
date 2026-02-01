"""
Corrective RAG (CRAG) retriever with relevance grading and fallback strategies.
Grades document relevance and triggers query refinement or web search on low quality.
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
from backend.modules.retrievers.reflective.schemas import CRAGRetrieverConfig


class CRAGRetriever(BaseRetriever):
    """
    Corrective RAG retriever implementing relevance-aware retrieval.

    Process:
    1. Retrieve documents
    2. Grade relevance of each document
    3. Filter out low-relevance documents
    4. If insufficient relevant docs:
       a. Rewrite query and retry (if enabled)
       b. Fallback to web search (if enabled)
    5. Return corrected document set

    Reference: https://arxiv.org/abs/2401.15884
    """

    base_retriever: BaseRetriever
    evaluator: BaseRelevanceEvaluator
    query_rewriter: Optional[BaseRetriever]
    web_search_retriever: Optional[BaseRetriever]
    config: CRAGRetrieverConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        base_retriever: BaseRetriever,
        evaluator: BaseRelevanceEvaluator,
        query_rewriter: Optional[BaseRetriever] = None,
        web_search_retriever: Optional[BaseRetriever] = None,
        config: Optional[CRAGRetrieverConfig] = None,
    ):
        """
        Initialize CRAG retriever.

        Args:
            base_retriever: Primary retriever
            evaluator: Relevance evaluator for grading
            query_rewriter: Optional query rewriter
            web_search_retriever: Optional web search fallback
            config: CRAG configuration
        """
        config = config or CRAGRetrieverConfig()

        super().__init__(
            base_retriever=base_retriever,
            evaluator=evaluator,
            query_rewriter=query_rewriter,
            web_search_retriever=web_search_retriever,
            config=config,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Synchronous retrieval (not recommended).

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Corrected list of relevant documents
        """
        raise NotImplementedError(
            "CRAGRetriever requires async operation. Use aget_relevant_documents."
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Async CRAG retrieval with correction.

        Args:
            query: Search query
            run_manager: Callback manager

        Returns:
            Corrected and filtered documents
        """
        current_query = query
        attempt = 0

        while attempt <= self.config.max_rewrites:
            logger.info(f"CRAG attempt {attempt + 1}: query='{current_query[:50]}...'")

            # Retrieve documents
            docs = await self.base_retriever.aget_relevant_documents(current_query)

            if not docs:
                logger.warning("No documents retrieved")
                return await self._fallback_retrieval(query)

            # Grade document relevance
            scores = await self.evaluator.evaluate(current_query, docs)

            # Filter by relevance threshold
            relevant_docs = []
            for doc, score in zip(docs, scores):
                if score >= self.config.relevance_threshold:
                    doc.metadata["crag_relevance_score"] = score
                    relevant_docs.append(doc)

            logger.info(
                f"CRAG grading: {len(relevant_docs)}/{len(docs)} docs relevant "
                f"(threshold: {self.config.relevance_threshold})"
            )

            # Check if we have enough relevant documents
            if len(relevant_docs) >= self.config.min_relevant_docs:
                logger.info(
                    f"Sufficient relevant docs found ({len(relevant_docs)} >= "
                    f"{self.config.min_relevant_docs})"
                )
                return relevant_docs

            # Insufficient relevant docs - try correction
            if attempt < self.config.max_rewrites and self.config.enable_query_rewrite:
                logger.info("Insufficient relevant docs, rewriting query...")
                current_query = await self._rewrite_query(current_query, docs, scores)
                attempt += 1
            else:
                # Max rewrites reached, try fallback
                logger.warning(
                    f"Max rewrites reached with {len(relevant_docs)} relevant docs"
                )
                break

        # Return what we have or try fallback
        if relevant_docs:
            return relevant_docs
        else:
            return await self._fallback_retrieval(query)

    async def _rewrite_query(
        self, query: str, docs: List[Document], scores: List[float]
    ) -> str:
        """
        Rewrite query to improve retrieval.

        Args:
            query: Original query
            docs: Retrieved documents
            scores: Relevance scores

        Returns:
            Rewritten query
        """
        if self.query_rewriter:
            try:
                rewritten_docs = await self.query_rewriter.aget_relevant_documents(
                    query
                )
                if rewritten_docs and rewritten_docs[0].page_content:
                    rewritten = rewritten_docs[0].page_content
                    logger.debug(f"Query rewritten: '{rewritten[:50]}...'")
                    return rewritten
            except Exception as e:
                logger.warning(f"Query rewriter failed: {e}")

        # Fallback: simple query expansion
        # Add terms from highest-scoring doc
        if docs and scores:
            max_score_idx = scores.index(max(scores))
            best_doc = docs[max_score_idx]
            key_terms = best_doc.page_content.split()[:5]
            expanded = f"{query} {' '.join(key_terms)}"
            logger.debug(f"Query expanded: '{expanded[:50]}...'")
            return expanded

        return query

    async def _fallback_retrieval(self, query: str) -> List[Document]:
        """
        Fallback retrieval strategy when primary retrieval fails.

        Args:
            query: Search query

        Returns:
            Documents from fallback source
        """
        if self.config.enable_web_search and self.web_search_retriever:
            logger.info("Using web search fallback")
            try:
                web_docs = await self.web_search_retriever.aget_relevant_documents(
                    query
                )
                # Mark as web search results
                for doc in web_docs:
                    doc.metadata["source"] = "web_search_fallback"
                return web_docs
            except Exception as e:
                logger.error(f"Web search fallback failed: {e}")

        logger.warning("No fallback available, returning empty results")
        return []
