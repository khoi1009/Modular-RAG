"""
Relevance evaluators for assessing document quality.
Supports LLM-based grading and embedding similarity scoring.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.types import ModelConfig


class BaseRelevanceEvaluator(ABC):
    """
    Base class for relevance evaluators.
    Evaluates how relevant documents are to a query.
    """

    @abstractmethod
    async def evaluate(self, query: str, documents: List[Document]) -> List[float]:
        """
        Evaluate relevance of documents to query.

        Args:
            query: Search query
            documents: Documents to evaluate

        Returns:
            List of relevance scores (0.0-1.0) for each document
        """
        pass

    async def evaluate_single(self, query: str, document: Document) -> float:
        """
        Evaluate single document relevance.

        Args:
            query: Search query
            document: Document to evaluate

        Returns:
            Relevance score (0.0-1.0)
        """
        scores = await self.evaluate(query, [document])
        return scores[0] if scores else 0.0


class LLMRelevanceEvaluator(BaseRelevanceEvaluator):
    """
    Uses LLM to grade document relevance.
    Asks LLM to score relevance on scale of 0-10.
    """

    def __init__(self, model_config: ModelConfig, batch_size: int = 10):
        """
        Initialize LLM evaluator.

        Args:
            model_config: LLM model configuration
            batch_size: Batch size for parallel evaluation
        """
        self.model_config = model_config
        self.batch_size = batch_size
        self.llm = model_gateway.get_llm_from_model_config(model_config, stream=False)

        # Grading prompt
        self.prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""You are a relevance grader. Assess how relevant the document is to the query.

Query: {query}

Document:
{document}

Score the relevance from 0 to 10, where:
- 0-2: Not relevant at all
- 3-5: Somewhat relevant
- 6-8: Relevant
- 9-10: Highly relevant

Respond with ONLY a number from 0 to 10. No explanation needed.""",
        )

    async def evaluate(self, query: str, documents: List[Document]) -> List[float]:
        """
        Evaluate documents using LLM.

        Args:
            query: Search query
            documents: Documents to evaluate

        Returns:
            Normalized relevance scores (0.0-1.0)
        """
        if not documents:
            return []

        scores = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_scores = await self._evaluate_batch(query, batch)
            scores.extend(batch_scores)

        return scores

    async def _evaluate_batch(
        self, query: str, documents: List[Document]
    ) -> List[float]:
        """Evaluate a batch of documents in parallel."""
        tasks = [self._evaluate_single_doc(query, doc) for doc in documents]
        return await asyncio.gather(*tasks)

    async def _evaluate_single_doc(self, query: str, document: Document) -> float:
        """Evaluate single document with LLM."""
        try:
            prompt_text = self.prompt.format(
                query=query, document=document.page_content[:1000]  # Limit length
            )

            response = await self.llm.ainvoke(prompt_text)
            score_text = response.content.strip()

            # Parse score
            score = float(score_text)
            # Normalize to 0-1 range
            normalized = min(max(score / 10.0, 0.0), 1.0)

            logger.debug(f"LLM relevance score: {normalized:.2f}")
            return normalized

        except Exception as e:
            logger.warning(f"Error evaluating relevance with LLM: {e}")
            # Return neutral score on error
            return 0.5


class EmbeddingSimilarityEvaluator(BaseRelevanceEvaluator):
    """
    Uses embedding cosine similarity as relevance score.
    Faster than LLM but potentially less accurate.
    """

    def __init__(self, embeddings: Embeddings):
        """
        Initialize embedding evaluator.

        Args:
            embeddings: Embedding model to use
        """
        self.embeddings = embeddings

    async def evaluate(self, query: str, documents: List[Document]) -> List[float]:
        """
        Evaluate documents using embedding similarity.

        Args:
            query: Search query
            documents: Documents to evaluate

        Returns:
            Cosine similarity scores (0.0-1.0)
        """
        if not documents:
            return []

        try:
            # Get query embedding
            query_embedding = await self.embeddings.aembed_query(query)

            # Get document embeddings
            doc_texts = [doc.page_content for doc in documents]
            doc_embeddings = await self.embeddings.aembed_documents(doc_texts)

            # Calculate cosine similarities
            scores = []
            for doc_emb in doc_embeddings:
                similarity = self._cosine_similarity(query_embedding, doc_emb)
                # Cosine similarity is in [-1, 1], normalize to [0, 1]
                normalized = (similarity + 1.0) / 2.0
                scores.append(normalized)

            logger.debug(f"Embedding similarity scores: {scores}")
            return scores

        except Exception as e:
            logger.warning(f"Error calculating embedding similarity: {e}")
            # Return neutral scores on error
            return [0.5] * len(documents)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)
