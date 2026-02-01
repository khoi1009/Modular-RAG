"""
LLM-based reranker using language models for relevance scoring.
Slower but more accurate than embedding-based reranking.
"""

import asyncio
from typing import Optional, Sequence

from langchain.callbacks.manager import Callbacks
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.language_models.chat_models import BaseChatModel

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.rerankers.advanced.schemas import LLMRerankerConfig
from backend.types import ModelConfig


class LLMReranker(BaseDocumentCompressor):
    """
    Reranker that uses LLM to score document relevance.

    Sends query + document to LLM and asks for relevance score.
    More accurate than embedding similarity but slower and more expensive.
    """

    llm: BaseChatModel
    config: LLMRerankerConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model_config: ModelConfig,
        config: Optional[LLMRerankerConfig] = None,
    ):
        """
        Initialize LLM reranker.

        Args:
            model_config: LLM model configuration
            config: LLM reranker configuration
        """
        if config is None:
            config = LLMRerankerConfig(model_name=model_config.name)

        llm = model_gateway.get_llm_from_model_config(model_config, stream=False)

        super().__init__(llm=llm, config=config)

        # Scoring prompt
        self.prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""Rate how relevant this document is to the query.

Query: {query}

Document:
{document}

Provide a relevance score from 0 to 10:
- 0-2: Not relevant
- 3-5: Somewhat relevant
- 6-8: Relevant
- 9-10: Highly relevant

Respond with ONLY a number from 0 to 10.""",
        )

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Synchronous reranking (not recommended, use async version).

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Reranked documents
        """
        raise NotImplementedError(
            "LLMReranker requires async operation. Use acompress_documents."
        )

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Async LLM-based reranking.

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Documents sorted by LLM relevance score
        """
        if not documents:
            return []

        logger.info(f"LLM reranking {len(documents)} documents")

        # Score documents in batches
        scored_docs = []

        for i in range(0, len(documents), self.config.batch_size):
            batch = list(documents[i : i + self.config.batch_size])
            batch_scores = await self._score_batch(query, batch)

            for doc, score in zip(batch, batch_scores):
                doc.metadata["llm_relevance_score"] = score
                doc.metadata["relevance_score"] = score
                scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        top_docs = [doc for doc, score in scored_docs[: self.config.top_k]]

        logger.info(
            f"LLM reranking complete: top score = "
            f"{scored_docs[0][1]:.2f if scored_docs else 0}"
        )

        return top_docs

    async def _score_batch(
        self, query: str, documents: List[Document]
    ) -> List[float]:
        """Score a batch of documents in parallel."""
        tasks = [self._score_document(query, doc) for doc in documents]
        return await asyncio.gather(*tasks)

    async def _score_document(self, query: str, document: Document) -> float:
        """Score a single document using LLM."""
        try:
            # Truncate document to avoid token limits
            doc_text = document.page_content[:2000]

            prompt_text = self.prompt.format(query=query, document=doc_text)

            response = await self.llm.ainvoke(prompt_text)
            score_text = response.content.strip()

            # Parse score
            score = float(score_text)

            # Normalize to 0-1 range
            normalized = min(max(score / 10.0, 0.0), 1.0)

            logger.debug(f"LLM score: {normalized:.2f}")
            return normalized

        except Exception as e:
            logger.warning(f"Error scoring document with LLM: {e}")
            # Return neutral score on error
            return 0.5
