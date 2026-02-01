"""
Multi-stage cascaded reranker for efficient large-scale reranking.
Uses progressively more powerful (and expensive) rerankers at each stage.
"""

from typing import List, Optional, Sequence, Tuple

from langchain.callbacks.manager import Callbacks
from langchain.docstore.document import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor

from backend.logger import logger
from backend.modules.rerankers.advanced.schemas import MultiStageRerankerConfig


class MultiStageReranker(BaseDocumentCompressor):
    """
    Cascaded multi-stage reranking pipeline.

    Example:
        Stage 1: Fast embedding reranker (100 → 20 docs)
        Stage 2: Powerful cross-encoder (20 → 5 docs)

    This approach reduces computational cost while maintaining quality.
    """

    stages: List[Tuple[BaseDocumentCompressor, int]]
    config: MultiStageRerankerConfig

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        stages: List[Tuple[BaseDocumentCompressor, int]],
        config: Optional[MultiStageRerankerConfig] = None,
    ):
        """
        Initialize multi-stage reranker.

        Args:
            stages: List of (compressor, top_k_to_pass) tuples
            config: Multi-stage reranker configuration
        """
        config = config or MultiStageRerankerConfig()

        super().__init__(stages=stages, config=config)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Apply multi-stage reranking pipeline.

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Reranked and filtered documents
        """
        if not documents:
            return []

        current_docs = list(documents)
        logger.info(
            f"Multi-stage reranking: {len(current_docs)} docs through "
            f"{len(self.stages)} stages"
        )

        for stage_idx, (compressor, top_k) in enumerate(self.stages):
            logger.debug(
                f"Stage {stage_idx + 1}/{len(self.stages)}: "
                f"{len(current_docs)} docs → top {top_k}"
            )

            # Apply reranker
            current_docs = compressor.compress_documents(
                current_docs, query, callbacks
            )

            # Keep top-k
            current_docs = current_docs[:top_k]

            # Add stage metadata
            for doc in current_docs:
                doc.metadata[f"stage_{stage_idx + 1}_rank"] = (
                    current_docs.index(doc) + 1
                )

        logger.info(f"Multi-stage reranking complete: {len(current_docs)} final docs")
        return current_docs

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Async multi-stage reranking.

        Args:
            documents: Documents to rerank
            query: Search query
            callbacks: Optional callbacks

        Returns:
            Reranked and filtered documents
        """
        if not documents:
            return []

        current_docs = list(documents)
        logger.info(
            f"Multi-stage reranking (async): {len(current_docs)} docs through "
            f"{len(self.stages)} stages"
        )

        for stage_idx, (compressor, top_k) in enumerate(self.stages):
            logger.debug(
                f"Stage {stage_idx + 1}/{len(self.stages)}: "
                f"{len(current_docs)} docs → top {top_k}"
            )

            # Apply reranker (async if available)
            if hasattr(compressor, "acompress_documents"):
                current_docs = await compressor.acompress_documents(
                    current_docs, query, callbacks
                )
            else:
                current_docs = compressor.compress_documents(
                    current_docs, query, callbacks
                )

            # Keep top-k
            current_docs = current_docs[:top_k]

            # Add stage metadata
            for doc in current_docs:
                doc.metadata[f"stage_{stage_idx + 1}_rank"] = (
                    current_docs.index(doc) + 1
                )

        logger.info(
            f"Multi-stage reranking complete (async): {len(current_docs)} final docs"
        )
        return current_docs
