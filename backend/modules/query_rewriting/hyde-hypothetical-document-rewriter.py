import asyncio
from typing import Dict, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_rewriting.base-query-rewriter import BaseQueryRewriter
from backend.modules.query_rewriting.schemas import RewriteResult
from backend.types import ModelConfig


class HyDEHypotheticalDocumentRewriter(BaseQueryRewriter):
    """
    HyDE (Hypothetical Document Embeddings) rewriter.
    Generates a hypothetical answer document that would answer the query,
    then uses that for retrieval instead of the original query.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.3}),
        )
        self.timeout = config.get("timeout", 15)

    async def rewrite(
        self, query: str, context: Optional[Dict] = None
    ) -> RewriteResult:
        """Generate hypothetical document that would answer the query"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query"],
                    template="""Write a detailed passage that would perfectly answer this question.
Write as if you are an expert providing the actual answer (not meta-commentary).

Question: {query}

Expert passage:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"query": query})

                hypothetical_doc = response.content.strip()

                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[hypothetical_doc],
                    strategy="hyde",
                    metadata={
                        "model": self.model_config.name,
                        "doc_length": len(hypothetical_doc.split()),
                    },
                )

        except asyncio.TimeoutError:
            logger.warning(f"HyDE timeout for query: {query}")
            return self._fallback_result(query)
        except Exception as e:
            logger.error(f"HyDE rewriting error: {e}")
            return self._fallback_result(query)

    def _fallback_result(self, query: str) -> RewriteResult:
        """Fallback to original query on error"""
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            strategy="hyde_fallback",
            metadata={"error": "fallback_to_original"},
        )
