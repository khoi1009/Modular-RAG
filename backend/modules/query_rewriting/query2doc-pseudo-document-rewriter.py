import asyncio
from typing import Dict, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_rewriting.base-query-rewriter import BaseQueryRewriter
from backend.modules.query_rewriting.schemas import RewriteResult
from backend.types import ModelConfig


class Query2DocPseudoDocumentRewriter(BaseQueryRewriter):
    """
    Query2Doc rewriter.
    Generates a pseudo-document from the query by expanding it
    with related terms and context, then uses for retrieval.
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
        """Generate pseudo-document from query"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query"],
                    template="""Expand this question into a detailed pseudo-document that includes:
- The question itself
- Key terms and concepts from the question
- Related terminology and synonyms
- Context that would help find relevant information

Question: {query}

Pseudo-document:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"query": query})

                pseudo_doc = response.content.strip()

                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[pseudo_doc],
                    strategy="query2doc",
                    metadata={
                        "model": self.model_config.name,
                        "doc_length": len(pseudo_doc.split()),
                    },
                )

        except asyncio.TimeoutError:
            logger.warning(f"Query2Doc timeout for query: {query}")
            return self._fallback_result(query)
        except Exception as e:
            logger.error(f"Query2Doc rewriting error: {e}")
            return self._fallback_result(query)

    def _fallback_result(self, query: str) -> RewriteResult:
        """Fallback to original query on error"""
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            strategy="query2doc_fallback",
            metadata={"error": "fallback_to_original"},
        )
