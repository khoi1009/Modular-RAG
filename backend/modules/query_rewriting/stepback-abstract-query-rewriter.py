import asyncio
from typing import Dict, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_rewriting.base-query-rewriter import BaseQueryRewriter
from backend.modules.query_rewriting.schemas import RewriteResult
from backend.types import ModelConfig


class StepBackAbstractQueryRewriter(BaseQueryRewriter):
    """
    Step-back prompting rewriter.
    Generates an abstract, higher-level version of the query
    to retrieve broader context before specific details.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 10)

    async def rewrite(
        self, query: str, context: Optional[Dict] = None
    ) -> RewriteResult:
        """Generate abstract step-back query"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query"],
                    template="""Given a specific question, generate a more general, abstract question that covers the broader topic.
The abstract question should help retrieve background information relevant to answering the specific question.

Specific question: {query}

Abstract question:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"query": query})

                abstract_query = response.content.strip()

                # Return both original and abstract queries for retrieval
                return RewriteResult(
                    original_query=query,
                    rewritten_queries=[query, abstract_query],
                    strategy="stepback",
                    metadata={
                        "model": self.model_config.name,
                        "abstract_query": abstract_query,
                    },
                )

        except asyncio.TimeoutError:
            logger.warning(f"Step-back timeout for query: {query}")
            return self._fallback_result(query)
        except Exception as e:
            logger.error(f"Step-back rewriting error: {e}")
            return self._fallback_result(query)

    def _fallback_result(self, query: str) -> RewriteResult:
        """Fallback to original query on error"""
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            strategy="stepback_fallback",
            metadata={"error": "fallback_to_original"},
        )
