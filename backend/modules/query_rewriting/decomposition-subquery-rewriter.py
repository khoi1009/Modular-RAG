import asyncio
import json
from typing import Dict, List, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_rewriting.base-query-rewriter import BaseQueryRewriter
from backend.modules.query_rewriting.schemas import RewriteResult
from backend.types import ModelConfig


class DecompositionSubqueryRewriter(BaseQueryRewriter):
    """
    Query decomposition rewriter.
    Breaks complex queries into simpler sub-queries
    for parallel retrieval and answering.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 15)
        self.max_subqueries = config.get("max_subqueries", 4)

    async def rewrite(
        self, query: str, context: Optional[Dict] = None
    ) -> RewriteResult:
        """Decompose query into sub-queries"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query", "max_subqueries"],
                    template="""Break down this complex question into simpler sub-questions that can be answered independently.
Return up to {max_subqueries} sub-questions as a JSON array.

Complex question: {query}

Return JSON array only:
["sub-question 1", "sub-question 2", ...]

JSON:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke(
                    {"query": query, "max_subqueries": self.max_subqueries}
                )

                subqueries = self._parse_subqueries(response.content)

                return RewriteResult(
                    original_query=query,
                    rewritten_queries=subqueries,
                    strategy="decomposition",
                    metadata={
                        "model": self.model_config.name,
                        "num_subqueries": len(subqueries),
                    },
                )

        except asyncio.TimeoutError:
            logger.warning(f"Decomposition timeout for query: {query}")
            return self._fallback_result(query)
        except Exception as e:
            logger.error(f"Decomposition rewriting error: {e}")
            return self._fallback_result(query)

    def _parse_subqueries(self, response: str) -> List[str]:
        """Parse sub-queries from LLM response"""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            subqueries = json.loads(response.strip())

            if isinstance(subqueries, list):
                return [sq.strip() for sq in subqueries if sq.strip()][:self.max_subqueries]
            return []

        except Exception as e:
            logger.error(f"Failed to parse subqueries: {e}")
            # Try line-by-line parsing as fallback
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            return [
                line.lstrip("0123456789.-) ")
                for line in lines
                if line and not line.startswith("{")
            ][:self.max_subqueries]

    def _fallback_result(self, query: str) -> RewriteResult:
        """Fallback to original query on error"""
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            strategy="decomposition_fallback",
            metadata={"error": "fallback_to_original"},
        )
