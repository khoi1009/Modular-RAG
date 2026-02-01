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


class MultiPerspectiveQueryRewriter(BaseQueryRewriter):
    """
    Multi-query rewriter.
    Generates multiple perspectives/variations of the same query
    to improve retrieval coverage.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.5}),
        )
        self.timeout = config.get("timeout", 15)
        self.num_variations = config.get("num_variations", 3)

    async def rewrite(
        self, query: str, context: Optional[Dict] = None
    ) -> RewriteResult:
        """Generate multiple query variations"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query", "num_variations"],
                    template="""Generate {num_variations} different ways to ask the same question.
Each variation should use different words but seek the same information.
Return as JSON array.

Original question: {query}

Return JSON array only:
["variation 1", "variation 2", "variation 3"]

JSON:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke(
                    {"query": query, "num_variations": self.num_variations}
                )

                variations = self._parse_variations(response.content)

                # Include original query
                all_queries = [query] + variations

                return RewriteResult(
                    original_query=query,
                    rewritten_queries=all_queries,
                    strategy="multi_query",
                    metadata={
                        "model": self.model_config.name,
                        "num_variations": len(variations),
                    },
                )

        except asyncio.TimeoutError:
            logger.warning(f"Multi-query timeout for query: {query}")
            return self._fallback_result(query)
        except Exception as e:
            logger.error(f"Multi-query rewriting error: {e}")
            return self._fallback_result(query)

    def _parse_variations(self, response: str) -> List[str]:
        """Parse query variations from LLM response"""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            variations = json.loads(response.strip())

            if isinstance(variations, list):
                return [v.strip() for v in variations if v.strip()][:self.num_variations]
            return []

        except Exception as e:
            logger.error(f"Failed to parse variations: {e}")
            # Try line-by-line parsing as fallback
            lines = [line.strip() for line in response.split("\n") if line.strip()]
            return [
                line.lstrip("0123456789.-) ")
                for line in lines
                if line and not line.startswith("{")
            ][:self.num_variations]

    def _fallback_result(self, query: str) -> RewriteResult:
        """Fallback to original query on error"""
        return RewriteResult(
            original_query=query,
            rewritten_queries=[query],
            strategy="multi_query_fallback",
            metadata={"error": "fallback_to_original"},
        )
