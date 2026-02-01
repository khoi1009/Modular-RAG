import asyncio
import json
from typing import Dict, Optional

import async_timeout
from langchain.prompts import PromptTemplate

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_analysis.base-query-analyzer import BaseQueryAnalyzer
from backend.modules.query_analysis.schemas import (
    QueryComplexity,
    QueryMetadata,
    QueryType,
)
from backend.types import ModelConfig


class LLMBasedQueryAnalyzer(BaseQueryAnalyzer):
    """Query analyzer using LLM via model gateway"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 10)

    async def analyze(
        self, query: str, context: Optional[Dict] = None
    ) -> QueryMetadata:
        """Analyze query using LLM to extract metadata"""
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["query"],
                    template="""Analyze this query and return JSON only:

Query: {query}

Return JSON with:
{{
  "query_type": "factual|comparison|temporal|spatial|analytical",
  "complexity": "simple|multi_hop|compositional",
  "complexity_score": 0.0-1.0,
  "intent": "retrieval-only|reasoning-required|verification-needed",
  "entities": ["entity1", "entity2"],
  "temporal_constraints": {{}},
  "spatial_constraints": {{}}
}}

JSON:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke({"query": query})

                # Parse LLM response
                result = self._parse_llm_response(response.content)
                return QueryMetadata(**result)

        except asyncio.TimeoutError:
            logger.warning(f"LLM analysis timeout for query: {query}")
            return self._fallback_analysis(query)
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return self._fallback_analysis(query)

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise

    def _fallback_analysis(self, query: str) -> QueryMetadata:
        """Fallback analysis using simple heuristics"""
        return QueryMetadata(
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            complexity_score=0.3,
            intent="retrieval-only",
            entities=[],
        )
