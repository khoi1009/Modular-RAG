"""LLM-based query router for flexible routing decisions."""
import json
from typing import Dict, Optional

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.orchestration.routing.base_query_router import BaseQueryRouter
from backend.modules.orchestration.routing.schemas import RoutingDecision
from backend.modules.query_analysis.schemas import QueryMetadata
from backend.modules.query_controllers.types import ModelConfig


class LLMBasedRouter(BaseQueryRouter):
    """Router that uses LLM to make routing decisions"""

    def __init__(self, model_config: Optional[ModelConfig] = None):
        """
        Initialize LLM-based router.

        Args:
            model_config: Optional model configuration, uses default if not provided
        """
        self.model_config = model_config or self._get_default_model_config()

    def _get_default_model_config(self) -> ModelConfig:
        """Get default model configuration for routing"""
        return ModelConfig(
            name="gpt-3.5-turbo",
            parameters={"temperature": 0.0, "max_tokens": 500},
        )

    def _build_routing_prompt(
        self, query: str, query_metadata: QueryMetadata
    ) -> str:
        """
        Build prompt for LLM routing decision.

        Args:
            query: Raw query string
            query_metadata: Analyzed query metadata

        Returns:
            Formatted prompt string
        """
        return f"""You are a query routing expert. Analyze the following query and metadata to determine the best retrieval strategy.

Query: {query}

Query Metadata:
- Type: {query_metadata.query_type}
- Complexity: {query_metadata.complexity}
- Complexity Score: {query_metadata.complexity_score}
- Intent: {query_metadata.intent}
- Entities: {', '.join(query_metadata.entities) if query_metadata.entities else 'None'}

Available Strategies:
1. simple-retrieval: Basic vector search, fast but less accurate
2. hybrid-search: Combines vector and keyword search, balanced
3. multi-hop-reasoning: For complex queries requiring multiple steps
4. reflective-retrieval: Self-correcting retrieval with verification

Available Preprocessing:
- hyde: Hypothetical document embeddings
- decomposition: Break into sub-queries
- stepback: Abstract reasoning
- query2doc: Expand with pseudo-documents

Respond ONLY with a valid JSON object in this exact format:
{{
  "controller_name": "pipeline_name",
  "retrieval_strategy": "strategy_name",
  "preprocessing_steps": ["step1", "step2"],
  "use_reranking": true/false,
  "max_iterations": 1-3,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

    async def route(
        self,
        query: str,
        query_metadata: QueryMetadata,
        context: Optional[Dict] = None,
    ) -> RoutingDecision:
        """
        Route query using LLM analysis.

        Args:
            query: Raw query string
            query_metadata: Analyzed query metadata
            context: Optional additional context

        Returns:
            RoutingDecision from LLM
        """
        try:
            # Get LLM
            llm = model_gateway.get_llm_from_model_config(
                self.model_config, stream=False
            )

            # Build prompt
            prompt = self._build_routing_prompt(query, query_metadata)

            # Invoke LLM
            response = await llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON response
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            routing_data = json.loads(response_text)

            # Create RoutingDecision
            decision = RoutingDecision(**routing_data)
            logger.info(f"LLM routing decision: {decision.controller_name} (confidence: {decision.confidence})")
            return decision

        except Exception as e:
            logger.error(f"Error in LLM routing: {e}, falling back to default")
            return self._fallback_routing(query_metadata)

    def _fallback_routing(self, query_metadata: QueryMetadata) -> RoutingDecision:
        """
        Fallback routing when LLM fails.

        Args:
            query_metadata: Query metadata

        Returns:
            Safe default routing decision
        """
        # Use complexity score for basic routing
        if query_metadata.complexity_score > 0.7:
            return RoutingDecision(
                controller_name="multi-hop-reasoning",
                retrieval_strategy="hybrid",
                preprocessing_steps=["decomposition"],
                use_reranking=True,
                max_iterations=2,
                confidence=0.6,
                reasoning="High complexity, using multi-hop (fallback)",
            )
        elif query_metadata.complexity_score > 0.4:
            return RoutingDecision(
                controller_name="hybrid-search",
                retrieval_strategy="hybrid",
                preprocessing_steps=[],
                use_reranking=True,
                max_iterations=1,
                confidence=0.7,
                reasoning="Medium complexity, using hybrid search (fallback)",
            )
        else:
            return RoutingDecision(
                controller_name="simple-retrieval",
                retrieval_strategy="vectorstore",
                preprocessing_steps=[],
                use_reranking=False,
                max_iterations=1,
                confidence=0.8,
                reasoning="Low complexity, using simple retrieval (fallback)",
            )
