"""LLM-as-Judge for evaluating answer quality without ground truth."""
import json
import re
from typing import Dict, Optional

from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.model_gateway.types import ModelConfig


class LLMJudge:
    """Uses LLM to evaluate answer quality on multiple dimensions."""

    def __init__(self, llm_config: ModelConfig):
        """Initialize LLM judge.

        Args:
            llm_config: Configuration for the LLM to use for judging
        """
        self.llm_config = llm_config
        self.llm = None

    async def _get_llm(self):
        """Lazy load LLM instance."""
        if self.llm is None:
            self.llm = model_gateway.get_llm_from_model_config(self.llm_config)
        return self.llm

    async def evaluate(
        self,
        query: str,
        predicted: str,
        ground_truth: Optional[str] = None,
        sources: Optional[list] = None
    ) -> Dict[str, float]:
        """Evaluate answer using LLM judge.

        Evaluates on multiple criteria:
        - Relevance: How well does answer address the question
        - Accuracy: Factual correctness (if ground truth provided)
        - Completeness: Thoroughness of coverage
        - Coherence: Structure and clarity

        Args:
            query: Original user question
            predicted: Generated answer to evaluate
            ground_truth: Optional reference answer for comparison
            sources: Optional list of source documents used

        Returns:
            Dictionary of scores (1-5 scale) for each criterion
        """
        llm = await self._get_llm()

        ground_truth_section = ""
        if ground_truth:
            ground_truth_section = f"""
Reference Answer (Ground Truth):
{ground_truth}
"""

        sources_section = ""
        if sources:
            sources_text = "\n".join([f"- {src}" for src in sources[:3]])
            sources_section = f"""
Sources Used:
{sources_text}
"""

        prompt = f"""Evaluate the following answer on a scale of 1-5 for each criterion.

Question: {query}

Answer to Evaluate:
{predicted}
{ground_truth_section}{sources_section}

Provide scores for:
1. Relevance (1-5): How well does the answer address the question?
2. Accuracy (1-5): How factually correct is the answer?
3. Completeness (1-5): How thoroughly does the answer cover the topic?
4. Coherence (1-5): How well-structured and clear is the answer?

Respond ONLY with valid JSON in this exact format:
{{
    "relevance": <score>,
    "accuracy": <score>,
    "completeness": <score>,
    "coherence": <score>,
    "explanation": "<brief explanation>"
}}"""

        response = await llm.ainvoke(prompt)
        scores = self._parse_scores(response.content)

        return scores

    def _parse_scores(self, response_text: str) -> Dict[str, float]:
        """Parse scores from LLM response.

        Args:
            response_text: Raw LLM response

        Returns:
            Dictionary of normalized scores
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
            if json_match:
                scores_dict = json.loads(json_match.group())
            else:
                scores_dict = json.loads(response_text)

            # Extract and normalize scores
            scores = {
                "relevance": self._normalize_score(scores_dict.get("relevance", 3)),
                "accuracy": self._normalize_score(scores_dict.get("accuracy", 3)),
                "completeness": self._normalize_score(scores_dict.get("completeness", 3)),
                "coherence": self._normalize_score(scores_dict.get("coherence", 3)),
            }

            # Calculate overall score
            scores["overall"] = sum(scores.values()) / len(scores)

            return scores

        except (json.JSONDecodeError, ValueError):
            # Fallback: try to extract numbers from text
            return self._fallback_parse(response_text)

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range from 1-5 scale.

        Args:
            score: Score in 1-5 range

        Returns:
            Normalized score in 0-1 range
        """
        # Clamp to valid range
        score = max(1, min(5, score))
        # Normalize to 0-1
        return (score - 1) / 4

    def _fallback_parse(self, response_text: str) -> Dict[str, float]:
        """Fallback parser when JSON parsing fails.

        Args:
            response_text: Raw response text

        Returns:
            Dictionary of scores with default values
        """
        scores = {
            "relevance": 0.5,
            "accuracy": 0.5,
            "completeness": 0.5,
            "coherence": 0.5,
        }

        # Try to extract individual scores
        patterns = {
            "relevance": r"relevance[:\s]+(\d+)",
            "accuracy": r"accuracy[:\s]+(\d+)",
            "completeness": r"completeness[:\s]+(\d+)",
            "coherence": r"coherence[:\s]+(\d+)",
        }

        for criterion, pattern in patterns.items():
            match = re.search(pattern, response_text.lower())
            if match:
                score = float(match.group(1))
                scores[criterion] = self._normalize_score(score)

        scores["overall"] = sum(scores.values()) / len(scores)

        return scores

    async def evaluate_batch(
        self,
        evaluations: list[Dict]
    ) -> list[Dict[str, float]]:
        """Evaluate multiple query-answer pairs in batch.

        Args:
            evaluations: List of dicts with 'query', 'predicted', 'ground_truth'

        Returns:
            List of score dictionaries
        """
        results = []

        for eval_item in evaluations:
            scores = await self.evaluate(
                query=eval_item["query"],
                predicted=eval_item["predicted"],
                ground_truth=eval_item.get("ground_truth"),
                sources=eval_item.get("sources")
            )
            results.append(scores)

        return results
