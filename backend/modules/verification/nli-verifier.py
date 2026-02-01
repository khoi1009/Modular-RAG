"""Optional Natural Language Inference verifier for entailment checking."""
import asyncio
from typing import Dict, List, Optional, Tuple

import async_timeout
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.types import ModelConfig


class NLIVerifier:
    """
    Natural Language Inference verifier using LLM for entailment checking.
    Can be used as alternative or complement to hallucination detection.
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        self.timeout = config.get("timeout", 20)

    async def verify_entailment(
        self, premise: str, hypothesis: str
    ) -> Tuple[str, float]:
        """
        Check if hypothesis is entailed by premise.

        Args:
            premise: Source text (from documents)
            hypothesis: Generated text to verify (claim or answer)

        Returns:
            Tuple of (label, confidence) where label is ENTAILMENT/NEUTRAL/CONTRADICTION
        """
        try:
            async with async_timeout.timeout(self.timeout):
                llm = model_gateway.get_llm_from_model_config(self.model_config)

                prompt = PromptTemplate(
                    input_variables=["premise", "hypothesis"],
                    template="""Determine the relationship between the premise and hypothesis.

Premise: {premise}

Hypothesis: {hypothesis}

Classify the relationship as one of:
- ENTAILMENT: The hypothesis logically follows from the premise
- NEUTRAL: The hypothesis is neither supported nor contradicted by the premise
- CONTRADICTION: The hypothesis contradicts the premise

Also provide a confidence score from 0.0 to 1.0.

Format your response as:
Label: [ENTAILMENT/NEUTRAL/CONTRADICTION]
Confidence: [0.0-1.0]

Response:""",
                )

                chain = prompt | llm
                response = await chain.ainvoke(
                    {"premise": premise, "hypothesis": hypothesis}
                )

                label, confidence = self._parse_nli_response(response.content)
                return label, confidence

        except asyncio.TimeoutError:
            logger.warning("NLI verification timeout")
            return "NEUTRAL", 0.5
        except Exception as e:
            logger.error(f"NLI verification error: {e}")
            return "NEUTRAL", 0.5

    async def verify_answer_entailment(
        self, answer: str, sources: List[Document]
    ) -> Tuple[bool, float]:
        """
        Verify if answer is entailed by source documents.

        Args:
            answer: Generated answer
            sources: Source documents

        Returns:
            Tuple of (is_entailed, confidence_score)
        """
        try:
            if not sources or not answer:
                return False, 0.0

            # Build premise from top sources
            premise = self._build_premise(sources)

            # Check entailment
            label, confidence = await self.verify_entailment(premise, answer)

            is_entailed = label == "ENTAILMENT"

            logger.debug(
                f"NLI verification - label: {label}, confidence: {confidence:.3f}"
            )

            return is_entailed, confidence

        except Exception as e:
            logger.error(f"Answer entailment verification error: {e}")
            return False, 0.5

    async def batch_verify(
        self, hypotheses: List[str], sources: List[Document]
    ) -> List[Tuple[bool, float]]:
        """
        Verify multiple hypotheses against sources in parallel.

        Args:
            hypotheses: List of claims/statements to verify
            sources: Source documents

        Returns:
            List of (is_entailed, confidence) tuples
        """
        premise = self._build_premise(sources)

        tasks = [self.verify_entailment(premise, hyp) for hyp in hypotheses]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert to boolean results
        verified_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch verification error: {result}")
                verified_results.append((False, 0.5))
            else:
                label, confidence = result
                is_entailed = label == "ENTAILMENT"
                verified_results.append((is_entailed, confidence))

        return verified_results

    def _build_premise(self, sources: List[Document]) -> str:
        """Build premise text from source documents"""
        premise_parts = []

        for i, doc in enumerate(sources[:3], 1):  # Use top 3 sources
            content = doc.page_content[:500]  # Truncate to manage token limits
            premise_parts.append(f"[Source {i}]: {content}")

        return "\n\n".join(premise_parts)

    def _parse_nli_response(self, response: str) -> Tuple[str, float]:
        """Parse NLI label and confidence from LLM response"""
        try:
            response = response.strip().upper()

            # Extract label
            label = "NEUTRAL"
            if "ENTAILMENT" in response and "CONTRADICTION" not in response:
                label = "ENTAILMENT"
            elif "CONTRADICTION" in response:
                label = "CONTRADICTION"
            elif "NEUTRAL" in response:
                label = "NEUTRAL"

            # Extract confidence score
            confidence = 0.5
            import re

            score_match = re.search(r"CONFIDENCE[:\s]+(\d*\.?\d+)", response)
            if score_match:
                confidence = float(score_match.group(1))
                # Normalize if needed
                if confidence > 1.0:
                    confidence = confidence / 10.0 if confidence <= 10.0 else 1.0
                confidence = max(0.0, min(1.0, confidence))

            return label, confidence

        except Exception as e:
            logger.error(f"NLI response parsing error: {e}")
            return "NEUTRAL", 0.5
