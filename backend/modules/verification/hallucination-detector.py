"""Detect hallucinated content in generated answers."""
import asyncio
from typing import Dict, List, Optional, Tuple

import async_timeout
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from backend.logger import logger
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.verification.schemas import Claim, ClaimVerification
from backend.types import ModelConfig


class HallucinationDetector:
    """Detect hallucinated content using LLM-as-Judge pattern"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.model_config = ModelConfig(
            name=config.get("model_name", "ollama/llama3"),
            parameters=config.get("model_parameters", {"temperature": 0.0}),
        )
        # Import with kebab-case module name
        from backend.modules.verification import claim_extractor as claim_extractor_module
        self.claim_extractor = claim_extractor_module.ClaimExtractor(config)
        self.threshold = config.get("threshold", 0.5)
        self.timeout = config.get("timeout", 30)

    async def detect(
        self, answer: str, sources: List[Document]
    ) -> Tuple[float, List[ClaimVerification]]:
        """
        Detect hallucinations in answer.
        Returns: (hallucination_score, claim_verifications)
        """
        try:
            async with async_timeout.timeout(self.timeout):
                # Extract claims from answer
                claims = await self.claim_extractor.extract_claims(answer)

                if not claims:
                    return 0.0, []

                # Verify each claim against sources
                verifications = []
                for claim in claims:
                    verification = await self._verify_claim(claim, sources)
                    verifications.append(verification)

                # Calculate overall hallucination score
                if verifications:
                    total_support = sum(v.support_score for v in verifications)
                    hallucination_score = 1.0 - (total_support / len(verifications))
                else:
                    hallucination_score = 0.0

                return hallucination_score, verifications

        except asyncio.TimeoutError:
            logger.warning("Hallucination detection timeout")
            return 0.0, []
        except Exception as e:
            logger.error(f"Hallucination detection error: {e}")
            return 0.0, []

    async def _verify_claim(
        self, claim: Claim, sources: List[Document]
    ) -> ClaimVerification:
        """Verify a single claim against source documents"""
        try:
            # Build context from sources
            context = self._build_context(sources)

            llm = model_gateway.get_llm_from_model_config(self.model_config)

            prompt = PromptTemplate(
                input_variables=["claim", "context"],
                template="""Given the following source documents, determine if the claim is supported.

Source Documents:
{context}

Claim: {claim}

Is this claim supported by the sources? Answer with:
- SUPPORTED: if the claim is directly stated or strongly implied
- PARTIAL: if the claim is partially supported
- UNSUPPORTED: if the claim contradicts or is not mentioned in sources

Answer (SUPPORTED/PARTIAL/UNSUPPORTED):""",
            )

            chain = prompt | llm
            response = await chain.ainvoke({"claim": claim.text, "context": context})

            # Parse response
            answer = response.content.strip().upper()
            if "SUPPORTED" in answer and "UNSUPPORTED" not in answer:
                support_score = 1.0
                is_supported = True
            elif "PARTIAL" in answer:
                support_score = 0.5
                is_supported = False
            else:
                support_score = 0.0
                is_supported = False

            # Find supporting sources
            supporting_sources = self._find_supporting_sources(claim, sources)

            return ClaimVerification(
                claim=claim,
                is_supported=is_supported,
                support_score=support_score,
                supporting_sources=supporting_sources,
                explanation=answer,
            )

        except Exception as e:
            logger.error(f"Claim verification error: {e}")
            return ClaimVerification(
                claim=claim,
                is_supported=False,
                support_score=0.0,
                supporting_sources=[],
                explanation="Verification failed",
            )

    def _build_context(self, sources: List[Document]) -> str:
        """Build context string from source documents"""
        context_parts = []
        for i, doc in enumerate(sources[:5], 1):  # Limit to top 5 sources
            content = doc.page_content[:500]  # Truncate long docs
            context_parts.append(f"[Source {i}]: {content}")
        return "\n\n".join(context_parts)

    def _find_supporting_sources(
        self, claim: Claim, sources: List[Document]
    ) -> List[str]:
        """Find source documents that support the claim"""
        supporting = []
        claim_lower = claim.text.lower()

        for doc in sources:
            content_lower = doc.page_content.lower()
            # Simple keyword matching - could be improved with embeddings
            if any(
                word in content_lower for word in claim_lower.split() if len(word) > 3
            ):
                doc_id = doc.metadata.get("_id", doc.metadata.get("source", "unknown"))
                supporting.append(doc_id)

        return supporting[:3]  # Return top 3
