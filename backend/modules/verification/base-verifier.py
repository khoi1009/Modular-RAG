"""Unified answer verification orchestrator combining all verification components."""
import time
from typing import Dict, List, Optional

from langchain_core.documents import Document

from backend.logger import logger
from backend.modules.verification.schemas import VerificationResult


class AnswerVerifier:
    """
    Unified answer verification system that orchestrates all verification components.
    Combines hallucination detection, source attribution, confidence scoring, and consistency checking.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize verifier with all sub-components.

        Args:
            config: Configuration dict with optional keys:
                - model_name: LLM model to use (default: "ollama/llama3")
                - model_parameters: Dict of model params
                - timeout: Timeout for verification operations
                - enable_nli: Whether to enable NLI verifier (default: False)
                - confidence_weights: Custom weights for confidence scoring
        """
        config = config or {}

        # Import components with kebab-case module names
        from backend.modules.verification import claim_extractor as claim_extractor_module
        from backend.modules.verification import confidence_scorer as confidence_scorer_module
        from backend.modules.verification import consistency_checker as consistency_checker_module
        from backend.modules.verification import hallucination_detector as hallucination_detector_module
        from backend.modules.verification import source_attributor as source_attributor_module

        # Initialize all verification components
        self.claim_extractor = claim_extractor_module.ClaimExtractor(config)
        self.hallucination_detector = hallucination_detector_module.HallucinationDetector(config)
        self.source_attributor = source_attributor_module.SourceAttributor(config)
        self.confidence_scorer = confidence_scorer_module.ConfidenceScorer(config)
        self.consistency_checker = consistency_checker_module.ConsistencyChecker(config)

        # Optional NLI verifier
        self.enable_nli = config.get("enable_nli", False)
        if self.enable_nli:
            from backend.modules.verification import nli_verifier as nli_verifier_module

            self.nli_verifier = nli_verifier_module.NLIVerifier(config)
        else:
            self.nli_verifier = None

        # Thresholds
        self.hallucination_threshold = config.get("hallucination_threshold", 0.5)
        self.min_confidence = config.get("min_confidence", 0.3)

    async def verify(
        self,
        query: str,
        answer: str,
        sources: List[Document],
        retrieval_scores: Optional[List[float]] = None,
    ) -> VerificationResult:
        """
        Verify an answer against source documents.

        Args:
            query: Original user query
            answer: Generated answer to verify
            sources: Retrieved source documents
            retrieval_scores: Optional relevance scores from retrieval

        Returns:
            VerificationResult with comprehensive verification metrics
        """
        start_time = time.time()

        try:
            logger.info(
                f"Starting answer verification - query length: {len(query)}, "
                f"answer length: {len(answer)}, sources: {len(sources)}"
            )

            # 1. Extract claims from answer
            claims = await self.claim_extractor.extract_claims(answer)
            logger.debug(f"Extracted {len(claims)} claims from answer")

            # 2. Detect hallucinations
            hallucination_score, claim_verifications = (
                await self.hallucination_detector.detect(answer, sources)
            )
            logger.debug(f"Hallucination score: {hallucination_score:.3f}")

            # 3. Attribute sources to claims
            attributions = await self.source_attributor.attribute(claims, sources)
            attribution_coverage = self.source_attributor.calculate_coverage(
                attributions
            )
            logger.debug(f"Attribution coverage: {attribution_coverage:.3f}")

            # 4. Check internal consistency
            consistency_score = await self.consistency_checker.check_internal_consistency(
                answer
            )
            logger.debug(f"Internal consistency: {consistency_score:.3f}")

            # 5. Optional: NLI verification
            if self.enable_nli and self.nli_verifier:
                nli_entailed, nli_confidence = (
                    await self.nli_verifier.verify_answer_entailment(answer, sources)
                )
                logger.debug(
                    f"NLI verification - entailed: {nli_entailed}, confidence: {nli_confidence:.3f}"
                )
                # Adjust hallucination score based on NLI
                if not nli_entailed:
                    hallucination_score = max(
                        hallucination_score, 1.0 - nli_confidence
                    )

            # 6. Calculate confidence score
            confidence = await self.confidence_scorer.score(
                query=query,
                answer=answer,
                sources=sources,
                retrieval_scores=retrieval_scores,
                hallucination_score=hallucination_score,
                consistency_score=consistency_score,
            )
            logger.debug(f"Overall confidence: {confidence:.3f}")

            # 7. Determine if answer is grounded
            is_grounded = self._determine_grounding(
                hallucination_score, attribution_coverage, consistency_score, confidence
            )

            # 8. Identify unsupported claims
            unsupported_claims = [
                cv.claim for cv in claim_verifications if not cv.is_supported
            ]

            # Calculate verification time
            verification_time_ms = int((time.time() - start_time) * 1000)

            result = VerificationResult(
                is_grounded=is_grounded,
                hallucination_score=hallucination_score,
                confidence=confidence,
                claim_verifications=claim_verifications,
                unsupported_claims=unsupported_claims,
                attribution_coverage=attribution_coverage,
                internal_consistency=consistency_score,
                verification_time_ms=verification_time_ms,
            )

            logger.info(
                f"Verification complete - grounded: {is_grounded}, "
                f"confidence: {confidence:.3f}, time: {verification_time_ms}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Answer verification error: {e}", exc_info=True)

            # Return safe default result on error
            verification_time_ms = int((time.time() - start_time) * 1000)
            return VerificationResult(
                is_grounded=False,
                hallucination_score=1.0,
                confidence=0.0,
                claim_verifications=[],
                unsupported_claims=[],
                attribution_coverage=0.0,
                internal_consistency=0.0,
                verification_time_ms=verification_time_ms,
            )

    def _determine_grounding(
        self,
        hallucination_score: float,
        attribution_coverage: float,
        consistency_score: float,
        confidence: float,
    ) -> bool:
        """
        Determine if answer is grounded based on multiple signals.

        An answer is considered grounded if:
        - Hallucination score is below threshold
        - Attribution coverage is reasonable
        - Internal consistency is acceptable
        - Overall confidence meets minimum
        """
        is_grounded = (
            hallucination_score < self.hallucination_threshold
            and attribution_coverage > 0.3
            and consistency_score > 0.5
            and confidence >= self.min_confidence
        )

        logger.debug(
            f"Grounding decision - hallucination: {hallucination_score:.3f} < {self.hallucination_threshold}, "
            f"coverage: {attribution_coverage:.3f} > 0.3, "
            f"consistency: {consistency_score:.3f} > 0.5, "
            f"confidence: {confidence:.3f} >= {self.min_confidence}, "
            f"result: {is_grounded}"
        )

        return is_grounded

    async def verify_with_citations(
        self,
        query: str,
        answer: str,
        sources: List[Document],
        retrieval_scores: Optional[List[float]] = None,
    ) -> tuple[VerificationResult, str]:
        """
        Verify answer and generate cited version.

        Args:
            query: Original user query
            answer: Generated answer
            sources: Source documents
            retrieval_scores: Optional retrieval scores

        Returns:
            Tuple of (VerificationResult, cited_answer)
        """
        # Perform standard verification
        result = await self.verify(query, answer, sources, retrieval_scores)

        # Generate citations
        claims = await self.claim_extractor.extract_claims(answer)
        attributions = await self.source_attributor.attribute(claims, sources)
        cited_answer = self.source_attributor.generate_citations(answer, attributions)

        return result, cited_answer
