"""Verification and Quality Control module for RAG answers."""

# Import schemas
from backend.modules.verification.schemas import (
    Claim,
    ClaimVerification,
    VerificationResult,
)

# Import main components - use dynamic imports for kebab-case modules
def _import_components():
    """Dynamically import components from kebab-case module files"""
    import importlib

    # Import claim extractor
    claim_extractor_module = importlib.import_module(
        "backend.modules.verification.claim-extractor"
    )
    ClaimExtractor = claim_extractor_module.ClaimExtractor

    # Import hallucination detector
    hallucination_detector_module = importlib.import_module(
        "backend.modules.verification.hallucination-detector"
    )
    HallucinationDetector = hallucination_detector_module.HallucinationDetector

    # Import source attributor
    source_attributor_module = importlib.import_module(
        "backend.modules.verification.source-attributor"
    )
    SourceAttributor = source_attributor_module.SourceAttributor

    # Import confidence scorer
    confidence_scorer_module = importlib.import_module(
        "backend.modules.verification.confidence-scorer"
    )
    ConfidenceScorer = confidence_scorer_module.ConfidenceScorer

    # Import consistency checker
    consistency_checker_module = importlib.import_module(
        "backend.modules.verification.consistency-checker"
    )
    ConsistencyChecker = consistency_checker_module.ConsistencyChecker

    # Import NLI verifier
    nli_verifier_module = importlib.import_module(
        "backend.modules.verification.nli-verifier"
    )
    NLIVerifier = nli_verifier_module.NLIVerifier

    # Import base verifier
    base_verifier_module = importlib.import_module(
        "backend.modules.verification.base-verifier"
    )
    AnswerVerifier = base_verifier_module.AnswerVerifier

    return {
        "ClaimExtractor": ClaimExtractor,
        "HallucinationDetector": HallucinationDetector,
        "SourceAttributor": SourceAttributor,
        "ConfidenceScorer": ConfidenceScorer,
        "ConsistencyChecker": ConsistencyChecker,
        "NLIVerifier": NLIVerifier,
        "AnswerVerifier": AnswerVerifier,
    }


# Import all components
_components = _import_components()
ClaimExtractor = _components["ClaimExtractor"]
HallucinationDetector = _components["HallucinationDetector"]
SourceAttributor = _components["SourceAttributor"]
ConfidenceScorer = _components["ConfidenceScorer"]
ConsistencyChecker = _components["ConsistencyChecker"]
NLIVerifier = _components["NLIVerifier"]
AnswerVerifier = _components["AnswerVerifier"]

__all__ = [
    # Schemas
    "Claim",
    "ClaimVerification",
    "VerificationResult",
    # Components
    "ClaimExtractor",
    "HallucinationDetector",
    "SourceAttributor",
    "ConfidenceScorer",
    "ConsistencyChecker",
    "NLIVerifier",
    # Main verifier
    "AnswerVerifier",
]
