"""Schemas for verification and quality control."""
from typing import List, Optional

from backend.types import ConfiguredBaseModel


class Claim(ConfiguredBaseModel):
    """A verifiable claim extracted from text"""
    text: str
    start_idx: int
    end_idx: int
    claim_type: str  # factual, opinion, conditional


class ClaimVerification(ConfiguredBaseModel):
    """Verification result for a single claim"""
    claim: Claim
    is_supported: bool
    support_score: float  # 0.0-1.0
    supporting_sources: List[str]  # Document IDs or references
    explanation: Optional[str] = None


class VerificationResult(ConfiguredBaseModel):
    """Complete verification result for an answer"""
    is_grounded: bool
    hallucination_score: float  # 0.0=grounded, 1.0=hallucinated
    confidence: float
    claim_verifications: List[ClaimVerification]
    unsupported_claims: List[Claim]
    attribution_coverage: float  # % of claims with sources
    internal_consistency: float
    verification_time_ms: int
