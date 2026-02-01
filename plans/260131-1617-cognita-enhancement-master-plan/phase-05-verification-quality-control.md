# Phase 5: Verification & Quality Control

**Duration:** Week 8 | **Priority:** P1 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [Phase 4: Orchestration Engine](phase-04-orchestration-engine.md)
- [Modular RAG Patterns Report](research/researcher-modular-rag-patterns-report.md)

## Overview

Ensure answer quality through hallucination detection, source attribution verification, confidence scoring, and consistency checking. Critical for production trustworthiness.

## Key Insights

From research:
- LLM-as-Judge: General purpose, ~75% accuracy
- NLI models: Check if answer is entailed by sources
- Claim extraction: Split answer into verifiable claims
- Reference-free: Prometheus, HHEM for hallucination-specific scoring

Current state:
- No hallucination detection
- No source attribution
- No confidence scoring
- No answer validation

## Requirements

### Functional
- Hallucination Detector: Check answer grounding in retrieved documents
- Source Attributor: Map claims to source documents with citations
- Confidence Scorer: Estimate answer reliability (0.0-1.0)
- Consistency Checker: Verify internal consistency and knowledge base alignment

### Non-Functional
- Verification latency < 500ms
- Async operations throughout
- Works with Ollama local models
- Graceful degradation (return unverified if fails)

## Architecture

### Module Structure
```
backend/modules/
└── verification/
    ├── __init__.py
    ├── base-verifier.py
    ├── hallucination-detector.py
    ├── source-attributor.py
    ├── confidence-scorer.py
    ├── consistency-checker.py
    ├── claim-extractor.py
    ├── nli-verifier.py
    └── schemas.py
```

### Verification Flow
```
Generated Answer + Retrieved Sources
    ↓
Claim Extractor → List[Claim]
    ↓
For each claim:
    ├── Hallucination Detector → is_grounded
    ├── Source Attributor → source_refs
    └── NLI Verifier → entailment_score
    ↓
Confidence Scorer → overall_confidence
    ↓
Consistency Checker → internal_consistency
    ↓
VerificationResult
```

## Related Code Files

### Files to Reference
- `backend/modules/orchestration/pipeline/pipeline-executor.py` - Integration point
- `backend/modules/model_gateway/model_gateway.py` - LLM access

### Files to Create
- `backend/modules/verification/__init__.py`
- `backend/modules/verification/base-verifier.py`
- `backend/modules/verification/hallucination-detector.py`
- `backend/modules/verification/source-attributor.py`
- `backend/modules/verification/confidence-scorer.py`
- `backend/modules/verification/consistency-checker.py`
- `backend/modules/verification/claim-extractor.py`
- `backend/modules/verification/nli-verifier.py`
- `backend/modules/verification/schemas.py`
- `tests/modules/verification/test_hallucination_detector.py`
- `tests/modules/verification/test_source_attributor.py`

## Implementation Steps

### Task 5.1: Claim Extraction & Hallucination Detection (Days 1-2)

1. Create `schemas.py`:
```python
class Claim(ConfiguredBaseModel):
    text: str
    start_idx: int
    end_idx: int
    claim_type: str  # factual, opinion, conditional

class ClaimVerification(ConfiguredBaseModel):
    claim: Claim
    is_supported: bool
    support_score: float  # 0.0 - 1.0
    supporting_sources: List[str]  # Document IDs
    explanation: Optional[str] = None

class VerificationResult(ConfiguredBaseModel):
    is_grounded: bool
    hallucination_score: float  # 0.0 = fully grounded, 1.0 = fully hallucinated
    confidence: float
    claim_verifications: List[ClaimVerification]
    unsupported_claims: List[Claim]
    attribution_coverage: float  # % of claims with sources
    internal_consistency: float
    verification_time_ms: int
```

2. Create `claim-extractor.py`:
```python
class ClaimExtractor:
    def __init__(self, llm_config: ModelConfig):
        self.llm = model_gateway.get_llm_from_model_config(llm_config)

    async def extract_claims(self, text: str) -> List[Claim]:
        prompt = """Extract verifiable factual claims from the following text.
        Return each claim as a separate item.

        Text: {text}

        Claims (one per line):"""

        response = await self.llm.ainvoke(prompt.format(text=text))
        return self._parse_claims(response.content, text)
```

3. Create `hallucination-detector.py`:
```python
class HallucinationDetector:
    def __init__(
        self,
        llm_config: ModelConfig,
        nli_verifier: Optional[NLIVerifier] = None,
        threshold: float = 0.5
    ):
        self.llm = model_gateway.get_llm_from_model_config(llm_config)
        self.nli_verifier = nli_verifier
        self.threshold = threshold

    async def detect(
        self, answer: str, sources: List[Document]
    ) -> Tuple[float, List[ClaimVerification]]:
        # Extract claims
        claims = await self.claim_extractor.extract_claims(answer)

        # Verify each claim
        verifications = []
        for claim in claims:
            if self.nli_verifier:
                # Use NLI model
                scores = await self.nli_verifier.verify(claim.text, sources)
                is_supported = max(scores) > self.threshold
            else:
                # Use LLM
                is_supported, score = await self._llm_verify(claim, sources)

            verifications.append(ClaimVerification(
                claim=claim,
                is_supported=is_supported,
                support_score=score,
                supporting_sources=self._find_supporting_sources(claim, sources)
            ))

        # Compute overall hallucination score
        hallucination_score = 1 - (
            sum(v.support_score for v in verifications) / len(verifications)
        ) if verifications else 0

        return hallucination_score, verifications
```

### Task 5.2: Source Attribution (Day 3)

1. Create `source-attributor.py`:
```python
class SourceAttributor:
    def __init__(self, similarity_threshold: float = 0.7):
        self.threshold = similarity_threshold

    async def attribute(
        self, claims: List[Claim], sources: List[Document]
    ) -> Dict[str, List[str]]:
        """Map each claim to supporting source document IDs"""
        attributions = {}

        for claim in claims:
            supporting = []
            for source in sources:
                similarity = await self._compute_similarity(claim.text, source.page_content)
                if similarity > self.threshold:
                    supporting.append(source.metadata.get("_id", "unknown"))
            attributions[claim.text] = supporting

        return attributions

    def generate_citations(
        self, answer: str, attributions: Dict[str, List[str]]
    ) -> str:
        """Insert citation markers into answer text"""
        # Add [1], [2], etc. after supported claims
        ...
```

### Task 5.3: Confidence Scoring (Day 4)

1. Create `confidence-scorer.py`:
```python
class ConfidenceScorer:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "retrieval_quality": 0.3,
            "source_agreement": 0.25,
            "hallucination_score": 0.25,
            "consistency_score": 0.2
        }

    async def score(
        self,
        query: str,
        answer: str,
        sources: List[Document],
        retrieval_scores: List[float],
        hallucination_score: float,
        consistency_score: float
    ) -> float:
        # Retrieval quality (average relevance score)
        retrieval_quality = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0

        # Source agreement (how many sources agree)
        source_agreement = await self._compute_source_agreement(answer, sources)

        # Combine with weights
        confidence = (
            self.weights["retrieval_quality"] * retrieval_quality +
            self.weights["source_agreement"] * source_agreement +
            self.weights["hallucination_score"] * (1 - hallucination_score) +
            self.weights["consistency_score"] * consistency_score
        )

        return min(max(confidence, 0.0), 1.0)
```

### Task 5.4: Consistency Checker (Day 5)

1. Create `consistency-checker.py`:
```python
class ConsistencyChecker:
    def __init__(self, llm_config: ModelConfig):
        self.llm = model_gateway.get_llm_from_model_config(llm_config)

    async def check_internal_consistency(self, answer: str) -> float:
        """Check if answer contradicts itself"""
        prompt = """Analyze the following text for internal contradictions.
        Rate consistency from 0 (contradictory) to 1 (fully consistent).

        Text: {answer}

        Consistency score (0-1):"""

        response = await self.llm.ainvoke(prompt.format(answer=answer))
        return self._parse_score(response.content)

    async def check_knowledge_consistency(
        self, answer: str, knowledge_base: Optional[List[str]] = None
    ) -> float:
        """Check if answer contradicts known facts"""
        if not knowledge_base:
            return 1.0  # No KB to check against

        # Compare against knowledge base entries
        ...
```

2. Create unified `base-verifier.py`:
```python
class AnswerVerifier:
    def __init__(self, config: Dict[str, Any]):
        self.hallucination_detector = HallucinationDetector(config)
        self.source_attributor = SourceAttributor(config)
        self.confidence_scorer = ConfidenceScorer(config)
        self.consistency_checker = ConsistencyChecker(config)

    async def verify(
        self, query: str, answer: str, sources: List[Document]
    ) -> VerificationResult:
        start_time = time.time()

        # Run verifications in parallel where possible
        hallucination_score, claim_verifications = await self.hallucination_detector.detect(
            answer, sources
        )

        attributions = await self.source_attributor.attribute(
            [cv.claim for cv in claim_verifications], sources
        )

        consistency = await self.consistency_checker.check_internal_consistency(answer)

        confidence = await self.confidence_scorer.score(
            query, answer, sources,
            retrieval_scores=[s.metadata.get("relevance_score", 0.5) for s in sources],
            hallucination_score=hallucination_score,
            consistency_score=consistency
        )

        return VerificationResult(
            is_grounded=hallucination_score < 0.3,
            hallucination_score=hallucination_score,
            confidence=confidence,
            claim_verifications=claim_verifications,
            unsupported_claims=[cv.claim for cv in claim_verifications if not cv.is_supported],
            attribution_coverage=len([a for a in attributions.values() if a]) / len(attributions) if attributions else 0,
            internal_consistency=consistency,
            verification_time_ms=int((time.time() - start_time) * 1000)
        )
```

## Todo List

- [ ] Create verification module structure
- [ ] Implement Claim schema
- [ ] Implement ClaimVerification schema
- [ ] Implement VerificationResult schema
- [ ] Implement ClaimExtractor
- [ ] Implement HallucinationDetector
- [ ] Implement SourceAttributor
- [ ] Implement ConfidenceScorer
- [ ] Implement ConsistencyChecker
- [ ] Implement unified AnswerVerifier
- [ ] Add NLI model support (optional)
- [ ] Register verification step in pipeline
- [ ] Write unit tests
- [ ] Integration test with orchestration

## Success Criteria

- Hallucination detection accuracy > 75%
- Source attribution coverage > 90%
- Confidence scores correlate with actual correctness
- Verification adds < 500ms latency
- Works with Ollama local models

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM verification unreliable | Medium | High | Use NLI model as primary |
| Claim extraction fails | Medium | Medium | Fallback to sentence splitting |
| Performance too slow | Medium | Medium | Parallelize, cache common patterns |

## Security Considerations

- Don't expose verification internals in API response
- Log verification failures for analysis
- Rate limit verification requests

## Next Steps

After Phase 5:
- Phase 6 (Observability) tracks verification metrics
- Phase 7 (Evaluation) benchmarks verification accuracy
- Phase 8 (Domain) adds domain-specific verification rules
