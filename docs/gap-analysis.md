# Cognita Gap Analysis - Modular RAG Paradigm

**Generated:** 2026-01-31 | **Phase:** 1 - Deep Code Inspection

## Overview

This analysis maps Cognita's current capabilities against the Modular RAG paradigm to identify gaps for enhancement phases.

## Component Status Matrix

| Component | Status | Current State | Gap Description | Phase |
|-----------|--------|---------------|-----------------|-------|
| **Indexing Module** | Done | Incremental indexing with hash-based change detection | None | - |
| **Pre-Retrieval** | Missing | No query processing | Need query rewriting, decomposition, expansion | 2 |
| **Retrieval Module** | Partial | Vector + basic reranking | Need hybrid search, BM25, advanced strategies | 3 |
| **Post-Retrieval** | Partial | Single-stage reranking via Infinity | Need multi-stage, fusion, compression | 3 |
| **Generation Module** | Done | LCEL chains with streaming | None | - |
| **Orchestration** | Missing | No adaptive routing | Need conditional pipelines, query routing | 4 |
| **Verification** | Missing | No QC | Need hallucination detection, confidence scoring | 5 |
| **Observability** | Missing | No tracing/metrics | Need full observability stack | 6 |
| **Evaluation** | Missing | No eval framework | Need automated RAG evaluation | 7 |

---

## Detailed Gap Analysis

### Gap 1: Pre-Retrieval Query Processing (Phase 2)

**Current State:** Query passed directly to retriever without processing.

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Query Rewriting | LLM-based query reformulation | P1 | Medium |
| Query Decomposition | Break complex queries into sub-queries | P1 | Medium |
| HyDE | Hypothetical Document Embeddings | P2 | Medium |
| Step-back Prompting | Generate abstract queries first | P2 | Low |
| Query Expansion | Add synonyms/related terms | P3 | Low |

**Required Changes:**
- New module: `backend/modules/query_processing/`
- Query processing pipeline before retrieval
- Configuration for processing strategies
- Integration with existing retriever factory

**Code Location:** `backend/modules/query_controllers/base.py:_get_retriever()`

---

### Gap 2: Advanced Retrieval Strategies (Phase 3)

**Current State:**
- Vector similarity search (Qdrant)
- Basic reranking via Infinity
- Multi-query retriever (query decomposition only)

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Hybrid Search | BM25 + Vector fusion | P1 | High |
| Multi-vector Retrieval | Multiple embeddings per chunk | P2 | Medium |
| Parent-Child Retrieval | Retrieve small, return large | P2 | Medium |
| Recursive Retrieval | Hierarchical document traversal | P3 | High |
| Semantic Caching | Cache similar queries | P2 | Medium |

**Required Changes:**
- BM25 implementation (Elasticsearch integration per plan)
- Fusion algorithms (RRF, linear combination)
- Extended VectorDB interface for multi-vector
- Caching layer for semantic similarity

**Code Locations:**
- `backend/modules/vector_db/base.py` - Add hybrid search methods
- `backend/modules/query_controllers/base.py` - New retriever types

---

### Gap 3: Orchestration Engine (Phase 4)

**Current State:** Fixed linear pipeline (retrieve → generate).

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Query Routing | Route to specialized retrievers | P1 | Medium |
| Conditional Pipelines | If-then logic in RAG flow | P1 | Medium |
| Adaptive Retrieval | Dynamic k, strategy selection | P2 | Medium |
| Self-RAG | Retrieve only when needed | P2 | High |
| Agent Orchestration | Multi-tool RAG agents | P3 | High |

**Required Changes:**
- New module: `backend/modules/orchestration/`
- Pipeline DSL or YAML configuration
- Router component with intent classification
- State machine for complex flows

**Code Location:** New module needed

---

### Gap 4: Verification & Quality Control (Phase 5)

**Current State:** No answer verification.

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Hallucination Detection | LLM-as-Judge verification | P1 | Medium |
| Source Attribution | Verify claims against sources | P1 | Medium |
| Confidence Scoring | Answer quality estimation | P2 | Medium |
| Fact Checking | External knowledge validation | P3 | High |
| Answer Regeneration | Retry on low confidence | P2 | Low |

**Required Changes:**
- New module: `backend/modules/verification/`
- Post-generation verification pipeline
- Confidence threshold configuration
- Retry/fallback mechanisms

**Code Location:** Integrate after generation in `base.py`

---

### Gap 5: Observability & Monitoring (Phase 6)

**Current State:** Basic logging only.

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Distributed Tracing | Trace across components | P1 | Medium |
| Metrics Collection | Latency, throughput, errors | P1 | Medium |
| LLM Token Tracking | Usage per request | P1 | Low |
| Dashboard | Real-time monitoring | P2 | Medium |
| Alerting | Threshold-based alerts | P2 | Low |

**Required Changes:**
- OpenTelemetry integration
- Prometheus metrics export
- Trace context propagation
- Grafana dashboard templates

**Code Location:** Middleware in `backend/server/`

---

### Gap 6: Evaluation Framework (Phase 7)

**Current State:** No automated evaluation.

**Missing Capabilities:**

| Technique | Description | Priority | Complexity |
|-----------|-------------|----------|------------|
| Retrieval Metrics | Precision, Recall, MRR, NDCG | P1 | Medium |
| Generation Metrics | Faithfulness, relevance, coherence | P1 | Medium |
| End-to-End Eval | Full pipeline scoring | P1 | Medium |
| Regression Testing | Track quality over time | P2 | Medium |
| A/B Testing | Compare configurations | P3 | High |

**Required Changes:**
- New module: `backend/modules/evaluation/`
- Eval dataset management
- Metrics calculation library
- Integration with CI/CD

**Code Location:** New module + CLI commands

---

## Implementation Priority Matrix

| Priority | Components | Effort | Impact | Dependencies |
|----------|------------|--------|--------|--------------|
| P1-Critical | Query Processing, Hybrid Retrieval | 4 weeks | High | Phase 1 complete |
| P1-Critical | Orchestration Engine | 2 weeks | High | Query + Retrieval |
| P1-High | Verification, Observability | 3 weeks | High | Orchestration |
| P2-Medium | Evaluation Framework | 2 weeks | Medium | Observability |
| P3-Nice | Domain Extensions | 2 weeks | Medium | All prior |

---

## Current vs Target Architecture

### Current
```
Query → VectorStore → Reranker → LLM → Response
```

### Target (Modular RAG)
```
Query → [Query Processing] → [Router] → [Hybrid Retrieval] → [Multi-stage Rerank]
                                              ↓
                              [Verification] ← [LLM Generation]
                                              ↓
                              [Observability] → Response
```

---

## Success Metrics Gaps

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Query latency (p95) | ~3-5s | <2s | Need caching, optimization |
| Hallucination rate | Unknown | <5% | Need detection system |
| Source attribution | Unknown | >95% | Need verification |
| Cache hit rate | 0% | >60% | Need semantic cache |
| Test coverage | Low | >80% | Need eval framework |

---

## Recommendations

1. **Phase 2-3 Parallel Execution**: Query Processing and Advanced Retrieval have no dependencies on each other
2. **Elasticsearch First**: Critical for BM25 hybrid search - add to Docker Compose
3. **LLM-as-Judge Pattern**: Use Ollama for verification to avoid external API costs
4. **OpenTelemetry**: Industry standard, integrates with existing infrastructure
5. **YAML Pipelines**: Human-readable orchestration configuration per validation
