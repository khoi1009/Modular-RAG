---
title: "Cognita Enhancement Master Plan"
description: "Transform Cognita into production-ready Modular RAG with orchestration, verification, and observability"
status: completed
priority: P1
effort: 12w
branch: main
tags: [rag, orchestration, retrieval, verification, observability, ollama]
created: 2026-01-31
---

# Cognita Enhancement Master Plan

**Timeline:** 12 Weeks | **Priority:** P1 | **Status:** Completed

## Vision

Transform Cognita from modular RAG infrastructure into a full Modular RAG system with:
- Advanced Orchestration Layer (adaptive retrieval, query routing, conditional pipelines)
- Query Intelligence (rewriting, decomposition, expansion via HyDE, Step-back)
- Verification & QC (hallucination detection, confidence scoring)
- Production Monitoring (observability, evaluation, feedback loops)
- Domain Extensions (water infrastructure intelligence)

## Phase Overview

| Phase | Name | Duration | Status | Depends On |
|-------|------|----------|--------|------------|
| 1 | [Deep Code Inspection](phase-01-deep-code-inspection.md) | Week 1 | completed | - |
| 2 | [Query Intelligence Layer](phase-02-query-intelligence-layer.md) | Weeks 2-3 | completed | Phase 1 |
| 3 | [Advanced Retrieval Layer](phase-03-advanced-retrieval-layer.md) | Weeks 4-5 | completed | Phase 1 |
| 4 | [Orchestration Engine](phase-04-orchestration-engine.md) | Weeks 6-7 | completed | Phases 2, 3 |
| 5 | [Verification & QC](phase-05-verification-quality-control.md) | Week 8 | completed | Phase 4 |
| 6 | [Observability & Monitoring](phase-06-observability-monitoring.md) | Week 9 | completed | Phase 4 |
| 7 | [Evaluation Framework](phase-07-evaluation-framework.md) | Week 10 | completed | Phase 6 |
| 8 | [Domain Extensions](phase-08-domain-extensions.md) | Week 11 | completed | Phases 4, 5 |
| 9 | [Performance Optimization](phase-09-performance-optimization.md) | Week 12 | completed | All prior |

## Key Architecture Decisions

1. **LLM Provider:** Prioritize Ollama (local models) - configured in `models_config.yaml`
2. **Deployment:** Docker Compose (dev) + Kubernetes (production)
3. **Base Classes:** Extend `ConfiguredBaseModel` with `model_config = ConfigDict(use_enum_values=True)`
4. **Chains:** Use LangChain Expression Language (LCEL) for RAG pipelines
5. **Async:** All I/O operations use `async/await` with `async_timeout`
6. **Module Pattern:** New modules in `backend/modules/{module_name}/` with `__init__.py` registry

## Success Metrics

- Query latency < 2s (p95)
- Hallucination rate < 5%
- Source attribution > 95%
- Cache hit rate > 60%
- Test coverage > 80%

## Reports

- [Architecture Report](research/researcher-architecture-report.md)
- [Modular RAG Patterns Report](research/researcher-modular-rag-patterns-report.md)

---

## Validation Summary

**Validated:** 2026-01-31
**Questions asked:** 6

### Confirmed Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| BM25 Implementation | Elasticsearch | Scalable, Docker-friendly, production-ready |
| Hallucination Detection | LLM-as-Judge | Works with Ollama, ~75% accuracy, no extra deps |
| Phase 2 & 3 Execution | Parallel | Both depend only on Phase 1, saves ~1 week |
| Domain Extensions | Generic Template | Create extensible pattern, water as example |
| Pipeline Config | YAML files | Human-readable, version-controllable |
| Test Coverage | 80% | Standard high-quality threshold |

### Action Items

- [ ] Update Phase 3 to use Elasticsearch for BM25 (replace rank_bm25 references)
- [ ] Update Phase 8 to create generic domain template with water as example
- [ ] Adjust timeline: Phases 2 & 3 run parallel (Weeks 2-4), Phase 4 starts Week 5
