# Phase 1: Deep Code Inspection & Architecture Mapping

**Duration:** Week 1 | **Priority:** P1 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [Architecture Report](research/researcher-architecture-report.md)
- [Modular RAG Patterns](research/researcher-modular-rag-patterns-report.md)
- [Cognita README](../../README.md)

## Overview

Understand every component of Cognita to build on solid foundation. Map existing patterns, identify extension points, create gap analysis against Modular RAG paradigm.

## Key Insights

From architecture research:
- Query Controllers extend `BaseQueryController`, use `@query_controller` decorator
- ModelGateway singleton manages chat/embedding/reranking/audio models
- 4 retriever strategies: vectorstore, contextual-compression, multi-query, contextual-compression-multi-query
- All models use `ConfiguredBaseModel` with Pydantic v2
- LCEL chains for RAG pipelines with async streaming

## Requirements

### Functional
- Document all existing query controllers and responsibilities
- Map data loader implementations and extension points
- Document parser modules and chunking strategies
- Analyze embedder integrations and vector DB abstraction
- Create dependency graph showing component interactions

### Non-Functional
- Documentation must be actionable for implementation phases
- Diagrams in Mermaid format for maintainability
- Extension points clearly identified with code references

## Architecture

### Current Stack Layers
```
API Layer (QueryControllers via FastAPI)
    |
Business Logic (LCEL RAG chains)
    |
Integration Layer (ModelGateway, VectorDB, MetadataStore)
    |
External Services (Ollama, Infinity, Vector DBs)
```

### Key Patterns Identified
1. **Decorator Registration**: `@query_controller("/route")` → FastAPI router
2. **Factory with Caching**: `model_gateway.get_llm_from_model_config()` with LRU
3. **Abstract Base Classes**: `BaseVectorDB`, `BaseDataLoader`, `BaseParser`
4. **Configuration-Driven**: YAML config → Pydantic models → runtime behavior

## Related Code Files

### Files to Analyze
- `backend/modules/query_controllers/base.py` - Base controller utilities
- `backend/modules/query_controllers/example/controller.py` - Implementation example
- `backend/modules/model_gateway/model_gateway.py` - Model factory singleton
- `backend/modules/vector_db/base.py` - VectorDB interface
- `backend/modules/dataloaders/loader.py` - DataLoader base
- `backend/modules/parsers/parser.py` - Parser base
- `backend/types.py` - Core type definitions

### Files to Create
- `docs/architecture-analysis.md` - Comprehensive findings
- `docs/architecture-diagram.mermaid` - Visual representation
- `docs/extension-points.md` - Injection points for new logic
- `docs/gap-analysis.md` - Missing Modular RAG components
- `docs/integration-patterns.md` - Code patterns and examples

## Implementation Steps

### Task 1.1: Core Architecture Analysis (Days 1-2)

1. Examine `backend/modules/` directory structure
2. Document each query controller:
   - `BasicRAGQueryController` - Standard RAG flow
   - `MultiModalQueryController` - Vision/audio support
3. Trace data flow: ingestion → embedding → storage → retrieval → generation
4. Map configuration loading from `models_config.yaml`
5. Document API layer: FastAPI routes, request/response schemas

### Task 1.2: Integration Patterns Study (Days 3-4)

1. Analyze query controller orchestration patterns:
   - Vector store retrieval via `_get_vector_store()`
   - LLM instantiation via `_get_llm()`
   - Retriever factory via `_get_retriever()`
2. Study retriever pattern:
   - Base classes and interfaces
   - Registration and discovery
   - Configuration flow
3. Document embedder integration:
   - Model swapping via config
   - Batching patterns
   - Error handling

### Task 1.3: Gap Analysis Report (Day 5)

Map current features to Modular RAG components:

| Component | Status | Notes |
|-----------|--------|-------|
| Indexing Module | Done | Incremental indexing supported |
| Pre-Retrieval (Query Processing) | Missing | No rewriting, decomposition |
| Retrieval Module | Partial | Basic vector + reranking |
| Post-Retrieval (Reranking) | Partial | Single-stage only |
| Generation Module | Done | LCEL chains |
| Orchestration Module | Missing | No adaptive routing |
| Verification Module | Missing | No hallucination detection |
| Observability | Missing | No tracing/metrics |

## Todo List

- [x] Document query controller architecture
- [x] Map data loader implementations
- [x] Document parser/chunking strategies
- [x] Analyze embedder integrations
- [x] Document vector DB abstraction
- [x] Create dependency graph
- [x] Write architecture-analysis.md
- [x] Create architecture-diagram.mermaid
- [x] Write extension-points.md
- [x] Write gap-analysis.md
- [x] Write integration-patterns.md

## Success Criteria

- All existing components documented with code references
- Mermaid diagram renders correctly
- Extension points identified for each new module
- Gap analysis prioritized by impact and complexity
- Documentation reviewed and actionable

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Undocumented code patterns | Medium | Medium | Read all source files, run tests |
| Hidden dependencies | Low | High | Trace imports, check requirements.txt |
| Outdated README | Medium | Low | Compare docs with actual code |

## Security Considerations

- Document any auth/authorization patterns found
- Identify sensitive config (API keys, credentials)
- Note any security middleware or decorators

## Next Steps

After Phase 1 completion:
1. Phase 2 (Query Intelligence) and Phase 3 (Advanced Retrieval) can start in parallel
2. Architecture docs inform all subsequent implementations
3. Gap analysis prioritizes feature development order
