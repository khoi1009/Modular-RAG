# Phase 2: Query Intelligence Layer

**Duration:** Weeks 2-3 | **Priority:** P1 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [Phase 1: Deep Code Inspection](phase-01-deep-code-inspection.md)
- [Modular RAG Patterns Report](research/researcher-modular-rag-patterns-report.md)
- [BaseQueryController](../../backend/modules/query_controllers/base.py)

## Overview

Add advanced query processing capabilities: analysis, rewriting (HyDE, Step-back, decomposition), and context enhancement. Enables intelligent preprocessing before retrieval.

## Key Insights

From research:
- HyDE: Generate hypothetical documents matching query → embed for retrieval
- Step-back: Abstract query to higher-level concepts → retrieve on both
- Query decomposition: Break complex queries into sub-queries → parallel retrieval
- Multi-query: Generate multiple perspectives → retrieve all → deduplicate

## Requirements

### Functional
- Query Analyzer: Detect type (factual/comparison/temporal), complexity, intent
- Query Rewriters: HyDE, Query2Doc, Step-back, Decomposition, Multi-query
- Context Enhancer: Query expansion, constraint extraction, context injection
- All rewriters use Ollama as default LLM provider

### Non-Functional
- Latency < 500ms for analysis, < 1s for rewriting
- Async operations throughout
- Configurable via YAML
- Fallback to original query on failure

## Architecture

### Module Structure
```
backend/modules/
├── query_analysis/
│   ├── __init__.py          # Registry
│   ├── base.py               # BaseQueryAnalyzer
│   ├── llm-analyzer.py       # LLM-based analysis
│   ├── fast-analyzer.py      # Lightweight classifiers
│   └── schemas.py            # Pydantic models
├── query_rewriting/
│   ├── __init__.py           # Registry + Factory
│   ├── base.py               # BaseQueryRewriter
│   ├── hyde-rewriter.py      # HyDE implementation
│   ├── query2doc-rewriter.py # Query2Doc
│   ├── stepback-rewriter.py  # Step-back prompting
│   ├── decomposition-rewriter.py
│   ├── multi-query-rewriter.py
│   └── schemas.py
└── context_enhancement/
    ├── __init__.py
    ├── query-expander.py     # Synonym/domain expansion
    ├── constraint-extractor.py
    └── context-injector.py
```

### Data Flow
```
User Query
    |
Query Analyzer → QueryMetadata (type, complexity, intent)
    |
Query Rewriter → RewriteResult (rewritten queries, strategy)
    |
Context Enhancer → EnhancedQuery (expanded, constraints, context)
    |
→ Retrieval Layer
```

## Related Code Files

### Files to Reference
- `backend/modules/query_controllers/base.py` - Integration point
- `backend/modules/model_gateway/model_gateway.py` - LLM access
- `backend/types.py` - Type patterns

### Files to Create
- `backend/modules/query_analysis/__init__.py`
- `backend/modules/query_analysis/base.py`
- `backend/modules/query_analysis/llm-analyzer.py`
- `backend/modules/query_analysis/fast-analyzer.py`
- `backend/modules/query_analysis/schemas.py`
- `backend/modules/query_rewriting/__init__.py`
- `backend/modules/query_rewriting/base.py`
- `backend/modules/query_rewriting/hyde-rewriter.py`
- `backend/modules/query_rewriting/query2doc-rewriter.py`
- `backend/modules/query_rewriting/stepback-rewriter.py`
- `backend/modules/query_rewriting/decomposition-rewriter.py`
- `backend/modules/query_rewriting/multi-query-rewriter.py`
- `backend/modules/query_rewriting/factory.py`
- `backend/modules/query_rewriting/schemas.py`
- `backend/modules/context_enhancement/__init__.py`
- `backend/modules/context_enhancement/query-expander.py`
- `backend/modules/context_enhancement/constraint-extractor.py`
- `backend/modules/context_enhancement/context-injector.py`
- `tests/modules/query_analysis/test_analyzers.py`
- `tests/modules/query_rewriting/test_rewriters.py`

## Implementation Steps

### Task 2.1: Query Analyzer Module (Days 1-3)

1. Create `backend/modules/query_analysis/schemas.py`:
```python
class QueryType(str, Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ANALYTICAL = "analytical"

class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MULTI_HOP = "multi_hop"
    COMPOSITIONAL = "compositional"

class QueryMetadata(ConfiguredBaseModel):
    query_type: QueryType
    complexity: QueryComplexity
    complexity_score: float  # 0.0 - 1.0
    intent: str  # retrieval-only, reasoning-required, verification-needed
    entities: List[str]
    temporal_constraints: Optional[Dict]
    spatial_constraints: Optional[Dict]
```

2. Create `base.py` with abstract interface:
```python
class BaseQueryAnalyzer(ABC):
    @abstractmethod
    async def analyze(self, query: str, context: Optional[Dict] = None) -> QueryMetadata:
        pass
```

3. Implement `llm-analyzer.py` using Ollama via model_gateway
4. Implement `fast-analyzer.py` using regex + simple heuristics
5. Write unit tests

### Task 2.2: Query Rewriting Module (Days 4-7)

1. Create `schemas.py`:
```python
class RewriteResult(ConfiguredBaseModel):
    original_query: str
    rewritten_queries: List[str]
    strategy: str
    metadata: Optional[Dict] = None
```

2. Create `base.py`:
```python
class BaseQueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str, context: Optional[Dict] = None) -> RewriteResult:
        pass
```

3. Implement rewriters:
   - `hyde-rewriter.py`: Generate hypothetical answer → use for retrieval
   - `query2doc-rewriter.py`: Generate pseudo-document from query
   - `stepback-rewriter.py`: Abstract to higher-level concepts
   - `decomposition-rewriter.py`: Break into sub-queries
   - `multi-query-rewriter.py`: Generate multiple perspectives

4. Create factory pattern in `factory.py`:
```python
class QueryRewriterFactory:
    _rewriters = {}

    @classmethod
    def register(cls, name: str, rewriter_cls: Type[BaseQueryRewriter]):
        cls._rewriters[name] = rewriter_cls

    @classmethod
    def create(cls, rewriter_type: str, config: Dict) -> BaseQueryRewriter:
        return cls._rewriters[rewriter_type](config)
```

### Task 2.3: Context Enhancement Module (Days 8-10)

1. Create `query-expander.py`:
   - Synonym expansion (WordNet or LLM)
   - Domain-specific term expansion (configurable dictionary)
   - Acronym expansion

2. Create `constraint-extractor.py`:
   - Extract time constraints (date parsing)
   - Extract geographic constraints
   - Domain-specific filters

3. Create `context-injector.py`:
   - User profile context
   - Session history context
   - Domain knowledge context

## Todo List

- [x] Create query_analysis module structure
- [x] Implement BaseQueryAnalyzer
- [x] Implement LLMAnalyzer with Ollama
- [x] Implement FastAnalyzer
- [x] Create query_rewriting module structure
- [x] Implement BaseQueryRewriter
- [x] Implement HyDERewriter
- [x] Implement Query2DocRewriter
- [x] Implement StepbackRewriter
- [x] Implement DecompositionRewriter
- [x] Implement MultiQueryRewriter
- [x] Create QueryRewriterFactory
- [x] Create context_enhancement module
- [x] Implement QueryExpander
- [x] Implement ConstraintExtractor
- [x] Implement ContextInjector
- [ ] Write unit tests for all modules
- [ ] Integration test with existing retrievers

## Success Criteria

- Query analyzer correctly classifies query types with >85% accuracy
- HyDE improves retrieval relevance on test set
- All rewriters work with Ollama local models
- Latency targets met
- Unit test coverage >80%

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM latency too high | Medium | High | Cache common patterns, use fast-analyzer fallback |
| Rewriting degrades quality | Medium | Medium | A/B test, fallback to original |
| Ollama model incompatibility | Low | High | Test with multiple models, document requirements |

## Security Considerations

- Sanitize user queries before LLM processing
- Rate limit query analysis to prevent abuse
- Log but don't expose internal rewriting logic

## Next Steps

After Phase 2:
- Integrate with Phase 3 (Advanced Retrieval) retrievers
- Feed into Phase 4 (Orchestration Engine) routing
- Use query metadata for adaptive pipeline selection
