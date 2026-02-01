# Phase 2: Query Intelligence Layer - Module Structure

## Overview

Implemented 3 major modules with 18 files totaling ~1,170 lines of production code.

## Module Tree

```
backend/modules/
├── query_analysis/                    # Query metadata extraction
│   ├── __init__.py                    # Exports: 6 symbols
│   ├── schemas.py                     # QueryType, QueryComplexity, QueryMetadata
│   ├── base-query-analyzer.py         # BaseQueryAnalyzer (ABC)
│   ├── llm-based-query-analyzer.py    # LLM-powered analysis via Ollama
│   └── fast-heuristic-query-analyzer.py  # Regex/pattern-based analysis
│
├── query_rewriting/                   # Query transformation strategies
│   ├── __init__.py                    # Exports: 8 symbols
│   ├── schemas.py                     # RewriteResult
│   ├── base-query-rewriter.py         # BaseQueryRewriter (ABC)
│   ├── hyde-hypothetical-document-rewriter.py      # Generate hypothetical docs
│   ├── stepback-abstract-query-rewriter.py         # Abstract to higher concepts
│   ├── decomposition-subquery-rewriter.py          # Break into sub-queries
│   ├── multi-perspective-query-rewriter.py         # Multiple perspectives
│   ├── query2doc-pseudo-document-rewriter.py       # Expand to pseudo-doc
│   └── query-rewriter-factory.py      # Factory with 5 registered rewriters
│
└── context_enhancement/               # Query enrichment
    ├── __init__.py                    # Exports: 6 symbols
    ├── synonym-domain-query-expander.py            # Synonym/domain expansion
    ├── temporal-spatial-constraint-extractor.py    # Time/location constraints
    └── session-domain-context-injector.py          # User/session context
```

## Component Breakdown

### 1. Query Analysis (5 files)

**Purpose**: Understand query characteristics before processing

| Component | Lines | Key Features |
|-----------|-------|--------------|
| Schemas | 28 | 3 enums (QueryType, QueryComplexity), 1 model |
| Base | 35 | Abstract interface, async analyze() |
| LLM Analyzer | 90 | Ollama integration, JSON parsing, fallback |
| Fast Analyzer | 140 | 15+ regex patterns, heuristic scoring |

**Capabilities**:
- Classify: factual, comparison, temporal, spatial, analytical
- Assess complexity: simple, multi-hop, compositional (0.0-1.0 score)
- Detect intent: retrieval-only, reasoning-required, verification-needed
- Extract: entities, temporal constraints, spatial constraints

### 2. Query Rewriting (9 files)

**Purpose**: Transform queries for better retrieval

| Component | Lines | Strategy | Use Case |
|-----------|-------|----------|----------|
| HyDE | 72 | Hypothetical docs | Semantic similarity |
| StepBack | 70 | Abstract concepts | Broader context |
| Decomposition | 110 | Sub-queries | Complex questions |
| Multi-Query | 110 | Perspectives | Coverage |
| Query2Doc | 65 | Pseudo-document | Term expansion |
| Factory | 48 | Registry pattern | Dynamic creation |

**All Rewriters Include**:
- Async LLM calls via model_gateway
- Timeout protection (10-15s)
- Fallback to original query on error
- Configurable model parameters
- JSON parsing with fallback

### 3. Context Enhancement (4 files)

**Purpose**: Enrich queries with additional context

| Component | Lines | Features |
|-----------|-------|----------|
| Query Expander | 145 | LLM/heuristic expansion, domain terms, acronyms |
| Constraint Extractor | 160 | Temporal (years, dates), spatial (locations), filters |
| Context Injector | 120 | User profile, session history, domain knowledge |

**Extraction Capabilities**:
- **Temporal**: Years (regex), dates (parsing), relative time (today, last year)
- **Spatial**: Location keywords, capitalized place names, prepositions
- **Context**: User role, preferences, expertise, recent queries

## Data Flow

```
User Query: "Compare AI frameworks used in 2023"
    |
    v
[Query Analyzer]
    ├─> Type: COMPARISON
    ├─> Complexity: MULTI_HOP (0.65)
    ├─> Intent: reasoning-required
    ├─> Entities: ["AI"]
    └─> Temporal: {years: [2023]}
    |
    v
[Query Rewriter - HyDE]
    └─> "In 2023, the leading AI frameworks include TensorFlow, PyTorch, and JAX.
         TensorFlow offers production-grade deployment, PyTorch excels in research..."
    |
    v
[Context Enhancer]
    ├─> Expanded: ["artificial intelligence", "machine learning", "deep learning"]
    ├─> Constraints: {year: [2023]}
    └─> Domain: "ai_ml"
    |
    v
→ To Retrieval Layer (Phase 3)
```

## Integration Points

### With Existing Cognita Components

1. **Model Gateway**:
   ```python
   from backend.modules.model_gateway.model_gateway import model_gateway
   llm = model_gateway.get_llm_from_model_config(model_config)
   ```

2. **Type System**:
   ```python
   from backend.types import ConfiguredBaseModel, ModelConfig
   class QueryMetadata(ConfiguredBaseModel): ...
   ```

3. **Async Patterns**:
   ```python
   async def analyze(self, query: str) -> QueryMetadata:
       async with async_timeout.timeout(self.timeout):
           # LLM call
   ```

### With Future Phases

- **Phase 3 (Advanced Retrieval)**: Rewritten queries feed retrievers
- **Phase 4 (Orchestration)**: Metadata drives routing logic
- **Phase 5 (Evaluation)**: Analysis helps generate test cases

## Configuration Examples

### Query Analyzer Config
```python
config = {
    "model_name": "ollama/llama3",
    "model_parameters": {"temperature": 0.0},
    "timeout": 10
}
analyzer = LLMBasedQueryAnalyzer(config)
```

### Query Rewriter Config
```python
config = {
    "model_name": "ollama/llama3",
    "model_parameters": {"temperature": 0.3},
    "timeout": 15,
    "max_subqueries": 4  # For decomposition
}
rewriter = QueryRewriterFactory.create("hyde", config)
```

### Context Enhancer Config
```python
config = {
    "use_llm": True,
    "model_name": "ollama/llama3",
    "timeout": 10,
    "domain_dictionary": {
        "healthcare": {
            "diagnosis": ["symptoms", "treatment", "prognosis"],
            "medication": ["dosage", "side effects", "interactions"]
        }
    }
}
expander = SynonymDomainQueryExpander(config)
```

## Performance Characteristics

### Latency Targets (from phase requirements)
- Query Analysis: < 500ms
- Query Rewriting: < 1s
- Context Enhancement: < 200ms

### Implementation Approach
- All operations async (non-blocking)
- Timeout protection on all LLM calls
- Fallback mechanisms prevent failures
- Stateless design enables caching

## Factory Pattern Usage

```python
# Available rewriters
QueryRewriterFactory.get_available_rewriters()
# Returns: ['hyde', 'query2doc', 'stepback', 'decomposition', 'multi_query']

# Create any rewriter dynamically
rewriter = QueryRewriterFactory.create('stepback', config)

# Register custom rewriter
class CustomRewriter(BaseQueryRewriter):
    async def rewrite(self, query, context):
        # Implementation
        pass

QueryRewriterFactory.register('custom', CustomRewriter)
```

## File Naming Convention

All files use kebab-case with descriptive names:
- `llm-based-query-analyzer.py` (not `llm_analyzer.py`)
- `hyde-hypothetical-document-rewriter.py` (not `hyde.py`)
- `temporal-spatial-constraint-extractor.py` (not `extractor.py`)

**Rationale**: Self-documenting for LLM tools (Glob, Grep, Search)

## Code Metrics

| Module | Files | Lines | Classes | Functions |
|--------|-------|-------|---------|-----------|
| query_analysis | 5 | ~300 | 4 | 15+ |
| query_rewriting | 9 | ~520 | 7 | 25+ |
| context_enhancement | 4 | ~350 | 6 | 20+ |
| **Total** | **18** | **~1,170** | **17** | **60+** |

## Status

- ✅ Implementation: Complete
- ✅ Module structure: Complete
- ✅ Integration points: Ready
- ⏳ Unit tests: Not started (separate phase)
- ⏳ Integration tests: Not started (separate phase)
- ⏳ Performance benchmarks: Not started
