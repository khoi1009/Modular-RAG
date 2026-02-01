# Phase 2 Quick Reference Guide

## Fast Import Reference

### Query Analysis
```python
# Analyzers
from backend.modules.query_analysis import (
    BaseQueryAnalyzer,
    LLMBasedQueryAnalyzer,
    FastHeuristicQueryAnalyzer,
)

# Schemas
from backend.modules.query_analysis import (
    QueryType,           # FACTUAL, COMPARISON, TEMPORAL, SPATIAL, ANALYTICAL
    QueryComplexity,     # SIMPLE, MULTI_HOP, COMPOSITIONAL
    QueryMetadata,       # Complete analysis result
)
```

### Query Rewriting
```python
# Factory (recommended)
from backend.modules.query_rewriting import QueryRewriterFactory

# Individual rewriters
from backend.modules.query_rewriting import (
    HyDEHypotheticalDocumentRewriter,
    StepBackAbstractQueryRewriter,
    DecompositionSubqueryRewriter,
    MultiPerspectiveQueryRewriter,
    Query2DocPseudoDocumentRewriter,
)

# Schema
from backend.modules.query_rewriting import RewriteResult
```

### Context Enhancement
```python
from backend.modules.context_enhancement import (
    SynonymDomainQueryExpander,
    TemporalSpatialConstraintExtractor,
    SessionDomainContextInjector,
)
```

## Usage Patterns

### Pattern 1: Simple Analysis
```python
# Fast heuristic analyzer (no LLM, < 50ms)
analyzer = FastHeuristicQueryAnalyzer()
metadata = await analyzer.analyze("What is machine learning?")

print(metadata.query_type)      # QueryType.FACTUAL
print(metadata.complexity)      # QueryComplexity.SIMPLE
print(metadata.complexity_score) # 0.3
```

### Pattern 2: LLM-Based Analysis
```python
# LLM analyzer (Ollama, ~500ms)
config = {
    "model_name": "ollama/llama3",
    "timeout": 10
}
analyzer = LLMBasedQueryAnalyzer(config)
metadata = await analyzer.analyze("Compare React vs Vue in 2024")

print(metadata.query_type)       # QueryType.COMPARISON
print(metadata.entities)         # ["React", "Vue"]
print(metadata.temporal_constraints)  # {"years": [2024]}
```

### Pattern 3: HyDE Rewriting (Most Important)
```python
# Generate hypothetical document for retrieval
rewriter = QueryRewriterFactory.create("hyde", {
    "model_name": "ollama/llama3",
    "timeout": 15
})

result = await rewriter.rewrite("What causes global warming?")

print(result.strategy)           # "hyde"
print(result.original_query)     # "What causes global warming?"
print(result.rewritten_queries[0])
# "Global warming is caused by greenhouse gas emissions
#  from fossil fuels, deforestation, and industrial processes..."
```

### Pattern 4: Query Decomposition
```python
# Break complex query into sub-queries
rewriter = QueryRewriterFactory.create("decomposition", {
    "max_subqueries": 3
})

result = await rewriter.rewrite(
    "How do neural networks work and what are their applications in healthcare?"
)

print(len(result.rewritten_queries))  # 2-3 sub-queries
# ["How do neural networks work?",
#  "What are neural network applications in healthcare?"]
```

### Pattern 5: Constraint Extraction
```python
# Extract temporal/spatial constraints
extractor = TemporalSpatialConstraintExtractor()
constraints = await extractor.extract(
    "AI startups in San Francisco during 2023"
)

print(constraints.temporal)  # {"years": [2023]}
print(constraints.spatial)   # {"potential_locations": ["San Francisco"]}
print(constraints.filters)   # {"year": [2023], "location": ["San Francisco"]}

# Use filters in vector store query
retriever_config.filter = constraints.filters
```

### Pattern 6: Context Injection
```python
# Inject user/session context
injector = SessionDomainContextInjector()

enhanced = await injector.inject(
    query="Latest research papers",
    user_profile={"role": "researcher", "expertise": "ML"},
    session_history=["neural networks", "transformers"],
    domain="ai_ml"
)

print(enhanced.injected_context)
# "User role: researcher | Recent queries: neural networks; transformers"

# Use enhanced query for retrieval
final_query = injector.format_query_with_context(
    query, enhanced
)
```

### Pattern 7: Full Pipeline
```python
async def intelligent_query_processing(user_query: str):
    # Step 1: Analyze
    analyzer = FastHeuristicQueryAnalyzer()
    metadata = await analyzer.analyze(user_query)

    # Step 2: Choose rewriter based on complexity
    if metadata.complexity == QueryComplexity.COMPOSITIONAL:
        rewriter_type = "decomposition"
    elif metadata.query_type == QueryType.COMPARISON:
        rewriter_type = "stepback"
    else:
        rewriter_type = "hyde"

    # Step 3: Rewrite
    rewriter = QueryRewriterFactory.create(rewriter_type)
    rewrite_result = await rewriter.rewrite(user_query)

    # Step 4: Extract constraints
    extractor = TemporalSpatialConstraintExtractor()
    constraints = await extractor.extract(user_query)

    # Step 5: Return processed query
    return {
        "metadata": metadata,
        "queries": rewrite_result.rewritten_queries,
        "filters": constraints.filters,
        "strategy": rewrite_result.strategy
    }
```

## Configuration Recipes

### Production Config (Balanced)
```python
PRODUCTION_CONFIG = {
    "analyzer": {
        "model_name": "ollama/llama3",
        "timeout": 8,
        "model_parameters": {"temperature": 0.0}
    },
    "rewriter": {
        "model_name": "ollama/llama3",
        "timeout": 12,
        "model_parameters": {"temperature": 0.3}
    }
}
```

### Fast Config (Low Latency)
```python
FAST_CONFIG = {
    "use_fast_analyzer": True,  # Heuristic only
    "rewriter": {
        "timeout": 5,
        "model_parameters": {"max_tokens": 512}
    }
}
```

### High Quality Config (Accuracy First)
```python
HIGH_QUALITY_CONFIG = {
    "analyzer": {
        "model_name": "ollama/llama3.1",  # Better model
        "timeout": 15,
        "model_parameters": {"temperature": 0.0}
    },
    "rewriter": {
        "model_name": "ollama/llama3.1",
        "timeout": 20,
        "model_parameters": {
            "temperature": 0.3,
            "max_tokens": 2048
        }
    }
}
```

## Error Handling

### Timeouts
```python
# All components have timeout protection
try:
    result = await rewriter.rewrite(query)
except asyncio.TimeoutError:
    # Automatic fallback in implementation
    # Returns original query
    pass
```

### LLM Failures
```python
# Components automatically fallback on LLM errors
result = await analyzer.analyze(query)
# If LLM fails, returns safe default:
# QueryMetadata(
#     query_type=QueryType.FACTUAL,
#     complexity=QueryComplexity.SIMPLE,
#     complexity_score=0.3,
#     intent="retrieval-only",
#     entities=[]
# )
```

## Rewriter Strategy Selection Guide

| Query Type | Best Rewriter | Reason |
|------------|---------------|--------|
| Factual ("What is X?") | HyDE | Generate answer-like doc |
| Comparison | StepBack | Get broader context first |
| Complex/Multi-part | Decomposition | Break into pieces |
| Ambiguous | Multi-Query | Try multiple angles |
| Short/Vague | Query2Doc | Expand with context |

## Performance Tips

### 1. Use Fast Analyzer for Real-time
```python
# Sub-100ms analysis
analyzer = FastHeuristicQueryAnalyzer()
```

### 2. Cache Analysis Results
```python
# Pseudocode
cache_key = hash(query)
if cache_key in metadata_cache:
    return metadata_cache[cache_key]
metadata = await analyzer.analyze(query)
metadata_cache[cache_key] = metadata
```

### 3. Parallel Rewriting
```python
# Run multiple rewriters in parallel
results = await asyncio.gather(
    hyde_rewriter.rewrite(query),
    stepback_rewriter.rewrite(query),
    multi_query_rewriter.rewrite(query)
)
# Use best result or combine
```

### 4. Selective Enhancement
```python
# Only enhance complex queries
if metadata.complexity_score > 0.6:
    result = await rewriter.rewrite(query)
else:
    result = RewriteResult(
        original_query=query,
        rewritten_queries=[query],
        strategy="passthrough"
    )
```

## Common Patterns

### Pattern: Adaptive Rewriting
```python
def select_rewriter(metadata: QueryMetadata) -> str:
    """Choose rewriter based on query characteristics"""
    if metadata.complexity == QueryComplexity.COMPOSITIONAL:
        return "decomposition"
    elif metadata.query_type == QueryType.COMPARISON:
        return "stepback"
    elif metadata.complexity_score > 0.5:
        return "multi_query"
    else:
        return "hyde"

rewriter_type = select_rewriter(metadata)
rewriter = QueryRewriterFactory.create(rewriter_type, config)
```

### Pattern: Constraint-Aware Retrieval
```python
async def retrieve_with_constraints(query: str, vector_store):
    # Extract constraints
    extractor = TemporalSpatialConstraintExtractor()
    constraints = await extractor.extract(query)

    # Apply to retrieval
    retriever_config = RetrieverConfig(
        k=10,
        filter=constraints.filters
    )

    return await vector_store.retrieve(query, retriever_config)
```

### Pattern: Context-Aware Expansion
```python
async def expand_with_domain(query: str, domain: str):
    expander = SynonymDomainQueryExpander({
        "domain_dictionary": {
            "healthcare": {"AI": ["medical imaging", "diagnosis"]},
            "finance": {"AI": ["fraud detection", "trading"]}
        }
    })

    expanded = await expander.expand(query, {"domain": domain})
    return query + " " + " ".join(expanded.expanded_terms)
```

## Testing Examples

### Unit Test Structure
```python
import pytest
from backend.modules.query_analysis import FastHeuristicQueryAnalyzer

@pytest.mark.asyncio
async def test_factual_query_classification():
    analyzer = FastHeuristicQueryAnalyzer()
    result = await analyzer.analyze("What is Python?")

    assert result.query_type == QueryType.FACTUAL
    assert result.complexity == QueryComplexity.SIMPLE

@pytest.mark.asyncio
async def test_comparison_query_classification():
    analyzer = FastHeuristicQueryAnalyzer()
    result = await analyzer.analyze("Compare Python vs Java")

    assert result.query_type == QueryType.COMPARISON
    assert "Python" in result.entities
    assert "Java" in result.entities
```

## Troubleshooting

### Issue: Slow LLM responses
**Solution**: Reduce timeout, use fast analyzer, or cache results

### Issue: Poor rewrite quality
**Solution**: Adjust temperature, try different rewriter, or use better model

### Issue: Memory usage high
**Solution**: Limit max_tokens, batch process, or use stateless pattern

### Issue: Import errors
**Solution**: Check kebab-case imports match file names
```python
# Correct
from backend.modules.query_analysis.llm-based-query-analyzer import LLMBasedQueryAnalyzer

# Wrong (will fail)
from backend.modules.query_analysis.llm_based_query_analyzer import LLMBasedQueryAnalyzer
```

## Next Steps

1. **Integrate with retrieval** (Phase 3)
2. **Add to orchestration** (Phase 4)
3. **Write tests**
4. **Benchmark performance**
5. **Fine-tune prompts**
