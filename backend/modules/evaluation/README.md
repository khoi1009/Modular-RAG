# Evaluation Framework

Comprehensive evaluation framework for assessing RAG pipeline quality through automated testing, metrics computation, and A/B experimentation.

## Overview

The evaluation module provides:

- **Dataset Management**: Store and version ground-truth Q&A pairs
- **Automated Evaluation**: Run queries through pipelines and compute metrics
- **Metrics Engine**: Retrieval and generation quality metrics
- **LLM-as-Judge**: Evaluate answers without ground truth
- **Regression Testing**: Catch quality degradations in CI/CD
- **A/B Testing**: Compare pipeline variants with statistical rigor

## Module Structure

```
evaluation/
├── datasets/          # Dataset management
│   ├── schemas.py            # Data models
│   ├── dataset-manager.py    # Storage and retrieval
│   └── dataset-loader.py     # Import utilities
├── metrics/           # Quality metrics
│   ├── retrieval-metrics.py  # Precision@K, Recall@K, MRR, NDCG
│   ├── generation-metrics.py # F1, BLEU, ROUGE, exact match
│   ├── llm-judge.py          # LLM-based evaluation
│   └── semantic-similarity.py # Embedding-based similarity
├── evaluator/         # Automated evaluation
│   ├── automated-evaluator.py # Run datasets through pipelines
│   └── regression-tester.py   # Compare against baselines
└── ab_testing/        # A/B experimentation
    ├── experiment-manager.py  # Create and manage experiments
    ├── variant-router.py      # Route traffic to variants
    └── statistical-analysis.py # Significance testing
```

## Quick Start

### 1. Create Evaluation Dataset

```python
from backend.modules.evaluation import DatasetManager, EvaluationSample, EvaluationDataset

# Create samples
samples = [
    EvaluationSample(
        id="sample_001",
        query="What are the benefits of RAG?",
        ground_truth_answer="RAG combines retrieval and generation...",
        ground_truth_sources=["doc_123", "doc_456"],
        difficulty="easy"
    ),
    # Add more samples...
]

# Create dataset
dataset = EvaluationDataset(
    name="rag-basics",
    version="1",
    domain="general",
    samples=samples
)

# Save dataset
manager = DatasetManager()
await manager.create_dataset(dataset)
```

### 2. Run Automated Evaluation

```python
from backend.modules.evaluation import AutomatedEvaluator

evaluator = AutomatedEvaluator(
    pipeline_executor=my_pipeline_executor,
    llm_judge_config=my_llm_config,
    enable_semantic_similarity=True
)

report = await evaluator.evaluate_dataset(
    dataset=dataset,
    pipeline_config_path="config/pipelines/my-pipeline.yaml",
    collection_name="my-collection"
)

print(f"Average F1: {report.aggregate_metrics['f1_mean']:.3f}")
print(f"Average Precision@5: {report.aggregate_metrics['precision@5_mean']:.3f}")
```

### 3. Run Regression Test

```python
from backend.modules.evaluation import RegressionTester

tester = RegressionTester(
    evaluator=evaluator,
    baseline_dir="data/evaluation/baselines"
)

result = await tester.run_regression_test(
    dataset=dataset,
    pipeline_config_path="config/pipelines/my-pipeline.yaml",
    collection_name="my-collection",
    save_as_baseline=True
)

if not result.passed:
    print(f"REGRESSION DETECTED: {len(result.regressions)} metrics degraded")
    for reg in result.regressions:
        print(f"  - {reg['metric']}: {reg['delta']:.4f} [{reg['severity']}]")
```

### 4. Create A/B Experiment

```python
from backend.modules.evaluation import ExperimentManager, Variant

manager = ExperimentManager()

experiment = await manager.create_experiment(
    name="retrieval-optimization",
    variants=[
        Variant(
            name="control",
            pipeline_config="config/pipelines/baseline.yaml",
            traffic_percentage=50.0
        ),
        Variant(
            name="reranked",
            pipeline_config="config/pipelines/with-reranking.yaml",
            traffic_percentage=50.0
        )
    ],
    metrics=["f1", "precision@5", "latency_p95_ms"]
)

# Get variant for user
variant = await manager.get_variant_for_user("retrieval-optimization", "user_123")
print(f"User assigned to: {variant.name}")
```

### 5. Analyze Experiment Results

```python
from backend.modules.evaluation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(confidence_level=0.95)

# Prepare variant data
variant_data = {
    "control": {
        "f1": [0.75, 0.78, 0.76, ...],
        "precision@5": [0.82, 0.85, 0.80, ...]
    },
    "reranked": {
        "f1": [0.80, 0.82, 0.81, ...],
        "precision@5": [0.88, 0.90, 0.87, ...]
    }
}

analysis = await analyzer.analyze_experiment(experiment, variant_data)

for rec in analysis.recommendations:
    print(rec)
```

## Metrics Reference

### Retrieval Metrics

- **Precision@K**: Proportion of top-K retrieved docs that are relevant
- **Recall@K**: Proportion of all relevant docs retrieved in top-K
- **MRR**: Mean Reciprocal Rank of first relevant document
- **NDCG@K**: Normalized Discounted Cumulative Gain (rank-aware)

### Generation Metrics

- **Exact Match**: Binary match with ground truth
- **F1 Score**: Token-level overlap with ground truth
- **BLEU**: N-gram overlap with brevity penalty
- **ROUGE**: Recall-oriented n-gram overlap (ROUGE-1, ROUGE-2, ROUGE-L)

### LLM Judge Metrics

- **Relevance**: How well answer addresses question (1-5)
- **Accuracy**: Factual correctness (1-5)
- **Completeness**: Thoroughness of coverage (1-5)
- **Coherence**: Structure and clarity (1-5)

### Semantic Similarity

- **Cosine Similarity**: Embedding-based similarity (-1 to 1)
- **BERTScore**: Sentence-level semantic similarity (0 to 1)

## Dataset Import Formats

### CSV Format

```csv
id,query,ground_truth_answer,ground_truth_sources,difficulty
sample_001,"What is RAG?","RAG combines retrieval...","doc_1|doc_2",easy
sample_002,"How does chunking work?","Chunking splits...","doc_3|doc_4",medium
```

Import:
```python
dataset = await manager.import_from_csv(
    csv_path="datasets/qa_pairs.csv",
    dataset_name="my-dataset",
    version="1"
)
```

### JSON Format

```json
{
  "name": "my-dataset",
  "version": "1",
  "samples": [
    {
      "id": "sample_001",
      "query": "What is RAG?",
      "ground_truth_answer": "RAG combines retrieval...",
      "ground_truth_sources": ["doc_1", "doc_2"],
      "difficulty": "easy"
    }
  ]
}
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run Regression Tests
  run: |
    python -m pytest tests/evaluation/test_regression.py
```

### Test Script

```python
# tests/evaluation/test_regression.py
import asyncio
from backend.modules.evaluation import RegressionTester, AutomatedEvaluator

async def test_regression():
    evaluator = AutomatedEvaluator()
    tester = RegressionTester(evaluator)

    dataset = await load_test_dataset()
    result = await tester.run_ci_test(
        dataset=dataset,
        pipeline_config_path="config/pipelines/production.yaml",
        collection_name="test-collection",
        fail_on_regression=True
    )

    assert result.passed, "Regression test failed"
```

## Performance Considerations

- Evaluation of 1000 samples takes < 10 minutes
- Use `max_concurrent` parameter to control parallelism
- Enable/disable expensive metrics (LLM judge, semantic similarity) as needed
- Cache embeddings for semantic similarity to reduce API calls

## Best Practices

1. **Version Datasets**: Always version evaluation datasets for reproducibility
2. **Baseline First**: Run and save baseline before making changes
3. **Statistical Rigor**: Use adequate sample sizes for A/B tests
4. **Multiple Metrics**: Don't rely on single metric - use comprehensive evaluation
5. **Domain-Specific**: Create domain-specific datasets for better coverage
6. **Regular Testing**: Integrate regression tests in CI/CD pipeline
7. **Monitor Latency**: Track latency alongside quality metrics

## Dependencies

Optional dependencies for full functionality:

```bash
pip install scipy  # For statistical analysis
pip install numpy  # For numerical computations
```

These are optional - the module provides fallback implementations if not available.
