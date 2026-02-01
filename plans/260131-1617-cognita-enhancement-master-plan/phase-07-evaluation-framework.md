# Phase 7: Evaluation Framework

**Duration:** Week 10 | **Priority:** P2 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [Phase 6: Observability & Monitoring](phase-06-observability-monitoring.md)
- [Modular RAG Patterns Report](research/researcher-modular-rag-patterns-report.md)

## Overview

Systematic evaluation and improvement through ground-truth datasets, automated metrics computation, LLM-as-Judge, A/B testing, and regression testing. Enables data-driven optimization.

## Key Insights

From research:
- LLM-as-Judge: Evaluate without ground truth
- Standard metrics: F1, BLEU, ROUGE, BERTScore
- Semantic similarity: Embedding-based comparison
- Regression testing: Catch quality degradation early

Current state:
- No evaluation datasets
- No automated evaluation pipeline
- No A/B testing framework
- No regression testing

## Requirements

### Functional
- Dataset Manager: Store/version ground-truth Q&A pairs
- Automated Evaluator: Run queries, compare against ground truth
- Metrics Engine: Compute retrieval and generation metrics
- A/B Testing: Route traffic to variants, track metrics
- Regression Testing: CI/CD integration for quality gates

### Non-Functional
- Evaluation run < 10 min for 1000 samples
- A/B tests with statistical significance
- Metrics stored with versioning
- Integration with CI/CD pipelines

## Architecture

### Module Structure
```
backend/modules/
└── evaluation/
    ├── __init__.py
    ├── datasets/
    │   ├── __init__.py
    │   ├── dataset-manager.py
    │   ├── dataset-loader.py
    │   └── schemas.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── retrieval-metrics.py
    │   ├── generation-metrics.py
    │   ├── llm-judge.py
    │   └── semantic-similarity.py
    ├── evaluator/
    │   ├── __init__.py
    │   ├── automated-evaluator.py
    │   └── regression-tester.py
    └── ab_testing/
        ├── __init__.py
        ├── experiment-manager.py
        ├── variant-router.py
        └── statistical-analysis.py

data/
└── evaluation/
    ├── general-qa/
    │   └── v1.json
    └── water-infrastructure/
        └── v1.json
```

### Evaluation Flow
```
Dataset (Q&A pairs)
    ↓
Evaluator.run(dataset, pipeline_config)
    ├── For each sample:
    │   ├── Query pipeline
    │   ├── Compare to ground truth
    │   └── Compute metrics
    ↓
Metrics Report
    ├── Retrieval: Precision@K, Recall@K, MRR, NDCG
    ├── Generation: F1, BLEU, ROUGE, BERTScore
    └── Quality: LLM-Judge scores
```

## Related Code Files

### Files to Reference
- `backend/modules/observability/metrics/collectors.py` - Metrics patterns
- `backend/modules/orchestration/pipeline/pipeline-executor.py` - Execution

### Files to Create
- `backend/modules/evaluation/__init__.py`
- `backend/modules/evaluation/datasets/__init__.py`
- `backend/modules/evaluation/datasets/dataset-manager.py`
- `backend/modules/evaluation/datasets/dataset-loader.py`
- `backend/modules/evaluation/datasets/schemas.py`
- `backend/modules/evaluation/metrics/__init__.py`
- `backend/modules/evaluation/metrics/retrieval-metrics.py`
- `backend/modules/evaluation/metrics/generation-metrics.py`
- `backend/modules/evaluation/metrics/llm-judge.py`
- `backend/modules/evaluation/metrics/semantic-similarity.py`
- `backend/modules/evaluation/evaluator/__init__.py`
- `backend/modules/evaluation/evaluator/automated-evaluator.py`
- `backend/modules/evaluation/evaluator/regression-tester.py`
- `backend/modules/evaluation/ab_testing/__init__.py`
- `backend/modules/evaluation/ab_testing/experiment-manager.py`
- `backend/modules/evaluation/ab_testing/variant-router.py`
- `backend/modules/evaluation/ab_testing/statistical-analysis.py`
- `data/evaluation/general-qa/v1.json`
- `tests/modules/evaluation/test_evaluator.py`
- `tests/modules/evaluation/test_ab_testing.py`

## Implementation Steps

### Task 7.1: Dataset Manager (Days 1-2)

1. Create `datasets/schemas.py`:
```python
class EvaluationSample(ConfiguredBaseModel):
    id: str
    query: str
    ground_truth_answer: str
    ground_truth_sources: List[str] = []
    difficulty: str = "medium"  # easy, medium, hard
    requires_retrieval: bool = True
    domain: Optional[str] = None
    metadata: Dict[str, Any] = {}

class EvaluationDataset(ConfiguredBaseModel):
    name: str
    version: str
    domain: Optional[str] = None
    description: Optional[str] = None
    samples: List[EvaluationSample]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EvaluationResult(ConfiguredBaseModel):
    sample_id: str
    query: str
    predicted_answer: str
    ground_truth_answer: str
    retrieved_sources: List[str]
    metrics: Dict[str, float]
    latency_ms: int
```

2. Create `datasets/dataset-manager.py`:
```python
class DatasetManager:
    def __init__(self, storage_path: str = "data/evaluation"):
        self.storage_path = Path(storage_path)

    async def create_dataset(self, dataset: EvaluationDataset) -> str:
        path = self.storage_path / dataset.domain / f"{dataset.name}_v{dataset.version}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataset.model_dump(), f, indent=2, default=str)
        return str(path)

    async def load_dataset(self, name: str, version: str, domain: Optional[str] = None) -> EvaluationDataset:
        ...

    async def list_datasets(self) -> List[Dict[str, str]]:
        ...

    async def import_from_csv(self, csv_path: str, dataset_name: str) -> EvaluationDataset:
        ...
```

### Task 7.2: Metrics Engine (Days 3-4)

1. Create `metrics/retrieval-metrics.py`:
```python
class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        return len([r for r in retrieved_k if r in relevant_set]) / k

    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        return len([r for r in retrieved_k if r in relevant_set]) / len(relevant_set) if relevant_set else 0

    @staticmethod
    def mrr(retrieved: List[str], relevant: List[str]) -> float:
        """Mean Reciprocal Rank"""
        relevant_set = set(relevant)
        for i, doc in enumerate(retrieved):
            if doc in relevant_set:
                return 1 / (i + 1)
        return 0

    @staticmethod
    def ndcg(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain"""
        ...
```

2. Create `metrics/generation-metrics.py`:
```python
class GenerationMetrics:
    @staticmethod
    def exact_match(predicted: str, ground_truth: str) -> float:
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def f1_score(predicted: str, ground_truth: str) -> float:
        pred_tokens = set(predicted.lower().split())
        gt_tokens = set(ground_truth.lower().split())
        if not pred_tokens or not gt_tokens:
            return 0.0
        precision = len(pred_tokens & gt_tokens) / len(pred_tokens)
        recall = len(pred_tokens & gt_tokens) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    @staticmethod
    async def bert_score(predicted: str, ground_truth: str) -> float:
        """Use sentence-transformers for semantic similarity"""
        ...

    @staticmethod
    def bleu_score(predicted: str, ground_truth: str) -> float:
        """BLEU score for text generation"""
        ...

    @staticmethod
    def rouge_score(predicted: str, ground_truth: str) -> Dict[str, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L"""
        ...
```

3. Create `metrics/llm-judge.py`:
```python
class LLMJudge:
    def __init__(self, llm_config: ModelConfig):
        self.llm = model_gateway.get_llm_from_model_config(llm_config)

    async def evaluate(
        self,
        query: str,
        predicted: str,
        ground_truth: Optional[str] = None,
        sources: Optional[List[str]] = None
    ) -> Dict[str, float]:
        prompt = """Evaluate the following answer on a scale of 1-5 for each criterion:
        - Relevance: How well does the answer address the question?
        - Accuracy: How factually correct is the answer?
        - Completeness: How thoroughly does the answer cover the topic?
        - Coherence: How well-structured and clear is the answer?

        Question: {query}
        Answer: {predicted}
        {ground_truth_section}

        Scores (JSON format):"""

        ground_truth_section = f"Ground Truth: {ground_truth}" if ground_truth else ""
        response = await self.llm.ainvoke(prompt.format(
            query=query, predicted=predicted, ground_truth_section=ground_truth_section
        ))
        return self._parse_scores(response.content)
```

### Task 7.3: Automated Evaluator (Day 5)

1. Create `evaluator/automated-evaluator.py`:
```python
class AutomatedEvaluator:
    def __init__(
        self,
        pipeline_executor: PipelineExecutor,
        metrics_engine: MetricsEngine,
        llm_judge: Optional[LLMJudge] = None
    ):
        self.executor = pipeline_executor
        self.metrics = metrics_engine
        self.judge = llm_judge

    async def evaluate_dataset(
        self,
        dataset: EvaluationDataset,
        pipeline_config: str,
        collection_name: str
    ) -> EvaluationReport:
        results = []

        for sample in dataset.samples:
            start = time.time()

            # Execute pipeline
            pipeline_result = await self.executor.execute(
                self._load_pipeline(pipeline_config),
                {"query": sample.query, "collection_name": collection_name}
            )

            # Compute metrics
            metrics = {
                "exact_match": GenerationMetrics.exact_match(
                    pipeline_result.answer, sample.ground_truth_answer
                ),
                "f1": GenerationMetrics.f1_score(
                    pipeline_result.answer, sample.ground_truth_answer
                ),
            }

            if sample.ground_truth_sources:
                retrieved_ids = [s.metadata["_id"] for s in pipeline_result.sources]
                metrics.update({
                    "precision@5": RetrievalMetrics.precision_at_k(
                        retrieved_ids, sample.ground_truth_sources, 5
                    ),
                    "recall@5": RetrievalMetrics.recall_at_k(
                        retrieved_ids, sample.ground_truth_sources, 5
                    ),
                    "mrr": RetrievalMetrics.mrr(retrieved_ids, sample.ground_truth_sources),
                })

            if self.judge:
                judge_scores = await self.judge.evaluate(
                    sample.query, pipeline_result.answer, sample.ground_truth_answer
                )
                metrics.update({f"judge_{k}": v for k, v in judge_scores.items()})

            results.append(EvaluationResult(
                sample_id=sample.id,
                query=sample.query,
                predicted_answer=pipeline_result.answer,
                ground_truth_answer=sample.ground_truth_answer,
                retrieved_sources=[s.metadata["_id"] for s in pipeline_result.sources],
                metrics=metrics,
                latency_ms=int((time.time() - start) * 1000)
            ))

        return EvaluationReport(
            dataset_name=dataset.name,
            pipeline_config=pipeline_config,
            results=results,
            aggregate_metrics=self._aggregate_metrics(results)
        )
```

2. Create `evaluator/regression-tester.py`:
```python
class RegressionTester:
    def __init__(self, evaluator: AutomatedEvaluator, baseline_path: str):
        self.evaluator = evaluator
        self.baseline_path = baseline_path

    async def run_regression_test(
        self, dataset: EvaluationDataset, pipeline_config: str
    ) -> RegressionResult:
        # Run current evaluation
        current = await self.evaluator.evaluate_dataset(dataset, pipeline_config)

        # Load baseline
        baseline = self._load_baseline(dataset.name, pipeline_config)

        # Compare
        regressions = []
        for metric, threshold in self.thresholds.items():
            current_val = current.aggregate_metrics.get(metric, 0)
            baseline_val = baseline.aggregate_metrics.get(metric, 0)
            if current_val < baseline_val - threshold:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_val,
                    "current": current_val,
                    "delta": current_val - baseline_val
                })

        return RegressionResult(
            passed=len(regressions) == 0,
            regressions=regressions,
            current_report=current,
            baseline_report=baseline
        )
```

### Task 7.4: A/B Testing Framework (Days 6-7)

1. Create `ab_testing/experiment-manager.py`:
```python
class Experiment(ConfiguredBaseModel):
    name: str
    variants: List[Variant]
    metrics: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # running, paused, completed

class Variant(ConfiguredBaseModel):
    name: str
    pipeline_config: str
    traffic_percentage: float

class ExperimentManager:
    def __init__(self, storage: ExperimentStorage):
        self.storage = storage
        self.active_experiments: Dict[str, Experiment] = {}

    async def create_experiment(self, experiment: Experiment) -> str:
        await self.storage.save(experiment)
        self.active_experiments[experiment.name] = experiment
        return experiment.name

    async def get_variant_for_user(self, experiment_name: str, user_id: str) -> Variant:
        experiment = self.active_experiments.get(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")

        # Deterministic assignment based on user_id
        hash_val = int(hashlib.md5(f"{experiment_name}:{user_id}".encode()).hexdigest(), 16)
        bucket = (hash_val % 100) / 100

        cumulative = 0
        for variant in experiment.variants:
            cumulative += variant.traffic_percentage / 100
            if bucket < cumulative:
                return variant

        return experiment.variants[-1]  # Fallback
```

2. Create `ab_testing/statistical-analysis.py`:
```python
class StatisticalAnalyzer:
    @staticmethod
    def compute_significance(
        control_values: List[float],
        treatment_values: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        from scipy import stats

        t_stat, p_value = stats.ttest_ind(control_values, treatment_values)
        significant = p_value < (1 - confidence_level)

        return {
            "control_mean": np.mean(control_values),
            "treatment_mean": np.mean(treatment_values),
            "lift": (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values),
            "p_value": p_value,
            "significant": significant,
            "confidence_level": confidence_level
        }

    async def analyze_experiment(self, experiment: Experiment) -> ExperimentAnalysis:
        # Load metrics for each variant
        results = {}
        for variant in experiment.variants:
            variant_metrics = await self._load_variant_metrics(experiment.name, variant.name)
            results[variant.name] = variant_metrics

        # Compute significance for each metric
        analysis = {}
        control = experiment.variants[0]
        for treatment in experiment.variants[1:]:
            for metric in experiment.metrics:
                analysis[f"{treatment.name}_{metric}"] = self.compute_significance(
                    results[control.name][metric],
                    results[treatment.name][metric]
                )

        return ExperimentAnalysis(experiment=experiment, results=analysis)
```

## Todo List

- [x] Create evaluation module structure
- [x] Implement EvaluationSample schema
- [x] Implement EvaluationDataset schema
- [x] Implement DatasetManager
- [x] Implement RetrievalMetrics
- [x] Implement GenerationMetrics
- [x] Implement LLMJudge
- [x] Implement AutomatedEvaluator
- [x] Implement RegressionTester
- [x] Implement ExperimentManager
- [x] Implement VariantRouter
- [x] Implement StatisticalAnalyzer
- [ ] Create sample evaluation datasets (deferred - no sample data as per requirements)
- [ ] Add CI/CD integration for regression tests (deferred - requires CI pipeline setup)
- [ ] Write comprehensive tests (deferred - test files not to be created as per requirements)

## Success Criteria

- Evaluation pipeline runs 1000 samples in < 10 min
- A/B tests achieve statistical significance detection
- Regression tests catch > 95% of quality degradations
- Metrics correlate with user satisfaction
- All evaluations reproducible

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Ground truth quality low | Medium | High | Human review, multiple annotators |
| LLM-Judge inconsistent | Medium | Medium | Calibration, ensemble judging |
| A/B test sample size insufficient | Medium | Medium | Power analysis before experiment |

## Security Considerations

- Secure evaluation dataset storage
- Rate limit evaluation API
- Audit log for experiment modifications

## Next Steps

After Phase 7:
- Phase 8 (Domain) creates domain-specific datasets
- Phase 9 (Performance) uses evaluation for optimization
- Continuous evaluation in production
