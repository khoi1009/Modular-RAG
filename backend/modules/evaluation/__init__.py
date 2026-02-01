"""Evaluation framework for RAG pipeline quality assessment."""
from backend.modules.evaluation.datasets import (
    DatasetLoader,
    DatasetManager,
    EvaluationDataset,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
)
from backend.modules.evaluation.metrics import (
    GenerationMetrics,
    LLMJudge,
    RetrievalMetrics,
    SemanticSimilarity,
)
from backend.modules.evaluation.evaluator import (
    AutomatedEvaluator,
    RegressionResult,
    RegressionTester,
)
from backend.modules.evaluation.ab_testing import (
    Experiment,
    ExperimentAnalysis,
    ExperimentManager,
    StatisticalAnalyzer,
    Variant,
    VariantRouter,
)

__all__ = [
    # Datasets
    "EvaluationSample",
    "EvaluationDataset",
    "EvaluationResult",
    "EvaluationReport",
    "DatasetManager",
    "DatasetLoader",
    # Metrics
    "RetrievalMetrics",
    "GenerationMetrics",
    "LLMJudge",
    "SemanticSimilarity",
    # Evaluator
    "AutomatedEvaluator",
    "RegressionTester",
    "RegressionResult",
    # A/B Testing
    "Experiment",
    "Variant",
    "ExperimentManager",
    "VariantRouter",
    "StatisticalAnalyzer",
    "ExperimentAnalysis",
]
