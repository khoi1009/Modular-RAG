"""Schemas for evaluation datasets and results."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import Field

from backend.types import ConfiguredBaseModel


class EvaluationSample(ConfiguredBaseModel):
    """Single evaluation sample with query and ground truth."""
    id: str
    query: str
    ground_truth_answer: str
    ground_truth_sources: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    requires_retrieval: bool = True
    domain: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(ConfiguredBaseModel):
    """Collection of evaluation samples for testing RAG pipelines."""
    name: str
    version: str
    domain: Optional[str] = None
    description: Optional[str] = None
    samples: List[EvaluationSample]
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(ConfiguredBaseModel):
    """Result of evaluating a single sample against pipeline."""
    sample_id: str
    query: str
    predicted_answer: str
    ground_truth_answer: str
    retrieved_sources: List[str] = Field(default_factory=list)
    metrics: Dict[str, float]
    latency_ms: int


class EvaluationReport(ConfiguredBaseModel):
    """Aggregated evaluation report for entire dataset."""
    dataset_name: str
    dataset_version: str
    pipeline_config: str
    collection_name: str
    results: List[EvaluationResult]
    aggregate_metrics: Dict[str, float]
    total_samples: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
