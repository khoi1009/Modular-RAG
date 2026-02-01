"""
Pydantic schemas for reflective retriever configurations.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RelevanceEvaluatorConfig(BaseModel):
    """Configuration for relevance evaluators."""

    evaluator_type: Literal["llm", "embedding"] = Field(
        default="embedding", description="Type of relevance evaluator"
    )
    model_name: Optional[str] = Field(
        default=None, description="Model to use for evaluation"
    )
    threshold: float = Field(
        default=0.7, description="Minimum relevance score threshold"
    )
    batch_size: int = Field(default=10, description="Batch size for evaluation")


class FeedbackRetrieverConfig(BaseModel):
    """Configuration for feedback-based retriever."""

    max_iterations: int = Field(
        default=3, description="Maximum refinement iterations"
    )
    quality_threshold: float = Field(
        default=0.7, description="Minimum average quality to accept results"
    )
    min_quality_threshold: float = Field(
        default=0.3, description="Floor quality threshold to continue iterating"
    )
    evaluator_config: RelevanceEvaluatorConfig = Field(
        default_factory=RelevanceEvaluatorConfig,
        description="Relevance evaluator configuration",
    )


class CRAGRetrieverConfig(BaseModel):
    """Configuration for Corrective RAG retriever."""

    relevance_threshold: float = Field(
        default=0.6, description="Relevance threshold for document grading"
    )
    min_relevant_docs: int = Field(
        default=3, description="Minimum relevant documents required"
    )
    enable_web_search: bool = Field(
        default=False, description="Enable web search fallback"
    )
    enable_query_rewrite: bool = Field(
        default=True, description="Enable query rewriting on low relevance"
    )
    max_rewrites: int = Field(
        default=2, description="Maximum query rewrite attempts"
    )
    evaluator_config: RelevanceEvaluatorConfig = Field(
        default_factory=RelevanceEvaluatorConfig,
        description="Relevance evaluator configuration",
    )
