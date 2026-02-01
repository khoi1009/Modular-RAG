"""
Pydantic schemas for advanced reranker configurations.
"""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class MultiStageRerankerConfig(BaseModel):
    """Configuration for multi-stage cascaded reranking."""

    stage_configs: List[Tuple[str, int]] = Field(
        default=[("fast", 20), ("powerful", 5)],
        description="List of (stage_name, top_k) tuples",
    )
    enable_score_fusion: bool = Field(
        default=False, description="Fuse scores across stages"
    )


class LLMRerankerConfig(BaseModel):
    """Configuration for LLM-based reranking."""

    model_name: str = Field(description="LLM model to use for reranking")
    top_k: int = Field(default=5, description="Number of top documents to return")
    batch_size: int = Field(default=10, description="Batch size for parallel scoring")
    temperature: float = Field(default=0.0, description="LLM temperature")
    score_type: Literal["binary", "scaled"] = Field(
        default="scaled", description="Type of relevance scoring"
    )


class DiversityRerankerConfig(BaseModel):
    """Configuration for diversity-aware reranking."""

    diversity_weight: float = Field(
        default=0.5, description="Weight for diversity vs relevance (0-1)"
    )
    top_k: int = Field(default=10, description="Number of documents to return")
    similarity_threshold: float = Field(
        default=0.8, description="Similarity threshold for diversity filtering"
    )
    use_mmr: bool = Field(
        default=True, description="Use Maximal Marginal Relevance algorithm"
    )
