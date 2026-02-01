"""
Pydantic schemas for hybrid retriever configurations.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class BM25Config(BaseModel):
    """Configuration for BM25 retriever."""

    k1: float = Field(default=1.5, description="BM25 k1 parameter (term saturation)")
    b: float = Field(default=0.75, description="BM25 b parameter (length normalization)")
    epsilon: float = Field(default=0.25, description="IDF floor value")


class HybridRetrieverConfig(BaseModel):
    """Configuration for hybrid vector + BM25 retriever."""

    fusion_strategy: Literal["rrf", "weighted"] = Field(
        default="rrf", description="Fusion strategy to combine results"
    )
    rrf_k: int = Field(default=60, description="RRF k parameter")
    vector_weight: float = Field(
        default=0.5, description="Weight for vector search (0-1)"
    )
    bm25_weight: Optional[float] = Field(
        default=None, description="Weight for BM25 (auto-computed if None)"
    )
    vector_top_k: int = Field(default=20, description="Number of vector results")
    bm25_top_k: int = Field(default=20, description="Number of BM25 results")
    final_top_k: int = Field(default=10, description="Final number of results")
    bm25_config: BM25Config = Field(
        default_factory=BM25Config, description="BM25 parameters"
    )
