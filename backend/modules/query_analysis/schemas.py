from enum import Enum
from typing import Dict, List, Optional

from backend.types import ConfiguredBaseModel


class QueryType(str, Enum):
    """Types of queries based on user intent and information need"""
    FACTUAL = "factual"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    ANALYTICAL = "analytical"


class QueryComplexity(str, Enum):
    """Complexity levels for query processing"""
    SIMPLE = "simple"  # Single-hop, straightforward lookup
    MULTI_HOP = "multi_hop"  # Requires multiple reasoning steps
    COMPOSITIONAL = "compositional"  # Multiple sub-questions combined


class QueryMetadata(ConfiguredBaseModel):
    """Metadata extracted from query analysis"""
    query_type: QueryType
    complexity: QueryComplexity
    complexity_score: float  # 0.0 - 1.0
    intent: str  # retrieval-only, reasoning-required, verification-needed
    entities: List[str] = []
    temporal_constraints: Optional[Dict] = None
    spatial_constraints: Optional[Dict] = None
