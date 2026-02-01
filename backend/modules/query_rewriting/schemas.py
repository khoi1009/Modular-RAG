from typing import Dict, List, Optional

from backend.types import ConfiguredBaseModel


class RewriteResult(ConfiguredBaseModel):
    """Result of query rewriting operation"""
    original_query: str
    rewritten_queries: List[str]
    strategy: str  # hyde, stepback, decomposition, multi-query
    metadata: Optional[Dict] = None
