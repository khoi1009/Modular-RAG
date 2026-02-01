"""Schemas for query routing."""
from typing import Any, Dict, List, Optional

from backend.types import ConfiguredBaseModel


class RoutingDecision(ConfiguredBaseModel):
    """Decision output from query router"""
    controller_name: str
    retrieval_strategy: str  # vectorstore, hybrid, reflective
    preprocessing_steps: List[str] = []  # hyde, decomposition, etc.
    use_reranking: bool = True
    max_iterations: int = 1
    fallback_strategy: Optional[str] = None
    confidence: float
    reasoning: str


class RoutingRule(ConfiguredBaseModel):
    """Rule definition for rule-based routing"""
    name: str
    conditions: List[Dict[str, Any]]  # Query metadata conditions
    action: RoutingDecision
    priority: int = 0
