"""Schemas for pipeline execution."""
from typing import Any, Dict, List, Optional

from backend.types import ConfiguredBaseModel


class PipelineStep(ConfiguredBaseModel):
    """Definition of a single pipeline step"""
    name: str
    module: str  # e.g., "query_rewriting.hyde"
    input: Optional[str] = None  # Context key for input
    output: str  # Context key for output
    condition: Optional[str] = None  # Expression to evaluate
    parallel: bool = False
    timeout_sec: int = 30
    retry_count: int = 0


class PipelineDefinition(ConfiguredBaseModel):
    """Definition of a complete pipeline"""
    name: str
    description: Optional[str] = None
    steps: List[PipelineStep]
    default_config: Dict[str, Any] = {}
