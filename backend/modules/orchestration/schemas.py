"""Core schemas for orchestration module."""
from typing import Any, Dict, List, Optional

from backend.types import ConfiguredBaseModel


class OrchestrationConfig(ConfiguredBaseModel):
    """Configuration for orchestration engine"""
    routing_rules_path: str = "config/routing-rules.yaml"
    pipelines_dir: str = "config/pipelines"
    default_pipeline: str = "simple-retrieval"
    enable_llm_routing: bool = False
    enable_ml_routing: bool = False
    max_routing_time_ms: int = 100
    enable_cost_tracking: bool = True


class PipelineResult(ConfiguredBaseModel):
    """Result from pipeline execution"""
    success: bool
    answer: Optional[str] = None
    sources: List = []
    context: Dict[str, Any] = {}
    execution_time_ms: int
    steps_executed: List[str]
    errors: List[str] = []
    cost_usd: Optional[float] = None
