"""Request/response schemas for water infrastructure controller."""
from typing import Optional

from backend.modules.query_controllers.types import ModelConfig, RetrieverConfig
from backend.types import ConfiguredBaseModel


class WaterInfrastructureQueryInput(ConfiguredBaseModel):
    """Input schema for water infrastructure queries."""

    query: str
    collection_name: str
    model_configuration: ModelConfig
    retriever_config: RetrieverConfig
    prompt_template: Optional[str] = None
    stream: bool = True
    enable_bi_analytics: bool = True
    enable_spatial_filter: bool = True
    internet_search_enabled: bool = False
