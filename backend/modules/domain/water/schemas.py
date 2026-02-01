"""Pydantic schemas for water infrastructure domain."""
from datetime import datetime
from typing import List, Optional, Tuple

from backend.types import ConfiguredBaseModel


class WaterEntities(ConfiguredBaseModel):
    """Extracted entities from water infrastructure queries."""

    asset_id: Optional[str] = None
    asset_type: Optional[str] = None  # pipe, pump, valve, meter, sensor
    location: Optional[str] = None
    date_range: Optional[Tuple[datetime, datetime]] = None
    maintenance_codes: List[str] = []
    zone: Optional[str] = None
    priority: Optional[str] = None  # critical, high, medium, low


class WaterQueryInput(ConfiguredBaseModel):
    """Input schema for water infrastructure queries."""

    query: str
    collection_name: str
    stream: bool = True
    model_configuration: Optional[dict] = None
    retriever_config: Optional[dict] = None
    enable_bi_analytics: bool = True
    enable_spatial_filter: bool = True
