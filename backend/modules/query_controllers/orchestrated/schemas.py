"""Schemas for orchestrated query controller."""
from typing import Optional

from pydantic import Field

from backend.modules.query_controllers.types import BaseQueryInput


class OrchestratedQueryInput(BaseQueryInput):
    """
    Input for orchestrated query controller.
    Extends base with orchestration-specific options.
    """

    enable_query_analysis: bool = Field(
        default=True,
        title="Enable query analysis for routing decisions",
    )

    force_pipeline: Optional[str] = Field(
        default=None,
        title="Force specific pipeline (bypass routing)",
    )

    enable_verification: bool = Field(
        default=False,
        title="Enable answer verification step",
    )

    max_iterations: Optional[int] = Field(
        default=None,
        title="Override max iterations for multi-hop reasoning",
    )
