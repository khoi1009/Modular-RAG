"""Base abstract router for query routing."""
from abc import ABC, abstractmethod
from typing import Dict, Optional

from backend.modules.orchestration.routing.schemas import RoutingDecision
from backend.modules.query_analysis.schemas import QueryMetadata


class BaseQueryRouter(ABC):
    """Abstract base class for query routers"""

    @abstractmethod
    async def route(
        self,
        query: str,
        query_metadata: QueryMetadata,
        context: Optional[Dict] = None,
    ) -> RoutingDecision:
        """
        Route a query to appropriate pipeline based on analysis.

        Args:
            query: Raw query string
            query_metadata: Analyzed query metadata
            context: Optional additional context

        Returns:
            RoutingDecision with pipeline and strategy info
        """
        pass
