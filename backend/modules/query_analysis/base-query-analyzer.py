from abc import ABC, abstractmethod
from typing import Dict, Optional

from backend.modules.query_analysis.schemas import QueryMetadata


class BaseQueryAnalyzer(ABC):
    """Abstract base class for query analyzers"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the query analyzer

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    async def analyze(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> QueryMetadata:
        """
        Analyze a query and extract metadata

        Args:
            query: The user's query string
            context: Optional context (user profile, session history, etc.)

        Returns:
            QueryMetadata with type, complexity, intent, entities
        """
        pass
