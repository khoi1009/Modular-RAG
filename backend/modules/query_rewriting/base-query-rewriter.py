from abc import ABC, abstractmethod
from typing import Dict, Optional

from backend.modules.query_rewriting.schemas import RewriteResult


class BaseQueryRewriter(ABC):
    """Abstract base class for query rewriters"""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the query rewriter

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    async def rewrite(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> RewriteResult:
        """
        Rewrite a query using a specific strategy

        Args:
            query: The original user query
            context: Optional context (metadata, domain info, etc.)

        Returns:
            RewriteResult with rewritten queries and metadata
        """
        pass
