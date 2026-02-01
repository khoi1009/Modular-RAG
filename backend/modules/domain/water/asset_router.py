"""Query routing for water infrastructure domain."""
from enum import Enum
from typing import Optional

from backend.modules.domain.water.schemas import WaterEntities


class QueryIntent(str, Enum):
    """Water infrastructure query intent types."""

    BI_ANALYTICS = "bi_analytics"  # SQL queries on daily_analysis table
    FAILURE_ANALYSIS = "failure"  # Asset failure patterns
    MAINTENANCE = "maintenance"  # Maintenance history/scheduling
    COMPLIANCE = "compliance"  # Regulatory compliance checks
    GENERAL = "general"  # General RAG queries


class WaterAssetRouter:
    """Route water infrastructure queries to appropriate handlers."""

    def __init__(self):
        # Keywords for intent classification
        self.intent_keywords = {
            QueryIntent.BI_ANALYTICS: [
                "average", "total", "sum", "count", "statistics", "trend",
                "compare", "comparison", "analysis", "report", "chart",
                "graph", "dashboard", "metrics", "kpi", "performance",
            ],
            QueryIntent.FAILURE_ANALYSIS: [
                "failure", "failed", "break", "broken", "leak", "burst",
                "malfunction", "outage", "incident", "emergency", "down",
                "not working", "stopped",
            ],
            QueryIntent.MAINTENANCE: [
                "maintenance", "service", "inspection", "repair", "replace",
                "preventive", "scheduled", "pm", "cm", "work order",
                "history", "last serviced", "when was",
            ],
            QueryIntent.COMPLIANCE: [
                "compliance", "regulation", "standard", "code", "permit",
                "violation", "inspection report", "audit", "regulatory",
                "requirement", "awwa", "epa",
            ],
        }

    def classify_intent(
        self,
        query: str,
        entities: Optional[WaterEntities] = None
    ) -> QueryIntent:
        """
        Classify query intent based on keywords and entities.

        Args:
            query: Natural language query
            entities: Extracted entities (optional)

        Returns:
            QueryIntent classification
        """
        query_lower = query.lower()

        # Check for BI analytics patterns
        if self._matches_keywords(query_lower, QueryIntent.BI_ANALYTICS):
            return QueryIntent.BI_ANALYTICS

        # Check for failure analysis patterns
        if self._matches_keywords(query_lower, QueryIntent.FAILURE_ANALYSIS):
            return QueryIntent.FAILURE_ANALYSIS

        # Check for maintenance patterns
        if self._matches_keywords(query_lower, QueryIntent.MAINTENANCE):
            return QueryIntent.MAINTENANCE

        # Check for compliance patterns
        if self._matches_keywords(query_lower, QueryIntent.COMPLIANCE):
            return QueryIntent.COMPLIANCE

        # Default to general RAG
        return QueryIntent.GENERAL

    def _matches_keywords(self, query: str, intent: QueryIntent) -> bool:
        """Check if query matches keywords for given intent."""
        keywords = self.intent_keywords.get(intent, [])
        return any(keyword in query for keyword in keywords)

    def should_use_sql_agent(self, intent: QueryIntent) -> bool:
        """Determine if SQL agent should be used for this intent."""
        return intent == QueryIntent.BI_ANALYTICS

    def should_use_temporal_filter(self, entities: WaterEntities) -> bool:
        """Determine if temporal filtering should be applied."""
        return entities.date_range is not None

    def should_use_spatial_filter(self, entities: WaterEntities) -> bool:
        """Determine if spatial filtering should be applied."""
        return entities.location is not None or entities.zone is not None
