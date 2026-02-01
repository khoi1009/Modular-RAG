import re
from typing import Dict, List, Optional

from backend.modules.query_analysis.base-query-analyzer import BaseQueryAnalyzer
from backend.modules.query_analysis.schemas import (
    QueryComplexity,
    QueryMetadata,
    QueryType,
)


class FastHeuristicQueryAnalyzer(BaseQueryAnalyzer):
    """Fast query analyzer using regex and heuristics"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for query classification"""
        self.comparison_patterns = [
            r"\b(compare|versus|vs|difference between|better than)\b",
            r"\b(which|what).*\b(better|best|worst|more|less)\b",
        ]
        self.temporal_patterns = [
            r"\b(when|before|after|during|since|until)\b",
            r"\b(\d{4}|\d{1,2}/\d{1,2}|yesterday|today|tomorrow)\b",
            r"\b(recent|latest|current|historical|past|future)\b",
        ]
        self.spatial_patterns = [
            r"\b(where|location|place|near|in|at)\b",
            r"\b(city|country|region|area|address)\b",
        ]
        self.analytical_patterns = [
            r"\b(why|how|explain|analyze|reason|cause)\b",
            r"\b(impact|effect|consequence|result)\b",
        ]

    async def analyze(
        self, query: str, context: Optional[Dict] = None
    ) -> QueryMetadata:
        """Analyze query using fast heuristics"""
        query_lower = query.lower()

        # Determine query type
        query_type = self._classify_query_type(query_lower)

        # Determine complexity
        complexity, score = self._assess_complexity(query)

        # Determine intent
        intent = self._determine_intent(query_lower)

        # Extract entities (simple capitalized words)
        entities = self._extract_entities(query)

        return QueryMetadata(
            query_type=query_type,
            complexity=complexity,
            complexity_score=score,
            intent=intent,
            entities=entities,
            temporal_constraints=self._extract_temporal(query_lower),
            spatial_constraints=self._extract_spatial(query_lower),
        )

    def _classify_query_type(self, query: str) -> QueryType:
        """Classify query type using pattern matching"""
        if any(re.search(p, query) for p in self.comparison_patterns):
            return QueryType.COMPARISON
        if any(re.search(p, query) for p in self.temporal_patterns):
            return QueryType.TEMPORAL
        if any(re.search(p, query) for p in self.spatial_patterns):
            return QueryType.SPATIAL
        if any(re.search(p, query) for p in self.analytical_patterns):
            return QueryType.ANALYTICAL
        return QueryType.FACTUAL

    def _assess_complexity(self, query: str):
        """Assess query complexity based on structure"""
        # Count indicators of complexity
        score = 0.0

        # Multiple questions or conjunctions
        conjunctions = len(re.findall(r"\b(and|or|but|however)\b", query.lower()))
        score += conjunctions * 0.15

        # Question words indicate multi-hop reasoning
        question_words = len(re.findall(r"\b(who|what|where|when|why|how)\b", query.lower()))
        if question_words > 1:
            score += 0.2

        # Length-based complexity
        words = query.split()
        if len(words) > 20:
            score += 0.3
        elif len(words) > 10:
            score += 0.15

        # Nested clauses
        if query.count(",") > 2 or query.count("(") > 0:
            score += 0.2

        # Normalize score
        score = min(score, 1.0)

        if score < 0.3:
            return QueryComplexity.SIMPLE, score
        elif score < 0.6:
            return QueryComplexity.MULTI_HOP, score
        else:
            return QueryComplexity.COMPOSITIONAL, score

    def _determine_intent(self, query: str) -> str:
        """Determine query intent"""
        if any(word in query for word in ["verify", "check", "confirm", "validate"]):
            return "verification-needed"
        if any(word in query for word in ["why", "how", "explain", "analyze"]):
            return "reasoning-required"
        return "retrieval-only"

    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities (simple capitalized words)"""
        words = query.split()
        # Find capitalized words that aren't at sentence start
        entities = [
            word.strip(".,?!\"'")
            for i, word in enumerate(words)
            if word[0].isupper() and (i > 0 or not query[0].isupper())
        ]
        return list(set(entities))[:5]  # Limit to 5 entities

    def _extract_temporal(self, query: str) -> Optional[Dict]:
        """Extract temporal constraints"""
        matches = re.findall(r"\b(\d{4})\b", query)
        if matches:
            return {"years": matches}
        return None

    def _extract_spatial(self, query: str) -> Optional[Dict]:
        """Extract spatial constraints"""
        if any(re.search(p, query) for p in self.spatial_patterns):
            return {"has_spatial": True}
        return None
