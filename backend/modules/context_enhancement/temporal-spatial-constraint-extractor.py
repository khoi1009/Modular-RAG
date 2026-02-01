import re
from datetime import datetime
from typing import Dict, List, Optional

from backend.logger import logger
from backend.types import ConfiguredBaseModel


class ExtractedConstraints(ConfiguredBaseModel):
    """Extracted temporal and spatial constraints"""
    temporal: Optional[Dict] = None
    spatial: Optional[Dict] = None
    filters: Dict = {}


class TemporalSpatialConstraintExtractor:
    """
    Extracts temporal and spatial constraints from queries
    to filter and rank retrieval results
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize regex patterns for constraint extraction"""
        # Temporal patterns
        self.year_pattern = r"\b(19|20)\d{2}\b"
        self.month_year_pattern = r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20)\d{2}\b"
        self.date_pattern = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        self.relative_time_pattern = r"\b(today|yesterday|tomorrow|last week|last month|last year|this week|this month|this year)\b"
        self.temporal_keywords = r"\b(before|after|during|since|until|recent|latest|current|historical|past|future)\b"

        # Spatial patterns
        self.country_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # Capitalized words
        self.location_keywords = r"\b(in|at|near|from|around|located in|based in)\b"
        self.spatial_keywords = r"\b(where|location|place|city|country|region|area|address)\b"

    async def extract(
        self, query: str, context: Optional[Dict] = None
    ) -> ExtractedConstraints:
        """Extract temporal and spatial constraints from query"""
        temporal = self._extract_temporal(query)
        spatial = self._extract_spatial(query)
        filters = self._build_filters(temporal, spatial)

        return ExtractedConstraints(
            temporal=temporal,
            spatial=spatial,
            filters=filters,
        )

    def _extract_temporal(self, query: str) -> Optional[Dict]:
        """Extract temporal constraints"""
        temporal = {}

        # Extract years
        years = re.findall(self.year_pattern, query)
        if years:
            temporal["years"] = [int(y) for y in years]

        # Extract dates
        dates = re.findall(self.date_pattern, query)
        if dates:
            temporal["dates"] = dates

        # Extract month-year combinations
        month_years = re.findall(self.month_year_pattern, query, re.IGNORECASE)
        if month_years:
            temporal["month_years"] = [f"{m} {y}" for m, y in month_years]

        # Extract relative time
        relative = re.findall(self.relative_time_pattern, query, re.IGNORECASE)
        if relative:
            temporal["relative_time"] = relative
            temporal["resolved_dates"] = self._resolve_relative_time(relative)

        # Detect temporal keywords
        keywords = re.findall(self.temporal_keywords, query, re.IGNORECASE)
        if keywords:
            temporal["keywords"] = list(set(keywords))

        return temporal if temporal else None

    def _extract_spatial(self, query: str) -> Optional[Dict]:
        """Extract spatial constraints"""
        spatial = {}

        # Detect spatial keywords
        keywords = re.findall(self.spatial_keywords, query, re.IGNORECASE)
        if keywords:
            spatial["keywords"] = list(set(keywords))

        # Detect location prepositions
        location_prepositions = re.findall(self.location_keywords, query, re.IGNORECASE)
        if location_prepositions:
            spatial["prepositions"] = list(set(location_prepositions))
            # Try to extract location names after prepositions
            spatial["potential_locations"] = self._extract_locations(query)

        return spatial if spatial else None

    def _extract_locations(self, query: str) -> List[str]:
        """Extract potential location names"""
        locations = []

        # Look for capitalized words that might be locations
        words = query.split()
        for i, word in enumerate(words):
            # Check if preceded by location keyword
            if i > 0 and words[i - 1].lower() in ["in", "at", "near", "from"]:
                # Get next 1-3 capitalized words
                location = []
                for j in range(i, min(i + 3, len(words))):
                    if words[j][0].isupper():
                        location.append(words[j].strip(".,?!"))
                    else:
                        break
                if location:
                    locations.append(" ".join(location))

        return locations[:3]  # Limit to 3 locations

    def _resolve_relative_time(self, relative_times: List[str]) -> Dict:
        """Resolve relative time expressions to actual dates"""
        now = datetime.now()
        resolved = {}

        for rel_time in relative_times:
            rel_time_lower = rel_time.lower()

            if rel_time_lower == "today":
                resolved["date"] = now.strftime("%Y-%m-%d")
            elif rel_time_lower == "yesterday":
                # Simple approximation
                resolved["date_approx"] = "yesterday"
            elif "last year" in rel_time_lower:
                resolved["year"] = now.year - 1
            elif "this year" in rel_time_lower:
                resolved["year"] = now.year

        return resolved

    def _build_filters(
        self, temporal: Optional[Dict], spatial: Optional[Dict]
    ) -> Dict:
        """Build metadata filters for retrieval"""
        filters = {}

        if temporal:
            # Add year filters
            if "years" in temporal:
                filters["year"] = temporal["years"]
            elif "resolved_dates" in temporal and "year" in temporal["resolved_dates"]:
                filters["year"] = [temporal["resolved_dates"]["year"]]

        if spatial:
            # Add location filters
            if "potential_locations" in spatial:
                filters["location"] = spatial["potential_locations"]

        return filters
