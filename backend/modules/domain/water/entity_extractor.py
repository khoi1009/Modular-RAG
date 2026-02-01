"""Entity extraction for water infrastructure queries."""
import re
from datetime import datetime
from typing import Optional

from backend.modules.domain.water.knowledge_base.acronyms import MAINTENANCE_CODE_PREFIXES
from backend.modules.domain.water.knowledge_base.asset_taxonomy import (
    ASSET_TAXONOMY,
    PRIORITY_LEVELS,
    normalize_asset_type,
)
from backend.modules.domain.water.schemas import WaterEntities
from backend.modules.domain.water.temporal_parser import TemporalParser


class WaterEntityExtractor:
    """Extract water infrastructure entities from natural language queries."""

    def __init__(self):
        self.temporal_parser = TemporalParser()

    def extract(self, query: str) -> WaterEntities:
        """
        Extract entities from query text.

        Performance target: < 100ms
        """
        entities = WaterEntities()

        # Extract asset ID (patterns like P-1234, PUMP-001, etc.)
        entities.asset_id = self._extract_asset_id(query)

        # Extract asset type
        entities.asset_type = normalize_asset_type(query)

        # Extract location
        entities.location = self._extract_location(query)

        # Extract date range
        entities.date_range = self.temporal_parser.parse(query)

        # Extract maintenance codes
        entities.maintenance_codes = self._extract_maintenance_codes(query)

        # Extract zone
        entities.zone = self._extract_zone(query)

        # Extract priority
        entities.priority = self._extract_priority(query)

        return entities

    def _extract_asset_id(self, query: str) -> Optional[str]:
        """Extract asset ID from query using regex patterns."""
        # Common patterns: P-1234, PUMP-001, V-456, METER_789
        patterns = [
            r'\b([A-Z]{1,4}[-_]\d{2,6})\b',  # P-1234, PUMP-001
            r'\b(ID[-:]?\d{3,8})\b',  # ID-12345
            r'\b([A-Z]{2,10}\d{2,6})\b',  # PUMP123
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location mentions from query."""
        # Look for common location patterns
        location_patterns = [
            r'\bat\s+([A-Z][a-zA-Z\s]{2,30}(?:Street|St|Ave|Avenue|Road|Rd|Blvd|Boulevard|Station|Plant))\b',
            r'\bin\s+([A-Z][a-zA-Z\s]{2,30}(?:Zone|Area|District|Sector))\b',
            r'\bnear\s+([A-Z][a-zA-Z\s]{2,30})\b',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()

        return None

    def _extract_maintenance_codes(self, query: str) -> list:
        """Extract maintenance codes from query."""
        codes = []

        # Look for maintenance code patterns
        for prefix, description in MAINTENANCE_CODE_PREFIXES.items():
            # Match codes like PM-001, CM-2023-045
            pattern = rf'\b({prefix}[-_]?\d{{2,6}})\b'
            matches = re.findall(pattern, query, re.IGNORECASE)
            codes.extend([m.upper() for m in matches])

            # Also match standalone prefixes if followed by descriptive text
            if re.search(rf'\b{prefix}\b', query, re.IGNORECASE):
                codes.append(prefix)

        return list(set(codes))  # Remove duplicates

    def _extract_zone(self, query: str) -> Optional[str]:
        """Extract zone/district information."""
        zone_patterns = [
            r'(?:zone|dma|district|sector)\s*[:#]?\s*([A-Z0-9-]{1,10})\b',
            r'\b(Zone\s+[A-Z0-9]{1,5})\b',
            r'\b(DMA[-\s]?\d{1,4})\b',
        ]

        for pattern in zone_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        return None

    def _extract_priority(self, query: str) -> Optional[str]:
        """Extract priority level from query."""
        query_lower = query.lower()

        for priority, keywords in PRIORITY_LEVELS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return priority

        return None
