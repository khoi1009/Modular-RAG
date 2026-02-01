"""Spatial filtering for water infrastructure documents."""
from typing import Dict, List, Optional

from langchain.schema import Document


class SpatialFilter:
    """Filter documents by geographic location or zone."""

    def filter_by_location(
        self,
        documents: List[Document],
        location: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> List[Document]:
        """
        Filter documents based on spatial criteria.

        Args:
            documents: List of retrieved documents
            location: Location string to match
            zone: Zone/district identifier to match

        Returns:
            Filtered list of documents
        """
        if not location and not zone:
            return documents

        filtered_docs = []

        for doc in documents:
            if self._matches_spatial_criteria(doc, location, zone):
                filtered_docs.append(doc)

        return filtered_docs

    def _matches_spatial_criteria(
        self,
        doc: Document,
        location: Optional[str],
        zone: Optional[str],
    ) -> bool:
        """Check if document matches spatial criteria."""
        metadata = doc.metadata

        # Check location match
        if location:
            doc_location = metadata.get("location", "").lower()
            doc_address = metadata.get("address", "").lower()

            if location.lower() in doc_location or location.lower() in doc_address:
                return True

        # Check zone match
        if zone:
            doc_zone = metadata.get("zone", "").upper()
            doc_dma = metadata.get("dma", "").upper()
            doc_district = metadata.get("district", "").upper()

            zone_upper = zone.upper()
            if zone_upper in [doc_zone, doc_dma, doc_district]:
                return True

        # If we have criteria but no match, filter out
        if location or zone:
            return False

        return True

    def create_spatial_metadata_filter(
        self,
        location: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Create vector DB filter dict for spatial criteria.

        This can be passed to retriever config to filter at query time.

        Args:
            location: Location string
            zone: Zone identifier

        Returns:
            Filter dict for vector DB or None
        """
        if not location and not zone:
            return None

        filters = []

        if location:
            # Create OR filter for location/address fields
            filters.append({
                "$or": [
                    {"location": {"$contains": location}},
                    {"address": {"$contains": location}},
                ]
            })

        if zone:
            # Create OR filter for zone/dma/district fields
            filters.append({
                "$or": [
                    {"zone": zone.upper()},
                    {"dma": zone.upper()},
                    {"district": zone.upper()},
                ]
            })

        # Combine all filters with AND
        if len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}
