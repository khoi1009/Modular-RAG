"""Attribute claims to source documents with citation support."""
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document

from backend.logger import logger
from backend.modules.verification.schemas import Claim


class SourceAttributor:
    """Map claims to supporting source documents and generate citations"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.similarity_threshold = config.get("similarity_threshold", 0.3)

    async def attribute(
        self, claims: List[Claim], sources: List[Document]
    ) -> Dict[str, List[str]]:
        """
        Map each claim to supporting source document IDs.
        Returns: Dict mapping claim text to list of source IDs
        """
        attributions = {}

        for claim in claims:
            supporting = await self._find_supporting_sources(claim, sources)
            attributions[claim.text] = supporting

        return attributions

    async def _find_supporting_sources(
        self, claim: Claim, sources: List[Document]
    ) -> List[str]:
        """Find source documents that support a claim"""
        supporting = []
        claim_lower = claim.text.lower()
        claim_words = set(
            word for word in claim_lower.split() if len(word) > 3
        )  # Filter short words

        if not claim_words:
            return []

        for doc in sources:
            content_lower = doc.page_content.lower()

            # Calculate word overlap score
            content_words = set(content_lower.split())
            overlap = len(claim_words.intersection(content_words))
            similarity = overlap / len(claim_words) if claim_words else 0

            if similarity > self.similarity_threshold:
                doc_id = doc.metadata.get("_id", doc.metadata.get("source", "unknown"))
                supporting.append((doc_id, similarity))

        # Sort by similarity and return top 3
        supporting.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in supporting[:3]]

    def generate_citations(
        self, answer: str, attributions: Dict[str, List[str]]
    ) -> str:
        """
        Insert citation markers into answer text.
        Example: "Paris is the capital [1][2]."
        """
        try:
            # Create source to citation number mapping
            all_sources = []
            for sources in attributions.values():
                all_sources.extend(sources)
            unique_sources = list(dict.fromkeys(all_sources))  # Preserve order

            if not unique_sources:
                return answer

            # Map source IDs to citation numbers
            source_to_num = {src: i + 1 for i, src in enumerate(unique_sources)}

            # Insert citations after claims
            cited_answer = answer
            for claim_text, sources in attributions.items():
                if not sources:
                    continue

                # Create citation string [1][2]
                citations = "".join(f"[{source_to_num[src]}]" for src in sources)

                # Find claim in answer and add citation
                # Use word boundaries to avoid partial matches
                pattern = re.escape(claim_text)
                cited_answer = re.sub(
                    f"({pattern})", f"\\1{citations}", cited_answer, count=1
                )

            return cited_answer

        except Exception as e:
            logger.error(f"Citation generation error: {e}")
            return answer

    def calculate_coverage(self, attributions: Dict[str, List[str]]) -> float:
        """Calculate percentage of claims with at least one source"""
        if not attributions:
            return 0.0

        claims_with_sources = sum(1 for sources in attributions.values() if sources)
        return claims_with_sources / len(attributions)
