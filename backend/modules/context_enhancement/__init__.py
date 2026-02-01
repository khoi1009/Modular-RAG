"""
Context Enhancement Module

Provides tools to enhance queries with additional context:
- Query expansion with synonyms and domain terms
- Temporal and spatial constraint extraction
- User profile and session context injection
"""

from backend.modules.context_enhancement.session-domain-context-injector import (
    EnhancedContext,
    SessionDomainContextInjector,
)
from backend.modules.context_enhancement.synonym-domain-query-expander import (
    ExpandedQuery,
    SynonymDomainQueryExpander,
)
from backend.modules.context_enhancement.temporal-spatial-constraint-extractor import (
    ExtractedConstraints,
    TemporalSpatialConstraintExtractor,
)

__all__ = [
    "SynonymDomainQueryExpander",
    "ExpandedQuery",
    "TemporalSpatialConstraintExtractor",
    "ExtractedConstraints",
    "SessionDomainContextInjector",
    "EnhancedContext",
]
