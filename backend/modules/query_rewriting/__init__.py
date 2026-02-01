"""
Query Rewriting Module

Provides query rewriters using various strategies:
- HyDE: Generate hypothetical documents for retrieval
- Step-back: Create abstract queries for broader context
- Decomposition: Break complex queries into sub-queries
- Multi-query: Generate multiple perspectives
- Query2Doc: Expand queries into pseudo-documents
"""

from backend.modules.query_rewriting.base-query-rewriter import BaseQueryRewriter
from backend.modules.query_rewriting.decomposition-subquery-rewriter import (
    DecompositionSubqueryRewriter,
)
from backend.modules.query_rewriting.hyde-hypothetical-document-rewriter import (
    HyDEHypotheticalDocumentRewriter,
)
from backend.modules.query_rewriting.multi-perspective-query-rewriter import (
    MultiPerspectiveQueryRewriter,
)
from backend.modules.query_rewriting.query2doc-pseudo-document-rewriter import (
    Query2DocPseudoDocumentRewriter,
)
from backend.modules.query_rewriting.query-rewriter-factory import QueryRewriterFactory
from backend.modules.query_rewriting.schemas import RewriteResult
from backend.modules.query_rewriting.stepback-abstract-query-rewriter import (
    StepBackAbstractQueryRewriter,
)

__all__ = [
    "BaseQueryRewriter",
    "HyDEHypotheticalDocumentRewriter",
    "StepBackAbstractQueryRewriter",
    "DecompositionSubqueryRewriter",
    "MultiPerspectiveQueryRewriter",
    "Query2DocPseudoDocumentRewriter",
    "QueryRewriterFactory",
    "RewriteResult",
]
