from typing import Dict, Type

from backend.logger import logger
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
from backend.modules.query_rewriting.stepback-abstract-query-rewriter import (
    StepBackAbstractQueryRewriter,
)


class QueryRewriterFactory:
    """Factory for creating query rewriter instances"""

    _rewriters: Dict[str, Type[BaseQueryRewriter]] = {}

    @classmethod
    def register(cls, name: str, rewriter_cls: Type[BaseQueryRewriter]):
        """Register a rewriter class"""
        cls._rewriters[name] = rewriter_cls
        logger.info(f"Registered query rewriter: {name}")

    @classmethod
    def create(cls, rewriter_type: str, config: Dict = None) -> BaseQueryRewriter:
        """Create a rewriter instance"""
        if rewriter_type not in cls._rewriters:
            raise ValueError(
                f"Unknown rewriter type: {rewriter_type}. "
                f"Available: {list(cls._rewriters.keys())}"
            )

        rewriter_cls = cls._rewriters[rewriter_type]
        return rewriter_cls(config or {})

    @classmethod
    def get_available_rewriters(cls) -> list:
        """Get list of available rewriter types"""
        return list(cls._rewriters.keys())


# Register built-in rewriters
QueryRewriterFactory.register("hyde", HyDEHypotheticalDocumentRewriter)
QueryRewriterFactory.register("query2doc", Query2DocPseudoDocumentRewriter)
QueryRewriterFactory.register("stepback", StepBackAbstractQueryRewriter)
QueryRewriterFactory.register("decomposition", DecompositionSubqueryRewriter)
QueryRewriterFactory.register("multi_query", MultiPerspectiveQueryRewriter)
