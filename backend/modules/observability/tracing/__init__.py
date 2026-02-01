"""Distributed tracing module for RAG query operations.

Provides lightweight (<10ms overhead) tracing with OpenTelemetry-compatible format.
Tracks query flow through retrieval, reranking, and generation stages.

Components:
    - SpanContext: Trace/span data structure with parent-child relationships
    - QueryTracer: Async tracer for query operations
    - TraceDecorators: Function decorators for automatic instrumentation
    - Exporters: Console, Jaeger, HTTP backends

Example:
    from backend.modules.observability.tracing import trace_query, QueryTracer

    @trace_query("embedding_generation")
    async def embed_query(text: str) -> List[float]:
        # Automatically traced with span context
        return await model.embed(text)
"""

from .exporters import (
    ConsoleExporter,
    HTTPExporter,
    JaegerExporter,
    SpanExporter,
)
from .query_tracer import QueryTracer
from .span_context import SpanContext
from .trace_decorators import trace_component, trace_query

__all__ = [
    "SpanContext",
    "QueryTracer",
    "trace_query",
    "trace_component",
    "SpanExporter",
    "ConsoleExporter",
    "JaegerExporter",
    "HTTPExporter",
]
