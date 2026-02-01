"""Observability module for Cognita RAG framework.

Provides comprehensive monitoring capabilities:
- Distributed tracing with OpenTelemetry-compatible spans
- Structured JSON logging with trace correlation
- Metrics collection (counters, gauges, histograms)
- Real-time dashboard and alerting

Components:
    - tracing: Span context, decorators, and exporters
    - logging: Structured logger, query replay, error tracking
    - metrics: Collection, storage, aggregation
    - dashboard: API endpoints and alert management
"""

from .logging.structured_logger import StructuredLogger, get_logger
from .metrics.metrics_collector import MetricsCollector, get_metrics_collector
from .tracing.span_context import SpanContext
from .tracing.trace_decorators import trace_query, trace_component

__all__ = [
    "StructuredLogger",
    "get_logger",
    "MetricsCollector",
    "get_metrics_collector",
    "SpanContext",
    "trace_query",
    "trace_component",
]
