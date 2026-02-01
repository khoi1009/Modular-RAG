"""Metrics collection and aggregation module for RAG performance monitoring.

Provides lightweight metrics collection for counters, gauges, and histograms.
Supports time-series storage and aggregation for dashboard visualization.

Components:
    - MetricsCollector: Counter, gauge, histogram collection
    - MetricsStorage: Time-series data persistence
    - MetricsAggregator: Percentiles, averages, rates calculation

Example:
    from backend.modules.observability.metrics import get_metrics_collector

    metrics = get_metrics_collector()

    # Track query count
    metrics.increment("queries.total", tags={"collection": "docs"})

    # Record latency
    metrics.histogram("query.latency_ms", 45.2, tags={"stage": "retrieval"})

    # Set active connections
    metrics.gauge("connections.active", 12)

    # Get query rate (queries/sec)
    rate = metrics.get_rate("queries.total", window=60)
"""

from .metrics_aggregator import MetricsAggregator, get_aggregator
from .metrics_collector import MetricsCollector, get_metrics_collector
from .metrics_storage import MetricsStorage, get_metrics_storage

__all__ = [
    "MetricsCollector",
    "get_metrics_collector",
    "MetricsStorage",
    "get_metrics_storage",
    "MetricsAggregator",
    "get_aggregator",
]
