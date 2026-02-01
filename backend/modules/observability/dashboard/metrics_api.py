"""FastAPI endpoints for observability dashboard metrics and traces.

Provides REST API for querying real-time metrics, trace data, and error statistics.
Integrates with MetricsCollector, QueryTracer, and ErrorTracker.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from ..logging.error_tracker import get_error_tracker
from ..metrics.metrics_aggregator import AggregatedStats, get_aggregator
from ..metrics.metrics_collector import get_metrics_collector
from ..metrics.metrics_storage import get_metrics_storage

router = APIRouter(prefix="/observability", tags=["observability"])


class MetricsSnapshot(BaseModel):
    """Current metrics snapshot response."""

    model_config = ConfigDict(use_enum_values=True)

    timestamp: str
    counters: Dict[str, float]
    gauges: Dict[str, float]
    histogram_counts: Dict[str, int]


class LatencyStats(BaseModel):
    """Latency statistics response."""

    model_config = ConfigDict(use_enum_values=True)

    metric_name: str
    stats: AggregatedStats
    sample_size: int


class ErrorSummary(BaseModel):
    """Error statistics summary."""

    model_config = ConfigDict(use_enum_values=True)

    total_unique_errors: int
    total_occurrences: int
    by_exception_type: Dict[str, int]
    by_component: Dict[str, int]


class TraceInfo(BaseModel):
    """Trace information response."""

    model_config = ConfigDict(use_enum_values=True)

    trace_id: str
    spans: List[Dict[str, Any]]
    total_duration_ms: int
    status: str


@router.get("/metrics/realtime", response_model=MetricsSnapshot)
async def get_realtime_metrics() -> Dict[str, Any]:
    """Get current snapshot of all metrics.

    Returns real-time counters, gauges, and histogram counts.

    Returns:
        MetricsSnapshot with current metrics
    """
    from datetime import datetime, timezone

    collector = get_metrics_collector()
    metrics = collector.get_all_metrics()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counters": metrics["counters"],
        "gauges": metrics["gauges"],
        "histogram_counts": metrics["histogram_counts"],
    }


@router.get("/metrics/latency", response_model=LatencyStats)
async def get_latency_stats(
    metric_name: str = Query(..., description="Metric name (e.g., query.latency_ms)"),
    tags: Optional[str] = Query(None, description="Tags as comma-separated key=value pairs"),
) -> Dict[str, Any]:
    """Get latency statistics (p50, p95, p99, mean, etc.) for a metric.

    Args:
        metric_name: Name of latency metric
        tags: Optional tags filter (e.g., "collection=docs,stage=retrieval")

    Returns:
        LatencyStats with aggregated statistics
    """
    collector = get_metrics_collector()
    aggregator = get_aggregator()

    # Parse tags
    tag_dict = None
    if tags:
        tag_dict = {}
        for pair in tags.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                tag_dict[key.strip()] = value.strip()

    # Get histogram values
    values = collector.get_histogram_values(metric_name, tags=tag_dict)

    if not values:
        raise HTTPException(status_code=404, detail=f"No data found for metric: {metric_name}")

    # Calculate statistics
    stats = aggregator.aggregate(values)

    return {
        "metric_name": metric_name,
        "stats": stats,
        "sample_size": len(values),
    }


@router.get("/metrics/rate")
async def get_metric_rate(
    metric_name: str = Query(..., description="Metric name"),
    window: int = Query(60, description="Time window in seconds"),
    tags: Optional[str] = Query(None, description="Tags filter"),
) -> Dict[str, Any]:
    """Get metric rate (value/sec) over time window.

    Args:
        metric_name: Name of counter metric
        window: Time window in seconds (default: 60)
        tags: Optional tags filter

    Returns:
        Rate as value per second
    """
    collector = get_metrics_collector()

    # Parse tags
    tag_dict = None
    if tags:
        tag_dict = {}
        for pair in tags.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                tag_dict[key.strip()] = value.strip()

    rate = collector.get_rate(metric_name, window=window, tags=tag_dict)

    return {
        "metric_name": metric_name,
        "rate_per_second": rate,
        "window_seconds": window,
        "tags": tag_dict,
    }


@router.get("/errors/summary", response_model=ErrorSummary)
async def get_error_summary() -> Dict[str, Any]:
    """Get aggregated error statistics.

    Returns summary of tracked errors by type and component.

    Returns:
        ErrorSummary with error statistics
    """
    tracker = get_error_tracker()
    stats = await tracker.get_stats()

    return stats


@router.get("/errors/top")
async def get_top_errors(
    limit: int = Query(10, description="Maximum errors to return"),
    sort_by: str = Query("count", description="Sort field: 'count' or 'last_seen'"),
) -> List[Dict[str, Any]]:
    """Get top errors sorted by count or recency.

    Args:
        limit: Maximum number of errors to return
        sort_by: Sort field ("count" or "last_seen")

    Returns:
        List of error records
    """
    tracker = get_error_tracker()
    errors = await tracker.get_top_errors(limit=limit, sort_by=sort_by)

    return [
        {
            "error_id": e.error_id,
            "exception_type": e.exception_type,
            "exception_message": e.exception_message,
            "count": e.count,
            "first_seen": e.first_seen,
            "last_seen": e.last_seen,
            "trace_ids": e.trace_ids[:5],  # Limit trace IDs for response size
        }
        for e in errors
    ]


@router.get("/errors/{error_id}")
async def get_error_details(error_id: str) -> Dict[str, Any]:
    """Get detailed error information including stack trace.

    Args:
        error_id: Error identifier

    Returns:
        Complete error record with stack trace
    """
    tracker = get_error_tracker()
    error = await tracker.get_error(error_id)

    if not error:
        raise HTTPException(status_code=404, detail=f"Error not found: {error_id}")

    return {
        "error_id": error.error_id,
        "exception_type": error.exception_type,
        "exception_message": error.exception_message,
        "stack_trace": error.stack_trace,
        "count": error.count,
        "first_seen": error.first_seen,
        "last_seen": error.last_seen,
        "trace_ids": error.trace_ids,
        "metadata": error.metadata,
    }


@router.get("/metrics/history")
async def get_metrics_history(
    start_time: Optional[str] = Query(None, description="Start timestamp (ISO 8601)"),
    end_time: Optional[str] = Query(None, description="End timestamp (ISO 8601)"),
    metric_name: Optional[str] = Query(None, description="Filter by metric name"),
) -> List[Dict[str, Any]]:
    """Get historical metrics from persistent storage.

    Args:
        start_time: Start timestamp (ISO 8601)
        end_time: End timestamp (ISO 8601)
        metric_name: Optional metric name filter

    Returns:
        List of metric snapshots
    """
    storage = get_metrics_storage()
    snapshots = await storage.query(
        start_time=start_time,
        end_time=end_time,
        metric_name=metric_name,
    )

    return [
        {
            "timestamp": s.timestamp,
            "metrics": s.metrics,
            "tags": s.tags,
        }
        for s in snapshots
    ]


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for observability service.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "cognita-observability",
    }
