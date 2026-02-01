"""Lightweight metrics collector for counters, gauges, and histograms.

Provides in-memory metrics collection with minimal overhead (<1ms per metric).
Supports tags for multi-dimensional metrics and rate calculations.
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class MetricPoint(BaseModel):
    """Single metric data point.

    Attributes:
        name: Metric name (e.g., "query.latency_ms")
        value: Metric value
        timestamp: Unix timestamp
        tags: Optional tags for multi-dimensional metrics
        metric_type: Type of metric (counter, gauge, histogram)
    """

    model_config = ConfigDict(use_enum_values=True)

    name: str
    value: float
    timestamp: float
    tags: Optional[Dict[str, str]] = None
    metric_type: str  # counter, gauge, histogram


class MetricsCollector:
    """In-memory metrics collector.

    Collects counters, gauges, and histograms with tag support.
    Maintains recent history for rate and aggregation calculations.

    Example:
        collector = MetricsCollector()

        # Counter: increment by 1 or N
        collector.increment("api.requests", tags={"endpoint": "/search"})
        collector.increment("api.errors", value=5)

        # Gauge: set current value
        collector.gauge("system.memory_mb", 1024.5)

        # Histogram: record distribution
        collector.histogram("query.latency_ms", 45.2)

        # Calculate rate
        rate = collector.get_rate("api.requests", window=60)  # requests/sec
    """

    def __init__(self, max_history: int = 10000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum data points to retain per metric
        """
        self.max_history = max_history
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = asyncio.Lock()

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Generate metric key from name and tags.

        Args:
            name: Metric name
            tags: Optional tags

        Returns:
            Metric key string
        """
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"

    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment counter metric.

        Args:
            name: Metric name
            value: Increment amount (default: 1)
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        timestamp = time.time()

        self._counters[key] += value
        self._history[key].append((timestamp, value))

    def gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set gauge metric to specific value.

        Args:
            name: Metric name
            value: Current value
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        timestamp = time.time()

        self._gauges[key] = value
        self._history[key].append((timestamp, value))

    def histogram(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record value in histogram distribution.

        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags
        """
        key = self._make_key(name, tags)
        timestamp = time.time()

        self._histograms[key].append(value)
        self._history[key].append((timestamp, value))

    def get_counter(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Get current counter value.

        Args:
            name: Metric name
            tags: Optional tags

        Returns:
            Counter value
        """
        key = self._make_key(name, tags)
        return self._counters.get(key, 0.0)

    def get_gauge(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get current gauge value.

        Args:
            name: Metric name
            tags: Optional tags

        Returns:
            Gauge value or None if not set
        """
        key = self._make_key(name, tags)
        return self._gauges.get(key)

    def get_histogram_values(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[float]:
        """Get histogram values.

        Args:
            name: Metric name
            tags: Optional tags

        Returns:
            List of recorded values
        """
        key = self._make_key(name, tags)
        return list(self._histograms.get(key, []))

    def get_rate(
        self,
        name: str,
        window: int = 60,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Calculate rate (value/sec) over time window.

        Args:
            name: Metric name
            window: Time window in seconds
            tags: Optional tags

        Returns:
            Rate as value per second
        """
        key = self._make_key(name, tags)
        history = self._history.get(key, deque())

        if not history:
            return 0.0

        now = time.time()
        cutoff = now - window

        # Sum values within window
        total = sum(value for ts, value in history if ts >= cutoff)

        # Count time points within window
        time_points = [(ts, value) for ts, value in history if ts >= cutoff]
        if not time_points:
            return 0.0

        # Calculate actual time span
        min_ts = min(ts for ts, _ in time_points)
        max_ts = max(ts for ts, _ in time_points)
        time_span = max_ts - min_ts

        if time_span == 0:
            return 0.0

        return total / time_span

    def get_average(
        self,
        name: str,
        window: int = 300,
        tags: Optional[Dict[str, str]] = None,
    ) -> float:
        """Calculate average value over time window.

        Args:
            name: Metric name
            window: Time window in seconds
            tags: Optional tags

        Returns:
            Average value
        """
        key = self._make_key(name, tags)
        history = self._history.get(key, deque())

        if not history:
            return 0.0

        now = time.time()
        cutoff = now - window

        # Get values within window
        values = [value for ts, value in history if ts >= cutoff]

        if not values:
            return 0.0

        return sum(values) / len(values)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get snapshot of all current metrics.

        Returns:
            Dictionary with counters, gauges, histograms
        """
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_counts": {k: len(v) for k, v in self._histograms.items()},
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._history.clear()


# Singleton instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector(max_history: int = 10000) -> MetricsCollector:
    """Get or create singleton metrics collector instance.

    Args:
        max_history: Maximum data points to retain

    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(max_history)
    return _metrics_collector
