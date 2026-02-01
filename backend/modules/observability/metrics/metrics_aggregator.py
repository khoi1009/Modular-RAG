"""Metrics aggregator for percentiles, averages, and statistical analysis.

Provides statistical aggregation functions for histogram data.
Calculates percentiles (p50, p95, p99), averages, min/max, and standard deviation.
"""

import math
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class AggregatedStats(BaseModel):
    """Aggregated statistics for metric values.

    Attributes:
        count: Number of data points
        min: Minimum value
        max: Maximum value
        mean: Average value
        median: Median (p50)
        p95: 95th percentile
        p99: 99th percentile
        stddev: Standard deviation
    """

    model_config = ConfigDict(use_enum_values=True)

    count: int
    min: float
    max: float
    mean: float
    median: float
    p95: float
    p99: float
    stddev: float


class MetricsAggregator:
    """Statistical aggregator for metrics analysis.

    Provides percentile, average, and statistical calculations for histogram data.
    Used for latency analysis and performance monitoring.

    Example:
        aggregator = MetricsAggregator()

        # Calculate stats for latency values
        latencies = [23.1, 45.2, 67.8, 120.5, 89.3]
        stats = aggregator.aggregate(latencies)

        print(f"p95 latency: {stats.p95}ms")
        print(f"mean latency: {stats.mean}ms")
    """

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """Calculate percentile value.

        Args:
            values: List of numeric values
            p: Percentile (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)

        if p == 0:
            return sorted_values[0]
        if p == 100:
            return sorted_values[-1]

        # Linear interpolation
        index = (p / 100) * (n - 1)
        lower_index = int(math.floor(index))
        upper_index = int(math.ceil(index))

        if lower_index == upper_index:
            return sorted_values[lower_index]

        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        fraction = index - lower_index

        return lower_value + (upper_value - lower_value) * fraction

    @staticmethod
    def mean(values: List[float]) -> float:
        """Calculate mean (average) value.

        Args:
            values: List of numeric values

        Returns:
            Mean value
        """
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def stddev(values: List[float]) -> float:
        """Calculate standard deviation.

        Args:
            values: List of numeric values

        Returns:
            Standard deviation
        """
        if not values or len(values) < 2:
            return 0.0

        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def aggregate(self, values: List[float]) -> AggregatedStats:
        """Calculate comprehensive statistics.

        Args:
            values: List of numeric values

        Returns:
            AggregatedStats with all statistical measures
        """
        if not values:
            return AggregatedStats(
                count=0,
                min=0.0,
                max=0.0,
                mean=0.0,
                median=0.0,
                p95=0.0,
                p99=0.0,
                stddev=0.0,
            )

        return AggregatedStats(
            count=len(values),
            min=min(values),
            max=max(values),
            mean=self.mean(values),
            median=self.percentile(values, 50),
            p95=self.percentile(values, 95),
            p99=self.percentile(values, 99),
            stddev=self.stddev(values),
        )

    def aggregate_by_percentiles(
        self,
        values: List[float],
        percentiles: List[float],
    ) -> dict[float, float]:
        """Calculate custom percentiles.

        Args:
            values: List of numeric values
            percentiles: List of percentiles to calculate (0-100)

        Returns:
            Dictionary mapping percentile to value
        """
        if not values:
            return {p: 0.0 for p in percentiles}

        return {p: self.percentile(values, p) for p in percentiles}

    def calculate_rate(
        self,
        total: float,
        time_window_seconds: float,
    ) -> float:
        """Calculate rate (value per second).

        Args:
            total: Total count/sum
            time_window_seconds: Time window in seconds

        Returns:
            Rate as value per second
        """
        if time_window_seconds == 0:
            return 0.0
        return total / time_window_seconds

    def calculate_error_rate(
        self,
        error_count: int,
        total_count: int,
    ) -> float:
        """Calculate error rate as percentage.

        Args:
            error_count: Number of errors
            total_count: Total number of requests

        Returns:
            Error rate as percentage (0-100)
        """
        if total_count == 0:
            return 0.0
        return (error_count / total_count) * 100


# Singleton instance
_aggregator: Optional[MetricsAggregator] = None


def get_aggregator() -> MetricsAggregator:
    """Get or create singleton aggregator instance.

    Returns:
        MetricsAggregator instance
    """
    global _aggregator
    if _aggregator is None:
        _aggregator = MetricsAggregator()
    return _aggregator
