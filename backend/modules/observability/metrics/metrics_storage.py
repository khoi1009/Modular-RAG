"""Time-series storage for metrics persistence and historical analysis.

Stores metrics in date-partitioned JSONL files for long-term retention.
Supports querying by time range and metric name.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class MetricSnapshot(BaseModel):
    """Point-in-time metrics snapshot.

    Attributes:
        timestamp: Snapshot timestamp
        metrics: Dictionary of metric name to value
        tags: Optional tags for categorization
    """

    model_config = ConfigDict(use_enum_values=True)

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metrics: Dict[str, float]
    tags: Optional[Dict[str, str]] = None


class MetricsStorage:
    """Persistent time-series storage for metrics.

    Stores metrics in date-partitioned JSONL files for efficient querying.
    Supports automatic cleanup of old data.

    Example:
        storage = MetricsStorage("./data/metrics")

        # Save snapshot
        await storage.save(MetricSnapshot(
            metrics={
                "queries.total": 1234,
                "query.latency_p50": 45.2,
                "query.latency_p95": 120.5
            },
            tags={"collection": "docs"}
        ))

        # Query time range
        snapshots = await storage.query(
            start_time="2026-02-01T00:00:00Z",
            end_time="2026-02-01T23:59:59Z"
        )
    """

    def __init__(
        self,
        storage_path: str = "./data/metrics",
        retention_days: int = 30,
    ):
        """Initialize metrics storage.

        Args:
            storage_path: Directory path for storing metrics
            retention_days: Days to retain metrics before cleanup
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._lock = asyncio.Lock()

    def _get_file_path(self, date: Optional[str] = None) -> Path:
        """Get file path for date-partitioned storage.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to JSONL file
        """
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.storage_path / f"metrics-{date}.jsonl"

    async def save(self, snapshot: MetricSnapshot) -> None:
        """Save metrics snapshot to storage.

        Args:
            snapshot: Metrics snapshot to save
        """
        date = snapshot.timestamp.split("T")[0]  # Extract YYYY-MM-DD
        file_path = self._get_file_path(date)

        async with self._lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(snapshot.model_dump_json() + "\n")

    async def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        metric_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[MetricSnapshot]:
        """Query metrics by time range and filters.

        Args:
            start_time: Start timestamp (ISO 8601)
            end_time: End timestamp (ISO 8601)
            metric_name: Filter by specific metric name
            tags: Filter by tags

        Returns:
            List of matching MetricSnapshot instances
        """
        # Determine date range
        if start_time:
            start_date = start_time.split("T")[0]
        else:
            start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if end_time:
            end_date = end_time.split("T")[0]
        else:
            end_date = start_date

        # Collect snapshots from all relevant files
        snapshots = []
        current_date = start_date

        async with self._lock:
            while current_date <= end_date:
                file_path = self._get_file_path(current_date)
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            snapshot_data = json.loads(line)
                            snapshot = MetricSnapshot(**snapshot_data)

                            # Apply filters
                            if start_time and snapshot.timestamp < start_time:
                                continue
                            if end_time and snapshot.timestamp > end_time:
                                continue
                            if metric_name and metric_name not in snapshot.metrics:
                                continue
                            if tags and (not snapshot.tags or not all(
                                snapshot.tags.get(k) == v for k, v in tags.items()
                            )):
                                continue

                            snapshots.append(snapshot)

                # Move to next date
                date_obj = datetime.fromisoformat(current_date)
                next_date = date_obj.replace(day=date_obj.day + 1)
                current_date = next_date.strftime("%Y-%m-%d")

                # Safety check to prevent infinite loop
                if len(current_date.split("-")[0]) > 4:
                    break

        return snapshots

    async def get_latest(
        self,
        metric_name: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[float]:
        """Get latest value for a metric.

        Args:
            metric_name: Metric name
            tags: Optional tags filter

        Returns:
            Latest metric value or None
        """
        snapshots = await self.query(metric_name=metric_name, tags=tags)
        if not snapshots:
            return None

        # Sort by timestamp descending
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots[0].metrics.get(metric_name)

    async def cleanup_old_data(self) -> int:
        """Remove metrics older than retention period.

        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now(timezone.utc).replace(
            day=datetime.now(timezone.utc).day - self.retention_days
        )
        cutoff_str = cutoff_date.strftime("%Y-%m-%d")

        deleted_count = 0
        async with self._lock:
            for file_path in self.storage_path.glob("metrics-*.jsonl"):
                # Extract date from filename
                date_str = file_path.stem.replace("metrics-", "")
                if date_str < cutoff_str:
                    file_path.unlink()
                    deleted_count += 1

        return deleted_count


# Singleton instance
_metrics_storage: Optional[MetricsStorage] = None


def get_metrics_storage(
    storage_path: str = "./data/metrics",
    retention_days: int = 30,
) -> MetricsStorage:
    """Get or create singleton metrics storage instance.

    Args:
        storage_path: Storage directory path
        retention_days: Days to retain metrics

    Returns:
        MetricsStorage instance
    """
    global _metrics_storage
    if _metrics_storage is None:
        _metrics_storage = MetricsStorage(storage_path, retention_days)
    return _metrics_storage
