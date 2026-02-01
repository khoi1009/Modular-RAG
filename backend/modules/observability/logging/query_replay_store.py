"""Persistent storage for query replay and debugging.

Stores queries, responses, and execution context for later replay.
Useful for debugging production issues and reproducing edge cases.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryRecord(BaseModel):
    """Recorded query execution for replay.

    Attributes:
        query_id: Unique query identifier
        trace_id: Associated trace ID
        timestamp: Query execution timestamp
        query_text: Original query string
        collection_name: Target collection
        k: Number of results requested
        filters: Optional metadata filters
        response: Query response data
        latency_ms: Total execution time
        metadata: Additional execution context
    """

    model_config = ConfigDict(use_enum_values=True)

    query_id: str
    trace_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query_text: str
    collection_name: str
    k: int = 10
    filters: Optional[Dict[str, Any]] = None
    response: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryReplayStore:
    """Persistent store for query records.

    Stores queries in JSONL format for easy replay and analysis.
    Supports filtering by collection, date range, and custom metadata.

    Example:
        store = QueryReplayStore("./data/query_replay")

        # Save query
        await store.save(QueryRecord(
            query_id="q123",
            trace_id="t456",
            query_text="What is RAG?",
            collection_name="docs",
            k=10,
            response={"results": [...]}
        ))

        # Replay query
        record = await store.get("q123")
        results = await replay_query(record)
    """

    def __init__(self, storage_path: str = "./data/query_replay"):
        """Initialize query replay store.

        Args:
            storage_path: Directory path for storing query records
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _get_file_path(self, date: Optional[str] = None) -> Path:
        """Get file path for date-partitioned storage.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Path to JSONL file for the date
        """
        if not date:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.storage_path / f"queries-{date}.jsonl"

    async def save(self, record: QueryRecord) -> None:
        """Save query record to storage.

        Args:
            record: Query record to save
        """
        date = record.timestamp.split("T")[0]  # Extract YYYY-MM-DD
        file_path = self._get_file_path(date)

        async with self._lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")

    async def get(self, query_id: str, date: Optional[str] = None) -> Optional[QueryRecord]:
        """Retrieve query record by ID.

        Args:
            query_id: Query identifier
            date: Date to search (YYYY-MM-DD), defaults to today

        Returns:
            QueryRecord if found, None otherwise
        """
        file_path = self._get_file_path(date)
        if not file_path.exists():
            return None

        async with self._lock:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    record_data = json.loads(line)
                    if record_data.get("query_id") == query_id:
                        return QueryRecord(**record_data)

        return None

    async def get_by_trace_id(self, trace_id: str, date: Optional[str] = None) -> Optional[QueryRecord]:
        """Retrieve query record by trace ID.

        Args:
            trace_id: Trace identifier
            date: Date to search (YYYY-MM-DD)

        Returns:
            QueryRecord if found, None otherwise
        """
        file_path = self._get_file_path(date)
        if not file_path.exists():
            return None

        async with self._lock:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    record_data = json.loads(line)
                    if record_data.get("trace_id") == trace_id:
                        return QueryRecord(**record_data)

        return None

    async def list_by_collection(
        self,
        collection_name: str,
        date: Optional[str] = None,
        limit: int = 100,
    ) -> List[QueryRecord]:
        """List queries for a specific collection.

        Args:
            collection_name: Collection to filter by
            date: Date to search (YYYY-MM-DD)
            limit: Maximum number of records to return

        Returns:
            List of QueryRecord instances
        """
        file_path = self._get_file_path(date)
        if not file_path.exists():
            return []

        records = []
        async with self._lock:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(records) >= limit:
                        break
                    record_data = json.loads(line)
                    if record_data.get("collection_name") == collection_name:
                        records.append(QueryRecord(**record_data))

        return records

    async def list_slow_queries(
        self,
        threshold_ms: int = 1000,
        date: Optional[str] = None,
        limit: int = 50,
    ) -> List[QueryRecord]:
        """List slow queries exceeding latency threshold.

        Args:
            threshold_ms: Latency threshold in milliseconds
            date: Date to search (YYYY-MM-DD)
            limit: Maximum number of records to return

        Returns:
            List of QueryRecord instances sorted by latency (descending)
        """
        file_path = self._get_file_path(date)
        if not file_path.exists():
            return []

        records = []
        async with self._lock:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    record_data = json.loads(line)
                    latency = record_data.get("latency_ms")
                    if latency and latency >= threshold_ms:
                        records.append(QueryRecord(**record_data))

        # Sort by latency descending
        records.sort(key=lambda r: r.latency_ms or 0, reverse=True)
        return records[:limit]


# Singleton instance
_replay_store: Optional[QueryReplayStore] = None


def get_replay_store(storage_path: str = "./data/query_replay") -> QueryReplayStore:
    """Get or create singleton replay store instance.

    Args:
        storage_path: Storage directory path

    Returns:
        QueryReplayStore instance
    """
    global _replay_store
    if _replay_store is None:
        _replay_store = QueryReplayStore(storage_path)
    return _replay_store
