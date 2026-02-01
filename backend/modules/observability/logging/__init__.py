"""Structured logging module with trace correlation and query replay.

Provides JSON-formatted logs with trace_id linking to distributed traces.
Supports query replay for debugging and error tracking for production monitoring.

Components:
    - StructuredLogger: JSON logger with trace correlation
    - QueryReplayStore: Persistent storage for query replay
    - ErrorTracker: Exception tracking and aggregation

Example:
    from backend.modules.observability.logging import get_logger

    logger = get_logger("retrieval")
    logger.info("Query received", trace_id=trace_id, query=query_text, k=10)

    # Output: {"timestamp": "2026-02-01T06:12:00Z", "level": "INFO",
    #          "service": "cognita", "component": "retrieval",
    #          "message": "Query received", "trace_id": "abc-123", ...}
"""

from .error_tracker import ErrorTracker, get_error_tracker
from .query_replay_store import QueryReplayStore, get_replay_store
from .structured_logger import StructuredLogger, get_logger

__all__ = [
    "StructuredLogger",
    "get_logger",
    "QueryReplayStore",
    "get_replay_store",
    "ErrorTracker",
    "get_error_tracker",
]
