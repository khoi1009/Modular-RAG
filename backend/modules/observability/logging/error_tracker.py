"""Exception tracking and aggregation for production error monitoring.

Tracks exceptions with stack traces, occurrence counts, and trace correlation.
Provides aggregated error statistics for alerting and debugging.
"""

import asyncio
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorRecord(BaseModel):
    """Recorded exception with context.

    Attributes:
        error_id: Unique error identifier (hash of exception type + message)
        first_seen: First occurrence timestamp
        last_seen: Most recent occurrence timestamp
        count: Total number of occurrences
        exception_type: Exception class name
        exception_message: Exception message
        stack_trace: Full stack trace
        trace_ids: Associated trace IDs for correlation
        metadata: Additional context (component, user_id, etc.)
    """

    model_config = ConfigDict(use_enum_values=True)

    error_id: str
    first_seen: str
    last_seen: str
    count: int = 1
    exception_type: str
    exception_message: str
    stack_trace: str
    trace_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorTracker:
    """In-memory error tracking and aggregation.

    Tracks exceptions with deduplication based on error signature.
    Provides aggregated statistics for monitoring and alerting.

    Example:
        tracker = ErrorTracker()

        try:
            result = await process_query(query)
        except Exception as e:
            await tracker.track(
                exception=e,
                trace_id=trace_id,
                component="retrieval",
                query=query_text
            )
            raise

        # Get top errors
        errors = await tracker.get_top_errors(limit=10)
    """

    def __init__(self, max_unique_errors: int = 1000):
        """Initialize error tracker.

        Args:
            max_unique_errors: Maximum unique error types to track
        """
        self.max_unique_errors = max_unique_errors
        self._errors: Dict[str, ErrorRecord] = {}
        self._lock = asyncio.Lock()

    def _generate_error_id(self, exception: Exception) -> str:
        """Generate unique error ID from exception.

        Args:
            exception: Exception instance

        Returns:
            Error ID (hash of type + message)
        """
        error_signature = f"{type(exception).__name__}:{str(exception)[:100]}"
        return str(hash(error_signature))

    async def track(
        self,
        exception: Exception,
        trace_id: Optional[str] = None,
        **metadata: Any,
    ) -> str:
        """Track exception occurrence.

        Args:
            exception: Exception to track
            trace_id: Optional trace ID for correlation
            **metadata: Additional context (component, user_id, etc.)

        Returns:
            Error ID
        """
        error_id = self._generate_error_id(exception)
        timestamp = datetime.now(timezone.utc).isoformat()
        stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        async with self._lock:
            if error_id in self._errors:
                # Update existing error
                error = self._errors[error_id]
                error.last_seen = timestamp
                error.count += 1
                if trace_id and trace_id not in error.trace_ids:
                    error.trace_ids.append(trace_id)
                # Merge metadata
                error.metadata.update(metadata)
            else:
                # Create new error record
                if len(self._errors) >= self.max_unique_errors:
                    # Remove oldest error by first_seen
                    oldest_id = min(self._errors.keys(), key=lambda k: self._errors[k].first_seen)
                    del self._errors[oldest_id]

                self._errors[error_id] = ErrorRecord(
                    error_id=error_id,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    count=1,
                    exception_type=type(exception).__name__,
                    exception_message=str(exception),
                    stack_trace=stack_trace,
                    trace_ids=[trace_id] if trace_id else [],
                    metadata=metadata,
                )

        return error_id

    async def get_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Get error record by ID.

        Args:
            error_id: Error identifier

        Returns:
            ErrorRecord if found, None otherwise
        """
        async with self._lock:
            return self._errors.get(error_id)

    async def get_top_errors(
        self,
        limit: int = 10,
        sort_by: str = "count",
    ) -> List[ErrorRecord]:
        """Get top errors sorted by count or recency.

        Args:
            limit: Maximum number of errors to return
            sort_by: Sort field ("count" or "last_seen")

        Returns:
            List of ErrorRecord instances
        """
        async with self._lock:
            errors = list(self._errors.values())

        if sort_by == "count":
            errors.sort(key=lambda e: e.count, reverse=True)
        elif sort_by == "last_seen":
            errors.sort(key=lambda e: e.last_seen, reverse=True)

        return errors[:limit]

    async def get_errors_by_component(
        self,
        component: str,
        limit: int = 10,
    ) -> List[ErrorRecord]:
        """Get errors filtered by component.

        Args:
            component: Component name to filter by
            limit: Maximum number of errors to return

        Returns:
            List of ErrorRecord instances
        """
        async with self._lock:
            filtered = [
                error
                for error in self._errors.values()
                if error.metadata.get("component") == component
            ]

        filtered.sort(key=lambda e: e.count, reverse=True)
        return filtered[:limit]

    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregate error statistics.

        Returns:
            Dictionary with error statistics
        """
        async with self._lock:
            total_unique = len(self._errors)
            total_occurrences = sum(error.count for error in self._errors.values())

            # Count by exception type
            by_type: Dict[str, int] = defaultdict(int)
            for error in self._errors.values():
                by_type[error.exception_type] += error.count

            # Count by component
            by_component: Dict[str, int] = defaultdict(int)
            for error in self._errors.values():
                component = error.metadata.get("component", "unknown")
                by_component[component] += error.count

        return {
            "total_unique_errors": total_unique,
            "total_occurrences": total_occurrences,
            "by_exception_type": dict(by_type),
            "by_component": dict(by_component),
        }

    async def clear(self) -> None:
        """Clear all tracked errors."""
        async with self._lock:
            self._errors.clear()


# Singleton instance
_error_tracker: Optional[ErrorTracker] = None


def get_error_tracker(max_unique_errors: int = 1000) -> ErrorTracker:
    """Get or create singleton error tracker instance.

    Args:
        max_unique_errors: Maximum unique error types to track

    Returns:
        ErrorTracker instance
    """
    global _error_tracker
    if _error_tracker is None:
        _error_tracker = ErrorTracker(max_unique_errors)
    return _error_tracker
