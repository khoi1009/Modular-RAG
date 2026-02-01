"""Structured JSON logger with trace correlation for production observability."""

import json
import logging
import sys
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class LogLevel(str, Enum):
    """Log severity levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry(BaseModel):
    """Structured log entry model.

    Attributes:
        timestamp: ISO 8601 timestamp in UTC
        level: Log severity level
        service: Service name (default: cognita)
        component: Component/module name (e.g., retrieval, embedding)
        message: Human-readable message
        trace_id: Optional trace ID for correlation with distributed traces
        span_id: Optional span ID for specific operation
        extra: Additional key-value metadata
    """

    model_config = ConfigDict(use_enum_values=True)

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    level: LogLevel
    service: str = "cognita"
    component: Optional[str] = None
    message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize log entry to JSON string."""
        data = self.model_dump(exclude_none=True)
        # Merge extra fields into top-level
        extra = data.pop("extra", {})
        data.update(extra)
        return json.dumps(data)


class StructuredLogger:
    """JSON structured logger with trace correlation.

    Outputs logs in JSON format for easy ingestion by log aggregation systems
    (e.g., ELK, Datadog, CloudWatch). Automatically links logs to distributed
    traces via trace_id field.

    Example:
        logger = StructuredLogger("retrieval")
        logger.info("Query executed", trace_id="abc-123", k=10, latency_ms=45)

        # Output: {"timestamp": "2026-02-01T06:12:00Z", "level": "INFO",
        #          "service": "cognita", "component": "retrieval",
        #          "message": "Query executed", "trace_id": "abc-123",
        #          "k": 10, "latency_ms": 45}
    """

    def __init__(
        self,
        component: Optional[str] = None,
        service: str = "cognita",
        output_stream=None,
    ):
        """Initialize structured logger.

        Args:
            component: Component/module name for log entries
            service: Service name (default: cognita)
            output_stream: Output stream (default: sys.stdout)
        """
        self.component = component
        self.service = service
        self.output_stream = output_stream or sys.stdout
        self._stdlib_logger = logging.getLogger(f"{service}.{component}" if component else service)

    def _log(
        self,
        level: LogLevel,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Internal logging method.

        Args:
            level: Log severity level
            message: Log message
            trace_id: Optional trace ID for correlation
            span_id: Optional span ID
            **extra: Additional metadata
        """
        entry = LogEntry(
            level=level,
            service=self.service,
            component=self.component,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            extra=extra,
        )

        json_log = entry.to_json()
        self.output_stream.write(json_log + "\n")
        self.output_stream.flush()

        # Also log to stdlib logger for compatibility
        stdlib_level = getattr(logging, level.value)
        self._stdlib_logger.log(stdlib_level, message, extra=extra)

    def debug(
        self,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, trace_id, span_id, **extra)

    def info(
        self,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, trace_id, span_id, **extra)

    def warning(
        self,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, trace_id, span_id, **extra)

    def error(
        self,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        exc_info: Optional[Exception] = None,
        **extra: Any,
    ) -> None:
        """Log error message with optional exception info."""
        if exc_info:
            extra["exception_type"] = type(exc_info).__name__
            extra["exception_message"] = str(exc_info)
        self._log(LogLevel.ERROR, message, trace_id, span_id, **extra)

    def critical(
        self,
        message: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, trace_id, span_id, **extra)


# Singleton logger instances
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(component: Optional[str] = None, service: str = "cognita") -> StructuredLogger:
    """Get or create a structured logger instance.

    Args:
        component: Component name for the logger
        service: Service name (default: cognita)

    Returns:
        StructuredLogger instance
    """
    key = f"{service}.{component}" if component else service
    if key not in _loggers:
        _loggers[key] = StructuredLogger(component=component, service=service)
    return _loggers[key]
