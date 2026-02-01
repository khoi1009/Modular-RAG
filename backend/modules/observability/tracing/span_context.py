"""Span context dataclass for distributed tracing."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpanContext:
    """Represents a single span in a distributed trace.

    Attributes:
        trace_id: Unique identifier for the entire trace
        span_id: Unique identifier for this span
        parent_span_id: Parent span ID for nested spans
        start_time: Timestamp when span started (seconds since epoch)
        end_time: Timestamp when span ended
        name: Human-readable span name
        attributes: Key-value metadata attached to span
        events: List of timestamped events within span
        status: Span status (OK, ERROR, CANCELLED)
    """

    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> int:
        """Calculate span duration in milliseconds.

        Returns:
            Duration in milliseconds, or 0 if span not yet ended
        """
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0

    @property
    def is_complete(self) -> bool:
        """Check if span has been completed."""
        return self.end_time is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization.

        Returns:
            Dictionary representation of span
        """
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "name": self.name,
            "attributes": self.attributes,
            "events": self.events,
            "status": self.status,
        }

    @classmethod
    def create_root_span(cls, name: str, attributes: Optional[Dict[str, Any]] = None) -> "SpanContext":
        """Create a new root span (no parent).

        Args:
            name: Name of the span
            attributes: Optional attributes to attach

        Returns:
            New root span context
        """
        return cls(
            trace_id=str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            name=name,
            attributes=attributes or {},
        )

    @classmethod
    def create_child_span(
        cls,
        parent: "SpanContext",
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "SpanContext":
        """Create a child span from a parent.

        Args:
            parent: Parent span context
            name: Name of the child span
            attributes: Optional attributes to attach

        Returns:
            New child span context
        """
        return cls(
            trace_id=parent.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent.span_id,
            name=name,
            attributes=attributes or {},
        )

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a timestamped event to the span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute.

        Args:
            key: Attribute key
            value: Attribute value
        """
        self.attributes[key] = value

    def set_status(self, status: str) -> None:
        """Set span status.

        Args:
            status: Status string (OK, ERROR, CANCELLED)
        """
        self.status = status

    def end(self, status: str = "OK") -> None:
        """End the span with a final status.

        Args:
            status: Final status (OK, ERROR, CANCELLED)
        """
        self.end_time = time.time()
        self.status = status
