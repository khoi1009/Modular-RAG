"""Query tracer for distributed tracing with context propagation."""

import asyncio
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from .exporters import SpanExporter, ConsoleExporter
from .span_context import SpanContext


class QueryTracer:
    """Distributed tracer for tracking query execution across components.

    Uses ContextVar for async-safe context propagation.
    """

    _current_context: ContextVar[Optional[SpanContext]] = ContextVar("span_context", default=None)
    _trace_stack: ContextVar[list] = ContextVar("trace_stack", default=[])

    def __init__(self, exporter: Optional[SpanExporter] = None):
        """Initialize query tracer.

        Args:
            exporter: Span exporter for sending traces. Defaults to ConsoleExporter.
        """
        self.exporter = exporter or ConsoleExporter(pretty_print=False)
        self.spans: Dict[str, SpanContext] = {}
        self._lock = asyncio.Lock()

    def start_trace(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        """Start a new trace (root span).

        Args:
            name: Name of the trace
            attributes: Optional attributes to attach

        Returns:
            New root span context
        """
        ctx = SpanContext.create_root_span(name, attributes)
        self._current_context.set(ctx)
        self.spans[ctx.span_id] = ctx

        # Initialize stack for this trace
        stack = self._trace_stack.get()
        if stack is None:
            stack = []
            self._trace_stack.set(stack)
        stack.append(ctx)

        return ctx

    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanContext:
        """Start a new child span within current trace.

        Args:
            name: Name of the span
            attributes: Optional attributes to attach

        Returns:
            New child span context
        """
        parent = self._current_context.get()

        if parent is None:
            # No parent, create root span
            return self.start_trace(name, attributes)

        ctx = SpanContext.create_child_span(parent, name, attributes)
        self._current_context.set(ctx)
        self.spans[ctx.span_id] = ctx

        # Push to stack
        stack = self._trace_stack.get()
        if stack is not None:
            stack.append(ctx)

        return ctx

    async def end_span(self, status: str = "OK") -> None:
        """End the current span.

        Args:
            status: Final status (OK, ERROR, CANCELLED)
        """
        ctx = self._current_context.get()
        if ctx is None:
            return

        ctx.end(status)

        # Export the completed span
        await self.exporter.export(ctx)

        # Pop from stack and restore parent
        stack = self._trace_stack.get()
        if stack and stack[-1].span_id == ctx.span_id:
            stack.pop()

            # Restore parent context
            if stack:
                self._current_context.set(stack[-1])
            else:
                self._current_context.set(None)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the current span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        ctx = self._current_context.get()
        if ctx:
            ctx.add_event(name, attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        ctx = self._current_context.get()
        if ctx:
            ctx.set_attribute(key, value)

    def get_current_span(self) -> Optional[SpanContext]:
        """Get the current active span.

        Returns:
            Current span context or None
        """
        return self._current_context.get()

    def get_trace_id(self) -> Optional[str]:
        """Get current trace ID.

        Returns:
            Trace ID or None if no active trace
        """
        ctx = self._current_context.get()
        return ctx.trace_id if ctx else None

    def get_span(self, span_id: str) -> Optional[SpanContext]:
        """Retrieve a span by ID.

        Args:
            span_id: Span ID to retrieve

        Returns:
            Span context or None if not found
        """
        return self.spans.get(span_id)

    def get_trace_spans(self, trace_id: str) -> list[SpanContext]:
        """Get all spans for a trace.

        Args:
            trace_id: Trace ID

        Returns:
            List of spans in the trace
        """
        return [span for span in self.spans.values() if span.trace_id == trace_id]

    async def flush(self) -> None:
        """Flush any buffered spans to exporter."""
        if hasattr(self.exporter, 'flush'):
            await self.exporter.flush()

    async def shutdown(self) -> None:
        """Shutdown tracer and cleanup resources."""
        await self.flush()
        await self.exporter.shutdown()


# Global tracer instance
_global_tracer: Optional[QueryTracer] = None


def get_tracer() -> QueryTracer:
    """Get or create the global tracer instance.

    Returns:
        Global QueryTracer instance
    """
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = QueryTracer()
    return _global_tracer


def set_tracer(tracer: QueryTracer) -> None:
    """Set the global tracer instance.

    Args:
        tracer: Tracer instance to use globally
    """
    global _global_tracer
    _global_tracer = tracer
