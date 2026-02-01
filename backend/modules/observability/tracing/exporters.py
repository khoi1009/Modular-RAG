"""Span exporters for sending traces to various backends."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import httpx

from .span_context import SpanContext

logger = logging.getLogger(__name__)


class SpanExporter(ABC):
    """Abstract base class for span exporters."""

    @abstractmethod
    async def export(self, span: SpanContext) -> None:
        """Export a span to the backend.

        Args:
            span: Span context to export
        """
        pass

    @abstractmethod
    async def export_batch(self, spans: List[SpanContext]) -> None:
        """Export multiple spans in a batch.

        Args:
            spans: List of span contexts to export
        """
        pass

    async def shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        pass


class ConsoleExporter(SpanExporter):
    """Exports spans to console output for debugging."""

    def __init__(self, pretty_print: bool = True):
        """Initialize console exporter.

        Args:
            pretty_print: Whether to format JSON output
        """
        self.pretty_print = pretty_print

    async def export(self, span: SpanContext) -> None:
        """Print span to console.

        Args:
            span: Span to export
        """
        if self.pretty_print:
            print(json.dumps(span.to_dict(), indent=2))
        else:
            print(json.dumps(span.to_dict()))

    async def export_batch(self, spans: List[SpanContext]) -> None:
        """Print multiple spans to console.

        Args:
            spans: Spans to export
        """
        span_dicts = [span.to_dict() for span in spans]
        if self.pretty_print:
            print(json.dumps(span_dicts, indent=2))
        else:
            print(json.dumps(span_dicts))


class JaegerExporter(SpanExporter):
    """Exports spans to Jaeger tracing backend."""

    def __init__(
        self,
        endpoint: str = "http://localhost:14268/api/traces",
        service_name: str = "cognita",
        batch_size: int = 100,
    ):
        """Initialize Jaeger exporter.

        Args:
            endpoint: Jaeger collector endpoint
            service_name: Service name for traces
            batch_size: Number of spans to batch before sending
        """
        self.endpoint = endpoint
        self.service_name = service_name
        self.batch_size = batch_size
        self.batch: List[SpanContext] = []
        self.client = httpx.AsyncClient(timeout=10.0)

    def _to_jaeger_format(self, span: SpanContext) -> Dict[str, Any]:
        """Convert span to Jaeger format.

        Args:
            span: Span to convert

        Returns:
            Jaeger-formatted span
        """
        return {
            "traceId": span.trace_id,
            "spanId": span.span_id,
            "operationName": span.name,
            "references": [
                {"refType": "CHILD_OF", "traceId": span.trace_id, "spanId": span.parent_span_id}
            ]
            if span.parent_span_id
            else [],
            "startTime": int(span.start_time * 1_000_000),  # microseconds
            "duration": span.duration_ms * 1000,  # microseconds
            "tags": [{"key": k, "type": "string", "value": str(v)} for k, v in span.attributes.items()],
            "logs": [
                {
                    "timestamp": int(event["timestamp"] * 1_000_000),
                    "fields": [
                        {"key": "event", "type": "string", "value": event["name"]},
                        *[
                            {"key": k, "type": "string", "value": str(v)}
                            for k, v in event.get("attributes", {}).items()
                        ],
                    ],
                }
                for event in span.events
            ],
        }

    async def export(self, span: SpanContext) -> None:
        """Export span to Jaeger.

        Args:
            span: Span to export
        """
        self.batch.append(span)
        if len(self.batch) >= self.batch_size:
            await self.flush()

    async def export_batch(self, spans: List[SpanContext]) -> None:
        """Export batch of spans to Jaeger.

        Args:
            spans: Spans to export
        """
        jaeger_spans = [self._to_jaeger_format(span) for span in spans]
        payload = {
            "data": [
                {
                    "process": {"serviceName": self.service_name, "tags": []},
                    "spans": jaeger_spans,
                }
            ]
        }

        try:
            response = await self.client.post(self.endpoint, json=payload)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to export spans to Jaeger: {e}")

    async def flush(self) -> None:
        """Flush buffered spans to Jaeger."""
        if self.batch:
            await self.export_batch(self.batch)
            self.batch.clear()

    async def shutdown(self) -> None:
        """Flush remaining spans and close client."""
        await self.flush()
        await self.client.aclose()


class HTTPExporter(SpanExporter):
    """Exports spans to custom HTTP endpoint."""

    def __init__(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        batch_size: int = 50,
    ):
        """Initialize HTTP exporter.

        Args:
            endpoint: HTTP endpoint URL
            headers: Optional HTTP headers
            batch_size: Number of spans to batch
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.batch_size = batch_size
        self.batch: List[SpanContext] = []
        self.client = httpx.AsyncClient(timeout=10.0)

    async def export(self, span: SpanContext) -> None:
        """Export span to HTTP endpoint.

        Args:
            span: Span to export
        """
        self.batch.append(span)
        if len(self.batch) >= self.batch_size:
            await self.flush()

    async def export_batch(self, spans: List[SpanContext]) -> None:
        """Export batch of spans.

        Args:
            spans: Spans to export
        """
        payload = {"spans": [span.to_dict() for span in spans]}

        try:
            response = await self.client.post(
                self.endpoint,
                json=payload,
                headers=self.headers,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to export spans to {self.endpoint}: {e}")

    async def flush(self) -> None:
        """Flush buffered spans."""
        if self.batch:
            await self.export_batch(self.batch)
            self.batch.clear()

    async def shutdown(self) -> None:
        """Flush and close client."""
        await self.flush()
        await self.client.aclose()
