"""Decorators for automatic tracing of functions."""

import functools
import inspect
from typing import Any, Callable, Optional

from .query_tracer import get_tracer


def traced(name: Optional[str] = None, attributes: Optional[dict] = None):
    """Decorator to automatically trace function execution.

    Args:
        name: Optional span name (defaults to function name)
        attributes: Optional attributes to attach to span

    Example:
        @traced("process_query")
        async def process_query(query: str):
            # Function is automatically traced
            return result
    """
    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs) -> Any:
                tracer = get_tracer()

                # Extract useful attributes from function arguments
                span_attrs = attributes.copy() if attributes else {}
                if args and hasattr(args[0], '__class__'):
                    span_attrs['class'] = args[0].__class__.__name__

                tracer.start_span(span_name, span_attrs)
                try:
                    result = await fn(*args, **kwargs)
                    await tracer.end_span("OK")
                    return result
                except Exception as e:
                    tracer.set_attribute("error.type", type(e).__name__)
                    tracer.set_attribute("error.message", str(e))
                    await tracer.end_span("ERROR")
                    raise

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs) -> Any:
                tracer = get_tracer()

                # Extract useful attributes
                span_attrs = attributes.copy() if attributes else {}
                if args and hasattr(args[0], '__class__'):
                    span_attrs['class'] = args[0].__class__.__name__

                tracer.start_span(span_name, span_attrs)
                try:
                    result = fn(*args, **kwargs)
                    # For sync functions, we need to handle end_span differently
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(tracer.end_span("OK"))
                        else:
                            loop.run_until_complete(tracer.end_span("OK"))
                    except RuntimeError:
                        # No event loop, create one
                        asyncio.run(tracer.end_span("OK"))
                    return result
                except Exception as e:
                    tracer.set_attribute("error.type", type(e).__name__)
                    tracer.set_attribute("error.message", str(e))
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(tracer.end_span("ERROR"))
                        else:
                            loop.run_until_complete(tracer.end_span("ERROR"))
                    except RuntimeError:
                        asyncio.run(tracer.end_span("ERROR"))
                    raise

            return sync_wrapper

    return decorator


def trace_method(name: Optional[str] = None):
    """Decorator for tracing class methods (includes class name in span).

    Args:
        name: Optional span name suffix (defaults to method name)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs) -> Any:
            tracer = get_tracer()
            class_name = self.__class__.__name__
            method_name = name or fn.__name__
            span_name = f"{class_name}.{method_name}"

            tracer.start_span(span_name, {"class": class_name, "method": method_name})
            try:
                result = await fn(self, *args, **kwargs)
                await tracer.end_span("OK")
                return result
            except Exception as e:
                tracer.set_attribute("error.type", type(e).__name__)
                tracer.set_attribute("error.message", str(e))
                await tracer.end_span("ERROR")
                raise

        return wrapper

    return decorator
