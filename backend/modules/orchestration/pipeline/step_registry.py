"""Registry for pipeline step functions."""
from typing import Callable, Dict

from backend.logger import logger


class StepRegistry:
    """
    Registry for pipeline step functions.
    Allows registering and retrieving step implementations.
    """

    _steps: Dict[str, Callable] = {}

    @classmethod
    def register(cls, module_path: str):
        """
        Decorator to register a step function.

        Args:
            module_path: Dot-separated path (e.g., "query_rewriting.hyde")

        Example:
            @StepRegistry.register("query_rewriting.hyde")
            async def hyde_rewrite(context):
                ...
        """

        def decorator(fn: Callable):
            cls._steps[module_path] = fn
            logger.debug(f"Registered pipeline step: {module_path}")
            return fn

        return decorator

    @classmethod
    def get(cls, module_path: str) -> Callable:
        """
        Get a registered step function.

        Args:
            module_path: Dot-separated path

        Returns:
            Registered step function

        Raises:
            ValueError: If step not found
        """
        if module_path not in cls._steps:
            available = ", ".join(cls._steps.keys())
            raise ValueError(
                f"Unknown step: {module_path}. Available steps: {available}"
            )
        return cls._steps[module_path]

    @classmethod
    def list_steps(cls) -> list:
        """List all registered step names"""
        return list(cls._steps.keys())

    @classmethod
    def clear(cls):
        """Clear all registered steps (mainly for testing)"""
        cls._steps.clear()
