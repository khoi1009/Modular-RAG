"""
Rate limiting for API calls with token bucket and adaptive algorithms.

Implements token bucket rate limiting with adaptive rate adjustment
based on observed error patterns.
"""

import asyncio
import time
from typing import Any, Dict

from backend.logger import logger


class RateLimiterMetrics:
    """Metrics tracking for rate limiter."""

    def __init__(self, name: str):
        self.name = name
        self.success_count = 0
        self.timeout_count = 0
        self.total_wait_time = 0.0

    def record_success(self) -> None:
        """Record successful token acquisition."""
        self.success_count += 1

    def record_timeout(self) -> None:
        """Record timeout waiting for tokens."""
        self.timeout_count += 1

    def record_wait(self, wait_time: float) -> None:
        """Record wait time."""
        self.total_wait_time += wait_time

    def get_stats(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "name": self.name,
            "success_count": self.success_count,
            "timeout_count": self.timeout_count,
            "total_wait_time": self.total_wait_time,
            "avg_wait_time": (
                self.total_wait_time / self.success_count
                if self.success_count > 0
                else 0.0
            ),
        }


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Implements classic token bucket algorithm for rate limiting with
    configurable refill rate and burst capacity.
    """

    def __init__(
        self,
        tokens_per_second: float,
        max_tokens: int,
        name: str = "default",
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            tokens_per_second: Token refill rate per second
            max_tokens: Maximum bucket capacity (burst size)
            name: Identifier for this rate limiter
        """
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = float(max_tokens)
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.name = name
        self.metrics = RateLimiterMetrics(name)

        logger.info(
            f"TokenBucketRateLimiter '{name}' initialized: "
            f"rate={tokens_per_second}/s, max_tokens={max_tokens}"
        )

    async def acquire(self, tokens: int = 1, timeout_sec: float = 30.0) -> bool:
        """
        Acquire tokens from bucket, wait if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout_sec: Maximum time to wait for tokens

        Returns:
            True if tokens acquired, False if timed out
        """
        start = time.time()

        while True:
            async with self.lock:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    wait_time = time.time() - start
                    self.metrics.record_success()
                    self.metrics.record_wait(wait_time)

                    if wait_time > 0.1:
                        logger.debug(
                            f"Rate limiter '{self.name}': acquired {tokens} tokens "
                            f"after {wait_time:.3f}s"
                        )
                    return True

            # Check timeout
            if time.time() - start > timeout_sec:
                logger.warning(
                    f"Rate limiter '{self.name}': timeout after {timeout_sec}s "
                    f"waiting for {tokens} tokens"
                )
                self.metrics.record_timeout()
                return False

            # Wait before retry
            wait_time = tokens / self.tokens_per_second
            await asyncio.sleep(min(wait_time, 0.1))

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update

        # Add tokens based on elapsed time
        self.tokens = min(
            self.max_tokens, self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with metrics and current state
        """
        return {
            "name": self.name,
            "tokens_per_second": self.tokens_per_second,
            "max_tokens": self.max_tokens,
            "current_tokens": self.tokens,
            "metrics": self.metrics.get_stats(),
        }


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts rate based on errors.

    Monitors API responses and automatically reduces rate when rate limit
    errors are detected, then gradually increases rate on success.
    """

    def __init__(self, base_rate: float, name: str):
        """
        Initialize adaptive rate limiter.

        Args:
            base_rate: Base tokens per second
            name: Identifier for this rate limiter
        """
        self.base_rate = base_rate
        self.current_rate = base_rate
        self.name = name
        self.rate_limiter = TokenBucketRateLimiter(
            base_rate, int(base_rate * 2), f"{name}_adaptive"
        )
        self.success_count = 0
        self.error_count = 0
        self.rate_adjustments = 0

        logger.info(f"AdaptiveRateLimiter '{name}' initialized: base_rate={base_rate}/s")

    async def acquire(self, tokens: int = 1, timeout_sec: float = 30.0) -> bool:
        """
        Acquire tokens with adaptive rate limiting.

        Args:
            tokens: Number of tokens to acquire
            timeout_sec: Maximum wait time

        Returns:
            True if tokens acquired, False if timed out
        """
        success = await self.rate_limiter.acquire(tokens, timeout_sec)

        if success:
            self.success_count += 1
            self._maybe_increase_rate()

        return success

    def record_error(self, error_type: str) -> None:
        """
        Record API error and adjust rate if needed.

        Args:
            error_type: Type of error ("rate_limit", "timeout", etc.)
        """
        self.error_count += 1

        if error_type == "rate_limit":
            self._decrease_rate()
        elif error_type == "timeout":
            # Mild decrease for timeouts
            self._decrease_rate(factor=0.8)

    def record_success(self) -> None:
        """Record successful API call."""
        self.success_count += 1
        self._maybe_increase_rate()

    def _decrease_rate(self, factor: float = 0.5) -> None:
        """
        Decrease rate limit.

        Args:
            factor: Multiplier for rate reduction (0-1)
        """
        old_rate = self.rate_limiter.tokens_per_second
        new_rate = max(old_rate * factor, 0.1)  # Minimum 0.1/s

        if new_rate != old_rate:
            self.rate_limiter.tokens_per_second = new_rate
            self.current_rate = new_rate
            self.rate_adjustments += 1

            logger.warning(
                f"Rate limiter '{self.name}': decreased rate "
                f"{old_rate:.2f} -> {new_rate:.2f} tokens/s"
            )

    def _maybe_increase_rate(self) -> None:
        """Gradually increase rate after sustained success."""
        # Increase rate every 100 successful calls if no recent errors
        if self.success_count % 100 == 0 and self.error_count == 0:
            old_rate = self.rate_limiter.tokens_per_second
            # Don't exceed 2x base rate
            new_rate = min(old_rate * 1.1, self.base_rate * 2)

            if new_rate > old_rate:
                self.rate_limiter.tokens_per_second = new_rate
                self.current_rate = new_rate
                self.rate_adjustments += 1

                logger.info(
                    f"Rate limiter '{self.name}': increased rate "
                    f"{old_rate:.2f} -> {new_rate:.2f} tokens/s"
                )

    def reset_error_count(self) -> None:
        """Reset error counter (called periodically)."""
        self.error_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get adaptive rate limiter statistics.

        Returns:
            Dictionary with metrics and state
        """
        base_stats = self.rate_limiter.get_stats()
        return {
            **base_stats,
            "base_rate": self.base_rate,
            "current_rate": self.current_rate,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "rate_adjustments": self.rate_adjustments,
        }
