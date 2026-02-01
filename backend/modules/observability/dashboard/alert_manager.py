"""Alert management system for threshold-based monitoring and notifications.

Defines alert rules with thresholds and triggers notifications when exceeded.
Supports multiple severity levels and notification channels.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..metrics.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(str, Enum):
    """Alert condition operators."""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="


class AlertRule(BaseModel):
    """Alert rule definition.

    Attributes:
        name: Rule name
        metric: Metric name to monitor
        condition: Comparison operator (>, <, >=, <=, ==, !=)
        threshold: Threshold value
        severity: Alert severity (info, warning, critical)
        window_seconds: Time window for aggregation
        enabled: Whether rule is active
        tags: Optional tags filter for metric
        metadata: Additional rule metadata
    """

    model_config = ConfigDict(use_enum_values=True)

    name: str
    metric: str
    condition: AlertCondition
    threshold: float
    severity: Severity = Severity.WARNING
    window_seconds: int = 60
    enabled: bool = True
    tags: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Alert(BaseModel):
    """Triggered alert instance.

    Attributes:
        alert_id: Unique alert identifier
        rule_name: Name of triggered rule
        metric: Metric that triggered alert
        current_value: Current metric value
        threshold: Threshold value from rule
        severity: Alert severity
        timestamp: Alert trigger time
        message: Human-readable alert message
    """

    model_config = ConfigDict(use_enum_values=True)

    alert_id: str
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: Severity
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: str


class AlertManager:
    """Alert management and notification system.

    Monitors metrics against defined rules and triggers alerts.
    Supports threshold-based alerting with multiple severity levels.

    Example:
        manager = AlertManager()

        # Define alert rule
        rule = AlertRule(
            name="high_latency",
            metric="query.latency_p95",
            condition=">",
            threshold=100.0,  # 100ms
            severity="critical",
            window_seconds=60
        )
        await manager.add_rule(rule)

        # Check alerts
        triggered = await manager.check_alerts()
        for alert in triggered:
            print(f"ALERT: {alert.message}")
    """

    def __init__(self):
        """Initialize alert manager."""
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._lock = asyncio.Lock()

    async def add_rule(self, rule: AlertRule) -> None:
        """Add or update alert rule.

        Args:
            rule: Alert rule to add
        """
        async with self._lock:
            self._rules[rule.name] = rule

    async def remove_rule(self, rule_name: str) -> None:
        """Remove alert rule.

        Args:
            rule_name: Name of rule to remove
        """
        async with self._lock:
            self._rules.pop(rule_name, None)

    async def get_rule(self, rule_name: str) -> Optional[AlertRule]:
        """Get alert rule by name.

        Args:
            rule_name: Rule name

        Returns:
            AlertRule if found, None otherwise
        """
        async with self._lock:
            return self._rules.get(rule_name)

    async def list_rules(self) -> List[AlertRule]:
        """List all alert rules.

        Returns:
            List of AlertRule instances
        """
        async with self._lock:
            return list(self._rules.values())

    def _evaluate_condition(
        self,
        current_value: float,
        condition: AlertCondition,
        threshold: float,
    ) -> bool:
        """Evaluate alert condition.

        Args:
            current_value: Current metric value
            condition: Comparison operator
            threshold: Threshold value

        Returns:
            True if condition met, False otherwise
        """
        if condition == AlertCondition.GREATER_THAN:
            return current_value > threshold
        elif condition == AlertCondition.LESS_THAN:
            return current_value < threshold
        elif condition == AlertCondition.GREATER_EQUAL:
            return current_value >= threshold
        elif condition == AlertCondition.LESS_EQUAL:
            return current_value <= threshold
        elif condition == AlertCondition.EQUAL:
            return current_value == threshold
        elif condition == AlertCondition.NOT_EQUAL:
            return current_value != threshold
        return False

    async def check_alerts(self) -> List[Alert]:
        """Check all rules and trigger alerts.

        Returns:
            List of newly triggered alerts
        """
        collector = get_metrics_collector()
        triggered_alerts = []

        async with self._lock:
            rules = list(self._rules.values())

        for rule in rules:
            if not rule.enabled:
                continue

            # Get current metric value
            current_value = collector.get_average(
                rule.metric,
                window=rule.window_seconds,
                tags=rule.tags,
            )

            # Evaluate condition
            if self._evaluate_condition(current_value, rule.condition, rule.threshold):
                alert_id = f"{rule.name}_{int(datetime.now(timezone.utc).timestamp())}"

                alert = Alert(
                    alert_id=alert_id,
                    rule_name=rule.name,
                    metric=rule.metric,
                    current_value=current_value,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.metric} {rule.condition.value} {rule.threshold} "
                    f"(current: {current_value:.2f})",
                )

                async with self._lock:
                    self._active_alerts[alert_id] = alert
                    self._alert_history.append(alert)

                triggered_alerts.append(alert)

                # Trigger notification
                await self.notify(alert)

        return triggered_alerts

    async def notify(self, alert: Alert) -> None:
        """Send alert notification.

        Args:
            alert: Alert to notify
        """
        # Log alert (can be extended to send emails, Slack, etc.)
        log_level = {
            Severity.INFO: logging.INFO,
            Severity.WARNING: logging.WARNING,
            Severity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)

        logger.log(log_level, f"ALERT TRIGGERED: {alert.message}")

        # TODO: Implement additional notification channels
        # - Email via SMTP
        # - Slack webhook
        # - PagerDuty integration
        # - Custom webhooks

    async def get_active_alerts(
        self,
        severity: Optional[Severity] = None,
    ) -> List[Alert]:
        """Get currently active alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        async with self._lock:
            alerts = list(self._active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    async def clear_alert(self, alert_id: str) -> None:
        """Clear/acknowledge active alert.

        Args:
            alert_id: Alert identifier
        """
        async with self._lock:
            self._active_alerts.pop(alert_id, None)

    async def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[Severity] = None,
    ) -> List[Alert]:
        """Get alert history.

        Args:
            limit: Maximum alerts to return
            severity: Optional severity filter

        Returns:
            List of historical alerts
        """
        async with self._lock:
            history = list(self._alert_history)

        if severity:
            history = [a for a in history if a.severity == severity]

        # Sort by timestamp descending
        history.sort(key=lambda a: a.timestamp, reverse=True)

        return history[:limit]


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create singleton alert manager instance.

    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
