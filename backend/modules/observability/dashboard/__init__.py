"""Dashboard module for real-time metrics visualization and alerting.

Provides FastAPI endpoints for metrics querying, trace visualization, and alert management.
Integrates with frontend dashboards for observability monitoring.

Components:
    - MetricsAPI: REST endpoints for metrics and traces
    - AlertManager: Alert rules and notification system

Example:
    from fastapi import FastAPI
    from backend.modules.observability.dashboard import metrics_router

    app = FastAPI()
    app.include_router(metrics_router, prefix="/api/observability")

    # Endpoints:
    # GET /api/observability/metrics/realtime
    # GET /api/observability/metrics/traces/{trace_id}
    # GET /api/observability/alerts/active
    # POST /api/observability/alerts/rules
"""

from .alert_manager import AlertManager, AlertRule, get_alert_manager
from .metrics_api import router as metrics_router

__all__ = [
    "metrics_router",
    "AlertManager",
    "AlertRule",
    "get_alert_manager",
]
