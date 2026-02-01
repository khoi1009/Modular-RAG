# Phase 6: Observability & Monitoring

**Duration:** Week 9 | **Priority:** P1 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [Phase 4: Orchestration Engine](phase-04-orchestration-engine.md)
- [Phase 5: Verification & QC](phase-05-verification-quality-control.md)

## Overview

Production-grade monitoring, distributed tracing, structured logging, and real-time dashboards. Essential for debugging, performance analysis, and operational excellence.

## Key Insights

Current state:
- Basic logging via `backend.logger`
- No distributed tracing
- No query replay capability
- No metrics collection
- No monitoring dashboard

Target state:
- OpenTelemetry-compatible tracing
- Structured JSON logging with correlation IDs
- Query replay for debugging
- Time-series metrics storage
- Real-time monitoring dashboard

## Requirements

### Functional
- Query Tracing: Trace every query through all pipeline steps
- Structured Logging: JSON logs with trace IDs, component tags
- Query Replay: Save/replay queries for debugging
- Metrics Dashboard: Real-time visualization of key metrics

### Non-Functional
- Tracing overhead < 10ms per query
- Log storage with 30-day retention
- Dashboard refresh < 5s
- Export to Jaeger, DataDog, or custom backend

## Architecture

### Module Structure
```
backend/modules/
└── observability/
    ├── __init__.py
    ├── tracing/
    │   ├── __init__.py
    │   ├── tracer.py
    │   ├── span-context.py
    │   ├── exporters.py
    │   └── decorators.py
    ├── logging/
    │   ├── __init__.py
    │   ├── structured-logger.py
    │   ├── query-replay.py
    │   └── error-tracker.py
    ├── metrics/
    │   ├── __init__.py
    │   ├── collectors.py
    │   ├── storage.py
    │   └── aggregators.py
    └── dashboard/
        ├── __init__.py
        ├── metrics-api.py
        └── alerting.py

frontend/src/components/
└── Monitoring/
    ├── MetricsDashboard.tsx
    ├── QueryTraceViewer.tsx
    └── AlertsPanel.tsx
```

### Tracing Flow
```
User Query
    ↓
Tracer.start_span("query_processing")
    ├── span: query_analysis (145ms)
    ├── span: routing (23ms)
    ├── span: retrieval (523ms)
    │   ├── span: vector_search (312ms)
    │   └── span: bm25_search (201ms)
    ├── span: reranking (89ms)
    ├── span: generation (1234ms)
    └── span: verification (234ms)
    ↓
Tracer.end_span()
    ↓
Exporter → Jaeger/DataDog/Custom
```

## Related Code Files

### Files to Reference
- `backend/logger.py` - Existing logging setup
- `backend/modules/orchestration/pipeline/pipeline-executor.py` - Add tracing

### Files to Create
- `backend/modules/observability/__init__.py`
- `backend/modules/observability/tracing/__init__.py`
- `backend/modules/observability/tracing/tracer.py`
- `backend/modules/observability/tracing/span-context.py`
- `backend/modules/observability/tracing/exporters.py`
- `backend/modules/observability/tracing/decorators.py`
- `backend/modules/observability/logging/__init__.py`
- `backend/modules/observability/logging/structured-logger.py`
- `backend/modules/observability/logging/query-replay.py`
- `backend/modules/observability/logging/error-tracker.py`
- `backend/modules/observability/metrics/__init__.py`
- `backend/modules/observability/metrics/collectors.py`
- `backend/modules/observability/metrics/storage.py`
- `backend/modules/observability/metrics/aggregators.py`
- `backend/modules/observability/dashboard/__init__.py`
- `backend/modules/observability/dashboard/metrics-api.py`
- `backend/modules/observability/dashboard/alerting.py`
- `frontend/src/components/Monitoring/MetricsDashboard.tsx`
- `tests/modules/observability/test_tracing.py`

## Implementation Steps

### Task 6.1: Query Tracing System (Days 1-2)

1. Create `tracing/span-context.py`:
```python
@dataclass
class SpanContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    name: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time) * 1000)
        return 0
```

2. Create `tracing/tracer.py`:
```python
class QueryTracer:
    _current_context: ContextVar[Optional[SpanContext]] = ContextVar("span_context", default=None)

    def __init__(self, exporter: SpanExporter):
        self.exporter = exporter
        self.spans: Dict[str, SpanContext] = {}

    def start_trace(self, name: str, attributes: Optional[Dict] = None) -> SpanContext:
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        ctx = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            name=name,
            attributes=attributes or {}
        )
        self._current_context.set(ctx)
        self.spans[span_id] = ctx
        return ctx

    def start_span(self, name: str, attributes: Optional[Dict] = None) -> SpanContext:
        parent = self._current_context.get()
        span_id = str(uuid.uuid4())
        ctx = SpanContext(
            trace_id=parent.trace_id if parent else str(uuid.uuid4()),
            span_id=span_id,
            parent_span_id=parent.span_id if parent else None,
            name=name,
            attributes=attributes or {}
        )
        self._current_context.set(ctx)
        self.spans[span_id] = ctx
        return ctx

    def end_span(self, status: str = "OK") -> None:
        ctx = self._current_context.get()
        if ctx:
            ctx.end_time = time.time()
            ctx.status = status
            self.exporter.export(ctx)
            # Restore parent context
            if ctx.parent_span_id:
                self._current_context.set(self.spans.get(ctx.parent_span_id))

    def add_event(self, name: str, attributes: Optional[Dict] = None) -> None:
        ctx = self._current_context.get()
        if ctx:
            ctx.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {}
            })
```

3. Create `tracing/decorators.py`:
```python
def traced(name: Optional[str] = None):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            span_name = name or fn.__name__
            tracer.start_span(span_name)
            try:
                result = await fn(*args, **kwargs)
                tracer.end_span("OK")
                return result
            except Exception as e:
                tracer.end_span("ERROR")
                raise
        return wrapper
    return decorator
```

4. Create `tracing/exporters.py` with Jaeger, Console, and Custom HTTP exporters

### Task 6.2: Structured Logging & Query Replay (Day 3)

1. Create `logging/structured-logger.py`:
```python
class StructuredLogger:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)

    def log(
        self,
        level: str,
        message: str,
        trace_id: Optional[str] = None,
        component: Optional[str] = None,
        **extra
    ):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": self.service_name,
            "message": message,
            "trace_id": trace_id or self._get_current_trace_id(),
            "component": component,
            **extra
        }
        self.logger.log(getattr(logging, level.upper()), json.dumps(record))
```

2. Create `logging/query-replay.py`:
```python
class QueryReplayStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    async def save_query(
        self,
        trace_id: str,
        query: str,
        context: Dict[str, Any],
        result: Any
    ):
        record = {
            "trace_id": trace_id,
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "context": context,
            "result": result
        }
        # Save to file or DB
        ...

    async def replay(self, trace_id: str) -> Dict:
        """Load and re-execute a saved query"""
        record = await self._load_record(trace_id)
        # Re-execute with same context
        ...
```

3. Create `logging/error-tracker.py` with exception grouping and Sentry integration

### Task 6.3: Metrics Collection (Day 4)

1. Create `metrics/collectors.py`:
```python
class MetricsCollector:
    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1, tags: Optional[Dict] = None):
        key = self._make_key(name, tags)
        self.counters[key] += value

    def gauge(self, name: str, value: float, tags: Optional[Dict] = None):
        key = self._make_key(name, tags)
        self.gauges[key] = value

    def histogram(self, name: str, value: float, tags: Optional[Dict] = None):
        key = self._make_key(name, tags)
        self.histograms[key].append(value)

# Predefined metrics
metrics = MetricsCollector()

# Usage
metrics.increment("queries_total", tags={"controller": "orchestrated"})
metrics.histogram("query_latency_ms", 1234, tags={"pipeline": "multi_hop"})
metrics.gauge("cache_hit_rate", 0.65)
```

2. Create `metrics/storage.py` for time-series storage (InfluxDB or PostgreSQL)
3. Create `metrics/aggregators.py` for percentiles, averages, rates

### Task 6.4: Monitoring Dashboard (Day 5)

1. Create `dashboard/metrics-api.py`:
```python
@router.get("/metrics/realtime")
async def get_realtime_metrics():
    return {
        "queries_per_second": metrics.get_rate("queries_total", window=60),
        "avg_latency_ms": metrics.get_average("query_latency_ms", window=300),
        "error_rate": metrics.get_rate("errors_total", window=300) / metrics.get_rate("queries_total", window=300),
        "hallucination_rate": metrics.get_average("hallucination_score", window=3600),
        "cache_hit_rate": metrics.get_gauge("cache_hit_rate")
    }

@router.get("/metrics/traces/{trace_id}")
async def get_trace(trace_id: str):
    return tracer.get_trace(trace_id)
```

2. Create `dashboard/alerting.py`:
```python
class AlertManager:
    def __init__(self, config: AlertConfig):
        self.rules = config.rules
        self.notifiers = config.notifiers

    async def check_alerts(self):
        for rule in self.rules:
            value = await metrics.get_metric(rule.metric)
            if rule.evaluate(value):
                await self.notify(rule, value)

# Example rules
rules = [
    AlertRule("high_error_rate", "error_rate > 0.05", severity="critical"),
    AlertRule("high_latency", "p95_latency_ms > 3000", severity="warning"),
    AlertRule("hallucination_spike", "hallucination_rate > 0.1", severity="critical"),
]
```

3. Create React dashboard components (if extending frontend)

## Todo List

- [ ] Create observability module structure
- [ ] Implement SpanContext
- [ ] Implement QueryTracer
- [ ] Implement @traced decorator
- [ ] Implement span exporters (Console, Jaeger, HTTP)
- [ ] Implement StructuredLogger
- [ ] Implement QueryReplayStore
- [ ] Implement ErrorTracker
- [ ] Implement MetricsCollector
- [ ] Implement metrics storage
- [ ] Implement metrics aggregators
- [ ] Create metrics API endpoints
- [ ] Implement AlertManager
- [ ] Integrate tracing with pipeline executor
- [ ] Write unit tests
- [ ] Create monitoring dashboard (optional)

## Success Criteria

- All pipeline steps traced with < 10ms overhead
- Logs in structured JSON format with trace IDs
- Query replay works for debugging
- Metrics available via API
- Alerts trigger correctly on threshold breaches

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tracing performance overhead | Medium | Medium | Sampling, async export |
| Log volume too high | Medium | Low | Log levels, rotation |
| Metrics storage growth | Medium | Medium | Retention policies, aggregation |

## Security Considerations

- Don't log sensitive query content in production
- Secure metrics API with authentication
- Mask PII in traces and logs

## Next Steps

After Phase 6:
- Phase 7 (Evaluation) uses metrics for benchmarking
- All subsequent phases integrate with tracing
- Dashboard informs optimization decisions
