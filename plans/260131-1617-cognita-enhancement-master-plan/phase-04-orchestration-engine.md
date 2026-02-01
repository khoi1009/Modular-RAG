# Phase 4: Orchestration Engine

**Duration:** Weeks 6-7 | **Priority:** P1 | **Status:** pending

## Context Links

- [Parent Plan](plan.md)
- [Phase 2: Query Intelligence](phase-02-query-intelligence-layer.md)
- [Phase 3: Advanced Retrieval](phase-03-advanced-retrieval-layer.md)
- [Architecture Report](research/researcher-architecture-report.md)

## Overview

Build the "Brain" of the enhanced Cognita: adaptive orchestration layer with intelligent query routing, conditional pipeline execution, and multi-hop reasoning coordination. This is the central component that ties together query intelligence and advanced retrieval.

## Key Insights

From research:
- Query routing: Rule-based (fast) → ML-based (learned) → LLM-based (flexible)
- Pipeline execution: YAML-defined, conditional branching, parallel execution
- Adaptive retrieval: Decide when/how much to retrieve based on query complexity

Current state:
- Query controllers are static, single-pipeline
- No routing logic between controllers
- No conditional execution within pipelines

## Requirements

### Functional
- Query Router: Route queries to appropriate pipelines based on analysis
- Pipeline Executor: Execute YAML-defined pipelines with conditionals
- Orchestrated Controller: Main entry point coordinating all layers
- Multi-hop reasoning support for complex queries

### Non-Functional
- Routing decision < 100ms
- Pipeline execution < 3s total
- Graceful degradation on component failure
- Cost tracking for LLM API calls
- Timeout handling per step

## Architecture

### Module Structure
```
backend/modules/
├── orchestration/
│   ├── __init__.py
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── base-router.py
│   │   ├── rule-based-router.py
│   │   ├── ml-based-router.py
│   │   ├── llm-based-router.py
│   │   └── schemas.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline-executor.py
│   │   ├── condition-evaluator.py
│   │   ├── step-registry.py
│   │   └── schemas.py
│   └── schemas.py
└── query_controllers/
    └── orchestrated/
        ├── __init__.py
        ├── orchestrated-controller.py
        └── schemas.py

config/
├── routing-rules.yaml
└── pipelines/
    ├── simple-retrieval.yaml
    ├── multi-hop-reasoning.yaml
    ├── hybrid-search.yaml
    └── self-reflective.yaml
```

### Orchestration Flow
```
User Query
    ↓
Query Analyzer (Phase 2)
    ↓
Query Router → RoutingDecision
    |           - controller_name
    |           - retrieval_strategy
    |           - preprocessing_steps
    |           - confidence
    ↓
Pipeline Executor
    ├── Load pipeline definition (YAML)
    ├── Evaluate conditions
    ├── Execute steps (sequential/parallel)
    └── Handle errors/retries
    ↓
Answer Verifier (Phase 5)
    ↓
Query Monitor (Phase 6)
    ↓
Response
```

## Related Code Files

### Files to Reference
- `backend/modules/query_controllers/base.py` - Extend patterns
- `backend/modules/query_controllers/example/controller.py` - Controller template
- `backend/server/decorator.py` - `@query_controller` decorator

### Files to Create
- `backend/modules/orchestration/__init__.py`
- `backend/modules/orchestration/schemas.py`
- `backend/modules/orchestration/routing/__init__.py`
- `backend/modules/orchestration/routing/base-router.py`
- `backend/modules/orchestration/routing/rule-based-router.py`
- `backend/modules/orchestration/routing/ml-based-router.py`
- `backend/modules/orchestration/routing/llm-based-router.py`
- `backend/modules/orchestration/routing/schemas.py`
- `backend/modules/orchestration/pipeline/__init__.py`
- `backend/modules/orchestration/pipeline/pipeline-executor.py`
- `backend/modules/orchestration/pipeline/condition-evaluator.py`
- `backend/modules/orchestration/pipeline/step-registry.py`
- `backend/modules/orchestration/pipeline/schemas.py`
- `backend/modules/query_controllers/orchestrated/__init__.py`
- `backend/modules/query_controllers/orchestrated/orchestrated-controller.py`
- `backend/modules/query_controllers/orchestrated/schemas.py`
- `config/routing-rules.yaml`
- `config/pipelines/simple-retrieval.yaml`
- `config/pipelines/multi-hop-reasoning.yaml`
- `config/pipelines/hybrid-search.yaml`
- `tests/modules/orchestration/test_routing.py`
- `tests/modules/orchestration/test_pipeline.py`

## Implementation Steps

### Task 4.1: Query Router (Days 1-4)

1. Create `routing/schemas.py`:
```python
class RoutingDecision(ConfiguredBaseModel):
    controller_name: str
    retrieval_strategy: str  # vectorstore, hybrid, reflective
    preprocessing_steps: List[str]  # hyde, decomposition, etc.
    use_reranking: bool = True
    max_iterations: int = 1
    fallback_strategy: Optional[str] = None
    confidence: float
    reasoning: str

class RoutingRule(ConfiguredBaseModel):
    name: str
    conditions: List[Dict[str, Any]]  # Query metadata conditions
    action: RoutingDecision
    priority: int = 0
```

2. Create `routing/base-router.py`:
```python
class BaseQueryRouter(ABC):
    @abstractmethod
    async def route(
        self, query: str, query_metadata: QueryMetadata, context: Optional[Dict] = None
    ) -> RoutingDecision:
        pass
```

3. Implement `rule-based-router.py`:
```python
class RuleBasedRouter(BaseQueryRouter):
    def __init__(self, rules_path: str):
        self.rules = self._load_rules(rules_path)

    async def route(self, query: str, query_metadata: QueryMetadata, context=None) -> RoutingDecision:
        for rule in sorted(self.rules, key=lambda r: -r.priority):
            if self._matches_conditions(query_metadata, rule.conditions):
                return rule.action
        return self._default_routing()
```

4. Implement `ml-based-router.py` using sklearn classifier
5. Implement `llm-based-router.py` using Ollama for complex cases

### Task 4.2: Pipeline Executor (Days 5-8)

1. Create `pipeline/schemas.py`:
```python
class PipelineStep(ConfiguredBaseModel):
    name: str
    module: str  # e.g., "query_rewriting.hyde"
    input: Optional[str] = None  # Context key for input
    output: str  # Context key for output
    condition: Optional[str] = None  # Expression to evaluate
    parallel: bool = False
    timeout_sec: int = 30
    retry_count: int = 0

class PipelineDefinition(ConfiguredBaseModel):
    name: str
    description: Optional[str] = None
    steps: List[PipelineStep]
    default_config: Dict[str, Any] = {}

class PipelineResult(ConfiguredBaseModel):
    success: bool
    answer: Optional[str] = None
    sources: List[Document] = []
    context: Dict[str, Any] = {}
    execution_time_ms: int
    steps_executed: List[str]
    errors: List[str] = []
```

2. Create `pipeline/step-registry.py`:
```python
class StepRegistry:
    _steps: Dict[str, Callable] = {}

    @classmethod
    def register(cls, module_path: str):
        def decorator(fn):
            cls._steps[module_path] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, module_path: str) -> Callable:
        if module_path not in cls._steps:
            raise ValueError(f"Unknown step: {module_path}")
        return cls._steps[module_path]
```

3. Create `pipeline/condition-evaluator.py`:
```python
class ConditionEvaluator:
    def evaluate(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition expression against context.
        Example: "query_metadata.complexity_score > 0.7"
        """
        # Safe expression evaluation
        ...
```

4. Create `pipeline/pipeline-executor.py`:
```python
class PipelineExecutor:
    def __init__(self, step_registry: StepRegistry, condition_evaluator: ConditionEvaluator):
        self.registry = step_registry
        self.evaluator = condition_evaluator

    async def execute(
        self, pipeline_def: PipelineDefinition, initial_context: Dict[str, Any]
    ) -> PipelineResult:
        context = {**initial_context}
        steps_executed = []
        start_time = time.time()

        for step in pipeline_def.steps:
            # Check condition
            if step.condition and not self.evaluator.evaluate(step.condition, context):
                continue

            # Get step function
            step_fn = self.registry.get(step.module)

            # Prepare input
            input_data = context.get(step.input) if step.input else context

            # Execute with timeout
            try:
                async with async_timeout.timeout(step.timeout_sec):
                    if step.parallel:
                        result = await self._execute_parallel(step_fn, input_data)
                    else:
                        result = await step_fn(input_data)

                context[step.output] = result
                steps_executed.append(step.name)
            except asyncio.TimeoutError:
                # Handle timeout
                ...

        return PipelineResult(
            success=True,
            answer=context.get("answer"),
            sources=context.get("sources", []),
            context=context,
            execution_time_ms=int((time.time() - start_time) * 1000),
            steps_executed=steps_executed
        )
```

### Task 4.3: Orchestrated Controller (Days 9-10)

1. Create `orchestrated-controller.py`:
```python
@query_controller("/orchestrated")
class OrchestratedQueryController(BaseQueryController):
    def __init__(self):
        self.query_analyzer = LLMQueryAnalyzer(config)
        self.router = RuleBasedRouter("config/routing-rules.yaml")
        self.pipeline_executor = PipelineExecutor(step_registry, condition_evaluator)

    @post("/answer")
    async def answer(self, request: OrchestratedQueryInput) -> StreamingResponse:
        # 1. Analyze query
        analysis = await self.query_analyzer.analyze(request.query)

        # 2. Route to pipeline
        routing = await self.router.route(request.query, analysis)

        # 3. Load pipeline
        pipeline = self._load_pipeline(routing.controller_name)

        # 4. Execute pipeline
        result = await self.pipeline_executor.execute(
            pipeline,
            {
                "query": request.query,
                "collection_name": request.collection_name,
                "query_metadata": analysis,
                "routing": routing,
            }
        )

        # 5. Return response
        if request.stream:
            return StreamingResponse(
                self._sse_wrap(self._stream_result(result)),
                media_type="text/event-stream"
            )
        return result.model_dump()
```

2. Create pipeline YAML files:
```yaml
# config/pipelines/multi-hop-reasoning.yaml
name: "multi_hop_reasoning"
description: "Pipeline for complex multi-hop queries"
steps:
  - name: "analyze_query"
    module: "query_analysis.llm_analyzer"
    output: "query_metadata"

  - name: "decompose"
    module: "query_rewriting.decomposition"
    condition: "query_metadata.complexity_score > 0.7"
    output: "sub_queries"

  - name: "retrieve_parallel"
    module: "retrievers.hybrid.vector_bm25"
    input: "sub_queries"
    parallel: true
    output: "documents"

  - name: "rerank"
    module: "rerankers.advanced.multi_stage"
    input: "documents"
    output: "ranked_docs"

  - name: "generate"
    module: "generation.lcel_chain"
    output: "answer"
```

## Todo List

- [ ] Create orchestration module structure
- [ ] Implement RoutingDecision schema
- [ ] Implement BaseQueryRouter
- [ ] Implement RuleBasedRouter
- [ ] Implement MLBasedRouter
- [ ] Implement LLMBasedRouter
- [ ] Create routing-rules.yaml
- [ ] Implement PipelineStep schema
- [ ] Implement StepRegistry
- [ ] Implement ConditionEvaluator
- [ ] Implement PipelineExecutor
- [ ] Create OrchestratedQueryController
- [ ] Create simple-retrieval.yaml pipeline
- [ ] Create multi-hop-reasoning.yaml pipeline
- [ ] Create hybrid-search.yaml pipeline
- [ ] Register steps from Phase 2 & 3 modules
- [ ] Write routing tests
- [ ] Write pipeline execution tests
- [ ] Integration test with full flow

## Success Criteria

- Router correctly classifies 90%+ of queries
- Pipeline executor handles all defined pipelines
- Graceful fallback on step failures
- Cost tracking accurate within 5%
- Latency targets met for each pipeline type

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Routing misclassification | Medium | Medium | Fallback to simple pipeline |
| Pipeline step timeout | Medium | High | Step-level timeouts, skip non-critical |
| Circular dependencies | Low | High | DAG validation at load time |
| Config parsing errors | Low | Medium | Schema validation, error messages |

## Security Considerations

- Validate pipeline YAML against schema (prevent injection)
- Sanitize condition expressions (no arbitrary code execution)
- Rate limit routing decisions per user
- Audit log for routing decisions

## Next Steps

After Phase 4:
- Phase 5 (Verification) adds verification step to pipelines
- Phase 6 (Observability) adds tracing/metrics to executor
- Phase 8 (Domain Extensions) adds specialized controllers
