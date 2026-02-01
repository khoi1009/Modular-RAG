Cognita Enhancement Master Plan
Transform Cognita into a World-Class Production-Ready Modular RAG Framework
Version: 1.0
Target Timeline: 12 Weeks
Focus: Building missing orchestration, advanced retrieval, and production monitoring capabilities

---

🎯 Strategic Vision
Transform Cognita from a "modular RAG infrastructure" into a "full Modular RAG system" by adding:

1. Advanced Orchestration Layer - Adaptive retrieval, query routing, conditional pipelines
2. Query Intelligence - Rewriting, decomposition, expansion (HyDE, Step-back, etc.)
3. Verification & Quality Control - Answer validation, hallucination detection, confidence scoring
4. Production Monitoring - Observability, evaluation, feedback loops
5. Domain-Specific Extensions - Water infrastructure intelligence for SEQ Water

---

📊 Current State Assessment (What Cognita Has)
✅ Strengths
• Clean modular architecture (loaders, parsers, embedders, retrievers)
• Production infrastructure (FastAPI, Docker, UI)
• Incremental indexing (critical for production)
• Vector DB abstraction (Qdrant, etc.)
• Basic reranking support
• Metadata management
⚠️ Gaps (What We'll Build)
• No adaptive retrieval orchestration
• No query preprocessing/rewriting modules
• No verification/hallucination detection
• Limited routing logic
• No observability/monitoring
• No evaluation framework
• No domain-specific controllers

---

🏗️ Architecture Enhancement Blueprint
┌─────────────────────────────────────────────────────────────┐
│ ENHANCED COGNITA STACK │
├─────────────────────────────────────────────────────────────┤
│ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ NEW: ORCHESTRATION LAYER │ │
│ │ - Adaptive Retrieval Engine │ │
│ │ - Query Router │ │
│ │ - Conditional Pipeline Executor │ │
│ │ - Multi-hop Reasoning Controller │ │
│ └─────────────────────────────────────────────────────┘ │
│ ↓ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ NEW: QUERY INTELLIGENCE LAYER │ │
│ │ - Query Rewriter (HyDE, Step-back, Decomposition) │ │
│ │ - Query Analyzer & Classifier │ │
│ │ - Context Enhancer │ │
│ └─────────────────────────────────────────────────────┘ │
│ ↓ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ EXISTING: RETRIEVAL LAYER (Enhanced) │ │
│ │ - Vector Retrievers ✓ │ │
│ │ - Rerankers ✓ │ │
│ │ - NEW: Hybrid Retrievers (Vector + BM25 + SQL) │ │
│ │ - NEW: Self-Reflective Retriever │ │
│ └─────────────────────────────────────────────────────┘ │
│ ↓ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ NEW: VERIFICATION & QC LAYER │ │
│ │ - Answer Validator │ │
│ │ - Hallucination Detector │ │
│ │ - Confidence Scorer │ │
│ │ - Source Attribution Verifier │ │
│ └─────────────────────────────────────────────────────┘ │
│ ↓ │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ NEW: OBSERVABILITY & EVALUATION │ │
│ │ - Query Tracing │ │
│ │ - Performance Metrics │ │
│ │ - A/B Testing Framework │ │
│ │ - Feedback Collection │ │
│ └─────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────┘

---

📋 PHASE 1: Deep Code Inspection & Architecture Mapping
Duration: Week 1
Goal: Understand every component of Cognita to build on solid foundation
Task 1.1: Core Architecture Analysis
Claude Code Prompt:
Please analyze the Cognita repository structure and create a comprehensive architecture map:

1. Examine `/backend/modules/` and document:
   - All existing query_controllers and their responsibilities
   - Data loader implementations and extension points
   - Parser modules and chunking strategies
   - Embedder integrations
   - Vector DB abstraction layer
   - Reranker implementations

2. Create a dependency graph showing:
   - How components interact
   - Data flow from ingestion to query response
   - Extension points and plugin interfaces

3. Document the API layer:
   - FastAPI route structure
   - Request/response schemas
   - Authentication/authorization flow

4. Analyze configuration system:
   - How components are configured
   - Environment variables
   - Runtime configuration options

Output:

- `/docs/architecture_analysis.md` with detailed findings
- `/docs/architecture_diagram.mermaid` visual representation
- `/docs/extension_points.md` listing all places we can inject new logic
  Task 1.2: Identify Integration Patterns
  Claude Code Prompt:
  Study how Cognita currently integrates components:

1. Examine existing query_controllers:
   - How do they orchestrate retrieval?
   - What's the pattern for adding new controllers?
   - How do they handle errors and fallbacks?

2. Study the retriever pattern:
   - Base classes and interfaces
   - How retrievers are registered and discovered
   - Configuration and initialization flow

3. Analyze the embedder integration:
   - How embedders are swapped
   - Batching and performance optimization
   - Error handling

Output:

- `/docs/integration_patterns.md` with code examples
- `/docs/best_practices.md` from current codebase
  Task 1.3: Gap Analysis Report
  Claude Code Prompt:
  Create a detailed gap analysis comparing Cognita's current capabilities against the Modular RAG research paradigm:

1. Map current features to Modular RAG components:
   - Indexing Module ✓
   - Pre-Retrieval Module (Query Processing) ❌
   - Retrieval Module ✓ (basic)
   - Post-Retrieval Module (Reranking) ⚠️ (partial)
   - Generation Module ✓
   - Orchestration Module ❌

2. For each missing/partial component:
   - Describe what's missing
   - Specify where it should be added in the codebase
   - Estimate complexity (Low/Medium/High)
   - List dependencies

Output:

- `/docs/gap_analysis.md`
- `/docs/implementation_priority.md` (prioritized by impact and complexity)

---

📋 PHASE 2: Query Intelligence Layer
Duration: Weeks 2-3
Goal: Add advanced query processing capabilities
Task 2.1: Query Analyzer Module
Claude Code Prompt:
Create a comprehensive Query Analyzer module at `/backend/modules/query_analysis/`:

Requirements:

1. Query Classifier:
   - Detect query type: factual, comparison, temporal, spatial, analytical
   - Identify complexity: simple, multi-hop, compositional
   - Extract intent: retrieval-only, reasoning-required, verification-needed

2. Query Metadata Extractor:
   - Named entities (dates, locations, asset IDs for water infrastructure)
   - Temporal constraints (date ranges)
   - Spatial constraints (geographic bounds)
   - Domain-specific patterns (pipe IDs, maintenance codes)

3. Query Complexity Scorer:
   - Estimate retrieval difficulty
   - Predict required context length
   - Suggest optimal retrieval strategy

Implementation:

- Base class: `BaseQueryAnalyzer`
- Default implementation using LLM (GPT-4/Claude)
- Fast implementation using lightweight classifiers
- Domain-specific analyzers (e.g., `WaterInfrastructureQueryAnalyzer`)

Files to create:

- `/backend/modules/query_analysis/__init__.py`
- `/backend/modules/query_analysis/base.py`
- `/backend/modules/query_analysis/llm_analyzer.py`
- `/backend/modules/query_analysis/fast_analyzer.py`
- `/backend/modules/query_analysis/schemas.py`
- `/tests/modules/query_analysis/test_analyzers.py`
  Task 2.2: Query Rewriting Module
  Claude Code Prompt:
  Implement advanced query rewriting techniques at `/backend/modules/query_rewriting/`:

Techniques to implement:

1. HyDE (Hypothetical Document Embeddings):
   - Generate hypothetical answer
   - Use answer for retrieval instead of question
   - Configurable LLM backend

2. Query2Doc:
   - Generate pseudo-document from query
   - Combine with original query for hybrid search

3. Step-Back Prompting:
   - Abstract query to higher-level concepts
   - Retrieve on both original and abstracted query

4. Query Decomposition:
   - Break complex queries into sub-queries
   - Handle temporal, compositional, and comparative queries
   - Return structured decomposition plan

5. Multi-Query Generation:
   - Generate multiple perspectives of same question
   - Retrieve using all variants
   - Merge results with deduplication

Implementation Structure:

```python
# Base rewriter interface
class BaseQueryRewriter(ABC):
    @abstractmethod
    async def rewrite(self, query: str, context: dict) -> RewriteResult:
        pass

# Specific rewriters
class HyDERewriter(BaseQueryRewriter):
    def __init__(self, llm_client, config):
        ...

    async def rewrite(self, query: str, context: dict) -> RewriteResult:
        # Generate hypothetical document
        ...

# Rewriter registry and factory
class QueryRewriterFactory:
    @staticmethod
    def create(rewriter_type: str, config: dict):
        ...
Files to create:
•	/backend/modules/query_rewriting/__init__.py
•	/backend/modules/query_rewriting/base.py
•	/backend/modules/query_rewriting/hyde.py
•	/backend/modules/query_rewriting/query2doc.py
•	/backend/modules/query_rewriting/stepback.py
•	/backend/modules/query_rewriting/decomposition.py
•	/backend/modules/query_rewriting/multi_query.py
•	/backend/modules/query_rewriting/factory.py
•	/backend/modules/query_rewriting/schemas.py
•	/tests/modules/query_rewriting/test_rewriters.py
•	/docs/query_rewriting_guide.md

### Task 2.3: Context Enhancement Module
**Claude Code Prompt:**
Build a context enhancement module to improve retrieval quality:
Features:
1.	Query Expansion:
o	Synonym expansion
o	Domain-specific term expansion
o	Acronym expansion (critical for technical domains)
2.	Context Injection:
o	Add user profile context
o	Add session history context
o	Add domain knowledge context
3.	Constraint Extraction:
o	Time constraints
o	Geographic constraints
o	Domain-specific filters
Implementation:
•	Integrate with existing metadata store
•	Support pluggable expansion dictionaries
•	LLM-based expansion for complex domains
Files:
•	/backend/modules/context_enhancement/__init__.py
•	/backend/modules/context_enhancement/expander.py
•	/backend/modules/context_enhancement/constraint_extractor.py
•	/backend/modules/context_enhancement/context_injector.py

---

## 📋 PHASE 3: Advanced Retrieval Layer
**Duration:** Weeks 4-5
**Goal:** Implement hybrid and self-reflective retrieval

### Task 3.1: Hybrid Retrieval System
**Claude Code Prompt:**
Extend Cognita's retrieval layer with hybrid capabilities at /backend/modules/retrievers/hybrid/:
Implement:
1.	Hybrid Vector + BM25 Retriever:
o	Combine dense (vector) and sparse (BM25) retrieval
o	Configurable fusion strategies: RRF, weighted, learned
o	Support for existing vector DB backends
2.	Hybrid Vector + SQL Retriever:
o	For structured + unstructured data
o	Query router decides which to use
o	Metadata filtering before vector search
3.	Multi-Stage Retrieval:
o	Stage 1: Fast, broad retrieval (BM25 or approximate vector)
o	Stage 2: Precise reranking (cross-encoder)
o	Stage 3: Optional LLM-based reranking
4.	Fusion Strategies:
o	Reciprocal Rank Fusion (RRF)
o	Weighted score combination
o	Learned fusion (simple logistic regression)
Integration:
•	Must work with existing vector DB abstraction
•	Add BM25 index alongside vector index
•	Support incremental indexing for BM25
Files:
•	/backend/modules/retrievers/hybrid/__init__.py
•	/backend/modules/retrievers/hybrid/vector_bm25.py
•	/backend/modules/retrievers/hybrid/vector_sql.py
•	/backend/modules/retrievers/hybrid/multi_stage.py
•	/backend/modules/retrievers/hybrid/fusion.py
•	/backend/modules/retrievers/hybrid/schemas.py
•	/tests/modules/retrievers/hybrid/test_hybrid_retrievers.py

### Task 3.2: Self-Reflective Retrieval
**Claude Code Prompt:**
Implement self-reflective retrieval patterns at /backend/modules/retrievers/reflective/:
Capabilities:
1.	Retrieval with Feedback:
o	Initial retrieval
o	LLM evaluates quality of results
o	If insufficient, reformulate and retrieve again
o	Maximum iteration limit (e.g., 3)
2.	Corrective RAG (CRAG):
o	Retrieve documents
o	Grade relevance
o	If low relevance: trigger web search or query rewrite
o	If high relevance: proceed to generation
3.	Self-RAG:
o	Retrieve multiple document sets
o	Generate candidate answers
o	Self-critique and select best answer
o	Include reflection tokens
4.	Adaptive Retrieval:
o	Decide when to retrieve (not always needed)
o	Decide how many documents to retrieve
o	Adjust based on query complexity
Implementation pattern:
class SelfReflectiveRetriever(BaseRetriever):
    def __init__(self, base_retriever, evaluator, max_iterations=3):
        self.base_retriever = base_retriever
        self.evaluator = evaluator  # LLM-based or heuristic
        self.max_iterations = max_iterations

    async def retrieve(self, query: str) -> RetrievalResult:
        for i in range(self.max_iterations):
            docs = await self.base_retriever.retrieve(query)
            quality_score = await self.evaluator.evaluate(query, docs)

            if quality_score > threshold:
                return docs
            else:
                query = await self.reformulate(query, docs)

        return docs  # Return best attempt
Files:
•	/backend/modules/retrievers/reflective/__init__.py
•	/backend/modules/retrievers/reflective/feedback_retriever.py
•	/backend/modules/retrievers/reflective/crag.py
•	/backend/modules/retrievers/reflective/self_rag.py
•	/backend/modules/retrievers/reflective/adaptive.py
•	/backend/modules/retrievers/reflective/evaluators.py

### Task 3.3: Advanced Reranking Pipeline
**Claude Code Prompt:**
Enhance the existing reranking capability at /backend/modules/rerankers/advanced/:
Add:
1.	Multi-Stage Reranking:
o	Stage 1: Fast model (e.g., MiniLM cross-encoder)
o	Stage 2: Powerful model (e.g., deberta-v3-large)
o	Configurable cascade strategy
2.	LLM-based Reranking:
o	Use GPT-4/Claude to score relevance
o	Generate explanations for ranking decisions
o	Support batch processing
3.	Diversity-Aware Reranking:
o	MMR (Maximal Marginal Relevance)
o	Ensure diverse perspectives in top results
o	Configurable diversity vs relevance tradeoff
4.	Metadata-Boosted Reranking:
o	Boost scores based on recency, authority, etc.
o	Domain-specific boosting rules
o	Configurable boost factors
Files:
•	/backend/modules/rerankers/advanced/__init__.py
•	/backend/modules/rerankers/advanced/multi_stage.py
•	/backend/modules/rerankers/advanced/llm_reranker.py
•	/backend/modules/rerankers/advanced/diversity.py
•	/backend/modules/rerankers/advanced/metadata_boost.py

---

## 📋 PHASE 4: Orchestration Engine (The Brain)
**Duration:** Weeks 6-7
**Goal:** Build the adaptive orchestration layer

### Task 4.1: Query Router
**Claude Code Prompt:**
Create an intelligent query router at /backend/modules/orchestration/routing/:
Router Types:
1.	Rule-Based Router:
o	Pattern matching on query
o	Keyword-based routing
o	Fast and deterministic
2.	ML-Based Router:
o	Classify query into categories
o	Route to specialized controllers
o	Train on historical query data
3.	LLM-Based Router:
o	Use GPT/Claude to decide routing
o	Complex reasoning for edge cases
o	Slower but more flexible
Router Destinations:
•	Simple retrieval (for straightforward questions)
•	Multi-hop reasoning (for complex questions)
•	Hybrid retrieval (for structured+unstructured)
•	External tools (calculator, SQL, web search)
•	Human escalation (for ambiguous queries)
Implementation:
class QueryRouter:
    def __init__(self, routing_strategy: str, config: dict):
        self.strategy = self._load_strategy(routing_strategy)

    async def route(self, query: str, context: dict) -> RoutingDecision:
        """
        Returns:
        - controller_name: str
        - retrieval_strategy: str
        - preprocessing_steps: List[str]
        - confidence: float
        """
        return await self.strategy.decide(query, context)

class RoutingDecision(BaseModel):
    controller_name: str
    retrieval_strategy: str
    preprocessing_steps: List[str]
    use_reranking: bool
    max_iterations: int
    fallback_strategy: Optional[str]
    confidence: float
    reasoning: str  # Explain why this route was chosen
Files:
•	/backend/modules/orchestration/routing/__init__.py
•	/backend/modules/orchestration/routing/base.py
•	/backend/modules/orchestration/routing/rule_based.py
•	/backend/modules/orchestration/routing/ml_based.py
•	/backend/modules/orchestration/routing/llm_based.py
•	/backend/modules/orchestration/routing/schemas.py
•	/config/routing_rules.yaml (for rule-based router)

### Task 4.2: Adaptive Pipeline Executor
**Claude Code Prompt:**
Build a flexible pipeline executor at /backend/modules/orchestration/pipeline/:
Features:
1.	Conditional Execution:
o	Execute steps based on intermediate results
o	Skip unnecessary steps
o	Early termination if answer found
2.	Iterative Refinement:
o	Loop until quality threshold met
o	Maximum iteration safety limit
o	Track improvement metrics
3.	Parallel Execution:
o	Run independent steps concurrently
o	Merge results intelligently
o	Handle failures gracefully
4.	Pipeline Definition DSL:
o	YAML-based pipeline definitions
o	Conditional branching
o	Error handling and retries
Example Pipeline Definition:
name: "multi_hop_reasoning"
steps:
  - name: "analyze_query"
    module: "query_analysis.llm_analyzer"
    output: "query_metadata"

  - name: "decompose"
    module: "query_rewriting.decomposition"
    condition: "query_metadata.complexity > 0.7"
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

  - name: "verify"
    module: "verification.hallucination_detector"
    input: "generated_answer"
    condition: "query_metadata.requires_verification"
    output: "verification_result"
Implementation:
class PipelineExecutor:
    async def execute(self, pipeline_def: dict, query: str) -> PipelineResult:
        context = {"query": query}

        for step in pipeline_def["steps"]:
            # Check condition
            if not self._evaluate_condition(step.get("condition"), context):
                continue

            # Execute step
            module = self._load_module(step["module"])
            input_data = self._get_input(step.get("input"), context)

            if step.get("parallel"):
                result = await self._execute_parallel(module, input_data)
            else:
                result = await module.execute(input_data)

            # Store output
            context[step["output"]] = result

        return PipelineResult(context)
Files:
•	/backend/modules/orchestration/pipeline/__init__.py
•	/backend/modules/orchestration/pipeline/executor.py
•	/backend/modules/orchestration/pipeline/condition_evaluator.py
•	/backend/modules/orchestration/pipeline/schemas.py
•	/config/pipelines/ (directory for pipeline definitions)
•	/config/pipelines/simple_retrieval.yaml
•	/config/pipelines/multi_hop.yaml
•	/config/pipelines/hybrid_search.yaml

### Task 4.3: Orchestration Controller
**Claude Code Prompt:**
Create the main orchestration controller at /backend/modules/query_controllers/orchestrated_controller.py:
This is the "Brain" that coordinates everything:
class OrchestratedQueryController:
    """
    Advanced query controller that orchestrates:
    - Query analysis
    - Routing decisions
    - Pipeline execution
    - Verification
    - Monitoring
    """

    def __init__(self, config: dict):
        self.query_analyzer = QueryAnalyzer(config)
        self.router = QueryRouter(config)
        self.pipeline_executor = PipelineExecutor(config)
        self.verifier = AnswerVerifier(config)
        self.monitor = QueryMonitor(config)

    async def answer_query(self, query: str, collection_name: str) -> QueryResponse:
        # 1. Analyze query
        analysis = await self.query_analyzer.analyze(query)

        # 2. Route to appropriate pipeline
        routing = await self.router.route(query, analysis)

        # 3. Load and execute pipeline
        pipeline = self._load_pipeline(routing.controller_name)
        result = await self.pipeline_executor.execute(pipeline, query)

        # 4. Verify answer quality
        verification = await self.verifier.verify(
            query,
            result.answer,
            result.sources
        )

        # 5. Log metrics
        await self.monitor.log(query, analysis, routing, result, verification)

        # 6. Return structured response
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            confidence=verification.confidence,
            routing_decision=routing,
            verification_results=verification,
            metadata=result.metadata
        )
This controller should support:
•	Multiple pipeline strategies
•	Graceful degradation
•	Timeout handling
•	Cost tracking (LLM API calls)
•	Performance monitoring

---

## 📋 PHASE 5: Verification & Quality Control
**Duration:** Week 8
**Goal:** Ensure answer quality and detect hallucinations

### Task 5.1: Answer Verification Module
**Claude Code Prompt:**
Create comprehensive answer verification at /backend/modules/verification/:
Implement:
1.	Hallucination Detection:
o	Check if answer is grounded in retrieved documents
o	Use NLI (Natural Language Inference) models
o	LLM-based verification as fallback
o	Generate attribution scores per sentence
2.	Source Attribution:
o	Map each claim in answer to source document
o	Identify unsupported claims
o	Generate citation links
3.	Confidence Scoring:
o	Estimate answer confidence based on:
	Retrieval quality scores
	LLM generation confidence
	Source agreement level
	Verification results
4.	Consistency Checking:
o	Check internal consistency of answer
o	Cross-check with knowledge base
o	Flag contradictions
Implementation:
class AnswerVerifier:
    def __init__(self, config: dict):
        self.nli_model = self._load_nli_model(config)
        self.llm_verifier = LLMVerifier(config)

    async def verify(self, query: str, answer: str, sources: List[Document]) -> VerificationResult:
        # Split answer into claims
        claims = self._extract_claims(answer)

        # Verify each claim
        claim_verifications = []
        for claim in claims:
            verification = await self._verify_claim(claim, sources)
            claim_verifications.append(verification)

        # Compute overall scores
        hallucination_score = self._compute_hallucination_score(claim_verifications)
        confidence = self._compute_confidence(claim_verifications)

        return VerificationResult(
            is_grounded=hallucination_score < threshold,
            hallucination_score=hallucination_score,
            confidence=confidence,
            claim_verifications=claim_verifications,
            unsupported_claims=[c for c in claim_verifications if not c.is_supported]
        )
Files:
•	/backend/modules/verification/__init__.py
•	/backend/modules/verification/hallucination_detector.py
•	/backend/modules/verification/source_attributor.py
•	/backend/modules/verification/confidence_scorer.py
•	/backend/modules/verification/consistency_checker.py
•	/backend/modules/verification/schemas.py
•	/models/nli/ (directory for NLI model weights)

### Task 5.2: Quality Metrics Module
**Claude Code Prompt:**
Build quality metrics collection at /backend/modules/metrics/:
Track:
1.	Retrieval Metrics:
o	Precision@K, Recall@K
o	MRR (Mean Reciprocal Rank)
o	NDCG (Normalized Discounted Cumulative Gain)
o	Retrieval latency
2.	Generation Metrics:
o	Answer length
o	Generation latency
o	Token usage
o	Perplexity (if available)
3.	End-to-End Metrics:
o	Total query latency
o	Cost per query
o	User satisfaction (from feedback)
o	Answer acceptance rate
4.	Quality Metrics:
o	Hallucination rate
o	Source attribution coverage
o	Confidence distribution
o	Consistency score
Storage:
•	Time-series database for metrics
•	Dashboard for visualization
•	Alerting for anomalies
Files:
•	/backend/modules/metrics/__init__.py
•	/backend/modules/metrics/retrieval_metrics.py
•	/backend/modules/metrics/generation_metrics.py
•	/backend/modules/metrics/quality_metrics.py
•	/backend/modules/metrics/storage.py

---

## 📋 PHASE 6: Observability & Monitoring
**Duration:** Week 9
**Goal:** Production-grade monitoring and debugging

### Task 6.1: Query Tracing System
**Claude Code Prompt:**
Implement distributed tracing at /backend/modules/observability/tracing/:
Features:
1.	Trace every query through the system:
o	Query reception
o	Analysis steps
o	Retrieval operations
o	Reranking
o	Generation
o	Verification
o	Response delivery
2.	Capture for each step:
o	Timestamp (start/end)
o	Latency
o	Input/output sizes
o	Cost (API calls)
o	Errors/warnings
o	Model used
3.	Integration:
o	OpenTelemetry compatible
o	Export to Jaeger, DataDog, or custom backend
o	Correlation IDs across services
Example trace:
{
  "trace_id": "abc123",
  "query": "What caused the pipe failure on 2024-01-15?",
  "spans": [
    {
      "name": "query_analysis",
      "start": "2024-01-31T10:00:00Z",
      "duration_ms": 145,
      "output": {"complexity": 0.6, "type": "causal"}
    },
    {
      "name": "hybrid_retrieval",
      "start": "2024-01-31T10:00:00.145Z",
      "duration_ms": 523,
      "metadata": {"num_docs": 50, "strategy": "vector_bm25"}
    },
    ...
  ]
}
Files:
•	/backend/modules/observability/tracing/__init__.py
•	/backend/modules/observability/tracing/tracer.py
•	/backend/modules/observability/tracing/exporters.py
•	/backend/modules/observability/tracing/decorators.py

### Task 6.2: Logging & Debugging
**Claude Code Prompt:**
Enhance logging infrastructure at /backend/modules/observability/logging/:
Requirements:
1.	Structured Logging:
o	JSON format
o	Include trace IDs
o	Severity levels
o	Component tags
2.	Query Replay:
o	Save all query inputs
o	Enable replay for debugging
o	Diff results across versions
3.	Error Tracking:
o	Capture exceptions with full context
o	Group similar errors
o	Integration with Sentry/similar
4.	Debug Mode:
o	Verbose logging of intermediate steps
o	Save intermediate results
o	Performance profiling
Files:
•	/backend/modules/observability/logging/__init__.py
•	/backend/modules/observability/logging/structured_logger.py
•	/backend/modules/observability/logging/query_replay.py
•	/backend/modules/observability/logging/error_tracker.py

### Task 6.3: Monitoring Dashboard
**Claude Code Prompt:**
Create monitoring dashboard at /backend/modules/observability/dashboard/:
Dashboard Components:
1.	Real-time Metrics:
o	Queries per second
o	Average latency
o	Error rate
o	Cost per query
2.	Quality Metrics:
o	Hallucination rate (rolling window)
o	Confidence distribution
o	User feedback scores
o	Source attribution rate
3.	System Health:
o	Vector DB status
o	LLM API status
o	Cache hit rates
o	Resource usage
4.	Alerts:
o	High error rate
o	Increased hallucination
o	Slow queries
o	High cost
Tech Stack:
•	Backend: FastAPI endpoints for metrics
•	Frontend: React dashboard (integrate with existing Cognita UI)
•	Storage: InfluxDB or PostgreSQL for time-series
•	Visualization: Recharts or similar
Files:
•	/backend/modules/observability/dashboard/__init__.py
•	/backend/modules/observability/dashboard/metrics_api.py
•	/backend/modules/observability/dashboard/alerting.py
•	/frontend/src/components/Monitoring/ (UI components)

---

## 📋 PHASE 7: Evaluation Framework
**Duration:** Week 10
**Goal:** Systematic evaluation and improvement

### Task 7.1: Evaluation Dataset Manager
**Claude Code Prompt:**
Create evaluation infrastructure at /backend/modules/evaluation/:
Features:
1.	Dataset Management:
o	Store ground-truth Q&A pairs
o	Import from various formats (JSON, CSV, etc.)
o	Version control for datasets
o	Support for different domains
2.	Automated Evaluation:
o	Run queries through system
o	Compare against ground truth
o	Generate evaluation reports
3.	Metrics Computation:
o	Exact match
o	F1 score
o	BLEU, ROUGE
o	Semantic similarity (BERTScore)
o	LLM-as-Judge
4.	Regression Testing:
o	Run on every code change
o	Track metric trends over time
o	Alert on degradation
Dataset Format:
{
  "datasets": [
    {
      "name": "water_infrastructure_qa",
      "domain": "water_utility",
      "version": "1.0",
      "samples": [
        {
          "id": "q1",
          "query": "What are the common causes of pipe failure?",
          "ground_truth_answer": "...",
          "ground_truth_sources": ["doc1", "doc2"],
          "difficulty": "easy",
          "required_retrieval": true
        }
      ]
    }
  ]
}
Files:
•	/backend/modules/evaluation/__init__.py
•	/backend/modules/evaluation/dataset_manager.py
•	/backend/modules/evaluation/evaluator.py
•	/backend/modules/evaluation/metrics.py
•	/backend/modules/evaluation/llm_judge.py
•	/backend/modules/evaluation/regression_test.py
•	/data/evaluation/ (directory for datasets)

### Task 7.2: A/B Testing Framework
**Claude Code Prompt:**
Implement A/B testing at /backend/modules/evaluation/ab_testing/:
Capabilities:
1.	Variant Management:
o	Define multiple pipeline variants
o	Route percentage of traffic to each
o	Gradual rollout support
2.	Experiment Tracking:
o	Assign users to variants
o	Track metrics per variant
o	Statistical significance testing
3.	Analysis:
o	Compare variants on all metrics
o	Generate comparison reports
o	Recommend winner
Example Experiment:
experiment = ABExperiment(
    name="hybrid_vs_vector_only",
    variants=[
        Variant(
            name="control",
            pipeline="simple_vector_retrieval",
            traffic_percentage=50
        ),
        Variant(
            name="treatment",
            pipeline="hybrid_vector_bm25",
            traffic_percentage=50
        )
    ],
    metrics=["latency", "hallucination_rate", "user_satisfaction"],
    duration_days=7
)
Files:
•	/backend/modules/evaluation/ab_testing/__init__.py
•	/backend/modules/evaluation/ab_testing/experiment.py
•	/backend/modules/evaluation/ab_testing/variant_router.py
•	/backend/modules/evaluation/ab_testing/analysis.py

---

## 📋 PHASE 8: Domain-Specific Extensions (SEQ Water)
**Duration:** Week 11
**Goal:** Build water infrastructure intelligence

### Task 8.1: Water Infrastructure Query Controller
**Claude Code Prompt:**
Create specialized controller at /backend/modules/query_controllers/water_infrastructure_controller.py:
Domain-Specific Features:
1.	Asset Type Detection:
o	Identify if query is about: pipes, pumps, valves, meters, etc.
o	Route to specialized retrievers based on asset type
2.	Temporal Query Handling:
o	Parse date ranges from queries
o	Filter documents by date metadata
o	Handle relative dates ("last month", "Q2 2024")
3.	Spatial Query Processing:
o	Extract geographic constraints
o	Filter by location metadata
o	Support radius queries
4.	Maintenance Code Understanding:
o	Recognize maintenance codes
o	Expand acronyms (specific to water utility)
o	Link to maintenance history
5.	Failure Analysis:
o	Route failure prediction queries to survival analysis module
o	Combine document retrieval with statistical models
o	Generate risk assessments
Implementation:
class WaterInfrastructureQueryController(BaseQueryController):
    async def answer_query(self, query: str, collection: str) -> QueryResponse:
        # 1. Detect domain-specific entities
        entities = await self.extract_water_entities(query)

        # 2. Route based on query type
        if entities.query_type == "failure_prediction":
            return await self.handle_failure_prediction(query, entities)
        elif entities.query_type == "maintenance_history":
            return await self.handle_maintenance_query(query, entities)
        elif entities.query_type == "compliance":
            return await self.handle_compliance_query(query, entities)
        else:
            return await self.handle_general_query(query, entities)

    async def handle_failure_prediction(self, query, entities):
        # Combine document retrieval with ML model
        docs = await self.retrieve_similar_failures(entities.asset_id)
        ml_prediction = await self.survival_model.predict(entities.asset_id)

        # Synthesize answer
        answer = await self.synthesize_failure_analysis(query, docs, ml_prediction)
        return answer
Files:
•	/backend/modules/query_controllers/water_infrastructure_controller.py
•	/backend/modules/domain/water/__init__.py
•	/backend/modules/domain/water/entity_extractor.py
•	/backend/modules/domain/water/asset_router.py
•	/backend/modules/domain/water/temporal_parser.py
•	/backend/modules/domain/water/spatial_filter.py
•	/backend/modules/domain/water/ml_integration.py

### Task 8.2: Water Utility Knowledge Base
**Claude Code Prompt:**
Build domain knowledge base at /backend/modules/domain/water/knowledge_base/:
Components:
1.	Acronym Dictionary:
o	Map water utility acronyms to full terms
o	Context-aware expansion
o	Regular expression patterns
2.	Asset Taxonomy:
o	Hierarchical classification of assets
o	Standardized naming conventions
o	Metadata schema definitions
3.	Maintenance Code Registry:
o	All maintenance codes and meanings
o	Severity classifications
o	Related failure modes
4.	Compliance Rules:
o	Regulatory requirements
o	Inspection schedules
o	Reporting standards
5.	Risk Factors:
o	Known failure predictors
o	Environmental factors
o	Material degradation patterns
Files:
•	/backend/modules/domain/water/knowledge_base/__init__.py
•	/backend/modules/domain/water/knowledge_base/acronyms.py
•	/backend/modules/domain/water/knowledge_base/asset_taxonomy.py
•	/backend/modules/domain/water/knowledge_base/maintenance_codes.py
•	/data/domain/water/acronyms.json
•	/data/domain/water/asset_taxonomy.json
•	/data/domain/water/maintenance_codes.json

---

## 📋 PHASE 9: Performance Optimization
**Duration:** Week 12
**Goal:** Production performance tuning

### Task 9.1: Caching Layer
**Claude Code Prompt:**
Implement intelligent caching at /backend/modules/cache/:
Cache Levels:
1.	Query Cache:
o	Cache exact query matches
o	Cache similar queries (semantic similarity > threshold)
o	TTL-based expiration
2.	Retrieval Cache:
o	Cache retrieval results
o	Invalidate on index updates
o	LRU eviction policy
3.	Embedding Cache:
o	Cache query embeddings
o	Reuse for similar queries
o	Persistent storage
4.	LLM Response Cache:
o	Cache expensive LLM calls
o	Invalidate based on prompt changes
o	Cost savings tracking
Implementation:
class MultiLevelCache:
    def __init__(self, config):
        self.query_cache = RedisCache(ttl=3600)
        self.retrieval_cache = RedisCache(ttl=7200)
        self.embedding_cache = PersistentCache()
        self.llm_cache = RedisCache(ttl=86400)

    async def get_or_compute_answer(self, query: str, compute_fn):
        # Check exact match
        cached = await self.query_cache.get(query)
        if cached:
            return cached

        # Check semantic similarity
        similar = await self.find_similar_query(query)
        if similar and similar.similarity > 0.95:
            return similar.answer

        # Compute and cache
        answer = await compute_fn()
        await self.query_cache.set(query, answer)
        return answer
Files:
•	/backend/modules/cache/__init__.py
•	/backend/modules/cache/multi_level_cache.py
•	/backend/modules/cache/redis_cache.py
•	/backend/modules/cache/persistent_cache.py
•	/backend/modules/cache/similarity_search.py

### Task 9.2: Batching & Parallelization
**Claude Code Prompt:**
Optimize processing at /backend/modules/optimization/:
Features:
1.	Batch Embedding:
o	Group multiple queries/documents
o	Single API call to embedding model
o	Optimal batch size tuning
2.	Parallel Retrieval:
o	Query multiple retrievers concurrently
o	Async/await patterns
o	Connection pooling
3.	Pipeline Parallelization:
o	Identify independent steps
o	Execute in parallel
o	Merge results efficiently
4.	Rate Limiting:
o	Respect API rate limits
o	Exponential backoff
o	Request queuing
Files:
•	/backend/modules/optimization/__init__.py
•	/backend/modules/optimization/batcher.py
•	/backend/modules/optimization/parallel_executor.py
•	/backend/modules/optimization/rate_limiter.py

---

## 🎯 Deliverables Checklist

### Code Deliverables
- [ ] Query Intelligence Layer (analysis, rewriting, enhancement)
- [ ] Advanced Retrieval Layer (hybrid, self-reflective, reranking)
- [ ] Orchestration Engine (routing, pipelines, controller)
- [ ] Verification & QC (hallucination detection, attribution, confidence)
- [ ] Observability (tracing, logging, dashboard)
- [ ] Evaluation Framework (datasets, metrics, A/B testing)
- [ ] Domain Extensions (water infrastructure controller, knowledge base)
- [ ] Performance Optimization (caching, batching, parallelization)

### Documentation Deliverables
- [ ] Architecture documentation
- [ ] API documentation (auto-generated from code)
- [ ] User guides for each module
- [ ] Configuration guides
- [ ] Deployment guides
- [ ] Performance tuning guides
- [ ] Domain-specific guides (water infrastructure)

### Testing Deliverables
- [ ] Unit tests (>80% coverage)
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Performance benchmarks
- [ ] Evaluation datasets
- [ ] Regression test suite

### Deployment Deliverables
- [ ] Docker Compose configuration
- [ ] Kubernetes manifests
- [ ] CI/CD pipelines
- [ ] Monitoring setup
- [ ] Backup and recovery procedures

---

## 🚀 Quick Start Prompts for Claude Code

### For Each Phase, Start With:

Phase [N]: [Phase Name]
Please analyze the current Cognita codebase structure and:
1.	Identify the best location to add [new module]
2.	Review existing patterns and interfaces
3.	Implement [specific feature] following Cognita's conventions
4.	Write comprehensive tests
5.	Update documentation
6.	Ensure backward compatibility
Specific requirements: [Copy requirements from relevant task above]
Please create:
•	All necessary Python files with proper imports
•	Pydantic schemas for data validation
•	Factory patterns for extensibility
•	Configuration files if needed
•	Unit tests with pytest
•	Integration with existing components
•	Documentation in docstrings
Follow these principles:
•	Use async/await for I/O operations
•	Type hints everywhere
•	Dependency injection for testability
•	Configuration over hardcoding
•	Fail gracefully with informative errors
•	Log all important operations

---

## 📊 Success Metrics

By the end of this enhancement, you should achieve:

**Technical Metrics:**
- [ ] Query latency < 2 seconds (p95)
- [ ] Hallucination rate < 5%
- [ ] Source attribution > 95%
- [ ] Cache hit rate > 60%
- [ ] Test coverage > 80%

**Capability Metrics:**
- [ ] Support 5+ retrieval strategies
- [ ] Support 3+ query rewriting techniques
- [ ] Adaptive retrieval working
- [ ] A/B testing framework functional
- [ ] Domain-specific controller operational

**Production Readiness:**
- [ ] Comprehensive observability
- [ ] Automated evaluation pipeline
- [ ] Multi-environment deployment
- [ ] Documentation complete
- [ ] Security audit passed

---

## 🎓 Learning Outcomes

By completing this plan, you will have:

1. ✅ Deep understanding of production RAG architecture
2. ✅ Hands-on experience with advanced retrieval techniques
3. ✅ Knowledge of LLM orchestration patterns
4. ✅ Expertise in system observability and monitoring
5. ✅ Domain-specific RAG implementation (water infrastructure)
6. ✅ Production deployment skills
7. ✅ A portfolio-worthy open-source contribution

---

## 📞 Support & Resources

- **Cognita GitHub**: https://github.com/truefoundry/cognita
- **Cognita Docs**: Check repo README and docs/
- **Modular RAG Papers**: See research references in documents
- **LangChain**: For comparison and integration ideas
- **LlamaIndex**: For additional retrieval patterns

---

**Next Step**: Start with Phase 1, Task 1.1. Use Claude Code to analyze the repository and generate the architecture documentation. This will be your foundation for all subsequent work.

```
