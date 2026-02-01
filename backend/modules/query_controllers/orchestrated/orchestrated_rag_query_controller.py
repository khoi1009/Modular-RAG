"""Orchestrated RAG query controller with intelligent routing and pipeline execution."""
import os
from typing import AsyncIterator

import yaml
from fastapi import Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.logger import logger
from backend.modules.orchestration.pipeline.condition_evaluator import (
    ConditionEvaluator,
)
from backend.modules.orchestration.pipeline.pipeline_executor import PipelineExecutor
from backend.modules.orchestration.pipeline.schemas import PipelineDefinition
from backend.modules.orchestration.pipeline.step_registry import StepRegistry
from backend.modules.orchestration.routing.rule_based_query_router import (
    RuleBasedRouter,
)
from backend.modules.query_analysis.llm_based_query_analyzer import (
    LLMBasedQueryAnalyzer,
)
from backend.modules.query_controllers.base import BaseQueryController
from backend.modules.query_controllers.orchestrated.schemas import (
    OrchestratedQueryInput,
)
from backend.modules.query_controllers.types import Answer, Docs
from backend.server.decorators import post, query_controller


@query_controller("/orchestrated")
class OrchestratedQueryController(BaseQueryController):
    """
    Orchestrated query controller that intelligently routes queries
    and executes adaptive pipelines.
    """

    def __init__(self):
        """Initialize orchestrated controller with routing and pipeline execution"""
        super().__init__()

        # Initialize query analyzer
        self.query_analyzer = LLMBasedQueryAnalyzer(
            config={
                "model_name": "ollama/llama3",
                "model_parameters": {"temperature": 0.0},
                "timeout": 10,
            }
        )

        # Initialize router
        self.router = RuleBasedRouter("config/routing-rules.yaml")

        # Initialize pipeline executor
        self.step_registry = StepRegistry()
        self.condition_evaluator = ConditionEvaluator()
        self.pipeline_executor = PipelineExecutor(
            self.step_registry, self.condition_evaluator
        )

        # Register pipeline steps
        self._register_pipeline_steps()

    def _register_pipeline_steps(self):
        """Register available pipeline steps from modules"""
        # Import and register steps from Phase 2 and Phase 3 modules

        # Retrieval steps
        @self.step_registry.register("retrievers.vectorstore")
        async def retrieve_vectorstore(context):
            """Basic vector store retrieval"""
            collection_name = context.get("collection_name")
            query = context.get("query")

            vector_store = await self._get_vector_store(collection_name)
            retriever = self._get_vector_store_retriever(
                vector_store,
                context.get("retriever_config"),
            )

            docs = await retriever.aget_relevant_documents(query)
            return docs

        # Generation step
        @self.step_registry.register("generation.lcel_chain")
        async def generate_answer(context):
            """Generate answer using LCEL chain"""
            documents = context.get("documents", [])
            query = context.get("query")
            model_config = context.get("model_configuration")
            prompt_template = context.get("prompt_template")

            # Get LLM
            llm = self._get_llm(model_config, stream=False)

            # Create prompt
            qa_prompt = self._get_prompt_template(
                input_variables=["context", "question"],
                template=prompt_template,
            )

            # Format context
            formatted_context = self._format_docs(documents)

            # Generate answer
            chain = qa_prompt | llm
            response = await chain.ainvoke(
                {"context": formatted_context, "question": query}
            )

            return response.content if hasattr(response, "content") else str(response)

        logger.info(f"Registered {len(self.step_registry.list_steps())} pipeline steps")

    def _load_pipeline(self, pipeline_name: str) -> PipelineDefinition:
        """
        Load pipeline definition from YAML file.

        Args:
            pipeline_name: Name of pipeline

        Returns:
            PipelineDefinition

        Raises:
            FileNotFoundError: If pipeline file not found
        """
        pipeline_path = f"config/pipelines/{pipeline_name}.yaml"

        if not os.path.exists(pipeline_path):
            logger.warning(f"Pipeline file not found: {pipeline_path}, using default")
            return self._get_default_pipeline()

        with open(pipeline_path, "r") as f:
            pipeline_data = yaml.safe_load(f)

        return PipelineDefinition(**pipeline_data)

    def _get_default_pipeline(self) -> PipelineDefinition:
        """Get default simple retrieval pipeline"""
        from backend.modules.orchestration.pipeline.schemas import PipelineStep

        return PipelineDefinition(
            name="simple-retrieval",
            description="Default simple retrieval pipeline",
            steps=[
                PipelineStep(
                    name="retrieve",
                    module="retrievers.vectorstore",
                    output="documents",
                    timeout_sec=10,
                ),
                PipelineStep(
                    name="generate",
                    module="generation.lcel_chain",
                    output="answer",
                    timeout_sec=30,
                ),
            ],
        )

    @post("/answer")
    async def answer(
        self,
        request: OrchestratedQueryInput = Body(),
    ):
        """
        Answer query using orchestrated pipeline execution.

        Args:
            request: Orchestrated query input

        Returns:
            StreamingResponse or dict with answer and docs
        """
        logger.info(f"Orchestrated query: {request.query}")

        # Step 1: Analyze query (if enabled)
        if request.enable_query_analysis:
            query_metadata = await self.query_analyzer.analyze(request.query)
            logger.info(
                f"Query analysis: type={query_metadata.query_type}, "
                f"complexity={query_metadata.complexity_score}"
            )
        else:
            # Skip analysis
            from backend.modules.query_analysis.schemas import (
                QueryComplexity,
                QueryMetadata,
                QueryType,
            )

            query_metadata = QueryMetadata(
                query_type=QueryType.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                complexity_score=0.3,
                intent="retrieval-only",
                entities=[],
            )

        # Step 2: Route to appropriate pipeline
        if request.force_pipeline:
            pipeline_name = request.force_pipeline
            logger.info(f"Using forced pipeline: {pipeline_name}")
        else:
            routing_decision = await self.router.route(
                request.query, query_metadata
            )
            pipeline_name = routing_decision.controller_name
            logger.info(
                f"Routing decision: {pipeline_name} "
                f"(confidence: {routing_decision.confidence})"
            )

        # Step 3: Load pipeline definition
        pipeline = self._load_pipeline(pipeline_name)

        # Step 4: Prepare initial context
        initial_context = {
            "query": request.query,
            "collection_name": request.collection_name,
            "query_metadata": query_metadata,
            "model_configuration": request.model_configuration,
            "prompt_template": request.prompt_template,
            "retriever_config": request.retriever_config,
        }

        # Step 5: Execute pipeline
        result = await self.pipeline_executor.execute(pipeline, initial_context)

        # Step 6: Return results
        if request.stream:
            return StreamingResponse(
                self._stream_orchestrated_result(result),
                media_type="text/event-stream",
            )
        else:
            return {
                "answer": result.answer,
                "docs": result.sources,
                "execution_time_ms": result.execution_time_ms,
                "steps_executed": result.steps_executed,
                "success": result.success,
                "errors": result.errors,
            }

    async def _stream_orchestrated_result(
        self, result
    ) -> AsyncIterator[str]:
        """
        Stream orchestrated result in SSE format.

        Args:
            result: PipelineResult

        Yields:
            SSE formatted strings
        """
        # Stream docs first
        if result.sources:
            docs_model = Docs(content=result.sources)
            yield "event: data\n"
            yield f"data: {docs_model.model_dump_json()}\n\n"

        # Stream answer
        if result.answer:
            answer_model = Answer(content=result.answer)
            yield "event: data\n"
            yield f"data: {answer_model.model_dump_json()}\n\n"

        # Stream end event
        yield "event: end\n"
