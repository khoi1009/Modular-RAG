"""Water infrastructure domain-specific query controller."""
import os
from typing import Dict, Optional

from fastapi import Body
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from backend.logger import logger
from backend.modules.domain.water.asset_router import QueryIntent, WaterAssetRouter
from backend.modules.domain.water.bi.sql_agent import WaterSQLAgent
from backend.modules.domain.water.bi.visualizer import DataVisualizer
from backend.modules.domain.water.entity_extractor import WaterEntityExtractor
from backend.modules.domain.water.spatial_filter import SpatialFilter
from backend.modules.query_controllers.base import BaseQueryController
from backend.modules.query_controllers.water_infrastructure.schemas import (
    WaterInfrastructureQueryInput,
)
from backend.server.decorators import post, query_controller


@query_controller("/water-infrastructure")
class WaterInfrastructureController(BaseQueryController):
    """Query controller for water infrastructure domain."""

    def __init__(self):
        super().__init__()
        self.entity_extractor = WaterEntityExtractor()
        self.asset_router = WaterAssetRouter()
        self.spatial_filter = SpatialFilter()

        # Initialize SQL agent if database URL is configured
        self.sql_agent = None
        db_url = os.getenv("WATER_BI_DATABASE_URL")
        if db_url:
            try:
                self.sql_agent = WaterSQLAgent(db_url)
                logger.info("Water BI SQL agent initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SQL agent: {e}")

    @post("/answer")
    async def answer(
        self,
        request: WaterInfrastructureQueryInput = Body(),
    ):
        """
        Answer water infrastructure queries with domain-specific routing.

        Supports:
        - BI analytics via SQL agent
        - Spatial filtering by location/zone
        - Temporal filtering by date ranges
        - Standard RAG for general queries
        """
        try:
            # Extract entities from query
            entities = self.entity_extractor.extract(request.query)
            logger.info(f"Extracted entities: {entities.model_dump()}")

            # Classify query intent
            intent = self.asset_router.classify_intent(request.query, entities)
            logger.info(f"Query intent classified as: {intent}")

            # Route to appropriate handler
            if intent == QueryIntent.BI_ANALYTICS and request.enable_bi_analytics:
                return await self._handle_bi_query(request, entities)
            elif intent == QueryIntent.FAILURE_ANALYSIS:
                return await self._handle_failure_query(request, entities)
            elif intent == QueryIntent.MAINTENANCE:
                return await self._handle_maintenance_query(request, entities)
            elif intent == QueryIntent.COMPLIANCE:
                return await self._handle_compliance_query(request, entities)
            else:
                return await self._handle_general_query(request, entities)

        except Exception as e:
            logger.error(f"Water infrastructure query failed: {e}")
            return {
                "error": str(e),
                "answer": f"Sorry, I encountered an error processing your query: {str(e)}",
                "docs": [],
            }

    async def _handle_bi_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
    ):
        """Handle BI analytics queries using SQL agent."""
        if not self.sql_agent:
            # Fall back to RAG if SQL agent not available
            logger.warning("SQL agent not available, falling back to RAG")
            return await self._handle_general_query(request, entities)

        try:
            # Get LLM for SQL generation
            llm = self._get_llm(request.model_configuration, stream=False)

            # Execute SQL query
            sql_result = self.sql_agent.query(request.query, llm)

            if "error" in sql_result:
                # Fall back to RAG on SQL error
                logger.warning(f"SQL query failed: {sql_result['error']}, falling back to RAG")
                return await self._handle_general_query(request, entities)

            # Format results for visualization
            chart_data = DataVisualizer.format_for_chart(sql_result["result"])

            # Also get context from vector store for additional info
            vector_store = await self._get_vector_store(request.collection_name)
            retriever = await self._get_retriever(
                vector_store=vector_store,
                retriever_name=request.retriever_config.retriever_name,
                retriever_config=request.retriever_config,
            )
            docs = await retriever.ainvoke(request.query)

            return {
                "answer": f"SQL Query Result:\n\n{sql_result.get('sql', 'N/A')}\n\nFound {sql_result.get('row_count', 0)} records.",
                "docs": self._enrich_context_for_non_stream_response({"context": docs}),
                "analytics": chart_data,
                "sql": sql_result.get("sql"),
                "intent": QueryIntent.BI_ANALYTICS,
            }

        except Exception as e:
            logger.error(f"BI query handling failed: {e}")
            return await self._handle_general_query(request, entities)

    async def _handle_failure_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
    ):
        """Handle asset failure analysis queries."""
        return await self._handle_domain_query(
            request,
            entities,
            intent=QueryIntent.FAILURE_ANALYSIS,
            prompt_suffix="\n\nFocus on failure patterns, root causes, and preventive measures.",
        )

    async def _handle_maintenance_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
    ):
        """Handle maintenance-related queries."""
        return await self._handle_domain_query(
            request,
            entities,
            intent=QueryIntent.MAINTENANCE,
            prompt_suffix="\n\nFocus on maintenance history, schedules, and best practices.",
        )

    async def _handle_compliance_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
    ):
        """Handle compliance and regulatory queries."""
        return await self._handle_domain_query(
            request,
            entities,
            intent=QueryIntent.COMPLIANCE,
            prompt_suffix="\n\nFocus on regulatory requirements, standards, and compliance status.",
        )

    async def _handle_general_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
    ):
        """Handle general RAG queries."""
        return await self._handle_domain_query(
            request,
            entities,
            intent=QueryIntent.GENERAL,
        )

    async def _handle_domain_query(
        self,
        request: WaterInfrastructureQueryInput,
        entities,
        intent: QueryIntent,
        prompt_suffix: str = "",
    ):
        """Common handler for domain-specific RAG queries."""
        # Get vector store
        vector_store = await self._get_vector_store(request.collection_name)

        # Apply spatial filtering if enabled
        retriever_config = request.retriever_config
        if request.enable_spatial_filter and (entities.location or entities.zone):
            # Add spatial filter to retriever config
            spatial_filter = self.spatial_filter.create_spatial_metadata_filter(
                location=entities.location,
                zone=entities.zone,
            )
            if spatial_filter:
                # Merge with existing filter
                existing_filter = retriever_config.filter or {}
                if existing_filter and spatial_filter:
                    retriever_config.filter = {"$and": [existing_filter, spatial_filter]}
                else:
                    retriever_config.filter = spatial_filter or existing_filter

        # Create prompt template
        template = request.prompt_template or self._get_default_prompt_template()
        template += prompt_suffix

        QA_PROMPT = self._get_prompt_template(
            input_variables=["context", "question"],
            template=template,
        )

        # Get LLM
        llm = self._get_llm(request.model_configuration, request.stream)

        # Get retriever
        retriever = await self._get_retriever(
            vector_store=vector_store,
            retriever_name=retriever_config.retriever_name,
            retriever_config=retriever_config,
        )

        # Build LCEL chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self._format_docs(x["context"]))
            )
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )

        if request.internet_search_enabled:
            rag_chain_with_source = (
                rag_chain_with_source | self._internet_search
            ).assign(answer=rag_chain_from_docs)
        else:
            rag_chain_with_source = rag_chain_with_source.assign(
                answer=rag_chain_from_docs
            )

        # Execute chain
        if request.stream:
            return StreamingResponse(
                self._sse_wrap(
                    self._stream_answer(rag_chain_with_source, request.query),
                ),
                media_type="text/event-stream",
            )
        else:
            outputs = await rag_chain_with_source.ainvoke(request.query)
            return {
                "answer": outputs["answer"],
                "docs": self._enrich_context_for_non_stream_response(outputs),
                "intent": intent,
                "entities": entities.model_dump(),
            }

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template for water infrastructure queries."""
        return """You are a water infrastructure expert assistant. Answer the question based on the provided context.

Context: {context}

Question: {question}

Answer:"""
