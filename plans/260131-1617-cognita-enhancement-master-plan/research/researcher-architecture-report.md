# Cognita Architecture Research Report

**Date:** 2026-01-31
**Report ID:** researcher-260131-1617-cognita-architecture

## Executive Summary

Cognita is a modular RAG (Retrieval-Augmented Generation) platform built on LangChain, featuring pluggable components for LLMs, embeddings, vector databases, and retrieval strategies. Architecture emphasizes extensibility through abstract base classes and registry patterns.

## 1. Core Architecture Patterns

### 1.1 Plugin/Module System
- **Base Classes**: Each subsystem (QueryControllers, VectorDB, ModelGateway) defines abstract base classes
- **Registration**: Controllers decorated with `@query_controller` decorator (FastAPI-based)
- **Factory Pattern**: ModelGateway uses factory methods with caching for model instances
- **Configuration-Driven**: YAML-based model configuration (`models_config.yaml`)

### 1.2 Layering
```
API Layer (QueryControllers)
    ↓
Business Logic (RAG chains via LCEL)
    ↓
Integration Layer (ModelGateway, VectorDB, Metadata Store)
    ↓
External Services (LLMs, Embeddings, Vector DBs)
```

## 2. Query Controllers: Extension Points & Registration

### 2.1 How They Work
- **Base Class**: `BaseQueryController` provides shared utilities
- **Inheritance Model**: Custom controllers extend `BaseQueryController`
- **Decorator Pattern**: `@query_controller("/route")` registers endpoint
- **Method Declaration**: `@post("/action")` decorates handler methods

### 2.2 Example Pattern (BasicRAGQueryController)
```python
@query_controller("/basic-rag")
class BasicRAGQueryController(BaseQueryController):
    @post("/answer")
    async def answer(self, request: ExampleQueryInput):
        # 1. Get vector store
        # 2. Create prompt template
        # 3. Get LLM from config
        # 4. Build retriever
        # 5. Construct LCEL chain
        # 6. Stream or invoke
```

### 2.3 Shared Utilities from Base
- `_get_vector_store()`: Async vector store retrieval
- `_get_llm()`: LLM instantiation from ModelConfig
- `_get_retriever()`: Multi-strategy retriever factory (vectorstore, contextual-compression, multi-query)
- `_get_prompt_template()`: LangChain PromptTemplate factory
- `_stream_answer()`: Async streaming with SSE wrapper
- `_internet_search()`: Optional web search integration (Brave API)

### 2.4 Required Metadata
Controllers expect these fields in retrieved docs:
- `_id`, `_data_source_fqn`, `_data_point_hash`, `filename`, `collection_name`, `page_number`, `relevance_score`

## 3. Model Gateway: LLM & Embeddings Integration

### 3.1 Architecture
Single `ModelGateway` instance (singleton) manages all model types:
- **Chat Models**: LLMs via ChatOpenAI (configurable base URL)
- **Embeddings**: OpenAI-compatible embedding APIs
- **Rerankers**: InfinityRerankerSvc wrapper
- **Audio Models**: AudioProcessingSvc wrapper

### 3.2 Configuration Loading
```python
# From YAML file (settings.MODELS_CONFIG_PATH)
model_providers:
  - provider_name: "openai"
    api_format: "openai"
    api_key_env_var: "OPENAI_API_KEY"
    llm_model_ids: ["gpt-4", "gpt-3.5-turbo"]
    embedding_model_ids: ["text-embedding-3-small"]
    reranking_model_ids: ["rerank-english-v2.0"]
    audio_model_ids: ["whisper-1"]
```

### 3.3 Model Name Convention
Format: `{provider_name}/{model_id}`
- Example: `openai/gpt-4`, `openai/text-embedding-3-small`

### 3.4 Caching Strategy
- **LLM Cache Key**: `(model_name, stream_bool)` tuple
- **Embedder Cache Key**: `model_name` string
- **Reranker Cache Key**: `(model_name, top_k)` tuple
- **Max Cache Size**: 50 instances per type

### 3.5 Required Methods for New Providers
To add new model provider:
1. Create ModelProviderConfig entry in YAML
2. Set `api_key_env_var` environment variable
3. Update `get_llm_from_model_config()` if non-OpenAI-compatible
4. Update `get_embedder_from_model_config()` if non-OpenAI API

**Current Implementation**: Hard-coded to OpenAI-compatible APIs only

## 4. Vector Database Abstraction

### 4.1 Base Interface (BaseVectorDB)
```python
# Core methods every implementation must provide:
- create_collection(collection_name, embeddings)
- upsert_documents(collection_name, documents, embeddings, incremental)
- get_collections() → List[str]
- delete_collection(collection_name)
- get_vector_store(collection_name, embeddings) → VectorStore
- get_vector_client()
- list_data_point_vectors(collection_name, data_source_fqn, batch_size)
- delete_data_point_vectors(collection_name, data_point_vectors, batch_size)
- get_embedding_dimensions(embeddings) → int
```

### 4.2 VectorStore Retrieval Strategies
Supported via `retriever_name` in request:
1. **vectorstore**: Direct similarity search
2. **contextual-compression**: With reranking
3. **multi-query**: LLM generates multiple queries
4. **contextual-compression-multi-query**: Combines both

### 4.3 Search Configuration (RetrieverConfig)
```python
search_type: "similarity" | "mmr"
k: int  # Documents to return
fetch_k: int  # Documents for MMR pre-filter
filter: Dict  # Metadata filters
```

## 5. Core Type Definitions (types.py)

### 5.1 Data Models
- **DataPoint**: Source data unit (hash, URI, metadata)
- **DataPointVector**: Vector store entry (vector_id, FQN, hash)
- **LoadedDataPoint**: Extends DataPoint with file paths

### 5.2 Model Types
- **ModelType Enum**: chat, embedding, reranking, audio
- **ModelConfig**: name, type, parameters dict
- **ModelProviderConfig**: provider metadata, API config, model IDs by type

### 5.3 Configuration Models
- **EmbedderConfig**: name, parameters
- **ParserConfig**: name (file extension), parameters
- **VectorDBConfig**: provider, local flag, URL, API key, config dict
- **RetrieverConfig**: search strategy, k, fetch_k, filters

### 5.4 Ingestion Pipeline Types
- **DataIngestionMode**: NONE, INCREMENTAL, FULL
- **DataIngestionRunStatus**: INITIALIZED → DATA_INGESTION_COMPLETED → COMPLETED
- **BaseDataIngestionRun**: Tracks collection, data_source, parser configs, mode
- **Collection**: Named vector DB collection with associated data sources

## 6. Retrieval-Augmented Generation Chain (LCEL)

### 6.1 Standard Pattern
```python
# 1. Parallel: Get context + pass question
setup_and_retrieval = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})

# 2. Optional: Internet search enrichment
| self._internet_search  # Inserts web search results

# 3. Format docs and create prompt
| lambda x: {context: format_docs(x["context"]), question: x["question"]}
| QA_PROMPT

# 4. LLM inference
| llm

# 5. Output parsing
| StrOutputParser()
```

### 6.2 Streaming
- Returns `AsyncIterator[BaseModel]` with SSE format
- Yields `Docs` for context chunks, `Answer` for LLM output
- Timeout: `GENERATION_TIMEOUT_SEC` (configurable)

## 7. Key Extension Points for New Features

### 7.1 Adding New Query Controller
1. Extend `BaseQueryController`
2. Use `@query_controller("/path")` decorator
3. Implement `answer()` method
4. Use base class utilities for vector store, LLM, retriever
5. Return dict with "answer" and "docs" keys or StreamingResponse

### 7.2 Adding New Vector DB Provider
1. Implement `BaseVectorDB` interface
2. Register in client factory (backend/modules/vector_db/client.py)
3. Add config entry with provider name
4. Must implement all 8 abstract methods

### 7.3 Adding New LLM/Embedding Provider
1. Add `ModelProviderConfig` entry in `models_config.yaml`
2. Set environment variable for API key
3. If non-OpenAI-compatible: Modify gateway methods
4. Caching automatic via ModelGateway

### 7.4 Custom Retrieval Strategy
1. Implement LangChain `Retriever` interface
2. Register in `_get_retriever()` method with new `retriever_name`
3. Add `RetrieverConfig` parameters as needed

## 8. Dependencies & Integration Points

### 8.1 External
- **LangChain**: LCEL chains, retrievers, prompt templates, document schema
- **Pydantic**: Type validation and serialization
- **FastAPI**: HTTP endpoints and decorators
- **Model APIs**: OpenAI-compatible (configurable endpoints)
- **Vector DBs**: Qdrant, Pinecone, etc. (via LangChain abstraction)

### 8.2 Internal
- **Metadata Store**: Collection, data source, data point metadata
- **Model Gateway**: Centralized model instance management
- **Vector DB Client**: Collection operations
- **Parser Modules**: Document extraction and chunking

## Unresolved Questions

1. **Provider Extensibility**: Is OpenAI-only API format limitation intentional? Path to support Anthropic, Cohere, other APIs?
2. **Model Gateway Async**: Why synchronous initialization in async context? Thread safety of caching?
3. **Metadata Requirements**: Are all 10 required metadata fields enforced? Validation layer?
4. **Retriever Composition**: Can multiple retrievers be chained/weighted? Current patterns only support predefined combinations.
5. **Error Handling**: What happens if model registration fails? Graceful degradation strategy?

---

**Report Type:** Architecture Analysis
**Scope:** Backend module interfaces and patterns
**Tools Used:** Code inspection, pattern analysis
