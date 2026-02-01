# Cognita Architecture Analysis

**Generated:** 2026-01-31 | **Phase:** 1 - Deep Code Inspection

## Executive Summary

Cognita is a production-ready RAG (Retrieval-Augmented Generation) framework built on LangChain/LlamaIndex. The codebase (~5,800 LOC) provides modular, extensible, API-driven architecture for scalable RAG systems.

## Core Philosophy

- **Modularity**: Separate concerns into independent, reusable components
- **Extensibility**: Registry-based patterns for swapping implementations
- **Production-Ready**: Incremental indexing, streaming, error handling, caching
- **Type-Safe**: Pydantic v2 models throughout

## Data Flow

```
Data Loading → Parsing & Chunking → Embedding → Vector Storage + Metadata
                                                            ↓
                        User Query → Retrieval → Reranking → LLM → Response
```

## Directory Structure

```
backend/modules/ (5,800 LOC total)
├── dataloaders/          # Data source integration (136 LOC)
│   ├── loader.py         # BaseDataLoader ABC + registry
│   ├── local_dir_loader.py
│   ├── web_loader.py
│   └── truefoundry_loader.py
│
├── parsers/              # Document chunking (142 LOC base)
│   ├── parser.py         # BaseParser ABC + registry + cache
│   ├── unstructured_io.py     # Universal doc parser
│   ├── multi_modal_parser.py  # Vision LLM analysis
│   ├── audio_parser.py        # Speech-to-text
│   ├── video_parser.py        # Audio+multimodal
│   └── web_parser.py          # HTML extraction
│
├── vector_db/            # VectorDB abstraction (98 LOC base)
│   ├── base.py           # BaseVectorDB ABC (7 methods)
│   ├── qdrant.py         # Qdrant implementation (primary)
│   └── [milvus, mongo, weaviate, singlestore].py (planned)
│
├── model_gateway/        # Unified model management
│   ├── model_gateway.py  # Singleton caching (350 LOC)
│   ├── reranker_svc.py   # Infinity-based reranking
│   └── audio_processing_svc.py
│
├── metadata_store/       # Collection & run metadata
│   ├── base.py           # BaseMetadataStore ABC (250 LOC)
│   └── prisma_store.py   # Prisma+PostgreSQL impl
│
└── query_controllers/    # RAG query execution
    ├── base.py           # BaseQueryController (250 LOC)
    ├── types.py          # Retriever configs
    ├── example/          # BasicRAGQueryController
    └── multimodal/       # MultiModalRAGQueryController
```

## Component Deep-Dives

### DataLoader System

**Location:** `backend/modules/dataloaders/loader.py`

**Registry Pattern:**
```python
LOADER_REGISTRY = {}
def register_dataloader(type: str, cls): LOADER_REGISTRY[type] = cls
def get_loader_for_data_source(type, *args, **kwargs): return LOADER_REGISTRY[type](*args, **kwargs)
```

**Implementations:**
- `"localdir"` → LocalDirLoader (copies from local path)
- `"web"` → WebLoader (downloads from HTTP URLs)
- `"truefoundry"` → TrueFoundryLoader (conditional on TFY_API_KEY)

**Key Abstraction:**
```python
class BaseDataLoader(ABC):
    async def load_full_data(data_source, dest_dir, batch_size=100)
    async def load_incremental_data(data_source, dest_dir, previous_snapshot, batch_size=100)
    @abstractmethod
    async def load_filtered_data(...) -> AsyncGenerator[List[LoadedDataPoint]]
```

### Parser System

**Location:** `backend/modules/parsers/parser.py`

**Registry with Multi-Mapping:**
```python
PARSER_REGISTRY = {}                    # name → parser class
PARSER_REGISTRY_EXTENSIONS = {}         # extension → [parser names]
```

**Registered Parsers:**
1. **UnstructuredIoParser** - Primary parser for PDF, DOCX, etc.
2. **MultiModalParser** - GPT-4 Vision for images/charts
3. **AudioParser** - faster-whisper-server integration
4. **VideoParser** - Combined audio + visual analysis
5. **WebParser** - HTML extraction

**Caching:** `cache_key = MD5(extension + parsers_map + args + kwargs)`

### VectorDB Abstraction

**Location:** `backend/modules/vector_db/base.py`

**Core Interface:**
```python
class BaseVectorDB(ABC):
    def create_collection(collection_name, embeddings)
    def upsert_documents(collection_name, documents, embeddings, incremental=True)
    def get_collections() -> List[str]
    def delete_collection(collection_name)
    def get_vector_store(collection_name, embeddings) -> VectorStore
    def list_data_point_vectors(collection_name, data_source_fqn, batch_size)
    def delete_data_point_vectors(collection_name, data_point_vectors, batch_size)
```

**Active Implementation:** Qdrant with cosine similarity, replication factor 3

### ModelGateway Singleton

**Location:** `backend/modules/model_gateway/model_gateway.py`

**Architecture:**
```python
class ModelGateway:
    _embedder_cache: Cache[model_name → Embeddings]      # max 50
    _llm_cache: Cache[(model_name, stream) → BaseChatModel]
    _reranker_cache: Cache[(model_name, top_k) → InfinityRerankerSvc]
    _audio_cache: Cache[model_name → AudioProcessingSvc]
```

**Services:**
- `get_embedder_from_model_config(model_name)` → LangChain Embeddings
- `get_llm_from_model_config(model_config, stream)` → ChatOpenAI
- `get_reranker_from_model_config(model_name, top_k)` → InfinityRerankerSvc
- `get_audio_model_from_model_config(model_name)` → AudioProcessingSvc

**Supported Providers:** OpenAI, Ollama, Infinity, Custom (via base_url)

### QueryController System

**Location:** `backend/modules/query_controllers/base.py`

**Registered Controllers:**
- `"basic-rag"` → BasicRAGQueryController (`example/controller.py:23-113`)
- `"multimodal"` → MultiModalRAGQueryController (`multimodal/controller.py:29-140`)

**Retriever Types:**
1. `"vectorstore"` - Simple similarity search
2. `"multi-query"` - Query decomposition via LLM
3. `"contextual-compression"` - Document reranking
4. `"contextual-compression-multi-query"` - Both combined

**LCEL Chain Pattern:**
```python
RunnableParallel({"context": retriever, "question": passthrough})
| Optional[internet_search]
| RunnablePassthrough.assign(context=format_docs)
| qa_prompt
| llm
| StrOutputParser()
```

### MetadataStore

**Location:** `backend/modules/metadata_store/base.py`

**Key Entities:**
- **Collections** - Configuration and embedder settings
- **DataSources** - Source references and associations
- **DataIngestionRuns** - Job tracking with status progression
- **RAGApplications** - User configurations

**Implementation:** PrismaMetadataStore (Prisma ORM + PostgreSQL)

## External Integrations

| Service | Purpose | Config |
|---------|---------|--------|
| OpenAI API | LLM, Embeddings | OPENAI_API_KEY |
| UnstructuredIO | Document parsing | UNSTRUCTURED_IO_URL |
| Infinity Server | Reranking | INFINITY_RERANKER_BASE_URL |
| faster-whisper | Audio transcription | AUDIO_PROCESSING_BASE_URL |
| Brave Search | Web search | BRAVE_API_KEY |
| Qdrant | Vector storage | VECTOR_DB_CONFIG |
| PostgreSQL | Metadata store | DATABASE_URL |

## Key Design Decisions

1. **Async/Await** - Non-blocking I/O throughout
2. **Registry-Based** - Plugin architecture without hard-coded imports
3. **FQN Pattern** - `"{type}::{uri}"` for unique identification
4. **Metadata Preservation** - Each chunk carries provenance
5. **Incremental Indexing** - Hash-based change detection
6. **Singleton ModelGateway** - Centralized model caching
7. **LCEL Chains** - Composable, streaming-friendly runnables
8. **Dual-Mode Response** - Batch (JSON) or streaming (SSE)

## Statistics

| Metric | Value |
|--------|-------|
| Total Module LOC | ~5,800 |
| Abstract Base Classes | 7 |
| Registry Implementations | 5 |
| Supported DataLoaders | 3 |
| Supported Parsers | 5 |
| Active VectorDBs | 1 (Qdrant) |
| QueryControllers | 2 |
| External Integrations | 8 |
