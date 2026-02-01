# Cognita Integration Patterns

**Generated:** 2026-01-31 | **Phase:** 1 - Deep Code Inspection

## Registry Pattern

Core extensibility mechanism used throughout Cognita.

### Pattern Structure

```python
# 1. Define registry
REGISTRY = {}

# 2. Registration function
def register_component(name: str, cls):
    REGISTRY[name] = cls

# 3. Factory function
def get_component(name: str, *args, **kwargs):
    return REGISTRY[name](*args, **kwargs)

# 4. Register in __init__.py
register_component("impl-name", ImplementationClass)
```

### Implementations

| Module | Registry | Registration | Factory |
|--------|----------|--------------|---------|
| DataLoaders | `LOADER_REGISTRY` | `register_dataloader()` | `get_loader_for_data_source()` |
| Parsers | `PARSER_REGISTRY` | `register_parser()` | `get_parser()` |
| VectorDB | `SUPPORTED_VECTOR_DBS` | Dict literal | `get_vector_db_client()` |
| QueryControllers | `QUERY_CONTROLLER_REGISTRY` | `register_query_controller()` | `list_query_controllers()` |
| MetadataStore | `METADATA_STORE_REGISTRY` | `register_metadata_store()` | `get_metadata_store_client()` |

## Singleton with Caching

ModelGateway implements singleton pattern with LRU caching.

**Location:** `backend/modules/model_gateway/model_gateway.py`

```python
class ModelGateway:
    _embedder_cache: Cache[str, Embeddings] = Cache(max_size=50)
    _llm_cache: Cache[Tuple[str, bool], BaseChatModel] = Cache(max_size=50)
    _reranker_cache: Cache[Tuple[str, int], InfinityRerankerSvc] = Cache(max_size=50)

    def get_embedder_from_model_config(self, model_name: str) -> Embeddings:
        if model_name in self._embedder_cache:
            return self._embedder_cache[model_name]
        embedder = self._create_embedder(model_name)
        self._embedder_cache[model_name] = embedder
        return embedder

# Singleton instance
model_gateway = ModelGateway()
```

## Decorator-Based Route Registration

QueryControllers use decorators for FastAPI integration.

**Location:** `backend/server/decorator.py`

```python
from backend.server.decorator import query_controller, post, get

@query_controller("/basic-rag")
class BasicRAGQueryController:

    @post("/answer")
    async def answer(self, request: ExampleQueryInput) -> StreamingResponse | Dict:
        # Implementation
        pass
```

**Route Registration Flow:**
1. `@query_controller("/path")` registers class in `QUERY_CONTROLLER_REGISTRY`
2. `@post("/answer")` adds method as POST endpoint
3. FastAPI router created at startup from registry

## Abstract Base Class Pattern

All components extend abstract base classes for interface enforcement.

### BaseDataLoader

```python
class BaseDataLoader(ABC):
    @abstractmethod
    async def load_filtered_data(
        self,
        data_source: DataSource,
        dest_dir: str,
        previous_snapshot: Dict[str, str],
        batch_size: int,
        data_ingestion_mode: DataIngestionMode
    ) -> AsyncGenerator[List[LoadedDataPoint], None]:
        """Must yield batches of loaded data points."""
        pass
```

### BaseParser

```python
class BaseParser(ABC):
    supported_file_extensions: List[str] = []

    @abstractmethod
    async def get_chunks(
        self,
        filepath: str,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> List[Document]:
        """Parse file and return chunked documents."""
        pass
```

### BaseVectorDB

```python
class BaseVectorDB(ABC):
    @abstractmethod
    def create_collection(self, collection_name: str, embeddings: Embeddings): pass

    @abstractmethod
    def upsert_documents(self, collection_name: str, documents: List[Document],
                         embeddings: Embeddings, incremental: bool = True): pass

    @abstractmethod
    def get_vector_store(self, collection_name: str, embeddings: Embeddings) -> VectorStore: pass
```

## LCEL Chain Composition

LangChain Expression Language for RAG pipelines.

**Location:** `backend/modules/query_controllers/example/controller.py`

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Build chain
qa_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | RunnablePassthrough.assign(context=lambda x: self._format_docs(x["context"]))
    | qa_prompt
    | llm
    | StrOutputParser()
)

# Execute
if stream:
    return StreamingResponse(self._stream_answer(qa_chain, query, docs))
else:
    result = await qa_chain.ainvoke(query)
    return {"answer": result, "docs": docs}
```

## Async Generator Pattern

Batch processing with async generators for memory efficiency.

**Location:** `backend/modules/dataloaders/loader.py`

```python
async def load_filtered_data(
    self, data_source, dest_dir, previous_snapshot, batch_size, data_ingestion_mode
) -> AsyncGenerator[List[LoadedDataPoint], None]:
    batch = []
    async for item in self._scan_source(data_source):
        if self._should_include(item, previous_snapshot, data_ingestion_mode):
            batch.append(await self._load_item(item, dest_dir))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

## Pydantic Configuration Pattern

All models use `ConfiguredBaseModel` with Pydantic v2.

**Location:** `backend/types.py`

```python
from pydantic import BaseModel, ConfigDict

class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
        validate_assignment=True
    )

class DataPoint(ConfiguredBaseModel):
    data_source_fqn: str
    data_point_uri: str
    data_point_hash: str
    metadata: Dict[str, Any] = {}
```

## Streaming Response Pattern

SSE (Server-Sent Events) for real-time response streaming.

**Location:** `backend/modules/query_controllers/base.py`

```python
async def _stream_answer(self, chain, query: str, docs: List[Document]):
    # Stream docs first
    yield self._sse_wrap({"docs": self._enrich_context(docs)})

    # Stream answer chunks
    async for chunk in chain.astream(query):
        yield self._sse_wrap(chunk)

    # End signal
    yield "event: end\n\n"

def _sse_wrap(self, data: Any) -> str:
    return f"event: data\ndata: {json.dumps(data)}\n\n"
```

## Configuration Loading Pattern

YAML config → Pydantic validation → Runtime behavior.

**Location:** `backend/settings.py`, `models_config.yaml`

```python
# Load config
with open("models_config.yaml") as f:
    config = yaml.safe_load(f)

# Validate with Pydantic
providers = [ModelProviderConfig(**p) for p in config["providers"]]

# Build registries
for provider in providers:
    for model_id in provider.embedding_model_ids:
        fqn = f"{provider.provider_name}/{model_id}"
        model_gateway.register_embedding_model(fqn, provider)
```

## Error Handling Pattern

Retry with exponential backoff for external services.

**Location:** `backend/modules/parsers/unstructured_io.py`

```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def parse_with_retry(self, filepath: str) -> List[Document]:
    async with aiohttp.ClientSession() as session:
        response = await session.post(self.api_url, data={"file": open(filepath, "rb")})
        if response.status != 200:
            raise ParseError(f"UnstructuredIO returned {response.status}")
        return self._process_response(await response.json())
```

## Incremental Processing Pattern

Hash-based change detection for efficient re-indexing.

**Location:** `backend/modules/vector_db/qdrant.py`

```python
def upsert_documents(self, collection_name, documents, embeddings, incremental=True):
    if incremental:
        # Fetch existing vectors by data_point_fqn
        existing = self.list_data_point_vectors(collection_name, data_source_fqn)
        existing_map = {v.data_point_fqn: v.data_point_hash for v in existing}

        # Filter to new/changed only
        to_insert = []
        for doc in documents:
            fqn = doc.metadata["_data_point_fqn"]
            new_hash = doc.metadata["_data_point_hash"]
            if fqn not in existing_map or existing_map[fqn] != new_hash:
                to_insert.append(doc)

        # Delete stale
        stale = [v for v in existing if v.data_point_hash != ...]
        self.delete_data_point_vectors(collection_name, stale)

        documents = to_insert

    # Insert new documents
    self._batch_insert(collection_name, documents, embeddings)
```

## Code References

| Pattern | Primary File | Line |
|---------|--------------|------|
| Registry | `backend/modules/dataloaders/loader.py` | 15-25 |
| Singleton | `backend/modules/model_gateway/model_gateway.py` | 45-60 |
| Decorator Route | `backend/server/decorator.py` | 1-50 |
| LCEL Chain | `backend/modules/query_controllers/example/controller.py` | 60-90 |
| Async Generator | `backend/modules/dataloaders/loader.py` | 45-70 |
| SSE Stream | `backend/modules/query_controllers/base.py` | 150-180 |
| Incremental | `backend/modules/vector_db/qdrant.py` | 80-120 |
