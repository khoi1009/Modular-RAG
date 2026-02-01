# Cognita Extension Points

**Generated:** 2026-01-31 | **Phase:** 1 - Deep Code Inspection

## Overview

Cognita's registry-based architecture provides clear extension points for adding new components without modifying core code.

## 1. Custom DataLoader

**Use Case:** Add new data source (S3, GCS, database, API)

**Steps:**

1. Create file: `backend/modules/dataloaders/custom_loader.py`

```python
from backend.modules.dataloaders.loader import BaseDataLoader, register_dataloader
from backend.types import DataSource, LoadedDataPoint, DataIngestionMode

class CustomLoader(BaseDataLoader):
    async def load_filtered_data(
        self,
        data_source: DataSource,
        dest_dir: str,
        previous_snapshot: dict,
        batch_size: int,
        data_ingestion_mode: DataIngestionMode
    ) -> AsyncGenerator[List[LoadedDataPoint], None]:
        # Implement loading logic
        batch = []
        async for item in self._fetch_from_source(data_source):
            loaded = LoadedDataPoint(
                data_source_fqn=f"custom::{data_source.uri}",
                data_point_uri=item.path,
                data_point_hash=item.hash,
                local_filepath=f"{dest_dir}/{item.path}",
                file_extension=item.extension
            )
            batch.append(loaded)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# Register
register_dataloader("custom", CustomLoader)
```

2. Import in `backend/modules/dataloaders/__init__.py`:

```python
from backend.modules.dataloaders.custom_loader import CustomLoader
```

---

## 2. Custom Parser

**Use Case:** Add new file format parser (CAD files, scientific data, etc.)

**Steps:**

1. Create file: `backend/modules/parsers/custom_parser.py`

```python
from backend.modules.parsers.parser import BaseParser, register_parser
from langchain.schema import Document

class CustomParser(BaseParser):
    supported_file_extensions = [".custom", ".cust"]

    async def get_chunks(
        self,
        filepath: str,
        metadata: dict = None,
        chunk_size: int = 2000,
        **kwargs
    ) -> List[Document]:
        # Read and parse file
        with open(filepath, 'r') as f:
            content = f.read()

        # Chunk content
        chunks = self._chunk_content(content, chunk_size)

        # Return documents with metadata
        return [
            Document(
                page_content=chunk,
                metadata={
                    **(metadata or {}),
                    "source": filepath,
                    "chunk_index": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]

# Register
register_parser("CustomParser", CustomParser)
```

2. Import in `backend/modules/parsers/__init__.py`:

```python
from backend.modules.parsers.custom_parser import CustomParser
```

---

## 3. Custom VectorDB

**Use Case:** Add vector database (Pinecone, Chroma, Weaviate)

**Steps:**

1. Create file: `backend/modules/vector_db/custom_db.py`

```python
from backend.modules.vector_db.base import BaseVectorDB
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings

class CustomVectorDB(BaseVectorDB):
    def __init__(self, config):
        self.client = CustomDBClient(config.url, config.api_key)

    def create_collection(self, collection_name: str, embeddings: Embeddings):
        dim = self.get_embedding_dimensions(embeddings)
        self.client.create_collection(collection_name, dimension=dim)

    def upsert_documents(self, collection_name: str, documents: list,
                         embeddings: Embeddings, incremental: bool = True):
        vectors = embeddings.embed_documents([d.page_content for d in documents])
        self.client.upsert(collection_name, vectors, documents)

    def get_vector_store(self, collection_name: str, embeddings: Embeddings) -> VectorStore:
        return CustomLangChainWrapper(self.client, collection_name, embeddings)

    def get_collections(self) -> list:
        return self.client.list_collections()

    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)

    def list_data_point_vectors(self, collection_name: str, data_source_fqn: str, batch_size: int):
        return self.client.scroll(collection_name, filter={"_data_point_fqn": data_source_fqn})

    def delete_data_point_vectors(self, collection_name: str, vectors: list, batch_size: int):
        ids = [v.data_point_vector_id for v in vectors]
        self.client.delete(collection_name, ids)
```

2. Register in `backend/modules/vector_db/__init__.py`:

```python
SUPPORTED_VECTOR_DBS = {
    "qdrant": QdrantVectorDB,
    "custom": CustomVectorDB,  # Add here
}
```

---

## 4. Custom QueryController

**Use Case:** Add specialized RAG pattern (agentic, graph-based, multi-hop)

**Steps:**

1. Create directory: `backend/modules/query_controllers/custom/`

2. Create `backend/modules/query_controllers/custom/types.py`:

```python
from backend.types import BaseQueryInput

class CustomQueryInput(BaseQueryInput):
    custom_param: str = "default"
    max_iterations: int = 3
```

3. Create `backend/modules/query_controllers/custom/controller.py`:

```python
from backend.server.decorator import query_controller, post
from backend.modules.query_controllers.base import BaseQueryController
from backend.modules.query_controllers.custom.types import CustomQueryInput

@query_controller("/custom-rag")
class CustomRAGQueryController(BaseQueryController):

    @post("/answer")
    async def answer(self, request: CustomQueryInput):
        # Get components
        vector_store = await self._get_vector_store(request.collection_name)
        llm = await self._get_llm(request.model_configuration)
        retriever = await self._get_retriever(
            vector_store,
            request.retriever_name,
            request.retriever_config
        )

        # Custom logic
        context = await retriever.aget_relevant_documents(request.query)

        # Build response
        chain = self._build_custom_chain(llm, request.prompt_template)

        if request.stream:
            return StreamingResponse(self._stream_answer(chain, request.query, context))

        result = await chain.ainvoke({"context": context, "query": request.query})
        return {"answer": result, "docs": context}
```

4. Register in `backend/modules/query_controllers/__init__.py`:

```python
from backend.modules.query_controllers.custom.controller import CustomRAGQueryController
register_query_controller("custom-rag", CustomRAGQueryController)
```

---

## 5. Custom Retriever Strategy

**Use Case:** Add new retrieval approach (HyDE, step-back, etc.)

**Location:** `backend/modules/query_controllers/base.py`

**Extension Point:** `_get_retriever()` method

```python
# Add to retriever factory in BaseQueryController
async def _get_retriever(self, vector_store, retriever_name, config):
    if retriever_name == "vectorstore":
        return vector_store.as_retriever(**config.dict())

    elif retriever_name == "hyde":  # New retriever
        from backend.modules.retrievers.hyde import HyDERetriever
        return HyDERetriever(
            vector_store=vector_store,
            llm=await self._get_llm(config.retriever_llm_configuration),
            **config.dict()
        )

    elif retriever_name == "multi-query":
        # ... existing code
```

---

## 6. Custom Model Provider

**Use Case:** Add LLM provider (Azure, Anthropic, local models)

**Steps:**

1. Add provider config in `models_config.yaml`:

```yaml
providers:
  - provider_name: "custom-provider"
    api_format: "openai"
    base_url: "https://custom-api.example.com/v1"
    api_key_env_var: "CUSTOM_API_KEY"
    default_headers:
      X-Custom-Header: "value"
    embedding_model_ids:
      - "custom-embed-001"
    llm_model_ids:
      - "custom-llm-001"
```

2. Models automatically registered via ModelGateway initialization

---

## 7. Custom MetadataStore

**Use Case:** Add metadata backend (SQLite, MongoDB, in-memory)

**Steps:**

1. Create file: `backend/modules/metadata_store/custom_store.py`

```python
from backend.modules.metadata_store.base import BaseMetadataStore, register_metadata_store

class CustomMetadataStore(BaseMetadataStore):
    @classmethod
    async def aconnect(cls, **config):
        instance = cls()
        instance.client = await CustomDBClient.connect(**config)
        return instance

    async def aget_collection_by_name(self, name: str):
        return await self.client.collections.find_one({"name": name})

    async def acreate_collection(self, collection):
        return await self.client.collections.insert_one(collection.dict())

    # ... implement all abstract methods

register_metadata_store("custom", CustomMetadataStore)
```

---

## Extension Point Summary

| Component | Base Class | Registry | Registration Function |
|-----------|------------|----------|----------------------|
| DataLoader | `BaseDataLoader` | `LOADER_REGISTRY` | `register_dataloader()` |
| Parser | `BaseParser` | `PARSER_REGISTRY` | `register_parser()` |
| VectorDB | `BaseVectorDB` | `SUPPORTED_VECTOR_DBS` | Dict assignment |
| QueryController | `BaseQueryController` | `QUERY_CONTROLLER_REGISTRY` | `register_query_controller()` |
| MetadataStore | `BaseMetadataStore` | `METADATA_STORE_REGISTRY` | `register_metadata_store()` |
| Retriever | N/A | N/A | Modify `_get_retriever()` |
| Model Provider | N/A | `model_gateway` | `models_config.yaml` |

## File Reference Summary

| Extension Type | Create File | Modify File |
|----------------|-------------|-------------|
| DataLoader | `backend/modules/dataloaders/custom.py` | `backend/modules/dataloaders/__init__.py` |
| Parser | `backend/modules/parsers/custom.py` | `backend/modules/parsers/__init__.py` |
| VectorDB | `backend/modules/vector_db/custom.py` | `backend/modules/vector_db/__init__.py` |
| QueryController | `backend/modules/query_controllers/custom/` | `backend/modules/query_controllers/__init__.py` |
| MetadataStore | `backend/modules/metadata_store/custom.py` | `backend/modules/metadata_store/__init__.py` |
