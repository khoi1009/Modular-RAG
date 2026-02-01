from typing import List, Optional, Literal
from urllib.parse import urlparse

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores.qdrant import Qdrant
from langchain.docstore.document import Document
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from backend.constants import DATA_POINT_FQN_METADATA_KEY, DATA_POINT_HASH_METADATA_KEY
from backend.logger import logger
from backend.modules.vector_db.base import BaseVectorDB
from backend.types import DataPointVector, QdrantClientConfig, VectorDBConfig

MAX_SCROLL_LIMIT = int(1e6)
BATCH_SIZE = 1000

# Quantization modes for enterprise-scale indexing
QuantizationMode = Literal["none", "scalar", "binary"]


class QuantizationConfig:
    """
    Configuration for vector quantization to support enterprise-scale datasets (10M+ docs).

    Binary Quantization achieves:
    - 32x memory reduction (1-bit per dimension vs 32-bit float)
    - Sub-30ms latency for large-scale retrieval
    - Requires rescoring for accuracy preservation
    """

    def __init__(
        self,
        mode: QuantizationMode = "none",
        rescore: bool = True,
        rescore_multiplier: float = 3.0,
        always_ram: bool = True,
    ):
        """
        Args:
            mode: Quantization mode - "none", "scalar", or "binary"
            rescore: Enable rescoring with full vectors (recommended for binary)
            rescore_multiplier: Fetch N*multiplier candidates for rescoring (2-3x recommended)
            always_ram: Keep quantized vectors in RAM for speed
        """
        self.mode = mode
        self.rescore = rescore
        self.rescore_multiplier = rescore_multiplier
        self.always_ram = always_ram

    def to_qdrant_config(self) -> Optional[models.QuantizationConfig]:
        """Convert to Qdrant quantization config."""
        if self.mode == "none":
            return None
        elif self.mode == "scalar":
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=self.always_ram,
                )
            )
        elif self.mode == "binary":
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=self.always_ram,
                )
            )
        return None


class QdrantVectorDB(BaseVectorDB):
    def __init__(self, config: VectorDBConfig):
        logger.debug(f"Connecting to qdrant using config: {config.model_dump()}")

        # Extract quantization config from VectorDBConfig.config if present
        config_dict = config.config or {}
        quantization_mode = config_dict.pop("quantization_mode", "none")
        rescore_multiplier = config_dict.pop("rescore_multiplier", 3.0)

        self.quantization_config = QuantizationConfig(
            mode=quantization_mode,
            rescore=quantization_mode == "binary",  # Always rescore for binary
            rescore_multiplier=rescore_multiplier,
        )

        if config.local is True:
            # TODO: make this path customizable
            self.qdrant_client = QdrantClient(
                path="./qdrant_db",
            )
        else:
            url = config.url
            api_key = config.api_key
            if not api_key:
                api_key = None
            qdrant_kwargs = QdrantClientConfig.model_validate(config_dict)
            if url.startswith("http://") or url.startswith("https://"):
                if qdrant_kwargs.port is None:
                    parsed_port = urlparse(url).port
                    if parsed_port:
                        qdrant_kwargs.port = parsed_port
                    else:
                        qdrant_kwargs.port = 443 if url.startswith("https://") else 6333
            self.qdrant_client = QdrantClient(
                url=url, api_key=api_key, **qdrant_kwargs.model_dump()
            )

        logger.info(f"[Qdrant] Initialized with quantization_mode={quantization_mode}")

    def create_collection(
        self,
        collection_name: str,
        embeddings: Embeddings,
        quantization_config: Optional[QuantizationConfig] = None,
    ):
        """
        Create a new collection with optional quantization for enterprise scale.

        Args:
            collection_name: Name of the collection
            embeddings: Embeddings model to determine vector size
            quantization_config: Optional quantization config (uses instance default if None)

        Binary Quantization enables:
        - 32x memory reduction for 10M+ document datasets
        - Sub-30ms retrieval latency
        - Accuracy preserved via rescoring mechanism
        """
        logger.debug(f"[Qdrant] Creating new collection {collection_name}")

        # Use provided config or instance default
        quant_config = quantization_config or self.quantization_config

        # Calculate embedding size
        partial_embeddings = embeddings.embed_documents(["Initial document"])
        vector_size = len(partial_embeddings[0])
        logger.debug(f"Vector size: {vector_size}")

        # Build collection parameters
        collection_params = {
            "collection_name": collection_name,
            "vectors_config": VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            "replication_factor": 3,
        }

        # Add quantization config if enabled
        qdrant_quant = quant_config.to_qdrant_config()
        if qdrant_quant:
            collection_params["quantization_config"] = qdrant_quant
            logger.info(
                f"[Qdrant] Creating collection with {quant_config.mode} quantization "
                f"(rescore_multiplier={quant_config.rescore_multiplier})"
            )

        self.qdrant_client.create_collection(**collection_params)

        self.qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        logger.debug(f"[Qdrant] Created new collection {collection_name}")

    def _get_records_to_be_upserted(
        self, collection_name: str, data_point_fqns: List[str], incremental: bool
    ):
        if not incremental:
            return []
        # For incremental deletion, we delete the documents with the same document_id
        logger.debug(
            f"[Qdrant] Incremental Ingestion: Fetching documents for {len(data_point_fqns)} data point fqns for collection {collection_name}"
        )
        stop = False
        offset = None
        record_ids_to_be_upserted = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                            match=models.MatchAny(
                                any=data_point_fqns,
                            ),
                        ),
                    ]
                ),
                limit=BATCH_SIZE,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            for record in records:
                record_ids_to_be_upserted.append(record.id)
                if len(record_ids_to_be_upserted) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset

        logger.debug(
            f"[Qdrant] Incremental Ingestion: collection={collection_name} Addition={len(data_point_fqns)}, Updates={len(record_ids_to_be_upserted)}"
        )
        return record_ids_to_be_upserted

    def upsert_documents(
        self,
        collection_name: str,
        documents,
        embeddings: Embeddings,
        incremental: bool = True,
    ):
        if len(documents) == 0:
            logger.warning("No documents to index")
            return
        # get record IDs to be upserted
        logger.debug(
            f"[Qdrant] Adding {len(documents)} documents to collection {collection_name}"
        )
        data_point_fqns = []
        for document in documents:
            if document.metadata.get(DATA_POINT_FQN_METADATA_KEY):
                data_point_fqns.append(
                    document.metadata.get(DATA_POINT_FQN_METADATA_KEY)
                )
        record_ids_to_be_upserted: List[str] = self._get_records_to_be_upserted(
            collection_name=collection_name,
            data_point_fqns=data_point_fqns,
            incremental=incremental,
        )

        # Add Documents
        Qdrant(
            client=self.qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
        ).add_documents(documents=documents)
        logger.debug(
            f"[Qdrant] Added {len(documents)} documents to collection {collection_name}"
        )

        # Delete Documents
        if len(record_ids_to_be_upserted):
            logger.debug(
                f"[Qdrant] Deleting {len(documents)} outdated documents from collection {collection_name}"
            )
            for i in range(0, len(record_ids_to_be_upserted), BATCH_SIZE):
                record_ids_to_be_processed = record_ids_to_be_upserted[
                    i : i + BATCH_SIZE
                ]
                self.qdrant_client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=record_ids_to_be_processed,
                    ),
                )
            logger.debug(
                f"[Qdrant] Deleted {len(documents)} outdated documents from collection {collection_name}"
            )

    def get_collections(self) -> List[str]:
        logger.debug("[Qdrant] Fetching collections")
        collections = self.qdrant_client.get_collections().collections
        logger.debug(f"[Qdrant] Fetched {len(collections)} collections")
        return [collection.name for collection in collections]

    def delete_collection(self, collection_name: str):
        logger.debug(f"[Qdrant] Deleting {collection_name} collection")
        self.qdrant_client.delete_collection(collection_name=collection_name)
        logger.debug(f"[Qdrant] Deleted {collection_name} collection")

    def get_vector_store(self, collection_name: str, embeddings: Embeddings):
        logger.debug(f"[Qdrant] Getting vector store for collection {collection_name}")
        return Qdrant(
            client=self.qdrant_client,
            embeddings=embeddings,
            collection_name=collection_name,
        )

    def search_with_rescoring(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[models.Filter] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> List[models.ScoredPoint]:
        """
        Search with rescoring for binary quantized collections.

        This implements the two-stage retrieval pattern:
        1. Fast retrieval using quantized vectors (2-3x candidates)
        2. Rescore using full float vectors for accuracy

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional Qdrant filter
            quantization_config: Optional config (uses instance default if None)

        Returns:
            List of ScoredPoint results with accurate scores
        """
        quant_config = quantization_config or self.quantization_config

        # Determine search parameters based on quantization mode
        search_params = models.SearchParams(hnsw_ef=128)

        if quant_config.mode == "binary" and quant_config.rescore:
            # Binary quantization with rescoring:
            # 1. Retrieve more candidates using fast binary search
            # 2. Rescore with full vectors
            oversample_k = int(top_k * quant_config.rescore_multiplier)

            search_params = models.SearchParams(
                hnsw_ef=128,
                quantization=models.QuantizationSearchParams(
                    ignore=False,  # Use quantized vectors for initial search
                    rescore=True,  # Rescore with full vectors
                    oversampling=quant_config.rescore_multiplier,
                ),
            )
            logger.debug(
                f"[Qdrant] Binary search with rescoring: "
                f"top_k={top_k}, oversample={oversample_k}"
            )
        elif quant_config.mode == "scalar":
            # Scalar quantization: optional rescoring
            search_params = models.SearchParams(
                hnsw_ef=128,
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=quant_config.rescore,
                ),
            )

        results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter_conditions,
            search_params=search_params,
            with_payload=True,
            with_vectors=False,
        )

        logger.debug(f"[Qdrant] Search returned {len(results)} results")
        return results

    async def asearch_with_rescoring(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter_conditions: Optional[models.Filter] = None,
        quantization_config: Optional[QuantizationConfig] = None,
    ) -> List[models.ScoredPoint]:
        """Async version of search_with_rescoring."""
        # Qdrant client search is fast, run sync in executor if needed
        return self.search_with_rescoring(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions,
            quantization_config=quantization_config,
        )

    def similarity_search_with_rescoring(
        self,
        collection_name: str,
        query: str,
        embeddings: Embeddings,
        top_k: int = 10,
        filter_conditions: Optional[models.Filter] = None,
    ) -> List[Document]:
        """
        High-level similarity search with automatic rescoring for binary quantization.

        Args:
            collection_name: Name of the collection
            query: Query text
            embeddings: Embeddings model
            top_k: Number of results
            filter_conditions: Optional filter

        Returns:
            List of LangChain Document objects
        """
        # Embed query
        query_vector = embeddings.embed_query(query)

        # Search with rescoring
        results = self.search_with_rescoring(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_conditions=filter_conditions,
        )

        # Convert to Documents
        documents = []
        for point in results:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            metadata["relevance_score"] = point.score

            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=metadata,
                )
            )

        return documents

    def get_vector_client(self):
        logger.debug("[Qdrant] Getting Qdrant client")
        return self.qdrant_client

    def list_data_point_vectors(
        self, collection_name: str, data_source_fqn: str, batch_size: int = BATCH_SIZE
    ) -> List[DataPointVector]:
        logger.debug(
            f"[Qdrant] Listing all data point vectors for collection {collection_name}"
        )
        stop = False
        offset = None
        data_point_vectors: List[DataPointVector] = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                with_payload=[
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                    f"metadata.{DATA_POINT_HASH_METADATA_KEY}",
                ],
                scroll_filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                            match=models.MatchText(
                                text=data_source_fqn,
                            ),
                        ),
                    ]
                ),
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                metadata: dict = record.payload.get("metadata")
                if (
                    metadata
                    and metadata.get(DATA_POINT_FQN_METADATA_KEY)
                    and metadata.get(DATA_POINT_HASH_METADATA_KEY)
                ):
                    data_point_vectors.append(
                        DataPointVector(
                            data_point_vector_id=record.id,
                            data_point_fqn=metadata.get(DATA_POINT_FQN_METADATA_KEY),
                            data_point_hash=metadata.get(DATA_POINT_HASH_METADATA_KEY),
                        )
                    )
                if len(data_point_vectors) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Listing {len(data_point_vectors)} data point vectors for collection {collection_name}"
        )
        return data_point_vectors

    def delete_data_point_vectors(
        self,
        collection_name: str,
        data_point_vectors: List[DataPointVector],
        batch_size: int = BATCH_SIZE,
    ):
        """
        Delete data point vectors from the collection
        """
        logger.debug(f"[Qdrant] Deleting {len(data_point_vectors)} data point vectors")
        vectors_to_be_deleted_count = len(data_point_vectors)
        deleted_vectors_count = 0
        for i in range(0, vectors_to_be_deleted_count, batch_size):
            data_point_vectors_to_be_processed = data_point_vectors[i : i + batch_size]
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[
                        document_vector_point.data_point_vector_id
                        for document_vector_point in data_point_vectors_to_be_processed
                    ],
                ),
            )
            deleted_vectors_count = deleted_vectors_count + len(
                data_point_vectors_to_be_processed
            )
            logger.debug(
                f"[Qdrant] Deleted [{deleted_vectors_count}/{vectors_to_be_deleted_count}] data point vectors"
            )
        logger.debug(
            f"[Qdrant] Deleted {vectors_to_be_deleted_count} data point vectors"
        )

    def list_documents_in_collection(
        self, collection_name: str, base_document_id: str = None
    ) -> List[str]:
        """
        List all documents in a collection
        """
        logger.debug(
            f"[Qdrant] Listing all documents with base document id {base_document_id} for collection {collection_name}"
        )
        stop = False
        offset = None
        document_ids_set = set()
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    should=(
                        [
                            models.FieldCondition(
                                key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                                match=models.MatchText(
                                    text=base_document_id,
                                ),
                            ),
                        ]
                        if base_document_id
                        else None
                    )
                ),
                limit=BATCH_SIZE,
                with_payload=[f"metadata.{DATA_POINT_FQN_METADATA_KEY}"],
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                if record.payload.get("metadata") and record.payload.get(
                    "metadata"
                ).get(DATA_POINT_FQN_METADATA_KEY):
                    document_ids_set.add(
                        record.payload.get("metadata").get(DATA_POINT_FQN_METADATA_KEY)
                    )
                if len(document_ids_set) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Found {len(document_ids_set)} documents with base document id {base_document_id} for collection {collection_name}"
        )
        return list(document_ids_set)

    def delete_documents(self, collection_name: str, document_ids: List[str]):
        """
        Delete documents from the collection
        """
        logger.debug(
            f"[Qdrant] Deleting {len(document_ids)} documents from collection {collection_name}"
        )
        try:
            self.qdrant_client.get_collection(collection_name=collection_name)
        except Exception as exp:
            logger.debug(exp)
            return
        # https://qdrant.tech/documentation/concepts/filtering/#full-text-match

        for i in range(0, len(document_ids), BATCH_SIZE):
            document_ids_to_be_processed = document_ids[i : i + BATCH_SIZE]
            self.qdrant_client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                                match=models.MatchAny(any=document_ids_to_be_processed),
                            ),
                        ],
                    )
                ),
            )
        logger.debug(
            f"[Qdrant] Deleted {len(document_ids)} documents from collection {collection_name}"
        )

    def list_document_vector_points(
        self, collection_name: str
    ) -> List[DataPointVector]:
        """
        List all documents in a collection
        """
        logger.debug(
            f"[Qdrant] Listing all document vector points for collection {collection_name}"
        )
        stop = False
        offset = None
        document_vector_points: List[DataPointVector] = []
        while stop is not True:
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=BATCH_SIZE,
                with_payload=[
                    f"metadata.{DATA_POINT_FQN_METADATA_KEY}",
                    f"metadata.{DATA_POINT_HASH_METADATA_KEY}",
                ],
                with_vectors=False,
                offset=offset,
            )
            for record in records:
                metadata: dict = record.payload.get("metadata")
                if (
                    metadata
                    and metadata.get(DATA_POINT_FQN_METADATA_KEY)
                    and metadata.get(DATA_POINT_HASH_METADATA_KEY)
                ):
                    document_vector_points.append(
                        DataPointVector(
                            point_id=record.id,
                            document_id=metadata.get(DATA_POINT_FQN_METADATA_KEY),
                            document_hash=metadata.get(DATA_POINT_HASH_METADATA_KEY),
                        )
                    )
                if len(document_vector_points) > MAX_SCROLL_LIMIT:
                    stop = True
                    break
            if next_offset is None:
                stop = True
            else:
                offset = next_offset
        logger.debug(
            f"[Qdrant] Listing {len(document_vector_points)} document vector points for collection {collection_name}"
        )
        return document_vector_points
