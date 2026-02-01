# Phase 9: Performance Optimization

**Duration:** Week 12 | **Priority:** P2 | **Status:** completed

## Context Links

- [Parent Plan](plan.md)
- [All Previous Phases](plan.md#phase-overview)
- [Architecture Report](research/researcher-architecture-report.md)

## Overview

Production performance tuning: multi-level caching, batch processing, parallel execution, rate limiting, and resource optimization. Final phase to ensure production readiness.

## Key Insights

Current state:
- Model gateway has basic LRU caching (50 instances)
- No query-level caching
- No embedding caching
- Sequential pipeline execution where parallel possible
- No rate limiting for external APIs

Target state:
- Multi-level caching (query, retrieval, embedding, LLM)
- Semantic cache for similar queries
- Batch embedding operations
- Parallel retriever execution
- Intelligent rate limiting with backoff

## Requirements

### Functional
- Query Cache: Exact match and semantic similarity caching
- Retrieval Cache: Cache retrieval results with invalidation
- Embedding Cache: Persistent embedding storage
- Batch Processing: Batch embedding and LLM calls
- Rate Limiting: Respect API limits with smart queuing

### Non-Functional
- Cache hit rate > 60% for repeated workloads
- Query latency < 2s (p95)
- Cost reduction > 30% via caching
- Zero data loss on cache failures

## Architecture

### Module Structure
```
backend/modules/
├── cache/
│   ├── __init__.py
│   ├── multi-level-cache.py
│   ├── query-cache.py
│   ├── semantic-cache.py
│   ├── retrieval-cache.py
│   ├── embedding-cache.py
│   └── cache-invalidation.py
└── optimization/
    ├── __init__.py
    ├── batch-processor.py
    ├── parallel-executor.py
    ├── rate-limiter.py
    └── resource-manager.py
```

### Caching Flow
```
User Query
    ↓
Query Cache → Hit? → Return cached response
    ↓ Miss
Semantic Cache → Similar query found? → Return adapted response
    ↓ Miss
Embedding Cache → Cached embedding? → Use cached
    ↓ Miss → Compute → Cache
Retrieval Cache → Cached results? → Use cached
    ↓ Miss → Retrieve → Cache
LLM Cache → Cached response? → Use cached
    ↓ Miss → Generate → Cache
    ↓
Store in Query Cache → Return response
```

## Related Code Files

### Files to Reference
- `backend/modules/model_gateway/model_gateway.py` - Existing caching
- `backend/modules/orchestration/pipeline/pipeline-executor.py` - Parallel execution
- `backend/settings.py` - Configuration

### Files to Create
- `backend/modules/cache/__init__.py`
- `backend/modules/cache/multi-level-cache.py`
- `backend/modules/cache/query-cache.py`
- `backend/modules/cache/semantic-cache.py`
- `backend/modules/cache/retrieval-cache.py`
- `backend/modules/cache/embedding-cache.py`
- `backend/modules/cache/cache-invalidation.py`
- `backend/modules/optimization/__init__.py`
- `backend/modules/optimization/batch-processor.py`
- `backend/modules/optimization/parallel-executor.py`
- `backend/modules/optimization/rate-limiter.py`
- `backend/modules/optimization/resource-manager.py`
- `tests/modules/cache/test_caching.py`
- `tests/modules/optimization/test_batching.py`

## Implementation Steps

### Task 9.1: Multi-Level Caching (Days 1-3)

1. Create `cache/query-cache.py`:
```python
class QueryCache:
    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        ttl_seconds: int = 3600,
        max_size: int = 10000
    ):
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.local_cache = LRUCache(maxsize=max_size)

    def _make_key(self, query: str, collection: str, config_hash: str) -> str:
        content = f"{query}:{collection}:{config_hash}"
        return f"qcache:{hashlib.sha256(content.encode()).hexdigest()}"

    async def get(
        self, query: str, collection: str, config_hash: str
    ) -> Optional[Dict]:
        key = self._make_key(query, collection, config_hash)

        # Check local cache first
        if key in self.local_cache:
            return self.local_cache[key]

        # Check Redis
        if self.redis:
            cached = await self.redis.get(key)
            if cached:
                result = json.loads(cached)
                self.local_cache[key] = result  # Populate local
                return result

        return None

    async def set(
        self, query: str, collection: str, config_hash: str, result: Dict
    ):
        key = self._make_key(query, collection, config_hash)

        # Store in local cache
        self.local_cache[key] = result

        # Store in Redis
        if self.redis:
            await self.redis.setex(key, self.ttl, json.dumps(result, default=str))
```

2. Create `cache/semantic-cache.py`:
```python
class SemanticCache:
    def __init__(
        self,
        embeddings: Embeddings,
        similarity_threshold: float = 0.95,
        max_entries: int = 5000
    ):
        self.embeddings = embeddings
        self.threshold = similarity_threshold
        self.cache: Dict[str, Tuple[List[float], Dict]] = {}  # query -> (embedding, result)
        self.index = None  # FAISS or similar for fast lookup

    async def get_similar(
        self, query: str, collection: str
    ) -> Optional[Tuple[str, Dict, float]]:
        """Find semantically similar cached query"""
        query_embedding = await self.embeddings.aembed_query(query)

        if self.index is None:
            return None

        # Search for similar
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k=1
        )

        if len(indices[0]) > 0 and distances[0][0] <= (1 - self.threshold):
            idx = indices[0][0]
            cached_query, cached_result = list(self.cache.items())[idx]
            similarity = 1 - distances[0][0]
            return cached_query, cached_result[1], similarity

        return None

    async def add(self, query: str, collection: str, result: Dict):
        """Add query result to semantic cache"""
        embedding = await self.embeddings.aembed_query(query)
        self.cache[query] = (embedding, result)
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.cache:
            return
        embeddings = np.array([v[0] for v in self.cache.values()]).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
```

3. Create `cache/embedding-cache.py`:
```python
class EmbeddingCache:
    def __init__(self, storage_path: str = "data/embedding_cache"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, List[float]] = {}

    def _text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    async def get(self, text: str) -> Optional[List[float]]:
        text_hash = self._text_hash(text)

        # Memory cache
        if text_hash in self.memory_cache:
            return self.memory_cache[text_hash]

        # Disk cache
        cache_file = self.storage_path / f"{text_hash}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file).tolist()
            self.memory_cache[text_hash] = embedding
            return embedding

        return None

    async def set(self, text: str, embedding: List[float]):
        text_hash = self._text_hash(text)
        self.memory_cache[text_hash] = embedding
        np.save(self.storage_path / f"{text_hash}.npy", np.array(embedding))

    async def get_batch(self, texts: List[str]) -> Tuple[List[List[float]], List[int]]:
        """Get cached embeddings, return (embeddings, missing_indices)"""
        embeddings = []
        missing = []
        for i, text in enumerate(texts):
            cached = await self.get(text)
            if cached:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                missing.append(i)
        return embeddings, missing
```

4. Create `cache/multi-level-cache.py`:
```python
class MultiLevelCache:
    def __init__(self, config: CacheConfig):
        self.query_cache = QueryCache(
            redis_client=config.redis_client,
            ttl_seconds=config.query_ttl
        )
        self.semantic_cache = SemanticCache(
            embeddings=config.embeddings,
            similarity_threshold=config.semantic_threshold
        )
        self.retrieval_cache = RetrievalCache(
            ttl_seconds=config.retrieval_ttl
        )
        self.embedding_cache = EmbeddingCache(
            storage_path=config.embedding_cache_path
        )
        self.metrics = CacheMetrics()

    async def get_or_compute(
        self,
        query: str,
        collection: str,
        config_hash: str,
        compute_fn: Callable
    ) -> Tuple[Dict, CacheHitInfo]:
        # Level 1: Exact query cache
        cached = await self.query_cache.get(query, collection, config_hash)
        if cached:
            self.metrics.record_hit("query")
            return cached, CacheHitInfo(level="query", exact=True)

        # Level 2: Semantic cache
        similar = await self.semantic_cache.get_similar(query, collection)
        if similar:
            cached_query, result, similarity = similar
            self.metrics.record_hit("semantic")
            return result, CacheHitInfo(
                level="semantic", exact=False, similarity=similarity
            )

        # Cache miss - compute
        self.metrics.record_miss()
        result = await compute_fn()

        # Store in caches
        await self.query_cache.set(query, collection, config_hash, result)
        await self.semantic_cache.add(query, collection, result)

        return result, CacheHitInfo(level="none")
```

### Task 9.2: Batch Processing (Day 4)

1. Create `optimization/batch-processor.py`:
```python
class BatchProcessor:
    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: int = 50
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.pending: Dict[str, List[Tuple[Any, asyncio.Future]]] = defaultdict(list)
        self.locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def process_with_batching(
        self,
        key: str,
        item: Any,
        batch_fn: Callable[[List[Any]], Awaitable[List[Any]]]
    ) -> Any:
        """Add item to batch, wait for batch processing, return result"""
        future = asyncio.Future()

        async with self.locks[key]:
            self.pending[key].append((item, future))

            if len(self.pending[key]) >= self.max_batch_size:
                await self._process_batch(key, batch_fn)
            else:
                # Schedule batch processing after wait time
                asyncio.create_task(self._delayed_process(key, batch_fn))

        return await future

    async def _delayed_process(self, key: str, batch_fn: Callable):
        await asyncio.sleep(self.max_wait_ms / 1000)
        async with self.locks[key]:
            if self.pending[key]:
                await self._process_batch(key, batch_fn)

    async def _process_batch(self, key: str, batch_fn: Callable):
        items_and_futures = self.pending[key]
        self.pending[key] = []

        items = [item for item, _ in items_and_futures]
        futures = [future for _, future in items_and_futures]

        try:
            results = await batch_fn(items)
            for future, result in zip(futures, results):
                future.set_result(result)
        except Exception as e:
            for future in futures:
                future.set_exception(e)

# Usage with embeddings
class BatchedEmbedder:
    def __init__(self, embedder: Embeddings, batch_processor: BatchProcessor):
        self.embedder = embedder
        self.processor = batch_processor

    async def embed_query(self, text: str) -> List[float]:
        return await self.processor.process_with_batching(
            "embed",
            text,
            self.embedder.aembed_documents
        )
```

### Task 9.3: Parallel Execution (Day 5)

1. Create `optimization/parallel-executor.py`:
```python
class ParallelExecutor:
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def execute_parallel(
        self,
        tasks: List[Tuple[Callable, Dict]],
        timeout_sec: float = 30.0
    ) -> List[Any]:
        """Execute tasks in parallel with concurrency limit"""
        async def run_with_semaphore(fn, kwargs):
            async with self.semaphore:
                return await fn(**kwargs)

        try:
            async with async_timeout.timeout(timeout_sec):
                results = await asyncio.gather(
                    *[run_with_semaphore(fn, kwargs) for fn, kwargs in tasks],
                    return_exceptions=True
                )
                return results
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Parallel execution timed out")

    async def execute_with_fallback(
        self,
        primary: Tuple[Callable, Dict],
        fallbacks: List[Tuple[Callable, Dict]],
        timeout_sec: float = 10.0
    ) -> Any:
        """Try primary, fall back to alternatives on failure"""
        try:
            async with async_timeout.timeout(timeout_sec):
                return await primary[0](**primary[1])
        except Exception as e:
            logger.warning(f"Primary failed: {e}, trying fallbacks")

        for fallback_fn, fallback_kwargs in fallbacks:
            try:
                async with async_timeout.timeout(timeout_sec):
                    return await fallback_fn(**fallback_kwargs)
            except Exception as e:
                logger.warning(f"Fallback failed: {e}")

        raise HTTPException(status_code=500, detail="All execution paths failed")
```

### Task 9.4: Rate Limiting (Days 6-7)

1. Create `optimization/rate-limiter.py`:
```python
class TokenBucketRateLimiter:
    def __init__(
        self,
        tokens_per_second: float,
        max_tokens: int,
        name: str = "default"
    ):
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = max_tokens
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        self.name = name
        self.metrics = RateLimiterMetrics(name)

    async def acquire(self, tokens: int = 1, timeout_sec: float = 30.0) -> bool:
        start = time.time()
        while True:
            async with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.metrics.record_success()
                    return True

            if time.time() - start > timeout_sec:
                self.metrics.record_timeout()
                return False

            # Wait and retry
            wait_time = tokens / self.tokens_per_second
            await asyncio.sleep(min(wait_time, 0.1))

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(
            self.max_tokens,
            self.tokens + elapsed * self.tokens_per_second
        )
        self.last_update = now

class AdaptiveRateLimiter:
    def __init__(self, base_rate: float, name: str):
        self.rate_limiter = TokenBucketRateLimiter(base_rate, int(base_rate * 2), name)
        self.success_count = 0
        self.error_count = 0
        self.base_rate = base_rate

    async def acquire_with_backoff(self, tokens: int = 1) -> bool:
        success = await self.rate_limiter.acquire(tokens)
        if success:
            self.success_count += 1
            self._maybe_increase_rate()
        return success

    def record_error(self, error_type: str):
        self.error_count += 1
        if error_type == "rate_limit":
            self._decrease_rate()

    def _decrease_rate(self):
        new_rate = max(self.rate_limiter.tokens_per_second * 0.5, 0.1)
        self.rate_limiter.tokens_per_second = new_rate
        logger.info(f"Rate decreased to {new_rate}/s due to rate limit")

    def _maybe_increase_rate(self):
        if self.success_count % 100 == 0 and self.error_count == 0:
            new_rate = min(
                self.rate_limiter.tokens_per_second * 1.1,
                self.base_rate * 2
            )
            self.rate_limiter.tokens_per_second = new_rate
```

2. Create `optimization/resource-manager.py`:
```python
class ResourceManager:
    def __init__(self, config: ResourceConfig):
        self.rate_limiters = {
            "openai": AdaptiveRateLimiter(config.openai_rate, "openai"),
            "ollama": AdaptiveRateLimiter(config.ollama_rate, "ollama"),
            "embedding": AdaptiveRateLimiter(config.embedding_rate, "embedding"),
        }
        self.cache = MultiLevelCache(config.cache_config)
        self.batch_processor = BatchProcessor(config.batch_size)
        self.parallel_executor = ParallelExecutor(config.max_concurrency)

    async def execute_with_resources(
        self,
        resource_type: str,
        operation: Callable,
        cache_key: Optional[str] = None,
        batch_key: Optional[str] = None
    ) -> Any:
        # Rate limit
        rate_limiter = self.rate_limiters.get(resource_type)
        if rate_limiter:
            await rate_limiter.acquire()

        # Check cache
        if cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                return cached

        # Execute
        try:
            result = await operation()

            # Cache result
            if cache_key:
                await self.cache.set(cache_key, result)

            return result
        except Exception as e:
            if rate_limiter and "rate" in str(e).lower():
                rate_limiter.record_error("rate_limit")
            raise
```

## Todo List

- [x] Create cache module structure
- [x] Implement QueryCache with Redis support
- [x] Implement SemanticCache with FAISS
- [x] Implement EmbeddingCache with persistence
- [x] Implement RetrievalCache with invalidation
- [x] Implement MultiLevelCache coordinator
- [x] Create optimization module structure
- [x] Implement BatchProcessor
- [x] Implement ParallelExecutor
- [x] Implement TokenBucketRateLimiter
- [x] Implement AdaptiveRateLimiter
- [x] Implement ResourceManager
- [ ] Integrate caching with pipeline executor
- [ ] Add cache metrics to observability
- [ ] Write performance benchmarks
- [ ] Write cache invalidation tests

## Success Criteria

- Cache hit rate > 60% for repeated workloads
- Query latency p95 < 2s
- Embedding batch efficiency > 80%
- Zero cache corruption incidents
- Cost savings > 30% from caching

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cache invalidation bugs | Medium | High | Comprehensive testing, TTL fallback |
| Semantic cache false positives | Medium | Medium | Higher threshold, confidence scoring |
| Rate limiter too aggressive | Low | Medium | Adaptive adjustment, monitoring |
| Memory pressure from caching | Medium | Medium | LRU eviction, size limits |

## Security Considerations

- Encrypt sensitive cached data
- Separate cache namespaces per user/tenant
- Rate limit cache invalidation API
- Audit cache access patterns

## Next Steps

After Phase 9:
- Deploy to production with feature flags
- Monitor performance metrics
- Tune cache parameters based on real traffic
- Document operational runbooks
