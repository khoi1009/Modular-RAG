# Modular RAG Implementation Patterns - Research Report

**Date:** 2026-01-31 | **Focus:** Architectural patterns for production RAG systems

---

## 1. HyDE (Hypothetical Document Embeddings)

**Pattern:** Zero-shot prompt LLM to generate 5 hypothetical documents matching query intent, embed each, average vectors.

**Implementation:**
- Query → LLM generates synthetic documents capturing textual patterns
- Each document embedded in shared vector space with real docs
- Single aggregated embedding for retrieval
- Works with any embedding model

**Benefits:** Query-time optimization; improves semantic match; no static per-document overhead.

**Alternative:** HyPE (per-document optimization) - static generation, 42% precision/45% recall improvement but no query-time cost.

**Python Integration:**
```python
# LangChain built-in support
from langchain.retrievers import HyDERetriever
retriever = HyDERetriever(llm=llm, base_embeddings=embeddings)
```

---

## 2. Self-Reflective RAG / CRAG Patterns

**Corrective RAG (CRAG) Workflow:**
- Retrieve documents → Relevance grader assesses quality
- **If confident (>threshold):** Refine via knowledge strips, filter irrelevant content
- **If uncertain/below threshold:** Fallback to web search for supplemental context
- Generate with refined docs

**Self-RAG Pattern:**
- Retrieval module: Grade doc relevance per query
- Generation module: Self-grade output, critique statements
- Adaptive retrieval: Decide whether to retrieve more docs mid-generation

**Python Stack:** LangGraph (workflow orchestration), LangChain (components), Tavily (web fallback).

**Async Pattern:**
```python
# LangGraph state machine
async def evaluate_documents(state):
    docs = state["documents"]
    grades = await asyncio.gather(
        *[grade_relevance(doc) for doc in docs]
    )
    state["filtered_docs"] = [d for d, g in zip(docs, grades) if g > threshold]
```

---

## 3. Query Decomposition & Multi-Hop Reasoning

**Strategy:** Break complex queries into sub-queries → retrieve per subquery → reason over answers.

**Tree-Based Decomposition (Latest):**
- Parse core queries, identify known/unknown entities
- Build tree hierarchy with consensus mechanism
- Adaptive leaf determination prevents over-decomposition
- Constrains reasoning space, reduces error propagation

**Multi-Hop Best Practices:**
1. **Hybrid retrieval:** Vector + keyword search
2. **Metadata filtering:** Pre-filter by relevance indicators
3. **Reranking:** Re-score retrieved docs for final set
4. **Structure-aware chunking:** Parent-document patterns

**Layering Strategy:**
- Base: Hybrid retrieval → Metadata/reranking
- Layer 1: Summarization, query expansion (HyDE)
- Layer 2: Multi-step reasoning, decomposition
- Layer 3: Grounding (CRAG), retrieval-based memory

```python
async def decompose_and_retrieve(query):
    subqueries = await decompose(query)
    results = await asyncio.gather(
        *[retrieve(sq) for sq in subqueries]
    )
    return synthesize_answers(results)
```

---

## 4. Hallucination Detection

**Four Primary Techniques:**

| Method | Accuracy | Precision | Recall | Cost |
|--------|----------|-----------|--------|------|
| LLM Prompt-based | >75% | High | Med | Low-Med |
| BERT Stochastic | High | Med | Highest | Medium |
| Semantic Similarity | Medium | High | Medium | Low |
| Token Similarity | Basic | Low | Low | Very Low |

**Reference-Free Models (No Ground Truth Needed):**
- **LLM-as-Judge:** General purpose
- **Prometheus, Lynx:** Specialized scorers
- **HHEM:** Hallucination-specific
- **TLM:** Trustworthy LM checks

**Advanced Frameworks:**
- **ReDeEP:** Mechanistic interpretability - decouples external context vs. parametric knowledge
- **MetaRAG:** Metamorphic testing, real-time unsupervised, black-box, no internals needed
- **LettuceDetect:** Handles context window constraints, computationally efficient

**Implementation Pattern:**
```python
# Post-generation check
async def detect_hallucination(generated, context):
    judge_score = await llm_judge(generated, context)
    groundedness = await check_semantic_similarity(generated, context)
    if judge_score < 0.6 or groundedness < 0.5:
        flag_hallucination(generated)
```

---

## 5. Ollama Integration Best Practices

**Architecture Pattern:**
- Ollama runs locally (no auth, no API costs, full privacy)
- Async calls via langchain-ollama with event loop overlap
- Embeddings + generation in single unified flow
- ChromaDB or Weaviate for vector storage

**Async RAG Pipeline:**
```python
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import asyncio

class AsyncRAGPipeline:
    def __init__(self, model="llama2", embed_model="nomic-embed-text"):
        self.llm = OllamaLLM(model=model)
        self.embeddings = OllamaEmbeddings(model=embed_model)

    async def embed_documents(self, docs):
        return await asyncio.gather(
            *[self.embeddings.embed_query(d) for d in docs]
        )

    async def generate_with_context(self, query, context):
        # Non-blocking - keeps system responsive
        return await self.llm.agenerate(prompt=self._build_prompt(query, context))
```

**Key Libraries:**
- `langchain_ollama` - Ollama bindings
- `langchain_community` - Vector stores (ChromaDB, Weaviate)
- `chromadb` - Lightweight vector DB
- `faiss` - Fast similarity search
- `asyncio` - Async orchestration

**Performance Considerations:**
- Load models into GPU memory once (persist across queries)
- Use smaller models (7B-13B) for responsive latency
- Batch embedding operations for throughput
- Non-blocking I/O for web/disk operations

**Model Selection:**
- **Llama 2/3.1 (8B):** Balanced speed/quality
- **Mistral (7B):** Fast reasoning
- **Nomic Embed Text:** Fast local embeddings
- **PHI-2:** Lightweight instruction following

---

## Integration Summary

**Recommended Stack:**
```
Query → HyDE expansion → Decompose (if multi-hop)
  ↓
Async retrieve (vector + keyword) → Rerank
  ↓
CRAG relevance check → Fallback web search if needed
  ↓
Self-reflection critique → Hallucination detection
  ↓
Generate (Ollama) → Score output → Return to user
```

**Async Pattern:**
Use `asyncio.gather()` for parallel: retrieval, embedding, reranking. Non-blocking LLM calls keep UI responsive.

**Local-First Advantage:**
Ollama + ChromaDB = Zero external dependencies, privacy, offline capability, deterministic behavior.

---

## Unresolved Questions

1. **Optimal decomposition depth:** How many levels for tree-based decomposition before diminishing returns?
2. **Consensus thresholds:** What consensus % required in tree validation across different domains?
3. **Model sizing:** For Ollama, quantization level (Q4/Q5) vs. latency trade-off in production?
4. **Fallback strategy:** When to trigger web search in CRAG - fixed threshold or adaptive per-query?

---

**Sources:**

- [Haystack - Hypothetical Document Embeddings (HyDE)](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde)
- [Zilliz - Better RAG with HyDE](https://zilliz.com/learn/improve-rag-and-information-retrieval-with-hyde-hypothetical-document-embeddings)
- [LangChain - Corrective RAG (CRAG)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- [LangChain Blog - Self-Reflective RAG with LangGraph](https://blog.langchain.com/agentic-rag-with-langgraph/)
- [Haystack - Query Decomposition](https://haystack.deepset.ai/blog/query-decomposition)
- [NVIDIA RAG Blueprint - Query Decomposition](https://docs.nvidia.com/rag/latest/query_decomposition.html)
- [AWS - Detect Hallucinations for RAG Systems](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/)
- [Cleanlab - Hallucination Detection Benchmarking](https://cleanlab.ai/blog/rag-tlm-hallucination-benchmarking/)
- [Microsoft Cosmos DB - Build RAG with LangChain and Ollama](https://devblogs.microsoft.com/cosmosdb/build-a-rag-application-with-langchain-and-local-llms-powered-by-ollama/)
- [DataCamp - RAG with Llama 3.1, Ollama, and LangChain](https://www.datacamp.com/tutorial/llama-3-1-rag)
