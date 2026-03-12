# Semantic Cache

NOMYO Router includes a built-in LLM semantic cache that can dramatically reduce inference costs and latency by reusing previous responses — either via exact-match or by finding semantically equivalent questions.

## How It Works

Every incoming request is checked against the cache before being forwarded to an LLM endpoint. On a cache hit the stored response is returned immediately, bypassing inference entirely.

Cache isolation is strict: each combination of **route + model + system prompt** gets its own namespace (a 16-character SHA-256 prefix). A question asked under one system prompt can never leak into a different user's or route's namespace.

Two lookup strategies are available:

| Mode | How it matches | When to use |
|---|---|---|
| **Exact match** | Normalized text hash | When you need 100% identical answers, zero false positives |
| **Semantic match** | Cosine similarity of sentence embeddings | When questions are paraphrased or phrased differently |

In semantic mode the embedding is a weighted average of:
- The **last user message** (70% by default) — captures what is being asked
- A **BM25-weighted summary of the chat history** (30% by default) — provides conversation context

This means "What is the capital of France?" and "Can you tell me the capital city of France?" will hit the same cache entry, but a question in a different conversation context will not.

### MoE Model Bypass

Models with names starting with `moe-` always bypass the cache entirely. Mixture-of-Experts models produce non-deterministic outputs by design and are not suitable for response caching.

### Privacy Protection

Responses that contain personally identifiable information (emails, UUIDs, numeric IDs ≥ 8 digits, or tokens extracted from `[Tags: identity]` lines in the system prompt) are stored **without a semantic embedding**. They remain reachable via exact-match within the same user-specific namespace but are invisible to cross-user semantic search.

If a personalized response is generated with a generic (shared) system prompt, it is skipped entirely — never stored.

---

## Docker Image Variants

| Tag | Semantic cache | Approx. image size | Includes |
|---|---|---|---|
| `latest` | No (exact match only) | ~300 MB | Base router only |
| `latest-semantic` | Yes | ~800 MB | `sentence-transformers`, `torch`, `all-MiniLM-L6-v2` model baked in |

Use the **`latest`** image when you only need exact-match deduplication. It has no heavy ML dependencies.

Use **`latest-semantic`** when you want to catch paraphrased or semantically equivalent questions. The `all-MiniLM-L6-v2` model is baked into the image at build time so no internet access is needed at runtime.

> **Note:** If you set `cache_similarity < 1.0` but use the `latest` image, the router falls back to exact-match caching and logs a warning. It will not error.

Build locally:

```bash
# Lean image (exact match only)
docker build -t nomyo-router .

# Semantic image (~500 MB larger)
docker build --build-arg SEMANTIC_CACHE=true -t nomyo-router:semantic .
```

---

## Dependencies

### Lean image / bare-metal (exact match)

No extra dependencies beyond the base `requirements.txt`. The `semantic-llm-cache` library provides exact-match storage with a no-op embedding provider.

### Semantic image / bare-metal (semantic match)

Additional heavy packages are required:

| Package | Purpose | Approx. size |
|---|---|---|
| `sentence-transformers` | Sentence embedding model wrapper | ~100 MB |
| `torch` (CPU) | PyTorch inference backend | ~700 MB |
| `numpy` | Vector math for weighted mean embeddings | ~20 MB |

Install for bare-metal semantic mode:

```bash
pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
```

The `all-MiniLM-L6-v2` model (~90 MB) is downloaded on first use (or baked in when using the `:semantic` Docker image).

---

## Configuration

All cache settings live in `config.yaml` and can be overridden with environment variables prefixed `NOMYO_ROUTER_`.

### Minimal — enable exact-match with in-memory backend

```yaml
# config.yaml
cache_enabled: true
```

### Full reference

```yaml
# config.yaml

# Master switch — set to true to enable the cache
cache_enabled: false

# Storage backend: "memory" | "sqlite" | "redis"
# memory  — fast, in-process, lost on restart
# sqlite  — persistent single-file, good for single-instance deployments
# redis   — persistent, shared across multiple router instances
cache_backend: memory

# Cosine similarity threshold for semantic matching
# 1.0  = exact match only (no sentence-transformers required)
# 0.95 = very close paraphrases only (recommended starting point)
# 0.85 = broader matching, higher false-positive risk
# Requires :semantic Docker image (or sentence-transformers installed)
cache_similarity: 1.0

# Time-to-live in seconds; null or omit to cache forever
cache_ttl: 3600

# SQLite backend: path to cache database file
cache_db_path: llm_cache.db

# Redis backend: connection URL
cache_redis_url: redis://localhost:6379/0

# Weight of chat-history embedding in the combined query vector (0.0–1.0)
# 0.3 = 30% history context, 70% last user message (default)
# Increase if your use case is highly conversation-dependent
cache_history_weight: 0.3
```

### Environment variable overrides

```bash
NOMYO_ROUTER_CACHE_ENABLED=true
NOMYO_ROUTER_CACHE_BACKEND=sqlite
NOMYO_ROUTER_CACHE_SIMILARITY=0.95
NOMYO_ROUTER_CACHE_TTL=7200
NOMYO_ROUTER_CACHE_DB_PATH=/data/llm_cache.db
NOMYO_ROUTER_CACHE_REDIS_URL=redis://redis:6379/0
NOMYO_ROUTER_CACHE_HISTORY_WEIGHT=0.3
```

### Per-request opt-in

The cache is opted in per-request via the `nomyo.cache` field in the request body. The global `cache_enabled` setting must also be `true`.

```json
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "nomyo": {
    "cache": true
  }
}
```

---

## Backend Comparison

| | `memory` | `sqlite` | `redis` |
|---|---|---|---|
| Persistence | No (lost on restart) | Yes | Yes |
| Multi-instance | No | No | Yes |
| Setup required | None | None | Redis server |
| Recommended for | Development, testing | Single-instance production | Multi-instance / HA production |

### SQLite — persistent single-instance

```yaml
cache_enabled: true
cache_backend: sqlite
cache_db_path: /data/llm_cache.db
cache_ttl: 86400   # 24 hours
```

Mount the path in Docker:

```bash
docker run -d \
  -v /path/to/data:/data \
  -e NOMYO_ROUTER_CACHE_ENABLED=true \
  -e NOMYO_ROUTER_CACHE_BACKEND=sqlite \
  -e NOMYO_ROUTER_CACHE_DB_PATH=/data/llm_cache.db \
  ghcr.io/nomyo-ai/nomyo-router:latest
```

### Redis — distributed / multi-instance

```yaml
cache_enabled: true
cache_backend: redis
cache_redis_url: redis://redis:6379/0
cache_similarity: 0.95
cache_ttl: 3600
```

Docker Compose example:

```yaml
services:
  nomyo-router:
    image: ghcr.io/nomyo-ai/nomyo-router:latest-semantic
    ports:
      - "12434:12434"
    environment:
      NOMYO_ROUTER_CACHE_ENABLED: "true"
      NOMYO_ROUTER_CACHE_BACKEND: redis
      NOMYO_ROUTER_CACHE_REDIS_URL: redis://redis:6379/0
      NOMYO_ROUTER_CACHE_SIMILARITY: "0.95"
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

---

## Code Examples

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:12434/v1",
    api_key="unused",
)

# First call — cache miss, hits the LLM
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_body={"nomyo": {"cache": True}},
)
print(response.choices[0].message.content)

# Second call (exact same question) — cache hit, returned instantly
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    extra_body={"nomyo": {"cache": True}},
)
print(response.choices[0].message.content)

# With semantic mode enabled, this also hits the cache:
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me the capital city of France"}],
    extra_body={"nomyo": {"cache": True}},
)
print(response.choices[0].message.content)
```

### Python (raw HTTP / httpx)

```python
import httpx

payload = {
    "model": "llama3.2",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum entanglement in one sentence."},
    ],
    "nomyo": {"cache": True},
}

resp = httpx.post("http://localhost:12434/v1/chat/completions", json=payload)
print(resp.json()["choices"][0]["message"]["content"])
```

### curl

```bash
curl -s http://localhost:12434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "nomyo": {"cache": true}
  }' | jq .choices[0].message.content
```

### Check cache statistics

```bash
curl -s http://localhost:12434/api/cache/stats | jq .
```

```json
{
  "hits": 42,
  "misses": 18,
  "hit_rate": 0.7,
  "semantic": true,
  "backend": "sqlite",
  "similarity_threshold": 0.95,
  "history_weight": 0.3
}
```

### Clear the cache

```bash
curl -X POST http://localhost:12434/api/cache/clear
```

---

## Use Cases Where Semantic Caching Helps Most

### FAQ and support bots

Users ask the same questions in slightly different ways. "How do I reset my password?", "I forgot my password", "password reset instructions" — all semantically equivalent. Semantic caching collapses these into a single LLM call.

**Recommended settings:** `cache_similarity: 0.90–0.95`, `cache_backend: sqlite` or `redis`, long TTL.

### Document Q&A / RAG pipelines

When the same document set is queried repeatedly by many users, common factual questions ("What is the contract start date?", "When does this contract begin?") hit the cache across users — as long as the system prompt is not user-personalized.

**Recommended settings:** `cache_similarity: 0.92`, `cache_backend: redis` for multi-user deployments.

### Code generation with repeated patterns

Development tools that generate boilerplate, explain errors, or convert code often receive nearly-identical prompts. Caching prevents re-running expensive large-model inference for the same error message phrased differently.

**Recommended settings:** `cache_similarity: 0.93`, `cache_backend: sqlite`.

### High-traffic chatbots with shared context

Public-facing bots where many users share the same system prompt (same product, same persona). Generic factual answers can be safely reused across users. The privacy guard ensures personalized responses are never shared.

**Recommended settings:** `cache_similarity: 0.90–0.95`, `cache_backend: redis`.

### Batch processing / data pipelines

When running LLM inference over large datasets with many near-duplicate or repeated inputs, caching can reduce inference calls dramatically and make pipelines idempotent.

**Recommended settings:** `cache_similarity: 0.95`, `cache_backend: sqlite`, `cache_ttl: null` (cache forever for deterministic batch runs).

---

## Tuning the Similarity Threshold

| `cache_similarity` | Behaviour | Risk |
|---|---|---|
| `1.0` | Exact match only | No false positives |
| `0.97` | Catches minor typos and punctuation differences | Very low |
| `0.95` | Catches clear paraphrases (recommended default) | Low |
| `0.90` | Catches broader rewordings | Moderate — may match different intents |
| `< 0.85` | Very aggressive matching | High false-positive rate, not recommended |

Start at `0.95` and lower gradually while monitoring cache hit rate via `/api/cache/stats`.

---

## Limitations

- **Streaming responses** cached from a non-streaming request are served as a single chunk (not token-by-token). This is indistinguishable to most clients.
- **MoE models** (`moe-*` prefix) are never cached.
- **Token counts** are not recorded for cache hits (the LLM was not called).
- The `memory` backend does not survive a router restart.
- Semantic search requires the `:semantic` image (or local `sentence-transformers` install); without it, `cache_similarity < 1.0` silently falls back to exact match.
