# Configuration Guide

## Configuration File

The NOMYO Router is configured via a YAML file (default: `config.yaml`). This file defines the Ollama endpoints, connection limits, and API keys.

### Basic Configuration

```yaml
# config.yaml
endpoints:
  - http://localhost:11434
  - http://ollama-server:11434

# Maximum concurrent connections *per endpoint‑model pair*
max_concurrent_connections: 2

# Optional router-level API key to secure the router and dashboard (leave blank to disable)
nomyo-router-api-key: ""
```

### Complete Example

```yaml
# config.yaml
endpoints:
  - http://192.168.0.50:11434
  - http://192.168.0.51:11434
  - http://192.168.0.52:11434
  - https://api.openai.com/v1

# Maximum concurrent connections *per endpoint‑model pair* (equals to OLLAMA_NUM_PARALLEL)
max_concurrent_connections: 2

# Optional router-level API key to secure the router and dashboard (leave blank to disable)
nomyo-router-api-key: ""

# API keys for remote endpoints
# Set an environment variable like OPENAI_KEY
# Confirm endpoints are exactly as in endpoints block
api_keys:
  "http://192.168.0.50:11434": "ollama"
  "http://192.168.0.51:11434": "ollama"
  "http://192.168.0.52:11434": "ollama"
  "https://api.openai.com/v1": "${OPENAI_KEY}"
```

## Configuration Options

### `endpoints`

**Type**: `list[str]`

**Description**: List of Ollama endpoint URLs. Can include both Ollama endpoints (`http://host:11434`) and OpenAI-compatible endpoints (`https://api.openai.com/v1`).

**Examples**:
```yaml
endpoints:
  - http://localhost:11434
  - http://ollama1:11434
  - http://ollama2:11434
  - https://api.openai.com/v1
  - https://api.anthropic.com/v1
```

**Notes**:
- Ollama endpoints use the standard `/api/` prefix
- OpenAI-compatible endpoints use `/v1` prefix
- The router automatically detects endpoint type based on URL pattern

### `max_concurrent_connections`

**Type**: `int`

**Default**: `1`

**Description**: Maximum number of concurrent connections allowed per endpoint-model pair. This corresponds to Ollama's `OLLAMA_NUM_PARALLEL` setting.

**Example**:
```yaml
max_concurrent_connections: 4
```

**Notes**:
- This setting controls how many requests can be processed simultaneously for a specific model on a specific endpoint
- When this limit is reached, the router will route requests to other endpoints with available capacity
- Higher values allow more parallel requests but may increase memory usage

### `router_api_key`

**Type**: `str` (optional)

**Description**: Shared secret that gates access to the NOMYO Router APIs and dashboard. When set, clients must send `Authorization: Bearer <key>` or an `api_key` query parameter.

**Example**:
```yaml
nomyo-router-api-key: "super-secret-value"
```

**Notes**:
- Leave this blank or omit it to disable router-level authentication.
- You can also set the `NOMYO_ROUTER_API_KEY` environment variable to avoid storing the key in plain text.

### `api_keys`

**Type**: `dict[str, str]`

**Description**: Mapping of endpoint URLs to API keys. Used for authenticating with remote endpoints.

**Example**:
```yaml
api_keys:
  "http://192.168.0.50:11434": "ollama"
  "https://api.openai.com/v1": "${OPENAI_KEY}"
```

**Environment Variables**:
- API keys can reference environment variables using `${VAR_NAME}` syntax
- The router will expand these references at startup
- Example: `${OPENAI_KEY}` will be replaced with the value of the `OPENAI_KEY` environment variable

## Environment Variables

### `NOMYO_ROUTER_CONFIG_PATH`

**Description**: Path to the configuration file. If not set, defaults to `config.yaml` in the current working directory.

**Example**:
```bash
export NOMYO_ROUTER_CONFIG_PATH=/etc/nomyo-router/config.yaml
```

### `NOMYO_ROUTER_DB_PATH`

**Description**: Path to the SQLite database file for storing token counts. If not set, defaults to `token_counts.db` in the current working directory.

**Example**:
```bash
export NOMYO_ROUTER_DB_PATH=/var/lib/nomyo-router/token_counts.db
```

### `NOMYO_ROUTER_API_KEY`

**Description**: Router-level API key. When set, all router endpoints and the dashboard require this key via `Authorization: Bearer <key>` or the `api_key` query parameter.

**Example**:
```bash
export NOMYO_ROUTER_API_KEY=your_router_api_key
```

### API-Specific Keys

You can set API keys directly as environment variables:

```bash
export OPENAI_KEY=your_openai_api_key
export ANTHROPIC_KEY=your_anthropic_api_key
```

## Configuration Best Practices

### Multiple Ollama Instances

For a cluster of Ollama instances:

```yaml
endpoints:
  - http://ollama-worker1:11434
  - http://ollama-worker2:11434
  - http://ollama-worker3:11434

max_concurrent_connections: 2
```

**Recommendation**: Set `max_concurrent_connections` to match your Ollama instances' `OLLAMA_NUM_PARALLEL` setting.

### Mixed Endpoints

Combining Ollama and OpenAI endpoints:

```yaml
endpoints:
  - http://localhost:11434
  - https://api.openai.com/v1

api_keys:
  "https://api.openai.com/v1": "${OPENAI_KEY}"
```

**Note**: The router will automatically route requests based on model availability across all endpoints.

### High Availability

For production deployments:

```yaml
endpoints:
  - http://ollama-primary:11434
  - http://ollama-secondary:11434
  - http://ollama-tertiary:11434

max_concurrent_connections: 3
```

**Recommendation**: Use multiple endpoints for redundancy and load distribution.

## Semantic LLM Cache

NOMYO Router can cache LLM responses and serve them directly — skipping endpoint selection, model load, and token generation entirely.

### How it works

1. On every cacheable request (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`) the cache is checked **before** choosing an endpoint.
2. On a **cache hit** the stored response is returned immediately as a single chunk (streaming or non-streaming — both work).
3. On a **cache miss** the request is forwarded normally. The response is stored in the cache after it completes.
4. **MOE requests** (`moe-*` model prefix) always bypass the cache.
5. **Token counts** are never recorded for cache hits.

### Cache key strategy

| Signal | How matched |
|---|---|
| `model + system_prompt` | Exact — hard context isolation per deployment |
| BM25-weighted embedding of chat history | Semantic — conversation context signal |
| Embedding of last user message | Semantic — the actual question |

The two semantic vectors are combined as a weighted mean (tuned by `cache_history_weight`) before cosine similarity comparison, staying at a single 384-dimensional vector compatible with the library's storage format.

### Quick start — exact match (lean image)

```yaml
cache_enabled: true
cache_backend: sqlite    # persists across restarts
cache_similarity: 1.0   # exact match only, no sentence-transformers needed
cache_ttl: 3600
```

### Quick start — semantic matching (:semantic image)

```yaml
cache_enabled: true
cache_backend: sqlite
cache_similarity: 0.90   # hit if ≥90% cosine similarity
cache_ttl: 3600
cache_history_weight: 0.3
```

Pull the semantic image:
```bash
docker pull ghcr.io/nomyo-ai/nomyo-router:latest-semantic
```

### Cache configuration options

#### `cache_enabled`

**Type**: `bool` | **Default**: `false`

Enable or disable the cache. All other cache settings are ignored when `false`.

#### `cache_backend`

**Type**: `str` | **Default**: `"memory"`

| Value | Description | Persists | Multi-replica |
|---|---|---|---|
| `memory` | In-process LRU dict | ❌ | ❌ |
| `sqlite` | File-based via `aiosqlite` | ✅ | ❌ |
| `redis` | Redis via `redis.asyncio` | ✅ | ✅ |

Use `redis` when running multiple router replicas behind a load balancer — all replicas share one warm cache.

#### `cache_similarity`

**Type**: `float` | **Default**: `1.0`

Cosine similarity threshold. `1.0` means exact match only (no embedding model needed). Values below `1.0` enable semantic matching, which requires the `:semantic` Docker image tag.

Recommended starting value for semantic mode: `0.90`.

#### `cache_ttl`

**Type**: `int | null` | **Default**: `3600`

Time-to-live for cache entries in seconds. Remove the key or set to `null` to cache forever.

#### `cache_db_path`

**Type**: `str` | **Default**: `"llm_cache.db"`

Path to the SQLite cache database. Only used when `cache_backend: sqlite`.

#### `cache_redis_url`

**Type**: `str` | **Default**: `"redis://localhost:6379/0"`

Redis connection URL. Only used when `cache_backend: redis`.

#### `cache_history_weight`

**Type**: `float` | **Default**: `0.3`

Weight of the BM25-weighted chat-history embedding in the combined cache key vector. `0.3` means the history contributes 30% and the final user message contributes 70% of the similarity signal. Only used when `cache_similarity < 1.0`.

### Cache management endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/cache/stats` | `GET` | Hit/miss counters, hit rate, current config |
| `/api/cache/invalidate` | `POST` | Clear all cache entries and reset counters |

```bash
# Check cache performance
curl http://localhost:12434/api/cache/stats

# Clear the cache
curl -X POST http://localhost:12434/api/cache/invalidate
```

Example stats response:
```json
{
  "enabled": true,
  "hits": 1547,
  "misses": 892,
  "hit_rate": 0.634,
  "semantic": true,
  "backend": "sqlite",
  "similarity_threshold": 0.9,
  "history_weight": 0.3
}
```

### Docker image variants

| Tag | Semantic cache | Image size |
|---|---|---|
| `latest` | ❌ exact match only | ~300 MB |
| `latest-semantic` | ✅ sentence-transformers + model pre-baked | ~800 MB |

Build locally:
```bash
# Lean (exact match)
docker build -t nomyo-router .

# Semantic (~500 MB larger, all-MiniLM-L6-v2 model baked in)
docker build --build-arg SEMANTIC_CACHE=true -t nomyo-router:semantic .
```

## Configuration Validation

The router validates the configuration at startup:

1. **Endpoint URLs**: Must be valid URLs
2. **API Keys**: Must be strings (can reference environment variables)
3. **Connection Limits**: Must be positive integers

If the configuration is invalid, the router will exit with an error message.

## Dynamic Configuration

The configuration is loaded at startup and cannot be changed without restarting the router. For production deployments, consider:

1. Using a configuration management system
2. Implementing a rolling restart strategy
3. Using environment variables for sensitive data

## Example Configurations

See the [examples](examples/) directory for ready-to-use configuration examples.


### Using the router API key

When `router_api_key`/`NOMYO_ROUTER_API_KEY` is set, clients must send it on every request:
- Header (recommended): Authorization: Bearer <router_key>
- Query param (fallback): ?api_key=<router_key>

Example:
```bash
curl -H "Authorization: Bearer $NOMYO_ROUTER_API_KEY" http://localhost:12434/api/tags
```
