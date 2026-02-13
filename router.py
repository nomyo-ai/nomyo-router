"""
title: NOMYO Router - an Ollama Proxy with Endpoint:Model aware routing
author: alpha-nerd-nomyo
author_url: https://github.com/nomyo-ai
version: 0.6
license: AGPL
"""
# -------------------------------------------------------------
import orjson, time, asyncio, yaml, ollama, openai, os, re, aiohttp, ssl, random, base64, io, enhance, secrets
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass
from datetime import datetime, timezone
from pathlib import Path

# Directory containing static files (relative to this script)
STATIC_DIR = Path(__file__).parent / "static"
from typing import Dict, Set, List, Optional
from urllib.parse import urlparse, parse_qsl, urlencode
from fastapi import FastAPI, Request, HTTPException
from fastapi_sse import sse_handler
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, JSONResponse, Response, HTMLResponse, RedirectResponse
from pydantic import Field
from pydantic_settings import BaseSettings
from collections import defaultdict
from PIL import Image

# ------------------------------------------------------------------
# In‑memory caches
# ------------------------------------------------------------------
# Successful results are cached for 300s
_models_cache: dict[str, tuple[Set[str], float]] = {}
_loaded_models_cache: dict[str, tuple[Set[str], float]] = {}
# Transient errors are cached separately per concern so that a failure
# in one path does not poison the other.
_available_error_cache: dict[str, float] = {}
_loaded_error_cache: dict[str, float] = {}

# ------------------------------------------------------------------
# Cache locks
# ------------------------------------------------------------------
_models_cache_lock = asyncio.Lock()
_loaded_models_cache_lock = asyncio.Lock()
_available_error_cache_lock = asyncio.Lock()
_loaded_error_cache_lock = asyncio.Lock()

# ------------------------------------------------------------------
# In-flight request tracking (prevents cache stampede)
# ------------------------------------------------------------------
_inflight_available_models: dict[str, asyncio.Task] = {}
_inflight_loaded_models: dict[str, asyncio.Task] = {}
_inflight_lock = asyncio.Lock()

# ------------------------------------------------------------------
# Queues
# ------------------------------------------------------------------
_subscribers: Set[asyncio.Queue] = set()
_subscribers_lock = asyncio.Lock()
token_queue: asyncio.Queue[tuple[str, str, int, int]] = asyncio.Queue()

# -------------------------------------------------------------
# Secret handling
# -------------------------------------------------------------
def _mask_secrets(text: str) -> str:
    """
    Mask common API key patterns to avoid leaking secrets in logs or error payloads.
    """
    if not text:
        return text
    # OpenAI-style keys (sk-...) and generic "api key" mentions
    text = re.sub(r"sk-[A-Za-z0-9]{4}[A-Za-z0-9_-]*", "sk-***redacted***", text)
    text = re.sub(r"(?i)(api[-_ ]key\\s*[:=]\\s*)([^\\s]+)", r"\\1***redacted***", text)
    return text

# ------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------
app_state = {
    "session": None,
    "connector": None,
}
token_worker_task: asyncio.Task | None = None
flush_task: asyncio.Task | None = None

# ------------------------------------------------------------------
# Token Count Buffer (for write-behind pattern)
# ------------------------------------------------------------------
# Structure: {endpoint: {model: (input_tokens, output_tokens)}}
token_buffer: dict[str, dict[str, tuple[int, int]]] = defaultdict(lambda: defaultdict(lambda: (0, 0)))
# Time series buffer with timestamp
time_series_buffer: list[dict[str, int | str]] = []
# Lock to protect buffer access from race conditions
buffer_lock = asyncio.Lock()

# Configuration for periodic flushing
FLUSH_INTERVAL = 10  # seconds

# -------------------------------------------------------------
# 1. Configuration loader
# -------------------------------------------------------------
class Config(BaseSettings):
    # List of Ollama endpoints
    endpoints: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:11434",
        ]
    )
    # List of llama-server endpoints (OpenAI-compatible with /v1/models status info)
    llama_server_endpoints: List[str] = Field(default_factory=list)
    # Max concurrent connections per endpoint‑model pair, see OLLAMA_NUM_PARALLEL
    max_concurrent_connections: int = 1

    api_keys: Dict[str, str] = Field(default_factory=dict)
    # Optional router-level API key used to gate access to this service and dashboard
    router_api_key: Optional[str] = Field(default=None, env="NOMYO_ROUTER_API_KEY")

    # Database configuration
    db_path: str = Field(default=os.getenv("NOMYO_ROUTER_DB_PATH", "token_counts.db"))

    class Config:
        # Load from `config.yaml` first, then from env variables
        env_prefix = "NOMYO_ROUTER_"
        yaml_file = Path("config.yaml")  # relative to cwd

    @classmethod
    def _expand_env_refs(cls, obj):
        """Recursively replace `${VAR}` with os.getenv('VAR')."""
        if isinstance(obj, dict):
            return {k: cls._expand_env_refs(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._expand_env_refs(v) for v in obj]
        if isinstance(obj, str):
            # Only expand if it is exactly ${VAR}
            m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", obj)
            if m:
                return os.getenv(m.group(1), "")
        return obj

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load the YAML file and create the Config instance."""
        if path.exists():
            with path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                cleaned = cls._expand_env_refs(data)
                if isinstance(cleaned, dict):
                    # Accept hyphenated config key and map it to the field name
                    key_aliases = [
                        # canonical field name
                        "router_api_key",
                        # lowercase, hyphen/underscore variants
                        "nomyo-router-api-key",
                        "nomyo_router_api_key",
                        "nomyo-router_api_key",
                        "nomyo_router-api_key",
                        # uppercase env-style variants
                        "NOMYO-ROUTER_API_KEY",
                        "NOMYO_ROUTER_API_KEY",
                    ]
                    for alias in key_aliases:
                        if alias in cleaned:
                            cleaned["router_api_key"] = cleaned.get("router_api_key", cleaned.pop(alias))
                            break
                    # If not present in YAML (or empty), fall back to env var explicitly
                    if not cleaned.get("router_api_key"):
                        env_key = os.getenv("NOMYO_ROUTER_API_KEY")
                        if env_key:
                            cleaned["router_api_key"] = env_key
            return cls(**cleaned)
        return cls()

def _config_path_from_env() -> Path:
    """
    Resolve the configuration file path. Defaults to `config.yaml`
    in the current working directory unless NOMYO_ROUTER_CONFIG_PATH
    is set.
    """
    candidate = os.getenv("NOMYO_ROUTER_CONFIG_PATH")
    if candidate:
        return Path(candidate).expanduser()
    return Path("config.yaml")

from db import TokenDatabase


# Create the global config object – it will be overwritten on startup
config = Config.from_yaml(_config_path_from_env())

# -------------------------------------------------------------
# 2. FastAPI application
# -------------------------------------------------------------
app = FastAPI()
sse_handler.app = app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
default_headers={
    "HTTP-Referer": "https://nomyo.ai",
    "X-Title": "NOMYO Router",
    }
        
# -------------------------------------------------------------
# Router-level authentication (optional)
# -------------------------------------------------------------
def _extract_router_api_key(request: Request) -> Optional[str]:
    """
    Extract the provided router API key from the Authorization header or `api_key`
    query parameter. The middleware uses this to gate access to API routes when
    a router_api_key is configured.
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        key = auth_header.split(" ", 1)[1].strip()
        if key:  # Ensure key is not empty
            return key
    query_key = request.query_params.get("api_key")
    if query_key:
        return query_key
    return None


def _strip_api_key_from_scope(request: Request) -> None:
    """
    Remove api_key from the ASGI scope query string to avoid leaking it in logs.
    """
    scope = request.scope
    raw_qs = scope.get("query_string", b"")
    if not raw_qs:
        return
    params = parse_qsl(raw_qs.decode("utf-8"), keep_blank_values=True)
    filtered = [(k, v) for (k, v) in params if k != "api_key"]
    scope["query_string"] = urlencode(filtered).encode("utf-8")


@app.middleware("http")
async def enforce_router_api_key(request: Request, call_next):
    """
    Enforce the optional NOMYO Router API key for all non-static requests.
    When `config.router_api_key` is set, clients must supply the key either in
    the Authorization header (`Bearer <key>`) or as `api_key` query parameter.
    """
    expected_key = config.router_api_key
    if not expected_key or request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path.startswith("/static") or path in {"/", "/favicon.ico"}:
        return await call_next(request)

    provided_key = _extract_router_api_key(request)
    # Strip the api_key query param from scope so access logs do not leak it
    _strip_api_key_from_scope(request)
    if provided_key is None:
        # No key provided but authentication is required - return 401
        headers = {}
        if "/api/" in path and path != "/api/usage-stream":
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Authorization, Content-Type",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            }
        return JSONResponse(
            content={"detail": "Missing NOMYO Router API key"},
            status_code=401,
            headers=headers,
        )

    if not secrets.compare_digest(str(provided_key), str(expected_key)):
        return JSONResponse(
            content={"detail": "Invalid NOMYO Router API key"},
            status_code=403,
        )

    response = await call_next(request)
    # Add CORS headers for authenticated API requests
    if "/api/" in path and path != "/api/usage-stream":
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response
        
# -------------------------------------------------------------
# 3. Global state: per‑endpoint per‑model active connection counters
# -------------------------------------------------------------
usage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
token_usage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
usage_lock = asyncio.Lock()  # protects access to usage_counts
token_usage_lock = asyncio.Lock()

# Database instance
db: "TokenDatabase" = None

# -------------------------------------------------------------
# 4. Helperfunctions 
# -------------------------------------------------------------
def _is_fresh(cached_at: float, ttl: int) -> bool:
    return (time.time() - cached_at) < ttl

async def _ensure_success(resp: aiohttp.ClientResponse) -> None:
    if resp.status >= 400:
        text = await resp.text()
        raise HTTPException(status_code=resp.status, detail=_mask_secrets(text))
    
def _format_connection_issue(url: str, error: Exception) -> str:
    """
    Provide a human-friendly error string for connection failures so operators
    know which endpoint and address failed from inside the container.
    """
    parsed = urlparse(url)
    host_hint = parsed.hostname or ""
    port_hint = parsed.port or ""

    if isinstance(error, aiohttp.ClientConnectorError):
        resolved_host = getattr(error, "host", host_hint) or host_hint or "?"
        resolved_port = getattr(error, "port", port_hint) or port_hint or "?"
        parts = [
            f"Failed to connect to {url} (resolved: {resolved_host}:{resolved_port}).",
            "Ensure the endpoint address is reachable from within the container.",
        ]
        if resolved_host in {"localhost", "127.0.0.1"}:
            parts.append(
                "Inside Docker, 'localhost' refers to the container itself; use "
                "'host.docker.internal' or a Docker network alias if the service "
                "runs on the host machine."
            )
        os_error = getattr(error, "os_error", None)
        if isinstance(os_error, OSError):
            errno = getattr(os_error, "errno", None)
            strerror = os_error.strerror or str(os_error)
            if errno is not None or strerror:
                parts.append(f"OS error [{errno}]: {strerror}.")
        elif os_error:
            parts.append(f"OS error: {os_error}.")
        parts.append(f"Original error: {error}.")
        return " ".join(parts)

    if isinstance(error, asyncio.TimeoutError):
        return (
            f"Timed out waiting for {url}. "
            "The remote endpoint may be offline or slow to respond."
        )

    return f"Error while contacting {url}: {error}"

def _normalize_llama_model_name(name: str) -> str:
    """Extract the model name from a huggingface-style identifier.
    e.g. 'unsloth/gpt-oss-20b-GGUF:F16' -> 'gpt-oss-20b-GGUF'
    """
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    if ":" in name:
        name = name.split(":")[0]
    return name

def _extract_llama_quant(name: str) -> str:
    """Extract the quantization level from a huggingface-style identifier.
    e.g. 'unsloth/gpt-oss-20b-GGUF:Q8_0' -> 'Q8_0'
    Returns empty string if no quant suffix is present.
    """
    if ":" in name:
        return name.rsplit(":", 1)[1]
    return ""

def _is_llama_model_loaded(item: dict) -> bool:
    """Return True if a llama-server /v1/models item has status 'loaded'.
    Handles both dict format ({"value": "loaded"}) and plain string ("loaded")."""
    status = item.get("status")
    if isinstance(status, dict):
        return status.get("value") == "loaded"
    if isinstance(status, str):
        return status == "loaded"
    return False

def is_ext_openai_endpoint(endpoint: str) -> bool:
    """
    Determine if an endpoint is an external OpenAI-compatible endpoint (not Ollama or llama-server).
    
    Returns True for:
    - External services like OpenAI.com, Groq, etc.
    
    Returns False for:
    - Ollama endpoints (without /v1, or with /v1 but default port 11434)
    - llama-server endpoints (explicitly configured in llama_server_endpoints)
    """
    # Check if it's a llama-server endpoint (has /v1 and is in the configured list)
    if endpoint in config.llama_server_endpoints:
        return False
    
    if "/v1" not in endpoint:
        return False
    
    base_endpoint = endpoint.replace('/v1', '')
    if base_endpoint in config.endpoints:
        return False  # It's Ollama's /v1
    
    # Check for default Ollama port
    if ':11434' in endpoint:
        return False  # It's Ollama
    
    return True  # It's an external OpenAI endpoint

def is_openai_compatible(endpoint: str) -> bool:
    """
    Return True if the endpoint speaks the OpenAI API (not native Ollama).
    This includes external OpenAI endpoints AND llama-server endpoints.
    """
    return "/v1" in endpoint or endpoint in config.llama_server_endpoints

async def token_worker() -> None:
    try:
        while True:
            endpoint, model, prompt, comp = await token_queue.get()
            # Calculate timestamp once before acquiring lock
            now = datetime.now(tz=timezone.utc)
            timestamp = int(datetime(now.year, now.month, now.day, now.hour, now.minute, tzinfo=timezone.utc).timestamp())

            # Accumulate counts in memory buffer (protected by lock)
            async with buffer_lock:
                token_buffer[endpoint][model] = (
                    token_buffer[endpoint].get(model, (0, 0))[0] + prompt,
                    token_buffer[endpoint].get(model, (0, 0))[1] + comp
                )

                # Add to time series buffer with timestamp (UTC)
                time_series_buffer.append({
                    'endpoint': endpoint,
                    'model': model,
                    'input_tokens': prompt,
                    'output_tokens': comp,
                    'total_tokens': prompt + comp,
                    'timestamp': timestamp
                })

            # Update in-memory counts for immediate reporting
            async with token_usage_lock:
                token_usage_counts[endpoint][model] += (prompt + comp)
                await publish_snapshot()
    except asyncio.CancelledError:
        # Gracefully handle task cancellation during shutdown
        print("[token_worker] Task cancelled, processing remaining queue items...")
        # Process any remaining items in the queue before exiting
        while not token_queue.empty():
            try:
                endpoint, model, prompt, comp = token_queue.get_nowait()
                # Calculate timestamp once before acquiring lock
                now = datetime.now(tz=timezone.utc)
                timestamp = int(datetime(now.year, now.month, now.day, now.hour, now.minute, tzinfo=timezone.utc).timestamp())

                async with buffer_lock:
                    token_buffer[endpoint][model] = (
                        token_buffer[endpoint].get(model, (0, 0))[0] + prompt,
                        token_buffer[endpoint].get(model, (0, 0))[1] + comp
                    )
                    time_series_buffer.append({
                        'endpoint': endpoint,
                        'model': model,
                        'input_tokens': prompt,
                        'output_tokens': comp,
                        'total_tokens': prompt + comp,
                        'timestamp': timestamp
                    })
                async with token_usage_lock:
                    token_usage_counts[endpoint][model] += (prompt + comp)
                    await publish_snapshot()
            except asyncio.QueueEmpty:
                break
        print("[token_worker] Task cancelled, remaining items processed.")
        raise

async def flush_buffer() -> None:
    """Periodically flush accumulated token counts to the database."""
    try:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL)

            # Flush token counts and time series (protected by lock)
            async with buffer_lock:
                if token_buffer:
                    # Copy buffer before releasing lock for DB operation
                    buffer_copy = {ep: dict(models) for ep, models in token_buffer.items()}
                    token_buffer.clear()
                else:
                    buffer_copy = None

                if time_series_buffer:
                    ts_copy = list(time_series_buffer)
                    time_series_buffer.clear()
                else:
                    ts_copy = None

            # Perform DB operations outside the lock to avoid blocking
            if buffer_copy:
                await db.update_batched_counts(buffer_copy)
            if ts_copy:
                await db.add_batched_time_series(ts_copy)
    except asyncio.CancelledError:
        # Gracefully handle task cancellation during shutdown
        print("[flush_buffer] Task cancelled, flushing remaining buffers...")
        # Flush any remaining data before exiting
        try:
            async with buffer_lock:
                if token_buffer:
                    buffer_copy = {ep: dict(models) for ep, models in token_buffer.items()}
                    token_buffer.clear()
                else:
                    buffer_copy = None
                if time_series_buffer:
                    ts_copy = list(time_series_buffer)
                    time_series_buffer.clear()
                else:
                    ts_copy = None
            if buffer_copy:
                await db.update_batched_counts(buffer_copy)
            if ts_copy:
                await db.add_batched_time_series(ts_copy)
            print("[flush_buffer] Task cancelled, remaining buffers flushed.")
        except Exception as e:
            print(f"[flush_buffer] Error during shutdown flush: {e}")
        raise

async def flush_remaining_buffers() -> None:
    """
    Flush any in-memory buffers to the database on shutdown.
    This is designed to be safely invoked during shutdown and should not raise.
    """
    try:
        flushed_entries = 0
        async with buffer_lock:
            if token_buffer:
                buffer_copy = {ep: dict(models) for ep, models in token_buffer.items()}
                flushed_entries += sum(len(v) for v in token_buffer.values())
                token_buffer.clear()
            else:
                buffer_copy = None
            if time_series_buffer:
                ts_copy = list(time_series_buffer)
                flushed_entries += len(time_series_buffer)
                time_series_buffer.clear()
            else:
                ts_copy = None
        # Perform DB operations outside the lock
        if buffer_copy:
            await db.update_batched_counts(buffer_copy)
        if ts_copy:
            await db.add_batched_time_series(ts_copy)
        if flushed_entries:
            print(f"[shutdown] Flushed {flushed_entries} in-memory entries to DB on shutdown.")
        else:
            print("[shutdown] No in-memory entries to flush on shutdown.")
    except Exception as e:
        # Do not raise during shutdown – log and continue teardown
        print(f"[shutdown] Error flushing remaining buffers: {e}")

class fetch:
    async def _fetch_available_models_internal(endpoint: str, api_key: Optional[str] = None) -> Set[str]:
        """
        Internal function that performs the actual HTTP request to fetch available models.
        This is called by available_models() after checking caches and in-flight requests.
        """
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}

        if endpoint in config.llama_server_endpoints and "/v1" not in endpoint:
            endpoint_url = f"{endpoint}/v1/models"
            key = "data"
        elif "/v1" in endpoint or endpoint in config.llama_server_endpoints:
            endpoint_url = f"{endpoint}/models"
            key = "data"
        else:
            endpoint_url = f"{endpoint}/api/tags"
            key = "models"

        client: aiohttp.ClientSession = app_state["session"]
        try:
            async with client.get(endpoint_url, headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()

                items = data.get(key, [])
                models = {item.get("id") or item.get("name") for item in items if item.get("id") or item.get("name")}

                # Update cache with lock protection
                async with _models_cache_lock:
                    _models_cache[endpoint] = (models, time.time())
                return models
        except Exception as e:
            # Treat any error as if the endpoint offers no models
            message = _format_connection_issue(endpoint_url, e)
            print(f"[fetch.available_models] {message}")
            # Update error cache with lock protection
            async with _available_error_cache_lock:
                _available_error_cache[endpoint] = time.time()
            return set()

    async def _refresh_available_models(endpoint: str, api_key: Optional[str] = None) -> None:
        """
        Background task to refresh available models cache without blocking the caller.
        Used for stale-while-revalidate pattern.
        """
        try:
            await fetch._fetch_available_models_internal(endpoint, api_key)
        except Exception as e:
            # Silently fail - cache will remain stale but functional
            print(f"[fetch._refresh_available_models] Background refresh failed for {endpoint}: {e}")

    async def available_models(endpoint: str, api_key: Optional[str] = None) -> Set[str]:
        """
        Query <endpoint>/api/tags and return a set of all model names that the
        endpoint *advertises* (i.e. is capable of serving).  This endpoint lists
        every model that is installed on the Ollama instance, regardless of
        whether the model is currently loaded into memory.

        Uses request coalescing to prevent cache stampede: if multiple requests
        arrive when cache is expired, only one actual HTTP request is made.

        Uses stale-while-revalidate: when the cache is between 300-600s old,
        the stale data is returned immediately while a background refresh runs.
        This prevents model blackouts caused by transient timeouts.

        If the request fails (e.g. timeout, 5xx, or malformed response), an empty
        set is returned.
        """
        # Check models cache with lock protection
        async with _models_cache_lock:
            if endpoint in _models_cache:
                models, cached_at = _models_cache[endpoint]

                # FRESH: < 300s old - return immediately
                if _is_fresh(cached_at, 300):
                    return models

                # STALE: 300-600s old - return stale data and refresh in background
                if _is_fresh(cached_at, 600):
                    asyncio.create_task(fetch._refresh_available_models(endpoint, api_key))
                    return models  # Return stale data immediately

                # EXPIRED: > 600s old - too stale, must refresh synchronously
                del _models_cache[endpoint]

        # Check error cache with lock protection
        async with _available_error_cache_lock:
            if endpoint in _available_error_cache:
                if _is_fresh(_available_error_cache[endpoint], 30):
                    # Still within the short error TTL – pretend nothing is available
                    return set()
                # Error expired – remove it
                del _available_error_cache[endpoint]

        # Request coalescing: check if another request is already fetching this endpoint
        async with _inflight_lock:
            if endpoint in _inflight_available_models:
                # Another request is already fetching - wait for it
                task = _inflight_available_models[endpoint]
            else:
                # Create new fetch task
                task = asyncio.create_task(fetch._fetch_available_models_internal(endpoint, api_key))
                _inflight_available_models[endpoint] = task

        try:
            # Wait for the fetch to complete (either ours or another request's)
            result = await task
            return result
        finally:
            # Clean up in-flight tracking (only if we created it)
            async with _inflight_lock:
                if _inflight_available_models.get(endpoint) == task:
                    _inflight_available_models.pop(endpoint, None)


    async def _fetch_loaded_models_internal(endpoint: str) -> Set[str]:
        """
        Internal function that performs the actual HTTP request to fetch loaded models.
        This is called by loaded_models() after checking caches and in-flight requests.
        
        For Ollama endpoints: queries /api/ps and returns model names
        For llama-server endpoints: queries /v1/models and filters for status.value == "loaded"
        """
        client: aiohttp.ClientSession = app_state["session"]
        
        # Check if this is a llama-server endpoint
        if endpoint in config.llama_server_endpoints:
            # Query /v1/models for llama-server
            try:
                async with client.get(f"{endpoint}/models") as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
                
                # Filter for loaded models only
                items = data.get("data", [])
                models = {
                    item.get("id")
                    for item in items
                    if item.get("id") and _is_llama_model_loaded(item)
                }

                # Update cache with lock protection
                async with _loaded_models_cache_lock:
                    _loaded_models_cache[endpoint] = (models, time.time())
                return models
            except Exception as e:
                # If anything goes wrong we simply assume the endpoint has no models
                message = _format_connection_issue(f"{endpoint}/models", e)
                print(f"[fetch.loaded_models] {message}")
                return set()
        else:
            # Original Ollama /api/ps logic
            try:
                async with client.get(f"{endpoint}/api/ps") as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
                # The response format is:
                #   {"models": [{"name": "model1"}, {"name": "model2"}]}
                models = {m.get("name") for m in data.get("models", []) if m.get("name")}

                # Update cache with lock protection
                async with _loaded_models_cache_lock:
                    _loaded_models_cache[endpoint] = (models, time.time())
                return models
            except Exception as e:
                # If anything goes wrong we simply assume the endpoint has no models
                message = _format_connection_issue(f"{endpoint}/api/ps", e)
                print(f"[fetch.loaded_models] {message}")
                return set()

    async def _refresh_loaded_models(endpoint: str) -> None:
        """
        Background task to refresh loaded models cache without blocking the caller.
        Used for stale-while-revalidate pattern.
        """
        try:
            await fetch._fetch_loaded_models_internal(endpoint)
        except Exception as e:
            # Silently fail - cache will remain stale but functional
            print(f"[fetch._refresh_loaded_models] Background refresh failed for {endpoint}: {e}")

    async def loaded_models(endpoint: str) -> Set[str]:
        """
        Query <endpoint>/api/ps and return a set of model names that are currently
        loaded on that endpoint. If the request fails (e.g. timeout, 5xx), an empty
        set is returned.

        Uses request coalescing to prevent cache stampede and stale-while-revalidate
        to serve requests immediately even when cache is stale (refreshing in background).
        """
        if is_ext_openai_endpoint(endpoint):
            return set()

        # Check loaded models cache with lock protection
        async with _loaded_models_cache_lock:
            if endpoint in _loaded_models_cache:
                models, cached_at = _loaded_models_cache[endpoint]

                # FRESH: < 10s old - return immediately
                if _is_fresh(cached_at, 30):
                    return models

                # STALE: 10-60s old - return stale data and refresh in background
                if _is_fresh(cached_at, 60):
                    # Kick off background refresh (fire-and-forget)
                    asyncio.create_task(fetch._refresh_loaded_models(endpoint))
                    return models  # Return stale data immediately

                # EXPIRED: > 60s old - too stale, must refresh synchronously
                del _loaded_models_cache[endpoint]

        # Check error cache with lock protection
        async with _loaded_error_cache_lock:
            if endpoint in _loaded_error_cache:
                if _is_fresh(_loaded_error_cache[endpoint], 30):
                    return set()
                # Error expired - remove it
                del _loaded_error_cache[endpoint]

        # Request coalescing: check if another request is already fetching this endpoint
        async with _inflight_lock:
            if endpoint in _inflight_loaded_models:
                # Another request is already fetching - wait for it
                task = _inflight_loaded_models[endpoint]
            else:
                # Create new fetch task
                task = asyncio.create_task(fetch._fetch_loaded_models_internal(endpoint))
                _inflight_loaded_models[endpoint] = task

        try:
            # Wait for the fetch to complete (either ours or another request's)
            result = await task
            return result
        finally:
            # Clean up in-flight tracking (only if we created it)
            async with _inflight_lock:
                if _inflight_loaded_models.get(endpoint) == task:
                    _inflight_loaded_models.pop(endpoint, None)

    async def endpoint_details(endpoint: str, route: str, detail: str, api_key: Optional[str] = None, skip_error_cache: bool = False) -> List[dict]:
        """
        Query <endpoint>/<route> to fetch <detail> and return a List of dicts with details
        for the corresponding Ollama endpoint. If the request fails we respond with "N/A" for detail.

        When ``skip_error_cache`` is False (the default), the call is short-circuited
        if the endpoint recently failed (recorded in ``_available_error_cache``).
        Pass ``skip_error_cache=True`` from health-check routes that must always probe.
        """
        # Fast-fail if the endpoint is known to be down (unless caller opts out)
        if not skip_error_cache:
            async with _available_error_cache_lock:
                if endpoint in _available_error_cache:
                    if _is_fresh(_available_error_cache[endpoint], 30):
                        return []

        client: aiohttp.ClientSession = app_state["session"]
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}

        request_url = f"{endpoint}{route}"
        try:
            async with client.get(request_url, headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()
            detail = data.get(detail, [])
            return detail
        except Exception as e:
            # If anything goes wrong we cannot reply details
            message = _format_connection_issue(request_url, e)
            print(f"[fetch.endpoint_details] {message}")
            # Record failure so subsequent calls skip this endpoint briefly
            async with _available_error_cache_lock:
                _available_error_cache[endpoint] = time.time()
            return []

def ep2base(ep):
    if "/v1" in ep:
        base_url = ep
    else:
        base_url = ep+"/v1"
    return base_url

def dedupe_on_keys(dicts, key_fields):
    """
    Helper function to deduplicate endpoint details based on given dict keys.
    """
    seen = set()
    out = []
    for d in dicts:
        # Build a tuple of the values for the chosen keys
        key = tuple(d.get(k) for k in key_fields)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out

async def increment_usage(endpoint: str, model: str) -> None:
    async with usage_lock:
        usage_counts[endpoint][model] += 1
        await publish_snapshot()

async def decrement_usage(endpoint: str, model: str) -> None:
    async with usage_lock:
        # Avoid negative counts
        current = usage_counts[endpoint].get(model, 0)
        if current > 0:
            usage_counts[endpoint][model] = current - 1
        # Optionally, clean up zero entries
        if usage_counts[endpoint].get(model, 0) == 0:
            usage_counts[endpoint].pop(model, None)
        #if not usage_counts[endpoint]:
        #    usage_counts.pop(endpoint, None)
        await publish_snapshot()

async def _make_chat_request(endpoint: str, model: str, messages: list, tools=None, stream: bool = False, think: bool = False, format=None, options=None, keep_alive: str = None) -> ollama.ChatResponse:
    """
    Helper function to make a chat request to a specific endpoint.
    Handles endpoint selection, client creation, usage tracking, and request execution.
    """
    use_openai = is_openai_compatible(endpoint)
    if use_openai:
        if ":latest" in model:
            model = model.split(":latest")[0]
        if messages:
            messages = transform_images_to_data_urls(messages)
            messages = transform_tool_calls_to_openai(messages)
        params = {
            "messages": messages,
            "model": model,
        }
        optional_params = {
            "tools": tools,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "max_tokens": options.get("num_predict") if options and "num_predict" in options else None,
            "frequency_penalty": options.get("frequency_penalty") if options and "frequency_penalty" in options else None,
            "presence_penalty": options.get("presence_penalty") if options and "presence_penalty" in options else None,
            "seed": options.get("seed") if options and "seed" in options else None,
            "stop": options.get("stop") if options and "stop" in options else None,
            "top_p": options.get("top_p") if options and "top_p" in options else None,
            "temperature": options.get("temperature") if options and "temperature" in options else None,
            "response_format": {"type": "json_schema", "json_schema": format} if format is not None else None
        }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=ep2base(endpoint), default_headers=default_headers, api_key=config.api_keys.get(endpoint, "no-key"))
    else:
        client = ollama.AsyncClient(host=endpoint)

    await increment_usage(endpoint, model)

    try:
        if use_openai:
            start_ts = time.perf_counter()
            response = await oclient.chat.completions.create(**params)
            if stream:
                # For streaming, we need to collect all chunks
                chunks = []
                tc_acc = {}  # accumulate tool-call deltas
                async for chunk in response:
                    chunks.append(chunk)
                    _accumulate_openai_tc_delta(chunk, tc_acc)
                    if chunk.usage is not None:
                        prompt_tok = chunk.usage.prompt_tokens or 0
                        comp_tok = chunk.usage.completion_tokens or 0
                        if prompt_tok != 0 or comp_tok != 0:
                            await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                # Convert to Ollama format
                if chunks:
                    response = rechunk.openai_chat_completion2ollama(chunks[-1], stream, start_ts)
                    # Inject fully-accumulated tool calls into the final response
                    if tc_acc and response.message:
                        response.message.tool_calls = _build_ollama_tool_calls(tc_acc)
            else:
                prompt_tok = response.usage.prompt_tokens or 0
                comp_tok = response.usage.completion_tokens or 0
                if prompt_tok != 0 or comp_tok != 0:
                    await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                response = rechunk.openai_chat_completion2ollama(response, stream, start_ts)
        else:
            response = await client.chat(model=model, messages=messages, tools=tools, stream=stream, think=think, format=format, options=options, keep_alive=keep_alive)
            if stream:
                # For streaming, collect all chunks
                chunks = []
                async for chunk in response:
                    chunks.append(chunk)
                    prompt_tok = chunk.prompt_eval_count or 0
                    comp_tok = chunk.eval_count or 0
                    if prompt_tok != 0 or comp_tok != 0:
                        await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                if chunks:
                    response = chunks[-1]
            else:
                prompt_tok = response.prompt_eval_count or 0
                comp_tok = response.eval_count or 0
                if prompt_tok != 0 or comp_tok != 0:
                    await token_queue.put((endpoint, model, prompt_tok, comp_tok))

        return response
    finally:
        await decrement_usage(endpoint, model)

def get_last_user_content(messages):
    """
    Given a list of dicts (e.g., messages from an API),
    return the 'content' of the last dict whose 'role' is 'user'.
    If no such dict exists, return None.
    """
    # Reverse iterate so we stop at the first match
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None

async def _make_moe_requests(model: str, messages: list, tools=None, think: bool = False, format=None, options=None, keep_alive: str = None) -> ollama.ChatResponse:
    """
    Helper function to make MOE (Multiple Opinions Ensemble) requests.
    Generates 3 responses, 3 critiques, and returns the final selected response.
    """
    query = get_last_user_content(messages)
    if not query:
        raise ValueError("No user query found in messages")

    if options is None:
        options = {}
    options["temperature"] = 1

    moe_reqs = []

    # Generate 3 responses
    response1_endpoint = await choose_endpoint(model)
    response1_task = asyncio.create_task(_make_chat_request(response1_endpoint, model, messages, tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    response2_endpoint = await choose_endpoint(model)
    response2_task = asyncio.create_task(_make_chat_request(response2_endpoint, model, messages, tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    response3_endpoint = await choose_endpoint(model)
    response3_task = asyncio.create_task(_make_chat_request(response3_endpoint, model, messages, tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    responses = await asyncio.gather(response1_task, response2_task, response3_task)

    for n, r in enumerate(responses):
        moe_req = enhance.moe(query, n, r.message.content)
        moe_reqs.append(moe_req)

    # Generate 3 critiques
    critique1_endpoint = await choose_endpoint(model)
    critique1_task = asyncio.create_task(_make_chat_request(critique1_endpoint, model, [{"role": "user", "content": moe_reqs[0]}], tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    critique2_endpoint = await choose_endpoint(model)
    critique2_task = asyncio.create_task(_make_chat_request(critique2_endpoint, model, [{"role": "user", "content": moe_reqs[1]}], tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    critique3_endpoint = await choose_endpoint(model)
    critique3_task = asyncio.create_task(_make_chat_request(critique3_endpoint, model, [{"role": "user", "content": moe_reqs[2]}], tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive))
    await asyncio.sleep(0.01)  # Small delay to allow usage count to update

    critiques = await asyncio.gather(critique1_task, critique2_task, critique3_task)

    # Select final response
    m = enhance.moe_select_candidate(query, critiques)

    # Generate final response
    final_endpoint = await choose_endpoint(model)
    return await _make_chat_request(final_endpoint, model, [{"role": "user", "content": m}], tools, stream=False, think=think, format=format, options=options, keep_alive=keep_alive)

def iso8601_ns():
    ns = time.time_ns()
    sec, ns_rem = divmod(ns, 1_000_000_000)
    dt = datetime.fromtimestamp(sec, tz=timezone.utc)
    return (
        f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T"
        f"{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}."
        f"{ns_rem:09d}Z"
    )

def is_base64(image_string):
    try:
        if isinstance(image_string, str) and base64.b64encode(base64.b64decode(image_string)) == image_string.encode():
            return True
    except Exception as e:
        return False

def resize_image_if_needed(image_data):
    try:
        # Check if already data-url
        if image_data.startswith("data:"):
            try:
                header, image_data = image_data.split(",", 1)
            except ValueError:
                pass
        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # Get current size
        width, height = image.size

        # Calculate the new dimensions while maintaining aspect ratio
        if width > 512 or height > 512:
            aspect_ratio = width / height
            if aspect_ratio > 1:  # Width is larger
                new_width = 512
                new_height = int(512 / aspect_ratio)
            else:  # Height is larger
                new_height = 512
                new_width = int(512 * aspect_ratio)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Encode the resized image back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        resized_image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return resized_image_data

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def transform_tool_calls_to_openai(message_list):
    """
    Ensure tool_calls in assistant messages conform to the OpenAI format:
    - Each tool call must have "type": "function"
    - Each tool call must have an "id"
    - arguments must be a JSON string, not a dict
    Also ensure tool-role messages have a tool_call_id.
    """
    # Track generated IDs so tool-role messages can reference them
    last_tool_call_ids = {}
    for msg in message_list:
        role = msg.get("role")
        if role == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                if "type" not in tc:
                    tc["type"] = "function"
                if "id" not in tc:
                    tc["id"] = f"call_{secrets.token_hex(16)}"
                func = tc.get("function", {})
                if isinstance(func.get("arguments"), dict):
                    func["arguments"] = orjson.dumps(func["arguments"]).decode("utf-8")
                # Remember the id for the following tool-role message
                name = func.get("name")
                if name:
                    last_tool_call_ids[name] = tc["id"]
        elif role == "tool":
            if "tool_call_id" not in msg:
                # Try to match by name from a preceding assistant tool_call
                name = msg.get("name") or msg.get("tool_name")
                if name and name in last_tool_call_ids:
                    msg["tool_call_id"] = last_tool_call_ids.pop(name)
    return message_list

def transform_images_to_data_urls(message_list):
    for message in message_list:
        if "images" in message:
            images = message.pop("images")
            if not isinstance(images, list):
                continue
            new_content = []
            for image in images:            #TODO: quality downsize if images are too big to fit into model context window size
                if not is_base64(image):
                    raise ValueError(f"Image string is not a valid base64 encoded string.")
                resized_image = resize_image_if_needed(image)
                if resized_image:
                    data_url = f"data:image/png;base64,{resized_image}"
                    #new_content.append({
                    #    "type": "text",
                    #    "text": ""
                    #})
                    new_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    })
            message["content"] = new_content

    return message_list

def _accumulate_openai_tc_delta(chunk, accumulator: dict) -> None:
    """Accumulate tool_call deltas from a single OpenAI streaming chunk.

    ``accumulator`` is a dict mapping tool-call *index* to
    ``{"id": str, "name": str, "arguments": str}`` where ``arguments``
    is the concatenation of all JSON fragments seen so far.
    """
    if not chunk.choices:
        return
    delta = chunk.choices[0].delta
    tc_deltas = getattr(delta, "tool_calls", None)
    if not tc_deltas:
        return
    for tc in tc_deltas:
        idx = tc.index
        if idx not in accumulator:
            accumulator[idx] = {
                "id": getattr(tc, "id", None) or f"call_{secrets.token_hex(16)}",
                "name": tc.function.name if tc.function else None,
                "arguments": "",
            }
        else:
            if getattr(tc, "id", None):
                accumulator[idx]["id"] = tc.id
            if tc.function and tc.function.name:
                accumulator[idx]["name"] = tc.function.name
        if tc.function and tc.function.arguments:
            accumulator[idx]["arguments"] += tc.function.arguments

def _build_ollama_tool_calls(accumulator: dict) -> list | None:
    """Convert accumulated tool-call data into Ollama-format tool_calls list."""
    if not accumulator:
        return None
    result = []
    for idx in sorted(accumulator.keys()):
        tc = accumulator[idx]
        try:
            args = orjson.loads(tc["arguments"]) if tc["arguments"] else {}
        except (orjson.JSONDecodeError, TypeError):
            args = {}
        result.append(ollama.Message.ToolCall(
            function=ollama.Message.ToolCall.Function(name=tc["name"], arguments=args)
        ))
    return result

class rechunk:
    def openai_chat_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.ChatResponse:
        now = time.perf_counter()
        if chunk.choices == [] and chunk.usage is not None:
            return ollama.ChatResponse(
                model=chunk.model,
                created_at=iso8601_ns(),
                done=True,
                done_reason='stop',
                total_duration=int((now - start_ts) * 1_000_000_000),
                load_duration=100000,
                prompt_eval_count=int(chunk.usage.prompt_tokens),
                prompt_eval_duration=int((now - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)),
                eval_count=int(chunk.usage.completion_tokens),
                eval_duration=int((now - start_ts) * 1_000_000_000),
                message=ollama.Message(role="assistant", content=""),
                )
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        if stream == True:
            thinking = (getattr(with_thinking.delta, "reasoning_content", None) or getattr(with_thinking.delta, "reasoning", None)) if with_thinking else None
            role = chunk.choices[0].delta.role or "assistant"
            content = chunk.choices[0].delta.content or ''
        else:
            thinking = (getattr(with_thinking.message, "reasoning_content", None) or getattr(with_thinking.message, "reasoning", None)) if with_thinking else None
            role = chunk.choices[0].message.role or "assistant"
            content = chunk.choices[0].message.content or ''
        # Convert OpenAI tool_calls to Ollama format
        # In streaming mode, tool_calls arrive as partial deltas across multiple chunks
        # (name only in first delta, arguments as incremental JSON fragments).
        # Callers must accumulate deltas and inject the final result; skip here.
        ollama_tool_calls = None
        if not stream:
            raw_tool_calls = getattr(with_thinking.message, "tool_calls", None) if with_thinking else None
            if raw_tool_calls:
                ollama_tool_calls = []
                for tc in raw_tool_calls:
                    try:
                        args = orjson.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {})
                    except (orjson.JSONDecodeError, TypeError):
                        args = {}
                    ollama_tool_calls.append(ollama.Message.ToolCall(
                        function=ollama.Message.ToolCall.Function(name=tc.function.name, arguments=args)
                    ))
        assistant_msg = ollama.Message(
            role=role,
            content=content,
            thinking=thinking,
            images=None,
            tool_name=None,
            tool_calls=ollama_tool_calls)
        rechunk = ollama.ChatResponse(
            model=chunk.model, 
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason, #if chunk.choices[0].finish_reason is not None else None,
            total_duration=int((now - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            load_duration=100000, 
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((now - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0, 
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((now - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            message=assistant_msg)
        return rechunk
    
    def openai_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.GenerateResponse:
        now = time.perf_counter()
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        thinking = getattr(with_thinking, "reasoning", None) if with_thinking else None
        rechunk = ollama.GenerateResponse(
            model=chunk.model,
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason,
            total_duration=int((now - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            load_duration=10000,
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((now - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0,
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((now - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            response=chunk.choices[0].text or '',
            thinking=thinking)
        return rechunk
    
    def openai_embeddings2ollama(chunk: dict) -> ollama.EmbeddingsResponse:
        rechunk = ollama.EmbeddingsResponse(embedding=chunk.data[0].embedding)
        return rechunk

    def openai_embed2ollama(chunk: dict, model: str) -> ollama.EmbedResponse:
        rechunk = ollama.EmbedResponse(
            model=model,
            created_at=iso8601_ns(),
            done=None,
            done_reason=None,
            total_duration=None,
            load_duration=None,
            prompt_eval_count=None,
            prompt_eval_duration=None,
            eval_count=None,
            eval_duration=None,
            embeddings=[chunk.data[0].embedding])
        return rechunk
    
# ------------------------------------------------------------------
# SSE Helpser
# ------------------------------------------------------------------
async def publish_snapshot():
    # NOTE: This function assumes usage_lock OR token_usage_lock is already held by the caller
    # Create a snapshot without acquiring the lock (caller must hold it)
    snapshot = orjson.dumps({
        "usage_counts": dict(usage_counts),  # Create a copy
        "token_usage_counts": dict(token_usage_counts)
    }, option=orjson.OPT_SORT_KEYS).decode("utf-8")

    # Distribute the snapshot (no lock needed here since we have a copy)
    async with _subscribers_lock:
        for q in _subscribers:
            # If the queue is full, drop the message to avoid back‑pressure.
            if q.full():
                try:
                    await q.get()
                except asyncio.QueueEmpty:
                    pass
            await q.put(snapshot)

async def close_all_sse_queues():
    for q in list(_subscribers):
        # sentinel value that the generator will recognise
        await q.put(None)

# ------------------------------------------------------------------
# Subscriber helpers
# ------------------------------------------------------------------
async def subscribe() -> asyncio.Queue:
    """
    Returns a new Queue that will receive every snapshot.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=10)
    async with _subscribers_lock:
        _subscribers.add(q)
    return q

async def unsubscribe(q: asyncio.Queue):
    async with _subscribers_lock:
        _subscribers.discard(q)

# ------------------------------------------------------------------
# Convenience wrapper – returns the current snapshot (for the proxy)
# ------------------------------------------------------------------
async def get_usage_counts() -> Dict:
    return dict(usage_counts)   # shallow copy

# -------------------------------------------------------------
# 5. Endpoint selection logic (respecting the configurable limit)
# -------------------------------------------------------------
async def choose_endpoint(model: str) -> str:
    """
    Determine which endpoint to use for the given model while respecting
    the `max_concurrent_connections` per endpoint‑model pair **and**
    ensuring that the chosen endpoint actually *advertises* the model.

    The selection algorithm:

    1️⃣  Query every endpoint for its advertised models (`/api/tags`).
    2️⃣  Build a list of endpoints that contain the requested model.
    3️⃣  For those endpoints, find those that have the model loaded
        (`/api/ps`) *and* still have a free slot.
    4️⃣  If none are both loaded and free, fall back to any endpoint
        from the filtered list that simply has a free slot and randomly 
        select one.
    5️⃣  If all are saturated, pick any endpoint from the filtered list
        (the request will queue on that endpoint).
    6️⃣  If no endpoint advertises the model at all, raise an error.
    """
    # 1️⃣  Gather advertised‑model sets for all endpoints concurrently
    #     Include both config.endpoints and config.llama_server_endpoints
    llama_eps_extra = [ep for ep in config.llama_server_endpoints if ep not in config.endpoints]
    all_endpoints = config.endpoints + llama_eps_extra

    tag_tasks = [fetch.available_models(ep) for ep in config.endpoints if not is_openai_compatible(ep)]
    tag_tasks += [fetch.available_models(ep, config.api_keys.get(ep)) for ep in config.endpoints if is_openai_compatible(ep)]
    tag_tasks += [fetch.available_models(ep, config.api_keys.get(ep)) for ep in llama_eps_extra]
    advertised_sets = await asyncio.gather(*tag_tasks)

    # 2️⃣  Filter endpoints that advertise the requested model
    candidate_endpoints = [
        ep for ep, models in zip(all_endpoints, advertised_sets)
        if model in models
    ]

    # 6️⃣
    if not candidate_endpoints:
        if ":latest" in model:  #ollama naming convention not applicable to openai/llama-server
            model_without_latest = model.split(":latest")[0]
            candidate_endpoints = [
                ep for ep, models in zip(all_endpoints, advertised_sets)
                if model_without_latest in models and (is_ext_openai_endpoint(ep) or ep in config.llama_server_endpoints)
            ]
        if not candidate_endpoints:
            # Only add :latest suffix if model doesn't already have a version suffix
            if ":" not in model:
                model = model + ":latest"
            candidate_endpoints = [
                ep for ep, models in zip(all_endpoints, advertised_sets)
                if model in models
            ]
        if not candidate_endpoints:
            raise RuntimeError(
                f"None of the configured endpoints ({', '.join(all_endpoints)}) "
                f"advertise the model '{model}'."
            )
    # 3️⃣  Among the candidates, find those that have the model *loaded*
    #      (concurrently, but only for the filtered list)
    load_tasks = [fetch.loaded_models(ep) for ep in candidate_endpoints]
    loaded_sets = await asyncio.gather(*load_tasks)

    # Protect all reads of usage_counts with the lock
    async with usage_lock:
        # Helper: get current usage count for (endpoint, model)
        def current_usage(ep: str) -> int:
            return usage_counts.get(ep, {}).get(model, 0)

        # 3️⃣ Endpoints that have the model loaded *and* a free slot
        loaded_and_free = [
            ep for ep, models in zip(candidate_endpoints, loaded_sets)
            if model in models and usage_counts.get(ep, {}).get(model, 0) < config.max_concurrent_connections
        ]

        if loaded_and_free:
            # Sort by per-model usage in DESCENDING order to ensure model affinity
            # Endpoints with higher usage (already handling this model) should be preferred
            # until they reach max_concurrent_connections
            loaded_and_free.sort(
                key=lambda ep: -usage_counts.get(ep, {}).get(model, 0)  # Negative for descending order
            )

            # If all endpoints have zero usage for this model, randomize to distribute
            # different models across different endpoints for better resource utilization
            if all(usage_counts.get(ep, {}).get(model, 0) == 0 for ep in loaded_and_free):
                return random.choice(loaded_and_free)

            return loaded_and_free[0]

        # 4️⃣ Endpoints among the candidates that simply have a free slot
        endpoints_with_free_slot = [
            ep for ep in candidate_endpoints
            if usage_counts.get(ep, {}).get(model, 0) < config.max_concurrent_connections
        ]

        if endpoints_with_free_slot:
            # Sort by per-model usage (descending) first to ensure model affinity
            # Even if the model isn't showing as "loaded" in /api/ps yet (e.g., during initial loading),
            # we want to send subsequent requests to the endpoint that already has connections for this model
            # Then by total endpoint usage (ascending) to balance idle endpoints
            endpoints_with_free_slot.sort(
                key=lambda ep: (
                    #-usage_counts.get(ep, {}).get(model, 0),  # Primary: per-model usage (descending - prefer endpoints with connections)
                    sum(usage_counts.get(ep, {}).values())    # Secondary: total endpoint usage (ascending - prefer idle endpoints)
                )
            )

            # If all endpoints have zero usage for this specific model, randomize to distribute
            # different models across different endpoints for better resource utilization
            if all(usage_counts.get(ep, {}).get(model, 0) == 0 for ep in endpoints_with_free_slot):
                return random.choice(endpoints_with_free_slot)

            return endpoints_with_free_slot[0]

        # 5️⃣ All candidate endpoints are saturated – pick one with lowest usages count (will queue)
        ep = min(candidate_endpoints, key=current_usage)
        return ep

# -------------------------------------------------------------
# 6. API route – Generate
# -------------------------------------------------------------
@app.post("/api/generate")
async def proxy(request: Request):
    """
    Proxy a generate request to Ollama and stream the response back to the client.
    """
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))
        
        model = payload.get("model")
        prompt = payload.get("prompt")
        suffix = payload.get("suffix")
        system = payload.get("system")
        template = payload.get("template")
        context = payload.get("context")
        stream = payload.get("stream")
        think = payload.get("think")
        raw = payload.get("raw")
        _format = payload.get("format")
        images = payload.get("images")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except orjson.JSONDecodeError as e:
        error_msg = f"Invalid JSON format in request body: {str(e)}. Please ensure the request is properly formatted."
        raise HTTPException(status_code=400, detail=error_msg) from e

    
    endpoint = await choose_endpoint(model)
    use_openai = is_openai_compatible(endpoint)
    if use_openai:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        params = {
            "prompt": prompt,
            "model": model,
        }

        optional_params = {
            "stream": stream,
            "max_tokens": options.get("num_predict") if options and "num_predict" in options else None,
            "frequency_penalty": options.get("frequency_penalty") if options and "frequency_penalty" in options else None,
            "presence_penalty": options.get("presence_penalty") if options and "presence_penalty" in options else None,
            "seed": options.get("seed") if options and "seed" in options else None,
            "stop": options.get("stop") if options and "stop" in options else None,
            "top_p": options.get("top_p") if options and "top_p" in options else None,
            "temperature": options.get("temperature") if options and "temperature" in options else None,
            "suffix": suffix,
            }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=ep2base(endpoint), default_headers=default_headers, api_key=config.api_keys.get(endpoint, "no-key"))
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)

    # 4. Async generator that streams data and decrements the counter
    async def stream_generate_response():
        try:
            if use_openai:
                start_ts = time.perf_counter()
                async_gen = await oclient.completions.create(**params)
            else:
                async_gen = await client.generate(model=model, prompt=prompt, suffix=suffix, system=system, template=template, context=context, stream=stream, think=think, raw=raw, format=_format, images=images, options=options, keep_alive=keep_alive)
            if stream == True:
                async for chunk in async_gen:
                    if use_openai:
                        chunk = rechunk.openai_completion2ollama(chunk, stream, start_ts)
                    prompt_tok = chunk.prompt_eval_count or 0
                    comp_tok   = chunk.eval_count or 0
                    if prompt_tok != 0 or comp_tok != 0:
                        await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = orjson.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                if use_openai:
                    response = rechunk.openai_completion2ollama(async_gen, stream, start_ts)
                    response = response.model_dump_json()
                else:
                    response = async_gen.model_dump_json()
                    prompt_tok = async_gen.prompt_eval_count or 0
                    comp_tok   = async_gen.eval_count or 0
                    if prompt_tok != 0 or comp_tok != 0:
                        await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                json_line = (
                    response
                    if hasattr(async_gen, "model_dump_json")
                    else orjson.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_generate_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 7. API route – Chat
# -------------------------------------------------------------
@app.post("/api/chat")
async def chat_proxy(request: Request):
    """
    Proxy a chat request to Ollama and stream the endpoint reply.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        messages = payload.get("messages")
        tools = payload.get("tools")
        stream = payload.get("stream")
        think = payload.get("think")
        _format = payload.get("format")
        keep_alive = payload.get("keep_alive")
        options = payload.get("options")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400, detail="Missing or invalid 'messages' field (must be a list)"
            )
        if options is not None and not isinstance(options, dict):
            raise HTTPException(
                status_code=400, detail="`options` must be a JSON object"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    if model.startswith("moe-"):
        model = model.split("moe-")[1]
        opt = True
    else:
        opt = False
    endpoint = await choose_endpoint(model)
    use_openai = is_openai_compatible(endpoint)
    if use_openai:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        if messages:
            messages = transform_images_to_data_urls(messages)
            messages = transform_tool_calls_to_openai(messages)
        params = {
            "messages": messages,
            "model": model,
            }
        optional_params = {
            "tools": tools,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
            "max_tokens": options.get("num_predict") if options and "num_predict" in options else None,
            "frequency_penalty": options.get("frequency_penalty") if options and "frequency_penalty" in options else None,
            "presence_penalty": options.get("presence_penalty") if options and "presence_penalty" in options else None,
            "seed": options.get("seed") if options and "seed" in options else None,
            "stop": options.get("stop") if options and "stop" in options else None,
            "top_p": options.get("top_p") if options and "top_p" in options else None,
            "temperature": options.get("temperature") if options and "temperature" in options else None,
            "response_format": {"type": "json_schema", "json_schema": _format} if _format is not None else None
            }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=ep2base(endpoint), default_headers=default_headers, api_key=config.api_keys.get(endpoint, "no-key"))
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams chat data and decrements the counter
    async def stream_chat_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if use_openai:
                start_ts = time.perf_counter()
                async_gen = await oclient.chat.completions.create(**params)
            else:
                if opt == True:
                    # Use the dedicated MOE helper function
                    async_gen = await _make_moe_requests(model, messages, tools, think, _format, options, keep_alive)
                else:
                    async_gen = await client.chat(model=model, messages=messages, tools=tools, stream=stream, think=think, format=_format, options=options, keep_alive=keep_alive)
            if stream == True:
                tc_acc = {}  # accumulate OpenAI tool-call deltas across chunks
                async for chunk in async_gen:
                    if use_openai:
                        _accumulate_openai_tc_delta(chunk, tc_acc)
                        chunk = rechunk.openai_chat_completion2ollama(chunk, stream, start_ts)
                        # Inject fully-accumulated tool calls only into the final chunk
                        if chunk.done and tc_acc and chunk.message:
                            chunk.message.tool_calls = _build_ollama_tool_calls(tc_acc)
                    # `chunk` can be a dict or a pydantic model – dump to JSON safely
                    prompt_tok = chunk.prompt_eval_count or 0
                    comp_tok   = chunk.eval_count or 0
                    if prompt_tok != 0 or comp_tok != 0:
                        await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = orjson.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                if use_openai:
                    response = rechunk.openai_chat_completion2ollama(async_gen, stream, start_ts)
                    response = response.model_dump_json()
                else:
                    response = async_gen.model_dump_json()
                    prompt_tok = async_gen.prompt_eval_count or 0
                    comp_tok   = async_gen.eval_count or 0
                    if prompt_tok != 0 or comp_tok != 0:
                        await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                json_line = (
                    response
                    if hasattr(async_gen, "model_dump_json")
                    else orjson.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    media_type = "application/x-ndjson" if stream else "application/json"
    return StreamingResponse(
        stream_chat_response(),
        media_type=media_type,
    )

# -------------------------------------------------------------
# 8. API route – Embedding - deprecated
# -------------------------------------------------------------
@app.post("/api/embeddings")
async def embedding_proxy(request: Request):
    """
    Proxy an embedding request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        prompt = payload.get("prompt")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    use_openai = is_openai_compatible(endpoint)
    if use_openai:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        client = openai.AsyncOpenAI(base_url=ep2base(endpoint), api_key=config.api_keys.get(endpoint, "no-key"))
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams embedding data and decrements the counter
    async def stream_embedding_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if use_openai:
                async_gen = await client.embeddings.create(input=prompt, model=model)
                async_gen = rechunk.openai_embeddings2ollama(async_gen)
            else:
                async_gen = await client.embeddings(model=model, prompt=prompt, options=options, keep_alive=keep_alive)
            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = orjson.dumps(async_gen)
            yield json_line.encode("utf-8") + b"\n"
        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_embedding_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 9. API route – Embed
# -------------------------------------------------------------
@app.post("/api/embed")
async def embed_proxy(request: Request):
    """
    Proxy an embed request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        _input = payload.get("input")
        truncate = payload.get("truncate")
        options = payload.get("options")
        keep_alive = payload.get("keep_alive")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not _input:
            raise HTTPException(
                status_code=400, detail="Missing required field 'input'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    use_openai = is_openai_compatible(endpoint)
    if use_openai:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        client = openai.AsyncOpenAI(base_url=ep2base(endpoint), api_key=config.api_keys.get(endpoint, "no-key"))
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams embed data and decrements the counter
    async def stream_embedding_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if use_openai:
                async_gen = await client.embeddings.create(input=_input, model=model)
                async_gen = rechunk.openai_embed2ollama(async_gen, model)
            else:
                async_gen = await client.embed(model=model, input=_input, truncate=truncate, options=options, keep_alive=keep_alive)
            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = orjson.dumps(async_gen)
            yield json_line.encode("utf-8") + b"\n"
        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_embedding_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 10. API route – Create
# -------------------------------------------------------------
@app.post("/api/create")
async def create_proxy(request: Request):
    """
    Proxy a create request to all Ollama endpoints and reply with deduplicated status.
    """
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        quantize = payload.get("quantize")
        from_ = payload.get("from")
        files = payload.get("files")
        adapters = payload.get("adapters")
        template = payload.get("template")
        license = payload.get("license")
        system = payload.get("system")
        parameters = payload.get("parameters")
        messages = payload.get("messages")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not from_ and not files:
            raise HTTPException(
                status_code=400, detail="You need to provide either from_ or files parameter!"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    
    status_lists = []

    for endpoint in config.endpoints:
        client = ollama.AsyncClient(host=endpoint)
        create = await client.create(model=model, quantize=quantize, from_=from_, files=files, adapters=adapters, template=template, license=license, system=system, parameters=parameters, messages=messages, stream=False)
        status_lists.append(create)

    combined_status = []
    for status_list in status_lists:
        combined_status += status_list

    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)

# -------------------------------------------------------------
# 11. API route – Show
# -------------------------------------------------------------
@app.post("/api/show")
async def show_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a model show request to Ollama and reply with ShowResponse.

    """
    try:
        body_bytes = await request.body()

        if not model:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    #await increment_usage(endpoint, model)

    client = ollama.AsyncClient(host=endpoint)

    # 3. Proxy a simple show request
    show = await client.show(model=model)

    # 4. Return ShowResponse
    return show

# -------------------------------------------------------------
@app.get("/api/token_counts")
async def token_counts_proxy():
    breakdown = []
    total = 0
    async for entry in db.load_token_counts():
        total += entry['total_tokens']
        breakdown.append({
            "endpoint": entry["endpoint"],
            "model": entry["model"],
            "input_tokens": entry["input_tokens"],
            "output_tokens": entry["output_tokens"],
            "total_tokens": entry["total_tokens"],
        })
    return {"total_tokens": total, "breakdown": breakdown}

@app.post("/api/aggregate_time_series_days")
async def aggregate_time_series_days_proxy(request: Request):
    """
    Aggregate time_series entries older than days into daily aggregates by endpoint/model/date.
    """
    try:
        body_bytes = await request.body()
        if not body_bytes:
            days = 30
            trim_old = False
        else:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            days = int(payload.get("days", 30))
            trim_old = bool(payload.get("trim_old", False))
    except Exception:
        days = 30
        trim_old = False
    aggregated = await db.aggregate_time_series_older_than(days, trim_old=trim_old)
    return {"status": "ok", "days": days, "trim_old": trim_old, "aggregated_groups": aggregated}

# 12. API route – Stats
# -------------------------------------------------------------
@app.post("/api/stats")
async def stats_proxy(request: Request, model: Optional[str] = None):
    """
    Return token usage statistics for a specific model.
    """
    try:
        body_bytes = await request.body()

        if not model:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # Get token counts from database
    token_data = await db.get_token_counts_for_model(model)

    if not token_data:
        raise HTTPException(
            status_code=404, detail="No token data found for this model"
        )

    # Get time series data for the last 30 days (43200 minutes = 30 days)
    # Assuming entries are grouped by minute, 30 days = 43200 entries max
    time_series = []
    endpoint_totals = defaultdict(int)  # Track tokens per endpoint
    
    async for entry in db.get_latest_time_series(limit=50000):
        if entry['model'] == model:
            time_series.append({
                'endpoint': entry['endpoint'],
                'timestamp': entry['timestamp'],
                'input_tokens': entry['input_tokens'],
                'output_tokens': entry['output_tokens'],
                'total_tokens': entry['total_tokens']
            })
            # Accumulate total tokens per endpoint
            endpoint_totals[entry['endpoint']] += entry['total_tokens']

    return {
        'model': model,
        'input_tokens': token_data['input_tokens'],
        'output_tokens': token_data['output_tokens'],
        'total_tokens': token_data['total_tokens'],
        'time_series': time_series,
        'endpoint_distribution': dict(endpoint_totals)
    }

# -------------------------------------------------------------
# 12. API route – Copy
# -------------------------------------------------------------
@app.post("/api/copy")
async def copy_proxy(request: Request, source: Optional[str] = None, destination: Optional[str] = None):
    """
    Proxy a model copy request to each Ollama endpoint and reply with Status Code.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not source and not destination:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            src = payload.get("source")
            dst = payload.get("destination")
        else:
            src = source
            dst = destination
        
        if not src:
            raise HTTPException(
                status_code=400, detail="Missing required field 'source'"
            )
        if not dst:
            raise HTTPException(
                status_code=400, detail="Missing required field 'destination'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 3. Iterate over all endpoints to copy the model on each endpoint
    status_list = []

    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 4. Proxy a simple copy request
            copy = await client.copy(source=src, destination=dst)
            status_list.append(copy.status)

    # 4. Return with 200 OK if all went well, 404 if a single endpoint failed
    return Response(status_code=404 if 404 in status_list else 200)

# -------------------------------------------------------------
# 13. API route – Delete
# -------------------------------------------------------------
@app.delete("/api/delete")
async def delete_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a model delete request to each Ollama endpoint and reply with Status Code.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not model:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints to delete the model on each endpoint
    status_list = []

    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 3. Proxy a simple copy request
            copy = await client.delete(model=model)
            status_list.append(copy.status)
    
    # 4. Return 200 0K, if a single enpoint fails, respond with 404
    return Response(status_code=404 if 404 in status_list else 200)   

# -------------------------------------------------------------
# 14. API route – Pull
# -------------------------------------------------------------
@app.post("/api/pull")
async def pull_proxy(request: Request, model: Optional[str] = None):
    """
    Proxy a pull request to all Ollama endpoint and report status back.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()

        if not model:
            payload = orjson.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
            insecure = payload.get("insecure")
        else:
            insecure = None

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints to pull the model
    status_list = []

    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 3. Proxy a simple pull request
            pull = await client.pull(model=model, insecure=insecure, stream=False)
            status_list.append(pull)

    combined_status = []
    for status in status_list:
        combined_status += status
    
    # 4. Report back a deduplicated status message
    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)

# -------------------------------------------------------------
# 15. API route – Push
# -------------------------------------------------------------
@app.post("/api/push")
async def push_proxy(request: Request):
    """
    Proxy a push request to Ollama and respond the deduplicated Ollama endpoint replies.
    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        insecure = payload.get("insecure")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints
    status_list = []

    for endpoint in config.endpoints:
        client = ollama.AsyncClient(host=endpoint)
        # 3. Proxy a simple push request
        push = await client.push(model=model, insecure=insecure, stream=False)
        status_list.append(push)

    combined_status = []
    for status in status_list:
        combined_status += status
    
    # 4. Report a deduplicated status
    final_status = list(dict.fromkeys(combined_status))

    return dict(final_status)


# -------------------------------------------------------------
# 16. API route – Version
# -------------------------------------------------------------
@app.get("/api/version")
async def version_proxy(request: Request):
    """
    Proxy a version request to Ollama and reply lowest version of all endpoints.

    """
    # 1. Query all endpoints for version
    tasks = [fetch.endpoint_details(ep, "/api/version", "version") for ep in config.endpoints if "/v1" not in ep]
    all_versions = await asyncio.gather(*tasks)

    def version_key(v):
        return tuple(map(int, v.split('.')))
    
    # 2. Return a JSONResponse with the min Version of all endpoints to maintain compatibility
    return JSONResponse(
        content={"version": str(min(all_versions, key=version_key))},
        status_code=200,
    )

# -------------------------------------------------------------
# 17. API route – tags
# -------------------------------------------------------------
@app.get("/api/tags")
async def tags_proxy(request: Request):
    """
    Proxy a tags request to Ollama endpoints and reply with a unique list of all models.

    """
    
    # 1. Query all endpoints for models
    tasks = [fetch.endpoint_details(ep, "/api/tags", "models") for ep in config.endpoints if "/v1" not in ep]
    tasks += [fetch.endpoint_details(ep, "/models", "data", config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    # Also query llama-server endpoints not already covered by config.endpoints
    llama_eps_for_tags = [ep for ep in config.llama_server_endpoints if ep not in config.endpoints]
    tasks += [fetch.endpoint_details(ep, "/models", "data", config.api_keys.get(ep)) for ep in llama_eps_for_tags]
    all_models = await asyncio.gather(*tasks)

    models = {'models': []}
    for modellist in all_models:
        for model in modellist:
            if not "model" in model.keys():  # Relable OpenAI models with Ollama Model.model from Model.id
                model['model'] = model['id'] + ":latest"
            else:
                model['id'] = model['model']
            if not "name" in model.keys():  # Relable OpenAI models with Ollama Model.name from Model.model to have model,name keys
                model['name'] = model['model']
            else:
                model['id'] = model['model']
        models['models'] += modellist
    
    # 2. Return a JSONResponse with a deduplicated list of unique models for inference
    return JSONResponse(
        content={"models": dedupe_on_keys(models['models'], ['digest','name','id'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 18. API route – ps
# -------------------------------------------------------------
@app.get("/api/ps")
async def ps_proxy(request: Request):
    """
    Proxy a ps request to all Ollama and llama-server endpoints and reply a unique list of all running models.

    For Ollama endpoints: queries /api/ps
    For llama-server endpoints: queries /v1/models with status.value == "loaded"
    """
    # 1. Query Ollama endpoints for running models via /api/ps
    ollama_tasks = [fetch.endpoint_details(ep, "/api/ps", "models") for ep in config.endpoints if "/v1" not in ep]
    # 2. Query llama-server endpoints for loaded models via /v1/models
    # Also query endpoints from llama_server_endpoints that may not be in config.endpoints
    all_llama_endpoints = set(config.llama_server_endpoints) | set(ep for ep in config.endpoints if ep in config.llama_server_endpoints)
    llama_tasks = [
        fetch.endpoint_details(ep, "/models", "data", config.api_keys.get(ep))
        for ep in all_llama_endpoints
    ]
    
    ollama_loaded = await asyncio.gather(*ollama_tasks) if ollama_tasks else []
    llama_loaded = await asyncio.gather(*llama_tasks) if llama_tasks else []

    models = {'models': []}
    # Add Ollama models (if any)
    if ollama_loaded:
        for modellist in ollama_loaded:
            models['models'] += modellist
    # Add llama-server models (filter for loaded only, if any)
    if llama_loaded:
        for modellist in llama_loaded:
            loaded_models = [item for item in modellist if _is_llama_model_loaded(item)]
            # Convert llama-server format to Ollama-like format for consistency
            for item in loaded_models:
                raw_id = item.get("id", "")
                normalized = _normalize_llama_model_name(raw_id)
                quant = _extract_llama_quant(raw_id)
                models['models'].append({
                    "name": normalized,
                    "id": normalized,
                    "digest": "",
                    "status": item.get("status"),
                    "details": {"quantization_level": quant} if quant else {}
                })
    
    # 3. Return a JSONResponse with deduplicated currently deployed models
    return JSONResponse(
        content={"models": dedupe_on_keys(models['models'], ['digest'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 18b. API route – ps details (backwards compatible)
# -------------------------------------------------------------
@app.get("/api/ps_details")
async def ps_details_proxy(request: Request):
    """
    Proxy a ps request to all Ollama and llama-server endpoints and reply with per-endpoint instances.
    This keeps /api/ps backward compatible while providing richer data.
    
    For Ollama endpoints: queries /api/ps
    For llama-server endpoints: queries /v1/models with status info
    """
    # 1. Query Ollama endpoints via /api/ps
    ollama_tasks = [(ep, fetch.endpoint_details(ep, "/api/ps", "models")) for ep in config.endpoints if "/v1" not in ep]
    # 2. Query llama-server endpoints via /v1/models
    # Also query endpoints from llama_server_endpoints that may not be in config.endpoints
    all_llama_endpoints = set(config.llama_server_endpoints) | set(ep for ep in config.endpoints if ep in config.llama_server_endpoints)
    llama_tasks = [
        (ep, fetch.endpoint_details(ep, "/models", "data", config.api_keys.get(ep)))
        for ep in all_llama_endpoints
    ]
    
    ollama_loaded = await asyncio.gather(*[task for _, task in ollama_tasks]) if ollama_tasks else []
    llama_loaded = await asyncio.gather(*[task for _, task in llama_tasks]) if llama_tasks else []

    models: list[dict] = []
    
    # Add Ollama models with endpoint info (if any)
    if ollama_loaded:
        for (endpoint, modellist) in zip([ep for ep, _ in ollama_tasks], ollama_loaded):
            for model in modellist:
                if isinstance(model, dict):
                    model_with_endpoint = dict(model)
                    model_with_endpoint["endpoint"] = endpoint
                    models.append(model_with_endpoint)
    
    # Add llama-server models with endpoint info and full status metadata (if any)
    if llama_loaded:
        for (endpoint, modellist) in zip([ep for ep, _ in llama_tasks], llama_loaded):
            # Filter for loaded models only
            loaded_models = [item for item in modellist if _is_llama_model_loaded(item)]
            for item in loaded_models:
                if isinstance(item, dict) and item.get("id"):
                    raw_id = item["id"]
                    normalized = _normalize_llama_model_name(raw_id)
                    quant = _extract_llama_quant(raw_id)
                    model_with_endpoint = {
                        "name": normalized,
                        "id": normalized,
                        "original_name": raw_id,
                        "digest": "",
                        "details": {"quantization_level": quant} if quant else {},
                        "endpoint": endpoint,
                        "status": item.get("status"),
                        "created": item.get("created"),
                        "owned_by": item.get("owned_by")
                    }
                    # Include full llama-server status details (args, preset)
                    status_info = item.get("status", {})
                    if isinstance(status_info, dict):
                        model_with_endpoint["llama_status_args"] = status_info.get("args")
                        model_with_endpoint["llama_status_preset"] = status_info.get("preset")
                    models.append(model_with_endpoint)

    return JSONResponse(content={"models": models}, status_code=200)

# -------------------------------------------------------------
# 19. Proxy usage route – for monitoring
# -------------------------------------------------------------
@app.get("/api/usage")
async def usage_proxy(request: Request):
    """
    Return a snapshot of the usage counter for each endpoint.
    Useful for debugging / monitoring.
    """
    return {"usage_counts": usage_counts,
            "token_usage_counts": token_usage_counts}

# -------------------------------------------------------------
# 20. Proxy config route – for monitoring and frontent usage
# -------------------------------------------------------------
@app.get("/api/config")
async def config_proxy(request: Request):
    """
    Return a simple JSON object that contains the configured
    Ollama endpoints and llama_server_endpoints. The front‑end uses this to display
    which endpoints are being proxied.
    """
    async def check_endpoint(url: str):
        client: aiohttp.ClientSession = app_state["session"]
        headers = None
        if "/v1" in url:
            headers = {"Authorization": "Bearer " + config.api_keys[url]}
            target_url = f"{url}/models"
        else:
            target_url = f"{url}/api/version"

        try:
            async with client.get(target_url, headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()
            if "/v1" in url:
                return {"url": url, "status": "ok", "version": "latest"}
            else:
                return {"url": url, "status": "ok", "version": data.get("version")}
        except Exception as e:
            detail = _format_connection_issue(target_url, e)
            return {"url": url, "status": "error", "detail": detail}

    # Check Ollama endpoints
    ollama_results = await asyncio.gather(*[check_endpoint(ep) for ep in config.endpoints])
    
    # Check llama-server endpoints
    llama_results = []
    if config.llama_server_endpoints:
        llama_results = await asyncio.gather(*[check_endpoint(ep) for ep in config.llama_server_endpoints])
    
    return {
        "endpoints": ollama_results,
        "llama_server_endpoints": llama_results,
        "require_router_api_key": bool(config.router_api_key),
    }

# -------------------------------------------------------------
# 21. API route – OpenAI compatible Embedding
# -------------------------------------------------------------
@app.post("/v1/embeddings")
async def openai_embedding_proxy(request: Request):
    """
    Proxy an OpenAI API compatible embedding request to Ollama and reply with embeddings.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        doc = payload.get("input")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not doc:
            raise HTTPException(
                status_code=400, detail="Missing required field 'input'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    if is_openai_compatible(endpoint):
        api_key = config.api_keys.get(endpoint, "no-key")
    else:
        api_key = "ollama"
    base_url = ep2base(endpoint)

    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=api_key)

    # 3. Async generator that streams embedding data and decrements the counter
    async_gen = await oclient.embeddings.create(input=doc, model=model)
            
    await decrement_usage(endpoint, model)

    # 5. Return a StreamingResponse backed by the generator
    return async_gen

# -------------------------------------------------------------
# 22. API route – OpenAI compatible Chat Completions
# -------------------------------------------------------------
@app.post("/v1/chat/completions")
async def openai_chat_completions_proxy(request: Request):
    """
    Proxy an OpenAI API compatible chat completions request to Ollama and reply with a streaming response.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        messages = payload.get("messages")
        frequency_penalty = payload.get("frequency_penalty")
        presence_penalty = payload.get("presence_penalty")
        response_format = payload.get("response_format")
        seed = payload.get("seed")
        stop = payload.get("stop")
        stream = payload.get("stream")
        stream_options = payload.get("stream_options")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        max_tokens = payload.get("max_tokens")
        max_completion_tokens = payload.get("max_completion_tokens")
        tools = payload.get("tools")

        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]

        params = {
            "messages": messages, 
            "model": model,
        }

        optional_params = {
            "tools": tools,
            "response_format": response_format,
            "stream_options": stream_options or {"include_usage": True },
            "max_completion_tokens": max_completion_tokens,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stop": stop,
            "stream": stream,
        }

        params.update({k: v for k, v in optional_params.items() if v is not None})
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not isinstance(messages, list):
            raise HTTPException(
                status_code=400, detail="Missing required field 'messages' (must be a list)"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys.get(endpoint, "no-key"))
    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ochat_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            try:
                async_gen = await oclient.chat.completions.create(**params)
            except openai.BadRequestError as e:
                # If tools are not supported by the model, retry without tools
                if "does not support tools" in str(e):
                    print(f"[openai_chat_completions_proxy] Model {model} doesn't support tools, retrying without tools")
                    params_without_tools = {k: v for k, v in params.items() if k != "tools"}
                    async_gen = await oclient.chat.completions.create(**params_without_tools)
                else:
                    raise
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else orjson.dumps(chunk)
                    )
                    if chunk.choices:
                        if chunk.choices[0].delta.content is not None:
                            yield f"data: {data}\n\n".encode("utf-8")
                    if chunk.usage is not None:
                        prompt_tok = chunk.usage.prompt_tokens or 0
                        comp_tok   = chunk.usage.completion_tokens or 0
                        if prompt_tok != 0 or comp_tok != 0:
                            local_model = model
                            if not is_ext_openai_endpoint(endpoint):
                                if not ":" in model:
                                    local_model = model if ":" in model else model + ":latest"
                            await token_queue.put((endpoint, local_model, prompt_tok, comp_tok))
                yield b"data: [DONE]\n\n"
            else:
                prompt_tok = async_gen.usage.prompt_tokens or 0
                comp_tok   = async_gen.usage.completion_tokens or 0
                if prompt_tok != 0 or comp_tok != 0:
                    await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else orjson.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_ochat_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 23. API route – OpenAI compatible Completions
# -------------------------------------------------------------
@app.post("/v1/completions")
async def openai_completions_proxy(request: Request):
    """
    Proxy an OpenAI API compatible chat completions request to Ollama and reply with a streaming response.

    """
    # 1. Parse and validate request
    try:
        body_bytes = await request.body()
        payload = orjson.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        prompt = payload.get("prompt")
        frequency_penalty = payload.get("frequency_penalty")
        presence_penalty = payload.get("presence_penalty")
        seed = payload.get("seed")
        stop = payload.get("stop")
        stream = payload.get("stream")
        stream_options = payload.get("stream_options")
        temperature = payload.get("temperature")
        top_p = payload.get("top_p")
        max_tokens = payload.get("max_tokens")
        max_completion_tokens = payload.get("max_completion_tokens")
        suffix = payload.get("suffix")

        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]

        params = {
            "prompt": prompt, 
            "model": model,
        }

        optional_params = {
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "stream_options": stream_options or {"include_usage": True },
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "suffix": suffix
        }

        params.update({k: v for k, v in optional_params.items() if v is not None})

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
        if not prompt:
            raise HTTPException(
                status_code=400, detail="Missing required field 'prompt'"
            )
    except orjson.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys.get(endpoint, "no-key"))

    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ocompletions_response(model=model):
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            async_gen = await oclient.completions.create(**params)
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else orjson.dumps(chunk)
                    )
                    if chunk.choices:
                        if chunk.choices[0].finish_reason == None:
                            yield f"data: {data}\n\n".encode("utf-8")
                    if chunk.usage is not None:
                            prompt_tok = chunk.usage.prompt_tokens or 0
                            comp_tok   = chunk.usage.completion_tokens or 0
                            if prompt_tok != 0 or comp_tok != 0:
                                local_model = model
                                if not is_ext_openai_endpoint(endpoint):
                                    if not ":" in model:
                                        local_model = model if ":" in model else model + ":latest"
                                await token_queue.put((endpoint, local_model, prompt_tok, comp_tok))
                # Final DONE event
                yield b"data: [DONE]\n\n"
            else:
                prompt_tok = async_gen.usage.prompt_tokens or 0
                comp_tok   = async_gen.usage.completion_tokens or 0
                if prompt_tok != 0 or comp_tok != 0:
                    await token_queue.put((endpoint, model, prompt_tok, comp_tok))
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else orjson.dumps(async_gen)
                )
                yield json_line.encode("utf-8") + b"\n"

        finally:
            # Ensure counter is decremented even if an exception occurs
            await decrement_usage(endpoint, model)

    # 4. Return a StreamingResponse backed by the generator
    return StreamingResponse(
        stream_ocompletions_response(),
        media_type="application/json",
    )

# -------------------------------------------------------------
# 24. OpenAI API compatible models endpoint
# -------------------------------------------------------------
@app.get("/v1/models")
async def openai_models_proxy(request: Request):
    """
    Proxy an OpenAI API models request to Ollama and llama-server endpoints and reply with a unique list of models.
    
    For Ollama endpoints: queries /api/tags (all models)
    For llama-server endpoints: queries /v1/models and filters for status.value == "loaded"
    """
    # 1. Query Ollama endpoints for all models via /api/tags
    ollama_tasks = [fetch.endpoint_details(ep, "/api/tags", "models") for ep in config.endpoints if "/v1" not in ep]
    # 2. Query llama-server endpoints for loaded models via /v1/models
    # Also query endpoints from llama_server_endpoints that may not be in config.endpoints
    all_llama_endpoints = set(config.llama_server_endpoints) | set(ep for ep in config.endpoints if ep in config.llama_server_endpoints)
    llama_tasks = [
        fetch.endpoint_details(ep, "/models", "data", config.api_keys.get(ep))
        for ep in all_llama_endpoints
    ]
    
    ollama_models = await asyncio.gather(*ollama_tasks) if ollama_tasks else []
    llama_models = await asyncio.gather(*llama_tasks) if llama_tasks else []
    
    models = {'data': []}
    
    # Add Ollama models (if any)
    if ollama_models:
        for modellist in ollama_models:
            for model in modellist:
                if not "id" in model.keys():  # Relable Ollama models with OpenAI Model.id from Model.name
                    model['id'] = model.get('name', model.get('id', ''))
                else:
                    model['name'] = model['id']
                models['data'].append(model)
    
    # Add llama-server models (filter for loaded only, if any)
    if llama_models:
        for modellist in llama_models:
            loaded_models = [item for item in modellist if _is_llama_model_loaded(item)]
            for model in loaded_models:
                if not "id" in model.keys():
                    model['id'] = model.get('name', model.get('id', ''))
                else:
                    model['name'] = model['id']
                models['data'].append(model)
    
    # 2. Return a JSONResponse with a deduplicated list of unique models for inference
    return JSONResponse(
        content={"data": dedupe_on_keys(models['data'], ['name'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 25. Serve the static front‑end
# -------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def redirect_favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the dynamic NOMYO Router dashboard listing the configured endpoints
    and the models details, availability & task status.
    """
    index_path = STATIC_DIR / "index.html"
    try:
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"), status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Page not found")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")

# -------------------------------------------------------------
# 26. Healthendpoint
# -------------------------------------------------------------
@app.get("/health")
async def health_proxy(request: Request):
    """
    Health‑check endpoint for monitoring the proxy.

    * Queries each configured endpoint for its `/api/version` response.
    * Returns a JSON object containing:
        - `status`: "ok" if every endpoint replied, otherwise "error".
        - `endpoints`: a mapping of endpoint URL → `{status, version|detail}`.
    * The HTTP status code is 200 when everything is healthy, 503 otherwise.
    """
    # Run all health checks in parallel
    tasks = [fetch.endpoint_details(ep, "/api/version", "version", skip_error_cache=True) for ep in config.endpoints] # if not is_ext_openai_endpoint(ep)]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    health_summary = {}
    overall_ok = True

    for ep, result in zip(config.endpoints, results):
        if isinstance(result, Exception):
            # Endpoint did not respond / returned an error
            health_summary[ep] = {"status": "error", "detail": str(result)}
            overall_ok = False
        else:
            # Successful response – report the reported version
            health_summary[ep] = {"status": "ok", "version": result}

    response_payload = {
        "status": "ok" if overall_ok else "error",
        "endpoints": health_summary,
    }

    http_status = 200 if overall_ok else 503
    return JSONResponse(content=response_payload, status_code=http_status)

# -------------------------------------------------------------
# 27. SSE route for usage broadcasts
# -------------------------------------------------------------
@app.get("/api/usage-stream")
async def usage_stream(request: Request):
    """
    Server‑Sent‑Events that emits a JSON payload every time the
    global `usage_counts` dictionary changes.
    """
    async def event_generator():
        # The queue that receives *every* new snapshot
        queue = await subscribe()
        try:
            while True:
                # If the client disconnects, cancel the loop
                if await request.is_disconnected():
                    break
                data = await queue.get()
                if data is None:
                    break
                # Send the data as a single SSE message
                yield f"data: {data}\n\n"
        finally:
            # Clean‑up: unsubscribe from the broadcast channel
            await unsubscribe(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------------------------------------------------
# 28. FastAPI startup/shutdown events
# -------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    global config, db
    # Load YAML config (or use defaults if not present)
    config_path = _config_path_from_env()
    config = Config.from_yaml(config_path)
    if config_path.exists():
        print(
            f"Loaded configuration from {config_path}:\n"
            f" endpoints={config.endpoints},\n"
            f" llama_server_endpoints={config.llama_server_endpoints},\n"
            f" max_concurrent_connections={config.max_concurrent_connections}"
        )
    else:
        print(
            f"No configuration file found at {config_path}. "
            "Falling back to default settings."
        )

    # Initialize database
    db = TokenDatabase(config.db_path)
    await db.init_db()

    # Load existing token counts from database
    async for count_entry in db.load_token_counts():
        endpoint = count_entry['endpoint']
        model = count_entry['model']
        input_tokens = count_entry['input_tokens']
        output_tokens = count_entry['output_tokens']
        total_tokens = count_entry['total_tokens']

        token_usage_counts[endpoint][model] = total_tokens

    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=512, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=120, sock_connect=15)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    app_state["connector"] = connector
    app_state["session"] = session
    token_worker_task = asyncio.create_task(token_worker())
    flush_task = asyncio.create_task(flush_buffer())

@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_all_sse_queues()
    await flush_remaining_buffers()
    await app_state["session"].close()
    if token_worker_task is not None:
        token_worker_task.cancel()
    if flush_task is not None:
        flush_task.cancel()
