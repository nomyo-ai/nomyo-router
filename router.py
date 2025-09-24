"""
title: NOMYO Router - an Ollama Proxy with Endpoint:Model aware routing
author: alpha-nerd-nomyo
author_url: https://github.com/nomyo-ai
version: 0.4
license: AGPL
"""
# -------------------------------------------------------------
import json, time, asyncio, yaml, ollama, openai, os, re, aiohttp, ssl, datetime, random, base64, io
from pathlib import Path
from typing import Dict, Set, List, Optional
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
# Transient errors are cached for 1s – the key stays until the
# timeout expires, after which the endpoint will be queried again.
_error_cache: dict[str, float] = {}

# ------------------------------------------------------------------
# SSE Queues
# ------------------------------------------------------------------
_subscribers: Set[asyncio.Queue] = set()
_subscribers_lock = asyncio.Lock()

# ------------------------------------------------------------------
# aiohttp Global Sessions
# ------------------------------------------------------------------
app_state = {
    "session": None,
    "connector": None,
}

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
    # Max concurrent connections per endpoint‑model pair, see OLLAMA_NUM_PARALLEL
    max_concurrent_connections: int = 1

    api_keys: Dict[str, str] = Field(default_factory=dict)

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
            return cls(**cleaned)
        return cls()

# Create the global config object – it will be overwritten on startup
config = Config()

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
# 3. Global state: per‑endpoint per‑model active connection counters
# -------------------------------------------------------------
usage_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
usage_lock = asyncio.Lock()  # protects access to usage_counts

# -------------------------------------------------------------
# 4. Helperfunctions 
# -------------------------------------------------------------
aiotimeout = aiohttp.ClientTimeout(total=5)

def _is_fresh(cached_at: float, ttl: int) -> bool:
    return (time.time() - cached_at) < ttl

async def _ensure_success(resp: aiohttp.ClientResponse) -> None:
    if resp.status >= 400:
        text = await resp.text()
        raise HTTPException(status_code=resp.status, detail=text)

class fetch:
    async def available_models(endpoint: str, api_key: Optional[str] = None) -> Set[str]:
        """
        Query <endpoint>/api/tags and return a set of all model names that the
        endpoint *advertises* (i.e. is capable of serving).  This endpoint lists
        every model that is installed on the Ollama instance, regardless of
        whether the model is currently loaded into memory.

        If the request fails (e.g. timeout, 5xx, or malformed response), an empty
        set is returned.
        """
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}

        if endpoint in _models_cache:
            models, cached_at = _models_cache[endpoint]
            if _is_fresh(cached_at, 300):
                return models
            else:
                # stale entry – drop it
                del _models_cache[endpoint]

        if endpoint in _error_cache:
            if _is_fresh(_error_cache[endpoint], 1):
                # Still within the short error TTL – pretend nothing is available
                return set()
            else:
                # Error expired – remove it
                del _error_cache[endpoint]

        if "/v1" in endpoint:
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
                
                if models:
                    _models_cache[endpoint] = (models, time.time())
                    return models
                else:
                    # Empty list – treat as “no models”, but still cache for 300s
                    _models_cache[endpoint] = (models, time.time())
                    return models
        except Exception as e:
            # Treat any error as if the endpoint offers no models
            print(f"[fetch.available_models] {endpoint} error: {e}")
            _error_cache[endpoint] = time.time()
            return set()


    async def loaded_models(endpoint: str) -> Set[str]:
        """
        Query <endpoint>/api/ps and return a set of model names that are currently
        loaded on that endpoint. If the request fails (e.g. timeout, 5xx), an empty
        set is returned.
        """
        client: aiohttp.ClientSession = app_state["session"]
        try:
            async with client.get(f"{endpoint}/api/ps") as resp:
                await _ensure_success(resp)
                data = await resp.json()
            # The response format is:
            #   {"models": [{"name": "model1"}, {"name": "model2"}]}
            models = {m.get("name") for m in data.get("models", []) if m.get("name")}
            return models
        except Exception:
            # If anything goes wrong we simply assume the endpoint has no models
            return set()

    async def endpoint_details(endpoint: str, route: str, detail: str, api_key: Optional[str] = None) -> List[dict]:
        """
        Query <endpoint>/<route> to fetch <detail> and return a List of dicts with details
        for the corresponding Ollama endpoint. If the request fails we respond with "N/A" for detail.
        """
        client: aiohttp.ClientSession = app_state["session"]
        headers = None
        if api_key is not None:
            headers = {"Authorization": "Bearer " + api_key}
        
        try:
            async with client.get(f"{endpoint}{route}", headers=headers) as resp:
                await _ensure_success(resp)
                data = await resp.json()
            detail = data.get(detail, [])
            return detail
        except Exception as e:
            # If anything goes wrong we cannot reply details
            print(e)
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

def iso8601_ns():
    ns_since_epoch = time.time_ns()
    dt = datetime.datetime.fromtimestamp(
        ns_since_epoch / 1_000_000_000,  # seconds
        tz=datetime.timezone.utc
    )
    iso8601_with_ns = (
        dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ns_since_epoch % 1_000_000_000:09d}Z"
    )
    return iso8601_with_ns

def is_base64(image_string):
    try:
        if isinstance(image_string, str) and base64.b64encode(base64.b64decode(image_string)) == image_string.encode():
            return True
    except Exception as e:
        return False

def resize_image_if_needed(image_data):
    try:
        # Decode the base64 image data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

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

            image = image.resize((new_width, new_height), Image.LANCZOS)

        # Encode the resized image back to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        resized_image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return resized_image_data

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

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

class rechunk:
    def openai_chat_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.ChatResponse:
        if chunk.choices == [] and chunk.usage is not None:
            return ollama.ChatResponse(
                model=chunk.model,
                created_at=iso8601_ns(),
                done=True,
                done_reason='stop',
                total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                load_duration=100000, 
                prompt_eval_count=int(chunk.usage.prompt_tokens),
                prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)), 
                eval_count=int(chunk.usage.completion_tokens),
                eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000),
                message={"role": "assistant"}
                )
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        if stream == True:
            thinking = getattr(with_thinking.delta, "reasoning", None) if with_thinking else None
            role = chunk.choices[0].delta.role or "assistant"
            content = chunk.choices[0].delta.content or ''
        else:
            thinking = getattr(with_thinking, "reasoning", None) if with_thinking else None
            role = chunk.choices[0].message.role or "assistant"
            content = chunk.choices[0].message.content or ''
        assistant_msg = ollama.Message(
            role=role,
            content=content,
            thinking=thinking,
            images=None,
            tool_name=None,
            tool_calls=None)
        rechunk = ollama.ChatResponse(
            model=chunk.model, 
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason, #if chunk.choices[0].finish_reason is not None else None,
            total_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            load_duration=100000, 
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0, 
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000) if chunk.usage is not None else 0,
            message=assistant_msg)
        return rechunk
    
    def openai_completion2ollama(chunk: dict, stream: bool, start_ts: float) -> ollama.GenerateResponse:
        with_thinking = chunk.choices[0] if chunk.choices[0] else None
        thinking = getattr(with_thinking, "reasoning", None) if with_thinking else None
        rechunk = ollama.GenerateResponse(
            model=chunk.model,
            created_at=iso8601_ns(),
            done=True if chunk.usage is not None else False,
            done_reason=chunk.choices[0].finish_reason,
            total_duration=int((time.perf_counter() - start_ts) * 1000) if chunk.usage is not None else 0,
            load_duration=10000,
            prompt_eval_count=int(chunk.usage.prompt_tokens) if chunk.usage is not None else 0,
            prompt_eval_duration=int((time.perf_counter() - start_ts) * 1_000_000_000 * (chunk.usage.prompt_tokens / chunk.usage.completion_tokens / 100)) if chunk.usage is not None and chunk.usage.completion_tokens != 0 else 0,
            eval_count=int(chunk.usage.completion_tokens) if chunk.usage is not None else 0,
            eval_duration=int((time.perf_counter() - start_ts) * 1000) if chunk.usage is not None else 0,
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
    async with usage_lock:
        snapshot = json.dumps({"usage_counts": usage_counts}, sort_keys=True)
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
    tag_tasks = [fetch.available_models(ep) for ep in config.endpoints if "/v1" not in ep]
    tag_tasks += [fetch.available_models(ep, config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    advertised_sets = await asyncio.gather(*tag_tasks)

    # 2️⃣  Filter endpoints that advertise the requested model
    candidate_endpoints = [
        ep for ep, models in zip(config.endpoints, advertised_sets)
        if model in models
    ]
    
    # 6️⃣ 
    if not candidate_endpoints:
        if ":latest" in model:  #ollama naming convention not applicable to openai
            model = model.split(":latest")
            model = model[0]
            candidate_endpoints = [
                ep for ep, models in zip(config.endpoints, advertised_sets)
                if model in models
            ]
        if not candidate_endpoints:
            raise RuntimeError(
                f"None of the configured endpoints ({', '.join(config.endpoints)}) "
                f"advertise the model '{model}'."
            )

    # 3️⃣  Among the candidates, find those that have the model *loaded*
    #      (concurrently, but only for the filtered list)
    load_tasks = [fetch.loaded_models(ep) for ep in candidate_endpoints]
    loaded_sets = await asyncio.gather(*load_tasks)
    
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
            ep = min(loaded_and_free, key=current_usage)
            return ep

        # 4️⃣ Endpoints among the candidates that simply have a free slot
        endpoints_with_free_slot = [
            ep for ep in candidate_endpoints
            if usage_counts.get(ep, {}).get(model, 0) < config.max_concurrent_connections
        ]

        if endpoints_with_free_slot:
            return random.choice(endpoints_with_free_slot)

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
        payload = json.loads(body_bytes.decode("utf-8"))
        
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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    
    endpoint = await choose_endpoint(model)
    is_openai_endpoint = "/v1" in endpoint
    if is_openai_endpoint:
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
            "sufix": suffix,
            }
        params.update({k: v for k, v in optional_params.items() if v is not None})
        oclient = openai.AsyncOpenAI(base_url=endpoint, default_headers=default_headers, api_key=config.api_keys[endpoint])
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)

    # 4. Async generator that streams data and decrements the counter
    async def stream_generate_response():
        try:
            if is_openai_endpoint:
                start_ts = time.perf_counter()
                async_gen = await oclient.completions.create(**params)
            else:
                async_gen = await client.generate(model=model, prompt=prompt, suffix=suffix, system=system, template=template, context=context, stream=stream, think=think, raw=raw, format=_format, images=images, options=options, keep_alive=keep_alive)
            if stream == True:
                async for chunk in async_gen:
                    if is_openai_endpoint:
                        chunk = rechunk.openai_completion2ollama(chunk, stream, start_ts)
                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = json.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                if is_openai_endpoint:
                    response = rechunk.openai_completion2ollama(async_gen, stream, start_ts)
                    response = response.model_dump_json()
                else:
                    response = async_gen.model_dump_json()
                json_line = (
                    response
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    is_openai_endpoint = "/v1" in endpoint
    if is_openai_endpoint:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        if messages:
            messages = transform_images_to_data_urls(messages)
        params = {
            "messages": messages, 
            "model": model,
            }
        optional_params = {
            "tools": tools,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream is not None else None,
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
        oclient = openai.AsyncOpenAI(base_url=endpoint, default_headers=default_headers, api_key=config.api_keys[endpoint])
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams chat data and decrements the counter
    async def stream_chat_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if is_openai_endpoint:
                start_ts = time.perf_counter()
                async_gen = await oclient.chat.completions.create(**params)
            else:
                async_gen = await client.chat(model=model, messages=messages, tools=tools, stream=stream, think=think, format=_format, options=options, keep_alive=keep_alive)
            if stream == True:
                async for chunk in async_gen:
                    if is_openai_endpoint:
                        chunk = rechunk.openai_chat_completion2ollama(chunk, stream, start_ts)
                    # `chunk` can be a dict or a pydantic model – dump to JSON safely
                    if hasattr(chunk, "model_dump_json"):
                        json_line = chunk.model_dump_json()
                    else:
                        json_line = json.dumps(chunk)
                    yield json_line.encode("utf-8") + b"\n"
            else:
                if is_openai_endpoint:
                    response = rechunk.openai_chat_completion2ollama(async_gen, stream, start_ts)
                    response = response.model_dump_json()
                else:
                    response = async_gen.model_dump_json()
                json_line = (
                    response
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    is_openai_endpoint = "/v1" in endpoint
    if is_openai_endpoint:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        client = openai.AsyncOpenAI(base_url=endpoint, api_key=config.api_keys[endpoint])
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams embedding data and decrements the counter
    async def stream_embedding_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if is_openai_endpoint:
                async_gen = await client.embeddings.create(input=prompt, model=model)
                async_gen = rechunk.openai_embeddings2ollama(async_gen)
            else:
                async_gen = await client.embeddings(model=model, prompt=prompt, options=options, keep_alive=keep_alive)
            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = json.dumps(async_gen)
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    is_openai_endpoint = "/v1" in endpoint
    if is_openai_endpoint:
        if ":latest" in model:
            model = model.split(":latest")
            model = model[0]
        client = openai.AsyncOpenAI(base_url=endpoint, api_key=config.api_keys[endpoint])
    else:
        client = ollama.AsyncClient(host=endpoint)
    await increment_usage(endpoint, model)
    # 3. Async generator that streams embed data and decrements the counter
    async def stream_embedding_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            if is_openai_endpoint:
                async_gen = await client.embeddings.create(input=_input, model=model)
                async_gen = rechunk.openai_embed2ollama(async_gen, model)
            else:
                async_gen = await client.embed(model=model, input=_input, truncate=truncate, options=options, keep_alive=keep_alive)
            if hasattr(async_gen, "model_dump_json"):
                json_line = async_gen.model_dump_json()
            else:
                json_line = json.dumps(async_gen)
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
    except json.JSONDecodeError as e:
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
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
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
            payload = json.loads(body_bytes.decode("utf-8"))
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
    except json.JSONDecodeError as e:
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
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
        
        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Iterate over all endpoints to delete the model on each endpoint
    status_list = []
    for endpoint in config.endpoints:
        if "/v1" not in endpoint:
            client = ollama.AsyncClient(host=endpoint)
            # 3. Proxy a simple copy request
            copy = await client.delete(model=model)
            status_list.append(copy.status)
    
    # 4. Retrun 200 0K, if a single enpoint fails, respond with 404
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
            payload = json.loads(body_bytes.decode("utf-8"))
            model = payload.get("model")
            insecure = payload.get("insecure")
        else:
            insecure = None

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
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
        payload = json.loads(body_bytes.decode("utf-8"))

        model = payload.get("model")
        insecure = payload.get("insecure")

        if not model:
            raise HTTPException(
                status_code=400, detail="Missing required field 'model'"
            )
    except json.JSONDecodeError as e:
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
    Proxy a ps request to all Ollama endpoints and reply a unique list of all running models.

    """
    # 1. Query all endpoints for running models
    tasks = [fetch.endpoint_details(ep, "/api/ps", "models") for ep in config.endpoints if "/v1" not in ep]
    loaded_models = await asyncio.gather(*tasks)

    models = {'models': []}
    for modellist in loaded_models:
        models['models'] += modellist
    
    # 2. Return a JSONResponse with deduplicated currently deployed models
    return JSONResponse(
        content={"models": dedupe_on_keys(models['models'], ['digest'])},
        status_code=200,
    )

# -------------------------------------------------------------
# 19. Proxy usage route – for monitoring
# -------------------------------------------------------------
@app.get("/api/usage")
async def usage_proxy(request: Request):
    """
    Return a snapshot of the usage counter for each endpoint.
    Useful for debugging / monitoring.
    """
    return {"usage_counts": usage_counts}

# -------------------------------------------------------------
# 20. Proxy config route – for monitoring and frontent usage
# -------------------------------------------------------------
@app.get("/api/config")
async def config_proxy(request: Request):
    """
    Return a simple JSON object that contains the configured
    Ollama endpoints. The front‑end uses this to display
    which endpoints are being proxied.
    """
    async def check_endpoint(url: str):
        try:
            client: aiohttp.ClientSession = app_state["session"]
            if "/v1" in url:
                headers = {"Authorization": "Bearer " + config.api_keys[url]}
                async with client.get(f"{url}/models", headers=headers) as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
            else:
                async with client.get(f"{url}/api/version") as resp:
                    await _ensure_success(resp)
                    data = await resp.json()
            if "/v1" in url:
                return {"url": url, "status": "ok", "version": "latest"}
            else:
                return {"url": url, "status": "ok", "version": data.get("version")}
        except Exception as e:
            return {"url": url, "status": "error", "detail": str(e)}

    results = await asyncio.gather(*[check_endpoint(ep) for ep in config.endpoints])
    return {"endpoints": results}

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
        payload = json.loads(body_bytes.decode("utf-8"))

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    if "/v1" in endpoint:
        api_key = config.api_keys[endpoint]
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
            "stream_options": stream_options,
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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys[endpoint])
    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ochat_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            async_gen = await oclient.chat.completions.create(**params)
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else json.dumps(chunk)
                    )
                    if chunk.choices[0].delta.content is not None:
                        yield f"data: {data}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            else:
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
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
        payload = json.loads(body_bytes.decode("utf-8"))

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
            "stream_options": stream_options,
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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    # 2. Endpoint logic
    endpoint = await choose_endpoint(model)
    await increment_usage(endpoint, model)
    base_url = ep2base(endpoint)
    oclient = openai.AsyncOpenAI(base_url=base_url, default_headers=default_headers, api_key=config.api_keys[endpoint])

    # 3. Async generator that streams completions data and decrements the counter
    async def stream_ocompletions_response():
        try:
            # The chat method returns a generator of dicts (or GenerateResponse)
            async_gen = await oclient.completions.create(**params)
            if stream == True:
                async for chunk in async_gen:
                    data = (
                        chunk.model_dump_json()
                        if hasattr(chunk, "model_dump_json")
                        else json.dumps(chunk)
                    )
                    yield f"data: {data}\n\n".encode("utf-8")
                # Final DONE event
                yield b"data: [DONE]\n\n"
            else:
                json_line = (
                    async_gen.model_dump_json()
                    if hasattr(async_gen, "model_dump_json")
                    else json.dumps(async_gen)
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
    Proxy an OpenAI API models request to Ollama endpoints and reply with a unique list of all models.

    """
    # 1. Query all endpoints for models
    tasks = [fetch.endpoint_details(ep, "/api/tags", "models") for ep in config.endpoints if "/v1" not in ep]
    tasks += [fetch.endpoint_details(ep, "/models", "data", config.api_keys[ep]) for ep in config.endpoints if "/v1" in ep]
    all_models = await asyncio.gather(*tasks)
    
    models = {'data': []}
    for modellist in all_models:
        for model in modellist:
            if not "id" in model.keys():  # Relable Ollama models with OpenAI Model.id from Model.name
                model['id'] = model['name']
            else:
                model['name'] = model['id']
        models['data'] += modellist
    
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
    return HTMLResponse(content=open("static/index.html", "r").read(), status_code=200)

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
    tasks = [fetch.endpoint_details(ep, "/api/version", "version") for ep in config.endpoints]

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
    global config
    # Load YAML config (or use defaults if not present)
    config = Config.from_yaml(Path("config.yaml"))
    print(f"Loaded configuration:\n endpoints={config.endpoints},\n "
          f"max_concurrent_connections={config.max_concurrent_connections}")
    
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=512, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=5, connect=5, sock_read=120, sock_connect=5)
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    app_state["connector"] = connector
    app_state["session"] = session


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await close_all_sse_queues()
    await app_state["session"].close()