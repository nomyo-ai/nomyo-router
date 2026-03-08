"""
LLM Semantic Cache for NOMYO Router.

Strategy:
- Namespace: sha256(route :: model :: system_prompt)[:16]  — exact context isolation
- Cache key:  hash(normalize(last_user_message), namespace) — exact lookup
- Embedding:  weighted mean of
                α  * embed(bm25_weighted(chat_history))   — conversation context
                1-α * embed(last_user_message)             — the actual question
  with α = cache_history_weight (default 0.3).
- Exact-match caching (similarity=1.0) uses DummyEmbeddingProvider — zero extra deps.
- Semantic caching (similarity<1.0) requires sentence-transformers.  If missing the
  library falls back to exact-match with a warning (lean Docker image behaviour).
- MOE models (moe-*) always bypass the cache.
- Token counts are never recorded for cache hits.
- Streaming cache hits are served as a single-chunk response.
"""

import hashlib
import math
import time
import warnings
from collections import Counter
from typing import Any, Optional

# Lazily resolved once at first embed() call
_semantic_available: Optional[bool] = None


def _check_sentence_transformers() -> bool:
    global _semantic_available
    if _semantic_available is None:
        try:
            import sentence_transformers  # noqa: F401
            _semantic_available = True
        except ImportError:
            _semantic_available = False
    return _semantic_available  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# BM25-weighted text representation of chat history
# ---------------------------------------------------------------------------

def _bm25_weighted_text(history: list[dict]) -> str:
    """
    Produce a BM25-importance-weighted text string from chat history turns.

    High-IDF (rare, domain-specific) terms are repeated proportionally to
    their BM25 score so the downstream sentence-transformer embedding
    naturally upweights topical signal and downweights stop words.
    """
    docs = [m.get("content", "") for m in history if m.get("content")]
    if not docs:
        return ""

    def _tok(text: str) -> list[str]:
        return [w.lower() for w in text.split() if len(w) > 2]

    tokenized = [_tok(d) for d in docs]
    N = len(tokenized)

    df: Counter = Counter()
    for tokens in tokenized:
        for term in set(tokens):
            df[term] += 1

    k1, b = 1.5, 0.75
    avg_dl = sum(len(t) for t in tokenized) / max(N, 1)

    term_scores: Counter = Counter()
    for tokens in tokenized:
        tf_c = Counter(tokens)
        dl = len(tokens)
        for term, tf in tf_c.items():
            idf = math.log((N + 1) / (df[term] + 1)) + 1.0
            score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
            term_scores[term] += score

    top = term_scores.most_common(50)
    if not top:
        return " ".join(docs)

    max_s = top[0][1]
    out: list[str] = []
    for term, score in top:
        out.extend([term] * max(1, round(3 * score / max_s)))
    return " ".join(out)


# ---------------------------------------------------------------------------
# LLMCache
# ---------------------------------------------------------------------------

class LLMCache:
    """
    Thin async wrapper around async-semantic-llm-cache that adds:
    - Route-aware namespace isolation
    - Two-vector weighted-mean embedding (history context + question)
    - Per-instance hit/miss counters
    - Graceful fallback when sentence-transformers is absent
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._backend: Any = None
        self._emb_cache: Any = None
        self._semantic: bool = False
        self._hits: int = 0
        self._misses: int = 0

    async def init(self) -> None:
        from semantic_llm_cache.similarity import EmbeddingCache

        # --- Backend ---
        backend_type: str = self._cfg.cache_backend
        if backend_type == "sqlite":
            from semantic_llm_cache.backends.sqlite import SQLiteBackend
            self._backend = SQLiteBackend(db_path=self._cfg.cache_db_path)
        elif backend_type == "redis":
            from semantic_llm_cache.backends.redis import RedisBackend
            self._backend = RedisBackend(url=self._cfg.cache_redis_url)
            await self._backend.ping()
        else:
            from semantic_llm_cache.backends.memory import MemoryBackend
            self._backend = MemoryBackend()

        # --- Embedding provider ---
        if self._cfg.cache_similarity < 1.0:
            if _check_sentence_transformers():
                from semantic_llm_cache.similarity import create_embedding_provider
                provider = create_embedding_provider("sentence-transformer")
                self._emb_cache = EmbeddingCache(provider=provider)
                self._semantic = True
                print(
                    f"[cache] Semantic cache ready "
                    f"(similarity≥{self._cfg.cache_similarity}, backend={backend_type})"
                )
            else:
                warnings.warn(
                    "[cache] sentence-transformers is not installed. "
                    "Falling back to exact-match caching (similarity=1.0). "
                    "Use the :semantic Docker image tag to enable semantic caching.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._emb_cache = EmbeddingCache()   # DummyEmbeddingProvider
                print(f"[cache] Exact-match cache ready (backend={backend_type}) [semantic unavailable]")
        else:
            self._emb_cache = EmbeddingCache()       # DummyEmbeddingProvider
            print(f"[cache] Exact-match cache ready (backend={backend_type})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _namespace(self, route: str, model: str, system: str) -> str:
        raw = f"{route}::{model}::{system}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_key(self, namespace: str, last_user: str) -> str:
        from semantic_llm_cache.utils import hash_prompt, normalize_prompt
        return hash_prompt(normalize_prompt(last_user), namespace)

    def _parse_messages(
        self, messages: list[dict]
    ) -> tuple[str, list[dict], str]:
        """
        Returns (system_prompt, prior_history_turns, last_user_message).
        Multimodal content lists are reduced to their text parts.
        """
        system = ""
        turns: list[dict] = []

        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if role == "system":
                system = content
            else:
                turns.append({"role": role, "content": content})

        last_user = ""
        for m in reversed(turns):
            if m["role"] == "user":
                last_user = m["content"]
                break

        # History = all turns before the final user message
        history = turns[:-1] if turns and turns[-1]["role"] == "user" else turns
        return system, history, last_user

    async def _build_embedding(
        self, history: list[dict], last_user: str
    ) -> list[float] | None:
        """
        Weighted mean of BM25-weighted history embedding and last-user embedding.
        Returns None when not in semantic mode.
        """
        if not self._semantic:
            return None

        import numpy as np

        alpha: float = self._cfg.cache_history_weight   # weight for history signal
        q_vec = np.array(await self._emb_cache.aencode(last_user), dtype=float)

        if not history:
            # No history → use question embedding alone (alpha has no effect)
            return q_vec.tolist()

        h_text = _bm25_weighted_text(history)
        h_vec = np.array(await self._emb_cache.aencode(h_text), dtype=float)

        combined = alpha * h_vec + (1.0 - alpha) * q_vec
        norm = float(np.linalg.norm(combined))
        if norm > 0.0:
            combined /= norm
        return combined.tolist()

    # ------------------------------------------------------------------
    # Public interface: chat (handles both Ollama and OpenAI message lists)
    # ------------------------------------------------------------------

    async def get_chat(
        self, route: str, model: str, messages: list[dict]
    ) -> bytes | None:
        """Return cached response bytes, or None on miss."""
        if not self._backend:
            return None

        system, history, last_user = self._parse_messages(messages)
        if not last_user:
            return None

        ns = self._namespace(route, model, system)
        key = self._cache_key(ns, last_user)

        # 1. Exact key match
        entry = await self._backend.get(key)
        if entry is not None:
            self._hits += 1
            return entry.response  # type: ignore[return-value]

        # 2. Semantic similarity match
        if self._semantic and self._cfg.cache_similarity < 1.0:
            emb = await self._build_embedding(history, last_user)
            result = await self._backend.find_similar(
                emb, threshold=self._cfg.cache_similarity, namespace=ns
            )
            if result is not None:
                _, matched, _ = result
                self._hits += 1
                return matched.response  # type: ignore[return-value]

        self._misses += 1
        return None

    async def set_chat(
        self, route: str, model: str, messages: list[dict], response_bytes: bytes
    ) -> None:
        """Store a response in the cache (fire-and-forget friendly)."""
        if not self._backend:
            return

        system, history, last_user = self._parse_messages(messages)
        if not last_user:
            return

        ns = self._namespace(route, model, system)
        key = self._cache_key(ns, last_user)

        emb = (
            await self._build_embedding(history, last_user)
            if self._semantic and self._cfg.cache_similarity < 1.0
            else None
        )

        from semantic_llm_cache.config import CacheEntry

        await self._backend.set(
            key,
            CacheEntry(
                prompt=last_user,
                response=response_bytes,
                embedding=emb,
                created_at=time.time(),
                ttl=self._cfg.cache_ttl,
                namespace=ns,
                hit_count=0,
            ),
        )

    # ------------------------------------------------------------------
    # Convenience wrappers for the generate route (prompt string, not messages)
    # ------------------------------------------------------------------

    async def get_generate(
        self, model: str, prompt: str, system: str = ""
    ) -> bytes | None:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return await self.get_chat("generate", model, messages)

    async def set_generate(
        self, model: str, prompt: str, system: str, response_bytes: bytes
    ) -> None:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        await self.set_chat("generate", model, messages, response_bytes)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "semantic": self._semantic,
            "backend": self._cfg.cache_backend,
            "similarity_threshold": self._cfg.cache_similarity,
            "history_weight": self._cfg.cache_history_weight,
        }

    async def clear(self) -> None:
        if self._backend:
            await self._backend.clear()
        self._hits = 0
        self._misses = 0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache: LLMCache | None = None


async def init_llm_cache(cfg: Any) -> LLMCache | None:
    """Initialise the module-level cache singleton. Returns None if disabled."""
    global _cache
    if not cfg.cache_enabled:
        print("[cache] Cache disabled (cache_enabled=false).")
        return None
    _cache = LLMCache(cfg)
    await _cache.init()
    return _cache


def get_llm_cache() -> LLMCache | None:
    return _cache


# ---------------------------------------------------------------------------
# Helper: convert a stored Ollama-format non-streaming response to an
# OpenAI SSE single-chunk stream (used when a streaming OpenAI request
# hits the cache whose entry was populated from a non-streaming response).
# ---------------------------------------------------------------------------

def openai_nonstream_to_sse(cached_bytes: bytes, model: str) -> bytes:
    """
    Wrap a stored OpenAI ChatCompletion JSON as a minimal single-chunk SSE stream.
    The stored entry always uses the non-streaming ChatCompletion format so that
    non-streaming cache hits can be served directly; this function adapts it for
    streaming clients.
    """
    import orjson, time as _time

    try:
        d = orjson.loads(cached_bytes)
        content = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
        chunk = {
            "id": d.get("id", "cache-hit"),
            "object": "chat.completion.chunk",
            "created": d.get("created", int(_time.time())),
            "model": d.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if d.get("usage"):
            chunk["usage"] = d["usage"]
        return f"data: {orjson.dumps(chunk).decode()}\n\ndata: [DONE]\n\n".encode()
    except Exception as exc:
        warnings.warn(
            f"[cache] openai_nonstream_to_sse: corrupt cache entry, returning empty stream: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return b"data: [DONE]\n\n"
