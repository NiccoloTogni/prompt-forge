"""
LLM response cache — a drop-in wrapper around any LLMClient.

Usage::

    from prompt_forge import CachedLLM, Project

    project = Project("my_project", llm=CachedLLM(my_llm))

    # Persist cache to disk across runs (requires: pip install diskcache)
    import diskcache
    project = Project("my_project", llm=CachedLLM(my_llm, cache=diskcache.Cache(".llm_cache")))

The cache key is a SHA-256 hash of the serialised messages and kwargs.
FilePart content is hashed by file bytes, not path, so stale hits after
file updates are not possible.

Token usage for cache hits is reported as zero — the call costs nothing.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from .llm.client import LLMClient, LLMMessage, LLMResponse, TextPart, FilePart

logger = logging.getLogger(__name__)


def _make_key(messages: list[LLMMessage], kwargs: dict) -> str:
    """Produce a deterministic SHA-256 cache key for a set of messages and kwargs."""
    parts = []
    for msg in messages:
        if isinstance(msg.content, str):
            parts.append({"role": msg.role, "content": msg.content})
        else:
            content_parts = []
            for p in msg.content:
                if isinstance(p, TextPart):
                    content_parts.append({"type": "text", "text": p.text})
                elif isinstance(p, FilePart):
                    if p.path is not None:
                        try:
                            data = Path(p.path).read_bytes()
                        except OSError:
                            data = str(p.path).encode()
                    else:
                        data = b""
                    content_parts.append({
                        "type": "file",
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "media_type": p.media_type,
                    })
            parts.append({"role": msg.role, "content": content_parts})

    payload = json.dumps({"messages": parts, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLLM:
    """
    Transparent cache wrapper around any LLMClient.

    Identical inputs always return the cached response without a network call.
    Implements the LLMClient protocol — pass it anywhere an LLMClient is accepted.

    Args:
        llm: The underlying LLMClient to wrap.
        cache: A dict-like object used as the cache store. Defaults to an
               in-memory dict (cleared when the process exits). Pass a
               persistent store — e.g. ``diskcache.Cache(".llm_cache")`` or
               a Redis client — to retain the cache across runs.
    """

    def __init__(self, llm: LLMClient, cache: Any = None):
        self.llm = llm
        self._cache: Any = cache if cache is not None else {}
        self.hits: int = 0
        self.misses: int = 0

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        key = _make_key(messages, kwargs)
        try:
            cached = self._cache[key]
            self.hits += 1
            logger.debug("Cache hit  (hits=%d misses=%d)", self.hits, self.misses)
            # Return zero usage — a cache hit costs no tokens.
            return LLMResponse(text=cached, usage={"input_tokens": 0, "output_tokens": 0})
        except KeyError:
            pass

        self.misses += 1
        logger.debug("Cache miss (hits=%d misses=%d)", self.hits, self.misses)
        response = self.llm.complete(messages, **kwargs)
        self._cache[key] = response.text
        return response

    @property
    def hit_rate(self) -> float | None:
        """Cache hit rate in [0, 1], or None if no calls have been made."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else None
