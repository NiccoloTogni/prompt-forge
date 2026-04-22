# CachedLLM Reference

`CachedLLM` is a transparent wrapper around any `LLMClient`. Identical inputs return a cached response without making a network call.

---

## Usage

```python
from prompt_forge import CachedLLM

# In-memory cache (cleared on process exit)
cached_llm = CachedLLM(my_llm)

# Persistent cache with diskcache (survives restarts)
import diskcache
cached_llm = CachedLLM(my_llm, cache=diskcache.Cache(".llm_cache"))

# Drop in wherever an LLMClient is accepted
pipeline = TrainingPipeline(llm=cached_llm, store=store, ...)
agent = InferenceAgent(llm=cached_llm, prompt_text="...")
```

---

## Constructor

```python
CachedLLM(llm: LLMClient, cache: Any = None)
```

| Parameter | Description |
|-----------|-------------|
| `llm` | The underlying LLMClient to wrap. |
| `cache` | A dict-like object used as the store. Defaults to an in-memory `dict`. Pass a persistent backend (e.g. `diskcache.Cache`, a Redis client) to retain the cache across runs. |

Any object supporting `__getitem__` / `__setitem__` and raising `KeyError` on misses works as a cache backend.

---

## Cache key

The key is a **SHA-256 hash** of the serialized messages and kwargs. This means:

- Two calls with identical text, roles, and kwargs always hit the cache.
- `FilePart` content is keyed by **file bytes** (not the path). If a file is updated on disk, the next call will be a cache miss and fetch a fresh response.
- Different `temperature` or other kwargs produce different keys.

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `hits` | `int` | Number of cache hits since construction. |
| `misses` | `int` | Number of cache misses since construction. |
| `hit_rate` | `float \| None` | `hits / (hits + misses)`. `None` if no calls have been made yet. |

---

## Token accounting

Cache hits return `usage={"input_tokens": 0, "output_tokens": 0}`. This means:

- `agent.tokens_used` only accumulates tokens from actual network calls.
- `TrainingReport.total_tokens_used` reflects real API spend.

---

## Persistent backends

```python
# diskcache — disk-backed, fast, supports TTL
import diskcache
cache = diskcache.Cache(".llm_cache", expire=3600)  # 1-hour TTL
cached_llm = CachedLLM(my_llm, cache=cache)

# shelve — stdlib, no extra dependencies
import shelve
cache = shelve.open(".llm_cache")
cached_llm = CachedLLM(my_llm, cache=cache)
```

`diskcache` is the recommended choice for most use cases.

---

## When to use caching

- **Development and iteration:** cache responses so you can re-run training loops without paying for repeated LLM calls on the same data.
- **Evaluation stability:** when running the same eval multiple times (e.g. comparing two prompt versions), caching ensures both see identical LLM outputs.
- **Cost control during debugging:** wrap the LLM with a cache while debugging pipeline logic, then remove the wrapper for production runs.

---

## Design notes

- `CachedLLM` only caches `complete()` calls — it has no knowledge of streaming or other methods.
- The cache is **not** thread-safe by default when using a plain `dict`. For concurrent inference (`max_workers > 1`), use `diskcache.Cache` or another thread-safe backend.
- Only `response.text` is cached — metadata like model version or latency is not preserved in cached responses.
