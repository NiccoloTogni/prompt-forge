# Retrievers Reference

Context retrievers are callables that fetch relevant background information before each inference call. The retrieved text is injected as `<retrieved_context>…</retrieved_context>` at the start of every user message.

## Retriever protocol

```python
Callable[[query: str, llm: LLMClient], str]
```

Any function or callable object with this signature can be used as a `context_retriever`. Exceptions are caught by `InferenceAgent` and logged as warnings — a failed retrieval never aborts inference.

---

## WebSearchRetriever

Built-in retriever that fetches live web search results.

```python
from prompt_forge.retrievers import WebSearchRetriever

retriever = WebSearchRetriever(
    provider="duckduckgo",
    num_results=3,
    rewrite_query=False,
)

agent = InferenceAgent(llm=llm, prompt_text="...", context_retriever=retriever)
```

### Constructor

```python
WebSearchRetriever(
    provider: str = "duckduckgo",
    api_key: str | None = None,
    num_results: int = 3,
    rewrite_query: bool = False,
    max_input_chars: int = 1000,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `provider` | `"duckduckgo"` | Search backend. One of `"duckduckgo"` or `"tavily"`. |
| `api_key` | `None` | API key. Required for `"tavily"`. |
| `num_results` | `3` | Number of results to fetch and include as context. |
| `rewrite_query` | `False` | When `True`, uses the LLM to distil a short search query from the raw input before searching. Recommended when inputs are long documents. |
| `max_input_chars` | `1000` | Maximum input characters passed to the query rewriter. Inputs are truncated to this length before rewriting. |

### Providers

#### DuckDuckGo (free, no key required)

```bash
pip install "prompt-forge[duckduckgo]"
```

```python
retriever = WebSearchRetriever(provider="duckduckgo")
```

No account or API key needed. Rate limits apply for high-volume usage.

#### Tavily (best quality, free tier available)

```bash
pip install "prompt-forge[tavily]"
```

```python
retriever = WebSearchRetriever(
    provider="tavily",
    api_key="tvly-your-key-here",
)
```

Designed for RAG applications. Returns more relevant, cleaner snippets than general-purpose search.

### Query rewriting

When `rewrite_query=True`, the LLM generates a focused search query before calling the search backend:

```python
retriever = WebSearchRetriever(
    provider="duckduckgo",
    rewrite_query=True,
    max_input_chars=500,  # truncate long inputs before rewriting
)
```

Use this when your inputs are long documents (invoices, contracts) and you want the search to target the key topic rather than the raw text. Query rewriting uses one extra LLM call per inference; the original query is used as fallback if rewriting fails.

---

## Custom retrievers

Any callable matching the protocol works:

```python
# Vector store retriever
def vector_retriever(query: str, llm) -> str:
    results = vector_store.search(query, top_k=5)
    return "\n\n".join(r.text for r in results)

agent = InferenceAgent(llm=llm, prompt_text="...", context_retriever=vector_retriever)
```

```python
# Class-based retriever with state
class PineconeRetriever:
    def __init__(self, index, top_k=5):
        self.index = index
        self.top_k = top_k

    def __call__(self, query: str, llm) -> str:
        embedding = embed(query)
        results = self.index.query(embedding, top_k=self.top_k)
        return "\n\n".join(m["text"] for m in results["matches"])
```

The `llm` parameter is available for LLM-assisted steps (query rewriting, answer synthesis) but can be ignored if not needed.

---

## Training consistency

If your production agent uses a retriever, the eval agent during training must use the same retriever. Otherwise, the eval scores measure a different distribution than production performance:

```python
retriever = WebSearchRetriever(provider="tavily", api_key="tvly-...")

# Production agent
agent = InferenceAgent(llm=llm, prompt_text="...", context_retriever=retriever)

# Training — pass the retriever to the config
report = pipeline.train(
    train_bundles,
    val_bundles=val_bundles,
    config=TrainingConfig(context_retriever=retriever),
)
```

The pipeline passes `config.context_retriever` to the internal eval `InferenceAgent` automatically.

---

## Retrieved context format

Retrieved text is injected into the user message before the input:

```
<retrieved_context>
[1] Title
https://example.com
Snippet text...

[2] Another title
https://example2.com
Another snippet...
</retrieved_context>

<input>
... the actual input ...
</input>
```

The system prompt should instruct the model on how to use this context. Example:

```
If a <retrieved_context> block is present, use it to inform your analysis,
but prioritise the information in the input over web results.
```

---

## Design notes

- **Query derivation:** for `run(input_text=...)`, the raw input text is the query. For file inputs, the file's text content is extracted via the file loader and used as the query (path string used as fallback if loading fails). For bundle inputs, text is extracted from all input roles. In all cases, `rewrite_query=True` can further distil the query before searching.
- **Empty results:** if the retriever returns an empty string, no `<retrieved_context>` tag is added to the message.
- **Caching with retrieval:** when using `CachedLLM`, cache keys include the full message content — including `<retrieved_context>`. If the retriever returns different results across calls (e.g. due to changing web content), the cache will miss. To make retrieval deterministic for caching, use a static knowledge base rather than live web search.
