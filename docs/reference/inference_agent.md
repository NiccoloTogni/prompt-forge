# InferenceAgent Reference

`InferenceAgent` is the production-facing component. It takes a trained prompt and uses it to process new inputs, with support for single calls, true batch inference, file inputs, structured output, and context retrieval.

---

## Constructor

```python
from prompt_forge import InferenceAgent

agent = InferenceAgent(
    llm=my_llm,
    prompt_text="You are an expert invoice extractor...",
    native_files=True,
    max_retries=3,
    retry_delay=1.0,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `LLMClient` | required | LLM client used for inference calls. |
| `prompt_text` | `str` | required | System prompt text. Usually loaded from a trained version. |
| `file_loader` | `FileLoader \| None` | `None` | Used to extract text when `native_files=False`. |
| `system_suffix` | `str` | `""` | Extra text appended to the system prompt on every call. |
| `llm_kwargs` | `dict \| None` | `None` | Extra kwargs forwarded to `llm.complete()` (e.g. `{"temperature": 0.0}`). |
| `output_schema` | `dict \| None` | `None` | JSON Schema dict. When set, outputs are parsed as JSON dicts. |
| `token_estimator` | `Callable[[str], int] \| None` | `None` | Token counter for chunking. Defaults to `len(text) // 4`. |
| `native_files` | `bool` | `True` | Pass files as `FilePart` content (requires multimodal LLM client). Set to `False` to extract text instead. |
| `max_retries` | `int` | `3` | Number of retry attempts after a failed LLM call. |
| `retry_delay` | `float` | `1.0` | Initial wait in seconds before the first retry. Doubles each attempt (exponential backoff). |
| `max_workers` | `int \| None` | `None` | Max concurrent LLM calls during per-input fallback. `None` = serial. |
| `context_retriever` | `Callable[[str, LLMClient], str] \| None` | `None` | Called before each inference call to fetch relevant context. See [Context retrieval](#context-retrieval). |

---

## Class methods

### `from_store(llm, store, version=None, **kwargs) → InferenceAgent`

Create an agent from a stored prompt version.

```python
agent = InferenceAgent.from_store(llm, store)           # latest version
agent = InferenceAgent.from_store(llm, store, version=3) # specific version
```

Automatically sets `output_schema` from the stored version if not overridden via `kwargs`.

### `from_project_dir(llm, project_dir, version=None, **kwargs) → InferenceAgent`

Convenience wrapper around `from_store`. Creates a `FileSystemStore` from the path.

```python
agent = InferenceAgent.from_project_dir(llm, "my_project/")
```

---

## Inference methods

### `run(input_text, input_file, input_files, extra_context) → str | dict`

Run inference on a single input. Provide exactly one of the first three arguments.

```python
# Text input
result = agent.run(input_text="What is the capital of France?")

# Single file
result = agent.run(input_file="path/to/document.pdf")

# Multiple named files
result = agent.run(input_files={"contract": "contract.pdf", "addendum": "addendum.pdf"})

# With additional context
result = agent.run(input_text="...", extra_context="Customer tier: premium")
```

Returns a `dict` if `output_schema` is set, otherwise a plain `str`.

### `run_batch(inputs, max_tokens=None) → list[str | dict]`

Run inference on multiple inputs in a **single LLM call** (true batch, not sequential). Chunked automatically when `max_tokens` is set.

```python
results = agent.run_batch([
    {"input_file": "doc1.pdf"},
    {"input_file": "doc2.pdf"},
    {"input_text": "inline input"},
])
```

Each element in `inputs` is a dict with keys matching `run()` parameters. Returns a list in the same order as `inputs`.

**Chunking:** when `max_tokens` is set, inputs are grouped into token-budget chunks and one LLM call is made per chunk. The token estimator is used to estimate input sizes.

**File parts fallback:** when any input contains native file parts (`FilePart`), batch XML wrapping is not possible. The agent falls back to per-input calls, optionally concurrent when `max_workers` is set.

### `run_bundle(bundle) → str`

Run inference on a single `ExampleBundle`. Output roles are excluded from the input. Always returns a plain string (code fences stripped) — intended for training/evaluation pipelines.

### `run_bundle_batch(bundles, max_tokens=None) → list[str]`

Run batch inference on a list of `ExampleBundle` objects. Same chunking and fallback behaviour as `run_batch`. Always returns plain strings.

---

## Context retrieval

The `context_retriever` hook enables RAG and web search patterns. It is called before every inference call and its return value is injected as `<retrieved_context>…</retrieved_context>` at the start of the user message — kept separate from `<additional_context>` (the `extra_context` parameter).

```python
def my_retriever(query: str, llm) -> str:
    results = vector_store.search(query, top_k=3)
    return "\n\n".join(r.text for r in results)

agent = InferenceAgent(
    llm=my_llm,
    prompt_text="...",
    context_retriever=my_retriever,
)
```

The query passed to the retriever is derived from the input:
- For `run(input_text=...)`: the input text itself.
- For `run(input_file=...)`: the file's text content (extracted via the file loader). Falls back to the path string if the file cannot be loaded.
- For `run(input_files=...)`: text content of all files joined with a space. Falls back to path strings for files that cannot be loaded.
- For `run_bundle(...)`: text extracted from all input roles.

Retriever exceptions are caught and logged as warnings — a failed retrieval never aborts inference. Empty strings returned by the retriever are silently ignored.

**Training consistency:** pass the same retriever to `TrainingConfig.context_retriever` so the eval agent during training sees the same distribution as the production agent.

### WebSearchRetriever

Built-in reference implementation for web search:

```python
from prompt_forge import WebSearchRetriever

retriever = WebSearchRetriever(
    provider="duckduckgo",  # or "tavily"
    num_results=3,
    rewrite_query=True,     # use LLM to rewrite the query before searching
)
agent = InferenceAgent(llm=llm, prompt_text="...", context_retriever=retriever)
```

See the `WebSearchRetriever` reference for full parameters and provider setup.

---

## Structured output

When `output_schema` is set, every inference call's output is parsed as JSON:

```python
schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
    }
}
agent = InferenceAgent(llm=llm, prompt_text="...", output_schema=schema)
result = agent.run(input_file="invoice.pdf")
# result is {"vendor": "Acme Corp", "total": 1234.50}
```

The agent adds a JSON-enforcement suffix to the system prompt and parses the response with `json.loads` (stripping markdown code fences first). Raises `ValueError` on parse failure.

**Note on trained prompts:** when you train with `TrainingConfig(output_schema=...)`, the optimizer is instructed to embed JSON formatting rules directly inside the prompt text. This means a trained prompt is self-contained — you can copy it into any platform (OpenAI playground, LangChain, etc.) without needing `InferenceAgent` to add anything. Setting `output_schema` on `InferenceAgent` is useful for the strict parsing guarantee, but is not required for the model to produce JSON.

---

## Token tracking

```python
print(agent.tokens_used)  # cumulative input + output tokens across all calls
```

Thread-safe. Reset by creating a new agent instance.

---

## Properties

### `prompt_info → str`

Human-readable summary of the loaded prompt:

```
Prompt: 42 lines, 1823 chars
Prompt: 15 lines, 620 chars, output_schema=['vendor', 'total', 'date']
```

---

## Batch format internals

Text-only batch calls use XML output tags that are more reliably produced by LLMs than JSON arrays:

```
<input id="1">…</input>
<input id="2">…</input>

<output id="1">…</output>
<output id="2">…</output>
```

If any `<output id="N">` tag is missing from the response, a `ValueError` is raised and the pipeline falls back to sequential per-input calls.

---

## Design notes

- **`native_files=True` (default):** files are passed as `FilePart` binary content parts. Requires an LLM client that supports multimodal input. Batch inference automatically falls back to sequential when file parts are present (XML wrapping doesn't work with binary content).
- **`native_files=False`:** requires an explicit `file_loader`. Text is extracted and sent inline. Works with any LLM but loses image/PDF fidelity.
- **`max_workers`** only applies to the per-input fallback path (when file parts are present). Pure-text batch calls always use the single-call XML format regardless of `max_workers`.
