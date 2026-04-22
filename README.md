<p align="center">
  <img src="resources/promptforge-logo.png" width="500" alt="PromptForge logo"/>
</p>

<!-- <h1 align="center">PromptForge</h1> -->
<p align="center"><em>Iterative, example-based prompt optimization — no fine-tuning required.</em></p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a>
</p>

---

## The Idea

Traditional approaches to making LLMs perform complex tasks:

| Approach | Problem |
|---|---|
| **Fine-tuning** | Expensive, requires GPU infra, opaque |
| **RAG** | Retrieves examples at runtime, doesn't generalize rules |
| **Manual prompting** | Doesn't scale, hard to cover all edge cases |

**PromptForge takes a different approach**: feed labeled examples to a *Prompt Engineering Agent* that distills patterns, rules, and edge cases into a comprehensive system prompt. The prompt *is* the learned model — human-readable, editable, and version-controlled.

```
[Seed prompt] + [Examples batch] → Optimizer LLM → [Improved prompt v1]
[Prompt v1]   + [Training log]   → Optimizer LLM → [Improved prompt v2]
     ...                    continues until convergence
```

---

## Installation

```bash
# Install from GitHub (PyPI release coming soon):
pip install git+https://github.com/NiccoloTogni/prompt-forge.git

# With optional extras:
pip install "prompt-forge[pdf] @ git+https://github.com/NiccoloTogni/prompt-forge.git"        # PDF
pip install "prompt-forge[excel] @ git+https://github.com/NiccoloTogni/prompt-forge.git"      # Excel
pip install "prompt-forge[docx] @ git+https://github.com/NiccoloTogni/prompt-forge.git"       # Word
pip install "prompt-forge[sqlalchemy] @ git+https://github.com/NiccoloTogni/prompt-forge.git" # SQL storage
pip install "prompt-forge[all] @ git+https://github.com/NiccoloTogni/prompt-forge.git"        # Everything
```

---

## Quick Start

### 1. Implement an LLM client

PromptForge is provider-agnostic. Wrap any LLM in the `LLMClient` protocol:

```python
from prompt_forge import LLMMessage, LLMResponse

class MyLLM:
    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # call your provider here
        ...
        return LLMResponse(text=..., usage={"input_tokens": ..., "output_tokens": ...})
```

<details>
<summary>Example: Azure OpenAI Chat Completions (text only)</summary>

```python
from openai import AzureOpenAI
from prompt_forge import LLMMessage, LLMResponse

class AzureClient:
    def __init__(self, deployment: str, **kwargs):
        self.client = AzureOpenAI(**kwargs)
        self.deployment = deployment

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            **kwargs,
        )
        return LLMResponse(
            text=resp.choices[0].message.content,
            usage={
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        )

llm = AzureClient(
    deployment="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
    api_key="your-key",
)
```
</details>

<details>
<summary>Example: Azure OpenAI Responses API (native file support)</summary>

Use this when you want to pass PDFs, images, and other files directly to the model
without text extraction. Requires `native_files=True` on the inference agent and a
model that supports the Responses API (api-version `2025-03-01-preview` or later).

```python
import base64
from openai import AzureOpenAI
from prompt_forge import LLMMessage, LLMResponse, TextPart, FilePart

class AzureResponsesClient:
    def __init__(self, deployment: str, **kwargs):
        self.client = AzureOpenAI(**kwargs)
        self.deployment = deployment

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        input_ = []
        for m in messages:
            if isinstance(m.content, str):
                input_.append({"role": m.role, "content": m.content})
            else:
                parts = []
                for part in m.content:
                    if isinstance(part, TextPart):
                        parts.append({"type": "input_text", "text": part.text})
                    elif isinstance(part, FilePart):
                        if part.file_id:
                            parts.append({"type": "input_file", "file_id": part.file_id})
                        else:
                            data = base64.b64encode(part.path.read_bytes()).decode()
                            mime = part.media_type or "application/octet-stream"
                            parts.append({
                                "type": "input_file",
                                "filename": part.path.name,
                                "file_data": f"data:{mime};base64,{data}",
                            })
                input_.append({"role": m.role, "content": parts})

        resp = self.client.responses.create(model=self.deployment, input=input_, **kwargs)
        return LLMResponse(
            text=resp.output_text,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )

llm = AzureResponsesClient(
    deployment="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2025-03-01-preview",
    api_key="your-key",
)

# Enable native file passing at inference time
agent = project.get_inference_agent(native_files=True)
result = agent.run(input_file="invoice.pdf")   # PDF passed natively — no text extraction
```

> **Note:** The Chat Completions client above does not handle `FilePart` content — use it only
> with `native_files=False`. Using `native_files=True` (the default) with a text-only client
> will raise an error from the provider when it receives unexpected content types.
</details>

---

### 2. Set up a project

```python
from prompt_forge import Project

project = Project("invoice_extraction", llm=llm)

# Define what an "example" looks like (role → file extension)
project.set_bundle_schema(
    input=".pdf",
    expected_output=".json",
)

# Optional: domain context helps the optimizer understand the task
project.set_context(
    "These are heat exchanger purchase orders from European manufacturers. "
    "Fields to extract: model, manufacturer, thermal capacity (kW), "
    "pressure rating (bar), material, price, delivery date. "
    "Units are metric. Prices in EUR unless stated otherwise."
)

# Starting point — can be very generic
project.set_seed_prompt(
    "You are a data extraction agent. Extract all relevant fields from "
    "the provided document and return them as structured JSON."
)
```

### 3. Load training examples

```
training_data/
    example_001/
        input.pdf
        expected_output.json
    example_002/
        input.pdf
        expected_output.json
    ...
```

```python
project.add_examples_from_directory("./training_data/")
print(f"Loaded {project.num_examples} examples")
```

### 4. Train

Split your examples into training and validation sets, then run the loop:

```python
from prompt_forge import TrainingConfig, train_val_split

train_bundles, val_bundles = train_val_split(project.bundles, val_ratio=0.2, seed=42)

report = project.train(
    train_bundles,
    val_bundles=val_bundles,
    config=TrainingConfig(
        batch_size=5,        # Examples per optimizer call
        max_iterations=20,   # Hard stop
        patience=3,          # Stop after 3 non-improving iterations
    ),
    eval_strategy="json_fields",   # Field-by-field JSON comparison
)

for r in report:
    status = "✓" if r.improved else "✗"
    before = f"{r.score_before:.2f}" if r.score_before is not None else "—"
    after  = f"{r.score_after:.2f}"  if r.score_after  is not None else "—"
    print(f"Iter {r.iteration}: {before} → {after} {status}")

# Training signals whether human review is recommended
if report.refinement_recommended:
    score_str = f"{report.final_score:.2f}" if report.final_score is not None else "unknown"
    print(f"Score {score_str} — consider reviewing and editing the prompt manually")
```

### 5. Run inference

```python
agent = project.get_inference_agent()
result = agent.run(input_file="new_invoice.pdf")
print(result)   # str, or dict if output_schema is set
```

---

## Features

### Structured JSON output

Declare that your task produces structured JSON and the optimizer will automatically generate prompts that enforce valid JSON output. At inference time, the agent parses and validates the response.

```python
project.set_output_schema({
    "invoice_number": "string",
    "supplier":       "string",
    "total_eur":      "number",
    "line_items":     "array",
    "delivery_date":  "string",
})

report = project.train(eval_strategy="json_fields")

agent = project.get_inference_agent()
result = agent.run(input_file="invoice.pdf")
print(result["total_eur"])   # dict, not a string
```

If the schema is not set explicitly, the optimizer **auto-detects** structured output by inspecting expected-output files: if ≥50% parse as JSON objects, it infers the schema from the union of their top-level keys.

You can also supply an exact JSON Schema object:

```python
project.set_output_schema({
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total_eur":      {"type": "number"},
    }
})
```

---

### Context window management

Prevent optimizer calls from exceeding your model's context window:

```python
report = project.train(
    config=TrainingConfig(max_tokens=100_000),   # Hard limit for the optimizer call
)
```

- If the **full batch** exceeds the budget, it is trimmed automatically and a `WARNING` is logged.
- If a **single example** is too large to fit on its own, training fails immediately with a clear error message identifying the offending example.

For precise token counting, provide a model-specific tokenizer:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

report = project.train(
    config=TrainingConfig(max_tokens=128_000),
    optimizer_kwargs={"token_estimator": lambda text: len(enc.encode(text))},
)
```

The default estimator uses `len(text) // 4` (~4 chars/token).

---

### Reproducibility

Set a `seed` to make batch selection deterministic across runs:

```python
report = project.train(
    config=TrainingConfig(seed=42),
)
```

---

### Retry and resilience

All LLM calls (optimizer, evaluator, inference agent) automatically retry on failure with exponential backoff. Configure globally via `TrainingConfig` or per-component:

```python
report = project.train(
    config=TrainingConfig(
        max_retries=5,      # attempts per call (default 3)
        retry_delay=2.0,    # initial wait in seconds, doubles each attempt (default 1.0)
    ),
)

# Or on the inference agent directly
agent = project.get_inference_agent(max_retries=3, retry_delay=1.0)
```

---

### LLM response caching

Wrap any LLM client with `CachedLLM` to cache responses by input hash. Identical calls are served from the cache without a network round-trip — token usage for cache hits is reported as zero.

```python
from prompt_forge import CachedLLM, Project

# In-memory cache (cleared on process exit)
project = Project("my_project", llm=CachedLLM(my_llm))

# Persistent disk cache across runs (requires: pip install diskcache)
import diskcache
project = Project("my_project", llm=CachedLLM(my_llm, cache=diskcache.Cache(".llm_cache")))
```

The cache key is a SHA-256 hash of the full message content and kwargs. `FilePart` inputs are keyed by file *bytes*, not path — so a stale hit after a file update is not possible. You can monitor efficiency via `cached_llm.hit_rate`.

The eval loop is the highest-value target: the same validation bundles are re-evaluated every iteration, so cache hit rates are near 100% once the val set has been seen once.

---

### Concurrent batch inference

When inputs contain native file parts (PDFs, images), the agent cannot batch them into a single call. Set `max_workers` to process them concurrently instead of sequentially:

```python
agent = project.get_inference_agent(max_workers=8)
results = agent.run_bundle_batch(bundles)   # up to 8 concurrent LLM calls

# Or via TrainingConfig for the eval loop
report = project.train(
    config=TrainingConfig(max_workers=8),
)
```

Results are always returned in the same order as the input bundles regardless of completion order.

---

### Custom optimizer prompts

The prompts used by the Prompt Engineering Agent and the consolidation step are fully configurable. Inspect the defaults to understand the expected format, then pass overrides:

```python
from prompt_forge import DEFAULT_OPTIMIZER_PROMPT, DEFAULT_CONSOLIDATION_PROMPT

# Inspect or extend the defaults
print(DEFAULT_OPTIMIZER_PROMPT)

report = project.train(
    optimizer_kwargs={
        "optimizer_prompt": my_custom_optimizer_prompt,
        "consolidation_prompt": my_custom_consolidation_prompt,
    },
)
```

---

### Context retrieval (RAG / web search hook)

Inject external context before each inference call by providing a `context_retriever`. The callable receives the query text and the LLM client, and returns a string that is prepended to the user message as `<retrieved_context>`:

```python
def my_retriever(query: str, llm) -> str:
    # query is the best available text representation of the input
    results = my_vector_store.search(query, top_k=3)
    return "\n\n".join(r.text for r in results)

agent = project.get_inference_agent(context_retriever=my_retriever)
result = agent.run(input_file="document.pdf")
```

The retriever can also call back into the LLM client for query rewriting or re-ranking:

```python
def rewriting_retriever(query: str, llm) -> str:
    # Use a cheap model call to turn the raw input into a focused search query
    rewrite_resp = llm.complete([
        LLMMessage(role="system", content="Extract a short search query from this input."),
        LLMMessage(role="user", content=query[:2000]),
    ])
    hits = my_vector_store.search(rewrite_resp.text.strip(), top_k=5)
    return "\n\n".join(h.text for h in hits)
```

If the retriever raises, the error is logged as a warning and inference continues without retrieved context — the retriever never blocks the main call.

> **Important:** if your production agent uses a context retriever, pass the same retriever to the training loop via `TrainingConfig(context_retriever=...)`. Without it, the eval agent sees a different input distribution than production and the training signal is misleading.

```python
report = project.train(
    train_bundles,
    val_bundles=val_bundles,
    config=TrainingConfig(context_retriever=my_retriever),
)
```

---

### Prompt consolidation

After many iterations the optimizer accumulates rules and the prompt grows. When you decide it has become unwieldy, call `consolidate()` explicitly to merge redundant and overlapping rules while preserving all distinct coverage:

```python
# After a training run, compress the latest prompt
project.consolidate()

# Or consolidate a specific version
project.consolidate(version=5)

# Then continue training from the consolidated baseline
report = project.train(train_bundles, val_bundles=val_bundles, config=config)
```

Consolidation saves the result as a new prompt version with a `[CONSOLIDATION]` entry in the training log, so the history stays complete and auditable. It is never triggered automatically — the decision is always yours.

---

### Evaluation strategies

```python
# Field-by-field JSON comparison — ideal for data extraction.
# Includes fuzzy matching: dates are normalised across formats,
# numbers tolerate minor floating-point differences, and a configurable
# numeric_tolerance handles rounding variation.
project.train(eval_strategy="json_fields")

# Token F1 similarity — robust to word order, good for free-text tasks
project.train(eval_strategy="similarity")

# LLM-as-judge — most flexible
project.train(eval_strategy="llm_judge")

# Exact string match
project.train(eval_strategy="exact_match")

# Skip evaluation — always accept new prompt (faster, less controlled)
project.train(eval_strategy="none")

# Custom evaluator
from prompt_forge import Evaluator, EvalResult

class MyEvaluator(Evaluator):
    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        score = my_comparison(actual, expected)
        return EvalResult(score=score, passed=score > 0.8)

project.train(eval_strategy=MyEvaluator())
```

`SimilarityEvaluator` supports three methods:

```python
from prompt_forge import SimilarityEvaluator

# Default: token F1 (ROUGE-1) — no dependencies, good for free-text
project.train(eval_strategy=SimilarityEvaluator(method="token"))

# Character-level difflib — only useful when exact character fidelity matters
project.train(eval_strategy=SimilarityEvaluator(method="char"))

# Semantic cosine similarity — best quality, requires an embedding function
embed = lambda text: my_embedding_model.encode(text).tolist()
project.train(eval_strategy=SimilarityEvaluator(method="embedding", embed_fn=embed))
```

---

### Train / validation split

Use `train_val_split` to create a reproducible split before training. Pass the val set explicitly so you control exactly which examples are used for scoring:

```python
from prompt_forge import train_val_split, TrainingConfig

train_bundles, val_bundles = train_val_split(
    project.bundles,
    val_ratio=0.2,   # 20% held out for evaluation
    seed=42,         # reproducible across runs
)

# Or fix the exact number of val examples:
train_bundles, val_bundles = train_val_split(project.bundles, val_size=10, seed=42)

report = project.train(
    train_bundles,
    val_bundles=val_bundles,
    config=TrainingConfig(batch_size=5),
)
```

Pass both `train_bundles` and `val_bundles` so the optimizer only sees training examples while scoring uses the held-out set. Omitting `train_bundles` falls back to all loaded examples.

> **Note:** if `val_bundles` is not provided, the evaluator is skipped and `min_improvement` / `patience` have no effect — all `max_iterations` will run.

---

### Batch selection strategies

```python
from prompt_forge import FailurePriorityBatchStrategy

# Focus on examples the current prompt fails on
project.train(batch_strategy=FailurePriorityBatchStrategy())
```

---

### Prompt versioning

```python
# List all versions
for v in project.list_versions():
    print(f"v{v.version}: score={v.eval_score:.2f}  {v.training_log_entry[:80]}")

# Get a specific version
v3 = project.get_prompt(version=3)
print(v3.prompt_text)
print(v3.output_schema)   # JSON schema if applicable

# Deploy a specific version
agent = project.get_inference_agent(version=3)
```

---

### Storage backends

```python
from prompt_forge import Project, SQLAlchemyStore

# Filesystem (default) — JSON files, human-readable, git-friendly
project = Project("my_project", llm=llm)

# Any SQL database via SQLAlchemy
# Requires: pip install "prompt-forge[sqlalchemy]"
# Plus the relevant DB driver, e.g. psycopg2, pyodbc, etc.

# PostgreSQL
project = Project("my_project", llm=llm,
    store=SQLAlchemyStore("postgresql+psycopg2://user:pass@host/db"))

# Azure SQL Server
project = Project("my_project", llm=llm,
    store=SQLAlchemyStore(
        "mssql+pyodbc://user:pass@server.database.windows.net/db"
        "?driver=ODBC+Driver+18+for+SQL+Server"
    ))

# SQLite file (single-file alternative to the default JSON layout)
project = Project("my_project", llm=llm,
    store=SQLAlchemyStore("sqlite:///./my_project/prompts.db"))

# Multiple projects sharing the same database — namespaced by project_name
store = SQLAlchemyStore("postgresql+psycopg2://user:pass@host/db", project_name="invoice_extraction")
project = Project("invoice_extraction", llm=llm, store=store)
```

---

### Custom file loaders

```python
from prompt_forge import get_default_loader

loader = get_default_loader()

def load_parquet(path) -> str:
    import pandas as pd
    return pd.read_parquet(path).to_string()

loader.register(".parquet", load_parquet)
project = Project("my_project", llm=llm, file_loader=loader)
```

Supported out of the box: `.txt`, `.md`, `.json`, `.csv`, `.pdf`, `.xlsx`, `.xls`, `.docx`.

---

### Training callbacks and multi-file examples

```python
def on_iteration(result):
    print(f"[Iter {result.iteration}] "
          f"{result.score_before:.3f} → {result.score_after:.3f} "
          f"{'✓ IMPROVED' if result.improved else '✗'}")
    print(f"  Learned: {result.learnings[:200]}")

# Multi-file examples: CSV data + template → specification document
project.set_bundle_schema(
    input_data=".csv",
    template=".docx",
    expected_output=".docx",
)

# Custom inference function for complex pipelines
def my_inference(prompt_text: str, bundle) -> str:
    contents = bundle.load_contents()
    input_text = preprocess(contents["input_data"].text)
    return call_my_pipeline(prompt_text, input_text)

project.train(
    config=TrainingConfig(batch_size=5, max_iterations=20),
    on_iteration=on_iteration,
    inference_fn=my_inference,
)
```

---

### Variable-length file bundles (variadic roles)

Some tasks have a fixed "main" file plus a variable number of attachments — e.g. an e-mail with N PDF attachments, or a product sheet with N reference images. Mark those roles as `variadic`:

```python
project.set_bundle_schema(
    mail=".txt",
    attachments=".pdf",
    expected_output=".json",
    variadic=["attachments"],   # 0..N files allowed for this role
)
```

Directory layout — all files whose name starts with the role are collected:

```
training_data/
    example_001/
        mail.txt
        attachments_1.pdf
        attachments_2.pdf
        attachments_3.pdf
        expected_output.json
    example_002/
        mail.txt
        # no attachments — that's fine for a variadic role
        expected_output.json
```

With `native_files=True` (the default) each file is passed as a separate `FilePart` inside the same XML tags. In text-extraction mode (`native_files=False`, requires an explicit `file_loader`), all files for a variadic role are concatenated into a single `<role>…</role>` block.

---

## Architecture

```
prompt_forge/
├── __init__.py              # Public API
├── project.py               # Project — main entry point
├── bundle.py                # ExampleBundle, BundleSchema, BundleCollection
├── utils.py                 # train_val_split and other helpers
├── _retry.py                # call_with_retry — exponential backoff used across all LLM calls
├── caching.py               # CachedLLM — transparent response cache wrapper
├── llm/
│   └── client.py            # LLMClient protocol (provider-agnostic)
├── file_loaders/
│   └── loader.py            # FileLoader with built-in + custom loaders
├── training/
│   ├── pipeline.py          # TrainingPipeline + TrainingReport
│   ├── optimizer.py         # PromptOptimizer — the PE agent
│   ├── prompt.py            # DEFAULT_OPTIMIZER_PROMPT, DEFAULT_CONSOLIDATION_PROMPT
│   ├── batch_strategy.py    # Batch selection strategies
│   └── training_log.py      # Compact training history
├── inference/
│   └── agent.py             # InferenceAgent — uses trained prompts
├── evaluation/
│   └── evaluator.py         # Evaluation strategies
└── storage/
    └── project_store.py     # FileSystemStore + SQLAlchemyStore backends
```

---

## License

MIT
