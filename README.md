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
<summary>Example: OpenAI / Azure OpenAI</summary>

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

### Prompt consolidation

After many iterations the optimizer accumulates rules and the prompt grows. Set `max_prompt_chars` to automatically trigger a consolidation step whenever the prompt exceeds that length — redundant and overlapping rules are merged while keeping all distinct coverage:

```python
report = project.train(
    config=TrainingConfig(max_prompt_chars=8_000),
)
```

---

### Evaluation strategies

```python
# Field-by-field JSON comparison — ideal for data extraction
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

## Architecture

```
prompt_forge/
├── __init__.py              # Public API
├── project.py               # Project — main entry point
├── bundle.py                # ExampleBundle, BundleSchema, BundleCollection
├── utils.py                 # train_val_split and other helpers
├── llm/
│   └── client.py            # LLMClient protocol (provider-agnostic)
├── file_loaders/
│   └── loader.py            # FileLoader with built-in + custom loaders
├── training/
│   ├── pipeline.py          # TrainingPipeline + TrainingReport
│   ├── optimizer.py         # PromptOptimizer — the PE agent
│   ├── prompt.py            # Default and consolidation meta-prompts
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
