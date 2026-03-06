<p align="center">
  <img src="resources/promptforge-logo.svg" width="200" alt="PromptForge logo"/>
</p>

<h1 align="center">PromptForge</h1>
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
pip install prompt-forge

# With file type support:
pip install prompt-forge[pdf]     # PDF loading (pdfplumber + OCR)
pip install prompt-forge[excel]   # Excel loading
pip install prompt-forge[docx]    # Word document loading
pip install prompt-forge[all]     # Everything
```

**Install directly from GitHub** (latest development version):

```bash
pip install git+https://github.com/NiccoloTogni/prompt-forge.git

# With extras:
pip install "prompt-forge[all] @ git+https://github.com/NiccoloTogni/prompt-forge.git"
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

```python
report = project.train(
    batch_size=5,                    # Examples per optimizer call
    max_iterations=20,               # Hard stop
    eval_strategy="json_fields",     # Field-by-field JSON comparison
    patience=3,                      # Stop after 3 non-improving iterations
)

for r in report:
    status = "✓" if r.improved else "✗"
    print(f"Iter {r.iteration}: {r.score_before:.2f} → {r.score_after:.2f} {status}")

# Training signals whether human review is recommended
if report.refinement_recommended:
    print(f"Score {report.final_score:.2f} — consider running project.refine()")
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

### Human interactive refinement

After automated training converges, use `project.refine()` to start an interactive session where you give direct feedback on the prompt:

```python
report = project.train(...)

if report.refinement_recommended:
    result = project.refine()
    print(f"Revised {result.num_revisions} times, saved versions: {result.saved_versions}")
```

**Session commands:**

| Input | Effect |
|---|---|
| Any text | Revise the prompt based on that feedback |
| `test` | Run on a random example and show the output |
| `test <id>` | Run on a specific example |
| `show` | Display the full current prompt |
| `save` | Save the current prompt as a new version |
| `done` / `quit` | End the session |

For non-CLI environments (Jupyter, web apps), pass custom `input_fn` / `output_fn` callbacks:

```python
result = project.refine(
    input_fn=my_widget.get_input,
    output_fn=my_widget.display,
)
```

Or use `InteractiveOptimizer` directly without a full project:

```python
from prompt_forge import InteractiveOptimizer

optimizer = InteractiveOptimizer(llm=my_llm, store=my_store, bundles=my_bundles)
result = optimizer.run_session(prompt_text=my_prompt)
```

---

### Context window management

Prevent optimizer calls from exceeding your model's context window:

```python
report = project.train(
    max_tokens=100_000,   # Hard limit for the optimizer call
)
```

- If the **full batch** exceeds the budget, it is trimmed automatically and a `WARNING` is logged.
- If a **single example** is too large to fit on its own, training fails immediately with a clear error message identifying the offending example.

For precise token counting, provide a model-specific tokenizer:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

report = project.train(
    max_tokens=128_000,
    optimizer_kwargs={"token_estimator": lambda text: len(enc.encode(text))},
)
```

The default estimator uses `len(text) // 4` (~4 chars/token).

---

### Evaluation strategies

```python
# Field-by-field JSON comparison — ideal for data extraction
project.train(eval_strategy="json_fields")

# Text similarity (difflib ratio)
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
from prompt_forge import FileSystemStore, SQLiteStore

# Filesystem (default) — JSON files, human-readable, git-friendly
project = Project("my_project", llm=llm)

# SQLite — better for querying history and metrics
store = SQLiteStore("./my_project/prompts.db")
project = Project("my_project", llm=llm, store=store)
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
├── llm/
│   └── client.py            # LLMClient protocol (provider-agnostic)
├── file_loaders/
│   └── loader.py            # FileLoader with built-in + custom loaders
├── training/
│   ├── pipeline.py          # TrainingPipeline + TrainingReport
│   ├── optimizer.py         # PromptOptimizer — the PE agent
│   ├── batch_strategy.py    # Batch selection strategies
│   └── training_log.py      # Compact training history
├── inference/
│   └── agent.py             # InferenceAgent — uses trained prompts
├── evaluation/
│   └── evaluator.py         # Evaluation strategies
├── interactive/
│   └── optimizer.py         # InteractiveOptimizer — human refinement
└── storage/
    └── project_store.py     # FileSystemStore + SQLiteStore backends
```

---

## License

MIT
