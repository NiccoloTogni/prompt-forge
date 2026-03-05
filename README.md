# APE — Automatic Prompt Engineering

A Python library for incrementally training system prompts from examples. Instead of updating model weights, APE updates prompts — producing human-readable, versionable, editable "learned knowledge".

## The Idea

Traditional approaches to making LLMs perform complex tasks:
- **Fine-tuning**: expensive, requires infrastructure, black box
- **RAG**: retrieves examples at runtime, doesn't generalize rules
- **Manual prompting**: doesn't scale, hard to cover all edge cases

**APE takes a different approach**: feed thousands of examples to a "Prompt Engineering Agent" that distills patterns, rules, and edge cases into a comprehensive system prompt. The prompt *is* the learned model — it's human-readable, editable, and version-controlled.

## Installation

```bash
pip install ape-toolkit

# With file type support:
pip install ape-toolkit[pdf]     # PDF loading (pdfplumber + OCR)
pip install ape-toolkit[excel]   # Excel loading
pip install ape-toolkit[all]     # Everything
```

## Quick Start

### 1. Set Up Your LLM Client

APE is provider-agnostic. You supply a client that implements the `LLMClient` protocol:

```python
from openai import AzureOpenAI
from ape import LLMMessage, LLMResponse

class AzureClient:
    """Example: Azure OpenAI wrapper."""

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
            raw=resp,
        )

llm = AzureClient(
    deployment="gpt-4o",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01",
    api_key="your-key",
)
```

### 2. Define a Project

```python
from ape import Project

project = Project("heat_exchanger_extraction", llm=llm)

# Define what an "example" looks like
project.set_bundle_schema(
    input=".pdf",
    expected_output=".json",
)

# Provide domain context
project.set_context(
    "These are heat exchanger purchase orders from European manufacturers. "
    "Fields to extract: model, manufacturer, thermal capacity (kW), "
    "pressure rating (bar), material, dimensions, price, delivery date. "
    "Units are metric. Prices in EUR unless stated otherwise."
)

# Starting point — can be very generic
project.set_seed_prompt(
    "You are a data extraction agent. Extract all relevant fields from "
    "the provided document and return them as structured JSON."
)
```

### 3. Load Training Examples

Organize examples in directories:

```
training_data/
    example_001/
        input.pdf           # The purchase order PDF
        expected_output.json # The correct extracted data
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
results = project.train(
    batch_size=5,          # Examples per iteration
    max_iterations=20,     # Maximum iterations
    eval_strategy="json_fields",  # Field-by-field JSON comparison
    patience=3,            # Stop after 3 iterations without improvement
)

# Check results
for r in results:
    print(f"Iteration {r.iteration}: "
          f"score {r.score_before:.2f} → {r.score_after:.2f} "
          f"({'✓' if r.improved else '✗'})")
```

### 5. Use in Production

```python
agent = project.get_inference_agent()
result = agent.run(input_file="new_order.pdf")
print(result)  # Extracted JSON
```

## How Training Works

```
Iteration 1:
  [Seed prompt] + [Batch of 5 examples] → Optimizer → [Improved prompt v1]
  
Iteration 2:  
  [Prompt v1] + [Training log] + [Next 5 examples] → Optimizer → [Prompt v2]
  
Iteration 3:
  [Prompt v2] + [Training log] + [Next 5 examples] → Optimizer → [Prompt v3]
  
  ...continues until convergence or max iterations
```

The **training log** is key: it's a compact summary of what was learned in each
iteration. This travels with the prompt so the optimizer doesn't forget previous
learnings when seeing new examples.

The **prompt itself is the compressed knowledge** from all examples seen so far.
You can inspect it, edit it, and understand exactly what the system has learned.


## Key Components

### Bundle Schema

Defines what training examples look like for your project:

```python
# Data extraction: PDF → JSON
project.set_bundle_schema(input=".pdf", expected_output=".json")

# Spec generation: CSV data + template → specification document
project.set_bundle_schema(
    input_data=".csv",
    template=".docx",
    expected_output=".docx",
)

# Translation: source text → translated text
project.set_bundle_schema(source=".txt", expected=".txt")
```

### Evaluation Strategies

```python
# Exact string match
project.train(eval_strategy="exact_match")

# Field-by-field JSON comparison (great for data extraction)
project.train(eval_strategy="json_fields")

# Text similarity (difflib)
project.train(eval_strategy="similarity")

# LLM-as-judge (most flexible, uses your LLM to score quality)
project.train(eval_strategy="llm_judge")

# No evaluation (always accept new prompt — faster, less controlled)
project.train(eval_strategy="none")

# Custom evaluator
from ape import Evaluator, EvalResult

class MyEvaluator(Evaluator):
    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        # Your custom logic
        score = my_comparison(actual, expected)
        return EvalResult(score=score, passed=score > 0.8)

project.train(eval_strategy=MyEvaluator())
```

### Custom File Loaders

```python
from ape import get_default_loader

loader = get_default_loader()

# Register a custom loader for Parquet files
def load_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_string()

loader.register(".parquet", load_parquet)

project = Project("my_project", llm=llm, file_loader=loader)
```

### Storage Backends

```python
from ape import Project, FileSystemStore, SQLiteStore

# Filesystem (default) — simple JSON files
project = Project("my_project", llm=llm)

# SQLite — better for querying history
store = SQLiteStore("./my_project/prompts.db")
project = Project("my_project", llm=llm, store=store)
```

### Prompt Versioning

```python
# List all versions
for v in project.list_versions():
    print(f"v{v.version}: score={v.eval_score}, created={v.created_at}")
    print(f"  Learned: {v.training_log_entry[:100]}")

# Get a specific version
v3 = project.get_prompt(version=3)
print(v3.prompt_text)

# Use a specific version for inference
agent = project.get_inference_agent(version=3)
```

### Inference with Few-Shot Examples

```python
agent = project.get_inference_agent(
    few_shot_examples=[
        {
            "input": "<input>...sample order text...</input>",
            "output": '{"model": "HX-500", "manufacturer": "Alfa Laval", ...}'
        }
    ]
)
```

### Custom Inference Function

If your task requires special processing beyond a simple LLM call:

```python
def my_inference(prompt_text: str, bundle) -> str:
    contents = bundle.load_contents()
    # Custom preprocessing, multi-step pipeline, etc.
    input_text = preprocess(contents["input"].text)
    result = call_my_pipeline(prompt_text, input_text)
    return postprocess(result)

project.train(inference_fn=my_inference)
```

### Monitoring Training Progress

```python
def on_iteration(result):
    print(f"[Iter {result.iteration}] "
          f"Score: {result.score_before:.3f} → {result.score_after:.3f} "
          f"{'✓ IMPROVED' if result.improved else '✗'}")
    print(f"  Learned: {result.learnings[:200]}")

project.train(on_iteration=on_iteration)
```


## Architecture

```
ape/
├── __init__.py              # Public API
├── project.py               # Project class — main entry point
├── bundle.py                # ExampleBundle, BundleSchema, BundleCollection
├── llm/
│   └── protocol.py          # LLMClient protocol (provider-agnostic)
├── file_loaders/
│   └── loader.py            # FileLoader with built-in + custom loaders
├── training/
│   ├── pipeline.py          # TrainingPipeline — orchestrates the loop
│   ├── optimizer.py         # PromptOptimizer — the PE agent
│   ├── batch_strategy.py    # Batch selection strategies
│   └── training_log.py      # Compact training history
├── inference/
│   └── agent.py             # InferenceAgent — uses trained prompts
├── evaluation/
│   └── evaluator.py         # Evaluation strategies
└── storage/
    └── project_store.py     # FileSystem + SQLite backends
```

## License

MIT
