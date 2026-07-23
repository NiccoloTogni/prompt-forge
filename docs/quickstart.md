# Quickstart

This guide walks you from zero to a trained prompt. We use invoice field extraction as the running example, but the same steps apply to any task.

**Prerequisites:** Python 3.10+, an LLM API you can call.

---

## Install

```bash
pip install git+https://github.com/NiccoloTogni/prompt-forge.git
pip install "prompt-forge[pdf] @ git+https://github.com/NiccoloTogni/prompt-forge.git"
```

---

## Step 1 — Implement an LLM client

prompt-forge is provider-agnostic. Any class with a `complete()` method works:

```python
from prompt_forge import LLMMessage, LLMResponse

class MyLLM:
    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        # call your provider here
        ...
        return LLMResponse(
            text=response.text,
            usage={"input_tokens": n_in, "output_tokens": n_out},
        )
```

**Azure OpenAI — Chat Completions (text only):**

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

**Azure OpenAI — Responses API (native PDF/image support):**

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
    deployment="gpt-5.4",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2025-03-01-preview",
    api_key="your-key",
)
```

> Use the Responses API client when passing PDFs or images natively (`native_files=True`). Use the Chat Completions client with `native_files=False` for text-only workflows.

---

## Step 2 — Prepare training data

Each example lives in its own subdirectory:

```
training_data/
    invoice_001/
        input.pdf
        expected_output.json
    invoice_002/
        input.pdf
        expected_output.json
    ...
```

`expected_output.json` is the ground truth your model should produce:

```json
{"vendor": "Acme Corp", "total": 1234.50, "date": "2024-01-15", "invoice_number": "INV-001"}
```

You need at least ~10 examples to get meaningful training signal. ~30+ is recommended for stable convergence.

---

## Step 3 — Set up a project

```python
from prompt_forge import Project

project = Project("invoice_extractor", llm=llm)

# What does one example look like?
project.set_bundle_schema(input=".pdf", expected_output=".json")

# Domain context helps the optimizer understand the task
project.set_context(
    "Purchase order invoices from European manufacturers. "
    "Extract: vendor name, total (EUR), invoice date, invoice number, line items."
)

# Seed prompt — can be very generic, the optimizer refines it
project.set_seed_prompt(
    "You are a data extraction agent. Extract all relevant fields from "
    "the provided document and return them as structured JSON."
)

project.add_examples_from_directory("./training_data/")
print(f"Loaded {project.num_examples} examples")
```

---

## Step 4 — Train

Split examples into training, validation, and test sets, then run the loop:

```python
from prompt_forge import TrainingConfig, train_val_test_split

train_bundles, val_bundles, test_bundles = train_val_test_split(
    project.bundles, val_ratio=0.2, test_ratio=0.2, seed=42
)

report = project.train(
    train_bundles,
    val_bundles=val_bundles,
    test_bundles=test_bundles,     # evaluated once at the end — unbiased score
    eval_strategy="json_fields",   # field-by-field JSON comparison
    config=TrainingConfig(
        batch_size=5,       # examples per optimizer call
        max_iterations=20,  # hard stop
        patience=3,         # stop after 3 iterations without val improvement
    ),
)

for r in report:
    status = "✓" if r.improved else "✗"
    before = f"{r.score_before:.2f}" if r.score_before is not None else "—"
    after  = f"{r.score_after:.2f}"  if r.score_after  is not None else "—"
    print(f"Iter {r.iteration}: {before} → {after} {status}")

print(f"\nFinal val score: {report.final_score:.2f} (version {report.final_version})")
if report.test_score is not None:
    print(f"Held-out test score: {report.test_score:.2f}")  # the number to report

# One LLM call — summarises recurring gaps across all iterations
# Tells you what training data to add for the next run
summary = report.aggregate_issues(llm)
if summary:
    print("\nRecurring issues:\n", summary)
```

Training saves a new prompt version after each accepted iteration. If interrupted, the next `train()` call restores state and continues from where it left off.

---

## Step 5 — Run inference

```python
agent = project.get_inference_agent()
result = agent.run(input_file="new_invoice.pdf")
print(result)
```

To use a specific version:

```python
agent = project.get_inference_agent(version=report.final_version)
```

---

## Common next steps

### Why a test set?

The training loop accepts or rejects each candidate prompt based on its validation score — it hill-climbs on the val set, so the final `report.final_score` is optimistically biased even with perfect train/val separation. The `test_bundles` you pass to `train()` are evaluated exactly once, on the final prompt, after the loop: `report.test_score` (with per-example detail in `report.test_example_scores`) is the honest generalization number, and it also drives `report.refinement_recommended` when present.

### Speed up development with LLM caching

```python
from prompt_forge import CachedLLM
import diskcache

cached_llm = CachedLLM(llm, cache=diskcache.Cache(".llm_cache"))
project = Project("invoice_extractor", llm=cached_llm)
```

The validation set is re-evaluated every iteration — cache hit rates approach 100% quickly, so training costs drop significantly after the first run.

### Consolidate a long prompt

After many iterations the prompt grows. When it feels unwieldy, compress it:

```python
project.consolidate()
# Then continue training from the new baseline
report = project.train(train_bundles, val_bundles=val_bundles, ...)
```

---

## Where to go next

| Topic | Reference |
|-------|-----------|
| Bundles, roles, variadic files | [reference/bundle.md](reference/bundle.md) |
| All TrainingConfig options | [reference/training_config.md](reference/training_config.md) |
| InferenceAgent methods and batch inference | [reference/inference_agent.md](reference/inference_agent.md) |
| Evaluation strategies | [reference/evaluators.md](reference/evaluators.md) |
| Storage backends | [reference/storage.md](reference/storage.md) |
| RAG and web search | [reference/retrievers.md](reference/retrievers.md) |
| LLM caching | [reference/caching.md](reference/caching.md) |
| Mental model | [concepts.md](concepts.md) |
