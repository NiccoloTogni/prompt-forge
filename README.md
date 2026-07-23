<p align="center">
  <img src="https://raw.githubusercontent.com/NiccoloTogni/prompt-forge/master/resources/promptforge-logo.png" width="500" alt="PromptForge logo"/>
</p>

<p align="center"><em>Iterative, example-based prompt optimization — no fine-tuning required.</em></p>

<p align="center">
  <a href="https://github.com/NiccoloTogni/prompt-forge/blob/master/docs/quickstart.md">Quickstart</a> ·
  <a href="https://github.com/NiccoloTogni/prompt-forge/blob/master/docs/concepts.md">Concepts</a> ·
  <a href="https://github.com/NiccoloTogni/prompt-forge/tree/master/docs/reference">Reference</a>
</p>

<p align="center">
  <a href="https://github.com/NiccoloTogni/prompt-forge/actions/workflows/ci.yml"><img src="https://github.com/NiccoloTogni/prompt-forge/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://pypi.org/project/prompt-forge/"><img src="https://img.shields.io/pypi/v/prompt-forge" alt="PyPI"/></a>
  <a href="https://github.com/NiccoloTogni/prompt-forge/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license"/></a>
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

> **Prefer zero setup?** [PromptFit](https://github.com/NiccoloTogni/promptfit) packages the same
> principles as a Claude Code skill — no Python package, no API key, same portable-prompt deliverable.
> prompt-forge is the library it graduates to: large jobs, native/binary file inputs, batch inference,
> RAG-matched eval, SQL storage.

---

## Installation

```bash
pip install prompt-forge

# Optional extras
pip install "prompt-forge[pdf]"         # PDF loading (pdfplumber + OCR)
pip install "prompt-forge[sqlalchemy]"  # SQL storage backend
pip install "prompt-forge[all]"         # everything

# Latest from source
pip install git+https://github.com/NiccoloTogni/prompt-forge.git
```

Supported file extras: `pdf`, `excel`, `docx`. Storage: `sqlalchemy`. Search: `duckduckgo`, `tavily`. Web UI: `app`.

---

## Quick Start

```python
from prompt_forge import Project, TrainingConfig, train_val_test_split

# 1. Wrap your LLM (any provider)
class MyLLM:
    def complete(self, messages, **kwargs):
        ...  # call your provider, return LLMResponse(text=..., usage={...})

project = Project("invoice_extractor", llm=MyLLM())

# 2. Define the example structure and a seed prompt
project.set_bundle_schema(input=".pdf", expected_output=".json")
project.set_context("Purchase order invoices from European manufacturers.")
project.set_seed_prompt("Extract all relevant fields and return structured JSON.")

# 3. Load examples (one subdirectory per example)
project.add_examples_from_directory("./training_data/")

# 4. Train — val drives accept/reject; test is scored once at the end (unbiased)
train, val, test = train_val_test_split(project.bundles, seed=42)
report = project.train(train, val_bundles=val, test_bundles=test,
                       eval_strategy="json_fields",
                       config=TrainingConfig(batch_size=5, max_iterations=20, patience=3))
print(report.test_score)  # held-out generalization score

# 5. Run inference with the trained prompt
agent = project.get_inference_agent()
result = agent.run(input_file="new_invoice.pdf")
```

See **[docs/quickstart.md](https://github.com/NiccoloTogni/prompt-forge/blob/master/docs/quickstart.md)** for the full step-by-step guide.

---

## Features

- **Iterative optimization** — batch-based training loop with early stopping and full prompt version history
- **Honest scoring** — train/val/test split with a held-out test score computed once, outside the accept/reject loop
- **Structured JSON output** — schema-aware prompts with automatic JSON parsing at inference time
- **Flexible evaluation** — exact match, JSON field comparison, token F1, embedding similarity, LLM-as-judge, or custom
- **Batch inference** — true single-call batching with chunking, concurrent fallback for file inputs
- **Context retrieval** — RAG / web search hook (`context_retriever`) with built-in `WebSearchRetriever`
- **LLM caching** — `CachedLLM` wrapper for zero-cost repeated calls during development
- **Prompt consolidation** — explicit compression of accumulated rules when the prompt grows unwieldy
- **Retry & resilience** — exponential backoff on all LLM calls, configurable per component
- **Storage backends** — filesystem (default, git-friendly) or any SQL database via SQLAlchemy
- **Provider-agnostic** — implement one `complete()` method to use any LLM

---

## Architecture

```
prompt_forge/
├── project.py               # Project — main entry point
├── bundle.py                # ExampleBundle, BundleSchema, BundleCollection
├── caching.py               # CachedLLM — transparent response cache wrapper
├── retrievers.py            # WebSearchRetriever (DuckDuckGo, Tavily)
├── utils.py                 # train_val_split and other helpers
├── _retry.py                # Exponential backoff used across all LLM calls
├── llm/client.py            # LLMClient protocol (provider-agnostic)
├── file_loaders/loader.py   # FileLoader with built-in + custom loaders
├── training/
│   ├── pipeline.py          # TrainingPipeline, TrainingConfig, TrainingReport
│   ├── optimizer.py         # PromptOptimizer — the PE agent
│   ├── batch_strategy.py    # Batch selection strategies
│   └── training_log.py      # Compact training history
├── inference/agent.py       # InferenceAgent — uses trained prompts
├── evaluation/evaluator.py  # Evaluation strategies
└── storage/project_store.py # FileSystemStore + SQLAlchemyStore
```

---

## License

MIT
