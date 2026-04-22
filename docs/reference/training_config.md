# Training Reference

This page covers the classes returned by and passed to `TrainingPipeline.train()`.

---

## TrainingConfig

Configuration for a single training run. All fields have defaults so you can start with `TrainingConfig()`.

```python
from prompt_forge import TrainingConfig

config = TrainingConfig(
    batch_size=5,
    max_iterations=30,
    patience=5,
    eval_train=False,
)
```

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `10` | Number of training examples per optimizer call. |
| `max_iterations` | `int` | `20` | Maximum training iterations before stopping. |
| `min_improvement` | `float` | `0.0` | Minimum score delta required to accept a new prompt version. Only meaningful when an evaluator is set. |
| `patience` | `int` | `5` | Stop early after this many consecutive non-improving iterations. Ignored when no evaluator is set. |
| `val_max_tokens` | `int \| None` | `None` | Token budget per eval batch call. `None` = no limit. |
| `auto_save` | `bool` | `True` | Save training state after each iteration for resumption. |
| `output_schema` | `dict \| None` | `None` | JSON Schema passed to the optimizer when the task requires structured output. |
| `refinement_threshold` | `float` | `0.8` | Scores below this flag `TrainingReport.refinement_recommended = True`. |
| `max_tokens` | `int \| None` | `None` | Per-call context window limit for the optimizer LLM call. |
| `max_total_tokens` | `int \| None` | `None` | Total token budget for the entire training run. Training stops early when exceeded. |
| `inference_temperature` | `float \| None` | `None` | Temperature for eval agent inference calls. `None` = use model default. Set to `0.0` for deterministic eval. |
| `seed` | `int \| None` | `None` | Random seed for reproducible batch selection. |
| `native_files` | `bool` | `True` | Pass input files natively as multimodal content. Set to `False` to extract text instead. |
| `max_retries` | `int` | `3` | Retry attempts per failed LLM call (optimizer + eval agent). |
| `retry_delay` | `float` | `1.0` | Initial retry wait in seconds. Doubles each attempt (exponential backoff). |
| `max_workers` | `int \| None` | `None` | Concurrent LLM calls during per-input batch fallback. `None` = serial. |
| `context_retriever` | `Callable \| None` | `None` | Retriever used by the eval agent. Should match the production agent's retriever for training/eval distribution consistency. |
| `eval_train` | `bool` | `False` | Also evaluate the new prompt on the training batch (accepted iterations only). Costs extra tokens — disabled by default. |

---

## TrainingPipeline

Orchestrates the incremental prompt training loop.

```python
from prompt_forge import TrainingPipeline

pipeline = TrainingPipeline(
    llm=my_llm,
    store=my_store,
    evaluator=my_evaluator,
    context="Domain context passed to the optimizer",
    on_iteration=lambda r: print(f"Iter {r.iteration}: {r.score_after:.3f}"),
)
report = pipeline.train(train_bundles, val_bundles=val_bundles, config=config)
```

### Constructor parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `llm` | `LLMClient` | LLM client for both optimization and eval inference. |
| `store` | `ProjectStore` | Storage backend for prompt versions and state. |
| `evaluator` | `Evaluator \| None` | Scoring strategy. `None` = accept all iterations (no evaluation). |
| `optimizer` | `PromptOptimizer \| None` | Custom optimizer. Defaults to `PromptOptimizer` with `llm` and `context`. |
| `batch_strategy` | `BatchStrategy \| None` | Batch selection strategy. Defaults to `RandomBatchStrategy`. |
| `file_loader` | `FileLoader \| None` | File loader for reading example files. |
| `context` | `str` | Domain context forwarded to the optimizer. |
| `inference_fn` | `Callable \| None` | Custom inference function `(prompt_text, bundle) → str`. Overrides the default LLM-based inference. |
| `on_iteration` | `Callable \| None` | Callback called with the `IterationResult` after each iteration. |

### `train(train_bundles, *, val_bundles=None, config=None) → TrainingReport`

Run the training loop. Accepts `BundleCollection` or `list[ExampleBundle]` for both arguments.

Raises `RuntimeError` if:
- `train_bundles` is empty.
- No seed prompt is found in the store.

### `consolidate(version=None) → PromptVersion`

Compress the current (or a specific) prompt by merging redundant rules. This is an explicit, user-triggered operation — **never called automatically**. The consolidated prompt is saved as a new version with `metadata["consolidation"] = True`.

```python
# After training has grown the prompt unwieldy
consolidated = pipeline.consolidate()
print(f"Compressed to v{consolidated.version}")
# Continue training from the new, shorter baseline
report2 = pipeline.train(more_bundles, ...)
```

---

## IterationResult

Result of a single training iteration. Returned in `TrainingReport.iterations`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `iteration` | `int` | 1-based iteration number. |
| `prompt_version` | `int` | Version number of the accepted prompt (unchanged on rejection). |
| `score_before` | `float \| None` | Val score before optimization. `None` if no evaluator or val set. |
| `score_after` | `float \| None` | Val score after optimization. `None` if no evaluator or val set. |
| `improved` | `bool` | Whether this iteration's prompt was accepted. |
| `learnings` | `str` | Key rules extracted by the optimizer from this batch. |
| `issues` | `str` | Outstanding gaps flagged by the optimizer. Empty string if none. |
| `batch_ids` | `list[str]` | Bundle IDs used in this iteration's training batch. |
| `train_score` | `float \| None` | Score on training batch. Set only when `eval_train=True` and iteration improved. |
| `val_example_scores` | `dict[str, float] \| None` | Per-example val scores: `bundle_id → score`. Set when improved and evaluator is active. |
| `train_example_scores` | `dict[str, float] \| None` | Per-example train scores. Set only when `eval_train=True` and iteration improved. |
| `tokens_used` | `int \| None` | Total tokens consumed in this iteration (optimizer + eval). |

---

## TrainingReport

Returned by `TrainingPipeline.train()`. Iterable — `for r in report` yields `IterationResult`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `iterations` | `list[IterationResult]` | All iteration results. |
| `final_version` | `int` | Version number of the best accepted prompt. |
| `final_score` | `float \| None` | Score of the final accepted prompt. `None` if no evaluator. |
| `refinement_recommended` | `bool` | `True` if `final_score` is below `refinement_threshold` or unknown. |
| `total_tokens_used` | `int` | Cumulative input + output tokens for the entire run. |

### Properties

#### `all_issues → list[tuple[int, str]]`

All non-empty issues flagged by the optimizer, as `(iteration, issues_text)` pairs.

### Methods

#### `aggregate_issues(llm) → str`

Summarise recurring issues across all iterations using one LLM call. Identifies distinct root causes and ranks them by frequency. Returns `""` if no issues were recorded.

```python
summary = report.aggregate_issues(llm)
if summary:
    print("Recurring training gaps:")
    print(summary)
```

---

## Displaying per-example scores

```python
for r in report:
    if r.improved and r.val_example_scores:
        worst = min(r.val_example_scores, key=r.val_example_scores.get)
        print(f"Iter {r.iteration}: worst example = {worst} ({r.val_example_scores[worst]:.2f})")
```

---

## Token budget management

```python
config = TrainingConfig(
    max_total_tokens=500_000,   # hard stop on total token spend
    max_iterations=50,          # also capped by iteration count
)
report = pipeline.train(...)
print(f"Used {report.total_tokens_used:,} tokens")
```

Training logs a warning and stops early when `max_total_tokens` is reached.

---

## Design notes

- **No evaluator:** when `evaluator=None`, the pipeline accepts all optimizer suggestions, runs all `max_iterations`, and ignores `patience`, `min_improvement`, and validation scoring. Useful when you want to run a fixed number of optimization passes without automated scoring.
- **No val set:** when `evaluator` is set but `val_bundles` is empty, a warning is logged and evaluation is skipped. The prompt is accepted unconditionally (same as no-evaluator mode).
- **Resumption:** `auto_save=True` (default) writes training state after each iteration. If training is interrupted, the next call to `train()` on the same pipeline instance will detect and log the restored state. The training log provides context to the optimizer so it avoids repeating the same changes.
- **Consolidation timing:** consolidate after a plateau (several iterations without improvement) or when the prompt exceeds a character count you're comfortable with. Consolidation does not reset training history — the optimizer still has access to prior learnings.
