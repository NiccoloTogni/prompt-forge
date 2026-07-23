# Concepts

The mental model behind prompt-forge. Read this before the reference docs to understand why things are designed the way they are.

---

## The prompt is the model

Traditional ML stores learned knowledge in weights — a binary artifact that requires the original framework to use. prompt-forge stores it in a system prompt — a text file that runs on any LLM API.

The optimizer (itself an LLM) reads labeled examples and extracts rules, edge case handling, and formatting instructions directly into the prompt text. After training, the prompt is self-contained: you can paste it into the OpenAI playground, use it with LangChain, or deploy it with any provider. No framework lock-in.

<!-- This has a concrete implication for `output_schema`: when you set one, the optimizer is instructed to embed JSON formatting rules inside the prompt text itself. The `InferenceAgent.output_schema` parameter adds a safety-net enforcement suffix at runtime, but the prompt already carries the instructions. A user can take the trained prompt and use it standalone. -->

---

## The training loop

Each iteration:

1. **Batch selection** — a subset of training bundles is drawn (randomly by default).
2. **Score current prompt** — the current prompt's validation score is `score_before`. It is computed once (at iteration 1, or when the prompt was accepted in a previous iteration) and reused — the baseline is never re-scored, which halves eval cost and keeps comparisons against a stable baseline instead of a freshly-sampled, noisy one.
3. **Optimize** — the optimizer receives the current prompt, the training batch with ground truth, and a compact summary of past iterations. It returns an improved prompt plus `learnings` (rules it extracted) and `issues` (gaps it could not resolve with the current data).
4. **Score new prompt** — the new prompt is run on the same validation set. This is `score_after`.
5. **Accept or reject** — accepted if `score_after >= score_before + min_improvement`. Accepted versions are saved; rejected ones are not.
6. **Early stopping** — if `patience` consecutive iterations produce no **strict** validation improvement, training stops. Note the asymmetry with acceptance: with `min_improvement=0` a tie is still accepted (the new prompt learned from a fresh batch the val set may not measure), but it counts toward `patience` — so training stops on a flat plateau instead of churning versions until `max_iterations`.
7. **Final test evaluation** — if `test_bundles` were provided, the final prompt is scored on them exactly once, after the loop (`TrainingReport.test_score`).

The optimizer accumulates context across iterations via the training log — a compact summary of past learnings. This prevents it from re-learning the same rules and allows it to build incrementally.

---

## Bundles and roles

A **bundle** is one training example. It is a named collection of files:

```python
ExampleBundle(
    bundle_id="invoice_001",
    files={
        "input": Path("invoice_001/input.pdf"),
        "expected_output": Path("invoice_001/expected_output.json"),
    }
)
```

Role names are how prompt-forge distinguishes inputs from ground truth. Any role whose name contains `"expected"` or `"output"` (case-insensitive) is treated as ground truth and excluded from inference — the model never sees it. All other roles are passed as inputs.

This convention means you can have multiple input roles (`email`, `attachments`) and the system automatically knows which file to compare against. If your ground truth role uses a non-standard name, rename it to include `"expected"` or `"output"`.

**Variadic roles** accept zero or more files per bundle — useful for tasks like email-with-attachments where the number of attachments varies. They are optional during validation.

---

## Train, validation, and test sets

These must be kept strictly separate:

- **Training set** — the only data the optimizer sees. Rules are extracted from these examples.
- **Validation set** — used to score each candidate prompt. Never shown to the optimizer.
- **Test set** — evaluated once at the very end to get an unbiased performance estimate. Never seen during training or validation.

If training and validation overlap, the optimizer can memorize specific examples rather than learning generalizable rules — the validation score will be inflated and performance on new data will be worse.

The test set matters for a subtler reason too: the loop *hill-climbs on the validation score* — every accept/reject decision selects for prompts that happen to do well on val. After many iterations the final validation score is optimistically biased, even with perfect train/val separation. Only a set the loop never used for any decision gives an honest generalization number.

Use `train_val_test_split` and pass the test set to `train()` — it is evaluated exactly once, on the final prompt:

```python
from prompt_forge import train_val_test_split

train, val, test = train_val_test_split(all_bundles, val_ratio=0.2, test_ratio=0.2, seed=42)
report = project.train(train, val_bundles=val, test_bundles=test, config=...)
print(report.final_score)  # val score — biased upward by prompt selection
print(report.test_score)   # held-out test score — the number to report
```

When `test_score` is available, `refinement_recommended` is based on it rather than the validation score.

---

## Prompt versioning

Every accepted iteration saves a `PromptVersion` to the store. Versions are numbered sequentially and linked via `parent_version`. Consolidated versions are marked with metadata and `[CONSOLIDATION]` in the log entry — the chain stays intact.

You can deploy any version independently:

```python
agent = InferenceAgent.from_store(llm, store, version=5)
```

The version history doubles as an audit trail: each version records the validation score, eval details, batch IDs, and learnings from the iteration that produced it.

---

## Evaluation

The evaluator compares actual model output to expected output and returns a score in [0, 1]. The training loop uses `mean_score` (average over the validation set) to decide whether a new prompt is better.

**No evaluator:** the pipeline accepts every optimizer suggestion and runs all `max_iterations`. Useful when you have no ground truth or the task is too open-ended for automatic scoring. `patience` and `min_improvement` are ignored.

**No validation set:** if you set an evaluator but pass no `val_bundles`, a warning is logged and evaluation is skipped — same behaviour as no evaluator.

The `pass_threshold` on each evaluator controls `EvalResult.passed` and `BatchEvalResult.pass_rate`, but does not affect which prompt versions are accepted — that uses `mean_score` only.

---

## The eval/production distribution gap

If your production agent uses a **context retriever** (RAG, web search), the model's input distribution at runtime includes retrieved context. If you train without the retriever, the eval agent sees a different distribution and the training signal is misleading — you are optimizing a prompt for inputs that differ from what production receives.

**Fix:** pass the same retriever to `TrainingConfig.context_retriever`. The pipeline creates the eval agent with this retriever, so training scores reflect production conditions.

```python
config = TrainingConfig(context_retriever=my_retriever)
```

This is the most important thing to get right when using retrieval — a mismatch here can make training converge in the wrong direction.

A second, milder gap: for text inputs the eval agent batches many validation examples into **one** LLM call (XML-wrapped), while production typically runs one call per input. The model can behave slightly differently on batched inputs, so validation scores may not exactly reflect single-call production behaviour. Native-file inputs already run per-input; for text inputs, pass a custom `inference_fn` to `TrainingPipeline` if you need eval to mirror production calls exactly (at N× the eval calls).

---

## Consolidation

The optimizer is naturally additive: each iteration appends new rules and edge cases. Over many iterations the prompt grows, which increases token cost per call and can degrade model adherence.

**Consolidation** compresses the prompt by merging redundant, overlapping, or contradictory rules while preserving distinct coverage. It is always explicit and user-triggered — never automatic — because it is a lossy operation and the right time to do it is a judgment call.

After consolidation, the compressed prompt is saved as a new version. Training continues from this baseline. The history stays complete — consolidation is just another version, not a history rewrite.

When to consolidate: after a plateau of non-improving iterations, or when the prompt has grown to a length that makes you uncomfortable.

---

## Batch inference

For text inputs, the agent batches all examples into a single LLM call using XML output tags:

```
<input id="1">…</input>  <input id="2">…</input>
                    ↓ one call ↓
<output id="1">…</output>  <output id="2">…</output>
```

This is more token-efficient than N sequential calls and avoids per-call latency. If the LLM response is missing any `<output id="N">` tag, the agent falls back to sequential per-input calls automatically.

For inputs with native file parts (PDFs, images), XML wrapping is not possible — the agent falls back to per-input calls, optionally concurrent via `max_workers`.

---

## Project vs TrainingPipeline

**`Project`** is the high-level entry point. It wires together the store, schema, LLM, evaluator, and pipeline into a single convenience object. Most users interact only with `Project`.

**`TrainingPipeline`** is the component `Project` delegates to internally. Use it directly when you need a custom batch selection strategy, a custom inference function, multiple projects sharing configuration, or more fine-grained control over when and how the store is used.

---

## Token cost model

Each training iteration spends tokens on:

- **Optimizer call** — current prompt + training batch + history → new prompt. Largest single cost.
- **Two eval passes** — validation set scored before and after optimization.
- **Optionally, one train-batch eval pass** — `eval_train=True` also scores the new prompt on the training batch. Off by default.

The same validation bundles are re-evaluated every iteration. Wrapping the LLM with `CachedLLM` caches responses by content hash — hit rates approach 100% after the first run, cutting eval costs to near zero for subsequent iterations.

Use `max_total_tokens` in `TrainingConfig` to set a hard budget. Training stops early and logs a warning when exceeded.
