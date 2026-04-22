# Evaluators Reference

Evaluators compare model output to ground truth and return a score between 0 and 1. They are passed to `TrainingPipeline` and used during the validation phase of each training iteration.

---

## EvalResult

Result of evaluating a single example.

```python
@dataclasses.dataclass
class EvalResult:
    score: float      # 0.0 to 1.0
    passed: bool
    details: dict     # evaluator-specific breakdown
    feedback: str     # human-readable explanation of failures
```

---

## BatchEvalResult

Aggregated result of evaluating a batch of examples.

```python
@dataclasses.dataclass
class BatchEvalResult:
    mean_score: float
    pass_rate: float
    individual_results: list[EvalResult]
    example_ids: list[str]       # bundle_id per example, parallel to individual_results
    failed_examples: list[dict]  # bundle_id + feedback for failures only
```

### Properties

#### `example_scores → dict[str, float]`

`bundle_id → score` for every evaluated example:

```python
result = evaluator.evaluate_batch(examples)
for bundle_id, score in result.example_scores.items():
    print(f"{bundle_id}: {score:.2f}")
```

### Methods

#### `to_dict() → dict`

```python
{
    "mean_score": 0.85,
    "pass_rate": 0.90,
    "num_examples": 10,
    "num_passed": 9,
    "example_scores": {"invoice_001": 1.0, "invoice_002": 0.5},
    "failed_examples": [{"bundle_id": "invoice_002", "feedback": "...", "score": 0.5}]
}
```

---

## Evaluator (base class)

```python
from prompt_forge.evaluation.evaluator import Evaluator, EvalResult

class MyEvaluator(Evaluator):
    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        score = my_scoring_logic(actual, expected)
        return EvalResult(score=score, passed=score >= 0.75)
```

### Abstract methods

#### `evaluate(actual, expected, **kwargs) → EvalResult`

Compare a single pair of actual vs expected strings. Must return an `EvalResult`.

### Inherited methods

#### `evaluate_batch(results) → BatchEvalResult`

Evaluate a batch of `(bundle_id, actual, expected)` triples. Calls `evaluate()` for each and aggregates. Provided by the base class — you only need to implement `evaluate()`.

---

## ExactMatchEvaluator

Exact string match after normalization.

```python
from prompt_forge import ExactMatchEvaluator

evaluator = ExactMatchEvaluator(
    normalize_whitespace=True,  # collapse whitespace before comparing
    case_sensitive=False,       # case-insensitive by default
)
```

Score is 1.0 on match, 0.0 otherwise. Good for structured, deterministic outputs.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `normalize_whitespace` | `True` | Collapse runs of whitespace to a single space. |
| `case_sensitive` | `False` | If `False`, compare lowercased strings. |

---

## JsonFieldEvaluator

Field-by-field JSON comparison with fuzzy matching.

```python
from prompt_forge import JsonFieldEvaluator

evaluator = JsonFieldEvaluator(
    pass_threshold=0.75,
    ignore_fields=["timestamp", "id"],
    case_sensitive=False,
    numeric_tolerance=0.01,
    normalize_dates=True,
    normalize_numbers=True,
)
```

Score = `correct_fields / total_fields`. Supports nested objects (dot-separated field paths in feedback). Good for data extraction tasks.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pass_threshold` | `0.75` | Minimum score to count as passed. |
| `ignore_fields` | `[]` | Field names to skip during comparison. |
| `case_sensitive` | `False` | String comparison is case-insensitive by default. |
| `numeric_tolerance` | `1e-9` | Max absolute difference for numeric equality. Set `0.01` to allow rounding. |
| `normalize_dates` | `True` | Treat date strings representing the same day as equal regardless of format (e.g. `"2024-01-15"` == `"15/01/2024"`). |
| `normalize_numbers` | `True` | Strip thousands separators before comparing numeric strings (e.g. `"1,234"` == `"1234"`). |

---

## SimilarityEvaluator

Text similarity with three methods.

```python
from prompt_forge import SimilarityEvaluator

# Token F1 (default — no dependencies)
evaluator = SimilarityEvaluator(method="token", pass_threshold=0.75)

# Character-level difflib
evaluator = SimilarityEvaluator(method="char")

# Embedding cosine similarity
evaluator = SimilarityEvaluator(method="embedding", embed_fn=my_embed_fn)
```

### Methods

| Method | Description | Dependencies |
|--------|-------------|--------------|
| `"token"` | Unigram F1 (ROUGE-1). Harmonic mean of token precision and recall. Robust to word order and paraphrasing. | None |
| `"char"` | `difflib.SequenceMatcher` ratio. Sensitive to exact character matches. | None |
| `"embedding"` | Cosine similarity between embedding vectors. Best for semantic equivalence. | `embed_fn` must be provided. |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pass_threshold` | `0.75` | Minimum score to count as passed. |
| `method` | `"token"` | One of `"token"`, `"char"`, `"embedding"`. |
| `embed_fn` | `None` | Required when `method="embedding"`. `Callable[[str], list[float]]`. |

---

## LLMJudgeEvaluator

Uses an LLM to score output quality.

```python
from prompt_forge import LLMJudgeEvaluator

evaluator = LLMJudgeEvaluator(
    llm=my_llm,
    pass_threshold=0.75,
    task_description="Extract invoice fields from PDF text",
)
```

The LLM is prompted to return a JSON object:

```json
{
    "score": 0.9,
    "correct_aspects": ["vendor correctly extracted", "date format correct"],
    "errors": ["total amount off by $0.01"],
    "feedback": "Nearly perfect extraction, minor rounding error on total"
}
```

Useful when the expected output is fuzzy (natural language tasks, paraphrasing). Adds LLM cost per evaluation call — consider using `SimilarityEvaluator(method="token")` for cheaper approximate scoring.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `llm` | required | LLM client for judging. |
| `pass_threshold` | `0.75` | Minimum score to count as passed. |
| `task_description` | `""` | Optional context injected into the judge prompt. |

---

## Implementing a custom evaluator

Subclass `Evaluator` and implement `evaluate()`:

```python
from prompt_forge.evaluation.evaluator import Evaluator, EvalResult

class RougeL(Evaluator):
    def __init__(self, pass_threshold: float = 0.5):
        self.pass_threshold = pass_threshold

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        score = self._rouge_l(actual, expected)
        return EvalResult(
            score=score,
            passed=score >= self.pass_threshold,
            feedback=f"ROUGE-L: {score:.2f}",
        )

    @staticmethod
    def _rouge_l(actual: str, expected: str) -> float:
        # ... your implementation ...
        ...
```

Pass it to `TrainingPipeline(evaluator=RougeL())`.

---

## Choosing an evaluator

| Task type | Recommended evaluator |
|-----------|----------------------|
| Structured JSON extraction | `JsonFieldEvaluator` |
| Exact string match (codes, IDs) | `ExactMatchEvaluator` |
| Free-text summarisation | `SimilarityEvaluator(method="token")` |
| Semantic equivalence | `SimilarityEvaluator(method="embedding")` |
| Complex quality judgment | `LLMJudgeEvaluator` |
| Multiple criteria | Custom evaluator composing the above |

---

## Design notes

- **`pass_threshold`** controls `EvalResult.passed` and `BatchEvalResult.pass_rate`. The training pipeline uses `mean_score` for iteration comparison, not `pass_rate`, so the threshold does not affect which prompt versions are accepted.
- **`JsonFieldEvaluator` with nested objects:** nested keys appear as dot-separated paths in feedback (e.g. `address.city`). Ignored via `ignore_fields` at any nesting level using the leaf key only (not the path).
- **`LLMJudgeEvaluator` parsing failure:** if the LLM response cannot be parsed as JSON, score defaults to 0.0 and `feedback` contains the raw response prefix. This is rare but can happen with low-quality models — switch to a stronger model if you observe it frequently.
- **Weighted field evaluation:** `JsonFieldEvaluator` weights all fields equally. If some fields matter more (e.g. `total_amount` is more important than `vendor_address`), implement a custom evaluator that weights scores per field.
