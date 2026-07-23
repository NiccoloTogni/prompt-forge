"""
Evaluation strategies for comparing actual vs expected output.

Built-in evaluators:
    - ExactMatchEvaluator: Exact string match (after normalization)
    - JsonFieldEvaluator: Field-by-field JSON comparison
    - SimilarityEvaluator: Token F1 / char difflib / embedding cosine similarity
    - LLMJudgeEvaluator: Uses an LLM to score output quality

Users can implement custom evaluators by subclassing Evaluator.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from typing import Any, Callable

from .._retry import call_with_retry
from ..llm.client import LLMClient, LLMMessage

DEFAULT_PASS_THRESHOLD = 0.75


@dataclasses.dataclass
class EvalResult:
    """Result of evaluating a single example."""

    score: float  # 0.0 to 1.0
    passed: bool
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    feedback: str = ""  # Human-readable explanation of what went wrong

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class BatchEvalResult:
    """Aggregated result of evaluating a batch of examples."""

    mean_score: float
    pass_rate: float
    individual_results: list[EvalResult]
    example_ids: list[str]           # Parallel to individual_results — bundle_id per example
    failed_examples: list[dict]      # Bundle IDs + feedback for failures only

    @property
    def example_scores(self) -> dict[str, float]:
        """Mapping of bundle_id → score for every evaluated example."""
        return dict(zip(self.example_ids, (r.score for r in self.individual_results)))

    def to_dict(self) -> dict:
        return {
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "num_examples": len(self.individual_results),
            "num_passed": sum(1 for r in self.individual_results if r.passed),
            "example_scores": self.example_scores,
            "failed_examples": self.failed_examples,
        }


class Evaluator(ABC):
    """Base class for evaluation strategies."""

    @abstractmethod
    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        """
        Compare actual output to expected output.

        Args:
            actual: The output produced by the inference agent.
            expected: The ground-truth expected output.

        Returns:
            EvalResult with score, pass/fail, and details.
        """
        ...

    def evaluate_batch(
        self,
        results: list[tuple[str, str, str]],  # (bundle_id, actual, expected)
    ) -> BatchEvalResult:
        """Evaluate a batch of results."""
        individual = []
        example_ids = []
        failed = []
        for bundle_id, actual, expected in results:
            result = self.evaluate(actual, expected)
            individual.append(result)
            example_ids.append(bundle_id)
            if not result.passed:
                failed.append({
                    "bundle_id": bundle_id,
                    "feedback": result.feedback,
                    "score": result.score,
                })

        scores = [r.score for r in individual]
        return BatchEvalResult(
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            pass_rate=sum(1 for r in individual if r.passed) / len(individual) if individual else 0.0,
            individual_results=individual,
            example_ids=example_ids,
            failed_examples=failed,
        )


class ExactMatchEvaluator(Evaluator):
    """
    Exact string match after normalization.

    Good for structured outputs where formatting is deterministic.
    """

    def __init__(self, normalize_whitespace: bool = True, case_sensitive: bool = False):
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        a = self._normalize(actual)
        e = self._normalize(expected)
        match = a == e
        return EvalResult(
            score=1.0 if match else 0.0,
            passed=match,
            feedback="" if match else f"Output does not match expected. First diff near position {self._first_diff_pos(a, e)}",
        )

    def _normalize(self, text: str) -> str:
        if self.normalize_whitespace:
            text = " ".join(text.split())
        if not self.case_sensitive:
            text = text.lower()
        return text.strip()

    def _first_diff_pos(self, a: str, b: str) -> int:
        for i, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                return i
        return min(len(a), len(b))


# Common date formats tried in order during date normalization.
_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d",
    "%d %B %Y", "%B %d, %Y",
    "%d %b %Y", "%b %d, %Y",
)


class JsonFieldEvaluator(Evaluator):
    """
    Field-by-field JSON comparison.

    Ideal for data extraction tasks. Reports which fields match,
    which are wrong, and which are missing.

    Fuzzy matching options (all enabled by default):
        - ``normalize_dates``: treat date strings that represent the same calendar day
          as equal, regardless of format (e.g. "2024-01-15" == "15/01/2024").
        - ``normalize_numbers``: strip thousands separators before comparing numeric
          strings (e.g. "1,234.56" == "1234.56").
        - ``numeric_tolerance``: maximum absolute difference for numeric values to be
          considered equal. Defaults to ``1e-9`` (effectively exact). Set a larger value
          (e.g. ``0.01``) to allow rounding differences.
    """

    def __init__(
        self,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        ignore_fields: list[str] | None = None,
        case_sensitive: bool = False,
        numeric_tolerance: float = 1e-9,
        normalize_dates: bool = True,
        normalize_numbers: bool = True,
    ):
        self.pass_threshold = pass_threshold
        self.ignore_fields = set(ignore_fields or [])
        self.case_sensitive = case_sensitive
        self.numeric_tolerance = numeric_tolerance
        self.normalize_dates = normalize_dates
        self.normalize_numbers = normalize_numbers

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        try:
            actual_data = json.loads(actual) if isinstance(actual, str) else actual
        except json.JSONDecodeError:
            return EvalResult(
                score=0.0, passed=False,
                feedback="Actual output is not valid JSON.",
                details={"error": "invalid_json_actual"},
            )
        try:
            expected_data = json.loads(expected) if isinstance(expected, str) else expected
        except json.JSONDecodeError:
            return EvalResult(
                score=0.0, passed=False,
                feedback="Expected output is not valid JSON.",
                details={"error": "invalid_json_expected"},
            )

        field_results = self._compare_fields(actual_data, expected_data)
        total = len(field_results)
        if total == 0:
            return EvalResult(score=1.0, passed=True, details={"fields": {}})

        correct = sum(1 for v in field_results.values() if v["match"])
        score = correct / total

        mismatches = {k: v for k, v in field_results.items() if not v["match"]}
        feedback_lines = []
        for field, info in mismatches.items():
            feedback_lines.append(
                f"  - {field}: expected '{info['expected']}', got '{info['actual']}'"
            )

        return EvalResult(
            score=score,
            passed=score >= self.pass_threshold,
            details={"fields": field_results, "correct": correct, "total": total},
            feedback=f"Field accuracy: {correct}/{total}\nMismatches:\n" + "\n".join(feedback_lines) if feedback_lines else "All fields match.",
        )

    def _compare_fields(
        self, actual: Any, expected: Any, prefix: str = ""
    ) -> dict[str, dict]:
        results = {}

        if isinstance(expected, dict):
            for key, exp_val in expected.items():
                if key in self.ignore_fields:
                    continue
                field_name = f"{prefix}.{key}" if prefix else key
                act_val = actual.get(key, "<MISSING>") if isinstance(actual, dict) else "<MISSING>"

                if isinstance(exp_val, dict):
                    sub = self._compare_fields(act_val if isinstance(act_val, dict) else {}, exp_val, field_name)
                    results.update(sub)
                elif isinstance(exp_val, list):
                    match = self._compare_values(act_val, exp_val)
                    results[field_name] = {"expected": exp_val, "actual": act_val, "match": match}
                else:
                    match = self._compare_values(act_val, exp_val)
                    results[field_name] = {"expected": exp_val, "actual": act_val, "match": match}
        else:
            field_name = prefix or "<root>"
            results[field_name] = {
                "expected": expected,
                "actual": actual,
                "match": self._compare_values(actual, expected),
            }

        return results

    def _compare_values(self, actual: Any, expected: Any) -> bool:
        if actual == expected:
            return True

        if isinstance(actual, str) and isinstance(expected, str):
            a, e = actual.strip(), expected.strip()
            if not self.case_sensitive:
                a, e = a.lower(), e.lower()
            if a == e:
                return True
            if self.normalize_dates:
                da, de = self._parse_date(actual.strip()), self._parse_date(expected.strip())
                if da is not None and de is not None and da == de:
                    return True
            if self.normalize_numbers:
                na, ne = self._parse_number(actual.strip()), self._parse_number(expected.strip())
                if na is not None and ne is not None:
                    return abs(na - ne) <= self.numeric_tolerance
            return False

        if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
            return abs(actual - expected) <= self.numeric_tolerance

        # Cross-type: one is a string, the other is a number
        na, ne = self._parse_number(str(actual)), self._parse_number(str(expected))
        if na is not None and ne is not None:
            return abs(na - ne) <= self.numeric_tolerance

        return str(actual).strip() == str(expected).strip()

    @staticmethod
    def _parse_date(text: str):
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _parse_number(text: str) -> float | None:
        try:
            return float(text.replace(",", "").replace(" ", ""))
        except ValueError:
            return None


class SimilarityEvaluator(Evaluator):
    """
    Text similarity evaluator with three methods:

    - ``"token"`` (default): Token F1 score (ROUGE-1 style). Measures unigram overlap
      between actual and expected. Robust to word order and paraphrasing; good for
      free-text tasks. No external dependencies.
    - ``"char"``: Character-level difflib ratio. Sensitive to exact character matches;
      only useful when character-level fidelity matters.
    - ``"embedding"``: Cosine similarity between vector embeddings. Best for semantic
      equivalence. Requires passing ``embed_fn``.

    Args:
        pass_threshold: Minimum score to count as passed (default 0.75).
        method: One of ``"token"``, ``"char"``, ``"embedding"``.
        embed_fn: Required when ``method="embedding"``. A callable that takes a string
                  and returns a list of floats (the embedding vector).
    """

    def __init__(
        self,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        method: str = "token",
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        if method not in ("token", "char", "embedding"):
            raise ValueError(f"method must be 'token', 'char', or 'embedding', got {method!r}")
        if method == "embedding" and embed_fn is None:
            raise ValueError("embed_fn is required when method='embedding'")
        self.pass_threshold = pass_threshold
        self.method = method
        self.embed_fn = embed_fn

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        if self.method == "token":
            score = self._token_f1(actual.strip(), expected.strip())
        elif self.method == "char":
            score = difflib.SequenceMatcher(None, actual.strip(), expected.strip()).ratio()
        else:  # embedding
            vec_a = self.embed_fn(actual.strip())
            vec_b = self.embed_fn(expected.strip())
            score = self._cosine(vec_a, vec_b)

        return EvalResult(
            score=score,
            passed=score >= self.pass_threshold,
            feedback=(
                f"Similarity ({self.method}): {score:.1%}"
                + ("" if score >= self.pass_threshold else f" (threshold: {self.pass_threshold:.1%})")
            ),
        )

    @staticmethod
    def _token_f1(actual: str, expected: str) -> float:
        """Unigram F1 (ROUGE-1): harmonic mean of token precision and recall."""
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\b\w+\b", text.lower())

        a_tokens = tokenize(actual)
        e_tokens = tokenize(expected)

        if not a_tokens and not e_tokens:
            return 1.0
        if not a_tokens or not e_tokens:
            return 0.0

        a_counts = Counter(a_tokens)
        e_counts = Counter(e_tokens)
        overlap = sum((a_counts & e_counts).values())

        precision = overlap / len(a_tokens)
        recall = overlap / len(e_tokens)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(x * x for x in vec_a))
        norm_b = math.sqrt(sum(x * x for x in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class LLMJudgeEvaluator(Evaluator):
    """
    Uses an LLM to judge output quality by comparing actual vs expected.

    The LLM is prompted to score accuracy and explain discrepancies.

    The judge call runs at temperature 0 by default — its score decides which
    prompt versions are accepted, so it must be as deterministic as the model
    allows. Consider passing a *different* LLM than the one used for inference:
    a model judging its own outputs is subject to self-preference bias.
    """

    def __init__(
        self,
        llm: LLMClient,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        task_description: str = "",
        temperature: float | None = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            llm: LLM client used for judging. Can (and often should) differ
                 from the inference LLM to avoid self-preference bias.
            pass_threshold: Scores at or above this count as passed.
            task_description: Optional task context injected into the judge prompt.
            temperature: Sampling temperature for the judge call. Defaults to
                         0.0 for deterministic scoring; None omits the kwarg.
            max_retries: Retries per failed judge LLM call (exponential backoff).
            retry_delay: Initial retry wait in seconds (doubles each attempt).
        """
        self.llm = llm
        self.pass_threshold = pass_threshold
        self.task_description = task_description
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        judge_prompt = f"""You are evaluating the quality of an AI-generated output against the expected correct output.

{f"Task context: {self.task_description}" if self.task_description else ""}

<expected_output>
{expected}
</expected_output>

<actual_output>
{actual}
</actual_output>

Evaluate how well the actual output matches the expected output.
Consider: accuracy of data, completeness, format correctness, and any errors.

Respond in this exact JSON format (no other text):
{{
    "score": <float between 0.0 and 1.0>,
    "correct_aspects": ["list of things done correctly"],
    "errors": ["list of specific errors or missing items"],
    "feedback": "brief overall assessment"
}}"""

        response = self._call_judge(judge_prompt)
        parsed = self._parse_judge_response(response.text)

        if parsed is None:
            # Re-ask once with a stricter instruction — a format slip must not
            # silently score the example as 0.
            retry_prompt = (
                judge_prompt
                + "\n\nIMPORTANT: your previous reply could not be parsed. "
                "Respond with ONLY the JSON object — no prose, no code fences."
            )
            response = self._call_judge(retry_prompt)
            parsed = self._parse_judge_response(response.text)

        if parsed is None:
            return EvalResult(
                score=0.0,
                passed=False,
                feedback=f"Judge response could not be parsed: {response.text[:200]}",
                details={"raw_response": response.text},
            )

        score, result = parsed
        return EvalResult(
            score=score,
            passed=score >= self.pass_threshold,
            details=result,
            feedback=result.get("feedback", ""),
        )

    def _call_judge(self, prompt: str):
        """One judge LLM call with retry/backoff, at the configured temperature."""
        llm_kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        return call_with_retry(
            lambda: self.llm.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                **llm_kwargs,
            ),
            max_retries=self.max_retries,
            delay=self.retry_delay,
        )

    @staticmethod
    def _parse_judge_response(text: str) -> tuple[float, dict] | None:
        """Extract (score, details) from the judge reply, or None if unparsable."""
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            result: dict = json.loads(text)
            score = float(result.get("score", 0.0))
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError, KeyError):
            return None
        return score, result
