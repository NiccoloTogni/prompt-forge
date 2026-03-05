"""
Evaluation strategies for comparing actual vs expected output.

Built-in evaluators:
    - ExactMatchEvaluator: Exact string match (after normalization)
    - JsonFieldEvaluator: Field-by-field JSON comparison
    - SimilarityEvaluator: Text similarity (difflib)
    - LLMJudgeEvaluator: Uses an LLM to score output quality

Users can implement custom evaluators by subclassing Evaluator.
"""

import dataclasses
import difflib
import json
from abc import ABC, abstractmethod
from typing import Any

from ..llm.client import LLMClient, LLMMessage


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
    failed_examples: list[dict]  # Bundle IDs + feedback for failures

    def to_dict(self) -> dict:
        return {
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "num_examples": len(self.individual_results),
            "num_passed": sum(1 for r in self.individual_results if r.passed),
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
        failed = []
        for bundle_id, actual, expected in results:
            result = self.evaluate(actual, expected)
            individual.append(result)
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


class JsonFieldEvaluator(Evaluator):
    """
    Field-by-field JSON comparison.

    Ideal for data extraction tasks. Reports which fields match,
    which are wrong, and which are missing.
    """

    def __init__(
        self,
        pass_threshold: float = 0.8,
        ignore_fields: list[str] | None = None,
        case_sensitive: bool = False,
    ):
        self.pass_threshold = pass_threshold
        self.ignore_fields = set(ignore_fields or [])
        self.case_sensitive = case_sensitive

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
        # String comparison with optional case insensitivity
        if isinstance(actual, str) and isinstance(expected, str):
            a = actual.strip()
            e = expected.strip()
            if not self.case_sensitive:
                a = a.lower()
                e = e.lower()
            return a == e
        # Numeric comparison with tolerance
        try:
            return abs(float(actual) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return str(actual).strip() == str(expected).strip()


class SimilarityEvaluator(Evaluator):
    """
    Text similarity using difflib SequenceMatcher.

    Good for free-text outputs where exact match is too strict.
    """

    def __init__(self, pass_threshold: float = 0.85):
        self.pass_threshold = pass_threshold

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        ratio = difflib.SequenceMatcher(None, actual.strip(), expected.strip()).ratio()
        return EvalResult(
            score=ratio,
            passed=ratio >= self.pass_threshold,
            feedback=f"Similarity: {ratio:.1%}" + ("" if ratio >= self.pass_threshold else f" (threshold: {self.pass_threshold:.1%})"),
        )


class LLMJudgeEvaluator(Evaluator):
    """
    Uses an LLM to judge output quality by comparing actual vs expected.

    The LLM is prompted to score accuracy and explain discrepancies.
    """

    def __init__(
        self,
        llm: LLMClient,
        pass_threshold: float = 0.8,
        task_description: str = "",
    ):
        self.llm = llm
        self.pass_threshold = pass_threshold
        self.task_description = task_description

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

        response = self.llm.complete(
            messages=[LLMMessage(role="user", content=judge_prompt)],
            temperature=0.0,
        )

        try:
            # Strip markdown code fences if present
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = json.loads(text.strip())
            score = float(result.get("score", 0.0))
            return EvalResult(
                score=score,
                passed=score >= self.pass_threshold,
                details=result,
                feedback=result.get("feedback", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: try to extract a score from the text
            return EvalResult(
                score=0.0,
                passed=False,
                feedback=f"Judge response could not be parsed: {response.text[:200]}",
                details={"raw_response": response.text},
            )
