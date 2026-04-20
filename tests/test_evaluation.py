"""
Tests for prompt_forge.evaluation.evaluator
"""

import json
import pytest
from unittest.mock import MagicMock

from prompt_forge.evaluation.evaluator import (
    EvalResult,
    BatchEvalResult,
    ExactMatchEvaluator,
    JsonFieldEvaluator,
    SimilarityEvaluator,
    LLMJudgeEvaluator,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_llm(response_text: str):
    """Return a mock LLMClient that always responds with response_text."""
    llm = MagicMock()
    llm.complete.return_value = MagicMock(text=response_text, usage=None)
    return llm


# ── EvalResult ────────────────────────────────────────────────────────────────

def test_eval_result_to_dict():
    r = EvalResult(score=0.9, passed=True, feedback="ok")
    d = r.to_dict()
    assert d["score"] == 0.9
    assert d["passed"] is True
    assert d["feedback"] == "ok"


# ── evaluate_batch (base) ─────────────────────────────────────────────────────

def test_evaluate_batch_mean_score():
    ev = ExactMatchEvaluator()
    results = ev.evaluate_batch([
        ("b1", "hello", "hello"),
        ("b2", "hello", "world"),
    ])
    assert isinstance(results, BatchEvalResult)
    assert results.mean_score == pytest.approx(0.5)
    assert results.pass_rate == pytest.approx(0.5)


def test_evaluate_batch_all_pass():
    ev = ExactMatchEvaluator()
    results = ev.evaluate_batch([("b1", "x", "x"), ("b2", "y", "y")])
    assert results.pass_rate == 1.0
    assert results.failed_examples == []


def test_evaluate_batch_failed_examples_structure():
    ev = ExactMatchEvaluator()
    results = ev.evaluate_batch([("b1", "a", "b")])
    assert len(results.failed_examples) == 1
    assert results.failed_examples[0]["bundle_id"] == "b1"
    assert "feedback" in results.failed_examples[0]
    assert "score" in results.failed_examples[0]


def test_evaluate_batch_empty():
    ev = ExactMatchEvaluator()
    results = ev.evaluate_batch([])
    assert results.mean_score == 0.0
    assert results.pass_rate == 0.0
    assert results.individual_results == []


def test_batch_eval_result_to_dict():
    ev = ExactMatchEvaluator()
    r = ev.evaluate_batch([("b1", "x", "x"), ("b2", "a", "b")])
    d = r.to_dict()
    assert d["num_examples"] == 2
    assert d["num_passed"] == 1
    assert d["mean_score"] == pytest.approx(0.5)


# ── ExactMatchEvaluator ───────────────────────────────────────────────────────

class TestExactMatch:
    def test_exact_match(self):
        ev = ExactMatchEvaluator()
        r = ev.evaluate("hello", "hello")
        assert r.passed is True
        assert r.score == 1.0

    def test_mismatch(self):
        ev = ExactMatchEvaluator()
        r = ev.evaluate("hello", "world")
        assert r.passed is False
        assert r.score == 0.0

    def test_case_insensitive_by_default(self):
        ev = ExactMatchEvaluator()
        assert ev.evaluate("Hello", "hello").passed is True

    def test_case_sensitive(self):
        ev = ExactMatchEvaluator(case_sensitive=True)
        assert ev.evaluate("Hello", "hello").passed is False
        assert ev.evaluate("hello", "hello").passed is True

    def test_whitespace_normalization(self):
        ev = ExactMatchEvaluator()
        assert ev.evaluate("hello   world", "hello world").passed is True

    def test_whitespace_normalization_disabled(self):
        ev = ExactMatchEvaluator(normalize_whitespace=False)
        assert ev.evaluate("hello   world", "hello world").passed is False

    def test_feedback_on_mismatch(self):
        ev = ExactMatchEvaluator()
        r = ev.evaluate("abc", "abd")
        assert r.feedback != ""

    def test_empty_strings_match(self):
        ev = ExactMatchEvaluator()
        assert ev.evaluate("", "").passed is True

    def test_strips_leading_trailing_whitespace(self):
        ev = ExactMatchEvaluator()
        assert ev.evaluate("  hello  ", "hello").passed is True


# ── JsonFieldEvaluator ────────────────────────────────────────────────────────

class TestJsonFieldEvaluator:
    def test_all_fields_match(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"a": "1", "b": "2"})
        expected = json.dumps({"a": "1", "b": "2"})
        r = ev.evaluate(actual, expected)
        assert r.passed is True
        assert r.score == 1.0

    def test_partial_match(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"a": "1", "b": "wrong"})
        expected = json.dumps({"a": "1", "b": "2"})
        r = ev.evaluate(actual, expected)
        assert r.score == pytest.approx(0.5)
        assert r.passed is False  # below default 0.75 threshold

    def test_custom_pass_threshold(self):
        ev = JsonFieldEvaluator(pass_threshold=0.4)
        actual = json.dumps({"a": "1", "b": "wrong"})
        expected = json.dumps({"a": "1", "b": "2"})
        r = ev.evaluate(actual, expected)
        assert r.passed is True  # 0.5 >= 0.4

    def test_missing_field(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"a": "1"})
        expected = json.dumps({"a": "1", "b": "2"})
        r = ev.evaluate(actual, expected)
        assert r.score < 1.0

    def test_invalid_actual_json(self):
        ev = JsonFieldEvaluator()
        r = ev.evaluate("not json", json.dumps({"a": "1"}))
        assert r.passed is False
        assert r.score == 0.0
        assert "not valid JSON" in r.feedback

    def test_invalid_expected_json(self):
        ev = JsonFieldEvaluator()
        r = ev.evaluate(json.dumps({"a": "1"}), "not json")
        assert r.passed is False
        assert r.score == 0.0

    def test_numeric_tolerance(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"price": 1.0})
        expected = json.dumps({"price": 1.0})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_case_insensitive_string_by_default(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"name": "ALICE"})
        expected = json.dumps({"name": "alice"})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_case_sensitive(self):
        ev = JsonFieldEvaluator(case_sensitive=True)
        actual = json.dumps({"name": "ALICE"})
        expected = json.dumps({"name": "alice"})
        r = ev.evaluate(actual, expected)
        assert r.score == 0.0

    def test_ignore_fields(self):
        ev = JsonFieldEvaluator(ignore_fields=["timestamp"])
        actual = json.dumps({"name": "alice", "timestamp": "wrong"})
        expected = json.dumps({"name": "alice", "timestamp": "right"})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_nested_fields(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"a": {"b": "1"}})
        expected = json.dumps({"a": {"b": "1"}})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_empty_json_objects(self):
        ev = JsonFieldEvaluator()
        r = ev.evaluate("{}", "{}")
        assert r.passed is True
        assert r.score == 1.0

    def test_feedback_lists_mismatches(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"a": "wrong"})
        expected = json.dumps({"a": "right"})
        r = ev.evaluate(actual, expected)
        assert "a" in r.feedback

    def test_accepts_dict_input(self):
        ev = JsonFieldEvaluator()
        r = ev.evaluate({"a": "1"}, {"a": "1"})
        assert r.score == 1.0

    # ── Fuzzy matching ────────────────────────────────────────────────────────

    def test_date_normalization_iso_vs_slash(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"date": "2024-01-15"})
        expected = json.dumps({"date": "15/01/2024"})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_date_normalization_iso_vs_us_slash(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"date": "2024-03-25"})
        expected = json.dumps({"date": "03/25/2024"})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_date_normalization_disabled(self):
        ev = JsonFieldEvaluator(normalize_dates=False)
        actual = json.dumps({"date": "2024-01-15"})
        expected = json.dumps({"date": "15/01/2024"})
        r = ev.evaluate(actual, expected)
        assert r.score == 0.0

    def test_different_dates_do_not_match(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"date": "2024-01-15"})
        expected = json.dumps({"date": "2024-01-16"})
        r = ev.evaluate(actual, expected)
        assert r.score == 0.0

    def test_number_string_normalization_thousands_sep(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"price": "1234.56"})
        expected = json.dumps({"price": "1,234.56"})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_number_string_normalization_disabled(self):
        ev = JsonFieldEvaluator(normalize_numbers=False)
        actual = json.dumps({"price": "1234.56"})
        expected = json.dumps({"price": "1,234.56"})
        r = ev.evaluate(actual, expected)
        assert r.score == 0.0

    def test_numeric_tolerance_within(self):
        ev = JsonFieldEvaluator(numeric_tolerance=0.01)
        actual = json.dumps({"price": 99.995})
        expected = json.dumps({"price": 100.0})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0

    def test_numeric_tolerance_exceeded(self):
        ev = JsonFieldEvaluator(numeric_tolerance=0.001)
        actual = json.dumps({"price": 99.995})
        expected = json.dumps({"price": 100.0})
        r = ev.evaluate(actual, expected)
        assert r.score == 0.0

    def test_cross_type_number_string_vs_float(self):
        ev = JsonFieldEvaluator()
        actual = json.dumps({"amount": "1234"})
        expected = json.dumps({"amount": 1234})
        r = ev.evaluate(actual, expected)
        assert r.score == 1.0


# ── SimilarityEvaluator ───────────────────────────────────────────────────────

class TestSimilarityEvaluator:
    def test_identical_strings(self):
        ev = SimilarityEvaluator()
        r = ev.evaluate("hello world", "hello world")
        assert r.score == pytest.approx(1.0)
        assert r.passed is True

    def test_completely_different(self):
        ev = SimilarityEvaluator()
        r = ev.evaluate("aaaa", "bbbb")
        assert r.score == pytest.approx(0.0)
        assert r.passed is False

    def test_partial_similarity(self):
        ev = SimilarityEvaluator()
        r = ev.evaluate("hello world", "hello earth")
        assert 0.0 < r.score < 1.0

    def test_custom_threshold(self):
        ev = SimilarityEvaluator(pass_threshold=0.5)
        r = ev.evaluate("hello world", "hello earth")
        # similarity should be above 0.5 for this pair
        assert r.passed is (r.score >= 0.5)

    def test_empty_strings(self):
        ev = SimilarityEvaluator()
        r = ev.evaluate("", "")
        assert r.score == pytest.approx(1.0)

    def test_feedback_includes_percentage(self):
        ev = SimilarityEvaluator()
        r = ev.evaluate("abc", "xyz")
        assert "%" in r.feedback


# ── LLMJudgeEvaluator ─────────────────────────────────────────────────────────

class TestLLMJudgeEvaluator:
    def _judge_response(self, score=0.9, feedback="Looks good"):
        return json.dumps({
            "score": score,
            "correct_aspects": ["field A"],
            "errors": [],
            "feedback": feedback,
        })

    def test_parses_valid_response(self):
        llm = make_llm(self._judge_response(score=0.9))
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("actual output", "expected output")
        assert r.score == pytest.approx(0.9)
        assert r.passed is True
        assert r.feedback == "Looks good"

    def test_below_threshold_fails(self):
        llm = make_llm(self._judge_response(score=0.5))
        ev = LLMJudgeEvaluator(llm=llm, pass_threshold=0.75)
        r = ev.evaluate("actual", "expected")
        assert r.passed is False

    def test_custom_threshold(self):
        llm = make_llm(self._judge_response(score=0.6))
        ev = LLMJudgeEvaluator(llm=llm, pass_threshold=0.5)
        r = ev.evaluate("actual", "expected")
        assert r.passed is True

    def test_handles_code_fence(self):
        response = "```json\n" + self._judge_response(score=0.8) + "\n```"
        llm = make_llm(response)
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("a", "b")
        assert r.score == pytest.approx(0.8)

    def test_handles_code_fence_no_language(self):
        response = "```\n" + self._judge_response(score=0.7) + "\n```"
        llm = make_llm(response)
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("a", "b")
        assert r.score == pytest.approx(0.7)

    def test_handles_trailing_whitespace_in_fence(self):
        response = "```json\n" + self._judge_response(score=0.75) + "\n```  "
        llm = make_llm(response)
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("a", "b")
        assert r.score == pytest.approx(0.75)

    def test_invalid_json_response_returns_zero(self):
        llm = make_llm("I cannot evaluate this.")
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("a", "b")
        assert r.score == 0.0
        assert r.passed is False
        assert "could not be parsed" in r.feedback

    def test_task_description_included_in_prompt(self):
        llm = make_llm(self._judge_response())
        ev = LLMJudgeEvaluator(llm=llm, task_description="Extract invoice data")
        ev.evaluate("actual", "expected")
        call_args = llm.complete.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        prompt_text = messages[0].content  # first message content
        assert "Extract invoice data" in prompt_text

    def test_details_stored_in_result(self):
        llm = make_llm(self._judge_response(score=0.9))
        ev = LLMJudgeEvaluator(llm=llm)
        r = ev.evaluate("a", "b")
        assert "score" in r.details
        assert "correct_aspects" in r.details
