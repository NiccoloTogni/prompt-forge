"""
Tests for the training-loop semantics:
    - Held-out test set evaluated exactly once, after the loop
    - Patience counts iterations without STRICT val improvement (plateau stops)
    - Baseline val eval is computed once and reused across iterations
"""

from pathlib import Path
from unittest.mock import MagicMock

from prompt_forge.bundle import ExampleBundle
from prompt_forge.evaluation.evaluator import (
    BatchEvalResult,
    EvalResult,
    Evaluator,
    ExactMatchEvaluator,
)
from prompt_forge.file_loaders import get_default_loader
from prompt_forge.llm.client import LLMResponse
from prompt_forge.storage.project_store import FileSystemStore, PromptVersion
from prompt_forge.training.pipeline import TrainingConfig, TrainingPipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_store(tmp_path: Path) -> FileSystemStore:
    store = FileSystemStore(tmp_path / "store")
    store.save_prompt_version(PromptVersion(
        version=1,
        prompt_text="seed",
        created_at="2024-01-01T00:00:00+00:00",
    ))
    return store


def _make_bundle(tmp_path: Path, name: str, input_text: str, output_text: str) -> ExampleBundle:
    d = tmp_path / name
    d.mkdir()
    (d / f"{name}_input.txt").write_text(input_text)
    (d / f"{name}_expected_output.txt").write_text(output_text)
    return ExampleBundle(bundle_id=name, files={
        "input": d / f"{name}_input.txt",
        "expected_output": d / f"{name}_expected_output.txt",
    })


def _make_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMResponse(
        text="<optimized_prompt>better</optimized_prompt><learnings>ok</learnings><issues></issues>",
        usage={"input_tokens": 1, "output_tokens": 1},
    )
    return llm


class ScriptedEvaluator(Evaluator):
    """Returns pre-scripted batch mean scores, one per evaluate_batch call.

    Repeats the last score once the script is exhausted. Counts calls so tests
    can assert how many evaluations actually ran.
    """

    def __init__(self, scores: list[float]):
        self.scores = list(scores)
        self.calls = 0
        self._last = scores[-1]

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        return EvalResult(score=self._last, passed=self._last >= 0.75)

    def evaluate_batch(self, results):
        self.calls += 1
        score = self.scores.pop(0) if self.scores else self._last
        self._last = score
        individual = [EvalResult(score=score, passed=score >= 0.75) for _ in results]
        return BatchEvalResult(
            mean_score=score,
            pass_rate=1.0 if score >= 0.75 else 0.0,
            individual_results=individual,
            example_ids=[r[0] for r in results],
            failed_examples=[],
        )


def _run(tmp_path, evaluator, *, with_test=False, **config_kwargs):
    store = _make_store(tmp_path)
    train_b = _make_bundle(tmp_path, "t", "train input", "train out")
    val_b = _make_bundle(tmp_path, "v", "val input", "val out")
    test_b = _make_bundle(tmp_path, "x", "test input", "test out") if with_test else None

    pipeline = TrainingPipeline(
        llm=_make_llm(),
        store=store,
        evaluator=evaluator,
        file_loader=get_default_loader(),
    )
    config_kwargs.setdefault("max_iterations", 3)
    return pipeline.train(
        [train_b],
        val_bundles=[val_b],
        test_bundles=[test_b] if test_b else None,
        config=TrainingConfig(native_files=False, **config_kwargs),
    )


# ── Held-out test set ─────────────────────────────────────────────────────────

def test_test_score_populated_when_test_bundles_given(tmp_path):
    report = _run(tmp_path, ExactMatchEvaluator(), with_test=True)
    assert report.test_score is not None
    assert 0.0 <= report.test_score <= 1.0
    assert report.test_example_scores is not None
    assert "x" in report.test_example_scores


def test_test_score_none_without_test_bundles(tmp_path):
    report = _run(tmp_path, ExactMatchEvaluator())
    assert report.test_score is None
    assert report.test_example_scores is None


def test_test_set_evaluated_exactly_once(tmp_path):
    evaluator = ScriptedEvaluator([0.5])
    report = _run(tmp_path, evaluator, with_test=True, max_iterations=3, patience=10)
    # 1 baseline + 3 candidates + 1 final test eval — the test set never
    # enters the per-iteration loop
    assert evaluator.calls == 5
    assert len(report.iterations) == 3


def test_test_skipped_without_evaluator(tmp_path):
    store = _make_store(tmp_path)
    train_b = _make_bundle(tmp_path, "t", "in", "out")
    test_b = _make_bundle(tmp_path, "x", "test in", "test out")
    pipeline = TrainingPipeline(
        llm=_make_llm(),
        store=store,
        evaluator=None,
        file_loader=get_default_loader(),
    )
    report = pipeline.train(
        [train_b],
        test_bundles=[test_b],
        config=TrainingConfig(max_iterations=1, native_files=False),
    )
    assert report.test_score is None


def test_refinement_recommended_uses_test_score(tmp_path):
    # Val and test always score 1.0 → above the 0.8 threshold → no refinement
    report = _run(tmp_path, ScriptedEvaluator([1.0]), with_test=True, patience=10)
    assert report.test_score == 1.0
    assert report.refinement_recommended is False

    # Constant 0.0 → below threshold → refinement recommended


def test_refinement_recommended_when_test_score_low(tmp_path):
    report = _run(tmp_path, ScriptedEvaluator([0.0]), with_test=True, patience=10)
    assert report.test_score == 0.0
    assert report.refinement_recommended is True


# ── Plateau / patience semantics ──────────────────────────────────────────────

def test_flat_plateau_triggers_early_stop(tmp_path):
    # Constant score: ties are accepted (min_improvement=0) but must not reset
    # patience — training stops after `patience` iterations, not max_iterations.
    evaluator = ScriptedEvaluator([0.5])
    report = _run(tmp_path, evaluator, max_iterations=10, patience=2)
    assert len(report.iterations) == 2
    # Tie-acceptance semantics preserved: the candidates were still accepted
    assert all(r.improved for r in report.iterations)


def test_strict_improvement_resets_patience(tmp_path):
    # baseline 0.1 → candidate 0.2 (strict gain, resets) → 0.2, 0.2 (plateau)
    evaluator = ScriptedEvaluator([0.1, 0.2, 0.2, 0.2])
    report = _run(tmp_path, evaluator, max_iterations=10, patience=2)
    assert len(report.iterations) == 3


def test_rejections_still_count_toward_patience(tmp_path):
    # baseline 0.5, every candidate scores 0.3 → rejected each time
    evaluator = ScriptedEvaluator([0.5, 0.3, 0.3])
    report = _run(tmp_path, evaluator, max_iterations=10, patience=2)
    assert len(report.iterations) == 2
    assert not any(r.improved for r in report.iterations)


# ── Baseline eval reuse ───────────────────────────────────────────────────────

def test_baseline_evaluated_once_and_reused(tmp_path):
    evaluator = ScriptedEvaluator([0.5, 0.3, 0.3])
    report = _run(tmp_path, evaluator, max_iterations=10, patience=2)
    # 1 baseline + 1 candidate per iteration — the baseline is never re-scored
    assert evaluator.calls == 1 + len(report.iterations)
    # The reused baseline score is reported unchanged in later iterations
    assert report.iterations[0].score_before == 0.5
    assert report.iterations[1].score_before == 0.5


def test_accepted_candidate_becomes_new_baseline(tmp_path):
    # iter1: 0.1 → 0.4 accepted; iter2's score_before must be 0.4 without re-eval
    evaluator = ScriptedEvaluator([0.1, 0.4, 0.4, 0.4])
    report = _run(tmp_path, evaluator, max_iterations=3, patience=10)
    assert report.iterations[1].score_before == 0.4
    # 1 baseline + 3 candidates (no per-iteration baseline re-eval)
    assert evaluator.calls == 4


def test_final_score_is_kept_prompt_score_not_last_candidate(tmp_path):
    # All candidates rejected: final_score must reflect the retained baseline
    # prompt (0.5), not the last rejected candidate (0.3).
    evaluator = ScriptedEvaluator([0.5, 0.3, 0.3])
    report = _run(tmp_path, evaluator, max_iterations=10, patience=2)
    assert report.final_score == 0.5
