"""
Tests for TrainingConfig.eval_train — optional train-batch scoring.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prompt_forge.bundle import ExampleBundle
from prompt_forge.evaluation.evaluator import ExactMatchEvaluator
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


def _run_pipeline(tmp_path, eval_train: bool):
    store = _make_store(tmp_path)
    train_b = _make_bundle(tmp_path, "t", "train input", "train out")
    val_b = _make_bundle(tmp_path, "v", "val input", "val out")

    pipeline = TrainingPipeline(
        llm=_make_llm(),
        store=store,
        evaluator=ExactMatchEvaluator(),
        file_loader=get_default_loader(),
    )
    return pipeline.train(
        [train_b],
        val_bundles=[val_b],
        config=TrainingConfig(
            max_iterations=1,
            native_files=False,
            eval_train=eval_train,
        ),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_train_score_none_by_default(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=False)
    assert all(r.train_score is None for r in report.iterations)


def test_train_score_populated_when_flag_set(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=True)
    improved_iters = [r for r in report.iterations if r.improved]
    assert improved_iters, "Expected at least one accepted iteration"
    assert all(r.train_score is not None for r in improved_iters)


def test_train_score_is_float_in_range(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=True)
    for r in report.iterations:
        if r.train_score is not None:
            assert 0.0 <= r.train_score <= 1.0


# ── Per-example scores ────────────────────────────────────────────────────────

def test_val_example_scores_populated(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=False)
    for r in report.iterations:
        if r.improved:
            assert r.val_example_scores is not None
            assert "v" in r.val_example_scores
            assert 0.0 <= r.val_example_scores["v"] <= 1.0


def test_val_example_scores_none_when_no_evaluator(tmp_path):
    store = _make_store(tmp_path)
    train_b = _make_bundle(tmp_path, "t", "train input", "train out")

    pipeline = TrainingPipeline(
        llm=_make_llm(),
        store=store,
        evaluator=None,
        file_loader=get_default_loader(),
    )
    report = pipeline.train(
        [train_b],
        config=TrainingConfig(max_iterations=1, native_files=False),
    )
    assert all(r.val_example_scores is None for r in report.iterations)


def test_train_example_scores_populated_when_flag_set(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=True)
    for r in report.iterations:
        if r.improved:
            assert r.train_example_scores is not None
            assert "t" in r.train_example_scores


def test_train_example_scores_none_when_flag_off(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=False)
    assert all(r.train_example_scores is None for r in report.iterations)


# ── aggregate_issues ──────────────────────────────────────────────────────────

def test_aggregate_issues_empty_when_no_issues(tmp_path):
    report = _run_pipeline(tmp_path, eval_train=False)
    # Default mock LLM returns no issues tag — all_issues will be empty
    llm = _make_llm()
    assert report.aggregate_issues(llm) == ""


def test_aggregate_issues_calls_llm_with_issues(tmp_path):
    from prompt_forge.training.pipeline import IterationResult, TrainingReport

    llm = MagicMock()
    llm.complete.return_value = LLMResponse(text="• Missing edge case X\n• Gap in Y", usage={})

    report = TrainingReport(
        iterations=[
            IterationResult(
                iteration=1, prompt_version=1, score_before=None, score_after=0.5,
                improved=True, learnings="", issues="Cannot handle edge case X",
                batch_ids=["a"],
            ),
            IterationResult(
                iteration=2, prompt_version=2, score_before=0.5, score_after=0.6,
                improved=True, learnings="", issues="Still missing coverage for Y",
                batch_ids=["b"],
            ),
        ],
        final_version=2,
        final_score=0.6,
        refinement_recommended=True,
    )

    result = report.aggregate_issues(llm)
    assert "Missing edge case X" in result or "Gap in Y" in result
    llm.complete.assert_called_once()
    # Both iteration issues should appear in the prompt sent to the LLM
    user_content = llm.complete.call_args[0][0][1].content
    assert "Cannot handle edge case X" in user_content
    assert "Still missing coverage for Y" in user_content
