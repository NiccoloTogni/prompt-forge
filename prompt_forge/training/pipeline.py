"""
Training Pipeline — orchestrates the full incremental training loop.

The pipeline coordinates:
    1. Batch selection
    2. Prompt optimization
    3. Evaluation
    4. Versioning
    5. Logging

It supports resumption — if training is interrupted, it picks up where
it left off using the stored training state.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from ..llm.client import LLMClient, LLMMessage
from ..bundle import BundleCollection, ExampleBundle
from ..file_loaders import FileLoader, get_default_loader
from ..storage.project_store import ProjectStore, PromptVersion
from ..evaluation.evaluator import Evaluator, BatchEvalResult
from ..inference.agent import InferenceAgent
from .optimizer import PromptOptimizer
from .batch_strategy import BatchStrategy, RandomBatchStrategy, FailurePriorityBatchStrategy
from .training_log import TrainingLog, LogEntry

logger = logging.getLogger(__name__)

# ── Module-level defaults ─────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MIN_IMPROVEMENT = 0.0        # Accept any improvement; ignored when evaluator is None
DEFAULT_PATIENCE = 5                 # Early-stop after N iters without strict val improvement; ignored when evaluator is None
DEFAULT_REFINEMENT_THRESHOLD = 0.8   # Scores below this flag refinement_recommended=True
DEFAULT_INFERENCE_TEMPERATURE = None  # None = use model default; set to 0.0 for deterministic eval
MAX_SUMMARY_ENTRIES = 15             # Number of recent iterations to include in optimizer context


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for a training run."""
    batch_size: int = DEFAULT_BATCH_SIZE
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT  # Ignored when evaluator is None
    patience: int = DEFAULT_PATIENCE                  # Ignored when evaluator is None
    val_max_tokens: int | None = None     # Max tokens per batch eval call (None = no limit)
    auto_save: bool = True
    output_schema: dict | None = None     # JSON schema if task requires structured output
    refinement_threshold: float = DEFAULT_REFINEMENT_THRESHOLD
    max_tokens: int | None = None         # Per-call context window limit for optimizer
    max_total_tokens: int | None = None   # Total token budget for the whole run
    inference_temperature: float | None = DEFAULT_INFERENCE_TEMPERATURE  # None = use model default
    seed: int | None = None               # Random seed for reproducible batch selection
    native_files: bool = True             # Pass input files natively to the LLM (requires multimodal client)
    max_retries: int = 3                  # Retries per failed LLM call (optimizer + eval agent)
    retry_delay: float = 1.0             # Initial retry wait in seconds (doubles each attempt)
    max_workers: int | None = None       # Concurrent LLM calls for per-input batch fallback (None = serial)
    context_retriever: "Callable[[str, Any], str] | None" = None  # Retriever used by the eval agent — must match production
    eval_train: bool = False             # Also evaluate the new prompt on the training batch (costs extra tokens)


@dataclasses.dataclass
class IterationResult:
    """Result of a single training iteration."""
    iteration: int
    prompt_version: int
    score_before: float | None
    score_after: float | None
    improved: bool
    learnings: str
    issues: str       # Outstanding gaps/contradictions flagged by the optimizer
    batch_ids: list[str]
    train_score: float | None = None                    # Score on training batch (only when TrainingConfig.eval_train=True)
    val_example_scores: dict[str, float] | None = None  # Per-example val scores: bundle_id → score
    train_example_scores: dict[str, float] | None = None  # Per-example train scores (eval_train=True only)
    tokens_used: int | None = None                      # Total tokens consumed in this iteration (optimizer + eval)


@dataclasses.dataclass
class TrainingReport:
    """
    Summary of a completed training run.

    Returned by TrainingPipeline.train() and Project.train().
    Iterable for backward compatibility: ``for r in report`` yields IterationResult.
    """
    iterations: list[IterationResult]
    final_version: int
    final_score: float | None
    refinement_recommended: bool  # True if score is below refinement_threshold or unknown
    total_tokens_used: int = 0   # Cumulative input + output tokens across the entire run
    test_score: float | None = None                     # One-shot score on held-out test_bundles (None if not provided)
    test_example_scores: dict[str, float] | None = None  # Per-example test scores: bundle_id → score

    @property
    def all_issues(self) -> list[tuple[int, str]]:
        """All non-empty issues flagged by the optimizer, as (iteration, issues) pairs."""
        return [
            (r.iteration, r.issues)
            for r in self.iterations
            if r.issues
        ]

    def aggregate_issues(self, llm: "Any") -> str:
        """
        Summarise recurring issues across all training iterations using one LLM call.

        Identifies distinct root causes and ranks them by how often they appeared,
        giving a direct signal of what the training data is missing coverage for.
        Returns an empty string if no issues were recorded.

        Args:
            llm: Any LLMClient — the same client used for training is fine.

        Returns:
            A concise bullet-point summary of recurring root causes, or ``""``
            if no issues were flagged during training.
        """
        issues = self.all_issues
        if not issues:
            return ""

        issues_text = "\n\n".join(
            f"[Iteration {iteration}]\n{text}" for iteration, text in issues
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a training analyst. The user will give you a list of issues "
                    "flagged by a prompt optimizer across multiple training iterations. "
                    "Identify the distinct recurring root causes and present them as a "
                    "concise bullet-point list ranked by frequency (most recurring first). "
                    "Merge semantically identical issues. Be specific and actionable — "
                    "each bullet should clearly describe what training data is missing "
                    "or what gap remains unresolved."
                ),
            ),
            LLMMessage(role="user", content=issues_text),
        ]
        from .._retry import call_with_retry
        response = call_with_retry(lambda: llm.complete(messages), max_retries=2, delay=1.0)
        return response.text.strip()

    def __iter__(self):
        return iter(self.iterations)

    def __len__(self):
        return len(self.iterations)


class TrainingPipeline:
    """
    Orchestrates the incremental prompt training loop.

    Holds infrastructure only — training and validation data are passed to train().

    Usage:
        pipeline = TrainingPipeline(
            llm=my_llm,
            store=my_store,
            evaluator=my_evaluator,
        )
        train_bundles, val_bundles = train_val_split(all_bundles, val_ratio=0.2, seed=42)
        pipeline.train(train_bundles, val_bundles=val_bundles, config=TrainingConfig(batch_size=5))
    """

    def __init__(
        self,
        llm: LLMClient,
        store: ProjectStore,
        evaluator: Evaluator | None = None,
        optimizer: PromptOptimizer | None = None,
        batch_strategy: BatchStrategy | None = None,
        file_loader: FileLoader | None = None,
        context: str = "",
        inference_fn: Callable[[str, ExampleBundle], str] | None = None,
        on_iteration: Callable[[IterationResult], None] | None = None,
    ):
        """
        Args:
            llm: LLM client for both optimization and inference.
            store: Storage backend for prompt versions and state.
            evaluator: Strategy for scoring outputs.
            optimizer: Custom prompt optimizer (default: PromptOptimizer with defaults).
            batch_strategy: How to select batches. Defaults to
                            FailurePriorityBatchStrategy when an evaluator is set
                            (so rejected batches actually get retried), otherwise
                            RandomBatchStrategy.
            file_loader: File loader for reading example files.
            context: Domain context passed to the optimizer.
            inference_fn: Custom function to run inference with the prompt.
                          Signature: (prompt_text, bundle) -> actual_output_str.
                          If None, a default LLM-based inference is used.
            on_iteration: Optional callback called after each iteration.
        """
        self.llm = llm
        self.store = store
        self.evaluator = evaluator
        self.file_loader = file_loader or get_default_loader()
        self.context = context
        self._custom_inference_fn = inference_fn  # None → use InferenceAgent (batch-capable)
        self.on_iteration = on_iteration

        self.optimizer = optimizer or PromptOptimizer(
            llm=llm,
            file_loader=self.file_loader,
            context=context,
        )
        if batch_strategy is not None:
            self.batch_strategy = batch_strategy
        elif evaluator is not None:
            # Failed batches are only retried if the strategy honours failed_ids
            self.batch_strategy = FailurePriorityBatchStrategy()
        else:
            self.batch_strategy = RandomBatchStrategy()
        self.training_log = TrainingLog()
        self._total_tokens: int = 0  # Running token counter across the whole training run
        self._eval_agent: InferenceAgent | None = None  # built by train(); consolidate() builds a default if needed

        # Restore state if available
        self._restore_state()

    def train(
        self,
        train_bundles: BundleCollection | list[ExampleBundle],
        *,
        val_bundles: BundleCollection | list[ExampleBundle] | None = None,
        test_bundles: BundleCollection | list[ExampleBundle] | None = None,
        config: TrainingConfig | None = None,
    ) -> TrainingReport:
        """
        Run the training loop.

        Args:
            train_bundles: Training examples used for optimization.
            val_bundles: Validation examples used for scoring (optional).
            test_bundles: Held-out examples evaluated exactly once, on the final
                          prompt, after the loop ends. Accept/reject decisions
                          hill-climb on the validation score, so only a set the
                          loop never sees gives an unbiased estimate of
                          generalization (reported as TrainingReport.test_score).
            config: Training configuration.

        Returns:
            TrainingReport with per-iteration results and refinement signal.
        """
        config = config or TrainingConfig()

        _tokens_at_start = self._total_tokens
        results: list[IterationResult] = []

        # Resolve train_bundles to a flat list
        if isinstance(train_bundles, BundleCollection):
            _train_list = train_bundles.bundles
        else:
            _train_list = list(train_bundles)

        if not _train_list:
            raise RuntimeError("train_bundles is empty — provide at least one training example.")

        # Build the eval agent once — reused across all iterations (prompt_text is updated per eval)
        llm_kwargs = {}
        if config.inference_temperature is not None:
            llm_kwargs["temperature"] = config.inference_temperature
        self._eval_agent = InferenceAgent(
            llm=self.llm,
            prompt_text="",  # set per evaluation call
            file_loader=self.file_loader,
            llm_kwargs=llm_kwargs,
            token_estimator=self.optimizer.token_estimator,
            native_files=config.native_files,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            max_workers=config.max_workers,
            context_retriever=config.context_retriever,
        )
        self.optimizer.max_retries = config.max_retries
        self.optimizer.retry_delay = config.retry_delay
        no_improvement_count = 0
        _failed_batch_ids: list[str] = []  # IDs from the last non-improving batch
        # Cached val eval of the current prompt — computed once, then reused
        # until a new prompt is accepted. Halves eval cost vs re-scoring the
        # baseline every iteration, and keeps accept/reject comparisons against
        # a stable baseline instead of a freshly-sampled (noisy) one.
        current_eval_result: BatchEvalResult | None = None

        # Get starting prompt
        current = self.store.get_latest_prompt()
        if current is None:
            raise RuntimeError(
                "No seed prompt found. Set a seed prompt before training: "
                "project.set_seed_prompt('...')"
            )

        current_prompt = current.prompt_text
        current_version = current.version

        # Resolve val_bundles to a flat list
        if isinstance(val_bundles, BundleCollection):
            _val_list = val_bundles.bundles
        elif val_bundles is not None:
            _val_list = list(val_bundles)
        else:
            _val_list = []

        # Resolve test_bundles to a flat list
        if isinstance(test_bundles, BundleCollection):
            _test_list = test_bundles.bundles
        elif test_bundles is not None:
            _test_list = list(test_bundles)
        else:
            _test_list = []

        if self.evaluator is not None and not _val_list:
            logger.warning(
                "An evaluator is set but no val_bundles were provided — "
                "evaluation will be skipped. Pass val_bundles to enable scoring."
            )
        if self.evaluator is None and _test_list:
            logger.warning(
                "test_bundles were provided but no evaluator is set — "
                "the final test evaluation will be skipped."
            )

        logger.info(
            f"Starting training: {len(_train_list)} train examples, "
            f"val_size={len(_val_list)}, "
            f"batch_size={config.batch_size}, max_iterations={config.max_iterations}"
        )

        # Continue numbering from any restored training log, so a resumed run
        # never produces duplicate iteration numbers in the history the
        # optimizer sees. max_iterations counts iterations of THIS run.
        _start_iteration = len(self.training_log.entries)

        for offset in range(1, config.max_iterations + 1):
            iteration = _start_iteration + offset
            logger.info(
                f"\n{'='*60}\nIteration {iteration} "
                f"({offset}/{config.max_iterations} this run)\n{'='*60}"
            )

            # 1. Select batch
            batch = self.batch_strategy.select_batch(
                bundles=_train_list,
                batch_size=config.batch_size,
                used_ids=self.training_log.get_all_used_bundle_ids(),
                failed_ids=_failed_batch_ids,
            )
            batch_ids = [b.bundle_id for b in batch]
            logger.info(f"Selected batch: {batch_ids}")

            # 2. Evaluate current prompt (optional, for scoring).
            #    Reuses the cached result — the current prompt was either scored
            #    at iteration 1 or is the accepted candidate of a previous
            #    iteration, whose eval we already have.
            score_before = None
            eval_feedback = ""
            if self.evaluator is not None and _val_list:
                if current_eval_result is None:
                    current_eval_result = self._evaluate_prompt(
                        current_prompt, _val_list, config.val_max_tokens
                    )
                score_before = current_eval_result.mean_score
                if current_eval_result.failed_examples:
                    feedback_lines = [
                        f"- {f['bundle_id']}: {f['feedback']}"
                        for f in current_eval_result.failed_examples[:10]
                    ]
                    eval_feedback = (
                        f"Current prompt scores {current_eval_result.mean_score:.2f} "
                        f"({current_eval_result.pass_rate:.0%} pass rate).\n"
                        f"Failures:\n" + "\n".join(feedback_lines)
                    )
                logger.info(f"Score before: {score_before:.3f}")

            # 3. Optimize
            tokens_before_iteration = self._total_tokens
            opt_result = self.optimizer.optimize(
                current_prompt=current_prompt,
                examples=batch,
                training_history=self.training_log.get_summary(max_entries=MAX_SUMMARY_ENTRIES),
                eval_feedback=eval_feedback,
                output_schema=config.output_schema,
                max_tokens=config.max_tokens,
            )
            # Accumulate optimizer tokens (inference tokens are added inside _default_inference)
            self._total_tokens += self._count_tokens(opt_result.usage)
            logger.info(f"Optimizer tokens this call: {self._count_tokens(opt_result.usage):,} "
                        f"(running total: {self._total_tokens:,})")

            # 4. Evaluate new prompt
            score_after = None
            improved = False
            made_progress = True  # strict val gain — drives the patience counter
            new_eval_result = None
            if self.evaluator is not None and _val_list:
                new_eval_result = self._evaluate_prompt(opt_result.new_prompt, _val_list, config.val_max_tokens)
                score_after = new_eval_result.mean_score
                improved = (
                    score_before is None
                    or score_after >= score_before + config.min_improvement
                )
                made_progress = improved and (
                    score_before is None or score_after > score_before
                )
                logger.info(f"Score after: {score_after:.3f} ({'improved' if improved else 'not improved'})")
            else:
                # No evaluator — always accept and run all iterations
                improved = True

            # 4b. Evaluate new prompt on training batch (optional, disabled by default)
            train_score = None
            train_example_scores = None
            if config.eval_train and self.evaluator is not None and improved:
                train_eval = self._evaluate_prompt(opt_result.new_prompt, batch, config.val_max_tokens)
                train_score = train_eval.mean_score
                train_example_scores = train_eval.example_scores
                logger.info(f"Train score: {train_score:.3f}  Val score: {score_after:.3f}")

            # 5. Accept or reject
            if improved:
                current_version += 1
                current_prompt = opt_result.new_prompt
                current_eval_result = new_eval_result  # accepted prompt's eval becomes the new baseline
                _failed_batch_ids = []

                # Save new version
                version = PromptVersion(
                    version=current_version,
                    prompt_text=current_prompt,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    parent_version=current_version - 1,
                    training_log_entry=opt_result.learnings,
                    eval_score=score_after,
                    eval_details=new_eval_result.to_dict() if new_eval_result is not None else None,
                    output_schema=opt_result.output_schema,
                    metadata={"batch_ids": batch_ids, "iteration": iteration},
                )
                self.store.save_prompt_version(version)
                logger.info(f"Saved prompt version {current_version}")
            else:
                _failed_batch_ids = batch_ids  # retried by FailurePriorityBatchStrategy (the default when an evaluator is set)

            # Patience counts iterations without a STRICT val improvement: a tie
            # can be accepted (min_improvement=0) but must not reset the counter,
            # otherwise training never stops on a flat plateau.
            if made_progress:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                logger.info(f"No val improvement ({no_improvement_count}/{config.patience})")

            # 6. Update training log
            log_entry = LogEntry(
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                batch_ids=batch_ids,
                score_before=score_before,
                score_after=score_after,
                learnings=opt_result.learnings,
                issues=opt_result.issues,
                prompt_version=current_version,
            )
            self.training_log.add_entry(log_entry)

            # 7. Save state for resumption
            if config.auto_save:
                self._save_state(iteration)

            # Build iteration result
            iteration_tokens = self._total_tokens - tokens_before_iteration
            iter_result = IterationResult(
                iteration=iteration,
                prompt_version=current_version,
                score_before=score_before,
                score_after=score_after,
                train_score=train_score,
                val_example_scores=new_eval_result.example_scores if new_eval_result is not None else None,
                train_example_scores=train_example_scores,
                improved=improved,
                learnings=opt_result.learnings,
                issues=opt_result.issues,
                batch_ids=batch_ids,
                tokens_used=iteration_tokens,
            )
            results.append(iter_result)

            if self.on_iteration:
                self.on_iteration(iter_result)

            # Token budget check
            if config.max_total_tokens is not None and self._total_tokens >= config.max_total_tokens:
                logger.warning(
                    f"Total token budget reached: {self._total_tokens:,} tokens used "
                    f"(max_total_tokens={config.max_total_tokens:,}). Stopping training early."
                )
                break

            # Early stopping (only when evaluator is active)
            if self.evaluator is not None and no_improvement_count >= config.patience:
                logger.info(f"Early stopping: no improvement for {config.patience} iterations.")
                break

        logger.info(f"\nTraining complete. Final version: {current_version}")
        # Score of the prompt actually kept — not of the last candidate, which
        # may have been rejected.
        if current_eval_result is not None:
            final_score = current_eval_result.mean_score
        else:
            final_score = results[-1].score_after if results else None

        # One-shot held-out test evaluation — runs outside the accept/reject
        # loop, so the score is not inflated by hill-climbing on the val set.
        test_score = None
        test_example_scores = None
        if self.evaluator is not None and _test_list:
            logger.info(f"Evaluating final prompt on held-out test set ({len(_test_list)} examples)...")
            test_eval = self._evaluate_prompt(current_prompt, _test_list, config.val_max_tokens)
            test_score = test_eval.mean_score
            test_example_scores = test_eval.example_scores
            logger.info(f"Test score: {test_score:.3f}")

        # The test score, when available, is the honest generalization signal.
        reference_score = test_score if test_score is not None else final_score
        return TrainingReport(
            iterations=results,
            final_version=current_version,
            final_score=final_score,
            test_score=test_score,
            test_example_scores=test_example_scores,
            refinement_recommended=(
                reference_score is None or reference_score < config.refinement_threshold
            ),
            total_tokens_used=self._total_tokens - _tokens_at_start,
        )

    def consolidate(
        self,
        version: int | None = None,
        *,
        val_bundles: BundleCollection | list[ExampleBundle] | None = None,
        val_max_tokens: int | None = None,
    ) -> PromptVersion:
        """
        Compress the current (or specified) prompt by merging redundant rules.

        This is an explicit, user-initiated operation — it is never called
        automatically during training. Call it when you decide the prompt has
        grown unwieldy, then continue training from the consolidated baseline.

        The consolidated prompt is saved as a new version with a metadata flag
        so the history stays honest.

        Consolidation is lossy: pass ``val_bundles`` (with an evaluator set on
        the pipeline) to re-score the consolidated prompt immediately — a score
        drop means the compression lost coverage. Without ``val_bundles`` the
        new version inherits the source's ``eval_score``, which describes the
        prompt *before* compression.

        Args:
            version: Prompt version to consolidate. Defaults to the latest.
            val_bundles: Validation examples to re-score the consolidated
                         prompt on (requires an evaluator).
            val_max_tokens: Token budget per batch eval call (as in training).

        Returns:
            The new PromptVersion containing the consolidated prompt.

        Raises:
            ValueError: If no prompt versions exist in the store.
        """
        source = (
            self.store.get_prompt_version(version)
            if version is not None
            else self.store.get_latest_prompt()
        )
        if source is None:
            raise ValueError("No prompt versions found. Train at least one iteration first.")

        logger.info(
            f"Consolidating prompt v{source.version} "
            f"({len(source.prompt_text):,} chars)..."
        )
        result = self.optimizer.consolidate(source.prompt_text)
        self._total_tokens += self._count_tokens(result.usage)

        # Re-score the consolidated prompt when possible — consolidation is
        # lossy, and inheriting the source's score would hide any coverage loss.
        new_score = source.eval_score
        eval_details = None
        rescored = False
        if val_bundles is not None and self.evaluator is not None:
            _val_list = (
                val_bundles.bundles
                if isinstance(val_bundles, BundleCollection)
                else list(val_bundles)
            )
            if _val_list:
                if self._eval_agent is None:
                    self._eval_agent = InferenceAgent(
                        llm=self.llm,
                        prompt_text="",  # set per evaluation call
                        file_loader=self.file_loader,
                        token_estimator=self.optimizer.token_estimator,
                    )
                eval_result = self._evaluate_prompt(result.new_prompt, _val_list, val_max_tokens)
                new_score = eval_result.mean_score
                eval_details = eval_result.to_dict()
                rescored = True
                if source.eval_score is not None and new_score < source.eval_score:
                    logger.warning(
                        f"Consolidation dropped the val score: "
                        f"{source.eval_score:.3f} → {new_score:.3f}. "
                        "Review the consolidated prompt before continuing."
                    )
                else:
                    logger.info(f"Consolidated prompt re-scored: {new_score:.3f}")
        elif val_bundles is not None:
            logger.warning(
                "val_bundles passed to consolidate() but no evaluator is set — "
                "re-scoring skipped; the new version inherits the source's score."
            )

        latest = self.store.get_latest_prompt()
        new_version_num = (latest.version if latest else 0) + 1

        consolidated = PromptVersion(
            version=new_version_num,
            prompt_text=result.new_prompt,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_version=source.version,
            training_log_entry=(
                f"[CONSOLIDATION] Compressed from v{source.version} "
                f"({len(source.prompt_text):,} → {len(result.new_prompt):,} chars)"
            ),
            eval_score=new_score,
            eval_details=eval_details,
            output_schema=source.output_schema,
            metadata={
                "consolidation": True,
                "source_version": source.version,
                "rescored": rescored,
            },
        )
        self.store.save_prompt_version(consolidated)
        logger.info(
            f"Saved consolidated prompt as v{new_version_num} "
            f"({len(result.new_prompt):,} chars)."
        )
        return consolidated

    def _evaluate_prompt(
        self,
        prompt_text: str,
        val_bundles: list[ExampleBundle],
        val_max_tokens: int | None = None,
    ) -> BatchEvalResult:
        """Run inference + evaluation on the validation set.

        Bundles whose expected output cannot be loaded are excluded from the
        batch (before inference, so no tokens are wasted on them) rather than
        scored as 0 — a broken example must not silently drag the mean down.
        Raises RuntimeError if no bundle is evaluable at all.

        Uses InferenceAgent.run_bundle_batch (single LLM call) by default.
        Falls back to sequential calls when a custom inference_fn is set,
        or if the batch call raises.
        """
        from ..bundle import ExampleBundle, is_output_role

        # Load expected outputs up front; exclude bundles that fail.
        evaluable: list[tuple[ExampleBundle, str]] = []
        skipped: list[str] = []
        for bundle in val_bundles:
            output_files = {
                role: path for role, path in bundle.files.items() if is_output_role(role)
            }
            try:
                contents = ExampleBundle(
                    bundle_id=bundle.bundle_id, files=output_files
                ).load_contents(self.file_loader)
                expected = next(iter(contents.values())).text if contents else ""
                evaluable.append((bundle, expected))
            except Exception as e:
                logger.warning(
                    f"Excluding {bundle.bundle_id} from evaluation — "
                    f"failed to load its expected output: {e}"
                )
                skipped.append(bundle.bundle_id)

        if not evaluable:
            raise RuntimeError(
                f"Evaluation is impossible: the expected output could not be loaded "
                f"for any of the {len(val_bundles)} bundles ({skipped}). "
                "Check the example files."
            )
        if skipped:
            logger.warning(
                f"Evaluating {len(evaluable)}/{len(val_bundles)} examples — "
                f"excluded due to load failures: {skipped}"
            )

        bundles = [b for b, _ in evaluable]
        if self._custom_inference_fn is None:
            # Default path: batch inference via InferenceAgent
            self._eval_agent.prompt_text = prompt_text
            tokens_before = self._eval_agent.tokens_used
            try:
                actuals = self._eval_agent.run_bundle_batch(bundles, val_max_tokens)
            except Exception as e:
                logger.warning(
                    f"Batch inference failed ({e}), falling back to sequential. "
                    f"Enable DEBUG logging to see the raw LLM response."
                )
                actuals = [self._eval_agent.run_bundle(b) for b in bundles]
            self._total_tokens += self._eval_agent.tokens_used - tokens_before
        else:
            # Custom inference function: always sequential
            actuals = [self._custom_inference_fn(prompt_text, b) for b in bundles]

        results = [
            (bundle.bundle_id, actual, expected)
            for (bundle, expected), actual in zip(evaluable, actuals)
        ]
        return self.evaluator.evaluate_batch(results)

    @staticmethod
    def _count_tokens(usage: dict[str, int] | None) -> int:
        """Sum input and output tokens from a usage dict."""
        if usage is None:
            return 0
        return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)

    def _save_state(self, iteration: int) -> None:
        """Save training state for resumption."""
        state = {
            "last_iteration": iteration,
            "training_log": self.training_log.to_dict(),
            "total_tokens_used": self._total_tokens,
        }
        self.store.save_training_state(state)

    def _restore_state(self) -> None:
        """Restore training state if available."""
        state = self.store.load_training_state()
        if state and "training_log" in state:
            self.training_log = TrainingLog.from_dict(state["training_log"])
            self._total_tokens = state.get("total_tokens_used", 0)
            logger.info(
                f"Restored training state: {len(self.training_log.entries)} previous iterations, "
                f"{self._total_tokens} tokens used so far"
            )
