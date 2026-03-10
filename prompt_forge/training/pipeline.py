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

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Callable

from ..llm.client import LLMClient
from ..bundle import BundleCollection, ExampleBundle
from ..file_loaders import FileLoader, get_default_loader
from ..storage.project_store import ProjectStore, PromptVersion
from ..evaluation.evaluator import Evaluator, BatchEvalResult
from ..inference.agent import InferenceAgent
from .optimizer import PromptOptimizer, OPTIMIZER_TEMPERATURE
from .batch_strategy import BatchStrategy, RandomBatchStrategy
from .training_log import TrainingLog, LogEntry

logger = logging.getLogger(__name__)

# ── Module-level defaults ─────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 10
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_MIN_IMPROVEMENT = 0.0        # Accept any improvement; ignored when evaluator is None
DEFAULT_PATIENCE = 5                 # Early-stop after N non-improving iters; ignored when evaluator is None
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
    optimizer_temperature: float = OPTIMIZER_TEMPERATURE  # Temperature for optimizer calls
    inference_temperature: float | None = DEFAULT_INFERENCE_TEMPERATURE  # None = use model default


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
    tokens_used: int | None = None  # Total tokens consumed in this iteration (optimizer + eval)


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

    @property
    def all_issues(self) -> list[tuple[int, str]]:
        """All non-empty issues flagged by the optimizer, as (iteration, issues) pairs."""
        return [
            (r.iteration, r.issues)
            for r in self.iterations
            if r.issues
        ]

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
        train_bundles, val_bundles = all_bundles.train_val_split(val_ratio=0.2, seed=42)
        pipeline.train(train_bundles, val_bundles, config=TrainingConfig(batch_size=5))
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
            batch_strategy: How to select batches (default: RandomBatchStrategy).
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
        self.inference_fn = inference_fn           # kept for backward compatibility
        self.on_iteration = on_iteration

        self.optimizer = optimizer or PromptOptimizer(
            llm=llm,
            file_loader=self.file_loader,
            context=context,
        )
        self.batch_strategy = batch_strategy or RandomBatchStrategy()
        self.training_log = TrainingLog()
        self._total_tokens: int = 0  # Running token counter across the whole training run

        # Restore state if available
        self._restore_state()

    def train(
        self,
        train_bundles: BundleCollection | list[ExampleBundle],
        *,
        val_bundles: BundleCollection | list[ExampleBundle] | None = None,
        config: TrainingConfig | None = None,
    ) -> TrainingReport:
        """
        Run the training loop.

        Args:
            train_bundles: Training examples used for optimization.
            val_bundles: Validation examples used for scoring (optional).
            config: Training configuration.

        Returns:
            TrainingReport with per-iteration results and refinement signal.
        """
        config = config or TrainingConfig()
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
        )
        no_improvement_count = 0
        _failed_batch_ids: list[str] = []  # IDs from the last non-improving batch

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

        if self.evaluator is not None and not _val_list:
            logger.warning(
                "An evaluator is set but no val_bundles were provided — "
                "evaluation will be skipped. Pass val_bundles to enable scoring."
            )

        logger.info(
            f"Starting training: {len(_train_list)} train examples, "
            f"val_size={len(_val_list)}, "
            f"batch_size={config.batch_size}, max_iterations={config.max_iterations}"
        )

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"\n{'='*60}\nIteration {iteration}/{config.max_iterations}\n{'='*60}")

            # 1. Select batch
            batch = self.batch_strategy.select_batch(
                bundles=_train_list,
                batch_size=config.batch_size,
                used_ids=self.training_log.get_all_used_bundle_ids(),
                failed_ids=_failed_batch_ids,
            )
            batch_ids = [b.bundle_id for b in batch]
            logger.info(f"Selected batch: {batch_ids}")

            # 2. Evaluate current prompt (optional, for scoring)
            score_before = None
            eval_feedback = ""
            if self.evaluator is not None and _val_list:
                eval_result = self._evaluate_prompt(current_prompt, _val_list, config.val_max_tokens)
                score_before = eval_result.mean_score
                if eval_result.failed_examples:
                    feedback_lines = [
                        f"- {f['bundle_id']}: {f['feedback']}"
                        for f in eval_result.failed_examples[:10]
                    ]
                    eval_feedback = (
                        f"Current prompt scores {eval_result.mean_score:.2f} "
                        f"({eval_result.pass_rate:.0%} pass rate).\n"
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
            new_eval_result = None
            if self.evaluator is not None and _val_list:
                new_eval_result = self._evaluate_prompt(opt_result.new_prompt, _val_list, config.val_max_tokens)
                score_after = new_eval_result.mean_score
                improved = (
                    score_before is None
                    or score_after >= score_before + config.min_improvement
                )
                logger.info(f"Score after: {score_after:.3f} ({'improved' if improved else 'not improved'})")
            else:
                # No evaluator — always accept and run all iterations
                improved = True

            # 5. Accept or reject
            if improved:
                current_version += 1
                current_prompt = opt_result.new_prompt
                no_improvement_count = 0
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
                no_improvement_count += 1
                _failed_batch_ids = batch_ids  # retry these with FailurePriorityBatchStrategy
                logger.info(f"Prompt not improved ({no_improvement_count}/{config.patience})")

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
        final_score = results[-1].score_after if results else None
        return TrainingReport(
            iterations=results,
            final_version=current_version,
            final_score=final_score,
            refinement_recommended=(
                final_score is None or final_score < config.refinement_threshold
            ),
            total_tokens_used=self._total_tokens,
        )

    def _evaluate_prompt(
        self,
        prompt_text: str,
        val_bundles: list[ExampleBundle],
        val_max_tokens: int | None = None,
    ) -> BatchEvalResult:
        """Run inference + evaluation on the validation set.

        Uses InferenceAgent.run_bundle_batch (single LLM call) by default.
        Falls back to sequential calls when a custom inference_fn is set,
        or if the batch call raises.
        """
        from ..bundle import is_output_role

        if self._custom_inference_fn is None:
            # Default path: batch inference via InferenceAgent
            self._eval_agent.prompt_text = prompt_text
            tokens_before = self._eval_agent.tokens_used
            try:
                actuals = self._eval_agent.run_bundle_batch(val_bundles, val_max_tokens)
            except Exception as e:
                logger.warning(f"Batch inference failed ({e}), falling back to sequential")
                actuals = [self._eval_agent.run_bundle(b) for b in val_bundles]
            self._total_tokens += self._eval_agent.tokens_used - tokens_before
        else:
            # Custom inference function: always sequential
            actuals = [self._custom_inference_fn(prompt_text, b) for b in val_bundles]

        results = []
        for bundle, actual in zip(val_bundles, actuals):
            try:
                contents = bundle.load_contents(self.file_loader)
                expected = next(
                    (content.text for role, content in contents.items() if is_output_role(role)),
                    "",
                )
                results.append((bundle.bundle_id, actual, expected))
            except Exception as e:
                logger.warning(f"Evaluation failed for {bundle.bundle_id}: {e}")
                results.append((bundle.bundle_id, "", f"[Error: {e}]"))

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
