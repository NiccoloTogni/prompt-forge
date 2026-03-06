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

from ..llm.client import LLMClient, LLMMessage
from ..bundle import BundleCollection, ExampleBundle
from ..file_loaders import FileLoader, get_default_loader
from ..storage.project_store import ProjectStore, PromptVersion
from ..evaluation.evaluator import Evaluator, BatchEvalResult
from .optimizer import PromptOptimizer
from .batch_strategy import BatchStrategy, RandomBatchStrategy
from .training_log import TrainingLog, LogEntry

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TrainingConfig:
    """Configuration for a training run."""
    batch_size: int = 10
    max_iterations: int = 20
    min_improvement: float = 0.00  # Minimum score improvement to accept new prompt
    patience: int = 5  # Stop after N iterations without improvement
    eval_sample_size: int | None = None  # How many examples to evaluate on (None = all)
    auto_save: bool = True
    output_schema: dict | None = None  # JSON schema if task requires structured output
    refinement_threshold: float = 0.8  # Scores below this recommend human refinement
    max_tokens: int | None = None  # Context window limit for optimizer; None disables check
    max_total_tokens: int | None = None  # Budget for the whole run; None = unlimited


@dataclasses.dataclass
class IterationResult:
    """Result of a single training iteration."""
    iteration: int
    prompt_version: int
    score_before: float | None
    score_after: float | None
    improved: bool
    learnings: str
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

    def __iter__(self):
        return iter(self.iterations)

    def __len__(self):
        return len(self.iterations)


class TrainingPipeline:
    """
    Orchestrates the incremental prompt training loop.

    Usage:
        pipeline = TrainingPipeline(
            llm=my_llm,
            store=my_store,
            bundles=my_bundles,
            evaluator=my_evaluator,
        )
        pipeline.train(config=TrainingConfig(batch_size=5, max_iterations=20))
    """

    def __init__(
        self,
        llm: LLMClient,
        store: ProjectStore,
        bundles: BundleCollection,
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
            bundles: Collection of training examples.
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
        self.bundles = bundles
        self.evaluator = evaluator
        self.file_loader = file_loader or get_default_loader()
        self.context = context
        self.inference_fn = inference_fn or self._default_inference
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
        config: TrainingConfig | None = None,
    ) -> TrainingReport:
        """
        Run the training loop.

        Args:
            config: Training configuration.

        Returns:
            List of results for each iteration.
        """
        config = config or TrainingConfig()
        results: list[IterationResult] = []
        no_improvement_count = 0

        # Get starting prompt
        current = self.store.get_latest_prompt()
        if current is None:
            raise RuntimeError(
                "No seed prompt found. Set a seed prompt before training: "
                "project.set_seed_prompt('...')"
            )

        current_prompt = current.prompt_text
        current_version = current.version

        logger.info(
            f"Starting training: {len(self.bundles)} examples, "
            f"batch_size={config.batch_size}, max_iterations={config.max_iterations}"
        )

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"\n{'='*60}\nIteration {iteration}/{config.max_iterations}\n{'='*60}")

            # 1. Select batch
            batch = self.batch_strategy.select_batch(
                bundles=self.bundles.bundles,
                batch_size=config.batch_size,
                used_ids=self.training_log.get_all_used_bundle_ids(),
            )
            batch_ids = [b.bundle_id for b in batch]
            logger.info(f"Selected batch: {batch_ids}")

            # 2. Evaluate current prompt (optional, for scoring)
            score_before = None
            eval_feedback = ""
            if self.evaluator is not None and config.eval_sample_size != 0:
                eval_result = self._evaluate_prompt(current_prompt, config.eval_sample_size)
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
                training_history=self.training_log.get_summary(max_entries=15),
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
            if self.evaluator is not None and config.eval_sample_size != 0:
                new_eval_result = self._evaluate_prompt(opt_result.new_prompt, config.eval_sample_size)
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
                logger.info(f"Prompt not improved ({no_improvement_count}/{config.patience})")

            # 6. Update training log
            log_entry = LogEntry(
                iteration=iteration,
                timestamp=datetime.now(timezone.utc).isoformat(),
                batch_ids=batch_ids,
                score_before=score_before,
                score_after=score_after,
                learnings=opt_result.learnings,
                errors_addressed=[],  # TODO: extract from eval feedback
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
        sample_size: int | None = None,
    ) -> BatchEvalResult:
        """Run inference + evaluation on a sample of examples."""
        bundles = self.bundles.bundles
        if sample_size and sample_size < len(bundles):
            import random
            bundles = random.sample(bundles, sample_size)

        results = []
        for bundle in bundles:
            try:
                contents = bundle.load_contents(self.file_loader)
                actual = self.inference_fn(prompt_text, bundle)
                expected = ""
                # Find the expected output role
                for role, content in contents.items():
                    if "expected" in role.lower() or "output" in role.lower():
                        expected = content.text
                        break
                results.append((bundle.bundle_id, actual, expected))
            except Exception as e:
                logger.warning(f"Evaluation failed for {bundle.bundle_id}: {e}")
                results.append((bundle.bundle_id, "", f"[Error: {e}]"))

        return self.evaluator.evaluate_batch(results)

    def _default_inference(self, prompt_text: str, bundle: ExampleBundle) -> str:
        """Default inference: use the LLM with the prompt to process the input."""
        contents = bundle.load_contents(self.file_loader)

        # Find input role(s) — anything that's not "expected*" or "output*"
        input_parts = []
        for role, content in contents.items():
            if not ("expected" in role.lower() or "output" in role.lower()):
                input_parts.append(f"<{role}>\n{content.text}\n</{role}>")

        user_content = "\n\n".join(input_parts)

        messages = [
            LLMMessage(role="system", content=prompt_text),
            LLMMessage(role="user", content=user_content),
        ]

        response = self.llm.complete(messages, temperature=0.0)
        if response.usage:
            self._total_tokens += (
                response.usage.get("input_tokens", 0)
                + response.usage.get("output_tokens", 0)
            )
        return response.text

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
