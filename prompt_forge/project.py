"""
Project — the main entry point for the PromptForge library.

A Project ties together all components: storage, bundles, training, inference.
It provides a high-level API for the full workflow.

    Usage:
        project = Project("my_project", llm=my_llm_client)
        project.set_bundle_schema(input=".pdf", expected_output=".json")
        project.add_examples_from_directory("./training_data/")
        project.set_seed_prompt("Extract all fields from the document...")
        project.train(batch_size=5, max_iterations=20)

        agent = project.get_inference_agent()
        result = agent.run(input_file="new_file.pdf")
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .llm.client import LLMClient
from .file_loaders import FileLoader, get_default_loader
from .bundle import BundleSchema, BundleCollection, ExampleBundle
from .storage.project_store import FileSystemStore, ProjectStore, PromptVersion
from .training.pipeline import TrainingPipeline, TrainingConfig, TrainingReport
from .training.optimizer import PromptOptimizer
from .training.batch_strategy import BatchStrategy, RandomBatchStrategy
from .evaluation.evaluator import (
    Evaluator,
    ExactMatchEvaluator,
    JsonFieldEvaluator,
    LLMJudgeEvaluator,
    SimilarityEvaluator,
)
from .inference.agent import InferenceAgent
from .interactive.optimizer import InteractiveOptimizer, InteractiveSessionResult

logger = logging.getLogger(__name__)


class Project:

    def __init__(
        self,
        name: str,
        llm: LLMClient,
        project_dir: str | Path | None = None,
        store: ProjectStore | None = None,
        file_loader: FileLoader | None = None,
    ):
        """
        Args:
            name: Project name (used as directory name if project_dir not specified).
            llm: LLM client for all operations.
            project_dir: Path to project directory. Defaults to ./{name}
            store: Custom storage backend. Defaults to FileSystemStore.
            file_loader: Custom file loader. Defaults to built-in loader.
        """
        self.name = name
        self.llm = llm
        self.file_loader = file_loader or get_default_loader()

        # Storage
        self.project_dir = Path(project_dir or f"./{name}")
        self.store = store or FileSystemStore(self.project_dir)

        # State
        self._schema: BundleSchema | None = None
        self._bundles: BundleCollection | None = None
        self._context: str = ""
        self._seed_prompt: str | None = None
        self._meta_prompt: str | None = None
        self._output_schema: dict | None = None

        # Try to load existing config
        self._load_config()

    # ── Schema & Examples ─────────────────────────────────────────────

    def set_bundle_schema(
        self,
        roles: dict[str, str] | None = None,
        role_descriptions: dict[str, str] | None = None,
        **kwargs: str,
    ) -> None:
        """
        Define the structure of example bundles.

        Can be called with a dict or with keyword arguments:
            project.set_bundle_schema(input=".pdf", expected_output=".json")
            project.set_bundle_schema(roles={"input": ".pdf", "expected_output": ".json"})

        Args:
            roles: Dict of role_name → file_extension.
            role_descriptions: Optional descriptions for each role.
            **kwargs: Alternative to roles dict (role_name=extension).
        """
        if roles is None:
            roles = kwargs
        if not roles:
            raise ValueError("Must specify at least one role.")

        self._schema = BundleSchema(
            roles=roles,
            role_descriptions=role_descriptions or {},
        )
        self._bundles = BundleCollection(schema=self._schema, loader=self.file_loader)
        self._save_config()
        logger.info(f"Bundle schema set: {roles}")

    def add_examples_from_directory(
        self, directory: str | Path, **kwargs
    ) -> int:
        """
        Load example bundles from a directory.

        See BundleCollection.add_from_directory for directory layout options.

        Returns:
            Number of bundles loaded.
        """
        if self._bundles is None:
            raise RuntimeError("Set a bundle schema first with set_bundle_schema().")
        count = self._bundles.add_from_directory(directory, **kwargs)
        self._save_config()
        logger.info(f"Loaded {count} example bundles from {directory}")
        return count

    def add_example(
        self,
        bundle_id: str,
        files: dict[str, str | Path],
        metadata: dict | None = None,
    ) -> None:
        """
        Add a single example bundle manually.

        Args:
            bundle_id: Unique ID for this example.
            files: Dict of role_name → file_path.
            metadata: Optional metadata.
        """
        if self._bundles is None:
            raise RuntimeError("Set a bundle schema first with set_bundle_schema().")
        bundle = ExampleBundle(
            bundle_id=bundle_id,
            files={role: Path(p) for role, p in files.items()},
            metadata=metadata or {},
        )
        self._bundles.add(bundle)

    # ── Prompt & Context ──────────────────────────────────────────────

    def set_seed_prompt(self, prompt: str) -> None:
        """
        Set the initial prompt to start training from.

        This is saved as version 0 in the prompt store.
        """
        self._seed_prompt = prompt
        # Save as version 0 if no versions exist yet
        if self.store.get_latest_prompt() is None:
            version = PromptVersion(
                version=0,
                prompt_text=prompt,
                created_at=datetime.now(timezone.utc).isoformat(),
                parent_version=None,
                training_log_entry="Seed prompt",
                metadata={"is_seed": True},
            )
            self.store.save_prompt_version(version)
        self._save_config()
        logger.info("Seed prompt set.")

    def set_context(self, context: str) -> None:
        """
        Set domain context that helps the optimizer understand the task.

        Example:
            project.set_context(
                "These are heat exchanger purchase orders from European manufacturers. "
                "Units are typically metric. Prices are in EUR."
            )
        """
        self._context = context
        self._save_config()

    def set_meta_prompt(self, meta_prompt: str) -> None:
        """
        Override the default meta-prompt used by the Prompt Engineering Agent.

        Only needed for advanced customization. The default works well for
        most use cases.
        """
        self._meta_prompt = meta_prompt
        self._save_config()

    def set_output_schema(self, schema: dict) -> None:
        """
        Declare that the task produces structured JSON output.

        The schema is stored with every trained prompt version and used by:
        - The PromptOptimizer, to ensure the optimized prompt instructs JSON output.
        - The InferenceAgent, to parse the LLM response and return a dict.

        Args:
            schema: A dict describing the expected JSON output. Can be a plain
                    field-description mapping or a full JSON Schema object.
                    Example: {"invoice_number": "string", "total": "number"}
        """
        if not isinstance(schema, dict):
            raise ValueError("output_schema must be a dict.")
        self._output_schema = schema
        self._save_config()
        logger.info(f"Output schema set: {list(schema.keys())}")

    # ── Training ──────────────────────────────────────────────────────

    def train(
        self,
        batch_size: int = 5,
        max_iterations: int = 20,
        eval_strategy: str | Evaluator | None = "llm_judge",
        batch_strategy: BatchStrategy | None = None,
        min_improvement: float = 0.01,
        patience: int = 3,
        eval_sample_size: int | None = None,
        inference_fn: Callable | None = None,
        on_iteration: Callable | None = None,
        optimizer_kwargs: dict | None = None,
        max_tokens: int | None = None,
        max_total_tokens: int | None = None,
    ) -> TrainingReport:
        """
        Run the incremental training loop.

        Args:
            batch_size: Examples per iteration.
            max_iterations: Maximum training iterations.
            eval_strategy: Evaluator instance or string shortcut
                          ("exact_match", "json_fields", "similarity", "llm_judge", "none").
            batch_strategy: Batch selection strategy (default: random).
            min_improvement: Minimum score delta to accept new prompt.
            patience: Stop after N iterations without improvement.
            eval_sample_size: Number of examples to evaluate on (None = all).
            inference_fn: Custom inference function (prompt, bundle) -> str.
            on_iteration: Callback after each iteration.
            optimizer_kwargs: Extra kwargs for the PromptOptimizer (e.g. token_estimator).
            max_tokens: Context window token limit per optimizer call. When set,
                        the batch is trimmed to fit (warning logged). If a single
                        example exceeds the budget alone, training fails with an error.
            max_total_tokens: Total token budget for the entire training run. Training
                              stops early with a warning when this limit is reached.

        Returns:
            TrainingReport with per-iteration results and refinement signal.
        """
        if self._bundles is None or len(self._bundles) == 0:
            raise RuntimeError("No training examples loaded.")
        if self.store.get_latest_prompt() is None:
            raise RuntimeError("No seed prompt set. Call set_seed_prompt() first.")

        evaluator = self._resolve_evaluator(eval_strategy)

        optimizer = PromptOptimizer(
            llm=self.llm,
            meta_prompt=self._meta_prompt,
            file_loader=self.file_loader,
            context=self._context,
            **(optimizer_kwargs or {}),
        )

        config = TrainingConfig(
            batch_size=batch_size,
            max_iterations=max_iterations,
            min_improvement=min_improvement,
            patience=patience,
            eval_sample_size=eval_sample_size,
            output_schema=self._output_schema,
            max_tokens=max_tokens,
            max_total_tokens=max_total_tokens,
        )

        pipeline = TrainingPipeline(
            llm=self.llm,
            store=self.store,
            bundles=self._bundles,
            evaluator=evaluator,
            optimizer=optimizer,
            batch_strategy=batch_strategy or RandomBatchStrategy(),
            file_loader=self.file_loader,
            context=self._context,
            inference_fn=inference_fn,
            on_iteration=on_iteration,
        )

        return pipeline.train(config)

    # ── Inference ─────────────────────────────────────────────────────

    def get_inference_agent(
        self,
        version: int | None = None,
        **kwargs,
    ) -> InferenceAgent:
        """
        Get an InferenceAgent loaded with a trained prompt.

        The output_schema is automatically loaded from the stored prompt version.
        Pass output_schema=... explicitly in kwargs to override it.

        Args:
            version: Specific prompt version, or None for latest.
            **kwargs: Additional args passed to InferenceAgent (e.g., output_schema, llm_kwargs).

        Returns:
            Ready-to-use InferenceAgent.
        """
        return InferenceAgent.from_store(
            llm=self.llm,
            store=self.store,
            version=version,
            file_loader=self.file_loader,
            **kwargs,
        )

    def refine(
        self,
        version: int | None = None,
        input_fn: Callable[[], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> InteractiveSessionResult:
        """
        Start an interactive human refinement session on a trained prompt.

        Shortcut for::

            optimizer = InteractiveOptimizer(llm=..., store=..., bundles=...)
            result = optimizer.run_session(prompt_text=...)

        Args:
            version: Prompt version to refine, or None for the latest.
            input_fn: Reads human input. Defaults to built-in input().
            output_fn: Displays output. Defaults to print().

        Returns:
            InteractiveSessionResult with the final prompt and session history.
        """
        prompt_version = self.get_prompt(version)
        if prompt_version is None:
            raise RuntimeError("No prompt found. Run train() or set_seed_prompt() first.")

        optimizer = InteractiveOptimizer(
            llm=self.llm,
            store=self.store,
            file_loader=self.file_loader,
            bundles=self._bundles,
            context=self._context,
        )
        return optimizer.run_session(
            prompt_text=prompt_version.prompt_text,
            input_fn=input_fn,
            output_fn=output_fn,
        )

    # ── Prompt History ────────────────────────────────────────────────

    def list_versions(self) -> list[PromptVersion]:
        """List all prompt versions."""
        return self.store.list_versions()

    def get_prompt(self, version: int | None = None) -> PromptVersion | None:
        """Get a specific prompt version, or the latest."""
        if version is not None:
            return self.store.get_prompt_version(version)
        return self.store.get_latest_prompt()

    @property
    def num_examples(self) -> int:
        """Number of loaded training examples."""
        return len(self._bundles) if self._bundles else 0

    @property
    def num_versions(self) -> int:
        """Number of prompt versions."""
        return len(self.store.list_versions())

    # ── Internal ──────────────────────────────────────────────────────

    def _resolve_evaluator(self, strategy: str | Evaluator | None) -> Evaluator | None:
        """Convert string shortcut to Evaluator instance. Returns None to disable evaluation."""
        if strategy is None or strategy == "none":
            return None
        if isinstance(strategy, Evaluator):
            return strategy
        mapping = {
            "exact_match": ExactMatchEvaluator,
            "json_fields": JsonFieldEvaluator,
            "similarity": SimilarityEvaluator,
            "llm_judge": lambda: LLMJudgeEvaluator(llm=self.llm, task_description=self._context),
        }
        factory = mapping.get(strategy)
        if factory is None:
            raise ValueError(
                f"Unknown eval strategy '{strategy}'. "
                f"Options: {list(mapping.keys()) + ['none']} or pass an Evaluator instance."
            )
        return factory() if callable(factory) else factory

    def _save_config(self) -> None:
        """Persist project configuration."""
        config = {
            "name": self.name,
            "context": self._context,
            "seed_prompt": self._seed_prompt,
            "meta_prompt": self._meta_prompt,
            "output_schema": self._output_schema,
            "schema": self._schema.to_dict() if self._schema else None,
            "bundles": self._bundles.to_dict() if self._bundles else None,
        }
        self.store.save_project_config(config)

    def _load_config(self) -> None:
        """Load project configuration if it exists."""
        config = self.store.load_project_config()
        if config is None:
            return
        self._context = config.get("context", "")
        self._seed_prompt = config.get("seed_prompt")
        self._meta_prompt = config.get("meta_prompt")
        self._output_schema = config.get("output_schema")
        if config.get("schema"):
            self._schema = BundleSchema.from_dict(config["schema"])
            self._bundles = BundleCollection(schema=self._schema, loader=self.file_loader)
            if config.get("bundles"):
                bundle_data = config["bundles"]
                for bid, bdata in bundle_data.get("bundles", {}).items():
                    bundle = ExampleBundle.from_dict(bdata)
                    self._bundles._bundles[bid] = bundle
        logger.info(f"Loaded project config: {self.name}")

    def __repr__(self) -> str:
        return (
            f"Project(name='{self.name}', "
            f"examples={self.num_examples}, "
            f"versions={self.num_versions})"
        )


