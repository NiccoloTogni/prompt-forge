"""
Project — the main entry point for the PromptForge library.

A Project ties together all components: storage, bundles, training, inference.
It provides a high-level API for the full workflow.

    Usage:
        project = Project("my_project", llm=my_llm_client)
        project.set_bundle_schema(input=".pdf", expected_output=".json")
        project.add_examples_from_directory("./training_data/")
        project.set_seed_prompt("Extract all fields from the document...")
        project.train(config=TrainingConfig(batch_size=5, max_iterations=20))

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
        self._optimizer_prompt: str | None = None
        self._consolidation_prompt: str | None = None
        self._output_schema: dict | None = None

        # Try to load existing config
        self._load_config()

    # ── Schema & Examples ─────────────────────────────────────────────

    def set_bundle_schema(
        self,
        roles: dict[str, str] | None = None,
        role_descriptions: dict[str, str] | None = None,
        variadic: list[str] | None = None,
        **kwargs: str,
    ) -> None:
        """
        Define the structure of example bundles.

        Can be called with a dict or with keyword arguments:
            project.set_bundle_schema(input=".pdf", expected_output=".json")
            project.set_bundle_schema(roles={"input": ".pdf", "expected_output": ".json"})
            project.set_bundle_schema(mail=".txt", attachments=".pdf", expected_output=".json",
                                      variadic=["attachments"])

        Args:
            roles: Dict of role_name → file_extension.
            role_descriptions: Optional descriptions for each role.
            variadic: List of role names that accept 0..N files instead of exactly one.
                      Variadic roles are optional in each bundle and their files are
                      collected as a list. Directory loading collects all matching files.
            **kwargs: Alternative to roles dict (role_name=extension).
        """
        if roles is None:
            roles = kwargs
        if not roles:
            raise ValueError("Must specify at least one role.")

        self._schema = BundleSchema(
            roles=roles,
            role_descriptions=role_descriptions or {},
            variadic_roles=set(variadic) if variadic else set(),
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
        # Always write version 0 — overwrite if it already exists so the store
        # stays in sync with the config (prevents drift on repeated calls).
        existing = self.store.get_prompt_version(0)
        if existing is None or existing.metadata.get("is_seed"):
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

    def set_optimizer_prompt(self, prompt: str) -> None:
        """
        Override the system prompt used by the Prompt Engineering Agent.

        Inspect ``DEFAULT_OPTIMIZER_PROMPT`` to understand the expected format
        and build custom variants on top of it.
        """
        self._optimizer_prompt = prompt
        self._save_config()

    def set_consolidation_prompt(self, prompt: str) -> None:
        """
        Override the prompt used when consolidating a grown system prompt.

        Inspect ``DEFAULT_CONSOLIDATION_PROMPT`` to understand the expected format.
        """
        self._consolidation_prompt = prompt
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
        train_bundles=None,
        *,
        val_bundles=None,
        test_bundles=None,
        config: TrainingConfig | None = None,
        eval_strategy: str | Evaluator | None = "llm_judge",
        batch_strategy: BatchStrategy | None = None,
        inference_fn: Callable | None = None,
        on_iteration: Callable | None = None,
        optimizer_kwargs: dict | None = None,
    ) -> TrainingReport:
        """
        Run the incremental training loop.

        Args:
            train_bundles: Training examples. Defaults to all loaded examples
                           (``project.bundles``). Pass an explicit subset — e.g. the
                           output of ``train_val_split`` — to keep validation examples
                           out of the optimizer's view.
            val_bundles: Validation examples used for scoring. When provided,
                         ``patience`` and ``min_improvement`` become effective.
            test_bundles: Held-out examples evaluated exactly once on the final
                          prompt, after training ends. The loop hill-climbs on
                          the validation score, so only a set it never sees gives
                          an unbiased generalization estimate — reported as
                          ``TrainingReport.test_score``. Use ``train_val_test_split``
                          to produce the three sets.
            config: Full training configuration. Use ``TrainingConfig`` to control
                    batch size, iterations, evaluation thresholds, token budgets,
                    temperatures, and more. Defaults to ``TrainingConfig()`` (all defaults).
            eval_strategy: Evaluator instance or string shortcut
                          ("exact_match", "json_fields", "similarity", "llm_judge", "none").
                          Pass None or "none" to disable evaluation — training always runs
                          max_iterations; min_improvement and patience are ignored.
            batch_strategy: Batch selection strategy (default: RandomBatchStrategy).
            inference_fn: Custom inference function ``(prompt_text, bundle) -> str``.
                          Defaults to an LLM call using the project's client.
            on_iteration: Optional callback called after each iteration with an
                          ``IterationResult``.
            optimizer_kwargs: Extra kwargs forwarded to ``PromptOptimizer``
                              (e.g. ``token_estimator``).

        Returns:
            TrainingReport with per-iteration results and refinement signal.
        """
        if self._bundles is None or len(self._bundles) == 0:
            raise RuntimeError("No training examples loaded.")
        if self.store.get_latest_prompt() is None:
            raise RuntimeError("No seed prompt set. Call set_seed_prompt() first.")

        if train_bundles is None:
            train_bundles = self._bundles

        config = config or TrainingConfig()
        # Inject output_schema from the project if not already set in config
        if config.output_schema is None and self._output_schema is not None:
            import dataclasses
            config = dataclasses.replace(config, output_schema=self._output_schema)

        evaluator = self._resolve_evaluator(eval_strategy)

        _opt_kwargs = dict(optimizer_kwargs or {})
        optimizer = PromptOptimizer(
            llm=self.llm,
            optimizer_prompt=_opt_kwargs.pop("optimizer_prompt", self._optimizer_prompt),
            consolidation_prompt=_opt_kwargs.pop("consolidation_prompt", self._consolidation_prompt),
            file_loader=self.file_loader,
            context=self._context,
            **_opt_kwargs,
        )

        pipeline = TrainingPipeline(
            llm=self.llm,
            store=self.store,
            evaluator=evaluator,
            optimizer=optimizer,
            batch_strategy=batch_strategy or RandomBatchStrategy(seed=config.seed),
            file_loader=self.file_loader,
            context=self._context,
            inference_fn=inference_fn,
            on_iteration=on_iteration,
        )

        return pipeline.train(
            train_bundles,
            val_bundles=val_bundles,
            test_bundles=test_bundles,
            config=config,
        )

    def consolidate(self, version: int | None = None):
        """
        Compress the current (or specified) prompt by merging redundant rules.

        Explicit, user-initiated operation — never called automatically during
        training. Call when the prompt has grown unwieldy, then continue training
        from the consolidated baseline.

        Args:
            version: Prompt version to consolidate. Defaults to the latest.

        Returns:
            The new PromptVersion containing the consolidated prompt.
        """
        optimizer = PromptOptimizer(
            llm=self.llm,
            consolidation_prompt=self._consolidation_prompt,
            file_loader=self.file_loader,
            context=self._context,
        )
        pipeline = TrainingPipeline(
            llm=self.llm,
            store=self.store,
            optimizer=optimizer,
        )
        return pipeline.consolidate(version=version)

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
    def bundles(self) -> BundleCollection | None:
        """The loaded example collection, or None if no schema has been set."""
        return self._bundles

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
            "optimizer_prompt": self._optimizer_prompt,
            "consolidation_prompt": self._consolidation_prompt,
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
        self._optimizer_prompt = config.get("optimizer_prompt")
        self._consolidation_prompt = config.get("consolidation_prompt")
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


