"""
APE — Automatic Prompt Engineering

A library for incrementally training system prompts from examples.
Instead of updating model weights, APE updates prompts — producing
human-readable, versionable, editable "learned knowledge".

Quick Start:
    from prompt_forge import Project

    project = Project("my_project", llm=my_llm_client)
    project.set_bundle_schema(input=".pdf", expected_output=".json")
    project.add_examples_from_directory("./training_data/")
    project.set_seed_prompt("Extract all fields from the document...")
    project.train(config=TrainingConfig(batch_size=5, max_iterations=20))

    agent = project.get_inference_agent()
    result = agent.run(input_file="new_file.pdf")
"""

__version__ = "0.1.0"

from .project import Project
from .utils import train_val_split
from .inference.agent import InferenceAgent
from .bundle import BundleSchema, BundleCollection, ExampleBundle
from .llm.client import LLMClient, LLMResponse, LLMMessage
from .file_loaders import FileLoader, FileContent, get_default_loader
from .storage.project_store import (
    FileSystemStore,
    SQLiteStore,
    PromptVersion,
)
from .training.pipeline import TrainingPipeline, TrainingConfig, TrainingReport, IterationResult
from .interactive.optimizer import InteractiveOptimizer, InteractiveSessionResult
from .training.optimizer import PromptOptimizer
from .training.batch_strategy import (
    BatchStrategy,
    RandomBatchStrategy,
    FailurePriorityBatchStrategy,
)
from .training.training_log import TrainingLog
from .evaluation.evaluator import (
    Evaluator,
    EvalResult,
    ExactMatchEvaluator,
    JsonFieldEvaluator,
    LLMJudgeEvaluator,
    SimilarityEvaluator,
)

__all__ = [
    # Core
    "Project",
    "InferenceAgent",
    "train_val_split",
    # LLM
    "LLMClient",
    "LLMResponse",
    "LLMMessage",
    # Bundles
    "BundleSchema",
    "BundleCollection",
    "ExampleBundle",
    # File loading
    "FileLoader",
    "FileContent",
    "get_default_loader",
    # Storage
    "FileSystemStore",
    "SQLiteStore",
    "PromptVersion",
    # Training
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingReport",
    "IterationResult",
    "PromptOptimizer",
    "BatchStrategy",
    "RandomBatchStrategy",
    "FailurePriorityBatchStrategy",
    "TrainingLog",
    # Interactive refinement
    "InteractiveOptimizer",
    "InteractiveSessionResult",
    # Evaluation
    "Evaluator",
    "EvalResult",
    "ExactMatchEvaluator",
    "JsonFieldEvaluator",
    "LLMJudgeEvaluator",
    "SimilarityEvaluator",
]
