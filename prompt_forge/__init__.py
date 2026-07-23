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

from __future__ import annotations

__version__ = "0.1.0"

from .project import Project
from .utils import train_val_split, train_val_test_split
from .inference.agent import InferenceAgent
from .bundle import BundleSchema, BundleCollection, ExampleBundle
from .llm.client import LLMClient, LLMResponse, LLMMessage, TextPart, FilePart, MessageContent
from .file_loaders import FileLoader, FileContent, get_default_loader
from .storage.project_store import (
    FileSystemStore,
    SQLAlchemyStore,
    SQLiteStore,
    PromptVersion,
)
from .training.pipeline import TrainingPipeline, TrainingConfig, TrainingReport, IterationResult
from .training.optimizer import PromptOptimizer
from .training.prompt import DEFAULT_OPTIMIZER_PROMPT, DEFAULT_CONSOLIDATION_PROMPT
from .training.batch_strategy import (
    BatchStrategy,
    RandomBatchStrategy,
    FailurePriorityBatchStrategy,
)
from .training.training_log import TrainingLog
from .caching import CachedLLM
from .retrievers import WebSearchRetriever
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
    "CachedLLM",
    "WebSearchRetriever",
    "train_val_split",
    "train_val_test_split",
    # LLM
    "LLMClient",
    "LLMResponse",
    "LLMMessage",
    "TextPart",
    "FilePart",
    "MessageContent",
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
    "SQLAlchemyStore",
    "SQLiteStore",  # deprecated
    "PromptVersion",
    # Training
    "TrainingPipeline",
    "TrainingConfig",
    "TrainingReport",
    "IterationResult",
    "PromptOptimizer",
    "DEFAULT_OPTIMIZER_PROMPT",
    "DEFAULT_CONSOLIDATION_PROMPT",
    "BatchStrategy",
    "RandomBatchStrategy",
    "FailurePriorityBatchStrategy",
    "TrainingLog",
    # Evaluation
    "Evaluator",
    "EvalResult",
    "ExactMatchEvaluator",
    "JsonFieldEvaluator",
    "LLMJudgeEvaluator",
    "SimilarityEvaluator",
]
