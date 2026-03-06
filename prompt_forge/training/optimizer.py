"""
Prompt Optimizer — the "Prompt Engineering Agent".

This is the heart of the library. It takes:
    - The current prompt
    - A batch of examples (input + expected output)
    - The training log (what was learned before)
    - Optional: evaluation results from testing the current prompt

And produces an improved version of the prompt.
"""

import json
import logging
from typing import Any, Callable

from .prompt import DEFAULT_META_PROMPT
from ..llm.client import LLMClient, LLMMessage
from ..file_loaders import FileLoader, get_default_loader
from ..bundle import ExampleBundle

MAX_PROMPT_LENGTH = 5000  # rough token estimate to prevent OOM errors
logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    The Prompt Engineering Agent.

    Analyzes examples and produces improved system prompts.
    """

    def __init__(
        self,
        llm: LLMClient,
        meta_prompt: str | None = None,
        file_loader: FileLoader | None = None,
        context: str = "",
        llm_kwargs: dict[str, Any] | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """
        Args:
            llm: The LLM client to use for optimization.
            meta_prompt: Custom meta-prompt for the optimizer. If None, uses default.
            file_loader: FileLoader for reading example files.
            context: Additional context about the task/domain.
            llm_kwargs: Additional kwargs to pass to llm.complete() (e.g., temperature).
            token_estimator: Callable that estimates token count for a string.
                             Defaults to len(text) // 4 (≈4 chars per token).
                             Provide a model-specific tokenizer for precise limits.
        """
        self.llm = llm
        self.meta_prompt = meta_prompt or DEFAULT_META_PROMPT
        self.file_loader = file_loader or get_default_loader()
        self.context = context
        self.llm_kwargs = llm_kwargs or {"temperature": 0.2}
        self.token_estimator = token_estimator or (lambda text: len(text) // 4)

    def optimize(
        self,
        current_prompt: str,
        examples: list[ExampleBundle],
        training_history: str = "",
        eval_feedback: str = "",
        output_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> "OptimizerResult":
        """
        Produce an improved prompt based on examples and feedback.

        Args:
            current_prompt: The current system prompt to improve.
            examples: Batch of examples to learn from.
            training_history: Summary of previous training iterations.
            eval_feedback: Feedback from evaluating the current prompt.
            output_schema: JSON schema for structured output tasks.
            max_tokens: Maximum total tokens (system + user) allowed for the
                        optimizer call. If any single example exceeds the budget
                        on its own, a ValueError is raised. If the full batch
                        exceeds the budget, it is trimmed and a warning is logged.

        Returns:
            OptimizerResult with the new prompt and learnings summary.
        """
        # Resolve schema: use explicit if given, otherwise auto-detect from examples
        resolved_schema = output_schema or self._detect_output_schema(examples)

        # Enforce context-window budget before building the final message
        if max_tokens is not None:
            examples = self._fit_examples_to_budget(
                examples=examples,
                current_prompt=current_prompt,
                training_history=training_history,
                eval_feedback=eval_feedback,
                output_schema=resolved_schema,
                max_tokens=max_tokens,
            )

        # Build the user message with all context
        user_content = self._build_user_message(
            current_prompt=current_prompt,
            examples=examples,
            training_history=training_history,
            eval_feedback=eval_feedback,
            output_schema=resolved_schema,
        )

        messages = [
            LLMMessage(role="system", content=self.meta_prompt),
            LLMMessage(role="user", content=user_content),
        ]

        logger.info(f"Optimizing prompt with {len(examples)} examples...")
        response = self.llm.complete(messages, **self.llm_kwargs)
        new_prompt = response.text.strip()

        # Summarize what was learned (second LLM call)
        learnings, learnings_usage = self._extract_learnings(current_prompt, new_prompt, examples)

        # Combine token usage from both calls
        combined_usage = self._sum_usage(response.usage, learnings_usage)

        return OptimizerResult(
            new_prompt=new_prompt,
            learnings=learnings,
            output_schema=resolved_schema,
            usage=combined_usage,
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate the token count for a string using the configured estimator."""
        return self.token_estimator(text)

    def _render_example(self, example: ExampleBundle) -> str:
        """Render a single example bundle to its string representation."""
        parts = [f"\n--- Example (ID: {example.bundle_id}) ---"]
        try:
            contents = example.load_contents(self.file_loader)
            for role, content in contents.items():
                parts.append(f"\n<{role}>\n{content.text}\n</{role}>")
        except Exception as e:
            parts.append(f"\n[Error loading example {example.bundle_id}: {e}]")
            logger.warning(f"Failed to load example {example.bundle_id}: {e}")
        return "\n".join(parts)

    def _fit_examples_to_budget(
        self,
        examples: list[ExampleBundle],
        current_prompt: str,
        training_history: str,
        eval_feedback: str,
        output_schema: dict | None,
        max_tokens: int,
    ) -> list[ExampleBundle]:
        """
        Trim the example batch to fit within the token budget.

        The total token count is estimated as:
            tokens(meta_prompt) + tokens(user_message_overhead) + tokens(examples)

        Raises:
            ValueError: If a single example already exceeds the token budget on its own.

        Returns:
            A (possibly shorter) list of examples that fits in the budget.
            if trimmed, a warning is logged.
        """
        # Build the overhead portion of the user message (everything except examples body)
        overhead_parts = []
        if self.context:
            overhead_parts.append(f"<task_context>\n{self.context}\n</task_context>")
        overhead_parts.append(f"<current_prompt>\n{current_prompt}\n</current_prompt>")
        if training_history:
            overhead_parts.append(f"<training_history>\n{training_history}\n</training_history>")
        if eval_feedback:
            overhead_parts.append(f"<evaluation_feedback>\n{eval_feedback}\n</evaluation_feedback>")
        if output_schema:
            overhead_parts.append(
                f"<structured_output_requirement>\n{json.dumps(output_schema, indent=2)}\n"
                "</structured_output_requirement>"
            )
        overhead_parts.append("<examples>")
        overhead_parts.append("</examples>")
        overhead_text = "\n\n".join(overhead_parts)

        fixed_tokens = (
            self._estimate_tokens(self.meta_prompt)
            + self._estimate_tokens(overhead_text)
        )
        remaining_budget = max_tokens - fixed_tokens

        if remaining_budget <= 0:
            raise ValueError(
                f"Token budget exhausted by the fixed overhead alone "
                f"({fixed_tokens} tokens > max_tokens={max_tokens}). "
                "Reduce context, training_history, or increase max_tokens."
            )

        # Check each example individually first — fail fast on oversized examples
        for example in examples:
            rendered = self._render_example(example)
            example_tokens = self._estimate_tokens(rendered)
            if example_tokens > remaining_budget:
                raise ValueError(
                    f"Example '{example.bundle_id}' is too large to fit in the context window "
                    f"({fixed_tokens + example_tokens} estimated tokens > max_tokens={max_tokens}). "
                    "Reduce example size or increase max_tokens."
                )

        # Greedily fill the batch
        fitted: list[ExampleBundle] = []
        used_tokens = 0
        for example in examples:
            rendered = self._render_example(example)
            example_tokens = self._estimate_tokens(rendered)
            if used_tokens + example_tokens > remaining_budget:
                logger.warning(
                    f"Batch trimmed from {len(examples)} to {len(fitted)} examples to fit within "
                    f"max_tokens={max_tokens} (estimated total: "
                    f"{fixed_tokens + used_tokens} tokens). "
                    "Consider reducing batch_size or increasing max_tokens."
                )
                break
            fitted.append(example)
            used_tokens += example_tokens

        return fitted

    def _detect_output_schema(self, examples: list[ExampleBundle]) -> dict | None:
        """
        Auto-detect if the expected outputs are JSON.

        Samples expected-output roles across examples. If ≥50% parse as JSON
        objects, returns a lightweight inferred schema with top-level keys.
        """
        json_examples: list[dict] = []
        total = 0

        for example in examples:
            try:
                contents = example.load_contents(self.file_loader)
            except Exception:
                continue

            for role, content in contents.items():
                if "expected" in role.lower() or "output" in role.lower():
                    total += 1
                    try:
                        parsed = json.loads(content.text.strip())
                        if isinstance(parsed, dict):
                            json_examples.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break  # only check the first matching role per example

        if total == 0 or len(json_examples) / total < 0.5:
            return None

        # Infer lightweight schema from union of top-level keys
        all_keys: dict[str, str] = {}
        for example in json_examples:
            for key, value in example.items():
                if key not in all_keys:
                    if isinstance(value, bool):
                        all_keys[key] = "boolean"
                    elif isinstance(value, int | float):
                        all_keys[key] = "number"
                    elif isinstance(value, list):
                        all_keys[key] = "array"
                    elif isinstance(value, dict):
                        all_keys[key] = "object"
                    else:
                        all_keys[key] = "string"

        return {"type": "object", "properties": all_keys}

    def _build_user_message(
        self,
        current_prompt: str,
        examples: list[ExampleBundle],
        training_history: str,
        eval_feedback: str,
        output_schema: dict | None = None,
    ) -> str:
        """Assemble the full context for the optimizer."""
        sections = []

        # Task context
        if self.context:
            sections.append(f"<task_context>\n{self.context}\n</task_context>")

        # Current prompt
        sections.append(f"<current_prompt>\n{current_prompt}\n</current_prompt>")

        # Training history
        if training_history:
            sections.append(f"<training_history>\n{training_history}\n</training_history>")

        # Evaluation feedback
        if eval_feedback:
            sections.append(f"<evaluation_feedback>\n{eval_feedback}\n</evaluation_feedback>")

        # Structured output requirement
        if output_schema:
            sections.append(
                "<structured_output_requirement>\n"
                "The task requires the AI agent to output valid JSON.\n"
                "Schema / expected structure:\n"
                f"{json.dumps(output_schema, indent=2)}\n\n"
                "IMPORTANT: The improved prompt you generate MUST:\n"
                "1. Explicitly instruct the agent to respond ONLY with valid JSON.\n"
                "2. Include the exact field names and their meanings.\n"
                "3. Specify how to handle missing or uncertain values (e.g., null).\n"
                "4. Prohibit any explanation text outside the JSON.\n"
                "</structured_output_requirement>"
            )

        # Examples
        sections.append("<examples>")
        for i, example in enumerate(examples, 1):
            sections.append(f"\n--- Example {i} (ID: {example.bundle_id}) ---")
            try:
                contents = example.load_contents(self.file_loader)
                for role, content in contents.items():
                    sections.append(f"\n<{role}>\n{content.text}\n</{role}>")
            except Exception as e:
                sections.append(f"\n[Error loading example {example.bundle_id}: {e}]")
                logger.warning(f"Failed to load example {example.bundle_id}: {e}")
        sections.append("\n</examples>")

        return "\n\n".join(sections)

    def _extract_learnings(
        self,
        old_prompt: str,
        new_prompt: str,
        examples: list[ExampleBundle],
    ) -> tuple[str, dict[str, int] | None]:
        """Ask the LLM to summarize what was learned in this iteration.

        Returns:
            (learnings_text, usage_dict)
        """
        messages = [
            LLMMessage(role="system", content=(
                "You are summarizing what was learned from a prompt optimization iteration. "
                "Be concise but specific. Focus on new rules, patterns, and edge cases discovered."
            )),
            LLMMessage(role="user", content=(
                f"The prompt was updated after analyzing {len(examples)} examples.\n\n"
                f"<old_prompt>\n{old_prompt[:MAX_PROMPT_LENGTH]}{'...' if len(old_prompt) > MAX_PROMPT_LENGTH else ''}\n</old_prompt>\n\n"
                f"<new_prompt>\n{new_prompt[:MAX_PROMPT_LENGTH]}{'...' if len(new_prompt) > MAX_PROMPT_LENGTH else ''}\n</new_prompt>\n\n"
                "Summarize the key changes and new learnings in 2-5 bullet points. "
                "Focus on WHAT was learned, not how the prompt changed syntactically."
            )),
        ]

        response = self.llm.complete(messages, temperature=0.0)
        return response.text.strip(), response.usage

    @staticmethod
    def _sum_usage(
        a: dict[str, int] | None,
        b: dict[str, int] | None,
    ) -> dict[str, int] | None:
        """Sum two usage dicts, returning None only if both are None."""
        if a is None and b is None:
            return None
        result: dict[str, int] = {}
        for key in ("input_tokens", "output_tokens"):
            result[key] = (a or {}).get(key, 0) + (b or {}).get(key, 0)
        return result


class OptimizerResult:
    """Result from a single optimization step."""

    def __init__(
        self,
        new_prompt: str,
        learnings: str,
        output_schema: dict | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.new_prompt = new_prompt
        self.learnings = learnings
        self.output_schema = output_schema
        self.usage = usage
