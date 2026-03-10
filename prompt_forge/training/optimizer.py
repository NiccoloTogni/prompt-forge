"""
Prompt Optimizer — the "Prompt Engineering Agent".

This is the heart of the library. It takes:
    - The current prompt
    - A batch of examples (input + expected output)
    - The training log (what was learned before)
    - Optional: evaluation results from testing the current prompt

And produces in a single LLM call:
    - An improved version of the prompt
    - A summary of what was learned
    - A list of issues/gaps that could not be resolved
"""

import json
import logging
import re
from typing import Callable

from .prompt import DEFAULT_META_PROMPT
from ..llm.client import LLMClient, LLMMessage
from ..file_loaders import FileLoader, get_default_loader
from ..bundle import ExampleBundle, is_output_role

# ── Module-level defaults ─────────────────────────────────────────────────────
OPTIMIZER_TEMPERATURE = 1   # Sampling temperature for the optimizer call
TOKEN_CHARS_PER_TOKEN = 4   # Chars-per-token ratio used by the built-in estimator

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """
    The Prompt Engineering Agent.

    Analyzes examples and produces in a single LLM call:
    - An improved system prompt
    - A summary of what was learned this iteration
    - A list of outstanding issues or gaps in the training data
    """

    def __init__(
        self,
        llm: LLMClient,
        meta_prompt: str | None = None,
        file_loader: FileLoader | None = None,
        context: str = "",
        temperature: float = OPTIMIZER_TEMPERATURE,
        token_estimator: Callable[[str], int] | None = None,
    ):
        """
        Args:
            llm: The LLM client to use for optimization.
            meta_prompt: Custom meta-prompt for the optimizer. If None, uses default.
            file_loader: FileLoader for reading example files.
            context: Additional context about the task/domain.
            temperature: Sampling temperature for the optimizer LLM call.
                         See OPTIMIZER_TEMPERATURE for the default value.
            token_estimator: Callable that estimates token count for a string.
                             Defaults to len(text) // TOKEN_CHARS_PER_TOKEN.
                             Provide a model-specific tokenizer for precise limits.
        """
        self.llm = llm
        self.meta_prompt = meta_prompt or DEFAULT_META_PROMPT
        self.file_loader = file_loader or get_default_loader()
        self.context = context
        self.temperature = temperature
        self.token_estimator = token_estimator or (lambda text: len(text) // TOKEN_CHARS_PER_TOKEN)
        self._detected_schema: dict | None = None  # cached after first successful detection

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

        A single LLM call returns the optimized prompt, a learnings summary,
        and a list of outstanding issues — all parsed from a structured XML response.

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
            OptimizerResult with the new prompt, learnings, and issues.
        """
        if output_schema is not None:
            resolved_schema = output_schema
        elif self._detected_schema is not None:
            resolved_schema = self._detected_schema
        else:
            self._detected_schema = self._detect_output_schema(examples)
            resolved_schema = self._detected_schema

        if max_tokens is not None:
            examples = self._fit_examples_to_budget(
                examples=examples,
                current_prompt=current_prompt,
                training_history=training_history,
                eval_feedback=eval_feedback,
                output_schema=resolved_schema,
                max_tokens=max_tokens,
            )

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
        response = self.llm.complete(messages, temperature=self.temperature)

        new_prompt, learnings, issues = self._parse_structured_response(
            response.text, current_prompt
        )

        return OptimizerResult(
            new_prompt=new_prompt,
            learnings=learnings,
            issues=issues,
            output_schema=resolved_schema,
            usage=response.usage,
        )

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_structured_response(
        text: str,
        fallback_prompt: str,
    ) -> tuple[str, str, str]:
        """
        Extract (optimized_prompt, learnings, issues) from the XML-tagged response.

        Falls back gracefully if the model doesn't follow the format exactly:
        - If <optimized_prompt> is missing, the full response is used as the prompt.
        - If <learnings> or <issues> are missing, they default to empty strings.
        """
        def extract(tag: str) -> str:
            match = re.search(
                rf"<{tag}>(.*?)</{tag}>",
                text,
                flags=re.DOTALL,
            )
            return match.group(1).strip() if match else ""

        new_prompt = extract("optimized_prompt")
        if not new_prompt:
            # Model didn't follow the format — treat the whole response as the prompt
            logger.warning(
                "Optimizer response missing <optimized_prompt> tag. "
                "Using full response as prompt. Consider reviewing your meta_prompt."
            )
            new_prompt = text.strip() or fallback_prompt

        learnings = extract("learnings")
        issues = extract("issues")
        if issues.lower() == "none":
            issues = ""

        return new_prompt, learnings, issues

    # ── Token budget ──────────────────────────────────────────────────────────

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

        Raises:
            ValueError: If a single example already exceeds the token budget on its own.

        Returns:
            A (possibly shorter) list of examples that fits in the budget.
            If trimmed, a warning is logged.
        """
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

        for example in examples:
            rendered = self._render_example(example)
            example_tokens = self._estimate_tokens(rendered)
            if example_tokens > remaining_budget:
                raise ValueError(
                    f"Example '{example.bundle_id}' is too large to fit in the context window "
                    f"({fixed_tokens + example_tokens} estimated tokens > max_tokens={max_tokens}). "
                    "Reduce example size or increase max_tokens."
                )

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

    # ── Schema detection ──────────────────────────────────────────────────────

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
                if is_output_role(role):
                    total += 1
                    try:
                        parsed = json.loads(content.text.strip())
                        if isinstance(parsed, dict):
                            json_examples.append(parsed)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break

        if total == 0 or len(json_examples) / total < 0.5:
            return None

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

    # ── Message building ──────────────────────────────────────────────────────

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

        if self.context:
            sections.append(f"<task_context>\n{self.context}\n</task_context>")

        sections.append(f"<current_prompt>\n{current_prompt}\n</current_prompt>")

        if training_history:
            sections.append(f"<training_history>\n{training_history}\n</training_history>")

        if eval_feedback:
            sections.append(f"<evaluation_feedback>\n{eval_feedback}\n</evaluation_feedback>")

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


class OptimizerResult:
    """Result from a single optimization step."""

    def __init__(
        self,
        new_prompt: str,
        learnings: str,
        issues: str = "",
        output_schema: dict | None = None,
        usage: dict[str, int] | None = None,
    ):
        self.new_prompt = new_prompt
        self.learnings = learnings
        self.issues = issues
        self.output_schema = output_schema
        self.usage = usage
