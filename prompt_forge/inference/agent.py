"""
Inference Agent — uses the trained prompt to perform tasks.

This is the production-facing component. It loads a versioned prompt
and uses it to process new inputs.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..llm.client import LLMClient, LLMMessage
from ..file_loaders import FileLoader, get_default_loader
from ..storage.project_store import ProjectStore, FileSystemStore

logger = logging.getLogger(__name__)


class InferenceAgent:
    """
    Uses a trained prompt to perform tasks on new inputs.

    Supports:
        - Loading a specific prompt version or the latest
        - Optional few-shot examples for additional context
        - Custom pre/post processing
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_text: str,
        file_loader: FileLoader | None = None,
        system_suffix: str = "",
        llm_kwargs: dict[str, Any] | None = None,
        output_schema: dict | None = None,
    ):
        """
        Args:
            llm: LLM client to use.
            prompt_text: The system prompt (usually from a trained version).
            file_loader: FileLoader for reading input files.
            system_suffix: Additional text appended to the system prompt.
            llm_kwargs: Additional kwargs for llm.complete().
        """
        self.llm = llm
        self.prompt_text = prompt_text
        self.file_loader = file_loader or get_default_loader()
        self.system_suffix = system_suffix
        self.llm_kwargs = llm_kwargs or {"temperature": 1}
        self.output_schema = output_schema

    @classmethod
    def from_store(
        cls,
        llm: LLMClient,
        store: ProjectStore,
        version: int | None = None,
        **kwargs,
    ) -> InferenceAgent:
        """
        Create an InferenceAgent from a stored prompt version.

        Args:
            llm: LLM client.
            store: Project storage backend.
            version: Specific version number, or None for latest.
            **kwargs: Additional arguments passed to InferenceAgent.__init__.
        """
        if version is not None:
            prompt_version = store.get_prompt_version(version)
            if prompt_version is None:
                raise ValueError(f"Prompt version {version} not found.")
        else:
            prompt_version = store.get_latest_prompt()
            if prompt_version is None:
                raise ValueError("No prompt versions found in store.")

        logger.info(f"Loaded prompt version {prompt_version.version} (score: {prompt_version.eval_score})")

        # Carry output_schema from the stored version unless caller overrides it
        if "output_schema" not in kwargs:
            kwargs["output_schema"] = prompt_version.output_schema

        return cls(
            llm=llm,
            prompt_text=prompt_version.prompt_text,
            **kwargs,
        )

    @classmethod
    def from_project_dir(
        cls,
        llm: LLMClient,
        project_dir: str | Path,
        version: int | None = None,
        **kwargs,
    ) -> InferenceAgent:
        """
        Convenience: create from a filesystem project directory.
        """
        store = FileSystemStore(project_dir)
        return cls.from_store(llm=llm, store=store, version=version, **kwargs)

    def run(
        self,
        input_text: str | None = None,
        input_file: str | Path | None = None,
        input_files: dict[str, str | Path] | None = None,
        extra_context: str = "",
    ) -> str | dict:
        """
        Run inference on a new input.

        Provide ONE of:
            - input_text: Direct text input
            - input_file: Path to a single input file
            - input_files: Dict of role_name → file_path for multi-file inputs

        Returns:
            A dict if output_schema is set (JSON parsed), otherwise a plain string.
        """
        # Build system prompt
        system = self.prompt_text
        if self.output_schema:
            system += "\n\nRespond ONLY with valid JSON. Do not include any explanation or text outside the JSON object."
        if self.system_suffix:
            system += "\n\n" + self.system_suffix

        # Build messages
        messages = [LLMMessage(role="system", content=system)]

        # Build user message
        user_content = self._build_user_content(
            input_text=input_text,
            input_file=input_file,
            input_files=input_files,
            extra_context=extra_context,
        )
        messages.append(LLMMessage(role="user", content=user_content))

        # Call LLM
        response = self.llm.complete(messages, **self.llm_kwargs)

        if self.output_schema is not None:
            return self._extract_json(response.text)
        return response.text

    @staticmethod
    def _extract_json(text: str) -> dict:
        """
        Parse JSON from LLM output, handling markdown code fences.

        Raises:
            ValueError: If the text cannot be parsed as a JSON object.
        """
        stripped = text.strip()
        # Strip ```json ... ``` or ``` ... ``` fences
        fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped)
        if fence_match:
            stripped = fence_match.group(1).strip()
        try:
            result = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM output could not be parsed as JSON: {exc}\nRaw output:\n{text}"
            ) from exc
        if not isinstance(result, dict):
            raise ValueError(
                f"Expected a JSON object but got {type(result).__name__}.\nRaw output:\n{text}"
            )
        return result

    def run_batch(
        self,
        inputs: list[dict],
    ) -> list[str]:
        """
        Run inference on multiple inputs.

        Args:
            inputs: List of dicts, each with keys matching run() parameters
                    (e.g., {"input_file": "path/to/file.pdf"}).

        Returns:
            List of output strings.
        """
        results = []
        for i, input_kwargs in enumerate(inputs):
            logger.info(f"Processing input {i+1}/{len(inputs)}")
            try:
                result = self.run(**input_kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed on input {i+1}: {e}")
                results.append(f"[Error: {e}]")
        return results

    def _build_user_content(
        self,
        input_text: str | None = None,
        input_file: str | Path | None = None,
        input_files: dict[str, str | Path] | None = None,
        extra_context: str = "",
    ) -> str:
        parts = []

        if extra_context:
            parts.append(f"<additional_context>\n{extra_context}\n</additional_context>")

        if input_text:
            parts.append(input_text)
        elif input_file:
            content = self.file_loader.load(input_file)
            parts.append(f"<input>\n{content.text}\n</input>")
        elif input_files:
            for role, path in input_files.items():
                content = self.file_loader.load(path)
                parts.append(f"<{role}>\n{content.text}\n</{role}>")
        else:
            raise ValueError("Provide one of: input_text, input_file, or input_files")

        return "\n\n".join(parts)

    @property
    def prompt_info(self) -> str:
        """Human-readable summary of the loaded prompt."""
        lines = len(self.prompt_text.splitlines())
        chars = len(self.prompt_text)
        schema_info = f", output_schema={list(self.output_schema.get('properties', self.output_schema).keys())}" if self.output_schema else ""
        return f"Prompt: {lines} lines, {chars} chars{schema_info}"
