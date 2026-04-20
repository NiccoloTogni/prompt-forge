"""
Inference Agent — uses the trained prompt to perform tasks.

This is the production-facing component. It loads a versioned prompt
and uses it to process new inputs.

Batch inference:
    run_batch() — production batch over file/text inputs, single LLM call (chunked if needed)
    run_bundle_batch() — training/eval batch over ExampleBundle objects, single LLM call
"""

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..llm.client import LLMClient, LLMMessage, LLMResponse, MessageContent, TextPart, FilePart
from ..file_loaders import FileLoader, get_default_loader
from ..storage.project_store import ProjectStore, FileSystemStore
from ..bundle import ExampleBundle, is_output_role
from .._retry import call_with_retry

logger = logging.getLogger(__name__)

TOKEN_CHARS_PER_TOKEN = 4

# MIME type map used when building FilePart objects
_MEDIA_TYPES: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".webp": "image/webp",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv":  "text/csv",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".json": "application/json",
}


def _infer_media_type(path: Path) -> str | None:
    return _MEDIA_TYPES.get(path.suffix.lower())


def _has_file_parts(content: MessageContent) -> bool:
    return isinstance(content, list) and any(isinstance(p, FilePart) for p in content)


class InferenceAgent:
    """
    Uses a trained prompt to perform tasks on new inputs.

    Supports:
        - Single and batch inference (true single-call, not sequential)
        - Automatic chunking when a token budget is set
        - ExampleBundle-based inference for training/evaluation pipelines
        - Loading a specific prompt version or the latest
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_text: str,
        file_loader: FileLoader | None = None,
        system_suffix: str = "",
        llm_kwargs: dict[str, Any] | None = None,
        output_schema: dict | None = None,
        token_estimator: Callable[[str], int] | None = None,
        native_files: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_workers: int | None = None,
        context_retriever: "Callable[[str, LLMClient], str] | None" = None,
    ):
        """
        Args:
            llm: LLM client to use.
            prompt_text: The system prompt (usually from a trained version).
            file_loader: FileLoader for reading input files (used when native_files=False).
            system_suffix: Additional text appended to the system prompt.
            llm_kwargs: Additional kwargs for llm.complete() (e.g. temperature).
            output_schema: If set, outputs are parsed as JSON dicts.
            token_estimator: Callable for estimating token counts (used for chunking).
                             Defaults to len(text) // TOKEN_CHARS_PER_TOKEN.
            native_files: When True (default), input files are passed directly to the
                          LLM as FilePart content parts. When False AND an explicit
                          file_loader is provided, files are extracted to text instead.
                          If False but no file_loader is given, native mode is used
                          regardless. Requires an LLM client that handles FilePart
                          (e.g. Azure OpenAI Responses API). Batch inference
                          automatically falls back to sequential calls when file parts
                          are present.
            max_retries: Number of times to retry a failed LLM call (default 3).
            retry_delay: Initial wait in seconds before the first retry; doubles
                         each subsequent attempt (exponential backoff).
            max_workers: Maximum number of concurrent LLM calls when the batch
                         falls back to per-input calls (i.e. when file parts are
                         present). None (default) keeps the serial behaviour.
            context_retriever: Optional callable ``(query: str, llm: LLMClient) -> str``
                               called before each inference call to fetch relevant
                               context (e.g. from a vector store or web search).
                               The returned string is injected as
                               ``<retrieved_context>…</retrieved_context>`` at the
                               start of the user message. The query is the best
                               available text representation of the input.
        """
        self.llm = llm
        self.prompt_text = prompt_text
        self.file_loader = file_loader or get_default_loader()
        self.system_suffix = system_suffix
        self.llm_kwargs = llm_kwargs or {}
        self.output_schema = output_schema
        self.token_estimator = token_estimator or (lambda text: len(text) // TOKEN_CHARS_PER_TOKEN)
        # Text extraction requires both an explicit opt-out AND an explicit loader.
        # Without a loader, fall back to native regardless of the flag.
        self.native_files = native_files or (file_loader is None)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_workers = max_workers
        self.context_retriever = context_retriever
        self.tokens_used: int = 0
        self._tokens_lock = threading.Lock()  # guards tokens_used under concurrent calls

    # ── Class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_store(
        cls,
        llm: LLMClient,
        store: ProjectStore,
        version: int | None = None,
        **kwargs,
    ) -> "InferenceAgent":
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

        if "output_schema" not in kwargs:
            kwargs["output_schema"] = prompt_version.output_schema

        return cls(llm=llm, prompt_text=prompt_version.prompt_text, **kwargs)

    @classmethod
    def from_project_dir(
        cls,
        llm: LLMClient,
        project_dir: str | Path,
        version: int | None = None,
        **kwargs,
    ) -> "InferenceAgent":
        """Convenience: create from a filesystem project directory."""
        store = FileSystemStore(project_dir)
        return cls.from_store(llm=llm, store=store, version=version, **kwargs)

    # ── Public inference API ──────────────────────────────────────────────────

    def run(
        self,
        input_text: str | None = None,
        input_file: str | Path | None = None,
        input_files: dict[str, str | Path] | None = None,
        extra_context: str = "",
    ) -> "str | dict":
        """
        Run inference on a new input.

        Provide ONE of:
            - input_text: Direct text input
            - input_file: Path to a single input file
            - input_files: Dict of role_name → file_path for multi-file inputs

        Returns:
            A dict if output_schema is set (JSON parsed), otherwise a plain string.
        """
        retrieved_context = ""
        if self.context_retriever is not None:
            retrieved_context = self._retrieve(self._input_query(input_text, input_file, input_files))
        user_content = self._build_user_content(input_text, input_file, input_files, extra_context, retrieved_context)
        system = self._build_system()
        if self.output_schema:
            system += "\n\nRespond ONLY with valid JSON. Do not include any explanation or text outside the JSON object."
        response = self._call_single(system, user_content)
        return self._post_process(response.text)

    def run_batch(
        self,
        inputs: list[dict],
        max_tokens: int | None = None,
    ) -> list:
        """
        Run inference on multiple inputs in a single LLM call (chunked if max_tokens is set).

        Args:
            inputs: List of dicts, each with keys matching run() parameters
                    (e.g., {"input_file": "path/to/file.pdf"}).
            max_tokens: Token budget per LLM call. If None, all inputs are sent at once.

        Returns:
            list[str] when output_schema is None.
            list[dict] when output_schema is set — each element is a parsed JSON object,
            equivalent to calling run() on each input individually.
        """
        user_contents = [self._build_user_content(**kw) for kw in inputs]
        return self._batch_call(user_contents, max_tokens)

    def run_bundle(self, bundle: ExampleBundle) -> str:
        """
        Run inference on a single ExampleBundle (input roles only, no JSON parsing).

        Intended for training/evaluation pipelines where the evaluator handles
        output parsing. Always returns a plain string with code fences stripped.
        """
        extra_context = self._retrieve(self._bundle_query(bundle)) if self.context_retriever else ""
        user_content = self._bundle_to_user_content(bundle, extra_context=extra_context)
        response = self._call_single(self._build_system(), user_content)
        return self._strip_code_fences(response.text)

    def run_bundle_batch(
        self,
        bundles: list[ExampleBundle],
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Run batch inference on ExampleBundles in a single LLM call (chunked if needed).

        Intended for training/evaluation pipelines. Always returns plain strings
        (code fences stripped) — the evaluator handles output parsing.

        Args:
            bundles: List of ExampleBundle objects.
            max_tokens: Token budget per LLM call. If None, all bundles are sent at once.

        Returns:
            List of output strings in the same order as the input bundles.
        """
        user_contents = [
            self._bundle_to_user_content(
                b,
                extra_context=self._retrieve(self._bundle_query(b)) if self.context_retriever else "",
            )
            for b in bundles
        ]
        return [
            json.dumps(o) if isinstance(o, dict) else str(o)
            for o in self._batch_call(user_contents, max_tokens)
        ]

    # ── Internal: batch call ──────────────────────────────────────────────────

    def _batch_call(self, user_contents: list[MessageContent], max_tokens: int | None) -> list:
        """Single-call (or chunked) batch over pre-built user content.

        Falls back to sequential calls when any content contains FilePart objects,
        since native file inputs cannot be wrapped in the XML batch format.

        Returns list[str] when output_schema is None, list[dict] when output_schema is set.
        """
        if not user_contents:
            return []

        # Native file content: can't XML-wrap binary parts — run per-input, optionally concurrent
        if any(_has_file_parts(c) for c in user_contents):
            logger.info(
                "Batch inference: %d inputs contain native file parts — running %s",
                len(user_contents),
                f"concurrently (max_workers={self.max_workers})" if self.max_workers else "sequentially",
            )
            if self.max_workers:
                return self._concurrent_call(user_contents)
            return [self._post_process(self._call_single(self._build_system(), c).text)
                    for c in user_contents]

        # All-text path — use existing batch / chunked logic.
        # Safe cast: file-part content already handled above; remaining items are str.
        from typing import cast
        text_contents = cast(list[str], user_contents)
        if len(text_contents) == 1:
            response = self._call_single(self._build_system(), text_contents[0])
            return [self._post_process(response.text)]
        if max_tokens is not None:
            return self._chunked_batch(text_contents, max_tokens)
        return self._single_batch_call(text_contents)

    def _concurrent_call(self, user_contents: list[MessageContent]) -> list:
        """Fire one LLM call per input concurrently, preserving input order."""
        system = self._build_system()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._call_single, system, c) for c in user_contents]
            return [self._post_process(f.result().text) for f in futures]

    def _single_batch_call(self, user_contents: list[str]) -> list:
        """Send all inputs in one LLM call; parse a JSON array response.

        Returns a list of strings (no output_schema) or dicts (output_schema set).
        Raises ValueError / json.JSONDecodeError on parse failure or wrong item count,
        which triggers the sequential fallback in the pipeline.
        """
        n = len(user_contents)
        input_blocks = "\n\n".join(
            f'<input id="{i}">\n{content}\n</input>'
            for i, content in enumerate(user_contents, 1)
        )

        if self.output_schema:
            format_instruction = (
                f"Return a JSON array of exactly {n} objects. "
                f"Each object must follow the output schema. "
                f"Respond with ONLY the JSON array, no surrounding text:\n"
                f'[{{"field": "value"}}, {{"field": "value"}}, ...]'
            )
        else:
            format_instruction = (
                f"Return a JSON array of exactly {n} strings, one per input, in order. "
                f"Respond with ONLY the JSON array, no surrounding text:\n"
                f'["output for input 1", "output for input 2", ...]'
            )

        user_message = (
            f"You will receive {n} independent inputs. Process each one according to your "
            f"instructions and produce an output for every input.\n\n"
            f"{format_instruction}\n\n"
            f"{input_blocks}"
        )
        response = self._call_single(self._build_system(), user_message)

        # Strip optional code fence, then parse the JSON array.
        # If the LLM prepends prose before the array, fall back to extracting
        # the first [...] block found anywhere in the response.
        text = response.text.strip()
        logger.debug(f"Batch inference raw response (first 500 chars): {text[:500]!r}")

        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fence_match:
            text = fence_match.group(1).strip()

        if not text.startswith("["):
            array_match = re.search(r"(\[[\s\S]*\])", text)
            if array_match:
                text = array_match.group(1).strip()

        outputs = json.loads(text)  # raises json.JSONDecodeError on failure → triggers fallback

        if not isinstance(outputs, list):
            raise ValueError(f"Batch inference: expected JSON array, got {type(outputs).__name__}")
        if len(outputs) != n:
            raise ValueError(f"Batch inference: expected {n} outputs, got {len(outputs)}")

        return outputs

    def _chunked_batch(self, user_contents: list[str], max_tokens: int) -> list:
        """Group inputs into token-budget chunks and run one batch call per chunk."""
        system_tokens = self.token_estimator(self._build_system())
        instruction_overhead = 300  # conservative estimate for batch wrapper text
        available = max_tokens - system_tokens - instruction_overhead

        if available <= 0:
            raise ValueError(
                f"System prompt alone ({system_tokens} tokens) exceeds "
                f"max_tokens={max_tokens}. Increase the token budget."
            )

        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for content in user_contents:
            tokens = self.token_estimator(content)
            if tokens > available:
                raise ValueError(
                    f"A single input ({tokens} tokens) exceeds the per-chunk budget "
                    f"({available} tokens). Increase max_tokens."
                )
            if current_tokens + tokens > available and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(content)
            current_tokens += tokens

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Batch inference: {len(user_contents)} inputs → {len(chunks)} chunk(s)")
        all_outputs: list = []
        for chunk in chunks:
            all_outputs.extend(self._single_batch_call(chunk))
        return all_outputs

    # ── Internal: helpers ─────────────────────────────────────────────────────

    def _call_single(self, system: str, user_content: MessageContent) -> LLMResponse:
        """Make a single LLM call (with retry) and accumulate token usage."""
        messages = [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user_content),
        ]
        response = call_with_retry(
            lambda: self.llm.complete(messages, **self.llm_kwargs),
            max_retries=self.max_retries,
            delay=self.retry_delay,
        )
        if response.usage:
            with self._tokens_lock:
                self.tokens_used += (
                    response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)
                )
        return response

    def _build_system(self) -> str:
        system = self.prompt_text
        if self.system_suffix:
            system += "\n\n" + self.system_suffix
        return system

    def _bundle_to_user_content(self, bundle: ExampleBundle, extra_context: str = "") -> MessageContent:
        if not self.native_files:
            sections = []
            if extra_context:
                sections.append(f"<retrieved_context>\n{extra_context}\n</retrieved_context>")
            contents = bundle.load_contents(self.file_loader)
            sections.extend(
                f"<{role}>\n{content.text}\n</{role}>"
                for role, content in contents.items()
                if not is_output_role(role)
            )
            return "\n\n".join(sections)
        parts: list[TextPart | FilePart] = []
        if extra_context:
            parts.append(TextPart(text=f"<retrieved_context>\n{extra_context}\n</retrieved_context>"))
        for role, raw_path in bundle.files.items():
            if is_output_role(role):
                continue
            parts.append(TextPart(text=f"<{role}>"))
            if isinstance(raw_path, list):
                for rp in raw_path:
                    p = Path(rp)
                    parts.append(FilePart(path=p, media_type=_infer_media_type(p)))
            else:
                p = Path(raw_path)  # normalise: may be str or Path
                parts.append(FilePart(path=p, media_type=_infer_media_type(p)))
            parts.append(TextPart(text=f"</{role}>"))
        return parts

    def _retrieve(self, query: str) -> str:
        """Call the context_retriever and return retrieved context (empty string on failure)."""
        if not self.context_retriever or not query:
            return ""
        try:
            return self.context_retriever(query, self.llm) or ""
        except Exception as exc:
            logger.warning("context_retriever raised an exception: %s", exc)
            return ""

    def _bundle_query(self, bundle: ExampleBundle) -> str:
        """Extract a text query from the input roles of a bundle for context retrieval."""
        parts = []
        for role, raw_path in bundle.files.items():
            if is_output_role(role):
                continue
            paths = raw_path if isinstance(raw_path, list) else [raw_path]
            for p in paths:
                try:
                    parts.append(self.file_loader.load(p).text)
                except Exception:
                    parts.append(str(p))
        return "\n".join(parts)

    @staticmethod
    def _input_query(
        input_text: str | None,
        input_file: "str | Path | None",
        input_files: "dict[str, str | Path] | None",
    ) -> str:
        """Derive a retrieval query from run() arguments."""
        if input_text:
            return input_text
        if input_file:
            return str(input_file)
        if input_files:
            return " ".join(str(p) for p in input_files.values())
        return ""

    def _post_process(self, text: str) -> "str | dict":
        """Parse JSON if output_schema is set; otherwise strip code fences."""
        if self.output_schema is not None:
            return self._extract_json(text)
        return self._strip_code_fences(text)

    def _build_user_content(
        self,
        input_text: str | None = None,
        input_file: str | Path | None = None,
        input_files: dict[str, str | Path] | None = None,
        extra_context: str = "",
        retrieved_context: str = "",
    ) -> MessageContent:
        if not self.native_files:
            # Text-extraction path (original behaviour)
            parts: list[str] = []
            if retrieved_context:
                parts.append(f"<retrieved_context>\n{retrieved_context}\n</retrieved_context>")
            if extra_context:
                parts.append(f"<additional_context>\n{extra_context}\n</additional_context>")
            if input_text is not None:
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

        # Native-file path — pass files directly as content parts
        multimodal: list[TextPart | FilePart] = []
        if retrieved_context:
            multimodal.append(TextPart(text=f"<retrieved_context>\n{retrieved_context}\n</retrieved_context>"))
        if extra_context:
            multimodal.append(TextPart(text=f"<additional_context>\n{extra_context}\n</additional_context>"))
        if input_text is not None:
            multimodal.append(TextPart(text=input_text))
        elif input_file:
            p = Path(input_file)
            multimodal.append(FilePart(path=p, media_type=_infer_media_type(p)))
        elif input_files:
            for role, path in input_files.items():
                p = Path(path)
                multimodal.append(TextPart(text=f"<{role}>"))
                multimodal.append(FilePart(path=p, media_type=_infer_media_type(p)))
        else:
            raise ValueError("Provide one of: input_text, input_file, or input_files")
        return multimodal

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        match = re.search(r"```(?:\w+)?\s*\n?([\s\S]*?)\s*```", stripped)
        return match.group(1).strip() if match else stripped

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse JSON from LLM output, handling markdown code fences."""
        stripped = text.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
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

    @property
    def prompt_info(self) -> str:
        """Human-readable summary of the loaded prompt."""
        lines = len(self.prompt_text.splitlines())
        chars = len(self.prompt_text)
        if self.output_schema:
            keys = list(self.output_schema.get("properties", self.output_schema).keys())
            schema_info = f", output_schema={keys}"
        else:
            schema_info = ""
        return f"Prompt: {lines} lines, {chars} chars{schema_info}"
