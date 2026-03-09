"""
Interactive Optimizer — human-in-the-loop prompt refinement.

This is a post-training refinement tool. After automated training converges
(or when a human wants to steer the prompt directly), an interactive session
lets the user give natural-language feedback and optionally test the prompt
on real examples.

Usage (CLI):
    optimizer = InteractiveOptimizer(llm=my_llm, store=project.store, bundles=project.bundles)
    result = optimizer.run_session(prompt_text=project.get_prompt().prompt_text)

Usage (Jupyter / custom UI):
    result = optimizer.run_session(
        prompt_text=...,
        input_fn=my_widget.get_input,
        output_fn=my_widget.display,
    )
"""

import dataclasses
import logging
import random
from datetime import datetime, timezone
from typing import Any, Callable

from ..llm.client import LLMClient, LLMMessage
from ..file_loaders import FileLoader, get_default_loader
from ..bundle import BundleCollection, ExampleBundle, is_output_role
from ..storage.project_store import ProjectStore, PromptVersion

logger = logging.getLogger(__name__)


DEFAULT_INTERACTIVE_META_PROMPT = """\
You are an expert Prompt Engineer working interactively with a human.

You will be given:
1. The current version of a system prompt
2. Direct feedback from the human describing what is wrong or what to improve
3. Optionally, a test input/output pair that illustrates a failure

Your job is to produce an improved version of the prompt that addresses the
human's feedback precisely.

CRITICAL RULES:
- Preserve ALL existing rules in the current prompt unless the feedback
  explicitly says a rule is wrong.
- Only ADD or REFINE based on the feedback — do not remove things not mentioned.
- Be specific: translate vague feedback into concrete, unambiguous rules.
- If a test output is provided, diagnose the root cause before fixing.

Respond with ONLY the improved system prompt text, nothing else.
Do not wrap it in markdown code fences or add any preamble."""


@dataclasses.dataclass
class InteractiveSessionResult:
    """Result of a completed interactive refinement session."""

    final_prompt: str
    saved_versions: list[int]   # version numbers saved to the store during this session
    num_revisions: int          # how many times the prompt was revised
    session_log: list[dict]     # [{type, content, timestamp}] full exchange history


class InteractiveOptimizer:
    """
    Human-in-the-loop prompt refinement.

    Runs an interactive session where a human can give feedback on the current
    prompt, test it on examples, and save improved versions — all without
    needing to run a full training loop.

    This is intended as a post-training refinement tool, but can also be used
    standalone without any training data.
    """

    def __init__(
        self,
        llm: LLMClient,
        store: ProjectStore | None = None,
        file_loader: FileLoader | None = None,
        bundles: BundleCollection | None = None,
        context: str = "",
        meta_prompt: str | None = None,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        """
        Args:
            llm: LLM client used for revisions.
            store: Storage backend. Required for the "save" command.
            file_loader: File loader for reading example files.
            bundles: Training/evaluation examples. Required for the "test" command.
            context: Domain context shown to the revision LLM.
            meta_prompt: Custom meta-prompt for the revision agent.
                         Defaults to DEFAULT_INTERACTIVE_META_PROMPT.
            llm_kwargs: Additional kwargs passed to llm.complete().
        """
        self.llm = llm
        self.store = store
        self.file_loader = file_loader or get_default_loader()
        self.bundles = bundles
        self.context = context
        self.meta_prompt = meta_prompt or DEFAULT_INTERACTIVE_META_PROMPT
        self.llm_kwargs = llm_kwargs or {"temperature": 0.3}

    # ── Public API ────────────────────────────────────────────────────

    def run_session(
        self,
        prompt_text: str,
        input_fn: Callable[[], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> InteractiveSessionResult:
        """
        Start an interactive refinement session.

        Args:
            prompt_text: The starting prompt (e.g. the latest trained version).
            input_fn: Callable that reads a line of input from the human.
                      Defaults to the built-in input().
            output_fn: Callable that displays a message to the human.
                       Defaults to print().

        Returns:
            InteractiveSessionResult with the final prompt and session history.
        """
        current_prompt = prompt_text
        saved_versions: list[int] = []
        num_revisions = 0
        session_log: list[dict] = []

        def log(type_: str, content: str) -> None:
            session_log.append({
                "type": type_,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        def say(msg: str) -> None:
            output_fn(msg)
            log("system", msg)

        # Welcome banner
        say(self._banner(current_prompt))

        # Main loop
        while True:
            try:
                raw = input_fn()
            except (EOFError, KeyboardInterrupt):
                say("\nSession interrupted.")
                break

            cmd = raw.strip()
            if not cmd:
                continue

            log("human", cmd)

            # Exit commands
            if cmd.lower() in ("done", "quit", "exit"):
                say("Session complete.")
                break

            # Show current prompt
            if cmd.lower() == "show":
                say(f"\n{'─'*60}\n{current_prompt}\n{'─'*60}")
                continue

            # Save command
            if cmd.lower() == "save":
                if self.store is None:
                    say("[!] No store configured — cannot save.")
                    continue
                version_num = self._save_version(current_prompt)
                saved_versions.append(version_num)
                say(f"[✓] Saved as version {version_num}.")
                log("system", f"Saved version {version_num}")
                continue

            # Test command: "test" or "test <bundle_id>"
            if cmd.lower() == "test" or cmd.lower().startswith("test "):
                bundle_id = cmd[5:].strip() if cmd.lower().startswith("test ") else None
                feedback = self._handle_test(bundle_id, current_prompt, say, input_fn)
                if feedback:
                    log("human", feedback)
                    say("[...] Revising prompt based on test feedback...")
                    current_prompt = self._revise(current_prompt, feedback)
                    num_revisions += 1
                    say(self._revision_summary(current_prompt))
                continue

            # Free-text feedback → revise
            say("[...] Revising prompt...")
            current_prompt = self._revise(current_prompt, cmd)
            num_revisions += 1
            say(self._revision_summary(current_prompt))

        return InteractiveSessionResult(
            final_prompt=current_prompt,
            saved_versions=saved_versions,
            num_revisions=num_revisions,
            session_log=session_log,
        )

    # ── Internal ──────────────────────────────────────────────────────

    def _revise(self, current_prompt: str, feedback: str, test_output: str = "") -> str:
        """Ask the LLM to revise the prompt based on human feedback."""
        user_parts = []

        if self.context:
            user_parts.append(f"<task_context>\n{self.context}\n</task_context>")
        user_parts.append(f"<current_prompt>\n{current_prompt}\n</current_prompt>")
        user_parts.append(f"<human_feedback>\n{feedback}\n</human_feedback>")

        if test_output:
            user_parts.append(
                f"<test_output_that_was_wrong>\n{test_output}\n</test_output_that_was_wrong>"
            )

        messages = [
            LLMMessage(role="system", content=self.meta_prompt),
            LLMMessage(role="user", content="\n\n".join(user_parts)),
        ]

        response = self.llm.complete(messages, **self.llm_kwargs)
        return response.text.strip()

    def _handle_test(
        self,
        bundle_id: str | None,
        current_prompt: str,
        say: Callable[[str], None],
        input_fn: Callable[[], str],
    ) -> str:
        """Run on an example, display result, and return optional feedback."""
        if self.bundles is None or len(self.bundles) == 0:
            say("[!] No examples configured — cannot run test.")
            return ""

        bundle = self._resolve_bundle(bundle_id)
        if bundle is None:
            say(f"[!] Example '{bundle_id}' not found.")
            return ""

        say(f"\n[Testing on example: {bundle.bundle_id}]")

        # Show input
        try:
            contents = bundle.load_contents(self.file_loader)
        except Exception as e:
            say(f"[!] Could not load example: {e}")
            return ""

        for role, content in contents.items():
            if not is_output_role(role):
                truncated = content.text[:800] + ("..." if len(content.text) > 800 else "")
                say(f"\n<{role}>\n{truncated}\n</{role}>")

        # Run inference
        say("\n[Running inference...]")
        try:
            actual = self._run_inference(current_prompt, bundle)
        except Exception as e:
            say(f"[!] Inference failed: {e}")
            return ""

        say(f"\n<output>\n{actual}\n</output>")

        # Ask for feedback
        say("\nFeedback on this output (press Enter to skip): ")
        try:
            feedback = input_fn().strip()
        except (EOFError, KeyboardInterrupt):
            return ""

        return feedback

    def _run_inference(self, prompt_text: str, bundle: ExampleBundle) -> str:
        """Run the LLM with prompt_text on the bundle's input role(s)."""
        contents = bundle.load_contents(self.file_loader)
        input_parts = []
        for role, content in contents.items():
            if not is_output_role(role):
                input_parts.append(f"<{role}>\n{content.text}\n</{role}>")

        messages = [
            LLMMessage(role="system", content=prompt_text),
            LLMMessage(role="user", content="\n\n".join(input_parts)),
        ]
        response = self.llm.complete(messages)
        return response.text

    def _resolve_bundle(self, bundle_id: str | None) -> ExampleBundle | None:
        """Return a specific bundle by ID, or a random one if ID is None."""
        all_bundles = self.bundles.bundles
        if not all_bundles:
            return None
        if bundle_id is None:
            return random.choice(all_bundles)
        for b in all_bundles:
            if b.bundle_id == bundle_id:
                return b
        return None

    def _save_version(self, prompt_text: str) -> int:
        """Save the current prompt as a new version in the store."""
        latest = self.store.get_latest_prompt()
        new_version_num = (latest.version + 1) if latest else 1
        version = PromptVersion(
            version=new_version_num,
            prompt_text=prompt_text,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_version=latest.version if latest else None,
            training_log_entry="Saved during interactive refinement session",
            metadata={"source": "interactive_session"},
        )
        self.store.save_prompt_version(version)
        logger.info(f"Saved interactive refinement as version {new_version_num}")
        return new_version_num

    def _banner(self, current_prompt: str) -> str:
        lines = len(current_prompt.splitlines())
        chars = len(current_prompt)
        return (
            "\n=== Interactive Prompt Refinement ===\n"
            f"Current prompt: {lines} lines, {chars} chars\n"
            "\nCommands:\n"
            "  <feedback>       Describe what to improve — the prompt will be revised\n"
            "  test             Test on a random example\n"
            "  test <id>        Test on a specific example by ID\n"
            "  show             Display the full current prompt\n"
            "  save             Save the current prompt as a new version\n"
            "  done             Finish the session\n"
        )

    def _revision_summary(self, new_prompt: str) -> str:
        lines = len(new_prompt.splitlines())
        chars = len(new_prompt)
        return f"[✓] Prompt revised ({lines} lines, {chars} chars). Type 'show' to view it."
