"""Interactive refinement page — chat-based human-in-the-loop prompt editing."""

import streamlit as st

from prompt_forge.app.state import require_project, require_llm
from prompt_forge.interactive.optimizer import DEFAULT_INTERACTIVE_META_PROMPT


def show() -> None:
    st.header("Interactive Refine")
    project = require_project()
    llm = require_llm()

    if project.get_prompt() is None:
        st.warning("No prompt versions found. Run training or set a seed prompt first.")
        return

    # ── Version selector ───────────────────────────────────────────────────────
    versions = project.list_versions()
    version_labels = [f"v{v.version}" + (" (latest)" if i == len(versions) - 1 else "")
                      for i, v in enumerate(versions)]
    selected_idx = st.selectbox("Start from version", range(len(versions)),
                                format_func=lambda i: version_labels[i],
                                index=len(versions) - 1)
    selected_version = versions[selected_idx]

    # Initialize session if switching versions
    current_key = f"refine_version_{selected_version.version}"
    if st.session_state.get("refine_version_key") != current_key:
        st.session_state.refine_version_key = current_key
        st.session_state.refine_prompt = selected_version.prompt_text
        st.session_state.refine_messages = []

    current_prompt: str = st.session_state.refine_prompt

    # ── Current prompt display ─────────────────────────────────────────────────
    with st.expander("Current prompt", expanded=False):
        st.code(current_prompt, language="text")

    col_save, col_reset = st.columns([1, 1])
    if col_save.button("Save as new version"):
        _save_version(project, current_prompt)

    if col_reset.button("Reset to selected version"):
        st.session_state.refine_prompt = selected_version.prompt_text
        st.session_state.refine_messages = []
        st.rerun()

    st.divider()

    # ── Chat history ───────────────────────────────────────────────────────────
    for msg in st.session_state.get("refine_messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ─────────────────────────────────────────────────────────────
    feedback = st.chat_input("Give feedback to refine the prompt...")
    if feedback:
        st.session_state.refine_messages.append({"role": "user", "content": feedback})
        with st.chat_message("user"):
            st.markdown(feedback)

        with st.chat_message("assistant"):
            with st.spinner("Revising prompt..."):
                new_prompt = _revise(llm, current_prompt, feedback)

            st.session_state.refine_prompt = new_prompt

            diff_summary = (
                f"Prompt updated ({len(new_prompt)} chars). "
                "Expand **Current prompt** above to review, or save it as a new version."
            )
            st.markdown(diff_summary)
            st.session_state.refine_messages.append(
                {"role": "assistant", "content": diff_summary}
            )
        st.rerun()


def _revise(llm, current_prompt: str, feedback: str) -> str:
    from prompt_forge import LLMMessage

    messages = [
        LLMMessage(role="system", content=DEFAULT_INTERACTIVE_META_PROMPT),
        LLMMessage(
            role="user",
            content=(
                f"Current prompt:\n<prompt>\n{current_prompt}\n</prompt>\n\n"
                f"Human feedback: {feedback}\n\n"
                "Please revise the prompt based on this feedback. "
                "Return only the revised prompt text, nothing else."
            ),
        ),
    ]
    response = llm.complete(messages, temperature=0.7)
    return response.text.strip()


def _save_version(project, prompt_text: str) -> None:
    from datetime import datetime, timezone
    from prompt_forge.storage.project_store import PromptVersion

    latest = project.get_prompt()
    new_version_num = (latest.version + 1) if latest else 1
    version = PromptVersion(
        version=new_version_num,
        prompt_text=prompt_text,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_version=latest.version if latest else None,
        training_log_entry="Saved from interactive refinement session",
        output_schema=project._output_schema,
    )
    project.store.save_prompt_version(version)
    st.success(f"Saved as version v{new_version_num}.")
