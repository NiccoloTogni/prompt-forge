"""Settings page — Azure OpenAI credentials and global config."""

from __future__ import annotations

import streamlit as st

from prompt_forge.app.state import load_app_config, save_app_config


def show() -> None:
    st.header("Settings")

    cfg = load_app_config()

    st.subheader("Azure OpenAI")
    with st.form("settings_form"):
        endpoint = st.text_input(
            "Endpoint URL",
            value=cfg.get("azure_endpoint", ""),
            placeholder="https://your-resource.cognitiveservices.azure.com/",
        )
        api_key = st.text_input(
            "API Key",
            value=cfg.get("azure_api_key", ""),
            type="password",
        )
        deployment = st.text_input(
            "Deployment name",
            value=cfg.get("azure_deployment", "gpt-4o"),
        )
        api_version = st.text_input(
            "API version",
            value=cfg.get("azure_api_version", "2024-12-01-preview"),
        )

        st.divider()
        st.subheader("Workspace")
        projects_dir = st.text_input(
            "Projects directory",
            value=cfg.get("projects_dir", "./projects"),
            help="Root folder where project data will be stored.",
        )

        submitted = st.form_submit_button("Save", type="primary")

    if submitted:
        new_cfg = {
            "azure_endpoint": endpoint.strip(),
            "azure_api_key": api_key.strip(),
            "azure_deployment": deployment.strip(),
            "azure_api_version": api_version.strip(),
            "projects_dir": projects_dir.strip(),
        }
        save_app_config(new_cfg)
        # Force LLM rebuild on next rerun
        st.session_state.pop("llm", None)
        st.success("Settings saved. LLM client will be rebuilt on the next page load.")

    # Connection test
    st.divider()
    if st.button("Test connection"):
        from prompt_forge.app.llm import build_llm
        from prompt_forge import LLMMessage

        st.session_state.pop("llm", None)
        llm = build_llm()
        if llm is None:
            st.error("Could not build LLM client. Check your credentials above.")
        else:
            try:
                resp = llm.complete(
                    [LLMMessage(role="user", content="Reply with the single word: OK")],
                    temperature=0.0,
                    max_tokens=5,
                )
                st.success(f"Connection OK. Response: `{resp.text.strip()}`")
                st.session_state.llm = llm
            except Exception as e:
                st.error(f"Connection failed: {e}")
