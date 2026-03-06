"""Session state helpers and app-level config persistence."""

import json
from pathlib import Path

import streamlit as st

_CONFIG_FILE = Path(".prompt_forge_ui.json")

_DEFAULTS: dict = {
    "azure_endpoint": "",
    "azure_api_key": "",
    "azure_deployment": "gpt-5.3-chat",
    "azure_api_version": "2024-12-01-preview",
    "projects_dir": "./projects",
}


# ── App config (persisted to disk) ────────────────────────────────────────────

def load_app_config() -> dict:
    if "app_config" not in st.session_state:
        if _CONFIG_FILE.exists():
            loaded = json.loads(_CONFIG_FILE.read_text())
            st.session_state.app_config = {**_DEFAULTS, **loaded}
        else:
            st.session_state.app_config = dict(_DEFAULTS)
    return st.session_state.app_config


def save_app_config(config: dict) -> None:
    st.session_state.app_config = config
    _CONFIG_FILE.write_text(json.dumps(config, indent=2))


# ── Project ────────────────────────────────────────────────────────────────────

def get_project():
    return st.session_state.get("project")


def set_project(project) -> None:
    st.session_state.project = project
    # Clear page-specific state when switching projects
    for key in ("training_running", "training_log", "training_report",
                "training_error", "refine_messages", "refine_prompt"):
        st.session_state.pop(key, None)


def require_project():
    p = get_project()
    if p is None:
        st.info("No project open. Go to **Projects** to create or open one.")
        st.stop()
    return p


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm():
    return st.session_state.get("llm")


def require_llm():
    llm = get_llm()
    if llm is None:
        st.info("LLM not configured. Go to **Settings** to enter your Azure credentials.")
        st.stop()
    return llm
