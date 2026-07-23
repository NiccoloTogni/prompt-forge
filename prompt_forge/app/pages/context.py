"""Context & Prompt page — seed prompt, domain context, output schema."""

from __future__ import annotations

import json

import streamlit as st

from prompt_forge.app.state import require_project


def show() -> None:
    st.header("Context & Prompt")
    project = require_project()

    # ── Seed prompt ────────────────────────────────────────────────────────────
    st.subheader("Seed prompt")
    st.caption(
        "The starting prompt for training. If a seed already exists (v0), "
        "editing here will overwrite it only if no training has been done yet."
    )
    current_seed = project._seed_prompt or ""
    new_seed = st.text_area("Seed prompt", value=current_seed, height=180)
    if st.button("Save seed prompt", type="primary"):
        if not new_seed.strip():
            st.error("Seed prompt cannot be empty.")
        else:
            project._seed_prompt = new_seed.strip()
            # Overwrite v0 if it's the only version
            versions = project.list_versions()
            if not versions or (len(versions) == 1 and versions[0].version == 0):
                project.set_seed_prompt(new_seed.strip())
                st.success("Seed prompt saved as v0.")
            else:
                project._save_config()
                st.success("Seed prompt updated (not re-saved as version — training already exists).")

    st.divider()

    # ── Domain context ─────────────────────────────────────────────────────────
    st.subheader("Domain context")
    st.caption("Helps the optimizer understand the task. Not used at inference time.")
    new_context = st.text_area("Context", value=project._context or "", height=120)
    if st.button("Save context"):
        project.set_context(new_context.strip())
        st.success("Context saved.")

    st.divider()

    # ── Output schema ──────────────────────────────────────────────────────────
    st.subheader("Output schema (optional)")
    st.caption(
        "If your task produces structured JSON, declare the expected fields here. "
        "Leave blank for plain-text output."
    )

    current_schema = project._output_schema
    schema_str = json.dumps(current_schema, indent=2) if current_schema else ""
    new_schema_str = st.text_area(
        "JSON schema (dict of field → type, or full JSON Schema)",
        value=schema_str,
        height=120,
        placeholder='{"invoice_number": "string", "total": "number"}',
    )

    col_save, col_clear = st.columns([1, 1])
    if col_save.button("Save schema"):
        if not new_schema_str.strip():
            st.error("Schema is empty. Use 'Clear schema' to remove it.")
        else:
            try:
                parsed = json.loads(new_schema_str)
                if not isinstance(parsed, dict):
                    st.error("Schema must be a JSON object.")
                else:
                    project.set_output_schema(parsed)
                    st.success(f"Output schema saved: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    if col_clear.button("Clear schema"):
        project._output_schema = None
        project._save_config()
        st.success("Output schema removed.")
