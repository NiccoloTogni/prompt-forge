"""Inference page — run the trained prompt on new inputs."""

import json
import tempfile
from pathlib import Path

import streamlit as st

from prompt_forge.app.state import require_project, require_llm


def show() -> None:
    st.header("Inference")
    project = require_project()
    llm = require_llm()

    versions = project.list_versions()
    if not versions:
        st.warning("No prompt versions. Run training or set a seed prompt first.")
        return

    # ── Version selector ───────────────────────────────────────────────────────
    version_nums = [v.version for v in reversed(versions)]
    selected_num = st.selectbox(
        "Prompt version",
        version_nums,
        format_func=lambda n: f"v{n}" + (" (latest)" if n == versions[-1].version else ""),
    )

    agent = project.get_inference_agent(version=selected_num)
    st.caption(agent.prompt_info)

    st.divider()

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode = st.radio("Input type", ["Text", "File", "Multiple files"], horizontal=True)

    result = None
    error = None

    if input_mode == "Text":
        text = st.text_area("Input text", height=180)
        if st.button("Run", type="primary") and text.strip():
            with st.spinner("Running inference..."):
                try:
                    result = agent.run(input_text=text.strip())
                except Exception as e:
                    error = str(e)

    elif input_mode == "File":
        uploaded = st.file_uploader("Upload input file")
        if st.button("Run", type="primary") and uploaded:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / uploaded.name
                tmp_path.write_bytes(uploaded.read())
                with st.spinner("Running inference..."):
                    try:
                        result = agent.run(input_file=tmp_path)
                    except Exception as e:
                        error = str(e)

    else:  # Multiple files
        schema = project._schema
        if schema is None:
            st.warning("No bundle schema defined. Set one in the **Examples** page.")
            return

        uploaded_files: dict[str, object] = {}
        for role in schema.roles:
            f = st.file_uploader(f"{role}", key=f"inf_{role}")
            if f:
                uploaded_files[role] = f

        if st.button("Run", type="primary") and len(uploaded_files) == len(schema.roles):
            with tempfile.TemporaryDirectory() as tmp:
                input_files: dict[str, Path] = {}
                for role, f in uploaded_files.items():
                    ext = schema.roles[role]
                    p = Path(tmp) / f"{role}{ext}"
                    p.write_bytes(f.read())
                    input_files[role] = p
                with st.spinner("Running inference..."):
                    try:
                        result = agent.run(input_files={r: str(p) for r, p in input_files.items()})
                    except Exception as e:
                        error = str(e)

    # ── Output ────────────────────────────────────────────────────────────────
    if error:
        st.error(f"Inference error: {error}")

    if result is not None:
        st.subheader("Output")
        if isinstance(result, dict):
            st.json(result)
        else:
            st.text_area("Result", value=str(result), height=240, disabled=True)
