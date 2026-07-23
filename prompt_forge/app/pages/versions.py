"""Versions page — browse prompt history and compare versions."""

from __future__ import annotations

import streamlit as st

from prompt_forge.app.state import require_project


def show() -> None:
    st.header("Versions")
    project = require_project()

    versions = project.list_versions()
    if not versions:
        st.info("No prompt versions yet. Set a seed prompt or run training.")
        return

    # ── Summary table ──────────────────────────────────────────────────────────
    rows = []
    for v in reversed(versions):
        rows.append({
            "Version": f"v{v.version}",
            "Score": f"{v.eval_score:.3f}" if v.eval_score is not None else "-",
            "Created": v.created_at[:19].replace("T", " "),
            "Parent": f"v{v.parent_version}" if v.parent_version is not None else "-",
            "Learned": (v.training_log_entry or "")[:100],
        })
    st.dataframe(rows, use_container_width=True)

    st.divider()

    # ── Version viewer ─────────────────────────────────────────────────────────
    st.subheader("Inspect version")
    version_nums = [v.version for v in reversed(versions)]
    selected_num = st.selectbox(
        "Select version",
        version_nums,
        format_func=lambda n: f"v{n}" + (" (latest)" if n == versions[-1].version else ""),
    )

    v = project.get_prompt(selected_num)
    if v is None:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{v.eval_score:.3f}" if v.eval_score is not None else "N/A")
    c2.metric("Created", v.created_at[:10])
    c3.metric("Parent", f"v{v.parent_version}" if v.parent_version is not None else "seed")

    if v.training_log_entry:
        st.subheader("What was learned")
        st.markdown(v.training_log_entry)

    st.subheader("Prompt text")
    st.code(v.prompt_text, language="text")

    if v.eval_details:
        with st.expander("Eval details"):
            st.json(v.eval_details)

    if v.output_schema:
        with st.expander("Output schema"):
            st.json(v.output_schema)

    # ── Side-by-side diff ─────────────────────────────────────────────────────
    if len(versions) >= 2:
        st.divider()
        st.subheader("Compare two versions")
        va_num, vb_num = st.columns(2)
        a = va_num.selectbox("Version A", version_nums, index=0, key="cmp_a",
                             format_func=lambda n: f"v{n}")
        b = vb_num.selectbox("Version B", version_nums, index=min(1, len(version_nums)-1),
                             key="cmp_b", format_func=lambda n: f"v{n}")

        if a != b:
            va = project.get_prompt(a)
            vb = project.get_prompt(b)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"v{a}")
                st.code(va.prompt_text if va else "", language="text")
            with col_b:
                st.caption(f"v{b}")
                st.code(vb.prompt_text if vb else "", language="text")
