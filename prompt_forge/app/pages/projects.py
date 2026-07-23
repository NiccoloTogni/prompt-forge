"""Projects page — create, open, and manage projects."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from prompt_forge.app.state import load_app_config, get_project, set_project, get_llm, require_llm


def show() -> None:
    st.header("Projects")

    cfg = load_app_config()
    projects_root = Path(cfg["projects_dir"])
    projects_root.mkdir(parents=True, exist_ok=True)

    col_list, col_new = st.columns([2, 1])

    # ── Existing projects ──────────────────────────────────────────────────────
    with col_list:
        st.subheader("Open project")
        existing = sorted(
            [d for d in projects_root.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not existing:
            st.info("No projects yet. Create one on the right.")
        else:
            for proj_dir in existing:
                name = proj_dir.name
                active = get_project() and get_project().name == name
                label = f"**{name}**" + (" (open)" if active else "")
                col_a, col_b = st.columns([4, 1])
                col_a.markdown(label)
                if col_b.button("Open", key=f"open_{name}", disabled=active):
                    _open_project(name, proj_dir)

    # ── Create new project ─────────────────────────────────────────────────────
    with col_new:
        st.subheader("New project")
        with st.form("new_project_form"):
            new_name = st.text_input("Project name", placeholder="my_project")
            submitted = st.form_submit_button("Create", type="primary")

        if submitted:
            new_name = new_name.strip().replace(" ", "_")
            if not new_name:
                st.error("Project name cannot be empty.")
            else:
                proj_dir = projects_root / new_name
                if proj_dir.exists():
                    st.warning(f"Project '{new_name}' already exists. Opening it.")
                _open_project(new_name, proj_dir)

    # ── Currently open project summary ────────────────────────────────────────
    project = get_project()
    if project:
        st.divider()
        st.subheader(f"Active: {project.name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Examples", project.num_examples)
        c2.metric("Versions", project.num_versions)
        latest = project.get_prompt()
        c3.metric("Latest version", f"v{latest.version}" if latest else "none")

        if latest:
            with st.expander("Current prompt (latest version)"):
                st.code(latest.prompt_text, language="text")


def _open_project(name: str, proj_dir: Path) -> None:
    llm = get_llm()
    if llm is None:
        st.error("Configure your LLM credentials in **Settings** first.")
        return

    from prompt_forge import Project

    project = Project(name=name, llm=llm, project_dir=proj_dir)
    set_project(project)
    st.success(f"Opened project '{name}'.")
    st.rerun()
