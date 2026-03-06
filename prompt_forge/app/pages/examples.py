"""Examples page — upload, browse, and delete training examples."""

import tempfile
from pathlib import Path

import streamlit as st

from prompt_forge.app.state import require_project


def show() -> None:
    st.header("Examples")
    project = require_project()

    # ── Schema setup ──────────────────────────────────────────────────────────
    st.subheader("Bundle schema")

    schema = project._schema
    if schema is None:
        with st.form("schema_form"):
            st.caption(
                "Define the roles in each example bundle. "
                "e.g. `input` → `.txt`, `expected_output` → `.txt`"
            )
            num_roles = st.number_input("Number of roles", 1, 8, 2)
            roles: dict[str, str] = {}
            for i in range(int(num_roles)):
                c1, c2 = st.columns(2)
                role_name = c1.text_input(f"Role {i+1} name", key=f"rn_{i}",
                                          placeholder="input")
                extension = c2.text_input(f"Extension", key=f"re_{i}",
                                          placeholder=".txt")
                if role_name and extension:
                    roles[role_name] = extension if extension.startswith(".") else f".{extension}"

            if st.form_submit_button("Set schema", type="primary"):
                if len(roles) < 1:
                    st.error("Define at least one role.")
                else:
                    project.set_bundle_schema(**roles)
                    st.success(f"Schema set: {roles}")
                    st.rerun()
    else:
        st.write({role: ext for role, ext in schema.roles.items()})
        if st.button("Reset schema (deletes all examples)"):
            project._schema = None
            project._bundles = None
            project._save_config()
            st.rerun()

    if schema is None:
        return

    st.divider()

    # ── Upload examples ────────────────────────────────────────────────────────
    st.subheader("Upload examples")
    st.caption(
        "Upload one or more complete bundles. Each bundle must contain one file per role. "
        "Files are grouped by bundle ID (you can name them freely)."
    )

    with st.form("upload_form"):
        bundle_id = st.text_input("Bundle ID", placeholder="example_001")
        uploaded: dict[str, object] = {}
        for role, ext in schema.roles.items():
            f = st.file_uploader(f"{role} ({ext})", key=f"upload_{role}")
            if f:
                uploaded[role] = f

        if st.form_submit_button("Add bundle"):
            if not bundle_id.strip():
                st.error("Bundle ID is required.")
            elif len(uploaded) != len(schema.roles):
                st.error(f"Upload a file for every role: {list(schema.roles.keys())}")
            else:
                _save_uploaded_bundle(project, bundle_id.strip(), uploaded, schema)

    st.divider()

    # ── Bundle list ────────────────────────────────────────────────────────────
    st.subheader(f"Loaded examples ({project.num_examples})")
    if project.num_examples == 0:
        st.info("No examples yet. Upload some above.")
        return

    bundles = project._bundles.bundles if project._bundles else []
    for bundle in bundles:
        with st.expander(bundle.bundle_id):
            for role, path in bundle.files.items():
                st.caption(f"**{role}**: `{path}`")
            if st.button("Delete", key=f"del_{bundle.bundle_id}"):
                project._bundles._bundles.pop(bundle.bundle_id, None)
                project._save_config()
                st.rerun()


def _save_uploaded_bundle(project, bundle_id: str, uploaded: dict, schema) -> None:
    projects_dir = Path(project.project_dir) / "examples" / bundle_id
    projects_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    for role, file_obj in uploaded.items():
        ext = schema.roles[role]
        dest = projects_dir / f"{role}{ext}"
        dest.write_bytes(file_obj.read())
        files[role] = dest

    project.add_example(bundle_id=bundle_id, files={r: str(p) for r, p in files.items()})
    project._save_config()
    st.success(f"Added bundle '{bundle_id}'.")
    st.rerun()
