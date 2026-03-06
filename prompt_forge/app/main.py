"""PromptForge Streamlit app — entry point and navigation."""

import sys
from pathlib import Path

# Ensure the package root is on sys.path when Streamlit runs this as a plain script
_pkg_root = str(Path(__file__).parent.parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import streamlit as st

from prompt_forge.app.state import load_app_config
from prompt_forge.app.llm import get_or_build_llm

from prompt_forge.app.pages.settings import show as show_settings
from prompt_forge.app.pages.projects import show as show_projects
from prompt_forge.app.pages.examples import show as show_examples
from prompt_forge.app.pages.context import show as show_context
from prompt_forge.app.pages.training import show as show_training
from prompt_forge.app.pages.refine import show as show_refine
from prompt_forge.app.pages.versions import show as show_versions
from prompt_forge.app.pages.inference import show as show_inference

st.set_page_config(
    page_title="PromptForge",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html("""
<style>
    /* Global font size bump */
    html, body, [class*="css"] { font-size: 17px; }

    /* Main content text */
    .stMarkdown p, .stText, div[data-testid="stText"] { font-size: 1.05rem; }

    /* Metric labels and values */
    [data-testid="stMetricLabel"] { font-size: 0.95rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }

    /* Dataframe text */
    .stDataFrame { font-size: 1rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { font-size: 1rem; }

    /* Code blocks */
    .stCodeBlock code { font-size: 0.95rem; }

    /* Chat messages */
    [data-testid="stChatMessage"] { font-size: 1.05rem; }
</style>
""")

# Bootstrap config and LLM on every rerun
load_app_config()
get_or_build_llm()

pages = {
    "Workspace": [
        st.Page(show_projects, title="Projects",          icon=":material/folder:",         url_path="projects"),
    ],
    "Project": [
        st.Page(show_examples,  title="Examples",         icon=":material/dataset:",        url_path="examples"),
        st.Page(show_context,   title="Context & Prompt", icon=":material/edit_note:",      url_path="context"),
        st.Page(show_training,  title="Training",         icon=":material/model_training:", url_path="training"),
        st.Page(show_refine,    title="Interactive Refine",icon=":material/chat:",          url_path="refine"),
        st.Page(show_versions,  title="Versions",         icon=":material/history:",        url_path="versions"),
        st.Page(show_inference, title="Inference",        icon=":material/bolt:",           url_path="inference"),
    ],
    "Config": [
        st.Page(show_settings,  title="Settings",         icon=":material/settings:",       url_path="settings"),
    ],
}

pg = st.navigation(pages)

# Logo in the sidebar header
_logo = Path(__file__).parent.parent.parent / "resources" / "promptforge-logo.svg"
if _logo.exists():
    st.logo(str(_logo), size="large")

# Sidebar: active project summary
with st.sidebar:
    from prompt_forge.app.state import get_project
    project = get_project()
    st.divider()
    if project:
        st.success(f"**{project.name}**")
        st.caption(f"{project.num_examples} examples · {project.num_versions} versions")
    else:
        st.warning("No project open")

pg.run()
