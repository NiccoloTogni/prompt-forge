"""PromptForge web UI — powered by Streamlit.

Install:
    pip install prompt-forge[app]

Run:
    prompt-forge-ui
or:
    python -m streamlit run $(python -c "from prompt_forge.app import _main_path; print(_main_path())")
"""

from pathlib import Path


def _main_path() -> str:
    return str(Path(__file__).parent / "main.py")


def run() -> None:
    """CLI entry point: prompt-forge-ui"""
    import sys
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print(
            "Streamlit is not installed. Run:\n"
            "    pip install prompt-forge[app]"
        )
        sys.exit(1)

    sys.argv = ["streamlit", "run", _main_path(), *sys.argv[1:]]
    stcli.main()
