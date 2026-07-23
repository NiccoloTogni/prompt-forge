"""Training page — configure, launch, and monitor training."""

from __future__ import annotations

import threading
import time

import streamlit as st

from prompt_forge.app.state import require_project, require_llm


def show() -> None:
    st.header("Training")
    project = require_project()
    require_llm()  # ensure LLM is configured

    if project.num_examples == 0:
        st.warning("No examples loaded. Go to **Examples** first.")
        return
    if project.get_prompt() is None:
        st.warning("No seed prompt set. Go to **Context & Prompt** first.")
        return

    running = st.session_state.get("training_running", False)

    # ── Config form (disabled while running) ───────────────────────────────────
    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        batch_size  = c1.number_input("Batch size",      min_value=1, max_value=50,         value=5)
        max_iter    = c2.number_input("Max iterations",  min_value=1, max_value=200,        value=10)
        patience    = c3.number_input("Patience",        min_value=1, max_value=50,         value=3)

        c4, c5 = st.columns(2)
        eval_strategy = c4.selectbox(
            "Eval strategy",
            ["none", "similarity", "exact_match", "json_fields", "llm_judge"],
        )
        max_tokens_m = c5.number_input(
            "Max total tokens (M, 0 = unlimited)",
            min_value=0.0, max_value=100.0, value=0.0, step=0.5,
        )

        submitted = st.form_submit_button(
            "Start training", type="primary", disabled=running
        )

    if submitted and not running:
        max_total = int(max_tokens_m * 1_000_000) if max_tokens_m > 0 else None
        _start_training(project, batch_size, max_iter, patience,
                        eval_strategy, max_total)

    # ── Live progress ──────────────────────────────────────────────────────────
    if running:
        st.info("Training in progress...")
        _render_log()
        time.sleep(1.5)
        st.rerun()

    elif st.session_state.get("training_error"):
        st.error(f"Training failed: {st.session_state.training_error}")

    elif st.session_state.get("training_report"):
        _render_report(st.session_state.training_report)
        # Also keep the partial log visible
        if st.session_state.get("training_log"):
            with st.expander("Iteration log"):
                _render_log()


def _start_training(project, batch_size, max_iter, patience, eval_strategy, max_total):
    st.session_state.training_running = True
    st.session_state.training_log = []
    st.session_state.training_report = None
    st.session_state.training_error = None

    def on_iteration(result):
        st.session_state.training_log.append(result)

    def run():
        try:
            from prompt_forge.training.pipeline import TrainingConfig
            report = project.train(
                config=TrainingConfig(
                    batch_size=batch_size,
                    max_iterations=max_iter,
                    patience=patience,
                    max_total_tokens=max_total,
                ),
                eval_strategy=eval_strategy if eval_strategy != "none" else None,
                on_iteration=on_iteration,
            )
            st.session_state.training_report = report
        except Exception as e:
            st.session_state.training_error = str(e)
        finally:
            st.session_state.training_running = False

    threading.Thread(target=run, daemon=True).start()
    st.rerun()


def _render_log():
    log = st.session_state.get("training_log", [])
    if not log:
        st.caption("Waiting for first iteration...")
        return
    rows = []
    for r in log:
        score = (
            f"{r.score_before:.3f} → {r.score_after:.3f}"
            if r.score_after is not None
            else "no eval"
        )
        rows.append({
            "Iter": r.iteration,
            "Version": f"v{r.prompt_version}",
            "Score": score,
            "Improved": "yes" if r.improved else "no",
            "Tokens": f"{r.tokens_used:,}" if r.tokens_used else "-",
        })
    st.dataframe(rows, use_container_width=True)


def _render_report(report):
    st.success(f"Training complete — final version: **v{report.final_version}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Iterations", len(report.iterations))
    c2.metric("Final score", f"{report.final_score:.3f}" if report.final_score is not None else "N/A")
    c3.metric("Total tokens", f"{report.total_tokens_used:,}")
    c4.metric("Refine recommended", "Yes" if report.refinement_recommended else "No")

    rows = []
    for r in report:
        rows.append({
            "Iter": r.iteration,
            "Version": f"v{r.prompt_version}",
            "Score before": f"{r.score_before:.3f}" if r.score_before is not None else "-",
            "Score after":  f"{r.score_after:.3f}"  if r.score_after  is not None else "-",
            "Improved": "yes" if r.improved else "no",
            "Tokens": f"{r.tokens_used:,}" if r.tokens_used else "-",
            "Learned": (r.learnings or "")[:120],
        })
    st.dataframe(rows, use_container_width=True)

    if report.refinement_recommended:
        st.info("Score is below the refinement threshold. Consider running an **Interactive Refine** session.")
