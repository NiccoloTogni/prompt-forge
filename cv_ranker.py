"""
cv_ranker.py — A simple GUI for ranking CVs using prompt-forge.

Usage:
    python cv_ranker.py

Requirements:
    pip install prompt-forge openai python-dotenv ttkthemes  # openai package includes AzureOpenAI
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from dotenv import load_dotenv
from openai import AzureOpenAI
from ttkthemes import ThemedTk

load_dotenv()

# ── prompt-forge imports ───────────────────────────────────────────────────────
from prompt_forge import Project, LLMMessage, LLMResponse
from prompt_forge.training.pipeline import TrainingConfig

# ── Ranking meta-prompt ────────────────────────────────────────────────────────
RANKING_META_PROMPT = """\
You are a technical recruiter maintaining a ranked shortlist of candidates for a job opening.

You will receive:
1. The current ranked list (which may be empty on the first call)
2. A set of new candidate CVs to evaluate

Your job:
- Evaluate each new CV against the hiring criteria in the header
- Insert each candidate into the correct position in the ranking
- For each candidate include: rank, name, a one-line fit summary, key strengths and gaps
- Remove candidates clearly below the minimum bar
- Keep the list clean and well-structured

Return ONLY the updated ranked list. No preamble, no explanation.
Use this exact format for each entry:

## #1 — [Candidate Name]
**Fit:** [one sentence]
**Strengths:** [comma-separated]
**Gaps:** [comma-separated or "none"]
""".strip()


# ── LLM factory ───────────────────────────────────────────────────────────────
def make_llm(api_key: str, model: str, endpoint: str):
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-02-01",
    )

    def llm(messages: list[LLMMessage]) -> LLMResponse:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=0.2,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )
    return llm


# ── App ────────────────────────────────────────────────────────────────────────
class CVRankerApp(ThemedTk):
    def __init__(self):
        super().__init__(theme="breeze")
        self.title("CV Ranker — prompt-forge")
        self.geometry("1100x700")
        self.minsize(800, 500)

        self._cvs: list[dict] = []   # {"name": str, "text": str}
        self._running = False

        self._build_ui()

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar: API settings
        top = ttk.Frame(self, padding=(10, 6))
        top.pack(fill="x", side="top")

        ttk.Label(top, text="Azure Endpoint:").pack(side="left")
        self._endpoint_var = tk.StringVar(value=os.environ.get("AZURE_OPENAI_ENDPOINT", ""))
        ttk.Entry(top, textvariable=self._endpoint_var, width=36).pack(side="left", padx=(4, 16))

        ttk.Label(top, text="API Key:").pack(side="left")
        self._api_key_var = tk.StringVar(value=os.environ.get("AZURE_OPENAI_API_KEY", ""))
        ttk.Entry(top, textvariable=self._api_key_var, width=28, show="*").pack(side="left", padx=(4, 16))

        ttk.Label(top, text="Deployment:").pack(side="left")
        self._model_var = tk.StringVar(value=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"))
        ttk.Entry(top, textvariable=self._model_var, width=16).pack(side="left", padx=(4, 16))

        ttk.Label(top, text="Batch size:").pack(side="left")
        self._batch_var = tk.IntVar(value=3)
        ttk.Spinbox(top, from_=1, to=20, textvariable=self._batch_var, width=5).pack(side="left", padx=(4, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Second row: project dir
        top2 = ttk.Frame(self, padding=(10, 4))
        top2.pack(fill="x", side="top")
        ttk.Label(top2, text="Project dir:").pack(side="left")
        self._project_dir_var = tk.StringVar(value=".cv_ranker_project")
        ttk.Entry(top2, textvariable=self._project_dir_var, width=40).pack(side="left", padx=(4, 4))
        ttk.Button(top2, text="Browse…", command=self._browse_project_dir).pack(side="left")
        ttk.Label(top2, text="(reuse the same dir to resume a previous ranking)", foreground="gray").pack(side="left", padx=(12, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        # Main area: left panel + right panel
        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(main, padding=4)
        right = ttk.Frame(main, padding=4)
        main.add(left, weight=1)
        main.add(right, weight=2)

        self._build_left(left)
        self._build_right(right)

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        status = ttk.Label(self, textvariable=self._status_var, anchor="w", padding=(10, 3))
        status.pack(fill="x", side="bottom")
        ttk.Separator(self, orient="horizontal").pack(fill="x", side="bottom")

    def _build_left(self, parent):
        # Hiring criteria
        criteria_frame = ttk.LabelFrame(parent, text="Hiring Criteria", padding=6)
        criteria_frame.pack(fill="x", pady=(0, 6))
        self._criteria_text = scrolledtext.ScrolledText(criteria_frame, height=8, wrap="word", font=("", 9))
        self._criteria_text.pack(fill="x")
        self._criteria_text.insert("1.0", (
            "We are hiring a Senior Machine Learning Engineer.\n\n"
            "Ideal profile:\n"
            "- Strong Python and MLOps experience\n"
            "- Production ML (not just academic)\n"
            "- LLMs / NLP is a strong plus\n"
            "- Team leadership preferred"
        ))

        # Add CV
        add_frame = ttk.LabelFrame(parent, text="Add Candidate CV", padding=6)
        add_frame.pack(fill="x", pady=(0, 6))

        name_row = ttk.Frame(add_frame)
        name_row.pack(fill="x", pady=(0, 4))
        ttk.Label(name_row, text="Name:").pack(side="left")
        self._cv_name_var = tk.StringVar()
        ttk.Entry(name_row, textvariable=self._cv_name_var, width=28).pack(side="left", padx=(4, 0))

        self._cv_text = scrolledtext.ScrolledText(add_frame, height=6, wrap="word", font=("", 9))
        self._cv_text.pack(fill="x", pady=(0, 4))

        btn_row = ttk.Frame(add_frame)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Add CV", command=self._add_cv).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Load from file…", command=self._load_cv_file).pack(side="left")

        # CV list
        list_outer = ttk.LabelFrame(parent, text="Loaded CVs", padding=6)
        list_outer.pack(fill="both", expand=True, pady=(0, 6))

        list_frame = ttk.Frame(list_outer)
        list_frame.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical")
        self._cv_listbox = tk.Listbox(
            list_frame, yscrollcommand=scrollbar.set,
            selectmode="single", font=("", 9), height=6,
        )
        scrollbar.config(command=self._cv_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self._cv_listbox.pack(fill="both", expand=True)

        list_btns = ttk.Frame(list_outer)
        list_btns.pack(fill="x", pady=(4, 0))
        ttk.Button(list_btns, text="Remove selected", command=self._remove_cv).pack(side="left", padx=(0, 4))
        ttk.Button(list_btns, text="Clear all", command=self._clear_cvs).pack(side="left")

        self._run_btn = ttk.Button(parent, text="▶  Run Ranking", command=self._run)
        self._run_btn.pack(fill="x", ipady=4)

    def _build_right(self, parent):
        result_frame = ttk.LabelFrame(parent, text="Ranked Output", padding=6)
        result_frame.pack(fill="both", expand=True)

        btn_row = ttk.Frame(result_frame)
        btn_row.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_row, text="Copy", command=self._copy_result).pack(side="right")
        ttk.Button(btn_row, text="Save…", command=self._save_result).pack(side="right", padx=(0, 4))

        self._result_text = scrolledtext.ScrolledText(
            result_frame, wrap="word", font=("Courier", 9),
            state="disabled", bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white",
        )
        self._result_text.pack(fill="both", expand=True)

    # ── CV management ──────────────────────────────────────────────────────────
    def _add_cv(self):
        name = self._cv_name_var.get().strip()
        text = self._cv_text.get("1.0", "end").strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a candidate name.")
            return
        if not text:
            messagebox.showwarning("Missing CV", "Please paste the CV text.")
            return
        self._cvs.append({"name": name, "text": text})
        self._cv_listbox.insert("end", name)
        self._cv_name_var.set("")
        self._cv_text.delete("1.0", "end")
        self._set_status(f"{len(self._cvs)} CV(s) loaded.")

    def _load_cv_file(self):
        paths = filedialog.askopenfilenames(
            title="Select CV files",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                self._cvs.append({"name": name, "text": text})
                self._cv_listbox.insert("end", name)
            except Exception as e:
                messagebox.showerror("Load error", str(e))
        self._set_status(f"{len(self._cvs)} CV(s) loaded.")

    def _remove_cv(self):
        sel = self._cv_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self._cvs.pop(idx)
        self._cv_listbox.delete(idx)
        self._set_status(f"{len(self._cvs)} CV(s) loaded.")

    def _clear_cvs(self):
        self._cvs.clear()
        self._cv_listbox.delete(0, "end")
        self._set_status("Ready.")

    def _browse_project_dir(self):
        path = filedialog.askdirectory(title="Select project directory")
        if path:
            self._project_dir_var.set(path)

    # ── Run ────────────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        if not self._cvs:
            messagebox.showwarning("No CVs", "Add at least one CV before running.")
            return
        api_key = self._api_key_var.get().strip()
        endpoint = self._endpoint_var.get().strip()
        if not api_key:
            messagebox.showwarning("API Key", "Please enter your Azure API key.")
            return
        if not endpoint:
            messagebox.showwarning("Endpoint", "Please enter your Azure OpenAI endpoint.")
            return
        criteria = self._criteria_text.get("1.0", "end").strip()
        if not criteria:
            messagebox.showwarning("Criteria", "Please fill in the hiring criteria.")
            return

        self._running = True
        self._run_btn.config(state="disabled")
        self._set_result("Running…")
        self._set_status("Ranking in progress…")

        threading.Thread(target=self._rank_thread, args=(api_key, endpoint, criteria), daemon=True).start()

    def _rank_thread(self, api_key: str, endpoint: str, criteria: str):
        try:
            llm = make_llm(api_key, self._model_var.get().strip(), endpoint)
            project_dir = self._project_dir_var.get().strip() or ".cv_ranker_project"

            project = Project(
                name="cv_ranking",
                llm=llm,
                project_dir=project_dir,
            )
            project.set_bundle_schema(input_fields=["cv"])

            # Only set seed prompt if no previous version exists (fresh start)
            existing_versions = project.list_versions()
            if not existing_versions:
                project.set_seed_prompt(
                    f"{criteria}\n\n---\nRANKED CANDIDATES\n(no candidates evaluated yet)"
                )
            else:
                self.after(0, self._set_status, f"Resuming from version {len(existing_versions)} — {len(self._cvs)} new CV(s) to process.")

            for cv in self._cvs:
                project.add_example(input=cv["text"])

            batch_size = max(1, self._batch_var.get())
            max_iter = max(1, (len(self._cvs) + batch_size - 1) // batch_size)

            def on_iteration(result):
                self.after(0, lambda: self._set_status(
                    f"Processed batch {result.iteration}/{max_iter} "
                    f"({result.tokens_used:,} tokens this batch)"
                ))

            report = project.train(
                eval_strategy=None,
                optimizer_kwargs={"meta_prompt": RANKING_META_PROMPT},
                on_iteration=on_iteration,
                config=TrainingConfig(
                    batch_size=batch_size,
                    max_iterations=max_iter,
                    inference_temperature=0.1,
                ),
            )

            self.after(0, self._set_result, report.best_prompt)
            total_versions = len(project.list_versions())
            self.after(0, self._set_status,
                       f"Done — {len(self._cvs)} CVs processed, "
                       f"{report.total_tokens_used:,} tokens used, "
                       f"{total_versions} total version(s) saved in {project_dir}.")

        except Exception as e:
            self.after(0, self._set_result, f"Error:\n\n{e}")
            self.after(0, self._set_status, "Failed — see output for details.")
        finally:
            self.after(0, self._finish_run)

    def _finish_run(self):
        self._running = False
        self._run_btn.config(state="normal")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _set_result(self, text: str):
        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("1.0", text)
        self._result_text.config(state="disabled")

    def _copy_result(self):
        text = self._result_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._set_status("Copied to clipboard.")

    def _save_result(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._result_text.get("1.0", "end").strip())
        self._set_status(f"Saved to {path}")


if __name__ == "__main__":
    app = CVRankerApp()
    app.mainloop()
