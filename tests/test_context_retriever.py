"""
Tests for the context_retriever hook on InferenceAgent and TrainingConfig.
"""

from pathlib import Path

import pytest

from prompt_forge.bundle import ExampleBundle
from prompt_forge.inference.agent import InferenceAgent
from prompt_forge.llm.client import LLMMessage, LLMResponse, TextPart, FilePart
from prompt_forge.file_loaders import get_default_loader
from prompt_forge.training.pipeline import TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

class CapturingLLM:
    def __init__(self, response_text: str = "answer"):
        self.calls: list[list[LLMMessage]] = []
        self.response_text = response_text

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(text=self.response_text, usage={})

    @property
    def last_user_content(self):
        return self.calls[-1][1].content


def make_agent(llm, retriever=None, native_files=False):
    return InferenceAgent(
        llm=llm,
        prompt_text="sys",
        native_files=native_files,
        file_loader=get_default_loader() if not native_files else None,
        context_retriever=retriever,
    )


def make_bundle(tmp_path: Path, name: str = "b") -> ExampleBundle:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"{name}_input.txt"
    p.write_text("hello world")
    return ExampleBundle(bundle_id=name, files={"input": p})


# ── No retriever — baseline unchanged ────────────────────────────────────────

def test_no_retriever_run(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm)
    agent.run(input_text="hi")
    assert "<retrieved_context>" not in llm.last_user_content


def test_no_retriever_bundle(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm)
    bundle = make_bundle(tmp_path)
    agent.run_bundle(bundle)
    assert "<retrieved_context>" not in llm.last_user_content


# ── Retriever injects context ─────────────────────────────────────────────────

def test_retriever_injected_in_run_batch():
    received_queries = []

    def retriever(query, llm_client):
        received_queries.append(query)
        return "batch snippet"

    llm = CapturingLLM(response_text='<output id="1">out1</output><output id="2">out2</output>')
    agent = InferenceAgent(
        llm=llm,
        prompt_text="sys",
        native_files=False,
        file_loader=get_default_loader(),
        context_retriever=retriever,
    )
    agent.run_batch([{"input_text": "query one"}, {"input_text": "query two"}])

    assert received_queries == ["query one", "query two"]
    # Both user messages sent in the single batched call contain retrieved context
    user_content = llm.calls[-1][1].content
    assert user_content.count("<retrieved_context>") == 2


def test_retriever_injected_in_run(tmp_path):
    llm = CapturingLLM()
    retriever = lambda query, llm_client: "relevant snippet"
    agent = make_agent(llm, retriever=retriever)
    agent.run(input_text="hi")
    assert "<retrieved_context>\nrelevant snippet\n</retrieved_context>" in llm.last_user_content


def test_retriever_injected_in_run_bundle(tmp_path):
    llm = CapturingLLM()
    retriever = lambda query, llm_client: "bundle context"
    agent = make_agent(llm, retriever=retriever)
    bundle = make_bundle(tmp_path)
    agent.run_bundle(bundle)
    assert "<retrieved_context>\nbundle context\n</retrieved_context>" in llm.last_user_content


def test_retriever_injected_in_run_bundle_batch(tmp_path):
    llm = CapturingLLM(response_text='<output id="1">r0</output><output id="2">r1</output>')
    call_count = [0]

    def retriever(query, llm_client):
        call_count[0] += 1
        return f"ctx_{call_count[0]}"

    agent = InferenceAgent(
        llm=llm,
        prompt_text="sys",
        native_files=False,
        file_loader=get_default_loader(),
        context_retriever=retriever,
    )
    b0 = make_bundle(tmp_path / "b0", "b0")
    b1 = make_bundle(tmp_path / "b1", "b1")
    agent.run_bundle_batch([b0, b1])
    assert call_count[0] == 2


# ── Retriever receives the query ──────────────────────────────────────────────

def test_retriever_receives_input_text():
    received = []
    llm = CapturingLLM()
    retriever = lambda query, llm_client: received.append(query) or ""
    agent = make_agent(llm, retriever=retriever)
    agent.run(input_text="search me")
    assert received == ["search me"]


def test_retriever_receives_bundle_text(tmp_path):
    received = []
    llm = CapturingLLM()
    retriever = lambda query, llm_client: received.append(query) or ""
    agent = make_agent(llm, retriever=retriever)
    bundle = make_bundle(tmp_path)
    agent.run_bundle(bundle)
    assert "hello world" in received[0]


# ── Retriever failure is silent ───────────────────────────────────────────────

def test_retriever_exception_does_not_propagate():
    def bad_retriever(query, llm_client):
        raise RuntimeError("vector store down")

    llm = CapturingLLM()
    agent = make_agent(llm, retriever=bad_retriever)
    result = agent.run(input_text="hi")  # should not raise
    assert result == "answer"
    assert "<retrieved_context>" not in llm.last_user_content


# ── retriever + extra_context in run() are both injected ─────────────────────

def test_retriever_merges_with_extra_context():
    llm = CapturingLLM()
    retriever = lambda query, llm_client: "retrieved"
    agent = make_agent(llm, retriever=retriever)
    agent.run(input_text="hi", extra_context="user hint")
    content = llm.last_user_content
    assert "retrieved" in content
    assert "user hint" in content


# ── TrainingConfig wires retriever to eval agent ─────────────────────────────

def test_training_config_context_retriever_default():
    assert TrainingConfig().context_retriever is None


def test_training_config_context_retriever_set():
    retriever = lambda q, llm: "ctx"
    config = TrainingConfig(context_retriever=retriever)
    assert config.context_retriever is retriever


def test_eval_agent_receives_context_retriever(tmp_path):
    """Eval agent built inside train() must carry the context_retriever."""
    from unittest.mock import MagicMock
    from prompt_forge.training.pipeline import TrainingPipeline
    from prompt_forge.storage.project_store import FileSystemStore
    from prompt_forge.llm.client import LLMResponse

    # Minimal LLM that records whether retriever was called during eval
    retriever_calls = []

    def retriever(query, llm_client):
        retriever_calls.append(query)
        return "retrieved"

    llm = MagicMock()
    # Optimizer response
    llm.complete.return_value = LLMResponse(
        text="<optimized_prompt>better</optimized_prompt><learnings>ok</learnings><issues></issues>",
        usage={"input_tokens": 1, "output_tokens": 1},
    )

    store = FileSystemStore(tmp_path / "store")
    store.save_prompt_version(__import__("prompt_forge.storage.project_store", fromlist=["PromptVersion"]).PromptVersion(
        version=1, prompt_text="seed", created_at="2024-01-01T00:00:00+00:00",
    ))

    # One training bundle and one val bundle (text files)
    (tmp_path / "t").mkdir(); (tmp_path / "v").mkdir()
    (tmp_path / "t" / "t_input.txt").write_text("train input")
    (tmp_path / "t" / "t_expected_output.txt").write_text("train out")
    (tmp_path / "v" / "v_input.txt").write_text("val input")
    (tmp_path / "v" / "v_expected_output.txt").write_text("val out")

    train_b = ExampleBundle(bundle_id="t", files={
        "input": tmp_path / "t" / "t_input.txt",
        "expected_output": tmp_path / "t" / "t_expected_output.txt",
    })
    val_b = ExampleBundle(bundle_id="v", files={
        "input": tmp_path / "v" / "v_input.txt",
        "expected_output": tmp_path / "v" / "v_expected_output.txt",
    })

    from prompt_forge.evaluation.evaluator import ExactMatchEvaluator

    pipeline = TrainingPipeline(
        llm=llm,
        store=store,
        evaluator=ExactMatchEvaluator(),
        file_loader=get_default_loader(),
    )
    pipeline.train(
        [train_b],
        val_bundles=[val_b],
        config=TrainingConfig(
            max_iterations=1,
            native_files=False,
            context_retriever=retriever,
        ),
    )

    assert retriever_calls, "context_retriever was never called by the eval agent"


# ── Native file path ──────────────────────────────────────────────────────────

def test_retriever_injected_native_bundle(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "b_input.pdf"
    p.write_bytes(b"%PDF")
    bundle = ExampleBundle(bundle_id="b", files={"input": p})

    llm = CapturingLLM()
    retriever = lambda query, llm_client: "native context"
    agent = InferenceAgent(
        llm=llm,
        prompt_text="sys",
        native_files=True,
        context_retriever=retriever,
    )
    agent.run_bundle(bundle)
    parts = llm.calls[-1][1].content
    assert isinstance(parts, list)
    text_parts = [p.text for p in parts if isinstance(p, TextPart)]
    assert any("native context" in t for t in text_parts)
