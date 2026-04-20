"""
Tests for the context_retriever hook on InferenceAgent.
"""

from pathlib import Path

import pytest

from prompt_forge.bundle import ExampleBundle
from prompt_forge.inference.agent import InferenceAgent
from prompt_forge.llm.client import LLMMessage, LLMResponse, TextPart, FilePart
from prompt_forge.file_loaders import get_default_loader


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
    llm = CapturingLLM(response_text='["r0", "r1"]')
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
