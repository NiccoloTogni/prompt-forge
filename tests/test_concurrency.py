"""
Tests for concurrent batch inference via ThreadPoolExecutor.
"""

import threading
from pathlib import Path

import pytest

from prompt_forge.bundle import ExampleBundle
from prompt_forge.inference.agent import InferenceAgent
from prompt_forge.llm.client import LLMMessage, LLMResponse, FilePart, TextPart
from prompt_forge.training.pipeline import TrainingConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

class ThreadTrackingLLM:
    """Records which thread handled each call and returns a fixed response."""

    def __init__(self, response_text: str = "output"):
        self.calls: list[LLMMessage] = []
        self.thread_ids: list[int] = []
        self.response_text = response_text
        self._lock = threading.Lock()

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        with self._lock:
            self.calls.append(messages)
            self.thread_ids.append(threading.get_ident())
        return LLMResponse(text=self.response_text, usage={"input_tokens": 5, "output_tokens": 2})


def make_agent(llm, max_workers=None, native_files=True):
    return InferenceAgent(
        llm=llm,
        prompt_text="You are helpful.",
        native_files=native_files,
        max_workers=max_workers,
    )


def make_file_bundle(tmp_path: Path, name: str) -> ExampleBundle:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / f"{name}.pdf"
    p.write_bytes(b"%PDF")
    return ExampleBundle(bundle_id=name, files={"input": p})


# ── Serial behaviour unchanged ────────────────────────────────────────────────

def test_serial_fallback_still_works(tmp_path):
    llm = ThreadTrackingLLM()
    agent = make_agent(llm, max_workers=None)
    bundles = [make_file_bundle(tmp_path / f"b{i}", f"b{i}") for i in range(3)]
    results = agent.run_bundle_batch(bundles)
    assert len(results) == 3
    assert llm.calls.__len__() == 3


# ── Concurrent mode ───────────────────────────────────────────────────────────

def test_concurrent_correct_result_count(tmp_path):
    llm = ThreadTrackingLLM(response_text="done")
    agent = make_agent(llm, max_workers=4)
    bundles = [make_file_bundle(tmp_path / f"c{i}", f"c{i}") for i in range(5)]
    results = agent.run_bundle_batch(bundles)
    assert len(results) == 5
    assert all(r == "done" for r in results)


def test_concurrent_uses_multiple_threads(tmp_path):
    import time

    class SlowTrackingLLM(ThreadTrackingLLM):
        """Adds a small delay so the pool must use multiple threads."""
        def complete(self, messages, **kwargs):
            time.sleep(0.02)
            return super().complete(messages, **kwargs)

    llm = SlowTrackingLLM()
    agent = make_agent(llm, max_workers=4)
    bundles = [make_file_bundle(tmp_path / f"t{i}", f"t{i}") for i in range(4)]
    agent.run_bundle_batch(bundles)
    assert len(set(llm.thread_ids)) >= 2


def test_concurrent_preserves_order(tmp_path):
    """Results must correspond to input order regardless of completion order."""
    import time
    from prompt_forge.llm.client import FilePart as _FilePart

    class EchoLLM:
        """Returns result_N where N is derived from the bundle's file path, not call order."""
        def complete(self, messages, **kwargs):
            time.sleep(0.01)
            user_parts = messages[1].content
            for part in user_parts:
                if isinstance(part, _FilePart):
                    idx = int(part.path.stem.lstrip("o"))
                    return LLMResponse(text=f"result_{idx}", usage={})
            return LLMResponse(text="unknown", usage={})

    agent = make_agent(EchoLLM(), max_workers=3)
    bundles = [make_file_bundle(tmp_path / f"o{i}", f"o{i}") for i in range(6)]
    results = agent.run_bundle_batch(bundles)
    assert results == [f"result_{i}" for i in range(6)]


def test_concurrent_accumulates_tokens(tmp_path):
    llm = ThreadTrackingLLM()  # returns usage={"input_tokens": 5, "output_tokens": 2}
    agent = make_agent(llm, max_workers=4)
    bundles = [make_file_bundle(tmp_path / f"tok{i}", f"tok{i}") for i in range(4)]
    agent.run_bundle_batch(bundles)
    assert agent.tokens_used == 4 * 7  # 4 bundles × (5 + 2) tokens


def test_concurrent_exception_propagates(tmp_path):
    class FailingLLM:
        def complete(self, messages, **kwargs):
            raise RuntimeError("provider down")

    agent = InferenceAgent(
        llm=FailingLLM(),
        prompt_text="sys",
        max_workers=2,
        max_retries=0,
    )
    bundles = [make_file_bundle(tmp_path / f"e{i}", f"e{i}") for i in range(2)]
    with pytest.raises(RuntimeError, match="provider down"):
        agent.run_bundle_batch(bundles)


# ── Text-only path unaffected ─────────────────────────────────────────────────

def test_text_batch_still_uses_single_call(tmp_path):
    """max_workers must not affect the text-only single-call batch path."""
    from prompt_forge.file_loaders import get_default_loader

    llm = ThreadTrackingLLM(response_text='<output id="1">r0</output><output id="2">r1</output><output id="3">r2</output>')
    agent = InferenceAgent(
        llm=llm,
        prompt_text="sys",
        native_files=False,
        file_loader=get_default_loader(),
        max_workers=4,
    )
    p0 = tmp_path / "x0.txt"; p0.write_text("a")
    p1 = tmp_path / "x1.txt"; p1.write_text("b")
    p2 = tmp_path / "x2.txt"; p2.write_text("c")
    bundles = [
        ExampleBundle(bundle_id="x0", files={"input": p0}),
        ExampleBundle(bundle_id="x1", files={"input": p1}),
        ExampleBundle(bundle_id="x2", files={"input": p2}),
    ]
    agent.run_bundle_batch(bundles)
    assert len(llm.calls) == 1  # single batched call, not 3


# ── TrainingConfig ────────────────────────────────────────────────────────────

def test_training_config_max_workers_default():
    assert TrainingConfig().max_workers is None


def test_training_config_max_workers_set():
    assert TrainingConfig(max_workers=8).max_workers == 8


def test_training_config_eval_train_default():
    assert TrainingConfig().eval_train is False
