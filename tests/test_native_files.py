"""
Tests for native file support in InferenceAgent and related helpers.
"""

import json
from pathlib import Path

import pytest

from prompt_forge.llm.client import LLMMessage, LLMResponse, TextPart, FilePart, MessageContent
from prompt_forge.inference.agent import InferenceAgent, _infer_media_type, _has_file_parts
from prompt_forge.bundle import ExampleBundle, BundleSchema
from prompt_forge.file_loaders import get_default_loader


# ── Helpers ───────────────────────────────────────────────────────────────────

class CapturingLLM:
    """Records every call to complete() so tests can inspect what was sent."""

    def __init__(self, response_text: str = "output"):
        self.calls: list[list[LLMMessage]] = []
        self.response_text = response_text

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(text=self.response_text, usage={"input_tokens": 10, "output_tokens": 5})

    @property
    def last_user_content(self) -> MessageContent:
        return self.calls[-1][-1].content  # last call, user message


def make_agent(llm, native_files: bool = True, output_schema=None, file_loader=None) -> InferenceAgent:
    return InferenceAgent(
        llm=llm,
        prompt_text="You are a helpful assistant.",
        native_files=native_files,
        output_schema=output_schema,
        file_loader=file_loader,
    )


def make_bundle(tmp_path: Path, roles: dict[str, str]) -> ExampleBundle:
    """Create an ExampleBundle with real temp files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = {}
    for role, content in roles.items():
        ext = ".json" if role.startswith("expected") else ".txt"
        p = tmp_path / f"{role}{ext}"
        p.write_text(content, encoding="utf-8")
        files[role] = p
    return ExampleBundle(bundle_id="test_bundle", files=files)


# ── _infer_media_type ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("ext,expected_mime", [
    (".pdf",  "application/pdf"),
    (".png",  "image/png"),
    (".jpg",  "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    (".csv",  "text/csv"),
    (".txt",  "text/plain"),
    (".json", "application/json"),
])
def test_infer_media_type_known(tmp_path, ext, expected_mime):
    p = tmp_path / f"file{ext}"
    p.touch()
    assert _infer_media_type(p) == expected_mime


def test_infer_media_type_unknown_returns_none(tmp_path):
    p = tmp_path / "file.xyz"
    p.touch()
    assert _infer_media_type(p) is None


def test_infer_media_type_case_insensitive(tmp_path):
    p = tmp_path / "file.PDF"
    p.touch()
    assert _infer_media_type(p) == "application/pdf"


# ── _has_file_parts ───────────────────────────────────────────────────────────

def test_has_file_parts_with_file_part(tmp_path):
    p = tmp_path / "f.pdf"
    p.touch()
    assert _has_file_parts([TextPart("hello"), FilePart(path=p)]) is True


def test_has_file_parts_text_only():
    assert _has_file_parts([TextPart("hello"), TextPart("world")]) is False


def test_has_file_parts_plain_string():
    assert _has_file_parts("just a string") is False


# ── FilePart / TextPart dataclasses ───────────────────────────────────────────

def test_file_part_defaults(tmp_path):
    p = tmp_path / "doc.pdf"
    p.touch()
    fp = FilePart(path=p)
    assert fp.media_type is None
    assert fp.file_id is None
    assert fp.type == "file"


def test_text_part_defaults():
    tp = TextPart(text="hello")
    assert tp.type == "text"


# ── _build_user_content (native_files=True) ───────────────────────────────────

def test_build_user_content_native_input_file(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")

    agent.run(input_file=p)

    content = llm.last_user_content
    assert isinstance(content, list)
    file_parts = [x for x in content if isinstance(x, FilePart)]
    assert len(file_parts) == 1
    assert file_parts[0].path == p
    assert file_parts[0].media_type == "application/pdf"


def test_build_user_content_native_input_text(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    agent.run(input_text="Hello world")

    # input_text is always passed as a plain string even in native mode
    content = llm.last_user_content
    assert isinstance(content, list)
    assert any(isinstance(p, TextPart) and "Hello world" in p.text for p in content)


def test_build_user_content_native_input_files_multi(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    p1 = tmp_path / "data.csv"
    p2 = tmp_path / "template.docx"
    p1.write_text("a,b,c", encoding="utf-8")
    p2.write_bytes(b"PK")  # fake docx

    agent.run(input_files={"data": str(p1), "template": str(p2)})

    content = llm.last_user_content
    assert isinstance(content, list)
    file_parts = [x for x in content if isinstance(x, FilePart)]
    assert len(file_parts) == 2
    paths = {fp.path for fp in file_parts}
    assert p1 in paths
    assert p2 in paths


def test_build_user_content_native_extra_context(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)
    p = tmp_path / "doc.txt"
    p.write_text("content", encoding="utf-8")

    agent.run(input_file=p, extra_context="some context")

    content = llm.last_user_content
    assert isinstance(content, list)
    text_parts = [x for x in content if isinstance(x, TextPart)]
    assert any("some context" in tp.text for tp in text_parts)


# ── _build_user_content (native_files=False + explicit loader) ────────────────

def test_build_user_content_text_extraction_explicit_loader(tmp_path):
    from prompt_forge.file_loaders import get_default_loader
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=False, file_loader=get_default_loader())
    p = tmp_path / "doc.txt"
    p.write_text("hello from file", encoding="utf-8")

    agent.run(input_file=p)

    content = llm.last_user_content
    assert isinstance(content, str)
    assert "hello from file" in content


def test_native_files_false_without_loader_falls_back_to_native(tmp_path):
    """native_files=False without an explicit loader should still use native mode."""
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=False)  # no file_loader → native regardless
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")

    agent.run(input_file=p)

    content = llm.last_user_content
    assert isinstance(content, list)
    assert any(isinstance(x, FilePart) for x in content)


# ── _bundle_to_user_content ───────────────────────────────────────────────────

def test_bundle_to_user_content_native(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    bundle = make_bundle(tmp_path, {"input": "hello", "expected_output": '{"k": "v"}'})
    agent.run_bundle(bundle)

    content = llm.last_user_content
    assert isinstance(content, list)
    # expected_output role is excluded
    file_parts = [x for x in content if isinstance(x, FilePart)]
    assert len(file_parts) == 1
    # opening and closing tags present
    text_parts = [x.text for x in content if isinstance(x, TextPart)]
    assert "<input>" in text_parts
    assert "</input>" in text_parts


def test_bundle_to_user_content_excludes_output_role(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    bundle = make_bundle(tmp_path, {
        "input": "hello",
        "expected_output": '{"result": 1}',
    })
    agent.run_bundle(bundle)

    content = llm.last_user_content
    assert isinstance(content, list)
    # Only input file should appear — expected_output must be absent
    assert all(
        "expected_output" not in x.text
        for x in content if isinstance(x, TextPart)
    )


def test_bundle_files_with_string_paths(tmp_path):
    """bundle.files values may be str — native path must handle both str and Path."""
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    p = tmp_path / "input.txt"
    p.write_text("content", encoding="utf-8")
    # Pass path as a string (not a Path object)
    bundle = ExampleBundle(bundle_id="b", files={"input": str(p)})
    agent.run_bundle(bundle)

    content = llm.last_user_content
    file_parts = [x for x in content if isinstance(x, FilePart)]
    assert len(file_parts) == 1
    assert isinstance(file_parts[0].path, Path)


# ── Batch inference fallback ──────────────────────────────────────────────────

def test_batch_falls_back_to_sequential_with_file_parts(tmp_path):
    llm = CapturingLLM(response_text="result")
    agent = make_agent(llm, native_files=True)

    bundles = [
        make_bundle(tmp_path / f"b{i}", {"input": f"text {i}"})
        for i in range(3)
    ]
    for b in bundles:
        (tmp_path / f"b{b.bundle_id}").mkdir(exist_ok=True)

    results = agent.run_bundle_batch(bundles)

    # Sequential: one LLM call per bundle
    assert len(llm.calls) == 3
    assert len(results) == 3


def test_batch_uses_single_call_without_file_parts(tmp_path):
    from prompt_forge.file_loaders import get_default_loader
    llm = CapturingLLM(response_text='<output id="1">r1</output><output id="2">r2</output><output id="3">r3</output>')
    agent = make_agent(llm, native_files=False, file_loader=get_default_loader())

    bundles = [
        make_bundle(tmp_path / f"c{i}", {"input": f"text {i}"})
        for i in range(3)
    ]

    agent.run_bundle_batch(bundles)

    # Single batched LLM call
    assert len(llm.calls) == 1


# ── TrainingConfig exposes native_files ───────────────────────────────────────

def test_training_config_native_files_default():
    from prompt_forge.training.pipeline import TrainingConfig
    assert TrainingConfig().native_files is True


def test_training_config_native_files_set():
    from prompt_forge.training.pipeline import TrainingConfig
    assert TrainingConfig(native_files=True).native_files is True
