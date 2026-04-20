"""
Tests for variadic role support in BundleSchema, ExampleBundle, BundleCollection,
and InferenceAgent._bundle_to_user_content.
"""

from pathlib import Path

import pytest

from prompt_forge.bundle import BundleSchema, ExampleBundle, BundleCollection
from prompt_forge.llm.client import LLMMessage, LLMResponse, TextPart, FilePart, MessageContent
from prompt_forge.inference.agent import InferenceAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

class CapturingLLM:
    def __init__(self, response_text: str = "output"):
        self.calls: list[list[LLMMessage]] = []
        self.response_text = response_text

    def complete(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(text=self.response_text, usage={"input_tokens": 10, "output_tokens": 5})

    @property
    def last_user_content(self) -> MessageContent:
        return self.calls[-1][-1].content


def make_agent(llm, native_files: bool = True, file_loader=None) -> InferenceAgent:
    return InferenceAgent(llm=llm, prompt_text="You are helpful.", native_files=native_files, file_loader=file_loader)


# ── BundleSchema.validate_bundle ──────────────────────────────────────────────

def test_variadic_role_absent_passes_validation(tmp_path):
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf", "expected_output": ".json"},
        variadic_roles={"attachments"},
    )
    mail = tmp_path / "mail.txt"
    mail.write_text("hi")
    bundle = ExampleBundle(bundle_id="b", files={"mail": mail, "expected_output": tmp_path / "out.json"})
    assert schema.validate_bundle(bundle) == []


def test_required_role_absent_fails_validation(tmp_path):
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf"},
        variadic_roles={"attachments"},
    )
    bundle = ExampleBundle(bundle_id="b", files={"attachments": []})
    errors = schema.validate_bundle(bundle)
    assert any("mail" in e for e in errors)


# ── BundleSchema extension validation ────────────────────────────────────────

def test_validate_bundle_wrong_extension_fails(tmp_path):
    schema = BundleSchema(roles={"input": ".pdf"})
    p = tmp_path / "input.txt"
    p.write_text("oops")
    bundle = ExampleBundle(bundle_id="b", files={"input": p})
    errors = schema.validate_bundle(bundle)
    assert any("input" in e and ".txt" in e for e in errors)


def test_validate_bundle_correct_extension_passes(tmp_path):
    schema = BundleSchema(roles={"input": ".pdf"})
    p = tmp_path / "input.pdf"
    p.write_bytes(b"%PDF")
    bundle = ExampleBundle(bundle_id="b", files={"input": p})
    assert schema.validate_bundle(bundle) == []


def test_validate_bundle_extension_case_insensitive(tmp_path):
    schema = BundleSchema(roles={"input": ".pdf"})
    p = tmp_path / "input.PDF"
    p.write_bytes(b"%PDF")
    bundle = ExampleBundle(bundle_id="b", files={"input": p})
    assert schema.validate_bundle(bundle) == []


def test_validate_bundle_variadic_wrong_extension(tmp_path):
    schema = BundleSchema(roles={"attachments": ".pdf"}, variadic_roles={"attachments"})
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.txt"  # wrong
    p1.write_bytes(b"%PDF")
    p2.write_text("oops")
    bundle = ExampleBundle(bundle_id="b", files={"attachments": [p1, p2]})
    errors = schema.validate_bundle(bundle)
    assert any(".txt" in e for e in errors)
    assert len(errors) == 1  # only b.txt is wrong


# ── BundleSchema serialisation ────────────────────────────────────────────────

def test_schema_to_dict_includes_variadic_roles():
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf"},
        variadic_roles={"attachments"},
    )
    d = schema.to_dict()
    assert "attachments" in d["variadic_roles"]


def test_schema_roundtrip():
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf", "expected_output": ".json"},
        variadic_roles={"attachments"},
    )
    restored = BundleSchema.from_dict(schema.to_dict())
    assert restored.variadic_roles == {"attachments"}
    assert restored.roles == schema.roles


def test_schema_from_dict_missing_variadic_roles_key():
    d = {"roles": {"input": ".txt"}}
    schema = BundleSchema.from_dict(d)
    assert schema.variadic_roles == set()


# ── ExampleBundle with list[Path] ─────────────────────────────────────────────

def test_bundle_files_list_path(tmp_path):
    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    p1.write_bytes(b"PDF1")
    p2.write_bytes(b"PDF2")
    bundle = ExampleBundle(bundle_id="b", files={"attachments": [p1, p2]})
    assert isinstance(bundle.files["attachments"], list)
    assert len(bundle.files["attachments"]) == 2


def test_bundle_to_dict_variadic_role():
    p1 = Path("/tmp/a.pdf")
    p2 = Path("/tmp/b.pdf")
    bundle = ExampleBundle(bundle_id="b", files={"attachments": [p1, p2]})
    d = bundle.to_dict()
    assert d["files"]["attachments"] == ["/tmp/a.pdf", "/tmp/b.pdf"]


def test_bundle_from_dict_list_value():
    d = {
        "bundle_id": "b",
        "files": {"attachments": ["/tmp/a.pdf", "/tmp/b.pdf"]},
        "metadata": {},
    }
    bundle = ExampleBundle.from_dict(d)
    assert isinstance(bundle.files["attachments"], list)
    assert bundle.files["attachments"] == [Path("/tmp/a.pdf"), Path("/tmp/b.pdf")]


def test_bundle_from_dict_scalar_value():
    d = {"bundle_id": "b", "files": {"input": "/tmp/x.txt"}, "metadata": {}}
    bundle = ExampleBundle.from_dict(d)
    assert bundle.files["input"] == Path("/tmp/x.txt")


# ── ExampleBundle.load_contents with variadic roles ───────────────────────────

def test_load_contents_variadic_concatenates(tmp_path):
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("first", encoding="utf-8")
    p2.write_text("second", encoding="utf-8")
    bundle = ExampleBundle(bundle_id="b", files={"docs": [p1, p2]})
    contents = bundle.load_contents()
    assert "first" in contents["docs"].text
    assert "second" in contents["docs"].text
    assert contents["docs"].metadata["file_count"] == 2


def test_load_contents_variadic_empty_list(tmp_path):
    bundle = ExampleBundle(bundle_id="b", files={"docs": []})
    contents = bundle.load_contents()
    assert contents["docs"].text == ""
    assert contents["docs"].metadata["file_count"] == 0


# ── BundleCollection._load_subdir_layout with variadic roles ─────────────────

def test_subdir_layout_collects_variadic_files(tmp_path):
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf"},
        variadic_roles={"attachments"},
    )
    subdir = tmp_path / "example_001"
    subdir.mkdir()
    (subdir / "mail.txt").write_text("hello")
    (subdir / "attachments_1.pdf").write_bytes(b"PDF1")
    (subdir / "attachments_2.pdf").write_bytes(b"PDF2")

    collection = BundleCollection(schema=schema)
    n = collection.add_from_directory(tmp_path)

    assert n == 1
    bundle = collection["example_001"]
    assert isinstance(bundle.files["attachments"], list)
    assert len(bundle.files["attachments"]) == 2


def test_subdir_layout_variadic_absent_still_loads(tmp_path):
    schema = BundleSchema(
        roles={"mail": ".txt", "attachments": ".pdf"},
        variadic_roles={"attachments"},
    )
    subdir = tmp_path / "example_001"
    subdir.mkdir()
    (subdir / "mail.txt").write_text("hello")
    # No attachments

    collection = BundleCollection(schema=schema)
    n = collection.add_from_directory(tmp_path)

    assert n == 1
    bundle = collection["example_001"]
    assert "attachments" not in bundle.files


def test_subdir_layout_non_variadic_takes_first(tmp_path):
    schema = BundleSchema(
        roles={"input": ".txt"},
        variadic_roles=set(),
    )
    subdir = tmp_path / "example_001"
    subdir.mkdir()
    (subdir / "input.txt").write_text("hello")

    collection = BundleCollection(schema=schema)
    n = collection.add_from_directory(tmp_path)
    assert n == 1
    assert isinstance(collection["example_001"].files["input"], Path)



# ── InferenceAgent._bundle_to_user_content with variadic roles ────────────────

def test_bundle_to_user_content_native_variadic(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    p1 = tmp_path / "a.pdf"
    p2 = tmp_path / "b.pdf"
    mail = tmp_path / "mail.txt"
    p1.write_bytes(b"PDF1")
    p2.write_bytes(b"PDF2")
    mail.write_text("hello")

    bundle = ExampleBundle(bundle_id="b", files={
        "mail": mail,
        "attachments": [p1, p2],
    })
    agent.run_bundle(bundle)

    content = llm.last_user_content
    assert isinstance(content, list)
    file_parts = [x for x in content if isinstance(x, FilePart)]
    # 1 mail + 2 attachments
    assert len(file_parts) == 3
    paths = {fp.path for fp in file_parts}
    assert p1 in paths
    assert p2 in paths
    assert mail in paths


def test_bundle_to_user_content_native_variadic_xml_tags(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    p = tmp_path / "a.pdf"
    p.write_bytes(b"PDF")
    bundle = ExampleBundle(bundle_id="b", files={"attachments": [p]})
    agent.run_bundle(bundle)

    content = llm.last_user_content
    text_parts = [x.text for x in content if isinstance(x, TextPart)]
    assert "<attachments>" in text_parts
    assert "</attachments>" in text_parts


def test_bundle_to_user_content_native_variadic_empty_list(tmp_path):
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=True)

    mail = tmp_path / "mail.txt"
    mail.write_text("hello")
    bundle = ExampleBundle(bundle_id="b", files={"mail": mail, "attachments": []})
    agent.run_bundle(bundle)

    content = llm.last_user_content
    file_parts = [x for x in content if isinstance(x, FilePart)]
    # Only the mail file — empty list emits tags but no FilePart
    assert len(file_parts) == 1


def test_bundle_to_user_content_text_mode_variadic(tmp_path):
    from prompt_forge.file_loaders import get_default_loader
    llm = CapturingLLM()
    agent = make_agent(llm, native_files=False, file_loader=get_default_loader())

    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("first doc", encoding="utf-8")
    p2.write_text("second doc", encoding="utf-8")
    bundle = ExampleBundle(bundle_id="b", files={"docs": [p1, p2]})
    agent.run_bundle(bundle)

    content = llm.last_user_content
    assert isinstance(content, str)
    assert "first doc" in content
    assert "second doc" in content


# ── Project.set_bundle_schema variadic parameter ──────────────────────────────

def test_set_bundle_schema_variadic_kwarg(tmp_path):
    from prompt_forge import Project

    class DummyLLM:
        def complete(self, messages, **kwargs):
            return LLMResponse(text="ok", usage={})

    project = Project("test", llm=DummyLLM(), project_dir=str(tmp_path))
    project.set_bundle_schema(
        mail=".txt",
        attachments=".pdf",
        expected_output=".json",
        variadic=["attachments"],
    )
    assert project._schema.variadic_roles == {"attachments"}
