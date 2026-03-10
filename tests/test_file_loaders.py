"""
Tests for prompt_forge.file_loaders
"""

import json
import pytest
from pathlib import Path

from prompt_forge.file_loaders import FileLoader, FileContent, get_default_loader


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def loader():
    return get_default_loader()


@pytest.fixture
def tmp_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world", encoding="utf-8")
    return f


@pytest.fixture
def tmp_json(tmp_path):
    f = tmp_path / "sample.json"
    f.write_text(json.dumps({"key": "value", "num": 42}), encoding="utf-8")
    return f


@pytest.fixture
def tmp_csv(tmp_path):
    f = tmp_path / "sample.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    return f


# ── FileContent ───────────────────────────────────────────────────────────────

def test_file_content_fields(loader, tmp_txt):
    content = loader.load(tmp_txt)
    assert isinstance(content, FileContent)
    assert content.text == "hello world"
    assert content.file_type == ".txt"
    assert content.source_path == str(tmp_txt)
    assert content.metadata["size_bytes"] == tmp_txt.stat().st_size


# ── Text loader ───────────────────────────────────────────────────────────────

def test_load_txt(loader, tmp_txt):
    assert loader.load(tmp_txt).text == "hello world"


def test_load_md(loader, tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\nBody text", encoding="utf-8")
    assert loader.load(f).text == "# Title\nBody text"


def test_load_txt_unicode_replacement(loader, tmp_path):
    f = tmp_path / "bad.txt"
    f.write_bytes(b"hello \xff world")  # invalid UTF-8 byte
    content = loader.load(f)
    assert "hello" in content.text
    assert "world" in content.text  # replacement char, not exception


# ── JSON loader ───────────────────────────────────────────────────────────────

def test_load_json_pretty_prints(loader, tmp_json):
    content = loader.load(tmp_json)
    parsed = json.loads(content.text)
    assert parsed["key"] == "value"
    assert parsed["num"] == 42
    assert "\n" in content.text  # pretty-printed


def test_load_json_preserves_unicode(loader, tmp_path):
    f = tmp_path / "unicode.json"
    f.write_text(json.dumps({"city": "München"}), encoding="utf-8")
    parsed = json.loads(loader.load(f).text)
    assert parsed["city"] == "München"


def test_load_invalid_json_raises(loader, tmp_path):
    f = tmp_path / "broken.json"
    f.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(Exception):  # json.JSONDecodeError
        loader.load(f)


# ── CSV loader ────────────────────────────────────────────────────────────────

def test_load_csv_pipe_separated(loader, tmp_csv):
    text = loader.load(tmp_csv).text
    assert "name | age" in text
    assert "Alice | 30" in text
    assert "Bob | 25" in text


def test_load_csv_empty(loader, tmp_path):
    f = tmp_path / "empty.csv"
    f.write_text("", encoding="utf-8")
    assert loader.load(f).text == ""


def test_load_tsv(loader, tmp_path):
    f = tmp_path / "data.tsv"
    f.write_text("col1\tcol2\nval1\tval2", encoding="utf-8")
    text = loader.load(f).text
    assert "col1" in text
    assert "val1" in text


# ── Extension handling ────────────────────────────────────────────────────────

def test_unsupported_extension_raises(loader, tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data")
    with pytest.raises(ValueError, match="No loader registered"):
        loader.load(f)


def test_load_accepts_string_path(loader, tmp_txt):
    content = loader.load(str(tmp_txt))
    assert content.text == "hello world"


def test_extension_is_case_insensitive(loader, tmp_path):
    f = tmp_path / "UPPER.TXT"
    f.write_text("uppercase ext", encoding="utf-8")
    content = loader.load(f)
    assert content.text == "uppercase ext"


# ── Custom loader registration ────────────────────────────────────────────────

def test_register_custom_loader(tmp_path):
    loader = FileLoader()
    loader.register(".parquet", lambda path: f"parquet:{path.name}")
    f = tmp_path / "data.parquet"
    f.write_bytes(b"\x00")  # dummy content
    content = loader.load(f)
    assert content.text == f"parquet:{f.name}"


def test_register_without_dot(tmp_path):
    loader = FileLoader()
    loader.register("xyz", lambda path: "xyz content")
    f = tmp_path / "file.xyz"
    f.write_bytes(b"\x00")
    content = loader.load(f)
    assert content.text == "xyz content"


def test_register_overrides_default(tmp_path):
    loader = FileLoader()
    loader.register(".txt", lambda path: "overridden")
    f = tmp_path / "file.txt"
    f.write_text("original", encoding="utf-8")
    assert loader.load(f).text == "overridden"


def test_custom_loaders_are_isolated():
    """Registering on one instance does not affect another."""
    a = FileLoader()
    b = FileLoader()
    a.register(".xyz", lambda p: "from a")
    assert ".xyz" not in b.supported_extensions


# ── can_load ──────────────────────────────────────────────────────────────────

def test_can_load_known_extension(loader):
    assert loader.can_load("file.txt") is True
    assert loader.can_load("file.json") is True
    assert loader.can_load("file.csv") is True


def test_can_load_unknown_extension(loader):
    assert loader.can_load("file.xyz") is False


def test_can_load_accepts_path_object(loader):
    assert loader.can_load(Path("file.txt")) is True


# ── supported_extensions ─────────────────────────────────────────────────────

def test_supported_extensions_sorted(loader):
    exts = loader.supported_extensions
    assert exts == sorted(exts)


def test_supported_extensions_includes_defaults(loader):
    exts = loader.supported_extensions
    for ext in (".txt", ".json", ".csv", ".md"):
        assert ext in exts


# ── get_default_loader ────────────────────────────────────────────────────────

def test_get_default_loader_returns_new_instance():
    a = get_default_loader()
    b = get_default_loader()
    assert a is not b


def test_get_default_loader_has_all_builtins():
    loader = get_default_loader()
    for ext in (".txt", ".json", ".csv", ".tsv", ".md", ".py"):
        assert loader.can_load(f"file{ext}")
