"""
Tests for prompt_forge.bundle and prompt_forge.utils.train_val_split
"""

import pytest
from pathlib import Path

from prompt_forge.bundle import (
    BundleSchema,
    BundleCollection,
    ExampleBundle,
    is_output_role,
)


# ── is_output_role ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,expected", [
    ("expected_output", True),
    ("output",          True),
    ("expected",        True),
    ("my_output_file",  True),
    ("expected_result", True),
    ("input",           False),
    ("source",          False),
    ("document",        False),
    ("",                False),
])
def test_is_output_role(role, expected):
    assert is_output_role(role) is expected


def test_is_output_role_case_insensitive():
    assert is_output_role("EXPECTED_OUTPUT") is True
    assert is_output_role("Output") is True


# ── BundleSchema ──────────────────────────────────────────────────────────────

def test_schema_validates_complete_bundle():
    schema = BundleSchema(roles={"input": ".txt", "expected_output": ".json"})
    bundle = ExampleBundle(
        bundle_id="b1",
        files={"input": Path("a.txt"), "expected_output": Path("b.json")},
    )
    assert schema.validate_bundle(bundle) == []


def test_schema_detects_missing_role():
    schema = BundleSchema(roles={"input": ".txt", "expected_output": ".json"})
    bundle = ExampleBundle(bundle_id="b1", files={"input": Path("a.txt")})
    errors = schema.validate_bundle(bundle)
    assert len(errors) == 1
    assert "expected_output" in errors[0]


def test_schema_round_trip():
    schema = BundleSchema(
        roles={"input": ".pdf", "output": ".json"},
        role_descriptions={"input": "Invoice PDF"},
    )
    d = schema.to_dict()
    restored = BundleSchema.from_dict(d)
    assert restored.roles == schema.roles
    assert restored.role_descriptions == schema.role_descriptions


def test_schema_from_dict_missing_descriptions():
    schema = BundleSchema.from_dict({"roles": {"input": ".txt"}})
    assert schema.role_descriptions == {}


# ── ExampleBundle ─────────────────────────────────────────────────────────────

def test_bundle_round_trip():
    bundle = ExampleBundle(
        bundle_id="b1",
        files={"input": Path("/data/a.txt"), "expected_output": Path("/data/b.json")},
        metadata={"source": "test"},
    )
    d = bundle.to_dict()
    restored = ExampleBundle.from_dict(d)
    assert restored.bundle_id == "b1"
    assert restored.files["input"] == Path("/data/a.txt")
    assert restored.metadata == {"source": "test"}


def test_bundle_load_contents(tmp_path):
    txt = tmp_path / "input.txt"
    txt.write_text("hello", encoding="utf-8")
    out = tmp_path / "expected_output.json"
    out.write_text('{"a": 1}', encoding="utf-8")

    bundle = ExampleBundle(
        bundle_id="b1",
        files={"input": txt, "expected_output": out},
    )
    contents = bundle.load_contents()
    assert contents["input"].text == "hello"
    assert '"a"' in contents["expected_output"].text


def test_bundle_load_contents_missing_file(tmp_path):
    bundle = ExampleBundle(
        bundle_id="b1",
        files={"input": tmp_path / "nonexistent.txt"},
    )
    with pytest.raises(Exception):
        bundle.load_contents()


# ── BundleCollection ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_schema():
    return BundleSchema(roles={"input": ".txt", "expected_output": ".txt"})


@pytest.fixture
def collection(simple_schema):
    return BundleCollection(schema=simple_schema)


def make_bundle(bundle_id, input_path=None, output_path=None):
    return ExampleBundle(
        bundle_id=bundle_id,
        files={
            "input": input_path or Path(f"/fake/{bundle_id}_input.txt"),
            "expected_output": output_path or Path(f"/fake/{bundle_id}_output.txt"),
        },
    )


def test_collection_add_valid_bundle(collection):
    collection.add(make_bundle("b1"))
    assert len(collection) == 1


def test_collection_add_invalid_bundle_raises(collection):
    bad = ExampleBundle(bundle_id="bad", files={"input": Path("x.txt")})
    with pytest.raises(ValueError, match="validation failed"):
        collection.add(bad)


def test_collection_len(collection):
    collection.add(make_bundle("b1"))
    collection.add(make_bundle("b2"))
    assert len(collection) == 2


def test_collection_getitem(collection):
    b = make_bundle("b1")
    collection.add(b)
    assert collection["b1"].bundle_id == "b1"


def test_collection_bundles_property(collection):
    collection.add(make_bundle("b1"))
    collection.add(make_bundle("b2"))
    assert len(collection.bundles) == 2
    assert all(isinstance(b, ExampleBundle) for b in collection.bundles)


def test_collection_get_batch(collection):
    for i in range(5):
        collection.add(make_bundle(f"b{i}"))
    batch = collection.get_batch(["b0", "b2", "b4"])
    assert [b.bundle_id for b in batch] == ["b0", "b2", "b4"]


def test_collection_get_batch_ignores_missing_ids(collection):
    collection.add(make_bundle("b1"))
    batch = collection.get_batch(["b1", "nonexistent"])
    assert len(batch) == 1


def test_collection_round_trip(collection):
    collection.add(make_bundle("b1"))
    collection.add(make_bundle("b2"))
    d = collection.to_dict()
    restored = BundleCollection.from_dict(d)
    assert len(restored) == 2
    assert "b1" in {b.bundle_id for b in restored.bundles}


# ── add_from_directory: subdirectory layout ───────────────────────────────────

def test_load_subdir_layout(tmp_path, simple_schema):
    for name in ("ex1", "ex2"):
        subdir = tmp_path / name
        subdir.mkdir()
        (subdir / "input.txt").write_text(f"input {name}")
        (subdir / "expected_output.txt").write_text(f"output {name}")

    col = BundleCollection(schema=simple_schema)
    count = col.add_from_directory(tmp_path)
    assert count == 2
    ids = {b.bundle_id for b in col.bundles}
    assert ids == {"ex1", "ex2"}


def test_load_subdir_layout_skips_incomplete(tmp_path):
    # Use distinct extensions so the fallback can't cross-assign
    schema = BundleSchema(roles={"input": ".txt", "expected_output": ".json"})

    good = tmp_path / "good"
    good.mkdir()
    (good / "input.txt").write_text("in")
    (good / "expected_output.json").write_text('{"a": 1}')

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "input.txt").write_text("in only")
    # missing expected_output.json

    col = BundleCollection(schema=schema)
    count = col.add_from_directory(tmp_path)
    assert count == 1
    assert col.bundles[0].bundle_id == "good"


def test_load_subdir_extension_fallback(tmp_path, simple_schema):
    """Files with the right extension but no role prefix still resolve."""
    subdir = tmp_path / "ex1"
    subdir.mkdir()
    (subdir / "input.txt").write_text("in")
    (subdir / "expected_output.txt").write_text("out")

    col = BundleCollection(schema=simple_schema)
    count = col.add_from_directory(tmp_path)
    assert count == 1


# ── add_from_directory: flat layout ──────────────────────────────────────────

def test_load_flat_layout(tmp_path, simple_schema):
    (tmp_path / "001_input.txt").write_text("input 1")
    (tmp_path / "001_expected_output.txt").write_text("output 1")
    (tmp_path / "002_input.txt").write_text("input 2")
    (tmp_path / "002_expected_output.txt").write_text("output 2")

    col = BundleCollection(schema=simple_schema)
    count = col.add_from_directory(tmp_path)
    assert count == 2
    assert {b.bundle_id for b in col.bundles} == {"001", "002"}


def test_load_flat_layout_ignores_files_without_underscore(tmp_path, simple_schema):
    (tmp_path / "readme.txt").write_text("ignore me")
    col = BundleCollection(schema=simple_schema)
    count = col.add_from_directory(tmp_path)
    assert count == 0


def test_add_from_directory_not_a_directory_raises(simple_schema, tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    col = BundleCollection(schema=simple_schema)
    with pytest.raises(ValueError, match="Not a directory"):
        col.add_from_directory(f)


