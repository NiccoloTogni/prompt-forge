"""
Tests for prompt_forge.storage.project_store
"""

import json
import warnings
import pytest
from datetime import datetime, timezone

from prompt_forge.storage.project_store import (
    FileSystemStore,
    SQLAlchemyStore,
    SQLiteStore,
    PromptVersion,
    ProjectStore,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_version(version=1, prompt_text="test prompt", **kwargs) -> PromptVersion:
    return PromptVersion(
        version=version,
        prompt_text=prompt_text,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_version=kwargs.get("parent_version"),
        training_log_entry=kwargs.get("training_log_entry", ""),
        eval_score=kwargs.get("eval_score"),
        eval_details=kwargs.get("eval_details"),
        output_schema=kwargs.get("output_schema"),
        metadata=kwargs.get("metadata", {}),
    )


@pytest.fixture
def fs_store(tmp_path):
    return FileSystemStore(tmp_path / "project")


@pytest.fixture
def sa_store():
    return SQLAlchemyStore("sqlite://")


# ── PromptVersion ─────────────────────────────────────────────────────────────

def test_prompt_version_round_trip():
    v = make_version(version=3, eval_score=0.9, metadata={"key": "val"})
    restored = PromptVersion.from_dict(v.to_dict())
    assert restored.version == 3
    assert restored.prompt_text == "test prompt"
    assert restored.eval_score == pytest.approx(0.9)
    assert restored.metadata == {"key": "val"}


def test_prompt_version_from_dict_ignores_unknown_fields():
    d = make_version().to_dict()
    d["unknown_field"] = "ignored"
    v = PromptVersion.from_dict(d)
    assert v.version == 1


def test_prompt_version_optional_fields_default_none():
    v = make_version()
    assert v.parent_version is None
    assert v.eval_score is None
    assert v.eval_details is None
    assert v.output_schema is None


# ── Protocol compliance ───────────────────────────────────────────────────────

def test_filesystem_store_implements_protocol(fs_store):
    assert isinstance(fs_store, ProjectStore)


def test_sqlalchemy_store_implements_protocol(sa_store):
    assert isinstance(sa_store, ProjectStore)


# ── Shared behaviour (parametrize over both backends) ─────────────────────────

@pytest.fixture(params=["fs", "sqlalchemy"])
def store(request, tmp_path):
    if request.param == "fs":
        yield FileSystemStore(tmp_path / "project")
    else:
        yield SQLAlchemyStore("sqlite://")


def test_get_latest_prompt_empty(store):
    assert store.get_latest_prompt() is None


def test_get_prompt_version_missing(store):
    assert store.get_prompt_version(99) is None


def test_list_versions_empty(store):
    assert store.list_versions() == []


def test_save_and_retrieve_version(store):
    store.save_prompt_version(make_version(version=1, prompt_text="hello"))
    retrieved = store.get_prompt_version(1)
    assert retrieved is not None
    assert retrieved.version == 1
    assert retrieved.prompt_text == "hello"


def test_get_latest_is_highest_version(store):
    store.save_prompt_version(make_version(version=1))
    store.save_prompt_version(make_version(version=2))
    store.save_prompt_version(make_version(version=3))
    assert store.get_latest_prompt().version == 3


def test_list_versions_returns_ascending_order(store):
    store.save_prompt_version(make_version(version=3))
    store.save_prompt_version(make_version(version=1))
    store.save_prompt_version(make_version(version=2))
    assert [v.version for v in store.list_versions()] == [1, 2, 3]


def test_save_overwrites_existing_version(store):
    store.save_prompt_version(make_version(version=1, prompt_text="original"))
    store.save_prompt_version(make_version(version=1, prompt_text="updated"))
    assert store.get_prompt_version(1).prompt_text == "updated"


def test_save_and_retrieve_eval_score(store):
    store.save_prompt_version(make_version(version=1, eval_score=0.85))
    assert store.get_prompt_version(1).eval_score == pytest.approx(0.85)


def test_save_and_retrieve_eval_details(store):
    details = {"mean_score": 0.9, "num_examples": 5}
    store.save_prompt_version(make_version(version=1, eval_details=details))
    assert store.get_prompt_version(1).eval_details["mean_score"] == pytest.approx(0.9)


def test_save_and_retrieve_output_schema(store):
    schema = {"type": "object", "properties": {"field": "string"}}
    store.save_prompt_version(make_version(version=1, output_schema=schema))
    assert store.get_prompt_version(1).output_schema == schema


def test_save_and_retrieve_metadata(store):
    store.save_prompt_version(make_version(version=1, metadata={"batch_ids": ["a", "b"], "iteration": 3}))
    meta = store.get_prompt_version(1).metadata
    assert meta["iteration"] == 3
    assert meta["batch_ids"] == ["a", "b"]


def test_save_and_retrieve_unicode_prompt(store):
    store.save_prompt_version(make_version(version=1, prompt_text="Extracte données: München, €42"))
    assert store.get_prompt_version(1).prompt_text == "Extracte données: München, €42"


def test_project_config_round_trip(store):
    store.save_project_config({"name": "myproject", "context": "some domain", "schema": None})
    loaded = store.load_project_config()
    assert loaded["name"] == "myproject"
    assert loaded["context"] == "some domain"


def test_load_project_config_missing(store):
    assert store.load_project_config() is None


def test_project_config_overwrite(store):
    store.save_project_config({"name": "v1"})
    store.save_project_config({"name": "v2"})
    assert store.load_project_config()["name"] == "v2"


def test_training_state_round_trip(store):
    store.save_training_state({"last_iteration": 5, "total_tokens_used": 1000})
    loaded = store.load_training_state()
    assert loaded["last_iteration"] == 5
    assert loaded["total_tokens_used"] == 1000


def test_load_training_state_missing(store):
    assert store.load_training_state() is None


def test_training_state_overwrite(store):
    store.save_training_state({"last_iteration": 1})
    store.save_training_state({"last_iteration": 5})
    assert store.load_training_state()["last_iteration"] == 5


# ── FileSystemStore specifics ─────────────────────────────────────────────────

def test_filesystem_creates_directories(tmp_path):
    FileSystemStore(tmp_path / "nested" / "project")
    assert (tmp_path / "nested" / "project").exists()
    assert (tmp_path / "nested" / "project" / "prompts").exists()


def test_filesystem_version_filename_format(tmp_path):
    store = FileSystemStore(tmp_path / "p")
    store.save_prompt_version(make_version(version=5))
    assert (tmp_path / "p" / "prompts" / "v0005.json").exists()


def test_filesystem_version_file_is_valid_json(tmp_path):
    store = FileSystemStore(tmp_path / "p")
    store.save_prompt_version(make_version(version=1, prompt_text="hello"))
    content = json.loads((tmp_path / "p" / "prompts" / "v0001.json").read_text())
    assert content["prompt_text"] == "hello"


# ── SQLAlchemyStore specifics ─────────────────────────────────────────────────

def test_sqlalchemy_context_manager():
    with SQLAlchemyStore("sqlite://") as store:
        store.save_prompt_version(make_version(version=1))
        assert store.get_prompt_version(1) is not None


def test_sqlalchemy_project_name_isolation():
    """Two projects in the same DB must not see each other's data."""
    store_a = SQLAlchemyStore("sqlite:///file:pf_test?mode=memory&cache=shared&uri=true", project_name="project_a")
    store_b = SQLAlchemyStore("sqlite:///file:pf_test?mode=memory&cache=shared&uri=true", project_name="project_b")
    store_a.save_prompt_version(make_version(version=1, prompt_text="from A"))
    assert store_b.get_prompt_version(1) is None
    assert store_b.list_versions() == []


def test_sqlalchemy_idempotent_schema_init(tmp_path):
    """Creating two stores against the same SQLite file should not raise."""
    db = f"sqlite:///{tmp_path / 'test.db'}"
    s1 = SQLAlchemyStore(db)
    s1.save_prompt_version(make_version(version=1))
    s2 = SQLAlchemyStore(db)
    assert s2.get_prompt_version(1) is not None


# ── SQLiteStore deprecation ───────────────────────────────────────────────────

def test_sqlite_store_emits_deprecation_warning(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = SQLiteStore(tmp_path / "legacy.db")
        store.close()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_sqlite_store_still_functional(tmp_path):
    """Deprecated but must still work for backwards compatibility."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with SQLiteStore(tmp_path / "legacy.db") as store:
            store.save_prompt_version(make_version(version=1, prompt_text="legacy"))
            assert store.get_prompt_version(1).prompt_text == "legacy"
