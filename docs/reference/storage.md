# Storage Reference

Storage backends persist prompt versions and training state. The `ProjectStore` protocol defines the interface — any conforming class can be used.

---

## PromptVersion

A single versioned snapshot of a prompt.

```python
@dataclasses.dataclass
class PromptVersion:
    version: int
    prompt_text: str
    created_at: str           # ISO 8601
    parent_version: int | None = None
    training_log_entry: str | None = None
    eval_score: float | None = None
    eval_details: dict | None = None
    output_schema: dict | None = None
    metadata: dict = {}
```

### Fields

| Field | Description |
|-------|-------------|
| `version` | Monotonically increasing integer. Version 1 is the seed prompt. |
| `prompt_text` | Full prompt text for this version. |
| `created_at` | ISO 8601 timestamp. |
| `parent_version` | Version this was derived from. `None` for the seed. |
| `training_log_entry` | Key learnings from the iteration that produced this version. `"[CONSOLIDATION]"` prefix marks consolidated versions. |
| `eval_score` | Validation score when this version was accepted. `None` if no evaluator was active. |
| `eval_details` | Full `BatchEvalResult.to_dict()` from the accepting iteration. |
| `output_schema` | JSON Schema dict if the task requires structured output. Carried forward across versions. |
| `metadata` | Arbitrary dict. Contains `{"batch_ids": [...], "iteration": N}` for trained versions and `{"consolidation": True, "source_version": N}` for consolidated versions. |

---

## ProjectStore (protocol)

```python
class ProjectStore(Protocol):
    def save_prompt_version(self, version: PromptVersion) -> None: ...
    def get_prompt_version(self, version: int) -> PromptVersion | None: ...
    def get_latest_prompt(self) -> PromptVersion | None: ...
    def list_versions(self) -> list[PromptVersion]: ...
    def save_project_config(self, config: dict) -> None: ...
    def load_project_config(self) -> dict | None: ...
    def save_training_state(self, state: dict) -> None: ...
    def load_training_state(self) -> dict | None: ...
```

Any class implementing these methods satisfies the protocol — you can use `isinstance(store, ProjectStore)` to check at runtime.

---

## FileSystemStore

JSON files on disk. The default backend — portable, human-readable, git-friendly.

```python
from prompt_forge import FileSystemStore

store = FileSystemStore("my_project/")
```

### Directory layout

```
my_project/
    config.json
    training_state.json
    prompts/
        v0001.json    # seed prompt
        v0002.json    # first trained version
        v0003.json    # consolidated
        ...
```

Each `vNNNN.json` file is a `PromptVersion.to_dict()` serialization. The directory is created on first use.

### Constructor

```python
FileSystemStore(project_dir: str | Path)
```

### Methods

All `ProjectStore` protocol methods are implemented. Additionally:

#### `list_versions() → list[PromptVersion]`

Returns all versions sorted by version number ascending.

### Git integration

The `prompts/` directory is plain JSON — commit it to track the prompt history alongside your code. The `training_state.json` file contains transient state (training log, token counter) and can be gitignored if you don't need resumption across machines.

---

## SQLAlchemyStore

SQL-backed storage supporting any SQLAlchemy-compatible database.

```python
from prompt_forge import SQLAlchemyStore

# PostgreSQL
store = SQLAlchemyStore("postgresql+psycopg2://user:pass@host/db")

# Azure SQL Server
store = SQLAlchemyStore(
    "mssql+pyodbc://user:pass@server.database.windows.net/db"
    "?driver=ODBC+Driver+18+for+SQL+Server"
)

# SQLite file
store = SQLAlchemyStore("sqlite:///my_project.db")

# SQLite in-memory (useful for tests)
store = SQLAlchemyStore("sqlite://")
```

Requires the `sqlalchemy` extra:

```bash
pip install "prompt-forge[sqlalchemy]"
```

### Constructor

```python
SQLAlchemyStore(
    connection_string: str,
    project_name: str = "default",
    **engine_kwargs,
)
```

| Parameter | Description |
|-----------|-------------|
| `connection_string` | Standard SQLAlchemy database URL. |
| `project_name` | Logical namespace within the database. Allows multiple projects to share one database. |
| `**engine_kwargs` | Forwarded to `sqlalchemy.create_engine()`. |

### Schema

Three tables are created automatically on first use (prefixed with `pf_` to avoid conflicts):

- `pf_prompt_versions` — primary key `(project_name, version)`
- `pf_project_config` — primary key `project_name`
- `pf_training_state` — primary key `project_name`

### Context manager

```python
with SQLAlchemyStore("postgresql://...") as store:
    store.save_prompt_version(...)
# connection pool is released on exit
```

`dispose()` can also be called manually.

---

## Setting a seed prompt

Before training, save a version 1 prompt:

```python
from prompt_forge import FileSystemStore, PromptVersion
from datetime import datetime, timezone

store = FileSystemStore("my_project/")
store.save_prompt_version(PromptVersion(
    version=1,
    prompt_text="You are an expert invoice extractor...",
    created_at=datetime.now(timezone.utc).isoformat(),
))
```

Or via `Project.set_seed_prompt()` which wraps this pattern.

---

## Inspecting version history

```python
for v in store.list_versions():
    consolidation = " [CONSOLIDATION]" if v.metadata.get("consolidation") else ""
    score = f" score={v.eval_score:.3f}" if v.eval_score is not None else ""
    print(f"v{v.version:04d}{consolidation}{score}: {v.created_at[:10]}")
```

---

## Implementing a custom backend

Implement all eight `ProjectStore` methods. The protocol uses `@runtime_checkable` so you can verify conformance:

```python
from prompt_forge.storage.project_store import ProjectStore

class RedisStore:
    def save_prompt_version(self, version): ...
    def get_prompt_version(self, version): ...
    def get_latest_prompt(self): ...
    def list_versions(self): ...
    def save_project_config(self, config): ...
    def load_project_config(self): ...
    def save_training_state(self, state): ...
    def load_training_state(self): ...

assert isinstance(RedisStore(), ProjectStore)  # passes
```

---

## Design notes

- **`FileSystemStore` is recommended** for most use cases. The JSON files are easily auditable and the directory can be version-controlled with git.
- **`SQLAlchemyStore`** is useful when prompts are part of a larger application database or when you need multi-user access. The `project_name` parameter lets you namespace projects within a shared database rather than maintaining separate database schemas.
- **`SQLiteStore`** (not documented above) is deprecated. Replace any existing usage with `SQLAlchemyStore("sqlite:///path/to/db")`.
- **`training_state.json`** stores the training log (history of past iterations) so the optimizer has context on previous changes, plus a running token counter. It is updated after every iteration when `auto_save=True`. If training is interrupted, the next call to `train()` restores this state automatically.
