"""
Project storage backends.

Two implementations:
    - FileSystemStore: JSON files on disk (default, portable)
    - SQLiteStore: SQLite database (optional, better for querying)
"""

import json
import sqlite3
import dataclasses
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclasses.dataclass
class PromptVersion:
    """A single versioned prompt snapshot."""

    version: int
    prompt_text: str
    created_at: str  # ISO format
    parent_version: int | None = None
    training_log_entry: str | None = None  # What was learned in this iteration
    eval_score: float | None = None
    eval_details: dict | None = None
    output_schema: dict | None = None  # JSON schema if task requires structured output
    metadata: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PromptVersion:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@runtime_checkable
class ProjectStore(Protocol):
    """Protocol for project storage backends."""

    def save_prompt_version(self, version: PromptVersion) -> None: ...
    def get_prompt_version(self, version: int) -> PromptVersion | None: ...
    def get_latest_prompt(self) -> PromptVersion | None: ...
    def list_versions(self) -> list[PromptVersion]: ...
    def save_project_config(self, config: dict) -> None: ...
    def load_project_config(self) -> dict | None: ...
    def save_training_state(self, state: dict) -> None: ...
    def load_training_state(self) -> dict | None: ...


class FileSystemStore:
    """
    JSON-file-based storage.

    Layout:
        project_dir/
            config.json
            training_state.json
            prompts/
                v001.json
                v002.json
                ...
    """

    def __init__(self, project_dir: str | Path):
        self.project_dir = Path(project_dir)
        self.prompts_dir = self.project_dir / "prompts"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.prompts_dir.mkdir(exist_ok=True)

    def save_prompt_version(self, version: PromptVersion) -> None:
        path = self.prompts_dir / f"v{version.version:04d}.json"
        path.write_text(
            json.dumps(version.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_prompt_version(self, version: int) -> PromptVersion | None:
        path = self.prompts_dir / f"v{version:04d}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PromptVersion.from_dict(data)

    def get_latest_prompt(self) -> PromptVersion | None:
        versions = self.list_versions()
        if not versions:
            return None
        return versions[-1]

    def list_versions(self) -> list[PromptVersion]:
        versions = []
        for path in sorted(self.prompts_dir.glob("v*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            versions.append(PromptVersion.from_dict(data))
        return versions

    def save_project_config(self, config: dict) -> None:
        path = self.project_dir / "config.json"
        path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_project_config(self) -> dict | None:
        path = self.project_dir / "config.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_training_state(self, state: dict) -> None:
        path = self.project_dir / "training_state.json"
        path.write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_training_state(self) -> dict | None:
        path = self.project_dir / "training_state.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


class SQLiteStore:
    """
    SQLite-backed storage. Better for querying prompt history and metrics.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS prompt_versions (
                version INTEGER PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parent_version INTEGER,
                training_log_entry TEXT,
                eval_score REAL,
                eval_details TEXT,
                output_schema TEXT,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS project_config (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                config TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state TEXT NOT NULL
            );
        """)
        self._conn.commit()
        # Migrate existing databases that may lack the output_schema column
        try:
            self._conn.execute("ALTER TABLE prompt_versions ADD COLUMN output_schema TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

    def save_prompt_version(self, version: PromptVersion) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO prompt_versions
               (version, prompt_text, created_at, parent_version,
                training_log_entry, eval_score, eval_details, output_schema, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                version.version,
                version.prompt_text,
                version.created_at,
                version.parent_version,
                version.training_log_entry,
                version.eval_score,
                json.dumps(version.eval_details) if version.eval_details else None,
                json.dumps(version.output_schema) if version.output_schema is not None else None,
                json.dumps(version.metadata),
            ),
        )
        self._conn.commit()

    def get_prompt_version(self, version: int) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT * FROM prompt_versions WHERE version = ?", (version,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def get_latest_prompt(self) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT * FROM prompt_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def list_versions(self) -> list[PromptVersion]:
        rows = self._conn.execute(
            "SELECT * FROM prompt_versions ORDER BY version ASC"
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def save_project_config(self, config: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO project_config (id, config) VALUES (1, ?)",
            (json.dumps(config, ensure_ascii=False),),
        )
        self._conn.commit()

    def load_project_config(self) -> dict | None:
        row = self._conn.execute(
            "SELECT config FROM project_config WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["config"])

    def save_training_state(self, state: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO training_state (id, state) VALUES (1, ?)",
            (json.dumps(state, ensure_ascii=False),),
        )
        self._conn.commit()

    def load_training_state(self) -> dict | None:
        row = self._conn.execute(
            "SELECT state FROM training_state WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["state"])

    def _row_to_version(self, row: sqlite3.Row) -> PromptVersion:
        return PromptVersion(
            version=row["version"],
            prompt_text=row["prompt_text"],
            created_at=row["created_at"],
            parent_version=row["parent_version"],
            training_log_entry=row["training_log_entry"],
            eval_score=row["eval_score"],
            eval_details=json.loads(row["eval_details"]) if row["eval_details"] else None,
            output_schema=json.loads(row["output_schema"]) if row["output_schema"] else None,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def close(self):
        self._conn.close()
