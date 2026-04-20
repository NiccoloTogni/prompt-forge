"""
Project storage backends.

Implementations:
    - FileSystemStore: JSON files on disk (default, portable, git-friendly)
    - SQLAlchemyStore: Any SQL database via a connection string (PostgreSQL,
                       MySQL, Azure SQL, SQLite, etc.) — requires sqlalchemy extra.
    - SQLiteStore: Deprecated. Use SQLAlchemyStore("sqlite:///path/to/db") instead.
"""

import json
import dataclasses
import warnings
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclasses.dataclass
class PromptVersion:
    """A single versioned prompt snapshot."""

    version: int
    prompt_text: str
    created_at: str  # ISO format
    parent_version: int | None = None
    training_log_entry: str | None = None
    eval_score: float | None = None
    eval_details: dict | None = None
    output_schema: dict | None = None
    metadata: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptVersion":
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
    JSON-file-based storage (default).

    Layout:
        project_dir/
            config.json
            training_state.json
            prompts/
                v0001.json
                v0002.json
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
        return PromptVersion.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def get_latest_prompt(self) -> PromptVersion | None:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def list_versions(self) -> list[PromptVersion]:
        return [
            PromptVersion.from_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in sorted(self.prompts_dir.glob("v*.json"))
        ]

    def save_project_config(self, config: dict) -> None:
        (self.project_dir / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_project_config(self) -> dict | None:
        path = self.project_dir / "config.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save_training_state(self, state: dict) -> None:
        (self.project_dir / "training_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load_training_state(self) -> dict | None:
        path = self.project_dir / "training_state.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class SQLAlchemyStore:
    """
    SQL-backed storage via SQLAlchemy.

    Supports any database that SQLAlchemy supports — PostgreSQL, MySQL,
    Azure SQL Server, SQLite, and more — via a standard connection string.

    Requires the ``sqlalchemy`` extra:
        pip install "prompt-forge[sqlalchemy]"

    Examples::

        # PostgreSQL
        SQLAlchemyStore("postgresql+psycopg2://user:pass@host/db")

        # Azure SQL Server
        SQLAlchemyStore(
            "mssql+pyodbc://user:pass@server.database.windows.net/db"
            "?driver=ODBC+Driver+18+for+SQL+Server"
        )

        # SQLite (file)
        SQLAlchemyStore("sqlite:///path/to/project.db")

        # SQLite (in-memory, useful for tests)
        SQLAlchemyStore("sqlite://")

    Args:
        connection_string: SQLAlchemy database URL.
        project_name: Logical project name used to namespace rows when multiple
                      projects share the same database. Defaults to "default".
        engine_kwargs: Extra keyword arguments forwarded to ``create_engine()``.
    """

    def __init__(
        self,
        connection_string: str,
        project_name: str = "default",
        **engine_kwargs,
    ):
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import StaticPool
        except ImportError:
            raise ImportError(
                "SQLAlchemyStore requires sqlalchemy. "
                'Install it with: pip install "prompt-forge[sqlalchemy]"'
            )

        self._project_name = project_name
        self._text = text

        # SQLite in-memory needs a shared connection pool to persist across calls
        if connection_string == "sqlite://" and "connect_args" not in engine_kwargs:
            engine_kwargs.setdefault("poolclass", StaticPool)
            engine_kwargs.setdefault("connect_args", {"check_same_thread": False})

        self._engine = create_engine(connection_string, **engine_kwargs)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(self._text("""
                CREATE TABLE IF NOT EXISTS pf_prompt_versions (
                    project_name    VARCHAR(255) NOT NULL,
                    version         INTEGER      NOT NULL,
                    prompt_text     TEXT         NOT NULL,
                    created_at      VARCHAR(64)  NOT NULL,
                    parent_version  INTEGER,
                    training_log_entry TEXT,
                    eval_score      FLOAT,
                    eval_details    TEXT,
                    output_schema   TEXT,
                    metadata        TEXT         NOT NULL DEFAULT '{}',
                    PRIMARY KEY (project_name, version)
                )
            """))
            conn.execute(self._text("""
                CREATE TABLE IF NOT EXISTS pf_project_config (
                    project_name VARCHAR(255) PRIMARY KEY,
                    config       TEXT         NOT NULL
                )
            """))
            conn.execute(self._text("""
                CREATE TABLE IF NOT EXISTS pf_training_state (
                    project_name VARCHAR(255) PRIMARY KEY,
                    state        TEXT         NOT NULL
                )
            """))

    # ── Prompt versions ───────────────────────────────────────────────

    def save_prompt_version(self, version: PromptVersion) -> None:
        params = {
            "project_name":       self._project_name,
            "version":            version.version,
            "prompt_text":        version.prompt_text,
            "created_at":         version.created_at,
            "parent_version":     version.parent_version,
            "training_log_entry": version.training_log_entry,
            "eval_score":         version.eval_score,
            "eval_details":       json.dumps(version.eval_details) if version.eval_details else None,
            "output_schema":      json.dumps(version.output_schema) if version.output_schema is not None else None,
            "metadata":           json.dumps(version.metadata),
        }
        with self._engine.begin() as conn:
            exists = conn.execute(
                self._text(
                    "SELECT 1 FROM pf_prompt_versions "
                    "WHERE project_name = :project_name AND version = :version"
                ),
                {"project_name": self._project_name, "version": version.version},
            ).first()
            if exists:
                conn.execute(
                    self._text("""
                        UPDATE pf_prompt_versions SET
                            prompt_text        = :prompt_text,
                            created_at         = :created_at,
                            parent_version     = :parent_version,
                            training_log_entry = :training_log_entry,
                            eval_score         = :eval_score,
                            eval_details       = :eval_details,
                            output_schema      = :output_schema,
                            metadata           = :metadata
                        WHERE project_name = :project_name AND version = :version
                    """),
                    params,
                )
            else:
                conn.execute(
                    self._text("""
                        INSERT INTO pf_prompt_versions
                            (project_name, version, prompt_text, created_at, parent_version,
                             training_log_entry, eval_score, eval_details, output_schema, metadata)
                        VALUES
                            (:project_name, :version, :prompt_text, :created_at, :parent_version,
                             :training_log_entry, :eval_score, :eval_details, :output_schema, :metadata)
                    """),
                    params,
                )

    def get_prompt_version(self, version: int) -> PromptVersion | None:
        sql = self._text(
            "SELECT * FROM pf_prompt_versions "
            "WHERE project_name = :p AND version = :v"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"p": self._project_name, "v": version}).mappings().first()
        return self._row_to_version(row) if row else None

    def get_latest_prompt(self) -> PromptVersion | None:
        sql = self._text(
            "SELECT * FROM pf_prompt_versions WHERE project_name = :p "
            "ORDER BY version DESC LIMIT 1"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"p": self._project_name}).mappings().first()
        return self._row_to_version(row) if row else None

    def list_versions(self) -> list[PromptVersion]:
        sql = self._text(
            "SELECT * FROM pf_prompt_versions WHERE project_name = :p "
            "ORDER BY version ASC"
        )
        with self._engine.connect() as conn:
            rows = conn.execute(sql, {"p": self._project_name}).mappings().all()
        return [self._row_to_version(r) for r in rows]

    # ── Config & state ────────────────────────────────────────────────

    def save_project_config(self, config: dict) -> None:
        value = json.dumps(config, ensure_ascii=False)
        with self._engine.begin() as conn:
            exists = conn.execute(
                self._text("SELECT 1 FROM pf_project_config WHERE project_name = :p"),
                {"p": self._project_name},
            ).first()
            if exists:
                conn.execute(
                    self._text("UPDATE pf_project_config SET config = :c WHERE project_name = :p"),
                    {"p": self._project_name, "c": value},
                )
            else:
                conn.execute(
                    self._text("INSERT INTO pf_project_config (project_name, config) VALUES (:p, :c)"),
                    {"p": self._project_name, "c": value},
                )

    def load_project_config(self) -> dict | None:
        sql = self._text(
            "SELECT config FROM pf_project_config WHERE project_name = :p"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"p": self._project_name}).mappings().first()
        return json.loads(row["config"]) if row else None

    def save_training_state(self, state: dict) -> None:
        value = json.dumps(state, ensure_ascii=False)
        with self._engine.begin() as conn:
            exists = conn.execute(
                self._text("SELECT 1 FROM pf_training_state WHERE project_name = :p"),
                {"p": self._project_name},
            ).first()
            if exists:
                conn.execute(
                    self._text("UPDATE pf_training_state SET state = :s WHERE project_name = :p"),
                    {"p": self._project_name, "s": value},
                )
            else:
                conn.execute(
                    self._text("INSERT INTO pf_training_state (project_name, state) VALUES (:p, :s)"),
                    {"p": self._project_name, "s": value},
                )

    def load_training_state(self) -> dict | None:
        sql = self._text(
            "SELECT state FROM pf_training_state WHERE project_name = :p"
        )
        with self._engine.connect() as conn:
            row = conn.execute(sql, {"p": self._project_name}).mappings().first()
        return json.loads(row["state"]) if row else None

    # ── Internal ──────────────────────────────────────────────────────

    def _row_to_version(self, row) -> PromptVersion:
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

    def dispose(self) -> None:
        """Release the connection pool."""
        self._engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()


class SQLiteStore:
    """
    .. deprecated::
        Use ``SQLAlchemyStore("sqlite:///path/to/db")`` instead.
        SQLiteStore will be removed in a future release.
    """

    def __init__(self, db_path: str | Path):
        warnings.warn(
            "SQLiteStore is deprecated and will be removed in a future release. "
            'Use SQLAlchemyStore("sqlite:///path/to/db") instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        import sqlite3

        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
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
        try:
            self._conn.execute("ALTER TABLE prompt_versions ADD COLUMN output_schema TEXT")
            self._conn.commit()
        except Exception:
            pass

    def save_prompt_version(self, version: PromptVersion) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO prompt_versions "
            "(version, prompt_text, created_at, parent_version, training_log_entry, "
            "eval_score, eval_details, output_schema, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version.version, version.prompt_text, version.created_at,
                version.parent_version, version.training_log_entry, version.eval_score,
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
        return self._row_to_version(row) if row else None

    def get_latest_prompt(self) -> PromptVersion | None:
        row = self._conn.execute(
            "SELECT * FROM prompt_versions ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return self._row_to_version(row) if row else None

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
        return json.loads(row["config"]) if row else None

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
        return json.loads(row["state"]) if row else None

    def _row_to_version(self, row) -> PromptVersion:
        d = dict(row) if not isinstance(row, dict) else row
        return PromptVersion(
            version=d["version"],
            prompt_text=d["prompt_text"],
            created_at=d["created_at"],
            parent_version=d["parent_version"],
            training_log_entry=d["training_log_entry"],
            eval_score=d["eval_score"],
            eval_details=json.loads(d["eval_details"]) if d["eval_details"] else None,
            output_schema=json.loads(d["output_schema"]) if d["output_schema"] else None,
            metadata=json.loads(d["metadata"]) if d["metadata"] else {},
        )

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
