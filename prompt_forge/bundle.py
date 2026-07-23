"""
Example Bundles — generic containers for training data.

A bundle is a collection of files with named roles. The roles are defined
by a BundleSchema that is project-specific.

Example schemas:
    - Data extraction: {"input": ".pdf", "expected_output": ".json"}
    - Spec generation: {"input_data": ".csv", "expected_output": ".docx"}
    - Translation:     {"source": ".txt", "expected": ".txt"}
    - Mail + attachments: {"mail": ".txt", "attachments": ".pdf"} with variadic_roles={"attachments"}

Variadic roles accept zero or more files per bundle (stored as list[Path]).
They are optional during validation and collected automatically during directory loading.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any
import logging

from .file_loaders import FileLoader, FileContent, get_default_loader

logger = logging.getLogger(__name__)

# Role names containing these keywords are treated as expected/output roles.
# All other roles are treated as input roles.
OUTPUT_ROLE_KEYWORDS: tuple[str, ...] = ("expected", "output")


def is_output_role(role: str) -> bool:
    """Return True if the role name indicates an expected/output role."""
    role_lower = role.lower()
    return any(kw in role_lower for kw in OUTPUT_ROLE_KEYWORDS)


@dataclasses.dataclass
class BundleSchema:
    """
    Defines the structure of example bundles for a project.

    Each key is a role name, the value describes the expected file type.
    Roles are used to match files within a bundle directory.

    Args:
        roles: Mapping of role_name → file extension (e.g., {"input": ".pdf", "expected_output": ".json"})
        role_descriptions: Optional human-readable descriptions for each role.
        variadic_roles: Set of role names that accept zero or more files instead of exactly one.
            Variadic roles are optional in bundles (absence is valid) and their files are
            collected as a list. Example: variadic_roles={"attachments"}.
    """

    roles: dict[str, str]  # role_name → expected extension
    role_descriptions: dict[str, str] = dataclasses.field(default_factory=dict)
    variadic_roles: set[str] = dataclasses.field(default_factory=set)

    def validate_bundle(self, bundle: ExampleBundle) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        for role, expected_ext in self.roles.items():
            if role in self.variadic_roles:
                raw = bundle.files.get(role)
                if raw is None:
                    continue  # variadic role absent — OK
                file_list: list = raw if isinstance(raw, list) else [raw]
                for p in file_list:
                    actual_ext = Path(p).suffix.lower()
                    if actual_ext != expected_ext.lower():
                        errors.append(
                            f"Role '{role}': '{Path(p).name}' has extension "
                            f"'{actual_ext}', expected '{expected_ext}'"
                        )
            else:
                if role not in bundle.files:
                    errors.append(f"Missing required role: '{role}'")
                else:
                    p = bundle.files[role]
                    actual_ext = Path(p).suffix.lower()
                    if actual_ext != expected_ext.lower():
                        errors.append(
                            f"Role '{role}': '{Path(p).name}' has extension "
                            f"'{actual_ext}', expected '{expected_ext}'"
                        )
        return errors

    def to_dict(self) -> dict:
        return {
            "roles": self.roles,
            "role_descriptions": self.role_descriptions,
            "variadic_roles": sorted(self.variadic_roles),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BundleSchema:
        return cls(
            roles=data["roles"],
            role_descriptions=data.get("role_descriptions", {}),
            variadic_roles=set(data.get("variadic_roles", [])),
        )


@dataclasses.dataclass
class ExampleBundle:
    """
    A single training example — a collection of files with named roles.

    Attributes:
        bundle_id: Unique identifier for this bundle.
        files: Mapping of role_name → file path (or list of paths for variadic roles).
        metadata: Optional extra info about this example.
    """

    bundle_id: str
    files: dict[str, Path | list[Path]]  # role_name → file path (or list for variadic roles)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def load_contents(self, loader: FileLoader | None = None) -> dict[str, FileContent]:
        """
        Load all files in the bundle using the given FileLoader.

        For variadic roles (list[Path]), all files are loaded and their text is
        concatenated with ``\\n\\n`` separators into a single FileContent.

        Returns:
            Mapping of role_name → FileContent.
        """
        if loader is None:
            loader = get_default_loader()
        contents: dict[str, FileContent] = {}
        for role, path in self.files.items():
            if isinstance(path, list):
                if not path:
                    contents[role] = FileContent(
                        text="", source_path="", file_type="", metadata={"file_count": 0}
                    )
                else:
                    loaded = [loader.load(p) for p in path]
                    combined = "\n\n".join(fc.text for fc in loaded)
                    contents[role] = FileContent(
                        text=combined,
                        source_path=str(path[0]),
                        file_type=path[0].suffix,
                        metadata={"file_count": len(path), "source_paths": [str(p) for p in path]},
                    )
            else:
                contents[role] = loader.load(path)
        return contents

    def to_dict(self) -> dict:
        def _serialise(v: Path | list[Path]) -> str | list[str]:
            return [str(p) for p in v] if isinstance(v, list) else str(v)

        return {
            "bundle_id": self.bundle_id,
            "files": {role: _serialise(path) for role, path in self.files.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExampleBundle:
        def _deserialise(v: str | list) -> Path | list[Path]:
            return [Path(p) for p in v] if isinstance(v, list) else Path(v)

        return cls(
            bundle_id=data["bundle_id"],
            files={role: _deserialise(v) for role, v in data["files"].items()},
            metadata=data.get("metadata", {}),
        )


class BundleCollection:
    """
    Manages a collection of ExampleBundles for a project.

    Supports loading from directories with auto-discovery based on schema.
    """

    def __init__(self, schema: BundleSchema, loader: FileLoader | None = None):
        self.schema = schema
        self.loader = loader or get_default_loader()
        self._bundles: dict[str, ExampleBundle] = {}

    @property
    def bundles(self) -> list[ExampleBundle]:
        return list(self._bundles.values())

    def __len__(self) -> int:
        return len(self._bundles)

    def __getitem__(self, bundle_id: str) -> ExampleBundle:
        return self._bundles[bundle_id]

    def add(self, bundle: ExampleBundle) -> None:
        """Add a single bundle to the collection."""
        errors = self.schema.validate_bundle(bundle)
        if errors:
            raise ValueError(
                f"Bundle '{bundle.bundle_id}' validation failed: {errors}"
            )
        self._bundles[bundle.bundle_id] = bundle

    def add_from_directory(
        self,
        directory: str | Path
    ) -> int:
        """
        Auto-discover bundles from a directory structure.

        Expects one subdirectory per bundle:

            directory/
                example_001/
                    input.pdf
                    expected_output.json
                example_002/
                    input.pdf
                    expected_output.json

        Files inside each subdirectory are matched to schema roles by name:
        a file named ``{role}.ext`` or ``{role}_*.ext`` is assigned to that role.
        For variadic roles all matching files are collected as a list.

        Returns the number of bundles loaded.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        subdirs = [d for d in sorted(directory.iterdir()) if d.is_dir()]
        if not subdirs:
            logger.warning(
                f"No subdirectories found in {directory}. "
                "add_from_directory expects one subdirectory per bundle."
            )
            return 0

        logger.info(f"Loading bundles from subdirectories in {directory}...")
        return self._load_subdir_layout(subdirs)

    def _load_subdir_layout(self, subdirs: list[Path]) -> int:
        """Each subdirectory is one bundle. Files are matched by role name in filename."""
        loaded = 0
        for subdir in subdirs:
            files: dict[str, Path | list[Path]] = {}
            for role, ext in self.schema.roles.items():
                candidates = list(subdir.glob(f"{role}.*")) + list(subdir.glob(f"{role}_*.*"))
                if not candidates:
                    candidates = [f for f in subdir.iterdir() if f.suffix.lower() == ext.lower()]

                if role in self.schema.variadic_roles:
                    if candidates:
                        files[role] = sorted(candidates)
                elif candidates:
                    files[role] = candidates[0]

            if files:
                bundle = ExampleBundle(
                    bundle_id=subdir.name,
                    files=files,
                    metadata={"source_dir": str(subdir)},
                )
                errors = self.schema.validate_bundle(bundle)
                if not errors:
                    self._bundles[bundle.bundle_id] = bundle
                    loaded += 1

        return loaded

    def get_batch(self, ids: list[str]) -> list[ExampleBundle]:
        """Get specific bundles by ID."""
        return [self._bundles[bid] for bid in ids if bid in self._bundles]

    def to_dict(self) -> dict:
        return {
            "schema": self.schema.to_dict(),
            "bundles": {bid: b.to_dict() for bid, b in self._bundles.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, loader: FileLoader | None = None) -> BundleCollection:
        schema = BundleSchema.from_dict(data["schema"])
        collection = cls(schema=schema, loader=loader)
        for bid, bdata in data.get("bundles", {}).items():
            bundle = ExampleBundle.from_dict(bdata)
            collection._bundles[bid] = bundle
        return collection
