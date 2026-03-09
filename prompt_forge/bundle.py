"""
Example Bundles — generic containers for training data.

A bundle is a collection of files with named roles. The roles are defined
by a BundleSchema that is project-specific.

Example schemas:
    - Data extraction: {"input": ".pdf", "expected_output": ".json"}
    - Spec generation: {"input_data": ".csv", "expected_output": ".docx"}
    - Translation:     {"source": ".txt", "expected": ".txt"}
"""

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
    """

    roles: dict[str, str]  # role_name → expected extension
    role_descriptions: dict[str, str] = dataclasses.field(default_factory=dict)

    def validate_bundle(self, bundle: ExampleBundle) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        for role in self.roles:
            if role not in bundle.files:
                errors.append(f"Missing required role: '{role}'")
        return errors

    def to_dict(self) -> dict:
        return {
            "roles": self.roles,
            "role_descriptions": self.role_descriptions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BundleSchema:
        return cls(
            roles=data["roles"],
            role_descriptions=data.get("role_descriptions", {}),
        )


@dataclasses.dataclass
class ExampleBundle:
    """
    A single training example — a collection of files with named roles.

    Attributes:
        bundle_id: Unique identifier for this bundle.
        files: Mapping of role_name → file path.
        metadata: Optional extra info about this example.
    """

    bundle_id: str
    files: dict[str, Path]  # role_name → file path
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def load_contents(self, loader: FileLoader | None = None) -> dict[str, FileContent]:
        """
        Load all files in the bundle using the given FileLoader.

        Returns:
            Mapping of role_name → FileContent.
        """
        if loader is None:
            loader = get_default_loader()
        contents = {}
        for role, path in self.files.items():
            contents[role] = loader.load(path)
        return contents

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "files": {role: str(path) for role, path in self.files.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExampleBundle:
        return cls(
            bundle_id=data["bundle_id"],
            files={role: Path(p) for role, p in data["files"].items()},
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

        Expects one of two layouts:

        1. (Default) Subdirectory layout (one subdirectory per bundle):
            directory/
                example_001/
                    input.pdf
                    expected_output.json
                example_002/
                    input.pdf
                    expected_output.json

        2. Flat layout with naming convention:
            directory/
                001_input.pdf
                001_expected_output.json
                002_input.pdf
                002_expected_output.json

        Returns the number of bundles loaded.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        loaded = 0

        # Try subdirectory layout first
        subdirs = [d for d in sorted(directory.iterdir()) if d.is_dir()]
        if subdirs:
            logger.info(f"Loading bundles from subdirectories in {directory}...")
            loaded = self._load_subdir_layout(subdirs)
        else:
            logger.info(f"Loading bundles from flat layout in {directory}...")
            loaded = self._load_flat_layout(directory)

        return loaded

    def _load_subdir_layout(self, subdirs: list[Path]) -> int:
        """Each subdirectory is one bundle. Files are matched by role name in filename."""
        loaded = 0
        for subdir in subdirs:
            files = {}
            for role, ext in self.schema.roles.items():
                # Look for files matching: role_name.ext or role_name*.ext
                candidates = list(subdir.glob(f"{role}.*")) + list(subdir.glob(f"{role}_*.*"))
                # Also try: any file with the right extension if there's only one role with that ext
                if not candidates:
                    candidates = [f for f in subdir.iterdir() if f.suffix.lower() == ext.lower()]

                if candidates:
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

    def _load_flat_layout(self, directory: Path) -> int:
        """
        Flat directory with prefix-based grouping.
        Files like: 001_input.pdf, 001_expected_output.json
        The prefix before the first underscore groups files into bundles.
        """
        loaded = 0
        groups: dict[str, dict[str, Path]] = {}

        for file_path in sorted(directory.iterdir()):
            if file_path.is_dir() or file_path.name.startswith("."):
                continue
            # Extract prefix (bundle id) and role from filename
            name = file_path.stem
            parts = name.split("_", 1)
            if len(parts) < 2:
                continue
            prefix, role_part = parts[0], parts[1]

            # Match role_part against schema roles
            for role in self.schema.roles:
                if role_part.lower().startswith(role.lower()):
                    if prefix not in groups:
                        groups[prefix] = {}
                    groups[prefix][role] = file_path
                    break

        for prefix, files in groups.items():
            bundle = ExampleBundle(
                bundle_id=prefix,
                files=files,
                metadata={"source_dir": str(directory)},
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
