# Bundle Reference

Bundles are the data containers for prompt-forge. Each bundle represents one training/evaluation example as a collection of named files.

---

## BundleSchema

Defines the expected structure of bundles for a project.

```python
from prompt_forge import BundleSchema

schema = BundleSchema(
    roles={"input": ".pdf", "expected_output": ".json"},
    role_descriptions={"input": "Invoice PDF", "expected_output": "Extracted fields as JSON"},
    variadic_roles={"attachments"},  # zero-or-more files
)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `roles` | `dict[str, str]` | `role_name → expected_extension` (e.g. `".pdf"`). Required. |
| `role_descriptions` | `dict[str, str]` | Human-readable description per role. Optional. |
| `variadic_roles` | `set[str]` | Roles that accept zero or more files. Files stored as `list[Path]`. |

### Methods

#### `validate_bundle(bundle) → list[str]`

Returns a list of validation error messages. An empty list means the bundle is valid. Used by `BundleCollection.add()` internally.

#### `to_dict() / from_dict(data)`

Serialization helpers for JSON round-trips.

### Role name conventions

Role names containing `"expected"` or `"output"` (case-insensitive) are treated as **output roles** — they are excluded from inference inputs and used as ground truth by the evaluator. All other roles are input roles.

---

## ExampleBundle

A single training or evaluation example.

```python
from pathlib import Path
from prompt_forge import ExampleBundle

bundle = ExampleBundle(
    bundle_id="invoice_001",
    files={
        "input": Path("data/001/input.pdf"),
        "expected_output": Path("data/001/expected_output.json"),
    },
    metadata={"source": "acme-corp"},
)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `bundle_id` | `str` | Unique identifier. Used in evaluation reports and batch tracking. |
| `files` | `dict[str, Path \| list[Path]]` | `role_name → path`. Variadic roles use `list[Path]`. |
| `metadata` | `dict` | Arbitrary key-value pairs. Not used internally, available to custom code. |

### Methods

#### `load_contents(loader=None) → dict[str, FileContent]`

Loads all files in the bundle using the given `FileLoader`. Returns `role_name → FileContent`. For variadic roles, all files are loaded and their text is concatenated with `\n\n` separators.

If `loader` is `None`, uses `get_default_loader()` (plain-text reader).

#### `to_dict() / from_dict(data)`

Serialization helpers. File paths are stored as strings.

---

## BundleCollection

Manages a collection of `ExampleBundle` objects for a project.

```python
from prompt_forge import BundleCollection, BundleSchema

schema = BundleSchema(roles={"input": ".pdf", "expected_output": ".json"})
collection = BundleCollection(schema=schema)
n = collection.add_from_directory("data/examples/")
print(f"Loaded {n} bundles")
```

### Constructor

```python
BundleCollection(schema: BundleSchema, loader: FileLoader | None = None)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `bundles` | `list[ExampleBundle]` | All bundles in the collection. |
| `schema` | `BundleSchema` | The schema used to validate bundles. |

### Methods

#### `add(bundle: ExampleBundle) → None`

Add a single bundle. Raises `ValueError` if it fails schema validation.

#### `add_from_directory(directory) → int`

Auto-discover bundles from a directory. Expects one subdirectory per bundle:

```
data/
    invoice_001/
        input.pdf
        expected_output.json
    invoice_002/
        input.pdf
        expected_output.json
```

Files inside each subdirectory are matched to schema roles by filename prefix: `{role}.ext` or `{role}_*.ext`. Returns the number of successfully loaded bundles. Bundles that fail schema validation are silently skipped (a warning is logged).

#### `get_batch(ids: list[str]) → list[ExampleBundle]`

Return specific bundles by ID. IDs not found in the collection are silently skipped.

#### `__len__() / __getitem__(bundle_id)`

Standard container operations.

#### `to_dict() / from_dict(data, loader=None)`

Serialization helpers.

---

## Train/val/test splits

`BundleCollection.bundles` returns a plain list, so standard Python is sufficient:

```python
from prompt_forge.training.pipeline import train_val_split

all_bundles = collection.bundles
# First split: hold out 20% as test
trainval, test = train_val_split(all_bundles, val_ratio=0.2, seed=42)
# Second split: hold out 20% of remaining as val
train, val = train_val_split(trainval, val_ratio=0.2, seed=42)

report = pipeline.train(train, val_bundles=val, ...)
# Evaluate final prompt on test (never seen during training)
agent = InferenceAgent.from_store(llm, store)
```

`train_val_split` is a helper in `prompt_forge.training.pipeline`.

---

## Variadic roles

Use variadic roles when an example has a variable number of files for one role (e.g. email attachments, supporting documents):

```python
schema = BundleSchema(
    roles={"email": ".txt", "attachments": ".pdf"},
    variadic_roles={"attachments"},
)

bundle = ExampleBundle(
    bundle_id="case_001",
    files={
        "email": Path("case_001/email.txt"),
        "attachments": [Path("case_001/doc1.pdf"), Path("case_001/doc2.pdf")],
    },
)
```

Variadic roles are optional during validation — a bundle with zero attachment files is valid.

---

## Design notes

- **Output role detection** is done by keyword matching (`"expected"` or `"output"` in the role name). If your schema uses a non-standard name for the ground truth file, it will be treated as an input. Rename it to include `"expected"` or `"output"`.
- **`bundle_id`** must be unique within a collection. `add_from_directory` uses the subdirectory name as the ID. If you add bundles manually, ensure IDs don't collide.
- **`FileContent.text`** for variadic roles is a single concatenated string. If you need to process files individually, iterate over `bundle.files[role]` yourself.
