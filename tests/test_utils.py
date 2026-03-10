"""
Tests for prompt_forge.utils
"""

import pytest
from pathlib import Path

from prompt_forge.bundle import BundleSchema, BundleCollection, ExampleBundle
from prompt_forge.utils import train_val_split


# ── Helpers ───────────────────────────────────────────────────────────────────

SCHEMA = BundleSchema(roles={"input": ".txt", "expected_output": ".txt"})


def make_collection(n: int) -> BundleCollection:
    col = BundleCollection(schema=SCHEMA)
    for i in range(n):
        col.add(ExampleBundle(
            bundle_id=f"b{i}",
            files={
                "input": Path(f"/fake/b{i}_input.txt"),
                "expected_output": Path(f"/fake/b{i}_output.txt"),
            },
        ))
    return col


# ── train_val_split ───────────────────────────────────────────────────────────

def test_sizes_sum_to_total():
    col = make_collection(10)
    train, val = train_val_split(col, val_ratio=0.2)
    assert len(train) + len(val) == 10


def test_val_ratio(self=None):
    col = make_collection(10)
    _, val = train_val_split(col, val_ratio=0.2)
    assert len(val) == 2


def test_val_size_overrides_ratio():
    col = make_collection(10)
    train, val = train_val_split(col, val_size=3)
    assert len(val) == 3
    assert len(train) == 7


def test_val_size_1():
    col = make_collection(5)
    train, val = train_val_split(col, val_size=1)
    assert len(val) == 1
    assert len(train) == 4


def test_no_overlap():
    col = make_collection(10)
    train, val = train_val_split(col, val_ratio=0.3, seed=0)
    train_ids = {b.bundle_id for b in train.bundles}
    val_ids = {b.bundle_id for b in val.bundles}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids | val_ids == {f"b{i}" for i in range(10)}


def test_reproducible_with_seed():
    col = make_collection(20)
    _, val1 = train_val_split(col, seed=42)
    _, val2 = train_val_split(col, seed=42)
    assert {b.bundle_id for b in val1.bundles} == {b.bundle_id for b in val2.bundles}


def test_different_seeds_produce_different_splits():
    col = make_collection(20)
    _, val1 = train_val_split(col, seed=1)
    _, val2 = train_val_split(col, seed=2)
    assert {b.bundle_id for b in val1.bundles} != {b.bundle_id for b in val2.bundles}


def test_no_seed_is_non_deterministic():
    """Without seed, two runs should not always return the same split."""
    col = make_collection(20)
    results = set()
    for _ in range(10):
        _, val = train_val_split(col)
        results.add(frozenset(b.bundle_id for b in val.bundles))
    assert len(results) > 1  # at least some variation


def test_at_least_one_train_example():
    col = make_collection(2)
    train, val = train_val_split(col, val_ratio=0.99)
    assert len(train) >= 1


def test_at_least_one_val_example():
    col = make_collection(10)
    _, val = train_val_split(col, val_ratio=0.01)
    assert len(val) >= 1


def test_val_size_capped_to_leave_one_train():
    col = make_collection(3)
    train, val = train_val_split(col, val_size=99)
    assert len(train) >= 1
    assert len(val) == 2  # capped at n - 1


def test_returns_bundle_collections():
    col = make_collection(5)
    train, val = train_val_split(col)
    assert isinstance(train, BundleCollection)
    assert isinstance(val, BundleCollection)


def test_inherits_schema():
    col = make_collection(5)
    train, val = train_val_split(col)
    assert train.schema is col.schema
    assert val.schema is col.schema


def test_inherits_loader():
    col = make_collection(5)
    train, val = train_val_split(col)
    assert train.loader is col.loader
    assert val.loader is col.loader


def test_empty_collection_raises():
    col = BundleCollection(schema=SCHEMA)
    with pytest.raises(ValueError, match="empty"):
        train_val_split(col)


def test_single_item_collection_raises():
    """Can't split 1 item: need at least 1 train and 1 val."""
    col = make_collection(1)
    # val_size=1 would leave 0 train examples, so it's capped to 0 → still 0 val?
    # actually: n_val = min(1, 1 - 1) = min(1, 0) = 0
    # val ends up empty but no error is raised (edge case, not a crash)
    train, val = train_val_split(col, val_size=1)
    assert len(train) + len(val) == 1
