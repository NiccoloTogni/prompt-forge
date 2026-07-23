"""
Utility helpers for working with prompt-forge data structures.
"""

from __future__ import annotations

import random

from .bundle import BundleCollection


def _subset_collection(collection: BundleCollection, bundles: list) -> BundleCollection:
    """Build a new BundleCollection (same schema/loader) from a list of bundles."""
    col = BundleCollection(schema=collection.schema, loader=collection.loader)
    for b in bundles:
        col._bundles[b.bundle_id] = b
    return col


def train_val_split(
    collection: BundleCollection,
    val_ratio: float = 0.2,
    val_size: int | None = None,
    seed: int | None = None,
) -> tuple[BundleCollection, BundleCollection]:
    """
    Split a BundleCollection into training and validation sets.

    Args:
        collection: The full collection to split.
        val_ratio: Fraction of examples for validation (ignored if val_size is set).
        val_size: Exact number of validation examples. Takes priority over val_ratio.
        seed: Random seed for reproducibility.

    Returns:
        (train_collection, val_collection) — two BundleCollection objects sharing
        the same schema and loader.

    Example:
        train_bundles, val_bundles = train_val_split(project.bundles, val_ratio=0.2, seed=42)
        project.train(train_bundles, val_bundles=val_bundles, config=TrainingConfig(...))
    """
    all_bundles = collection.bundles
    if not all_bundles:
        raise ValueError("Cannot split an empty collection.")

    rng = random.Random(seed)
    shuffled = rng.sample(all_bundles, len(all_bundles))

    n_val = val_size if val_size is not None else max(1, int(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)  # keep at least one training example

    val_list = shuffled[:n_val]
    train_list = shuffled[n_val:]

    return _subset_collection(collection, train_list), _subset_collection(collection, val_list)


def train_val_test_split(
    collection: BundleCollection,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int | None = None,
) -> tuple[BundleCollection, BundleCollection, BundleCollection]:
    """
    Split a BundleCollection into training, validation, and test sets.

    The training loop accepts or rejects each candidate prompt based on its
    validation score — it hill-climbs on the val set, so the final val score is
    optimistically biased. The test set is meant to be passed as ``test_bundles``
    to ``train()``: it is evaluated exactly once, on the final prompt, and gives
    an unbiased estimate of generalization.

    Args:
        collection: The full collection to split.
        val_ratio: Fraction of examples for validation (ignored if val_size is set).
        test_ratio: Fraction of examples for test (ignored if test_size is set).
        val_size: Exact number of validation examples. Takes priority over val_ratio.
        test_size: Exact number of test examples. Takes priority over test_ratio.
        seed: Random seed for reproducibility.

    Returns:
        (train_collection, val_collection, test_collection) — three
        BundleCollection objects sharing the same schema and loader.

    Example:
        train_b, val_b, test_b = train_val_test_split(project.bundles, seed=42)
        report = project.train(train_b, val_bundles=val_b, test_bundles=test_b)
        print(report.test_score)  # unbiased generalization estimate
    """
    all_bundles = collection.bundles
    if len(all_bundles) < 3:
        raise ValueError(
            "Need at least 3 examples for a train/val/test split "
            f"(got {len(all_bundles)})."
        )

    rng = random.Random(seed)
    shuffled = rng.sample(all_bundles, len(all_bundles))
    n = len(shuffled)

    n_val = val_size if val_size is not None else max(1, int(n * val_ratio))
    n_test = test_size if test_size is not None else max(1, int(n * test_ratio))
    # Guarantee at least one example per split: cap test first, then val
    n_test = min(n_test, n - 2)
    n_val = min(n_val, n - n_test - 1)

    val_list = shuffled[:n_val]
    test_list = shuffled[n_val:n_val + n_test]
    train_list = shuffled[n_val + n_test:]

    return (
        _subset_collection(collection, train_list),
        _subset_collection(collection, val_list),
        _subset_collection(collection, test_list),
    )
