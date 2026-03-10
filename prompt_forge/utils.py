"""
Utility helpers for working with prompt-forge data structures.
"""

import random

from .bundle import BundleCollection


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
        project.train(config=TrainingConfig(...), val_bundles=val_bundles)
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

    train_col = BundleCollection(schema=collection.schema, loader=collection.loader)
    for b in train_list:
        train_col._bundles[b.bundle_id] = b

    val_col = BundleCollection(schema=collection.schema, loader=collection.loader)
    for b in val_list:
        val_col._bundles[b.bundle_id] = b

    return train_col, val_col
