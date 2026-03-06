"""
Batch strategies — how to select examples for each training iteration.

The strategy determines which examples are shown to the optimizer in each
iteration. Good strategies maximize learning per iteration.
"""

import random
from abc import ABC, abstractmethod

from ..bundle import ExampleBundle


class BatchStrategy(ABC):
    """Base class for batch selection strategies."""

    @abstractmethod
    def select_batch(
        self,
        bundles: list[ExampleBundle],
        batch_size: int,
        used_ids: set[str] | None = None,
        failed_ids: list[str] | None = None,
    ) -> list[ExampleBundle]:
        """
        Select a batch of examples for the next training iteration.

        Args:
            bundles: All available example bundles.
            batch_size: Number of examples to select.
            used_ids: IDs of bundles already used in previous iterations.
            failed_ids: IDs of bundles that the current prompt gets wrong.

        Returns:
            List of selected ExampleBundles.
        """
        ...


class RandomBatchStrategy(BatchStrategy):
    """
    Simple random selection.

    Prioritizes unseen examples, falls back to random selection from
    all examples when all have been seen.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def select_batch(
        self,
        bundles: list[ExampleBundle],
        batch_size: int,
        used_ids: set[str] | None = None,
        failed_ids: list[str] | None = None,
    ) -> list[ExampleBundle]:
        _ = failed_ids  # Not used in this strategy
        used_ids = used_ids or set()
        batch_size = min(batch_size, len(bundles))

        # Prioritize unseen examples
        unseen = [b for b in bundles if b.bundle_id not in used_ids]

        if len(unseen) >= batch_size:
            return self.rng.sample(unseen, batch_size)

        # Use all unseen + fill the rest randomly from seen
        batch = list(unseen)
        seen = [b for b in bundles if b.bundle_id in used_ids]
        remaining = batch_size - len(batch)
        if seen and remaining > 0:
            batch.extend(self.rng.sample(seen, min(remaining, len(seen))))

        return batch


class FailurePriorityBatchStrategy(BatchStrategy):
    """
    Prioritizes examples that the current prompt fails on.

    Mixes failed examples with unseen examples to balance
    fixing known issues with discovering new patterns.
    """

    def __init__(self, failure_ratio: float = 0.5, seed: int | None = None):
        """
        Args:
            failure_ratio: Proportion of the batch to fill with failed examples (0-1).
            seed: Random seed.
        """
        self.failure_ratio = failure_ratio
        self.rng = random.Random(seed)

    def select_batch(
        self,
        bundles: list[ExampleBundle],
        batch_size: int,
        used_ids: set[str] | None = None,
        failed_ids: list[str] | None = None,
    ) -> list[ExampleBundle]:
        used_ids = used_ids or set()
        failed_ids = failed_ids or []
        batch_size = min(batch_size, len(bundles))

        bundles_by_id = {b.bundle_id: b for b in bundles}

        # Split into categories
        failed = [bundles_by_id[fid] for fid in failed_ids if fid in bundles_by_id]
        unseen = [b for b in bundles if b.bundle_id not in used_ids and b.bundle_id not in failed_ids]
        rest = [b for b in bundles if b.bundle_id in used_ids and b.bundle_id not in failed_ids]

        # Calculate slots
        failure_slots = min(int(batch_size * self.failure_ratio), len(failed))
        remaining_slots = batch_size - failure_slots

        batch = self.rng.sample(failed, failure_slots) if failure_slots > 0 else []

        # Fill remaining with unseen first, then rest
        pool = unseen + rest
        if pool and remaining_slots > 0:
            batch.extend(self.rng.sample(pool, min(remaining_slots, len(pool))))

        return batch
