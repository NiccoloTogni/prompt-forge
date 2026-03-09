"""
Training Log — compact memory of what was learned across iterations.

The training log travels with the prompt. When the optimizer sees a new
batch of examples, it also sees the log of all previous learnings. This
prevents forgetting.
"""

import dataclasses


@dataclasses.dataclass
class LogEntry:
    """A single training iteration entry."""

    iteration: int
    timestamp: str
    batch_ids: list[str]
    score_before: float | None
    score_after: float | None
    learnings: str       # What the optimizer learned from this batch
    prompt_version: int
    issues: str = ""     # Outstanding gaps/contradictions flagged by the optimizer; default
                         # allows loading log entries saved before this field was added

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LogEntry:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TrainingLog:
    """
    Manages the cumulative training log.

    The log is designed to stay compact — it summarizes what was learned
    rather than storing raw examples. This lets it fit in context alongside
    the prompt and new examples.
    """

    def __init__(self):
        self._entries: list[LogEntry] = []

    @property
    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def add_entry(self, entry: LogEntry) -> None:
        self._entries.append(entry)

    def get_summary(self, max_entries: int | None = None) -> str:
        """
        Generate a compact text summary of the training history.

        This is injected into the optimizer's context so it knows what
        has been learned previously.
        """
        entries = self._entries
        if max_entries and len(entries) > max_entries:
            # Keep first few + last few for context
            keep_start = max_entries // 3
            keep_end = max_entries - keep_start
            entries = entries[:keep_start] + entries[-keep_end:]

        if not entries:
            return "No training history yet."

        lines = [f"=== Training History ({len(self._entries)} iterations) ===\n"]
        for entry in entries:
            score_change = ""
            if entry.score_before is not None and entry.score_after is not None:
                delta = entry.score_after - entry.score_before
                score_change = f" | Score: {entry.score_before:.2f} → {entry.score_after:.2f} ({delta:+.2f})"

            lines.append(f"--- Iteration {entry.iteration}{score_change} ---")
            lines.append(f"Batch: {', '.join(entry.batch_ids[:5])}{'...' if len(entry.batch_ids) > 5 else ''}")
            lines.append(f"Learnings: {entry.learnings}")
            if entry.issues:
                lines.append(f"Issues: {entry.issues}")
            lines.append("")

        return "\n".join(lines)

    def get_all_used_bundle_ids(self) -> set[str]:
        """Get all bundle IDs that have been used in training so far."""
        ids: set[str] = set()
        for entry in self._entries:
            ids.update(entry.batch_ids)
        return ids

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self._entries]}

    @classmethod
    def from_dict(cls, data: dict) -> TrainingLog:
        log = cls()
        for entry_data in data.get("entries", []):
            log._entries.append(LogEntry.from_dict(entry_data))
        return log
