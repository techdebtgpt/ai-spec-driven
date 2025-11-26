from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass
class ContextExpansionStep:
    trigger: str
    included_paths: List[Path]
    excluded_paths: List[Path] = field(default_factory=list)


class ContextRetriever:
    """
    Minimal iterative context expansion engine.

    Today it simply tracks which files/directories have been requested and
    records the reasoning payload. Future iterations can integrate embeddings
    or semantic search without changing the interface.
    """

    def __init__(self) -> None:
        self._included: Set[Path] = set()
        self._excluded: Set[Path] = set()
        self.steps: List[ContextExpansionStep] = []

    def include(self, trigger: str, paths: List[Path]) -> ContextExpansionStep:
        normalized = [path.resolve() for path in paths]
        self._included.update(normalized)
        step = ContextExpansionStep(trigger=trigger, included_paths=normalized)
        self.steps.append(step)
        return step

    def exclude(self, trigger: str, paths: List[Path]) -> ContextExpansionStep:
        normalized = [path.resolve() for path in paths]
        self._excluded.update(normalized)
        step = ContextExpansionStep(trigger=trigger, included_paths=[], excluded_paths=normalized)
        self.steps.append(step)
        return step

    def current_context(self) -> List[Path]:
        return sorted(self._included - self._excluded)


