from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    CLARIFYING = "CLARIFYING"
    PLANNING = "PLANNING"
    SPEC_PENDING = "SPEC_PENDING"
    IMPLEMENTING = "IMPLEMENTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


@dataclass
class Task:
    id: str
    repo_path: Path
    branch: str
    description: str
    status: TaskStatus = TaskStatus.CREATED
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "repo_path": str(self.repo_path),
            "branch": self.branch,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Task":
        return cls(
            id=raw["id"],
            repo_path=Path(raw["repo_path"]),
            branch=raw["branch"],
            description=raw["description"],
            status=TaskStatus(raw["status"]),
            created_at=datetime.fromisoformat(raw["created_at"]),
            updated_at=datetime.fromisoformat(raw["updated_at"]),
            metadata=raw.get("metadata", {}),
        )


class ClarificationStatus(str, Enum):
    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    OVERRIDDEN = "OVERRIDDEN"


@dataclass
class ClarificationItem:
    id: str
    task_id: str
    question: str
    answer: Optional[str] = None
    status: ClarificationStatus = ClarificationStatus.PENDING


class PlanStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class PlanStep:
    description: str
    target_files: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "target_files": list(self.target_files),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PlanStep":
        return cls(
            description=raw.get("description", ""),
            target_files=list(raw.get("target_files", [])),
            notes=raw.get("notes"),
        )


@dataclass
class Plan:
    id: str
    task_id: str
    steps: List[PlanStep] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    refactor_suggestions: List[str] = field(default_factory=list)
    status: PlanStatus = PlanStatus.PENDING


class BoundarySpecStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    SKIPPED = "SKIPPED"


@dataclass
class BoundarySpec:
    id: str
    task_id: str
    boundary_name: str
    human_description: str
    diagram_text: str
    machine_spec: Dict[str, Any]
    status: BoundarySpecStatus = BoundarySpecStatus.PENDING
    plan_step: Optional[str] = None


class PatchStatus(str, Enum):
    PENDING = "PENDING"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"


class PatchKind(str, Enum):
    IMPLEMENTATION = "IMPLEMENTATION"
    REFACTOR = "REFACTOR"


@dataclass
class Patch:
    id: str
    task_id: str
    step_reference: str
    diff: str
    rationale: str
    alternatives: List[str] = field(default_factory=list)
    status: PatchStatus = PatchStatus.PENDING
    kind: PatchKind = PatchKind.IMPLEMENTATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "step_reference": self.step_reference,
            "diff": self.diff,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
            "status": self.status.value,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> Patch:
        return cls(
            id=raw["id"],
            task_id=raw["task_id"],
            step_reference=raw["step_reference"],
            diff=raw["diff"],
            rationale=raw["rationale"],
            alternatives=raw.get("alternatives", []),
            status=PatchStatus(raw.get("status", PatchStatus.PENDING.value)),
            kind=PatchKind(raw.get("kind", PatchKind.IMPLEMENTATION.value)),
        )


class TestSuggestionStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    DISMISSED = "DISMISSED"


@dataclass
class TestSuggestion:
    id: str
    task_id: str
    description: str
    suggestion_type: str  # UNIT, INTEGRATION, etc.
    related_files: List[str] = field(default_factory=list)
    skeleton_code: Optional[str] = None
    status: TestSuggestionStatus = TestSuggestionStatus.NEW


@dataclass
class LogEntry:
    id: str
    task_id: str
    timestamp: datetime
    entry_type: str
    payload: Dict[str, Any]


class RefactorSuggestionStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class RefactorSuggestion:
    id: str
    task_id: str
    description: str
    rationale: str
    scope: List[str] = field(default_factory=list)
    status: RefactorSuggestionStatus = RefactorSuggestionStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "description": self.description,
            "rationale": self.rationale,
            "scope": self.scope,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "RefactorSuggestion":
        return cls(
            id=raw["id"],
            task_id=raw["task_id"],
            description=raw["description"],
            rationale=raw["rationale"],
            scope=raw.get("scope", []),
            status=RefactorSuggestionStatus(raw.get("status", RefactorSuggestionStatus.PENDING.value)),
        )
