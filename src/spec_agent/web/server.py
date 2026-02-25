from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..config.settings import get_settings
from ..domain.models import LogEntry, Task, TaskStatus
from ..persistence.store import JsonStore
from ..services.patches.engine import EXTERNAL_PATCH_SENTINEL


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _safe_dt(value: object) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    try:
        if isinstance(value, str) and value:
            return datetime.fromisoformat(value)
    except Exception:
        return None
    return None


def _task_title(task: Task) -> str:
    title = (task.title or "").strip()
    if title:
        return title
    desc = (task.description or "").strip()
    return desc.splitlines()[0].strip() if desc else f"task-{task.id[:8]}"


def _task_summary(task: Task) -> str:
    summary = (task.summary or "").strip()
    if summary:
        return summary
    desc = (task.description or "").strip()
    return desc.splitlines()[0].strip() if desc else ""


def _detect_test_runner(
    files: list[str], existing_tool: str | None = None
) -> tuple[str, list[str]]:
    """Return ``(command, base_args)`` for running tests against *files*."""
    if existing_tool:
        tool = existing_tool.strip().lower()
        if tool in ("jest", "npx jest"):
            return ("npx", ["jest"])
        if tool == "pytest":
            return ("pytest", [])
        if tool in ("dotnet", "dotnet test"):
            return ("dotnet", ["test"])
    ext_map: dict[str, tuple[str, list[str]]] = {
        ".cs": ("dotnet", ["test"]),
        ".py": ("pytest", []),
        ".ts": ("npx", ["jest"]),
        ".tsx": ("npx", ["jest"]),
        ".js": ("npx", ["jest"]),
        ".jsx": ("npx", ["jest"]),
    }
    for f in files:
        suffix = Path(f).suffix.lower()
        if suffix in ext_map:
            return ext_map[suffix]
    return ("pytest", [])


def _filter_test_files(files: list[str]) -> list[str]:
    """Keep only paths that look like test files."""
    indicators = ("test", "tests", "spec", "_test.", ".test.", ".spec.")
    return [f for f in files if any(ind in f.lower() for ind in indicators)]


def _render_plan_markdown(task: Task) -> str:
    """
    Render a demo-friendly markdown view of the approved plan using task metadata.

    We do this in the web dashboard because exported plan files can live outside the
    dashboard process' accessible filesystem.
    """
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    repo_summary = meta.get("repository_summary") or {}
    if not isinstance(repo_summary, dict):
        repo_summary = {}
    plan_preview = meta.get("plan_preview") or {}
    if not isinstance(plan_preview, dict):
        plan_preview = {}
    plan_data = meta.get("plan") or {}
    if not isinstance(plan_data, dict):
        plan_data = {}
    if not plan_preview.get("steps") and plan_data:
        plan_preview = plan_data

    steps = plan_preview.get("steps") or []
    risks = plan_preview.get("risks") or []
    refactors = plan_preview.get("refactors") or []

    title = (task.title or "").strip() or (task.description or "").strip().splitlines()[0].strip() if (task.description or "").strip() else "Plan"
    if len(title) > 120:
        title = title[:117].rstrip() + "..."

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- **Repo**: `{task.repo_path}`")
    lines.append(f"- **Branch**: `{task.branch}`")
    lines.append("")

    repo_summary = meta.get("repository_summary") or {}
    snapshot_lines: list[str] = []
    languages = repo_summary.get("top_languages") or []
    if languages:
        snapshot_lines.append(f"- Languages: {', '.join(languages[:5])}")
    modules = repo_summary.get("top_modules") or []
    if modules:
        module_names = [str(m).split('(', 1)[0].strip() for m in modules[:5]]
        snapshot_lines.append(f"- Key modules: {', '.join(module_names)}")
    hotspots = repo_summary.get("hotspots") or []
    if hotspots:
        hotspot_names = [h.get("path") for h in hotspots if h.get("path")]
        if hotspot_names:
            snapshot_lines.append(f"- Hotspots: {', '.join(hotspot_names[:5])}")
    tests_flag = repo_summary.get("has_tests")
    if tests_flag is not None:
        snapshot_lines.append(f"- Tests detected: {'yes' if tests_flag else 'no'}")
    scoped = (repo_summary.get("scoped_context") or {})
    impact = scoped.get("impact") or {}
    directories = impact.get("top_directories") or []
    if directories:
        snapshot_lines.append(f"- Scoped directories: {', '.join(directories[:6])}")
    files_sample = impact.get("files_sample") or []
    if files_sample:
        sample_list = ", ".join(files_sample[:6])
        snapshot_lines.append(f"- Sample files: {sample_list}")
    if snapshot_lines:
        lines.append("## Repository snapshot")
        lines.append("")
        lines.extend(snapshot_lines)
        lines.append("")

    desc = (task.description or "").strip()
    lines.append("## Change request")
    lines.append("")
    lines.append(desc or "_(missing description)_")
    lines.append("")

    # Acceptance criteria (derived for demo friendliness)
    ac = _derive_acceptance_criteria(task, meta.get("clarifications") or [], steps if isinstance(steps, list) else [])
    if ac:
        lines.append("## Acceptance criteria")
        lines.append("")
        lines.extend([f"- {c}" for c in ac[:20]])
        lines.append("")

    # Impacted files (sample) if bounded context exists
    bounded_ctx = meta.get("bounded_context") or {}
    if isinstance(bounded_ctx, dict):
        scoped = bounded_ctx.get("manual") or bounded_ctx.get("plan_targets") or {}
        if isinstance(scoped, dict):
            impact = scoped.get("impact") or {}
            if isinstance(impact, dict):
                files_sample = impact.get("files_sample") or []
                if isinstance(files_sample, list) and files_sample:
                    lines.append("## Impacted files (sample)")
                    lines.append("")
                    lines.extend([f"- `{p}`" for p in files_sample[:50]])
                    lines.append("")

    lines.append("## Plan steps")
    lines.append("")
    if isinstance(steps, list) and steps:
        for idx, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                d = (step.get("description") or "").strip() or "(missing step description)"
                lines.append(f"{idx}. {d}")
                targets = step.get("target_files") or []
                if targets:
                    lines.append(f"   - Targets: {', '.join(f'`{t}`' for t in targets)}")
                notes = (step.get("notes") or "").strip()
                if notes:
                    lines.append(f"   - Notes: {notes}")
            else:
                lines.append(f"{idx}. {str(step)}")
    else:
        lines.append("_No plan steps generated._")
    lines.append("")

    # Scenarios (Given / When / Then) – helpful for future tests.
    has_tests = bool((meta.get("repository_summary") or {}).get("has_tests", False)) if isinstance(meta.get("repository_summary"), dict) else False

    def _infer_step_scenario(description: str, notes: str | None = None) -> list[str]:
        desc2 = (description or "").strip()
        if not desc2:
            return []
        desc_l = desc2.lower()
        given = "Given the codebase is indexed and the current behavior is understood"
        if desc_l.startswith("implement"):
            when = f"When we {desc_l}"
            then = "Then the new component exists and is ready to be integrated"
        elif desc_l.startswith("update") or desc_l.startswith("refactor"):
            when = f"When we {desc_l}"
            then = "Then the existing flow uses the updated configuration path"
        elif "test" in desc_l and has_tests:
            when = f"When we {desc_l}"
            then = "Then automated tests validate both happy-path and failure cases"
        elif desc_l.startswith("document"):
            when = f"When we {desc_l}"
            then = "Then the team can maintain the setup confidently without tribal knowledge"
        else:
            when = f"When we {desc_l}"
            then = "Then the change is implemented with clear acceptance criteria"
        out = [f"Given: {given}", f"When:  {when}", f"Then:  {then}"]
        n = (notes or "").strip()
        if n:
            out.append(f"And:   {n}")
        return out

    scenario_lines: list[str] = []
    if isinstance(steps, list):
        for idx, step in enumerate(steps, start=1):
            if isinstance(step, dict):
                d = (step.get("description") or "").strip()
                notes = step.get("notes")
            else:
                d = str(step).strip()
                notes = None
            if not d:
                continue
            sc = _infer_step_scenario(d, notes if isinstance(notes, str) else None)
            if not sc:
                continue
            if scenario_lines:
                scenario_lines.append("")
            scenario_lines.append(f"### Step {idx}: {d}")
            scenario_lines.extend([f"- {line}" for line in sc])

    if scenario_lines:
        lines.append("## Scenarios (Given / When / Then)")
        lines.append("")
        lines.extend(scenario_lines)
        lines.append("")

    # Test plan (simple, demo-friendly)
    if has_tests:
        lines.append("## Test plan")
        lines.append("")
        lines.append("- Run the existing automated test suite.")
        lines.append("- Add or update tests to cover the new/changed behavior.")
        lines.append("- Validate at least one happy-path and one failure-path scenario.")
        lines.append("")
    else:
        lines.append("## Test plan")
        lines.append("")
        lines.append("- Run basic smoke checks for the changed flow.")
        lines.append("- If the repo has a preferred test framework, add a minimal regression test for the key behavior.")
        lines.append("")

    if isinstance(risks, list) and risks:
        lines.append("## Risks")
        lines.append("")
        lines.extend([f"- {str(r).strip()}" for r in risks if str(r).strip()] or ["_None_"])
        lines.append("")

    if isinstance(refactors, list) and refactors:
        lines.append("## Refactor suggestions")
        lines.append("")
        lines.extend([f"- {str(r).strip()}" for r in refactors if str(r).strip()] or ["_None_"])
        lines.append("")

    specs = meta.get("boundary_specs") or []
    if isinstance(specs, list):
        lines.append("## Boundary specs (summary)")
        lines.append("")
        if specs:
            lines.append(f"- Count: {len(specs)}")
            for spec in specs:
                name = (spec.get("boundary_name") or "Unknown").strip()
                status = (spec.get("status") or "").strip()
                plan_step = (spec.get("plan_step") or "").strip()
                desc = (spec.get("human_description") or "").strip().splitlines()[0] if isinstance(spec, dict) else ""
                if desc and len(desc) > 140:
                    desc = desc[:137].rstrip() + "..."
                status_suffix = f" ({status})" if status else ""
                step_suffix = f" — step: {plan_step}" if plan_step else ""
                desc_suffix = f" — {desc}" if desc else ""
                lines.append(f"- `{name}`{status_suffix}{step_suffix}{desc_suffix}")
        else:
            lines.append("_None_")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _derive_acceptance_criteria(task: Task, clarifications: list[dict], plan_steps: list[object]) -> list[str]:
    """
    Derive human-friendly acceptance criteria.

    Priority:
    1) Explicit `task.metadata["acceptance_criteria"]` (if provided)
    2) Plan steps (Then/outcome lines inferred from the plan)
    3) High-signal clarified constraints (avoid low-signal answers like "I don't know")
    """
    meta = task.metadata if isinstance(task.metadata, dict) else {}

    # 1) Explicit
    meta_ac = meta.get("acceptance_criteria")
    if isinstance(meta_ac, list):
        explicit = [str(x).strip() for x in meta_ac if str(x).strip()]
        if explicit:
            return explicit[:20]

    has_tests = bool((meta.get("repository_summary") or {}).get("has_tests", False)) if isinstance(meta.get("repository_summary"), dict) else False

    def _infer_step_scenario(description: str, notes: str | None = None) -> list[str]:
        desc = (description or "").strip()
        if not desc:
            return []
        desc_l = desc.lower()
        notes_clean = (notes or "").strip()
        given = "Given the codebase is indexed and the current behavior is understood"
        if desc_l.startswith("implement"):
            when = f"When we {desc_l}"
            then = "Then the new component exists and is ready to be integrated"
        elif desc_l.startswith("update") or desc_l.startswith("refactor"):
            when = f"When we {desc_l}"
            then = "Then the existing flow uses the updated configuration path"
        elif "test" in desc_l and has_tests:
            when = f"When we {desc_l}"
            then = "Then automated tests validate both happy-path and failure cases"
        elif desc_l.startswith("document"):
            when = f"When we {desc_l}"
            then = "Then the team can maintain the setup confidently without tribal knowledge"
        else:
            when = f"When we {desc_l}"
            then = "Then the change is implemented with clear acceptance criteria"
        out = [f"Given: {given}", f"When:  {when}", f"Then:  {then}"]
        if notes_clean:
            out.append(f"And:   {notes_clean}")
        return out

    # 2) From plan steps (best default; derived from description)
    derived: list[str] = []
    if isinstance(plan_steps, list) and plan_steps:
        for step in plan_steps[:8]:
            if isinstance(step, dict):
                desc = (step.get("description") or "").strip()
                notes = step.get("notes")
                notes_s = str(notes).strip() if isinstance(notes, str) else None
            else:
                desc = str(step).strip()
                notes_s = None
            if not desc:
                continue
            scenario = _infer_step_scenario(desc, notes_s)
            then_lines = [l for l in scenario if l.startswith("Then:")]
            if then_lines:
                # "Then:  X" -> "X"
                then_text = then_lines[0].split("Then:", 1)[1].strip()
                if then_text:
                    derived.append(then_text)
            else:
                # fallback to the step description
                derived.append(desc)
        # De-dupe while preserving order
        seen = set()
        deduped = []
        for item in derived:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item.strip())
        if deduped:
            return deduped[:10]

    # 3) High-signal clarification constraints only (fallback)
    low_signal = {
        "all",
        "you tell me",
        "idk",
        "i don't know",
        "i dont know",
        "dont know",
        "no",
        "no i think",
        "not sure",
        "maybe",
        "i think",
        "i am knew so i need to know",
        "i need to know",
    }
    constraints: list[str] = []
    for it in clarifications:
        if not isinstance(it, dict):
            continue
        if str(it.get("status") or "").strip().upper() != "ANSWERED":
            continue
        answer = str(it.get("answer") or "").strip()
        if not answer:
            continue
        a_l = answer.lower().strip()
        if a_l in low_signal:
            continue
        # Heuristic: include only answers that look like a requirement/constraint.
        looks_like_constraint = any(tok in a_l for tok in ("must", "should", "at least", "no ", "only ", "min", "maximum", "max ")) or any(ch.isdigit() for ch in a_l)
        if not looks_like_constraint:
            continue
        constraints.append(answer if len(answer) <= 140 else answer[:140].rstrip() + "…")

    # De-dupe
    seen = set()
    out: list[str] = []
    for c in constraints:
        k = c.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out[:10]


def _derive_scenarios(task: Task, plan_steps: list[object]) -> list[dict]:
    """
    Derive Given/When/Then scenarios from plan steps for UI display.
    """
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    has_tests = bool((meta.get("repository_summary") or {}).get("has_tests", False)) if isinstance(meta.get("repository_summary"), dict) else False

    def _infer_step_scenario(description: str, notes: str | None = None) -> list[str]:
        desc = (description or "").strip()
        if not desc:
            return []
        desc_l = desc.lower()
        notes_clean = (notes or "").strip()
        given = "Given the codebase is indexed and the current behavior is understood"
        if desc_l.startswith("implement"):
            when = f"When we {desc_l}"
            then = "Then the new component exists and is ready to be integrated"
        elif desc_l.startswith("update") or desc_l.startswith("refactor"):
            when = f"When we {desc_l}"
            then = "Then the existing flow uses the updated configuration path"
        elif "test" in desc_l and has_tests:
            when = f"When we {desc_l}"
            then = "Then automated tests validate both happy-path and failure cases"
        elif desc_l.startswith("document"):
            when = f"When we {desc_l}"
            then = "Then the team can maintain the setup confidently without tribal knowledge"
        else:
            when = f"When we {desc_l}"
            then = "Then the change is implemented with clear acceptance criteria"
        out = [f"Given: {given}", f"When:  {when}", f"Then:  {then}"]
        if notes_clean:
            out.append(f"And:   {notes_clean}")
        return out

    scenarios: list[dict] = []
    if not isinstance(plan_steps, list):
        return scenarios
    for idx, step in enumerate(plan_steps[:10], start=1):
        if isinstance(step, dict):
            desc = (step.get("description") or "").strip()
            notes = step.get("notes")
            notes_s = str(notes).strip() if isinstance(notes, str) else None
        else:
            desc = str(step).strip()
            notes_s = None
        if not desc:
            continue
        lines = _infer_step_scenario(desc, notes_s)
        if not lines:
            continue
        scenarios.append({"index": idx, "title": desc, "lines": lines})
    return scenarios

def _latest_logs(logs: Iterable[LogEntry], *, limit: int) -> list[LogEntry]:
    items: list[LogEntry] = []
    for e in logs:
        if not _safe_dt(getattr(e, "timestamp", None)):
            continue
        items.append(e)
    items.sort(key=lambda e: e.timestamp, reverse=True)
    return items[: max(1, int(limit or 50))]


def _patch_counts(task: Task) -> Dict[str, int]:
    patch_state = task.metadata.get("patch_queue_state") if isinstance(task.metadata, dict) else None
    pending = applied = rejected = 0
    if isinstance(patch_state, list):
        for item in patch_state:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "")
            if status == "PENDING":
                pending += 1
            elif status == "APPLIED":
                applied += 1
            elif status == "REJECTED":
                rejected += 1
    return {"pending": pending, "applied": applied, "rejected": rejected, "total": pending + applied + rejected}


def _pending_clarifications(task: Task) -> int:
    items = task.metadata.get("clarifications", []) if isinstance(task.metadata, dict) else []
    if not isinstance(items, list):
        return 0
    pending = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().upper()
        if status == "PENDING" or status == "":
            pending += 1
    return pending


def _task_bucket(status: TaskStatus) -> str:
    if status in {TaskStatus.CANCELLED}:
        return "failed"
    if status in {TaskStatus.COMPLETED}:
        return "completed"
    if status in {TaskStatus.CREATED, TaskStatus.CLARIFYING, TaskStatus.PLANNING, TaskStatus.SPEC_PENDING, TaskStatus.IMPLEMENTING, TaskStatus.VERIFYING}:
        return "running"
    return "running"


_MAX_DIFF_DISPLAY_BYTES = 50_000

_WORKFLOW_ORDER: list[Tuple[str, str]] = [
    ("TASK_SPECIFICATION", "Task Specification"),
    ("CLARIFYING", "Clarifying Questions"),
    ("PLANNING", "Context Analysis & Preplan"),
    ("CODEGEN", "Code Generation"),
    ("RUN_TESTS", "Run Tests"),
]


def _infer_workflow(task: Task, logs: list[LogEntry]) -> list[Dict[str, Any]]:
    """
    Build a mockup-like workflow timeline from task metadata + log events.

    We don't have precise per-step runtimes, so we approximate durations using the gaps
    between key events.
    """
    # Find key timestamps.
    created_at = task.created_at
    plan_generated_at: datetime | None = None
    plan_approved_at: datetime | None = None
    patches_generated_at: datetime | None = None

    for entry in sorted([e for e in logs if e.task_id == task.id], key=lambda e: e.timestamp):
        if entry.entry_type == "PLAN_GENERATED":
            plan_generated_at = entry.timestamp
        elif entry.entry_type == "PLAN_APPROVED":
            plan_approved_at = entry.timestamp
        elif entry.entry_type == "PATCHES_GENERATED":
            patches_generated_at = entry.timestamp

    pending_clarifications = _pending_clarifications(task)
    has_plan = False
    if isinstance(task.metadata, dict):
        has_plan = bool(task.metadata.get("plan_preview") or task.metadata.get("plan"))
    pending_specs = task.metadata.get("pending_specs") if isinstance(task.metadata, dict) else None
    pending_specs_count = len(pending_specs) if isinstance(pending_specs, list) else 0
    plan_approved = bool(task.metadata.get("plan_approved")) if isinstance(task.metadata, dict) else False
    patches = _patch_counts(task)
    external_patch_pending = False
    if isinstance(task.metadata, dict):
        patch_queue = task.metadata.get("patch_queue_state")
        if isinstance(patch_queue, list):
            for patch in patch_queue:
                if not isinstance(patch, dict):
                    continue
                status = str(patch.get("status") or "").strip().upper()
                if status and status != "PENDING":
                    continue
                if patch.get("applied_diff"):
                    continue
                diff_text = str(patch.get("diff") or "")
                if diff_text.startswith(EXTERNAL_PATCH_SENTINEL):
                    external_patch_pending = True
                    break

    def _dur(a: datetime | None, b: datetime | None) -> float | None:
        if not a or not b:
            return None
        return max(0.0, (b - a).total_seconds())

    # Step boundaries (best-effort).
    spec_start = created_at
    spec_end = created_at  # task creation is immediate in our system
    clar_start = created_at
    clar_end = plan_generated_at
    plan_start = plan_generated_at or created_at
    plan_end = plan_approved_at if plan_approved_at else (plan_generated_at or None)
    approve_start = plan_generated_at
    approve_end = plan_approved_at
    codegen_start = plan_approved_at
    codegen_end = patches_generated_at

    # Determine active step.
    active_key = "TASK_SPECIFICATION"
    if pending_clarifications > 0:
        active_key = "CLARIFYING"
    elif has_plan and not plan_approved:
        active_key = "PLANNING"
    elif plan_approved and (patches.get("total", 0) == 0):
        active_key = "CODEGEN"
    elif plan_approved and (patches.get("pending", 0) > 0):
        active_key = "CODEGEN"

    _wf_meta = task.metadata if isinstance(task.metadata, dict) else {}

    steps: list[Dict[str, Any]] = []
    for key, label in _WORKFLOW_ORDER:
        done = False
        started_at: datetime | None = None
        ended_at: datetime | None = None
        chips: list[Dict[str, str]] = []

        if key == "TASK_SPECIFICATION":
            done = True
            started_at, ended_at = spec_start, spec_end
        elif key == "CLARIFYING":
            done = pending_clarifications == 0
            started_at, ended_at = clar_start, clar_end
            if pending_clarifications > 0:
                chips.append({"kind": "warning", "text": f"{pending_clarifications} pending"})
        elif key == "PLANNING":
            done = has_plan and plan_approved
            started_at = plan_generated_at
            ended_at = plan_approved_at if plan_approved else plan_generated_at
            if pending_specs_count > 0:
                chips.append({"kind": "warning", "text": f"{pending_specs_count} specs pending"})
            if plan_approved:
                chips.append({"kind": "success", "text": "user approved"})
        elif key == "CODEGEN":
            done = (patches.get("total", 0) > 0) or task.status in {TaskStatus.VERIFYING, TaskStatus.COMPLETED}
            started_at, ended_at = codegen_start, codegen_end
            if patches.get("total", 0) > 0:
                chips.append({"kind": "neutral", "text": f"{patches['total']} patches"})
            if external_patch_pending:
                chips.append({"kind": "warning", "text": "External edit required"})
        elif key == "RUN_TESTS":
            test_exit = _wf_meta.get("test_exit_code")
            has_test_output = bool(_wf_meta.get("test_output"))
            done = task.status in {TaskStatus.COMPLETED} or (task.status == TaskStatus.VERIFYING and has_test_output)
            started_at = patches_generated_at
            ended_at = None
            if task.status == TaskStatus.VERIFYING and not has_test_output:
                active_key = "RUN_TESTS"
            elif task.status == TaskStatus.VERIFYING and has_test_output:
                ended_at = task.updated_at
            elif task.status == TaskStatus.COMPLETED:
                ended_at = task.updated_at

        status = "pending"
        if done:
            status = "done"
        elif key == active_key:
            status = "active"

        step_entry: Dict[str, Any] = {
                "key": key,
                "label": label,
                "status": status,
                "started_at": _iso(started_at) if started_at else None,
                "ended_at": _iso(ended_at) if ended_at else None,
                "duration_seconds": _dur(started_at, ended_at),
                "chips": chips,
        }
        if key == "RUN_TESTS" and isinstance(_wf_meta.get("test_exit_code"), int) and _wf_meta.get("test_exit_code") != 0:
            step_entry["test_failed"] = True
        steps.append(step_entry)

    return steps


def _serialize_task(task: Task, latest_log: LogEntry | None) -> Dict[str, Any]:
    patches = _patch_counts(task)
    duration_seconds: float | None = None
    try:
        delta = (task.updated_at - task.created_at).total_seconds()
        if delta > 0:
            duration_seconds = round(delta, 1)
    except Exception:
        pass
    return {
        "id": task.id,
        "client": (task.client or "").strip() or None,
        "title": _task_title(task),
        "summary": _task_summary(task),
        "description": task.description,
        "status": task.status.value,
        "bucket": "failed" if (
            isinstance(task.metadata, dict)
            and isinstance(task.metadata.get("test_exit_code"), int)
            and task.metadata["test_exit_code"] != 0
        ) else _task_bucket(task.status),
        "created_at": _iso(task.created_at),
        "updated_at": _iso(task.updated_at),
        "repo_path": str(task.repo_path),
        "branch": task.branch,
        "last_event": latest_log.entry_type if latest_log else None,
        "last_event_at": _iso(latest_log.timestamp) if latest_log else None,
        "patch_counts": patches,
        "plan_approved": bool(task.metadata.get("plan_approved")) if isinstance(task.metadata, dict) else False,
        "duration_seconds": duration_seconds,
    }


def _serialize_log(entry: LogEntry) -> Dict[str, Any]:
    return {
        "id": entry.id,
        "task_id": entry.task_id,
        "timestamp": _iso(entry.timestamp),
        "entry_type": entry.entry_type,
        "payload": entry.payload,
    }


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Spec Agent Dashboard</title>
  <link rel="stylesheet" href="/styles.css" />
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="logo">R</div>
      <div class="name">RiSpec</div>
    </div>
    <div class="status">
      <span class="dot live"></span>
      <span id="liveLabel">Live</span>
    </div>
  </div>

  <div class="grid">
    <aside class="panel tasks">
      <div class="panelHeader">
        <div class="panelTitle">Tasks</div>
        <div class="tabs">
          <button class="tab active" data-filter="all">All</button>
          <button class="tab" data-filter="running">Running</button>
          <button class="tab" data-filter="failed">Failed</button>
        </div>
        <div class="range">
          <button class="pill" data-range="5m">5m</button>
          <button class="pill" data-range="1h">1h</button>
          <button class="pill active" data-range="today">Today</button>
          <button class="pill" data-range="all">All time</button>
        </div>
      </div>
      <div id="taskList" class="taskList"></div>
    </aside>

    <main class="panel workflow">
      <div class="panelHeader">
        <div class="panelTitle">Workflow</div>
        <div class="panelSubtitle" id="workflowSubtitle">Select a task</div>
      </div>
      <div id="workflowSteps" class="steps"></div>
    </main>

    <section class="panel details">
      <div class="panelHeader" style="display:flex;align-items:center;gap:10px">
        <div class="panelTitle">Evidence &amp; Details</div>
        <div class="detailActions">
          <button class="iconBtn" id="btnRefresh" title="Refresh">&#x21bb;</button>
          <button class="iconBtn" id="btnPin" title="Pin"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M5 1v4L3 7v1h4v6l1 1 1-1V8h4V7l-2-2V1"/></svg></button>
          <button class="iconBtn" id="btnTag" title="Tag"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 1h6l7 7-6 6-7-7V1z"/><circle cx="4.5" cy="4.5" r="1" fill="currentColor"/></svg></button>
        </div>
      </div>
      <div id="detailsBody" class="detailsBody">
        <div class="empty">Select a step to view details</div>
      </div>
    </section>
  </div>

  <script src="/app.js"></script>
</body>
</html>
"""


def _plan_html(task_id: str) -> str:
    """
    Full-page plan view for demos: renders the plan markdown in a dedicated page.
    """
    safe_task_id = (task_id or "").replace('"', "").replace("'", "").strip()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Plan</title>
  <link rel="stylesheet" href="/styles.css" />
</head>
<body class="planBody">
  <div class="topbar">
    <div class="brand">
      <div class="logo">R</div>
      <div class="name">RiSpec</div>
    </div>
    <div class="status">
      <a class="backLink" href="/">← Back to dashboard</a>
    </div>
  </div>

  <main class="planWrap">
    <div class="planHeader">
      <div class="planTitle" id="planTitle">Plan</div>
      <div class="planSubtitle" id="planSubtitle">Loading…</div>
      <div class="planActions">
        <a class="pillLink" id="downloadLink" target="_blank" rel="noreferrer">Open / download markdown</a>
      </div>
    </div>
    <pre class="diff planContent" id="planContent">Loading…</pre>
  </main>

  <script>
    const TASK_ID = "{safe_task_id}";
    const download = document.getElementById('downloadLink');
    download.href = `/api/plan-markdown/${{encodeURIComponent(TASK_ID)}}`;

    fetch(`/api/task/${{encodeURIComponent(TASK_ID)}}`, {{ cache: 'no-store' }})
      .then(r => r.ok ? r.json() : Promise.reject(new Error('task not found')))
      .then(data => {{
        const t = data.task || {{}};
        const rawClient = (t.client && String(t.client).trim()) || '';
        const clientLabel = rawClient
          ? (rawClient.toLowerCase() === 'cursor' ? 'Cursor' : rawClient.toLowerCase() === 'claude' ? 'Claude' : rawClient)
          : 'CLI';
        document.getElementById('planTitle').textContent = t.title ? t.title : 'Plan';
        document.getElementById('planSubtitle').textContent = `${{clientLabel}} · ${{t.repo_path || '—'}} · ${{t.branch || '—'}}`;
      }})
      .catch(() => {{
        document.getElementById('planSubtitle').textContent = 'Task not found';
      }});

    fetch(`/api/plan-markdown/${{encodeURIComponent(TASK_ID)}}`, {{ cache: 'no-store' }})
      .then(r => r.ok ? r.text() : Promise.reject(new Error('plan not found')))
      .then(txt => {{
        document.getElementById('planContent').textContent = txt || '—';
      }})
      .catch(() => {{
        document.getElementById('planContent').textContent = 'Plan markdown not available.';
      }});
  </script>
</body>
</html>
"""


_STYLES_CSS = """
:root{
  --bg:#070a12;
  --panel:#0b1120;
  --panel2:#0a1020;
  --border:#131b2d;
  --text:#e5e7eb;
  --muted:#94a3b8;
  --accent:#3b82f6;
  --good:#22c55e;
  --warn:#f59e0b;
  --info:#38bdf8;
  --bad:#ef4444;
  --chip:#111a2f;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;
  background: radial-gradient(1200px 600px at 30% -20%, rgba(59,130,246,.25), transparent 60%),
              radial-gradient(900px 500px at 110% 30%, rgba(34,197,94,.12), transparent 55%),
              var(--bg);
  color:var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
.planBody{
  overflow:auto;
}
.backLink{
  color: var(--muted);
  text-decoration:none;
  font-size:13px;
}
.backLink:hover{color:#e2e8f0}
.planWrap{
  max-width: 1100px;
  margin: 0 auto;
  padding: 16px;
}
.planHeader{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:14px;
  margin-top: 14px;
}
.planTitle{font-size:16px; font-weight:800; color:#f1f5f9}
.planSubtitle{margin-top:6px; font-size:12px; color:var(--muted)}
.planActions{margin-top:10px}
.pillLink{
  display:inline-block;
  font-size:12px;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid var(--border);
  background:rgba(148,163,184,.08);
  color:#e2e8f0;
  text-decoration:none;
}
.pillLink:hover{border-color:rgba(59,130,246,.35)}
.planContent{
  margin-top: 14px;
  max-height: none;
  width: 100%;
}
.topbar{
  height:56px;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 16px;
  border-bottom:1px solid var(--border);
  background:rgba(7,10,18,.6);
  backdrop-filter: blur(10px);
}
.brand{display:flex; align-items:center; gap:10px}
.logo{
  width:28px;height:28px;border-radius:8px;
  background:linear-gradient(135deg, #2563eb, #7c3aed);
  display:grid;place-items:center;
  font-weight:700;
}
.name{font-weight:600; letter-spacing:.2px}
.status{display:flex; align-items:center; gap:8px; color:var(--muted); font-size:13px}
.dot{width:8px; height:8px; border-radius:999px; background:var(--muted); display:inline-block}
.dot.live{background:var(--good)}

.grid{
  height:calc(100% - 56px);
  display:grid;
  grid-template-columns: 360px 360px 1fr;
  gap:0;
}
.panel{
  border-right:1px solid var(--border);
  background:linear-gradient(180deg, rgba(11,17,32,.92), rgba(10,16,32,.82));
  min-height:0;
}
.panel.details{border-right:none}
.panelHeader{
  padding:14px 14px 10px 14px;
  border-bottom:1px solid var(--border);
}
.panelTitle{font-size:14px; font-weight:700; color:#f1f5f9}
.panelSubtitle{margin-top:4px; font-size:12px; color:var(--muted)}
.tabs{display:flex; gap:8px; margin-top:12px}
.tab{
  font-size:12px;
  padding:6px 10px;
  border-radius:8px;
  border:1px solid var(--border);
  background:transparent;
  color:var(--muted);
  cursor:pointer;
}
.tab.active{background:rgba(59,130,246,.15); color:#dbeafe; border-color:rgba(59,130,246,.35)}
.range{display:flex; gap:8px; margin-top:10px}
.pill{
  font-size:12px;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid var(--border);
  background:transparent;
  color:var(--muted);
  cursor:pointer;
}
.pill.active{background:rgba(148,163,184,.1); color:#e2e8f0}

.taskList{padding:10px; overflow:auto; height:calc(100% - 110px)}
.taskCard{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:12px;
  margin-bottom:10px;
  cursor:pointer;
}
.taskCard:hover{border-color:rgba(59,130,246,.35)}
.taskCard.selected{outline:2px solid rgba(59,130,246,.35)}
.taskRow{display:flex; align-items:center; justify-content:space-between; gap:10px}
.taskRow .taskTitle{flex:1; min-width:0}
.taskRow .timeLabel{flex-shrink:0; color:#cbd5ff; font-size:12px}
.timeLabel{color:#cbd5ff;font-size:12px}
.taskTitle{
  font-size:13px;
  font-weight:650;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}
.taskMeta{margin-top:8px; display:flex; gap:8px; flex-wrap:wrap}
.chip{
  font-size:11px;
  padding:4px 8px;
  border-radius:999px;
  border:1px solid var(--border);
  background:rgba(17,26,47,.55);
  color:var(--muted);
}
.chip.purple{border-color:rgba(124,58,237,.35); color:#e9d5ff; background:rgba(124,58,237,.12)}
.chip.green{border-color:rgba(34,197,94,.35); color:#bbf7d0; background:rgba(34,197,94,.10)}
.chip.yellow{border-color:rgba(245,158,11,.35); color:#fde68a; background:rgba(245,158,11,.12)}
.chip.blue{border-color:rgba(56,189,248,.40); color:#e0f2fe; background:rgba(56,189,248,.14)}
.chip.red{border-color:rgba(239,68,68,.35); color:#fecaca; background:rgba(239,68,68,.10)}
.stepDot{width:12px;height:12px;border-radius:999px;border:2px solid var(--muted);flex-shrink:0;margin-left:auto}
.stepDot.good{border-color:var(--good);background:var(--good)}
.stepDot.bad{border-color:var(--bad);background:var(--bad)}
.stepDot.active{border-color:var(--accent);background:transparent}
.dot2{width:10px;height:10px;border-radius:999px; background:var(--muted)}
.dot2.good{background:var(--good)}
.dot2.warn{background:var(--warn)}
.dot2.bad{background:var(--bad)}
.dot2.blue{background:var(--info)}

.steps{padding:12px; overflow:auto; height:calc(100% - 58px)}
.step{
  display:flex;
  gap:12px;
  padding:12px;
  border-radius:12px;
  border:1px solid transparent;
}
.step:hover{background:rgba(148,163,184,.06)}
.step.selected{border-color:rgba(59,130,246,.35); background:rgba(59,130,246,.08)}
.rail{
  width:8px;
  border-radius:999px;
  background:rgba(148,163,184,.15);
  position:relative;
  overflow:hidden;
}
.rail .fill{position:absolute; left:0; top:0; width:100%; height:100%; background:rgba(34,197,94,.65)}
.rail .fill.active{background:rgba(59,130,246,.75)}
.stepBody{flex:1}
.stepLabelRow{display:flex;align-items:center;gap:8px}
.stepIcon{width:18px;height:18px;color:var(--muted);flex-shrink:0}
.stepIcon svg{width:18px;height:18px}
.detailHeaderRow{display:flex;align-items:center;gap:8px}
.detailHeaderIcon{width:18px;height:18px;color:var(--muted);flex-shrink:0}
.detailHeaderIcon svg{width:18px;height:18px}
.stepLabel{font-size:13px; font-weight:650}
.stepChips{margin-top:8px; display:flex; gap:8px; flex-wrap:wrap}
.small{font-size:12px; color:var(--muted); margin-top:6px}
.detailsBody{padding:16px}
.details{overflow:auto}
.detailActions{display:flex;gap:6px;margin-left:auto}
.iconBtn{
  width:28px;height:28px;border-radius:6px;
  border:1px solid var(--border);
  background:transparent;
  color:var(--muted);
  cursor:pointer;
  display:grid;place-items:center;
  font-size:14px;
  padding:0;
}
.iconBtn:hover{background:rgba(148,163,184,.1);color:#e2e8f0}
.iconBtn svg{width:14px;height:14px}
.card{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:14px;
}
.cardTitle{font-weight:750; font-size:16px}
.cardBody{margin-top:10px; color:var(--muted); font-size:13px; line-height:1.5}
.banner{
  border:1px solid rgba(34,197,94,.25);
  background:rgba(34,197,94,.10);
  color:#bbf7d0;
  padding:10px 12px;
  border-radius:12px;
  margin-bottom:12px;
  font-size:13px;
  font-weight:650;
}
.banner.warn{
  border-color:rgba(245,158,11,.30);
  background:rgba(245,158,11,.10);
  color:#fde68a;
}
.sectionTitle{
  margin-top:16px;
  margin-bottom:8px;
  font-size:12px;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:var(--muted);
}
.taskHero{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:14px;
}
.taskHeroTop{display:flex; align-items:flex-start; justify-content:space-between; gap:12px}
.taskHeroTitle{font-weight:800; font-size:16px; color:#f1f5f9}
.taskHeroDesc{margin-top:8px; color:var(--muted); font-size:13px; line-height:1.5}
.overviewHero{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:12px 14px;
  margin-bottom:16px;
}
.overviewTitle{font-weight:700; font-size:15px; color:#f1f5f9}
.overviewSubtitle{margin-top:6px; font-size:12px; color:var(--muted)}
.overviewGrid{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap:10px;
  margin-bottom:16px;
}
.infoCard{
  border:1px solid var(--border);
  background:rgba(11,17,32,.5);
  border-radius:10px;
  padding:10px 12px;
}
.infoLabel{font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em}
.infoValue{margin-top:4px; font-size:13px; color:#e2e8f0}
.ctaBox{
  border:1px solid rgba(124,58,237,.45);
  background:rgba(79,70,229,.12);
  border-radius:12px;
  padding:12px 14px;
  margin-top:12px;
  font-size:13px;
  color:#e2e8f0;
}
.copyBtn{
  margin-top:10px;
  padding:6px 10px;
  border-radius:8px;
  border:1px solid rgba(124,58,237,.6);
  background:rgba(124,58,237,.2);
  color:#e5e7eb;
  font-size:12px;
  cursor:pointer;
}
.actionBar{display:flex;gap:8px;margin-top:14px;padding-top:10px;border-top:1px solid var(--border)}
.actionBtn{
  padding:6px 14px;
  border-radius:8px;
  border:1px solid rgba(124,58,237,.6);
  background:rgba(124,58,237,.18);
  color:#e5e7eb;
  font-size:12px;
  cursor:pointer;
  display:inline-flex;align-items:center;gap:6px;
  transition:background .15s,border-color .15s;
}
.actionBtn:hover{background:rgba(124,58,237,.35);border-color:rgba(124,58,237,.8)}
.actionBtn:disabled{opacity:.55;cursor:default}
.actionBtn.success{border-color:rgba(34,197,94,.5);background:rgba(34,197,94,.12);color:#86efac}
.actionBtn.fail{border-color:rgba(239,68,68,.5);background:rgba(239,68,68,.12);color:#fca5a5}
.prio{
  font-size:11px;
  font-weight:800;
  color:#fecaca;
  background:rgba(239,68,68,.12);
  border:1px solid rgba(239,68,68,.28);
  border-radius:999px;
  padding:3px 10px;
  flex:0 0 auto;
}
.acItem{
  border:1px solid var(--border);
  background:rgba(10,16,32,.35);
  border-radius:10px;
  padding:10px 12px;
  margin-top:10px;
  display:flex;
  gap:10px;
  align-items:flex-start;
  font-size:13px;
}
.acN{
  width:22px;height:22px;border-radius:999px;
  background:rgba(59,130,246,.18);
  border:1px solid rgba(59,130,246,.30);
  display:grid; place-items:center;
  color:#dbeafe;
  font-weight:800;
  flex:0 0 auto;
  font-size:12px;
}
.acText{flex:1 1 auto; color:#e2e8f0}
.qcard{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:12px;
  margin-top:10px;
}
.qhead{font-size:13px; font-weight:650; display:flex; gap:10px; align-items:flex-start}
.qbadge{
  font-size:11px;
  font-weight:800;
  color:#e2e8f0;
  background:rgba(124,58,237,.18);
  border:1px solid rgba(124,58,237,.35);
  border-radius:999px;
  padding:2px 8px;
  flex:0 0 auto;
  margin-top:1px;
}
.qtext{flex:1 1 auto; line-height:1.35}
.qans{margin-top:10px; font-size:13px}
.qans.ok{color:#bbf7d0}
.qans.muted{color:var(--muted)}
.qans .qicon{margin-right:8px}
.tree{
  border:1px solid var(--border);
  background:rgba(10,16,32,.45);
  border-radius:12px;
  padding:10px;
  margin-top:10px;
}
.treeDir{
  font-weight:650;
  font-size:13px;
  margin-bottom:4px;
  display:flex;
  align-items:center;
  gap:6px;
}
.treeDir::before{
  content:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 16 16' fill='none' stroke='%2394a3b8' stroke-width='1.5'%3E%3Cpath d='M2 4h4l2 2h6v7H2z' rx='1'/%3E%3C/svg%3E");
  font-size:0;
  vertical-align:middle;
}
.treeFile{
  font-size:12px;
  color:var(--muted);
  padding:2px 0 2px 18px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
  position:relative;
  display:flex;
  justify-content:space-between;
  align-items:center;
}
.treeFile::before{
  content:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 16 16' fill='none' stroke='%2394a3b8' stroke-width='1.5'%3E%3Cpath d='M4 2h5l3 3v9H4z'/%3E%3Cline x1='9' y1='2' x2='9' y2='5'/%3E%3Cline x1='9' y1='5' x2='12' y2='5'/%3E%3C/svg%3E");
  position:absolute;
  left:0;
  top:2px;
  font-size:0;
}
.badge-modify{
  font-size:10px;
  font-weight:700;
  color:#fde68a;
  background:rgba(245,158,11,.15);
  border:1px solid rgba(245,158,11,.35);
  border-radius:999px;
  padding:1px 7px;
  flex-shrink:0;
  margin-left:8px;
}
.badge-create{
  font-size:10px;
  font-weight:700;
  color:#bbf7d0;
  background:rgba(34,197,94,.15);
  border:1px solid rgba(34,197,94,.35);
  border-radius:999px;
  padding:1px 7px;
  flex-shrink:0;
  margin-left:8px;
}
.planItem{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:10px 12px;
  margin-top:10px;
  display:flex;
  gap:10px;
  align-items:flex-start;
  font-size:13px;
}
.planN{
  width:22px;
  height:22px;
  border-radius:999px;
  display:grid;
  place-items:center;
  background:rgba(59,130,246,.18);
  border:1px solid rgba(59,130,246,.35);
  color:#dbeafe;
  font-weight:700;
  font-size:12px;
  flex:0 0 auto;
}
.fileRow{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:10px;
  padding:8px 10px;
  margin-top:8px;
  color:#e2e8f0;
  font-size:12px;
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.patchList{
  border:1px solid var(--border);
  background:rgba(10,16,32,.35);
  border-radius:12px;
  padding:8px;
  margin-top:10px;
}
.patchRow{
  padding:8px 10px;
  border-radius:10px;
  color:#e2e8f0;
  font-size:12px;
  cursor:pointer;
}
.patchRow:hover{background:rgba(148,163,184,.08)}
.patchRow.selected{background:rgba(59,130,246,.18); border:1px solid rgba(59,130,246,.30)}
.diff{
  margin-top:10px;
  border:1px solid var(--border);
  background:rgba(0,0,0,.35);
  border-radius:12px;
  padding:12px;
  overflow:auto;
  max-height:360px;
  font-size:12px;
  line-height:1.4;
  color:#e5e7eb;
}
.diff-file-header{color:#94a3b8}
.diff-hunk{color:#fbbf24}
.diff-add{color:#4ade80}
.diff-del{color:#f87171}
.diff-context{color:#e5e7eb}
.kv{display:grid; grid-template-columns: 140px 1fr; row-gap:10px; column-gap:12px}
.k{color:var(--muted); font-size:12px}
.v{font-size:12px; color:#e2e8f0; word-break:break-word}
.empty{color:var(--muted); font-size:13px; padding:40px 0; text-align:center}
"""


_APP_JS = r"""
let state = {
  tasks: [],
  selectedTaskId: null,
  selectedStepKey: null,
  selectedPatchId: null,
  selectedTaskData: null,
  filter: 'all',
  range: 'all',
  lastNow: null,
};

const $ = (id) => document.getElementById(id);

const STEP_ICONS = {
  TASK_SPECIFICATION: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="2" width="10" height="12" rx="1.5"/><line x1="5.5" y1="5" x2="10.5" y2="5"/><line x1="5.5" y1="8" x2="10.5" y2="8"/><line x1="5.5" y1="11" x2="8.5" y2="11"/></svg>',
  CLARIFYING: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2 3h12v8H5l-3 3V3z" rx="1.5"/><circle cx="5" cy="7" r="0.5" fill="currentColor"/><circle cx="8" cy="7" r="0.5" fill="currentColor"/><circle cx="11" cy="7" r="0.5" fill="currentColor"/></svg>',
  PLANNING: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="12" height="12" rx="1.5"/><line x1="5" y1="5" x2="11" y2="5"/><line x1="5" y1="8" x2="11" y2="8"/><line x1="5" y1="11" x2="9" y2="11"/></svg>',
  CODEGEN: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="5,4 2,8 5,12"/><polyline points="11,4 14,8 11,12"/><line x1="9" y1="3" x2="7" y2="13"/></svg>',
  RUN_TESTS: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 2l1 5h6l1-5"/><path d="M5 7v5a3 3 0 006 0V7"/><line x1="3" y1="2" x2="13" y2="2"/></svg>',
};

function renderHighlightedDiff(pre, text) {
  pre.innerHTML = '';
  const lines = (text || '').split('\n');
  lines.forEach((line, i) => {
    const span = document.createElement('span');
    if (line.startsWith('+++') || line.startsWith('---')) {
      span.className = 'diff-file-header';
    } else if (line.startsWith('@@')) {
      span.className = 'diff-hunk';
    } else if (line.startsWith('+')) {
      span.className = 'diff-add';
    } else if (line.startsWith('-')) {
      span.className = 'diff-del';
    } else {
      span.className = 'diff-context';
    }
    span.textContent = line;
    pre.appendChild(span);
    if (i < lines.length - 1) pre.appendChild(document.createTextNode('\n'));
  });
}

function clientLabelFrom(raw) {
  const val = (raw && String(raw).trim()) || '';
  if (!val) return 'CLI';
  if (val.toLowerCase() === 'cursor') return 'Cursor';
  if (val.toLowerCase() === 'claude') return 'Claude';
  return val;
}

// Match CLI dashboard: short title so "X ago" timestamp stays visible (28 chars like dashboard.py)
const TASK_TITLE_MAX_LEN = 28;
function resolveTaskTitle(task) {
  if (!task) return 'Untitled';
  const raw = (task.title && task.title.trim())
    || (task.summary && task.summary.trim())
    || ((task.description || '').split(/\n/)[0] || '').trim();
  if (!raw) return 'Untitled';
  return raw.length > TASK_TITLE_MAX_LEN ? raw.slice(0, TASK_TITLE_MAX_LEN - 1) + '…' : raw;
}

function fmtCount(value) {
  if (typeof value === 'number' && !Number.isNaN(value)) {
    return value.toLocaleString();
  }
  return '—';
}

function normalizeRisk(raw) {
  const cleanStr = (v) => (typeof v === 'string' ? v.trim() : '');

  const asFields = (obj) => {
    const title = cleanStr(obj.id) || cleanStr(obj.name) || cleanStr(obj.title) || 'Risk';
    const description = cleanStr(obj.description) || cleanStr(obj.desc) || cleanStr(obj.summary) || '';
    const mitigation =
      cleanStr(obj.mitigation) ||
      cleanStr(obj.mitigation_strategy) ||
      cleanStr(obj.mitigationStrategy) ||
      '';
    return { title: title || 'Risk', description, mitigation };
  };

  if (!raw) return { title: 'Risk', description: '', mitigation: '' };

  if (typeof raw === 'object') {
    return asFields(raw);
  }

  if (typeof raw === 'string') {
    const trimmed = raw.trim();

    // Try JSON parsing (both proper JSON and common single-quote dict strings)
    const attempts = [trimmed, trimmed.replace(/'/g, '"')];
    for (const candidate of attempts) {
      try {
        const parsed = JSON.parse(candidate);
        if (parsed && typeof parsed === 'object') {
          return asFields(parsed);
        }
      } catch (e) {
        /* ignore */
      }
    }

    const idMatch = trimmed.match(/['"]?id['"]?\s*:\s*['"]([^'"}]+)['"]/i);
    const descMatch = trimmed.match(/['"]?description['"]?\s*:\s*['"]([^'"}]+)['"]/i);
    const mitMatch = trimmed.match(/['"]?mitigation['"]?\s*:\s*['"]([^'"}]+)['"]/i);

    return {
      title: (idMatch && idMatch[1].trim()) || 'Risk',
      description: (descMatch && descMatch[1].trim()) || trimmed,
      mitigation: (mitMatch && mitMatch[1].trim()) || '',
    };
  }

  return { title: 'Risk', description: String(raw), mitigation: '' };
}

/**
 * Render a compact "bounded context tree" grouped by directory.
 *
 * This is intentionally shallow and demo-friendly: we group by the
 * immediate directory and list a small sample of files under each,
 * using the existing `.tree`, `.treeDir`, `.treeFile` styles.
 */
function renderPathTree(container, paths, maxDirs = 8, maxFilesPerDir = 6, fileBadges = null) {
  const clean = (paths || []).filter(p => typeof p === 'string' && p.trim());
  if (!clean.length) return;

  // If there are no directory separators (e.g. C# namespaces like
  // "Pbp.Payments.CardStore.Domain"), a tree would look fake. Fall back
  // to a simple flat list in that case.
  const hasSlash = clean.some(p => String(p).includes('/'));
  if (!hasSlash) {
    const box = document.createElement('div');
    box.className = 'tree';
    clean.slice(0, maxFilesPerDir * maxDirs).forEach(p => {
      const row = document.createElement('div');
      row.className = 'fileRow';
      row.textContent = p;
      box.appendChild(row);
    });
    container.appendChild(box);
    return;
  }

  const tree = document.createElement('div');
  tree.className = 'tree';

  const byDir = {};
  clean.forEach(raw => {
    const full = String(raw).trim().replace(/\\/g, '/');
    if (!full) return;
    const idx = full.lastIndexOf('/');
    const dir = idx >= 0 ? full.slice(0, idx) : '(root)';
    const file = idx >= 0 ? full.slice(idx + 1) : full;
    if (!file) return;
    if (!byDir[dir]) byDir[dir] = [];
    if (!byDir[dir].includes(file)) {
      byDir[dir].push(file);
    }
  });

  const dirs = Object.keys(byDir).sort().slice(0, maxDirs);
  dirs.forEach(dir => {
    const dirEl = document.createElement('div');
    dirEl.className = 'treeDir';
    dirEl.textContent = dir;
    tree.appendChild(dirEl);

    byDir[dir].slice(0, maxFilesPerDir).forEach(file => {
      const fileEl = document.createElement('div');
      fileEl.className = 'treeFile';
      const nameSpan = document.createElement('span');
      nameSpan.textContent = file;
      nameSpan.style.overflow = 'hidden';
      nameSpan.style.textOverflow = 'ellipsis';
      nameSpan.style.whiteSpace = 'nowrap';
      fileEl.appendChild(nameSpan);
      const fullPath = dir === '(root)' ? file : dir + '/' + file;
      if (fileBadges && fileBadges[fullPath]) {
        const badge = document.createElement('span');
        badge.className = fileBadges[fullPath] === 'CREATE' ? 'badge-create' : 'badge-modify';
        badge.textContent = fileBadges[fullPath] === 'CREATE' ? '+ CREATE' : 'MODIFY';
        fileEl.appendChild(badge);
      }
      tree.appendChild(fileEl);
    });
  });

  if (!dirs.length) return;
  container.appendChild(tree);
}

function renderOverview(container, task, detail) {
  const data = detail || {};
  const gitInfo = data.git_info || {};
  const hero = document.createElement('div');
  hero.className = 'overviewHero';
  const title = document.createElement('div');
  title.className = 'overviewTitle';
  title.textContent = data.repo_path || task.repo_path || '—';
  const subtitle = document.createElement('div');
  subtitle.className = 'overviewSubtitle';
  const clientLabel = clientLabelFrom(task.client);
  subtitle.textContent = `${clientLabel} · ${data.branch || task.branch || '—'}`;
  hero.appendChild(title);
  hero.appendChild(subtitle);
  container.appendChild(hero);

  const metricsGrid = document.createElement('div');
  metricsGrid.className = 'overviewGrid';
  const metrics = [
    { label: 'Files indexed', value: fmtCount(data.file_count) },
    { label: 'Directories', value: fmtCount(data.directory_count) },
    { label: 'Tests detected', value: data.has_tests ? 'Yes' : 'Not detected' },
  ];
  metrics.forEach(item => {
    const card = document.createElement('div');
    card.className = 'infoCard';
    const k = document.createElement('div');
    k.className = 'infoLabel';
    k.textContent = item.label;
    const v = document.createElement('div');
    v.className = 'infoValue';
    v.textContent = item.value;
    card.appendChild(k);
    card.appendChild(v);
    metricsGrid.appendChild(card);
  });
  container.appendChild(metricsGrid);

  const contextGrid = document.createElement('div');
  contextGrid.className = 'overviewGrid';
  const contextItems = [
    { label: 'Primary languages', value: (data.languages || []).join(', ') || '—' },
    { label: 'Frameworks', value: (data.frameworks || []).join(', ') || '—' },
    { label: 'Top modules', value: (data.top_modules || []).join(', ') || '—' },
    { label: 'Top directories', value: (data.top_directories || []).join(', ') || '—' },
  ];
  if (gitInfo.current_commit) {
    contextItems.push({ label: 'Commit', value: String(gitInfo.current_commit).slice(0, 12) });
  }
  if (gitInfo.commit_author) {
    contextItems.push({ label: 'Author', value: gitInfo.commit_author });
  }
  if (gitInfo.commit_message) {
    const msg = String(gitInfo.commit_message);
    contextItems.push({
      label: 'Commit message',
      value: msg.length > 80 ? msg.slice(0, 77) + '…' : msg
    });
  }
  if (gitInfo.remote_url) {
    contextItems.push({ label: 'Remote', value: gitInfo.remote_url });
  }
  if (gitInfo.commit_date) {
    contextItems.push({ label: 'Commit date', value: gitInfo.commit_date });
  }
  const hotspotSummary = (data.hotspots || []).map(h => h.path || '').filter(Boolean).slice(0,3).join(', ');
  contextItems.push({ label: 'Hotspots', value: hotspotSummary || '—' });
  contextItems.forEach(item => {
    const card = document.createElement('div');
    card.className = 'infoCard';
    const k = document.createElement('div');
    k.className = 'infoLabel';
    k.textContent = item.label;
    const v = document.createElement('div');
    v.className = 'infoValue';
    v.textContent = item.value;
    card.appendChild(k);
    card.appendChild(v);
    contextGrid.appendChild(card);
  });
  container.appendChild(contextGrid);

  const scopeTitle = document.createElement('div');
  scopeTitle.className = 'sectionTitle';
  scopeTitle.textContent = 'Scoped directories';
  container.appendChild(scopeTitle);
  const scopeMeta = document.createElement('div');
  scopeMeta.className = 'small';
  const scopeCount = typeof data.scope_file_count === 'number' ? data.scope_file_count : (data.files_sample || []).length;
  scopeMeta.textContent = `${scopeCount || 0} files affected`;
  container.appendChild(scopeMeta);
  const scopeList = data.scoped_directories && data.scoped_directories.length ? data.scoped_directories : (data.files_sample || []);
  scopeList.slice(0, 12).forEach(item => {
    const row = document.createElement('div');
    row.className = 'fileRow';
    row.textContent = item;
    container.appendChild(row);
  });

  const hotspots = data.hotspots || [];
  if (hotspots.length) {
    const sec = document.createElement('div');
    sec.className = 'sectionTitle';
    sec.textContent = 'Hotspots';
    container.appendChild(sec);
    hotspots.slice(0, 6).forEach(h => {
      const row = document.createElement('div');
      row.className = 'fileRow';
      const reason = h.reason ? ` · ${h.reason}` : '';
      row.textContent = `${h.path || '—'}${reason}`;
      container.appendChild(row);
    });
  }

  const tests = data.test_paths_sample || [];
  if (tests.length) {
    const sec = document.createElement('div');
    sec.className = 'sectionTitle';
    sec.textContent = 'Test files (sample)';
    container.appendChild(sec);
    tests.slice(0, 8).forEach(t => {
      const row = document.createElement('div');
      row.className = 'fileRow';
      row.textContent = t;
      container.appendChild(row);
    });
  }

  if (data.clarifications) {
    const clar = document.createElement('div');
    clar.className = 'sectionTitle';
    clar.textContent = 'Clarifications';
    container.appendChild(clar);
    const summary = document.createElement('div');
    summary.className = 'small';
    summary.textContent = `${data.clarifications.answered || 0} answered · ${data.clarifications.pending || 0} pending`;
    container.appendChild(summary);
  }

  if (data.external_patches && data.external_patches.length) {
    const extTitle = document.createElement('div');
    extTitle.className = 'sectionTitle';
    extTitle.textContent = 'External edits required';
    container.appendChild(extTitle);
    data.external_patches.slice(0, 4).forEach(p => {
      const box = document.createElement('div');
      box.className = 'ctaBox';
      const info = document.createElement('div');
      info.innerHTML = `<strong>${p.step_reference || 'Patch'}</strong><br/>Apply externally, then sync.`;
      box.appendChild(info);
      if (p.instructions) {
        const txt = document.createElement('div');
        txt.className = 'small';
        txt.style.marginTop = '6px';
        txt.textContent = p.instructions;
        box.appendChild(txt);
      }
      if (p.sync_command) {
        const btn = document.createElement('button');
        btn.className = 'copyBtn';
        btn.textContent = 'Copy sync command';
        btn.onclick = () => copyText(p.sync_command, btn);
        box.appendChild(btn);
      }
      container.appendChild(box);
    });
  }
}

function parseRangeToMinutes(range) {
  if (range === '5m') return 5;
  if (range === '1h') return 60;
  if (range === 'today') return 24 * 60;
  if (range === 'all') return 0; // special: all time
  return 60;
}

function minutesAgo(iso, nowIso) {
  try {
    const t = new Date(iso).getTime();
    const n = new Date(nowIso).getTime();
    return Math.max(0, Math.floor((n - t) / 60000));
  } catch (_) { return 0; }
}

function fmtAgo(iso, nowIso) {
  const m = minutesAgo(iso, nowIso);
  if (m < 1) return 'just now';
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h >= 24) {
    const d = Math.floor(h / 24);
    return `${d}d ago`;
  }
  return `${h}h ago`;
}

function statusDot(task) {
  // Map bucket/status into dot color similar to mock
  if (task.bucket === 'failed') return 'bad';
  if (task.status === 'COMPLETED') return 'good';
  if (task.status === 'IMPLEMENTING' || task.status === 'VERIFYING') return 'info';
  if (task.status === 'CLARIFYING' || task.status === 'SPEC_PENDING') return 'warn';
  return 'blue';
}

function chipForTask(task) {
  const chips = [];
  // Show duration if available
  if (task.duration_seconds && task.duration_seconds > 0) {
    chips.push({ cls: 'chip', text: `${task.duration_seconds}s` });
  }
  // Show patch counts if any
  if (task.patch_counts && task.patch_counts.total > 0) {
    chips.push({ cls: 'purple', text: `${task.patch_counts.total} patches` });
  }
  // Show status as chip
  let cls = 'chip';
  if (task.status === 'COMPLETED') cls = 'chip green';
  else if (task.status === 'IMPLEMENTING' || task.status === 'VERIFYING') cls = 'chip blue';
  else if (task.status === 'CLARIFYING' || task.status === 'SPEC_PENDING') cls = 'chip yellow';
  else if (task.bucket === 'failed') cls = 'chip red';
  chips.push({ cls, text: task.status });
  return chips;
}

function renderTasks() {
  const list = $('taskList');
  list.innerHTML = '';
  const nowIso = state.lastNow || new Date().toISOString();
  const maxMin = parseRangeToMinutes(state.range);

  const tasks = (state.tasks || [])
    .filter(t => {
      if (state.filter === 'all') return true;
      return (t.bucket || 'running') === state.filter;
    })
    .filter(t => {
      // range filter: based on updated_at
      if (!maxMin || maxMin <= 0) return true;
      const m = minutesAgo(t.updated_at, nowIso);
      return m <= maxMin;
    })
    .sort((a,b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime());

  if (!tasks.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'No tasks in this range. Try “Today” or widen the range.';
    list.appendChild(empty);
    return;
  }

  tasks.forEach(t => {
    const card = document.createElement('div');
    card.className = 'taskCard' + (t.id === state.selectedTaskId ? ' selected' : '');
    card.onclick = () => {
      state.selectedTaskId = t.id;
      state.selectedStepKey = null;
      loadTask(t.id);
      renderTasks();
    };

    const row = document.createElement('div');
    row.className = 'taskRow';

    const left = document.createElement('div');
    left.style.display = 'flex';
    left.style.alignItems = 'center';
    left.style.gap = '10px';
    const dot = document.createElement('span');
    dot.className = 'dot2 ' + statusDot(t);
    const title = document.createElement('div');
    title.className = 'taskTitle';
    const clientLabel = clientLabelFrom(t.client);
    const shortTitle = resolveTaskTitle(t);
    title.textContent = `${clientLabel} · ${shortTitle}`;
    left.appendChild(dot);
    left.appendChild(title);

    const right = document.createElement('div');
    right.className = 'timeLabel';
    right.textContent = fmtAgo(t.updated_at, nowIso);

    row.appendChild(left);
    row.appendChild(right);

    const meta = document.createElement('div');
    meta.className = 'taskMeta';
    chipForTask(t).forEach(c => {
      const chip = document.createElement('span');
      chip.className = 'chip ' + (c.cls === 'chip' ? '' : c.cls);
      chip.textContent = c.text;
      meta.appendChild(chip);
    });

    card.appendChild(row);
    card.appendChild(meta);
    list.appendChild(card);
  });
}

function renderWorkflow(task, workflow) {
  if (task) {
    const clientLabel = clientLabelFrom(task.client);
    $('workflowSubtitle').textContent = `${clientLabel} · ${task.title}`;
  } else {
    $('workflowSubtitle').textContent = 'Select a task';
  }
  const stepsEl = $('workflowSteps');
  stepsEl.innerHTML = '';

  if (!task) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Select a task to view workflow';
    stepsEl.appendChild(empty);
    return;
  }

  (workflow || []).forEach(step => {
    const el = document.createElement('div');
    el.className = 'step' + (state.selectedStepKey === step.key ? ' selected' : '');
    el.onclick = () => {
      state.selectedStepKey = step.key;
      renderWorkflow(task, workflow);
      // Prefer cached step details if we have them; otherwise show a loading state.
      const cached = (state.selectedTaskData && state.selectedTaskData.step_details) ? state.selectedTaskData.step_details : {};
      const detail = (cached && cached[step.key]) ? cached[step.key] : null;
      renderDetails(task, step, detail);
    };

    const rail = document.createElement('div');
    rail.className = 'rail';
    const fill = document.createElement('div');
    fill.className = 'fill' + (step.status === 'active' ? ' active' : '');
    fill.style.height = step.status === 'done' ? '100%' : (step.status === 'active' ? '50%' : '0%');
    rail.appendChild(fill);

    const body = document.createElement('div');
    body.className = 'stepBody';
    const labelRow = document.createElement('div');
    labelRow.className = 'stepLabelRow';
    const iconSvg = STEP_ICONS[step.key];
    if (iconSvg) {
      const iconWrap = document.createElement('span');
      iconWrap.className = 'stepIcon';
      iconWrap.innerHTML = iconSvg;
      labelRow.appendChild(iconWrap);
    }
    const label = document.createElement('div');
    label.className = 'stepLabel';
    label.textContent = step.label;
    labelRow.appendChild(label);

    const chips = document.createElement('div');
    chips.className = 'stepChips';
    const dur = document.createElement('span');
    dur.className = 'chip';
    if (typeof step.duration_seconds === 'number') {
      dur.textContent = `${Math.round(step.duration_seconds * 10) / 10}s`;
    } else {
      dur.textContent = '—';
    }
    chips.appendChild(dur);

    (step.chips || []).forEach(c => {
      const text = (c.text || '').trim();
      if (!text || text.toLowerCase() === 'ended') return; // hide noisy "ended" chip
      const ch = document.createElement('span');
      let cls = 'chip';
      if (c.kind === 'success') cls += ' green';
      if (c.kind === 'warning') cls += ' yellow';
      if (c.kind === 'danger') cls += ' red';
      if (c.kind === 'neutral') cls += ' purple';
      ch.className = cls;
      ch.textContent = text.length > 8 ? text.slice(0, 8) + '…' : text;
      chips.appendChild(ch);
    });

    // Step completion dot
    const stepDot = document.createElement('span');
    let dotCls = 'stepDot';
    if (step.test_failed) dotCls += ' bad';
    else if (step.status === 'done') dotCls += ' good';
    else if (step.status === 'active') dotCls += ' active';
    stepDot.className = dotCls;
    labelRow.appendChild(stepDot);

    body.appendChild(labelRow);
    body.appendChild(chips);

    el.appendChild(rail);
    el.appendChild(body);
    stepsEl.appendChild(el);
  });
}

function renderDetails(task, step, detail) {
  const el = $('detailsBody');
  el.innerHTML = '';
  if (!task) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Select a task to view details';
    el.appendChild(empty);
    return;
  }

  const cached = (state.selectedTaskData && state.selectedTaskData.step_details) || {};
  let currentStep = step || { key: 'OVERVIEW', label: 'Task overview', status: 'info' };
  let d = detail || cached[currentStep.key] || {};

  if (currentStep.key === 'OVERVIEW') {
    renderOverview(el, task, d);
    return;
  }

  if (!step) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Select a step to view details';
    el.appendChild(empty);
    return;
  }

  // Header
  const head = document.createElement('div');
  head.style.marginBottom = '14px';
  const h1Row = document.createElement('div');
  h1Row.className = 'detailHeaderRow';
  const hIcon = STEP_ICONS[currentStep.key];
  if (hIcon) {
    const hIconWrap = document.createElement('span');
    hIconWrap.className = 'detailHeaderIcon';
    hIconWrap.innerHTML = hIcon;
    h1Row.appendChild(hIconWrap);
  }
  const h1Label = document.createElement('span');
  h1Label.style.fontWeight = '700';
  h1Label.style.fontSize = '14px';
  h1Label.textContent = currentStep.label;
  h1Row.appendChild(h1Label);
  const hDot = document.createElement('span');
  let hDotCls = 'dot2';
  if (currentStep.test_failed) hDotCls += ' bad';
  else if (currentStep.status === 'done') hDotCls += ' good';
  else if (currentStep.status === 'active') hDotCls += ' blue';
  hDot.className = hDotCls;
  h1Row.appendChild(hDot);
  const h2 = document.createElement('div');
  h2.className = 'small';
  const dur = typeof currentStep.duration_seconds === 'number' ? `${Math.round(currentStep.duration_seconds * 10) / 10}s` : '—';
  h2.textContent = dur;
  head.appendChild(h1Row);
  head.appendChild(h2);
  el.appendChild(head);

  d = d || {};

  // Step-specific rendering (mockup-style)
  if (currentStep.key === 'TASK_SPECIFICATION') {
    const hero = document.createElement('div');
    hero.className = 'taskHero';

    const top = document.createElement('div');
    top.className = 'taskHeroTop';

    const left = document.createElement('div');
    const title = document.createElement('div');
    title.className = 'taskHeroTitle';
    title.textContent = d.title || task.title || 'Task';
    const desc = document.createElement('div');
    desc.className = 'taskHeroDesc';
    desc.textContent = d.description || task.description || '—';
    left.appendChild(title);
    left.appendChild(desc);

    top.appendChild(left);

    if (d.priority) {
      const pr = document.createElement('span');
      pr.className = 'prio';
      pr.textContent = String(d.priority).toUpperCase();
      top.appendChild(pr);
    }

    hero.appendChild(top);
    el.appendChild(hero);

    const criteria = d.acceptance_criteria || [];
    if (criteria.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = 'Acceptance criteria';
      el.appendChild(sec);

      criteria.slice(0, 20).forEach((c, idx) => {
        const row = document.createElement('div');
        row.className = 'acItem';
        const n = document.createElement('span');
        n.className = 'acN';
        n.textContent = String(idx + 1);
        const txt = document.createElement('span');
        txt.className = 'acText';
        txt.textContent = String(c);
        row.appendChild(n);
        row.appendChild(txt);
        el.appendChild(row);
      });
    }
    return;
  }

  if (currentStep.key === 'CLARIFYING') {
    const items = d.items || [];
    const summary = document.createElement('div');
    summary.className = 'small';
    summary.textContent = `${d.answered || 0} answered · ${d.pending || 0} pending`;
    el.appendChild(summary);

    items.forEach((it, idx) => {
      const q = document.createElement('div');
      q.className = 'qcard';
      const qh = document.createElement('div');
      qh.className = 'qhead';
      const badge = document.createElement('span');
      badge.className = 'qbadge';
      badge.textContent = `Q${idx + 1}`;
      const qt = document.createElement('span');
      qt.className = 'qtext';
      // Hide clarification IDs in the UI (keep it human-friendly).
      qt.textContent = `${it.question || ''}`.trim();
      qh.appendChild(badge);
      qh.appendChild(qt);
      const ans = document.createElement('div');
      ans.className = 'qans ' + (it.status === 'ANSWERED' ? 'ok' : 'muted');
      const icon = document.createElement('span');
      icon.className = 'qicon';
      icon.textContent = it.status === 'ANSWERED' ? '✓' : '○';
      const at = document.createElement('span');
      at.textContent = it.status === 'ANSWERED' ? (it.answer || '—') : '(pending)';
      ans.appendChild(icon);
      ans.appendChild(at);
      q.appendChild(qh);
      q.appendChild(ans);
      el.appendChild(q);
    });

    const footer = document.createElement('div');
    footer.className = 'small';
    footer.style.marginTop = '12px';
    footer.textContent = `${d.answered || 0} answered · ${d.pending || 0} pending`;
    el.appendChild(footer);
    return;
  }

  if (currentStep.key === 'PLANNING') {
    // Approval banner
    if (d.plan_approved) {
      const banner = document.createElement('div');
      banner.className = 'banner';
      banner.textContent = 'User approved this plan';
      el.appendChild(banner);
    } else {
      const banner = document.createElement('div');
      banner.className = 'banner warn';
      banner.textContent = 'Plan not approved yet';
      el.appendChild(banner);
    }

    // Plan markdown link (if available)
    if (d.plan_markdown_path) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = 'Plan document';
      el.appendChild(sec);

      const card = document.createElement('div');
      card.className = 'card';

      const small = document.createElement('div');
      small.className = 'small';
      const fileName = String(d.plan_markdown_path).split('/').slice(-1)[0];
      small.textContent = fileName;

      const actions = document.createElement('div');
      actions.style.marginTop = '6px';
      const link = document.createElement('a');
      link.href = `/api/plan-markdown/${encodeURIComponent(task.id)}`;
      link.target = '_blank';
      link.rel = 'noreferrer';
      link.textContent = 'Open / download markdown';
      actions.appendChild(link);

      const spacer = document.createElement('span');
      spacer.textContent = '  ·  ';
      spacer.style.color = 'var(--muted)';
      actions.appendChild(spacer);

      const full = document.createElement('a');
      full.href = `/plan/${encodeURIComponent(task.id)}`;
      full.textContent = 'View full page';
      actions.appendChild(full);

      card.appendChild(small);
      card.appendChild(actions);
      el.appendChild(card);
    }

    // Impacted scope: prefer plan_target_files (with MODIFY/CREATE badges),
    // fall back to directory-level targets, then file sample.
    const bc = d.bounded_context || {};
    const planTargetFiles = (bc.plan_target_files || []);
    const bcAllowed = bc.allowed_files || [];
    const targets = (bc.targets || []);
    const files = (bc.files_sample || []);
    const total = (typeof bc.file_count === 'number') ? bc.file_count : files.length;
    const list = planTargetFiles.length ? planTargetFiles : (targets.length ? targets : files);
    if (list.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = planTargetFiles.length ? 'Bounded context tree' : (targets.length ? 'Bounded context tree' : 'Impacted files (sample)');
      el.appendChild(sec);

      const meta = document.createElement('div');
      meta.className = 'small';
      meta.textContent = planTargetFiles.length ? `${planTargetFiles.length} files affected` : `${total} files affected`;
      el.appendChild(meta);

      if (planTargetFiles.length) {
        // Build MODIFY/CREATE badges using same logic as CODEGEN
        const allowedSet = new Set(bcAllowed);
        const badges = {};
        if (allowedSet.size) {
          planTargetFiles.forEach(f => { badges[f] = allowedSet.has(f) ? 'MODIFY' : 'CREATE'; });
        }
        renderPathTree(el, planTargetFiles, 12, 8, Object.keys(badges).length ? badges : null);
      } else if (targets.length) {
        renderPathTree(el, list, 12, 8);
      } else {
        list.slice(0, 25).forEach(p => {
          const row = document.createElement('div');
          row.className = 'fileRow';
          row.textContent = p;
          el.appendChild(row);
        });
      }
    }

    // Execution plan
    const steps = d.plan_steps || [];
    if (steps.length) {
      const sec2 = document.createElement('div');
      sec2.className = 'sectionTitle';
      sec2.textContent = 'Execution plan';
      el.appendChild(sec2);
      steps.forEach((s, idx) => {
        const li = document.createElement('div');
        li.className = 'planItem';
        const n = document.createElement('span');
        n.className = 'planN';
        n.textContent = String(idx + 1);
        const txt = document.createElement('span');

        // Be robust to different step payload shapes so we never show "[object Object]".
        let label = "";
        if (s && typeof s === "object") {
          const candidate =
            (typeof s.description === "string" && s.description) ||
            (typeof s.title === "string" && s.title) ||
            (typeof s.summary === "string" && s.summary) ||
            (typeof s.text === "string" && s.text) ||
            (typeof s.name === "string" && s.name);
          if (candidate) {
            label = candidate;
          } else {
            try {
              label = JSON.stringify(s);
            } catch {
              label = String(s);
            }
          }
        } else {
          label = String(s ?? "");
        }

        txt.textContent = label.trim();
        li.appendChild(n);
        li.appendChild(txt);
        el.appendChild(li);
      });
    }

    // Boundary specs
    const specs = d.boundary_specs || [];
    if (specs.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = 'Boundary specs';
      el.appendChild(sec);

      specs.slice(0, 20).forEach(s => {
        const card = document.createElement('div');
        card.className = 'card';

        const title = document.createElement('div');
        title.className = 'sectionTitle';
        const status = s.status ? ` (${s.status})` : '';
        title.textContent = `${s.name}${status}`;
        card.appendChild(title);

        if (s.plan_step) {
          const step = document.createElement('div');
          step.className = 'small';
          step.textContent = `Plan step: ${s.plan_step}`;
          card.appendChild(step);
        }

        if (s.summary || s.description) {
          const desc = document.createElement('div');
          desc.className = 'small';
          desc.style.marginTop = '6px';
          desc.textContent = s.description || s.summary;
          card.appendChild(desc);
        }

        const meta = document.createElement('div');
        meta.className = 'small';
        meta.style.marginTop = '8px';
        const actors = (s.actors || []).join(', ');
        const interfaces = (s.interfaces || []).join(', ');
        const invariants = (s.invariants || []).join('; ');
        meta.textContent = [
          actors ? `Actors: ${actors}` : '',
          interfaces ? `Interfaces: ${interfaces}` : '',
          invariants ? `Invariants: ${invariants}` : '',
        ].filter(Boolean).join(' • ');
        if (meta.textContent) card.appendChild(meta);

        if (s.diagram) {
          const pre = document.createElement('pre');
          pre.className = 'diff';
          pre.textContent = s.diagram;
          pre.style.marginTop = '8px';
          card.appendChild(pre);
        }

        el.appendChild(card);
      });
    }

    // Scenarios
    const scenarios = d.scenarios || [];
    if (scenarios.length) {
      const sec3 = document.createElement('div');
      sec3.className = 'sectionTitle';
      sec3.textContent = 'Scenarios (Given / When / Then)';
      el.appendChild(sec3);
      scenarios.slice(0, 8).forEach(sc => {
        const box = document.createElement('div');
        box.className = 'qcard';
        const h = document.createElement('div');
        h.className = 'qhead';
        h.textContent = `Step ${sc.index}: ${sc.title}`;
        const body = document.createElement('div');
        body.className = 'small';
        body.style.marginTop = '8px';
        body.style.whiteSpace = 'pre-line';
        body.textContent = (sc.lines || []).join('\n');
        box.appendChild(h);
        box.appendChild(body);
        el.appendChild(box);
      });
    }

    // Risks
    const risks = d.risks || [];
    if (risks.length) {
      const sec4 = document.createElement('div');
      sec4.className = 'sectionTitle';
      sec4.textContent = 'Risks';
      el.appendChild(sec4);
      risks.slice(0, 20).forEach(r => {
        const card = document.createElement('div');
        card.className = 'qcard';

        const info = normalizeRisk(r);
        const title = info.title || 'Risk';
        const desc = info.description || '';
        const mitigation = info.mitigation || '';

        const h = document.createElement('div');
        h.className = 'qhead';
        h.textContent = title || 'Risk';

        const body = document.createElement('div');
        body.className = 'small';
        body.style.marginTop = '6px';
        body.textContent = desc || '—';

        card.appendChild(h);
        card.appendChild(body);

        if (mitigation) {
          const mit = document.createElement('div');
          mit.className = 'small muted';
          mit.style.marginTop = '6px';
          mit.textContent = `Mitigation: ${mitigation}`;
          card.appendChild(mit);
        }

        el.appendChild(card);
      });
    }

    // Refactor suggestions
    const ref = d.refactors || [];
    if (ref.length) {
      const sec5 = document.createElement('div');
      sec5.className = 'sectionTitle';
      sec5.textContent = 'Refactor suggestions';
      el.appendChild(sec5);
      ref.slice(0, 20).forEach(r => {
        const row = document.createElement('div');
        row.className = 'fileRow';
        row.textContent = r;
        el.appendChild(row);
      });
    }
    return;
  }

  if (currentStep.key === 'CODEGEN') {
    const patches = d.patches || [];
    const pending = (patches || []).filter(p => p.status === 'PENDING');
    const banner = document.createElement('div');
    banner.className = pending.length ? 'banner warn' : 'banner';
    banner.textContent = pending.length ? '\u2298 Not applied' : '\u2714 Applied / no pending patches';
    el.appendChild(banner);

    // Files touched (from diffs) with MODIFY/CREATE badges
    const files = d.files_touched || [];
    if (files.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = `Files touched (${files.length})`;
      el.appendChild(sec);
      const allowedSet = new Set(d.allowed_files || []);
      const badges = {};
      if (allowedSet.size) {
        files.forEach(f => { badges[f] = allowedSet.has(f) ? 'MODIFY' : 'CREATE'; });
      }
      renderPathTree(el, files, 12, 30, Object.keys(badges).length ? badges : null);
    }

    // Patch list + diff preview
    if (patches.length) {
      const sec2 = document.createElement('div');
      sec2.className = 'sectionTitle';
      sec2.textContent = 'Unified diff';
      el.appendChild(sec2);

      const chosen = patches.find(p => p.id === state.selectedPatchId) || pending[0] || patches[0];
      state.selectedPatchId = chosen ? chosen.id : null;

      const list = document.createElement('div');
      list.className = 'patchList';
      patches.slice(0, 20).forEach(p => {
        const row = document.createElement('div');
        row.className = 'patchRow' + (p.id === state.selectedPatchId ? ' selected' : '');
        const via = (p.applied_via && String(p.applied_via).trim()) ? ` · ${p.applied_via}` : '';
        row.textContent = `${p.id.slice(0,8)} · ${p.status}${via} · ${p.step_reference}`;
      row.onclick = () => {
          state.selectedPatchId = p.id;
          renderDetails(task, step, d);
        };
        list.appendChild(row);
      });
      el.appendChild(list);

      if (chosen && chosen.requires_external) {
        const cta = document.createElement('div');
        cta.className = 'ctaBox';
        const msg = document.createElement('div');
        msg.innerHTML = `<strong>External edit required</strong><br/>Apply this step in your editor, then sync the diff.`;
        cta.appendChild(msg);
        if (chosen.external_instructions) {
          const txt = document.createElement('div');
          txt.className = 'small';
          txt.style.marginTop = '6px';
          txt.textContent = chosen.external_instructions;
          cta.appendChild(txt);
        }
        if (chosen.sync_command) {
          const btn = document.createElement('button');
          btn.className = 'copyBtn';
          btn.textContent = 'Copy sync command';
          btn.onclick = () => copyText(chosen.sync_command, btn);
          cta.appendChild(btn);
        }
        el.appendChild(cta);
      }

      const pre = document.createElement('pre');
      pre.className = 'diff';
      if (chosen && chosen.diff) {
        renderHighlightedDiff(pre, chosen.diff);
      } else {
        pre.textContent = '—';
      }
      el.appendChild(pre);
    }

    // Action bar with Re-run tests / Re-run from here
    {
      const bar = document.createElement('div');
      bar.className = 'actionBar';
      const testFiles = d.test_files || [];
      if (testFiles.length) {
        const btn = document.createElement('button');
        btn.className = 'actionBtn';
        btn.innerHTML = '\u25b6 Re-run tests';
        btn.onclick = async () => {
          btn.disabled = true;
          btn.textContent = 'Running tests\u2026';
          try {
            const res = await fetch(`/api/task/${encodeURIComponent(task.id)}/run-tests`, {method:'POST'});
            const result = await res.json();
            if (result.exit_code === 0) {
              btn.className = 'actionBtn success';
              btn.textContent = '\u2714 Tests passed';
            } else {
              btn.className = 'actionBtn fail';
              btn.textContent = '\u2718 Tests failed';
            }
          } catch(e) {
            btn.className = 'actionBtn fail';
            btn.textContent = 'Error running tests';
          }
          setTimeout(() => { btn.className='actionBtn'; btn.innerHTML='\u25b6 Re-run tests'; btn.disabled=false; }, 4000);
          setTimeout(tick, 500);
        };
        bar.appendChild(btn);
      }
      el.appendChild(bar);
    }
    return;
  }

  if (currentStep.key === 'RUN_TESTS') {
    // Run / Re-run tests button
    {
      const bar = document.createElement('div');
      bar.style.marginBottom = '12px';
      const hasResults = !!(d.tool_name || d.output || typeof d.exit_code === 'number');
      const testFiles = d.test_files || [];
      const btn = document.createElement('button');
      btn.className = 'actionBtn';
      btn.innerHTML = hasResults ? '\u25b6 Re-run tests' : '\u25b6 Run tests';
      if (!testFiles.length && !hasResults) {
        btn.disabled = true;
        btn.title = 'No test files detected in patches';
      }
      btn.onclick = async () => {
        btn.disabled = true;
        btn.textContent = 'Running tests\u2026';
        try {
          const res = await fetch(`/api/task/${encodeURIComponent(task.id)}/run-tests`, {method:'POST'});
          const result = await res.json();
          if (result.exit_code === 0) {
            btn.className = 'actionBtn success';
            btn.textContent = '\u2714 Tests passed';
          } else {
            btn.className = 'actionBtn fail';
            btn.textContent = '\u2718 Tests failed';
          }
        } catch(e) {
          btn.className = 'actionBtn fail';
          btn.textContent = 'Error running tests';
        }
        setTimeout(() => { btn.className='actionBtn'; btn.innerHTML=hasResults?'\u25b6 Re-run tests':'\u25b6 Run tests'; btn.disabled=false; }, 4000);
        setTimeout(tick, 500);
      };
      bar.appendChild(btn);
      el.appendChild(bar);
    }

    if (d.tool_name) {
      const toolSec = document.createElement('div');
      toolSec.className = 'sectionTitle';
      toolSec.textContent = 'Tool';
      el.appendChild(toolSec);
      const toolCard = document.createElement('div');
      toolCard.className = 'card';
      const toolLabel = d.tool_version ? `${d.tool_name} ${d.tool_version}` : d.tool_name;
      toolCard.textContent = toolLabel;
      el.appendChild(toolCard);
    }
    if (d.arguments) {
      const argSec = document.createElement('div');
      argSec.className = 'sectionTitle';
      argSec.textContent = 'Arguments';
      el.appendChild(argSec);
      const argPre = document.createElement('pre');
      argPre.className = 'diff';
      argPre.textContent = typeof d.arguments === 'string' ? d.arguments : JSON.stringify(d.arguments, null, 2);
      el.appendChild(argPre);
    }
    if (d.output) {
      const outSec = document.createElement('div');
      outSec.className = 'sectionTitle';
      outSec.textContent = 'Output';
      el.appendChild(outSec);
      const outPre = document.createElement('pre');
      outPre.className = 'diff';
      outPre.textContent = d.output;
      outPre.style.maxHeight = '400px';
      el.appendChild(outPre);
    }
    if (typeof d.exit_code === 'number') {
      const exitSec = document.createElement('div');
      exitSec.className = 'sectionTitle';
      exitSec.textContent = 'Exit code';
      el.appendChild(exitSec);
      const exitEl = document.createElement('div');
      exitEl.className = 'card';
      exitEl.style.display = 'flex';
      exitEl.style.alignItems = 'center';
      exitEl.style.gap = '8px';
      const exitDot = document.createElement('span');
      exitDot.className = 'dot2 ' + (d.exit_code === 0 ? 'good' : 'bad');
      exitEl.appendChild(exitDot);
      const exitText = document.createElement('span');
      exitText.style.color = d.exit_code === 0 ? 'var(--good)' : 'var(--bad)';
      exitText.style.fontWeight = '700';
      exitText.textContent = `Exit code: ${d.exit_code}`;
      exitEl.appendChild(exitText);
      el.appendChild(exitEl);
    }
    if (!d.tool_name && !d.arguments && !d.output && typeof d.exit_code !== 'number') {
      const hint = document.createElement('div');
      hint.className = 'small';
      hint.textContent = 'No test execution data available yet.';
      el.appendChild(hint);
    }
    return;
  }

  const fallback = document.createElement('div');
  fallback.className = 'small';
  fallback.textContent = 'No renderer for this step yet.';
  el.appendChild(fallback);
}

async function copyText(value, btn) {
  if (!value) return;
  const previous = btn ? btn.textContent : null;
  try {
    await navigator.clipboard.writeText(value);
    if (btn) {
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = previous || 'Copy', 1200);
    }
  } catch (e) {
    if (btn) {
      btn.textContent = 'Copy failed';
      setTimeout(() => btn.textContent = previous || 'Copy', 1200);
    }
  }
}

async function fetchState() {
  const maxMin = parseRangeToMinutes(state.range);
  const url = `/api/state?minutes=${encodeURIComponent(maxMin)}&filter=${encodeURIComponent(state.filter)}`;
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error('failed to fetch state');
  return await res.json();
}

async function loadTask(taskId) {
  const res = await fetch(`/api/task/${encodeURIComponent(taskId)}`, { cache: 'no-store' });
  if (!res.ok) return;
  const data = await res.json();
  state.selectedTaskData = data;
  state.selectedPatchId = null;
  renderWorkflow(data.task, data.workflow);
  state.selectedStepKey = null;
  const overview = (data.step_details && data.step_details.OVERVIEW) ? data.step_details.OVERVIEW : null;
  renderDetails(data.task, { key: 'OVERVIEW', label: 'Task overview', status: 'info' }, overview);
}

function wireUI() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.filter = btn.getAttribute('data-filter');
      renderTasks();
    });
  });
  document.querySelectorAll('.pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.range = btn.getAttribute('data-range');
      renderTasks();
    });
  });
}

let _tickRunning = false;
let _lastStateFP = '';
let _lastDetailFP = '';

function _fp(data) {
  const t = data.tasks || [];
  return t.length + ':' + (t.length ? t[0].updated_at || '' : '') + ':' + (data.now || '');
}
function _dfp(d) {
  if (!d || !d.task) return '';
  const t = d.task;
  return t.id + ':' + (t.updated_at || '') + ':' + (t.last_event_at || '');
}

async function tick() {
  if (_tickRunning) return;
  _tickRunning = true;
  try {
    const data = await fetchState();
    state.lastNow = data.now;
    state.tasks = data.tasks || [];
    const fp = _fp(data);
    if (fp !== _lastStateFP) { _lastStateFP = fp; renderTasks(); }

    if (state.selectedTaskId) {
      const res = await fetch(`/api/task/${encodeURIComponent(state.selectedTaskId)}`, { cache: 'no-store' });
      if (res.ok) {
        const tdata = await res.json();
        const dfp = _dfp(tdata);
        const changed = dfp !== _lastDetailFP;
        _lastDetailFP = dfp;
        state.selectedTaskData = tdata;
        if (changed) {
          renderWorkflow(tdata.task, tdata.workflow);
          if (state.selectedStepKey) {
            const step = (tdata.workflow || []).find(s => s.key === state.selectedStepKey);
            if (step) {
              const detail = (tdata.step_details || {})[step.key] || null;
              renderDetails(tdata.task, step, detail);
            }
          } else {
            const overview = (tdata.step_details && tdata.step_details.OVERVIEW) ? tdata.step_details.OVERVIEW : null;
            renderDetails(tdata.task, { key: 'OVERVIEW', label: 'Task overview', status: 'info' }, overview);
          }
        }
      }
    }
  } catch (e) { console.warn(e); }
  finally { _tickRunning = false; }
}

wireUI();
const btnRefresh = $('btnRefresh');
if (btnRefresh) btnRefresh.addEventListener('click', () => tick());
tick();
setInterval(tick, 5000);
"""


class _DashboardApi:
    def __init__(self) -> None:
        settings = get_settings()
        self.store = JsonStore(settings.state_dir)
        self._cache: tuple[list[Task], list[LogEntry]] | None = None
        self._cache_time: float = 0.0
        self._cache_tasks_mtime: float = 0
        self._cache_logs_mtime: float = 0

    def _load(self) -> tuple[list[Task], list[LogEntry]]:
        now = time.monotonic()
        tasks_path = self.store.tasks_file
        logs_path = self.store.logs_file
        tasks_mtime = tasks_path.stat().st_mtime if tasks_path.exists() else 0
        logs_mtime = logs_path.stat().st_mtime if logs_path.exists() else 0

        if (self._cache is not None
                and (now - self._cache_time) < 5.0
                and tasks_mtime == self._cache_tasks_mtime
                and logs_mtime == self._cache_logs_mtime):
            return self._cache

        tasks = self.store.load_tasks()
        logs = self.store.load_logs()
        self._cache = (tasks, logs)
        self._cache_time = now
        self._cache_tasks_mtime = tasks_mtime
        self._cache_logs_mtime = logs_mtime
        return tasks, logs

    def state(self, *, minutes: int, filter_bucket: str) -> Dict[str, Any]:
        tasks, logs = self._load()
        now = _utcnow()

        # Filter by time window based on task.updated_at.
        # minutes <= 0 means "all time".
        cutoff = None if minutes <= 0 else (now.timestamp() - float(minutes * 60))
        filtered_tasks: list[Task] = []
        for t in tasks:
            if cutoff is not None and t.updated_at.timestamp() < cutoff:
                continue
            if filter_bucket and filter_bucket != "all":
                if _task_bucket(t.status) != filter_bucket:
                    continue
            filtered_tasks.append(t)

        latest_by_task: dict[str, LogEntry] = {}
        for entry in logs:
            ts = _safe_dt(getattr(entry, "timestamp", None))
            if not ts:
                continue
            prev = latest_by_task.get(entry.task_id)
            if not prev or ts > prev.timestamp:
                latest_by_task[entry.task_id] = entry

        # Keep payload light – strip description (only needed in detail endpoint).
        data_tasks = []
        for t in filtered_tasks:
            td = _serialize_task(t, latest_by_task.get(t.id))
            td.pop("description", None)
            data_tasks.append(td)

        # Provide a small recent activity list (global).
        recent = _latest_logs(logs, limit=20)
        return {
            "now": _iso(now),
            "tasks": data_tasks,
            "activity": [_serialize_log(e) for e in recent],
        }

    def task(self, task_id: str) -> Dict[str, Any]:
        tasks, logs = self._load()
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise KeyError("task not found")

        # Latest log
        latest: LogEntry | None = None
        for entry in logs:
            if entry.task_id != task_id:
                continue
            if not latest or entry.timestamp > latest.timestamp:
                latest = entry

        workflow = _infer_workflow(task, logs)
        step_details = _build_step_details(task, logs)
        return {
            "now": _iso(_utcnow()),
            "task": _serialize_task(task, latest),
            "workflow": workflow,
            "step_details": step_details,
            "activity": [_serialize_log(e) for e in _latest_logs([l for l in logs if l.task_id == task_id], limit=30)],
        }

    def plan_markdown(self, task_id: str) -> Dict[str, Any]:
        """
        Serve the exported plan markdown (if present) for a task.
        """
        tasks, _ = self._load()
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise KeyError("task not found")
        meta = task.metadata if isinstance(task.metadata, dict) else {}
        path_str = str(meta.get("plan_markdown_path") or "").strip()

        # Always provide markdown for the dashboard:
        # - Prefer the exported file if readable
        # - Otherwise, fall back to rendering from stored task metadata
        content = ""
        if path_str:
            try:
                p = Path(path_str)
                if p.exists() and p.is_file():
                    content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                content = ""

        if not content:
            content = _render_plan_markdown(task)

        if len(content) > 200_000:
            content = content[:200_000] + "\n\n… (truncated)\n"
        return {"path": path_str or None, "content": content}

    def run_tests(self, task_id: str) -> Dict[str, Any]:
        """Execute tests for the given task and persist results."""
        tasks, _ = self._load()
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            raise KeyError("task not found")

        meta = task.metadata if isinstance(task.metadata, dict) else {}
        repo_path = str(task.repo_path or "").strip()
        if not repo_path or not Path(repo_path).is_dir():
            return {"error": "Repository path not found", "exit_code": -1}

        # Collect files_touched from patches
        files_touched: list[str] = []
        patches_raw = meta.get("patch_queue_state") or []
        if isinstance(patches_raw, list):
            for p in patches_raw:
                if not isinstance(p, dict):
                    continue
                applied_diff = p.get("applied_diff")
                raw_diff = str(p.get("diff") or "")
                diff = str(applied_diff or raw_diff)
                for f in _extract_files_from_diff(diff):
                    if f not in files_touched:
                        files_touched.append(f)

        test_files = _filter_test_files(files_touched)
        if not test_files:
            return {"error": "No test files found in patches", "exit_code": -1}

        existing_tool = meta.get("test_tool")
        command, base_args = _detect_test_runner(test_files, existing_tool)

        # Build command based on runner
        if command == "dotnet":
            # dotnet test --filter "FullyQualifiedName~Class1|FullyQualifiedName~Class2"
            class_names = []
            for tf in test_files:
                stem = Path(tf).stem
                class_names.append(f"FullyQualifiedName~{stem}")
            filter_expr = "|".join(class_names)
            cmd = [command] + base_args + ["--filter", filter_expr]
        elif command == "pytest":
            cmd = [command] + base_args + test_files
        else:
            # jest / npx jest
            cmd = [command] + base_args + test_files

        tool_name = base_args[0] if base_args else command
        arguments = " ".join(cmd[len([command] + base_args):]) if cmd else ""

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=300,
            )
            output = result.stdout
            if result.stderr:
                output = output + "\n" + result.stderr if output else result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            output = "Test execution timed out after 300 seconds."
            exit_code = -1
        except FileNotFoundError:
            output = f"Test runner '{command}' not found. Make sure it is installed."
            exit_code = -1
        except Exception as exc:
            output = f"Error running tests: {exc}"
            exit_code = -1

        # Truncate very long output
        if len(output) > 100_000:
            output = output[:100_000] + "\n\n... (output truncated)"

        # Persist results
        if not isinstance(task.metadata, dict):
            task.metadata = {}
        task.metadata["test_tool"] = tool_name
        task.metadata["test_output"] = output
        task.metadata["test_exit_code"] = exit_code
        task.metadata["test_arguments"] = arguments
        self.store.upsert_task(task)
        self._cache = None

        return {
            "tool": tool_name,
            "arguments": arguments,
            "output": output,
            "exit_code": exit_code,
        }


def _extract_files_from_diff(diff: str) -> list[str]:
    """
    Best-effort parse of changed file paths from a unified diff.
    """
    if not diff:
        return []
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            parts = line.split()
            path = parts[1] if len(parts) > 1 else ""
            if path.startswith("a/") or path.startswith("b/"):
                path = path[2:]
            if not path or path == "/dev/null":
                continue
            if path not in files:
                files.append(path)
    return files


def _build_step_details(task: Task, logs: list[LogEntry]) -> Dict[str, Any]:
    """
    Produce mockup-like, per-step detail payloads.
    """
    meta = task.metadata if isinstance(task.metadata, dict) else {}
    clarifications = meta.get("clarifications") or []
    if not isinstance(clarifications, list):
        clarifications = []
    answered = 0
    pending = 0
    items: list[Dict[str, Any]] = []
    for it in clarifications:
        if not isinstance(it, dict):
            continue
        status = str(it.get("status") or "").strip().upper()
        if status == "ANSWERED":
            answered += 1
        elif status == "PENDING" or status == "":
            pending += 1
        # Do not expose internal clarification IDs in the web UI payload.
        items.append(
            {
                "question": str(it.get("question") or ""),
                "answer": str(it.get("answer") or ""),
                "status": status or "PENDING",
            }
        )

    # Optional priority (demo-friendly, can be set by client)
    priority = None
    pr = meta.get("priority")
    if isinstance(pr, str) and pr.strip():
        priority = pr.strip()

    plan_preview = meta.get("plan_preview") or {}
    if not isinstance(plan_preview, dict):
        plan_preview = {}
    if not plan_preview.get("steps"):
        plan_fallback = meta.get("plan") if isinstance(meta, dict) else {}
        if isinstance(plan_fallback, dict):
            plan_preview = plan_fallback
    plan_steps = (plan_preview.get("steps") or []) if isinstance(plan_preview, dict) else []
    risks = (plan_preview.get("risks") or []) if isinstance(plan_preview, dict) else []
    refactors = (plan_preview.get("refactors") or []) if isinstance(plan_preview, dict) else []
    if not isinstance(risks, list):
        risks = []
    if not isinstance(refactors, list):
        refactors = []

    # Extract union of target_files across all plan steps for bounded context tree
    plan_target_files: list[str] = []
    if isinstance(plan_steps, list):
        seen_tf: set[str] = set()
        for step in plan_steps:
            if not isinstance(step, dict):
                continue
            for tf in (step.get("target_files") or []):
                if isinstance(tf, str) and tf.strip() and tf not in seen_tf:
                    plan_target_files.append(tf)
                    seen_tf.add(tf)

    acceptance_criteria = _derive_acceptance_criteria(task, clarifications, plan_steps if isinstance(plan_steps, list) else [])
    scenarios = _derive_scenarios(task, plan_steps if isinstance(plan_steps, list) else [])

    specs = meta.get("boundary_specs") if isinstance(meta, dict) else []
    spec_summary: list[dict] = []
    if isinstance(specs, list):
        for spec in specs:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("boundary_name") or "Unknown").strip()
            status = str(spec.get("status") or "").strip()
            plan_step = str(spec.get("plan_step") or "").strip()
            human_description = str(spec.get("human_description") or "").strip()
            desc = human_description.splitlines()[0] if human_description else ""
            if desc and len(desc) > 140:
                desc = desc[:137].rstrip() + "..."
            machine = spec.get("machine_spec") or {}
            actors = machine.get("actors") or []
            interfaces = machine.get("interfaces") or []
            invariants = machine.get("invariants") or []
            diagram = str(spec.get("diagram_text") or "").strip()
            spec_summary.append(
                {
                    "name": name,
                    "status": status,
                    "plan_step": plan_step,
                    "summary": desc,
                    "description": human_description,
                    "actors": actors,
                    "interfaces": interfaces,
                    "invariants": invariants,
                    "diagram": diagram,
                }
            )

    bounded_ctx = meta.get("bounded_context") or {}
    if not isinstance(bounded_ctx, dict):
        bounded_ctx = {}
    scoped = bounded_ctx.get("manual") or bounded_ctx.get("plan_targets") or {}
    if not isinstance(scoped, dict):
        scoped = {}
    # Prefer showing "targets" (what we intentionally scoped to) rather than huge file samples.
    raw_targets = list((scoped.get("targets") or {}).keys()) if isinstance(scoped.get("targets"), dict) else []
    if not isinstance(raw_targets, list):
        raw_targets = []
    scope = (scoped.get("scope") or {}) if isinstance(scoped, dict) else {}
    allowed_files = scope.get("allowed_files") if isinstance(scope, dict) else []
    if not isinstance(allowed_files, list):
        allowed_files = []
    impact = (scoped.get("impact") or {}) if isinstance(scoped, dict) else {}
    files_sample = impact.get("files_sample") or []
    if not isinstance(files_sample, list):
        files_sample = []
    # Fallback: if no impact sample is present, show a small slice of allowed files.
    if not files_sample and allowed_files:
        files_sample = list(allowed_files[:20])

    # Keep the web UI tight: only show targets (or a very small file sample as fallback).
    # Also hide obvious build artifacts and test outputs.
    def _is_noise(rel: str) -> bool:
        p = (rel or "").replace("\\", "/").lower()
        return any(seg in p for seg in ("/bin/", "/obj/", "/node_modules/", "/dist/", "/build/"))

    targets_clean = [t for t in raw_targets if t and not _is_noise(t)]
    files_sample_clean = [f for f in files_sample if f and not _is_noise(f)]

    patches_raw = meta.get("patch_queue_state") or []
    external_patch_notes: list[Dict[str, str]] = []
    patches: list[Dict[str, Any]] = []
    files_touched: list[str] = []
    if isinstance(patches_raw, list):
        for p in patches_raw:
            if not isinstance(p, dict):
                continue
            applied_diff = p.get("applied_diff")
            raw_diff = str(p.get("diff") or "")
            diff = str(applied_diff or raw_diff)
            patch_id = str(p.get("id") or "")
            step_reference = str(p.get("step_reference") or "")
            requires_external = bool(not applied_diff and raw_diff.startswith(EXTERNAL_PATCH_SENTINEL))
            sync_command = ""
            external_message = ""
            if requires_external:
                external_message = raw_diff[len(EXTERNAL_PATCH_SENTINEL) :].strip()
                diff = external_message or diff or "External edit required."
                sync_command = f"./spec-agent sync-external {task.id} --patch-id {patch_id}"
                external_patch_notes.append(
                    {
                        "patch_id": patch_id,
                        "step_reference": step_reference,
                        "sync_command": sync_command,
                        "instructions": external_message,
                    }
                )
            for f in _extract_files_from_diff(diff):
                if f not in files_touched:
                    files_touched.append(f)
            patches.append(
                {
                    "id": patch_id,
                    "status": str(p.get("status") or ""),
                    "kind": str(p.get("kind") or ""),
                    "step_reference": step_reference,
                    "diff": (diff[:_MAX_DIFF_DISPLAY_BYTES] + "\n\n... (diff truncated)")
                           if len(diff) > _MAX_DIFF_DISPLAY_BYTES else diff,
                    "rationale": str(p.get("rationale") or ""),
                    "applied_via": str(p.get("applied_via") or ""),
                    "applied_at": p.get("applied_at"),
                    "has_applied_diff": bool(applied_diff),
                    "requires_external": requires_external,
                    "sync_command": sync_command,
                    "external_instructions": external_message,
                }
            )

    plan_approved = bool(meta.get("plan_approved", False))
    plan_md_path = str(meta.get("plan_markdown_path") or "").strip()
    # Generate a preview from task metadata to avoid filesystem issues.
    raw = _render_plan_markdown(task)
    if len(raw) > 12_000:
        raw = raw[:12_000] + "\n\n… (preview truncated)\n"
    plan_md_preview: str | None = raw

    repo_summary = meta.get("repository_summary") or {}
    if not isinstance(repo_summary, dict):
        repo_summary = {}
    languages = repo_summary.get("top_languages") or []
    if not isinstance(languages, list):
        languages = []
    frameworks = repo_summary.get("frameworks") or []
    if not isinstance(frameworks, list):
        frameworks = []
    hotspots = repo_summary.get("hotspots") or []
    if not isinstance(hotspots, list):
        hotspots = []
    hotspot_items: list[Dict[str, Any]] = []
    for item in hotspots:
        if not isinstance(item, dict):
            continue
        hotspot_items.append(
            {
                "path": str(item.get("path") or ""),
                "reason": str(item.get("reason") or ""),
                "lines": item.get("lines"),
            }
        )
    top_modules_summary = repo_summary.get("top_modules") or []
    if not isinstance(top_modules_summary, list):
        top_modules_summary = []
    top_dirs_summary = repo_summary.get("top_directories") or []
    if not isinstance(top_dirs_summary, list):
        top_dirs_summary = []
    test_sample = repo_summary.get("test_paths_sample") or []
    if not isinstance(test_sample, list):
        test_sample = []
    git_info = meta.get("git_info") or {}
    if not isinstance(git_info, dict):
        git_info = {}

    overview = {
        "repo_path": str(task.repo_path),
        "branch": task.branch,
        "client": (task.client or "").strip(),
        "languages": languages[:5],
        "frameworks": frameworks[:5],
        "has_tests": bool(repo_summary.get("has_tests")),
        "hotspots": [h for h in hotspot_items if h.get("path")][:8],
        "scoped_directories": (impact.get("top_directories") or [])[:8],
        "files_sample": files_sample_clean[:12],
        "scope_file_count": len(allowed_files) if allowed_files else len(files_sample_clean),
        "clarifications": {"answered": answered, "pending": pending},
        "external_patches": external_patch_notes[:8],
        "boundary_specs": spec_summary,
        "file_count": repo_summary.get("file_count"),
        "directory_count": repo_summary.get("directory_count"),
        "top_modules": top_modules_summary[:6],
        "top_directories": top_dirs_summary[:6],
        "test_paths_sample": test_sample[:10],
        "git_info": {
            "current_commit": git_info.get("current_commit"),
            "commit_message": git_info.get("commit_message"),
            "commit_author": git_info.get("commit_author"),
            "commit_date": git_info.get("commit_date"),
            "remote_url": git_info.get("remote_url"),
            "current_branch": git_info.get("current_branch") or task.branch,
        },
    }

    return {
        "OVERVIEW": overview,
        "TASK_SPECIFICATION": {
            "title": _task_title(task),
            "description": (task.description or "").strip(),
            "acceptance_criteria": acceptance_criteria[:20],
            "priority": priority,
        },
        "CLARIFYING": {"answered": answered, "pending": pending, "items": items},
        "PLANNING": {
            "plan_steps": plan_steps if isinstance(plan_steps, list) else [],
            "plan_approved": plan_approved,
            "approved_at": meta.get("plan_approved_at"),
            "plan_markdown_path": plan_md_path or None,
            "plan_markdown_preview": plan_md_preview,
            "bounded_context": {
                "file_count": len(plan_target_files) if plan_target_files else len(allowed_files),
                "targets": targets_clean[:25],
                "files_sample": files_sample_clean[:25],
                "plan_target_files": plan_target_files,
                "allowed_files": allowed_files[:200],
            },
            "boundary_specs": spec_summary,
            "scenarios": scenarios,
            "risks": [str(r).strip() for r in risks if str(r).strip()][:20],
            "refactors": [str(r).strip() for r in refactors if str(r).strip()][:20],
        },
        "CODEGEN": {"patches": patches, "files_touched": files_touched, "allowed_files": allowed_files[:50], "test_files": _filter_test_files(files_touched)},
        "RUN_TESTS": {
            "tool_name": meta.get("test_tool") or None,
            "arguments": meta.get("test_arguments") or None,
            "output": meta.get("test_output") or None,
            "exit_code": meta.get("test_exit_code") if isinstance(meta.get("test_exit_code"), int) else None,
            "tool_version": meta.get("test_tool_version") or None,
            "test_files": _filter_test_files(files_touched),
        },
    }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "SpecAgentDashboard/0.1"

    def _send(self, code: int, body: bytes, *, content_type: str) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected before we finished writing the response; safe to ignore.
            return

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=str, separators=(",", ":")).encode("utf-8")
        self._send(code, body, content_type="application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/" or path == "/index.html":
            self._send(200, _INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path.startswith("/plan/"):
            task_id = path.split("/plan/", 1)[1].strip("/")
            self._send(200, _plan_html(task_id).encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/styles.css":
            self._send(200, _STYLES_CSS.encode("utf-8"), content_type="text/css; charset=utf-8")
            return
        if path == "/app.js":
            self._send(200, _APP_JS.encode("utf-8"), content_type="application/javascript; charset=utf-8")
            return

        # API endpoints
        if path == "/api/state":
            qs = parse_qs(parsed.query or "")
            minutes = int((qs.get("minutes") or ["60"])[0])
            filter_bucket = str((qs.get("filter") or ["all"])[0])
            try:
                payload = self.server.api.state(minutes=minutes, filter_bucket=filter_bucket)  # type: ignore[attr-defined]
                self._send_json(200, payload)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if path.startswith("/api/task/"):
            task_id = path.split("/api/task/", 1)[1].strip("/")
            try:
                payload = self.server.api.task(task_id)  # type: ignore[attr-defined]
                self._send_json(200, payload)
            except KeyError:
                self._send_json(404, {"error": "task not found"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        if path.startswith("/api/plan-markdown/"):
            task_id = path.split("/api/plan-markdown/", 1)[1].strip("/")
            try:
                payload = self.server.api.plan_markdown(task_id)  # type: ignore[attr-defined]
                body = (payload.get("content") or "").encode("utf-8")
                self._send(200, body, content_type="text/markdown; charset=utf-8")
            except KeyError:
                self._send(404, b"Not Found", content_type="text/plain; charset=utf-8")
            except Exception as exc:
                self._send(500, str(exc).encode("utf-8"), content_type="text/plain; charset=utf-8")
            return

        self._send(404, b"Not Found", content_type="text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        # POST /api/task/{task_id}/run-tests
        if path.startswith("/api/task/") and path.endswith("/run-tests"):
            parts = path.split("/")
            # /api/task/{id}/run-tests → parts: ['', 'api', 'task', '{id}', 'run-tests']
            task_id = parts[3] if len(parts) >= 5 else ""
            try:
                payload = self.server.api.run_tests(task_id)  # type: ignore[attr-defined]
                self._send_json(200, payload)
            except KeyError:
                self._send_json(404, {"error": "task not found"})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - stdlib signature
        # Keep server quiet by default.
        return


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, DashboardRequestHandler)
        self.api = _DashboardApi()


def run_dashboard_server(*, host: str = "127.0.0.1", port: int = 8844) -> None:
    httpd = DashboardServer((host, port))
    try:
        httpd.serve_forever(poll_interval=0.25)
    finally:
        httpd.server_close()
