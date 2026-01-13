from __future__ import annotations

import time
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..domain.models import LogEntry, Task, TaskStatus
from ..workflow.orchestrator import TaskOrchestrator


console = Console()


TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.CANCELLED}
WORKFLOW_ORDER: list[TaskStatus] = [
    TaskStatus.CREATED,
    TaskStatus.CLARIFYING,
    TaskStatus.PLANNING,
    TaskStatus.SPEC_PENDING,
    TaskStatus.IMPLEMENTING,
    TaskStatus.VERIFYING,
    TaskStatus.COMPLETED,
    TaskStatus.CANCELLED,
]


@dataclass(frozen=True)
class TaskLogSummary:
    entry_type: str
    timestamp: datetime


def _safe_dt(dt: object) -> datetime | None:
    return dt if isinstance(dt, datetime) else None


def _fmt_ts(dt: datetime | None) -> str:
    if not dt:
        return "—"
    try:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "—"


def _latest_log_by_task(logs: Iterable[LogEntry]) -> Dict[str, TaskLogSummary]:
    latest: Dict[str, TaskLogSummary] = {}
    for entry in logs:
        ts = _safe_dt(getattr(entry, "timestamp", None))
        if not ts:
            continue
        existing = latest.get(entry.task_id)
        if not existing or ts > existing.timestamp:
            latest[entry.task_id] = TaskLogSummary(entry_type=entry.entry_type, timestamp=ts)
    return latest


def _status_style(status: TaskStatus) -> str:
    if status in TERMINAL_STATUSES:
        return "dim"
    if status in {TaskStatus.CLARIFYING, TaskStatus.SPEC_PENDING}:
        return "yellow"
    if status in {TaskStatus.PLANNING}:
        return "cyan"
    if status in {TaskStatus.IMPLEMENTING, TaskStatus.VERIFYING}:
        return "green"
    return "white"


def _build_tasks_table(tasks: list[Task], *, selected_task_id: str | None = None) -> Table:
    table = Table(title="Tasks", show_header=True, header_style="bold magenta", show_lines=False)
    table.add_column("Client", width=10)
    table.add_column("Title", no_wrap=True, width=28)
    table.add_column("Status", width=12)
    table.add_column("Updated", width=19)
    table.add_column("Summary")

    if not tasks:
        table.add_row("—", "—", "—", "No tasks yet")
        return table

    for task in tasks:
        is_selected = selected_task_id and task.id == selected_task_id
        style = "reverse" if is_selected else ""
        title = (task.title or "").strip()
        if not title:
            title = (task.description or "").splitlines()[0].strip() if task.description else f"task-{task.id[:8]}"
        summary = (task.summary or "").strip()
        if not summary:
            summary = (task.description or "").splitlines()[0].strip() if task.description else ""
        table.add_row(
            (task.client or "—")[:10],
            title[:28],
            Text(task.status.value, style=_status_style(task.status)),
            task.updated_at.isoformat(timespec="seconds"),
            summary[:80],
            style=style,
        )

    return table


def _pending_clarifications(task: Task) -> int:
    items = task.metadata.get("clarifications", []) if isinstance(task.metadata, dict) else []
    pending = 0
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().upper()
            if status == "PENDING" or status == "":
                pending += 1
    return pending


def _patch_counts(task: Task) -> tuple[int, int, int]:
    patch_state = task.metadata.get("patch_queue_state") if isinstance(task.metadata, dict) else None
    patch_pending = 0
    patch_applied = 0
    patch_rejected = 0
    if isinstance(patch_state, list):
        for item in patch_state:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("status") or "")
            if raw == "PENDING":
                patch_pending += 1
            elif raw == "APPLIED":
                patch_applied += 1
            elif raw == "REJECTED":
                patch_rejected += 1
    return patch_pending, patch_applied, patch_rejected


def _build_workflow_panel(task: Task | None) -> Panel:
    if not task:
        return Panel.fit("Select a task to view workflow.", title="Workflow", border_style="blue")

    # Derived workflow: combines TaskStatus with key metadata to show the
    # "real" stage engineers care about (plan approved, patches generated, etc).
    pending_clarifications = _pending_clarifications(task)
    plan_preview = task.metadata.get("plan_preview") if isinstance(task.metadata, dict) else None
    has_plan = bool(plan_preview)
    pending_specs = task.metadata.get("pending_specs") if isinstance(task.metadata, dict) else None
    pending_specs_count = len(pending_specs) if isinstance(pending_specs, list) else 0
    plan_approved = bool((task.metadata.get("plan_approved") if isinstance(task.metadata, dict) else False))
    patch_pending, patch_applied, patch_rejected = _patch_counts(task)
    patches_generated = isinstance(task.metadata.get("patch_queue_state"), list) if isinstance(task.metadata, dict) else False

    steps: list[tuple[str, bool]] = []
    steps.append((f"Clarifications ({pending_clarifications} pending)", pending_clarifications == 0))
    steps.append(("Plan generated", has_plan))
    # Only meaningful once a plan exists; treat "no specs" as already resolved.
    steps.append((f"Boundary specs ({pending_specs_count} pending)", pending_specs_count == 0))
    steps.append(("Plan approved", plan_approved))
    steps.append((f"Patches generated ({patch_pending + patch_applied + patch_rejected})", patches_generated))
    steps.append((f"Patches pending={patch_pending} applied={patch_applied} rejected={patch_rejected}", patches_generated and patch_pending == 0))
    steps.append(("Completed", task.status == TaskStatus.COMPLETED))

    # Determine the active step: first incomplete (or fallback to status)
    active_index = next((i for i, (_, done) in enumerate(steps) if not done), len(steps) - 1)
    lines: list[Text] = []
    for idx, (label_text, done) in enumerate(steps):
        if done:
            marker = Text("✓ ", style="green")
            label = Text(label_text, style="green")
        elif idx == active_index:
            marker = Text("▶ ", style="cyan")
            label = Text(label_text, style="cyan")
        else:
            marker = Text("• ", style="dim")
            label = Text(label_text, style="dim")
        lines.append(Text.assemble(marker, label))

    body = Text("\n").join(lines) if lines else Text("—")
    return Panel(Align.left(body), title="Workflow", border_style="blue")


def _task_title(task: Task) -> str:
    title = (task.title or "").strip()
    if title:
        return title
    return (task.description or "").splitlines()[0].strip() if task.description else f"task-{task.id[:8]}"


def _build_activity_panel(
    tasks: list[Task],
    logs: list[LogEntry],
    *,
    selected_task_id: str | None,
    limit: int = 12,
) -> Panel:
    """
    Recent activity based on reasoning logs.

    - If a task is selected: show its latest events
    - Otherwise: show global latest events across all tasks
    """
    if not logs:
        return Panel.fit("—", title="Recent activity", border_style="bright_black")

    tasks_by_id: dict[str, Task] = {t.id: t for t in tasks}

    # Filter logs
    filtered: list[LogEntry] = []
    for entry in logs:
        ts = _safe_dt(getattr(entry, "timestamp", None))
        if not ts:
            continue
        if selected_task_id:
            if entry.task_id == selected_task_id:
                filtered.append(entry)
        else:
            filtered.append(entry)

    if not filtered:
        title = "Recent activity (this task)" if selected_task_id else "Recent activity"
        return Panel.fit("—", title=title, border_style="bright_black")

    filtered.sort(key=lambda e: e.timestamp, reverse=True)
    filtered = filtered[: max(1, int(limit or 12))]

    lines: list[Text] = []
    for entry in filtered:
        ts = _fmt_ts(entry.timestamp)
        task = tasks_by_id.get(entry.task_id)
        label = f"{entry.task_id[:8]}"
        if task:
            label = f"{(task.client or '—')[:10]} · {_task_title(task)[:28]}"
        lines.append(
            Text.assemble(
                Text(ts, style="dim"),
                Text("  "),
                Text(label, style="bold"),
                Text("  "),
                Text(str(entry.entry_type or "—")),
            )
        )

    body = Text("\n").join(lines) if lines else Text("—")
    panel_title = "Recent activity (this task)" if selected_task_id else "Recent activity"
    return Panel(Align.left(body), title=panel_title, border_style="bright_black")


def _build_details_panel(task: Task | None, *, last_log: TaskLogSummary | None) -> Panel:
    if not task:
        return Panel.fit("—", title="Details", border_style="bright_black")

    patch_pending, patch_applied, patch_rejected = _patch_counts(task)
    title = (task.title or "").strip()
    if not title:
        title = (task.description or "").splitlines()[0].strip() if task.description else f"task-{task.id[:8]}"
    summary = (task.summary or "").strip()
    if not summary:
        summary = (task.description or "").splitlines()[0].strip() if task.description else ""

    details = [
        ("Client", task.client or "—"),
        ("Title", title),
        ("Reference", task.id[:8]),
        ("Repo", str(task.repo_path)),
        ("Branch", task.branch),
        ("Status", task.status.value),
        ("Updated", task.updated_at.isoformat(timespec="seconds")),
        ("Last event", (last_log.entry_type if last_log else "—")),
        ("Last event at", _fmt_ts(last_log.timestamp if last_log else None)),
        ("Patches", f"pending={patch_pending} applied={patch_applied} rejected={patch_rejected}"),
    ]

    text = Text()
    for k, v in details:
        text.append(f"{k}: ", style="bold")
        text.append(f"{v}\n")

    if summary:
        text.append("\n")
        text.append("Summary:\n", style="bold")
        text.append(summary[:240] + ("…" if len(summary) > 240 else ""))

    desc = (task.description or "").strip()
    if desc and desc.strip() != summary.strip():
        text.append("\n")
        text.append("Description:\n", style="bold")
        text.append(desc[:600] + ("…" if len(desc) > 600 else ""))

    return Panel(Align.left(text), title="Details", border_style="bright_black")


def _build_layout(
    tasks: list[Task],
    *,
    selected: Task | None,
    latest_logs: Dict[str, TaskLogSummary],
    logs: list[LogEntry],
) -> Layout:
    selected_id = selected.id if selected else None

    layout = Layout()
    layout.split_row(
        Layout(name="left", ratio=3),
        Layout(name="middle", ratio=2),
        Layout(name="right", ratio=3),
    )

    layout["left"].update(_build_tasks_table(tasks, selected_task_id=selected_id))
    layout["middle"].update(_build_workflow_panel(selected))
    layout["right"].split_column(
        Layout(name="details", ratio=3),
        Layout(name="activity", ratio=2),
    )
    layout["right"]["details"].update(
        _build_details_panel(selected, last_log=(latest_logs.get(selected.id) if selected else None))
    )
    layout["right"]["activity"].update(
        _build_activity_panel(tasks, logs, selected_task_id=selected_id, limit=12)
    )
    return layout


def run_task_dashboard(
    orchestrator: TaskOrchestrator,
    *,
    task_id: str | None = None,
    status: TaskStatus | None = None,
    show_all: bool = False,
    refresh_seconds: float = 1.0,
) -> None:
    """
    Live dashboard for tasks.

    Exit with Ctrl+C (or type 'q' + Enter).
    """

    refresh_seconds = max(0.2, float(refresh_seconds or 1.0))

    stop_event = threading.Event()

    def _listen_for_quit() -> None:
        """
        Best-effort quit handling for environments where Ctrl+C isn't convenient.
        Uses line-based input: type 'q' then Enter.
        """
        try:
            while not stop_event.is_set():
                line = sys.stdin.readline()
                if not line:
                    return
                if line.strip().lower() in {"q", "quit", "exit"}:
                    stop_event.set()
                    return
        except Exception:
            return

    threading.Thread(target=_listen_for_quit, daemon=True).start()

    def _select(tasks: list[Task]) -> Task | None:
        if not tasks:
            return None
        if task_id:
            for t in tasks:
                if t.id == task_id:
                    return t
        # Default: most recently updated
        return max(tasks, key=lambda t: t.updated_at)

    with Live(console=console, auto_refresh=False, screen=True) as live:
        while True:
            if stop_event.is_set():
                break
            try:
                tasks = orchestrator.list_tasks(status=status)
                if not show_all:
                    tasks = [t for t in tasks if t.status not in TERMINAL_STATUSES]

                # Sort: most recently updated first
                tasks.sort(key=lambda t: t.updated_at, reverse=True)

                all_logs = orchestrator.store.load_logs()
                latest_logs = _latest_log_by_task(all_logs)
                selected = _select(tasks)

                layout = _build_layout(tasks[:20], selected=selected, latest_logs=latest_logs, logs=all_logs)
                live.update(layout, refresh=True)
                time.sleep(refresh_seconds)
            except KeyboardInterrupt:
                break

