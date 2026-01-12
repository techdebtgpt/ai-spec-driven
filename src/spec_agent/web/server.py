from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from ..config.settings import get_settings
from ..domain.models import LogEntry, Task, TaskStatus
from ..persistence.store import JsonStore


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


_WORKFLOW_ORDER: list[Tuple[str, str]] = [
    ("TASK_SPECIFICATION", "Task specification"),
    ("CLARIFYING", "Clarifying questions"),
    ("PLANNING", "Context analysis & plan"),
    ("APPROVAL", "User approval"),
    ("CODEGEN", "Code generation"),
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
    has_plan = bool(task.metadata.get("plan_preview")) if isinstance(task.metadata, dict) else False
    pending_specs = task.metadata.get("pending_specs") if isinstance(task.metadata, dict) else None
    pending_specs_count = len(pending_specs) if isinstance(pending_specs, list) else 0
    plan_approved = bool(task.metadata.get("plan_approved")) if isinstance(task.metadata, dict) else False
    patches = _patch_counts(task)

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
    elif has_plan and (pending_specs_count > 0) and not plan_approved:
        active_key = "PLANNING"
    elif has_plan and not plan_approved:
        active_key = "APPROVAL"
    elif plan_approved and (patches.get("total", 0) == 0):
        active_key = "CODEGEN"
    elif plan_approved and (patches.get("pending", 0) > 0):
        active_key = "CODEGEN"

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
            done = pending_clarifications == 0 and (plan_generated_at is not None or has_plan)
            started_at, ended_at = clar_start, clar_end
            if pending_clarifications > 0:
                chips.append({"kind": "warning", "text": f"{pending_clarifications} pending"})
        elif key == "PLANNING":
            done = has_plan
            started_at = plan_generated_at
            ended_at = plan_generated_at
            if pending_specs_count > 0:
                chips.append({"kind": "warning", "text": f"{pending_specs_count} specs pending"})
        elif key == "APPROVAL":
            done = plan_approved
            started_at, ended_at = approve_start, approve_end
            if plan_approved:
                chips.append({"kind": "success", "text": "user approved"})
        elif key == "CODEGEN":
            done = (patches.get("total", 0) > 0) or task.status in {TaskStatus.VERIFYING, TaskStatus.COMPLETED}
            started_at, ended_at = codegen_start, codegen_end
            if patches.get("total", 0) > 0:
                chips.append({"kind": "neutral", "text": f"{patches['total']} patches"})

        status = "pending"
        if done:
            status = "done"
        elif key == active_key:
            status = "active"

        steps.append(
            {
                "key": key,
                "label": label,
                "status": status,
                "started_at": _iso(started_at) if started_at else None,
                "ended_at": _iso(ended_at) if ended_at else None,
                "duration_seconds": _dur(started_at, ended_at),
                "chips": chips,
            }
        )

    return steps


def _serialize_task(task: Task, latest_log: LogEntry | None) -> Dict[str, Any]:
    patches = _patch_counts(task)
    return {
        "id": task.id,
        "client": (task.client or "").strip() or None,
        "title": _task_title(task),
        "summary": _task_summary(task),
        "description": task.description,
        "status": task.status.value,
        "bucket": _task_bucket(task.status),
        "created_at": _iso(task.created_at),
        "updated_at": _iso(task.updated_at),
        "repo_path": str(task.repo_path),
        "branch": task.branch,
        "last_event": latest_log.entry_type if latest_log else None,
        "last_event_at": _iso(latest_log.timestamp) if latest_log else None,
        "patch_counts": patches,
        "plan_approved": bool(task.metadata.get("plan_approved")) if isinstance(task.metadata, dict) else False,
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
      <div class="panelHeader">
        <div class="panelTitle">Evidence &amp; Details</div>
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
.chip.yellow{border-color:rgba(245,158,11,.35); color:#fde68a; background:rgba(245,158,11,.10)}
.chip.red{border-color:rgba(239,68,68,.35); color:#fecaca; background:rgba(239,68,68,.10)}
.dot2{width:10px;height:10px;border-radius:999px; background:var(--muted)}
.dot2.good{background:var(--good)}
.dot2.warn{background:var(--warn)}
.dot2.bad{background:var(--bad)}
.dot2.blue{background:var(--accent)}

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
.stepLabel{font-size:13px; font-weight:650}
.stepChips{margin-top:8px; display:flex; gap:8px; flex-wrap:wrap}
.small{font-size:12px; color:var(--muted); margin-top:6px}
.detailsBody{padding:16px}
.details{overflow:auto}
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
.qcard{
  border:1px solid var(--border);
  background:rgba(10,16,32,.55);
  border-radius:12px;
  padding:12px;
  margin-top:10px;
}
.qhead{font-size:13px; font-weight:650}
.qans{margin-top:10px; font-size:13px}
.qans.ok{color:#bbf7d0}
.qans.muted{color:var(--muted)}
.tree{
  border:1px solid var(--border);
  background:rgba(10,16,32,.45);
  border-radius:12px;
  padding:10px;
  margin-top:10px;
}
.treeDir{font-weight:650; font-size:13px; margin-bottom:6px}
.treeFile{font-size:12px; color:var(--muted); padding:2px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
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
  filter: 'all',
  range: 'today',
  lastNow: null,
};

const $ = (id) => document.getElementById(id);

function parseRangeToMinutes(range) {
  if (range === '5m') return 5;
  if (range === '1h') return 60;
  if (range === 'today') return 24 * 60;
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
  return `${h}h ago`;
}

function statusDot(task) {
  // Map bucket/status into dot color similar to mock
  if (task.bucket === 'failed') return 'bad';
  if (task.status === 'IMPLEMENTING' || task.status === 'VERIFYING') return 'good';
  if (task.status === 'CLARIFYING' || task.status === 'SPEC_PENDING') return 'warn';
  return 'blue';
}

function chipForTask(task) {
  const chips = [];
  // Show patch counts if any
  if (task.patch_counts && task.patch_counts.total > 0) {
    chips.push({ cls: 'purple', text: `${task.patch_counts.total} patches` });
  }
  // Show status as chip
  chips.push({ cls: 'chip', text: task.status });
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
    title.textContent = `${t.client ? t.client : '—'} · ${t.title}`;
    left.appendChild(dot);
    left.appendChild(title);

    const right = document.createElement('div');
    right.style.color = 'var(--muted)';
    right.style.fontSize = '12px';
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
  $('workflowSubtitle').textContent = task ? `${task.client || '—'} · ${task.title}` : 'Select a task';
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
      renderDetails(task, step, null);
    };

    const rail = document.createElement('div');
    rail.className = 'rail';
    const fill = document.createElement('div');
    fill.className = 'fill' + (step.status === 'active' ? ' active' : '');
    fill.style.height = step.status === 'done' ? '100%' : (step.status === 'active' ? '50%' : '0%');
    rail.appendChild(fill);

    const body = document.createElement('div');
    body.className = 'stepBody';
    const label = document.createElement('div');
    label.className = 'stepLabel';
    label.textContent = step.label;

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
      const ch = document.createElement('span');
      let cls = 'chip';
      if (c.kind === 'success') cls += ' green';
      if (c.kind === 'warning') cls += ' yellow';
      if (c.kind === 'danger') cls += ' red';
      if (c.kind === 'neutral') cls += ' purple';
      ch.className = cls;
      ch.textContent = c.text;
      chips.appendChild(ch);
    });

    const small = document.createElement('div');
    small.className = 'small';
    if (step.ended_at) {
      small.textContent = `Ended: ${step.ended_at}`;
    } else if (step.started_at) {
      small.textContent = `Started: ${step.started_at}`;
    } else {
      small.textContent = '—';
    }

    body.appendChild(label);
    body.appendChild(chips);
    body.appendChild(small);

    el.appendChild(rail);
    el.appendChild(body);
    stepsEl.appendChild(el);
  });
}

function renderDetails(task, step, detail) {
  const el = $('detailsBody');
  el.innerHTML = '';
  if (!task || !step) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = 'Select a step to view details';
    el.appendChild(empty);
    return;
  }

  // Header
  const head = document.createElement('div');
  head.style.marginBottom = '14px';
  const h1 = document.createElement('div');
  h1.style.fontWeight = '700';
  h1.style.fontSize = '14px';
  h1.textContent = step.label;
  const h2 = document.createElement('div');
  h2.className = 'small';
  const dur = typeof step.duration_seconds === 'number' ? `${Math.round(step.duration_seconds * 10) / 10}s` : '—';
  h2.textContent = `${dur} · ${step.status.toUpperCase()}`;
  head.appendChild(h1);
  head.appendChild(h2);
  el.appendChild(head);

  const d = detail || {};

  // Step-specific rendering (mockup-style)
  if (step.key === 'TASK_SPECIFICATION') {
    const card = document.createElement('div');
    card.className = 'card';
    const title = document.createElement('div');
    title.className = 'cardTitle';
    title.textContent = d.title || task.title;
    const desc = document.createElement('div');
    desc.className = 'cardBody';
    desc.textContent = d.description || task.description || '—';
    card.appendChild(title);
    card.appendChild(desc);
    el.appendChild(card);
    return;
  }

  if (step.key === 'CLARIFYING') {
    const items = d.items || [];
    const summary = document.createElement('div');
    summary.className = 'small';
    summary.textContent = `${d.answered || 0} answered · ${d.pending || 0} pending`;
    el.appendChild(summary);

    items.forEach(it => {
      const q = document.createElement('div');
      q.className = 'qcard';
      const qh = document.createElement('div');
      qh.className = 'qhead';
      qh.textContent = `${it.id || ''}  ${it.question || ''}`.trim();
      const ans = document.createElement('div');
      ans.className = 'qans ' + (it.status === 'ANSWERED' ? 'ok' : 'muted');
      ans.textContent = it.status === 'ANSWERED' ? (it.answer || '—') : '(pending)';
      q.appendChild(qh);
      q.appendChild(ans);
      el.appendChild(q);
    });
    return;
  }

  if (step.key === 'PLANNING') {
    if (d.plan_approved) {
      const banner = document.createElement('div');
      banner.className = 'banner';
      banner.textContent = 'User approved this plan';
      el.appendChild(banner);
    }

    // Bounded context
    const bc = d.bounded_context || {};
    const allowed = (bc.allowed_files || []);
    if (allowed.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = 'Bounded context (files)';
      el.appendChild(sec);

      // Group by top-level directory
      const groups = {};
      allowed.forEach(p => {
        const parts = (p || '').split('/');
        const key = parts.length ? parts[0] : '(root)';
        groups[key] = groups[key] || [];
        groups[key].push(p);
      });
      Object.keys(groups).slice(0, 30).forEach(k => {
        const g = document.createElement('div');
        g.className = 'tree';
        const gh = document.createElement('div');
        gh.className = 'treeDir';
        gh.textContent = k;
        g.appendChild(gh);
        groups[k].slice(0, 40).forEach(p => {
          const f = document.createElement('div');
          f.className = 'treeFile';
          f.textContent = p;
          g.appendChild(f);
        });
        el.appendChild(g);
      });
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
        txt.textContent = (s.description || String(s)).trim();
        li.appendChild(n);
        li.appendChild(txt);
        el.appendChild(li);
      });
    }
    return;
  }

  if (step.key === 'APPROVAL') {
    const banner = document.createElement('div');
    banner.className = d.plan_approved ? 'banner' : 'banner warn';
    banner.textContent = d.plan_approved ? 'Plan approved' : 'Plan not approved yet';
    el.appendChild(banner);
    return;
  }

  if (step.key === 'CODEGEN') {
    const patches = d.patches || [];
    const pending = (patches || []).filter(p => p.status === 'PENDING');
    const banner = document.createElement('div');
    banner.className = pending.length ? 'banner warn' : 'banner';
    banner.textContent = pending.length ? 'Not applied' : 'Applied / no pending patches';
    el.appendChild(banner);

    // Files touched (from diffs)
    const files = d.files_touched || [];
    if (files.length) {
      const sec = document.createElement('div');
      sec.className = 'sectionTitle';
      sec.textContent = `Files touched (${files.length})`;
      el.appendChild(sec);
      files.slice(0, 30).forEach(f => {
        const row = document.createElement('div');
        row.className = 'fileRow';
        row.textContent = f;
        el.appendChild(row);
      });
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
        row.textContent = `${p.id.slice(0,8)} · ${p.status} · ${p.step_reference}`;
        row.onclick = () => {
          state.selectedPatchId = p.id;
          renderDetails(task, step, detail);
        };
        list.appendChild(row);
      });
      el.appendChild(list);

      const pre = document.createElement('pre');
      pre.className = 'diff';
      pre.textContent = (chosen && chosen.diff) ? chosen.diff : '—';
      el.appendChild(pre);
    }
    return;
  }

  const fallback = document.createElement('div');
  fallback.className = 'small';
  fallback.textContent = 'No renderer for this step yet.';
  el.appendChild(fallback);
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
  renderWorkflow(data.task, data.workflow);
  renderDetails(data.task, null, null);
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

async function tick() {
  try {
    const data = await fetchState();
    state.lastNow = data.now;
    state.tasks = data.tasks || [];
    renderTasks();

    // Keep selected task view fresh
    if (state.selectedTaskId) {
      const res = await fetch(`/api/task/${encodeURIComponent(state.selectedTaskId)}`, { cache: 'no-store' });
      if (res.ok) {
        const tdata = await res.json();
        renderWorkflow(tdata.task, tdata.workflow);
        if (state.selectedStepKey) {
          const step = (tdata.workflow || []).find(s => s.key === state.selectedStepKey);
          if (step) {
            const detail = (tdata.step_details || {})[step.key] || null;
            renderDetails(tdata.task, step, detail);
          }
        }
      }
    }
  } catch (e) {
    // If state dir missing, etc.
    console.warn(e);
  }
}

wireUI();
tick();
setInterval(tick, 1200);
"""


class _DashboardApi:
    def __init__(self) -> None:
        settings = get_settings()
        self.store = JsonStore(settings.state_dir)

    def _load(self) -> tuple[list[Task], list[LogEntry]]:
        tasks = self.store.load_tasks()
        logs = self.store.load_logs()
        return tasks, logs

    def state(self, *, minutes: int, filter_bucket: str) -> Dict[str, Any]:
        tasks, logs = self._load()
        now = _utcnow()

        # filter by time window based on task.updated_at
        cutoff = now.timestamp() - float(minutes * 60)
        filtered_tasks: list[Task] = []
        for t in tasks:
            if t.updated_at.timestamp() < cutoff:
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

        # Keep payload light.
        data_tasks = [_serialize_task(t, latest_by_task.get(t.id)) for t in filtered_tasks]

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
        items.append(
            {
                "id": str(it.get("id") or ""),
                "question": str(it.get("question") or ""),
                "answer": str(it.get("answer") or ""),
                "status": status or "PENDING",
            }
        )

    plan_preview = meta.get("plan_preview") or {}
    plan_steps = (plan_preview.get("steps") or []) if isinstance(plan_preview, dict) else []
    bounded_ctx = meta.get("bounded_context") or {}
    if not isinstance(bounded_ctx, dict):
        bounded_ctx = {}
    scoped = bounded_ctx.get("manual") or bounded_ctx.get("plan_targets") or {}
    if not isinstance(scoped, dict):
        scoped = {}
    scope = (scoped.get("scope") or {}) if isinstance(scoped, dict) else {}
    allowed_files = scope.get("allowed_files") if isinstance(scope, dict) else []
    if not isinstance(allowed_files, list):
        allowed_files = []

    patches_raw = meta.get("patch_queue_state") or []
    patches: list[Dict[str, Any]] = []
    files_touched: list[str] = []
    if isinstance(patches_raw, list):
        for p in patches_raw:
            if not isinstance(p, dict):
                continue
            diff = str(p.get("diff") or "")
            for f in _extract_files_from_diff(diff):
                if f not in files_touched:
                    files_touched.append(f)
            patches.append(
                {
                    "id": str(p.get("id") or ""),
                    "status": str(p.get("status") or ""),
                    "kind": str(p.get("kind") or ""),
                    "step_reference": str(p.get("step_reference") or ""),
                    "diff": diff,
                    "rationale": str(p.get("rationale") or ""),
                }
            )

    plan_approved = bool(meta.get("plan_approved", False))

    return {
        "TASK_SPECIFICATION": {
            "title": _task_title(task),
            "description": (task.description or "").strip(),
        },
        "CLARIFYING": {"answered": answered, "pending": pending, "items": items},
        "PLANNING": {
            "plan_steps": plan_steps if isinstance(plan_steps, list) else [],
            "plan_approved": plan_approved,
            "bounded_context": {"allowed_files": allowed_files, "targets": scope.get("targets") if isinstance(scope, dict) else []},
        },
        "APPROVAL": {"plan_approved": plan_approved, "approved_at": meta.get("plan_approved_at")},
        "CODEGEN": {"patches": patches, "files_touched": files_touched},
    }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "SpecAgentDashboard/0.1"

    def _send(self, code: int, body: bytes, *, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self._send(code, body, content_type="application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        path = parsed.path or "/"

        if path == "/" or path == "/index.html":
            self._send(200, _INDEX_HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
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

        self._send(404, b"Not Found", content_type="text/plain; charset=utf-8")

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


