"""
MCP Server for Spec Agent - Enables Cursor/Claude integration.

This module exposes spec-agent's core functionality as MCP tools,
allowing AI assistants to drive spec-driven development workflows.

Usage:
    # Run directly
    python -m spec_agent.mcp_server

    # Or via the installed script
    spec-agent-mcp
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .domain.models import ClarificationStatus, TaskStatus
from .workflow.orchestrator import TaskOrchestrator

LOG = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP(
    "spec-agent",
    instructions="Spec-driven development agent for making changes in large/legacy repositories. "
    "Use index_repository first, then create_task, answer clarifications, generate_plan, "
    "approve specs, approve plan, and finally generate/approve patches.",
)

# Lazy-loaded orchestrator (created on first use)
_orchestrator: TaskOrchestrator | None = None


def get_orchestrator() -> TaskOrchestrator:
    """Get or create the TaskOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TaskOrchestrator()
    return _orchestrator


# =============================================================================
# Repository Indexing Tools
# =============================================================================


@mcp.tool()
def index_repository(repo_path: str, branch: str = "main") -> dict:
    """
    Index a repository to prepare for spec-driven development.

    This analyzes the codebase structure, identifies key modules, languages,
    and creates a semantic index for intelligent planning.

    Args:
        repo_path: Absolute path to the repository root
        branch: Git branch to index (default: main)

    Returns:
        Summary including repo name, languages, modules, and semantic analysis
    """
    orchestrator = get_orchestrator()
    result = orchestrator.index_repository(
        repo_path=Path(repo_path),
        branch=branch,
    )
    # Return a simplified view for the AI
    return {
        "repo_name": result.get("repo_name"),
        "repo_path": result.get("repo_path"),
        "branch": result.get("branch"),
        "indexed_at": result.get("indexed_at"),
        "languages": (result.get("repository_summary") or {}).get("top_languages", []),
        "modules": (result.get("repository_summary") or {}).get("top_modules", []),
        "has_semantic_index": result.get("semantic_index") is not None,
    }


@mcp.tool()
def get_repository_summary() -> dict:
    """
    Get the summary of the currently indexed repository.

    Returns information about languages, modules, hotspots, and architecture.
    Must have run index_repository first.

    Returns:
        Repository summary with languages, modules, architecture info
    """
    orchestrator = get_orchestrator()
    try:
        index_data = orchestrator.get_cached_repository_index()
    except ValueError as e:
        return {"error": str(e), "hint": "Run index_repository first"}

    summary = index_data.get("repository_summary", {})
    semantic = index_data.get("semantic_index", {})

    return {
        "repo_name": index_data.get("repo_name"),
        "repo_path": index_data.get("repo_path"),
        "branch": index_data.get("branch"),
        "languages": summary.get("top_languages", []),
        "modules": summary.get("top_modules", []),
        "hotspots": summary.get("hotspots", [])[:5],
        "architecture": (semantic.get("repository") or {}).get("architectureStyle"),
        "frameworks": (semantic.get("repository") or {}).get("frameworks", []),
        "domains": [d.get("name") for d in (semantic.get("domains") or [])[:5]],
    }


# =============================================================================
# Task Management Tools
# =============================================================================


@mcp.tool()
def create_task(description: str, title: Optional[str] = None, client: Optional[str] = None) -> dict:
    """
    Create a new development task from the indexed repository.

    This generates clarifying questions to understand the requirements better.
    Answer the clarifications before generating a plan.

    Args:
        description: Detailed description of what changes you want to make
        title: Optional short title for the task
        client: Optional editor/chat client driving this task (e.g. cursor, claude, terminal)

    Returns:
        Task ID and any clarifying questions that need answers
    """
    orchestrator = get_orchestrator()
    task = orchestrator.create_task_from_index(
        description=description,
        title=title,
        client=client or "mcp",
    )

    clarifications = task.metadata.get("clarifications", [])
    pending = [c for c in clarifications if c.get("status") == "PENDING"]

    return {
        "task_id": task.id,
        "status": task.status.value,
        "title": task.title,
        "clarifications_pending": len(pending),
        "clarifications": [
            {"id": c["id"], "question": c["question"]}
            for c in pending
        ],
        "next_step": "answer_clarification" if pending else "generate_plan",
    }


@mcp.tool()
def list_tasks(status_filter: Optional[str] = None) -> list:
    """
    List all spec-agent tasks.

    Args:
        status_filter: Optional filter by status (CLARIFYING, PLANNING, APPROVED, PATCHING, COMPLETED)

    Returns:
        List of tasks with their IDs, titles, and statuses
    """
    orchestrator = get_orchestrator()
    status = TaskStatus(status_filter) if status_filter else None
    tasks = orchestrator.list_tasks(status=status)

    return [
        {
            "task_id": t.id,
            "title": t.title or t.description[:50],
            "status": t.status.value,
            "repo_path": str(t.repo_path),
            "branch": t.branch,
            "updated_at": t.updated_at.isoformat(),
        }
        for t in tasks
    ]


@mcp.tool()
def get_task_status(task_id: str) -> dict:
    """
    Get detailed status of a specific task.

    Shows the current workflow stage, pending items, and next steps.

    Args:
        task_id: UUID of the task

    Returns:
        Detailed task status including patches, specs, and git status
    """
    orchestrator = get_orchestrator()
    return orchestrator.get_task_status(task_id)


# =============================================================================
# Clarification Tools
# =============================================================================


@mcp.tool()
def get_clarifications(task_id: str) -> list:
    """
    Get all clarification questions for a task.

    These questions help refine the requirements before planning.

    Args:
        task_id: UUID of the task

    Returns:
        List of clarification questions with their status
    """
    orchestrator = get_orchestrator()
    clarifications = orchestrator.list_clarifications(task_id)
    return [
        {
            "id": c.get("id"),
            "question": c.get("question"),
            "status": c.get("status"),
            "answer": c.get("answer"),
        }
        for c in clarifications
    ]


@mcp.tool()
def answer_clarification(task_id: str, clarification_id: str, answer: str) -> dict:
    """
    Answer a clarification question.

    Args:
        task_id: UUID of the task
        clarification_id: ID of the clarification question
        answer: Your answer to the question

    Returns:
        Updated clarification status and remaining pending count
    """
    orchestrator = get_orchestrator()
    orchestrator.update_clarification(
        task_id,
        clarification_id,
        answer=answer,
        status=ClarificationStatus.ANSWERED,
    )

    # Check remaining
    clarifications = orchestrator.list_clarifications(task_id)
    pending = [c for c in clarifications if c.get("status") == "PENDING"]

    return {
        "status": "answered",
        "clarification_id": clarification_id,
        "pending_count": len(pending),
        "ready_for_plan": len(pending) == 0,
        "next_step": "generate_plan" if len(pending) == 0 else "answer_clarification",
    }


@mcp.tool()
def skip_clarification(task_id: str, clarification_id: str) -> dict:
    """
    Skip/override a clarification question (proceed without answering).

    Args:
        task_id: UUID of the task
        clarification_id: ID of the clarification question to skip

    Returns:
        Updated status
    """
    orchestrator = get_orchestrator()
    orchestrator.update_clarification(
        task_id,
        clarification_id,
        answer="",
        status=ClarificationStatus.OVERRIDDEN,
    )

    clarifications = orchestrator.list_clarifications(task_id)
    pending = [c for c in clarifications if c.get("status") == "PENDING"]

    return {
        "status": "skipped",
        "pending_count": len(pending),
        "ready_for_plan": len(pending) == 0,
    }


# =============================================================================
# Planning Tools
# =============================================================================


@mcp.tool()
def generate_plan(task_id: str, fast: bool = False) -> dict:
    """
    Generate an implementation plan for the task.

    Creates a step-by-step plan with boundary specifications that define
    the contracts and invariants for the changes. Automatically freezes
    scope to create a final plan ready for approval.

    Args:
        task_id: UUID of the task
        fast: Skip rationale enhancement for faster execution

    Returns:
        Plan steps, boundary specs, and any identified risks
    """
    orchestrator = get_orchestrator()

    if orchestrator.has_pending_clarifications(task_id):
        return {
            "error": "Clarifications pending",
            "hint": "Answer all clarifications before generating a plan",
            "pending_clarifications": orchestrator.list_clarifications(task_id),
        }

    # auto_freeze_scope=True creates a final plan directly (more natural flow)
    result = orchestrator.generate_plan(task_id, skip_rationale_enhancement=fast, auto_freeze_scope=True)

    plan = result.get("plan", {})
    steps = plan.get("steps", [])

    return {
        "task_id": task_id,
        "steps": [
            {
                "index": i + 1,
                "description": s.get("description") if isinstance(s, dict) else str(s),
                "target_files": s.get("target_files", []) if isinstance(s, dict) else [],
            }
            for i, s in enumerate(steps)
        ],
        "risks": plan.get("risks", []),
        "pending_specs": result.get("pending_specs", []),
        "next_step": "get_boundary_specs" if result.get("pending_specs") else "approve_plan",
    }


@mcp.tool()
def get_boundary_specs(task_id: str) -> list:
    """
    Get boundary specifications for a task.

    Boundary specs define the contracts, actors, interfaces, and invariants
    that must be maintained during implementation.

    Args:
        task_id: UUID of the task

    Returns:
        List of boundary specs with their approval status
    """
    orchestrator = get_orchestrator()
    specs = orchestrator.get_boundary_specs(task_id)

    return [
        {
            "id": s.get("id"),
            "boundary_name": s.get("boundary_name"),
            "status": s.get("status"),
            "description": s.get("human_description"),
            "actors": (s.get("machine_spec") or {}).get("actors", []),
            "interfaces": (s.get("machine_spec") or {}).get("interfaces", []),
            "invariants": (s.get("machine_spec") or {}).get("invariants", []),
        }
        for s in specs
    ]


@mcp.tool()
def approve_spec(task_id: str, spec_id: str) -> dict:
    """
    Approve a boundary specification.

    Args:
        task_id: UUID of the task
        spec_id: ID of the boundary spec to approve

    Returns:
        Updated spec status and remaining pending count
    """
    orchestrator = get_orchestrator()
    result = orchestrator.approve_spec(task_id, spec_id)

    specs = orchestrator.get_boundary_specs(task_id)
    pending = [s for s in specs if s.get("status") == "PENDING"]

    return {
        "spec_id": result["spec_id"],
        "status": "APPROVED",
        "pending_count": len(pending),
        "all_specs_resolved": len(pending) == 0,
        "next_step": "approve_plan" if len(pending) == 0 else "approve_spec",
    }


@mcp.tool()
def approve_all_specs(task_id: str) -> dict:
    """
    Approve all pending boundary specifications at once.

    Args:
        task_id: UUID of the task

    Returns:
        Count of approved specs
    """
    orchestrator = get_orchestrator()
    result = orchestrator.approve_all_specs(task_id)
    return {
        "approved_count": result.get("approved_count", 0),
        "next_step": "approve_plan",
    }


@mcp.tool()
def skip_spec(task_id: str, spec_id: str) -> dict:
    """
    Skip a boundary specification (proceed without enforcing it).

    Args:
        task_id: UUID of the task
        spec_id: ID of the boundary spec to skip

    Returns:
        Updated status
    """
    orchestrator = get_orchestrator()
    result = orchestrator.skip_spec(task_id, spec_id)

    specs = orchestrator.get_boundary_specs(task_id)
    pending = [s for s in specs if s.get("status") == "PENDING"]

    return {
        "spec_id": result["spec_id"],
        "status": "SKIPPED",
        "pending_count": len(pending),
        "all_specs_resolved": len(pending) == 0,
    }


@mcp.tool()
def approve_plan(task_id: str) -> dict:
    """
    Approve the implementation plan.

    All boundary specs must be resolved (approved or skipped) before approving.
    After approval, you can generate patches.

    Args:
        task_id: UUID of the task

    Returns:
        Approval confirmation and next steps
    """
    orchestrator = get_orchestrator()

    # Check for pending specs
    specs = orchestrator.get_boundary_specs(task_id)
    pending = [s for s in specs if s.get("status") == "PENDING"]
    if pending:
        return {
            "error": "Pending boundary specs",
            "pending_count": len(pending),
            "hint": "Approve or skip all boundary specs before approving the plan",
            "pending_specs": [s.get("boundary_name") for s in pending],
        }

    result = orchestrator.approve_plan(task_id)
    return {
        "status": "approved",
        "task_id": task_id,
        "next_step": "generate_patches",
    }


# =============================================================================
# Patch Generation & Review Tools
# =============================================================================


@mcp.tool()
def generate_patches(task_id: str, fast: bool = False) -> dict:
    """
    Generate code patches for the approved plan.

    Creates incremental diffs that implement each plan step.
    Review and approve patches before they are applied.

    Args:
        task_id: UUID of the task
        fast: Skip rationale enhancement for faster execution

    Returns:
        Count of generated patches and test suggestions
    """
    orchestrator = get_orchestrator()
    result = orchestrator.generate_patches(task_id, skip_rationale_enhancement=fast)

    return {
        "task_id": task_id,
        "patch_count": result.get("patch_count", 0),
        "test_count": result.get("test_count", 0),
        "next_step": "list_patches",
    }


@mcp.tool()
def list_patches(task_id: str) -> list:
    """
    List all patches for a task.

    Args:
        task_id: UUID of the task

    Returns:
        List of patches with their status and preview
    """
    orchestrator = get_orchestrator()
    patches = orchestrator.list_patches(task_id)

    return [
        {
            "id": p.id,
            "step_reference": p.step_reference,
            "status": p.status.value,
            "kind": p.kind.value,
            "diff_preview": p.diff[:500] + ("..." if len(p.diff) > 500 else ""),
            "rationale": p.rationale[:200] + ("..." if len(p.rationale) > 200 else ""),
        }
        for p in patches
    ]


@mcp.tool()
def get_patch_details(task_id: str, patch_id: str) -> dict:
    """
    Get full details of a specific patch including complete diff.

    Args:
        task_id: UUID of the task
        patch_id: UUID of the patch

    Returns:
        Complete patch details including full diff and rationale
    """
    orchestrator = get_orchestrator()
    patches = orchestrator.list_patches(task_id)
    patch = next((p for p in patches if p.id == patch_id), None)

    if not patch:
        return {"error": f"Patch {patch_id} not found"}

    return {
        "id": patch.id,
        "step_reference": patch.step_reference,
        "status": patch.status.value,
        "kind": patch.kind.value,
        "diff": patch.diff,
        "rationale": patch.rationale,
    }


@mcp.tool()
def get_next_pending_patch(task_id: str) -> dict:
    """
    Get the next patch that needs review.

    Args:
        task_id: UUID of the task

    Returns:
        Next pending patch or message if none remain
    """
    orchestrator = get_orchestrator()
    patch = orchestrator.get_next_pending_patch(task_id)

    if not patch:
        return {
            "message": "No pending patches",
            "all_reviewed": True,
        }

    return {
        "id": patch.id,
        "step_reference": patch.step_reference,
        "status": patch.status.value,
        "diff": patch.diff,
        "rationale": patch.rationale,
    }


@mcp.tool()
def approve_patch(task_id: str, patch_id: str) -> dict:
    """
    Approve and apply a patch to the working tree.

    This will modify files in the repository.

    Args:
        task_id: UUID of the task
        patch_id: UUID of the patch to approve

    Returns:
        Confirmation and remaining patch count
    """
    orchestrator = get_orchestrator()
    patch = orchestrator.approve_patch(task_id, patch_id)

    # Count remaining
    patches = orchestrator.list_patches(task_id)
    pending = [p for p in patches if p.status.value == "PENDING"]

    return {
        "patch_id": patch.id,
        "status": "applied",
        "pending_count": len(pending),
        "all_applied": len(pending) == 0,
        "next_step": "approve_patch" if pending else "task_complete",
    }


@mcp.tool()
def reject_patch(task_id: str, patch_id: str) -> dict:
    """
    Reject a patch and regenerate the plan.

    This will discard the current patch queue and create a new plan
    that accounts for the rejection.

    Args:
        task_id: UUID of the task
        patch_id: UUID of the patch to reject

    Returns:
        Confirmation that plan was regenerated
    """
    orchestrator = get_orchestrator()
    orchestrator.reject_patch(task_id, patch_id)

    return {
        "patch_id": patch_id,
        "status": "rejected",
        "message": "Plan regenerated - review new patches",
        "next_step": "list_patches",
    }


# =============================================================================
# Workflow Helpers
# =============================================================================


@mcp.tool()
def get_workflow_status(task_id: str) -> dict:
    """
    Get a comprehensive workflow status for a task.

    Shows where you are in the spec-driven workflow and what to do next.

    Args:
        task_id: UUID of the task

    Returns:
        Current stage, completed steps, and recommended next action
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    # Gather status
    clarifications = orchestrator.list_clarifications(task_id)
    pending_clarifications = [c for c in clarifications if c.get("status") == "PENDING"]

    specs = orchestrator.get_boundary_specs(task_id)
    pending_specs = [s for s in specs if s.get("status") == "PENDING"]

    patches = orchestrator.list_patches(task_id)
    pending_patches = [p for p in patches if p.status.value == "PENDING"]
    applied_patches = [p for p in patches if p.status.value == "APPLIED"]

    # Determine stage and next action
    if pending_clarifications:
        stage = "clarifying"
        next_action = "answer_clarification"
        hint = f"Answer {len(pending_clarifications)} clarification question(s)"
    elif task.status == TaskStatus.CLARIFYING:
        stage = "ready_to_plan"
        next_action = "generate_plan"
        hint = "Generate implementation plan"
    elif pending_specs:
        stage = "reviewing_specs"
        next_action = "approve_spec"
        hint = f"Review {len(pending_specs)} boundary spec(s)"
    elif task.status == TaskStatus.PLANNING:
        stage = "ready_to_approve"
        next_action = "approve_plan"
        hint = "Approve the plan to enable patch generation"
    elif not patches:
        stage = "ready_to_generate"
        next_action = "generate_patches"
        hint = "Generate implementation patches"
    elif pending_patches:
        stage = "reviewing_patches"
        next_action = "approve_patch"
        hint = f"Review {len(pending_patches)} pending patch(es)"
    else:
        stage = "completed"
        next_action = None
        hint = "All patches applied!"

    return {
        "task_id": task_id,
        "title": task.title,
        "stage": stage,
        "status": task.status.value,
        "next_action": next_action,
        "hint": hint,
        "progress": {
            "clarifications": {
                "total": len(clarifications),
                "pending": len(pending_clarifications),
            },
            "specs": {
                "total": len(specs),
                "pending": len(pending_specs),
            },
            "patches": {
                "total": len(patches),
                "pending": len(pending_patches),
                "applied": len(applied_patches),
            },
        },
    }


# =============================================================================
# Entry Point
# =============================================================================


def run():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
