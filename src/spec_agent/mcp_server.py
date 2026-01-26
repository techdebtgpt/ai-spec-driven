"""
MCP Server for Spec Agent - Prompt-Based Architecture.

This module exposes spec-agent's core functionality using MCP prompts for generation
and tools for state management. LLM calls are delegated to the Cursor/Claude client
rather than running in the backend.

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
    instructions="Spec-driven development agent using prompt-based architecture. "
    "Use index_repository first, then create_task. For LLM-based generation, invoke the "
    "appropriate prompts (generate_clarifications_prompt, generate_plan_prompt, etc.), then "
    "submit results using submit_* tools.",
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
# MCP PROMPTS - LLM Generation Tasks
# =============================================================================


@mcp.prompt()
def generate_clarifications_prompt(task_id: str):
    """
    Generate clarifying questions for a task.

    This prompt guides Claude to generate 3-5 clarification questions
    to better understand the task requirements before planning.

    Args:
        task_id: UUID of the task

    Returns:
        Prompt messages for Claude to generate clarifications
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    # Get repository context
    try:
        repo_index = orchestrator.get_cached_repository_index()
        repo_summary = repo_index.get("repository_summary", {})
        languages = repo_summary.get("top_languages", [])
        modules = repo_summary.get("top_modules", [])

        repo_context = f"""Repository Context:
- Primary languages: {', '.join([l['language'] for l in languages[:3]])}
- Key modules: {', '.join(modules[:5])}
"""
    except ValueError:
        repo_context = ""

    prompt_content = f"""You are a senior software architect helping to clarify requirements for a software development task.

Task Details:
{repo_context}
Title: {task.title or "N/A"}
Description: {task.description}

Your job is to generate 3-5 clarifying questions that will help understand:
1. Edge cases and constraints
2. Performance or scalability requirements
3. Integration points with existing systems
4. Error handling expectations
5. Testing requirements

For each question, provide:
- The question text (clear and specific)
- Rationale: Why this question is important
- Default answer: A reasonable default if the user skips

Return your response as a JSON object:
{{
  "clarifications": [
    {{
      "question": "the question text",
      "rationale": "why it matters",
      "default_answer": "suggested default"
    }}
  ]
}}

IMPORTANT: After generating the clarifications, call the tool:
submit_clarifications(task_id="{task_id}", clarifications_json=<your_json>)
"""

    return [{"role": "user", "content": prompt_content}]


@mcp.prompt()
def generate_plan_prompt(task_id: str):
    """
    Generate an implementation plan for a task.

    This prompt guides Claude to create a step-by-step implementation plan
    based on the task description and answered clarifications.

    Args:
        task_id: UUID of the task

    Returns:
        Prompt messages for Claude to generate a plan
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    clarifications = orchestrator.list_clarifications(task_id)

    # Format clarifications
    clarifications_text = "\n".join([
        f"Q: {c['question']}\nA: {c.get('answer', c.get('default_answer', 'N/A'))}"
        for c in clarifications
    ])

    # Get repository context
    try:
        repo_index = orchestrator.get_cached_repository_index()
        repo_summary = repo_index.get("repository_summary", {})
        languages = repo_summary.get("top_languages", [])
        modules = repo_summary.get("top_modules", [])
        hotspots = repo_summary.get("hotspots", [])

        repo_context = f"""Repository Context:
- Primary languages: {', '.join([l['language'] for l in languages[:3]])}
- Key modules: {', '.join(modules[:5])}
- Code hotspots: {', '.join([h.get('file', '') for h in hotspots[:5]])}
"""
    except ValueError:
        repo_context = ""

    prompt_content = f"""You are a senior software architect creating a detailed implementation plan.

{repo_context}

Task Details:
Title: {task.title or "N/A"}
Description: {task.description}

Clarifications:
{clarifications_text or "None provided"}

Your job is to create a step-by-step implementation plan that:
1. Breaks the task into logical, incremental steps
2. Identifies which files need to be modified for each step
3. Considers dependencies between steps
4. Identifies potential risks

Each step should be implementable independently and testable.

Return your response as a JSON object:
{{
  "steps": [
    {{
      "description": "what to implement in this step",
      "target_files": ["file1.py", "file2.js"],
      "reasoning": "why this step is needed and how it fits in"
    }}
  ],
  "risks": [
    "potential issue 1",
    "potential issue 2"
  ],
  "dependencies": ["external library needed", "..."]
}}

IMPORTANT: After generating the plan, call the tool:
submit_plan(task_id="{task_id}", plan_json=<your_json>)
"""

    return [{"role": "user", "content": prompt_content}]


@mcp.prompt()
def generate_patch_prompt(task_id: str, step_index: int):
    """
    Generate a code patch for a specific plan step.

    This prompt guides Claude to create the actual code changes (as a unified diff)
    for implementing a single step from the plan.

    Args:
        task_id: UUID of the task
        step_index: Zero-based index of the step to implement

    Returns:
        Prompt messages for Claude to generate a patch
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {})
    steps = plan_data.get("steps", [])

    if step_index >= len(steps):
        return [{"role": "user", "content": f"Error: Step index {step_index} out of range (plan has {len(steps)} steps)"}]

    step = steps[step_index]
    step_desc = step.get("description") if isinstance(step, dict) else str(step)
    target_files = step.get("target_files", []) if isinstance(step, dict) else []
    reasoning = step.get("reasoning", "") if isinstance(step, dict) else ""

    prompt_content = f"""You are an expert software engineer implementing a specific code change.

Task: {task.title or "N/A"}

Plan Step #{step_index + 1}:
Description: {step_desc}
Target Files: {', '.join(target_files) if target_files else 'Not specified - infer from description'}
Reasoning: {reasoning}

Your job is to:
1. Read the target files (or infer them from the description)
2. Implement the changes needed for this step
3. Create a unified diff patch showing the changes
4. Explain the rationale for your implementation choices

Guidelines:
- Follow the existing code style and patterns
- Add appropriate error handling
- Consider edge cases mentioned in clarifications
- Keep changes minimal and focused on this step only
- Include comments where logic isn't self-evident

After implementing the changes:
1. Generate a unified diff (use `git diff` or create manually)
2. Write a rationale explaining:
   - What you changed and why
   - Design decisions made
   - Trade-offs considered
   - How this addresses the step requirements

Then call:
submit_patch(task_id="{task_id}", step_index={step_index}, diff="<unified_diff>", rationale="<explanation>")
"""

    return [{"role": "user", "content": prompt_content}]


# =============================================================================
# Repository Indexing Tools (State Management)
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
# Task Management Tools (State Management)
# =============================================================================


@mcp.tool()
def create_task(description: str, title: Optional[str] = None, client: Optional[str] = None) -> dict:
    """
    Create a new development task (does NOT auto-generate clarifications).

    After creating the task, invoke the generate_clarifications_prompt to create
    clarifying questions using Claude's LLM.

    Args:
        description: Detailed description of what changes you want to make
        title: Optional short title for the task
        client: Optional editor/chat client driving this task (e.g. cursor, claude)

    Returns:
        Task ID and next step instructions
    """
    from uuid import uuid4
    from .domain.models import Task, TaskStatus

    orchestrator = get_orchestrator()

    # Get repo info from cached index
    try:
        index_data = orchestrator.get_cached_repository_index()
        repo_path = Path(index_data.get("repo_path", "."))
        branch = index_data.get("branch", "main")
        repository_summary = index_data.get("repository_summary", {})
    except ValueError:
        return {
            "error": "No repository index found",
            "hint": "Run index_repository first"
        }

    # Create task manually WITHOUT auto-generating clarifications
    task = Task(
        id=str(uuid4()),
        repo_path=repo_path,
        branch=branch,
        title=title or description[:50],
        summary="",
        client=client or "mcp",
        description=description,
        status=TaskStatus.CLARIFYING,
    )

    # Store minimal metadata
    task.metadata["repository_summary"] = repository_summary
    task.metadata["clarifications"] = []  # Empty - will be populated by prompt

    # Save the task
    orchestrator.store.upsert_task(task)

    return {
        "task_id": task.id,
        "status": "created",
        "title": task.title,
        "next_step": f"Invoke generate_clarifications_prompt with task_id='{task.id}' to create clarification questions",
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
# Clarification Tools (State Management + Submission)
# =============================================================================


@mcp.tool()
def submit_clarifications(task_id: str, clarifications_json: str) -> dict:
    """
    Submit generated clarification questions to the task.

    This tool stores the clarifications generated by Claude via the
    generate_clarifications_prompt.

    Args:
        task_id: UUID of the task
        clarifications_json: JSON string with clarifications array

    Returns:
        Confirmation and next step instructions
    """
    orchestrator = get_orchestrator()

    # Parse the clarifications
    try:
        data = json.loads(clarifications_json)
        clarifications = data.get("clarifications", [])
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    if not clarifications:
        return {"error": "No clarifications provided in JSON"}

    # Store clarifications in task metadata
    task = orchestrator._get_task(task_id)
    task.metadata["clarifications"] = [
        {
            "id": f"clarif-{i}",
            "question": c["question"],
            "rationale": c.get("rationale", ""),
            "default_answer": c.get("default_answer", ""),
            "status": "PENDING",
            "answer": None,
        }
        for i, c in enumerate(clarifications)
    ]
    task.status = TaskStatus.CLARIFYING
    orchestrator.store.upsert_task(task)

    return {
        "status": "clarifications_stored",
        "count": len(clarifications),
        "task_id": task_id,
        "next_step": "User should answer clarifications using answer_clarification tool",
    }


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
            "default_answer": c.get("default_answer"),
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
        "next_step": f"Invoke generate_plan_prompt with task_id='{task_id}'" if len(pending) == 0 else "answer_clarification",
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

    # Get the default answer and use it
    clarifications = orchestrator.list_clarifications(task_id)
    clarif = next((c for c in clarifications if c.get("id") == clarification_id), None)
    default_answer = clarif.get("default_answer", "") if clarif else ""

    orchestrator.update_clarification(
        task_id,
        clarification_id,
        answer=default_answer,
        status=ClarificationStatus.OVERRIDDEN,
    )

    clarifications = orchestrator.list_clarifications(task_id)
    pending = [c for c in clarifications if c.get("status") == "PENDING"]

    return {
        "status": "skipped",
        "used_default": default_answer,
        "pending_count": len(pending),
        "ready_for_plan": len(pending) == 0,
    }


# =============================================================================
# Planning Tools (State Management + Submission)
# =============================================================================


@mcp.tool()
def submit_plan(task_id: str, plan_json: str) -> dict:
    """
    Submit a generated implementation plan.

    This tool stores the plan generated by Claude via the generate_plan_prompt.

    Args:
        task_id: UUID of the task
        plan_json: JSON string with steps, risks, and dependencies

    Returns:
        Confirmation and next step instructions
    """
    orchestrator = get_orchestrator()

    # Parse the plan
    try:
        plan_data = json.loads(plan_json)
        steps = plan_data.get("steps", [])
        risks = plan_data.get("risks", [])
        dependencies = plan_data.get("dependencies", [])
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    if not steps:
        return {"error": "Plan must include at least one step"}

    # Store plan in task metadata
    task = orchestrator._get_task(task_id)
    task.metadata["plan"] = {
        "steps": steps,
        "risks": risks,
        "dependencies": dependencies,
    }
    task.metadata["plan_approved"] = False
    task.status = TaskStatus.PLANNING
    orchestrator.store.upsert_task(task)

    return {
        "status": "plan_stored",
        "steps_count": len(steps),
        "task_id": task_id,
        "next_step": "User should review the plan and call approve_plan when ready",
    }


@mcp.tool()
def get_plan(task_id: str) -> dict:
    """
    Get the implementation plan for a task.

    Args:
        task_id: UUID of the task

    Returns:
        Plan steps, risks, and dependencies
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {})

    steps = plan_data.get("steps", [])

    return {
        "task_id": task_id,
        "approved": task.metadata.get("plan_approved", False),
        "steps": [
            {
                "index": i + 1,
                "description": s.get("description") if isinstance(s, dict) else str(s),
                "target_files": s.get("target_files", []) if isinstance(s, dict) else [],
            }
            for i, s in enumerate(steps)
        ],
        "risks": plan_data.get("risks", []),
        "dependencies": plan_data.get("dependencies", []),
    }


@mcp.tool()
def approve_plan(task_id: str) -> dict:
    """
    Approve the implementation plan.

    After approval, you can start generating patches for each step.

    Args:
        task_id: UUID of the task

    Returns:
        Approval confirmation and next steps
    """
    orchestrator = get_orchestrator()

    task = orchestrator._get_task(task_id)
    task.metadata["plan_approved"] = True
    task.status = TaskStatus.IMPLEMENTING
    orchestrator.store.upsert_task(task)

    plan_data = task.metadata.get("plan", {})
    steps_count = len(plan_data.get("steps", []))

    return {
        "status": "approved",
        "task_id": task_id,
        "steps_count": steps_count,
        "next_step": f"Invoke generate_patch_prompt with task_id='{task_id}' and step_index=0 to start implementing",
    }


# =============================================================================
# Patch Generation & Review Tools (State Management + Submission)
# =============================================================================


@mcp.tool()
def submit_patch(task_id: str, step_index: int, diff: str, rationale: str) -> dict:
    """
    Submit a generated code patch.

    This tool stores the patch generated by Claude via the generate_patch_prompt.

    Args:
        task_id: UUID of the task
        step_index: Zero-based index of the step this patch implements
        diff: Unified diff showing the code changes
        rationale: Explanation of what changed and why

    Returns:
        Confirmation and next step instructions
    """
    orchestrator = get_orchestrator()

    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {})
    steps = plan_data.get("steps", [])

    if step_index >= len(steps):
        return {"error": f"Step index {step_index} out of range (plan has {len(steps)} steps)"}

    step = steps[step_index]
    step_desc = step.get("description") if isinstance(step, dict) else str(step)

    # Store patch
    from uuid import uuid4
    from .domain.models import Patch, PatchKind, PatchStatus

    patch = Patch(
        id=str(uuid4()),
        task_id=task_id,
        step_reference=step_desc,
        diff=diff,
        rationale=rationale,
        status=PatchStatus.PENDING,
        kind=PatchKind.IMPLEMENTATION,
        alternatives=[],
    )

    # Store in task metadata
    if "patches" not in task.metadata:
        task.metadata["patches"] = []
    task.metadata["patches"].append({
        "id": patch.id,
        "step_index": step_index,
        "step_reference": step_desc,
        "diff": diff,
        "rationale": rationale,
        "status": "PENDING",
    })
    task.status = TaskStatus.IMPLEMENTING
    orchestrator.store.upsert_task(task)

    # Check if there are more steps
    next_step_index = step_index + 1
    if next_step_index < len(steps):
        next_step_msg = f"Invoke generate_patch_prompt with task_id='{task_id}' and step_index={next_step_index}"
    else:
        next_step_msg = "All patches generated! Review and approve patches using list_patches and approve_patch"

    return {
        "status": "patch_stored",
        "patch_id": patch.id,
        "step_index": step_index,
        "task_id": task_id,
        "next_step": next_step_msg,
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
    task = orchestrator._get_task(task_id)
    patches = task.metadata.get("patches", [])

    return [
        {
            "id": p["id"],
            "step_index": p.get("step_index", -1),
            "step_reference": p["step_reference"],
            "status": p["status"],
            "diff_preview": p["diff"][:500] + ("..." if len(p["diff"]) > 500 else ""),
            "rationale": p["rationale"][:200] + ("..." if len(p["rationale"]) > 200 else ""),
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
    task = orchestrator._get_task(task_id)
    patches = task.metadata.get("patches", [])

    patch = next((p for p in patches if p["id"] == patch_id), None)

    if not patch:
        return {"error": f"Patch {patch_id} not found"}

    return {
        "id": patch["id"],
        "step_index": patch.get("step_index", -1),
        "step_reference": patch["step_reference"],
        "status": patch["status"],
        "diff": patch["diff"],
        "rationale": patch["rationale"],
    }


@mcp.tool()
def approve_patch(task_id: str, patch_id: str) -> dict:
    """
    Approve a patch (marks it as approved, but does NOT auto-apply to files).

    After approving all patches, the user can apply them manually or use sync_external_patch.

    Args:
        task_id: UUID of the task
        patch_id: UUID of the patch to approve

    Returns:
        Confirmation and remaining patch count
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    patches = task.metadata.get("patches", [])

    # Update patch status
    for p in patches:
        if p["id"] == patch_id:
            p["status"] = "APPROVED"
            break

    orchestrator.store.upsert_task(task)

    # Count remaining
    pending = [p for p in patches if p["status"] == "PENDING"]

    return {
        "patch_id": patch_id,
        "status": "approved",
        "pending_count": len(pending),
        "all_approved": len(pending) == 0,
        "next_step": "approve_patch" if pending else "Apply patches manually in your editor, then use sync_external_patch",
    }


@mcp.tool()
def sync_external_patch(task_id: str, patch_id: str | None = None, client: str | None = None, include_staged: bool = True) -> dict:
    """
    Sync/record code changes applied in an external editor (Cursor/Claude) back into spec-agent.

    Use this after you apply the patch changes in your editor,
    so the task dashboard reflects the real diff + files touched.

    Args:
        task_id: UUID of the task
        patch_id: Optional UUID of the patch to mark as applied
        client: Optional client label (e.g. "cursor" or "claude")
        include_staged: Include staged changes (git diff --cached) in the recorded diff

    Returns:
        Summary of synced diff and patch status update (if patch_id provided)
    """
    orchestrator = get_orchestrator()
    return orchestrator.sync_external_patch(
        task_id,
        patch_id=patch_id,
        client=client,
        include_staged=include_staged,
    )


@mcp.tool()
def reject_patch(task_id: str, patch_id: str) -> dict:
    """
    Reject a patch.

    This marks the patch as rejected. The user can then regenerate it
    by invoking generate_patch_prompt again for the same step.

    Args:
        task_id: UUID of the task
        patch_id: UUID of the patch to reject

    Returns:
        Confirmation
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    patches = task.metadata.get("patches", [])

    # Update patch status
    rejected_step_index = -1
    for p in patches:
        if p["id"] == patch_id:
            p["status"] = "REJECTED"
            rejected_step_index = p.get("step_index", -1)
            break

    orchestrator.store.upsert_task(task)

    return {
        "patch_id": patch_id,
        "status": "rejected",
        "message": f"Patch rejected. Invoke generate_patch_prompt again with step_index={rejected_step_index} to regenerate",
        "next_step": f"generate_patch_prompt(task_id='{task_id}', step_index={rejected_step_index})",
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
    clarifications = task.metadata.get("clarifications", [])
    pending_clarifications = [c for c in clarifications if c.get("status") == "PENDING"]

    plan_data = task.metadata.get("plan", {})
    plan_approved = task.metadata.get("plan_approved", False)

    patches = task.metadata.get("patches", [])
    pending_patches = [p for p in patches if p["status"] == "PENDING"]
    approved_patches = [p for p in patches if p["status"] == "APPROVED"]

    # Determine stage and next action
    if pending_clarifications:
        stage = "clarifying"
        next_action = "answer_clarification"
        hint = f"Answer {len(pending_clarifications)} clarification question(s)"
    elif not plan_data:
        stage = "ready_to_plan"
        next_action = f"generate_plan_prompt(task_id='{task_id}')"
        hint = "Invoke generate_plan_prompt to create implementation plan"
    elif not plan_approved:
        stage = "ready_to_approve"
        next_action = "approve_plan"
        hint = "Review the plan and approve it"
    elif not patches:
        stage = "ready_to_generate"
        next_action = f"generate_patch_prompt(task_id='{task_id}', step_index=0)"
        hint = "Invoke generate_patch_prompt to start creating patches"
    elif pending_patches:
        stage = "reviewing_patches"
        next_action = "approve_patch"
        hint = f"Review {len(pending_patches)} pending patch(es)"
    else:
        stage = "completed"
        next_action = None
        hint = "All patches approved! Apply them in your editor."

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
            "plan": {
                "exists": bool(plan_data),
                "approved": plan_approved,
                "steps_count": len(plan_data.get("steps", [])),
            },
            "patches": {
                "total": len(patches),
                "pending": len(pending_patches),
                "approved": len(approved_patches),
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
