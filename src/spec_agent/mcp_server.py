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
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from .domain.models import ClarificationStatus, PatchStatus, TaskStatus, utcnow
from .workflow.orchestrator import TaskOrchestrator

LOG = logging.getLogger(__name__)

# Initialize the MCP server
mcp = FastMCP(
    "spec-agent",
    instructions=(
        "Spec-Agent: AI-Driven Development Lifecycle engine. "
        "This system implements a rigorous, pillar-based workflow that transforms requirements into production-ready code. "
        "THE SIX PILLARS: "
        "1) DISCOVERY — index_repository + create_task + answer_clarifications: Understand the codebase and requirements. "
        "2) ARCHITECTURE — generate_plan_prompt + submit_plan + approve_plan: Design a concrete implementation plan. "
        "3) QUALITY GATE — generate_test_suggestions_prompt + submit_test_suggestions: Define test coverage BEFORE writing code. "
        "4) IMPLEMENTATION — generate_patch_prompt + submit_patch (for each step): Generate focused, incremental patches. "
        "5) VERIFICATION — approve_patch + sync_external_patch: Review changes and sync applied diffs. "
        "6) DELIVERY — complete_task: Finalize and close. "
        "IMPORTANT: Follow the pillars in order. Do not skip ahead. "
        "After submit_clarifications, you MUST call answer_clarifications_prompt then answer_all_clarifications with answers for EVERY question. "
        "Do not plan or write code until all clarifications are answered. "
        "After applying ALL changes in your editor, call sync_external_patch(task_id) "
        "WITHOUT patch_id to mark all patches as applied at once."
    ),
)

# Lazy-loaded orchestrator (created on first use)
_orchestrator: TaskOrchestrator | None = None

# In-memory cache for quick_repo_summary (keyed by repo_path string)
_quick_summary_cache: Dict[str, Tuple[float, dict]] = {}
_QUICK_SUMMARY_TTL = 300  # seconds


def get_orchestrator() -> TaskOrchestrator:
    """Get or create the TaskOrchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TaskOrchestrator()
    return _orchestrator


def _infer_repo_and_branch(
    repo_path: Optional[str] = None, branch: Optional[str] = None
) -> Tuple[Path, str]:
    """
    Infer repo_path and branch, defaulting to current working tree and git branch.

    - repo_path: explicit absolute path if provided, else cwd
    - branch: explicit if provided, else `git rev-parse --abbrev-ref HEAD`, else 'main'
    """
    repo = Path(repo_path or os.getcwd()).resolve()
    if branch:
        return repo, branch

    try:
        current_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(repo),
            text=True,
        ).strip()
    except Exception:
        current_branch = "main"

    return repo, current_branch


def _infer_step_scenario(description: str, notes: Optional[str] = None, *, has_tests: bool = True) -> list[str]:
    """
    Create a small, human-readable Given/When/Then scenario for a plan step.

    This mirrors the CLI/web scenario heuristic so test prompts align with
    plan scenarios.
    """
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

    lines = [
        f"Given: {given}",
        f"When:  {when}",
        f"Then:  {then}",
    ]
    if notes_clean:
        lines.append(f"And:   {notes_clean}")
    return lines


def _test_system_prompt() -> str:
    return (
        "You are executing the Quality Gate pillar of the AI-Driven Development Lifecycle. "
        "Design comprehensive automated tests that achieve full coverage of changed and impacted code paths. "
        "Map tests directly to plan scenarios as acceptance criteria. Cover happy paths, edge cases, "
        "negative cases, and error handling. Match the repository's existing test framework and conventions. "
        "If coverage gaps remain (external dependencies, nondeterminism, missing harness), call them out "
        "explicitly with proposed mitigations."
    )


# =============================================================================
# MCP PROMPTS - LLM Generation Tasks
# =============================================================================


@mcp.prompt()
def generate_clarifications_prompt(task_id: str):
    """
    Generate clarifying questions for a task.

    This prompt guides client LLM to generate 3-5 clarification questions
    to better understand the task requirements before planning.

    Args:
        task_id: UUID of the task

    Returns:
        Prompt messages for client LLM to generate clarifications
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
def answer_clarifications_prompt(task_id: str):
    """
    Answer all pending clarification questions for a task.
    
    This prompt guides client LLM to automatically answer clarification questions
    based on the task description and repository context, so planning can proceed.
    
    Args:
        task_id: UUID of the task
        
    Returns:
        Prompt messages for client LLM to answer clarifications
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    clarifications = orchestrator.list_clarifications(task_id)
    pending = [
        c
        for c in clarifications
        if (c.get("status") or ClarificationStatus.PENDING.value) == ClarificationStatus.PENDING.value
    ]
    
    if not pending:
        return [
            {
                "role": "user",
                "content": "All clarifications are already answered. You can proceed to generate_plan_prompt."
            }
        ]
    
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
- Hotspots: {', '.join([h['path'] for h in hotspots[:5]]) if hotspots else 'None'}
"""
    except ValueError:
        repo_context = ""
    
    # Format pending questions with IDs for the batch payload
    questions_text = "\n".join([
        f"Q{i+1}. clarification_id={c.get('id')!r}\n   {c.get('question', '')}\n   Default: {c.get('default_answer', 'N/A')}"
        for i, c in enumerate(pending)
    ])
    ids_list = [c.get("id") for c in pending]
    
    prompt_content = f"""You are helping to answer clarification questions for a software development task.

Task Details:
{repo_context}
Title: {task.title or "N/A"}
Description: {task.description}

Pending Clarification Questions (you MUST answer every one):
{questions_text}

Your job is to answer each question based on:
1. The task description and requirements
2. The repository structure and existing patterns
3. Common software engineering practices
4. The default answer provided (if reasonable)

IMPORTANT: You MUST answer ALL {len(pending)} pending questions in ONE call. Do not proceed to planning or code until all are answered.

Call answer_all_clarifications ONCE with a JSON payload containing an answer for every clarification_id:
- Required clarification_ids: {ids_list}

Example format:
answer_all_clarifications(task_id="{task_id}", answers_json='{{"answers": [{{"clarification_id": "{pending[0].get("id")}", "answer": "your answer for Q1"}}, {{"clarification_id": "...", "answer": "..."}}, ...]}}')

You must include exactly one entry per pending question. After answer_all_clarifications succeeds, you can proceed to generate_plan_prompt.
"""
    
    return [{"role": "user", "content": prompt_content}]


@mcp.prompt()
def generate_plan_prompt(task_id: str):
    """
    Generate an implementation plan for a task.

    This prompt guides client LLM to create a step-by-step implementation plan
    based on the task description and answered clarifications.

    Args:
        task_id: UUID of the task

    Returns:
        Prompt messages for client LLM to generate a plan
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    clarifications = orchestrator.list_clarifications(task_id)
    pending = [
        c
        for c in clarifications
        if (c.get("status") or ClarificationStatus.PENDING.value) == ClarificationStatus.PENDING.value
    ]

    if pending:
        pending_preview = "\n".join(f"- {c.get('question', '').strip()}" for c in pending[:5])
        return [
            {
                "role": "user",
                "content": (
                    "Clarification questions are still pending. You MUST answer them before generating a plan.\n"
                    f"Pending count: {len(pending)}\n"
                    f"IMPORTANT: Invoke answer_clarifications_prompt(task_id='{task_id}'), then call answer_all_clarifications with a JSON answer for EVERY pending question. Do not call generate_plan_prompt until all are answered.\n"
                    + (f"\n\nPending questions:\n{pending_preview}" if pending_preview else "")
                ),
            }
        ]

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

    prompt_content = f"""You are an expert software architect executing the Architecture pillar of the AI-Driven Development Lifecycle.

{repo_context}

Task: {task.title or "N/A"}
Description: {task.description}

Clarifications:
{clarifications_text or "None provided"}

Produce a precise implementation plan:
1. Break the task into 3-7 logical, incremental steps
2. For each step, identify the exact files to modify
3. Order steps by dependency — earlier steps should not depend on later ones
4. Identify risks and external dependencies

Each step MUST be independently implementable and verifiable.

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
def generate_test_suggestions_prompt(task_id: str):
    """
    Generate test suggestions based on the plan steps and scenarios.

    This prompt guides client LLM to propose tests that map to plan scenarios
    and aim for full coverage.
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {}) if isinstance(task.metadata, dict) else {}
    steps = plan_data.get("steps", []) if isinstance(plan_data, dict) else []

    if not steps:
        return [
            {
                "role": "user",
                "content": "No plan found. Generate and submit a plan first (generate_plan_prompt + submit_plan).",
            }
        ]

    repo_summary = task.metadata.get("repository_summary", {}) if isinstance(task.metadata, dict) else {}
    has_tests = bool(repo_summary.get("has_tests", False)) if isinstance(repo_summary, dict) else False
    languages = repo_summary.get("top_languages", []) if isinstance(repo_summary, dict) else []
    test_paths = repo_summary.get("test_paths_sample", []) if isinstance(repo_summary, dict) else []

    def _lang_list(raw) -> list[str]:
        langs: list[str] = []
        for item in raw or []:
            if isinstance(item, dict) and item.get("language"):
                langs.append(str(item.get("language")))
            else:
                langs.append(str(item))
        return langs

    step_lines: list[str] = []
    scenario_lines: list[str] = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            desc = (step.get("description") or "").strip() or "(missing step description)"
            targets = step.get("target_files") or []
            notes = step.get("notes")
        else:
            desc = str(step).strip() or "(missing step description)"
            targets = []
            notes = None

        step_lines.append(f"{idx}. {desc}")
        if targets:
            step_lines.append(f"   Targets: {', '.join(str(t) for t in targets)}")
        if notes:
            step_lines.append(f"   Notes: {notes}")

        scenario = _infer_step_scenario(desc, notes, has_tests=has_tests)
        if scenario:
            if scenario_lines:
                scenario_lines.append("")  # spacer
            scenario_lines.append(f"Step {idx}: {desc}")
            scenario_lines.extend([f"- {line}" for line in scenario])

    clarifications = task.metadata.get("clarifications") if isinstance(task.metadata, dict) else None
    clarifications_text = ""
    if isinstance(clarifications, list) and clarifications:
        clarifications_text = "\n".join(
            [
                f"Q: {c.get('question')}\nA: {c.get('answer', c.get('default_answer', 'N/A'))}"
                for c in clarifications
            ]
        )

    repo_context = f"""Repository Context:
- Primary languages: {', '.join(_lang_list(languages)[:3]) or 'unknown'}
- Has tests: {has_tests}
- Sample test files: {', '.join(str(p) for p in test_paths[:5]) or 'none detected'}
"""

    prompt_content = f"""Generate test suggestions that map 1:1 to the plan steps and scenarios.

Task Details:
Title: {task.title or "N/A"}
Description: {task.description}

{repo_context}

Plan Steps:
{chr(10).join(step_lines)}

Scenarios (Given / When / Then):
{chr(10).join(scenario_lines) if scenario_lines else "None"}

Clarifications:
{clarifications_text or "None provided"}

Requirements:
- Use the scenarios above as acceptance criteria.
- Aim for 100% code coverage of changed and impacted code paths.
- Include happy-path, edge-case, and negative tests per scenario.
- Identify existing test files to update when applicable.
- Provide skeleton code matching the repo's test framework.

Return a JSON object:
{{
  "tests": [
    {{
      "step_index": 1,
      "description": "Test scenario description",
      "expected_behavior": "What should be verified",
      "test_type": "UNIT or INTEGRATION",
      "related_files": ["file1.py", "file2.py"],
      "existing_tests_to_update": ["tests/test_file.py"],
      "skeleton_code": "Complete test skeleton code"
    }}
  ],
  "coverage_gaps": [
    "Describe any gaps that prevent full coverage and how to mitigate them"
  ]
}}

IMPORTANT: Return ONLY valid JSON (no markdown code blocks).

After generating the tests, call:
submit_test_suggestions(task_id="{task_id}", suggestions_json=<your_json>)
"""

    return [
        {"role": "system", "content": _test_system_prompt()},
        {"role": "user", "content": prompt_content},
    ]


@mcp.prompt()
def generate_patch_prompt(task_id: str, step_index: int):
    """
    Generate a code patch for a specific plan step.

    This prompt guides client LLM to create the actual code changes (as a unified diff)
    for implementing a single step from the plan.

    Args:
        task_id: UUID of the task
        step_index: Zero-based index of the step to implement

    Returns:
        Prompt messages for client LLM to generate a patch
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {})
    steps = plan_data.get("steps", [])

    if step_index >= len(steps):
        return [{"role": "user", "content": f"Error: Step index {step_index} out of range (plan has {len(steps)} steps)"}]

    step = steps[step_index]

    # Skip codegen steps (e.g. test steps on repos with no test framework)
    if isinstance(step, dict) and step.get("skip_codegen"):
        reason = step.get("skip_codegen_reason", "step does not require code generation")
        return [{"role": "user", "content": (
            f"Step #{step_index + 1} is marked as skip_codegen ({reason}). "
            f"No patch is needed. Proceed to the next step: "
            f"generate_patch_prompt(task_id='{task_id}', step_index={step_index + 1})"
        )}]

    step_desc = step.get("description") if isinstance(step, dict) else str(step)
    target_files = step.get("target_files", []) if isinstance(step, dict) else []
    reasoning = step.get("reasoning", "") if isinstance(step, dict) else ""

    prompt_content = f"""You are executing the Implementation pillar of the AI-Driven Development Lifecycle.

Task: {task.title or "N/A"}

Plan Step #{step_index + 1}:
Description: {step_desc}
Target Files: {', '.join(target_files) if target_files else 'Infer from description'}
Reasoning: {reasoning}

Implement this step precisely:
1. Read the target files
2. Make the minimum changes needed for this step
3. Produce a unified diff
4. Write a rationale explaining your decisions

Rules:
- Follow existing code style and patterns exactly
- Keep changes minimal and focused on THIS step only
- Do not add unnecessary comments, docstrings, or formatting changes

Then call: submit_patch(task_id="{task_id}", step_index={step_index}, diff="<unified_diff>", rationale="<explanation>")
"""

    return [{"role": "user", "content": prompt_content}]


# =============================================================================
# Repository Indexing Tools (State Management)
# =============================================================================


@mcp.tool()
def index_repository(repo_path: Optional[str] = None, branch: Optional[str] = None) -> dict:
    """
    Index a repository to prepare for spec-driven development.

    This analyzes the codebase structure, identifies key modules, languages,
    and creates a semantic index for intelligent planning.

    Args:
        repo_path: Absolute path to the repository root. If not provided,
            defaults to the current working directory.
        branch: Git branch to index. If not provided, defaults to the current
            git branch, or 'main' if it cannot be detected.

    Returns:
        Summary including repo name, languages, modules, and semantic analysis
    """
    orchestrator = get_orchestrator()
    repo, resolved_branch = _infer_repo_and_branch(repo_path, branch)
    result = orchestrator.index_repository(
        repo_path=repo,
        branch=resolved_branch,
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
def quick_repo_summary(repo_path: Optional[str] = None, branch: Optional[str] = None) -> dict:
    """
    Quickly summarize a repository without building the full semantic index.

    This is a fast, non-LLM pass used in MCP flows to give the chat agent and
    user a basic understanding of the repo before planning. It scans files,
    applies .gitignore, counts languages, detects tests and hotspots, and
    returns a lightweight summary that can be rendered directly in chat.

    Args:
        repo_path: Absolute path to the repository root. If not provided,
            defaults to the current working directory.
        branch: Git branch to label in the summary. If not provided, defaults
            to the current git branch, or 'main' if it cannot be detected.

    Returns:
        Repository summary with languages, modules, file counts, and hotspots.
    """
    orchestrator = get_orchestrator()
    repo, resolved_branch = _infer_repo_and_branch(repo_path, branch)

    cache_key = str(repo)
    now = time.monotonic()

    # Check in-memory cache (TTL-based)
    if cache_key in _quick_summary_cache:
        cached_at, cached_result = _quick_summary_cache[cache_key]
        if now - cached_at < _QUICK_SUMMARY_TTL:
            # Update branch in case it changed between calls
            cached_result["branch"] = resolved_branch
            return cached_result

    # Use the ContextIndexer directly with quick=True to skip Serena + import graph.
    summary = orchestrator.context_indexer.summarize_repository(
        repo_path=repo,
        include_serena_semantic_tree=False,
        quick=True,
    )

    if not isinstance(summary, dict):
        summary = {}

    result = {
        "repo_name": repo.name,
        "repo_path": str(repo),
        "branch": resolved_branch,
        "indexed_at": datetime.now().isoformat(),
        "file_count": summary.get("file_count"),
        "directory_count": summary.get("directory_count"),
        "languages": summary.get("top_languages", []),
        "modules": summary.get("top_modules", []),
        "has_tests": summary.get("has_tests", False),
        "test_paths_sample": summary.get("test_paths_sample", [])[:10],
        "hotspots": summary.get("hotspots", [])[:10],
    }

    # Store in cache
    _quick_summary_cache[cache_key] = (now, result)

    return result


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

    # Be defensive: repository_summary / semantic_index may be None depending on
    # how indexing was run or if it partially failed.
    summary = index_data.get("repository_summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    semantic = index_data.get("semantic_index") or {}
    if not isinstance(semantic, dict):
        semantic = {}

    repo_semantic = semantic.get("repository") or {}
    domains = semantic.get("domains") or []
    if not isinstance(domains, list):
        domains = []

    return {
        "repo_name": index_data.get("repo_name"),
        "repo_path": index_data.get("repo_path"),
        "branch": index_data.get("branch"),
        "languages": summary.get("top_languages", []),
        "modules": summary.get("top_modules", []),
        "hotspots": (summary.get("hotspots") or [])[:5],
        "architecture": repo_semantic.get("architectureStyle"),
        "frameworks": repo_semantic.get("frameworks", []),
        "domains": [d.get("name") for d in domains[:5] if isinstance(d, dict)],
    }


def _repo_summary_from_index(index_data: dict) -> dict:
    """Build a compact repo summary for inclusion in tool responses (e.g. run_ticket_workflow, create_task)."""
    summary = index_data.get("repository_summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    semantic = index_data.get("semantic_index") or {}
    if not isinstance(semantic, dict):
        semantic = {}
    repo_semantic = semantic.get("repository") or {}
    domains = semantic.get("domains") or []
    if not isinstance(domains, list):
        domains = []
    return {
        "repo_name": index_data.get("repo_name"),
        "repo_path": index_data.get("repo_path"),
        "branch": index_data.get("branch"),
        "languages": summary.get("top_languages", []),
        "modules": summary.get("top_modules", []),
        "hotspots": (summary.get("hotspots") or [])[:5],
        "architecture": repo_semantic.get("architectureStyle"),
        "frameworks": repo_semantic.get("frameworks", []),
        "domains": [d.get("name") for d in domains[:5] if isinstance(d, dict)],
    }


# =============================================================================
# Task Management Tools (State Management)
# =============================================================================


@mcp.tool()
def run_ticket_workflow(
    description: Optional[str] = None,
    title: Optional[str] = None,
    client: Optional[str] = "cursor",
    repo_path: Optional[str] = None,
    branch: Optional[str] = None,
    generate_patches: bool = True,
) -> dict:
    """
    End-to-end ticket handler for MCP clients.

    Given a natural-language ticket, this tool:
    1) Indexes the current working tree (if needed)
    2) Creates a task bound to repo/branch
    3) Surfaces clarifications immediately
    4) Optionally auto-generates plan and patches (fast defaults)
    """
    if not (description and description.strip()):
        return {
            "error": "description is required",
            "hint": "Call run_ticket_workflow with a non-empty description (e.g. the user request).",
        }

    orchestrator = get_orchestrator()

    repo, resolved_branch = _infer_repo_and_branch(repo_path, branch)

    index_data = orchestrator.index_repository(
        repo_path=repo,
        branch=resolved_branch,
        include_serena_semantic_tree=False,
    )

    task = orchestrator.create_task_from_index(
        description=description,
        title=title,
        summary=None,
        client=(client or "cursor"),
        index_data=index_data,
    )

    result: dict[str, Any] = {
        "task_id": task.id,
        "short_id": task.id[:8],
        "repo_path": str(task.repo_path),
        "branch": task.branch,
        "client": task.client,
        "status": task.status.value,
        "next_steps": [],
        "repository_summary": _repo_summary_from_index(index_data),
    }

    clarifications = task.metadata.get("clarifications") or []
    if clarifications:
        result["clarifications"] = [
            {
                "id": c.get("id"),
                "question": c.get("question"),
                "status": c.get("status", "PENDING"),
                "default_answer": c.get("default_answer"),
            }
            for c in clarifications
            if isinstance(c, dict)
        ]
        result["next_steps"].append("answer_clarifications")
        return result

    if not generate_patches:
        result["next_steps"].append("plan")
        return result

    plan_payload = orchestrator.generate_plan(task.id, skip_rationale_enhancement=True)
    plan_data = plan_payload.get("plan") or {}
    pending_specs = plan_payload.get("pending_specs") or []
    result["plan"] = {
        "steps": plan_data.get("steps", []),
        "risks": plan_data.get("risks", []),
        "refactors": plan_data.get("refactors", []),
        "pending_specs": pending_specs,
    }

    if pending_specs:
        result["next_steps"].append("resolve_specs")
        return result

    orchestrator.approve_plan(task.id)
    patches_payload = orchestrator.generate_patches(task.id, skip_rationale_enhancement=True)
    result["patches"] = {
        "patch_count": patches_payload.get("patch_count", 0),
        "test_count": patches_payload.get("test_count", 0),
        "tests_skipped_reason": patches_payload.get("tests_skipped_reason"),
    }
    result["next_steps"].append("review_patches")
    return result


@mcp.tool()
def create_task(description: Optional[str] = None, title: Optional[str] = None, client: Optional[str] = None) -> dict:
    """
    Create a new development task using the same workflow as the CLI.

    This delegates to the TaskOrchestrator so that:
    - Repository summary and git metadata are populated from the cached index
    - Clarifying questions are auto-generated and stored on the task
    - The web dashboard (RiSpec) immediately shows the same clarifying questions
      and context that the CLI workflow would.

    Args:
        description: Detailed description of what changes you want to make
        title: Optional short title for the task
        client: Optional editor/chat client driving this task (e.g. cursor, claude)

    Returns:
        Task ID and next step instructions
    """
    orchestrator = get_orchestrator()

    if not (description and description.strip()):
        return {
            "error": "description is required",
            "hint": "Call create_task with a non-empty description describing the changes you want.",
        }

    # Ensure we have a repository index (same precondition as CLI create-task-from-index)
    try:
        index_data = orchestrator.get_cached_repository_index()
    except ValueError:
        return {
            "error": "No repository index found",
            "hint": "Run index_repository first",
        }

    # Use the orchestrator's create_task_from_index so MCP / Cursor tasks follow
    # the exact same lifecycle as CLI-created tasks.
    task = orchestrator.create_task_from_index(
        description=description,
        title=title,
        client=client or "mcp",
        index_data=index_data,
    )

    return {
        "task_id": task.id,
        "status": "created",
        "title": task.title,
        "summary": task.summary,
        "repository_summary": _repo_summary_from_index(index_data),
        # Clarifications are already generated and stored; next step is to review/answer them.
        "next_step": "Review and answer clarifications using get_clarifications and answer_clarification",
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

    This tool stores the clarifications generated by client LLM via the
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
        "next_step": "Invoke answer_clarifications_prompt, then call answer_all_clarifications with answers for EVERY question (all at once). Do not proceed to planning until all are answered.",
        "pending_count": len(clarifications),
        "pending_ids": [f"clarif-{i}" for i in range(len(clarifications))],
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
def answer_all_clarifications(task_id: str, answers_json: str) -> dict:
    """
    Answer ALL pending clarification questions in one call.

    You MUST provide an answer for every pending question. Planning cannot proceed
    until all are answered. Use get_clarifications first to see pending IDs.

    Args:
        task_id: UUID of the task
        answers_json: JSON string with format {"answers": [{"clarification_id": "clarif-0", "answer": "your answer"}, ...]}

    Returns:
        Status and next step; error if any pending question is missing an answer
    """
    orchestrator = get_orchestrator()
    pending = [
        c
        for c in orchestrator.list_clarifications(task_id)
        if (c.get("status") or ClarificationStatus.PENDING.value) == ClarificationStatus.PENDING.value
    ]
    if not pending:
        return {
            "status": "already_answered",
            "task_id": task_id,
            "next_step": f"Invoke generate_plan_prompt with task_id='{task_id}'",
        }

    try:
        data = json.loads(answers_json)
        answers = data.get("answers", [])
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "pending_count": len(pending)}

    pending_ids = {c.get("id") for c in pending}
    provided_ids = {a.get("clarification_id") for a in answers if a.get("clarification_id")}
    missing = pending_ids - provided_ids
    if missing:
        return {
            "error": f"Missing answers for clarification(s): {sorted(missing)}. You must answer every pending question.",
            "pending_ids": sorted(pending_ids),
            "pending_count": len(pending),
        }

    for a in answers:
        cid = a.get("clarification_id")
        if cid not in pending_ids:
            continue
        ans = (a.get("answer") or "").strip()
        orchestrator.update_clarification(
            task_id,
            cid,
            answer=ans or "(no answer)",
            status=ClarificationStatus.ANSWERED,
        )

    return {
        "status": "all_answered",
        "task_id": task_id,
        "answered_count": len(pending),
        "ready_for_plan": True,
        "next_step": f"Invoke generate_plan_prompt with task_id='{task_id}'",
    }


@mcp.tool()
def answer_clarification(task_id: str, clarification_id: str, answer: str) -> dict:
    """
    Answer a single clarification question.

    Prefer answer_all_clarifications when multiple questions are pending, so all
    are answered in one step before planning.

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
        "next_step": (
            f"Invoke generate_plan_prompt with task_id='{task_id}'"
            if len(pending) == 0
            else f"Call answer_all_clarifications with task_id='{task_id}' and answers for ALL remaining {len(pending)} question(s) before planning"
        ),
    }


# =============================================================================
# Planning Tools (State Management + Submission)
# =============================================================================


@mcp.tool()
def submit_plan(task_id: str, plan_json: str) -> dict:
    """
    Submit a generated implementation plan.

    This tool stores the plan generated by client LLM via the generate_plan_prompt.

    Args:
        task_id: UUID of the task
        plan_json: JSON string with steps, risks, and dependencies

    Returns:
        Confirmation and next step instructions
    """
    orchestrator = get_orchestrator()

    if orchestrator.has_pending_clarifications(task_id):
        pending = orchestrator.list_clarifications(task_id)
        pending_count = len([
            c for c in pending if (c.get("status") or ClarificationStatus.PENDING.value) == ClarificationStatus.PENDING.value
        ])
        return {
            "error": "Clarifications are still pending. You must answer ALL of them before submitting a plan.",
            "pending_count": pending_count,
            "next_step": "Call answer_all_clarifications with answers for every pending question, then re-run generate_plan_prompt and submit_plan",
        }

    # Parse the plan
    try:
        plan_data = json.loads(plan_json)
        steps = plan_data.get("steps", [])
        risks = plan_data.get("risks", [])
        dependencies = plan_data.get("dependencies", [])
        refactors = plan_data.get("refactors", [])
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}

    if not steps:
        return {"error": "Plan must include at least one step"}

    # Store plan in task metadata (keep both legacy `plan` and dashboard-friendly `plan_preview`)
    task = orchestrator._get_task(task_id)
    plan_preview = {
        "id": plan_data.get("id") or str(uuid4()),
        "steps": steps,
        "risks": risks,
        "refactors": refactors,
    }
    task.metadata["plan"] = {
        "steps": steps,
        "risks": risks,
        "dependencies": dependencies,
        "refactors": refactors,
    }
    task.metadata["plan_preview"] = plan_preview
    task.metadata["plan_stage"] = "FINAL"
    task.metadata.pop("plan_markdown_path", None)
    task.metadata["plan_approved"] = False
    task.status = TaskStatus.PLANNING
    task.touch()
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
def get_plan_overview(task_id: str) -> dict:
    """
    Return a dashboard-friendly snapshot of the plan for chat clients.

    This surfaces the same details users see in the CLI/web plan review
    so they can confirm the execution plan before or while generating patches.
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    plan_preview = task.metadata.get("plan_preview", {}) if isinstance(task.metadata, dict) else {}
    plan_data = task.metadata.get("plan", {}) if isinstance(task.metadata, dict) else {}

    steps = plan_preview.get("steps") if isinstance(plan_preview, dict) else None
    if not steps:
        steps = plan_data.get("steps", []) if isinstance(plan_data, dict) else []

    risks = plan_preview.get("risks") if isinstance(plan_preview, dict) else None
    if risks is None:
        risks = plan_data.get("risks", []) if isinstance(plan_data, dict) else []

    refactors = plan_preview.get("refactors") if isinstance(plan_preview, dict) else None
    if refactors is None:
        refactors = plan_data.get("refactors", []) if isinstance(plan_data, dict) else []

    return {
        "task_id": task_id,
        "approved": bool(task.metadata.get("plan_approved")) if isinstance(task.metadata, dict) else False,
        "steps": [
            {
                "index": i + 1,
                "description": s.get("description") if isinstance(s, dict) else str(s),
                "target_files": s.get("target_files", []) if isinstance(s, dict) else [],
                "notes": s.get("notes") if isinstance(s, dict) else None,
            }
            for i, s in enumerate(steps)
        ],
        "risks": risks or [],
        "refactors": refactors or [],
        "plan_markdown_path": task.metadata.get("plan_markdown_path") if isinstance(task.metadata, dict) else None,
    }


@mcp.tool()
def get_test_system_prompt() -> dict:
    """
    Return the raw system prompt used for test suggestions.
    """
    return {"system_prompt": _test_system_prompt()}


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
    plan_data = task.metadata.get("plan", {}) if isinstance(task.metadata, dict) else {}
    if not isinstance(plan_data, dict):
        plan_data = {}
    refactors = plan_data.get("refactors", []) if isinstance(plan_data, dict) else []

    # Ensure dashboard-friendly plan preview is present for exports/UI
    plan_preview = task.metadata.get("plan_preview") if isinstance(task.metadata, dict) else {}
    if not isinstance(plan_preview, dict) or not plan_preview.get("steps"):
        task.metadata["plan_preview"] = {
            "id": plan_data.get("id") or str(uuid4()),
            "steps": plan_data.get("steps", []),
            "risks": plan_data.get("risks", []),
            "refactors": refactors,
        }

    task.metadata["plan_approved"] = True
    task.status = TaskStatus.IMPLEMENTING
    task.touch()
    task.metadata["plan_approved_at"] = task.updated_at.isoformat()
    orchestrator.store.upsert_task(task)

    # Auto-apply patches for skip_codegen steps (e.g. test steps on repos with no test framework)
    steps = plan_data.get("steps", [])
    for idx, step in enumerate(steps):
        if isinstance(step, dict) and step.get("skip_codegen"):
            patch = orchestrator.add_patch_for_step(
                task_id=task_id,
                step_index=idx,
                diff="",
                rationale=f"Auto-passed: {step.get('skip_codegen_reason', 'step does not require code generation')}",
            )
            # Mark immediately as APPLIED
            patches = orchestrator._load_patch_queue(task)
            for p in patches:
                if p.id == patch.id:
                    p.status = PatchStatus.APPLIED
                    p.applied_via = "auto-skip"
                    p.applied_at = utcnow()
            orchestrator._persist_patch_queue(task, patches)

    plan_markdown_path: str | None = None
    try:
        exported = orchestrator.export_approved_plan_markdown(task_id)
        plan_markdown_path = str(exported)
    except Exception as exc:  # pragma: no cover - best-effort export
        LOG.warning("Could not export plan markdown for %s: %s", task_id, exc)

    # Reload task to surface any metadata updates from export
    task = orchestrator._get_task(task_id)
    plan_preview = task.metadata.get("plan_preview", {}) if isinstance(task.metadata, dict) else {}
    steps = plan_preview.get("steps", []) if isinstance(plan_preview, dict) else []
    steps_count = len(steps)

    return {
        "status": "approved",
        "task_id": task_id,
        "steps_count": steps_count,
        "plan_markdown_path": plan_markdown_path,
        "next_step": (
            f"RECOMMENDED: Invoke generate_test_suggestions_prompt(task_id='{task_id}') to generate test cases "
            f"based on the plan BEFORE writing patches. Then call submit_test_suggestions with the generated JSON. "
            f"After tests are defined, invoke generate_patch_prompt(task_id='{task_id}', step_index=0) to start implementing."
        ),
        "alternative_next_step": f"Or skip tests and go directly to generate_patch_prompt(task_id='{task_id}', step_index=0)",
    }


# =============================================================================
# Patch Generation & Review Tools (State Management + Submission)
# =============================================================================


@mcp.tool()
def submit_patch(task_id: str, step_index: int, diff: str, rationale: str) -> dict:
    """
    Submit a generated code patch.

    This tool stores the patch generated by client LLM via the generate_patch_prompt.

    Args:
        task_id: UUID of the task
        step_index: Zero-based index of the step this patch implements
        diff: Unified diff showing the code changes
        rationale: Explanation of what changed and why

    Returns:
        Confirmation and next step instructions
    """
    orchestrator = get_orchestrator()

    # Check if this step is marked as skip_codegen (already auto-applied)
    task = orchestrator._get_task(task_id)
    plan_data = task.metadata.get("plan", {})
    steps = plan_data.get("steps", [])
    step = steps[step_index] if step_index < len(steps) else None
    if isinstance(step, dict) and step.get("skip_codegen"):
        return {
            "status": "skipped",
            "reason": step.get("skip_codegen_reason", "step does not require code generation"),
            "next_step": "Proceed to next step or approve patches",
        }

    # Append the patch into the orchestrator's patch queue so CLI and MCP stay aligned.
    try:
        patch = orchestrator.add_patch_for_step(
            task_id=task_id,
            step_index=step_index,
            diff=diff,
            rationale=rationale,
        )
    except Exception as exc:
        return {"error": str(exc)}

    # Check if there are more steps remaining in the plan
    plan_data = task.metadata.get("plan", {})
    steps = plan_data.get("steps", [])
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
def submit_test_suggestions(task_id: str, suggestions_json: str) -> dict:
    """
    Submit generated test suggestions for a task.

    Stores the full suggestions payload and a compact description list.
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    try:
        data = json.loads(suggestions_json)
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON: {exc}"}

    tests = data.get("tests") or data.get("suggestions") or []
    if not isinstance(tests, list) or not tests:
        return {"error": "No tests provided in JSON (expected tests array)."}

    descriptions: list[str] = []
    for test in tests:
        if isinstance(test, dict) and test.get("description"):
            descriptions.append(str(test.get("description")))
        else:
            descriptions.append(str(test))

    if not isinstance(task.metadata, dict):
        task.metadata = {}

    task.metadata["test_suggestions_raw"] = tests
    task.metadata["test_suggestions"] = descriptions
    task.metadata["test_suggestions_coverage_gaps"] = data.get("coverage_gaps", [])
    task.touch()
    orchestrator.store.upsert_task(task)

    return {
        "status": "test_suggestions_stored",
        "task_id": task_id,
        "test_count": len(descriptions),
        "coverage_gaps": data.get("coverage_gaps", []),
        "next_step": f"Tests stored! Use get_test_suggestions(task_id='{task_id}') to review, then proceed to generate_patch_prompt for implementation.",
    }


@mcp.tool()
def get_test_suggestions(task_id: str) -> dict:
    """
    Get the test suggestions for a task.

    Returns the full test suggestions including skeleton code, coverage gaps,
    and mapping to plan steps. Use this after generate_test_suggestions_prompt
    and submit_test_suggestions to review the proposed tests.

    Args:
        task_id: UUID of the task

    Returns:
        Test suggestions with descriptions, skeleton code, and coverage gaps
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    if not isinstance(task.metadata, dict):
        return {"error": "No metadata found for task", "task_id": task_id}

    raw_tests = task.metadata.get("test_suggestions_raw", [])
    descriptions = task.metadata.get("test_suggestions", [])
    coverage_gaps = task.metadata.get("test_suggestions_coverage_gaps", [])

    if not raw_tests and not descriptions:
        return {
            "error": "No test suggestions found for this task",
            "task_id": task_id,
            "hint": f"Generate tests first by invoking generate_test_suggestions_prompt(task_id='{task_id}'), then call submit_test_suggestions with the generated JSON.",
        }

    # Get plan context for cross-reference
    plan_data = task.metadata.get("plan", {}) if isinstance(task.metadata, dict) else {}
    plan_steps = plan_data.get("steps", []) if isinstance(plan_data, dict) else []

    return {
        "task_id": task_id,
        "test_count": len(raw_tests) if raw_tests else len(descriptions),
        "tests": [
            {
                "index": i + 1,
                "step_index": t.get("step_index") if isinstance(t, dict) else None,
                "description": t.get("description") if isinstance(t, dict) else str(t),
                "expected_behavior": t.get("expected_behavior") if isinstance(t, dict) else None,
                "test_type": t.get("test_type") if isinstance(t, dict) else None,
                "related_files": t.get("related_files", []) if isinstance(t, dict) else [],
                "existing_tests_to_update": t.get("existing_tests_to_update", []) if isinstance(t, dict) else [],
                "skeleton_code": t.get("skeleton_code") if isinstance(t, dict) else None,
            }
            for i, t in enumerate(raw_tests if raw_tests else descriptions)
        ],
        "coverage_gaps": coverage_gaps,
        "plan_steps_count": len(plan_steps),
        "next_step": f"Review the test suggestions above, then proceed to generate_patch_prompt(task_id='{task_id}', step_index=0) to implement the changes.",
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
            "step_index": idx,
            "step_reference": p.step_reference,
            "status": p.status.value if hasattr(p.status, "value") else str(p.status),
            "diff_preview": p.diff[:500] + ("..." if len(p.diff) > 500 else ""),
            "rationale": p.rationale[:200] + ("..." if len(p.rationale) > 200 else ""),
        }
        for idx, p in enumerate(patches)
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
        "step_index": next((idx for idx, p in enumerate(patches) if p.id == patch_id), -1),
        "step_reference": patch.step_reference,
        "status": patch.status.value if hasattr(patch.status, "value") else str(patch.status),
        "diff": patch.diff,
        "rationale": patch.rationale,
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
    try:
        patch = orchestrator.approve_patch(task_id, patch_id)
    except Exception as exc:
        return {"error": str(exc)}

    patches = orchestrator.list_patches(task_id)
    pending = [p for p in patches if p.status == PatchStatus.PENDING]
    approved = [p for p in patches if p.status == PatchStatus.APPLIED]

    if pending:
        next_step = f"approve_patch for remaining {len(pending)} patch(es)"
    else:
        # All patches approved - need to apply and sync
        next_step = (
            f"IMPORTANT: Apply the approved patch(es) in your editor, then call "
            f"sync_external_patch(task_id='{task_id}') to sync all changes. "
            f"The web UI will not show applied changes until you call sync_external_patch."
        )

    return {
        "patch_id": patch_id,
        "status": "approved",
        "pending_count": len(pending),
        "all_approved": len(pending) == 0,
        "next_step": next_step,
        "applied_via": getattr(patch, "applied_via", None),
        "sync_required": len(pending) == 0,
        "sync_instruction": (
            f"After applying changes in your editor, call: sync_external_patch(task_id='{task_id}')"
            if len(pending) == 0
            else None
        ),
    }


@mcp.tool()
def sync_external_patch(task_id: str, patch_id: str | None = None, client: str | None = None, include_staged: bool = True) -> dict:
    """
    Sync/record code changes applied in an external editor (Cursor/Claude) back into spec-agent.

    Use this after you apply the patch changes in your editor,
    so the task dashboard reflects the real diff + files touched.

    When all patches are synced, the task is automatically marked as COMPLETED.

    Args:
        task_id: UUID of the task
        patch_id: Optional UUID of the patch to mark as applied
        client: Optional client label (e.g. "cursor" or "claude")
        include_staged: Include staged changes (git diff --cached) in the recorded diff

    Returns:
        Summary of synced diff and patch status update (if patch_id provided)
    """
    orchestrator = get_orchestrator()
    result = orchestrator.sync_external_patch(
        task_id,
        patch_id=patch_id,
        client=client,
        include_staged=include_staged,
    )

    # Reload task to get current status (may have been auto-completed)
    task = orchestrator._get_task(task_id)
    result["task_status"] = task.status.value
    result["task_completed"] = task.status == TaskStatus.COMPLETED

    # Check remaining patches
    try:
        patches = orchestrator.list_patches(task_id)
        pending_patches = [p for p in patches if p.status == PatchStatus.PENDING]
        result["pending_patches_count"] = len(pending_patches)
    except Exception:
        pending_patches = []
        result["pending_patches_count"] = 0

    # Enhance the response with clearer messaging
    if result.get("task_completed"):
        result["message"] = (
            f"Task completed! All patches have been synced and applied. "
            f"The web dashboard now shows the task as COMPLETED."
        )
        result["next_step"] = "Task is complete! You can view the final state in the web dashboard."
    elif result.get("has_diff"):
        if patch_id:
            result["message"] = (
                f"Successfully synced changes! Patch {patch_id[:8]} marked as APPLIED. "
                f"The web dashboard will now show the actual diff and files touched."
            )
        else:
            result["message"] = (
                f"Successfully synced changes! The web dashboard will now show the actual diff and files touched."
            )
        if pending_patches:
            result["next_step"] = (
                f"{len(pending_patches)} patch(es) remaining. Apply the next patch and call sync_external_patch again."
            )
        else:
            result["next_step"] = "All patches synced! Check the web dashboard for final status."
    else:
        result["message"] = (
            "No changes detected in git diff. If you applied changes, make sure they are saved and visible in git status."
        )
        result["next_step"] = "No changes detected. Make sure you've saved your file changes and they appear in git diff."

    return result


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
    patches = orchestrator.list_patches(task_id)
    patch = next((p for p in patches if p.id == patch_id), None)
    if not patch:
        return {"error": f"Patch {patch_id} not found"}

    rejected_step_index = next((idx for idx, p in enumerate(patches) if p.id == patch_id), -1)

    orchestrator.reject_patch(task_id, patch_id)

    return {
        "patch_id": patch_id,
        "status": "rejected",
        "message": f"Patch rejected. Invoke generate_patch_prompt again with step_index={rejected_step_index} to regenerate",
        "next_step": f"generate_patch_prompt(task_id='{task_id}', step_index={rejected_step_index})",
    }


@mcp.tool()
def complete_task(task_id: str, sync_changes: bool = True, force: bool = False) -> dict:
    """
    Mark a task as completed.

    Use this after all changes have been applied and synced. This explicitly
    marks the task as COMPLETED in the web dashboard.

    The plan must be approved before a task can be completed. Use force=True
    to override this check (not recommended).

    If sync_changes is True (default), this will first capture any uncommitted
    git changes before marking the task complete.

    Args:
        task_id: UUID of the task
        sync_changes: If True, capture current git diff before completing
        force: If True, complete even if plan is not approved (not recommended)

    Returns:
        Confirmation that the task is marked as COMPLETED
    """
    orchestrator = get_orchestrator()
    task = orchestrator._get_task(task_id)

    # Check if plan is approved
    plan_approved = task.metadata.get("plan_approved", False) if isinstance(task.metadata, dict) else False
    if not plan_approved and not force:
        return {
            "error": "Cannot complete task: plan is not approved",
            "task_id": task_id,
            "plan_approved": False,
            "hint": f"Approve the plan first using approve_plan(task_id='{task_id}'), or use force=True to override.",
        }

    # Optionally sync any remaining changes first
    sync_result = None
    if sync_changes:
        try:
            sync_result = orchestrator.sync_external_patch(
                task_id,
                patch_id=None,
                client=task.client or "mcp",
                include_staged=True,
            )
        except Exception as exc:
            LOG.warning("Could not sync changes before completing task %s: %s", task_id, exc)

    # Mark task as completed
    task.status = TaskStatus.COMPLETED
    task.touch()
    orchestrator.store.upsert_task(task)

    return {
        "task_id": task_id,
        "status": "COMPLETED",
        "message": "Task marked as COMPLETED. The web dashboard now shows the task as finished.",
        "plan_approved": plan_approved,
        "sync_result": {
            "has_diff": sync_result.get("has_diff") if sync_result else False,
            "files": sync_result.get("files", []) if sync_result else [],
        } if sync_changes else None,
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

    # Test suggestions state
    test_suggestions = task.metadata.get("test_suggestions", [])
    test_suggestions_raw = task.metadata.get("test_suggestions_raw", [])
    has_tests = bool(test_suggestions or test_suggestions_raw)
    test_count = len(test_suggestions_raw) if test_suggestions_raw else len(test_suggestions)

    # Prefer the orchestrator's patch queue so CLI and MCP stay in sync
    try:
        patch_queue = orchestrator.list_patches(task_id)
    except Exception:
        patch_queue = []

    pending_patches = [p for p in patch_queue if p.status == PatchStatus.PENDING]
    approved_patches = [p for p in patch_queue if p.status == PatchStatus.APPLIED]
    # Check for approved patches that haven't been synced (no applied_diff)
    approved_unsynced = [p for p in patch_queue if p.status == PatchStatus.APPLIED and not p.applied_diff]

    # Determine stage and next action
    if task.status == TaskStatus.CLARIFYING and not clarifications:
        stage = "clarifying"
        next_action = f"generate_clarifications_prompt(task_id='{task_id}')"
        hint = "Invoke generate_clarifications_prompt to create clarification questions"
    elif pending_clarifications:
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
    elif plan_approved and not has_tests and not patch_queue:
        # Plan approved but no tests generated yet - recommend test generation
        stage = "ready_to_test"
        next_action = f"generate_test_suggestions_prompt(task_id='{task_id}')"
        hint = (
            "RECOMMENDED: Generate test cases before implementing. "
            f"Invoke generate_test_suggestions_prompt(task_id='{task_id}'), then submit_test_suggestions. "
            "Or skip tests and go directly to generate_patch_prompt."
        )
    elif not patch_queue:
        stage = "ready_to_generate"
        next_action = f"generate_patch_prompt(task_id='{task_id}', step_index=0)"
        hint = f"Invoke generate_patch_prompt to start creating patches{' (tests already generated)' if has_tests else ''}"
    elif pending_patches:
        stage = "reviewing_patches"
        next_action = "approve_patch"
        hint = f"Review {len(pending_patches)} pending patch(es)"
    elif approved_unsynced:
        stage = "syncing_patches"
        next_action = f"sync_external_patch(task_id='{task_id}', patch_id='{approved_unsynced[0].id}')"
        hint = f"IMPORTANT: {len(approved_unsynced)} approved patch(es) need syncing. Apply changes in your editor, then call sync_external_patch to update the web dashboard."
    elif task.status == TaskStatus.COMPLETED:
        stage = "completed"
        next_action = None
        hint = "Task COMPLETED! All patches have been applied and synced. The web dashboard shows the final state."
    else:
        stage = "completed"
        next_action = None
        hint = "All patches approved and synced! Task complete."

    out: dict[str, Any] = {
        "task_id": task_id,
        "title": task.title,
        "stage": stage,
        "status": task.status.value,
        "task_completed": task.status == TaskStatus.COMPLETED,
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
            "tests": {
                "generated": has_tests,
                "count": test_count,
            },
            "patches": {
                "total": len(patch_queue),
                "pending": len(pending_patches),
                "approved": len(approved_patches),
                "unsynced": len(approved_unsynced),
            },
        },
    }
    try:
        index_data = orchestrator.get_cached_repository_index()
        out["repository_summary"] = _repo_summary_from_index(index_data)
    except ValueError:
        pass
    return out


# =============================================================================
# Entry Point
# =============================================================================


def run():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    run()
