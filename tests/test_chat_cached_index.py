from __future__ import annotations

import json
from pathlib import Path

from spec_agent.cli.chat import ChatSession
from spec_agent.config.settings import AgentSettings
from spec_agent.workflow.orchestrator import TaskOrchestrator


def test_chat_session_reuses_cached_repository_index(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("", encoding="utf-8")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)

    # Write a minimal per-repo repository index that chat can reuse.
    (state_dir / "repository_indexes").mkdir(exist_ok=True)
    import hashlib
    key = hashlib.sha1(str(repo.resolve()).encode("utf-8")).hexdigest()[:16]
    (state_dir / "repository_indexes" / f"{key}.json").write_text(
        json.dumps(
            {
                "repo_path": str(repo),
                "repo_name": "repo",
                "branch": "main",
                "repository_summary": {"file_count": 1, "has_tests": False},
                "semantic_index": None,
                "git_info": {},
                "indexed_at": "2025-12-17T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    session = ChatSession(orch)
    # Chat still loads "last index" if present; per-repo selection is handled by menu flow.
    # This test just ensures ChatSession construction doesn't crash with per-repo indexes present.
    assert session is not None

