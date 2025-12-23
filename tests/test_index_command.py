"""
Tests for the new index and start commands.
"""
from pathlib import Path

from spec_agent.workflow.orchestrator import TaskOrchestrator
from spec_agent.persistence.store import JsonStore


def test_index_repository_creates_index(tmp_path: Path) -> None:
    """Test that index_repository creates and saves an index."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    (repo_path / ".gitignore").write_text("", encoding="utf-8")

    # Create a simple file
    (repo_path / "test.py").write_text("print('hello')\n", encoding="utf-8")

    # Create orchestrator with custom state dir
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    from spec_agent.config.settings import AgentSettings

    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)

    # Index the repository (git metadata may be missing; index should still succeed)
    index_data = orch.index_repository(repo_path=repo_path, branch="main")

    # Verify index was created
    assert index_data is not None
    assert "repository_summary" in index_data
    assert "repo_path" in index_data
    assert "branch" in index_data
    assert index_data["branch"] == "main"
    assert Path(index_data["repo_path"]).resolve() == repo_path.resolve()

    # Verify index was saved to disk
    store = JsonStore(state_dir)
    loaded_index = store.load_repository_index()
    assert Path(loaded_index["repo_path"]).resolve() == repo_path.resolve()
    assert loaded_index["branch"] == "main"


def test_index_repository_resolves_subdir_to_repo_root_and_detects_tests(tmp_path: Path) -> None:
    """
    If the user indexes a nested folder (e.g. ".../src"), we should still treat the
    repository root as the repo path so test detection can see sibling /tests.
    """
    repo_root = tmp_path / "test_repo"
    repo_root.mkdir()
    (repo_root / ".gitignore").write_text("", encoding="utf-8")

    # Create /src (indexed path) and /tests (where tests live)
    src_dir = repo_root / "src"
    src_dir.mkdir()
    (src_dir / "app.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")

    tests_dir = repo_root / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "from src.app import hello\n\ndef test_hello():\n    assert hello() == 'hi'\n",
        encoding="utf-8",
    )

    # Create orchestrator with custom state dir
    state_dir = tmp_path / "state_2"
    state_dir.mkdir()
    from spec_agent.config.settings import AgentSettings

    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)

    # Index from the nested directory
    index_data = orch.index_repository(repo_path=src_dir, branch="main")

    assert Path(index_data["repo_path"]).resolve() == repo_root.resolve()
    assert index_data.get("requested_path") is not None
    assert Path(index_data["requested_path"]).resolve() == src_dir.resolve()
    assert bool((index_data.get("repository_summary") or {}).get("has_tests", False)) is True

    # Verify index was saved to disk with the resolved repo root
    store = JsonStore(state_dir)
    loaded_index = store.load_repository_index()
    assert Path(loaded_index["repo_path"]).resolve() == repo_root.resolve()


def test_create_task_from_index(tmp_path: Path) -> None:
    """Test that create_task_from_index uses the saved index."""
    repo_path = tmp_path / "test_repo_3"
    repo_path.mkdir()
    (repo_path / ".gitignore").write_text("", encoding="utf-8")

    # Create a simple file
    (repo_path / "test.py").write_text("print('hello')\n", encoding="utf-8")

    # Create orchestrator with custom state dir
    state_dir = tmp_path / "state_3"
    state_dir.mkdir()
    from spec_agent.config.settings import AgentSettings

    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)

    # First, index the repository
    index_data = orch.index_repository(repo_path=repo_path, branch="main")

    # Now create a task from the index
    task = orch.create_task_from_index(description="Add a new feature")

    # Verify task was created with index data
    assert task is not None
    assert task.description == "Add a new feature"
    assert task.repo_path.resolve() == repo_path.resolve()
    assert task.branch == "main"
    assert "repository_summary" in task.metadata
    # Repository summary should match the indexed data
    assert task.metadata["repository_summary"] == index_data["repository_summary"]


def test_create_task_from_index_without_index_fails(tmp_path: Path) -> None:
    """Test that create_task_from_index fails if no index exists."""
    state_dir = tmp_path / "state_4"
    state_dir.mkdir()
    from spec_agent.config.settings import AgentSettings

    settings = AgentSettings(state_dir=state_dir)
    orch = TaskOrchestrator(settings=settings)

    # Try to create task without indexing first
    try:
        orch.create_task_from_index(description="This should fail")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "No repository index found" in str(e)

