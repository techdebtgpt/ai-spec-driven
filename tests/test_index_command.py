"""
Tests for the new index and start commands.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import json

from spec_agent.workflow.orchestrator import TaskOrchestrator
from spec_agent.persistence.store import JsonStore


def test_index_repository_creates_index():
    """Test that index_repository creates and saves an index."""
    with TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        
        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
        
        # Create a simple file
        (repo_path / "test.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, capture_output=True)
        
        # Create orchestrator with custom state dir
        state_dir = Path(tmpdir) / "state"
        state_dir.mkdir()
        from spec_agent.config.settings import AgentSettings
        settings = AgentSettings(state_dir=state_dir)
        orch = TaskOrchestrator(settings=settings)
        
        # Index the repository
        index_data = orch.index_repository(repo_path=repo_path, branch="main")
        
        # Verify index was created
        assert index_data is not None
        assert "repository_summary" in index_data
        assert "repo_path" in index_data
        assert "branch" in index_data
        assert index_data["branch"] == "main"
        assert str(repo_path) in index_data["repo_path"]
        
        # Verify index was saved to disk
        store = JsonStore(state_dir)
        loaded_index = store.load_repository_index()
        assert Path(loaded_index["repo_path"]).resolve() == repo_path.resolve()
        assert loaded_index["branch"] == "main"


def test_create_task_from_index():
    """Test that create_task_from_index uses the saved index."""
    with TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "test_repo"
        repo_path.mkdir()
        
        # Initialize a git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
        
        # Create a simple file
        (repo_path / "test.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=repo_path, check=True, capture_output=True)
        
        # Create orchestrator with custom state dir
        state_dir = Path(tmpdir) / "state"
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


def test_create_task_from_index_without_index_fails():
    """Test that create_task_from_index fails if no index exists."""
    with TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        state_dir.mkdir()
        from spec_agent.config.settings import AgentSettings
        settings = AgentSettings(state_dir=state_dir)
        orch = TaskOrchestrator(settings=settings)
        
        # Try to create task without indexing first
        try:
            task = orch.create_task_from_index(description="This should fail")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "No repository index found" in str(e)

