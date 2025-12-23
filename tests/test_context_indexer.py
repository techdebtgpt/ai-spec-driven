from pathlib import Path

from spec_agent.config.settings import AgentSettings
from spec_agent.services.context.indexer import ContextIndexer


def test_summarize_repository_counts_tmp_path(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("import os\n\nprint('hi')\n")
    (repo / "b.js").write_text("import foo from 'bar';\n")
    (repo / "README.md").write_text("# Readme\n")
    (repo / "nested").mkdir()
    (repo / "nested" / "big.py").write_text("print('x')\n" * 600)

    settings = AgentSettings()
    indexer = ContextIndexer(settings)

    summary = indexer.summarize_repository(repo)

    assert summary["file_count"] == 4
    assert summary["directory_count"] >= 1
    assert any("python" in item for item in summary["top_languages"])
    assert summary["hotspots"], "Expected the large file to be flagged as a hotspot"


def test_summarize_targets_limits_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("print('a')\n")
    (repo / "b.py").write_text("print('b')\n")
    (repo / "nested").mkdir()
    (repo / "nested" / "c.py").write_text("print('c')\n" * 600)

    settings = AgentSettings()
    indexer = ContextIndexer(settings)

    summary = indexer.summarize_targets(repo, ["nested"])
    aggregate = summary["aggregate"]

    assert aggregate["file_count"] == 1
    assert "nested/c.py" in summary["targets"]["nested"]["hotspots"][0]["path"]


def test_summarize_repository_detects_dotnet_test_projects(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    # Typical dotnet convention: Foo.Tests/*.csproj
    tests_proj_dir = repo / "CardStoreService.Tests"
    tests_proj_dir.mkdir()
    (tests_proj_dir / "CardStoreService.Tests.csproj").write_text("<Project></Project>\n")

    # Another common convention: Foo.Tests.Unit
    tests_unit_dir = repo / "Pbp.Payments.CardStore.Tests.Unit"
    tests_unit_dir.mkdir()
    (tests_unit_dir / "ExampleTests.cs").write_text("public class ExampleTests {}\n")

    settings = AgentSettings()
    indexer = ContextIndexer(settings)

    summary = indexer.summarize_repository(repo)
    assert summary["has_tests"] is True

