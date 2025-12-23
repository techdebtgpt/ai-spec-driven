from pathlib import Path

from spec_agent.services.tests.suggester import TestSuggester as _TestSuggester


def test_detect_test_files_skips_large_directories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_example.py").write_text("def test_example():\n    assert True\n")

    ignored_dir = repo / "node_modules"
    ignored_dir.mkdir()
    (ignored_dir / "component.test.js").write_text("describe('skip', () => {});\n")

    suggester = _TestSuggester()

    has_tests, files, framework = suggester._detect_test_files(repo, repo_context=None)

    assert has_tests is True
    assert files == ["tests/test_example.py"]
    assert framework == "pytest"
