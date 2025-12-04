from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import networkx as nx
import pathspec

from ...config.settings import AgentSettings


class ContextIndexer:
    """
    Repository inventory + hotspot detection stub.

    For the MVP iteration we:
      - Count files, directories, and language distribution.
      - Build a lightweight import graph for Python/TS files (best-effort).
      - Flag large files purely by line count until richer heuristics arrive.
    """

    def __init__(self, settings: AgentSettings) -> None:
        self.settings = settings

    def _load_gitignore(self, repo_path: Path) -> Optional[pathspec.PathSpec]:
        """
        Load and parse .gitignore file if it exists.

        Returns a PathSpec matcher for filtering paths, or None if no .gitignore.
        """
        gitignore_path = repo_path / ".gitignore"
        if not gitignore_path.exists():
            return None

        try:
            with gitignore_path.open("r", encoding="utf-8", errors="ignore") as f:
                patterns = f.read().splitlines()
            # Filter out empty lines and comments
            patterns = [p.strip() for p in patterns if p.strip() and not p.strip().startswith("#")]
            return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        except OSError:
            return None

    def summarize_repository(self, repo_path: Path) -> Dict[str, any]:
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        # Load gitignore patterns
        gitignore_spec = self._load_gitignore(repo_path)

        file_counter = 0
        directory_counter = 0
        language_hits: Counter[str] = Counter()
        hotspots: List[Dict[str, any]] = []
        import_edges: Dict[str, List[str]] = defaultdict(list)

        for path in repo_path.rglob("*"):
            # Get relative path for gitignore matching
            relative_path = path.relative_to(repo_path)

            # Skip if matches gitignore patterns
            if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
                continue

            # Always skip .git directory
            if ".git" in path.parts:
                continue

            if path.is_dir():
                directory_counter += 1
                continue

            file_counter += 1
            language = self._detect_language(path)
            if language:
                language_hits[language] += 1

            line_count = self._count_lines(path)
            if line_count >= self.settings.hotspot_loc_threshold:
                hotspots.append(
                    {"path": str(relative_path), "reason": f"{line_count} LOC"}
                )

            if language in {"python", "typescript"}:
                imports = self._extract_imports(path)
                if imports:
                    import_edges[str(relative_path)].extend(imports)

        graph = self._build_graph(import_edges)
        top_modules = [
            f"{node} (fan-in={graph.in_degree(node)})"
            for node, _ in sorted(
                graph.in_degree(), key=lambda pair: pair[1], reverse=True
            )[: self.settings.top_module_count]
        ]

        return {
            "file_count": file_counter,
            "directory_count": directory_counter,
            "top_languages": [f"{lang} ({count})" for lang, count in language_hits.most_common(3)],
            "hotspots": hotspots,
            "top_modules": top_modules,
        }

    def _detect_language(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        for language, extensions in self.settings.DEFAULT_LANGUAGES.items():
            if suffix in extensions:
                return language
        return None

    @staticmethod
    def _count_lines(path: Path) -> int:
        try:
            return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            return 0

    @staticmethod
    def _extract_imports(path: Path) -> List[str]:
        imports: List[str] = []
        try:
            for line in path.open("r", encoding="utf-8", errors="ignore"):
                line = line.strip()
                if line.startswith("import "):
                    imports.append(line.replace("import", "").strip())
                elif line.startswith("from "):
                    parts = line.split()
                    if len(parts) >= 2:
                        imports.append(parts[1])
        except OSError:
            pass
        return imports

    @staticmethod
    def _build_graph(edges: Dict[str, Iterable[str]]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for source, targets in edges.items():
            for target in targets:
                graph.add_edge(source, target)
        return graph


