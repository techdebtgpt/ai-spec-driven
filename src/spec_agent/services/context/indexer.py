from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import sys

import networkx as nx
import pathspec  # pyright: ignore[reportMissingImports]

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

    def summarize_repository(self, repo_path: Path, include_serena_semantic_tree: bool = False) -> Dict[str, any]:
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        # Try to enhance language and module detection with Serena if available
        serena_info = self._detect_languages_with_serena(repo_path)
        serena_languages = serena_info.get("languages", []) if serena_info else []
        serena_modules = serena_info.get("modules", []) if serena_info else []
        serena_namespaces = serena_info.get("namespaces", []) if serena_info else []
        serena_directories = serena_info.get("top_directories", []) if serena_info else []
        serena_project_type = serena_info.get("project_type") if serena_info else None
        serena_file_extensions = serena_info.get("file_extensions", {}) if serena_info else {}
        serena_semantic_modules = serena_info.get("semantic_modules", []) if serena_info else []
        serena_semantic_dependencies = serena_info.get("semantic_dependencies", []) if serena_info else []

        # Optionally export a full semantic symbol tree via Serena (can be expensive)
        serena_semantic_tree: Optional[Dict[str, any]] = None
        if include_serena_semantic_tree:
            serena_semantic_tree = self._export_semantic_tree_with_serena(repo_path)
        
        # Load gitignore patterns
        gitignore_spec = self._load_gitignore(repo_path)

        file_counter = 0
        directory_counter = 0
        language_hits: Counter[str] = Counter()
        extension_hits: Counter[str] = Counter()
        hotspots: List[Dict[str, any]] = []
        import_edges: Dict[str, List[str]] = defaultdict(list)
        
        # Basic test harness detection (best-effort). We use this to avoid suggesting
        # test work when the repo clearly has no tests.
        has_tests = False
        test_paths_sample: List[str] = []
        
        # Track file sizes for statistics
        total_size_bytes = 0

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
                # Heuristic: common test directories
                if not has_tests:
                    dir_name = path.name.lower()
                    # Common cross-language test dirs + dotnet conventions like Foo.Tests / Foo.IntegrationTests
                    if (
                        dir_name in {"test", "tests", "__tests__", "spec", "specs"}
                        # dotnet naming often includes ".Tests." segments (e.g. Foo.Tests.Unit)
                        or ".tests." in dir_name
                        or ".test." in dir_name
                        or dir_name.endswith(".test")
                        or dir_name.endswith(".tests")
                        or dir_name.endswith(".integrationtest")
                        or dir_name.endswith(".integrationtests")
                        or dir_name.endswith(".unittest")
                        or dir_name.endswith(".unittests")
                    ):
                        has_tests = True
                        if len(test_paths_sample) < 10:
                            test_paths_sample.append(str(relative_path).replace("\\", "/"))
                continue

            file_counter += 1
            
            # Track file extension
            if path.suffix:
                extension_hits[path.suffix.lower()] += 1
            
            # Track file size
            try:
                file_size = path.stat().st_size
                total_size_bytes += file_size
            except OSError:
                pass
            
            language = self._detect_language(path)
            if language:
                language_hits[language] += 1
            
            # Heuristic: common test file naming patterns (cross-language)
            if not has_tests:
                name_lower = path.name.lower()
                rel_str = str(relative_path).replace("\\", "/")
                rel_lower = rel_str.lower()
                if (
                    name_lower.startswith("test_")
                    or name_lower.endswith("_test.py")
                    or name_lower.endswith("_test.go")
                    or name_lower.endswith(".spec.ts")
                    or name_lower.endswith(".spec.tsx")
                    or name_lower.endswith(".test.ts")
                    or name_lower.endswith(".test.tsx")
                    or name_lower.endswith(".spec.js")
                    or name_lower.endswith(".test.js")
                    or "/test/" in f"/{rel_lower}/"
                    or "/tests/" in f"/{rel_lower}/"
                    # .NET conventions: csproj names/folders often contain ".Tests" or "IntegrationTests"
                    or name_lower.endswith(".tests.csproj")
                    or name_lower.endswith(".test.csproj")
                    or name_lower.endswith("tests.cs")
                    or name_lower.endswith("test.cs")
                    or "integrationtests" in name_lower
                    or "unittests" in name_lower
                    or ".tests." in rel_lower
                    or ".test." in rel_lower
                    or "/.tests/" in f"/{rel_lower}/"
                    or "/.test/" in f"/{rel_lower}/"
                ):
                    has_tests = True
                    if len(test_paths_sample) < 10:
                        test_paths_sample.append(rel_str)

            if path.suffix.lower() not in {".dll", ".exe"}:
                line_count = self._count_lines(path)
                if line_count >= self.settings.hotspot_loc_threshold:
                    hotspots.append(
                        {"path": str(relative_path), "reason": f"{line_count} LOC", "lines": line_count}
                    )

            if language in {"python", "typescript"}:
                imports = self._extract_imports(path)
                if imports:
                    import_edges[str(relative_path)].extend(imports)

        # Store a shallow directory tree for later "logical target" → path resolution,
        # without re-scanning the repo during planning.
        directory_structure = self._build_directory_structure(repo_path, gitignore_spec, max_depth=3)

        graph = self._build_graph(import_edges)
        top_modules = [
            f"{node} (fan-in={graph.in_degree(node)})"
            for node, _ in sorted(
                graph.in_degree(), key=lambda pair: pair[1], reverse=True
            )[: self.settings.top_module_count]
        ]
        
        dependency_graph_summary = {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "top_fan_in": [
                {"module": node, "references": graph.in_degree(node)}
                for node, _ in sorted(graph.in_degree(), key=lambda pair: pair[1], reverse=True)[:10]
            ],
            "top_fan_out": [
                {"module": node, "references": graph.out_degree(node)}
                for node, _ in sorted(graph.out_degree(), key=lambda pair: pair[1], reverse=True)[:10]
            ],
        }

        # Enhance modules with Serena-detected modules/namespaces/directories
        if not top_modules or (len(top_modules) < self.settings.top_module_count):
            # Add Serena-detected namespaces (for C# projects)
            for ns in serena_namespaces[:self.settings.top_module_count]:
                ns_display = f"{ns} (namespace)"
                if ns_display not in top_modules:
                    top_modules.append(ns_display)
            
            # Add Serena-detected modules (for Python/JS projects)
            for mod in serena_modules[:self.settings.top_module_count]:
                mod_display = f"{mod} (module)"
                if mod_display not in top_modules:
                    top_modules.append(mod_display)
            
            # Add top-level directories as modules if we still need more
            if len(top_modules) < self.settings.top_module_count:
                for dir_name in serena_directories[:self.settings.top_module_count]:
                    dir_display = f"{dir_name} (directory)"
                    if dir_display not in top_modules:
                        top_modules.append(dir_display)
            
            # Add semantic modules if available
            if len(top_modules) < self.settings.top_module_count and serena_semantic_modules:
                for semantic_module in serena_semantic_modules:
                    module_name = semantic_module.get("module")
                    if not module_name:
                        continue
                    semantic_display = f"{module_name} (semantic)"
                    if semantic_display not in top_modules:
                        top_modules.append(semantic_display)
                    if len(top_modules) >= self.settings.top_module_count:
                        break

            # Limit to top_module_count
            top_modules = top_modules[:self.settings.top_module_count]

        # Build top languages list from extension-based detection
        top_languages_list = [f"{lang} ({count})" for lang, count in language_hits.most_common(3)]
        
        # Enhance languages with Serena-detected languages if we don't have enough
        if serena_languages and len(top_languages_list) < 3:
            # Extract language names already in the list (to avoid duplicates)
            existing_lang_names = set()
            for item in top_languages_list:
                # Extract language name from format like "csharp (69)" or "python (42)"
                lang_name = item.split(" (")[0].lower()
                existing_lang_names.add(lang_name)
            
            # Add Serena-detected languages that aren't already in the list
            for lang in serena_languages:
                if len(top_languages_list) >= 3:
                    break
                lang_lower = lang.lower()
                # Check if this language is already represented
                if lang_lower not in existing_lang_names:
                    top_languages_list.append(f"{lang} (detected by Serena)")
                    existing_lang_names.add(lang_lower)
        
        # Fallback: If no languages detected by extension-based method, use Serena
        if not top_languages_list and serena_languages:
            for lang in serena_languages[:3]:
                top_languages_list.append(f"{lang} (detected by Serena)")
        
        # Build detailed language breakdown with file counts
        language_details = []
        for lang, count in language_hits.most_common(10):
            language_details.append({"language": lang, "file_count": count})
        
        # Add Serena-detected languages that might not have been counted by extension
        if serena_languages:
            existing_langs = {detail["language"].lower() for detail in language_details}
            for lang in serena_languages:
                if lang.lower() not in existing_langs:
                    language_details.append({"language": lang, "file_count": 0, "detected_by": "serena"})
        
        # Detect frameworks based on common patterns
        frameworks = self._detect_frameworks(repo_path)
        
        # Determine project type with heuristics around detected frameworks/languages.
        project_type = serena_project_type
        if project_type:
            project_type = project_type.strip()
            if project_type.lower() == "node.js" and ".NET" in frameworks:
                project_type = ".NET"
            elif ".net" not in project_type.lower() and ".NET" in frameworks:
                project_type = ".NET"
        if not project_type and ".NET" in frameworks:
            project_type = ".NET"
        if not project_type and language_hits.get("csharp"):
            project_type = ".NET"
        if project_type and project_type.lower() == "node.js" and not frameworks:
            project_type = None

        # Build comprehensive response with enhanced data
        return {
            # Basic counts
            "file_count": file_counter,
            "directory_count": directory_counter,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            
            # Language information
            "top_languages": top_languages_list if top_languages_list else [],
            "language_details": language_details,
            "all_languages": list(language_hits.keys()),
            
            # Project type and structure
            "project_type": project_type,
            "frameworks": frameworks,
            "top_file_extensions": [f"{ext} ({count})" for ext, count in extension_hits.most_common(10)],
            "has_tests": bool(has_tests),
            "test_paths_sample": test_paths_sample,
            
            # Modules and namespaces
            "top_modules": top_modules if top_modules else [],
            "namespaces": serena_namespaces[:10] if serena_namespaces else [],
            "top_directories": serena_directories[:10] if serena_directories else [],
            
            # Code quality indicators
            "hotspots": hotspots,
            
            # Serena-specific data
            "serena_enabled": bool(serena_info),
            "serena_file_extensions": serena_file_extensions,
            "serena_semantic_modules": serena_semantic_modules,
            "serena_semantic_dependencies": serena_semantic_dependencies,
            "serena_semantic_tree": serena_semantic_tree,
            
            # Dependency graph insight
            "dependency_graph": dependency_graph_summary,

            # Shallow directory tree (for plan-target resolution / UX)
            "directory_structure": directory_structure,
        }

    def summarize_targets(
        self,
        repo_path: Path,
        targets: List[str],
        *,
        include_serena_semantic_tree: bool = False,
        include_file_list: bool = False,
        max_file_list: int = 5000,
    ) -> Dict[str, any]:
        """
        Produce scoped summaries for a subset of repository paths.
        
        Args:
            repo_path: Root of the repository
            targets: Relative or absolute paths pointing to files/directories of interest
            include_file_list: When true, include a bounded allowlist of files covered by these targets.
            max_file_list: Maximum number of files to include in the allowlist (defensive cap).
        
        Returns:
            Dictionary with per-target stats plus aggregate totals.
        """
        if not targets:
            return {"targets": {}, "aggregate": {}}

        repo_path = repo_path.resolve()
        gitignore_spec = self._load_gitignore(repo_path)

        target_summaries: Dict[str, Dict[str, any]] = {}
        aggregate_languages: Counter[str] = Counter()
        aggregate_files = 0
        aggregate_size = 0
        aggregate_hotspots: List[Dict[str, any]] = []
        impacted_top_directories: List[str] = []
        impacted_namespaces: List[str] = []
        impacted_files: List[str] = []
        allowed_files: List[str] = []
        allowed_files_set: set[str] = set()
        allowlist_truncated = False

        for raw_target in targets:
            target_path = Path(raw_target)
            if not target_path.is_absolute():
                target_path = repo_path / target_path

            try:
                relative_display = str(target_path.relative_to(repo_path))
            except ValueError:
                relative_display = str(target_path)

            if not target_path.exists():
             
                continue

            summary = self._summarize_single_target(
                repo_path,
                target_path,
                gitignore_spec,
                include_files=include_file_list,
                max_files=max_file_list,
            )
            if not summary:
                continue

            target_summaries[relative_display] = summary
            aggregate_files += summary.get("file_count", 0)
            aggregate_size += summary.get("total_size_bytes", 0)
            aggregate_languages.update(summary.get("language_counts", {}))
            aggregate_hotspots.extend(
                {"path": f"{relative_display}/{item['path']}", "lines": item["lines"]}
                for item in summary.get("hotspots", [])
            )

            # Collect impact signals
            for rel_file in (summary.get("files_sample") or []):
                if rel_file not in impacted_files:
                    impacted_files.append(rel_file)
                top_dir = rel_file.split("/", 1)[0] if "/" in rel_file else rel_file
                if top_dir and top_dir not in impacted_top_directories:
                    impacted_top_directories.append(top_dir)
            for ns in (summary.get("csharp_namespaces") or []):
                if ns not in impacted_namespaces:
                    impacted_namespaces.append(ns)

            if include_file_list:
                for rel_file in (summary.get("files_all") or []):
                    normalized = str(rel_file).replace("\\", "/")
                    if normalized in allowed_files_set:
                        continue
                    if len(allowed_files) >= max_file_list:
                        allowlist_truncated = True
                        break
                    allowed_files_set.add(normalized)
                    allowed_files.append(normalized)

        aggregate = {
            "file_count": aggregate_files,
            "total_size_bytes": aggregate_size,
            "top_languages": [
                f"{lang} ({count})"
                for lang, count in aggregate_languages.most_common(5)
            ],
            "hotspots": aggregate_hotspots[:10],
        }

        # Optional: scoped Serena semantic tree export (symbols overview per file for targets)
        serena_semantic_tree = None
        if include_serena_semantic_tree:
            serena_semantic_tree = self._export_semantic_tree_with_serena(repo_path, targets=targets)

        impact = {
            "top_directories": impacted_top_directories[:20],
            "namespaces": impacted_namespaces[:50],
            "files_sample": impacted_files[:200],
        }

        return {
            "targets": target_summaries,
            "aggregate": aggregate,
            "impact": impact,
            "serena_semantic_tree": serena_semantic_tree,
            "scope": {
                "targets": [str(t) for t in targets],
                "frozen": bool(include_file_list),
                "allowed_files": allowed_files if include_file_list else [],
                "allowed_files_truncated": allowlist_truncated,
                "max_file_list": max_file_list,
            },
        }

    def _summarize_single_target(
        self,
        repo_path: Path,
        target_path: Path,
        gitignore_spec: Optional[pathspec.PathSpec],
        *,
        include_files: bool = False,
        max_files: int = 5000,
    ) -> Dict[str, any]:
        """
        Summarize a single directory or file relative to repo root.
        """
        paths_to_scan: List[Path] = []
        if target_path.is_dir():
            paths_to_scan = sorted(target_path.rglob("*"))
        else:
            paths_to_scan = [target_path]

        file_counter = 0
        language_hits: Counter[str] = Counter()
        hotspots: List[Dict[str, any]] = []
        total_size_bytes = 0
        files_sample: List[str] = []
        files_all: List[str] = []
        files_all_set: set[str] = set()
        csharp_namespaces: List[str] = []

        for path in paths_to_scan:
            if path.is_dir():
                continue

            relative_path = path.relative_to(repo_path)
            if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
                continue

            if ".git" in path.parts:
                continue

            file_counter += 1
            try:
                file_size = path.stat().st_size
                total_size_bytes += file_size
            except OSError:
                pass

            if len(files_sample) < 200:
                try:
                    files_sample.append(str(relative_path).replace("\\", "/"))
                except Exception:
                    pass

            if include_files and len(files_all) < max_files:
                try:
                    normalized = str(relative_path).replace("\\", "/")
                    if normalized not in files_all_set:
                        files_all_set.add(normalized)
                        files_all.append(normalized)
                except Exception:
                    pass

            if path.suffix.lower() == ".cs" and len(csharp_namespaces) < 50:
                namespace = self._extract_csharp_namespace(path)
                if namespace and namespace not in csharp_namespaces:
                    csharp_namespaces.append(namespace)

            language = self._detect_language(path)
            if language:
                language_hits[language] += 1

            line_count = self._count_lines(path)
            if line_count >= self.settings.hotspot_loc_threshold:
                hotspots.append(
                    {"path": str(relative_path), "lines": line_count}
                )

        if not file_counter:
            return {}

        return {
            "file_count": file_counter,
            "total_size_bytes": total_size_bytes,
            "top_languages": [
                f"{lang} ({count})"
                for lang, count in language_hits.most_common(3)
            ],
            "language_counts": dict(language_hits),
            "hotspots": hotspots[:5],
            "files_sample": files_sample,
            "files_all": files_all if include_files else [],
            "csharp_namespaces": csharp_namespaces,
        }

    @staticmethod
    def _extract_csharp_namespace(path: Path) -> Optional[str]:
        """
        Best-effort namespace extraction for C# files to help "impacted modules" reporting.
        """
        try:
            # Only scan the top of the file; namespace is usually near top
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for _ in range(250):
                    line = f.readline()
                    if not line:
                        break
                    stripped = line.strip()
                    if not stripped or stripped.startswith("//"):
                        continue
                    if stripped.startswith("namespace "):
                        candidate = stripped[len("namespace ") :].strip().rstrip("{").strip()
                        # Handle file-scoped namespace "namespace X.Y;"
                        candidate = candidate.rstrip(";").strip()
                        return candidate or None
        except OSError:
            return None
        return None

    def _build_directory_structure(self, repo_path: Path, gitignore_spec: Optional[pathspec.PathSpec], max_depth: int = 3) -> Dict[str, any]:
        """
        Build a hierarchical directory structure with file counts and sizes.
        
        Args:
            repo_path: Root path of the repository
            gitignore_spec: Gitignore patterns to respect
            max_depth: Maximum depth to traverse (default 3 levels)
        
        Returns:
            Dictionary representing the directory tree structure
        """
        def build_tree(current_path: Path, depth: int = 0) -> Dict[str, any]:
            if depth > max_depth:
                return None
            
            relative_path = current_path.relative_to(repo_path) if current_path != repo_path else Path(".")
            
            # Skip if matches gitignore
            if gitignore_spec and gitignore_spec.match_file(str(relative_path)):
                return None
            
            # Skip .git directory
            if ".git" in current_path.parts:
                return None
            
            node = {
                "name": current_path.name if current_path != repo_path else ".",
                "path": str(relative_path),
                "type": "directory" if current_path.is_dir() else "file",
                "depth": depth,
            }
            
            if current_path.is_file():
                try:
                    node["size"] = current_path.stat().st_size
                    node["extension"] = current_path.suffix.lower() if current_path.suffix else None
                except OSError:
                    node["size"] = 0
                return node
            
            # It's a directory
            children = []
            file_count = 0
            dir_count = 0
            total_size = 0
            
            try:
                for child in sorted(current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
                    # Skip .git
                    if child.name == ".git":
                        continue
                    
                    child_relative = child.relative_to(repo_path)
                    if gitignore_spec and gitignore_spec.match_file(str(child_relative)):
                        continue
                    
                    child_node = build_tree(child, depth + 1)
                    if child_node:
                        children.append(child_node)
                        if child_node["type"] == "file":
                            file_count += 1
                            total_size += child_node.get("size", 0)
                        else:
                            dir_count += 1
                            file_count += child_node.get("file_count", 0)
                            total_size += child_node.get("total_size", 0)
            except (OSError, PermissionError):
                pass
            
            node["children"] = children
            node["file_count"] = file_count
            node["dir_count"] = dir_count
            node["total_size"] = total_size
            
            return node
        
        return build_tree(repo_path)

    def _detect_language(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        for language, extensions in self.settings.DEFAULT_LANGUAGES.items():
            if suffix in extensions:
                return language
        return None

    def _detect_frameworks(self, repo_path: Path) -> List[str]:
        """
        Detect common frameworks based on project files and directory structure.
        """
        frameworks = []
        
        # Check for common framework indicator files
        framework_indicators = {
            # Python frameworks
            "requirements.txt": ["Python"],
            "pyproject.toml": ["Python"],
            "setup.py": ["Python"],
            "Pipfile": ["Python"],
            "poetry.lock": ["Poetry"],
            "manage.py": ["Django"],
            "flask": ["Flask"],  # Directory
            "fastapi": ["FastAPI"],  # Directory or imports
            
            # JavaScript/TypeScript frameworks
            "package.json": [],  # Will check contents
            "package-lock.json": ["npm"],
            "yarn.lock": ["Yarn"],
            "pnpm-lock.yaml": ["pnpm"],
            "next.config.js": ["Next.js"],
            "nuxt.config.js": ["Nuxt.js"],
            "angular.json": ["Angular"],
            "vue.config.js": ["Vue.js"],
            "svelte.config.js": ["Svelte"],
            
            # .NET frameworks
            "*.csproj": [".NET"],
            "*.sln": [".NET"],
            "*.cs": ["C#"],
            
            # Java frameworks
            "pom.xml": ["Maven", "Java"],
            "build.gradle": ["Gradle", "Java"],
            "build.gradle.kts": ["Gradle", "Kotlin"],
            
            # Ruby frameworks
            "Gemfile": ["Ruby"],
            "Rakefile": ["Ruby", "Rake"],
            "config.ru": ["Rack"],
            
            # Go
            "go.mod": ["Go"],
            "go.sum": ["Go"],
            
            # Rust
            "Cargo.toml": ["Rust"],
            "Cargo.lock": ["Rust"],
            
            # PHP
            "composer.json": ["PHP", "Composer"],
            
            # Mobile
            "*.xcodeproj": ["iOS", "Swift"],
            "*.xcworkspace": ["iOS", "Swift"],
            "Podfile": ["CocoaPods", "iOS"],
            
            # DevOps/Infrastructure
            "Dockerfile": ["Docker"],
            "docker-compose.yml": ["Docker Compose"],
            "terraform": ["Terraform"],  # Directory
            "*.tf": ["Terraform"],
            "kubernetes": ["Kubernetes"],  # Directory
            "*.yaml": [],  # Will check for k8s manifests
        }
        
        for pattern, fw_list in framework_indicators.items():
            if "*" in pattern:
                # Glob pattern
                matches = list(repo_path.glob(f"**/{pattern}"))[:3]  # Limit to avoid too many
                if matches:
                    frameworks.extend(fw_list)
            else:
                # Exact filename or directory
                if (repo_path / pattern).exists():
                    frameworks.extend(fw_list)
        
        # Check package.json for specific frameworks
        package_json = repo_path / "package.json"
        if package_json.exists():
            try:
                import json
                with package_json.open() as f:
                    pkg_data = json.load(f)
                    deps = {**pkg_data.get("dependencies", {}), **pkg_data.get("devDependencies", {})}
                    
                    if "react" in deps or "react-dom" in deps:
                        frameworks.append("React")
                    if "vue" in deps:
                        frameworks.append("Vue.js")
                    if "angular" in deps or "@angular/core" in deps:
                        frameworks.append("Angular")
                    if "svelte" in deps:
                        frameworks.append("Svelte")
                    if "next" in deps:
                        frameworks.append("Next.js")
                    if "express" in deps:
                        frameworks.append("Express.js")
                    if "nestjs" in deps or "@nestjs/core" in deps:
                        frameworks.append("NestJS")
                    if "typescript" in deps:
                        frameworks.append("TypeScript")
            except (json.JSONDecodeError, OSError):
                pass
        
        # Remove duplicates and return
        return sorted(list(set(frameworks)))

    def _detect_languages_with_serena(self, repo_path: Path) -> Optional[Dict[str, any]]:
        """
        Optionally use Serena to detect languages and modules in the repository.
        Returns a dict with 'languages' and 'modules' keys, or None if unavailable.
        """
        import logging
        import time
        import os
        logger = logging.getLogger(__name__)
        
        # Check if Serena is enabled
        # Also check environment variable directly in case settings didn't pick it up
        serena_enabled_env = os.getenv("SPEC_AGENT_SERENA_ENABLED", "").lower() in {"1", "true", "yes"}
        serena_enabled = self.settings.serena_enabled or serena_enabled_env
        
        if not serena_enabled:
            logger.debug(f"Serena not enabled (settings={self.settings.serena_enabled}, env={os.getenv('SPEC_AGENT_SERENA_ENABLED')}), skipping language detection")
            return None
        
        # Try to use Serena's language detection script
        try:
            # Calculate path: __file__ is in src/spec_agent/services/context/indexer.py
            # Go up: context -> services -> spec_agent -> src -> project root
            language_detection_script = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "serena_language_detection.py"
            if not language_detection_script.exists():
                logger.debug(f"Serena language detection script not found: {language_detection_script}")
                return None
            
            logger.debug(f"Starting Serena language detection for {repo_path}")
            logger.info(f"Starting Serena language detection for {repo_path}")
            start_time = time.time()
            
            # Call the language detection script
            import subprocess
            # Capture both stdout and stderr so we can see logs
            result = subprocess.run(
                [sys.executable, str(language_detection_script), str(repo_path)],
                capture_output=True,
                text=True,
                timeout=60,  # 60 second timeout (Serena initialization can take time)
            )
            
            elapsed = time.time() - start_time
            
            # Only print stderr if there are errors (WARNING or ERROR level)
            # Filter out non-critical language server initialization errors
            if result.stderr and isinstance(result.stderr, str):
                # Filter to only show warnings and errors, not INFO logs
                # Skip non-critical language server initialization errors
                non_critical_patterns = [
                    'Language server stderr reader thread terminated',
                    'Language server stdout read process terminated',
                    'Language server startup',
                    'Language server initialization failed',
                    'LanguageServerTerminatedException',
                    'Error starting language server for language',
                    'Language server startup (language=',
                    'init_language_server_manager',
                ]
                for line in result.stderr.split('\n'):
                    if line.strip() and ('WARNING' in line or 'ERROR' in line or 'CRITICAL' in line):
                        # Skip non-critical language server errors
                        if any(pattern in line for pattern in non_critical_patterns):
                            continue
                        logger.debug(f"Serena detection output: {line}")
            
            logger.debug(f"Serena detection completed in {elapsed:.2f}s (returncode: {result.returncode})")
            logger.info(f"Serena language detection completed in {elapsed:.2f}s (returncode: {result.returncode})")
            
            if result.returncode == 0:
                import json
                try:
                    data = json.loads(result.stdout)
                    # Only return if we got actual data (not just error messages)
                    if data.get("languages") or data.get("modules") or data.get("namespaces"):
                        modules_list = data.get("modules", [])
                        namespaces_list = data.get("namespaces", [])
                        logger.info(f"Serena detected: {len(data.get('languages', []))} languages, {len(modules_list)} modules, {len(namespaces_list)} namespaces")
                        return {
                            "languages": data.get("languages", []),
                            "modules": modules_list,
                            "namespaces": namespaces_list,
                            "top_directories": data.get("top_directories", []),
                            "file_extensions": data.get("file_extensions", {}),
                            "project_type": data.get("project_type"),
                            "semantic_modules": data.get("semantic_modules", []),
                            "semantic_dependencies": data.get("semantic_dependencies", []),
                        }
                    else:
                        error_msg = data.get("error", "Unknown error")
                        logger.warning(f"Serena language detection returned no data: {error_msg}")
                except json.JSONDecodeError:
                    # If stdout isn't valid JSON, check stderr for errors
                    if result.stderr and isinstance(result.stderr, str):
                        logger.warning(f"Serena language detection returned invalid JSON. stderr: {result.stderr[:500]}")
                    if result.stdout and isinstance(result.stdout, str):
                        logger.debug(f"stdout: {result.stdout[:500]}")
            else:
                logger.warning(f"Serena language detection failed with returncode {result.returncode}")
                if result.stderr and isinstance(result.stderr, str):
                    logger.warning(f"stderr: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.warning(f"Serena language detection timed out after {elapsed:.2f}s")
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as exc:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.warning(f"Serena language detection failed after {elapsed:.2f}s: {exc}")
        except Exception as exc:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Serena language detection error after {elapsed:.2f}s: {exc}", exc_info=True)
        
        return None

    def _export_semantic_tree_with_serena(
        self,
        repo_path: Path,
        targets: Optional[List[str]] = None,
    ) -> Optional[Dict[str, any]]:
        """
        Export a full Serena semantic tree (symbol overview per file) for sharing.

        This shells out to scripts/serena_semantic_tree.py to keep MCP dependency optional.
        """
        import json
        import logging
        import os
        import subprocess
        import time

        logger = logging.getLogger(__name__)

        # Check if Serena is enabled (settings or env)
        serena_enabled_env = os.getenv("SPEC_AGENT_SERENA_ENABLED", "").lower() in {"1", "true", "yes"}
        serena_enabled = self.settings.serena_enabled or serena_enabled_env
        if not serena_enabled:
            logger.info("Serena not enabled; skipping semantic tree export")
            return None

        # Resolve script path from this file location
        script_path = (
            Path(__file__).parent.parent.parent.parent.parent
            / "scripts"
            / "serena_semantic_tree.py"
        )
        if not script_path.exists():
            logger.warning("Serena semantic tree script not found: %s", script_path)
            return None

        # Longer timeout than language detection: full tree can take time
        timeout_seconds = self.settings.serena_timeout_seconds
        env_timeout = os.getenv("SPEC_AGENT_SERENA_SEMANTIC_TREE_TIMEOUT")
        if env_timeout:
            try:
                timeout_seconds = int(env_timeout)
            except ValueError:
                pass
        if timeout_seconds < 120:
            timeout_seconds = 120

        logger.info("Starting Serena semantic tree export (timeout=%ss)", timeout_seconds)
        start_time = time.time()

        try:
            cmd = [sys.executable, str(script_path), str(repo_path)]
            for target in (targets or []):
                cmd.extend(["--target", str(target)])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Serena semantic tree export timed out after %ss", timeout_seconds)
            return {"error": "timeout", "tree": None, "timeout_seconds": timeout_seconds}
        except Exception as exc:
            logger.warning("Serena semantic tree export failed: %s", exc)
            return {"error": str(exc), "tree": None}

        elapsed = time.time() - start_time
        if result.returncode != 0:
            stderr_excerpt = (result.stderr or "")[:500]
            logger.warning("Serena semantic tree export failed (rc=%s) in %.2fs", result.returncode, elapsed)
            if stderr_excerpt:
                logger.debug("Serena semantic tree stderr: %s", stderr_excerpt)

        try:
            data = json.loads(result.stdout)
        except Exception:
            # If stdout isn't JSON, return a minimal error payload
            return {
                "error": "invalid_json_from_serena_semantic_tree",
                "returncode": result.returncode,
                "stdout_excerpt": (result.stdout or "")[:500],
                "stderr_excerpt": (result.stderr or "")[:500],
                "elapsed_seconds": round(elapsed, 3),
                "tree": None,
            }

        # Attach a small bit of execution metadata
        if isinstance(data, dict):
            data.setdefault("execution", {})
            if isinstance(data["execution"], dict):
                data["execution"].update(
                    {
                        "returncode": result.returncode,
                        "elapsed_seconds": round(elapsed, 3),
                    }
                )
        return data if isinstance(data, dict) else {"tree": None, "raw": data}

    def _match_serena_language(self, path: Path, serena_languages: List[str]) -> Optional[str]:
        """
        Match a file path against Serena-detected languages.
        This is a helper for future use - currently we use extension-based detection.
        """
        # For now, this is a placeholder
        # The main language detection happens in _detect_language based on extensions
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
