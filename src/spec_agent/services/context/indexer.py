from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import os
import subprocess
import json
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

    def summarize_repository(self, repo_path: Path) -> Dict[str, any]:
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        # Try to enhance language and module detection with Serena if available
        serena_info = self._detect_languages_with_serena(repo_path)
        serena_languages = serena_info.get("languages", []) if serena_info else []
        serena_modules = serena_info.get("modules", []) if serena_info else []
        serena_namespaces = serena_info.get("namespaces", []) if serena_info else []
        serena_directories = serena_info.get("top_directories", []) if serena_info else []
        
        # Load gitignore patterns
        gitignore_spec = self._load_gitignore(repo_path)

        file_counter = 0
        directory_counter = 0
        language_hits: Counter[str] = Counter()
        hotspots: List[Dict[str, any]] = []
        import_edges: Dict[str, List[str]] = defaultdict(list)
        
        # If Serena detected languages, we'll use them to enhance detection
        # The extension-based detection below will still work, but Serena helps
        # identify languages that might be missed or confirm project types

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
        
        return {
            "file_count": file_counter,
            "directory_count": directory_counter,
            "top_languages": top_languages_list if top_languages_list else [],
            "hotspots": hotspots,
            "top_modules": top_modules if top_modules else [],
        }

    def _detect_language(self, path: Path) -> str | None:
        suffix = path.suffix.lower()
        for language, extensions in self.settings.DEFAULT_LANGUAGES.items():
            if suffix in extensions:
                return language
        return None

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


