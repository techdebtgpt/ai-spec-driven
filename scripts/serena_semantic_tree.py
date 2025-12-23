#!/usr/bin/env python3
"""
Generate a semantic symbol tree for an entire repository using Serena MCP.

This script is intended to be called from Spec Agent's indexer (via subprocess)
so the core package can remain usable without the optional MCP dependency.

Output is a JSON payload containing:
  - stats and configuration used
  - a hierarchical tree (directories + files)
  - per-file symbol overview (best-effort)
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]

try:
    import pathspec  # type: ignore[reportMissingImports]

    PATHSPEC_AVAILABLE = True
except Exception:
    pathspec = None  # type: ignore[assignment]
    PATHSPEC_AVAILABLE = False


CODE_EXTENSIONS: Tuple[str, ...] = (
    ".py",
    ".pyi",
    ".pyw",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mjs",
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hpp",
    ".cs",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".tf",
)


def _get_serena_mcp_command(repo_path: Path) -> list[str]:
    """
    Get the command to start Serena MCP server with the project path.

    Compatible with the other Serena helper scripts in this repository.
    """
    cmd = os.getenv("SERENA_MCP_COMMAND")
    if cmd:
        import shlex

        base_cmd = shlex.split(cmd)
        if "--project" not in base_cmd:
            base_cmd.extend(["--project", str(repo_path)])
        return base_cmd

    import shutil

    uvx_path = shutil.which("uvx")
    if uvx_path:
        return [
            uvx_path,
            "--from",
            "git+https://github.com/oraios/serena",
            "serena",
            "start-mcp-server",
            "--project",
            str(repo_path),
            "--transport",
            "stdio",
            "--enable-web-dashboard",
            "false",
            "--enable-gui-log-window",
            "false",
        ]

    serena_path = shutil.which("serena")
    if serena_path:
        return [
            serena_path,
            "start-mcp-server",
            "--project",
            str(repo_path),
            "--transport",
            "stdio",
            "--enable-web-dashboard",
            "false",
            "--enable-gui-log-window",
            "false",
        ]

    return [
        "uvx",
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--project",
        str(repo_path),
        "--transport",
        "stdio",
        "--enable-web-dashboard",
        "false",
        "--enable-gui-log-window",
        "false",
    ]


def _should_skip_path(path: Path) -> bool:
    # Always skip VCS metadata
    if ".git" in path.parts:
        return True
    return False


def _load_gitignore(repo_path: Path) -> Any:
    if not PATHSPEC_AVAILABLE:
        return None
    gitignore_path = repo_path / ".gitignore"
    if not gitignore_path.exists():
        return None
    try:
        patterns = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        patterns = [p.strip() for p in patterns if p.strip() and not p.strip().startswith("#")]
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except Exception:
        return None


def _collect_code_files(repo_path: Path) -> List[Path]:
    """
    Collect code files from the filesystem (fast, reliable), then ask Serena for symbols.
    """
    gitignore_spec = _load_gitignore(repo_path)
    files: List[Path] = []
    for p in repo_path.rglob("*"):
        if p.is_dir():
            continue
        if _should_skip_path(p):
            continue
        if gitignore_spec:
            try:
                rel = p.relative_to(repo_path)
                if gitignore_spec.match_file(str(rel)):
                    continue
            except Exception:
                pass
        if p.suffix.lower() not in CODE_EXTENSIONS:
            continue
        files.append(p)
    files.sort(key=lambda x: str(x))
    return files


def _normalize_targets(repo_path: Path, raw_targets: List[str]) -> List[Path]:
    targets: List[Path] = []
    for raw in raw_targets:
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (repo_path / p).resolve()
        else:
            p = p.resolve()
        try:
            p.relative_to(repo_path)
        except Exception:
            continue
        if p.exists():
            targets.append(p)

    # de-dupe
    unique: List[Path] = []
    seen = set()
    for t in targets:
        key = str(t)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def _collect_code_files_scoped(repo_path: Path, targets: List[Path]) -> List[Path]:
    """
    Collect code files only under given targets (files or directories).
    """
    if not targets:
        return _collect_code_files(repo_path)

    gitignore_spec = _load_gitignore(repo_path)
    files: List[Path] = []

    def _maybe_add(p: Path) -> None:
        if p.is_dir():
            return
        if _should_skip_path(p):
            return
        if p.suffix.lower() not in CODE_EXTENSIONS:
            return
        if gitignore_spec:
            try:
                rel = p.relative_to(repo_path)
                if gitignore_spec.match_file(str(rel)):
                    return
            except Exception:
                pass
        files.append(p)

    for target in targets:
        if target.is_file():
            _maybe_add(target)
            continue
        for p in target.rglob("*"):
            _maybe_add(p)

    files.sort(key=lambda x: str(x))
    # de-dupe
    unique: List[Path] = []
    seen = set()
    for f in files:
        key = str(f)
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique


def _content_to_text(content: Optional[List[Any]]) -> str:
    """
    Convert Serena MCP tool result content to text (best-effort).
    """
    if not content:
        return ""
    chunks: List[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
            continue
        json_value = getattr(item, "json", None)
        if json_value is not None:
            try:
                chunks.append(json.dumps(json_value))
                continue
            except Exception:
                pass
        chunks.append(str(item))
    return "\n".join(chunks)


def _parse_symbol_lines(overview_text: str, max_symbols: int) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of symbol names from Serena's overview text.

    We do not depend on a stable Serena output format; we primarily keep the
    raw overview (truncated) for sharing.
    """
    symbols: List[Dict[str, Any]] = []
    if not overview_text:
        return symbols

    for line in overview_text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        # Skip obvious noise lines
        lower = cleaned.lower()
        if any(
            token in lower
            for token in (
                "traceback",
                "validation error",
                "error",
                "warning",
                "exception",
            )
        ):
            continue

        # Common bullet formats: "- Foo", "* Foo", "• Foo"
        cleaned = cleaned.lstrip("-*• ").strip()
        if not cleaned:
            continue

        # Heuristic: first token is often the symbol name
        name = cleaned.split()[0].strip("():,{}[]")
        if not name or len(name) < 2:
            continue

        symbols.append({"name": name, "line": line.strip()})
        if len(symbols) >= max_symbols:
            break

    # De-duplicate by name, preserving order
    seen = set()
    unique: List[Dict[str, Any]] = []
    for sym in symbols:
        n = sym.get("name")
        if not n or n in seen:
            continue
        seen.add(n)
        unique.append(sym)
    return unique


def _insert_file_into_tree(tree_root: Dict[str, Any], rel_path: str, file_payload: Dict[str, Any]) -> None:
    parts = [p for p in rel_path.split("/") if p]
    node = tree_root
    current_path = []
    for i, part in enumerate(parts):
        current_path.append(part)
        is_leaf = i == len(parts) - 1
        children: List[Dict[str, Any]] = node.setdefault("children", [])

        # Find existing child
        existing = None
        for child in children:
            if child.get("name") == part and child.get("type") == ("file" if is_leaf else "directory"):
                existing = child
                break

        if existing is None:
            existing = {
                "name": part,
                "path": "/".join(current_path),
                "type": "file" if is_leaf else "directory",
            }
            if not is_leaf:
                existing["children"] = []
            children.append(existing)

        if is_leaf:
            existing.update(file_payload)
        node = existing


def _sort_tree(node: Dict[str, Any]) -> None:
    children = node.get("children")
    if not isinstance(children, list):
        return
    children.sort(key=lambda c: (c.get("type") != "directory", c.get("name", "")))
    for child in children:
        _sort_tree(child)


@dataclass
class ExportConfig:
    max_files: int
    per_file_timeout_seconds: float
    concurrency: int
    max_overview_chars: int
    max_symbols_per_file: int
    targets: List[Path]


async def _export_semantic_tree_async(repo_path: Path, config: ExportConfig) -> Dict[str, Any]:
    started_at = time.time()

    if not MCP_AVAILABLE:
        return {
            "error": "MCP library not available (install spec-agent with serena extras)",
            "tree": None,
        }

    files = _collect_code_files_scoped(repo_path, config.targets)
    total_discovered = len(files)
    if config.max_files > 0:
        files = files[: config.max_files]

    tree_root: Dict[str, Any] = {"name": ".", "path": ".", "type": "directory", "children": []}

    serena_cmd = _get_serena_mcp_command(repo_path)
    server_params = StdioServerParameters(command=serena_cmd[0], args=serena_cmd[1:])

    semaphore = asyncio.Semaphore(max(1, config.concurrency))

    file_results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=60.0)

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}

            if "get_symbols_overview" not in tool_names:
                return {
                    "error": "Serena MCP server does not expose get_symbols_overview",
                    "available_tools": sorted(tool_names),
                    "tree": None,
                }

            async def _process_file(path: Path) -> None:
                rel = str(path.relative_to(repo_path)).replace("\\", "/")
                async with semaphore:
                    try:
                        coro = session.call_tool(
                            "get_symbols_overview",
                            arguments={"relative_path": rel},
                        )
                        result = await asyncio.wait_for(coro, timeout=config.per_file_timeout_seconds)
                        overview_text = _content_to_text(result.content)
                        overview_trimmed = (
                            overview_text[: config.max_overview_chars] if config.max_overview_chars > 0 else overview_text
                        )
                        symbols = _parse_symbol_lines(overview_text, config.max_symbols_per_file)

                        payload = {
                            "overview": overview_trimmed,
                            "symbol_count_estimate": len(symbols),
                            "symbols": symbols,
                        }
                        _insert_file_into_tree(
                            tree_root,
                            rel_path=rel,
                            file_payload=payload,
                        )
                        file_results.append({"path": rel, "symbols": len(symbols)})
                    except asyncio.TimeoutError:
                        failures.append({"path": rel, "error": "timeout"})
                    except Exception as exc:
                        failures.append({"path": rel, "error": str(exc)})

            # Process files in a bounded-concurrency way
            await asyncio.gather(*(_process_file(p) for p in files))

    _sort_tree(tree_root)
    elapsed = time.time() - started_at

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_path": str(repo_path),
        "config": {
            "max_files": config.max_files,
            "per_file_timeout_seconds": config.per_file_timeout_seconds,
            "concurrency": config.concurrency,
            "max_overview_chars": config.max_overview_chars,
            "max_symbols_per_file": config.max_symbols_per_file,
        },
        "stats": {
            "discovered_files": total_discovered,
            "indexed_files": len(file_results),
            "failed_files": len(failures),
            "elapsed_seconds": round(elapsed, 3),
        },
        "failures": failures[:200],
        "tree": tree_root,
    }


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Serena semantic symbol tree as JSON")
    parser.add_argument("repo_path", type=str, help="Path to repository")
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Limit export to a file/dir under the repo (repeatable). Paths are relative to repo_path unless absolute.",
    )
    parser.add_argument("--max-files", type=int, default=_read_int_env("SPEC_AGENT_SERENA_SEMANTIC_TREE_MAX_FILES", 1500))
    parser.add_argument(
        "--per-file-timeout-seconds",
        type=float,
        default=_read_float_env("SPEC_AGENT_SERENA_SEMANTIC_TREE_PER_FILE_TIMEOUT", 12.0),
    )
    parser.add_argument("--concurrency", type=int, default=_read_int_env("SPEC_AGENT_SERENA_SEMANTIC_TREE_CONCURRENCY", 4))
    parser.add_argument(
        "--max-overview-chars",
        type=int,
        default=_read_int_env("SPEC_AGENT_SERENA_SEMANTIC_TREE_MAX_OVERVIEW_CHARS", 2000),
    )
    parser.add_argument(
        "--max-symbols-per-file",
        type=int,
        default=_read_int_env("SPEC_AGENT_SERENA_SEMANTIC_TREE_MAX_SYMBOLS_PER_FILE", 200),
    )
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(json.dumps({"error": f"Repository path does not exist: {repo_path}"}))
        return 1

    scoped_targets = _normalize_targets(repo_path, list(args.target or []))

    config = ExportConfig(
        max_files=args.max_files,
        per_file_timeout_seconds=args.per_file_timeout_seconds,
        concurrency=args.concurrency,
        max_overview_chars=args.max_overview_chars,
        max_symbols_per_file=args.max_symbols_per_file,
        targets=scoped_targets,
    )

    try:
        payload = asyncio.run(_export_semantic_tree_async(repo_path, config))
        print(json.dumps(payload, indent=2))
        return 0 if not payload.get("error") else 2
    except KeyboardInterrupt:
        print(json.dumps({"error": "interrupted"}))
        return 130
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

