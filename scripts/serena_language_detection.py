#!/usr/bin/env python3
"""
Use Serena MCP to detect languages in a repository.

This script connects to Serena's MCP server and uses its tools to analyze
the repository structure and detect programming languages.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Set up logging - use WARNING level to suppress verbose INFO logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None
    stdio_client = None


def _get_serena_mcp_command(repo_path: Path) -> list[str]:
    """Get the command to start Serena MCP server with the project path."""
    cmd = os.getenv("SERENA_MCP_COMMAND")
    if cmd:
        import shlex
        base_cmd = shlex.split(cmd)
        if "--project" not in base_cmd:
            base_cmd.extend(["--project", str(repo_path)])
        return base_cmd
    
    # Try to find uvx first
    import shutil
    uvx_path = shutil.which("uvx")
    if uvx_path:
        return [
            uvx_path,
            "--from", "git+https://github.com/oraios/serena",
            "serena", "start-mcp-server",
            "--project", str(repo_path),
            "--transport", "stdio",
            "--enable-web-dashboard", "false",
            "--enable-gui-log-window", "false",
        ]
    
    # Try to find serena directly
    serena_path = shutil.which("serena")
    if serena_path:
        return [
            serena_path,
            "start-mcp-server",
            "--project", str(repo_path),
            "--transport", "stdio",
            "--enable-web-dashboard", "false",
            "--enable-gui-log-window", "false",
        ]
    
    # Default: use uvx (will fail with clear error if not found)
    return [
        "uvx",
        "--from", "git+https://github.com/oraios/serena",
        "serena", "start-mcp-server",
        "--project", str(repo_path),
        "--transport", "stdio",
        "--enable-web-dashboard", "false",
        "--enable-gui-log-window", "false",
    ]


CODE_FILE_PATTERN = re.compile(
    r"([A-Za-z0-9_\-./\\]+?\.(?:py|pyi|pyw|ts|tsx|js|jsx|mjs|c|cc|cpp|cxx|h|hpp|cs|java|go|rb|php|rs|swift|kt|kts|scala|tf|yaml|yml|json))"
)


def _module_name_from_path(path_str: str) -> str:
    normalized = path_str.replace("\\", "/").lstrip("./")
    if not normalized:
        return "root"
    parts = normalized.split("/")
    if len(parts) > 1 and parts[0]:
        return parts[0]
    filename = Path(normalized).stem
    return filename or normalized


def _extract_paths_from_text(text: str) -> List[str]:
    if not text:
        return []
    matches = CODE_FILE_PATTERN.findall(text)
    paths: List[str] = []
    for match in matches:
        candidate = match.strip().strip("',.:()[]")
        if not candidate:
            continue
        cleaned = candidate.replace("\\", "/")
        if cleaned not in paths:
            paths.append(cleaned)
    return paths


def _parse_symbol_names(text: str, max_symbols: int = 8) -> List[str]:
    symbols: List[str] = []
    if not text:
        return symbols
    for line in text.splitlines():
        cleaned = line.strip(" -*\t")
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(keyword in lowered for keyword in ["error", "warning", "validation", "traceback"]):
            continue
        cleaned = cleaned.replace("()", "")
        tokens = cleaned.replace(":", " ").split()
        if not tokens:
            continue
        # Prefer tokens that resemble symbol identifiers
        candidate = tokens[-1]
        if candidate.lower() in {"class", "def", "function", "module", "namespace"} and len(tokens) > 1:
            candidate = tokens[-2]
        candidate = candidate.strip("{}()")
        if not candidate:
            continue
        if candidate not in symbols:
            symbols.append(candidate)
        if len(symbols) >= max_symbols:
            break
    return symbols


def _content_to_text(content: Optional[List]) -> str:
    if not content:
        return ""
    chunks: List[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            chunks.append(text)
            continue
        json_value = getattr(item, "json", None)
        if json_value:
            try:
                chunks.append(json.dumps(json_value))
                continue
            except Exception:
                pass
        if hasattr(item, "type") and hasattr(item, "data"):
            chunks.append(str(item.data))
            continue
        chunks.append(str(item))
    return "\n".join(chunks)


async def _detect_languages_async(repo_path: Path) -> Dict[str, any]:
    """
    Use Serena MCP tools to detect languages in the repository.
    Returns a dict with language information.
    """
    start_time = time.time()
    logger.info(f"Starting Serena language detection for: {repo_path}")
    
    if not MCP_AVAILABLE:
        logger.error("MCP library not available")
        return {"languages": [], "modules": [], "namespaces": [], "top_directories": [], "error": "MCP library not available"}
    
    serena_cmd = _get_serena_mcp_command(repo_path)
    logger.info(f"Serena command: {' '.join(serena_cmd)}")
    
    # Create server parameters for stdio transport
    server_params = StdioServerParameters(
        command=serena_cmd[0],
        args=serena_cmd[1:],
    )
    
    try:
        logger.info("Connecting to Serena MCP server...")
        connect_start = time.time()
        
        # Use a timeout for the entire operation (compatible with Python < 3.11)
        async with stdio_client(server_params) as (read, write):
            connect_time = time.time() - connect_start
            logger.info(f"Connected to Serena MCP server in {connect_time:.2f}s")
            
            async with ClientSession(read, write) as session:
                # Initialize the session with timeout
                init_start = time.time()
                try:
                    logger.info("Initializing Serena session...")
                    await asyncio.wait_for(session.initialize(), timeout=45.0)
                    init_time = time.time() - init_start
                    logger.info(f"Session initialized in {init_time:.2f}s")
                except asyncio.TimeoutError:
                    logger.error(f"Session initialization timed out after {time.time() - init_start:.2f}s")
                    return {"languages": [], "modules": [], "namespaces": [], "top_directories": [], "error": "Serena MCP initialization timed out"}
                
                # List available tools
                tools_start = time.time()
                logger.info("Listing available tools...")
                tools_result = await session.list_tools()
                tools = {tool.name: tool for tool in tools_result.tools}
                tools_time = time.time() - tools_start
                logger.info(f"Found {len(tools)} tools in {tools_time:.2f}s: {', '.join(list(tools.keys())[:5])}...")
                
                language_info = {
                    "languages": [],
                    "file_extensions": {},
                    "project_type": None,
                    "modules": [],
                    "namespaces": [],
                    "top_directories": [],
                    "semantic_modules": [],
                    "semantic_dependencies": [],
                }
                
                # Try to get symbols overview which often includes language info
                if "get_symbols_overview" in tools:
                    try:
                        overview_result = await session.call_tool(
                            "get_symbols_overview",
                            arguments={},
                        )
                        if overview_result.content:
                            for item in overview_result.content:
                                text = item.text if hasattr(item, 'text') else str(item)
                                # Parse the overview to extract language information
                                # This is format-dependent, so we'll do best-effort parsing
                                if text:
                                    language_info["raw_overview"] = text[:500]  # First 500 chars
                    except Exception as exc:
                        sys.stderr.write(f"Warning: get_symbols_overview failed: {exc}\n")
                
                # Try to list directory to see file types
                if "list_dir" in tools:
                    try:
                        dir_start = time.time()
                        logger.info("Calling list_dir...")
                        dir_result = await session.call_tool(
                            "list_dir",
                            arguments={"relative_path": "."},
                        )
                        dir_time = time.time() - dir_start
                        logger.info(f"list_dir completed in {dir_time:.2f}s")
                        if dir_result.content:
                            # Analyze file extensions from directory listing
                            extensions = set()
                            for item in dir_result.content:
                                text = item.text if hasattr(item, 'text') else str(item)
                                # Skip error messages
                                if 'error' in text.lower() or 'validation error' in text.lower():
                                    continue
                                # Extract file extensions from the listing
                                for line in text.split('\n'):
                                    line = line.strip()
                                    # Also look for .csproj files in the listing
                                    if '.csproj' in line.lower() and line not in csproj_files:
                                        # Extract the .csproj filename
                                        parts = line.split()
                                        for part in parts:
                                            if '.csproj' in part.lower():
                                                # Clean up the path
                                                csproj_path = part.strip('"').strip("'")
                                                if csproj_path not in csproj_files:
                                                    csproj_files.append(csproj_path)
                                                    logger.info(f"Found .csproj file in list_dir: {csproj_path}")
                                    if '.' in line and '/' not in line[:10]:  # Likely a filename
                                        parts = line.split()
                                        for part in parts:
                                            if '.' in part:
                                                ext = part.split('.')[-1].lower()
                                                if len(ext) <= 5 and ext.isalpha():
                                                    extensions.add(ext)
                            
                            # Map extensions to languages
                            extension_to_lang = {
                                'py': 'python', 'pyw': 'python', 'pyi': 'python',
                                'ts': 'typescript', 'tsx': 'typescript',
                                'js': 'javascript', 'jsx': 'javascript', 'mjs': 'javascript',
                                'cs': 'csharp', 'csx': 'csharp',
                                'java': 'java',
                                'go': 'go',
                                'rs': 'rust',
                                'cpp': 'cpp', 'cc': 'cpp', 'cxx': 'cpp', 'hpp': 'cpp',
                                'c': 'c', 'h': 'c',
                                'rb': 'ruby',
                                'php': 'php',
                                'swift': 'swift',
                                'kt': 'kotlin', 'kts': 'kotlin',
                                'scala': 'scala',
                                'tf': 'terraform', 'tfvars': 'terraform',
                                'yaml': 'yaml', 'yml': 'yaml',
                                'json': 'json',
                                'xml': 'xml',
                                'html': 'html', 'htm': 'html',
                                'css': 'css', 'scss': 'css', 'sass': 'css',
                            }
                            
                            detected_languages = set()
                            for ext in extensions:
                                if ext in extension_to_lang:
                                    detected_languages.add(extension_to_lang[ext])
                            
                            language_info["languages"] = sorted(list(detected_languages))
                            language_info["file_extensions"] = {ext: extension_to_lang.get(ext, ext) for ext in extensions}
                    except Exception as exc:
                        logger.warning(f"list_dir failed: {exc}")
                
                # Try to find project files that indicate language
                project_file_patterns = {
                    'package.json': 'javascript',
                    'package-lock.json': 'javascript',
                    'yarn.lock': 'javascript',
                    'requirements.txt': 'python',
                    'pyproject.toml': 'python',
                    'Pipfile': 'python',
                    'go.mod': 'go',
                    'Cargo.toml': 'rust',
                    'pom.xml': 'java',
                    'build.gradle': 'java',
                    '*.csproj': 'csharp',
                    '*.sln': 'csharp',
                    '*.xcodeproj': 'swift',
                    'Gemfile': 'ruby',
                    'composer.json': 'php',
                }
                
                csproj_files = []
                if "find_file" in tools:
                    find_start = time.time()
                    logger.info(f"Searching for project files with {len(project_file_patterns)} patterns...")
                    for i, (pattern, lang) in enumerate(project_file_patterns.items()):
                        try:
                            pattern_start = time.time()
                            result = await session.call_tool(
                                "find_file",
                                arguments={
                                    "file_mask": pattern,
                                    "relative_path": ".",
                                },
                            )
                            pattern_time = time.time() - pattern_start
                            if pattern_time > 1.0:  # Log slow patterns
                                logger.warning(f"Pattern '{pattern}' took {pattern_time:.2f}s")
                            if result.content:
                                for item in result.content:
                                    text = item.text if hasattr(item, 'text') else str(item)
                                    if text.strip():
                                        if lang not in language_info["languages"]:
                                            language_info["languages"].append(lang)
                                        # Detect project type
                                        if not language_info["project_type"]:
                                            if pattern.endswith('.csproj') or pattern.endswith('.sln'):
                                                language_info["project_type"] = ".NET"
                                            elif pattern == 'package.json':
                                                language_info["project_type"] = "Node.js"
                                            elif pattern in ['requirements.txt', 'pyproject.toml']:
                                                language_info["project_type"] = "Python"
                                        
                                        # Collect .csproj files to extract namespaces
                                        if pattern.endswith('.csproj'):
                                            # Parse JSON response if it's JSON
                                            try:
                                                parsed = json.loads(text)
                                                if isinstance(parsed, dict) and "files" in parsed:
                                                    csproj_files.extend(parsed["files"])
                                                elif isinstance(parsed, list):
                                                    csproj_files.extend(parsed)
                                            except (json.JSONDecodeError, TypeError):
                                                # Not JSON, treat as plain text
                                                for line in text.split('\n'):
                                                    line = line.strip()
                                                    if '.csproj' in line.lower() and line not in csproj_files:
                                                        # Extract just the filename/path
                                                        if '/' in line:
                                                            csproj_files.append(line.split()[-1] if ' ' in line else line)
                                                        else:
                                                            csproj_files.append(line)
                        except Exception as exc:
                            logger.debug(f"Pattern '{pattern}' failed: {exc}")
                            pass  # Continue with other patterns
                    
                    find_time = time.time() - find_start
                    logger.info(f"find_file searches completed in {find_time:.2f}s")
                
                # For C# projects, extract namespaces from .csproj file names
                # .csproj files often follow the pattern: Namespace.Project.csproj
                if csproj_files and "csharp" in language_info.get("languages", []):
                    logger.info(f"Found {len(csproj_files)} .csproj files: {csproj_files[:5]}")
                    for csproj in csproj_files:
                        # Extract namespace from filename like "Pbp.Payments.CardStore.Api.csproj"
                        # Handle both full paths and just filenames
                        csproj_str = str(csproj)
                        filename = csproj_str.split('/')[-1] if '/' in csproj_str else csproj_str
                        # Clean up filename (remove any extra whitespace or quotes)
                        filename = filename.strip().strip('"').strip("'")
                        if filename.endswith('.csproj'):
                            # Remove .csproj extension
                            name_without_ext = filename[:-7]
                            # Split by dots to get namespace components
                            parts = name_without_ext.split('.')
                            # Take first 2 parts as namespace (e.g., "Pbp.Payments" from "Pbp.Payments.CardStore.Api")
                            if len(parts) >= 2:
                                namespace = '.'.join(parts[:2])
                                if namespace not in language_info["namespaces"]:
                                    language_info["namespaces"].append(namespace)
                                    logger.info(f"Extracted namespace: {namespace} from {filename}")
                            # Also add the project name as a module
                            if len(parts) >= 1:
                                project_name = parts[-1]
                                if project_name not in language_info["modules"]:
                                    language_info["modules"].append(project_name)
                                    logger.info(f"Extracted module: {project_name} from {filename}")
                elif "csharp" in language_info.get("languages", []):
                    logger.warning("C# detected but no .csproj files found")
                
                # Try to detect modules/namespaces using get_symbols_overview
                # Note: get_symbols_overview requires a file path, not a directory
                # So we'll skip it for now and rely on .csproj file parsing
                # If we have .cs files, we could call it on specific files, but that's expensive
                # For now, skip get_symbols_overview and rely on .csproj parsing
                if False and "get_symbols_overview" in tools:  # Disabled - requires file path, not directory
                    try:
                        overview_start = time.time()
                        logger.info("Calling get_symbols_overview...")
                        # This would need a specific file path, not "."
                        overview_result = await session.call_tool(
                            "get_symbols_overview",
                            arguments={"relative_path": "."},
                        )
                        overview_time = time.time() - overview_start
                        logger.info(f"get_symbols_overview completed in {overview_time:.2f}s")
                        if overview_result.content:
                            modules_set = set(language_info.get("modules", []))  # Preserve existing modules
                            namespaces_set = set(language_info.get("namespaces", []))  # Preserve existing namespaces
                            for item in overview_result.content:
                                text = item.text if hasattr(item, 'text') else str(item)
                                # Skip error messages
                                if 'error' in text.lower() or 'validation error' in text.lower():
                                    continue
                                # Parse symbols to extract namespaces/modules
                                # Format varies, but we look for common patterns
                                for line in text.split('\n'):
                                    line = line.strip()
                                    # C# namespace pattern: namespace X.Y.Z
                                    if 'namespace' in line.lower():
                                        parts = line.split()
                                        if len(parts) > 1:
                                            ns = parts[1].rstrip('{').strip()
                                            if ns:
                                                namespaces_set.add(ns)
                                    # Python module pattern: from X.Y import or import X.Y
                                    if 'from' in line.lower() or 'import' in line.lower():
                                        # Extract module name
                                        if 'from' in line.lower():
                                            parts = line.split('from')
                                            if len(parts) > 1:
                                                mod = parts[1].split()[0].split('.')[0]
                                                if mod and not mod.startswith('#'):
                                                    modules_set.add(mod)
                                        elif 'import' in line.lower():
                                            parts = line.split('import')
                                            if len(parts) > 1:
                                                mod = parts[1].split()[0].split('.')[0]
                                                if mod and not mod.startswith('#'):
                                                    modules_set.add(mod)
                            
                            # Merge with existing, don't overwrite
                            language_info["modules"] = sorted(list(modules_set))[:10]
                            language_info["namespaces"] = sorted(list(namespaces_set))[:10]
                    except Exception as exc:
                        logger.warning(f"get_symbols_overview for modules failed: {exc}")
                
                # Try to detect top-level directories that might be modules
                if "list_dir" in tools:
                    try:
                        dir_result = await session.call_tool(
                            "list_dir",
                            arguments={"relative_path": "."},
                        )
                        if dir_result.content:
                            top_dirs = []
                            for item in dir_result.content:
                                text = item.text if hasattr(item, 'text') else str(item)
                                # Skip error messages
                                if 'error' in text.lower() or 'validation error' in text.lower() or 'For further information' in text:
                                    continue
                                # Parse JSON response if it's JSON
                                try:
                                    parsed = json.loads(text)
                                    if isinstance(parsed, dict) and "files" in parsed:
                                        for file_path in parsed["files"]:
                                            if '/' in str(file_path):
                                                first_dir = str(file_path).split('/')[0].strip()
                                                if first_dir and first_dir not in ['', '.', '..'] and first_dir not in top_dirs:
                                                    if not first_dir.startswith('.'):
                                                        top_dirs.append(first_dir)
                                    elif isinstance(parsed, list):
                                        for item in parsed:
                                            if '/' in str(item):
                                                first_dir = str(item).split('/')[0].strip()
                                                if first_dir and first_dir not in ['', '.', '..'] and first_dir not in top_dirs:
                                                    if not first_dir.startswith('.'):
                                                        top_dirs.append(first_dir)
                                except (json.JSONDecodeError, TypeError):
                                    # Not JSON, treat as plain text
                                    for line in text.split('\n'):
                                        line = line.strip()
                                        # Skip error messages and URLs
                                        if 'error' in line.lower() or 'http' in line.lower() or 'For further information' in line:
                                            continue
                                        # Look for directory indicators
                                        if line and not line.startswith('.') and '/' not in line:
                                            # Check if it looks like a module directory
                                            if any(ext in line for ext in ['.csproj', '.py', '.ts', '.js']):
                                                # Extract directory name before extension
                                                dir_name = line.split('.')[0]
                                                if dir_name and dir_name not in top_dirs:
                                                    top_dirs.append(dir_name)
                                        elif '/' in line:
                                            # Extract first directory component
                                            first_dir = line.split('/')[0].strip()
                                            if first_dir and first_dir not in ['', '.', '..'] and first_dir not in top_dirs:
                                                if not first_dir.startswith('.'):
                                                    top_dirs.append(first_dir)
                            
                            language_info["top_directories"] = top_dirs[:10]
                    except Exception as exc:
                        sys.stderr.write(f"Warning: list_dir for modules failed: {exc}\n")
                
                # Try to find symbols to extract namespaces/modules
                if "find_symbol" in tools and language_info.get("languages"):
                    try:
                        # For C# projects, look for common namespace patterns
                        if "csharp" in language_info.get("languages", []):
                            common_namespaces = ["System", "Microsoft", "Pbp", "Api", "Service", "Domain"]
                            for ns_prefix in common_namespaces[:5]:  # Limit to avoid too many calls
                                try:
                                    symbol_result = await session.call_tool(
                                        "find_symbol",
                                        arguments={"symbol_name": ns_prefix},
                                    )
                                    if symbol_result.content:
                                        for item in symbol_result.content:
                                            text = item.text if hasattr(item, 'text') else str(item)
                                            # Extract namespace from symbol results
                                            if 'namespace' in text.lower():
                                                for line in text.split('\n'):
                                                    if 'namespace' in line.lower():
                                                        parts = line.split('namespace')
                                                        if len(parts) > 1:
                                                            ns = parts[1].split()[0].rstrip('{').strip()
                                                            if ns and ns not in language_info["namespaces"]:
                                                                language_info["namespaces"].append(ns)
                                except Exception:
                                    continue
                    except Exception as exc:
                        logger.warning(f"find_symbol for modules failed: {exc}")
                
                # Build semantic module summaries via get_symbols_overview
                if "get_symbols_overview" in tools:
                    candidate_files: List[str] = []

                    async def _collect_candidate_files() -> List[str]:
                        collected: List[str] = []
                        if "find_file" in tools:
                            search_patterns = [
                                "*.py",
                                "*.ts",
                                "*.tsx",
                                "*.js",
                                "*.jsx",
                                "*.cs",
                                "*.java",
                                "*.go",
                                "*.rs",
                            ]
                            for pattern in search_patterns:
                                if len(collected) >= 20:
                                    break
                                try:
                                    result = await session.call_tool(
                                        "find_file",
                                        arguments={
                                            "file_mask": pattern,
                                            "relative_path": ".",
                                        },
                                    )
                                    text = _content_to_text(result.content)
                                    for path in _extract_paths_from_text(text):
                                        if path not in collected:
                                            collected.append(path)
                                        if len(collected) >= 20:
                                            break
                                except Exception:
                                    continue
                        return collected

                    candidate_files = await _collect_candidate_files()

                    module_file_map: Dict[str, str] = {}
                    for file_path in candidate_files:
                        module_name = _module_name_from_path(file_path)
                        if module_name in module_file_map:
                            continue
                        module_file_map[module_name] = file_path
                        if len(module_file_map) >= 5:
                            break

                    semantic_modules: List[Dict[str, any]] = []
                    dependency_tracker: Dict[str, set] = defaultdict(set)

                    async def _collect_references(symbol_name: str, file_path: str) -> Dict[str, List[str]]:
                        referencing_files: List[str] = []
                        referencing_modules: List[str] = []
                        if "find_referencing_symbols" not in tools:
                            return {"files": referencing_files, "modules": referencing_modules}

                        # Try with relative_path hint first, then without if it fails
                        reference_args = [
                            {"symbol_name": symbol_name, "relative_path": file_path},
                            {"symbol_name": symbol_name},
                        ]
                        result = None
                        for args in reference_args:
                            try:
                                result = await session.call_tool(
                                    "find_referencing_symbols",
                                    arguments=args,
                                )
                                break
                            except Exception:
                                result = None
                                continue

                        if not result:
                            return {"files": referencing_files, "modules": referencing_modules}

                        text = _content_to_text(result.content)
                        referencing_files = _extract_paths_from_text(text)[:5]
                        referenced_modules = []
                        for ref_path in referencing_files:
                            module = _module_name_from_path(ref_path)
                            if module not in referenced_modules:
                                referenced_modules.append(module)
                        return {"files": referencing_files, "modules": referenced_modules}

                    for module_name, file_path in module_file_map.items():
                        try:
                            overview_result = await session.call_tool(
                                "get_symbols_overview",
                                arguments={"relative_path": file_path},
                            )
                        except Exception as exc:
                            logger.debug(f"get_symbols_overview failed for {file_path}: {exc}")
                            continue

                        overview_text = _content_to_text(overview_result.content)
                        symbol_names = _parse_symbol_names(overview_text)
                        module_entry = {
                            "module": module_name,
                            "file": file_path,
                            "symbol_count": len(symbol_names),
                            "top_symbols": [],
                            "referenced_by_modules": [],
                            "raw_overview": overview_text[:500] if overview_text else "",
                        }

                        referenced_accumulator: set = set()
                        for symbol_name in symbol_names[:3]:
                            symbol_info = {"symbol": symbol_name}
                            references = await _collect_references(symbol_name, file_path)
                            if references["files"]:
                                symbol_info["referenced_by_files"] = references["files"]
                            if references["modules"]:
                                symbol_info["referenced_by_modules"] = references["modules"]
                                for ref_module in references["modules"]:
                                    referenced_accumulator.add(ref_module)
                                    dependency_tracker[module_name].add(ref_module)
                            module_entry["top_symbols"].append(symbol_info)

                        if referenced_accumulator:
                            module_entry["referenced_by_modules"] = sorted(referenced_accumulator)[:5]

                        semantic_modules.append(module_entry)

                    if semantic_modules:
                        language_info["semantic_modules"] = semantic_modules
                        dependencies_summary = []
                        for module, refs in dependency_tracker.items():
                            dependencies_summary.append(
                                {
                                    "module": module,
                                    "referenced_by": sorted(refs)[:10],
                                    "reference_count": len(refs),
                                }
                            )
                        language_info["semantic_dependencies"] = dependencies_summary

                total_time = time.time() - start_time
                logger.info(f"Serena language detection completed in {total_time:.2f}s")
                logger.info(f"Detected: {len(language_info.get('languages', []))} languages, {len(language_info.get('modules', []))} modules, {len(language_info.get('namespaces', []))} namespaces")
                
                return language_info
    except asyncio.TimeoutError:
        total_time = time.time() - start_time
        logger.error(f"Serena MCP operation timed out after {total_time:.2f}s")
        return {"languages": [], "modules": [], "namespaces": [], "top_directories": [], "error": "Serena MCP operation timed out"}
    except FileNotFoundError as exc:
        total_time = time.time() - start_time
        logger.error(f"Serena MCP server not found after {total_time:.2f}s: {exc}")
        return {"languages": [], "modules": [], "namespaces": [], "top_directories": [], "error": f"Serena MCP server not found: {exc}"}
    except Exception as exc:
        total_time = time.time() - start_time
        logger.error(f"Serena MCP call failed after {total_time:.2f}s: {exc}", exc_info=True)
        return {"languages": [], "modules": [], "namespaces": [], "top_directories": [], "error": f"Serena MCP call failed: {exc}"}


def detect_languages(repo_path: Path) -> Dict[str, any]:
    """
    Synchronous wrapper to detect languages using Serena.
    """
    if not MCP_AVAILABLE:
        return {"languages": [], "error": "MCP library not available"}
    
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # We're in an async context, need to use a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _detect_languages_async(repo_path))
            return future.result()
    except RuntimeError:
        # No running loop, safe to use asyncio.run
        return asyncio.run(_detect_languages_async(repo_path))


def main() -> int:
    """CLI entry point for testing."""
    if len(sys.argv) < 2:
        print("Usage: serena_language_detection.py <repo_path>", file=sys.stderr)
        return 1
    
    repo_path = Path(sys.argv[1])
    if not repo_path.exists():
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        return 1
    
    result = detect_languages(repo_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
