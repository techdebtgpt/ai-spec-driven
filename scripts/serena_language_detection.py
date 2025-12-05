#!/usr/bin/env python3
"""
Use Serena MCP to detect languages in a repository.

This script connects to Serena's MCP server and uses its tools to analyze
the repository structure and detect programming languages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
