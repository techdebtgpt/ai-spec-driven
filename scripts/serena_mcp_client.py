#!/usr/bin/env python3
"""
Real Serena MCP client that generates patches via the Serena MCP server.

This script bridges Spec Agent's JSON payload format to Serena's MCP protocol.
It expects Serena MCP server to be running and accessible.

Usage:
    # Set SERENA_MCP_SERVER_URL if your server is not at the default location
    export SERENA_MCP_SERVER_URL="http://localhost:8000"
    
    # Use as delegate
    export SERENA_PATCH_DELEGATE="python scripts/serena_mcp_client.py"
    
    # Or use directly
    echo '{"repo_path":"/path/to/repo","plan_id":"...","step_description":"..."}' \
      | python scripts/serena_mcp_client.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import httpx
except ImportError:
    httpx = None

try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None
    stdio_client = None


def _call_serena_via_mcp_stdio(
    repo_path: Path, plan_id: str, step_description: str
) -> Dict[str, Any]:
    """
    Call Serena MCP server via stdio transport.
    
    Assumes Serena is available as 'serena' command or via uvx.
    """
    # Try to find Serena command
    serena_cmd = os.getenv("SERENA_CMD", "serena")
    
    # Check if we should use uvx
    if not _command_exists(serena_cmd):
        serena_cmd = "uvx"
        serena_args = ["--from", "git+https://github.com/oraios/serena", "serena"]
    else:
        serena_args = []
    
    # Build the prompt for Serena
    prompt = f"""Generate a code patch for the following task:

Repository: {repo_path}
Plan ID: {plan_id}
Step: {step_description}

Please provide:
1. A unified diff showing the changes
2. A rationale explaining why this change is correct
3. Alternative approaches if applicable

Focus on incremental changes (<30 lines of code)."""
    
    try:
        # Call Serena MCP server via stdio
        # This is a simplified version - adjust based on actual Serena MCP API
        result = subprocess.run(
            [serena_cmd] + serena_args + ["mcp", "call", "propose_patch"],
            input=json.dumps({
                "repo_path": str(repo_path),
                "prompt": prompt,
            }),
            text=True,
            capture_output=True,
            check=True,
            timeout=120,
        )
        
        response = json.loads(result.stdout)
        return {
            "diff": response.get("diff", ""),
            "rationale": response.get("rationale", "Generated via Serena MCP."),
            "alternatives": response.get("alternatives", []),
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Serena MCP call failed: {exc}") from exc


def _call_serena_via_http(
    repo_path: Path, plan_id: str, step_description: str
) -> Dict[str, Any]:
    """
    Call Serena MCP server via HTTP (if it exposes an HTTP endpoint).
    """
    if httpx is None:
        raise RuntimeError("httpx is required for HTTP transport. Install with: pip install httpx")
    
    server_url = os.getenv("SERENA_MCP_SERVER_URL", "http://localhost:8000")
    
    prompt = f"""Generate a code patch for the following task:

Repository: {repo_path}
Plan ID: {plan_id}
Step: {step_description}

Please provide:
1. A unified diff showing the changes
2. A rationale explaining why this change is correct
3. Alternative approaches if applicable

Focus on incremental changes (<30 lines of code)."""
    
    try:
        response = httpx.post(
            f"{server_url}/mcp/call",
            json={
                "method": "propose_patch",
                "params": {
                    "repo_path": str(repo_path),
                    "prompt": prompt,
                },
            },
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "diff": data.get("result", {}).get("diff", ""),
            "rationale": data.get("result", {}).get("rationale", "Generated via Serena MCP."),
            "alternatives": data.get("result", {}).get("alternatives", []),
        }
    except Exception as exc:
        raise RuntimeError(f"Serena HTTP call failed: {exc}") from exc


def _call_serena_via_python_mcp(
    repo_path: Path, plan_id: str, step_description: str
) -> Dict[str, Any]:
    """
    Call Serena using Python MCP client library.
    """
    if ClientSession is None or stdio_client is None:
        raise RuntimeError(
            "MCP client library is required. Install with: pip install mcp"
        )
    
    serena_cmd = os.getenv("SERENA_CMD", "serena")
    if not _command_exists(serena_cmd):
        serena_cmd = "uvx"
    
    # This is a placeholder - adjust based on actual MCP client API
    # The actual implementation depends on Serena's MCP server interface
    raise NotImplementedError(
        "Python MCP client integration not yet implemented. "
        "Use stdio or HTTP transport instead."
    )


def _command_exists(cmd: str) -> bool:
    """Check if a command exists in PATH."""
    try:
        subprocess.run(
            ["which", cmd] if sys.platform != "win32" else ["where", cmd],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main() -> int:
    """Main entry point."""
    payload = json.loads(sys.stdin.read() or "{}")
    repo_path = Path(payload.get("repo_path", "."))
    plan_id = payload.get("plan_id", "unknown-plan")
    step_description = payload.get("step_description", "unspecified step")
    
    # Determine transport method
    transport = os.getenv("SERENA_TRANSPORT", "stdio").lower()
    
    try:
        if transport == "http":
            response = _call_serena_via_http(repo_path, plan_id, step_description)
        elif transport == "mcp":
            response = _call_serena_via_python_mcp(repo_path, plan_id, step_description)
        else:  # stdio (default)
            response = _call_serena_via_mcp_stdio(repo_path, plan_id, step_description)
        
        json.dump(response, sys.stdout)
        return 0
    except Exception as exc:
        error_response = {
            "diff": "",
            "rationale": f"Serena integration failed: {exc}",
            "alternatives": [
                "Check that Serena MCP server is running",
                "Verify SERENA_CMD or SERENA_MCP_SERVER_URL environment variables",
                "Review error logs for details",
            ],
        }
        json.dump(error_response, sys.stdout)
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

