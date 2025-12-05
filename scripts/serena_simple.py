#!/usr/bin/env python3
"""
Simple Serena integration that calls Serena CLI directly.

This is a practical implementation that works with Serena when it's installed.
It assumes Serena has a CLI command that can generate patches.

Usage:
    # Install Serena first (if not already installed):
    # uvx --from git+https://github.com/oraios/serena serena --help
    
    # Set as delegate:
    export SERENA_PATCH_DELEGATE="python scripts/serena_simple.py"
    
    # Or use directly:
    echo '{"repo_path":"/path/to/repo","plan_id":"...","step_description":"..."}' \
      | python scripts/serena_simple.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _find_serena_command() -> list[str]:
    """Find the Serena command to use."""
    # Check for explicit command
    explicit_cmd = os.getenv("SERENA_CMD")
    if explicit_cmd:
        return shlex.split(explicit_cmd) if isinstance(explicit_cmd, str) else explicit_cmd
    
    # Try direct 'serena' command
    try:
        subprocess.run(["serena", "--version"], capture_output=True, check=True, timeout=5)
        return ["serena"]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    
    # Fall back to uvx
    return ["uvx", "--from", "git+https://github.com/oraios/serena", "serena"]


def _call_serena_cli(repo_path: Path, plan_id: str, step_description: str) -> Dict[str, Any]:
    """
    Call Serena to generate a patch.
    
    Serena is an MCP server, not a direct CLI. This function tries to:
    1. Use Serena's MCP integration script if available
    2. Fall back to trying direct commands (in case Serena adds CLI support)
    
    For proper MCP integration, use serena_mcp_integration.py instead.
    """
    serena_cmd = _find_serena_command()
    
    # Build the instruction for Serena
    instruction = f"""Generate a code patch for:

Step: {step_description}
Repository: {repo_path}
Plan ID: {plan_id}

Requirements:
- Incremental change (<30 lines of code)
- Include unified diff format
- Provide rationale
- Suggest alternatives if applicable"""
    
    try:
        # Pattern 1: Try serena start-mcp-server with a wrapper
        # This is a fallback - proper MCP integration should use serena_mcp_integration.py
        try:
            # Check if we can at least verify Serena is available
            result = subprocess.run(
                serena_cmd + ["start-mcp-server", "--help"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            # If we get here, Serena is available but we need MCP client
            raise RuntimeError(
                "Serena MCP server is available. Use serena_mcp_integration.py for proper integration. "
                "Install MCP library: pip install mcp"
            )
        except subprocess.CalledProcessError:
            # Command exists but --help failed, continue
            pass
        except FileNotFoundError:
            raise RuntimeError(
                "Serena not found. Install with: uvx --from git+https://github.com/oraios/serena serena start-mcp-server --help"
            )
        
        # Pattern 2: Try any direct CLI commands (if Serena adds them in future)
        # For now, Serena is MCP-only, so this will likely fail
        try:
            result = subprocess.run(
                serena_cmd + ["--help"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            # If we see available commands, we could try them
            # For now, guide user to MCP integration
            raise RuntimeError(
                "Serena is available but requires MCP integration. "
                "Use: export SERENA_PATCH_DELEGATE='python scripts/serena_mcp_integration.py'"
            )
        except subprocess.CalledProcessError:
            pass
        
        raise RuntimeError(
            "Serena requires MCP integration. "
            "Install MCP: pip install mcp, then use serena_mcp_integration.py"
        )
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Serena command timed out")
    except RuntimeError:
        raise  # Re-raise our custom errors
    except Exception as exc:
        raise RuntimeError(f"Serena call failed: {exc}") from exc


def main() -> int:
    """Main entry point."""
    payload = json.loads(sys.stdin.read() or "{}")
    repo_path = Path(payload.get("repo_path", "."))
    plan_id = payload.get("plan_id", "unknown-plan")
    step_description = payload.get("step_description", "unspecified step")
    
    if not repo_path.exists():
        error_response = {
            "diff": "",
            "rationale": f"Repository path does not exist: {repo_path}",
            "alternatives": ["Verify the repository path is correct"],
        }
        json.dump(error_response, sys.stdout)
        return 1
    
    try:
        response = _call_serena_cli(repo_path, plan_id, step_description)
        json.dump(response, sys.stdout)
        return 0
    except Exception as exc:
        error_response = {
            "diff": "",
            "rationale": f"Serena integration failed: {exc}",
            "alternatives": [
                "Install Serena: uvx --from git+https://github.com/oraios/serena serena --help",
                "Check that SERENA_CMD environment variable points to Serena executable",
                "Verify Serena CLI is working: serena --version",
                "Review error logs for details",
            ],
        }
        json.dump(error_response, sys.stdout)
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

