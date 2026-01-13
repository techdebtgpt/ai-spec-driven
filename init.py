#!/usr/bin/env python3
"""
Spec-Agent Initialization Script
This script sets up the development environment for the spec-driven-development-agent
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


def _upsert_env_entries(entries: Dict[str, str]) -> Optional[Path]:
    """
    Persist key/value pairs to ~/.spec_agent/env, replacing existing lines when necessary.
    """

    entries = {key: value for key, value in entries.items() if value is not None}
    if not entries:
        return None

    state_dir = Path.home() / ".spec_agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    env_path = state_dir / "env"

    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    def upsert(key: str, value: str) -> None:
        entry = f"{key}={value}"
        for idx, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[idx] = entry
                break
        else:
            lines.append(entry)

    for key, value in entries.items():
        upsert(key, value)

    env_path.write_text("\n".join(lines) + "\n")
    return env_path


def ensure_serena_env(venv_path: Path, project_root: Path) -> None:
    """
    Preconfigure Serena wrapper environment variables so future CLI invocations
    automatically shell out to scripts/serena_patch_wrapper.py.
    """

    wrapper_script = project_root / "scripts" / "serena_patch_wrapper.py"
    mcp_integration = project_root / "scripts" / "serena_mcp_integration.py"
    
    if not wrapper_script.exists():
        return

    python_name = "python.exe" if sys.platform == "win32" else "python"
    python_bin = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / python_name
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    raw_command = f'{python_bin} {wrapper_script}'
    escaped_command = raw_command.replace('"', '\\"')
    
    # Set up delegate to MCP integration if it exists
    delegate_command = None
    if mcp_integration.exists():
        delegate_raw = f'{python_bin} {mcp_integration}'
        delegate_command = delegate_raw.replace('"', '\\"')

    env_entries = {
        "SPEC_AGENT_SERENA_ENABLED": "1",
        "SPEC_AGENT_SERENA_COMMAND": f'"{escaped_command}"',
        "SPEC_AGENT_SERENA_TIMEOUT": "120",
    }
    
    if delegate_command:
        env_entries["SERENA_PATCH_DELEGATE"] = f'"{delegate_command}"'

    env_path = _upsert_env_entries(env_entries)
    if env_path:
        # Keep setup output clean; only mention in verbose mode.
        if os.getenv("SPEC_AGENT_SETUP_VERBOSE") == "1":
            print(f"Configured Serena integration at {env_path}")


def ensure_openai_env_from_envvars() -> None:
    """
    If SPEC_AGENT_OPENAI_* vars are present in the current environment, persist them so
    teammates don't need to re-export keys for every CLI run.
    """

    entries = {
        "SPEC_AGENT_OPENAI_API_KEY": os.getenv("SPEC_AGENT_OPENAI_API_KEY"),
        "SPEC_AGENT_OPENAI_MODEL": os.getenv("SPEC_AGENT_OPENAI_MODEL"),
        "SPEC_AGENT_OPENAI_BASE_URL": os.getenv("SPEC_AGENT_OPENAI_BASE_URL"),
        "SPEC_AGENT_OPENAI_TIMEOUT": os.getenv("SPEC_AGENT_OPENAI_TIMEOUT"),
    }
    env_path = _upsert_env_entries(entries)
    if env_path:
        if os.getenv("SPEC_AGENT_SETUP_VERBOSE") == "1":
            print(f"Captured OpenAI defaults from environment at {env_path}")


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    quiet = not verbose
    if verbose:
        os.environ["SPEC_AGENT_SETUP_VERBOSE"] = "1"

    def _print_step(msg: str) -> None:
        print(msg)

    def _print_ok(msg: str) -> None:
        print(f"✓ {msg}")

    def _run(cmd: Sequence[str], *, cwd: Path | None = None, label: str | None = None) -> None:
        """
        Run a command. In quiet mode, suppress stdout/stderr unless it fails.
        """
        if label:
            _print_step(label)
        try:
            if quiet:
                subprocess.run(
                    list(cmd),
                    check=True,
                    cwd=str(cwd) if cwd else None,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                subprocess.run(list(cmd), check=True, cwd=str(cwd) if cwd else None)
        except subprocess.CalledProcessError as exc:
            # Re-run in verbose mode to surface the real error output.
            if quiet:
                print("✗ Failed. Re-running with full output for troubleshooting:\n")
                subprocess.run(list(cmd), check=False, cwd=str(cwd) if cwd else None)
            raise exc

    # Get the project root directory
    project_root = Path(__file__).parent.resolve()
    venv_path = project_root / ".venv"
    
    _print_step("Initializing Spec Agent...")
    if quiet:
        _print_step("(Tip: re-run with --verbose to see full install logs)")
    print()
    
    # Check Python version
    _print_step("Checking Python version...")
    if sys.version_info < (3, 11):
        print(f"Error: Python 3.11+ is required. Found Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        sys.exit(1)
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    _print_ok(f"Found Python {python_version}")
    print()
    
    # Create virtual environment if it doesn't exist
    if venv_path.exists():
        _print_ok("Virtual environment already exists")
    else:
        _print_step("Creating virtual environment...")
        try:
            _run([sys.executable, "-m", "venv", str(venv_path)], label=None)
            _print_ok("Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e}")
            sys.exit(1)
    print()
    
    # Determine the pip executable path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
        activate_cmd = ".venv\\Scripts\\activate"
    else:
        pip_path = venv_path / "bin" / "pip"
        activate_cmd = "source .venv/bin/activate"
    
    # Upgrade pip
    _print_step("Upgrading pip...")
    try:
        args = [str(pip_path), "install", "--upgrade", "pip"]
        if quiet:
            args.append("--quiet")
        _run(args, cwd=project_root, label=None)
        _print_ok("pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
    print()
    
    # Install the package with dev and serena dependencies
    _print_step("Installing dependencies (dev + serena)...")
    try:
        install_cmd = [str(pip_path), "install", "-e", f"{project_root}[dev,serena]"]
        if quiet:
            install_cmd.append("--quiet")
        _run(install_cmd, cwd=project_root, label=None)
        _print_ok("Installation complete")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)
    print()
    
    # Verify MCP library is installed (should be via serena dependencies, but double-check)
    _print_step("Verifying Serena dependencies...")
    try:
        _run([str(pip_path), "show", "mcp"], cwd=project_root, label=None)
        _print_ok("MCP library verified (required for Serena integration)")
    except subprocess.CalledProcessError:
        # MCP not found, try to install it
        _print_step("MCP library not found, installing...")
        try:
            cmd = [str(pip_path), "install", "mcp>=1.12.3"]
            if quiet:
                cmd.append("--quiet")
            _run(cmd, cwd=project_root, label=None)
            _print_ok("MCP library installed")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Could not install MCP library: {e}")
            print("Serena integration will not be available without MCP")
    print()
    
    # Verify installation
    _print_step("Verifying installation...")
    spec_agent_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "spec-agent"
    
    if spec_agent_path.exists():
        _print_ok("spec-agent command is available")
        ensure_serena_env(venv_path, project_root)
        ensure_openai_env_from_envvars()
        print()
        _print_ok("Setup complete! You can now use spec-agent.")
        print()
        print("Next:")
        print(f"- Activate venv: {activate_cmd}")
        print("- Help: spec-agent --help")
        print("- Example: spec-agent start --description \"Your task description\"")
        print()
        print("Optional:")
        print("- Serena requires `uv` installed (e.g. `brew install uv`)")
    else:
        print("Warning: spec-agent command not found. You may need to activate the venv:")
        print(f"  {activate_cmd}")


if __name__ == "__main__":
    main()

