#!/usr/bin/env python3
"""
Spec-Agent Initialization Script
This script sets up the development environment for the spec-driven-development-agent
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


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
    if not wrapper_script.exists():
        return

    python_name = "python.exe" if sys.platform == "win32" else "python"
    python_bin = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / python_name
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    raw_command = f'{python_bin} {wrapper_script}'
    escaped_command = raw_command.replace('"', '\\"')

    env_path = _upsert_env_entries(
        {
            "SPEC_AGENT_SERENA_ENABLED": "1",
            "SPEC_AGENT_SERENA_COMMAND": f'"{escaped_command}"',
        }
    )
    if env_path:
        print(f"Configured Serena wrapper env at {env_path}")


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
        print(f"Captured OpenAI defaults from environment at {env_path}")


def main():
    # Get the project root directory
    project_root = Path(__file__).parent.resolve()
    venv_path = project_root / ".venv"
    
    print("Initializing Spec-Driven Development Agent...")
    print()
    
    # Check Python version
    print("Checking Python version...")
    if sys.version_info < (3, 11):
        print(f"Error: Python 3.11+ is required. Found Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        sys.exit(1)
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Found Python {python_version}")
    print()
    
    # Create virtual environment if it doesn't exist
    if venv_path.exists():
        print("Virtual environment already exists, skipping creation...")
    else:
        print("Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            print("Virtual environment created")
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
    print("Upgrading pip...")
    try:
        subprocess.run([str(pip_path), "install", "--upgrade", "pip", "--quiet"], check=True)
        print("pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
    print()
    
    # Install the package with dev dependencies
    print("Installing spec-agent with dev dependencies...")
    try:
        subprocess.run([str(pip_path), "install", "-e", f"{project_root}[dev]"], check=True, cwd=project_root)
        print("Installation complete")
    except subprocess.CalledProcessError as e:
        print(f"Error installing package: {e}")
        sys.exit(1)
    print()
    
    # Verify installation
    print("Verifying installation...")
    spec_agent_path = venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "spec-agent"
    
    if spec_agent_path.exists():
        print("spec-agent command is available")
        ensure_serena_env(venv_path, project_root)
        ensure_openai_env_from_envvars()
        print()
        print("Setup complete! You can now use spec-agent.")
        print()
        print("To activate the virtual environment in your current shell, run:")
        print(f"  {activate_cmd}")
        print()
        print("To see available commands, run:")
        print("  spec-agent --help")
        print()
        print("Example usage:")
        print('  spec-agent start /path/to/repo --branch main --description "Your task description"')
    else:
        print("Warning: spec-agent command not found. You may need to activate the venv:")
        print(f"  {activate_cmd}")


if __name__ == "__main__":
    main()

