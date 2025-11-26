from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from pathlib import Path
from typing import ClassVar, Dict, Tuple


@dataclass
class AgentSettings:
    """
    Central configuration aligned with the MVP document.
    """

    state_dir: Path = Path.home() / ".spec_agent"
    repo_upper_bound_loc: int = 1_000_000
    indexing_timeout_seconds: int = 60
    max_patch_lines: int = 30
    top_module_count: int = 5
    hotspot_loc_threshold: int = 500
    hotspot_fan_out_threshold: int = 15
    serena_enabled: bool = False
    serena_command: Tuple[str, ...] = ()
    serena_timeout_seconds: int = 120

    DEFAULT_LANGUAGES: ClassVar[Dict[str, tuple[str, ...]]] = {
        "python": (".py",),
        "typescript": (".ts", ".tsx"),
        "javascript": (".js", ".jsx"),
        "shell": (".sh",),
        "markdown": (".md",),
        "go": (".go",),
    }


def get_settings() -> AgentSettings:
    settings = AgentSettings()

    serena_enabled = os.getenv("SPEC_AGENT_SERENA_ENABLED")
    if serena_enabled is not None:
        settings.serena_enabled = serena_enabled.lower() in {"1", "true", "yes"}

    raw_command = os.getenv("SPEC_AGENT_SERENA_COMMAND")
    if raw_command:
        settings.serena_command = tuple(shlex.split(raw_command))

    timeout = os.getenv("SPEC_AGENT_SERENA_TIMEOUT")
    if timeout:
        try:
            settings.serena_timeout_seconds = int(timeout)
        except ValueError:
            pass

    return settings


