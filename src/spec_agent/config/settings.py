from __future__ import annotations

from dataclasses import dataclass
import os
import shlex
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple


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
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: Optional[str] = None
    openai_timeout_seconds: int = 60

    DEFAULT_LANGUAGES: ClassVar[Dict[str, tuple[str, ...]]] = {
        "python": (".py", ".pyw", ".pyi"),
        "typescript": (".ts", ".tsx"),
        "javascript": (".js", ".jsx", ".mjs", ".cjs"),
        "csharp": (".cs", ".csx"),
        "java": (".java",),
        "go": (".go",),
        "rust": (".rs",),
        "cpp": (".cpp", ".cc", ".cxx", ".hpp", ".h"),
        "c": (".c", ".h"),
        "shell": (".sh", ".bash", ".zsh"),
        "powershell": (".ps1", ".psm1", ".psd1"),
        "ruby": (".rb",),
        "php": (".php",),
        "swift": (".swift",),
        "kotlin": (".kt", ".kts"),
        "scala": (".scala",),
        "markdown": (".md", ".markdown"),
        "yaml": (".yaml", ".yml"),
        "json": (".json",),
        "xml": (".xml",),
        "html": (".html", ".htm"),
        "css": (".css", ".scss", ".sass"),
        "terraform": (".tf", ".tfvars"),
        "dockerfile": (".dockerfile",),
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

    openai_api_key = os.getenv("SPEC_AGENT_OPENAI_API_KEY")
    if openai_api_key:
        settings.openai_api_key = openai_api_key

    openai_model = os.getenv("SPEC_AGENT_OPENAI_MODEL")
    if openai_model:
        settings.openai_model = openai_model

    openai_base = os.getenv("SPEC_AGENT_OPENAI_BASE_URL")
    if openai_base:
        settings.openai_base_url = openai_base

    openai_timeout = os.getenv("SPEC_AGENT_OPENAI_TIMEOUT")
    if openai_timeout:
        try:
            settings.openai_timeout_seconds = int(openai_timeout)
        except ValueError:
            pass

    return settings


