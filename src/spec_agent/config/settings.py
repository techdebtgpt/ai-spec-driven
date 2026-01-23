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
    llm_provider: str = "openai"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_timeout_seconds: Optional[int] = None
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: Optional[str] = None
    openai_timeout_seconds: int = 60
    prefer_external_edits: bool = True

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

    # Optional: isolate Spec Agent state (tasks/index/logs) per repo/branch/session.
    # Example:
    #   export SPEC_AGENT_STATE_DIR="$HOME/.spec_agent/cardstore-api/UB-10496"
    raw_state_dir = os.getenv("SPEC_AGENT_STATE_DIR")
    if raw_state_dir:
        try:
            settings.state_dir = Path(raw_state_dir).expanduser().resolve()
        except Exception:
            # Fall back to default if invalid path
            pass

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

    llm_provider = os.getenv("SPEC_AGENT_LLM_PROVIDER")
    if llm_provider:
        settings.llm_provider = llm_provider.lower()

    llm_api_key = os.getenv("SPEC_AGENT_LLM_API_KEY")
    if llm_api_key:
        settings.llm_api_key = llm_api_key

    llm_model = os.getenv("SPEC_AGENT_LLM_MODEL")
    if llm_model:
        settings.llm_model = llm_model

    llm_base = os.getenv("SPEC_AGENT_LLM_BASE_URL")
    if llm_base:
        settings.llm_base_url = llm_base

    llm_timeout = os.getenv("SPEC_AGENT_LLM_TIMEOUT")
    if llm_timeout:
        try:
            settings.llm_timeout_seconds = int(llm_timeout)
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

    prefer_external = os.getenv("SPEC_AGENT_EXTERNAL_EDITS_ONLY")
    if prefer_external is not None:
        settings.prefer_external_edits = prefer_external.lower() in {"1", "true", "yes", "on"}

    return settings
