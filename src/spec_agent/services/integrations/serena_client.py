from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


LOG = logging.getLogger(__name__)


class SerenaToolError(RuntimeError):
    """
    Raised when the Serena subprocess fails or returns malformed payloads.
    """


@dataclass
class SerenaPatchProposal:
    """
    Normalized structure returned by the Serena integration.
    """

    diff: str
    rationale: str
    alternatives: List[str]


class SerenaToolClient:
    """
    Minimal adapter that shells out to a configured Serena command.

    The command must accept a JSON payload via stdin and emit JSON via stdout
    using the following shape:

    {
        "diff": "<unified diff>",
        "rationale": "...",
        "alternatives": ["...", "..."]
    }

    This keeps the integration flexible: teams can point the command to a shell
    script that proxies `serena` MCP calls, or even to a mock during tests.
    """

    def __init__(self, command: Sequence[str], timeout_seconds: int) -> None:
        if not command:
            raise ValueError("Serena command must be provided when integration is enabled.")
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    def request_patch(self, repo_path: Path, step_description: str, plan_id: str) -> SerenaPatchProposal:
        payload = {
            "repo_path": str(repo_path),
            "plan_id": plan_id,
            "step_description": step_description,
        }

        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                check=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:  # pragma: no cover - timeout path
            raise SerenaToolError(f"Serena command timed out: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            LOG.error("Serena command failed: %s", exc.stderr)
            raise SerenaToolError("Serena command failed; see stderr for details.") from exc

        try:
            response = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise SerenaToolError("Serena command returned invalid JSON.") from exc

        diff = response.get("diff")
        rationale = response.get("rationale", "Generated via Serena integration.")
        alternatives = response.get("alternatives", [])
        if not diff:
            raise SerenaToolError("Serena response did not provide a diff.")

        if not isinstance(alternatives, list):
            alternatives = [str(alternatives)]

        return SerenaPatchProposal(diff=diff, rationale=rationale, alternatives=[str(item) for item in alternatives])


