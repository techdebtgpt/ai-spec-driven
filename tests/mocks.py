from __future__ import annotations

from typing import List


class DummyRepo:
    """
    Minimal git repo stub used for orchestrator tests.

    Tests can inject predictable stdout responses so we avoid calling the real git binary.
    """

    def __init__(self) -> None:
        self.branch = "main"
        self.commit = "abc123"
        self.status = ""
        self.applied_diffs: List[str] = []

    def set_branch(self, name: str) -> None:
        self.branch = name

    def set_commit(self, sha: str) -> None:
        self.commit = sha

    def set_status(self, status: str) -> None:
        self.status = status

    def apply(self, diff: str) -> None:
        self.applied_diffs.append(diff)

    # Helpers invoked via monkeypatched subprocess.run in tests.
    def run(self, args: List[str], **kwargs: object):
        cmd = " ".join(args)
        if cmd.endswith("rev-parse --abbrev-ref HEAD"):
            return _CompletedProcess(self.branch)
        if cmd.endswith("rev-parse HEAD"):
            return _CompletedProcess(self.commit)
        if cmd.endswith("status --short"):
            return _CompletedProcess(self.status)
        if cmd.startswith("git apply"):
            self.apply(str(kwargs.get("input", "")))
            return _CompletedProcess("")

        raise AssertionError(f"Unexpected command {cmd}")


class _CompletedProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""

