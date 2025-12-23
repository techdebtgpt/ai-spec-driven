# Spec Agent (Spec‑Driven Development Agent)

Spec Agent is a **boundary-aware, spec-driven workflow** for making changes in large/legacy repositories. It indexes a repo, turns a change request into a plan + boundary specifications, and then generates **incremental, reviewable patches** you can approve or reject.

## What it does

- **Repository indexing**: builds a lightweight summary (languages, hotspots, etc.) and optionally a semantic snapshot (when enabled).
- **Task workflow**: create a task, answer clarifying questions, generate a plan, and lock scope.
- **Boundary specs**: proposes “contracts” (actors, interfaces, invariants) that gate patch generation.
- **Patch queue**: generates and applies changes as a sequence of small diffs with rationale.
- **Persistence**: stores tasks, indexes, specs, patches, and logs under `~/.spec_agent` (or an overridden state dir).

## Requirements

- Python **3.11+**
- Git (recommended; Spec Agent degrades gracefully if git metadata is unavailable)
- Optional integrations:
  - **Serena** (MCP server) for semantic tooling and patch generation
  - **OpenAI** (or compatible) for LLM-backed planning/rationale generation

## Install / Setup

### Recommended (repo-local launcher)

From the project root:

```bash
./spec-agent --setup
```

This will create `.venv/`, install dependencies, and expose the CLI as `spec-agent` inside the venv. The launcher then proxies all commands:

```bash
./spec-agent --help
```

### Manual (editable install)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
spec-agent --help
```

## Quickstart (non-interactive CLI)

```bash
# 1) Index a repository
./spec-agent index /path/to/repo --branch main

# 2) Create a task
./spec-agent start --description "Describe the change you want"

# 3) Answer clarifications (if any)
./spec-agent clarifications <task-id>

# 4) Generate a plan (and boundary specs)
./spec-agent plan <task-id>

# 5) Review/resolve boundary specs (required before plan approval)
./spec-agent specs <task-id>
./spec-agent approve-spec <task-id> <spec-id>   # or: ./spec-agent skip-spec <task-id> <spec-id>

# 6) Approve the plan and generate patches
./spec-agent approve-plan <task-id>
./spec-agent generate-patches <task-id>

# 7) Review/apply patches
./spec-agent patches <task-id>

# 8) Inspect status
./spec-agent status <task-id>
```

## Quickstart (interactive)

```bash
./spec-agent chat
```

Note: The interactive chat session only starts when you run the `chat` command. Running `./spec-agent` without `chat` uses the standard CLI interface.

## Command overview

Run `./spec-agent --help` for the authoritative list. Common commands:

- **Indexing & context**: `index`, `context-summary`, `bounded-index`, `context`
- **Tasks**: `start`, `tasks`, `task-edit`, `status`, `logs`, `clean-logs`, `clean-tasks`
- **Planning & specs**: `plan`, `specs`, `approve-spec`, `skip-spec`, `spec-edit`, `spec-regenerate`, `approve-plan`
- **Patches**: `generate-patches`, `patches`, `ask-patch`
- **Refactors**: `refactors`
- **Interactive**: `chat`

## State and configuration

By default, Spec Agent stores state under:

- `~/.spec_agent/` (tasks, logs, repository indexes, etc.)
- `~/.spec_agent/env` is sourced by the launcher (`./spec-agent`) if present.

To isolate state (per repo / branch / session), set:

- `SPEC_AGENT_STATE_DIR=/custom/path`

### OpenAI configuration (optional)

- `SPEC_AGENT_OPENAI_API_KEY`
- `SPEC_AGENT_OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `SPEC_AGENT_OPENAI_BASE_URL` (optional)
- `SPEC_AGENT_OPENAI_TIMEOUT` (seconds; default: `60`)

### Serena configuration (optional)

- `SPEC_AGENT_SERENA_ENABLED=1`
- `SPEC_AGENT_SERENA_COMMAND="python $(pwd)/scripts/serena_patch_wrapper.py"`
- `SPEC_AGENT_SERENA_TIMEOUT=120`

Delegate used by the wrapper (recommended):

- `SERENA_PATCH_DELEGATE="python $(pwd)/scripts/serena_mcp_integration.py"`

If you need to override how Serena is started:

- `SERENA_MCP_COMMAND="uvx --from git+https://github.com/oraios/serena serena start-mcp-server"`

## Development

```bash
./spec-agent --setup
./.venv/bin/python -m ruff check .
./.venv/bin/python -m pytest
```
