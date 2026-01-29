# Spec Agent (Spec‑Driven Development Agent)

Spec Agent is a **boundary-aware, spec-driven workflow** for making changes in large/legacy repositories. It indexes a repo, turns a change request into a plan + boundary specifications, and then generates **incremental, reviewable patches** you can approve or reject.
# This tool enables spec-driven development workflows for large codebases
# Spec Agent - Purpose

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

# NOTE: When Serena code generation is disabled, the patch queue will emit
# "EXTERNAL_EDIT_REQUIRED" entries. Apply those steps in your editor and sync:
# ./spec-agent sync-external <task-id> --patch-id <patch-id>

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

## MCP Integration (Cursor / Claude Desktop)

Spec Agent can be used as an MCP server, allowing AI assistants like Cursor and Claude to drive the entire spec-driven workflow.

### Setup

1. Install with MCP support:

```bash
pip install -e ".[serena]"  # includes mcp dependency
```

2. Configure your AI assistant (see below).

3. Restart the assistant to load the MCP server.

### Cursor Configuration

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "spec-agent": {
      "command": "/path/to/ai-spec-driven/.venv/bin/python",
      "args": ["-m", "spec_agent.mcp_server"],
      "env": {
        "SPEC_AGENT_OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

```json
{
  "mcpServers": {
    "spec-agent": {
      "command": "/path/to/ai-spec-driven/.venv/bin/python",
      "args": ["-m", "spec_agent.mcp_server"],
      "env": {
        "SPEC_AGENT_OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

### Available MCP Tools

| Category | Tools |
|----------|-------|
| **Repository** | `index_repository`, `get_repository_summary` |
| **Tasks** | `create_task`, `list_tasks`, `get_task_status` |
| **Clarifications** | `get_clarifications`, `answer_clarification` |
| **Planning** | `generate_plan`, `get_boundary_specs`, `approve_spec`, `approve_all_specs`, `skip_spec`, `approve_plan` |
| **Patches** | `generate_patches`, `list_patches`, `get_patch_details`, `get_next_pending_patch`, `approve_patch`, `sync_external_patch`, `reject_patch` |
| **Workflow** | `get_workflow_status` |

### Example Usage in Cursor/Claude

Once configured, you can ask the AI assistant:

> "Index this repository and create a task to add user authentication"

The AI will automatically call the MCP tools to:
1. Index the codebase
2. Create a task with clarifying questions
3. Generate an implementation plan
4. Create boundary specs
5. Generate and apply patches

See `mcp-config-examples/` for more configuration examples.

## Demo: apply patches in Cursor/Claude, then sync back to the dashboard

By default, Spec Agent can apply patches itself via `approve_patch`. For demos where you want
Cursor/Claude to do the code edits (and Spec Agent simply *tracks* what changed), use the external
sync flow:

- **Generate patches** (Spec Agent creates diffs to review)
- **Apply changes in your editor** (Cursor/Claude updates files)
- **Sync back to Spec Agent** so the web dashboard shows the real diff + files touched

### CLI flow

1) Generate patches:

- `./spec-agent generate-patches <task-id>`

2) Apply the change in your editor (Cursor/Claude).

3) Sync the result back to Spec Agent (optionally attach it to a specific patch):

- `./spec-agent sync-external <task-id> --patch-id <patch-id> --client cursor`

If you omit `--patch-id`, Spec Agent will still record the `git diff` in the task history.

### MCP flow (Cursor / Claude Desktop)

After applying changes in the editor, call:

- `sync_external_patch(task_id, patch_id?, client?)`

Then open the web dashboard: the **Code generation** step will display the synced diff and label
the patch as applied via `cursor`/`claude`.

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
- `SPEC_AGENT_EXTERNAL_EDITS_ONLY=1` to skip placeholder files and require `sync-external` for every patch

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
