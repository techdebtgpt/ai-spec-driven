# Spec-Driven Development Agent
AI-assisted, boundary-aware change-management tool for senior engineers working in single legacy repositories. The repo is organized so each epic in the spec has an obvious home in the codebase and can grow iteratively.

## High-Level Goals
- Context acquisition for large mono-repos, including dependency/call graph stubs, legacy hotspot detection, and incremental retrieval.
- Human-first workflow: clarification questions, change plans, boundary specs, and gated patch generation.
- Incremental, reviewable diffs with rationale, refactor suggestions, and test recommendations.
- Persistent, auditable history of every question, decision, spec, patch, and test suggestion.

## Repository Layout (Iteration 0)
```
spec-driven-development-agent/
├── pyproject.toml
├── README.md
├── src/spec_agent/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli/app.py                     # Typer CLI entrypoints (start/list/plan/etc.)
│   ├── config/settings.py             # Runtime configuration + thresholds
│   ├── domain/models.py               # Dataclasses/enums mirroring the spec data model
│   ├── workflow/orchestrator.py       # State machine implementing the gated lifecycle
│   ├── persistence/store.py           # JSON-backed persistence for tasks/logs/specs
│   ├── tracing/reasoning_log.py       # Structured logging of agent actions
│   └── services/
│       ├── context/indexer.py         # Repository inventory + legacy hotspot scaffolding
│       ├── context/retriever.py       # Iterative context expansion policy
│       ├── planning/clarifier.py      # Clarity assessment + question generation
│       ├── planning/plan_builder.py   # Structured change plan generator
│       ├── specs/boundary_manager.py  # Boundary detection + spec proposal shells
│       ├── patches/engine.py          # Patch step workflow + rationale hooks
│       └── tests/suggester.py         # Test case recommendation surface
└── tests/
    └── test_context_indexer.py
```

> **Iteration focus:** Provide runnable scaffolding (CLI + orchestrator + services + persistence) that reflects all epics, even if components currently stub responses. Each subsequent iteration can flesh out the underlying analysis/LLM layers without reshaping the architecture.

## Getting Started

### Quick Setup (Recommended)
```bash
cd spec-driven-development-agent
./spec-agent --setup
```

This unified launcher will:
- Check Python version (requires Python 3.11+)
- Create a virtual environment (`.venv`)
- Activate the virtual environment
- Install the package with dev dependencies
- Verify the installation

After setup, the same launcher proxies every CLI command. Example:
```bash
./spec-agent start /path/to/repo --branch main --description "Investigate auth bug"
```

### Command Reference
- `./spec-agent --setup`: bootstrap the environment (venv + dependencies) and verify the CLI command.
- `./spec-agent --help`: show available subcommands and options.
- `./spec-agent start <repo> --branch <branch> --description "<text>"`: create a new task and capture contextual data.
- `./spec-agent tasks [--status <status>]`: list recorded tasks, optionally filtered by status.
- `./spec-agent plan <task_id>`: generate the plan, boundary specs, patch queue, and suggested tests for an existing task.

### Alternative: Python Script
```bash
cd spec-driven-development-agent
python3 init.py
```

### Alternative: Bash Script
```bash
cd spec-driven-development-agent
./spec-agent-init.sh
```

### Manual Setup
```bash
cd spec-driven-development-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Usage
```bash
# Prefer the launcher (auto-activates the venv)
./spec-agent --help
./spec-agent tasks
./spec-agent start /path/to/repo --branch main --description "Investigate auth bug"

# Or manually activate the virtual environment
source .venv/bin/activate
spec-agent --help
```

## LLM-Ready Configuration

We now ship the plumbing needed to talk to OpenAI (or an API-compatible gateway) without
enabling it in the CLI yet. This keeps the default CLI behavior deterministic while letting
teammates wire native LLM calls inside their own agents or experiments.

1. Environment variables recognized by `AgentSettings`:
   - `SPEC_AGENT_OPENAI_API_KEY` (required for making calls)
   - `SPEC_AGENT_OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
   - `SPEC_AGENT_OPENAI_BASE_URL` (optional override for Azure/OpenRouter/etc.)
   - `SPEC_AGENT_OPENAI_TIMEOUT` (defaults to 60 seconds)
2. Use `spec_agent.services.llm.openai_client.OpenAILLMClient` to obtain a reusable client.
3. Pass that client into any service that accepts an optional `llm_client` parameter
   (`Clarifier`, `PlanBuilder`, `BoundaryManager`, `TestSuggester`) to opt into LLM behavior.
4. Quick-start checklist for teammates:
   ```bash
   # Copy the template and open it in an editor
   cp spec_agent_env.example ~/.spec_agent/env
   nano ~/.spec_agent/env   # or use VS Code, TextEdit, etc.

   # Fill in your Serena + OpenAI values, then save

   # Optional: rerun setup to verify everything is wired
   ./spec-agent --setup
   ```
   The CLI reads `~/.spec_agent/env` automatically. Alternatively, export `SPEC_AGENT_OPENAI_*` before
   running `./spec-agent --setup` and the bootstrapper will write them for you.

By default, the orchestrator still instantiates these services without an LLM, so the CLI
remains fully offline. Future PRs can enable the integration by constructing the client and
injecting it where needed.

## Optional: Serena-backed LLM Integration

If you plan to plug an LLM into the workflow, you can delegate semantic retrieval and editing to [Serena](https://github.com/oraios/serena). The orchestrator now supports an opt-in mode that shells out to a Serena command for patch generation:

1. Run `./spec-agent --setup` (or `./spec-agent --setup --reconfigure`). The launcher now writes `~/.spec_agent/env` with default Serena settings pointing at the local wrapper, so future `./spec-agent <cmd>` invocations pick them up automatically.
2. Install and configure Serena (for example via `uvx --from git+https://github.com/oraios/serena serena start-mcp-server --help`).
3. Provide a wrapper command that accepts JSON via stdin and emits JSON with `diff`, `rationale`, and `alternatives` keys.
   - Example wrapper: `scripts/serena_patch_wrapper.py` (see below). It can forward requests to another command via `SERENA_PATCH_DELEGATE` or return a deterministic placeholder diff for smoke testing.
4. Override any defaults by editing `~/.spec_agent/env` (e.g., update `SPEC_AGENT_SERENA_COMMAND` or set `SPEC_AGENT_SERENA_TIMEOUT=120`).

When enabled, `PatchEngine` uses the configured Serena command to request diffs per plan step. Failures automatically fall back to the built-in placeholder patches, so the workflow remains stable even if Serena is unavailable.

To verify the wrapper without wiring a full LLM, run:

```bash
# One-off smoke test
echo '{"repo_path":"/Users/Ina/payments/cardStore/cardstore-infrastructure","plan_id":"smoke","step_description":"Demo Serena step"}' \
  | python scripts/serena_patch_wrapper.py | jq

# Full CLI integration (placeholder diff):
export SPEC_AGENT_SERENA_ENABLED=1
export SPEC_AGENT_SERENA_COMMAND="python /Users/Ina/repos/ai-learning/spec-driven-development-agent/scripts/serena_patch_wrapper.py"
./spec-agent plan <task-id>

# To forward to a real Serena MCP workflow, point the wrapper at your command:
SERENA_PATCH_DELEGATE='/path/to/serena-mcp-wrapper --repo ${PWD}' \
  ./spec-agent plan <task-id>
```

## Next Iterations
- Connect `ContextIndexer` to language-specific parsers for dependency + call graphs.
- Implement real LLM calls inside `Clarifier`, `PlanBuilder`, and `BoundaryManager`.
- Add Git integration in `patches.engine` for diff previews and approvals.
- Expand tests to cover workflow transitions, persistence durability, and failure cases.

The project keeps all epic responsibilities visible from day one, making it easier to invest in the highest-risk components without re-architecting later. 


