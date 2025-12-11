# Spec-Driven Development Agent
AI-assisted, boundary-aware change-management tool for senior engineers working in single legacy repositories. The repo is organized so each epic in the spec has an obvious home in the codebase and can grow iteratively.

## High-Level Goals
- Context acquisition for large mono-repos, including dependency/call graph stubs, legacy hotspot detection, and incremental retrieval.
- Human-first workflow: clarification questions, change plans, boundary specs, and gated patch generation.
- Incremental, reviewable diffs with rationale, refactor suggestions, and test recommendations.
- Persistent, auditable history of every question, decision, spec, patch, and test suggestion.

## Repository Layout
```
spec-driven-development-agent/
├── pyproject.toml
├── README.md
├── scripts/
│   ├── serena_mcp_integration.py      # Serena MCP server integration for code generation
│   ├── serena_language_detection.py   # Repository language detection utilities
│   ├── serena_mcp_client.py           # MCP client helper utilities
│   ├── serena_patch_wrapper.py        # Wrapper for Serena patch generation
│   └── serena_simple.py               # Simple Serena integration fallback
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
│       ├── context/
│       │   ├── indexer.py             # Repository inventory + legacy hotspot scaffolding
│       │   └── retriever.py           # Iterative context expansion policy
│       ├── integrations/
│       │   └── serena_client.py       # Serena tool client adapter
│       ├── llm/
│       │   └── openai_client.py       # OpenAI LLM client implementation
│       ├── planning/
│       │   ├── clarifier.py           # Clarity assessment + question generation
│       │   ├── plan_builder.py        # Structured change plan generator
│       │   └── refactor_advisor.py    # Refactor suggestion generator
│       ├── patches/
│       │   └── engine.py              # Patch step workflow + rationale hooks
│       ├── specs/
│       │   └── boundary_manager.py   # Boundary detection + spec proposal
│       └── tests/
│           └── suggester.py           # Test case recommendation surface
└── tests/
    ├── test_context_indexer.py
    ├── test_patch_workflow.py
    └── mocks.py                       # Test utilities and mocks
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
./spec-agent index /path/to/repo --branch main
./spec-agent start --description "Investigate auth bug"
```

### Command Reference

**Setup & General:**
- `./spec-agent --setup`: bootstrap the environment (venv + dependencies) and verify the CLI command.
- `./spec-agent --help`: show available subcommands and options.

**Core Workflow:**
- `./spec-agent index <repo> [--branch <branch>]`: index a repository and save the context for later use.
- `./spec-agent start --description "<text>"`: create a new task using the previously indexed repository.
- `./spec-agent tasks [--status <status>]`: list recorded tasks, optionally filtered by status.
- `./spec-agent plan <task_id>`: generate the plan, boundary specs, patch queue, and suggested tests for an existing task.

**Boundary Specs:**
- `./spec-agent specs <task_id>`: view detailed boundary specifications for a task.
- `./spec-agent approve-spec <task_id> <spec_id>`: approve a boundary specification.
- `./spec-agent skip-spec <task_id> <spec_id>`: skip (override) a boundary specification.

**Patch Management:**
- `./spec-agent patches <task_id> [--list]`: inspect and approve/reject incremental patches for a task.

**Refactoring:**
- `./spec-agent refactors <task_id> [--list]`: inspect and accept/reject refactor suggestions.

**Status:**
- `./spec-agent status <task_id>`: display branch/commit alignment and uncommitted changes for a task's working tree.

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

## Epic Status

### Epic 2 - Boundary-Aware Change Management 

**Completed:**
- **2.2 Plan Agent**: Generates structured implementation plans with steps, risks, and refactor suggestions
- **2.4 Boundary Specification Proposal**: LLM-powered boundary spec generation with Mermaid diagrams and machine-readable contracts
- **2.5 Spec Approval Gate**: CLI commands for reviewing and approving boundary specs before patch generation

**In Progress:**
- **2.1 Context Engine**: Basic repository summarization implemented; dependency graph analysis pending
- **2.3 Boundary Detection**: Keyword-based detection implemented; sophisticated dependency-based detection pending

### Epic 3 - Serena Integration & Boundary-Aware Code Generation 

**Completed:**
- **Serena MCP Integration**: Full integration with Serena's MCP server for semantic code editing
- **Language Detection**: Automatic repository language detection for context-aware code generation
- **Module Detection**: Module identification and targeting for code changes
- **Boundary Spec Integration**: Approved boundary specs automatically included in code generation prompts
- **Patch Generation & Application**: Unified diff generation with git apply integration and fallback mechanisms

**Key Features:**
- Boundary specs (actors, interfaces, invariants) are passed to Serena during code generation
- Code generation respects architectural contracts defined in boundary specs
- Proper diff format detection (existing files vs new files)
- Comprehensive error handling and diagnostics

### Command Reference

**Task Management:**
- `./spec-agent start <repo> --branch <branch> --description "<text>"`: Create a new task
- `./spec-agent tasks [--status <status>]`: List all tasks, optionally filtered by status
- `./spec-agent status <task-id>`: Show task status and git state

**Planning & Boundary Specs:**
- `./spec-agent plan <task-id>`: Generate plan, boundary specs, patch queue, and test suggestions
- `./spec-agent specs <task-id>`: View detailed boundary specifications
- `./spec-agent approve-spec <task-id> <spec-id>`: Approve a boundary spec
- `./spec-agent skip-spec <task-id> <spec-id>`: Skip/override a boundary spec

**Patch Management:**
- `./spec-agent patches <task-id>`: Review and approve/reject patches interactively
- `./spec-agent patches <task-id> --list`: List all patches without reviewing

**Refactor Management:**
- `./spec-agent refactors <task-id>`: Review refactor suggestions
- `./spec-agent refactors <task-id> --list`: List refactor suggestions

### Workflow Example

```bash
# 1. Start a new task
./spec-agent start /path/to/repo --branch master --description "add OAuth2 configuration"

# 2. Generate plan (creates boundary specs if needed)
./spec-agent plan <task-id>

# 3. Review boundary specs
./spec-agent specs <task-id>

# 4. Approve boundary specs (required before patch generation)
./spec-agent approve-spec <task-id> <spec-id>

# 5. Review and approve patches (boundary specs automatically included)
./spec-agent patches <task-id>

# 6. Check status
./spec-agent status <task-id>
```

## Optional: Serena-backed LLM Integration

If you plan to plug an LLM into the workflow, you can delegate semantic retrieval and editing to [Serena](https://github.com/oraios/serena). The orchestrator now supports an opt-in mode that shells out to a Serena command for patch generation.

### Quick Setup

1. **Install Serena and MCP library**:
   ```bash
   # Install MCP library (required for Serena integration)
   pip install mcp
   # Or install with optional dependencies:
   pip install -e ".[serena]"
   
   # Install uv (provides uvx command) - required to run Serena
   # On macOS/Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or via Homebrew: brew install uv
   # Or via pip: pip install uv
   
   # Verify Serena is available
   uvx --from git+https://github.com/oraios/serena serena start-mcp-server --help
   
   # Alternative: Install Serena directly (if you prefer)
   # pip install serena-agent
   ```

2. **Enable Serena integration**:
   ```bash
   export SPEC_AGENT_SERENA_ENABLED=1
   export SPEC_AGENT_SERENA_COMMAND="python $(pwd)/scripts/serena_patch_wrapper.py"
   ```

3. **Configure real Serena** (choose one method):

   **Method A: Use Serena MCP integration** (recommended - this is how Serena works):
   ```bash
   # Install MCP library (already done in step 1)
   pip install mcp
   
   # Install uv (if not already installed) - provides uvx command
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or: brew install uv
   # Or: pip install uv
   
   # Use the MCP integration script
   export SERENA_PATCH_DELEGATE="python $(pwd)/scripts/serena_mcp_integration.py"
   
   # Optionally customize Serena MCP server command (if you installed Serena differently):
   # export SERENA_MCP_COMMAND="serena start-mcp-server"  # if installed via pip
   # export SERENA_MCP_COMMAND="uvx --from git+https://github.com/oraios/serena serena start-mcp-server"  # default
   ```

   **Method B: Use the simple wrapper** (fallback, guides to MCP):
   ```bash
   export SERENA_PATCH_DELEGATE="python $(pwd)/scripts/serena_simple.py"
   # This will detect Serena and guide you to use MCP integration
   ```

   **Method C: Point directly to a custom command**:
   ```bash
   export SERENA_PATCH_DELEGATE="your-custom-serena-wrapper"
   # Must accept JSON via stdin and emit JSON with diff/rationale/alternatives
   ```

4. **Test the integration**:
   ```bash
   # Test the wrapper
   echo '{"repo_path":"/path/to/repo","plan_id":"test","step_description":"Test step"}' \
     | python scripts/serena_patch_wrapper.py | jq
   
   # Full workflow test
   ./spec-agent start /path/to/repo --branch main --description "Test task"
   ./spec-agent plan <task-id>
   ./spec-agent patches <task-id>  # Should show real Serena-generated patches!
   ```

### Configuration Files

The launcher writes `~/.spec_agent/env` with default Serena settings. You can edit this file to persist your configuration:

```bash
# Edit ~/.spec_agent/env
SPEC_AGENT_SERENA_ENABLED=1
SPEC_AGENT_SERENA_COMMAND="python /path/to/scripts/serena_patch_wrapper.py"
SPEC_AGENT_SERENA_TIMEOUT=120
SERENA_PATCH_DELEGATE="python /path/to/scripts/serena_simple.py"
```

### How It Works

- `PatchEngine` uses the configured Serena command to request diffs per plan step
- The wrapper (`serena_patch_wrapper.py`) handles the JSON protocol conversion
- The MCP integration (`serena_mcp_integration.py`) connects to Serena's MCP server and uses its semantic code editing tools
- **Boundary specs are automatically included**: When boundary specs are approved, they're passed to Serena with the step description, ensuring generated code respects architectural contracts (actors, interfaces, invariants)
- Language detection automatically identifies the repository's primary language for context-aware code generation
- Proper diff format generation: Existing files use `--- a/filename` format, new files use `--- /dev/null` format
- Failures provide detailed diagnostics and fallback mechanisms
- Serena provides IDE-like tools (find_symbol, edit_code, etc.) that work at the semantic level rather than text-based edits

### Note on Serena Architecture

Serena is an **MCP server** that provides semantic code editing tools. It doesn't have a direct CLI for generating patches. Instead:
- Serena's MCP server provides tools like `find_symbol`, `edit_code`, `get_file_contents`, etc.
- An LLM or agent orchestrates these tools to make changes
- The `serena_mcp_integration.py` script provides a basic integration, but for full functionality, you may want to enhance it to properly chain Serena's tools or integrate with an LLM that can orchestrate them

## Next Iterations

**Epic 2 Remaining:**
- **2.1 Context Engine**: Enhance with dependency graph analysis using language-specific parsers
- **2.3 Boundary Detection**: Implement sophisticated boundary detection using dependency graphs and LLM-assisted semantic analysis

**Future Enhancements:**
- Enhanced boundary visualization
- Test generation based on boundary specs
- Improved refactor suggestion algorithms
- Better conflict resolution for patches

The project keeps all epic responsibilities visible from day one, making it easier to invest in the highest-risk components without re-architecting later. 


