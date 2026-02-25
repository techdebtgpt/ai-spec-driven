# Spec Agent MCP Integration

This directory contains example configuration files for integrating spec-agent with AI assistants via MCP (Model Context Protocol).

## Quick Setup

### 1. Install with MCP support

```bash
cd /path/to/ai-spec-driven
pip install -e ".[serena]"  # includes mcp dependency
```

Or with uv:

```bash
uv pip install -e ".[serena]"
```

### 2. Configure your AI assistant

#### For Cursor

Copy the configuration to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "spec-agent": {
      "command": "python",
      "args": ["-m", "spec_agent.mcp_server"],
      "cwd": "/path/to/ai-spec-driven",
      "env": {
        "SPEC_AGENT_OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

#### For Claude Desktop

Copy to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS):

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

### 3. Restart your AI assistant

After adding the configuration, restart Cursor or Claude Desktop to load the MCP server.

## Available Tools

The MCP server exposes these tools for spec-driven development:

### Repository Management
- `index_repository` - Index a codebase for analysis
- `get_repository_summary` - Get indexed repository info

### Task Management
- `create_task` - Create a new development task
- `list_tasks` - List all tasks
- `get_task_status` - Get task details

### Clarifications
- `get_clarifications` - Get clarifying questions
- `answer_clarification` - Answer a question

### Planning
- `generate_plan` - Generate implementation plan
- `get_boundary_specs` - Get boundary specifications
- `approve_spec` / `skip_spec` - Resolve specs
- `approve_all_specs` - Approve all specs at once
- `approve_plan` - Approve the plan

### Patch Generation
- `generate_patches` - Generate code patches
- `list_patches` - List all patches
- `get_patch_details` - Get full patch content
- `get_next_pending_patch` - Get next patch to review
- `approve_patch` - Apply a patch
- `reject_patch` - Reject and regenerate

### Workflow
- `get_workflow_status` - Get comprehensive workflow state

## Example Workflow

Here's how an AI assistant would use these tools:

```
1. index_repository("/path/to/repo", "main")
2. create_task("Add user authentication to the API")
3. answer_clarification(task_id, q_id, "Use JWT tokens")
4. generate_plan(task_id)
5. approve_all_specs(task_id)
6. approve_plan(task_id)
7. generate_patches(task_id)
8. approve_patch(task_id, patch_id)  # for each patch
```

## Environment Variables

- `SPEC_AGENT_OPENAI_API_KEY` - OpenAI API key (required for LLM features)
- `SPEC_AGENT_OPENAI_MODEL` - Model to use (default: gpt-4.1-mini)
- `SPEC_AGENT_STATE_DIR` - Override state directory (default: ~/.spec_agent)

## Testing the Server

Run the MCP server directly to test:

```bash
cd /path/to/ai-spec-driven
python -m spec_agent.mcp_server
```

Or use the installed script:

```bash
spec-agent-mcp
```
