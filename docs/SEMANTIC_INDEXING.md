# Semantic Repository Indexing

## Overview

The semantic indexing feature generates a language-agnostic, architectural representation of a repository. This index serves as long-term memory for AI agents that implement features across the codebase.

## Schema

The index follows the schema defined in `templates/index-schema.json`. The schema captures:

### Repository Information
- Name, path, primary languages
- Frameworks, build tools
- Architecture style
- Entry points and test frameworks

### Structure
- **Modules**: Logical components with their types (api, service, library, cli, worker, test, infra)
- **Responsibilities**: What each module does
- **Dependencies**: Which modules depend on each other

### Domains
- **Domain Concepts**: Business entities, value objects
- **Commands & Events**: Domain-driven design patterns
- **Services**: Domain services and their relationships

### Public Interfaces
- **HTTP APIs**: Routes, handlers, request/response models
- **CLI Commands**: Command definitions and handlers
- **Events**: Publishers and consumers

### Key Components
- Controllers, services, repositories
- Integration points
- Workers and utilities
- Dependencies and relationships

### Cross-Cutting Concerns
- Logging, authentication, authorization
- Validation, configuration
- Observability, error handling

### External Integrations
- HTTP clients, message queues
- Databases, SDKs
- Purpose and usage locations

### Rules & Constraints
- Architectural conventions
- Framework-enforced patterns
- Code standards

### Testing
- Test types (unit, integration, e2e)
- Coverage focus areas
- Test data strategies

### Quality Signals
- Code generation quality
- Legacy hotspots
- High-risk areas

## Usage

### Index a Repository

```bash
./spec-agent index /path/to/repository --branch main
```

This command:
1. Analyzes the repository structure (files, directories, languages)
2. Generates a semantic index using an LLM
3. Saves both the basic summary and semantic index to `.spec-agent/repository_index.json`

### View the Index

The index is displayed in the terminal after generation, showing:
- Repository overview (name, path, commit info)
- Statistics (file count, size, languages)
- Project structure (modules, namespaces, directories)
- Semantic analysis (architecture, domains, interfaces, components)

### Use the Index

Once indexed, you can create tasks that leverage the semantic understanding:

```bash
./spec-agent start --description "Add authentication to the user management API"
```

The agent will use the semantic index to:
- Understand the codebase architecture
- Identify relevant modules and boundaries
- Generate plans that respect existing patterns
- Make changes consistent with the domain model

## How It Works

### 1. Basic Analysis
The `ContextIndexer` performs fast, local analysis:
- File and directory counts
- Language detection
- Framework identification
- Module/namespace discovery
- Directory structure mapping

### 2. Semantic Analysis
The `SemanticIndexer` uses an LLM to perform deep analysis:
- Collects representative files (entry points, models, configs, docs)
- Sends them to the LLM with the schema and instructions
- LLM analyzes architectural patterns, responsibilities, and boundaries
- Returns structured JSON following the schema

### 3. Storage
Both analyses are stored together in `repository_index.json`:
- `repository_summary`: Fast stats and basic structure
- `semantic_index`: Deep architectural understanding
- `git_info`: Commit information for reproducibility

## System Prompt

The semantic indexer uses this prompt to guide the LLM:

```
You are an expert software architect and codebase analyst.

Your task is to analyze the given repository and generate a
LANGUAGE-AGNOSTIC semantic index of the codebase.

The index must follow the provided JSON schema EXACTLY.
Do not invent new fields.
Do not include raw source code.
Do not include full AST dumps.
Do not include private helpers unless they are architecturally significant.

Focus on:
- Structure
- Responsibilities
- Boundaries
- Public interfaces
- Domain concepts
- Cross-cutting concerns
- Constraints and conventions

This index will be used as long-term memory for AI agents
that will later implement large features across the repository.

Guidelines:
- Be concise but informative.
- Prefer intent and responsibility over implementation details.
- Infer architecture and domain concepts when possible.
- If something is unclear, make a reasonable assumption and note it.
- Omit anything that would quickly become stale.

Output ONLY valid JSON.
Do NOT add explanations or markdown.
Do NOT wrap the JSON in code blocks.
```

## Configuration

### LLM Settings
The semantic indexer requires an LLM client. Configure in your environment:

```bash
export OPENAI_API_KEY=your-key-here
export OPENAI_MODEL=gpt-4  # or gpt-4-turbo
```

### Token Budget
The indexer uses up to 8000 output tokens for comprehensive analysis. For large repositories, this ensures complete coverage of all architectural aspects.

### File Selection
The indexer intelligently selects representative files:
- Entry points (main.py, app.py, index.js, etc.)
- Domain models and entities
- Configuration files (package.json, pyproject.toml, etc.)
- Documentation (README.md, ARCHITECTURE.md)
- Up to 30 files total, with line limits per file

## Benefits

1. **Language Agnostic**: Works with any programming language
2. **Architectural Focus**: Captures intent, not implementation
3. **Long-Term Stability**: Focuses on concepts that don't change frequently
4. **AI-Friendly**: Structured format optimized for LLM consumption
5. **Human Readable**: JSON format can be reviewed and edited manually

## Limitations

- Requires LLM API access (OpenAI)
- First-time indexing takes 1-2 minutes for large repositories
- Quality depends on LLM model capabilities
- May need manual refinement for complex architectures

## Future Enhancements

- Incremental updates (re-index only changed modules)
- Multiple schema versions for different use cases
- Export to other formats (Markdown, diagram tools)
- Integration with documentation generators
- Validation tools to ensure schema compliance

