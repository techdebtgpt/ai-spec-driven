# Templates

This folder contains template files used by the spec-agent.

## Files

### index-schema.json

The JSON schema that defines the structure for semantic repository indexes.

This schema is used by the `SemanticIndexer` service to guide LLM analysis of codebases. It ensures that all semantic indexes follow a consistent, structured format that captures:

- Repository metadata and architecture
- Module structure and dependencies
- Domain concepts and boundaries
- Public interfaces (APIs, CLI, events)
- Key components and their relationships
- Cross-cutting concerns
- External integrations
- Testing strategies
- Quality signals

The schema is automatically loaded by the semantic indexer when generating repository indexes.

**Path**: `templates/index-schema.json`  
**Used by**: `src/spec_agent/services/context/semantic_indexer.py`  
**Format**: JSON Schema (draft-07)

## Usage

The templates in this folder are automatically discovered by the spec-agent at runtime. You don't need to manually reference them - the code knows where to find them.

If you want to customize the semantic index structure for your organization, you can modify `index-schema.json` to add additional fields or change the focus areas.

