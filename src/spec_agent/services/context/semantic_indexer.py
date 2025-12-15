"""
Semantic repository indexer that generates a language-agnostic semantic index.

This module analyzes codebases and produces a structured semantic index
following the schema defined in index-schema.json.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from ...services.llm.openai_client import OpenAILLMClient
from ...config.settings import AgentSettings


class SemanticIndexer:
    """
    Generate a semantic, language-agnostic index of a repository.
    
    This indexer uses an LLM to analyze the codebase structure, boundaries,
    responsibilities, and architectural patterns, producing a structured
    JSON output that follows the index-schema.json format.
    """

    SYSTEM_PROMPT = """You are an expert software architect and codebase analyst.

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

Analyze this repository and produce a semantic index using the schema below.

Guidelines:
- Be concise but informative.
- Prefer intent and responsibility over implementation details.
- Infer architecture and domain concepts when possible.
- If something is unclear, make a reasonable assumption and note it.
- Omit anything that would quickly become stale.

Output ONLY valid JSON.
Do NOT add explanations or markdown.
Do NOT wrap the JSON in code blocks."""

    def __init__(self, llm_client: OpenAILLMClient, settings: AgentSettings):
        self.llm_client = llm_client
        self.settings = settings

    def generate_semantic_index(
        self, 
        repo_path: Path, 
        basic_summary: Dict[str, Any],
        schema_path: Path | None = None
    ) -> Dict[str, Any]:
        """
        Generate a semantic index for the repository.
        
        Args:
            repo_path: Path to the repository root
            basic_summary: Basic file/directory statistics from ContextIndexer
            schema_path: Optional path to index-schema.json
            
        Returns:
            Semantic index following the schema format
        """
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        # Load the schema
        if schema_path is None:
            # Default to schema in templates folder
            schema_path = Path(__file__).parent.parent.parent.parent.parent / "templates" / "index-schema.json"
        
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        # Collect representative files for analysis
        representative_files = self._collect_representative_files(repo_path, basic_summary)
        
        # Build the analysis prompt
        user_prompt = self._build_analysis_prompt(
            repo_path=repo_path,
            basic_summary=basic_summary,
            representative_files=representative_files,
            schema=schema
        )

        # Call LLM to generate semantic index
        response = ""
        try:
            response = self.llm_client.complete(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_output_tokens=8000,  # Large token budget for comprehensive analysis
            )
            
            # Parse the JSON response
            semantic_index = json.loads(response)
            
            # Add generation metadata
            semantic_index["generatedAt"] = datetime.now().isoformat()
            semantic_index["schemaVersion"] = schema.get("schemaVersion", "1.0")
            
            return semantic_index
            
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\n\nResponse: {response}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate semantic index: {e}")

    def _collect_representative_files(
        self, 
        repo_path: Path, 
        basic_summary: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Collect representative files that illustrate the codebase structure.
        
        Strategy:
        - Entry points (main files, CLI definitions, API routers)
        - Domain models
        - Configuration files
        - README and docs
        - Test examples
        """
        representative_files = {}
        
        # Common entry point patterns
        entry_point_patterns = [
            "main.py", "app.py", "__main__.py", "index.js", "index.ts",
            "server.py", "server.js", "api.py", "routes.py", "Program.cs",
            "Application.java", "Main.java"
        ]
        
        # Domain/model patterns
        model_patterns = [
            "**/models.py", "**/model.py", "**/domain/*.py",
            "**/entities/*.py", "**/entity/*.py",
            "**/*Model.cs", "**/*Entity.cs", "**/*Domain.cs",
            "**/*Model.java", "**/*Entity.java"
        ]
        
        # Configuration patterns
        config_patterns = [
            "config.py", "settings.py", "configuration.py",
            "appsettings.json", "application.properties",
            "package.json", "pyproject.toml", "pom.xml", "build.gradle"
        ]
        
        # Documentation patterns
        doc_patterns = [
            "README.md", "README.rst", "ARCHITECTURE.md",
            "docs/index.md", "docs/architecture.md"
        ]
        
        max_files = 30  # Limit to prevent token overflow
        collected = 0
        
        # Collect entry points
        for pattern in entry_point_patterns:
            for file_path in repo_path.rglob(pattern):
                if collected >= max_files:
                    break
                if self._should_include_file(file_path, repo_path):
                    content = self._read_file_safely(file_path, max_lines=200)
                    if content:
                        relative_path = file_path.relative_to(repo_path)
                        representative_files[str(relative_path)] = content
                        collected += 1
        
        # Collect configuration files
        for pattern in config_patterns:
            if collected >= max_files:
                break
            for file_path in repo_path.glob(pattern):
                if self._should_include_file(file_path, repo_path):
                    content = self._read_file_safely(file_path, max_lines=100)
                    if content:
                        relative_path = file_path.relative_to(repo_path)
                        representative_files[str(relative_path)] = content
                        collected += 1
        
        # Collect documentation
        for pattern in doc_patterns:
            if collected >= max_files:
                break
            for file_path in repo_path.glob(pattern):
                if self._should_include_file(file_path, repo_path):
                    content = self._read_file_safely(file_path, max_lines=150)
                    if content:
                        relative_path = file_path.relative_to(repo_path)
                        representative_files[str(relative_path)] = content
                        collected += 1
        
        # Collect some model files
        for pattern in model_patterns:
            if collected >= max_files:
                break
            for file_path in repo_path.glob(pattern):
                if collected >= max_files:
                    break
                if self._should_include_file(file_path, repo_path):
                    content = self._read_file_safely(file_path, max_lines=150)
                    if content:
                        relative_path = file_path.relative_to(repo_path)
                        representative_files[str(relative_path)] = content
                        collected += 1
        
        return representative_files

    def _should_include_file(self, file_path: Path, repo_path: Path) -> bool:
        """Check if a file should be included in analysis."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return False
        
        # Skip common ignored directories
        ignored_dirs = {
            'node_modules', '__pycache__', 'venv', 'env', '.venv',
            'dist', 'build', 'target', 'bin', 'obj', '.git'
        }
        if any(part in ignored_dirs for part in file_path.parts):
            return False
        
        # Must be a file
        if not file_path.is_file():
            return False
        
        # Check file size (skip very large files)
        try:
            if file_path.stat().st_size > 1_000_000:  # 1MB limit
                return False
        except OSError:
            return False
        
        return True

    def _read_file_safely(self, file_path: Path, max_lines: int = 200) -> str | None:
        """Read a file safely with line limit."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n... (truncated at {max_lines} lines)")
                        break
                    lines.append(line)
                return ''.join(lines)
        except Exception:
            return None

    def _build_analysis_prompt(
        self,
        repo_path: Path,
        basic_summary: Dict[str, Any],
        representative_files: Dict[str, str],
        schema: Dict[str, Any]
    ) -> str:
        """Build the prompt for LLM analysis."""
        
        prompt_parts = [
            "# Repository Analysis Request\n",
            f"\n## Repository: {repo_path.name}",
            f"Path: {repo_path}\n",
            "\n## Basic Statistics:",
            f"- Files: {basic_summary.get('file_count', 0)}",
            f"- Directories: {basic_summary.get('directory_count', 0)}",
            f"- Languages: {', '.join(basic_summary.get('top_languages', []))}",
            f"- Top Extensions: {', '.join(basic_summary.get('top_file_extensions', [])[:5])}",
        ]
        
        if basic_summary.get('frameworks'):
            prompt_parts.append(f"- Detected Frameworks: {', '.join(basic_summary['frameworks'])}")
        
        if basic_summary.get('top_modules'):
            prompt_parts.append(f"\n## Top Modules/Namespaces:")
            for module in basic_summary['top_modules'][:10]:
                prompt_parts.append(f"  - {module}")
        
        prompt_parts.append("\n## Representative Files:\n")
        
        for file_path, content in list(representative_files.items())[:25]:
            prompt_parts.append(f"\n### File: {file_path}")
            prompt_parts.append(f"```\n{content[:2000]}\n```\n")
        
        prompt_parts.append("\n## Schema to Follow:\n")
        prompt_parts.append(f"```json\n{json.dumps(schema, indent=2)}\n```\n")
        
        prompt_parts.append("\n## Instructions:")
        prompt_parts.append("Analyze the repository above and generate a semantic index that follows the schema exactly.")
        prompt_parts.append("Focus on architectural patterns, responsibilities, boundaries, and domain concepts.")
        prompt_parts.append("Output ONLY the JSON index, no additional text or formatting.")
        
        return '\n'.join(prompt_parts)


