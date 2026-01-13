from __future__ import annotations

import json
import logging
import re
import os
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from ...domain.models import BoundarySpec, Patch, Plan, PlanStep, TestSuggestion

LOG = logging.getLogger(__name__)


class TestSuggester:
    """
    Provides candidate test updates based on the approved plan and patches.
    
    Epic 4.2: Test Case Suggestions
    - Identifies existing tests that should be updated
    - Suggests new test cases (unit and/or integration)
    - Provides test skeletons matching the repo's test framework
    """

    def __init__(self, llm_client: Optional[object] = None) -> None:
        """
        Initialize TestSuggester with optional LLM client.
        
        Args:
            llm_client: Optional OpenAILLMClient for AI-powered test suggestions
        """
        self.llm_client = llm_client

    def _detect_test_files(self, repo_path: Optional[Path], repo_context: Optional[dict]) -> tuple[bool, List[str], Optional[str]]:
        """
        Detect if test files exist in the repository.
        
        Returns:
            tuple: (has_tests, test_file_paths, test_framework)
        """
        test_file_paths: List[str] = []
        seen_paths: set[str] = set()
        test_framework: Optional[str] = None
        
        if not repo_path or not repo_path.exists():
            return (False, [], None)
        
        # Common test directory names
        test_dirs = ["tests", "test", "__tests__", "spec", "specs", "Tests", "Test"]
        
        try:
            max_results = 50
            skip_dirs = {
                ".git",
                "node_modules",
                "dist",
                "build",
                ".venv",
                "venv",
                "__pycache__",
                ".tox",
            }
            
            for root, dirs, files in os.walk(repo_path):
                relative_parts = Path(root).relative_to(repo_path).parts
                if any(part.startswith(".") for part in relative_parts if part):
                    continue
                
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in skip_dirs
                ]
                
                for file_name in files:
                    if file_name.startswith("."):
                        continue
                    file_path = str(Path(root, file_name).relative_to(repo_path))
                    
                    if file_name.startswith("test_") or file_name.endswith("_test.py"):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "pytest"
                    elif file_name.endswith((".test.js", ".test.ts", ".spec.js", ".spec.ts")):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "jest"
                    elif file_name.endswith("_test.go"):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "go_test"
                    elif file_name.endswith("_test.rs"):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "rust_test"
                    elif file_name.endswith(("Test.java", "Tests.java")):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "junit"
                    elif file_name.endswith(("Test.cs", "Tests.cs")):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                        test_framework = test_framework or "nunit"
                    
                    if any(test_dir in Path(file_path).parts for test_dir in test_dirs):
                        if file_path not in seen_paths:
                            test_file_paths.append(file_path)
                            seen_paths.add(file_path)
                    
                    if len(test_file_paths) >= max_results:
                        break
                if len(test_file_paths) >= max_results:
                    break
        except Exception as exc:
            LOG.warning("Failed to detect test files: %s", exc)
        
        # Also check repo_context for test framework info
        if repo_context and not test_framework:
            test_framework = repo_context.get("test_framework")
        
        has_tests = len(test_file_paths) > 0
        return (has_tests, test_file_paths[:10], test_framework)  # Limit to 10 for performance

    def suggest(
        self,
        plan: Plan,
        patches: Optional[List[Patch]] = None,
        repo_context: Optional[dict] = None,
        repo_path: Optional[Path] = None,
    ) -> List[TestSuggestion]:
        """
        Suggest test cases based on the plan and patches.

        Args:
            plan: The implementation plan
            patches: Optional list of patches to analyze for test suggestions
            repo_context: Optional repository context (language, test framework, etc.)
            repo_path: Optional repository path for test file detection

        Returns:
            List of test suggestions
        """
        suggestions: List[TestSuggestion] = []
        
        # Detect if tests exist in the repository
        has_tests, test_file_paths, detected_framework = self._detect_test_files(repo_path, repo_context)
        
        # If no tests found, add a special suggestion about test coverage
        if not has_tests:
            coverage_suggestion = self._create_test_coverage_suggestion(
                plan=plan,
                repo_context=repo_context,
                detected_framework=detected_framework,
            )
            suggestions.append(coverage_suggestion)
        
        for step in plan.steps:
            # Find related patch if available
            related_patch = None
            if patches:
                related_patch = next(
                    (p for p in patches if p.step_reference == step.description),
                    None
                )
            
            if self.llm_client:
                try:
                    suggestion = self._suggest_with_llm(
                        step=step,
                        plan=plan,
                        patch=related_patch,
                        repo_context=repo_context,
                        has_tests=has_tests,
                        test_file_paths=test_file_paths,
                    )
                    suggestions.append(suggestion)
                except Exception as exc:
                    LOG.warning("LLM test suggestion failed for step '%s': %s", step.description, exc)
                    # Fall back to template
                    suggestion = self._suggest_template(step, plan, related_patch)
                    suggestions.append(suggestion)
            else:
                suggestion = self._suggest_template(step, plan, related_patch)
                suggestions.append(suggestion)
        
        return suggestions

    def _create_test_coverage_suggestion(
        self,
        plan: Plan,
        repo_context: Optional[dict],
        detected_framework: Optional[str],
    ) -> TestSuggestion:
        """Create a special suggestion when no tests are found."""
        primary_lang = repo_context.get("primary_language", "unknown") if repo_context else "unknown"
        
        # Determine appropriate test framework based on language
        framework_suggestion = detected_framework
        if not framework_suggestion:
            framework_map = {
                "python": "pytest",
                "javascript": "jest",
                "typescript": "jest",
                "java": "junit",
                "csharp": "nunit",
                "go": "go_test",
                "rust": "rust_test",
            }
            framework_suggestion = framework_map.get(primary_lang, "pytest")
        
        description = f"""⚠️ **No tests found in repository**

This repository does not appear to have any test files. It is strongly recommended to add test coverage for the changes being made.

**Recommendation:**
- Set up a test framework ({framework_suggestion} recommended for {primary_lang})
- Create test files for the new/changed functionality
- Aim for good test coverage to ensure code quality and prevent regressions

**Benefits of adding tests:**
- Catch bugs early in development
- Prevent regressions when making changes
- Document expected behavior
- Enable confident refactoring
- Improve code quality and maintainability"""
        
        # Generate a basic test setup skeleton
        skeleton = self._generate_test_setup_skeleton(framework_suggestion, primary_lang)
        
        return TestSuggestion(
            id=str(uuid4()),
            task_id=plan.task_id,
            description=description,
            suggestion_type="COVERAGE",
            related_files=[],
            skeleton_code=skeleton,
        )

    def _generate_test_setup_skeleton(self, framework: str, language: str) -> str:
        """Generate a test setup skeleton based on framework."""
        if framework == "pytest" or language == "python":
            return """# pytest setup example
# Install: pip install pytest pytest-cov

# tests/conftest.py (optional)
import pytest

# tests/test_example.py
def test_example():
    \"\"\"Example test to get started.\"\"\"
    assert True

# Run tests: pytest tests/
# With coverage: pytest --cov=src tests/
"""
        elif framework == "jest" or language in ["javascript", "typescript"]:
            return """// Jest setup example
// Install: npm install --save-dev jest @types/jest

// package.json
{
  "scripts": {
    "test": "jest",
    "test:coverage": "jest --coverage"
  }
}

// __tests__/example.test.js
describe('Example', () => {
  test('should work', () => {
    expect(true).toBe(true);
  });
});
"""
        elif framework == "junit" or language == "java":
            return """// JUnit setup example
// Add to pom.xml or build.gradle

// src/test/java/ExampleTest.java
import org.junit.Test;
import static org.junit.Assert.*;

public class ExampleTest {
    @Test
    public void testExample() {
        assertTrue(true);
    }
}
"""
        elif framework == "nunit" or language == "csharp":
            return """// NUnit setup example
// Install: dotnet add package NUnit

// Tests/ExampleTests.cs
using NUnit.Framework;

namespace Tests
{
    public class ExampleTests
    {
        [Test]
        public void TestExample()
        {
            Assert.IsTrue(true);
        }
    }
}
"""
        else:
            return f"""# Test setup for {framework}
# Add test framework configuration
# Create test files following {framework} conventions
"""

    def _suggest_with_llm(
        self,
        step: PlanStep,
        plan: Plan,
        patch: Optional[Patch],
        repo_context: Optional[dict],
        has_tests: bool = True,
        test_file_paths: Optional[List[str]] = None,
    ) -> TestSuggestion:
        """Generate test suggestion using LLM."""
        # Build context
        patch_context = ""
        if patch:
            patch_context = f"""
Code Change:
```
{patch.diff[:1500]}
```

Rationale: {patch.rationale}
"""

        repo_info = ""
        if repo_context:
            primary_lang = repo_context.get("primary_language", "unknown")
            repo_info = f"\nRepository: {primary_lang}\n"
            if repo_context.get("test_framework"):
                repo_info += f"Test Framework: {repo_context.get('test_framework')}\n"
        
        # Add test detection info
        if not has_tests:
            repo_info += "\n⚠️ **IMPORTANT: No test files detected in this repository.**\n"
            repo_info += "It is strongly recommended to add test coverage for these changes.\n"
        elif test_file_paths and len(test_file_paths) > 0:
            repo_info += f"\nExisting test files found: {len(test_file_paths)} test file(s)\n"
            repo_info += f"Sample test files: {', '.join(test_file_paths[:3])}\n"
        
        prompt = f"""Suggest test cases for this implementation step.

Plan Step: {step.description}
{step.notes if step.notes else ""}
{patch_context}{repo_info}

Provide test suggestions that include:

1. **Test Description**: Clear scenario description
2. **Expected Behavior**: What should be verified
3. **Test Type**: UNIT or INTEGRATION
4. **Test Skeleton**: Code skeleton matching the repository's test framework style
5. **Existing Tests to Update**: Identify existing test files that should be updated based on the code changes

For identifying existing tests:
- Look for test files that test the same modules/files being changed
- Consider integration tests that might be affected
- List test file paths (e.g., "tests/test_module.py", "src/__tests__/component.test.js")

For the test skeleton:
- Match the repository's test framework (pytest, unittest, jest, etc.)
- Include proper imports and setup
- Use appropriate assertion style
- Include comments explaining the test

Return JSON:
{{
  "description": "Test scenario description",
  "expected_behavior": "What should be verified",
  "test_type": "UNIT or INTEGRATION",
  "skeleton_code": "Complete test skeleton code",
  "related_files": ["file1.py", "file2.py"],
  "existing_tests_to_update": ["test_file1.py", "test_file2.py"]
}}

**IMPORTANT**: 
- Return ONLY valid JSON (no markdown code blocks)
- Escape all quotes in strings using \\"
- Escape all newlines in strings using \\n
- Escape all backslashes using \\\\
- The skeleton_code field must be a properly escaped JSON string

Return only valid JSON, no markdown."""

        try:
            response_text = self.llm_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a test engineer. Suggest comprehensive test cases with proper skeletons. Return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_output_tokens=800,
            )
            
            # Use robust JSON parsing
            test_data = self._parse_llm_json_response(response_text)

            # Determine related files
            related_files = test_data.get("related_files", step.target_files)
            if not related_files:
                related_files = step.target_files
            
            # Identify existing tests that should be updated (Epic 4.2)
            existing_tests = test_data.get("existing_tests_to_update", [])
            if existing_tests:
                # Add note about existing tests in description
                description = test_data.get("description", f"Verify behavior for: {step.description}")
                description += f"\n\n**Existing tests to update:** {', '.join(existing_tests)}"
            elif not has_tests:
                # No tests found - emphasize the need for test coverage
                description = test_data.get("description", f"Verify behavior for: {step.description}")
                description += "\n\n⚠️ **No tests found in repository** - This is a new test that should be created to add test coverage."
            else:
                description = test_data.get("description", f"Verify behavior for: {step.description}")

            return TestSuggestion(
                id=str(uuid4()),
                task_id=plan.task_id,
                description=description,
                suggestion_type=test_data.get("test_type", "UNIT" if step.target_files else "INTEGRATION"),
                related_files=related_files,
                skeleton_code=test_data.get("skeleton_code", self._default_skeleton(step)),
            )

        except Exception as exc:
            error_msg = str(exc)
            # Include more context for JSON parsing errors
            if "JSON" in error_msg or "json" in error_msg or "Unterminated" in error_msg:
                LOG.error("Failed to generate test suggestion with LLM due to JSON parsing error: %s. Falling back to template suggestion.", exc)
            else:
                LOG.error("Failed to generate test suggestion with LLM: %s. Falling back to template suggestion.", exc)
            return self._suggest_template(step, plan, patch)

    def _suggest_template(
        self,
        step: PlanStep,
        plan: Plan,
        patch: Optional[Patch],
    ) -> TestSuggestion:
        """Generate template test suggestion."""
        test_type = "UNIT" if step.target_files else "INTEGRATION"
        
        description = f"Verify behavior for: {step.description}"
        if patch:
            description += f"\n\nRelated change: {patch.step_reference}"
        
        return TestSuggestion(
            id=str(uuid4()),
            task_id=plan.task_id,
            description=description,
            suggestion_type=test_type,
            related_files=step.target_files,
            skeleton_code=self._default_skeleton(step),
        )

    def _default_skeleton(self, step: PlanStep) -> str:
        """Generate default test skeleton based on step."""
        # Default to pytest-style for Python, but can be enhanced
        return f"""def test_{step.description.lower().replace(' ', '_').replace('-', '_')[:50]}():
    \"\"\"
    Test: {step.description}
    
    Expected behavior:
    - Verify the implementation works correctly
    - Check edge cases if applicable
    \"\"\"
    # TODO: Implement test
    assert True
"""

    def _parse_llm_json_response(self, response_text: str) -> dict:
        """
        Parse JSON response from LLM with robust error handling.
        
        Handles common issues:
        - Markdown code blocks
        - Unescaped quotes in strings
        - Unescaped newlines
        - Trailing commas
        - Partial JSON responses
        """
        if not response_text or not response_text.strip():
            return {}
        
        # Clean up markdown code blocks
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block markers
            lines = text.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        # Try direct JSON parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            LOG.debug("Initial JSON parse failed at position %d: %s. Response preview: %s", 
                     e.pos if hasattr(e, 'pos') else -1, e.msg, text[max(0, (e.pos if hasattr(e, 'pos') else 0) - 100):(e.pos if hasattr(e, 'pos') else 0) + 100] if hasattr(e, 'pos') else text[:200])
        
        # Try to extract JSON object using regex
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to fix common JSON issues
        try:
            # Try to escape unescaped quotes in string values
            # This is tricky, so we'll use a more lenient approach
            # Use json5-like parsing or try to extract fields manually
            
            # Last resort: try to extract key-value pairs manually
            result = {}
            
            # Extract description
            desc_match = re.search(r'"description"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
            if not desc_match:
                desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', text)
            if desc_match:
                result["description"] = desc_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            # Extract expected_behavior
            behavior_match = re.search(r'"expected_behavior"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
            if not behavior_match:
                behavior_match = re.search(r'"expected_behavior"\s*:\s*"([^"]*)"', text)
            if behavior_match:
                result["expected_behavior"] = behavior_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            
            # Extract test_type
            type_match = re.search(r'"test_type"\s*:\s*"([^"]*)"', text)
            if type_match:
                result["test_type"] = type_match.group(1)
            
            # Extract skeleton_code - this is the tricky one with multiline strings
            # The LLM might return unescaped newlines, so we need a more robust extraction
            skeleton_start = text.find('"skeleton_code"')
            if skeleton_start != -1:
                # Find the colon after skeleton_code
                colon_pos = text.find(':', skeleton_start)
                if colon_pos != -1:
                    # Skip whitespace
                    value_start = colon_pos + 1
                    while value_start < len(text) and text[value_start] in ' \t\n':
                        value_start += 1
                    
                    if value_start < len(text) and text[value_start] == '"':
                        # Extract string content, handling escaped quotes and newlines
                        quote_start = value_start
                        quote_end = quote_start + 1
                        escaped = False
                        
                        while quote_end < len(text):
                            if escaped:
                                escaped = False
                            elif text[quote_end] == '\\':
                                escaped = True
                            elif text[quote_end] == '"':
                                # Found closing quote
                                skeleton_content = text[quote_start + 1:quote_end]
                                # Unescape the content
                                result["skeleton_code"] = (
                                    skeleton_content
                                    .replace('\\"', '"')
                                    .replace('\\n', '\n')
                                    .replace('\\t', '\t')
                                    .replace('\\r', '\r')
                                    .replace('\\\\', '\\')
                                )
                                break
                            quote_end += 1
                    else:
                        # Might be a multiline string without proper escaping
                        # Try to extract until we hit the next JSON key or closing brace
                        # Look for the next key pattern or end of object
                        next_key_pattern = r'"[a-z_]+"\s*:'
                        next_match = re.search(next_key_pattern, text[value_start:])
                        if next_match:
                            # Extract up to the next key
                            content_end = value_start + next_match.start()
                            # Try to find the last quote before the next key
                            last_quote = text.rfind('"', value_start, content_end)
                            if last_quote > value_start:
                                skeleton_content = text[value_start + 1:last_quote]
                                result["skeleton_code"] = skeleton_content.replace('\\"', '"').replace('\\n', '\n')
                        else:
                            # Extract until closing brace
                            brace_pos = text.find('}', value_start)
                            if brace_pos != -1:
                                # Find last quote before closing brace
                                last_quote = text.rfind('"', value_start, brace_pos)
                                if last_quote > value_start:
                                    skeleton_content = text[value_start + 1:last_quote]
                                    result["skeleton_code"] = skeleton_content.replace('\\"', '"').replace('\\n', '\n')
            
            # Extract related_files (array)
            files_match = re.search(r'"related_files"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if files_match:
                files_str = files_match.group(1)
                files = [f.strip().strip('"') for f in re.findall(r'"([^"]*)"', files_str)]
                result["related_files"] = files
            
            # Extract existing_tests_to_update (array)
            tests_match = re.search(r'"existing_tests_to_update"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if tests_match:
                tests_str = tests_match.group(1)
                tests = [t.strip().strip('"') for t in re.findall(r'"([^"]*)"', tests_str)]
                result["existing_tests_to_update"] = tests
            
            if result:
                return result
        except Exception as e:
            LOG.debug("Manual JSON extraction failed: %s", e)
        
        # If all else fails, return empty dict with better error info
        error_pos = len(text)
        try:
            # Try to find where the error might be
            json.loads(text)
        except json.JSONDecodeError as e:
            error_pos = e.pos if hasattr(e, 'pos') else len(text)
            LOG.warning("Could not parse JSON response at position %d: %s. Response preview around error: %s", 
                       error_pos, e.msg, text[max(0, error_pos - 150):error_pos + 150])
        else:
            LOG.warning("Could not parse JSON response, returning empty dict. Response preview: %s", text[:200])
        return {}
