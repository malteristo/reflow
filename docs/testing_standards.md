# Research Agent Testing Standards

> **Practical guide for AI-assisted test development following TDD principles**

## Overview

This document provides testing standards for the Research Agent project, optimized for AI-assisted development. Instead of extensive templates, we focus on clear patterns and conventions that enable efficient test generation.

**Quick References:**
- **Copy-paste code snippets**: [Testing Patterns Reference](testing_patterns_reference.md)
- **Test execution**: Use `python scripts/test_runner.py <category>` for organized testing

## File Organization & Naming Conventions

### Test File Structure
```
src/research_agent_backend/tests/
├── unit/                          # Isolated component tests
│   ├── test_cli_commands.py       # CLI command testing
│   ├── test_embedding_service.py  # Core service testing
│   └── test_vector_store.py       # Database operation testing
├── integration/                   # Component interaction tests  
│   ├── test_rag_pipeline.py       # End-to-end query flow
│   └── test_cli_integration.py    # CLI workflow testing
├── e2e/                          # Full system tests
│   └── test_mcp_workflows.py      # Complete MCP server workflows
├── fixtures/                     # Shared test data and utilities
│   ├── sample_documents/          # Test markdown files
│   ├── mock_configs.py           # Configuration fixtures
│   └── test_data.py              # Common test data generators
└── conftest.py                   # Pytest configuration and global fixtures
```

### Naming Conventions

**Test Files**: `test_<module_name>.py`
- `test_embedding_service.py` - Tests for `embedding_service.py`
- `test_cli_commands.py` - Tests for CLI command modules

**Test Classes**: `Test<ComponentName>`
- `TestEmbeddingService` - Tests for EmbeddingService class
- `TestCLICommands` - Tests for CLI command functions

**Test Methods**: `test_<behavior_description>`
- `test_generate_embeddings_returns_correct_dimensions()`
- `test_cli_command_handles_invalid_input_gracefully()`

**Test Markers**: Use pytest markers for organization
```python
@pytest.mark.unit           # Fast, isolated tests
@pytest.mark.integration    # Slower, multi-component tests  
@pytest.mark.cli            # CLI-specific tests
@pytest.mark.mcp            # MCP server tests
@pytest.mark.async          # Async operation tests
@pytest.mark.tdd_red        # TDD Red phase (failing by design)
@pytest.mark.tdd_green      # TDD Green phase (minimal implementation)
```

## Test Execution Workflow

### Quick Test Commands

Use the test runner script for organized test execution:

```bash
# Run specific test categories
python scripts/test_runner.py unit          # Fast unit tests only
python scripts/test_runner.py integration   # Integration tests
python scripts/test_runner.py cli           # CLI command tests
python scripts/test_runner.py mcp           # MCP server tests
python scripts/test_runner.py async         # Async operation tests

# TDD workflow support
python scripts/test_runner.py red           # Run failing tests (TDD Red phase)
python scripts/test_runner.py green         # Run minimal implementation tests

# Coverage and comprehensive testing
python scripts/test_runner.py coverage      # Run with HTML coverage report
python scripts/test_runner.py all           # Run all tests
```

### Watch Mode Development
```bash
# Continuous testing during development
python scripts/test_runner.py watch
# Requires: pip install pytest-watch
```

## TDD Red-Green-Refactor Workflow

### AI-Assisted TDD Process

1. **Red Phase**: Write failing tests that define desired behavior
```bash
python scripts/test_runner.py red  # Run tests marked as @pytest.mark.tdd_red
```

2. **Green Phase**: Write minimal implementation to make tests pass
```bash
python scripts/test_runner.py green  # Run tests marked as @pytest.mark.tdd_green
```

3. **Refactor Phase**: Improve code while keeping tests green
```bash
python scripts/test_runner.py unit  # Ensure all unit tests still pass
```

### Example TDD Workflow for AI Development

```python
# RED: Write failing test first
@pytest.mark.tdd_red
def test_embedding_service_generates_correct_dimensions():
    """Test that embeddings have expected dimensionality."""
    service = EmbeddingService(test_config)
    
    embeddings = service.generate_embeddings(["test text"])
    
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384  # Expected dimension
    # This will fail initially since EmbeddingService doesn't exist

# GREEN: Minimal implementation
@pytest.mark.tdd_green  
def test_minimal_embedding_implementation():
    """Minimal implementation to pass the test."""
    class EmbeddingService:
        def __init__(self, config):
            pass
        def generate_embeddings(self, texts):
            return [[0.0] * 384 for _ in texts]  # Hard-coded minimal implementation
    
    service = EmbeddingService({})
    embeddings = service.generate_embeddings(["test"])
    assert len(embeddings[0]) == 384

# REFACTOR: Proper implementation (unmarked - regular test)
def test_production_embedding_service():
    """Test with proper implementation."""
    service = EmbeddingService(production_config)
    embeddings = service.generate_embeddings(["test text"])
    
    # Now tests real functionality
    assert isinstance(embeddings[0][0], float)
    assert all(-1 <= x <= 1 for x in embeddings[0])  # Valid embedding range
```

## Component-Specific Testing Patterns

### CLI Testing (Typer-based)

**Key Patterns:**
- Use `typer.testing.CliRunner` for command invocation
- Mock file system operations with `pytest-mock`
- Test both success and error paths
- Validate output formatting and exit codes

**AI Prompt for CLI Tests:**
> "Generate pytest tests for the CLI command `<command_name>` that uses Typer. Include tests for: success case with valid inputs, error handling for invalid inputs, file validation, and interactive prompts. Use CliRunner and appropriate mocks."

### MCP Server Testing (FastMCP)

**Key Patterns:**  
- Test tool registration and discovery
- Validate tool execution with various inputs
- Test resource management and cleanup
- Use async test fixtures for server lifecycle

**AI Prompt for MCP Tests:**
> "Create pytest tests for the MCP tool `<tool_name>` using FastMCP framework. Include tests for: tool registration, parameter validation, successful execution, error handling, and resource cleanup. Use async/await patterns."

### Async Operation Testing

**Key Patterns:**
- Use `pytest-asyncio` for async test support
- Test concurrent operations and timeouts
- Validate proper resource cleanup
- Test async context managers

**AI Prompt for Async Tests:**
> "Write async pytest tests for `<async_function>`. Include tests for: normal execution, concurrent operations, timeout handling, exception propagation, and async context manager cleanup. Use pytest.mark.asyncio."

### Core Service Testing

**Key Patterns:**
- Mock external dependencies (models, databases, APIs)
- Test configuration loading and validation
- Validate error handling and edge cases
- Use dependency injection for testability

**AI Prompt for Service Tests:**
> "Generate unit tests for the service class `<ServiceName>`. Mock all external dependencies, test configuration handling, validate core business logic, and include error cases. Use pytest fixtures for setup."

## Coverage Requirements

### Minimum Coverage Standards
- **New code**: 95% line coverage required
- **Modified code**: Existing coverage must be maintained
- **Critical paths**: 100% coverage for error handling and edge cases

### Coverage Reporting
```bash
# Generate coverage report
python scripts/test_runner.py coverage

# View HTML report
open htmlcov/index.html  # Generated after coverage run
```

### Coverage Configuration
Coverage settings are defined in `pyproject.toml`:
- Source: `src/` directory
- Excludes: test files, __pycache__, virtual environments
- Fail threshold: 95%

## AI Development Integration

### Efficient Test Generation Prompts

**For new features:**
> "Following TDD principles, write failing tests first for [feature description]. Use appropriate pytest markers (@pytest.mark.unit, @pytest.mark.cli, etc.) and follow the naming conventions in the testing standards."

**For refactoring:**
> "Update tests for [component] after refactoring. Ensure all existing behavior is preserved while adding tests for new functionality. Use mocks for external dependencies."

**For bug fixes:**
> "Write a test that reproduces the bug in [component], then implement the fix. The test should fail initially (red), pass after the fix (green), and serve as regression protection."

### Pattern Application

Instead of copying templates, use these targeted approaches:

1. **Identify the component type** (CLI, MCP, Service, Async)
2. **Reference the appropriate pattern** from [Testing Patterns Reference](testing_patterns_reference.md)
3. **Adapt the pattern** to your specific functionality
4. **Follow TDD workflow** with red-green-refactor cycles

### Integration with TaskMaster

When working on testing subtasks:

```bash
# Start testing subtask
task-master set-status --id=X.Y --status=in-progress

# Log TDD progress
task-master update-subtask --id=X.Y --prompt="RED: Created failing tests for..."
task-master update-subtask --id=X.Y --prompt="GREEN: Minimal implementation passes tests..."  
task-master update-subtask --id=X.Y --prompt="REFACTOR: Enhanced implementation with proper logic..."

# Complete testing
task-master set-status --id=X.Y --status=done
```

## Quality Assurance

### Pre-commit Testing
```bash
# Run before committing
python scripts/test_runner.py unit      # Fast feedback
python scripts/test_runner.py coverage  # Ensure coverage requirements
```

### CI/CD Integration
The pytest configuration supports automated testing in CI environments:
- All tests run with coverage reporting
- Strict marker enforcement prevents typos
- Async tests handled automatically
- HTML coverage reports generated for review

---

**Remember**: Tests are documentation of behavior. Write them clearly, name them descriptively, and keep them focused on a single behavior per test method. 