# TDD Workflow Guide for Research Agent Backend

This guide explains how to follow Test-Driven Development (TDD) principles when working on the Research Agent backend.

## TDD Cycle Overview

TDD follows the **Red-Green-Refactor** cycle:

1. **RED**: Write a failing test that defines desired behavior
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Improve code quality while keeping tests green

## Step-by-Step TDD Workflow

### Phase 1: RED - Write Failing Tests

1. **Understand the requirement** from the task or user story
2. **Write a test** that describes the expected behavior
3. **Run the test** and confirm it fails (RED)
4. **Commit the failing test** with clear message

```python
# Example RED phase test
def test_document_chunker_splits_by_headers():
    """Test that document chunker respects markdown headers."""
    chunker = MarkdownChunker(chunk_size=512)
    document = "# Header 1\nContent 1\n## Header 2\nContent 2"
    
    chunks = chunker.chunk_document(document)
    
    assert len(chunks) == 2
    assert chunks[0].content == "# Header 1\nContent 1"
    assert chunks[1].content == "## Header 2\nContent 2"
```

### Phase 2: GREEN - Make Tests Pass

1. **Write minimal code** to make the test pass
2. **Don't optimize** yet - focus on making it work
3. **Run tests** and confirm they pass (GREEN)
4. **Commit working code** with clear message

```python
# Example GREEN phase implementation
class MarkdownChunker:
    def __init__(self, chunk_size: int):
        self.chunk_size = chunk_size
    
    def chunk_document(self, document: str) -> List[Chunk]:
        # Minimal implementation just to pass the test
        parts = document.split('\n## ')
        if len(parts) > 1:
            chunks = [Chunk(content=parts[0])]
            for part in parts[1:]:
                chunks.append(Chunk(content=f"## {part}"))
            return chunks
        return [Chunk(content=document)]
```

### Phase 3: REFACTOR - Improve Code Quality

1. **Improve code structure** without changing behavior
2. **Add error handling** and edge case support
3. **Optimize performance** if needed
4. **Run tests frequently** to ensure no regressions
5. **Commit refactored code** with clear message

```python
# Example REFACTOR phase improvement
class MarkdownChunker:
    def __init__(self, chunk_size: int):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        self.chunk_size = chunk_size
        self._header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def chunk_document(self, document: str) -> List[Chunk]:
        """Split document into chunks respecting markdown structure."""
        if not document.strip():
            return []
        
        headers = self._find_headers(document)
        return self._split_by_headers(document, headers)
    
    def _find_headers(self, document: str) -> List[HeaderInfo]:
        """Find all markdown headers in the document."""
        # More sophisticated header detection
        pass
    
    def _split_by_headers(self, document: str, headers: List[HeaderInfo]) -> List[Chunk]:
        """Split document at header boundaries."""
        # Improved splitting logic
        pass
```

## Task Master Integration

When working with Task Master, integrate TDD into your workflow:

### Starting a Subtask
```bash
# Set status to in-progress
task-master set-status --id=X.Y --status=in-progress
```

### RED Phase Documentation
```bash
# Document the failing tests you've written
task-master update-subtask --id=X.Y --prompt="RED PHASE: Created failing tests for [functionality]:
- test_feature_does_expected_behavior()
- test_feature_handles_edge_cases()
- test_feature_validates_input()

All tests failing as expected since [module] not implemented."
```

### GREEN Phase Documentation
```bash
# Document minimal implementation
task-master update-subtask --id=X.Y --prompt="GREEN PHASE: Implemented minimal [module]:
- Basic [functionality] working
- All tests now passing
- Coverage: [percentage]%

Ready for refactoring phase."
```

### REFACTOR Phase Documentation
```bash
# Document improvements made
task-master update-subtask --id=X.Y --prompt="REFACTOR PHASE: Enhanced [module]:
- Improved error handling
- Added input validation
- Optimized [specific area]
- Added comprehensive docstrings
- Coverage maintained at [percentage]%"
```

### Completing the Subtask
```bash
# Mark as complete
task-master set-status --id=X.Y --status=done
```

## Testing Guidelines

### Test Naming Convention
- Use descriptive names that explain the behavior being tested
- Start with `test_` prefix
- Include the scenario: `test_function_name_when_condition_then_result`

### Test Structure
```python
def test_function_name_when_condition_then_result():
    """Test description explaining what behavior is being verified."""
    # Arrange - Set up test data
    input_data = create_test_data()
    
    # Act - Execute the function under test
    result = function_under_test(input_data)
    
    # Assert - Verify the expected outcome
    assert result == expected_result
```

### Coverage Requirements
- **New code**: 100% line coverage required
- **Modified code**: Maintain existing coverage + test new functionality
- **Critical paths**: Focus extra attention on error handling and edge cases

### Test Categories
Use pytest markers to categorize tests:
- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Component interaction tests
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.slow` - Tests that take longer to run

## Common TDD Patterns

### Testing Exceptions
```python
def test_function_raises_exception_for_invalid_input():
    """Test that function raises appropriate exception for invalid input."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_under_test(invalid_input)
```

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function_returns_expected_result():
    """Test async function behavior."""
    result = await async_function_under_test()
    assert result is not None
```

### Testing with Mocks
```python
@patch('module.external_dependency')
def test_function_with_external_dependency(mock_dependency):
    """Test function that depends on external service."""
    mock_dependency.return_value = expected_response
    
    result = function_under_test()
    
    assert result == expected_result
    mock_dependency.assert_called_once()
```

## Best Practices

1. **Write tests first** - Always start with RED phase
2. **Keep tests simple** - One assertion per test when possible
3. **Test behavior, not implementation** - Focus on what, not how
4. **Use descriptive test names** - Make intent clear
5. **Clean up resources** - Use fixtures for setup/teardown
6. **Run tests frequently** - After every small change
7. **Commit often** - After each TDD cycle completion

## Tools and Commands

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest src/research_agent_backend/tests/test_module.py

# Run tests with coverage
pytest --cov=src/research_agent_backend --cov-report=term-missing

# Run only unit tests
pytest -m unit

# Run tests and watch for changes
pytest-watch
```

### Coverage Analysis
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Check coverage meets threshold
pytest --cov=src --cov-fail-under=95
```

Remember: **If you're not writing tests first, you're not doing TDD!** 