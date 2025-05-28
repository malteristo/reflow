# Pytest Markers for Research Agent Backend

This document describes the pytest markers used in the Research Agent backend test suite.

## Standard Markers

### @pytest.mark.unit
**Purpose**: Mark tests that test individual components in isolation

**Characteristics**:
- Fast execution (< 1 second per test)
- No external dependencies (database, network, filesystem)
- Use mocks/stubs for dependencies
- Focus on testing single functions/methods

**Example**:
```python
@pytest.mark.unit
def test_embedding_service_validates_input():
    """Test that embedding service validates input parameters."""
    service = EmbeddingService()
    with pytest.raises(ValueError):
        service.embed_text("")
```

### @pytest.mark.integration
**Purpose**: Mark tests that verify component interactions

**Characteristics**:
- Medium execution time (1-10 seconds per test)
- May use temporary databases or files
- Test multiple components working together
- Verify data flow between modules

**Example**:
```python
@pytest.mark.integration
def test_document_processing_pipeline():
    """Test complete document processing from ingestion to storage."""
    processor = DocumentProcessor()
    vector_store = VectorStore(temp_db_path)
    
    result = processor.ingest_and_store(document, vector_store)
    assert result.success
```

### @pytest.mark.e2e
**Purpose**: Mark end-to-end tests that verify complete workflows

**Characteristics**:
- Slower execution (10+ seconds per test)
- Test complete user scenarios
- May require external services (mocked or real)
- Verify system behavior from user perspective

**Example**:
```python
@pytest.mark.e2e
def test_complete_query_workflow():
    """Test complete workflow from query to results."""
    # Test entire RAG pipeline
    result = query_engine.search("machine learning", collections=["docs"])
    assert len(result.documents) > 0
```

### @pytest.mark.slow
**Purpose**: Mark tests that take significant time to execute

**Characteristics**:
- Execution time > 5 seconds
- Often involves large datasets or complex computations
- Run separately from main test suite
- Useful for performance testing

**Example**:
```python
@pytest.mark.slow
def test_large_document_batch_processing():
    """Test processing of large document batches."""
    documents = generate_large_document_set(1000)
    result = processor.process_batch(documents)
    assert result.processed_count == 1000
```

## Specialized Markers

### @pytest.mark.tdd_red
**Purpose**: Mark tests in the RED phase of TDD cycle

**Usage**: Temporarily mark failing tests during development
```python
@pytest.mark.tdd_red
def test_new_feature_not_implemented_yet():
    """This test will fail until feature is implemented."""
    pytest.skip("RED PHASE: Feature not implemented")
```

### @pytest.mark.tdd_green
**Purpose**: Mark tests in the GREEN phase of TDD cycle

**Usage**: Mark tests with minimal implementation
```python
@pytest.mark.tdd_green
def test_basic_functionality_works():
    """Basic implementation test."""
    result = basic_function()
    assert result is not None  # Minimal assertion
```

### @pytest.mark.performance
**Purpose**: Mark tests that verify performance requirements

**Characteristics**:
- Measure execution time
- Verify memory usage
- Test scalability limits

**Example**:
```python
@pytest.mark.performance
def test_embedding_generation_performance():
    """Test that embedding generation meets performance requirements."""
    start_time = time.time()
    embeddings = service.embed_batch(large_text_list)
    duration = time.time() - start_time
    assert duration < 10.0  # Must complete within 10 seconds
```

### @pytest.mark.requires_gpu
**Purpose**: Mark tests that require GPU acceleration

**Usage**: Skip on systems without GPU support
```python
@pytest.mark.requires_gpu
def test_gpu_accelerated_embeddings():
    """Test GPU-accelerated embedding generation."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    # GPU-specific test logic
```

### @pytest.mark.requires_network
**Purpose**: Mark tests that require network connectivity

**Usage**: Skip in offline environments
```python
@pytest.mark.requires_network
def test_external_api_integration():
    """Test integration with external API."""
    # Network-dependent test logic
```

## Running Tests by Marker

### Run specific marker groups:
```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run unit and integration tests
pytest -m "unit or integration"

# Skip slow tests
pytest -m "not slow"

# Run only TDD red phase tests
pytest -m tdd_red

# Run performance tests
pytest -m performance
```

### Combine markers:
```bash
# Run unit tests that are not slow
pytest -m "unit and not slow"

# Run integration tests that don't require network
pytest -m "integration and not requires_network"
```

## Custom Test Execution Profiles

### Development Profile (Fast Feedback)
```bash
# Quick tests for development feedback
pytest -m "unit and not slow" --tb=short
```

### CI/CD Profile (Comprehensive)
```bash
# Full test suite for CI/CD
pytest -m "not requires_gpu and not requires_network" --cov=src --cov-fail-under=95
```

### Performance Profile
```bash
# Performance and slow tests
pytest -m "performance or slow" --tb=short
```

## Marker Best Practices

1. **Use multiple markers** when appropriate:
   ```python
   @pytest.mark.integration
   @pytest.mark.slow
   def test_large_database_operation():
       pass
   ```

2. **Document marker rationale** in test docstrings:
   ```python
   @pytest.mark.requires_network
   def test_api_call():
       """Test API integration. Marked requires_network due to external dependency."""
       pass
   ```

3. **Keep marker usage consistent** across the test suite

4. **Use descriptive marker names** that clearly indicate their purpose

5. **Update CI/CD configurations** when adding new markers

## Configuration in pyproject.toml

Current marker configuration:
```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "slow: Slow running tests",
    "tdd_red: TDD Red phase tests",
    "tdd_green: TDD Green phase tests", 
    "performance: Performance tests",
    "requires_gpu: Tests requiring GPU",
    "requires_network: Tests requiring network",
]
``` 