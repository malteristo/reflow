# Testing Patterns Quick Reference

> **Copy-paste code snippets for common testing patterns**

## CLI Testing Patterns

### Basic Command Test
```python
@pytest.mark.cli
class TestCLICommands:
    def setup_method(self):
        self.runner = CliRunner()
    
    def test_command_success(self):
        result = self.runner.invoke(app, ["command", "--arg", "value"])
        assert result.exit_code == 0
        assert "Success" in result.stdout
```

### Interactive Prompt Testing
```python
@patch('typer.prompt')
def test_interactive_prompt(self, mock_prompt):
    mock_prompt.return_value = "user_input"
    result = self.runner.invoke(app, ["interactive-command"])
    assert result.exit_code == 0
    mock_prompt.assert_called_once()
```

### File Validation Testing
```python
def test_file_validation(self, temp_dir):
    test_file = temp_dir / "test.txt"
    test_file.write_text("content")
    
    result = self.runner.invoke(app, ["command", str(test_file)])
    assert result.exit_code == 0
```

## MCP Server Testing Patterns

### Tool Registration Test
```python
@pytest.mark.mcp
class TestMCPTools:
    async def test_tool_registration(self):
        server = FastMCP("TestServer")
        
        @server.tool()
        def test_tool(input_text: str) -> str:
            return f"Processed: {input_text}"
        
        async with Client(server) as client:
            tools = await client.list_tools()
            assert "test_tool" in [t.name for t in tools.tools]
```

### Tool Execution Test
```python
async def test_tool_execution(self):
    server = FastMCP("TestServer")
    
    @server.tool()
    def echo_tool(message: str) -> str:
        return f"Echo: {message}"
    
    async with Client(server) as client:
        result = await client.call_tool("echo_tool", {"message": "test"})
        assert isinstance(result[0], TextContent)
        assert result[0].text == "Echo: test"
```

### Resource Management Test
```python
async def test_resource_access(self):
    server = FastMCP("TestServer")
    
    @server.resource("config://version")
    def get_version() -> str:
        return "1.0.0"
    
    async with Client(server) as client:
        resources = await client.list_resources()
        assert "config://version" in [r.uri for r in resources.resources]
```

## Async Testing Patterns

### Basic Async Function Test
```python
@pytest.mark.async
class TestAsyncOperations:
    async def test_async_function(self):
        result = await async_function("input")
        assert result == expected_output
```

### Concurrent Operations Test
```python
async def test_concurrent_execution(self):
    tasks = [async_function(f"input_{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 5
    assert all(r is not None for r in results)
```

### Timeout Handling Test
```python
async def test_timeout_handling(self):
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(slow_async_function(), timeout=0.1)
```

### Async Context Manager Test
```python
async def test_async_context_manager(self):
    async with AsyncResource() as resource:
        result = await resource.process_data("input")
        assert result is not None
    # Resource should be properly cleaned up
```

## Core Service Testing Patterns

### Service with Mocked Dependencies
```python
@pytest.mark.unit
class TestEmbeddingService:
    def setup_method(self):
        self.mock_config = Mock()
        self.service = EmbeddingService(self.mock_config)
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_generation(self, mock_transformer):
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_transformer.return_value = mock_model
        
        embeddings = self.service.generate_embeddings(["test text"])
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
```

### Database Operation Test
```python
@patch('chromadb.Client')
def test_vector_store_operations(self, mock_client):
    mock_collection = Mock()
    mock_client.return_value.get_or_create_collection.return_value = mock_collection
    
    service = VectorStoreService(mock_client)
    service.add_documents(["doc1", "doc2"], [[0.1, 0.2], [0.3, 0.4]])
    
    mock_collection.add.assert_called_once()
```

### Configuration Loading Test
```python
def test_config_loading(self, temp_dir):
    config_file = temp_dir / "config.json"
    config_file.write_text('{"embedding_model": "test-model"}')
    
    config = load_config(config_file)
    assert config.embedding_model == "test-model"
```

## Error Handling Patterns

### Exception Testing
```python
def test_service_exception_handling(self):
    with patch('requests.post') as mock_post:
        mock_post.side_effect = ConnectionError("Network error")
        
        with pytest.raises(ServiceError) as exc_info:
            service.call_external_api("data")
        
        assert "Network error" in str(exc_info.value)
```

### Async Exception Testing
```python
async def test_async_exception_handling(self):
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.side_effect = asyncio.TimeoutError()
        
        with pytest.raises(ServiceTimeoutError):
            await async_service.process_request("data")
```

### Validation Error Testing
```python
def test_input_validation(self):
    with pytest.raises(ValidationError) as exc_info:
        service.process_invalid_input(None)
    
    assert "Required field" in str(exc_info.value)
```

## Integration Testing Patterns

### End-to-End Pipeline Test
```python
@pytest.mark.integration
class TestRAGPipeline:
    def setup_method(self):
        self.pipeline = RAGPipeline(test_config)
    
    async def test_complete_query_flow(self):
        # Setup test documents
        await self.pipeline.ingest_documents(["test doc content"])
        
        # Execute query
        results = await self.pipeline.query("test query")
        
        # Validate results
        assert len(results) > 0
        assert all(r.relevance_score > 0.5 for r in results)
```

### Service Integration Test
```python
async def test_embedding_to_vector_store_integration(self):
    embedding_service = EmbeddingService(config)
    vector_store = VectorStore(config)
    
    # Generate embeddings
    embeddings = embedding_service.generate_embeddings(["test text"])
    
    # Store in vector database
    doc_ids = vector_store.add_embeddings(embeddings, ["test text"])
    
    # Query back
    results = vector_store.query(embeddings[0], top_k=1)
    assert len(results) == 1
    assert results[0].id == doc_ids[0]
```

## TDD Red-Green-Refactor Patterns

### Red Phase Test (Failing)
```python
@pytest.mark.tdd_red
def test_new_feature_behavior(self):
    """This test defines the desired behavior but will fail initially."""
    result = new_feature("input")
    assert result == "expected_output"
    # This will fail because new_feature() doesn't exist yet
```

### Green Phase Test (Minimal Implementation)
```python
@pytest.mark.tdd_green  
def test_minimal_implementation(self):
    """Test with minimal implementation to make test pass."""
    def new_feature(input_data):
        return "expected_output"  # Hard-coded minimal implementation
    
    result = new_feature("input")
    assert result == "expected_output"
```

### Refactor Phase Test (Enhanced Implementation)
```python
def test_refactored_implementation(self):
    """Test with proper implementation after refactoring."""
    def new_feature(input_data):
        # Proper implementation with real logic
        processed = process_input(input_data)
        return generate_output(processed)
    
    result = new_feature("input")
    assert result == "expected_output"
```

## Common Fixtures

### Temporary Directory Fixture
```python
@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
```

### Mock Configuration Fixture
```python
@pytest.fixture
def mock_config():
    return {
        "embedding_model": "test-model",
        "chunk_size": 512,
        "vector_db_path": ":memory:"
    }
```

### Async Test Server Fixture
```python
@pytest.fixture
async def test_mcp_server():
    server = FastMCP("TestServer")
    yield server
    # Cleanup handled automatically
```

### Mock Database Fixture
```python
@pytest.fixture
def mock_vector_store():
    with patch('chromadb.Client') as mock_client:
        mock_collection = Mock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield mock_collection
```

## Test Data Patterns

### Sample Document Creation
```python
def create_test_document(content: str = "Test content") -> Path:
    """Create a temporary test document."""
    test_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
    test_file.write(f"# Test Document\n\n{content}")
    test_file.close()
    return Path(test_file.name)
```

### Mock Embedding Data
```python
def generate_mock_embeddings(count: int, dimensions: int = 384) -> List[List[float]]:
    """Generate mock embedding vectors."""
    return [[0.1 * i * j for j in range(dimensions)] for i in range(count)]
```

### Sample Configuration
```python
def create_test_config() -> dict:
    """Create test configuration."""
    return {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "vector_db_path": ":memory:",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2"
    }
``` 