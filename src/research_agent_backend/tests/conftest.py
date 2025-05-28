"""Shared test fixtures and configuration for Research Agent backend tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

# Configure asyncio for pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture
def temp_vector_store():
    """Provide a temporary vector store for testing."""
    # Create temporary directory for vector store
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Mock vector store configuration
    mock_store = Mock()
    mock_store.path = temp_path
    mock_store.is_temporary = True
    mock_store.collection_names = []
    
    yield mock_store
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        {
            "id": "doc_1",
            "content": "This is a sample document about machine learning.",
            "metadata": {"source": "test", "category": "ml"}
        },
        {
            "id": "doc_2", 
            "content": "Another document discussing natural language processing.",
            "metadata": {"source": "test", "category": "nlp"}
        },
        {
            "id": "doc_3",
            "content": "A third document covering computer vision topics.",
            "metadata": {"source": "test", "category": "cv"}
        }
    ]


@pytest.fixture
def mock_embedding_service():
    """Provide a mock embedding service for testing."""
    mock_service = Mock()
    
    # Configure mock to return consistent embeddings
    mock_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_service.embed_batch.return_value = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ]
    mock_service.model_name = "test-embedding-model"
    mock_service.embedding_dim = 5
    
    return mock_service


@pytest.fixture
def test_config():
    """Provide test configuration settings."""
    return {
        "embedding_model": "test-model",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "vector_store": {
            "type": "chromadb",
            "persist_directory": None  # Use in-memory for tests
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(levelname)s: %(message)s"
        }
    }


@pytest.fixture
def integration_config(test_config):
    """Configuration optimized for integration testing."""
    config = test_config.copy()
    config.update({
        "vector_store": {
            "provider": "chromadb",
            "path": tempfile.mkdtemp(),
            "embedding_function": "sentence-transformers",
            "collection_name": "integration_test"
        },
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 10,
            "max_length": 512,
            "dimension": 384  # Standard dimension for all-MiniLM-L6-v2
        },
        "chunking": {
            "strategy": "recursive",
            "chunk_size": 256,
            "chunk_overlap": 50
        },
        "data_preparation": {
            "normalization": "unit_vector",
            "dimensionality_reduction": None,
            "batch_size": 100
        }
    })
    return config 