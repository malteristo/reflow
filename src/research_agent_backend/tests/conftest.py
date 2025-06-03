"""Shared test fixtures and configuration for Research Agent backend tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List
import numpy as np

# Configure asyncio for pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(autouse=True)
def ensure_integration_patches_cleanup():
    """
    Global fixture to ensure integration patches are always cleaned up.
    
    This prevents integration test patches from persisting across tests,
    which was causing interface mismatch errors in the test suite.
    """
    # Yield to run the test
    yield
    
    # Cleanup after each test
    try:
        from research_agent_backend.core.integration_pipeline import ensure_patches_removed
        ensure_patches_removed()
    except ImportError:
        # Module might not be available in all test contexts
        pass


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    yield temp_path
    
    # Cleanup temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)


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
def sample_embeddings():
    """Provide sample embedding vectors for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4]
    ]


@pytest.fixture
def sample_metadata():
    """Provide sample metadata for testing."""
    return [
        {
            "document_type": "markdown",
            "source_path": "/docs/ml/basics.md",
            "created_at": "2024-01-01T00:00:00Z",
            "team_id": "team1",
            "priority": 5,
            "tags": ["python", "ml"]
        },
        {
            "document_type": "code",
            "source_path": "/src/models/neural_net.py",
            "created_at": "2024-01-02T00:00:00Z",
            "team_id": "team2",
            "priority": 8,
            "tags": ["python", "deep-learning"]
        },
        {
            "document_type": "markdown",
            "source_path": "/docs/nlp/intro.md",
            "created_at": "2024-01-03T00:00:00Z",
            "team_id": "team1",
            "priority": 3,
            "tags": ["nlp", "transformers"]
        }
    ]


@pytest.fixture
def config_manager():
    """Provide a mock ConfigManager for testing."""
    mock_config = Mock()
    mock_config.get.return_value = {
        "vector_store": {
            "type": "chromadb",
            "persist_directory": None
        },
        "query": {
            "max_results": 100,
            "similarity_threshold": 0.0,
            "enable_caching": True
        }
    }
    mock_config.project_root = "/tmp/test_project"
    return mock_config


@pytest.fixture
def in_memory_chroma_manager():
    """Provide a mock ChromaDBManager for testing."""
    mock_chroma = Mock()
    mock_chroma.is_connected = True
    mock_chroma.client = Mock()
    mock_chroma.query_collection.return_value = {
        "ids": [["doc1", "doc2", "doc3"]],
        "documents": [["Sample doc 1", "Sample doc 2", "Sample doc 3"]],
        "metadatas": [[{"type": "test"}, {"type": "test"}, {"type": "test"}]],
        "distances": [[0.1, 0.2, 0.3]]
    }
    mock_chroma.get_collection.return_value = Mock()
    mock_chroma.list_collections.return_value = [
        {"name": "test_collection", "metadata": {}}
    ]
    mock_chroma.health_check.return_value = {"status": "healthy"}
    return mock_chroma


@pytest.fixture
def collection_type_manager():
    """Provide a mock CollectionTypeManager for testing."""
    mock_ctm = Mock()
    mock_ctm.get_collection_type_config.return_value = {
        "max_documents": 1000,
        "hnsw_parameters": {"ef": 100, "M": 16}
    }
    mock_ctm.determine_collection_type.return_value = "FUNDAMENTAL"
    mock_ctm.get_available_types.return_value = ["FUNDAMENTAL", "PROJECT_SPECIFIC"]
    return mock_ctm


@pytest.fixture
def data_preparation_manager():
    """Provide a mock DataPreparationManager for testing."""
    mock_dpm = Mock()
    mock_dpm.clean_text.return_value = "Cleaned and normalized text"
    mock_dpm.normalize_vector.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_dpm.process_document.return_value = {
        "cleaned_content": "Processed content",
        "metadata": {"processed": True}
    }
    return mock_dpm


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