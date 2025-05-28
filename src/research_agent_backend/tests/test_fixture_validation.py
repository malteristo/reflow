"""Test to validate that fixtures work correctly in practice."""

import pytest
from pathlib import Path


def test_temp_vector_store_fixture(temp_vector_store):
    """Test that temp_vector_store fixture provides expected functionality."""
    assert temp_vector_store is not None
    assert hasattr(temp_vector_store, 'path')
    assert hasattr(temp_vector_store, 'is_temporary')
    assert temp_vector_store.is_temporary is True
    assert isinstance(temp_vector_store.path, Path)


def test_sample_documents_fixture(sample_documents):
    """Test that sample_documents fixture provides test data."""
    assert sample_documents is not None
    assert isinstance(sample_documents, list)
    assert len(sample_documents) == 3
    
    # Check first document structure
    doc = sample_documents[0]
    assert 'id' in doc
    assert 'content' in doc
    assert 'metadata' in doc
    assert doc['id'] == 'doc_1'


def test_mock_embedding_service_fixture(mock_embedding_service):
    """Test that mock_embedding_service fixture provides expected interface."""
    assert mock_embedding_service is not None
    assert hasattr(mock_embedding_service, 'embed_text')
    assert hasattr(mock_embedding_service, 'embed_batch')
    assert hasattr(mock_embedding_service, 'model_name')
    assert hasattr(mock_embedding_service, 'embedding_dim')
    
    # Test that mock methods return expected values
    embedding = mock_embedding_service.embed_text.return_value
    assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert mock_embedding_service.embedding_dim == 5


def test_test_config_fixture(test_config):
    """Test that test_config fixture provides configuration dictionary."""
    assert test_config is not None
    assert isinstance(test_config, dict)
    assert 'embedding_model' in test_config
    assert 'chunk_size' in test_config
    assert 'vector_store' in test_config
    assert test_config['chunk_size'] == 512


def test_utilities_work_correctly():
    """Test that test utilities work as expected."""
    from .utils import create_test_document, create_test_embeddings
    
    # Test document creation
    doc = create_test_document()
    assert 'title' in doc
    assert 'content' in doc
    assert 'metadata' in doc
    
    # Test embedding creation
    embeddings = create_test_embeddings(dimension=3, count=2)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 3
    assert len(embeddings[1]) == 3 