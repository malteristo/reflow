"""
Tests for Collection Metadata Model Change Integration.

This module tests the enhanced CollectionMetadata class with model fingerprint tracking
and reindex management capabilities for model change detection integration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.research_agent_backend.models.metadata_schema.collection_metadata import (
    CollectionMetadata,
    ReindexStatus
)
from src.research_agent_backend.models.metadata_schema.enums import CollectionType


class TestCollectionMetadataModelIntegration:
    """Test model integration features of CollectionMetadata."""
    
    def test_default_model_integration_fields(self):
        """Test that new model integration fields have proper defaults."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        assert metadata.embedding_model_fingerprint is None
        assert metadata.model_name is None
        assert metadata.model_version is None
        assert metadata.reindex_status == "not_required"
        assert metadata.last_reindex_timestamp is None
        assert metadata.original_document_count == 0
    
    def test_update_model_fingerprint(self):
        """Test updating model fingerprint information."""
        metadata = CollectionMetadata(collection_name="test-collection")
        original_updated_at = metadata.updated_at
        
        # Small delay to ensure timestamp difference
        with patch('src.research_agent_backend.models.metadata_schema.collection_metadata.datetime') as mock_datetime:
            future_time = original_updated_at + timedelta(seconds=1)
            mock_datetime.utcnow.return_value = future_time
            
            metadata.update_model_fingerprint(
                fingerprint="abc123def456",
                model_name="text-embedding-3-small",
                model_version="1.0.0"
            )
        
        assert metadata.embedding_model_fingerprint == "abc123def456"
        assert metadata.model_name == "text-embedding-3-small"
        assert metadata.model_version == "1.0.0"
        assert metadata.updated_at > original_updated_at
    
    def test_set_reindex_status_completed(self):
        """Test setting reindex status to completed updates timestamp."""
        metadata = CollectionMetadata(collection_name="test-collection")
        original_updated_at = metadata.updated_at
        
        with patch('src.research_agent_backend.models.metadata_schema.collection_metadata.datetime') as mock_datetime:
            future_time = original_updated_at + timedelta(seconds=1)
            mock_datetime.utcnow.return_value = future_time
            
            metadata.set_reindex_status("completed")
        
        assert metadata.reindex_status == "completed"
        assert metadata.last_reindex_timestamp == future_time
        assert metadata.updated_at == future_time
    
    def test_set_reindex_status_failed(self):
        """Test setting reindex status to failed updates timestamp."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        with patch('src.research_agent_backend.models.metadata_schema.collection_metadata.datetime') as mock_datetime:
            future_time = datetime.utcnow() + timedelta(seconds=1)
            mock_datetime.utcnow.return_value = future_time
            
            metadata.set_reindex_status("failed")
        
        assert metadata.reindex_status == "failed"
        assert metadata.last_reindex_timestamp == future_time
    
    def test_set_reindex_status_in_progress_no_timestamp(self):
        """Test setting reindex status to in_progress doesn't update reindex timestamp."""
        metadata = CollectionMetadata(collection_name="test-collection")
        original_reindex_timestamp = metadata.last_reindex_timestamp
        
        metadata.set_reindex_status("in_progress")
        
        assert metadata.reindex_status == "in_progress"
        assert metadata.last_reindex_timestamp == original_reindex_timestamp  # Should be None still
    
    def test_set_reindex_status_no_timestamp_update(self):
        """Test setting reindex status with update_timestamp=False."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        metadata.set_reindex_status("completed", update_timestamp=False)
        
        assert metadata.reindex_status == "completed"
        assert metadata.last_reindex_timestamp is None
    
    def test_requires_reindexing_no_fingerprint(self):
        """Test requires_reindexing returns True when no fingerprint is stored."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        result = metadata.requires_reindexing("current-fingerprint")
        
        assert result is True
    
    def test_requires_reindexing_fingerprint_mismatch(self):
        """Test requires_reindexing returns True when fingerprints don't match."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "old-fingerprint"
        
        result = metadata.requires_reindexing("new-fingerprint")
        
        assert result is True
    
    def test_requires_reindexing_status_pending(self):
        """Test requires_reindexing returns True when status is pending."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "same-fingerprint"
        metadata.reindex_status = "pending"
        
        result = metadata.requires_reindexing("same-fingerprint")
        
        assert result is True
    
    def test_requires_reindexing_status_failed(self):
        """Test requires_reindexing returns True when status is failed."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "same-fingerprint"
        metadata.reindex_status = "failed"
        
        result = metadata.requires_reindexing("same-fingerprint")
        
        assert result is True
    
    def test_requires_reindexing_false(self):
        """Test requires_reindexing returns False when no reindexing is needed."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "same-fingerprint"
        metadata.reindex_status = "completed"
        
        result = metadata.requires_reindexing("same-fingerprint")
        
        assert result is False
    
    def test_is_model_compatible_no_stored_info(self):
        """Test is_model_compatible returns False when no model info is stored."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        result = metadata.is_model_compatible("fingerprint", "model-name")
        
        assert result is False
    
    def test_is_model_compatible_true(self):
        """Test is_model_compatible returns True when all conditions match."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "test-fingerprint"
        metadata.model_name = "test-model"
        metadata.reindex_status = "completed"
        
        result = metadata.is_model_compatible("test-fingerprint", "test-model")
        
        assert result is True
    
    def test_is_model_compatible_fingerprint_mismatch(self):
        """Test is_model_compatible returns False when fingerprints don't match."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "old-fingerprint"
        metadata.model_name = "test-model"
        metadata.reindex_status = "completed"
        
        result = metadata.is_model_compatible("new-fingerprint", "test-model")
        
        assert result is False
    
    def test_is_model_compatible_name_mismatch(self):
        """Test is_model_compatible returns False when model names don't match."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "test-fingerprint"
        metadata.model_name = "old-model"
        metadata.reindex_status = "completed"
        
        result = metadata.is_model_compatible("test-fingerprint", "new-model")
        
        assert result is False
    
    def test_is_model_compatible_not_completed(self):
        """Test is_model_compatible returns False when reindex status is not completed."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "test-fingerprint"
        metadata.model_name = "test-model"
        metadata.reindex_status = "pending"
        
        result = metadata.is_model_compatible("test-fingerprint", "test-model")
        
        assert result is False
    
    def test_get_reindex_progress_info(self):
        """Test get_reindex_progress_info returns complete status information."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "test-fingerprint"
        metadata.model_name = "test-model"
        metadata.model_version = "1.0.0"
        metadata.reindex_status = "completed"
        metadata.document_count = 100
        metadata.original_document_count = 95
        
        timestamp = datetime.utcnow()
        metadata.last_reindex_timestamp = timestamp
        
        result = metadata.get_reindex_progress_info()
        
        assert result["status"] == "completed"
        assert result["last_reindex"] == timestamp.isoformat()
        assert result["model_fingerprint"] == "test-fingerprint"
        assert result["model_name"] == "test-model"
        assert result["model_version"] == "1.0.0"
        assert result["document_count"] == 100
        assert result["original_document_count"] == 95
        assert result["requires_reindexing"] is False
    
    def test_get_reindex_progress_info_no_timestamp(self):
        """Test get_reindex_progress_info handles None timestamp gracefully."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        result = metadata.get_reindex_progress_info()
        
        assert result["last_reindex"] is None
    
    def test_to_dict_includes_model_fields(self):
        """Test to_dict includes all model integration fields."""
        metadata = CollectionMetadata(collection_name="test-collection")
        metadata.embedding_model_fingerprint = "test-fingerprint"
        metadata.model_name = "test-model"
        metadata.model_version = "1.0.0"
        metadata.reindex_status = "completed"
        metadata.original_document_count = 50
        
        timestamp = datetime.utcnow()
        metadata.last_reindex_timestamp = timestamp
        
        result = metadata.to_dict()
        
        assert result["embedding_model_fingerprint"] == "test-fingerprint"
        assert result["model_name"] == "test-model"
        assert result["model_version"] == "1.0.0"
        assert result["reindex_status"] == "completed"
        assert result["last_reindex_timestamp"] == timestamp.isoformat()
        assert result["original_document_count"] == 50
    
    def test_to_dict_none_timestamp(self):
        """Test to_dict handles None reindex timestamp correctly."""
        metadata = CollectionMetadata(collection_name="test-collection")
        
        result = metadata.to_dict()
        
        assert result["last_reindex_timestamp"] is None
    
    def test_backward_compatibility(self):
        """Test that existing functionality remains intact with new fields."""
        metadata = CollectionMetadata(
            collection_name="backward-compat-test",
            collection_type=CollectionType.GENERAL,
            description="Test collection",
            embedding_model="old-model",
            embedding_dimension=384
        )
        
        # Test existing methods still work
        metadata.update_stats(document_count=10, chunk_count=50, size_bytes=1024)
        
        assert metadata.collection_name == "backward-compat-test"
        assert metadata.document_count == 10
        assert metadata.chunk_count == 50
        assert metadata.total_size_bytes == 1024
        
        # New fields should have defaults
        assert metadata.embedding_model_fingerprint is None
        assert metadata.reindex_status == "not_required"


class TestReindexStatusEnum:
    """Test the ReindexStatus type alias."""
    
    def test_valid_reindex_statuses(self):
        """Test that all valid reindex statuses are accepted."""
        metadata = CollectionMetadata(collection_name="test")
        
        valid_statuses = ["pending", "in_progress", "completed", "failed", "not_required"]
        
        for status in valid_statuses:
            metadata.set_reindex_status(status)  # type: ignore
            assert metadata.reindex_status == status
    
    def test_reindex_status_in_progress_info(self):
        """Test reindex progress info when status is in_progress."""
        metadata = CollectionMetadata(collection_name="test")
        metadata.set_reindex_status("in_progress")
        
        info = metadata.get_reindex_progress_info()
        
        assert info["status"] == "in_progress"
        assert info["requires_reindexing"] is True  # in_progress means still needs completion 