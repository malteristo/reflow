"""
Integration tests for Model Change Detection with Document Chunking Pipeline.

Tests the integration between the model change detection system and the document
processing pipeline, ensuring proper model tracking, cache invalidation, and
re-chunking workflows.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.research_agent_backend.core.document_processor import (
    ChunkConfig,
    RecursiveChunker,
    ChunkingPipelineWithModelDetection,
    ModelChangeIntegration,
    ModelAwareChunkResult
)
from src.research_agent_backend.core.model_change_detection import (
    ModelFingerprint,
    ModelChangeEvent,
    ModelChangeDetector
)


class TestModelChangeIntegration:
    """Test suite for model change detection integration with document processing."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service for testing."""
        service = Mock()
        service.get_model_info.return_value = {
            'model_name': 'test-model-v1',
            'model_version': '1.0.0',
            'model_path': '/path/to/model',
            'dimension': 384
        }
        service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='test-model-v1',
            model_type='local',
            version='1.0.0', 
            checksum='abc123def456'
        )
        return service

    @pytest.fixture
    def chunk_config(self):
        """Create a basic chunk configuration."""
        return ChunkConfig(
            chunk_size=256,
            chunk_overlap=50,
            min_chunk_size=50
        )

    @pytest.fixture
    def chunker(self, chunk_config):
        """Create a recursive chunker for testing."""
        return RecursiveChunker(chunk_config)

    @pytest.fixture
    def model_integration(self, mock_embedding_service):
        """Create a model change integration instance."""
        return ModelChangeIntegration(mock_embedding_service)

    @pytest.fixture
    def chunking_pipeline(self, mock_embedding_service):
        """Create a chunking pipeline with model detection."""
        config = ChunkConfig(chunk_size=256, chunk_overlap=50, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        return ChunkingPipelineWithModelDetection(chunker, mock_embedding_service)

    def test_model_integration_initialization(self, model_integration, mock_embedding_service):
        """Test that model integration initializes correctly."""
        assert model_integration.embedding_service == mock_embedding_service
        assert model_integration.model_detector is not None
        assert isinstance(model_integration.model_detector, ModelChangeDetector)

    def test_get_current_model_fingerprint(self, model_integration):
        """Test fingerprint generation from embedding service."""
        fingerprint = model_integration.get_current_model_fingerprint()
        
        assert fingerprint is not None
        assert fingerprint.model_name == 'test-model-v1'
        assert fingerprint.version == '1.0.0'
        assert fingerprint.checksum == 'abc123def456'

    def test_get_model_fingerprint_without_service(self):
        """Test fingerprint generation without embedding service."""
        integration = ModelChangeIntegration(embedding_service=None)
        fingerprint = integration.get_current_model_fingerprint()
        
        assert fingerprint is None

    def test_get_model_fingerprint_fallback(self, mock_embedding_service):
        """Test fingerprint generation using fallback method."""
        # Remove the generate_model_fingerprint method to test fallback
        del mock_embedding_service.generate_model_fingerprint
        
        integration = ModelChangeIntegration(mock_embedding_service)
        fingerprint = integration.get_current_model_fingerprint()
        
        assert fingerprint is not None
        assert fingerprint.model_name == 'test-model-v1'
        assert fingerprint.version == '1.0.0'
        assert len(fingerprint.checksum) == 32  # MD5 hash length

    def test_register_current_model(self, model_integration):
        """Test model registration with change detector."""
        success = model_integration.register_current_model()
        
        assert success is True
        assert model_integration._cached_fingerprint is not None

    def test_check_for_model_changes_no_change(self, model_integration):
        """Test model change detection when no change occurred."""
        # Register the model first
        model_integration.register_current_model()
        
        # Check for changes (should be none)
        change_detected = model_integration.check_for_model_changes()
        
        # Since we're using the same fingerprint, no change should be detected
        assert change_detected is False

    def test_check_for_model_changes_with_change(self, model_integration, mock_embedding_service):
        """Test model change detection when model actually changed."""
        # Register initial model
        model_integration.register_current_model()
        
        # Change the model fingerprint
        mock_embedding_service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='test-model-v2',
            model_type='local',
            version='2.0.0',
            checksum='xyz789abc123'
        )
        
        # Check for changes
        change_detected = model_integration.check_for_model_changes()
        
        assert change_detected is True

    def test_enhance_chunk_with_model_info(self, model_integration, chunker):
        """Test chunk enhancement with model information."""
        # Create a basic chunk
        base_chunks = chunker.chunk_text("This is a test document for chunking.")
        base_chunk = base_chunks[0]
        
        # Enhance with model info
        enhanced_chunk = model_integration.enhance_chunk_with_model_info(base_chunk)
        
        assert isinstance(enhanced_chunk, ModelAwareChunkResult)
        assert enhanced_chunk.model_fingerprint == 'abc123def456'
        assert enhanced_chunk.model_name == 'test-model-v1'
        assert enhanced_chunk.model_version == '1.0.0'
        assert enhanced_chunk.created_at is not None
        
        # Check metadata enhancement
        assert 'model_fingerprint' in enhanced_chunk.processing_metadata
        assert 'model_name' in enhanced_chunk.processing_metadata
        assert 'model_tracked_at' in enhanced_chunk.processing_metadata

    def test_should_invalidate_chunks_no_change(self, model_integration, chunker):
        """Test chunk invalidation when no model change occurred."""
        # Register model and create chunks
        model_integration.register_current_model()
        base_chunks = chunker.chunk_text("Test content for invalidation testing.")
        enhanced_chunks = [model_integration.enhance_chunk_with_model_info(chunk) for chunk in base_chunks]
        
        # Check if invalidation is needed (should be False)
        should_invalidate = model_integration.should_invalidate_chunks(enhanced_chunks)
        
        assert should_invalidate is False

    def test_should_invalidate_chunks_with_change(self, model_integration, chunker, mock_embedding_service):
        """Test chunk invalidation when model changed."""
        # Register initial model and create chunks
        model_integration.register_current_model()
        base_chunks = chunker.chunk_text("Test content for invalidation testing.")
        enhanced_chunks = [model_integration.enhance_chunk_with_model_info(chunk) for chunk in base_chunks]
        
        # Change the model
        mock_embedding_service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='new-model',
            model_type='local',
            version='2.0.0',
            checksum='new_fingerprint_hash'
        )
        
        # Register the new model
        model_integration.register_current_model()
        
        # Check if invalidation is needed (should be True)
        should_invalidate = model_integration.should_invalidate_chunks(enhanced_chunks)
        
        assert should_invalidate is True


class TestChunkingPipelineWithModelDetection:
    """Test suite for the model-aware chunking pipeline."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.get_model_info.return_value = {
            'model_name': 'pipeline-test-model',
            'model_version': '1.0.0'
        }
        service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='pipeline-test-model',
            model_type='local',
            version='1.0.0',
            checksum='pipeline123hash'
        )
        return service

    @pytest.fixture
    def chunking_pipeline(self, mock_embedding_service):
        """Create a chunking pipeline with model detection."""
        config = ChunkConfig(chunk_size=256, chunk_overlap=50, min_chunk_size=50)
        chunker = RecursiveChunker(config)
        return ChunkingPipelineWithModelDetection(chunker, mock_embedding_service)

    def test_pipeline_initialization(self, chunking_pipeline):
        """Test that the pipeline initializes correctly."""
        assert chunking_pipeline.chunker is not None
        assert chunking_pipeline.model_integration is not None
        assert chunking_pipeline.logger is not None

    def test_chunk_text_with_model_tracking(self, chunking_pipeline):
        """Test text chunking with model tracking."""
        text = "This is a longer test document that should be split into multiple chunks for testing the model-aware chunking pipeline functionality."
        
        chunks = chunking_pipeline.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ModelAwareChunkResult) for chunk in chunks)
        assert all(chunk.model_fingerprint == 'pipeline123hash' for chunk in chunks)
        assert all(chunk.model_name == 'pipeline-test-model' for chunk in chunks)

    def test_chunk_text_with_model_change_detection(self, chunking_pipeline, mock_embedding_service):
        """Test chunking when a model change is detected."""
        # First chunking operation
        text1 = "First document to be chunked."
        chunks1 = chunking_pipeline.chunk_text(text1)
        
        # Change the model
        mock_embedding_service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='updated-model',
            model_type='local',
            version='2.0.0', 
            checksum='updated123hash'
        )
        
        # Second chunking operation (should detect model change)
        text2 = "Second document with new model."
        chunks2 = chunking_pipeline.chunk_text(text2)
        
        # Verify chunks from second operation have new model info
        assert all(chunk.model_fingerprint == 'updated123hash' for chunk in chunks2)
        assert all(chunk.model_name == 'updated-model' for chunk in chunks2)

    def test_invalidate_cache_if_needed(self, chunking_pipeline, mock_embedding_service):
        """Test cache invalidation functionality."""
        # Create some chunks with current model
        text = "Content for cache invalidation testing."
        chunks = chunking_pipeline.chunk_text(text)
        
        # Initially, no invalidation should be needed
        assert chunking_pipeline.invalidate_cache_if_needed(chunks) is False
        
        # Change the model
        mock_embedding_service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='cache-test-new-model',
            model_type='local',
            version='3.0.0',
            checksum='cache123new'
        )
        
        # Re-register to update the model
        chunking_pipeline.model_integration.register_current_model()
        
        # Now invalidation should be recommended
        assert chunking_pipeline.invalidate_cache_if_needed(chunks) is True

    def test_get_model_status(self, chunking_pipeline):
        """Test model status reporting."""
        status = chunking_pipeline.get_model_status()
        
        assert 'current_model' in status
        assert 'change_detected' in status
        assert 'model_registered' in status
        
        assert status['current_model']['name'] == 'pipeline-test-model'
        assert status['current_model']['fingerprint'] == 'pipeline123hash'
        assert status['model_registered'] is True

    def test_chunk_document_sections(self, chunking_pipeline):
        """Test document section chunking with model tracking."""
        # Mock document sections
        sections = [
            Mock(content="First section content", title="Section 1"),
            Mock(content="Second section content with more text", title="Section 2")
        ]
        
        chunks = chunking_pipeline.chunk_document_sections(sections)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, ModelAwareChunkResult) for chunk in chunks)
        assert all(chunk.model_fingerprint is not None for chunk in chunks)


class TestModelChangeIntegrationEdgeCases:
    """Test edge cases and error conditions for model change integration."""

    @pytest.fixture
    def chunk_config_basic(self):
        """Create a basic chunk configuration for edge case testing."""
        return ChunkConfig(
            chunk_size=128,
            chunk_overlap=20,
            min_chunk_size=32
        )

    @pytest.fixture
    def chunker(self, chunk_config_basic):
        """Create a recursive chunker for edge case testing."""
        return RecursiveChunker(chunk_config_basic)

    @pytest.fixture
    def mock_embedding_service_for_edge_cases(self):
        """Create a mock embedding service for edge case testing."""
        service = Mock()
        service.get_model_info.return_value = {
            'model_name': 'edge-case-model',
            'model_version': '1.0.0'
        }
        service.generate_model_fingerprint.return_value = ModelFingerprint(
            model_name='edge-case-model',
            model_type='local',
            version='1.0.0',
            checksum='edge123case'
        )
        return service

    @pytest.fixture
    def model_integration(self, mock_embedding_service_for_edge_cases):
        """Create a model change integration instance for edge case testing."""
        return ModelChangeIntegration(mock_embedding_service_for_edge_cases)

    def test_integration_with_failing_embedding_service(self):
        """Test integration when embedding service raises exceptions."""
        failing_service = Mock()
        failing_service.generate_model_fingerprint.side_effect = Exception("Service error")
        failing_service.get_model_info.side_effect = Exception("Info error")
        
        integration = ModelChangeIntegration(failing_service)
        
        # Should handle exceptions gracefully
        fingerprint = integration.get_current_model_fingerprint()
        assert fingerprint is None
        
        registration_success = integration.register_current_model()
        assert registration_success is False

    def test_chunk_enhancement_without_fingerprint(self, chunker):
        """Test chunk enhancement when no model fingerprint is available."""
        integration = ModelChangeIntegration(embedding_service=None)
        
        base_chunks = chunker.chunk_text("Test content")
        base_chunk = base_chunks[0]
        
        enhanced_chunk = integration.enhance_chunk_with_model_info(base_chunk)
        
        assert isinstance(enhanced_chunk, ModelAwareChunkResult)
        assert enhanced_chunk.model_fingerprint is None
        assert enhanced_chunk.model_name is None
        assert enhanced_chunk.model_version is None

    def test_invalidation_with_mixed_chunk_types(self, model_integration, chunker):
        """Test invalidation logic with mixed chunk types."""
        # Create a mix of regular and model-aware chunks
        base_chunks = chunker.chunk_text("Mixed chunk types test.")
        mixed_chunks = [
            base_chunks[0],  # Regular ChunkResult
            model_integration.enhance_chunk_with_model_info(base_chunks[0])  # ModelAwareChunkResult
        ]
        
        # Should handle mixed types gracefully
        should_invalidate = model_integration.should_invalidate_chunks(mixed_chunks)
        assert isinstance(should_invalidate, bool) 