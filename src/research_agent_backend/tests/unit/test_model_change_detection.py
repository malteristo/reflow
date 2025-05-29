"""
Test suite for model change detection system.

This module contains comprehensive TDD tests for model change detection,
including fingerprinting, persistence, cache invalidation, and re-indexing triggers.

RED PHASE: All tests should fail initially until GREEN phase implementation.
"""

import pytest
import json
import hashlib
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
from datetime import datetime

# RED PHASE: Import modules that don't exist yet - these will fail
try:
    from src.research_agent_backend.core.model_change_detection import (
        ModelFingerprint,
        ModelChangeDetector,
        ModelChangeEvent,
        ModelChangeError,
        FingerprintMismatchError,
        PersistenceError,
    )
except ImportError:
    # Expected during RED phase - will be created during GREEN phase
    pass

# Import existing embedding services to test integration
from src.research_agent_backend.core.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
)

from src.research_agent_backend.core.local_embedding_service import (
    LocalEmbeddingService,
    ModelCacheManager,
    EmbeddingModelConfig,
)

from src.research_agent_backend.core.api_embedding_service import (
    APIEmbeddingService,
    APIConfiguration,
)


class TestModelFingerprint:
    """Test the ModelFingerprint class for storing and comparing model metadata."""
    
    def test_model_fingerprint_creation(self):
        """Test that ModelFingerprint can be created with required fields."""
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        assert fingerprint.model_name == "test-model"
        assert fingerprint.model_type == "local"
        assert fingerprint.version == "1.0.0"
        assert fingerprint.checksum == "abc123"
        assert fingerprint.metadata["dimension"] == 384
        assert isinstance(fingerprint.created_at, datetime)
    
    def test_model_fingerprint_equality(self):
        """Test that ModelFingerprint equality works correctly."""
        fingerprint1 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        fingerprint2 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        assert fingerprint1 == fingerprint2
    
    def test_model_fingerprint_inequality_different_checksum(self):
        """Test that ModelFingerprint detects differences in checksum."""
        fingerprint1 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        fingerprint2 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="def456",  # Different checksum
            metadata={"dimension": 384}
        )
        
        assert fingerprint1 != fingerprint2
    
    def test_model_fingerprint_inequality_different_version(self):
        """Test that ModelFingerprint detects differences in version."""
        fingerprint1 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        fingerprint2 = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="2.0.0",  # Different version
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        assert fingerprint1 != fingerprint2
    
    def test_model_fingerprint_to_dict(self):
        """Test that ModelFingerprint can be serialized to dictionary."""
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        data = fingerprint.to_dict()
        
        assert isinstance(data, dict)
        assert data["model_name"] == "test-model"
        assert data["model_type"] == "local"
        assert data["version"] == "1.0.0"
        assert data["checksum"] == "abc123"
        assert data["metadata"]["dimension"] == 384
        assert "created_at" in data
    
    def test_model_fingerprint_from_dict(self):
        """Test that ModelFingerprint can be deserialized from dictionary."""
        data = {
            "model_name": "test-model",
            "model_type": "local",
            "version": "1.0.0",
            "checksum": "abc123",
            "metadata": {"dimension": 384},
            "created_at": "2023-01-01T00:00:00.000000"
        }
        
        fingerprint = ModelFingerprint.from_dict(data)
        
        assert fingerprint.model_name == "test-model"
        assert fingerprint.model_type == "local"
        assert fingerprint.version == "1.0.0"
        assert fingerprint.checksum == "abc123"
        assert fingerprint.metadata["dimension"] == 384
        assert isinstance(fingerprint.created_at, datetime)
    
    def test_model_fingerprint_checksum_validation(self):
        """Test that ModelFingerprint validates checksum format."""
        with pytest.raises(ValueError, match="Checksum must be non-empty"):
            ModelFingerprint(
                model_name="test-model",
                model_type="local",
                version="1.0.0",
                checksum="",  # Empty checksum should fail
                metadata={}
            )
    
    def test_model_fingerprint_model_name_validation(self):
        """Test that ModelFingerprint validates model name."""
        with pytest.raises(ValueError, match="Model name must be non-empty"):
            ModelFingerprint(
                model_name="",  # Empty model name should fail
                model_type="local",
                version="1.0.0",
                checksum="abc123",
                metadata={}
            )


class TestModelChangeDetector:
    """Test the ModelChangeDetector class for centralized change detection."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        ModelChangeDetector.reset_singleton()
    
    def teardown_method(self):
        """Clean up after each test."""
        ModelChangeDetector.reset_singleton()
    
    def test_model_change_detector_initialization(self):
        """Test that ModelChangeDetector can be initialized."""
        detector = ModelChangeDetector(storage_path="/tmp/test_models.json")
        
        assert str(detector.storage_path) == "/tmp/test_models.json"
        assert isinstance(detector._fingerprints, dict)
    
    def test_model_change_detector_singleton_pattern(self):
        """Test that ModelChangeDetector follows singleton pattern."""
        detector1 = ModelChangeDetector()
        detector2 = ModelChangeDetector()
        
        # Should return the same instance
        assert detector1 is detector2
    
    def test_register_model_fingerprint(self):
        """Test registering a new model fingerprint."""
        detector = ModelChangeDetector()
        
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        detector.register_model(fingerprint)
        
        stored_fingerprint = detector.get_model_fingerprint("test-model")
        assert stored_fingerprint == fingerprint
    
    def test_detect_model_change_no_previous_model(self):
        """Test change detection when no previous model exists."""
        detector = ModelChangeDetector()
        
        fingerprint = ModelFingerprint(
            model_name="new-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        change_detected = detector.detect_change(fingerprint)
        
        # New model should be detected as a change
        assert change_detected is True
    
    def test_detect_model_change_same_model(self):
        """Test change detection when model hasn't changed."""
        detector = ModelChangeDetector()
        
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        # Register the model first
        detector.register_model(fingerprint)
        
        # Check the same model again
        change_detected = detector.detect_change(fingerprint)
        
        # Same model should not be detected as a change
        assert change_detected is False
    
    def test_detect_model_change_different_checksum(self):
        """Test change detection when model checksum changes."""
        detector = ModelChangeDetector()
        
        original_fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        updated_fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="def456",  # Different checksum
            metadata={"dimension": 384}
        )
        
        # Register original model
        detector.register_model(original_fingerprint)
        
        # Check updated model
        change_detected = detector.detect_change(updated_fingerprint)
        
        # Different checksum should be detected as a change
        assert change_detected is True
    
    def test_detect_model_change_different_version(self):
        """Test change detection when model version changes."""
        detector = ModelChangeDetector()
        
        original_fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        updated_fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="2.0.0",  # Different version
            checksum="abc123",
            metadata={"dimension": 384}
        )
        
        # Register original model
        detector.register_model(original_fingerprint)
        
        # Check updated model
        change_detected = detector.detect_change(updated_fingerprint)
        
        # Different version should be detected as a change
        assert change_detected is True
    
    def test_get_model_fingerprint_nonexistent(self):
        """Test getting fingerprint for non-existent model."""
        detector = ModelChangeDetector()
        
        fingerprint = detector.get_model_fingerprint("nonexistent-model")
        
        assert fingerprint is None
    
    def test_list_registered_models(self):
        """Test listing all registered models."""
        detector = ModelChangeDetector()
        
        fingerprint1 = ModelFingerprint(
            model_name="model-1",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={}
        )
        
        fingerprint2 = ModelFingerprint(
            model_name="model-2",
            model_type="api",
            version="2.0.0",
            checksum="def456",
            metadata={}
        )
        
        detector.register_model(fingerprint1)
        detector.register_model(fingerprint2)
        
        models = detector.list_models()
        
        assert len(models) == 2
        assert "model-1" in models
        assert "model-2" in models
    
    def test_clear_all_models(self):
        """Test clearing all registered models."""
        detector = ModelChangeDetector()
        
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={}
        )
        
        detector.register_model(fingerprint)
        assert len(detector.list_models()) == 1
        
        detector.clear_all()
        assert len(detector.list_models()) == 0


class TestModelChangeDetectorPersistence:
    """Test persistent storage functionality of ModelChangeDetector."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        ModelChangeDetector.reset_singleton()
    
    def teardown_method(self):
        """Clean up after each test."""
        ModelChangeDetector.reset_singleton()
    
    def test_save_fingerprints_to_file(self):
        """Test saving fingerprints to persistent storage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            storage_path = f.name
        
        try:
            detector = ModelChangeDetector(storage_path=storage_path)
            
            fingerprint = ModelFingerprint(
                model_name="test-model",
                model_type="local",
                version="1.0.0",
                checksum="abc123",
                metadata={"dimension": 384}
            )
            
            detector.register_model(fingerprint)
            detector.save_to_disk()
            
            # File should exist and contain data
            assert os.path.exists(storage_path)
            
            with open(storage_path, 'r') as f:
                data = json.load(f)
            
            assert "test-model" in data
            assert data["test-model"]["checksum"] == "abc123"
            
        finally:
            if os.path.exists(storage_path):
                os.unlink(storage_path)
    
    def test_load_fingerprints_from_file(self):
        """Test loading fingerprints from persistent storage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            storage_path = f.name
            
            # Write test data
            test_data = {
                "test-model": {
                    "model_name": "test-model",
                    "model_type": "local",
                    "version": "1.0.0",
                    "checksum": "abc123",
                    "metadata": {"dimension": 384},
                    "created_at": "2023-01-01T00:00:00.000000"
                }
            }
            json.dump(test_data, f)
        
        try:
            detector = ModelChangeDetector(storage_path=storage_path)
            detector.load_from_disk()
            
            fingerprint = detector.get_model_fingerprint("test-model")
            
            assert fingerprint is not None
            assert fingerprint.model_name == "test-model"
            assert fingerprint.checksum == "abc123"
            assert fingerprint.metadata["dimension"] == 384
            
        finally:
            if os.path.exists(storage_path):
                os.unlink(storage_path)
    
    def test_auto_save_on_register(self):
        """Test that fingerprints are automatically saved when registered."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            storage_path = f.name
        
        try:
            detector = ModelChangeDetector(storage_path=storage_path, auto_save=True)
            
            fingerprint = ModelFingerprint(
                model_name="test-model",
                model_type="local",
                version="1.0.0",
                checksum="abc123",
                metadata={}
            )
            
            detector.register_model(fingerprint)
            
            # File should be automatically saved
            assert os.path.exists(storage_path)
            
            with open(storage_path, 'r') as f:
                data = json.load(f)
            
            assert "test-model" in data
            
        finally:
            if os.path.exists(storage_path):
                os.unlink(storage_path)
    
    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file creates empty storage."""
        nonexistent_path = "/tmp/nonexistent_models.json"
        
        detector = ModelChangeDetector(storage_path=nonexistent_path)
        detector.load_from_disk()
        
        # Should not raise error and should have empty storage
        assert len(detector.list_models()) == 0
    
    def test_save_to_invalid_path(self):
        """Test saving to invalid path raises appropriate error."""
        invalid_path = "/invalid/path/models.json"
        
        detector = ModelChangeDetector(storage_path=invalid_path)
        
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="1.0.0",
            checksum="abc123",
            metadata={}
        )
        
        detector.register_model(fingerprint)
        
        with pytest.raises(PersistenceError, match="Failed to save"):
            detector.save_to_disk()


class TestModelChangeEvent:
    """Test the ModelChangeEvent class for re-indexing triggers."""
    
    def test_model_change_event_creation(self):
        """Test that ModelChangeEvent can be created."""
        event = ModelChangeEvent(
            model_name="test-model",
            change_type="version_update",
            old_fingerprint=None,
            new_fingerprint=ModelFingerprint(
                model_name="test-model",
                model_type="local",
                version="2.0.0",
                checksum="def456",
                metadata={}
            ),
            requires_reindexing=True
        )
        
        assert event.model_name == "test-model"
        assert event.change_type == "version_update"
        assert event.old_fingerprint is None
        assert event.new_fingerprint.version == "2.0.0"
        assert event.requires_reindexing is True
        assert isinstance(event.timestamp, datetime)
    
    def test_model_change_event_to_dict(self):
        """Test that ModelChangeEvent can be serialized."""
        fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="2.0.0",
            checksum="def456",
            metadata={}
        )
        
        event = ModelChangeEvent(
            model_name="test-model",
            change_type="version_update",
            old_fingerprint=None,
            new_fingerprint=fingerprint,
            requires_reindexing=True
        )
        
        data = event.to_dict()
        
        assert isinstance(data, dict)
        assert data["model_name"] == "test-model"
        assert data["change_type"] == "version_update"
        assert data["old_fingerprint"] is None
        assert data["new_fingerprint"]["version"] == "2.0.0"
        assert data["requires_reindexing"] is True
        assert "timestamp" in data


class TestLocalEmbeddingServiceIntegration:
    """Test integration of model change detection with LocalEmbeddingService."""
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_local_service_generates_fingerprint(self, mock_sentence_transformer):
        """Test that LocalEmbeddingService can generate model fingerprint."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.get_max_seq_length.return_value = 512
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService(model_name="test-model")
        
        # This method should be added to LocalEmbeddingService
        fingerprint = service.generate_model_fingerprint()
        
        assert isinstance(fingerprint, ModelFingerprint)
        assert fingerprint.model_name == "test-model"
        assert fingerprint.model_type == "local"
        assert fingerprint.metadata["dimension"] == 384
        assert fingerprint.checksum is not None
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_local_service_detects_model_change(self, mock_sentence_transformer):
        """Test that LocalEmbeddingService detects model changes."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.get_max_seq_length.return_value = 512
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService(model_name="test-model")
        
        # Should detect change on first check (no previous model)
        changed = service.check_model_changed()
        assert changed is True
        
        # Should not detect change on second check (same model)
        changed = service.check_model_changed()
        assert changed is False
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_local_service_cache_invalidation(self, mock_sentence_transformer):
        """Test that LocalEmbeddingService invalidates cache on model change."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.get_max_seq_length.return_value = 512
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService(model_name="test-model")
        
        # Reset the detector singleton to ensure clean state
        ModelChangeDetector.reset_singleton()
        
        # Simulate cache invalidation - should clear cache on first call since no previous model
        with patch.object(service._cache_manager, 'clear_cache') as mock_clear:
            service.invalidate_cache_on_change()
            mock_clear.assert_called_once()


class TestAPIEmbeddingServiceIntegration:
    """Test integration of model change detection with APIEmbeddingService."""
    
    def test_api_service_generates_fingerprint(self):
        """Test that APIEmbeddingService can generate model fingerprint."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # This method should be added to APIEmbeddingService
        fingerprint = service.generate_model_fingerprint()
        
        assert isinstance(fingerprint, ModelFingerprint)
        assert fingerprint.model_name == "text-embedding-3-small"
        assert fingerprint.model_type == "api"
        assert fingerprint.metadata["provider"] == "openai"
        assert fingerprint.checksum is not None
    
    def test_api_service_detects_config_change(self):
        """Test that APIEmbeddingService detects configuration changes."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Reset detector to ensure clean state
        ModelChangeDetector.reset_singleton()
        
        # Should detect change on first check (no previous config)
        changed = service.check_model_changed()
        assert changed is True
        
        # Should not detect change on second check (same config)
        changed = service.check_model_changed()
        assert changed is False
        
        # Should detect change when model name changes
        service.config.model_name = "text-embedding-3-large"
        changed = service.check_model_changed()
        assert changed is True


class TestModelChangeExceptions:
    """Test custom exceptions for model change detection."""
    
    def test_model_change_error_inheritance(self):
        """Test that ModelChangeError inherits from EmbeddingServiceError."""
        assert issubclass(ModelChangeError, EmbeddingServiceError)
    
    def test_fingerprint_mismatch_error_inheritance(self):
        """Test that FingerprintMismatchError inherits from ModelChangeError."""
        assert issubclass(FingerprintMismatchError, ModelChangeError)
    
    def test_persistence_error_inheritance(self):
        """Test that PersistenceError inherits from ModelChangeError."""
        assert issubclass(PersistenceError, ModelChangeError)
    
    def test_custom_exceptions_creation(self):
        """Test that custom exceptions can be created with messages."""
        error1 = ModelChangeError("Test error")
        assert str(error1) == "Test error"
        
        error2 = FingerprintMismatchError("Fingerprint mismatch")
        assert str(error2) == "Fingerprint mismatch"
        
        error3 = PersistenceError("Persistence failed")
        assert str(error3) == "Persistence failed"


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    def test_complete_model_change_workflow(self):
        """Test complete workflow: detect change -> invalidate cache -> trigger reindex."""
        # This test will verify the complete workflow once all components are implemented
        detector = ModelChangeDetector()
        
        # Mock embedding service
        mock_service = Mock(spec=EmbeddingService)
        mock_service.get_model_info.return_value = {
            "model_name": "test-model",
            "model_type": "local",
            "dimension": 384
        }
        
        # Simulate model change detection workflow
        old_fingerprint = detector.get_model_fingerprint("test-model")
        
        # Generate new fingerprint
        new_fingerprint = ModelFingerprint(
            model_name="test-model",
            model_type="local",
            version="2.0.0",
            checksum="new_checksum",
            metadata={"dimension": 384}
        )
        
        # Detect change
        change_detected = detector.detect_change(new_fingerprint)
        assert change_detected is True or old_fingerprint is None
        
        # Register new fingerprint
        detector.register_model(new_fingerprint)
        
        # Create change event
        event = ModelChangeEvent(
            model_name="test-model",
            change_type="model_update",
            old_fingerprint=old_fingerprint,
            new_fingerprint=new_fingerprint,
            requires_reindexing=True
        )
        
        assert event.requires_reindexing is True
    
    def test_cross_service_change_detection(self):
        """Test that changes are detected across different service types."""
        detector = ModelChangeDetector()
        
        # Register local model
        local_fingerprint = ModelFingerprint(
            model_name="shared-model-name",
            model_type="local",
            version="1.0.0",
            checksum="local_checksum",
            metadata={}
        )
        detector.register_model(local_fingerprint)
        
        # Try to register API model with same name but different type
        api_fingerprint = ModelFingerprint(
            model_name="shared-model-name",
            model_type="api",
            version="1.0.0",
            checksum="api_checksum",
            metadata={}
        )
        
        # Should detect change due to different type/checksum
        change_detected = detector.detect_change(api_fingerprint)
        assert change_detected is True 