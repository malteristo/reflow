"""
Test suite for embedding service implementations.

This module contains comprehensive TDD tests for the abstract base class
and concrete implementations of embedding services.
"""

import pytest
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# RED PHASE: Import the modules that don't exist yet - these will fail
from src.research_agent_backend.core.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)


class TestEmbeddingServiceAbstractBase:
    """Test the abstract base class for embedding services."""
    
    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that the abstract base class cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            EmbeddingService()
    
    def test_abstract_methods_are_defined(self):
        """Test that all required abstract methods are properly defined."""
        # Check that EmbeddingService has the required abstract methods
        abstract_methods = getattr(EmbeddingService, '__abstractmethods__', set())
        expected_methods = {
            'embed_text',
            'embed_batch', 
            'get_model_info',
            'get_embedding_dimension',
            'is_model_available'
        }
        assert expected_methods.issubset(abstract_methods)
    
    def test_embedding_service_inheritance_structure(self):
        """Test that EmbeddingService properly inherits from ABC."""
        assert issubclass(EmbeddingService, ABC)
        assert hasattr(EmbeddingService, '__abstractmethods__')


class TestEmbeddingServiceInterface:
    """Test the expected interface and behavior patterns for embedding services."""
    
    def test_embed_text_method_signature(self):
        """Test that embed_text has the correct method signature."""
        # This will fail until we implement the abstract method
        assert hasattr(EmbeddingService, 'embed_text')
        
        # Create a concrete test implementation to validate signature
        class TestEmbeddingService(EmbeddingService):
            def embed_text(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {"model_name": "test", "dimension": 3}
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return True
        
        service = TestEmbeddingService()
        result = service.embed_text("test")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    def test_embed_batch_method_signature(self):
        """Test that embed_batch has the correct method signature."""
        assert hasattr(EmbeddingService, 'embed_batch')
        
        class TestEmbeddingService(EmbeddingService):
            def embed_text(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {"model_name": "test", "dimension": 3}
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return True
        
        service = TestEmbeddingService()
        result = service.embed_batch(["text1", "text2"])
        assert isinstance(result, list)
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(isinstance(x, float) for embedding in result for x in embedding)
    
    def test_get_model_info_method_signature(self):
        """Test that get_model_info returns expected dictionary structure."""
        assert hasattr(EmbeddingService, 'get_model_info')
        
        class TestEmbeddingService(EmbeddingService):
            def embed_text(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {
                    "model_name": "test-model",
                    "dimension": 3,
                    "max_seq_length": 512,
                    "model_type": "local"
                }
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return True
        
        service = TestEmbeddingService()
        info = service.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "dimension" in info


class TestEmbeddingServiceExceptions:
    """Test the custom exceptions for embedding services."""
    
    def test_embedding_service_error_inheritance(self):
        """Test that EmbeddingServiceError inherits from Exception."""
        assert issubclass(EmbeddingServiceError, Exception)
    
    def test_model_not_found_error_inheritance(self):
        """Test that ModelNotFoundError inherits from EmbeddingServiceError."""
        assert issubclass(ModelNotFoundError, EmbeddingServiceError)
    
    def test_embedding_dimension_error_inheritance(self):
        """Test that EmbeddingDimensionError inherits from EmbeddingServiceError."""
        assert issubclass(EmbeddingDimensionError, EmbeddingServiceError)
    
    def test_batch_processing_error_inheritance(self):
        """Test that BatchProcessingError inherits from EmbeddingServiceError."""
        assert issubclass(BatchProcessingError, EmbeddingServiceError)
    
    def test_custom_exceptions_can_be_instantiated(self):
        """Test that custom exceptions can be created with messages."""
        error1 = EmbeddingServiceError("Test error")
        assert str(error1) == "Test error"
        
        error2 = ModelNotFoundError("Model not found")
        assert str(error2) == "Model not found"
        
        error3 = EmbeddingDimensionError("Dimension mismatch")
        assert str(error3) == "Dimension mismatch"
        
        error4 = BatchProcessingError("Batch failed")
        assert str(error4) == "Batch failed"


class TestEmbeddingServiceValidation:
    """Test validation methods and error handling patterns."""
    
    def test_embedding_dimension_consistency(self):
        """Test that embedding dimensions are consistent across calls."""
        class TestEmbeddingService(EmbeddingService):
            def embed_text(self, text: str) -> List[float]:
                return [0.1, 0.2, 0.3]  # Always 3 dimensions
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {"model_name": "test", "dimension": 3}
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return True
        
        service = TestEmbeddingService()
        
        # Single embedding dimension should match get_embedding_dimension
        single_embedding = service.embed_text("test")
        assert len(single_embedding) == service.get_embedding_dimension()
        
        # Batch embeddings should all have same dimension
        batch_embeddings = service.embed_batch(["test1", "test2", "test3"])
        expected_dim = service.get_embedding_dimension()
        for embedding in batch_embeddings:
            assert len(embedding) == expected_dim
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        class TestEmbeddingService(EmbeddingService):
            def embed_text(self, text: str) -> List[float]:
                if not text or text.strip() == "":
                    raise EmbeddingServiceError("Cannot embed empty text")
                return [0.1, 0.2, 0.3]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                return [self.embed_text(text) for text in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {"model_name": "test", "dimension": 3}
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return True
        
        service = TestEmbeddingService()
        
        # Empty text should raise error
        with pytest.raises(EmbeddingServiceError, match="Cannot embed empty text"):
            service.embed_text("")
        
        # Empty batch should return empty list
        result = service.embed_batch([])
        assert result == []


class TestEmbeddingServiceModelInfo:
    """Test model information and availability checking."""
    
    def test_model_availability_check(self):
        """Test that model availability can be checked."""
        class TestEmbeddingService(EmbeddingService):
            def __init__(self, model_available: bool = True):
                self._model_available = model_available
            
            def embed_text(self, text: str) -> List[float]:
                if not self.is_model_available():
                    raise ModelNotFoundError("Model not available")
                return [0.1, 0.2, 0.3]
            
            def embed_batch(self, texts: List[str]) -> List[List[float]]:
                return [self.embed_text(text) for text in texts]
            
            def get_model_info(self) -> Dict[str, Any]:
                return {"model_name": "test", "dimension": 3}
            
            def get_embedding_dimension(self) -> int:
                return 3
            
            def is_model_available(self) -> bool:
                return self._model_available
        
        # Test with available model
        available_service = TestEmbeddingService(model_available=True)
        assert available_service.is_model_available() is True
        result = available_service.embed_text("test")
        assert isinstance(result, list)
        
        # Test with unavailable model
        unavailable_service = TestEmbeddingService(model_available=False)
        assert unavailable_service.is_model_available() is False
        with pytest.raises(ModelNotFoundError, match="Model not available"):
            unavailable_service.embed_text("test") 