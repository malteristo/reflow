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
import requests
import json
import os

# RED PHASE: Import the modules that don't exist yet - these will fail
from src.research_agent_backend.core.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)

# RED PHASE: Import API Embedding Service that doesn't exist yet
try:
    from src.research_agent_backend.core.api_embedding_service import (
        APIEmbeddingService,
        APIConfiguration,
        APIError,
        RateLimitError,
        AuthenticationError,
    )
except ImportError:
    # Expected during RED phase - will be created during GREEN phase
    pass


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


# NEW TDD RED PHASE TESTS FOR API EMBEDDING SERVICE
class TestAPIEmbeddingServiceInheritance:
    """Test API Embedding Service inheritance and basic structure."""
    
    def test_api_embedding_service_inheritance(self):
        """Test that APIEmbeddingService properly inherits from EmbeddingService."""
        # This will fail during RED phase until we implement APIEmbeddingService
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService
        assert issubclass(APIEmbeddingService, EmbeddingService)
    
    def test_api_embedding_service_abstract_methods_implemented(self):
        """Test that APIEmbeddingService implements all required abstract methods."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService
        
        # Check that all abstract methods are implemented (not abstract)
        abstract_methods = getattr(APIEmbeddingService, '__abstractmethods__', set())
        assert len(abstract_methods) == 0, f"APIEmbeddingService has unimplemented abstract methods: {abstract_methods}"


class TestAPIConfiguration:
    """Test API configuration and credential management."""
    
    def test_api_configuration_creation(self):
        """Test creation of API configuration objects."""
        from src.research_agent_backend.core.api_embedding_service import APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        assert config.provider == "openai"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.model_name == "text-embedding-3-small"
    
    def test_api_configuration_from_environment(self):
        """Test loading API configuration from environment variables."""
        from src.research_agent_backend.core.api_embedding_service import APIConfiguration
        
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "env-test-key",
            "OPENAI_BASE_URL": "https://custom.openai.com/v1"
        }):
            config = APIConfiguration.from_environment("openai")
            assert config.api_key == "env-test-key"
            assert config.base_url == "https://custom.openai.com/v1"
    
    def test_api_configuration_missing_key_error(self):
        """Test error handling when API key is missing."""
        from src.research_agent_backend.core.api_embedding_service import APIConfiguration, AuthenticationError
        
        with pytest.raises(AuthenticationError, match="API key is required"):
            APIConfiguration(
                provider="openai",
                api_key=None,
                base_url="https://api.openai.com/v1",
                model_name="text-embedding-3-small"
            )


class TestAPIEmbeddingServiceInitialization:
    """Test API Embedding Service initialization and configuration."""
    
    def test_api_embedding_service_initialization_with_config(self):
        """Test initialization with explicit configuration."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        assert service.config == config
        assert service.is_model_available() is True  # Mock will return True initially
    
    def test_api_embedding_service_initialization_from_env(self):
        """Test initialization using environment variables."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            service = APIEmbeddingService.from_environment("openai")
            assert service.config.api_key == "env-key"
            assert service.config.provider == "openai"
    
    @patch('requests.Session')
    def test_api_embedding_service_session_configuration(self, mock_session_class):
        """Test HTTP session configuration with proper headers and timeout."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        mock_session = Mock()
        mock_session_class.return_value = mock_session
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Verify session was configured properly
        mock_session_class.assert_called_once()
        assert hasattr(service, '_session')


class TestAPIEmbeddingServiceTextEmbedding:
    """Test single text embedding functionality via API."""
    
    @patch('requests.Session.post')
    def test_embed_text_successful_request(self, mock_post):
        """Test successful single text embedding request."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}],
            "model": "text-embedding-3-small",
            "usage": {"total_tokens": 5}
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_text("Hello world")
        
        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "embeddings" in call_args[0][0]  # URL contains embeddings endpoint
        
        request_data = call_args[1]["json"]
        assert request_data["input"] == "Hello world"
        assert request_data["model"] == "text-embedding-3-small"
        
        # Verify result
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    @patch('requests.Session.post')
    def test_embed_text_empty_input_error(self, mock_post):
        """Test error handling for empty text input."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="Cannot embed empty text"):
            service.embed_text("")
        
        with pytest.raises(EmbeddingServiceError, match="Cannot embed empty text"):
            service.embed_text("   ")
        
        # Verify no API call was made
        mock_post.assert_not_called()
    
    @patch('requests.Session.post')
    def test_embed_text_api_error_handling(self, mock_post):
        """Test handling of various API errors."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration, APIError
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Test 401 Unauthorized
        mock_response_401 = Mock()
        mock_response_401.status_code = 401
        mock_response_401.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = mock_response_401
        
        with pytest.raises(AuthenticationError, match="Invalid API key"):
            service.embed_text("test")
        
        # Test 429 Rate Limit
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_post.return_value = mock_response_429
        
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            service.embed_text("test")
        
        # Test 500 Server Error
        mock_response_500 = Mock()
        mock_response_500.status_code = 500
        mock_response_500.json.return_value = {"error": {"message": "Internal server error"}}
        mock_post.return_value = mock_response_500
        
        with pytest.raises(APIError, match="Internal server error"):
            service.embed_text("test")
    
    @patch('requests.Session.post')
    def test_embed_text_network_error_handling(self, mock_post):
        """Test handling of network connectivity errors."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Test connection error
        mock_post.side_effect = requests.ConnectionError("Connection failed")
        
        with pytest.raises(EmbeddingServiceError, match="Connection failed"):
            service.embed_text("test")
        
        # Test timeout error
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        with pytest.raises(EmbeddingServiceError, match="Request timed out"):
            service.embed_text("test")


class TestAPIEmbeddingServiceBatchEmbedding:
    """Test batch text embedding functionality via API."""
    
    @patch('requests.Session.post')
    def test_embed_batch_successful_request(self, mock_post):
        """Test successful batch embedding request."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
                {"embedding": [0.7, 0.8, 0.9]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"total_tokens": 15}
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        texts = ["Hello", "world", "test"]
        result = service.embed_batch(texts)
        
        # Verify API call was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]["json"]
        assert request_data["input"] == texts
        assert request_data["model"] == "text-embedding-3-small"
        
        # Verify result
        expected = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        assert result == expected
        assert len(result) == 3
        assert all(isinstance(embedding, list) for embedding in result)
        assert all(isinstance(x, float) for embedding in result for x in embedding)
    
    @patch('requests.Session.post')
    def test_embed_batch_empty_list(self, mock_post):
        """Test handling of empty batch input."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_batch([])
        
        assert result == []
        mock_post.assert_not_called()
    
    @patch('requests.Session.post')
    def test_embed_batch_with_empty_strings(self, mock_post):
        """Test error handling when batch contains empty strings."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(BatchProcessingError, match="Batch contains empty texts"):
            service.embed_batch(["Hello", "", "world"])
        
        mock_post.assert_not_called()
    
    @patch('requests.Session.post')
    def test_embed_batch_chunking_large_batches(self, mock_post):
        """Test automatic chunking of large batches."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Mock responses for chunked requests
        def mock_response_side_effect(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            # Each chunk gets 3 embeddings
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                    {"embedding": [0.7, 0.8, 0.9]}
                ],
                "model": "text-embedding-3-small",
                "usage": {"total_tokens": 15}
            }
            return mock_response
        
        mock_post.side_effect = mock_response_side_effect
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            max_batch_size=3  # Force chunking at 3 items
        )
        
        service = APIEmbeddingService(config)
        texts = [f"text{i}" for i in range(7)]  # 7 texts should create 3 chunks (3, 3, 1)
        result = service.embed_batch(texts)
        
        # Should make multiple API calls due to chunking
        assert mock_post.call_count >= 2  # At least 2 chunks for 7 items with max_batch_size=3
        assert len(result) == 7  # Should return embeddings for all 7 texts


class TestAPIEmbeddingServiceModelInfo:
    """Test model information and metadata retrieval."""
    
    @patch('requests.Session.get')
    def test_get_model_info_successful(self, mock_get):
        """Test successful model information retrieval."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Mock successful model info response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "text-embedding-3-small",
            "object": "model",
            "created": 1677649963,
            "owned_by": "openai"
        }
        mock_get.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        info = service.get_model_info()
        
        # Verify API call was made
        mock_get.assert_called_once()
        
        # Verify response structure
        assert isinstance(info, dict)
        assert info["model_name"] == "text-embedding-3-small"
        assert info["model_type"] == "api"
        assert info["provider"] == "openai"
        assert "dimension" in info  # Should be populated from config or API
    
    def test_get_embedding_dimension_from_config(self):
        """Test getting embedding dimension from configuration."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            embedding_dimension=1536  # Explicit dimension
        )
        
        service = APIEmbeddingService(config)
        dimension = service.get_embedding_dimension()
        
        assert dimension == 1536
        assert isinstance(dimension, int)
    
    @patch('requests.Session.post')
    def test_get_embedding_dimension_from_api(self, mock_post):
        """Test getting embedding dimension from API response."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Mock API response with embedding to determine dimension
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1] * 1536}],  # 1536-dimensional embedding
            "model": "text-embedding-3-small",
            "usage": {"total_tokens": 5}
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
            # No explicit dimension - should be detected from API
        )
        
        service = APIEmbeddingService(config)
        dimension = service.get_embedding_dimension()
        
        assert dimension == 1536
    
    def test_is_model_available_with_valid_config(self):
        """Test model availability check with valid configuration."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        assert service.is_model_available() is True
    
    def test_is_model_available_with_invalid_config(self):
        """Test model availability check with invalid configuration."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # Configuration creation should raise AuthenticationError with None API key
        with pytest.raises(AuthenticationError):
            APIConfiguration(
                provider="openai",
                api_key=None,  # Invalid - no API key
                base_url="https://api.openai.com/v1",
                model_name="text-embedding-3-small"
            )


class TestAPIEmbeddingServiceRetryLogic:
    """Test retry logic and resilience features."""
    
    @patch('requests.Session.post')
    @patch('time.sleep')  # Mock sleep to speed up tests
    def test_retry_on_temporary_failure(self, mock_sleep, mock_post):
        """Test retry logic for temporary API failures."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration
        
        # First call fails with 500, second succeeds
        mock_responses = [
            Mock(status_code=500, json=lambda: {"error": {"message": "Temporary error"}}),
            Mock(status_code=200, json=lambda: {"data": [{"embedding": [0.1, 0.2, 0.3]}]})
        ]
        mock_post.side_effect = mock_responses
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            max_retries=2,
            retry_delay=0.1
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_text("test")
        
        # Should have made 2 API calls (1 failed, 1 successful)
        assert mock_post.call_count == 2
        assert result == [0.1, 0.2, 0.3]
    
    @patch('requests.Session.post')
    @patch('time.sleep')
    def test_retry_exhaustion(self, mock_sleep, mock_post):
        """Test behavior when all retry attempts are exhausted."""
        from src.research_agent_backend.core.api_embedding_service import APIEmbeddingService, APIConfiguration, APIError
        
        # All calls fail with 500
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "Persistent error"}}
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            model_name="text-embedding-3-small",
            max_retries=3,
            retry_delay=0.1
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(APIError, match="Persistent error"):
            service.embed_text("test")
        
        # Should have made max_retries + 1 attempts
        assert mock_post.call_count == 4  # 1 initial + 3 retries


class TestAPIEmbeddingServiceCustomExceptions:
    """Test custom exceptions for API-specific errors."""
    
    def test_api_error_inheritance(self):
        """Test that APIError inherits from EmbeddingServiceError."""
        from src.research_agent_backend.core.api_embedding_service import APIError
        assert issubclass(APIError, EmbeddingServiceError)
    
    def test_rate_limit_error_inheritance(self):
        """Test that RateLimitError inherits from APIError."""
        from src.research_agent_backend.core.api_embedding_service import RateLimitError, APIError
        assert issubclass(RateLimitError, APIError)
    
    def test_authentication_error_inheritance(self):
        """Test that AuthenticationError inherits from APIError."""
        from src.research_agent_backend.core.api_embedding_service import AuthenticationError, APIError
        assert issubclass(AuthenticationError, APIError)
    
    def test_custom_api_exceptions_creation(self):
        """Test that custom API exceptions can be created with context."""
        from src.research_agent_backend.core.api_embedding_service import (
            APIError, RateLimitError, AuthenticationError
        )
        
        api_error = APIError("General API error", status_code=500)
        assert str(api_error) == "General API error"
        assert api_error.status_code == 500
        
        rate_limit_error = RateLimitError("Rate limit exceeded", retry_after=60)
        assert str(rate_limit_error) == "Rate limit exceeded"
        assert rate_limit_error.retry_after == 60
        
        auth_error = AuthenticationError("Invalid API key")
        assert str(auth_error) == "Invalid API key" 