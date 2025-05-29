"""
Test suite for API Embedding Service.

This module contains comprehensive TDD tests for the APIEmbeddingService,
including configuration, API integration, error handling, and model change detection.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from requests import HTTPError, ConnectionError, Timeout
from typing import Dict, Any, List

from src.research_agent_backend.core.api_embedding_service import (
    APIEmbeddingService,
    APIConfiguration,
    APIError,
    RateLimitError,
    AuthenticationError,
    BatchProcessingError,
)

from src.research_agent_backend.core.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    BatchProcessingError,
)

from src.research_agent_backend.core.model_change_detection import (
    ModelFingerprint,
    ModelChangeDetector,
)


class TestAPIEmbeddingServiceModelChangeDetection:
    """Test model change detection integration for API embedding service."""
    
    def setup_method(self):
        """Reset singleton state before each test."""
        ModelChangeDetector.reset_singleton()
    
    def teardown_method(self):
        """Clean up after each test."""
        ModelChangeDetector.reset_singleton()
    
    def test_generate_model_fingerprint_openai(self):
        """Test fingerprint generation for OpenAI models."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small",
            embedding_dimension=1536
        )
        
        service = APIEmbeddingService(config)
        fingerprint = service.generate_model_fingerprint()
        
        assert isinstance(fingerprint, ModelFingerprint)
        assert fingerprint.model_name == "text-embedding-3-small"
        assert fingerprint.model_type == "api"
        assert "provider" in fingerprint.metadata
        assert fingerprint.metadata["provider"] == "openai"
        assert fingerprint.checksum is not None
        assert len(fingerprint.checksum) >= 3
    
    def test_generate_model_fingerprint_anthropic(self):
        """Test fingerprint generation for Anthropic models."""
        config = APIConfiguration(
            provider="anthropic",
            api_key="test-key",
            model_name="claude-3-haiku-20240307",
            embedding_dimension=768
        )
        
        service = APIEmbeddingService(config)
        fingerprint = service.generate_model_fingerprint()
        
        assert fingerprint.model_name == "claude-3-haiku-20240307"
        assert fingerprint.metadata["provider"] == "anthropic"
    
    def test_generate_model_fingerprint_custom_provider(self):
        """Test fingerprint generation for custom providers."""
        config = APIConfiguration(
            provider="custom",
            api_key="test-key",
            base_url="https://custom-api.example.com",
            model_name="custom-model",
            embedding_dimension=512
        )
        
        service = APIEmbeddingService(config)
        fingerprint = service.generate_model_fingerprint()
        
        assert fingerprint.model_name == "custom-model"
        assert fingerprint.metadata["provider"] == "custom"
        assert fingerprint.metadata["base_url"] == "https://custom-api.example.com"
    
    def test_check_model_changed_first_time(self):
        """Test model change detection on first check."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # First check should detect change (no previous model)
        changed = service.check_model_changed()
        assert changed is True
    
    def test_check_model_changed_same_model(self):
        """Test no change detected for same model."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # First check
        changed1 = service.check_model_changed()
        assert changed1 is True
        
        # Second check with same model
        changed2 = service.check_model_changed()
        assert changed2 is False
    
    def test_check_model_changed_different_model(self):
        """Test change detected for different model."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # First check
        service.check_model_changed()
        
        # Change model configuration
        service.config.model_name = "text-embedding-3-large"
        
        # Should detect change
        changed = service.check_model_changed()
        assert changed is True
    
    def test_check_model_changed_different_provider(self):
        """Test change detected for different provider."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # First check
        service.check_model_changed()
        
        # Change provider
        service.config.provider = "anthropic"
        
        # Should detect change
        changed = service.check_model_changed()
        assert changed is True
    
    def test_invalidate_cache_on_change_with_change(self):
        """Test cache invalidation when model changes."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Set a cached dimension first
        service._cached_dimension = 1536
        
        # First call should detect change and trigger cache invalidation
        service.invalidate_cache_on_change()
        
        # Cached dimension should have been cleared
        assert service._cached_dimension is None
    
    def test_invalidate_cache_on_change_no_change(self):
        """Test no cache invalidation when model hasn't changed."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Set a cached dimension
        service._cached_dimension = 1536
        
        # Mock check_model_changed to return False (no change detected)
        with patch.object(service, 'check_model_changed', return_value=False):
            # This call should not detect change, no invalidation
            service.invalidate_cache_on_change()
        
        # Cached dimension should be preserved since no change was detected
        assert service._cached_dimension == 1536


class TestAPIEmbeddingServiceErrorHandling:
    """Test comprehensive error handling scenarios."""
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_connection_error(self, mock_post):
        """Test handling of connection errors."""
        mock_post.side_effect = ConnectionError("Connection failed")
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="Connection failed"):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        mock_post.side_effect = Timeout("Request timed out")
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="Request timed out"):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_401_authentication_error(self, mock_post):
        """Test handling of authentication errors."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "Invalid API key"}
        mock_response.raise_for_status.side_effect = HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="invalid-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(AuthenticationError):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_429_rate_limit_error(self, mock_post):
        """Test handling of rate limit errors."""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_response.raise_for_status.side_effect = HTTPError("429 Too Many Requests")
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(RateLimitError):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_500_server_error(self, mock_post):
        """Test handling of server errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_response.raise_for_status.side_effect = HTTPError("500 Internal Server Error")
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(APIError):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="Failed to parse API response"):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_missing_data_field(self, mock_post):
        """Test handling of response missing data field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"usage": {"tokens": 10}}  # Missing 'data' field
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="Invalid API response: missing or empty 'data' field"):
            service.embed_text("test text")
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_text_missing_embedding_field(self, mock_post):
        """Test handling of response missing embedding field."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"index": 0}]  # Missing 'embedding' field
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        with pytest.raises(EmbeddingServiceError, match="API response missing 'embedding' field"):
            service.embed_text("test text")


class TestAPIEmbeddingServiceProviderSpecific:
    """Test provider-specific functionality."""
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_openai_provider_request_format(self, mock_post):
        """Test OpenAI-specific request formatting."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_text("test text")
        
        # Verify OpenAI-specific request format
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        assert request_data['model'] == "text-embedding-3-small"
        assert request_data['input'] == "test text"
        assert request_data['encoding_format'] == "float"
        assert result == [0.1, 0.2, 0.3]
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_anthropic_provider_request_format(self, mock_post):
        """Test Anthropic-specific request formatting."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.4, 0.5, 0.6]}]
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="anthropic",
            api_key="test-key",
            model_name="claude-3-haiku-20240307",
            base_url="https://api.anthropic.com/v1"
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_text("test text")
        
        # Verify Anthropic-specific request format
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        request_data = call_args[1]['json']
        
        assert request_data['model'] == "claude-3-haiku-20240307"
        assert request_data['input'] == "test text"
        assert 'encoding_format' not in request_data  # Anthropic doesn't use this
        assert result == [0.4, 0.5, 0.6]
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_custom_provider_request_format(self, mock_post):
        """Test custom provider request formatting."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.7, 0.8, 0.9]}]
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="custom",
            api_key="test-key",
            model_name="custom-model",
            base_url="https://custom-api.example.com/embeddings"
        )
        
        service = APIEmbeddingService(config)
        result = service.embed_text("test text")
        
        # Verify custom provider uses base URL
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://custom-api.example.com/embeddings"
        assert result == [0.7, 0.8, 0.9]


class TestAPIEmbeddingServiceBatchProcessingEdgeCases:
    """Test edge cases in batch processing."""
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_batch_large_batch_chunking(self, mock_post):
        """Test batch chunking for large batches."""
        # Mock responses for multiple chunks
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2]},
                {"embedding": [0.3, 0.4]},
                {"embedding": [0.5, 0.6]}
            ]
        }
        
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "data": [
                {"embedding": [0.7, 0.8]},
                {"embedding": [0.9, 1.0]}
            ]
        }
        
        mock_post.side_effect = [mock_response1, mock_response2]
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small",
            max_batch_size=3
        )
        
        service = APIEmbeddingService(config)
        
        # Test with 5 texts (should be chunked into 3 + 2)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = service.embed_batch(texts)
        
        assert len(result) == 5
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]
        assert result[2] == [0.5, 0.6]
        assert result[3] == [0.7, 0.8]
        assert result[4] == [0.9, 1.0]
        
        # Verify two API calls were made
        assert mock_post.call_count == 2
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_embed_batch_partial_failure_handling(self, mock_post):
        """Test handling of partial failures in batch processing."""
        # First chunk succeeds, second fails
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
        
        mock_response2 = Mock()
        mock_response2.status_code = 500
        mock_response2.raise_for_status.side_effect = HTTPError("500 Internal Server Error")
        
        # Configure side_effect for both calls plus ensure iteration doesn't fail
        mock_post.side_effect = [mock_response1, mock_response2]
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small",
            max_batch_size=1
        )
        
        service = APIEmbeddingService(config)
        
        # Should fail on second chunk with BatchProcessingError
        with pytest.raises(BatchProcessingError, match="Failed to process chunk"):
            service.embed_batch(["text1", "text2"])


class TestAPIConfigurationEdgeCases:
    """Test edge cases in API configuration."""
    
    def test_api_configuration_with_all_optional_fields(self):
        """Test configuration with all optional fields."""
        config = APIConfiguration(
            provider="custom",
            api_key="test-key",
            model_name="custom-model",
            base_url="https://custom.example.com/v1/embeddings",
            timeout=60,
            max_retries=5,
            max_batch_size=50,
            embedding_dimension=512,
            retry_delay=2.0
        )
        
        assert config.provider == "custom"
        assert config.base_url == "https://custom.example.com/v1/embeddings"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.max_batch_size == 50
        assert config.embedding_dimension == 512
        assert config.retry_delay == 2.0
    
    def test_api_configuration_default_values(self):
        """Test configuration with default values."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_batch_size == 100
        assert config.retry_delay == 1.0
    
    def test_api_configuration_validation_empty_api_key(self):
        """Test validation fails for empty API key."""
        with pytest.raises(AuthenticationError, match="API key is required"):
            APIConfiguration(
                provider="openai",
                api_key="",
                model_name="text-embedding-3-small"
            )
    
    def test_api_configuration_validation_empty_provider(self):
        """Test validation fails for empty provider."""
        with pytest.raises(ValueError, match="Provider must be specified"):
            APIConfiguration(
                provider="",
                api_key="test-key",
                model_name="text-embedding-3-small"
            )


class TestAPIEmbeddingServiceUtilityMethods:
    """Test utility and helper methods."""
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small",
            embedding_dimension=1536
        )
        
        service = APIEmbeddingService(config)
        model_info = service.get_model_info()
        
        assert model_info["model_name"] == "text-embedding-3-small"
        assert model_info["provider"] == "openai"
        assert model_info["model_type"] == "api"
        assert model_info["dimension"] == 1536
    
    def test_get_embedding_dimension_from_config(self):
        """Test embedding dimension retrieval from config."""
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small",
            embedding_dimension=1536
        )
        
        service = APIEmbeddingService(config)
        dimension = service.get_embedding_dimension()
        
        assert dimension == 1536
    
    @patch('src.research_agent_backend.core.api_embedding_service.requests.Session.post')
    def test_is_model_available_true(self, mock_post):
        """Test model availability check returns True for successful connection."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
        mock_post.return_value = mock_response
        
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        available = service.is_model_available()
        
        assert available is True
    
    def test_is_model_available_false(self):
        """Test model availability check returns False for incomplete configuration."""
        # Create a valid config first
        config = APIConfiguration(
            provider="openai",
            api_key="test-key",
            model_name="text-embedding-3-small"
        )
        
        service = APIEmbeddingService(config)
        
        # Now modify the config to make it invalid for availability check
        service.config.api_key = ""  # Clear the API key after creation
        
        # is_model_available should return False for missing API key
        available = service.is_model_available()
        assert available is False 