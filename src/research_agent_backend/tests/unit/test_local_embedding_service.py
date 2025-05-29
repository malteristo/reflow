"""
Test suite for LocalEmbeddingService implementation.

This module contains comprehensive TDD tests for the local embedding service
using sentence-transformers models.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
from typing import List, Dict, Any
import tempfile
import os

# RED PHASE: Import the modules that don't exist yet - these will fail
from src.research_agent_backend.core.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
    EmbeddingDimensionError,
    BatchProcessingError,
)

from src.research_agent_backend.core.local_embedding_service import (
    LocalEmbeddingService,
    ModelCacheManager,
    EmbeddingModelConfig,
)


def clear_model_cache():
    """Helper function to clear the model cache between tests."""
    try:
        ModelCacheManager().clear_cache()
    except:
        pass


class TestLocalEmbeddingServiceInitialization:
    """Test initialization and configuration of LocalEmbeddingService."""
    
    def test_local_embedding_service_inheritance(self):
        """Test that LocalEmbeddingService properly inherits from EmbeddingService."""
        assert issubclass(LocalEmbeddingService, EmbeddingService)
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_default_model_initialization(self, mock_sentence_transformer):
        """Test initialization with default model (multi-qa-MiniLM-L6-cos-v1)."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        
        # Verify default model is loaded with cache_folder parameter
        mock_sentence_transformer.assert_called_once_with('multi-qa-MiniLM-L6-cos-v1', cache_folder=None)
        assert service.is_model_available() is True
        assert service.get_embedding_dimension() == 384
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_custom_model_initialization(self, mock_sentence_transformer):
        """Test initialization with custom model."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService(model_name='BAAI/bge-base-en-v1.5')
        
        mock_sentence_transformer.assert_called_once_with('BAAI/bge-base-en-v1.5', cache_folder=None)
        assert service.get_embedding_dimension() == 768
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_model_initialization_failure(self, mock_sentence_transformer):
        """Test handling of model initialization failures."""
        clear_model_cache()
        mock_sentence_transformer.side_effect = Exception("Model not found")
        
        with pytest.raises(ModelNotFoundError, match="Failed to load model"):
            LocalEmbeddingService(model_name='nonexistent-model')
    
    def test_model_cache_directory_configuration(self):
        """Test that model cache directory can be configured."""
        clear_model_cache()
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_st.return_value = mock_model
                service = LocalEmbeddingService(cache_dir=temp_dir)
                assert service._cache_dir == temp_dir


class TestLocalEmbeddingServiceEmbedding:
    """Test embedding generation functionality."""
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_text_single_input(self, mock_sentence_transformer):
        """Test embedding generation for single text input."""
        clear_model_cache()
        mock_model = Mock()
        # Configure mock to return actual numpy array
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        result = service.embed_text("Hello world")
        
        mock_model.encode.assert_called_once_with("Hello world", convert_to_tensor=False)
        assert result == [0.1, 0.2, 0.3, 0.4]
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_text_empty_input(self, mock_sentence_transformer):
        """Test handling of empty text input."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        
        with pytest.raises(EmbeddingServiceError, match="Cannot embed empty text"):
            service.embed_text("")
        
        with pytest.raises(EmbeddingServiceError, match="Cannot embed empty text"):
            service.embed_text("   ")
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_text_model_error(self, mock_sentence_transformer):
        """Test handling of model encoding errors."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        
        with pytest.raises(EmbeddingServiceError, match="Failed to generate embedding"):
            service.embed_text("test")
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_batch_multiple_inputs(self, mock_sentence_transformer):
        """Test batch embedding generation."""
        clear_model_cache()
        mock_model = Mock()
        # Configure mock to return actual numpy array
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        texts = ["Hello", "world", "test"]
        result = service.embed_batch(texts)
        
        mock_model.encode.assert_called_once_with(texts, convert_to_tensor=False)
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_batch_empty_list(self, mock_sentence_transformer):
        """Test batch embedding with empty list."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        result = service.embed_batch([])
        
        assert result == []
        mock_model.encode.assert_not_called()
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_embed_batch_with_empty_strings(self, mock_sentence_transformer):
        """Test batch embedding with some empty strings."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 3
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        texts = ["valid text", "", "   ", "another valid"]
        
        with pytest.raises(BatchProcessingError, match="Batch contains empty texts"):
            service.embed_batch(texts)


class TestLocalEmbeddingServiceModelInfo:
    """Test model information and metadata functionality."""
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_get_model_info_default_model(self, mock_sentence_transformer):
        """Test getting model information for default model."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.get_max_seq_length.return_value = 512
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        info = service.get_model_info()
        
        expected_info = {
            "model_name": "multi-qa-MiniLM-L6-cos-v1",
            "dimension": 384,
            "max_seq_length": 512,
            "model_type": "local",
            "library": "sentence-transformers"
        }
        assert info == expected_info
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_get_model_info_custom_model(self, mock_sentence_transformer):
        """Test getting model information for custom model."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.get_max_seq_length.return_value = 256
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService(model_name='custom-model')
        info = service.get_model_info()
        
        assert info["model_name"] == "custom-model"
        assert info["dimension"] == 768
        assert info["max_seq_length"] == 256
        assert info["model_type"] == "local"
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension."""
        clear_model_cache()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        assert service.get_embedding_dimension() == 384
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_is_model_available_true(self, mock_sentence_transformer):
        """Test model availability check when model is loaded."""
        clear_model_cache()
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        service = LocalEmbeddingService()
        assert service.is_model_available() is True
    
    def test_is_model_available_false(self):
        """Test model availability check when model is not loaded."""
        service = LocalEmbeddingService.__new__(LocalEmbeddingService)
        service._model = None
        assert service.is_model_available() is False


class TestLocalEmbeddingServiceCaching:
    """Test model caching functionality."""
    
    def test_model_caching_enabled(self):
        """Test that model caching is enabled by default."""
        clear_model_cache()
        
        with patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            # Create two services with same model - should use cache
            service1 = LocalEmbeddingService()
            service2 = LocalEmbeddingService()
            
            # Model should only be loaded once due to caching
            assert mock_st.call_count == 1
    
    def test_model_cache_different_models(self):
        """Test that different models are cached separately."""
        clear_model_cache()
        
        with patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model
            
            service1 = LocalEmbeddingService(model_name='model1')
            service2 = LocalEmbeddingService(model_name='model2')
            
            # Different models should be loaded separately
            assert mock_st.call_count == 2
            calls = mock_st.call_args_list
            assert calls[0][0][0] == 'model1'
            assert calls[1][0][0] == 'model2'


class TestModelCacheManager:
    """Test the ModelCacheManager utility class."""
    
    def test_model_cache_manager_singleton(self):
        """Test that ModelCacheManager is a singleton."""
        cache1 = ModelCacheManager()
        cache2 = ModelCacheManager()
        assert cache1 is cache2
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_get_or_load_model_new_model(self, mock_sentence_transformer):
        """Test loading a new model through cache manager."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        cache = ModelCacheManager()
        cache.clear_cache()  # Ensure clean state
        model = cache.get_or_load_model('test-model')
        
        assert model is mock_model
        mock_sentence_transformer.assert_called_once_with('test-model', cache_folder=None)
    
    @patch('src.research_agent_backend.core.local_embedding_service.SentenceTransformer')
    def test_get_or_load_model_cached_model(self, mock_sentence_transformer):
        """Test retrieving a cached model."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        cache = ModelCacheManager()
        cache.clear_cache()  # Ensure clean state
        model1 = cache.get_or_load_model('test-model')
        model2 = cache.get_or_load_model('test-model')
        
        assert model1 is model2
        # Model should only be loaded once
        mock_sentence_transformer.assert_called_once()
    
    def test_clear_cache(self):
        """Test clearing the model cache."""
        cache = ModelCacheManager()
        cache._cache['test'] = Mock()
        
        cache.clear_cache()
        assert len(cache._cache) == 0


class TestEmbeddingModelConfig:
    """Test the EmbeddingModelConfig data class."""
    
    def test_embedding_model_config_default_values(self):
        """Test default configuration values."""
        config = EmbeddingModelConfig()
        
        assert config.model_name == "multi-qa-MiniLM-L6-cos-v1"
        assert config.cache_dir is None
        assert config.device == "auto"
        assert config.normalize_embeddings is True
    
    def test_embedding_model_config_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingModelConfig(
            model_name="custom-model",
            cache_dir="/custom/cache",
            device="cpu",
            normalize_embeddings=False
        )
        
        assert config.model_name == "custom-model"
        assert config.cache_dir == "/custom/cache"
        assert config.device == "cpu"
        assert config.normalize_embeddings is False
    
    def test_embedding_model_config_validation(self):
        """Test configuration validation."""
        # Test invalid device
        with pytest.raises(ValueError, match="Device must be"):
            EmbeddingModelConfig(device="invalid")
        
        # Test empty model name
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            EmbeddingModelConfig(model_name="") 