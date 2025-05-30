"""
Test suite for re-ranking service implementation following TDD approach.

Tests cover:
- Cross-encoder model initialization and configuration
- Single result re-ranking implementation
- Batch processing optimization
- Integration with retrieval pipeline
- Performance evaluation

This follows TDD Red-Green-Refactor methodology.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from research_agent_backend.core.reranker import (
    RerankerService,
    RankedResult,
    RerankerConfig
)
from research_agent_backend.core.integration_pipeline.models import SearchResult
from research_agent_backend.utils.config import ConfigManager


class TestRerankerService:
    """Test suite for RerankerService following TDD methodology."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with reranker settings."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            'reranker.model_name': 'cross-encoder/ms-marco-MiniLM-L6-v2',
            'reranker.batch_size': 32,
            'reranker.max_length': 512,
            'reranker.device': 'cpu',
            'reranker.cache_size': 1000,
            'reranker.enable_caching': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            SearchResult(
                content="Python is a programming language that emphasizes readability and simplicity.",
                metadata={"source": "doc1.md", "type": "technical"},
                relevance_score=0.85,
                document_id="doc1",
                chunk_id="chunk1"
            ),
            SearchResult(
                content="Machine learning is a subset of artificial intelligence that uses algorithms.",
                metadata={"source": "doc2.md", "type": "technical"},
                relevance_score=0.78,
                document_id="doc2", 
                chunk_id="chunk2"
            ),
            SearchResult(
                content="Data structures are ways to organize and store data efficiently.",
                metadata={"source": "doc3.md", "type": "technical"},
                relevance_score=0.72,
                document_id="doc3",
                chunk_id="chunk3"
            )
        ]

    # RED PHASE: Cross-Encoder Model Initialization Tests
    
    def test_reranker_service_initialization_with_default_model(self, mock_config_manager):
        """Test RerankerService can be initialized with default cross-encoder model."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            reranker = RerankerService(config_manager=mock_config_manager)
            
            # Should initialize with default model
            assert reranker.model_name == 'cross-encoder/ms-marco-MiniLM-L6-v2'
            assert reranker.device == 'cpu'
            assert reranker.batch_size == 32
            mock_cross_encoder.assert_called_once_with('cross-encoder/ms-marco-MiniLM-L6-v2')
    
    def test_reranker_service_initialization_with_custom_model(self):
        """Test RerankerService can be initialized with custom model configuration."""
        config = RerankerConfig(
            model_name='mixedbread-ai/mxbai-rerank-xsmall-v1',
            batch_size=16,
            max_length=256,
            device='cpu'
        )
        
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            reranker = RerankerService(config=config)
            
            assert reranker.model_name == 'mixedbread-ai/mxbai-rerank-xsmall-v1'
            assert reranker.batch_size == 16
            assert reranker.max_length == 256
            mock_cross_encoder.assert_called_once_with('mixedbread-ai/mxbai-rerank-xsmall-v1')
    
    def test_reranker_service_handles_model_loading_errors(self, mock_config_manager):
        """Test RerankerService handles model loading errors gracefully."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder', side_effect=Exception("Model not found")):
            with pytest.raises(Exception) as exc_info:
                RerankerService(config_manager=mock_config_manager)
            
            assert "Model not found" in str(exc_info.value)
    
    def test_reranker_config_creation_and_validation(self):
        """Test RerankerConfig dataclass creation and validation."""
        config = RerankerConfig(
            model_name='test-model',
            batch_size=64,
            max_length=1024,
            device='cpu',
            enable_caching=False
        )
        
        assert config.model_name == 'test-model'
        assert config.batch_size == 64
        assert config.max_length == 1024
        assert config.device == 'cpu'
        assert config.enable_caching == False
    
    def test_reranker_config_default_values(self):
        """Test RerankerConfig has appropriate default values."""
        config = RerankerConfig()
        
        assert config.model_name == 'cross-encoder/ms-marco-MiniLM-L6-v2'
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.device == 'cpu'
        assert config.enable_caching == True
        assert config.cache_size == 1000

    # RED PHASE: Single Result Re-ranking Tests
    
    def test_rerank_single_query_document_pair(self, mock_config_manager):
        """Test re-ranking a single query-document pair returns correct score."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.85])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            score = reranker.score_pair("python programming", "Python is a programming language")
            
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            # Score will be normalized via sigmoid, so check it's reasonable
            assert score > 0.5  # Should be reasonably high for good match
            mock_model.predict.assert_called_once()
    
    def test_rerank_single_result_with_metadata_preservation(self, mock_config_manager, sample_search_results):
        """Test re-ranking preserves original result metadata while updating score."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.92])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            original_result = sample_search_results[0]
            ranked_result = reranker.rerank_single(
                query="python programming language",
                search_result=original_result
            )
            
            # Check RankedResult structure
            assert isinstance(ranked_result, RankedResult)
            assert ranked_result.original_result == original_result
            assert isinstance(ranked_result.rerank_score, float)
            assert 0.0 <= ranked_result.rerank_score <= 1.0
            assert ranked_result.original_score == 0.85
            assert ranked_result.rank == 1  # Default rank for single result
    
    def test_score_normalization_within_valid_range(self, mock_config_manager):
        """Test that scores are normalized to valid 0-1 range."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            # Test extreme values
            mock_model.predict.return_value = np.array([-5.2, 8.7, 0.5])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            scores = reranker._normalize_scores(np.array([-5.2, 8.7, 0.5]))
            
            assert all(0.0 <= score <= 1.0 for score in scores)
            assert len(scores) == 3

    # RED PHASE: Batch Processing Tests
    
    def test_rerank_multiple_results_batch_processing(self, mock_config_manager, sample_search_results):
        """Test re-ranking multiple results efficiently with batch processing."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.95, 0.88, 0.76])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            ranked_results = reranker.rerank_results(
                query="python programming concepts",
                candidates=sample_search_results,
                top_n=3
            )
            
            # Check batch processing results
            assert len(ranked_results) == 3
            assert all(isinstance(result, RankedResult) for result in ranked_results)
            
            # Should be sorted by rerank_score (descending)
            scores = [result.rerank_score for result in ranked_results]
            assert scores == sorted(scores, reverse=True)
            
            # Check rank assignment
            for i, result in enumerate(ranked_results):
                assert result.rank == i + 1
    
    def test_batch_size_optimization_for_large_inputs(self, mock_config_manager):
        """Test that large result sets are processed in appropriate batches."""
        # Create large dataset
        large_results = []
        for i in range(100):
            result = SearchResult(
                content=f"Document {i} content about various topics",
                metadata={"source": f"doc{i}.md"},
                relevance_score=0.5 + (i % 50) / 100,
                document_id=f"doc{i}",
                chunk_id=f"chunk{i}"
            )
            large_results.append(result)
        
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            # Mock predict to return appropriate number of scores
            mock_model.predict.return_value = np.random.uniform(0.4, 0.9, size=100)
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            ranked_results = reranker.rerank_results(
                query="test query",
                candidates=large_results,
                top_n=10
            )
            
            assert len(ranked_results) == 10
            assert mock_model.predict.call_count >= 1  # Should be called for batches
    
    def test_empty_candidates_handling(self, mock_config_manager):
        """Test handling of empty candidate list."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            reranker = RerankerService(config_manager=mock_config_manager)
            ranked_results = reranker.rerank_results(
                query="test query",
                candidates=[],
                top_n=5
            )
            
            assert ranked_results == []

    # RED PHASE: Integration and Performance Tests
    
    def test_integration_with_search_pipeline(self, mock_config_manager, sample_search_results):
        """Test integration with existing search pipeline components."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.91, 0.84, 0.77])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            
            # Simulate pipeline integration
            enhanced_results = reranker.enhance_search_results(
                query="python data structures",
                search_results=sample_search_results
            )
            
            assert len(enhanced_results) == 3
            assert all(hasattr(result, 'rerank_score') for result in enhanced_results)
            assert all(hasattr(result, 'original_score') for result in enhanced_results)
    
    def test_performance_metrics_collection(self, mock_config_manager, sample_search_results):
        """Test that performance metrics are collected during re-ranking."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.90, 0.82, 0.75])
            mock_cross_encoder.return_value = mock_model
            
            reranker = RerankerService(config_manager=mock_config_manager)
            
            # Enable metrics collection
            ranked_results = reranker.rerank_results(
                query="test query",
                candidates=sample_search_results,
                top_n=3,
                collect_metrics=True
            )
            
            metrics = reranker.get_last_operation_metrics()
            assert 'processing_time_ms' in metrics
            assert 'batch_count' in metrics
            assert 'candidates_processed' in metrics
            assert metrics['candidates_processed'] == 3
    
    def test_caching_functionality_for_repeated_queries(self, mock_config_manager):
        """Test that caching works for repeated query-document pairs."""
        with patch('research_agent_backend.core.reranker.service.CrossEncoder') as mock_cross_encoder:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([0.88])
            mock_cross_encoder.return_value = mock_model
            
            config = RerankerConfig(enable_caching=True, cache_size=100)
            reranker = RerankerService(config=config)
            
            # First call should invoke model
            score1 = reranker.score_pair("test query", "test document")
            assert mock_model.predict.call_count == 1
            
            # Second call with same inputs should use cache
            score2 = reranker.score_pair("test query", "test document")
            assert mock_model.predict.call_count == 1  # Should not increase
            assert score1 == score2 