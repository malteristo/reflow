"""
Test suite for re-ranking service pipeline integration components.

Tests cover:
- Pipeline processor for intercepting search results
- Integration with IntegratedSearchEngine
- Configuration for top_n and re-ranking parameters
- Data flow optimization between retrieval and re-ranking
- Logging and monitoring hooks for observability

This follows TDD Red-Green-Refactor methodology for pipeline integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Dict, Any
import numpy as np

from research_agent_backend.core.reranker.pipeline import (
    RerankerPipelineProcessor,
    PipelineConfig,
    PipelineResult
)
from research_agent_backend.core.reranker import RerankerService, RerankerConfig
from research_agent_backend.core.integration_pipeline.models import SearchResult
from research_agent_backend.core.integration_pipeline import IntegratedSearchEngine
from research_agent_backend.utils.config import ConfigManager


class TestRerankerPipelineProcessor:
    """Test suite for RerankerPipelineProcessor following TDD methodology."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with pipeline settings."""
        config = Mock(spec=ConfigManager)
        config.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': True,
            'pipeline.rerank_top_k': 20,
            'pipeline.rerank_top_n': 5,
            'pipeline.rerank_threshold': 0.1,
            'pipeline.enable_logging': True,
            'pipeline.enable_monitoring': True,
            'reranker.model_name': 'cross-encoder/ms-marco-MiniLM-L6-v2',
            'reranker.batch_size': 32,
            'reranker.device': 'cpu',
            'reranker.enable_caching': True
        }.get(key, default)
        return config
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for pipeline testing."""
        return [
            SearchResult(
                content="Python programming fundamentals and best practices for beginners",
                metadata={"source": "python_guide.md", "type": "tutorial"},
                relevance_score=0.82,
                document_id="doc1",
                chunk_id="chunk1"
            ),
            SearchResult(
                content="Advanced Python data structures including dictionaries and lists",
                metadata={"source": "data_structures.md", "type": "reference"},
                relevance_score=0.78,
                document_id="doc2",
                chunk_id="chunk2"
            ),
            SearchResult(
                content="Machine learning with Python: libraries and frameworks",
                metadata={"source": "ml_python.md", "type": "tutorial"},
                relevance_score=0.75,
                document_id="doc3",
                chunk_id="chunk3"
            ),
            SearchResult(
                content="Web development using Python Flask and Django frameworks",
                metadata={"source": "web_dev.md", "type": "tutorial"},
                relevance_score=0.71,
                document_id="doc4",
                chunk_id="chunk4"
            ),
            SearchResult(
                content="Python testing strategies with pytest and unittest",
                metadata={"source": "testing.md", "type": "reference"},
                relevance_score=0.68,
                document_id="doc5",
                chunk_id="chunk5"
            )
        ]

    # RED PHASE: Pipeline Configuration Tests
    
    def test_pipeline_config_creation_and_validation(self):
        """Test PipelineConfig dataclass creation and validation."""
        config = PipelineConfig(
            enable_reranking=True,
            rerank_top_k=30,
            rerank_top_n=8,
            rerank_threshold=0.2,
            enable_logging=True,
            enable_monitoring=True
        )
        
        assert config.enable_reranking == True
        assert config.rerank_top_k == 30
        assert config.rerank_top_n == 8
        assert config.rerank_threshold == 0.2
        assert config.enable_logging == True
        assert config.enable_monitoring == True
    
    def test_pipeline_config_default_values(self):
        """Test PipelineConfig has appropriate default values."""
        config = PipelineConfig()
        
        assert config.enable_reranking == True
        assert config.rerank_top_k == 20
        assert config.rerank_top_n == 5
        assert config.rerank_threshold == 0.1
        assert config.enable_logging == False
        assert config.enable_monitoring == False

    # RED PHASE: Pipeline Processor Initialization Tests
    
    def test_pipeline_processor_initialization_with_config_manager(self, mock_config_manager):
        """Test RerankerPipelineProcessor can be initialized with ConfigManager."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            
            assert processor.config.enable_reranking == True
            assert processor.config.rerank_top_k == 20
            assert processor.config.rerank_top_n == 5
            assert processor.config.enable_logging == True
            mock_reranker_service.assert_called_once()
    
    def test_pipeline_processor_initialization_with_direct_config(self):
        """Test RerankerPipelineProcessor can be initialized with direct configuration."""
        pipeline_config = PipelineConfig(
            enable_reranking=True,
            rerank_top_k=15,
            rerank_top_n=3,
            enable_monitoring=True
        )
        reranker_config = RerankerConfig(model_name='test-model', batch_size=16)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            processor = RerankerPipelineProcessor(
                pipeline_config=pipeline_config,
                reranker_config=reranker_config
            )
            
            assert processor.config.rerank_top_k == 15
            assert processor.config.rerank_top_n == 3
            assert processor.config.enable_monitoring == True
            mock_reranker_service.assert_called_once_with(config=reranker_config)
    
    def test_pipeline_processor_handles_initialization_errors(self, mock_config_manager):
        """Test RerankerPipelineProcessor handles initialization errors gracefully."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService', side_effect=Exception("Service init failed")):
            with pytest.raises(Exception) as exc_info:
                RerankerPipelineProcessor(config_manager=mock_config_manager)
            
            assert "Service init failed" in str(exc_info.value)

    # RED PHASE: Search Result Interception Tests
    
    def test_intercept_and_rerank_search_results(self, mock_config_manager, sample_search_results):
        """Test pipeline processor can intercept search results and apply re-ranking."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = [
                Mock(rerank_score=0.95, original_score=0.82, rank=1),
                Mock(rerank_score=0.88, original_score=0.78, rank=2),
                Mock(rerank_score=0.82, original_score=0.75, rank=3)
            ]
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            result = processor.process_search_results(
                query="python programming",
                search_results=sample_search_results
            )
            
            # Should call reranker service with appropriate parameters
            mock_service.rerank_results.assert_called_once_with(
                query="python programming",
                candidates=sample_search_results,
                top_n=5,  # From config
                collect_metrics=True
            )
            
            assert isinstance(result, PipelineResult)
            assert len(result.reranked_results) == 3
            assert result.original_count == 5
            assert result.reranked_count == 3
    
    def test_process_with_disabled_reranking(self, mock_config_manager, sample_search_results):
        """Test pipeline processor passes through results when re-ranking is disabled."""
        mock_config_manager.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': False,
            'pipeline.enable_logging': False
        }.get(key, default)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService'):
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results
            )
            
            assert result.reranked_results == sample_search_results
            assert result.original_count == 5
            assert result.reranked_count == 5
            assert result.processing_time_ms >= 0
            assert not result.reranking_applied
    
    def test_top_k_limiting_before_reranking(self, mock_config_manager, sample_search_results):
        """Test pipeline limits candidates to top_k before applying re-ranking."""
        mock_config_manager.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': True,
            'pipeline.rerank_top_k': 3,  # Limit to 3 candidates
            'pipeline.rerank_top_n': 2,
            'pipeline.enable_logging': False,
            'pipeline.enable_monitoring': False,
            'reranker.model_name': 'test-model'
        }.get(key, default)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = [Mock(), Mock()]
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results  # 5 results
            )
            
            # Should only pass first 3 results to reranker
            called_args = mock_service.rerank_results.call_args
            assert len(called_args[1]['candidates']) == 3
    
    def test_threshold_filtering_of_results(self, mock_config_manager, sample_search_results):
        """Test pipeline applies relevance threshold filtering."""
        mock_config_manager.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': True,
            'pipeline.rerank_threshold': 0.8,  # High threshold
            'pipeline.rerank_top_k': 20,
            'pipeline.rerank_top_n': 5,
            'pipeline.enable_logging': False,
            'pipeline.enable_monitoring': False,
            'reranker.model_name': 'test-model'
        }.get(key, default)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            # Return results with scores above and below threshold
            mock_service.rerank_results.return_value = [
                Mock(rerank_score=0.85, rank=1),  # Above threshold
                Mock(rerank_score=0.75, rank=2),  # Below threshold
            ]
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results
            )
            
            # Should filter out results below threshold
            assert len(result.reranked_results) == 1
            assert result.reranked_results[0].rerank_score == 0.85
            assert result.threshold_filtered_count == 1

    # RED PHASE: Integration with IntegratedSearchEngine Tests
    
    def test_enhanced_search_engine_integration(self, mock_config_manager):
        """Test integration with IntegratedSearchEngine for end-to-end pipeline."""
        import asyncio
        from unittest.mock import AsyncMock
        
        with patch('research_agent_backend.core.reranker.pipeline.enhanced_search.IntegratedSearchEngine') as mock_search_engine:
            with patch('research_agent_backend.core.reranker.pipeline.enhanced_search.RerankerPipelineProcessor') as mock_processor:
                
                # Mock search engine results - make it async
                mock_engine = Mock()
                mock_engine.search = AsyncMock(return_value=[
                    SearchResult(content="Result 1", metadata={}, relevance_score=0.8, document_id="1", chunk_id="1"),
                    SearchResult(content="Result 2", metadata={}, relevance_score=0.7, document_id="2", chunk_id="2")
                ])
                mock_search_engine.return_value = mock_engine
                
                # Mock pipeline processor
                mock_proc = Mock()
                mock_proc.process_search_results.return_value = Mock(
                    reranked_results=[Mock(content="Reranked 1"), Mock(content="Reranked 2")],
                    processing_time_ms=50.0
                )
                mock_processor.return_value = mock_proc
                
                # Test enhanced search engine
                from research_agent_backend.core.reranker.pipeline import EnhancedSearchEngine
                enhanced_engine = EnhancedSearchEngine(config_manager=mock_config_manager)
                
                results = enhanced_engine.search_with_reranking(
                    query="test query",
                    top_k=10,
                    rerank_top_n=5
                )
                
                # Should call pipeline processor
                mock_proc.process_search_results.assert_called_once()
                assert len(results) == 2

    # RED PHASE: Data Flow Optimization Tests
    
    def test_efficient_data_passing_between_stages(self, mock_config_manager, sample_search_results):
        """Test efficient data flow between retrieval and re-ranking stages."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = []
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            
            # Monitor data transformations
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results,
                preserve_original_order=True
            )
            
            # Should preserve original data structure efficiently
            assert hasattr(result, 'data_transformation_metrics')
            assert result.data_transformation_metrics['serialization_overhead_ms'] < 10.0
            assert result.data_transformation_metrics['memory_usage_efficient'] == True
    
    def test_zero_copy_data_passing_optimization(self, mock_config_manager, sample_search_results):
        """Test that data passing minimizes serialization/deserialization overhead."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = []
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            
            # Test that SearchResult objects are passed by reference
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results
            )
            
            # Should use minimal serialization
            call_args = mock_reranker_service.return_value.rerank_results.call_args
            passed_candidates = call_args[1]['candidates']
            
            # Should be the same objects, not copies
            assert passed_candidates is sample_search_results or passed_candidates == sample_search_results

    # RED PHASE: Logging and Monitoring Tests
    
    def test_comprehensive_pipeline_logging(self, mock_config_manager, sample_search_results):
        """Test comprehensive logging throughout the pipeline process."""
        mock_config_manager.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': True,
            'pipeline.enable_logging': True,
            'pipeline.enable_monitoring': False,
            'pipeline.rerank_top_n': 3,
            'reranker.model_name': 'test-model'
        }.get(key, default)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = []
            mock_service.get_last_operation_metrics.return_value = {}
            mock_reranker_service.return_value = mock_service
            
            with patch('research_agent_backend.core.reranker.pipeline.processor.logger') as mock_logger:
                processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
                processor.process_search_results(
                    query="test query",
                    search_results=sample_search_results
                )
                
                # Check that logging calls were made
                assert mock_logger.info.call_count >= 2
                assert mock_logger.debug.call_count >= 2
    
    def test_performance_monitoring_metrics_collection(self, mock_config_manager, sample_search_results):
        """Test that performance metrics are collected for monitoring."""
        mock_config_manager.get.side_effect = lambda key, default=None: {
            'pipeline.enable_reranking': True,
            'pipeline.enable_monitoring': True,
            'pipeline.enable_logging': False,
            'pipeline.rerank_top_n': 3
        }.get(key, default)
        
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.return_value = []
            mock_service.get_last_operation_metrics.return_value = {
                'processing_time_ms': 45.2,
                'cache_hit_rate': 0.75,
                'candidates_processed': 5
            }
            mock_reranker_service.return_value = mock_service
            
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            result = processor.process_search_results(
                query="test query",
                search_results=sample_search_results
            )
            
            # Should collect comprehensive monitoring metrics
            assert hasattr(result, 'performance_metrics')
            assert 'total_processing_time_ms' in result.performance_metrics
            assert 'reranking_time_ms' in result.performance_metrics
            assert 'cache_hit_rate' in result.performance_metrics
            assert 'throughput_docs_per_second' in result.performance_metrics
    
    def test_error_handling_and_monitoring_hooks(self, mock_config_manager, sample_search_results):
        """Test proper error handling with monitoring hooks for observability."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService') as mock_reranker_service:
            mock_service = Mock()
            mock_service.rerank_results.side_effect = Exception("Reranking failed")
            mock_reranker_service.return_value = mock_service
            
            with patch('research_agent_backend.core.reranker.pipeline.processor.logger') as mock_logger:
                processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
                result = processor.process_search_results(
                    query="test query",
                    search_results=sample_search_results
                )
                
                # Should handle error gracefully and log it
                mock_logger.error.assert_called()
                assert result.error_occurred == True
                assert "Reranking failed" in result.error_message
                assert result.reranked_results == sample_search_results  # Fallback to original

    # RED PHASE: Configuration Integration Tests
    
    def test_dynamic_configuration_updates(self, mock_config_manager):
        """Test that pipeline configuration can be updated dynamically."""
        with patch('research_agent_backend.core.reranker.pipeline.processor.RerankerService'):
            processor = RerankerPipelineProcessor(config_manager=mock_config_manager)
            
            # Update configuration
            new_config = PipelineConfig(
                enable_reranking=False,
                rerank_top_k=50,
                rerank_top_n=10
            )
            
            processor.update_config(new_config)
            
            assert processor.config.enable_reranking == False
            assert processor.config.rerank_top_k == 50
            assert processor.config.rerank_top_n == 10
    
    def test_configuration_validation_and_defaults(self):
        """Test configuration validation and fallback to defaults."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            PipelineConfig(rerank_top_k=-1)  # Invalid negative value
        
        with pytest.raises(ValueError):
            PipelineConfig(rerank_top_n=0)  # Invalid zero value
        
        with pytest.raises(ValueError):
            PipelineConfig(rerank_threshold=-0.1)  # Invalid negative threshold 