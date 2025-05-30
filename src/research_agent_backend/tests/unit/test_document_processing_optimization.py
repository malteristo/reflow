"""
Unit tests for document processing pipeline optimization features.

Tests the enhanced document processing pipeline with advanced chunking strategies,
streaming processing, batch optimization, parallel processing, and performance monitoring.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

from src.research_agent_backend.core.document_insertion.manager import DocumentInsertionManager
from src.research_agent_backend.core.document_processor.chunking.chunker import RecursiveChunker
from src.research_agent_backend.core.document_processor.chunking.config import ChunkConfig, BoundaryStrategy
from src.research_agent_backend.models.metadata_schema import DocumentMetadata
from src.research_agent_backend.utils.config import ConfigManager


@pytest.fixture
def large_document_content():
    """Generate large document content for testing streaming processing."""
    # Create a 50KB document with realistic structure
    sections = []
    for i in range(100):
        section = f"""
## Section {i}: Advanced Features

This is section {i} of our comprehensive document. This section covers various aspects
of advanced features including implementation details, best practices, and common pitfalls.

### Key Points

1. **Feature Implementation**: The implementation should follow best practices and ensure
   proper error handling and validation.

2. **Performance Considerations**: When implementing large-scale features, performance
   becomes a critical factor that needs careful consideration.

3. **Scalability**: The system should be designed to handle growing data volumes and
   user loads without degrading performance.

### Code Examples

```python
def advanced_feature_{i}():
    '''Example implementation of advanced feature {i}.'''
    return "This is a sample implementation"
```

### Summary

Section {i} demonstrates the importance of careful design and implementation when
building advanced features. The next section will cover additional aspects.
"""
        sections.append(section)
    
    return "\n".join(sections)


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for testing."""
    config = Mock(spec=ConfigManager)
    
    # Mock the get method that ConfigManager actually uses
    def mock_get(key: str, default=None):
        config_data = {
            "chunking_strategy.chunk_size": 1000,
            "chunking_strategy.chunk_overlap": 200,
            "chunking_strategy.boundary_strategy": "intelligent",
            "chunking_strategy.preserve_code_blocks": True,
            "chunking_strategy.preserve_tables": True,
            "embedding.batch_size": 32,
            "embedding.model_name": "test-model",
            "embedding.dimension": 384,
            "processing.max_workers": 4,
            "processing.batch_size": 100,
            "processing.memory_limit_mb": 512,
            "processing.enable_streaming": True,
            "processing.enable_caching": True
        }
        return config_data.get(key, default)
    
    config.get = Mock(side_effect=mock_get)
    
    # Mock get_config method that data preparation manager uses
    def mock_get_config(section: str):
        configs = {
            "data_preparation": {
                "cleaning": {"min_length": 10},
                "normalization": {"enabled": True},
                "dimensionality": {"reduction_enabled": False}
            }
        }
        return configs.get(section, {})
    
    config.get_config = Mock(side_effect=mock_get_config)
    
    # Mock the config property that returns full config
    config.config = {
        "chunking_strategy": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "boundary_strategy": "intelligent",
            "preserve_code_blocks": True,
            "preserve_tables": True
        },
        "embedding": {
            "batch_size": 32,
            "model_name": "test-model",
            "dimension": 384
        },
        "processing": {
            "max_workers": 4,
            "batch_size": 100,
            "memory_limit_mb": 512,
            "enable_streaming": True,
            "enable_caching": True
        }
    }
    
    return config


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    vector_store = Mock()
    vector_store.add_documents.return_value = None
    vector_store.get_collection_info.return_value = {"count": 0}
    return vector_store


@pytest.fixture  
def mock_data_preparation_manager():
    """Mock data preparation manager for testing."""
    data_prep = Mock()
    
    # Mock prepare method to return text unchanged
    def mock_prepare(text, metadata, collection_type):
        return text, [0.1] * 384, {"processed": True}  # text, embedding, metadata
    
    data_prep.prepare.side_effect = mock_prepare
    return data_prep


@pytest.fixture
def mock_collection_type_manager():
    """Mock collection type manager for testing."""
    collection_mgr = Mock()
    collection_mgr.get_collection_type.return_value = "project_specific"
    return collection_mgr


class TestOptimizedDocumentInsertionManager:
    """Test optimized document insertion manager."""
    
    def test_should_use_advanced_recursive_chunker(self, mock_config_manager, mock_vector_store, mock_data_preparation_manager, mock_collection_type_manager):
        """Test that optimized manager uses RecursiveChunker instead of basic DocumentChunker."""
        # This test should FAIL initially - current implementation uses basic chunker
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True  # New parameter for optimization
        )
        
        # Should use RecursiveChunker for optimized processing
        assert hasattr(manager, 'recursive_chunker')
        assert isinstance(manager.recursive_chunker, RecursiveChunker)
        
        # Should configure chunker with advanced settings
        assert manager.recursive_chunker.config.boundary_strategy == BoundaryStrategy.INTELLIGENT
        assert manager.recursive_chunker.config.preserve_code_blocks == True
        assert manager.recursive_chunker.config.preserve_tables == True
    
    def test_should_support_streaming_processing_for_large_documents(self, mock_config_manager, mock_vector_store, large_document_content, mock_data_preparation_manager, mock_collection_type_manager):
        """Test streaming processing capability for large documents."""
        # This test should FAIL initially - no streaming support exists
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True
        )
        
        metadata = DocumentMetadata(title="Large Document", user_id="test_user")
        
        # Should support streaming mode for large documents
        result = manager.insert_document_streaming(
            text=large_document_content,
            metadata=metadata,
            collection_name="test_collection",
            chunk_size=1000,
            stream_buffer_size=8192  # Process in 8KB chunks
        )
        
        assert result.success == True
        assert result.chunk_count > 0
        assert hasattr(result, 'memory_peak_mb')
        assert result.memory_peak_mb < 100  # Should use less than 100MB for streaming
        assert hasattr(result, 'processing_method')
        assert result.processing_method == "streaming"
    
    def test_should_optimize_batch_embedding_generation(self, mock_config_manager, mock_vector_store, mock_data_preparation_manager, mock_collection_type_manager):
        """Test optimized batch embedding generation."""
        # This test should FAIL initially - no batch optimization exists
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True
        )
        
        # Create multiple documents for batch processing
        documents = []
        for i in range(10):
            documents.append({
                "text": f"Document {i} content with multiple sentences. This is the second sentence.",
                "metadata": DocumentMetadata(title=f"Doc {i}", user_id="test_user"),
                "collection_name": "test_collection"
            })
        
        # Should support optimized batch insertion
        results = manager.insert_documents_batch_optimized(documents)
        
        assert len(results.successful_insertions) == 10
        assert results.batch_processing_time < 5.0  # Should be faster than sequential
        assert hasattr(results, 'embedding_batch_efficiency')
        assert results.embedding_batch_efficiency > 0.7  # Should achieve >70% efficiency
        assert hasattr(results, 'chunks_per_batch')
        assert results.chunks_per_batch >= 5  # Should batch multiple chunks together
    
    def test_should_support_parallel_document_processing(self, mock_config_manager, mock_vector_store, mock_data_preparation_manager, mock_collection_type_manager):
        """Test parallel processing of multiple documents."""
        # This test should FAIL initially - no parallel processing exists
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True,
            max_workers=4
        )
        
        documents = []
        for i in range(20):
            documents.append({
                "text": f"Document {i} with substantial content for parallel processing test.",
                "metadata": DocumentMetadata(title=f"Parallel Doc {i}", user_id="test_user"),
                "collection_name": "test_collection"
            })
        
        # Should support parallel processing
        start_time = time.time()
        results = manager.insert_documents_parallel(documents, max_workers=4)
        processing_time = time.time() - start_time
        
        assert len(results.successful_insertions) == 20
        assert results.parallel_processing == True
        assert results.workers_used == 4
        assert processing_time < 10.0  # Should be faster than sequential
        assert hasattr(results, 'parallelization_efficiency')
        assert results.parallelization_efficiency > 0.5  # Should achieve >50% efficiency
    
    def test_should_implement_intelligent_caching_layer(self, mock_config_manager, mock_vector_store, mock_data_preparation_manager, mock_collection_type_manager):
        """Test intelligent caching for repeated operations."""
        # This test should FAIL initially - no caching layer exists
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True,
            enable_caching=True
        )
        
        metadata = DocumentMetadata(title="Cached Document", user_id="test_user")
        document_text = "This is a document that will be processed multiple times for caching test."
        
        # First insertion - should compute embeddings
        result1 = manager.insert_document(
            text=document_text,
            metadata=metadata,
            collection_name="test_collection"
        )
        
        # Second insertion with same content - should use cache
        result2 = manager.insert_document(
            text=document_text,
            metadata=metadata,
            collection_name="test_collection"
        )
        
        assert result1.success == True
        assert result2.success == True
        assert hasattr(result1, 'cache_hit')
        assert result1.cache_hit == False
        assert result2.cache_hit == True
        assert hasattr(manager, 'cache_stats')
        assert manager.cache_stats['hits'] >= 1
        assert manager.cache_stats['misses'] >= 1
    
    def test_should_collect_performance_metrics(self, mock_config_manager, mock_vector_store, mock_data_preparation_manager, mock_collection_type_manager):
        """Test comprehensive performance metrics collection."""
        # This test should FAIL initially - no metrics collection exists
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_preparation_manager,
            config_manager=mock_config_manager,
            collection_type_manager=mock_collection_type_manager,
            enable_optimization=True,
            enable_metrics=True
        )
        
        documents = []
        for i in range(5):
            documents.append({
                "text": f"Document {i} for metrics testing with various lengths and complexity.",
                "metadata": DocumentMetadata(title=f"Metrics Doc {i}", user_id="test_user"),
                "collection_name": "test_collection"
            })
        
        # Process documents and collect metrics
        results = manager.insert_documents_batch_optimized(documents)
        metrics = manager.get_performance_metrics()
        
        # Should collect comprehensive metrics
        assert 'chunking_performance' in metrics
        assert 'embedding_performance' in metrics
        assert 'storage_performance' in metrics
        assert 'memory_usage' in metrics
        assert 'processing_efficiency' in metrics
        
        # Chunking metrics
        assert metrics['chunking_performance']['chunks_per_second'] > 0
        assert metrics['chunking_performance']['average_chunk_size'] > 0
        assert metrics['chunking_performance']['boundary_detection_time_ms'] >= 0
        
        # Embedding metrics
        assert metrics['embedding_performance']['embeddings_per_second'] > 0
        assert metrics['embedding_performance']['batch_efficiency'] >= 0
        assert metrics['embedding_performance']['cache_hit_ratio'] >= 0
        
        # Memory metrics
        assert metrics['memory_usage']['peak_memory_mb'] >= 0
        assert metrics['memory_usage']['average_memory_mb'] >= 0
        assert metrics['memory_usage']['memory_efficiency'] >= 0


class TestAdvancedChunkingIntegration:
    """Test advanced chunking strategy integration."""
    
    def test_should_integrate_recursive_chunker_with_boundary_strategies(self, mock_config_manager):
        """Test integration of RecursiveChunker with different boundary strategies."""
        # This test should FAIL initially - integration doesn't exist
        from src.research_agent_backend.core.document_insertion.optimization import OptimizedChunkingService
        
        service = OptimizedChunkingService(config_manager=mock_config_manager)
        
        text = """
        # Advanced Features
        
        This document covers advanced features including:
        
        1. Intelligent boundary detection
        2. Content-aware chunking
        3. Performance optimization
        
        ## Code Examples
        
        ```python
        def example_function():
            return "Hello, World!"
        ```
        
        ## Tables
        
        | Feature | Status | Priority |
        |---------|--------|----------|
        | Chunking | Done | High |
        | Caching | Pending | Medium |
        """
        
        metadata = DocumentMetadata(title="Test Document", user_id="test_user")
        
        # Should support different boundary strategies
        for strategy in [BoundaryStrategy.INTELLIGENT, BoundaryStrategy.MARKUP_AWARE, BoundaryStrategy.SENTENCE_ONLY]:
            chunks, chunk_metadata = service.chunk_document_advanced(
                text=text,
                metadata=metadata,
                boundary_strategy=strategy,
                preserve_code_blocks=True,
                preserve_tables=True
            )
            
            assert len(chunks) > 0
            assert len(chunk_metadata) == len(chunks)
            
            # Code blocks should be preserved
            code_chunks = [chunk for chunk in chunks if '```python' in chunk]
            assert len(code_chunks) <= 1  # Code should be in single chunk
            
            # Tables should be preserved
            table_chunks = [chunk for chunk in chunks if '|' in chunk and '---' in chunk]
            assert len(table_chunks) <= 1  # Table should be in single chunk
    
    def test_should_support_semantic_chunking_strategy(self, mock_config_manager):
        """Test semantic chunking strategy implementation."""
        # This test should FAIL initially - semantic chunking not implemented
        from src.research_agent_backend.core.document_insertion.optimization import SemanticChunkingService
        
        service = SemanticChunkingService(config_manager=mock_config_manager)
        
        text = """
        Introduction to Machine Learning
        
        Machine learning is a subset of artificial intelligence that focuses on algorithms
        that can learn from data. The field has grown rapidly in recent years.
        
        Types of Machine Learning
        
        There are three main types of machine learning: supervised learning, unsupervised
        learning, and reinforcement learning. Each type has its own use cases and methods.
        
        Applications in Industry
        
        Machine learning is used in many industries including healthcare, finance, and
        technology. Companies use it for recommendation systems, fraud detection, and more.
        """
        
        metadata = DocumentMetadata(title="ML Document", user_id="test_user")
        
        # Should create semantically coherent chunks
        chunks, chunk_metadata = service.chunk_document_semantic(
            text=text,
            metadata=metadata,
            target_chunk_size=500,
            semantic_similarity_threshold=0.7
        )
        
        assert len(chunks) >= 2  # Should create multiple semantic chunks
        
        # Each chunk should be semantically coherent
        for i, chunk in enumerate(chunks):
            assert len(chunk.strip()) > 100  # Meaningful chunk size
            assert hasattr(chunk_metadata[i], 'semantic_coherence_score')
            assert chunk_metadata[i].semantic_coherence_score >= 0.7


class TestStreamingProcessing:
    """Test streaming processing for large documents."""
    
    def test_should_process_large_files_with_streaming(self, mock_config_manager, mock_vector_store, large_document_content):
        """Test streaming processing for large files without loading entire content into memory."""
        # This test should FAIL initially - no streaming processor exists
        from src.research_agent_backend.core.document_insertion.optimization import StreamingDocumentProcessor
        
        processor = StreamingDocumentProcessor(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            stream_buffer_size=8192,  # 8KB buffer
            max_memory_mb=50  # 50MB memory limit
        )
        
        # Create temporary file with large content
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(large_document_content)
            temp_file = f.name
        
        try:
            metadata = DocumentMetadata(title="Large File", user_id="test_user")
            
            # Should process file with streaming
            result = processor.process_file_streaming(
                file_path=Path(temp_file),
                metadata=metadata,
                collection_name="test_collection"
            )
            
            assert result.success == True
            assert result.chunk_count > 10  # Should create many chunks
            assert hasattr(result, 'peak_memory_mb')
            assert result.peak_memory_mb < 100  # Should stay under memory limit
            assert hasattr(result, 'stream_chunks_processed')
            assert result.stream_chunks_processed > 0
            
        finally:
            os.unlink(temp_file)
    
    def test_should_handle_memory_pressure_gracefully(self, mock_config_manager, mock_vector_store):
        """Test graceful handling of memory pressure during processing."""
        # This test should FAIL initially - no memory pressure handling exists
        from src.research_agent_backend.core.document_insertion.optimization import StreamingDocumentProcessor
        
        processor = StreamingDocumentProcessor(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store,
            max_memory_mb=10  # Very low memory limit
        )
        
        # Generate content that would exceed memory limit if loaded entirely
        large_text = "This is a sentence that will be repeated many times. " * 10000
        metadata = DocumentMetadata(title="Memory Test", user_id="test_user")
        
        # Should handle memory pressure gracefully
        result = processor.process_text_streaming(
            text=large_text,
            metadata=metadata,
            collection_name="test_collection"
        )
        
        assert result.success == True
        assert hasattr(result, 'memory_pressure_events')
        assert hasattr(result, 'gc_collections_triggered')
        assert result.peak_memory_mb <= 20  # Should stay reasonably low


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_should_optimize_embedding_batch_sizes(self, mock_config_manager, mock_vector_store):
        """Test dynamic optimization of embedding batch sizes."""
        # This test should FAIL initially - no batch size optimization exists
        from src.research_agent_backend.core.document_insertion.optimization import EmbeddingBatchOptimizer
        
        optimizer = EmbeddingBatchOptimizer(config_manager=mock_config_manager)
        
        # Test with different chunk sizes and quantities
        test_cases = [
            (10, 500),   # 10 chunks, 500 chars each
            (100, 200),  # 100 chunks, 200 chars each
            (50, 1000),  # 50 chunks, 1000 chars each
        ]
        
        for chunk_count, chunk_size in test_cases:
            chunks = [f"Test chunk {i} " + "content " * (chunk_size // 8) for i in range(chunk_count)]
            
            # Should determine optimal batch size
            optimal_batch_size = optimizer.determine_optimal_batch_size(
                chunks=chunks,
                target_latency_ms=500,
                max_memory_mb=100
            )
            
            assert optimal_batch_size > 0
            assert optimal_batch_size <= len(chunks)
            
            # Should batch process embeddings efficiently
            embeddings, metrics = optimizer.generate_embeddings_optimized(
                chunks=chunks,
                batch_size=optimal_batch_size
            )
            
            assert len(embeddings) == len(chunks)
            assert metrics['batch_efficiency'] > 0.5
            assert metrics['total_processing_time'] > 0
    
    def test_should_implement_adaptive_chunk_sizing(self, mock_config_manager):
        """Test adaptive chunk sizing based on content analysis."""
        # This test should FAIL initially - no adaptive sizing exists
        from src.research_agent_backend.core.document_insertion.optimization import AdaptiveChunkSizer
        
        sizer = AdaptiveChunkSizer(config_manager=mock_config_manager)
        
        # Test different content types
        test_documents = [
            ("Dense technical text with many concepts and terminology.", "technical"),
            ("Simple narrative story with basic vocabulary and structure.", "narrative"),
            ("Code documentation with examples and API references.", "code_docs"),
            ("Mathematical formulas and scientific notation content.", "scientific")
        ]
        
        for content, content_type in test_documents:
            # Should determine optimal chunk size for content type
            optimal_size = sizer.determine_optimal_chunk_size(
                content=content * 100,  # Repeat to make it longer
                content_type=content_type,
                target_overlap_ratio=0.2
            )
            
            assert optimal_size > 200
            assert optimal_size < 2000
            
            # Different content types should get different optimal sizes
            assert hasattr(sizer, 'content_type_profiles')
            assert content_type in sizer.content_type_profiles


class TestCachingLayer:
    """Test intelligent caching layer."""
    
    def test_should_cache_chunk_embeddings(self, mock_config_manager):
        """Test caching of chunk embeddings for repeated content."""
        # This test should FAIL initially - no embedding cache exists
        from src.research_agent_backend.core.document_insertion.optimization import EmbeddingCache
        
        cache = EmbeddingCache(
            config_manager=mock_config_manager,
            max_cache_size_mb=100,
            cache_ttl_hours=24
        )
        
        # Test caching behavior
        chunks = [
            "This is a repeated chunk that should be cached.",
            "Another chunk with different content.",
            "This is a repeated chunk that should be cached.",  # Duplicate
            "Final chunk with unique content."
        ]
        
        # First round - should miss cache
        embeddings1, cache_stats1 = cache.get_embeddings_with_caching(chunks)
        
        assert len(embeddings1) == 4
        assert cache_stats1['cache_hits'] == 0
        assert cache_stats1['cache_misses'] == 3  # Only 3 unique chunks
        
        # Second round - should hit cache for repeated content
        embeddings2, cache_stats2 = cache.get_embeddings_with_caching(chunks)
        
        assert len(embeddings2) == 4
        assert cache_stats2['cache_hits'] >= 2  # Should hit for duplicates
        assert cache_stats2['total_requests'] == 4
    
    def test_should_cache_chunking_results(self, mock_config_manager):
        """Test caching of chunking results for repeated content."""
        # This test should FAIL initially - no chunking cache exists
        from src.research_agent_backend.core.document_insertion.optimization import ChunkingCache
        
        cache = ChunkingCache(config_manager=mock_config_manager)
        
        text = "This is a document that will be chunked multiple times with the same parameters."
        chunking_config = {
            "chunk_size": 1000,
            "overlap": 200,
            "boundary_strategy": "intelligent"
        }
        
        # First chunking - should compute
        chunks1, metadata1, cache_hit1 = cache.get_chunks_with_caching(
            text=text,
            config=chunking_config
        )
        
        assert len(chunks1) > 0
        assert cache_hit1 == False
        
        # Second chunking with same text and config - should hit cache
        chunks2, metadata2, cache_hit2 = cache.get_chunks_with_caching(
            text=text,
            config=chunking_config
        )
        
        assert chunks2 == chunks1
        assert cache_hit2 == True
        assert cache.get_cache_stats()['hits'] >= 1 