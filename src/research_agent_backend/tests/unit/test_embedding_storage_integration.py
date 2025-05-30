"""
Unit tests for advanced embedding and storage integration enhancements.

Tests the enhanced integration between embedding services and vector storage
including advanced model management, multi-provider support, hybrid search,
storage backend flexibility, metadata indexing, and intelligent caching.

Follows TDD methodology: RED-GREEN-REFACTOR
"""

import pytest
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional

from src.research_agent_backend.core.embedding_service import EmbeddingService
from src.research_agent_backend.exceptions.vector_store_exceptions import VectorStoreError
from src.research_agent_backend.utils.config import ConfigManager


class TestAdvancedEmbeddingModelManagement:
    """Test advanced embedding model management with fallback mechanisms."""
    
    def test_embedding_service_manager_initialization(self):
        """Test initialization of advanced embedding service manager."""
        # This test should fail - EmbeddingServiceManager doesn't exist yet
        from src.research_agent_backend.core.enhanced_embedding import EmbeddingServiceManager
        
        config = {
            "primary_service": "local",
            "fallback_services": ["api_openai", "api_anthropic"],
            "auto_fallback": True,
            "health_check_interval": 30
        }
        
        manager = EmbeddingServiceManager(config)
        
        assert manager.primary_service is not None
        assert len(manager.fallback_services) == 2
        assert manager.auto_fallback is True
        assert manager.health_check_interval == 30
    
    def test_automatic_fallback_on_primary_failure(self):
        """Test automatic fallback when primary embedding service fails."""
        from src.research_agent_backend.core.enhanced_embedding import EmbeddingServiceManager
        
        config = {
            "primary_service": "local",
            "fallback_services": ["api_openai"],
            "auto_fallback": True
        }
        
        manager = EmbeddingServiceManager(config)
        
        # Mock primary service failure
        manager.primary_service.embed_text = Mock(side_effect=Exception("Model not available"))
        manager.fallback_services[0].embed_text = Mock(return_value=[0.1, 0.2, 0.3])
        
        result = manager.embed_text("test text")
        
        assert result == [0.1, 0.2, 0.3]
        assert manager.current_active_service == manager.fallback_services[0]
        assert manager.primary_service.embed_text.called
        assert manager.fallback_services[0].embed_text.called
    
    def test_health_check_and_recovery(self):
        """Test health checking and recovery of primary service."""
        from src.research_agent_backend.core.enhanced_embedding import EmbeddingServiceManager
        
        manager = EmbeddingServiceManager({
            "primary_service": "local",
            "fallback_services": ["api_openai"],
            "health_check_interval": 1
        })
        
        # Simulate primary service recovery
        manager.primary_service.is_model_available = Mock(return_value=True)
        manager.check_and_recover_primary()
        
        assert manager.current_active_service == manager.primary_service


class TestMultiProviderEmbeddingSupport:
    """Test simultaneous use of multiple embedding services for different document types."""
    
    def test_document_type_based_service_selection(self):
        """Test selection of embedding service based on document type."""
        from src.research_agent_backend.core.enhanced_embedding import MultiProviderEmbeddingCoordinator
        
        config = {
            "document_type_mappings": {
                "code": "local_code_model",
                "research_paper": "api_openai_large",
                "general": "local_general"
            }
        }
        
        coordinator = MultiProviderEmbeddingCoordinator(config)
        
        # Test service selection for different document types
        code_service = coordinator.get_service_for_document_type("code")
        research_service = coordinator.get_service_for_document_type("research_paper")
        general_service = coordinator.get_service_for_document_type("general")
        
        assert code_service.model_name == "local_code_model"
        assert research_service.model_name == "api_openai_large"
        assert general_service.model_name == "local_general"
    
    def test_batch_processing_across_multiple_services(self):
        """Test batch processing that spans multiple embedding services."""
        from src.research_agent_backend.core.enhanced_embedding import MultiProviderEmbeddingCoordinator
        
        coordinator = MultiProviderEmbeddingCoordinator({})
        
        documents = [
            {"text": "def hello():", "type": "code"},
            {"text": "Research findings show...", "type": "research_paper"},
            {"text": "General information", "type": "general"}
        ]
        
        results = coordinator.embed_batch_multi_provider(documents)
        
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
        assert len(coordinator.service_usage_stats) > 0


class TestAdvancedVectorSearchFeatures:
    """Test enhanced vector search capabilities including hybrid search."""
    
    def test_hybrid_search_initialization(self):
        """Test initialization of hybrid search combining dense and sparse vectors."""
        from src.research_agent_backend.core.enhanced_search import HybridSearchEngine
        
        config = {
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "fusion_method": "rrf",  # Reciprocal Rank Fusion
            "max_results": 100
        }
        
        search_engine = HybridSearchEngine(config)
        
        assert search_engine.dense_weight == 0.7
        assert search_engine.sparse_weight == 0.3
        assert search_engine.fusion_method == "rrf"
        assert search_engine.max_results == 100
    
    def test_hybrid_search_execution(self):
        """Test execution of hybrid search with result fusion."""
        from src.research_agent_backend.core.enhanced_search import HybridSearchEngine
        
        search_engine = HybridSearchEngine({
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "fusion_method": "weighted_score"
        })
        
        query = "machine learning algorithms"
        collection_name = "test_collection"
        
        results = search_engine.hybrid_search(
            query=query,
            collection_name=collection_name,
            top_k=10
        )
        
        assert len(results) <= 10
        assert all("dense_score" in result for result in results)
        assert all("sparse_score" in result for result in results)
        assert all("fused_score" in result for result in results)
    
    def test_custom_similarity_metrics(self):
        """Test custom similarity metrics for specialized domains."""
        from src.research_agent_backend.core.enhanced_search import CustomSimilarityEngine
        
        engine = CustomSimilarityEngine()
        
        # Test semantic similarity for code
        code_similarity = engine.calculate_semantic_similarity(
            "def sort_list(arr):",
            "function sortArray(list):",
            domain="code"
        )
        
        # Test similarity for academic papers
        paper_similarity = engine.calculate_semantic_similarity(
            "neural network architecture",
            "deep learning model structure",
            domain="academic"
        )
        
        assert 0 <= code_similarity <= 1
        assert 0 <= paper_similarity <= 1
        assert paper_similarity > 0.5  # Should be high semantic similarity


class TestStorageBackendFlexibility:
    """Test multi-vector store support and migration tools."""
    
    def test_multi_backend_storage_manager_initialization(self):
        """Test initialization of multi-backend storage manager."""
        from src.research_agent_backend.core.enhanced_storage import MultiBackendStorageManager
        
        config = {
            "primary_backend": "chromadb",
            "secondary_backends": ["sqlite_vec"],
            "replication_strategy": "async",
            "consistency_level": "eventual"
        }
        
        manager = MultiBackendStorageManager(config)
        
        assert manager.primary_backend.backend_type == "chromadb"
        assert len(manager.secondary_backends) == 1
        assert manager.replication_strategy == "async"
        assert manager.consistency_level == "eventual"
    
    def test_storage_migration_between_backends(self):
        """Test migration of data between different storage backends."""
        from src.research_agent_backend.core.enhanced_storage import StorageMigrationTool
        
        migration_tool = StorageMigrationTool()
        
        # Mock source and destination managers
        source_config = {"type": "chromadb", "path": "/tmp/source"}
        dest_config = {"type": "sqlite_vec", "path": "/tmp/dest"}
        
        migration_result = migration_tool.migrate_collection(
            source_config=source_config,
            dest_config=dest_config,
            collection_name="test_collection",
            batch_size=1000
        )
        
        assert migration_result.success is True
        assert migration_result.documents_migrated > 0
        assert migration_result.migration_time_seconds > 0
        assert migration_result.validation_passed is True
    
    def test_automatic_backend_failover(self):
        """Test automatic failover to secondary backend on primary failure."""
        from src.research_agent_backend.core.enhanced_storage import MultiBackendStorageManager
        
        manager = MultiBackendStorageManager({
            "primary_backend": "chromadb",
            "secondary_backends": ["sqlite_vec"],
            "auto_failover": True
        })
        
        # Mock primary backend failure
        manager.primary_backend.add_documents = Mock(side_effect=VectorStoreError("Connection failed"))
        manager.secondary_backends[0].add_documents = Mock(return_value={"success": True})
        
        result = manager.add_documents(
            collection_name="test",
            chunks=["test content"],
            embeddings=[[0.1, 0.2, 0.3]]
        )
        
        assert result["success"] is True
        assert manager.current_active_backend == manager.secondary_backends[0]


class TestEnhancedMetadataIndexing:
    """Test advanced metadata indexing and semantic enrichment."""
    
    def test_semantic_metadata_enrichment(self):
        """Test automatic semantic enrichment of document metadata."""
        from src.research_agent_backend.core.enhanced_metadata import SemanticMetadataEnricher
        
        enricher = SemanticMetadataEnricher()
        
        document_content = "This paper presents a novel approach to neural network optimization using gradient descent."
        base_metadata = {
            "source": "paper.pdf",
            "author": "Smith et al.",
            "created_at": "2024-01-01"
        }
        
        enriched_metadata = enricher.enrich_metadata(document_content, base_metadata)
        
        assert "semantic_tags" in enriched_metadata
        assert "content_type" in enriched_metadata
        assert "domain" in enriched_metadata
        assert "complexity_score" in enriched_metadata
        assert enriched_metadata["domain"] in ["academic", "technical", "research"]
    
    def test_advanced_metadata_indexing(self):
        """Test advanced indexing of metadata for fast filtering."""
        from src.research_agent_backend.core.enhanced_metadata import AdvancedMetadataIndex
        
        index = AdvancedMetadataIndex()
        
        # Add documents with rich metadata
        documents = [
            {
                "id": "doc1",
                "content": "AI research paper",
                "metadata": {
                    "domain": "AI",
                    "complexity": 8.5,
                    "tags": ["machine learning", "neural networks"]
                }
            },
            {
                "id": "doc2", 
                "content": "Simple tutorial",
                "metadata": {
                    "domain": "education",
                    "complexity": 3.0,
                    "tags": ["tutorial", "beginner"]
                }
            }
        ]
        
        for doc in documents:
            index.add_document(doc["id"], doc["metadata"])
        
        # Test complex filtering
        results = index.filter_documents({
            "domain": "AI",
            "complexity": {"$gte": 8.0},
            "tags": {"$in": ["neural networks"]}
        })
        
        assert len(results) == 1
        assert results[0] == "doc1"
    
    def test_metadata_semantic_search(self):
        """Test semantic search within metadata fields."""
        from src.research_agent_backend.core.enhanced_metadata import MetadataSemanticSearch
        
        search = MetadataSemanticSearch()
        
        metadata_corpus = [
            {"id": "doc1", "title": "Deep learning fundamentals", "abstract": "Introduction to neural networks"},
            {"id": "doc2", "title": "Machine learning basics", "abstract": "Overview of ML algorithms"},
            {"id": "doc3", "title": "Natural language processing", "abstract": "Text analysis techniques"}
        ]
        
        for doc in metadata_corpus:
            search.index_metadata(doc["id"], doc)
        
        results = search.search_metadata("artificial intelligence neural networks", top_k=2)
        
        assert len(results) <= 2
        assert results[0]["id"] == "doc1"  # Should rank highest for neural network query


class TestIntelligentCachingLayer:
    """Test multi-level caching and cache warming strategies."""
    
    def test_multi_level_cache_initialization(self):
        """Test initialization of multi-level caching system."""
        from src.research_agent_backend.core.enhanced_caching import MultiLevelCacheManager
        
        config = {
            "l1_cache": {"type": "memory", "size": 1000, "ttl": 300},
            "l2_cache": {"type": "disk", "size": 10000, "ttl": 3600},
            "l3_cache": {"type": "distributed", "size": 100000, "ttl": 86400}
        }
        
        cache_manager = MultiLevelCacheManager(config)
        
        assert cache_manager.l1_cache.cache_type == "memory"
        assert cache_manager.l2_cache.cache_type == "disk"
        assert cache_manager.l3_cache.cache_type == "distributed"
        assert cache_manager.l1_cache.max_size == 1000
    
    def test_intelligent_cache_warming(self):
        """Test intelligent cache warming based on usage patterns."""
        from src.research_agent_backend.core.enhanced_caching import IntelligentCacheWarmer
        
        warmer = IntelligentCacheWarmer()
        
        # Historical usage data
        usage_patterns = [
            {"query": "machine learning", "frequency": 50, "last_used": "2024-01-15"},
            {"query": "neural networks", "frequency": 30, "last_used": "2024-01-14"},
            {"query": "deep learning", "frequency": 40, "last_used": "2024-01-16"}
        ]
        
        warmer.analyze_usage_patterns(usage_patterns)
        warming_plan = warmer.generate_warming_plan(target_cache_size=100)
        
        assert len(warming_plan) > 0
        assert warming_plan[0]["query"] == "machine learning"  # Highest frequency
        assert all("priority" in item for item in warming_plan)
    
    def test_cache_performance_optimization(self):
        """Test cache performance optimization and hit rate improvement."""
        from src.research_agent_backend.core.enhanced_caching import CachePerformanceOptimizer
        
        optimizer = CachePerformanceOptimizer()
        
        cache_stats = {
            "hit_rate": 0.65,
            "miss_rate": 0.35,
            "eviction_rate": 0.1,
            "memory_usage": 0.8
        }
        
        optimization_recommendations = optimizer.analyze_performance(cache_stats)
        
        assert "recommendations" in optimization_recommendations
        assert "projected_hit_rate" in optimization_recommendations
        assert optimization_recommendations["projected_hit_rate"] > cache_stats["hit_rate"]
    
    def test_embedding_cache_with_model_fingerprinting(self):
        """Test embedding cache that invalidates on model changes."""
        from src.research_agent_backend.core.enhanced_caching import ModelAwareCacheManager
        
        cache = ModelAwareCacheManager()
        
        # Cache embeddings with model fingerprint
        model_fingerprint = "model_v1_abc123"
        text = "test text for caching"
        embedding = [0.1, 0.2, 0.3]
        
        cache.cache_embedding(text, embedding, model_fingerprint)
        
        # Retrieve with same model fingerprint
        cached_result = cache.get_cached_embedding(text, model_fingerprint)
        assert cached_result == embedding
        
        # Retrieve with different model fingerprint (should miss)
        cached_result = cache.get_cached_embedding(text, "model_v2_def456")
        assert cached_result is None


class TestIntegrationOptimization:
    """Test integration-level optimizations and performance improvements."""
    
    def test_pipeline_coordination_optimization(self):
        """Test optimization of coordination between embedding and storage pipelines."""
        from src.research_agent_backend.core.enhanced_integration import OptimizedPipelineCoordinator
        
        coordinator = OptimizedPipelineCoordinator({
            "parallel_processing": True,
            "batch_optimization": True,
            "adaptive_batching": True
        })
        
        documents = [
            {"content": "Document 1", "metadata": {"type": "text"}},
            {"content": "Document 2", "metadata": {"type": "code"}},
            {"content": "Document 3", "metadata": {"type": "text"}}
        ]
        
        results = coordinator.process_documents_optimized(documents, collection_name="test")
        
        assert results.success is True
        assert results.processing_time_seconds > 0
        assert results.optimization_metrics["parallel_processing_used"] is True
        assert results.optimization_metrics["batch_efficiency"] > 0.8
    
    def test_resource_usage_monitoring(self):
        """Test monitoring and optimization of resource usage."""
        from src.research_agent_backend.core.enhanced_integration import ResourceMonitor
        
        monitor = ResourceMonitor()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Simulate some processing
        time.sleep(0.1)
        
        # Get metrics
        metrics = monitor.get_current_metrics()
        
        assert "memory_usage_mb" in metrics
        assert "cpu_usage_percent" in metrics
        assert "cache_hit_rate" in metrics
        assert "active_connections" in metrics
        
        monitor.stop_monitoring()


# Integration test combining multiple enhancements
class TestIntegratedEnhancements:
    """Test integration of all enhancements working together."""
    
    def test_end_to_end_enhanced_workflow(self):
        """Test complete enhanced workflow from document ingestion to search."""
        from src.research_agent_backend.core.enhanced_integration import EnhancedWorkflowManager
        
        workflow = EnhancedWorkflowManager({
            "embedding_strategy": "multi_provider",
            "storage_strategy": "multi_backend",
            "caching_strategy": "intelligent",
            "search_strategy": "hybrid"
        })
        
        # Document ingestion with enhancements
        documents = [
            {"content": "AI research paper content", "type": "research"},
            {"content": "Code implementation details", "type": "code"}
        ]
        
        ingestion_result = workflow.ingest_documents_enhanced(documents, collection_name="enhanced_test")
        
        assert ingestion_result.success is True
        assert ingestion_result.enhancements_applied["multi_provider_embedding"] is True
        assert ingestion_result.enhancements_applied["intelligent_caching"] is True
        
        # Enhanced search
        search_result = workflow.search_enhanced(
            query="machine learning algorithms",
            collection_name="enhanced_test",
            search_type="hybrid"
        )
        
        assert len(search_result.results) > 0
        assert search_result.enhancements_applied["hybrid_search"] is True
        assert search_result.performance_metrics["cache_hit_rate"] >= 0 