"""
Integration tests for the RAG pipeline end-to-end functionality.

Tests the complete RAG query processing pipeline including query context parsing,
embedding generation, vector search, re-ranking, and result formatting.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import from the proper package structure
from research_agent_backend.core.rag_query_engine import (
    RAGQueryEngine,
    QueryResult,
    QueryContext,
    QueryIntent,
    ContextualFilter
)


class TestRAGPipelineIntegration:
    """Test suite for the complete RAG pipeline end-to-end functionality."""
    
    def setup_method(self):
        """Set up test fixtures for pipeline testing."""
        self.mock_query_manager = Mock()
        self.mock_embedding_service = Mock()
        self.mock_reranker = Mock()
        
        self.rag_engine = RAGQueryEngine(
            query_manager=self.mock_query_manager,
            embedding_service=self.mock_embedding_service,
            reranker=self.mock_reranker
        )
        
        # Mock successful pipeline responses
        self._setup_mock_responses()
    
    def _setup_mock_responses(self):
        """Set up standard mock responses for pipeline testing."""
        # Mock embedding service response
        self.mock_embedding_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock vector search results
        self.mock_search_results = [
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "metadata": {"source": "ml_basics.md", "collection": "ai", "author": "test"},
                "distance": 0.15
            },
            {
                "content": "Deep learning uses neural networks with multiple layers",
                "metadata": {"source": "deep_learning.md", "collection": "ai", "author": "test"},
                "distance": 0.25
            }
        ]
        self.mock_query_manager.query.return_value = self.mock_search_results
        
        # Mock reranker response
        self.mock_ranked_results = [
            Mock(score=0.9, document=Mock(content="Machine learning is a subset of artificial intelligence")),
            Mock(score=0.8, document=Mock(content="Deep learning uses neural networks with multiple layers"))
        ]
        self.mock_reranker.rerank.return_value = self.mock_ranked_results

    def test_main_query_method_basic_execution(self):
        """Test the main query() method executes the complete pipeline successfully."""
        query_text = "What is machine learning?"
        collections = ["ai", "technology"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections,
            top_k=10
        )
        
        # Verify result is QueryResult type
        assert isinstance(result, QueryResult)
        
        # Verify result structure
        assert hasattr(result, 'query_context')
        assert hasattr(result, 'results')
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'execution_stats')
        assert hasattr(result, 'feedback')
        
        # Verify query context was parsed
        assert isinstance(result.query_context, QueryContext)
        assert result.query_context.original_query == query_text
        assert result.query_context.intent == QueryIntent.INFORMATION_SEEKING
        
        # Verify all pipeline components were called
        self.mock_embedding_service.embed_text.assert_called_once()
        self.mock_query_manager.query.assert_called_once()
        self.mock_reranker.rerank.assert_called_once()
        
        # Verify execution stats are populated
        assert "execution_time_ms" in result.execution_stats
        assert "total_candidates" in result.execution_stats
        assert "filtered_candidates" in result.execution_stats
        assert "final_results" in result.execution_stats
        assert "reranking_enabled" in result.execution_stats
        assert "collections_searched" in result.execution_stats
        assert result.execution_stats["collections_searched"] == 2
        assert result.execution_stats["reranking_enabled"] is True
        
        # Verify metadata is populated
        assert "query_intent" in result.metadata
        assert "key_terms" in result.metadata
        assert "search_collections" in result.metadata
        assert "processing_pipeline" in result.metadata
        assert result.metadata["search_collections"] == collections
        
        # Verify feedback was generated
        assert result.feedback is not None

    def test_query_method_with_reranking_disabled(self):
        """Test query method with reranking disabled."""
        query_text = "Python tutorials"
        collections = ["programming"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections,
            enable_reranking=False
        )
        
        # Verify reranking was not called
        self.mock_reranker.rerank.assert_not_called()
        assert result.execution_stats["reranking_enabled"] is False
        
        # Should still have results from vector search
        assert isinstance(result, QueryResult)
        assert result.query_context.original_query == query_text

    def test_query_method_with_feedback_disabled(self):
        """Test query method with feedback generation disabled."""
        query_text = "JavaScript frameworks"
        collections = ["web"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections,
            include_feedback=False
        )
        
        # Verify feedback is None
        assert result.feedback is None
        
        # Should still have other components
        assert isinstance(result, QueryResult)
        assert result.query_context is not None

    def test_query_method_input_validation(self):
        """Test query method input validation."""
        # Test empty query
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            self.rag_engine.query(
                query_text="",
                collections=["test"]
            )
        
        # Test None query
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            self.rag_engine.query(
                query_text=None,
                collections=["test"]
            )
        
        # Test empty collections
        with pytest.raises(ValueError, match="Collections list cannot be empty"):
            self.rag_engine.query(
                query_text="test query",
                collections=[]
            )

    def test_query_method_error_handling(self):
        """Test query method error handling and graceful degradation."""
        # Mock embedding service failure
        self.mock_embedding_service.embed_text.side_effect = Exception("Embedding failed")
        
        query_text = "Test query"
        collections = ["test"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # Should return error result instead of raising
        assert isinstance(result, QueryResult)
        assert result.results == []
        assert "error" in result.metadata
        assert "error" in result.execution_stats
        assert result.execution_stats["failed_at"] == "query_execution"
        
        # Should still have basic query context
        assert result.query_context.original_query == query_text

    def test_query_method_with_filters(self):
        """Test query method with metadata filters extraction."""
        query_text = "Python tutorials from programming collection"
        collections = ["programming", "docs"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # Verify filters were extracted
        assert len(result.query_context.filters) > 0
        collection_filter = next(
            (f for f in result.query_context.filters if f.field == "collection"),
            None
        )
        assert collection_filter is not None
        assert collection_filter.value == "programming"

    def test_query_method_execution_phases(self):
        """Test that all execution phases are completed in order."""
        query_text = "machine learning basics"
        collections = ["ai"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # Verify processing pipeline phases
        expected_phases = [
            "context_parsing",
            "embedding_generation", 
            "vector_search",
            "metadata_filtering",
            "reranking",
            "feedback_generation"
        ]
        
        assert "processing_pipeline" in result.metadata
        pipeline = result.metadata["processing_pipeline"]
        
        # Check that all expected phases are present
        for phase in expected_phases:
            assert phase in pipeline

    def test_query_method_different_intents(self):
        """Test query method with different query intents."""
        # Test comparative query
        result = self.rag_engine.query(
            query_text="Compare React vs Vue performance",
            collections=["web"]
        )
        assert result.query_context.intent == QueryIntent.COMPARATIVE_ANALYSIS
        
        # Test tutorial query  
        result = self.rag_engine.query(
            query_text="How to implement authentication in Django",
            collections=["web"]
        )
        assert result.query_context.intent == QueryIntent.TUTORIAL_SEEKING

    def test_query_result_serialization(self):
        """Test that QueryResult can be serialized to/from dict."""
        query_text = "Test serialization"
        collections = ["test"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # Test to_dict()
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "query_context" in result_dict
        assert "results" in result_dict
        assert "metadata" in result_dict
        assert "execution_stats" in result_dict
        
        # Test from_dict()
        recreated_result = QueryResult.from_dict(result_dict)
        assert isinstance(recreated_result, QueryResult)
        assert recreated_result.query_context.original_query == query_text

    def test_prd_compliance_fr_rq_005(self):
        """Test FR-RQ-005 compliance: Complete query processing pipeline."""
        query_text = "test query processing"
        collections = ["test"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # FR-RQ-005: Query processing pipeline
        assert isinstance(result.query_context, QueryContext)
        assert result.query_context.intent is not None
        assert isinstance(result.query_context.key_terms, list)
        assert isinstance(result.execution_stats, dict)
        assert "execution_time_ms" in result.execution_stats

    def test_prd_compliance_fr_rq_006(self):
        """Test FR-RQ-006 compliance: Enhanced query embedding generation."""
        query_text = "enhanced embedding test"
        collections = ["test"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections
        )
        
        # FR-RQ-006: Enhanced embedding generation 
        self.mock_embedding_service.embed_text.assert_called_once()
        
        # Verify the call was made with enhanced query
        call_args = self.mock_embedding_service.embed_text.call_args
        enhanced_query = call_args[0][0]
        assert isinstance(enhanced_query, str)
        assert len(enhanced_query) >= len(query_text)  # Should be enhanced

    def test_prd_compliance_fr_rq_008(self):
        """Test FR-RQ-008 compliance: Cross-encoder re-ranking integration."""
        query_text = "reranking test"
        collections = ["test"]
        
        result = self.rag_engine.query(
            query_text=query_text,
            collections=collections,
            enable_reranking=True
        )
        
        # FR-RQ-008: Cross-encoder re-ranking
        self.mock_reranker.rerank.assert_called_once()
        assert result.execution_stats["reranking_enabled"] is True
        
        # Verify reranking was called with correct parameters
        call_args = self.mock_reranker.rerank.call_args
        assert call_args[0][0] == query_text  # Original query
        assert isinstance(call_args[0][1], list)  # Candidates list 