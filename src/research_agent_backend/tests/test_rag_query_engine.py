"""
Tests for RAG Query Engine - Query Context Parsing (Red Phase).

This module tests the RAG Query Engine's ability to parse and extract context
from user queries, preparing them for the RAG pipeline.
"""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# These imports will fail initially (Red phase)
from research_agent_backend.core.rag_query_engine import (
    RAGQueryEngine,
    QueryContext,
    QueryIntent,
    ContextualFilter
)


class TestQueryContextParsing:
    """Test suite for query context parsing functionality (Red Phase)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_query_manager = Mock()
        self.mock_embedding_service = Mock()
        self.mock_reranker = Mock()
        
        self.rag_engine = RAGQueryEngine(
            query_manager=self.mock_query_manager,
            embedding_service=self.mock_embedding_service,
            reranker=self.mock_reranker
        )
    
    def test_parse_simple_query(self):
        """Test parsing of simple text queries."""
        query = "What is machine learning?"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert isinstance(context, QueryContext)
        assert context.original_query == query
        assert context.intent == QueryIntent.INFORMATION_SEEKING
        assert context.key_terms == ["machine learning"]
        assert context.filters == []
        assert context.preferences == {}
    
    def test_parse_query_with_collection_filter(self):
        """Test parsing queries with collection specifications."""
        query = "Show me Python tutorials from programming collection"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.original_query == query
        assert context.intent == QueryIntent.INFORMATION_SEEKING
        assert context.key_terms == ["Python", "tutorials"]
        assert len(context.filters) == 1
        assert context.filters[0].field == "collection"
        assert context.filters[0].value == "programming"
        assert context.filters[0].operator == "equals"
    
    def test_parse_query_with_multiple_filters(self):
        """Test parsing queries with multiple metadata filters."""
        query = "Find recent articles about AI from 2024 in research collection"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.key_terms == ["recent", "articles", "AI"]
        assert len(context.filters) >= 2
        
        # Check for collection filter
        collection_filters = [f for f in context.filters if f.field == "collection"]
        assert len(collection_filters) == 1
        assert collection_filters[0].value == "research"
        
        # Check for date filter
        date_filters = [f for f in context.filters if f.field == "date" or f.field == "year"]
        assert len(date_filters) >= 1
    
    def test_parse_comparative_query(self):
        """Test parsing queries that compare concepts."""
        query = "Compare React vs Vue.js performance"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.intent == QueryIntent.COMPARATIVE_ANALYSIS
        assert "React" in context.key_terms
        assert "Vue.js" in context.key_terms
        assert "performance" in context.key_terms
        assert context.preferences.get("comparison_mode") is True
    
    def test_parse_tutorial_request(self):
        """Test parsing tutorial/how-to queries."""
        query = "How to implement authentication in Django?"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.intent == QueryIntent.TUTORIAL_SEEKING
        assert context.key_terms == ["implement", "authentication", "Django"]
        assert context.preferences.get("tutorial_format") is True
    
    def test_parse_code_search_query(self):
        """Test parsing code-specific search queries."""
        query = "Show me Python function examples for data validation"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.intent == QueryIntent.CODE_SEARCH
        assert "Python" in context.key_terms
        assert "function" in context.key_terms
        assert "data validation" in context.key_terms
        assert context.preferences.get("content_type") == "code"
    
    def test_parse_query_with_preferences(self):
        """Test parsing queries with user preferences."""
        query = "Explain machine learning basics in simple terms with examples"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.preferences.get("complexity_level") == "beginner"
        assert context.preferences.get("include_examples") is True
        assert "explain" in context.key_terms
    
    def test_parse_empty_query(self):
        """Test handling of empty or invalid queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.rag_engine.parse_query_context("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.rag_engine.parse_query_context(None)
    
    def test_parse_query_extract_entities(self):
        """Test entity extraction from queries."""
        query = "Find documentation about TensorFlow 2.0 GPU setup"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.entities["technology"] == ["TensorFlow"]
        assert context.entities["version"] == ["2.0"]
        assert context.entities["hardware"] == ["GPU"]
    
    def test_parse_query_with_temporal_context(self):
        """Test parsing queries with time-based context."""
        query = "Latest updates on React hooks from last month"
        
        context = self.rag_engine.parse_query_context(query)
        
        assert context.temporal_context is not None
        assert context.temporal_context["period"] == "last_month"
        assert context.temporal_context["recency_preference"] is True


class TestQueryIntent:
    """Test suite for query intent classification."""
    
    def test_query_intent_enum_values(self):
        """Test that QueryIntent enum has expected values."""
        assert QueryIntent.INFORMATION_SEEKING
        assert QueryIntent.COMPARATIVE_ANALYSIS
        assert QueryIntent.TUTORIAL_SEEKING
        assert QueryIntent.CODE_SEARCH
        assert QueryIntent.TROUBLESHOOTING
    
    def test_intent_classification_information_seeking(self):
        """Test classification of information-seeking queries."""
        queries = [
            "What is machine learning?",
            "Define artificial intelligence",
            "Tell me about Python frameworks"
        ]
        
        for query in queries:
            intent = QueryIntent.classify(query)
            assert intent == QueryIntent.INFORMATION_SEEKING
    
    def test_intent_classification_comparative(self):
        """Test classification of comparative queries."""
        queries = [
            "React vs Vue performance",
            "Compare Python and JavaScript",
            "Difference between SQL and NoSQL"
        ]
        
        for query in queries:
            intent = QueryIntent.classify(query)
            assert intent == QueryIntent.COMPARATIVE_ANALYSIS
    
    def test_intent_classification_tutorial(self):
        """Test classification of tutorial queries."""
        queries = [
            "How to install Docker",
            "Step by step guide for React setup",
            "Tutorial for machine learning"
        ]
        
        for query in queries:
            intent = QueryIntent.classify(query)
            assert intent == QueryIntent.TUTORIAL_SEEKING


class TestContextualFilter:
    """Test suite for contextual filter extraction."""
    
    def test_contextual_filter_creation(self):
        """Test creation of contextual filters."""
        filter_obj = ContextualFilter(
            field="collection",
            value="programming",
            operator="equals",
            confidence=0.9
        )
        
        assert filter_obj.field == "collection"
        assert filter_obj.value == "programming"
        assert filter_obj.operator == "equals"
        assert filter_obj.confidence == 0.9
    
    def test_contextual_filter_from_text(self):
        """Test extraction of filters from text."""
        text = "from programming collection"
        
        filters = ContextualFilter.extract_from_text(text)
        
        assert len(filters) == 1
        assert filters[0].field == "collection"
        assert filters[0].value == "programming"
    
    def test_temporal_filter_extraction(self):
        """Test extraction of temporal filters."""
        text = "from last week"
        
        filters = ContextualFilter.extract_from_text(text)
        
        temporal_filters = [f for f in filters if f.field == "date"]
        assert len(temporal_filters) >= 1
        assert temporal_filters[0].operator == "greater_than"


class TestQueryContext:
    """Test suite for QueryContext data structure."""
    
    def test_query_context_creation(self):
        """Test creation of QueryContext objects."""
        context = QueryContext(
            original_query="test query",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["test"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        assert context.original_query == "test query"
        assert context.intent == QueryIntent.INFORMATION_SEEKING
        assert context.key_terms == ["test"]
    
    def test_query_context_to_dict(self):
        """Test conversion of QueryContext to dictionary."""
        context = QueryContext(
            original_query="test query",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["test"],
            filters=[],
            preferences={"example": True},
            entities={"technology": ["Python"]},
            temporal_context=None
        )
        
        context_dict = context.to_dict()
        
        assert isinstance(context_dict, dict)
        assert context_dict["original_query"] == "test query"
        assert context_dict["preferences"]["example"] is True
    
    def test_query_context_from_dict(self):
        """Test creation of QueryContext from dictionary."""
        context_dict = {
            "original_query": "test query",
            "intent": "INFORMATION_SEEKING",
            "key_terms": ["test"],
            "filters": [],
            "preferences": {},
            "entities": {},
            "temporal_context": None
        }
        
        context = QueryContext.from_dict(context_dict)
        
        assert context.original_query == "test query"
        assert context.intent == QueryIntent.INFORMATION_SEEKING 


class TestRAGQueryEmbeddingGeneration:
    """Test suite for query embedding generation functionality (Red Phase)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_query_manager = Mock()
        self.mock_embedding_service = Mock()
        self.mock_reranker = Mock()
        
        self.rag_engine = RAGQueryEngine(
            query_manager=self.mock_query_manager,
            embedding_service=self.mock_embedding_service,
            reranker=self.mock_reranker
        )
    
    def test_generate_query_embedding_simple(self):
        """Test basic query embedding generation."""
        query_context = QueryContext(
            original_query="What is machine learning?",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["machine learning"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        # Mock embedding service response
        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.mock_embedding_service.generate_embeddings.return_value = expected_embedding
        
        # Generate embedding
        embedding = self.rag_engine.generate_query_embedding(query_context)
        
        # Verify the embedding service was called correctly
        self.mock_embedding_service.generate_embeddings.assert_called_once()
        assert embedding == expected_embedding
        assert len(embedding) == 5
    
    def test_generate_embedding_with_enhanced_context(self):
        """Test embedding generation with enhanced query context."""
        query_context = QueryContext(
            original_query="Compare React vs Vue.js performance",
            intent=QueryIntent.COMPARATIVE_ANALYSIS,
            key_terms=["React", "Vue.js", "performance"],
            filters=[ContextualFilter("collection", "frontend", "equals", 0.9)],
            preferences={"comparison_mode": True},
            entities={"technology": ["React", "Vue.js"]},
            temporal_context={"period": "recent", "recency_preference": True}
        )
        
        expected_embedding = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.mock_embedding_service.generate_embeddings.return_value = expected_embedding
        
        embedding = self.rag_engine.generate_query_embedding(query_context)
        
        # Should enhance the query with context for better embedding
        call_args = self.mock_embedding_service.generate_embeddings.call_args[0]
        enhanced_query = call_args[0]
        
        assert "React" in enhanced_query
        assert "Vue.js" in enhanced_query
        assert "comparison" in enhanced_query.lower()
        assert embedding == expected_embedding
    
    def test_generate_embedding_with_metadata_context(self):
        """Test embedding generation incorporating metadata context."""
        query_context = QueryContext(
            original_query="Show me Python tutorials from programming collection",
            intent=QueryIntent.TUTORIAL_SEEKING,
            key_terms=["Python", "tutorials"],
            filters=[ContextualFilter("collection", "programming", "equals", 0.9)],
            preferences={"tutorial_format": True},
            entities={"technology": ["Python"]},
            temporal_context=None
        )
        
        expected_embedding = [0.3, 0.6, 0.9, 0.2, 0.5]
        self.mock_embedding_service.generate_embeddings.return_value = expected_embedding
        
        embedding = self.rag_engine.generate_query_embedding(query_context)
        
        # Verify context enhancement
        call_args = self.mock_embedding_service.generate_embeddings.call_args[0]
        enhanced_query = call_args[0]
        
        assert "tutorial" in enhanced_query.lower()
        assert "Python" in enhanced_query
        assert embedding == expected_embedding
    
    def test_generate_embedding_error_handling(self):
        """Test error handling during embedding generation."""
        query_context = QueryContext(
            original_query="Test query",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["test"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        # Mock embedding service to raise exception
        self.mock_embedding_service.generate_embeddings.side_effect = Exception("Embedding service error")
        
        with pytest.raises(Exception, match="Failed to generate query embedding"):
            self.rag_engine.generate_query_embedding(query_context)
    
    def test_query_enhancement_intent_based(self):
        """Test query enhancement based on different intents."""
        # Test tutorial intent enhancement
        tutorial_context = QueryContext(
            original_query="How to implement authentication",
            intent=QueryIntent.TUTORIAL_SEEKING,
            key_terms=["implement", "authentication"],
            filters=[],
            preferences={"tutorial_format": True},
            entities={},
            temporal_context=None
        )
        
        self.mock_embedding_service.generate_embeddings.return_value = [0.1, 0.2]
        self.rag_engine.generate_query_embedding(tutorial_context)
        
        call_args = self.mock_embedding_service.generate_embeddings.call_args[0]
        enhanced_query = call_args[0]
        
        assert "tutorial" in enhanced_query.lower() or "how to" in enhanced_query.lower()
    
    def test_query_enhancement_with_entities(self):
        """Test query enhancement with extracted entities."""
        query_context = QueryContext(
            original_query="TensorFlow 2.0 GPU setup",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["setup"],
            filters=[],
            preferences={},
            entities={
                "technology": ["TensorFlow"],
                "version": ["2.0"],
                "hardware": ["GPU"]
            },
            temporal_context=None
        )
        
        self.mock_embedding_service.generate_embeddings.return_value = [0.1, 0.2]
        self.rag_engine.generate_query_embedding(query_context)
        
        call_args = self.mock_embedding_service.generate_embeddings.call_args[0]
        enhanced_query = call_args[0]
        
        assert "TensorFlow" in enhanced_query
        assert "2.0" in enhanced_query
        assert "GPU" in enhanced_query
    
    def test_query_enhancement_with_temporal_context(self):
        """Test query enhancement with temporal context."""
        query_context = QueryContext(
            original_query="Latest React updates",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["React", "updates"],
            filters=[],
            preferences={},
            entities={},
            temporal_context={"period": "recent", "recency_preference": True}
        )
        
        self.mock_embedding_service.generate_embeddings.return_value = [0.1, 0.2]
        self.rag_engine.generate_query_embedding(query_context)
        
        call_args = self.mock_embedding_service.generate_embeddings.call_args[0]
        enhanced_query = call_args[0]
        
        assert "recent" in enhanced_query.lower() or "latest" in enhanced_query.lower()
    
    def test_embedding_caching_behavior(self):
        """Test that similar queries can leverage embedding caching."""
        query_context1 = QueryContext(
            original_query="What is machine learning?",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["machine learning"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        query_context2 = QueryContext(
            original_query="What is machine learning?",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["machine learning"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        expected_embedding = [0.1, 0.2, 0.3]
        self.mock_embedding_service.generate_embeddings.return_value = expected_embedding
        
        # Generate embeddings for the same query twice
        embedding1 = self.rag_engine.generate_query_embedding(query_context1)
        embedding2 = self.rag_engine.generate_query_embedding(query_context2)
        
        # Both should return the same result
        assert embedding1 == embedding2
        assert embedding1 == expected_embedding 


class TestRAGVectorSearchExecution:
    """Test suite for vector search execution functionality (Red Phase)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_query_manager = Mock()
        self.mock_embedding_service = Mock()
        self.mock_reranker = Mock()
        
        self.rag_engine = RAGQueryEngine(
            query_manager=self.mock_query_manager,
            embedding_service=self.mock_embedding_service,
            reranker=self.mock_reranker
        )
    
    def test_execute_vector_search_simple(self):
        """Test basic vector search execution."""
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        collections = ["test_collection"]
        
        # Mock search results
        mock_results = [
            {"id": "doc1", "distance": 0.1, "text": "Test content 1"},
            {"id": "doc2", "distance": 0.3, "text": "Test content 2"}
        ]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=5
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["distance"] == 0.1
        self.mock_query_manager.query.assert_called_once()
    
    def test_execute_vector_search_with_filters(self):
        """Test vector search with metadata filters."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["docs", "papers"]
        filters = {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "author": "John Doe"
        }
        
        mock_results = [{"id": "filtered_doc", "distance": 0.2}]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=10,
            filters=filters
        )
        
        assert len(results) == 1
        # Verify filters were passed to query manager
        call_args = self.mock_query_manager.query.call_args
        assert "filters" in call_args.kwargs or filters in call_args.args
    
    def test_execute_vector_search_multiple_collections(self):
        """Test search across multiple collections."""
        query_embedding = [0.5, 0.6, 0.7]
        collections = ["collection1", "collection2", "collection3"]
        
        # Mock different results from each collection
        mock_results = [
            {"id": "doc1", "distance": 0.1, "collection": "collection1"},
            {"id": "doc2", "distance": 0.2, "collection": "collection2"},
            {"id": "doc3", "distance": 0.15, "collection": "collection3"}
        ]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=20
        )
        
        assert len(results) == 3
        # Verify collections parameter was used
        call_args = self.mock_query_manager.query.call_args
        assert collections in call_args.args or "collections" in call_args.kwargs
    
    def test_execute_vector_search_distance_threshold(self):
        """Test search with distance threshold filtering."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        distance_threshold = 0.5
        
        # Mock results with various distances
        mock_results = [
            {"id": "close_doc", "distance": 0.2},
            {"id": "medium_doc", "distance": 0.4},
            {"id": "far_doc", "distance": 0.8}  # Should be filtered out
        ]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=10,
            distance_threshold=distance_threshold
        )
        
        # Only documents within threshold should be returned
        assert all(result["distance"] <= distance_threshold for result in results)
    
    def test_execute_vector_search_empty_results(self):
        """Test search with no matching results."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["empty_collection"]
        
        self.mock_query_manager.query.return_value = []
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=5
        )
        
        assert results == []
        assert isinstance(results, list)
    
    def test_execute_vector_search_error_handling(self):
        """Test error handling during vector search."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        
        # Mock query manager to raise an exception
        self.mock_query_manager.query.side_effect = Exception("Vector search failed")
        
        with pytest.raises(Exception) as exc_info:
            self.rag_engine.execute_vector_search(
                query_embedding=query_embedding,
                collections=collections,
                top_k=5
            )
        
        assert "Vector search failed" in str(exc_info.value)
    
    def test_execute_vector_search_top_k_limiting(self):
        """Test that top_k parameter limits results correctly."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        top_k = 3
        
        # Mock more results than top_k
        mock_results = [
            {"id": f"doc{i}", "distance": 0.1 * i} for i in range(10)
        ]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=top_k
        )
        
        # Should be limited to top_k results
        assert len(results) <= top_k
    
    def test_execute_vector_search_result_metadata_preservation(self):
        """Test that search preserves important metadata from query manager."""
        query_embedding = [0.1, 0.2, 0.3]
        collections = ["test_collection"]
        
        # Mock results with rich metadata
        mock_results = [
            {
                "id": "doc1",
                "distance": 0.1,
                "metadata": {
                    "title": "Test Document",
                    "author": "Test Author",
                    "timestamp": "2023-01-01"
                },
                "content": "Test content"
            }
        ]
        self.mock_query_manager.query.return_value = mock_results
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=5
        )
        
        assert len(results) == 1
        assert "metadata" in results[0]
        assert results[0]["metadata"]["title"] == "Test Document"
        assert "content" in results[0] 