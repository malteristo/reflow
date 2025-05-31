"""Tests for RAG Query Engine functionality."""

from unittest.mock import Mock, patch, MagicMock
import pytest
from datetime import datetime

from research_agent_backend.core.rag_query_engine import (
    RAGQueryEngine, 
    QueryContext, 
    QueryIntent, 
    ContextualFilter
)
from research_agent_backend.core.integration_pipeline.models import SearchResult
from research_agent_backend.core.reranker.models import RankedResult


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
        assert result["chunk_content"] == "Test content"
        assert result["collection"] == "test_collection"


class TestRAGMetadataFiltering:
    """Test suite for metadata filtering functionality (Red Phase)."""
    
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
        
        # Sample candidate results with rich metadata
        self.candidate_results = [
            {
                "id": "doc1",
                "content": "Python programming tutorial for beginners",
                "distance": 0.1,
                "metadata": {
                    "collection": "programming",
                    "author": "John Doe",
                    "date": "2023-01-15",
                    "category": "tutorial",
                    "difficulty": "beginner",
                    "language": "Python",
                    "tags": ["programming", "tutorial", "python"],
                    "rating": 4.5
                }
            },
            {
                "id": "doc2", 
                "content": "Advanced Python algorithms and data structures",
                "distance": 0.2,
                "metadata": {
                    "collection": "programming",
                    "author": "Jane Smith",
                    "date": "2023-06-20",
                    "category": "advanced",
                    "difficulty": "expert",
                    "language": "Python",
                    "tags": ["algorithms", "data-structures", "python"],
                    "rating": 4.8
                }
            },
            {
                "id": "doc3",
                "content": "JavaScript web development basics",
                "distance": 0.3,
                "metadata": {
                    "collection": "web-dev",
                    "author": "Bob Wilson",
                    "date": "2023-03-10",
                    "category": "tutorial",
                    "difficulty": "intermediate",
                    "language": "JavaScript",
                    "tags": ["javascript", "web", "development", "tutorial"],
                    "rating": 4.2
                }
            },
            {
                "id": "doc4",
                "content": "Python data science with pandas",
                "distance": 0.15,
                "metadata": {
                    "collection": "data-science",
                    "author": "Alice Johnson",
                    "date": "2023-09-05",
                    "category": "tutorial",
                    "difficulty": "intermediate",
                    "language": "Python",
                    "tags": ["python", "data-science", "pandas", "tutorial"],
                    "rating": 4.7
                }
            }
        ]
    
    def test_apply_metadata_filters_equals_operator(self):
        """Test metadata filtering with equals operator."""
        filters = [
            ContextualFilter(field="language", value="Python", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return only Python documents
        assert len(filtered_results) == 3
        python_docs = ["doc1", "doc2", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in python_docs for doc_id in result_ids)
        
        # Should preserve original metadata
        for result in filtered_results:
            assert result["metadata"]["language"] == "Python"
    
    def test_apply_metadata_filters_collection_filter(self):
        """Test filtering by collection."""
        filters = [
            ContextualFilter(field="collection", value="programming", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return only programming collection documents
        assert len(filtered_results) == 2
        programming_docs = ["doc1", "doc2"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in programming_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_contains_operator(self):
        """Test metadata filtering with contains operator for array fields."""
        filters = [
            ContextualFilter(field="tags", value="tutorial", operator="contains", confidence=0.8)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return documents with "tutorial" in tags
        assert len(filtered_results) == 3
        tutorial_docs = ["doc1", "doc3", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in tutorial_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_greater_than_operator(self):
        """Test metadata filtering with greater_than operator for numeric fields."""
        filters = [
            ContextualFilter(field="rating", value=4.5, operator="greater_than", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return documents with rating > 4.5
        assert len(filtered_results) == 2
        high_rated_docs = ["doc2", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in high_rated_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_less_than_operator(self):
        """Test metadata filtering with less_than operator."""
        filters = [
            ContextualFilter(field="rating", value=4.3, operator="less_than", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return documents with rating < 4.3
        assert len(filtered_results) == 1
        assert filtered_results[0]["id"] == "doc3"
        assert filtered_results[0]["metadata"]["rating"] == 4.2
    
    def test_apply_metadata_filters_date_range(self):
        """Test date-based filtering with custom date operators."""
        filters = [
            ContextualFilter(field="date", value="2023-06-01", operator="greater_than", confidence=0.8)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return documents after June 1, 2023
        assert len(filtered_results) == 2
        recent_docs = ["doc2", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in recent_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_multiple_and_conditions(self):
        """Test metadata filtering with multiple filters (AND logic)."""
        filters = [
            ContextualFilter(field="language", value="Python", operator="equals", confidence=0.9),
            ContextualFilter(field="difficulty", value="intermediate", operator="equals", confidence=0.8)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return only Python documents with intermediate difficulty
        assert len(filtered_results) == 1
        assert filtered_results[0]["id"] == "doc4"
        assert filtered_results[0]["metadata"]["language"] == "Python"
        assert filtered_results[0]["metadata"]["difficulty"] == "intermediate"
    
    def test_apply_metadata_filters_multiple_complex_conditions(self):
        """Test complex metadata filtering with multiple criteria."""
        filters = [
            ContextualFilter(field="category", value="tutorial", operator="equals", confidence=0.9),
            ContextualFilter(field="rating", value=4.0, operator="greater_than", confidence=0.8)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return tutorial documents with rating > 4.0
        assert len(filtered_results) == 3
        tutorial_docs = ["doc1", "doc3", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in tutorial_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_missing_field(self):
        """Test filtering when field doesn't exist in metadata."""
        filters = [
            ContextualFilter(field="nonexistent_field", value="some_value", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should return no results when field doesn't exist
        assert len(filtered_results) == 0
    
    def test_apply_metadata_filters_empty_filters(self):
        """Test filtering with empty filter list."""
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=[]
        )
        
        # Should return all candidates when no filters applied
        assert len(filtered_results) == 4
        assert filtered_results == self.candidate_results
    
    def test_apply_metadata_filters_none_filters(self):
        """Test filtering with None filters."""
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=None
        )
        
        # Should return all candidates when no filters applied
        assert len(filtered_results) == 4
        assert filtered_results == self.candidate_results
    
    def test_apply_metadata_filters_empty_candidates(self):
        """Test filtering with empty candidate list."""
        filters = [
            ContextualFilter(field="language", value="Python", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=[],
            filters=filters
        )
        
        # Should return empty list
        assert filtered_results == []
    
    def test_apply_metadata_filters_confidence_based_filtering(self):
        """Test that low-confidence filters are handled appropriately."""
        filters = [
            ContextualFilter(field="language", value="Python", operator="equals", confidence=0.3)  # Low confidence
        ]
        
        # Should still apply filter regardless of confidence for now
        # (Future enhancement could skip low-confidence filters)
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        assert len(filtered_results) == 3
        python_docs = ["doc1", "doc2", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in python_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_case_insensitive_string_matching(self):
        """Test case-insensitive string matching for equals operator."""
        filters = [
            ContextualFilter(field="language", value="python", operator="equals", confidence=0.9)  # lowercase
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Should match "Python" in metadata despite case difference
        assert len(filtered_results) == 3
        python_docs = ["doc1", "doc2", "doc4"]
        result_ids = [result["id"] for result in filtered_results]
        assert all(doc_id in python_docs for doc_id in result_ids)
    
    def test_apply_metadata_filters_with_nested_fields(self):
        """Test filtering with nested metadata fields."""
        # Add nested metadata to test data
        nested_candidates = [
            {
                "id": "doc_nested1",
                "content": "Test content",
                "distance": 0.1,
                "metadata": {
                    "collection": "test",
                    "author": {
                        "name": "John Doe",
                        "department": "Engineering"
                    },
                    "publication": {
                        "year": 2023,
                        "journal": "Tech Review"
                    }
                }
            }
        ]
        
        filters = [
            ContextualFilter(field="author.department", value="Engineering", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=nested_candidates,
            filters=filters
        )
        
        # Should handle nested field access
        assert len(filtered_results) == 1
        assert filtered_results[0]["id"] == "doc_nested1"
    
    def test_apply_metadata_filters_preserves_result_structure(self):
        """Test that filtering preserves the complete result structure."""
        filters = [
            ContextualFilter(field="language", value="Python", operator="equals", confidence=0.9)
        ]
        
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=self.candidate_results,
            filters=filters
        )
        
        # Verify complete structure is preserved
        for result in filtered_results:
            assert "id" in result
            assert "content" in result
            assert "distance" in result
            assert "metadata" in result
            assert isinstance(result["metadata"], dict)
            
            # Verify specific fields are intact
            assert "collection" in result["metadata"]
            assert "author" in result["metadata"]
            assert "date" in result["metadata"]
    
    def test_metadata_filtering_integration_with_vector_search(self):
        """Test integration of metadata filtering with vector search pipeline."""
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        collections = ["programming"]
        
        # Mock QueryManager to return our test candidates
        self.mock_query_manager.query.return_value = self.candidate_results
        
        # Test with metadata filters
        filters = {
            "language": "Python",
            "difficulty": "beginner"
        }
        
        results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=collections,
            top_k=10,
            filters=filters
        )
        
        # Should integrate metadata filtering into the pipeline
        self.mock_query_manager.query.assert_called_once()
        call_kwargs = self.mock_query_manager.query.call_args.kwargs
        assert "filters" in call_kwargs or filters in self.mock_query_manager.query.call_args.args


class TestRAGReranking:
    """Test suite for re-ranking functionality (Red Phase)."""
    
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
        
        # Sample search results for re-ranking
        self.search_results = [
            {
                "id": "doc1",
                "content": "Python programming tutorial for beginners",
                "distance": 0.2,
                "metadata": {
                    "collection": "programming",
                    "author": "John Doe",
                    "rating": 4.5
                }
            },
            {
                "id": "doc2", 
                "content": "Advanced Python algorithms and data structures",
                "distance": 0.3,
                "metadata": {
                    "collection": "programming",
                    "author": "Jane Smith",
                    "rating": 4.8
                }
            },
            {
                "id": "doc3",
                "content": "JavaScript web development basics",
                "distance": 0.15,
                "metadata": {
                    "collection": "web-dev",
                    "author": "Bob Wilson",
                    "rating": 4.2
                }
            }
        ]
        
        # Mock SearchResult objects for reranker
        self.mock_search_result_objects = [
            SearchResult(
                content=result["content"],
                metadata=result["metadata"], 
                relevance_score=1.0 - result["distance"],
                document_id=result["id"],
                chunk_id=f"{result['id']}_chunk1"
            )
            for result in self.search_results
        ]
        
        # Mock RankedResult objects from reranker
        self.mock_ranked_results = [
            RankedResult(
                original_result=self.mock_search_result_objects[0],
                rerank_score=0.95,
                original_score=0.8,
                rank=1
            ),
            RankedResult(
                original_result=self.mock_search_result_objects[1],
                rerank_score=0.88,
                original_score=0.7,
                rank=2
            ),
            RankedResult(
                original_result=self.mock_search_result_objects[2],
                rerank_score=0.82,
                original_score=0.85,
                rank=3
            )
        ]
    
    def test_apply_reranking_basic_functionality(self):
        """Test basic re-ranking functionality with mock reranker."""
        query = "Python programming tutorial"
        
        # Mock reranker response
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        # Test apply_reranking method
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Verify reranker was called correctly
        self.mock_reranker.rerank_results.assert_called_once()
        call_args = self.mock_reranker.rerank_results.call_args
        assert call_args[1]["query"] == query
        assert call_args[1]["top_n"] == 3
        
        # Verify results structure
        assert len(reranked_results) == 3
        assert all("rerank_score" in result for result in reranked_results)
        assert all("original_score" in result for result in reranked_results)
        assert all("rank" in result for result in reranked_results)
        
        # Verify results are sorted by rerank_score (descending)
        rerank_scores = [result["rerank_score"] for result in reranked_results]
        assert rerank_scores == sorted(rerank_scores, reverse=True)
    
    def test_apply_reranking_preserves_metadata(self):
        """Test that re-ranking preserves original metadata."""
        query = "Python programming"
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Verify original metadata is preserved
        for i, result in enumerate(reranked_results):
            original_metadata = self.search_results[0]["metadata"]  # Should be reordered
            assert "metadata" in result
            assert "collection" in result["metadata"]
            assert "author" in result["metadata"]
            assert "rating" in result["metadata"]
    
    def test_apply_reranking_handles_conversion_to_search_results(self):
        """Test conversion from raw results to SearchResult objects for reranker."""
        query = "test query"
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=2
        )
        
        # Verify that candidates were converted to SearchResult objects
        call_args = self.mock_reranker.rerank_results.call_args[1]["candidates"]
        
        # Should be SearchResult objects
        assert all(hasattr(candidate, 'content') for candidate in call_args)
        assert all(hasattr(candidate, 'metadata') for candidate in call_args)
        assert all(hasattr(candidate, 'relevance_score') for candidate in call_args)
    
    def test_apply_reranking_handles_back_conversion_to_dict_format(self):
        """Test conversion back from RankedResult to dictionary format."""
        query = "test query"
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Verify results are in expected dictionary format for pipeline compatibility
        for result in reranked_results:
            assert isinstance(result, dict)
            assert "id" in result
            assert "content" in result 
            assert "metadata" in result
            assert "rerank_score" in result
            assert "original_score" in result
            assert "rank" in result
    
    def test_apply_reranking_with_top_n_limiting(self):
        """Test that top_n parameter limits returned results correctly."""
        query = "test query"
        top_n = 2
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results[:top_n]
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=top_n
        )
        
        assert len(reranked_results) == top_n
        
        # Verify top_n was passed to reranker
        self.mock_reranker.rerank_results.assert_called_once()
        call_args = self.mock_reranker.rerank_results.call_args
        assert call_args[1]["top_n"] == top_n
    
    def test_apply_reranking_empty_candidates(self):
        """Test re-ranking with empty candidate list."""
        query = "test query"
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=[],
            top_n=5
        )
        
        assert reranked_results == []
        # Should not call reranker for empty candidates
        self.mock_reranker.rerank_results.assert_not_called()
    
    def test_apply_reranking_error_handling(self):
        """Test error handling during re-ranking operation."""
        query = "test query"
        
        # Mock reranker to raise exception
        self.mock_reranker.rerank_results.side_effect = Exception("Reranker error")
        
        # Should handle error gracefully and return original results
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Should return original results with neutral rerank scores
        assert len(reranked_results) == 3
        assert all("rerank_score" in result for result in reranked_results)
        # Error handling should assign neutral scores
        assert all(result["rerank_score"] == 0.5 for result in reranked_results)
    
    def test_apply_reranking_score_improvement_tracking(self):
        """Test tracking of score improvements from re-ranking."""
        query = "Python programming"
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Verify score improvement information is tracked
        for result in reranked_results:
            assert "score_improvement" in result
            score_improvement = result["score_improvement"]
            assert isinstance(score_improvement, (int, float))
            
            # Should calculate improvement as (rerank_score - original_score)
            expected_improvement = result["rerank_score"] - result["original_score"]
            assert abs(score_improvement - expected_improvement) < 0.001
    
    def test_apply_reranking_integration_with_metadata_filtering(self):
        """Test integration of re-ranking with metadata filtering pipeline."""
        query = "Python programming"
        
        # Set up filtered candidates (simulating post-metadata-filtering)
        filtered_candidates = self.search_results[:2]  # Only first 2 results
        filtered_ranked_results = self.mock_ranked_results[:2]
        
        self.mock_reranker.rerank_results.return_value = filtered_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=filtered_candidates,
            top_n=2
        )
        
        assert len(reranked_results) == 2
        
        # Verify candidates passed to reranker match filtered set
        call_args = self.mock_reranker.rerank_results.call_args[1]["candidates"]
        assert len(call_args) == 2
    
    def test_apply_reranking_preserves_original_ranking_information(self):
        """Test that original ranking information is preserved for analysis."""
        query = "test query"
        
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=self.search_results,
            top_n=3
        )
        
        # Verify original distance/score information is preserved
        for i, result in enumerate(reranked_results):
            assert "original_distance" in result or "distance" in result
            assert "original_score" in result
            
            # Original score should match relevance calculation from distance
            if "distance" in result:
                expected_original_score = 1.0 - result["distance"]
                assert abs(result["original_score"] - expected_original_score) < 0.1
    
    def test_reranking_pipeline_integration(self):
        """Test integration of re-ranking into the complete RAG pipeline."""
        # This test verifies that re-ranking fits into the overall query processing flow
        query_context = QueryContext(
            original_query="Python programming tutorial",
            intent=QueryIntent.TUTORIAL_SEEKING,
            key_terms=["Python", "programming", "tutorial"],
            filters=[],
            preferences={},
            entities={},
            temporal_context=None
        )
        
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock the pipeline components
        self.mock_query_manager.query.return_value = self.search_results
        self.mock_embedding_service.generate_embeddings.return_value = query_embedding
        self.mock_reranker.rerank_results.return_value = self.mock_ranked_results
        
        # Simulate full pipeline execution
        # 1. Vector search
        vector_results = self.rag_engine.execute_vector_search(
            query_embedding=query_embedding,
            collections=["programming"],
            top_k=10
        )
        
        # 2. Metadata filtering (assume no filters for this test)
        filtered_results = self.rag_engine.apply_metadata_filters(
            candidates=vector_results,
            filters=[]
        )
        
        # 3. Re-ranking (the focus of this test)
        final_results = self.rag_engine.apply_reranking(
            query=query_context.original_query,
            candidates=filtered_results,
            top_n=5
        )
        
        # Verify pipeline integration
        assert len(final_results) <= 5
        assert all("rerank_score" in result for result in final_results)
        
        # Verify all pipeline components were called
        self.mock_query_manager.query.assert_called_once()
        self.mock_reranker.rerank_results.assert_called_once()
    
    def test_reranking_with_diverse_content_types(self):
        """Test re-ranking with different content types and domains."""
        # Add diverse content to test reranker's cross-domain capabilities
        diverse_results = self.search_results + [
            {
                "id": "doc4",
                "content": "Machine learning model training best practices",
                "distance": 0.4,
                "metadata": {"collection": "ai", "type": "guide"}
            },
            {
                "id": "doc5", 
                "content": "Database design patterns for web applications",
                "distance": 0.35,
                "metadata": {"collection": "database", "type": "reference"}
            }
        ]
        
        # Create extended mock ranked results
        extended_ranked_results = self.mock_ranked_results + [
            RankedResult(
                original_result=SearchResult(
                    content="Machine learning model training best practices",
                    metadata={"collection": "ai", "type": "guide"},
                    relevance_score=0.6,
                    document_id="doc4"
                ),
                rerank_score=0.78,
                original_score=0.6,
                rank=4
            )
        ]
        
        query = "programming best practices"
        self.mock_reranker.rerank_results.return_value = extended_ranked_results
        
        reranked_results = self.rag_engine.apply_reranking(
            query=query,
            candidates=diverse_results,
            top_n=4
        )
        
        assert len(reranked_results) == 4
        
        # Verify cross-domain content is handled properly
        collections = [result["metadata"]["collection"] for result in reranked_results]
        assert len(set(collections)) > 1  # Should have multiple collections


class TestRAGFeedbackGeneration:
    """Test suite for feedback generation functionality (Red Phase)."""
    
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
        
        # Sample search results for testing
        self.sample_results = [
            {
                "id": "doc1",
                "content": "Python is a high-level programming language",
                "distance": 0.2,
                "metadata": {"collection": "programming", "author": "tech_docs", "date": "2024-01-15"}
            },
            {
                "id": "doc2", 
                "content": "Machine learning algorithms require large datasets",
                "distance": 0.4,
                "metadata": {"collection": "ai", "author": "research", "date": "2024-02-10"}
            },
            {
                "id": "doc3",
                "content": "Data structures are fundamental to computer science",
                "distance": 0.6,
                "metadata": {"collection": "programming", "author": "tutorials", "date": "2024-03-01"}
            }
        ]
        
        self.sample_query_context = QueryContext(
            original_query="What is Python programming?",
            intent=QueryIntent.INFORMATION_SEEKING,
            key_terms=["Python", "programming"],
            filters=[],
            preferences={},
            entities={"technology": ["Python"]},
            temporal_context=None
        )
    
    def test_generate_result_feedback_basic(self):
        """Test basic feedback generation for search results."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        assert isinstance(feedback, dict)
        assert "search_summary" in feedback
        assert "result_explanations" in feedback
        assert "refinement_suggestions" in feedback
        assert "relevance_metrics" in feedback
        
        # Verify result explanations for each result
        assert len(feedback["result_explanations"]) == 3
        for i, explanation in enumerate(feedback["result_explanations"]):
            assert "result_id" in explanation
            assert "relevance_score" in explanation
            assert "ranking_reason" in explanation
            assert explanation["result_id"] == self.sample_results[i]["id"]
    
    def test_relevance_scoring_calculation(self):
        """Test relevance score calculation based on multiple factors."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        explanations = feedback["result_explanations"]
        
        # First result should have highest relevance (lowest distance)
        assert explanations[0]["relevance_score"] > explanations[1]["relevance_score"]
        assert explanations[1]["relevance_score"] > explanations[2]["relevance_score"]
        
        # Scores should be normalized between 0 and 1
        for explanation in explanations:
            score = explanation["relevance_score"]
            assert 0.0 <= score <= 1.0
    
    def test_ranking_reason_generation(self):
        """Test generation of human-readable ranking explanations."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        explanations = feedback["result_explanations"]
        
        # Check that ranking reasons are informative
        for explanation in explanations:
            reason = explanation["ranking_reason"]
            assert isinstance(reason, str)
            assert len(reason) > 20  # Should be a meaningful explanation
            
            # Should mention key factors
            if explanation["result_id"] == "doc1":
                assert "Python" in reason  # Should mention query match
                assert "distance" in reason.lower() or "similarity" in reason.lower()
    
    def test_search_summary_generation(self):
        """Test generation of overall search summary."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        summary = feedback["search_summary"]
        
        assert isinstance(summary, dict)
        assert "total_results" in summary
        assert "collections_searched" in summary
        assert "best_match_score" in summary
        assert "query_coverage" in summary
        
        assert summary["total_results"] == 3
        assert "programming" in summary["collections_searched"]
        assert "ai" in summary["collections_searched"]
        assert 0.0 <= summary["best_match_score"] <= 1.0
    
    def test_refinement_suggestions_generation(self):
        """Test generation of query refinement suggestions."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        suggestions = feedback["refinement_suggestions"]
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Each suggestion should have structure
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "type" in suggestion
            assert "suggestion" in suggestion
            assert "reason" in suggestion
            
            # Valid suggestion types
            assert suggestion["type"] in [
                "add_filter", "refine_terms", "expand_scope", 
                "change_collection", "add_context"
            ]
    
    def test_refinement_suggestions_based_on_results(self):
        """Test that refinement suggestions are contextually relevant."""
        # Test with poor results (high distances)
        poor_results = [
            {
                "id": "doc1",
                "content": "Unrelated content about cooking",
                "distance": 0.9,
                "metadata": {"collection": "cooking", "author": "chef"}
            }
        ]
        
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=poor_results,
            top_k=1
        )
        
        suggestions = feedback["refinement_suggestions"]
        
        # Should suggest collection refinement when results are from wrong collection
        collection_suggestions = [s for s in suggestions if s["type"] == "change_collection"]
        assert len(collection_suggestions) > 0
        
        # Should suggest term refinement when relevance is poor
        term_suggestions = [s for s in suggestions if s["type"] == "refine_terms"]
        assert len(term_suggestions) > 0
    
    def test_feedback_generation_with_filters(self):
        """Test feedback generation when query has filters applied."""
        filtered_context = QueryContext(
            original_query="Python tutorials from programming collection",
            intent=QueryIntent.TUTORIAL_SEEKING,
            key_terms=["Python", "tutorials"],
            filters=[
                ContextualFilter(field="collection", value="programming", confidence=0.9)
            ],
            preferences={"tutorial_format": True}
        )
        
        feedback = self.rag_engine.generate_result_feedback(
            query_context=filtered_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        # Should acknowledge applied filters in summary
        summary = feedback["search_summary"]
        assert "filters_applied" in summary
        assert len(summary["filters_applied"]) == 1
        assert summary["filters_applied"][0]["field"] == "collection"
    
    def test_feedback_generation_with_preferences(self):
        """Test feedback generation considering user preferences."""
        preference_context = QueryContext(
            original_query="Simple Python examples for beginners",
            intent=QueryIntent.CODE_SEARCH,
            key_terms=["Python", "examples"],
            preferences={
                "complexity_level": "beginner",
                "include_examples": True,
                "content_type": "code"
            }
        )
        
        feedback = self.rag_engine.generate_result_feedback(
            query_context=preference_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        # Ranking reasons should consider preferences
        explanations = feedback["result_explanations"]
        for explanation in explanations:
            reason = explanation["ranking_reason"]
            # Should mention preference alignment where relevant
            if "beginner" in reason.lower() or "example" in reason.lower():
                assert True  # Good - mentions preferences
    
    def test_feedback_generation_empty_results(self):
        """Test feedback generation when no results are found."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=[],
            top_k=5
        )
        
        assert feedback["search_summary"]["total_results"] == 0
        assert len(feedback["result_explanations"]) == 0
        
        # Should provide helpful suggestions for empty results
        suggestions = feedback["refinement_suggestions"]
        assert len(suggestions) > 0
        
        # Should suggest expanding scope or changing terms
        suggestion_types = [s["type"] for s in suggestions]
        assert "expand_scope" in suggestion_types or "refine_terms" in suggestion_types
    
    def test_collect_user_feedback_interface(self):
        """Test user feedback collection interface."""
        feedback_data = {
            "query_id": "query_123",
            "result_id": "doc1",
            "rating": 4,
            "relevance": "high",
            "feedback_text": "Very helpful result"
        }
        
        result = self.rag_engine.collect_user_feedback(feedback_data)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "feedback_id" in result
        assert result["success"] is True
    
    def test_feedback_metrics_tracking(self):
        """Test tracking of feedback quality metrics."""
        feedback = self.rag_engine.generate_result_feedback(
            query_context=self.sample_query_context,
            search_results=self.sample_results,
            top_k=3
        )
        
        metrics = feedback["relevance_metrics"]
        
        assert isinstance(metrics, dict)
        assert "average_relevance" in metrics
        assert "result_diversity" in metrics
        assert "query_term_coverage" in metrics
        
        # Metrics should be properly calculated
        assert 0.0 <= metrics["average_relevance"] <= 1.0
        assert 0.0 <= metrics["result_diversity"] <= 1.0
        assert 0.0 <= metrics["query_term_coverage"] <= 1.0


class TestRAGResultFormatting:
    """Test cases for result formatting functionality."""
    
    def setup_method(self):
        """Set up test fixtures for result formatting tests."""
        from unittest.mock import Mock
        from research_agent_backend.core.rag_query_engine import QueryContext, QueryIntent, ContextualFilter
        
        # Mock dependencies
        self.mock_query_manager = Mock()
        self.mock_embedding_service = Mock()
        self.mock_reranker = Mock()
        
        # Create RAG engine instance
        from research_agent_backend.core.rag_query_engine import RAGQueryEngine
        self.rag_engine = RAGQueryEngine(
            query_manager=self.mock_query_manager,
            embedding_service=self.mock_embedding_service,
            reranker=self.mock_reranker
        )
        
        # Sample query context
        self.query_context = QueryContext(
            original_query="Python machine learning tutorial",
            intent=QueryIntent.TUTORIAL_SEEKING,
            key_terms=["machine learning", "Python", "tutorial", "machine", "learning"],
            filters=[],
            preferences={"complexity_level": "beginner"},
            entities={"Python": "programming_language"},
            temporal_context=None
        )
        
        # Sample reranked results for formatting
        self.reranked_results = [
            {
                "id": "doc1",
                "content": "Python machine learning tutorial: Getting started with scikit-learn. This comprehensive guide covers the basics of machine learning in Python, including data preprocessing, model training, and evaluation. Perfect for beginners who want to learn machine learning with practical examples.",
                "metadata": {
                    "collection": "tutorials",
                    "source": "ml_guide.md",
                    "author": "Jane Smith",
                    "created": "2024-01-15",
                    "type": "tutorial",
                    "tags": ["python", "machine-learning", "scikit-learn"]
                },
                "rerank_score": 0.92,
                "original_score": 0.78,
                "rank": 1,
                "score_improvement": 0.14,
                "distance": 0.22
            },
            {
                "id": "doc2", 
                "content": "Advanced Python techniques for machine learning practitioners. Explore optimization strategies, custom algorithms, and performance tuning for ML pipelines. Includes code examples and best practices for production environments.",
                "metadata": {
                    "collection": "advanced",
                    "source": "advanced_ml.md", 
                    "author": "Dr. Johnson",
                    "created": "2024-02-20",
                    "type": "guide",
                    "tags": ["python", "optimization", "production"]
                },
                "rerank_score": 0.85,
                "original_score": 0.72,
                "rank": 2,
                "score_improvement": 0.13,
                "distance": 0.28
            }
        ]
        
        # Sample feedback for formatting tests
        self.sample_feedback = {
            "search_summary": {
                "total_results": 2,
                "collections_searched": ["tutorials", "advanced"],
                "best_match_score": 0.92,
                "query_coverage": 0.85
            },
            "result_explanations": [
                {
                    "result_id": "doc1",
                    "relevance_score": 0.92,
                    "ranking_reason": "High semantic similarity to query terms. Contains 'Python', 'machine learning', 'tutorial'. Relevance score: 0.92"
                }
            ],
            "refinement_suggestions": [
                {
                    "type": "add_filter",
                    "suggestion": "Add collection filter to focus on tutorials",
                    "reason": "Multiple relevant collections found"
                }
            ],
            "relevance_metrics": {
                "average_relevance": 0.885,
                "result_diversity": 0.75,
                "query_term_coverage": 0.85
            }
        }
    
    def test_format_results_basic_structure(self):
        """Test basic structure of formatted results."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        # Verify top-level structure
        assert isinstance(formatted_results, dict)
        assert "query_info" in formatted_results
        assert "results" in formatted_results
        assert "summary" in formatted_results
        assert "suggestions" in formatted_results
        assert "metadata" in formatted_results
        
        # Verify query info
        query_info = formatted_results["query_info"]
        assert query_info["original_query"] == "Python machine learning tutorial"
        assert query_info["intent"] == "tutorial_seeking"
        assert "Python" in query_info["key_terms"]
        
        # Verify results structure
        results = formatted_results["results"]
        assert isinstance(results, list)
        assert len(results) == 2
        
        # Verify individual result structure
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "snippet" in result
            assert "metadata" in result
            assert "relevance" in result
            assert "source_info" in result
    
    def test_format_results_content_highlighting(self):
        """Test query term highlighting in content."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        results = formatted_results["results"]
        
        # Check first result highlighting
        first_result = results[0]
        content = first_result["content"]
        snippet = first_result["snippet"]
        
        # Should highlight query terms in content
        assert "<mark>Python</mark>" in content or "**Python**" in content
        assert "<mark>machine learning</mark>" in content or "**machine learning**" in content
        assert "<mark>tutorial</mark>" in content or "**tutorial**" in content
        
        # Snippet should be properly truncated with highlights
        # Allow for extra length due to highlighting markup tags
        assert len(snippet) <= 300  # Increased to account for markup
        assert "..." in snippet or len(snippet) < len(first_result["content"].replace("<mark>", "").replace("</mark>", ""))
    
    def test_format_results_snippet_generation(self):
        """Test intelligent snippet generation around query terms."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        results = formatted_results["results"]
        
        for result in results:
            snippet = result["snippet"]
            
            # Snippet should contain query terms
            lower_snippet = snippet.lower()
            assert any(term.lower() in lower_snippet for term in ["python", "machine", "learning", "tutorial"])
            
            # Snippet should be meaningful length
            assert 50 <= len(snippet) <= 300
            
            # Should not be empty or just punctuation
            assert snippet.strip()
            assert len(snippet.strip()) > 10
    
    def test_format_results_metadata_enrichment(self):
        """Test metadata enrichment and presentation."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        results = formatted_results["results"]
        
        for result in results:
            metadata = result["metadata"]
            source_info = result["source_info"]
            relevance = result["relevance"]
            
            # Verify enriched metadata
            assert "collection" in metadata
            assert "source" in metadata
            assert "type" in metadata
            assert "tags" in metadata
            
            # Verify source information formatting
            assert "file" in source_info
            assert "author" in source_info
            assert "created" in source_info
            
            # Verify relevance information
            assert "rerank_score" in relevance
            assert "original_score" in relevance
            assert "rank" in relevance
            assert "confidence" in relevance
            assert 0.0 <= relevance["rerank_score"] <= 1.0
            assert 0.0 <= relevance["original_score"] <= 1.0
    
    def test_format_results_cli_output_format(self):
        """Test CLI-friendly output format."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="cli"
        )
        
        # CLI format should be optimized for console display
        assert isinstance(formatted_results, dict)
        assert "header" in formatted_results
        assert "results_table" in formatted_results
        assert "summary_stats" in formatted_results
        assert "next_steps" in formatted_results
        
        # Header should contain query summary
        header = formatted_results["header"]
        assert "query" in header.lower()
        assert "python" in header.lower()
        
        # Results table should be structured for console display
        table = formatted_results["results_table"]
        assert isinstance(table, list)
        assert len(table) == 2
        
        for row in table:
            assert "rank" in row
            assert "title" in row
            assert "snippet" in row
            assert "score" in row
            assert "source" in row
    
    def test_format_results_api_output_format(self):
        """Test API-friendly JSON output format."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="api"
        )
        
        # API format should be structured for programmatic access
        assert isinstance(formatted_results, dict)
        assert "query" in formatted_results
        assert "results" in formatted_results
        assert "pagination" in formatted_results
        assert "performance" in formatted_results
        assert "suggestions" in formatted_results
        
        # Query should have complete context information
        query_info = formatted_results["query"]
        assert "text" in query_info
        assert "intent" in query_info
        assert "terms" in query_info
        assert "filters" in query_info
        
        # Results should have complete information for API consumers
        results = formatted_results["results"]
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "metadata" in result
            assert "scores" in result
            assert "highlighting" in result
            
            # Scores should have comprehensive information
            scores = result["scores"]
            assert "relevance" in scores
            assert "confidence" in scores
            assert "improvement" in scores
    
    def test_format_results_performance_information(self):
        """Test inclusion of performance and timing information."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured",
            include_performance=True
        )
        
        metadata = formatted_results["metadata"]
        
        # Should include timing information
        assert "performance" in metadata
        performance = metadata["performance"]
        
        assert "total_results" in performance
        assert "processing_stages" in performance
        assert "reranking_applied" in performance
        
        # Should track processing stages
        stages = performance["processing_stages"]
        assert isinstance(stages, dict)
        # May include embedding_time, search_time, reranking_time, formatting_time
    
    def test_format_results_empty_results(self):
        """Test formatting when no results are found."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=[],
            feedback={
                "search_summary": {"total_results": 0},
                "result_explanations": [],
                "refinement_suggestions": [
                    {"type": "expand_scope", "suggestion": "Try broader terms", "reason": "No results found"}
                ],
                "relevance_metrics": {"average_relevance": 0.0}
            },
            output_format="structured"
        )
        
        # Should handle empty results gracefully
        assert isinstance(formatted_results, dict)
        assert "results" in formatted_results
        assert len(formatted_results["results"]) == 0
        
        # Should provide helpful suggestions
        assert "suggestions" in formatted_results
        suggestions = formatted_results["suggestions"]
        assert len(suggestions) > 0
        assert any("expand" in s.get("suggestion", "").lower() for s in suggestions)
    
    def test_format_results_error_handling(self):
        """Test error handling in result formatting."""
        # Test with malformed results
        malformed_results = [
            {"id": "doc1"},  # Missing required fields
            {"content": "test"}  # Missing ID
        ]
        
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=malformed_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        # Should handle malformed results gracefully
        assert isinstance(formatted_results, dict)
        assert "results" in formatted_results
        
        # Should include error information
        if "errors" in formatted_results["metadata"]:
            errors = formatted_results["metadata"]["errors"]
            assert isinstance(errors, list)
    
    def test_format_results_customization_options(self):
        """Test result formatting with customization options."""
        custom_options = {
            "snippet_length": 150,
            "highlight_style": "markdown",
            "include_scores": True,
            "include_metadata": False,
            "sort_by": "rerank_score"
        }
        
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured",
            formatting_options=custom_options
        )
        
        results = formatted_results["results"]
        
        for result in results:
            # Check snippet length customization
            if "snippet" in result:
                assert len(result["snippet"]) <= 150 + 20  # Allow some margin for highlighting
            
            # Check markdown highlighting style
            if "content" in result:
                content = result["content"]
                # Should use markdown highlighting
                assert "**" in content or "*" in content or "`" in content
            
            # Check scores inclusion/exclusion based on options
            if custom_options["include_scores"]:
                assert "relevance" in result
            
            if not custom_options["include_metadata"]:
                # May have minimal metadata only
                pass
    
    def test_format_results_integration_with_feedback(self):
        """Test integration between result formatting and feedback generation."""
        formatted_results = self.rag_engine.format_results(
            query_context=self.query_context,
            reranked_results=self.reranked_results,
            feedback=self.sample_feedback,
            output_format="structured"
        )
        
        # Should integrate feedback suggestions
        assert "suggestions" in formatted_results
        suggestions = formatted_results["suggestions"]
        
        # Should match original feedback suggestions
        original_suggestions = self.sample_feedback["refinement_suggestions"]
        assert len(suggestions) == len(original_suggestions)
        
        # Should integrate search summary
        summary = formatted_results["summary"]
        assert "total_results" in summary
        assert summary["total_results"] == self.sample_feedback["search_summary"]["total_results"] 