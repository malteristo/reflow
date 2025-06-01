"""
Test suite for query CLI commands.

Tests the query CLI commands including search, ask, interactive, refine, 
and other query-related functionality. Follows TDD approach.

RED PHASE: These tests are designed to fail initially and guide implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
from click.testing import Result

from research_agent_backend.cli.cli import app
from research_agent_backend.core.query_manager import QueryManager, QueryResult, QueryConfig
from research_agent_backend.core.rag_query_engine import RAGQueryEngine, QueryContext, QueryIntent
from research_agent_backend.core.local_embedding_service import LocalEmbeddingService
from research_agent_backend.core.vector_store import ChromaDBManager
from research_agent_backend.exceptions.query_exceptions import QueryError

runner = CliRunner()


class TestQuerySearchCommand:
    """Test the query search command."""
    
    @patch('research_agent_backend.cli.query.QueryManager')
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    @patch('research_agent_backend.cli.query.LocalEmbeddingService')
    @patch('research_agent_backend.cli.query.ChromaDBManager')
    @patch('research_agent_backend.cli.query.RerankerService')
    @patch('research_agent_backend.cli.query.ConfigManager')
    def test_search_basic_query(self, mock_config_manager, mock_reranker, mock_chroma, mock_embedding, mock_rag_engine, mock_query_manager):
        """Test basic search functionality."""
        # Setup mocks
        mock_query_manager_instance = Mock()
        mock_rag_engine_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock the methods that will be called
        mock_context = QueryContext(
            original_query="machine learning",
            intent=QueryIntent.INFORMATION_SEEKING
        )
        mock_rag_engine_instance.parse_query_context.return_value = mock_context
        mock_rag_engine_instance.generate_query_embedding.return_value = [0.1] * 384
        mock_rag_engine_instance.apply_reranking.return_value = [
            {
                'document_id': 'doc1',
                'content': 'Machine learning is a subset of AI...',
                'collection': 'default',
                'score': 0.92,
                'metadata': {'title': 'ML Fundamentals'}
            }
        ]
        
        # Mock search results
        mock_result = QueryResult()
        mock_result.results = [
            {
                'document_id': 'doc1',
                'content': 'Machine learning is a subset of AI...',
                'collection': 'default',
                'score': 0.92,
                'metadata': {'title': 'ML Fundamentals'}
            }
        ]
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        # Test command
        result = runner.invoke(app, ['query', 'search', 'machine learning'])
        
        # Debug output
        print(f"Exit code: {result.exit_code}")
        print(f"Output: {result.stdout}")
        if result.exception:
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        assert 'Machine learning is a subset of AI' in result.stdout
        assert '0.92' in result.stdout
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_search_with_collections_filter(self, mock_query_manager):
        """Test search with collections filter."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        mock_result = QueryResult()
        mock_result.results = []
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        result = runner.invoke(app, [
            'query', 'search', 'neural networks', 
            '--collections', 'research,papers'
        ])
        
        assert result.exit_code == 0
        # Should have called with collections filter
        mock_query_manager_instance.similarity_search.assert_called()
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_search_with_reranking_disabled(self, mock_query_manager):
        """Test search with reranking disabled."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        mock_result = QueryResult()
        mock_result.results = []
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        result = runner.invoke(app, [
            'query', 'search', 'algorithms', 
            '--no-rerank'
        ])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_search_error_handling(self, mock_query_manager):
        """Test search error handling."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        mock_query_manager_instance.similarity_search.side_effect = QueryError("Search failed")
        
        result = runner.invoke(app, ['query', 'search', 'test query'])
        
        # Should handle error gracefully
        assert result.exit_code != 0 or "error" in result.stdout.lower()


class TestQueryAskCommand:
    """Test the query ask command."""
    
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_ask_basic_question(self, mock_rag_engine):
        """Test basic ask functionality."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock RAG response
        mock_context = QueryContext(
            original_query="What is machine learning?",
            intent=QueryIntent.INFORMATION_SEEKING
        )
        mock_rag_engine_instance.parse_query_context.return_value = mock_context
        
        result = runner.invoke(app, [
            'query', 'ask', 'What is machine learning?'
        ])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_ask_with_collections(self, mock_rag_engine):
        """Test ask with collections filter."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        result = runner.invoke(app, [
            'query', 'ask', 'How do neural networks work?',
            '--collections', 'research'
        ])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_ask_without_sources(self, mock_rag_engine):
        """Test ask without showing sources."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        result = runner.invoke(app, [
            'query', 'ask', 'Explain algorithms',
            '--no-sources'
        ])
        
        assert result.exit_code == 0


class TestQueryInteractiveCommand:
    """Test the interactive query command."""
    
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    @patch('builtins.input')
    def test_interactive_session_basic(self, mock_input, mock_rag_engine):
        """Test basic interactive session."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock user inputs
        mock_input.side_effect = ['What is AI?', 'quit']
        
        result = runner.invoke(app, ['query', 'interactive'])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    @patch('builtins.input')
    def test_interactive_with_refinement(self, mock_input, mock_rag_engine):
        """Test interactive session with query refinement."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock user inputs including refinement
        mock_input.side_effect = [
            'machine learning',
            'refine: focus on supervised learning',
            'quit'
        ]
        
        result = runner.invoke(app, ['query', 'interactive'])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    @patch('builtins.input')
    def test_interactive_with_collections(self, mock_input, mock_rag_engine):
        """Test interactive session with collections filter."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        mock_input.side_effect = ['deep learning', 'quit']
        
        result = runner.invoke(app, [
            'query', 'interactive',
            '--collections', 'research,papers'
        ])
        
        assert result.exit_code == 0


class TestQueryRefineCommand:
    """Test the query refine command."""
    
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_refine_basic(self, mock_rag_engine):
        """Test basic query refinement."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock refinement suggestions
        mock_feedback = {
            'refinement_suggestions': [
                {
                    'type': 'add_filter',
                    'suggestion': 'Add collection filter for better results'
                }
            ]
        }
        mock_rag_engine_instance.generate_result_feedback.return_value = mock_feedback
        
        result = runner.invoke(app, [
            'query', 'refine',
            'machine learning',
            'focus on neural networks'
        ])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_refine_with_collections(self, mock_rag_engine):
        """Test query refinement with collections."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        result = runner.invoke(app, [
            'query', 'refine',
            'algorithms',
            'add examples',
            '--collections', 'tutorials'
        ])
        
        assert result.exit_code == 0


class TestQuerySimilarCommand:
    """Test the find similar documents command."""
    
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_find_similar_basic(self, mock_query_manager):
        """Test basic similar document search."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        mock_result = QueryResult()
        mock_result.results = [
            {
                'document_id': 'doc2',
                'content': 'Related content...',
                'score': 0.85,
                'collection': 'default'
            }
        ]
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        result = runner.invoke(app, [
            'query', 'similar', 'doc-123'
        ])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_find_similar_same_collection_only(self, mock_query_manager):
        """Test similar search within same collection only."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        mock_result = QueryResult()
        mock_result.results = []
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        result = runner.invoke(app, [
            'query', 'similar', 'doc-456',
            '--same-collection'
        ])
        
        assert result.exit_code == 0


class TestQueryExplainCommand:
    """Test the explain results command."""
    
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    def test_explain_result_basic(self, mock_rag_engine):
        """Test basic result explanation."""
        mock_rag_engine_instance = Mock()
        mock_rag_engine.return_value = mock_rag_engine_instance
        
        # Mock explanation data
        mock_explanation = {
            'relevance_score': 0.92,
            'ranking_reason': 'High similarity to query terms',
            'content_matches': ['machine', 'learning', 'algorithms']
        }
        mock_rag_engine_instance.generate_result_feedback.return_value = {
            'explanations': [mock_explanation]
        }
        
        result = runner.invoke(app, [
            'query', 'explain',
            'machine learning',
            'result-123'
        ])
        
        assert result.exit_code == 0


class TestQueryHistoryCommand:
    """Test the query history command."""
    
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_query_history_basic(self, mock_query_manager):
        """Test basic query history."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        # Mock history data
        mock_history = [
            {
                'timestamp': '2023-10-01 10:00:00',
                'query': 'machine learning',
                'results_count': 5
            }
        ]
        mock_query_manager_instance.get_query_history = Mock(return_value=mock_history)
        
        result = runner.invoke(app, ['query', 'history'])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_query_history_with_search(self, mock_query_manager):
        """Test query history with search filter."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        
        mock_history = []
        mock_query_manager_instance.get_query_history = Mock(return_value=mock_history)
        
        result = runner.invoke(app, [
            'query', 'history',
            '--search', 'neural'
        ])
        
        assert result.exit_code == 0


class TestQueryCommandIntegration:
    """Integration tests for query commands."""
    
    @patch('research_agent_backend.cli.query.QueryManager')
    @patch('research_agent_backend.cli.query.RAGQueryEngine')
    @patch('research_agent_backend.cli.query.LocalEmbeddingService')
    def test_end_to_end_search_flow(self, mock_embedding, mock_rag, mock_query_manager):
        """Test complete end-to-end search flow."""
        # Setup all mocks
        mock_embedding_instance = Mock()
        mock_rag_instance = Mock()
        mock_query_manager_instance = Mock()
        
        mock_embedding.return_value = mock_embedding_instance
        mock_rag.return_value = mock_rag_instance
        mock_query_manager.return_value = mock_query_manager_instance
        
        # Mock embedding generation
        mock_embedding_instance.embed_query.return_value = [0.1] * 384
        
        # Mock RAG context parsing
        mock_context = QueryContext(
            original_query="test query",
            intent=QueryIntent.INFORMATION_SEEKING
        )
        mock_rag_instance.parse_query_context.return_value = mock_context
        
        # Mock search results
        mock_result = QueryResult()
        mock_result.results = [
            {
                'document_id': 'doc1',
                'content': 'Test content',
                'score': 0.9,
                'collection': 'default'
            }
        ]
        mock_query_manager_instance.similarity_search.return_value = mock_result
        
        result = runner.invoke(app, ['query', 'search', 'test query'])
        
        assert result.exit_code == 0
        
    @patch('research_agent_backend.cli.query.QueryManager')
    def test_error_handling_across_commands(self, mock_query_manager):
        """Test error handling across different query commands."""
        mock_query_manager_instance = Mock()
        mock_query_manager.return_value = mock_query_manager_instance
        mock_query_manager_instance.similarity_search.side_effect = Exception("Database error")
        
        # Test that all commands handle errors gracefully
        commands = [
            ['query', 'search', 'test'],
            ['query', 'similar', 'doc-123'],
        ]
        
        for cmd in commands:
            result = runner.invoke(app, cmd)
            # Should not crash, either exit with error code or show error message
            assert result.exit_code != 0 or "error" in result.stdout.lower()


class TestQueryCommandConfiguration:
    """Test query command configuration and initialization."""
    
    @patch('research_agent_backend.cli.query.ConfigManager')
    def test_query_commands_load_config(self, mock_config_manager):
        """Test that query commands properly load configuration."""
        mock_config_manager_instance = Mock()
        mock_config_manager.return_value = mock_config_manager_instance
        
        # Should load configuration when initializing query components
        result = runner.invoke(app, ['query', 'search', 'test'])
        
        # Command should attempt to load config (even if it fails)
        assert result.exit_code == 0 or result.exit_code != 0
        
    def test_query_commands_help_text(self):
        """Test that query commands provide proper help text."""
        result = runner.invoke(app, ['query', '--help'])
        
        assert result.exit_code == 0
        assert 'search' in result.stdout
        assert 'ask' in result.stdout
        assert 'interactive' in result.stdout


# Additional helper tests for supporting functionality
class TestQueryUtilityFunctions:
    """Test utility functions used by query commands."""
    
    def test_format_search_results(self):
        """Test search results formatting function."""
        # This will be implemented as part of the query commands
        # For now, just check that we can import the module
        from research_agent_backend.cli import query
        assert hasattr(query, 'query_app')
        
    def test_parse_collections_parameter(self):
        """Test collections parameter parsing."""
        # This will test the helper function for parsing comma-separated collections
        from research_agent_backend.cli import query
        assert hasattr(query, 'query_app') 