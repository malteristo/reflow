"""
Tests for the Research Agent Knowledge Base Utility CLI commands.

This module tests the utility commands including status and rebuild-index
that need implementation in subtask 8.8.

Implements TDD testing for utility command functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typer.testing import CliRunner

from research_agent_backend.cli.cli import app
from research_agent_backend.core.document_insertion import (
    DocumentInsertionManager,
    InsertionResult,
    BatchInsertionResult,
    create_document_insertion_manager
)
from research_agent_backend.core.vector_store import create_chroma_manager
from research_agent_backend.core.vector_store.types import HealthStatus, CollectionStats


class TestStatusCommand:
    """Test suite for the status CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_status_command_basic_functionality(self):
        """Test basic status command functionality."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            
            # Create proper HealthStatus mock
            health_status = HealthStatus(
                status='healthy',
                connected=True,
                persist_directory='/test',
                collections_count=3,
                collections=['default', 'research', 'archive'],
                timestamp='2024-01-15T10:00:00Z',
                errors=[]
            )
            mock_chroma.health_check.return_value = health_status
            
            # Mock list_collections to return collection info objects
            mock_collection_infos = []
            for name in ['default', 'research', 'archive']:
                mock_collection_info = Mock()
                mock_collection_info.name = name
                mock_collection_infos.append(mock_collection_info)
            mock_chroma.list_collections.return_value = mock_collection_infos
            
            # Mock get_collection_stats for each collection
            def get_collection_stats_side_effect(collection_name):
                stats_map = {
                    'default': CollectionStats(
                        name='default',
                        id='default-id',
                        document_count=25,
                        metadata={},
                        timestamp='2024-01-15T10:00:00Z',
                        storage_size_bytes=1024 * 1024 * 3  # 3MB
                    ),
                    'research': CollectionStats(
                        name='research',
                        id='research-id',
                        document_count=15,
                        metadata={},
                        timestamp='2024-01-15T10:00:00Z',
                        storage_size_bytes=1024 * 1024 * 2  # 2MB
                    ),
                    'archive': CollectionStats(
                        name='archive',
                        id='archive-id',
                        document_count=8,
                        metadata={},
                        timestamp='2024-01-15T10:00:00Z',
                        storage_size_bytes=1024 * 1024 * 1  # 1MB
                    )
                }
                return stats_map[collection_name]
            
            mock_chroma.get_collection_stats.side_effect = get_collection_stats_side_effect
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            # Debug output
            print(f"Exit code: {result.exit_code}")
            print(f"Stdout: {result.stdout}")
            print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            assert "Knowledge Base Status" in result.stdout
            assert "Total Documents: 48" in result.stdout  # 25 + 15 + 8 = 48
            assert "Total Storage: 6.0 MB" in result.stdout  # 3 + 2 + 1 = 6
            assert "Collections: 3" in result.stdout
            assert "default" in result.stdout
            assert "research" in result.stdout
            assert "archive" in result.stdout
            assert "Health: OK" in result.stdout
    
    def test_status_command_shows_collection_details(self):
        """Test that status command shows detailed collection information."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_collection_info.return_value = {
                'default': {'document_count': 25, 'size_mb': 3.2, 'last_updated': '2024-01-15'},
                'research': {'document_count': 15, 'size_mb': 2.1, 'last_updated': '2024-01-14'}
            }
            mock_chroma.get_total_documents.return_value = 40
            mock_chroma.get_total_storage_size.return_value = 5.3
            mock_chroma.health_check.return_value = True
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            assert result.exit_code == 0
            # Should show collection details in table format
            assert "25" in result.stdout  # default collection doc count
            assert "15" in result.stdout  # research collection doc count
            assert "3.2" in result.stdout  # default collection size
            assert "2.1" in result.stdout  # research collection size
    
    def test_status_command_health_check(self):
        """Test status command health check functionality."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.health_check.return_value = False
            mock_chroma.get_health_details.return_value = {
                'database_accessible': True,
                'embeddings_service': False,
                'storage_writable': True,
                'last_error': 'Embedding service timeout'
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            assert result.exit_code == 0
            assert "Health: DEGRADED" in result.stdout or "Health: ERROR" in result.stdout
            assert "Embedding service timeout" in result.stdout
    
    def test_status_command_empty_knowledge_base(self):
        """Test status command with empty knowledge base."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_collection_info.return_value = {}
            mock_chroma.get_total_documents.return_value = 0
            mock_chroma.get_total_storage_size.return_value = 0.0
            mock_chroma.health_check.return_value = True
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            assert result.exit_code == 0
            assert "Total Documents: 0" in result.stdout
            assert "No collections found" in result.stdout or "Collections: 0" in result.stdout
    
    def test_status_command_system_metrics(self):
        """Test status command includes system performance metrics."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_collection_info.return_value = {'default': {'document_count': 10, 'size_mb': 1.0}}
            mock_chroma.get_total_documents.return_value = 10
            mock_chroma.get_total_storage_size.return_value = 1.0
            mock_chroma.health_check.return_value = True
            mock_chroma.get_performance_metrics.return_value = {
                'avg_query_time_ms': 45.2,
                'total_queries': 150,
                'cache_hit_rate': 0.78
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            assert result.exit_code == 0
            assert "Performance Metrics" in result.stdout
            assert "45.2" in result.stdout  # avg query time
            assert "150" in result.stdout   # total queries
            assert "78%" in result.stdout   # cache hit rate
    
    def test_status_command_configuration_info(self):
        """Test status command shows current configuration."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            with patch('research_agent_backend.cli.knowledge_base.get_config_manager') as mock_get_config:
                mock_chroma = Mock()
                mock_chroma.get_collection_info.return_value = {}
                mock_chroma.get_total_documents.return_value = 0
                mock_chroma.health_check.return_value = True
                mock_create_chroma.return_value = mock_chroma
                
                mock_config = Mock()
                mock_config.get_embedding_model_name.return_value = "multi-qa-MiniLM-L6-cos-v1"
                mock_config.get_vector_store_type.return_value = "chromadb"
                mock_config.get_chunk_size.return_value = 512
                mock_get_config.return_value = mock_config
                
                result = self.runner.invoke(app, ["kb", "status"])
                
                assert result.exit_code == 0
                assert "Configuration" in result.stdout
                assert "multi-qa-MiniLM-L6-cos-v1" in result.stdout
                assert "chromadb" in result.stdout
                assert "512" in result.stdout
    
    def test_status_command_error_handling(self):
        """Test status command error handling."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_create_chroma.side_effect = Exception("Database connection failed")
            
            result = self.runner.invoke(app, ["kb", "status"])
            
            assert result.exit_code != 0
            assert "Error" in result.stdout
            assert "Database connection failed" in result.stdout


class TestRebuildIndexCommand:
    """Test suite for the rebuild-index CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_rebuild_index_basic_functionality(self):
        """Test basic rebuild-index command functionality."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
                with patch('builtins.input', return_value='y'):  # Confirm rebuild
                    mock_manager = Mock(spec=DocumentInsertionManager)
                    mock_manager.rebuild_collection_index.return_value = {
                        'success': True, 
                        'documents_processed': 45,
                        'processing_time_seconds': 12.5,
                        'embeddings_regenerated': 45
                    }
                    mock_create_manager.return_value = mock_manager
                    
                    mock_chroma = Mock()
                    # Mock list_collections to return collection objects with name attributes
                    mock_collection_1 = Mock()
                    mock_collection_1.name = 'default'
                    mock_collection_2 = Mock()
                    mock_collection_2.name = 'research'
                    mock_chroma.list_collections.return_value = [mock_collection_1, mock_collection_2]
                    
                    # Mock get_collection_stats for before/after stats
                    def get_collection_stats_side_effect(collection_name):
                        return CollectionStats(
                            name=collection_name,
                            id=f'{collection_name}-id',
                            document_count=25 if collection_name == 'default' else 20,
                            metadata={},
                            timestamp='2024-01-15T10:00:00Z',
                            storage_size_bytes=1024 * 1024 * 2  # 2MB
                        )
                    mock_chroma.get_collection_stats.side_effect = get_collection_stats_side_effect
                    
                    mock_create_chroma.return_value = mock_chroma
                    
                    result = self.runner.invoke(app, ["kb", "rebuild-index"])
                    
                    # Debug output
                    print(f"Exit code: {result.exit_code}")
                    print(f"Stdout: {result.stdout}")
                    print(f"Exception: {result.exception}")
                    
                    assert result.exit_code == 0
                    assert "Index Rebuild Plan" in result.stdout
                    assert "Total documents to process: 45" in result.stdout  # 25 + 20
                    assert "Continue with rebuild? (y/N):" in result.stdout
                    assert "Rebuild completed successfully" in result.stdout
    
    def test_rebuild_index_specific_collection(self):
        """Test rebuild-index with specific collection."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            with patch('builtins.input', return_value='y'):  # Confirm rebuild
                mock_manager = Mock(spec=DocumentInsertionManager)
                mock_manager.rebuild_collection_index.return_value = {
                    'success': True,
                    'collection': 'research',
                    'documents_processed': 15,
                    'processing_time_seconds': 4.2
                }
                mock_create_manager.return_value = mock_manager
                
                result = self.runner.invoke(app, [
                    "kb", "rebuild-index",
                    "--collection", "research"
                ])
                
                assert result.exit_code == 0
                assert "Successfully rebuilt index for collection 'research'" in result.stdout
                assert "15 documents processed" in result.stdout
                assert "4.2" in result.stdout
    
    def test_rebuild_index_with_confirm_flag(self):
        """Test rebuild-index with --confirm flag to skip prompt."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_manager.rebuild_index.return_value = {
                'success': True,
                'documents_processed': 30,
                'processing_time_seconds': 8.1
            }
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "rebuild-index",
                "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "Successfully rebuilt index" in result.stdout
            assert "30 documents processed" in result.stdout
            # Should not show confirmation prompt
            assert "Are you sure" not in result.stdout
    
    def test_rebuild_index_user_cancellation(self):
        """Test rebuild-index when user cancels the operation."""
        with patch('builtins.input', return_value='n'):  # Cancel rebuild
            result = self.runner.invoke(app, ["kb", "rebuild-index"])
            
            assert result.exit_code == 0
            assert "Operation cancelled" in result.stdout or "Cancelled" in result.stdout
    
    def test_rebuild_index_dry_run_mode(self):
        """Test rebuild-index with dry-run mode."""
        result = self.runner.invoke(app, [
            "--dry-run",
            "kb", "rebuild-index",
            "--collection", "test-collection"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "Would rebuild index" in result.stdout
        assert "test-collection" in result.stdout
    
    def test_rebuild_index_progress_tracking(self):
        """Test that rebuild-index shows progress during processing."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            with patch('builtins.input', return_value='y'):  # Confirm rebuild
                mock_manager = Mock(spec=DocumentInsertionManager)
                
                # Simulate progress callback
                def mock_rebuild_index(progress_callback=None, collection_name=None):
                    if progress_callback:
                        progress_callback(10, 50, "Processing embeddings...")
                        progress_callback(25, 50, "Updating index...")
                        progress_callback(50, 50, "Complete")
                    
                    return {
                        'success': True,
                        'documents_processed': 50,
                        'processing_time_seconds': 15.3
                    }
                
                mock_manager.rebuild_index.side_effect = mock_rebuild_index
                mock_create_manager.return_value = mock_manager
                
                result = self.runner.invoke(app, ["kb", "rebuild-index", "--confirm"])
                
                assert result.exit_code == 0
                assert "Processing" in result.stdout or "Progress" in result.stdout
    
    def test_rebuild_index_handles_errors(self):
        """Test rebuild-index error handling."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            with patch('builtins.input', return_value='y'):  # Confirm rebuild
                mock_manager = Mock(spec=DocumentInsertionManager)
                mock_manager.rebuild_index.return_value = {
                    'success': False,
                    'error_message': 'Embedding service unavailable',
                    'documents_processed': 0
                }
                mock_create_manager.return_value = mock_manager
                
                result = self.runner.invoke(app, ["kb", "rebuild-index", "--confirm"])
                
                assert result.exit_code != 0
                assert "Failed to rebuild index" in result.stdout
                assert "Embedding service unavailable" in result.stdout
    
    def test_rebuild_index_validates_collection_exists(self):
        """Test rebuild-index validates that specified collection exists."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.collection_exists.return_value = False
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "rebuild-index",
                "--collection", "non-existent-collection",
                "--confirm"
            ])
            
            assert result.exit_code != 0
            assert "Collection 'non-existent-collection' not found" in result.stdout
    
    def test_rebuild_index_shows_before_after_stats(self):
        """Test rebuild-index shows before/after statistics."""
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
                with patch('builtins.input', return_value='y'):  # Confirm rebuild
                    mock_manager = Mock(spec=DocumentInsertionManager)
                    mock_manager.rebuild_index.return_value = {
                        'success': True,
                        'documents_processed': 30,
                        'processing_time_seconds': 7.5,
                        'before_stats': {'total_embeddings': 150, 'total_chunks': 150},
                        'after_stats': {'total_embeddings': 148, 'total_chunks': 148}
                    }
                    mock_create_manager.return_value = mock_manager
                    
                    mock_chroma = Mock()
                    mock_chroma.get_total_documents.return_value = 30
                    mock_create_chroma.return_value = mock_chroma
                    
                    result = self.runner.invoke(app, ["kb", "rebuild-index", "--confirm"])
                    
                    assert result.exit_code == 0
                    assert "Before: 150" in result.stdout or "150 embeddings" in result.stdout
                    assert "After: 148" in result.stdout or "148 embeddings" in result.stdout


class TestUtilityCommandIntegration:
    """Test suite for utility command integration and workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_status_and_rebuild_workflow(self):
        """Test typical workflow of checking status then rebuilding index."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
                # First check status
                mock_chroma = Mock()
                mock_chroma.get_collection_info.return_value = {'default': {'document_count': 10, 'size_mb': 1.0}}
                mock_chroma.get_total_documents.return_value = 10
                mock_chroma.health_check.return_value = False  # Health issue detected
                mock_create_chroma.return_value = mock_chroma
                
                status_result = self.runner.invoke(app, ["kb", "status"])
                assert status_result.exit_code == 0
                assert "DEGRADED" in status_result.stdout or "ERROR" in status_result.stdout
                
                # Then rebuild index to fix issues
                with patch('builtins.input', return_value='y'):
                    mock_manager = Mock(spec=DocumentInsertionManager)
                    mock_manager.rebuild_index.return_value = {
                        'success': True,
                        'documents_processed': 10,
                        'processing_time_seconds': 3.0
                    }
                    mock_create_manager.return_value = mock_manager
                    
                    rebuild_result = self.runner.invoke(app, ["kb", "rebuild-index"])
                    assert rebuild_result.exit_code == 0
                    assert "Successfully rebuilt index" in rebuild_result.stdout
    
    def test_utility_commands_respect_global_options(self):
        """Test that utility commands respect global CLI options."""
        # Test verbose mode
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_collection_info.return_value = {}
            mock_chroma.get_total_documents.return_value = 0
            mock_chroma.health_check.return_value = True
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, ["--verbose", "kb", "status"])
            assert result.exit_code == 0
            
        # Test dry-run mode for both commands
        status_dry = self.runner.invoke(app, ["--dry-run", "kb", "status"])
        assert status_dry.exit_code == 0
        
        rebuild_dry = self.runner.invoke(app, ["--dry-run", "kb", "rebuild-index"])
        assert rebuild_dry.exit_code == 0
        assert "DRY RUN" in rebuild_dry.stdout 