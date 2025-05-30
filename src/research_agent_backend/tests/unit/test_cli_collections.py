"""
Tests for the Research Agent Collections CLI commands.

This module tests the collection management commands including
creation, listing, info retrieval, and deletion functionality.

Implements TDD testing for FR-KB-005: Collection management.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from research_agent_backend.cli.cli import app
from research_agent_backend.core.vector_store import (
    ChromaDBManager,
    CollectionManager,
    CollectionInfo,
    CollectionStats,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    VectorStoreError
)
from research_agent_backend.models.metadata_schema import CollectionType


class TestCreateCollectionCommand:
    """Test suite for the create-collection CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_create_collection_basic_functionality(self):
        """Test basic create collection command functionality."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock successful collection creation
            mock_collection = Mock()
            mock_collection.name = "test-collection"
            mock_collection_manager.create_collection.return_value = mock_collection
            
            result = self.runner.invoke(app, [
                "collections", "create", "test-collection",
                "--description", "Test collection description",
                "--type", "general"
            ])
            
            assert result.exit_code == 0
            assert "Successfully created collection" in result.stdout
            assert "test-collection" in result.stdout
            
            # Verify collection manager was called correctly
            mock_collection_manager.create_collection.assert_called_once()
            call_args = mock_collection_manager.create_collection.call_args
            assert call_args[1]['name'] == "test-collection"
            assert call_args[1]['collection_type'] == CollectionType.GENERAL
    
    def test_create_collection_with_different_types(self):
        """Test create collection with different collection types."""
        test_cases = [
            ("fundamental", CollectionType.FUNDAMENTAL),
            ("project-specific", CollectionType.PROJECT_SPECIFIC),
            ("general", CollectionType.GENERAL),
            ("reference", CollectionType.REFERENCE),
            ("temporary", CollectionType.TEMPORARY)
        ]
        
        for type_str, expected_enum in test_cases:
            with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
                mock_manager = Mock(spec=ChromaDBManager)
                mock_collection_manager = Mock(spec=CollectionManager)
                mock_manager.collection_manager = mock_collection_manager
                mock_create_manager.return_value = mock_manager
                
                mock_collection = Mock()
                mock_collection.name = f"test-{type_str}"
                mock_collection_manager.create_collection.return_value = mock_collection
                
                result = self.runner.invoke(app, [
                    "collections", "create", f"test-{type_str}",
                    "--type", type_str
                ])
                
                assert result.exit_code == 0
                call_args = mock_collection_manager.create_collection.call_args
                assert call_args[1]['collection_type'] == expected_enum
    
    def test_create_collection_already_exists_error(self):
        """Test create collection when collection already exists."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection already exists error
            mock_collection_manager.create_collection.side_effect = CollectionAlreadyExistsError(
                "Collection 'existing-collection' already exists"
            )
            
            result = self.runner.invoke(app, [
                "collections", "create", "existing-collection"
            ])
            
            assert result.exit_code == 1
            assert "already exists" in result.stdout.lower()
    
    def test_create_collection_dry_run_mode(self):
        """Test create collection in dry-run mode."""
        with patch('research_agent_backend.cli.collections._get_global_config') as mock_config:
            mock_config.return_value = {"dry_run": True}
            
            result = self.runner.invoke(app, [
                "collections", "create", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "DRY RUN" in result.stdout
            assert "Would create collection" in result.stdout


class TestListCollectionsCommand:
    """Test suite for the list-collections CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_list_collections_basic_functionality(self):
        """Test basic list collections command functionality."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection list
            mock_collections = [
                CollectionInfo(
                    name="default",
                    id="default-id",
                    metadata={"collection_type": "general", "description": "Default collection"},
                    count=15,
                    created_at="2024-01-01T00:00:00",
                    owner_id="user1",
                    team_id=None
                ),
                CollectionInfo(
                    name="research",
                    id="research-id", 
                    metadata={"collection_type": "fundamental", "description": "Research papers"},
                    count=42,
                    created_at="2024-01-02T00:00:00",
                    owner_id="user1",
                    team_id=None
                )
            ]
            mock_collection_manager.list_collections.return_value = mock_collections
            
            result = self.runner.invoke(app, ["collections", "list"])
            
            assert result.exit_code == 0
            assert "default" in result.stdout
            assert "research" in result.stdout
            assert "general" in result.stdout
            assert "fundamental" in result.stdout
    
    def test_list_collections_with_stats(self):
        """Test list collections with statistics."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            mock_collections = [
                CollectionInfo(
                    name="test-collection",
                    id="test-id",
                    metadata={"collection_type": "general"},
                    count=25,
                    created_at="2024-01-01T00:00:00",
                    owner_id="user1",
                    team_id=None
                )
            ]
            mock_collection_manager.list_collections.return_value = mock_collections
            
            result = self.runner.invoke(app, ["collections", "list", "--stats"])
            
            assert result.exit_code == 0
            assert "25" in result.stdout  # Document count
            assert "Documents" in result.stdout  # Stats header
    
    def test_list_collections_filter_by_type(self):
        """Test list collections filtered by type."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock filtered collections (only fundamental type)
            mock_collections = [
                CollectionInfo(
                    name="research",
                    id="research-id",
                    metadata={"collection_type": "fundamental"},
                    count=42,
                    created_at="2024-01-01T00:00:00",
                    owner_id="user1",
                    team_id=None
                )
            ]
            mock_collection_manager.get_collections_by_type.return_value = mock_collections
            
            result = self.runner.invoke(app, [
                "collections", "list", "--type", "fundamental"
            ])
            
            assert result.exit_code == 0
            assert "research" in result.stdout
            
            # Verify filter was applied
            mock_collection_manager.get_collections_by_type.assert_called_once_with(CollectionType.FUNDAMENTAL)
    
    def test_list_collections_empty_result(self):
        """Test list collections when no collections exist."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            mock_collection_manager.list_collections.return_value = []
            
            result = self.runner.invoke(app, ["collections", "list"])
            
            assert result.exit_code == 0
            assert "No collections found" in result.stdout or "empty" in result.stdout.lower()


class TestCollectionInfoCommand:
    """Test suite for the collection-info CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_collection_info_basic_functionality(self):
        """Test basic collection info command functionality."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection stats
            mock_stats = CollectionStats(
                name="test-collection",
                id="test-id",
                document_count=25,
                metadata={
                    "collection_type": "general",
                    "description": "Test collection",
                    "created_at": "2024-01-01T00:00:00",
                    "owner_id": "user1"
                },
                timestamp="2024-01-01T00:00:00",
                collection_type="general",
                owner_id="user1",
                team_id=None
            )
            mock_collection_manager.get_collection_stats.return_value = mock_stats
            
            result = self.runner.invoke(app, ["collections", "info", "test-collection"])
            
            assert result.exit_code == 0
            assert "test-collection" in result.stdout
            assert "25" in result.stdout  # Document count
            assert "general" in result.stdout  # Collection type
            assert "Test collection" in result.stdout  # Description
    
    def test_collection_info_not_found(self):
        """Test collection info for non-existent collection."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection not found error
            mock_collection_manager.get_collection_stats.side_effect = CollectionNotFoundError(
                "Collection 'non-existent' not found"
            )
            
            result = self.runner.invoke(app, ["collections", "info", "non-existent"])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()


class TestDeleteCollectionCommand:
    """Test suite for the delete-collection CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_delete_collection_with_confirmation(self):
        """Test delete collection with confirmation flag."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "collections", "delete", "test-collection", "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "Successfully deleted" in result.stdout
            mock_collection_manager.delete_collection.assert_called_once_with("test-collection")
    
    def test_delete_collection_interactive_confirmation(self):
        """Test delete collection with interactive confirmation."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock user confirming deletion
            with patch('typer.confirm', return_value=True):
                result = self.runner.invoke(app, [
                    "collections", "delete", "test-collection"
                ])
                
                assert result.exit_code == 0
                assert "Successfully deleted" in result.stdout
                mock_collection_manager.delete_collection.assert_called_once_with("test-collection")
    
    def test_delete_collection_interactive_cancel(self):
        """Test delete collection with interactive cancellation."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock user canceling deletion
            with patch('typer.confirm', return_value=False):
                result = self.runner.invoke(app, [
                    "collections", "delete", "test-collection"
                ])
                
                assert result.exit_code == 0
                assert "Operation cancelled" in result.stdout
                mock_collection_manager.delete_collection.assert_not_called()
    
    def test_delete_collection_not_found(self):
        """Test delete collection for non-existent collection."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection not found error
            mock_collection_manager.delete_collection.side_effect = CollectionNotFoundError(
                "Collection 'non-existent' not found"
            )
            
            result = self.runner.invoke(app, [
                "collections", "delete", "non-existent", "--confirm"
            ])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_delete_collection_dry_run_mode(self):
        """Test delete collection in dry-run mode."""
        with patch('research_agent_backend.cli.collections._get_global_config') as mock_config:
            mock_config.return_value = {"dry_run": True}
            
            result = self.runner.invoke(app, [
                "collections", "delete", "test-collection", "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "DRY RUN" in result.stdout
            assert "Would delete collection" in result.stdout


class TestRenameCollectionCommand:
    """Test suite for the rename-collection CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_rename_collection_basic_functionality(self):
        """Test basic rename collection command functionality."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collection existence check and rename operation
            mock_collection = Mock()
            mock_collection.name = "old-collection"
            mock_collection_manager.get_collection.side_effect = [
                mock_collection,  # Original collection exists
                CollectionNotFoundError("new-collection not found")  # New name available
            ]
            
            result = self.runner.invoke(app, [
                "collections", "rename", "old-collection", "new-collection"
            ])
            
            assert result.exit_code == 0
            assert "Successfully renamed" in result.stdout
    
    def test_rename_collection_source_not_found(self):
        """Test rename collection when source collection doesn't exist."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock source collection not found
            mock_collection_manager.get_collection.side_effect = CollectionNotFoundError(
                "Collection 'non-existent' not found"
            )
            
            result = self.runner.invoke(app, [
                "collections", "rename", "non-existent", "new-name"
            ])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_rename_collection_target_exists(self):
        """Test rename collection when target name already exists."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock both collections exist
            mock_collection = Mock()
            mock_collection_manager.get_collection.return_value = mock_collection
            
            result = self.runner.invoke(app, [
                "collections", "rename", "old-collection", "existing-collection"
            ])
            
            assert result.exit_code == 1
            assert "already exists" in result.stdout.lower()


class TestMoveDocumentsCommand:
    """Test suite for the move-documents CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_move_documents_basic_functionality(self):
        """Test basic move documents command functionality."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_document_manager = Mock()
            mock_manager.collection_manager = mock_collection_manager
            mock_manager.document_manager = mock_document_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock successful collections existence check
            mock_collection = Mock()
            mock_collection_manager.get_collection.return_value = mock_collection
            
            # Mock document move operation
            mock_document_manager.move_documents.return_value = {
                "moved_count": 5,
                "total_documents": 5,
                "success": True
            }
            
            result = self.runner.invoke(app, [
                "collections", "move-documents", "source-collection", "target-collection",
                "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "Successfully moved" in result.stdout
            assert "5 documents" in result.stdout
    
    def test_move_documents_with_pattern(self):
        """Test move documents with pattern filtering."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_document_manager = Mock()
            mock_manager.collection_manager = mock_collection_manager
            mock_manager.document_manager = mock_document_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collections exist
            mock_collection = Mock()
            mock_collection_manager.get_collection.return_value = mock_collection
            
            # Mock pattern-based move
            mock_document_manager.move_documents.return_value = {
                "moved_count": 3,
                "total_documents": 10,
                "success": True
            }
            
            result = self.runner.invoke(app, [
                "collections", "move-documents", "source", "target",
                "--pattern", "*.pdf",
                "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "3 documents" in result.stdout
            assert "*.pdf" in result.stdout or "pdf" in result.stdout
    
    def test_move_documents_source_not_found(self):
        """Test move documents when source collection doesn't exist."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_manager.collection_manager = mock_collection_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock source collection not found
            mock_collection_manager.get_collection.side_effect = [
                CollectionNotFoundError("Source collection not found"),
                Mock()  # Target would exist
            ]
            
            result = self.runner.invoke(app, [
                "collections", "move-documents", "non-existent", "target", "--confirm"
            ])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_move_documents_interactive_confirmation(self):
        """Test move documents with interactive confirmation."""
        with patch('research_agent_backend.cli.collections.create_chroma_manager') as mock_create_manager:
            mock_manager = Mock(spec=ChromaDBManager)
            mock_collection_manager = Mock(spec=CollectionManager)
            mock_document_manager = Mock()
            mock_manager.collection_manager = mock_collection_manager
            mock_manager.document_manager = mock_document_manager
            mock_create_manager.return_value = mock_manager
            
            # Mock collections exist
            mock_collection = Mock()
            mock_collection_manager.get_collection.return_value = mock_collection
            
            # Mock user confirming move
            with patch('typer.confirm', return_value=True):
                mock_document_manager.move_documents.return_value = {
                    "moved_count": 2,
                    "total_documents": 2,
                    "success": True
                }
                
                result = self.runner.invoke(app, [
                    "collections", "move-documents", "source", "target"
                ])
                
                assert result.exit_code == 0
                assert "Successfully moved" in result.stdout 