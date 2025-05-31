"""
Tests for the Research Agent Projects CLI commands.

This module tests the project-specific knowledge management commands including
project-collection linking, default collection management, and project context detection.

Implements TDD testing for FR-KB-005: Project and collection management (Task 10).
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
    CollectionNotFoundError,
    VectorStoreError
)
from research_agent_backend.models.metadata_schema import CollectionType


class TestLinkCollectionCommand:
    """Test suite for the link-collection CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_link_collection_basic_functionality(self):
        """Test basic link collection to project functionality."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            # Mock successful collection linking
            mock_manager.link_collection.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "Successfully linked collection" in result.stdout
            assert "test-collection" in result.stdout
            assert "test-project" in result.stdout
            
            # Verify manager was called correctly (with description=None)
            mock_manager.link_collection.assert_called_once_with(
                project_name="test-project",
                collection_name="test-collection",
                description=None
            )
    
    def test_link_collection_with_description(self):
        """Test link collection with custom description."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.link_collection.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "test-project", "test-collection",
                "--description", "Custom link description"
            ])
            
            assert result.exit_code == 0
            mock_manager.link_collection.assert_called_once_with(
                project_name="test-project",
                collection_name="test-collection",
                description="Custom link description"
            )
    
    def test_link_collection_project_not_found(self):
        """Test link collection when project doesn't exist."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            # Mock project not found error
            from research_agent_backend.exceptions.project_exceptions import ProjectNotFoundError
            mock_manager.link_collection.side_effect = ProjectNotFoundError(
                "nonexistent-project"
            )
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "nonexistent-project", "test-collection"
            ])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_link_collection_collection_not_found(self):
        """Test link collection when collection doesn't exist."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            mock_manager.link_collection.side_effect = CollectionNotFoundError(
                "Collection 'nonexistent-collection' not found"
            )
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "test-project", "nonexistent-collection"
            ])
            
            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()
    
    def test_link_collection_already_linked(self):
        """Test link collection when already linked to project."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            from research_agent_backend.exceptions.project_exceptions import CollectionAlreadyLinkedError
            mock_manager.link_collection.side_effect = CollectionAlreadyLinkedError(
                "test-collection", "test-project"
            )
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 1
            assert "already linked" in result.stdout.lower()
    
    def test_link_collection_dry_run_mode(self):
        """Test link collection in dry-run mode."""
        with patch('research_agent_backend.cli.projects._get_dry_run_status') as mock_dry_run:
            mock_dry_run.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "link-collection", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "DRY RUN" in result.stdout
            assert "Would link collection" in result.stdout


class TestUnlinkCollectionCommand:
    """Test suite for the unlink-collection CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_unlink_collection_basic_functionality(self):
        """Test basic unlink collection from project functionality."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.unlink_collection.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "unlink-collection", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "Successfully unlinked collection" in result.stdout
            assert "test-collection" in result.stdout
            assert "test-project" in result.stdout
            
            mock_manager.unlink_collection.assert_called_once_with(
                project_name="test-project",
                collection_name="test-collection"
            )
    
    def test_unlink_collection_with_confirmation(self):
        """Test unlink collection with confirmation prompt."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.unlink_collection.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "unlink-collection", "test-project", "test-collection",
                "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "Successfully unlinked collection" in result.stdout
    
    def test_unlink_collection_not_linked(self):
        """Test unlink collection when not linked to project."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            from research_agent_backend.exceptions.project_exceptions import CollectionNotLinkedError
            mock_manager.unlink_collection.side_effect = CollectionNotLinkedError(
                "test-collection", "test-project"
            )
            
            result = self.runner.invoke(app, [
                "projects", "unlink-collection", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 1
            assert "not linked" in result.stdout.lower()


class TestSetDefaultCollectionsCommand:
    """Test suite for the set-default-collections CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_set_default_collections_single_collection(self):
        """Test setting a single default collection for project."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.set_default_collections.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "set-default-collections", "test-project", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "Successfully set default collections" in result.stdout
            assert "test-collection" in result.stdout
            
            mock_manager.set_default_collections.assert_called_once_with(
                project_name="test-project",
                collection_names=["test-collection"],
                append=False
            )
    
    def test_set_default_collections_multiple_collections(self):
        """Test setting multiple default collections for project."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.set_default_collections.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "set-default-collections", "test-project", 
                "collection1,collection2,collection3"
            ])
            
            assert result.exit_code == 0
            mock_manager.set_default_collections.assert_called_once_with(
                project_name="test-project",
                collection_names=["collection1", "collection2", "collection3"],
                append=False
            )
    
    def test_set_default_collections_append_mode(self):
        """Test setting default collections in append mode."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.set_default_collections.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "set-default-collections", "test-project", "new-collection",
                "--append"
            ])
            
            assert result.exit_code == 0
            mock_manager.set_default_collections.assert_called_once_with(
                project_name="test-project",
                collection_names=["new-collection"],
                append=True
            )
    
    def test_set_default_collections_clear_existing(self):
        """Test clearing existing default collections."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.clear_default_collections.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "set-default-collections", "test-project", "",
                "--clear"
            ])
            
            assert result.exit_code == 0
            assert "Cleared default collections" in result.stdout
            mock_manager.clear_default_collections.assert_called_once_with(
                "test-project"
            )


class TestListProjectCollectionsCommand:
    """Test suite for the list-project-collections CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_list_project_collections_basic_functionality(self):
        """Test basic list project collections functionality."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            # Mock project collections data
            mock_project_info = Mock()
            mock_project_info.linked_collections = [
                Mock(collection_name="collection1", description="First collection", is_default=True, document_count=42),
                Mock(collection_name="collection2", description="Second collection", is_default=False, document_count=24)
            ]
            mock_manager.get_project_collections.return_value = mock_project_info
            
            result = self.runner.invoke(app, [
                "projects", "list-project-collections", "test-project"
            ])
            
            assert result.exit_code == 0
            assert "collection1" in result.stdout
            assert "collection2" in result.stdout
            assert "First collection" in result.stdout
            assert "Second collection" in result.stdout
            
            mock_manager.get_project_collections.assert_called_once_with(
                "test-project"
            )
    
    def test_list_project_collections_with_stats(self):
        """Test list project collections with statistics."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            mock_project_info = Mock()
            mock_project_info.linked_collections = [
                Mock(
                    collection_name="collection1", 
                    description="First collection", 
                    is_default=True,
                    document_count=42,
                    last_updated="2024-01-01T00:00:00"
                )
            ]
            mock_manager.get_project_collections.return_value = mock_project_info
            
            result = self.runner.invoke(app, [
                "projects", "list-project-collections", "test-project",
                "--stats"
            ])
            
            assert result.exit_code == 0
            assert "42" in result.stdout  # Document count
            assert "Documents" in result.stdout  # Stats header
    
    def test_list_project_collections_defaults_only(self):
        """Test list only default collections for project."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            mock_project_info = Mock()
            mock_project_info.default_collections = [
                Mock(collection_name="default-collection", description="Default collection")
            ]
            mock_manager.get_project_collections.return_value = mock_project_info
            
            result = self.runner.invoke(app, [
                "projects", "list-project-collections", "test-project",
                "--defaults-only"
            ])
            
            assert result.exit_code == 0
            assert "default-collection" in result.stdout
    
    def test_list_project_collections_empty_project(self):
        """Test list collections for project with no linked collections."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            mock_project_info = Mock()
            mock_project_info.linked_collections = []
            mock_manager.get_project_collections.return_value = mock_project_info
            
            result = self.runner.invoke(app, [
                "projects", "list-project-collections", "test-project"
            ])
            
            assert result.exit_code == 0
            assert "No collections linked" in result.stdout


class TestProjectContextDetection:
    """Test suite for project context detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_detect_project_from_path(self):
        """Test automatic project detection from file path."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            # Mock project detection from path
            mock_manager.detect_project_from_path.return_value = "detected-project"
            
            result = self.runner.invoke(app, [
                "projects", "detect-context",
                "--path", "/users/test/projects/detected-project/docs/file.md"
            ])
            
            assert result.exit_code == 0
            assert "detected-project" in result.stdout
            assert "Detected project" in result.stdout
            
            mock_manager.detect_project_from_path.assert_called_once_with(
                "/users/test/projects/detected-project/docs/file.md"
            )
    
    def test_detect_project_no_match(self):
        """Test project detection when no project matches path."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            mock_manager.detect_project_from_path.return_value = None
            
            result = self.runner.invoke(app, [
                "projects", "detect-context",
                "--path", "/users/test/random/path/file.md"
            ])
            
            assert result.exit_code == 0
            assert "No project detected" in result.stdout
    
    def test_set_explicit_project_context(self):
        """Test setting explicit project context."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.set_active_project.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "set-context", "explicit-project"
            ])
            
            assert result.exit_code == 0
            assert "Set project context" in result.stdout
            assert "explicit-project" in result.stdout
            
            mock_manager.set_active_project.assert_called_once_with("explicit-project")


class TestProjectMetadataStorage:
    """Test suite for project metadata storage and retrieval."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_create_project_metadata(self):
        """Test creating project with metadata storage."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.create_project.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "create", "new-project",
                "--description", "Test project description",
                "--tags", "research,ai,test"
            ])
            
            assert result.exit_code == 0
            assert "Successfully created project" in result.stdout
            assert "new-project" in result.stdout
            
            mock_manager.create_project.assert_called_once_with(
                name="new-project",
                description="Test project description",
                tags=["research", "ai", "test"]
            )
    
    def test_update_project_metadata(self):
        """Test updating project metadata."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            mock_manager.update_project_metadata.return_value = True
            
            result = self.runner.invoke(app, [
                "projects", "update", "existing-project",
                "--description", "Updated description",
                "--add-tags", "new-tag"
            ])
            
            assert result.exit_code == 0
            assert "Successfully updated project" in result.stdout
            
            mock_manager.update_project_metadata.assert_called_once()
    
    def test_get_project_metadata(self):
        """Test retrieving project metadata."""
        with patch('research_agent_backend.cli.projects.create_project_manager') as mock_create_manager:
            mock_manager = Mock()
            mock_create_manager.return_value = mock_manager
            
            # Mock project metadata
            mock_project = Mock()
            mock_project.name = "test-project"
            mock_project.description = "Test description"
            mock_project.tags = ["tag1", "tag2"]
            mock_project.created_at = "2024-01-01T00:00:00"
            mock_project.linked_collections_count = 3
            mock_project.total_documents = 157
            mock_project.status.value = "active"
            
            mock_manager.get_project_metadata.return_value = mock_project
            
            result = self.runner.invoke(app, [
                "projects", "info", "test-project"
            ])
            
            assert result.exit_code == 0
            assert "test-project" in result.stdout
            assert "Test description" in result.stdout
            assert "tag1" in result.stdout
            assert "157" in result.stdout  # Document count 