"""
Tests for the Research Agent Knowledge Base CLI commands.

This module tests the knowledge base management commands including
document ingestion, listing, and removal functionality.

Implements TDD testing for FR-KB-002: Document ingestion and management.
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
from research_agent_backend.models.metadata_schema import DocumentMetadata


class TestIngestFolderCommand:
    """Test suite for the ingest-folder CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = None
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_folder_structure(self):
        """Create a temporary folder structure with test documents."""
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        # Create test markdown files
        (temp_path / "doc1.md").write_text("# Document 1\nThis is the first document.")
        (temp_path / "doc2.md").write_text("# Document 2\nThis is the second document.")
        (temp_path / "readme.txt").write_text("This is a text file.")
        
        # Create subdirectory with more files
        subdir = temp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc3.md").write_text("# Document 3\nThis is in a subdirectory.")
        (subdir / "doc4.txt").write_text("Another text file in subdirectory.")
        
        return temp_path
    
    def test_ingest_folder_basic_functionality(self):
        """Test basic ingest-folder command functionality."""
        # This test will fail because the command is not implemented yet
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 2
            mock_result.total_documents = 2
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection",
                "--pattern", "*.md"
            ])
            
            # Should succeed and show progress
            assert result.exit_code == 0
            assert "Successfully ingested" in result.stdout
            assert "2 documents" in result.stdout
            assert "test-collection" in result.stdout
            
            # Should have called document insertion manager
            mock_create_manager.assert_called_once()
            mock_manager.insert_batch.assert_called_once()
    
    def test_ingest_folder_recursive_mode(self):
        """Test ingest-folder with recursive mode enabled."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 3  # Including subdirectory
            mock_result.total_documents = 3
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection",
                "--pattern", "*.md",
                "--recursive"
            ])
            
            assert result.exit_code == 0
            assert "3 documents" in result.stdout
            
            # Should have found files in subdirectories
            call_kwargs = mock_manager.insert_batch.call_args[1]  # keyword arguments
            documents = call_kwargs['documents']
            assert len(documents) == 3
    
    def test_ingest_folder_non_recursive_mode(self):
        """Test ingest-folder with recursive mode disabled."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 2  # Only root level
            mock_result.total_documents = 2
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection",
                "--pattern", "*.md",
                "--no-recursive"
            ])
            
            assert result.exit_code == 0
            assert "2 documents" in result.stdout
            
            # Should only find files in root directory
            call_kwargs = mock_manager.insert_batch.call_args[1]  # keyword arguments
            documents = call_kwargs['documents']
            assert len(documents) == 2
    
    def test_ingest_folder_pattern_filtering(self):
        """Test ingest-folder with different file patterns."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 1  # Only .txt files
            mock_result.total_documents = 1
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection",
                "--pattern", "*.txt",
                "--no-recursive"
            ])
            
            assert result.exit_code == 0
            assert "1 documents" in result.stdout
    
    def test_ingest_folder_progress_tracking(self):
        """Test that ingest-folder shows progress during processing."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            
            # Simulate progress callback
            def mock_insert_batch(documents, collection_name, progress_callback=None):
                if progress_callback:
                    progress_callback(1, 2, 1)  # processed, total, batch
                    progress_callback(2, 2, 2)
                
                result = BatchInsertionResult()
                result.success = True
                result.successful_insertions = 2
                result.total_documents = 2
                return result
            
            mock_manager.insert_batch.side_effect = mock_insert_batch
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "Processing" in result.stdout or "Progress" in result.stdout
    
    def test_ingest_folder_error_handling(self):
        """Test ingest-folder error handling for various failure scenarios."""
        # Test non-existent folder
        result = self.runner.invoke(app, [
            "kb", "ingest-folder", "/non/existent/path",
            "--collection", "test-collection"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_ingest_folder_empty_folder(self):
        """Test ingest-folder with empty folder."""
        self.temp_dir = tempfile.mkdtemp()
        
        result = self.runner.invoke(app, [
            "kb", "ingest-folder", self.temp_dir,
            "--collection", "test-collection"
        ])
        
        assert result.exit_code == 0
        assert "No matching files found" in result.stdout or "0 documents" in result.stdout
    
    def test_ingest_folder_dry_run_mode(self):
        """Test ingest-folder with dry-run mode."""
        folder_path = self.create_test_folder_structure()
        
        result = self.runner.invoke(app, [
            "--dry-run",
            "kb", "ingest-folder", str(folder_path),
            "--collection", "test-collection"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "Would ingest folder" in result.stdout
    
    def test_ingest_folder_metadata_extraction(self):
        """Test that ingest-folder properly extracts metadata from files."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 2
            mock_result.total_documents = 2
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection",
                "--pattern", "*.md"
            ])
            
            assert result.exit_code == 0
            
            # Check that documents were passed with proper metadata
            call_kwargs = mock_manager.insert_batch.call_args[1]  # keyword arguments
            documents = call_kwargs['documents']
            for doc in documents:
                assert "text" in doc
                assert "metadata" in doc
                assert isinstance(doc["metadata"], DocumentMetadata)
                assert doc["metadata"].source_path
                assert doc["metadata"].title
    
    def test_ingest_folder_batch_processing(self):
        """Test that ingest-folder uses batch processing for efficiency."""
        folder_path = self.create_test_folder_structure()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = BatchInsertionResult()
            mock_result.success = True
            mock_result.successful_insertions = 2
            mock_result.total_documents = 2
            mock_manager.insert_batch.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "ingest-folder", str(folder_path),
                "--collection", "test-collection"
            ])
            
            assert result.exit_code == 0
            
            # Should call insert_batch, not individual insert_document calls
            mock_manager.insert_batch.assert_called_once()
            mock_manager.insert_document.assert_not_called()


class TestAddDocumentCommand:
    """Test suite for the add-document CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_file = None
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_file and Path(self.temp_file).exists():
            Path(self.temp_file).unlink()
    
    def create_test_document(self, content="# Test Document\nThis is a test document."):
        """Create a temporary test document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(content)
            self.temp_file = f.name
        return self.temp_file
    
    def test_add_document_basic_functionality(self):
        """Test basic add-document command functionality."""
        doc_path = self.create_test_document()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = InsertionResult()
            mock_result.success = True
            mock_result.document_id = "test-doc-123"
            mock_result.chunk_count = 1
            mock_manager.insert_document.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "add-document", doc_path,
                "--collection", "test-collection"
            ])
            
            assert result.exit_code == 0
            assert "Successfully added document" in result.stdout
            assert "test-doc-123" in result.stdout
            assert "test-collection" in result.stdout
    
    def test_add_document_force_mode(self):
        """Test add-document with force mode enabled."""
        doc_path = self.create_test_document()
        
        with patch('research_agent_backend.cli.knowledge_base.create_document_insertion_manager') as mock_create_manager:
            mock_manager = Mock(spec=DocumentInsertionManager)
            mock_result = InsertionResult()
            mock_result.success = True
            mock_result.document_id = "test-doc-123"
            mock_manager.insert_document.return_value = mock_result
            mock_create_manager.return_value = mock_manager
            
            result = self.runner.invoke(app, [
                "kb", "add-document", doc_path,
                "--collection", "test-collection",
                "--force"
            ])
            
            assert result.exit_code == 0
            assert "Successfully added document" in result.stdout
    
    def test_add_document_file_not_found(self):
        """Test add-document with non-existent file."""
        result = self.runner.invoke(app, [
            "kb", "add-document", "/non/existent/file.md",
            "--collection", "test-collection"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()
    
    def test_add_document_dry_run_mode(self):
        """Test add-document with dry-run mode."""
        doc_path = self.create_test_document()
        
        result = self.runner.invoke(app, [
            "--dry-run",
            "kb", "add-document", doc_path,
            "--collection", "test-collection"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "Would add document" in result.stdout


class TestListDocumentsCommand:
    """Test suite for the list-documents CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_list_documents_basic_functionality(self):
        """Test basic list-documents command functionality."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_documents.return_value = {
                'ids': ['doc1', 'doc2'],
                'metadatas': [
                    {'title': 'Document 1', 'source_path': '/path/doc1.md'},
                    {'title': 'Document 2', 'source_path': '/path/doc2.md'}
                ]
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "list-documents"
            ])
            
            assert result.exit_code == 0
            assert "Document 1" in result.stdout
            assert "Document 2" in result.stdout
            assert "doc1" in result.stdout
            assert "doc2" in result.stdout
    
    def test_list_documents_with_collection_filter(self):
        """Test list-documents with collection filter."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_documents.return_value = {
                'ids': ['doc1'],
                'metadatas': [
                    {'title': 'Document 1', 'source_path': '/path/doc1.md'}
                ]
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "list-documents",
                "--collection", "specific-collection"
            ])
            
            assert result.exit_code == 0
            assert "Document 1" in result.stdout
    
    def test_list_documents_with_limit(self):
        """Test list-documents with limit parameter."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_documents.return_value = {
                'ids': ['doc1', 'doc2'],
                'metadatas': [
                    {'title': 'Document 1', 'source_path': '/path/doc1.md'},
                    {'title': 'Document 2', 'source_path': '/path/doc2.md'}
                ]
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "list-documents",
                "--limit", "10"
            ])
            
            assert result.exit_code == 0
            # Should call get_documents with limit parameter
            mock_chroma.get_documents.assert_called_with(
                collection_name=None,
                limit=10,
                include=['documents', 'metadatas']
            )
    
    def test_list_documents_empty_result(self):
        """Test list-documents with no documents found."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.get_documents.return_value = {
                'ids': [],
                'metadatas': []
            }
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "list-documents"
            ])
            
            assert result.exit_code == 0
            assert "No documents found" in result.stdout or "0 documents" in result.stdout


class TestRemoveDocumentCommand:
    """Test suite for the remove-document CLI command."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
    
    def test_remove_document_basic_functionality(self):
        """Test basic remove-document command functionality."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.delete_documents.return_value = Mock(success_count=1)
            mock_create_chroma.return_value = mock_chroma
            
            with patch('builtins.input', return_value='y'):  # Confirm deletion
                result = self.runner.invoke(app, [
                    "kb", "remove-document", "doc-123"
                ])
            
            assert result.exit_code == 0
            assert "Successfully removed document" in result.stdout
            assert "doc-123" in result.stdout
    
    def test_remove_document_with_confirm_flag(self):
        """Test remove-document with --confirm flag to skip prompt."""
        with patch('research_agent_backend.cli.knowledge_base.create_chroma_manager') as mock_create_chroma:
            mock_chroma = Mock()
            mock_chroma.delete_documents.return_value = Mock(success_count=1)
            mock_create_chroma.return_value = mock_chroma
            
            result = self.runner.invoke(app, [
                "kb", "remove-document", "doc-123",
                "--confirm"
            ])
            
            assert result.exit_code == 0
            assert "Successfully removed document" in result.stdout
    
    def test_remove_document_user_cancellation(self):
        """Test remove-document when user cancels the operation."""
        with patch('builtins.input', return_value='n'):  # Cancel deletion
            result = self.runner.invoke(app, [
                "kb", "remove-document", "doc-123"
            ])
        
        assert result.exit_code == 0
        assert "Operation cancelled" in result.stdout or "Cancelled" in result.stdout
    
    def test_remove_document_dry_run_mode(self):
        """Test remove-document with dry-run mode."""
        result = self.runner.invoke(app, [
            "--dry-run",
            "kb", "remove-document", "doc-123"
        ])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "Would remove document" in result.stdout 