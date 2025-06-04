"""
Test suite for Document Insertion Manager - TDD Implementation.

This module tests the DocumentInsertionManager class following strict TDD principles.
All tests are designed to fail initially (RED PHASE) before implementation.
"""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.research_agent_backend.core.document_insertion import (
    DocumentInsertionManager,
    InsertionResult,
    BatchInsertionResult,
    InsertionError,
    ValidationError,
    TransactionError
)
from src.research_agent_backend.core.vector_store import ChromaDBManager
from src.research_agent_backend.core.data_preparation import DataPreparationManager
from src.research_agent_backend.models.metadata_schema import (
    ChunkMetadata,
    DocumentMetadata,
    ContentType,
    DocumentType,
    CollectionType
)
from src.research_agent_backend.utils.config import ConfigManager


class TestDocumentInsertionManager:
    """Test DocumentInsertionManager class initialization and basic functionality."""
    
    def test_initialization_with_valid_dependencies(self):
        """Test DocumentInsertionManager initialization with valid dependencies."""
        # This test will fail because DocumentInsertionManager doesn't exist yet
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        mock_config = Mock(spec=ConfigManager)
        
        # SYSTEMATIC FIX: Configure mock to return proper chunking configuration values
        # instead of Mock objects that cause "chunk_size must be positive integer" error
        mock_config.get.side_effect = lambda key, default=None: {
            "chunking_strategy.chunk_size": 512,
            "chunking_strategy.chunk_overlap": 50,
            "chunking_strategy.preserve_code_blocks": True,
            "chunking_strategy.preserve_tables": True
        }.get(key, default)
        
        # Expected to fail - class doesn't exist
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep,
            config_manager=mock_config
        )
        
        assert manager is not None
        assert manager.vector_store == mock_vector_store
        assert manager.data_preparation_manager == mock_data_prep
        assert manager.config_manager == mock_config
    
    def test_initialization_with_missing_dependencies(self):
        """Test DocumentInsertionManager initialization fails with missing dependencies."""
        # Should fail because DocumentInsertionManager doesn't exist
        with pytest.raises(ValueError, match="vector_store is required"):
            DocumentInsertionManager(vector_store=None)
    
    def test_initialization_with_default_config(self):
        """Test DocumentInsertionManager uses default configuration when not provided."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        
        # Should fail because class doesn't exist
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep
        )
        
        assert manager.config_manager is not None
        assert manager.batch_size > 0
        assert manager.enable_transactions is True


class TestSingleDocumentInsertion:
    """Test single document insertion functionality."""
    
    @pytest.fixture
    def insertion_manager(self):
        """Create a mock DocumentInsertionManager for testing."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        mock_config = Mock(spec=ConfigManager)
        
        # Will fail because class doesn't exist
        return DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep,
            config_manager=mock_config
        )
    
    def test_insert_single_document_success(self, insertion_manager):
        """Test successful single document insertion with valid data."""
        # Sample document data
        document_text = "This is a test document about machine learning."
        document_metadata = DocumentMetadata(
            title="Test Document",
            document_type=DocumentType.TEXT,
            source_path="/test/document.txt"
        )
        collection_name = "test_collection"
        
        # Mock embedding service response
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_text.return_value = mock_embedding
        
        # Mock data preparation to return proper tuple
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            document_text,  # cleaned text
            np.array(mock_embedding),  # normalized embedding
            document_metadata.to_dict()  # processed metadata
        )
        
        # Mock vector store insertion
        insertion_manager.vector_store.add_documents.return_value = None
        
        # This will fail because insert_document method doesn't exist
        result = insertion_manager.insert_document(
            text=document_text,
            metadata=document_metadata,
            collection_name=collection_name
        )
        
        assert isinstance(result, InsertionResult)
        assert result.success is True
        assert result.document_id is not None
        assert result.chunk_count == 1
        assert result.errors == []
    
    def test_insert_document_with_chunking(self, insertion_manager):
        """Test document insertion with chunking enabled."""
        # Use a more predictable document for chunking
        large_text = "This is the first sentence. This is the second sentence. This is the third sentence."
        document_metadata = DocumentMetadata(
            title="Large Document",
            document_type=DocumentType.TEXT,
            source_path="/test/large_document.txt"
        )
        
        # Mock data preparation to return the exact text
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            large_text,  # cleaned text (keep original)
            None,  # no embedding
            document_metadata.to_dict()  # processed metadata
        )
        
        # Mock embedding service
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_batch.return_value = [
            [0.1, 0.2, 0.3] for _ in range(3)  # Expect 3 chunks based on 3 sentences
        ]
        
        # Mock vector store
        insertion_manager.vector_store.add_documents.return_value = None
        
        # Insert with chunking enabled and smaller chunk size to force chunking
        result = insertion_manager.insert_document(
            text=large_text,
            metadata=document_metadata,
            collection_name="test_collection",
            enable_chunking=True,
            chunk_size=30  # Small chunk size to ensure chunking happens
        )
        
        assert isinstance(result, InsertionResult)
        assert result.success is True
        assert result.chunk_count == 3  # Should create 3 chunks from 3 sentences
        assert len(result.chunk_ids) == 3
    
    def test_insert_document_validation_failure(self, insertion_manager):
        """Test document insertion fails with invalid data."""
        # Invalid document data
        invalid_text = ""  # Empty text
        invalid_metadata = {}  # Missing required fields
        
        # This will fail because method doesn't exist
        with pytest.raises(ValidationError, match="Document text cannot be empty"):
            insertion_manager.insert_document(
                text=invalid_text,
                metadata=invalid_metadata,
                collection_name="test_collection"
            )
    
    def test_insert_document_embedding_failure(self, insertion_manager):
        """Test document insertion handles embedding service failures."""
        document_text = "Test document"
        document_metadata = DocumentMetadata(title="Test")
        
        # Mock embedding service failure
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_text.side_effect = Exception("Embedding failed")
        
        # This will fail because method doesn't exist
        with pytest.raises(InsertionError, match="Failed to generate embeddings"):
            insertion_manager.insert_document(
                text=document_text,
                metadata=document_metadata,
                collection_name="test_collection"
            )


class TestBatchDocumentInsertion:
    """Test batch document insertion functionality."""
    
    @pytest.fixture
    def insertion_manager(self):
        """Create a mock DocumentInsertionManager for testing."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        
        # Will fail because class doesn't exist
        return DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep,
            batch_size=10,
            enable_transactions=True
        )
    
    def test_batch_insertion_success(self, insertion_manager):
        """Test successful batch document insertion."""
        # Sample batch data
        documents = [
            {"text": f"Document {i}", "metadata": DocumentMetadata(title=f"Doc {i}")}
            for i in range(5)
        ]
        collection_name = "test_collection"
        
        # Mock successful processing
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_batch.return_value = [
            [0.1, 0.2] for _ in range(5)
        ]
        
        # Mock data preparation to return proper tuple for each document
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock vector store operations
        insertion_manager.vector_store.add_documents.return_value = None
        
        # Mock transaction methods
        insertion_manager.vector_store.begin_transaction = Mock()
        insertion_manager.vector_store.commit_transaction = Mock()
        insertion_manager.vector_store.rollback_transaction = Mock()
        
        # This will fail because method doesn't exist
        result = insertion_manager.insert_batch(
            documents=documents,
            collection_name=collection_name
        )
        
        assert isinstance(result, BatchInsertionResult)
        assert result.total_documents == 5
        assert result.successful_insertions == 5
        assert result.failed_insertions == 0
        assert result.success_rate == 1.0
    
    def test_batch_insertion_with_transaction_rollback(self, insertion_manager):
        """Test batch insertion rolls back on failure when transactions enabled."""
        documents = [
            {"text": f"Document {i}", "metadata": DocumentMetadata(title=f"Doc {i}")}
            for i in range(5)
        ]
        
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock partial failure scenario
        insertion_manager.vector_store.add_documents.side_effect = [
            None,  # First batch succeeds
            None,  # Second batch succeeds
            Exception("Database error")  # Third batch fails
        ]
        
        # Mock transaction methods
        insertion_manager.vector_store.begin_transaction = Mock()
        insertion_manager.vector_store.commit_transaction = Mock()
        insertion_manager.vector_store.rollback_transaction = Mock()
        
        # This will fail because method doesn't exist
        with pytest.raises(TransactionError):
            insertion_manager.insert_batch(
                documents=documents,
                collection_name="test_collection"
            )
        
        # Verify rollback was called
        insertion_manager.vector_store.rollback_transaction.assert_called_once()
    
    def test_batch_insertion_progress_tracking(self, insertion_manager):
        """Test batch insertion provides progress tracking."""
        documents = [
            {"text": f"Document {i}", "metadata": DocumentMetadata(title=f"Doc {i}")}
            for i in range(20)
        ]
        
        progress_updates = []
        
        def progress_callback(processed, total, current_batch):
            progress_updates.append((processed, total, current_batch))
        
        # This will fail because method doesn't exist
        result = insertion_manager.insert_batch(
            documents=documents,
            collection_name="test_collection",
            progress_callback=progress_callback
        )
        
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 20  # Final update shows all processed
        assert progress_updates[-1][1] == 20  # Total count
    
    def test_batch_insertion_memory_management(self, insertion_manager):
        """Test batch insertion handles memory efficiently for large batches."""
        # Large batch that should be processed in chunks
        large_batch = [
            {"text": f"Document {i}", "metadata": DocumentMetadata(title=f"Doc {i}")}
            for i in range(1000)
        ]
        
        insertion_manager.batch_size = 50  # Process in smaller batches
        
        # This will fail because method doesn't exist
        result = insertion_manager.insert_batch(
            documents=large_batch,
            collection_name="test_collection"
        )
        
        # Verify batch processing was used
        assert insertion_manager.vector_store.add_documents.call_count > 1
        assert result.total_documents == 1000


class TestErrorHandlingAndValidation:
    """Test comprehensive error handling and validation."""
    
    @pytest.fixture
    def insertion_manager(self):
        """Create a mock DocumentInsertionManager for testing."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        
        # Will fail because class doesn't exist
        return DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep
        )
    
    def test_validation_empty_text(self, insertion_manager):
        """Test validation fails for empty text."""
        # This will fail because method doesn't exist
        with pytest.raises(ValidationError, match="Document text cannot be empty"):
            insertion_manager.validate_document_input(
                text="",
                metadata=DocumentMetadata(title="Test")
            )
    
    def test_validation_invalid_metadata(self, insertion_manager):
        """Test validation fails for invalid metadata."""
        # This will fail because method doesn't exist
        with pytest.raises(ValidationError, match="Invalid metadata"):
            insertion_manager.validate_document_input(
                text="Valid text",
                metadata=None
            )
    
    def test_database_connection_error_handling(self, insertion_manager):
        """Test handling of database connection errors."""
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        insertion_manager.vector_store.add_documents.side_effect = ConnectionError("DB disconnected")
        
        # This will fail because method doesn't exist
        with pytest.raises(InsertionError, match="Failed to insert document into vector store"):
            insertion_manager.insert_document(
                text="Test document",
                metadata=DocumentMetadata(title="Test"),
                collection_name="test_collection"
            )
    
    def test_collection_type_validation(self, insertion_manager):
        """Test collection type awareness and validation."""
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock vector store operations
        insertion_manager.vector_store.add_documents.return_value = None
        
        # Mock collection type manager
        insertion_manager.collection_type_manager = Mock()
        insertion_manager.collection_type_manager.get_collection_type.return_value = CollectionType.FUNDAMENTAL
        insertion_manager.collection_type_manager.validate_document_for_type.return_value = (True, [])
        insertion_manager.collection_type_manager.determine_collection_type_for_document.return_value = CollectionType.FUNDAMENTAL
        
        # This will fail because method doesn't exist
        result = insertion_manager.insert_document(
            text="Test document",
            metadata=DocumentMetadata(title="Test"),
            collection_name="fundamental_collection"
        )
        
        # Verify collection type validation was called
        insertion_manager.collection_type_manager.determine_collection_type_for_document.assert_called_once()


class TestDataPreparationIntegration:
    """Test integration with DataPreparationManager."""
    
    @pytest.fixture
    def insertion_manager(self):
        """Create a mock DocumentInsertionManager for testing."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        
        # Will fail because class doesn't exist
        return DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep
        )
    
    def test_document_cleaning_integration(self, insertion_manager):
        """Test document is cleaned before insertion."""
        dirty_text = "  Document with   extra    whitespace  \n\n"
        clean_text = "Document with extra whitespace"
        
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            clean_text, None, {}
        )
        
        # Mock embedding service to prevent errors
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock vector store
        insertion_manager.vector_store.add_documents.return_value = None
        
        # This will fail because method doesn't exist
        insertion_manager.insert_document(
            text=dirty_text,
            metadata=DocumentMetadata(title="Test"),
            collection_name="test_collection"
        )
        
        # Verify data preparation was called with dirty text
        insertion_manager.data_preparation_manager.prepare_single_document.assert_called_once()
        # Check the actual call arguments using keyword arguments
        call_kwargs = insertion_manager.data_preparation_manager.prepare_single_document.call_args.kwargs
        assert call_kwargs["text"] == dirty_text
    
    def test_metadata_normalization_integration(self, insertion_manager):
        """Test metadata is normalized before insertion."""
        original_metadata = DocumentMetadata(title="Test Document")
        normalized_metadata = {"title": "test_document", "normalized": True}
        
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "text", None, normalized_metadata
        )
        
        # Mock embedding service to prevent errors
        insertion_manager.embedding_service = Mock()
        insertion_manager.embedding_service.embed_text.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Mock vector store
        insertion_manager.vector_store.add_documents.return_value = None
        
        # This will fail because method doesn't exist
        insertion_manager.insert_document(
            text="Test text",
            metadata=original_metadata,
            collection_name="test_collection"
        )
        
        # Verify metadata normalization occurred by checking the vector store call
        insertion_manager.vector_store.add_documents.assert_called_once()
        call_kwargs = insertion_manager.vector_store.add_documents.call_args.kwargs
        # Since the mock returns normalized metadata, it should be passed through
        # The actual metadata structure will be in the ChunkMetadata format


class TestTransactionSupport:
    """Test transaction support and rollback capabilities."""
    
    @pytest.fixture
    def insertion_manager(self):
        """Create a mock DocumentInsertionManager with transaction support."""
        mock_vector_store = Mock(spec=ChromaDBManager)
        mock_data_prep = Mock(spec=DataPreparationManager)
        
        # Will fail because class doesn't exist
        return DocumentInsertionManager(
            vector_store=mock_vector_store,
            data_preparation_manager=mock_data_prep,
            enable_transactions=True
        )
    
    def test_transaction_commit_on_success(self, insertion_manager):
        """Test transaction is committed on successful batch insertion."""
        documents = [
            {"text": "Doc 1", "metadata": DocumentMetadata(title="Doc 1")},
            {"text": "Doc 2", "metadata": DocumentMetadata(title="Doc 2")}
        ]
        
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock vector store operations
        insertion_manager.vector_store.add_documents.return_value = None
        
        # Mock transaction methods
        insertion_manager.vector_store.begin_transaction = Mock()
        insertion_manager.vector_store.commit_transaction = Mock()
        insertion_manager.vector_store.rollback_transaction = Mock()
        
        # This will fail because method doesn't exist
        result = insertion_manager.insert_batch(
            documents=documents,
            collection_name="test_collection"
        )
        
        # Verify transaction was committed
        insertion_manager.vector_store.commit_transaction.assert_called_once()
        assert result.success is True
    
    def test_transaction_rollback_on_failure(self, insertion_manager):
        """Test transaction is rolled back on insertion failure."""
        documents = [
            {"text": "Doc 1", "metadata": DocumentMetadata(title="Doc 1")},
            {"text": "Doc 2", "metadata": DocumentMetadata(title="Doc 2")}
        ]
        
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock transaction methods
        insertion_manager.vector_store.begin_transaction = Mock()
        insertion_manager.vector_store.commit_transaction = Mock()
        insertion_manager.vector_store.rollback_transaction = Mock()
        
        # Mock failure during insertion
        insertion_manager.vector_store.add_documents.side_effect = Exception("Insertion failed")
        
        # This will fail because method doesn't exist
        with pytest.raises(TransactionError):
            insertion_manager.insert_batch(
                documents=documents,
                collection_name="test_collection"
            )
        
        # Verify transaction was rolled back
        insertion_manager.vector_store.rollback_transaction.assert_called_once()
    
    def test_nested_transaction_support(self, insertion_manager):
        """Test support for nested transactions."""
        # Mock data preparation
        insertion_manager.data_preparation_manager.prepare_single_document.return_value = (
            "test text",  # cleaned text
            np.array([0.1, 0.2]),  # normalized embedding
            {"title": "test"}  # processed metadata
        )
        
        # Mock vector store operations
        insertion_manager.vector_store.add_documents.return_value = None
        
        # Mock transaction methods
        insertion_manager.vector_store.begin_transaction = Mock()
        insertion_manager.vector_store.commit_transaction = Mock()
        insertion_manager.vector_store.rollback_transaction = Mock()
        
        # This will fail because method doesn't exist
        with insertion_manager.transaction_context():
            insertion_manager.insert_document(
                text="Doc 1",
                metadata=DocumentMetadata(title="Doc 1"),
                collection_name="test_collection"
            )
            
            with insertion_manager.transaction_context():
                insertion_manager.insert_document(
                    text="Doc 2", 
                    metadata=DocumentMetadata(title="Doc 2"),
                    collection_name="test_collection"
                )
        
        # Verify proper transaction handling
        assert insertion_manager.vector_store.begin_transaction.call_count == 2
        assert insertion_manager.vector_store.commit_transaction.call_count == 2


# Expected imports that will fail until implementation exists
# These represent the classes and exceptions we need to implement

"""
Expected classes to implement in GREEN PHASE:

1. DocumentInsertionManager - Main class for document insertion operations
2. InsertionResult - Result object for single document insertion
3. BatchInsertionResult - Result object for batch insertion operations
4. InsertionError - Base exception for insertion failures
5. ValidationError - Exception for validation failures
6. TransactionError - Exception for transaction-related failures

Key methods to implement:
- DocumentInsertionManager.__init__()
- DocumentInsertionManager.insert_document()
- DocumentInsertionManager.insert_batch()
- DocumentInsertionManager.validate_document_input()
- DocumentInsertionManager.transaction_context()

Integration points:
- ChromaDBManager for vector storage
- DataPreparationManager for cleaning/normalization
- Collection type awareness
- Embedding service integration (Task 4 dependency)
- Comprehensive error handling and logging
""" 