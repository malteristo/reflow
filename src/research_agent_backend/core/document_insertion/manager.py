"""
Main DocumentInsertionManager orchestration class.

This module provides the primary interface for document insertion operations,
orchestrating validation, preparation, chunking, embedding, and storage.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from uuid import uuid4

from ...core.vector_store import ChromaDBManager
from ...core.data_preparation import DataPreparationManager
from ...core.collection_type_manager import CollectionTypeManager
from ...models.metadata_schema import DocumentMetadata
from ...utils.config import ConfigManager

from .exceptions import (
    InsertionError, 
    ValidationError, 
    TransactionError,
    InsertionResult, 
    BatchInsertionResult
)
from .validation import DocumentValidator, DocumentPreparationService
from .chunking import DocumentChunker, ChunkMetadataFactory
from .embeddings import EmbeddingService
from .transactions import TransactionManager


class DocumentInsertionManager:
    """
    Document Insertion Manager for vector database operations.
    
    Provides comprehensive document insertion capabilities with:
    - Single document insertion with metadata validation
    - Batch insertion with transaction support and progress tracking
    - Integration with data preparation and collection type systems
    - Vector generation and embedding integration
    - Comprehensive error handling and validation
    
    Implements strict TDD methodology with test-driven development.
    """
    
    def __init__(
        self,
        vector_store: ChromaDBManager,
        data_preparation_manager: Optional[DataPreparationManager] = None,
        config_manager: Optional[ConfigManager] = None,
        collection_type_manager: Optional[CollectionTypeManager] = None,
        batch_size: int = 100,
        enable_transactions: bool = True,
        embedding_service: Optional[Any] = None
    ) -> None:
        """
        Initialize DocumentInsertionManager.
        
        Args:
            vector_store: ChromaDBManager instance for vector storage
            data_preparation_manager: Manager for data cleaning and normalization
            config_manager: Configuration manager (creates default if None)
            collection_type_manager: Collection type manager for type-aware operations
            batch_size: Default batch size for batch operations
            enable_transactions: Enable transaction support for batch operations
            embedding_service: Embedding service for vector generation (Task 4 dependency)
            
        Raises:
            ValueError: If required dependencies are missing
        """
        if vector_store is None:
            raise ValueError("vector_store is required")
        
        self.vector_store = vector_store
        self.data_preparation_manager = data_preparation_manager
        self.config_manager = config_manager or ConfigManager()
        self.collection_type_manager = collection_type_manager
        self.batch_size = batch_size
        self.enable_transactions = enable_transactions
        self.embedding_service = embedding_service
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize service components
        self.validator = DocumentValidator(logger=self.logger)
        self.preparation_service = DocumentPreparationService(
            data_preparation_manager=data_preparation_manager,
            collection_type_manager=collection_type_manager,
            logger=self.logger
        )
        self.chunker = DocumentChunker(logger=self.logger)
        self.embedding_svc = EmbeddingService(
            embedding_service=embedding_service,
            logger=self.logger
        )
        self.transaction_manager = TransactionManager(
            vector_store=vector_store,
            logger=self.logger
        )
        
        # Initialize embedded services if not provided
        if self.data_preparation_manager is None:
            from ...core.data_preparation import create_data_preparation_manager
            self.data_preparation_manager = create_data_preparation_manager(
                config_manager=self.config_manager
            )
            self.preparation_service.data_preparation_manager = self.data_preparation_manager
        
        if self.collection_type_manager is None:
            from ...core.collection_type_manager import create_collection_type_manager
            self.collection_type_manager = create_collection_type_manager(
                config_manager=self.config_manager
            )
            self.preparation_service.collection_type_manager = self.collection_type_manager
        
        self.logger.info("DocumentInsertionManager initialized successfully")
    
    def insert_document(
        self,
        text: str,
        metadata: Union[DocumentMetadata, Dict[str, Any]],
        collection_name: str,
        enable_chunking: bool = False,
        chunk_size: Optional[int] = None,
        document_id: Optional[str] = None
    ) -> InsertionResult:
        """
        Insert a single document into the vector database.
        
        Args:
            text: Document text content
            metadata: Document metadata (DocumentMetadata or dict)
            collection_name: Target collection name
            enable_chunking: Whether to chunk large documents
            chunk_size: Custom chunk size (uses config default if None)
            document_id: Custom document ID (generates UUID if None)
            
        Returns:
            InsertionResult with insertion status and details
            
        Raises:
            ValidationError: If document validation fails
            InsertionError: If insertion operation fails
        """
        start_time = datetime.utcnow()
        result = InsertionResult()
        
        try:
            # Validate input
            self.validator.validate_document_input(text, metadata)
            
            # Generate document ID if not provided
            if document_id is None:
                document_id = str(uuid4())
            result.document_id = document_id
            
            # Convert metadata to DocumentMetadata if needed
            if isinstance(metadata, dict):
                doc_metadata = DocumentMetadata(**metadata)
            else:
                doc_metadata = metadata
            
            # Data preparation and cleaning
            cleaned_text, processed_embedding, processed_metadata = self.preparation_service.prepare_document(
                text, doc_metadata, collection_name
            )
            
            if cleaned_text is None:
                raise ValidationError("Document was filtered out during preparation")
            
            # Handle chunking if enabled
            if enable_chunking:
                chunks, chunk_metadata_list = self.chunker.chunk_document(
                    cleaned_text, doc_metadata, chunk_size
                )
                embeddings = self.embedding_svc.generate_embeddings_batch(chunks)
                result.chunk_count = len(chunks)
                result.chunk_ids = [chunk_meta.chunk_id for chunk_meta in chunk_metadata_list]
            else:
                # Single chunk insertion
                chunks = [cleaned_text]
                embeddings = [self.embedding_svc.generate_embedding(cleaned_text)]
                chunk_metadata_list = [ChunkMetadataFactory.create_chunk_metadata(doc_metadata, 0)]
                result.chunk_count = 1
                result.chunk_ids = [chunk_metadata_list[0].chunk_id]
            
            # Convert chunk metadata to ChromaDB format
            chromadb_metadata = [
                chunk_meta.to_chromadb_metadata() for chunk_meta in chunk_metadata_list
            ]
            
            # Insert into vector store
            try:
                self.vector_store.add_documents(
                    collection_name=collection_name,
                    chunks=chunks,
                    embeddings=embeddings,
                    metadata=chromadb_metadata,
                    ids=result.chunk_ids
                )
                
                result.success = True
                result.metadata = processed_metadata or {}
                
                self.logger.info(
                    f"Successfully inserted document {document_id} with {result.chunk_count} chunks"
                )
                
            except Exception as e:
                raise InsertionError(f"Failed to insert document into vector store: {e}") from e
                
        except ValidationError as e:
            result.errors.append(str(e))
            self.logger.error(f"Document validation failed: {e}")
            raise
        except InsertionError as e:
            result.errors.append(str(e))
            self.logger.error(f"Document insertion failed: {e}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error during document insertion: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg)
            raise InsertionError(error_msg) from e
        finally:
            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def insert_batch(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str,
        enable_transaction_rollback: bool = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> BatchInsertionResult:
        """
        Insert multiple documents in batches with transaction support.
        
        Args:
            documents: List of documents with 'text' and 'metadata' keys
            collection_name: Target collection name
            enable_transaction_rollback: Enable rollback on failure (uses instance default if None)
            progress_callback: Optional callback for progress tracking (processed, total, current_batch)
            
        Returns:
            BatchInsertionResult with batch insertion status and details
            
        Raises:
            TransactionError: If transaction fails and rollback is enabled
            InsertionError: If batch insertion fails
        """
        if enable_transaction_rollback is None:
            enable_transaction_rollback = self.enable_transactions
        
        start_time = datetime.utcnow()
        result = BatchInsertionResult()
        result.total_documents = len(documents)
        result.transaction_id = str(uuid4()) if enable_transaction_rollback else None
        
        try:
            # Begin transaction if enabled
            if enable_transaction_rollback:
                self.transaction_manager.begin_transaction()
            
            # Process documents in batches
            processed_count = 0
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                current_batch_num = (i // self.batch_size) + 1
                
                try:
                    # Process current batch
                    batch_results = self._process_document_batch(batch, collection_name)
                    
                    # Update results
                    for batch_result in batch_results:
                        if batch_result.success:
                            result.successful_insertions += 1
                            if batch_result.document_id:
                                result.document_ids.append(batch_result.document_id)
                        else:
                            result.failed_insertions += 1
                            result.failed_documents.append({
                                'document': batch[batch_results.index(batch_result)],
                                'errors': batch_result.errors
                            })
                            result.errors.extend(batch_result.errors)
                            
                            # If any document fails and rollback is enabled, raise TransactionError
                            if enable_transaction_rollback and batch_result.errors:
                                self.transaction_manager.rollback_transaction()
                                raise TransactionError(
                                    f"Document insertion failed in batch {current_batch_num}, "
                                    f"rolling back transaction: {batch_result.errors[0]}"
                                )
                    
                    processed_count += len(batch)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(processed_count, result.total_documents, current_batch_num)
                    
                    self.logger.debug(
                        f"Processed batch {current_batch_num}, "
                        f"{processed_count}/{result.total_documents} documents"
                    )
                    
                except TransactionError:
                    # Re-raise transaction errors immediately
                    raise
                except Exception as e:
                    if enable_transaction_rollback:
                        self.transaction_manager.rollback_transaction()
                        raise TransactionError(
                            f"Batch insertion failed, rolling back: {e}"
                        ) from e
                    else:
                        # Continue processing remaining batches without rollback
                        error_msg = f"Batch {current_batch_num} failed: {e}"
                        result.errors.append(error_msg)
                        result.failed_insertions += len(batch)
                        self.logger.error(error_msg)
            
            # Commit transaction if enabled and successful
            if enable_transaction_rollback:
                self.transaction_manager.commit_transaction()
            
            result.success = result.failed_insertions == 0
            
            self.logger.info(
                f"Batch insertion completed: {result.successful_insertions}/{result.total_documents} "
                f"documents inserted successfully"
            )
            
        except TransactionError:
            # Re-raise transaction errors
            raise
        except Exception as e:
            if enable_transaction_rollback:
                self.transaction_manager.rollback_transaction()
                raise TransactionError(f"Batch insertion failed: {e}") from e
            else:
                error_msg = f"Batch insertion failed: {e}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
                raise InsertionError(error_msg) from e
        finally:
            # Calculate processing time
            end_time = datetime.utcnow()
            result.processing_time_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def validate_document_input(
        self,
        text: str,
        metadata: Union[DocumentMetadata, Dict[str, Any], None]
    ) -> None:
        """
        Validate document input before insertion.
        
        Args:
            text: Document text content
            metadata: Document metadata
            
        Raises:
            ValidationError: If validation fails
        """
        return self.validator.validate_document_input(text, metadata)
    
    @property
    def transaction_context(self):
        """Access to transaction context manager."""
        return self.transaction_manager.transaction_context
    
    def _process_document_batch(
        self, 
        batch: List[Dict[str, Any]], 
        collection_name: str
    ) -> List[InsertionResult]:
        """Process a batch of documents."""
        results = []
        
        for doc in batch:
            try:
                result = self.insert_document(
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    collection_name=collection_name
                )
                results.append(result)
            except Exception as e:
                # Create failed result
                failed_result = InsertionResult()
                failed_result.success = False
                failed_result.errors.append(str(e))
                results.append(failed_result)
        
        return results 