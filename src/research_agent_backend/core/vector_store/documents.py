"""
Document Operations for ChromaDB Vector Store.

This module handles document CRUD operations including insertion, updating,
deletion, and batch processing with comprehensive validation.

Implements FR-ST-002: Document operations with metadata support.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb

from ..collection_type_manager import CollectionTypeManager
from .types import (
    BatchResult,
    EmbeddingVector,
    DocumentId,
    MetadataDict,
    DocumentInsertionError,
    MetadataValidationError,
    EmbeddingDimensionError,
    CollectionNotFoundError,
    VectorStoreError,
)


class DocumentManager:
    """
    Document Management for ChromaDB Collections.
    
    Handles document CRUD operations with validation and batch processing.
    """
    
    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_type_manager: CollectionTypeManager
    ) -> None:
        """
        Initialize Document Manager.
        
        Args:
            client: ChromaDB client instance
            collection_type_manager: Collection type manager for validation
        """
        self.client = client
        self.collection_type_manager = collection_type_manager
        self.logger = logging.getLogger(__name__)
        self._collections_cache: Dict[str, chromadb.Collection] = {}
    
    def get_collection(self, name: str) -> chromadb.Collection:
        """Get collection instance with caching."""
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            collection = self.client.get_collection(name)
            self._collections_cache[name] = collection
            return collection
        except Exception as e:
            error_msg = f"Collection '{name}' not found: {e}"
            self.logger.error(error_msg)
            raise CollectionNotFoundError(error_msg) from e
    
    def add_documents(
        self,
        collection_name: str,
        chunks: List[str],
        embeddings: List[EmbeddingVector],
        metadata: Optional[List[MetadataDict]] = None,
        ids: Optional[List[DocumentId]] = None
    ) -> BatchResult:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection name
            chunks: List of text chunks to add
            embeddings: List of embedding vectors for each chunk
            metadata: List of metadata dictionaries for each chunk
            ids: List of document IDs (generates UUIDs if None)
            
        Returns:
            BatchResult with operation statistics
            
        Raises:
            DocumentInsertionError: If insertion fails
            CollectionNotFoundError: If collection doesn't exist
            MetadataValidationError: If metadata structure is invalid
            EmbeddingDimensionError: If embedding dimensions are incompatible
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Validate input lengths
            if len(chunks) != len(embeddings):
                raise DocumentInsertionError("Number of chunks must match number of embeddings")
            
            # Validate metadata structure if provided
            if metadata is not None:
                if len(metadata) != len(chunks):
                    raise DocumentInsertionError("Number of metadata entries must match number of chunks")
                
                # Check for invalid metadata entries
                for i, meta in enumerate(metadata):
                    if meta is None:
                        raise MetadataValidationError(f"Metadata entry {i} cannot be None")
                    if not isinstance(meta, dict):
                        raise MetadataValidationError(f"Metadata entry {i} must be a dictionary")
            
            # Check embedding dimensions consistency
            if embeddings:
                first_dim = len(embeddings[0])
                for i, embedding in enumerate(embeddings):
                    if len(embedding) != first_dim:
                        raise EmbeddingDimensionError(
                            f"Embedding {i} has dimension {len(embedding)}, expected {first_dim}"
                        )
                
                # Check consistency with existing embeddings in collection
                try:
                    existing_count = collection.count()
                    if existing_count > 0:
                        # Get a sample to check dimension
                        sample_results = collection.get(limit=1, include=['embeddings'])
                        if (sample_results['embeddings'] is not None and 
                            len(sample_results['embeddings']) > 0):
                            existing_dim = len(sample_results['embeddings'][0])
                            if first_dim != existing_dim:
                                raise EmbeddingDimensionError(
                                    f"New embeddings have dimension {first_dim}, "
                                    f"but collection expects dimension {existing_dim}"
                                )
                except Exception as e:
                    # If we can't check existing dimensions, log but continue
                    self.logger.warning(f"Could not validate embedding dimensions: {e}")
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid4()) for _ in chunks]
            elif len(ids) != len(chunks):
                raise DocumentInsertionError("Number of IDs must match number of chunks")
            
            # Prepare metadata with consistent structure
            if metadata is None:
                # ChromaDB requires non-empty metadata dictionaries
                # Provide minimal metadata with document index
                metadata = [{"document_index": i} for i in range(len(chunks))]
            else:
                # Ensure no metadata entries are empty
                for i, meta in enumerate(metadata):
                    if not meta:  # Empty dict or None
                        metadata[i] = {"document_index": i}
            
            # Add documents to collection
            collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata
            )
            
            # Create success result
            result = BatchResult(
                success_count=len(chunks),
                error_count=0,
                total_count=len(chunks),
                errors=[],
                success_ids=ids
            )
            
            self.logger.info(f"Successfully added {len(chunks)} documents to collection: {collection_name}")
            return result
            
        except (DocumentInsertionError, MetadataValidationError, EmbeddingDimensionError, CollectionNotFoundError):
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            error_msg = f"Failed to add documents to collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            
            # Check if this is a ChromaDB embedding dimension error
            if "dimension" in str(e).lower() and ("expecting" in str(e).lower() or "got" in str(e).lower()):
                raise EmbeddingDimensionError(str(e)) from e
            else:
                raise DocumentInsertionError(error_msg) from e
    
    def update_documents(
        self,
        collection_name: str,
        ids: List[DocumentId],
        chunks: Optional[List[str]] = None,
        embeddings: Optional[List[EmbeddingVector]] = None,
        metadata: Optional[List[MetadataDict]] = None
    ) -> BatchResult:
        """
        Update existing documents in a collection.
        
        Args:
            collection_name: Target collection name
            ids: List of document IDs to update
            chunks: Updated text chunks (optional)
            embeddings: Updated embedding vectors (optional)
            metadata: Updated metadata (optional)
            
        Returns:
            BatchResult with operation statistics
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Validate that all provided lists have the same length as IDs
            if chunks is not None and len(chunks) != len(ids):
                raise DocumentInsertionError("Number of chunks must match number of IDs")
            if embeddings is not None and len(embeddings) != len(ids):
                raise DocumentInsertionError("Number of embeddings must match number of IDs")
            if metadata is not None and len(metadata) != len(ids):
                raise DocumentInsertionError("Number of metadata entries must match number of IDs")
            
            # Update documents
            collection.update(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata
            )
            
            result = BatchResult(
                success_count=len(ids),
                error_count=0,
                total_count=len(ids),
                errors=[],
                success_ids=ids
            )
            
            self.logger.info(f"Successfully updated {len(ids)} documents in collection: {collection_name}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to update documents in collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise DocumentInsertionError(error_msg) from e
    
    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[DocumentId]] = None,
        where: Optional[MetadataDict] = None
    ) -> BatchResult:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Target collection name
            ids: Specific document IDs to delete (optional)
            where: Metadata filter for documents to delete (optional)
            
        Returns:
            BatchResult with operation statistics
            
        Note:
            Either ids or where must be provided, not both.
        """
        try:
            collection = self.get_collection(collection_name)
            
            if ids is None and where is None:
                raise DocumentInsertionError("Either 'ids' or 'where' parameter must be provided")
            if ids is not None and where is not None:
                raise DocumentInsertionError("Cannot provide both 'ids' and 'where' parameters")
            
            # Get documents before deletion for result statistics
            if ids is not None:
                # Count specific IDs
                delete_count = len(ids)
                deleted_ids = ids
            else:
                # Get documents matching filter
                try:
                    matching_docs = collection.get(where=where, include=['documents'])
                    deleted_ids = matching_docs['ids'] if matching_docs['ids'] else []
                    delete_count = len(deleted_ids)
                except Exception:
                    # If we can't get the documents, we'll estimate
                    delete_count = 0
                    deleted_ids = []
            
            # Delete documents
            collection.delete(ids=ids, where=where)
            
            result = BatchResult(
                success_count=delete_count,
                error_count=0,
                total_count=delete_count,
                errors=[],
                success_ids=deleted_ids
            )
            
            self.logger.info(f"Successfully deleted {delete_count} documents from collection: {collection_name}")
            return result
            
        except Exception as e:
            error_msg = f"Failed to delete documents from collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise DocumentInsertionError(error_msg) from e
    
    def get_documents(
        self,
        collection_name: str,
        ids: Optional[List[DocumentId]] = None,
        where: Optional[MetadataDict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents from a collection.
        
        Args:
            collection_name: Collection to query
            ids: Specific document IDs to retrieve
            where: Metadata filter for documents
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            include: Data to include ('documents', 'metadatas', 'embeddings')
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Set default include if not provided
            if include is None:
                include = ['documents', 'metadatas']
            
            # Get documents
            results = collection.get(
                ids=ids,
                where=where,
                limit=limit,
                offset=offset,
                include=include
            )
            
            self.logger.debug(f"Retrieved {len(results.get('ids', []))} documents from collection: {collection_name}")
            return results
            
        except Exception as e:
            error_msg = f"Failed to get documents from collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def count_documents(
        self,
        collection_name: str,
        where: Optional[MetadataDict] = None
    ) -> int:
        """
        Count documents in a collection.
        
        Args:
            collection_name: Collection to count
            where: Optional metadata filter
            
        Returns:
            Number of documents matching criteria
        """
        try:
            collection = self.get_collection(collection_name)
            
            if where is None:
                # Total count
                count = collection.count()
            else:
                # Count with filter - get documents and count results
                results = collection.get(where=where, limit=1, include=[])
                if results and 'ids' in results:
                    # This is an approximation - ChromaDB doesn't have a direct count with filter
                    # We'd need to get all matching documents to get exact count
                    all_results = collection.get(where=where, include=[])
                    count = len(all_results.get('ids', []))
                else:
                    count = 0
            
            return count
            
        except Exception as e:
            error_msg = f"Failed to count documents in collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def determine_collection_for_document(
        self,
        document_metadata: MetadataDict,
        chunk_metadata: Optional[MetadataDict] = None
    ) -> str:
        """
        Determine the appropriate collection for a document based on its metadata.
        
        Args:
            document_metadata: Document-level metadata
            chunk_metadata: Optional chunk-level metadata
            
        Returns:
            Collection name to use for this document
        """
        # Determine collection type
        collection_type = self.collection_type_manager.determine_collection_type(
            document_metadata=document_metadata,
            chunk_metadata=chunk_metadata
        )
        
        # Extract project information for naming
        project_name = None
        if collection_type.value == 'project-specific':  # Using .value to get string
            project_name = document_metadata.get('project_name') or document_metadata.get('team_id')
        
        # Generate collection name
        collection_name = self.collection_type_manager.create_collection_name(
            collection_type=collection_type,
            project_name=project_name
        )
        
        return collection_name 