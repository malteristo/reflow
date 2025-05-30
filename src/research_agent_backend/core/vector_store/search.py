"""
Search and Query Operations for ChromaDB Vector Store.

This module handles vector similarity search operations including query execution,
filter validation, and result formatting.

Implements FR-ST-002: Vector search operations with metadata filtering.
"""

import logging
from typing import Any, Dict, List, Optional

import chromadb

from .types import (
    SearchResult,
    FilterDict,
    EmbeddingVector,
    CollectionNotFoundError,
    QueryError,
    VectorStoreError,
)


class SearchManager:
    """
    Search and Query Manager for ChromaDB Collections.
    
    Handles vector similarity search operations with filtering and result formatting.
    """
    
    def __init__(self, client: chromadb.ClientAPI) -> None:
        """
        Initialize Search Manager.
        
        Args:
            client: ChromaDB client instance
        """
        self.client = client
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
    
    def query_collection(
        self,
        collection_name: str,
        query_embedding: EmbeddingVector,
        k: int = 10,
        filters: Optional[FilterDict] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        include_distances: bool = True
    ) -> SearchResult:
        """
        Query a collection for similar documents.
        
        Args:
            collection_name: Collection to query
            query_embedding: Query vector
            k: Number of results to return
            filters: Metadata filters to apply
            include_metadata: Include metadata in results
            include_documents: Include document text in results
            include_distances: Include similarity distances in results
            
        Returns:
            SearchResult with query results
            
        Raises:
            QueryError: If query execution fails
            CollectionNotFoundError: If collection doesn't exist
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Validate filter structure if provided
            if filters is not None:
                self._validate_query_filters(filters)
            
            # Prepare include list
            include_list = []
            if include_metadata:
                include_list.append('metadatas')
            if include_documents:
                include_list.append('documents')
            if include_distances:
                include_list.append('distances')
            
            # Execute query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filters,
                include=include_list
            )
            
            # Format results
            search_result = SearchResult(
                collection=collection_name,
                ids=results['ids'][0] if results['ids'] else [],
                query_embedding_dimension=len(query_embedding),
                results_count=len(results['ids'][0]) if results['ids'] else 0,
                metadatas=results['metadatas'][0] if include_metadata and 'metadatas' in results else None,
                documents=results['documents'][0] if include_documents and 'documents' in results else None,
                distances=results['distances'][0] if include_distances and 'distances' in results else None
            )
            
            self.logger.debug(f"Query returned {search_result.results_count} results from {collection_name}")
            return search_result
            
        except CollectionNotFoundError:
            # Re-raise collection not found
            raise
        except QueryError:
            # Re-raise our query errors
            raise
        except Exception as e:
            error_msg = f"Failed to query collection '{collection_name}': {e}"
            self.logger.error(error_msg)
            raise QueryError(error_msg) from e
    
    def multi_collection_query(
        self,
        collection_names: List[str],
        query_embedding: EmbeddingVector,
        k: int = 10,
        filters: Optional[FilterDict] = None,
        include_metadata: bool = True,
        include_documents: bool = True,
        include_distances: bool = True
    ) -> List[SearchResult]:
        """
        Query multiple collections for similar documents.
        
        Args:
            collection_names: Collections to query
            query_embedding: Query vector
            k: Number of results to return per collection
            filters: Metadata filters to apply
            include_metadata: Include metadata in results
            include_documents: Include document text in results
            include_distances: Include similarity distances in results
            
        Returns:
            List of SearchResult objects, one per collection
        """
        results = []
        
        for collection_name in collection_names:
            try:
                result = self.query_collection(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    k=k,
                    filters=filters,
                    include_metadata=include_metadata,
                    include_documents=include_documents,
                    include_distances=include_distances
                )
                results.append(result)
                
            except CollectionNotFoundError:
                self.logger.warning(f"Collection '{collection_name}' not found, skipping")
                continue
            except Exception as e:
                self.logger.error(f"Error querying collection '{collection_name}': {e}")
                # Continue with other collections
                continue
        
        return results
    
    def _validate_query_filters(self, filters: FilterDict) -> None:
        """
        Validate query filter structure.
        
        Args:
            filters: Filter dictionary to validate
            
        Raises:
            QueryError: If filter structure is invalid
        """
        try:
            # Check for invalid operators
            invalid_operators = ["$invalid"]
            
            def check_filter_dict(filter_dict):
                if not isinstance(filter_dict, dict):
                    return
                
                for key, value in filter_dict.items():
                    if key in invalid_operators:
                        raise QueryError(f"Invalid filter operator: {key}")
                    
                    if isinstance(value, dict):
                        check_filter_dict(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                check_filter_dict(item)
            
            check_filter_dict(filters)
            
        except Exception as e:
            if isinstance(e, QueryError):
                raise
            else:
                raise QueryError(f"Invalid filter structure: {e}") from e
    
    def validate_embedding_dimension(
        self,
        collection_name: str,
        embedding: EmbeddingVector
    ) -> bool:
        """
        Validate that an embedding has the correct dimension for a collection.
        
        Args:
            collection_name: Collection name
            embedding: Embedding vector to validate
            
        Returns:
            True if dimension is valid, False otherwise
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Check if collection has any documents
            count = collection.count()
            if count == 0:
                # Empty collection, any dimension is valid for first document
                return True
            
            # Get a sample to check dimension
            sample_results = collection.get(limit=1, include=['embeddings'])
            if sample_results['embeddings'] and len(sample_results['embeddings']) > 0:
                expected_dim = len(sample_results['embeddings'][0])
                actual_dim = len(embedding)
                return actual_dim == expected_dim
            
            # If we can't determine dimension, assume valid
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not validate embedding dimension: {e}")
            return True
    
    def get_similar_documents_by_id(
        self,
        collection_name: str,
        document_id: str,
        k: int = 10,
        filters: Optional[FilterDict] = None
    ) -> Optional[SearchResult]:
        """
        Find documents similar to a specific document in the collection.
        
        Args:
            collection_name: Collection to search
            document_id: ID of the reference document
            k: Number of similar documents to return
            filters: Optional metadata filters
            
        Returns:
            SearchResult with similar documents, or None if document not found
        """
        try:
            collection = self.get_collection(collection_name)
            
            # Get the reference document's embedding
            doc_result = collection.get(
                ids=[document_id],
                include=['embeddings']
            )
            
            if not doc_result['ids'] or not doc_result['embeddings']:
                self.logger.warning(f"Document '{document_id}' not found in collection '{collection_name}'")
                return None
            
            # Use the document's embedding as query
            query_embedding = doc_result['embeddings'][0]
            
            # Query for similar documents (excluding the original)
            search_result = self.query_collection(
                collection_name=collection_name,
                query_embedding=query_embedding,
                k=k + 1,  # Get one extra to account for excluding original
                filters=filters
            )
            
            # Remove the original document from results
            if document_id in search_result.ids:
                idx = search_result.ids.index(document_id)
                search_result.ids.pop(idx)
                if search_result.metadatas:
                    search_result.metadatas.pop(idx)
                if search_result.documents:
                    search_result.documents.pop(idx)
                if search_result.distances:
                    search_result.distances.pop(idx)
                search_result.results_count -= 1
            
            # Trim to requested size
            if len(search_result.ids) > k:
                search_result.ids = search_result.ids[:k]
                if search_result.metadatas:
                    search_result.metadatas = search_result.metadatas[:k]
                if search_result.documents:
                    search_result.documents = search_result.documents[:k]
                if search_result.distances:
                    search_result.distances = search_result.distances[:k]
                search_result.results_count = k
            
            return search_result
            
        except Exception as e:
            error_msg = f"Failed to find similar documents for '{document_id}': {e}"
            self.logger.error(error_msg)
            raise QueryError(error_msg) from e 