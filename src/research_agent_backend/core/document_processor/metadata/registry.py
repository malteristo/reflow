"""
Metadata Registry Module

Provides storage and querying capabilities for document metadata.
Supports complex queries and metadata management operations.

Implements FR-KB-003.3: Metadata storage and querying.
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

from .types import DocumentMetadata

logger = logging.getLogger(__name__)


class MetadataRegistry:
    """Registry for storing and querying document metadata.
    
    Provides efficient storage and retrieval of document metadata with
    support for complex queries and filtering operations.
    """
    
    def __init__(self):
        """Initialize empty metadata registry."""
        self.documents: Dict[str, DocumentMetadata] = {}
    
    def register(self, metadata: DocumentMetadata) -> None:
        """Register document metadata.
        
        Args:
            metadata: DocumentMetadata instance to register
        """
        if not isinstance(metadata, DocumentMetadata):
            raise ValueError(f"Expected DocumentMetadata, got {type(metadata)}")
        
        self.documents[metadata.document_id] = metadata
        logger.debug("Registered metadata for document: %s", metadata.document_id)
    
    def update(self, metadata: DocumentMetadata) -> None:
        """Update existing document metadata.
        
        Args:
            metadata: Updated DocumentMetadata instance
        """
        if not isinstance(metadata, DocumentMetadata):
            raise ValueError(f"Expected DocumentMetadata, got {type(metadata)}")
        
        self.documents[metadata.document_id] = metadata
        logger.debug("Updated metadata for document: %s", metadata.document_id)
    
    def remove(self, document_id: str) -> None:
        """Remove document from registry.
        
        Args:
            document_id: ID of document to remove
        """
        if document_id in self.documents:
            del self.documents[document_id]
            logger.debug("Removed metadata for document: %s", document_id)
        else:
            logger.warning("Attempted to remove non-existent document: %s", document_id)
    
    def has_document(self, document_id: str) -> bool:
        """Check if document exists in registry.
        
        Args:
            document_id: ID of document to check
            
        Returns:
            True if document exists, False otherwise
        """
        return document_id in self.documents
    
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata by ID.
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            DocumentMetadata instance or None if not found
        """
        return self.documents.get(document_id)
    
    def get_document_count(self) -> int:
        """Get total number of documents.
        
        Returns:
            Number of documents in registry
        """
        return len(self.documents)
    
    def get_all_documents(self) -> List[DocumentMetadata]:
        """Get all document metadata.
        
        Returns:
            List of all DocumentMetadata instances
        """
        return list(self.documents.values())
    
    def query_by_tags(self, tags: List[str], match_all: bool = False) -> List[DocumentMetadata]:
        """Find documents by tags.
        
        Args:
            tags: List of tags to search for
            match_all: If True, document must have all tags; if False, any tag matches
            
        Returns:
            List of matching DocumentMetadata instances
        """
        if not tags:
            return []
        
        matching_docs = []
        for doc in self.documents.values():
            doc_tags = set(doc.tags)
            search_tags = set(tags)
            
            if match_all:
                # Document must have all searched tags
                if search_tags.issubset(doc_tags):
                    matching_docs.append(doc)
            else:
                # Document must have at least one searched tag
                if search_tags.intersection(doc_tags):
                    matching_docs.append(doc)
        
        logger.debug("Found %d documents matching tags: %s (match_all=%s)", 
                    len(matching_docs), tags, match_all)
        return matching_docs
    
    def query_by_author(self, author: str) -> List[DocumentMetadata]:
        """Find documents by author.
        
        Args:
            author: Author name to search for
            
        Returns:
            List of matching DocumentMetadata instances
        """
        matching_docs = [
            doc for doc in self.documents.values() 
            if doc.author and doc.author.lower() == author.lower()
        ]
        
        logger.debug("Found %d documents by author: %s", len(matching_docs), author)
        return matching_docs
    
    def query_by_metadata(self, key_path: str, value: Any) -> List[DocumentMetadata]:
        """Find documents by metadata value using dot notation.
        
        Args:
            key_path: Dot-separated path to metadata field (e.g., 'frontmatter.category')
            value: Value to search for
            
        Returns:
            List of matching DocumentMetadata instances
        """
        matching_docs = []
        for doc in self.documents.values():
            doc_value = self._get_nested_value(doc, key_path)
            if doc_value == value:
                matching_docs.append(doc)
        
        logger.debug("Found %d documents with %s = %s", 
                    len(matching_docs), key_path, value)
        return matching_docs
    
    def query_by_title_pattern(self, pattern: str, case_sensitive: bool = False) -> List[DocumentMetadata]:
        """Find documents by title pattern.
        
        Args:
            pattern: String pattern to search for in titles
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching DocumentMetadata instances
        """
        matching_docs = []
        search_pattern = pattern if case_sensitive else pattern.lower()
        
        for doc in self.documents.values():
            if doc.title:
                title = doc.title if case_sensitive else doc.title.lower()
                if search_pattern in title:
                    matching_docs.append(doc)
        
        logger.debug("Found %d documents matching title pattern: %s", 
                    len(matching_docs), pattern)
        return matching_docs
    
    def query_by_custom_filter(self, filter_func: Callable[[DocumentMetadata], bool]) -> List[DocumentMetadata]:
        """Find documents using a custom filter function.
        
        Args:
            filter_func: Function that takes DocumentMetadata and returns bool
            
        Returns:
            List of matching DocumentMetadata instances
        """
        matching_docs = [doc for doc in self.documents.values() if filter_func(doc)]
        
        logger.debug("Found %d documents matching custom filter", len(matching_docs))
        return matching_docs
    
    def get_all_tags(self) -> List[str]:
        """Get all unique tags across all documents.
        
        Returns:
            Sorted list of unique tags
        """
        all_tags = set()
        for doc in self.documents.values():
            all_tags.update(doc.tags)
        
        return sorted(list(all_tags))
    
    def get_all_authors(self) -> List[str]:
        """Get all unique authors across all documents.
        
        Returns:
            Sorted list of unique authors
        """
        authors = {doc.author for doc in self.documents.values() if doc.author}
        return sorted(list(authors))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with various statistics about the registry
        """
        total_docs = len(self.documents)
        docs_with_title = sum(1 for doc in self.documents.values() if doc.title)
        docs_with_author = sum(1 for doc in self.documents.values() if doc.author)
        docs_with_tags = sum(1 for doc in self.documents.values() if doc.tags)
        docs_with_frontmatter = sum(1 for doc in self.documents.values() if doc.frontmatter)
        docs_with_inline_metadata = sum(1 for doc in self.documents.values() if doc.inline_metadata)
        
        all_tags = self.get_all_tags()
        all_authors = self.get_all_authors()
        
        return {
            'total_documents': total_docs,
            'documents_with_title': docs_with_title,
            'documents_with_author': docs_with_author,
            'documents_with_tags': docs_with_tags,
            'documents_with_frontmatter': docs_with_frontmatter,
            'documents_with_inline_metadata': docs_with_inline_metadata,
            'unique_tags': len(all_tags),
            'unique_authors': len(all_authors),
            'most_common_tags': self._get_most_common_tags(5),
            'coverage_percentages': {
                'title': (docs_with_title / total_docs * 100) if total_docs > 0 else 0,
                'author': (docs_with_author / total_docs * 100) if total_docs > 0 else 0,
                'tags': (docs_with_tags / total_docs * 100) if total_docs > 0 else 0,
                'frontmatter': (docs_with_frontmatter / total_docs * 100) if total_docs > 0 else 0,
                'inline_metadata': (docs_with_inline_metadata / total_docs * 100) if total_docs > 0 else 0
            }
        }
    
    def clear(self) -> None:
        """Clear all documents from registry."""
        self.documents.clear()
        logger.debug("Cleared all documents from registry")
    
    def _get_nested_value(self, doc: DocumentMetadata, key_path: str) -> Any:
        """Get nested value from document using dot notation.
        
        Args:
            doc: DocumentMetadata instance
            key_path: Dot-separated path to the value
            
        Returns:
            The value at the specified path, or None if not found
        """
        parts = key_path.split('.')
        value = doc
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        
        return value
    
    def _get_most_common_tags(self, limit: int = 10) -> List[tuple]:
        """Get most common tags with their counts.
        
        Args:
            limit: Maximum number of tags to return
            
        Returns:
            List of (tag, count) tuples sorted by count descending
        """
        tag_counts = {}
        for doc in self.documents.values():
            for tag in doc.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort by count descending and limit results
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:limit]


class MetadataQuery:
    """Helper class for querying metadata registry.
    
    Provides convenient methods for common query patterns and
    complex query composition.
    """
    
    def find_by_tags(self, registry: MetadataRegistry, tags: List[str], match_all: bool = False) -> List[DocumentMetadata]:
        """Find documents by tags.
        
        Args:
            registry: MetadataRegistry to search
            tags: List of tags to search for
            match_all: If True, document must have all tags
            
        Returns:
            List of matching documents
        """
        return registry.query_by_tags(tags, match_all)
    
    def find_by_author(self, registry: MetadataRegistry, author: str) -> List[DocumentMetadata]:
        """Find documents by author.
        
        Args:
            registry: MetadataRegistry to search
            author: Author name to search for
            
        Returns:
            List of matching documents
        """
        return registry.query_by_author(author)
    
    def find_by_metadata(self, registry: MetadataRegistry, key_path: str, value: Any) -> List[DocumentMetadata]:
        """Find documents by metadata value.
        
        Args:
            registry: MetadataRegistry to search
            key_path: Dot-separated path to metadata field
            value: Value to search for
            
        Returns:
            List of matching documents
        """
        return registry.query_by_metadata(key_path, value)
    
    def find_recent_documents(
        self, 
        registry: MetadataRegistry, 
        date_field: str = "frontmatter.date",
        limit: int = 10
    ) -> List[DocumentMetadata]:
        """Find most recent documents based on a date field.
        
        Args:
            registry: MetadataRegistry to search
            date_field: Dot-separated path to date field
            limit: Maximum number of documents to return
            
        Returns:
            List of most recent documents
        """
        docs_with_dates = []
        for doc in registry.get_all_documents():
            date_value = registry._get_nested_value(doc, date_field)
            if date_value:
                docs_with_dates.append((doc, date_value))
        
        # Sort by date descending (assuming ISO date strings)
        docs_with_dates.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in docs_with_dates[:limit]]
    
    def find_documents_with_metadata_keys(
        self, 
        registry: MetadataRegistry, 
        required_keys: List[str]
    ) -> List[DocumentMetadata]:
        """Find documents that have all specified metadata keys.
        
        Args:
            registry: MetadataRegistry to search
            required_keys: List of required metadata keys
            
        Returns:
            List of documents with all required keys
        """
        def has_all_keys(doc: DocumentMetadata) -> bool:
            for key in required_keys:
                if registry._get_nested_value(doc, key) is None:
                    return False
            return True
        
        return registry.query_by_custom_filter(has_all_keys) 