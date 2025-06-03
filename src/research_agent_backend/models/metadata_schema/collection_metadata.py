"""
Collection-level metadata for vector database collections.

This module provides the CollectionMetadata class for tracking metadata
associated with vector database collections.

Supports collection management, team-based access control, and model change detection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal

from .enums import CollectionType, AccessPermission

# Type alias for reindex status
ReindexStatus = Literal["pending", "in_progress", "completed", "failed", "not_required"]


@dataclass
class CollectionMetadata:
    """
    Metadata for vector database collections.
    
    Supports collection management, team-based access control, and model change detection.
    Enhanced with model fingerprint tracking for automatic re-indexing workflows.
    """
    collection_name: str = ""
    collection_type: CollectionType = CollectionType.GENERAL
    description: str = ""
    
    # Vector database configuration
    embedding_model: str = ""
    embedding_dimension: int = 0
    distance_metric: str = "cosine"
    
    # Model change detection integration
    embedding_model_fingerprint: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    reindex_status: ReindexStatus = "not_required"
    last_reindex_timestamp: Optional[datetime] = None
    original_document_count: int = 0  # Validation metric for re-indexing completeness
    
    # HNSW parameters
    hnsw_construction_ef: int = 100
    hnsw_m: int = 16
    
    # Collection statistics
    document_count: int = 0
    chunk_count: int = 0
    total_size_bytes: int = 0
    
    # Team scalability
    owner_id: str = ""
    team_id: Optional[str] = None
    team_permissions: Dict[str, List[AccessPermission]] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_indexed_at: Optional[datetime] = None
    
    def update_stats(self, document_count: int = None, chunk_count: int = None, size_bytes: int = None) -> None:
        """Update collection statistics."""
        if document_count is not None:
            self.document_count = document_count
        if chunk_count is not None:
            self.chunk_count = chunk_count
        if size_bytes is not None:
            self.total_size_bytes = size_bytes
        self.updated_at = datetime.utcnow()
    
    def update_model_fingerprint(self, fingerprint: str, model_name: str, model_version: str) -> None:
        """
        Update model fingerprint information.
        
        Args:
            fingerprint: Model fingerprint checksum
            model_name: Name of the embedding model
            model_version: Version of the embedding model
        """
        self.embedding_model_fingerprint = fingerprint
        self.model_name = model_name
        self.model_version = model_version
        self.updated_at = datetime.utcnow()
    
    def set_reindex_status(self, status: ReindexStatus, update_timestamp: bool = True) -> None:
        """
        Set the reindex status.
        
        Args:
            status: New reindex status
            update_timestamp: Whether to update the reindex timestamp
        """
        self.reindex_status = status
        if update_timestamp and status in ("completed", "failed"):
            self.last_reindex_timestamp = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def requires_reindexing(self, current_fingerprint: str) -> bool:
        """
        Check if collection requires re-indexing based on model fingerprint.
        
        Args:
            current_fingerprint: Current model fingerprint to compare against
            
        Returns:
            True if re-indexing is required, False otherwise
        """
        if not self.embedding_model_fingerprint:
            # No stored fingerprint means this is a new collection
            return True
        
        if self.embedding_model_fingerprint != current_fingerprint:
            # Fingerprint mismatch indicates model change
            return True
        
        if self.reindex_status in ("pending", "failed"):
            # Explicit reindex requirement
            return True
        
        return False
    
    def is_model_compatible(self, fingerprint: str, model_name: str) -> bool:
        """
        Check if the collection is compatible with the given model.
        
        Args:
            fingerprint: Model fingerprint to check
            model_name: Model name to check
            
        Returns:
            True if compatible, False otherwise
        """
        if not self.embedding_model_fingerprint or not self.model_name:
            # No stored model info - compatibility unknown
            return False
        
        return (
            self.embedding_model_fingerprint == fingerprint and
            self.model_name == model_name and
            self.reindex_status == "completed"
        )
    
    def get_reindex_progress_info(self) -> Dict[str, Any]:
        """
        Get information about reindex progress and status.
        
        Returns:
            Dictionary with reindex status information
        """
        return {
            "status": self.reindex_status,
            "last_reindex": self.last_reindex_timestamp.isoformat() if self.last_reindex_timestamp else None,
            "model_fingerprint": self.embedding_model_fingerprint,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "document_count": self.document_count,
            "original_document_count": self.original_document_count,
            "requires_reindexing": self.requires_reindexing(self.embedding_model_fingerprint or "")
        }
    
    def add_team_permission(self, user_id: str, permissions: List[AccessPermission]) -> None:
        """Add team member permissions."""
        self.team_permissions[user_id] = permissions
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "collection_name": self.collection_name,
            "collection_type": str(self.collection_type),
            "description": self.description,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.distance_metric,
            # Model change detection fields
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "reindex_status": self.reindex_status,
            "last_reindex_timestamp": self.last_reindex_timestamp.isoformat() if self.last_reindex_timestamp else None,
            "original_document_count": self.original_document_count,
            # HNSW parameters
            "hnsw_construction_ef": self.hnsw_construction_ef,
            "hnsw_m": self.hnsw_m,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "total_size_bytes": self.total_size_bytes,
            "owner_id": self.owner_id,
            "team_id": self.team_id,
            "team_permissions": {
                user_id: [str(perm) for perm in perms]
                for user_id, perms in self.team_permissions.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
        } 