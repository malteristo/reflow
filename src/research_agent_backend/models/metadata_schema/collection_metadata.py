"""
Collection-level metadata for vector database collections.

This module provides the CollectionMetadata class for tracking metadata
associated with vector database collections.

Supports collection management and team-based access control.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import CollectionType, AccessPermission


@dataclass
class CollectionMetadata:
    """
    Metadata for vector database collections.
    
    Supports collection management and team-based access control.
    """
    collection_name: str = ""
    collection_type: CollectionType = CollectionType.GENERAL
    description: str = ""
    
    # Vector database configuration
    embedding_model: str = ""
    embedding_dimension: int = 0
    distance_metric: str = "cosine"
    
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