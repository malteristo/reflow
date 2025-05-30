"""
Document-level metadata for source documents.

This module provides the DocumentMetadata class for tracking metadata
associated with source documents in the vector database.

Supports future team scalability and comprehensive document tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .enums import DocumentType, AccessPermission


@dataclass
class DocumentMetadata:
    """
    Metadata for source documents.
    
    Supports future team scalability and comprehensive document tracking.
    """
    document_id: str = field(default_factory=lambda: str(uuid4()))
    title: str = ""
    document_type: DocumentType = DocumentType.UNKNOWN
    source_path: str = ""
    file_size_bytes: Optional[int] = None
    last_modified: Optional[datetime] = None
    author: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Team scalability fields
    user_id: str = ""
    team_id: Optional[str] = None
    access_permissions: List[AccessPermission] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_timestamp(self) -> None:
        """Update the modified timestamp."""
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "document_type": str(self.document_type),
            "source_path": self.source_path,
            "file_size_bytes": self.file_size_bytes,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "author": self.author,
            "description": self.description,
            "tags": self.tags,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "access_permissions": [str(perm) for perm in self.access_permissions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        } 