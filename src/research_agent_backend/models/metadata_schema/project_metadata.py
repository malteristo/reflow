"""
Project metadata models for Research Agent.

This module defines data classes for project metadata including
project information, collection links, and project context.

Implements FR-KB-005: Project metadata storage and management.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ProjectStatus(Enum):
    """Project status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


@dataclass
class ProjectMetadata:
    """
    Metadata for a project in the knowledge base.
    
    Contains project identification, description, linked collections,
    and configuration information.
    """
    name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    owner_id: Optional[str] = None
    team_id: Optional[str] = None
    
    # Collection linkage
    linked_collections: List[str] = field(default_factory=list)
    default_collections: List[str] = field(default_factory=list)
    
    # Project-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics (computed)
    linked_collections_count: int = 0
    total_documents: int = 0
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.updated_at is None:
            self.updated_at = self.created_at
        
        # Update computed fields
        self.linked_collections_count = len(self.linked_collections)


@dataclass
class CollectionLinkMetadata:
    """
    Metadata for a collection linked to a project.
    
    Contains information about the relationship between
    a collection and a project.
    """
    collection_name: str
    project_name: str
    description: Optional[str] = None
    is_default: bool = False
    linked_at: Optional[str] = None
    linked_by: Optional[str] = None
    
    # Statistics (computed from collection)
    document_count: Optional[int] = None
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.linked_at is None:
            self.linked_at = datetime.utcnow().isoformat()


@dataclass
class ProjectInfo:
    """
    Complete project information including metadata and linked collections.
    
    Used for displaying project details and managing project state.
    """
    metadata: ProjectMetadata
    linked_collections: List[CollectionLinkMetadata] = field(default_factory=list)
    default_collections: List[CollectionLinkMetadata] = field(default_factory=list)
    
    @property
    def name(self) -> str:
        """Get project name."""
        return self.metadata.name
    
    @property
    def description(self) -> Optional[str]:
        """Get project description."""
        return self.metadata.description
    
    @property
    def tags(self) -> List[str]:
        """Get project tags."""
        return self.metadata.tags
    
    @property
    def created_at(self) -> Optional[str]:
        """Get project creation time."""
        return self.metadata.created_at
    
    @property
    def linked_collections_count(self) -> int:
        """Get count of linked collections."""
        return len(self.linked_collections)
    
    @property
    def total_documents(self) -> int:
        """Get total document count across all linked collections."""
        return sum(
            coll.document_count or 0 
            for coll in self.linked_collections
        )


@dataclass
class ProjectContext:
    """
    Project context information for queries and operations.
    
    Contains the active project and its associated collections
    for use in knowledge base operations.
    """
    active_project: Optional[str] = None
    default_collections: List[str] = field(default_factory=list)
    project_paths: Dict[str, str] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if a project context is active."""
        return self.active_project is not None
    
    def get_collections_for_query(self, explicit_collections: Optional[List[str]] = None) -> List[str]:
        """
        Get collections to use for a query.
        
        Returns explicit collections if provided, otherwise default collections.
        """
        if explicit_collections:
            return explicit_collections
        return self.default_collections if self.default_collections else [] 