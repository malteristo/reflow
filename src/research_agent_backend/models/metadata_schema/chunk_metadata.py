"""
Chunk-level metadata for document chunks in vector database.

This module provides the ChunkMetadata class for tracking metadata
associated with individual document chunks stored in ChromaDB.

Implements FR-KB-002: Rich metadata for document chunks including
source_document_id, document_title, header_hierarchy, chunk_sequence_id,
content_type, and code_language.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from .enums import ContentType, AccessPermission
from .header_hierarchy import HeaderHierarchy


logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """
    Metadata for document chunks stored in vector database.
    
    Implements FR-KB-002: Rich metadata for document chunks including
    source_document_id, document_title, header_hierarchy, chunk_sequence_id,
    content_type, and code_language.
    """
    # Core chunk identification
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    source_document_id: str = ""
    document_title: str = ""
    chunk_sequence_id: int = 0
    
    # Content classification
    content_type: ContentType = ContentType.PROSE
    code_language: Optional[str] = None
    
    # Markdown structure context
    header_hierarchy: HeaderHierarchy = field(default_factory=HeaderHierarchy)
    
    # Chunk properties
    chunk_size: int = 0
    start_char_index: Optional[int] = None
    end_char_index: Optional[int] = None
    
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
    
    def to_chromadb_metadata(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Convert to ChromaDB-compatible metadata.
        
        ChromaDB only accepts str, int, float, or bool values for metadata.
        Complex objects are JSON serialized to strings.
        """
        metadata = {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "document_title": self.document_title,
            "chunk_sequence_id": self.chunk_sequence_id,
            "content_type": str(self.content_type),
            "code_language": self.code_language or "",
            "header_hierarchy": self.header_hierarchy.to_json(),
            "chunk_size": self.chunk_size,
            "start_char_index": self.start_char_index or 0,
            "end_char_index": self.end_char_index or 0,
            "user_id": self.user_id,
            "team_id": self.team_id or "",
            "access_permissions": ",".join(str(perm) for perm in self.access_permissions),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        
        # Ensure all values are ChromaDB-compatible types
        validated_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                validated_metadata[key] = value
            else:
                validated_metadata[key] = str(value)
        
        return validated_metadata
    
    @classmethod
    def from_chromadb_metadata(cls, metadata: Dict[str, Any]) -> 'ChunkMetadata':
        """Create ChunkMetadata from ChromaDB metadata."""
        chunk_metadata = cls()
        
        # Basic fields
        chunk_metadata.chunk_id = metadata.get("chunk_id", "")
        chunk_metadata.source_document_id = metadata.get("source_document_id", "")
        chunk_metadata.document_title = metadata.get("document_title", "")
        chunk_metadata.chunk_sequence_id = int(metadata.get("chunk_sequence_id", 0))
        
        # Content type
        content_type_str = metadata.get("content_type", "prose")
        try:
            chunk_metadata.content_type = ContentType(content_type_str)
        except ValueError:
            chunk_metadata.content_type = ContentType.UNKNOWN
        
        chunk_metadata.code_language = metadata.get("code_language") or None
        
        # Header hierarchy
        hierarchy_json = metadata.get("header_hierarchy", "")
        if hierarchy_json:
            chunk_metadata.header_hierarchy = HeaderHierarchy.from_json(hierarchy_json)
        
        # Chunk properties
        chunk_metadata.chunk_size = int(metadata.get("chunk_size", 0))
        chunk_metadata.start_char_index = int(metadata.get("start_char_index", 0)) or None
        chunk_metadata.end_char_index = int(metadata.get("end_char_index", 0)) or None
        
        # Team fields
        chunk_metadata.user_id = metadata.get("user_id", "")
        chunk_metadata.team_id = metadata.get("team_id") or None
        
        # Access permissions
        permissions_str = metadata.get("access_permissions", "")
        if permissions_str:
            permission_strs = permissions_str.split(",")
            permissions = []
            for perm_str in permission_strs:
                try:
                    permissions.append(AccessPermission(perm_str.strip()))
                except ValueError:
                    continue
            chunk_metadata.access_permissions = permissions
        
        # Timestamps
        try:
            chunk_metadata.created_at = datetime.fromisoformat(metadata.get("created_at", ""))
        except (ValueError, TypeError):
            chunk_metadata.created_at = datetime.utcnow()
        
        try:
            chunk_metadata.updated_at = datetime.fromisoformat(metadata.get("updated_at", ""))
        except (ValueError, TypeError):
            chunk_metadata.updated_at = datetime.utcnow()
        
        return chunk_metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "source_document_id": self.source_document_id,
            "document_title": self.document_title,
            "chunk_sequence_id": self.chunk_sequence_id,
            "content_type": str(self.content_type),
            "code_language": self.code_language,
            "header_hierarchy": self.header_hierarchy.to_dict(),
            "chunk_size": self.chunk_size,
            "start_char_index": self.start_char_index,
            "end_char_index": self.end_char_index,
            "user_id": self.user_id,
            "team_id": self.team_id,
            "access_permissions": [str(perm) for perm in self.access_permissions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update chunk metadata from dictionary data.
        
        Useful for integrating with hybrid chunker results (FR-KB-002.1).
        Updates only the provided fields, leaving others unchanged.
        
        Args:
            data: Dictionary with metadata fields to update
        """
        # Update header hierarchy if provided
        if 'header_hierarchy' in data:
            hierarchy_data = data['header_hierarchy']
            if isinstance(hierarchy_data, list):
                self.header_hierarchy.headers = hierarchy_data
            elif isinstance(hierarchy_data, HeaderHierarchy):
                self.header_hierarchy = hierarchy_data
        
        # Update content type if provided
        if 'content_type' in data:
            content_type_value = data['content_type']
            if isinstance(content_type_value, str):
                try:
                    self.content_type = ContentType(content_type_value)
                except ValueError:
                    logger.warning(f"Invalid content type: {content_type_value}")
            elif isinstance(content_type_value, ContentType):
                self.content_type = content_type_value
        
        # Update other fields directly
        field_mappings = {
            'source_document_id': 'source_document_id',
            'document_title': 'document_title', 
            'code_language': 'code_language',
            'section_title': 'section_title',  # Custom field for hybrid chunker
            'section_level': 'section_level',   # Custom field for hybrid chunker
            'is_atomic_unit': 'is_atomic_unit', # Custom field for hybrid chunker
            'chunk_size': 'chunk_size',
            'start_char_index': 'start_char_index',
            'end_char_index': 'end_char_index'
        }
        
        for data_key, attr_name in field_mappings.items():
            if data_key in data and data[data_key] is not None:
                setattr(self, attr_name, data[data_key])
        
        # Update timestamp
        self.update_timestamp()
        
        logger.debug(f"Updated chunk metadata from dict with keys: {list(data.keys())}") 