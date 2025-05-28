"""
Collection Type Management for Research Agent Vector Database.

This module provides collection type-specific configuration management,
including indexing strategies, HNSW parameters, and data routing logic.

Implements FR-KB-005: Collection management with type-specific configurations.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..models.metadata_schema import CollectionType, CollectionMetadata, AccessPermission
from ..utils.config import ConfigManager


logger = logging.getLogger(__name__)


@dataclass
class CollectionTypeConfig:
    """Configuration for a specific collection type."""
    
    # Basic properties
    collection_type: CollectionType
    description: str
    default_name_prefix: str
    
    # Vector configuration
    embedding_dimension: int = 384  # Default for multi-qa-MiniLM-L6-cos-v1
    distance_metric: str = "cosine"
    
    # HNSW index parameters (optimized per collection type)
    hnsw_construction_ef: int = 100
    hnsw_m: int = 16
    hnsw_search_ef: int = 50
    
    # Performance and behavior settings
    batch_insert_size: int = 100
    enable_auto_compaction: bool = True
    max_documents_per_collection: Optional[int] = None
    
    # Access and permissions
    default_permissions: List[AccessPermission] = field(default_factory=lambda: [AccessPermission.READ])
    allow_public_access: bool = False
    
    # Data routing rules
    content_type_patterns: List[str] = field(default_factory=list)
    document_type_patterns: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    
    def to_chromadb_metadata(self) -> Dict[str, Any]:
        """Convert to ChromaDB collection metadata."""
        return {
            "hnsw:space": self.distance_metric,
            "hnsw:construction_ef": self.hnsw_construction_ef,
            "hnsw:M": self.hnsw_m,
            "hnsw:search_ef": self.hnsw_search_ef,
            "collection_type": str(self.collection_type),
            "batch_insert_size": self.batch_insert_size,
            "enable_auto_compaction": self.enable_auto_compaction,
        }
    
    def create_collection_metadata(
        self,
        collection_name: str,
        owner_id: str = "",
        team_id: Optional[str] = None,
        **kwargs
    ) -> CollectionMetadata:
        """Create CollectionMetadata instance for this collection type."""
        return CollectionMetadata(
            collection_name=collection_name,
            collection_type=self.collection_type,
            description=kwargs.get('description', self.description),
            embedding_dimension=self.embedding_dimension,
            distance_metric=self.distance_metric,
            hnsw_construction_ef=self.hnsw_construction_ef,
            hnsw_m=self.hnsw_m,
            owner_id=owner_id,
            team_id=team_id,
            **kwargs
        )


class CollectionTypeManager:
    """
    Manager for collection type configurations and operations.
    
    Provides collection type-specific configuration, validation, and routing
    logic for the Research Agent vector database system.
    
    Implements FR-KB-005: Collection type management and data organization.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        """Initialize collection type manager."""
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize collection type configurations
        self._collection_configs: Dict[CollectionType, CollectionTypeConfig] = {}
        self._initialize_default_configurations()
        
        # Load custom configurations from config
        self._load_custom_configurations()
    
    def _initialize_default_configurations(self) -> None:
        """Initialize default collection type configurations."""
        
        # FUNDAMENTAL: Core knowledge that's rarely updated, optimized for retrieval
        self._collection_configs[CollectionType.FUNDAMENTAL] = CollectionTypeConfig(
            collection_type=CollectionType.FUNDAMENTAL,
            description="Core foundational knowledge for long-term reference",
            default_name_prefix="fundamental",
            embedding_dimension=384,
            distance_metric="cosine",
            # Optimized for read-heavy workloads
            hnsw_construction_ef=200,  # Higher for better index quality
            hnsw_m=32,  # Higher for better recall
            hnsw_search_ef=100,  # Higher for better search quality
            batch_insert_size=50,  # Smaller batches for careful insertion
            enable_auto_compaction=True,
            max_documents_per_collection=10000,
            default_permissions=[AccessPermission.READ],
            allow_public_access=True,
            content_type_patterns=["reference", "documentation", "standards"],
            document_type_patterns=["markdown", "pdf", "docx"],
            metadata_filters={"document_type": "reference"}
        )
        
        # PROJECT_SPECIFIC: Project-focused knowledge, balanced read/write
        self._collection_configs[CollectionType.PROJECT_SPECIFIC] = CollectionTypeConfig(
            collection_type=CollectionType.PROJECT_SPECIFIC,
            description="Project-specific knowledge and documentation",
            default_name_prefix="project",
            embedding_dimension=384,
            distance_metric="cosine",
            # Balanced for read/write operations
            hnsw_construction_ef=100,
            hnsw_m=16,
            hnsw_search_ef=50,
            batch_insert_size=100,
            enable_auto_compaction=True,
            max_documents_per_collection=5000,
            default_permissions=[AccessPermission.READ, AccessPermission.WRITE],
            allow_public_access=False,
            content_type_patterns=["code-block", "prose", "list"],
            document_type_patterns=["markdown", "code", "text"],
            metadata_filters={"team_id": "project_team"}
        )
        
        # GENERAL: General-purpose collection, optimized for frequent updates
        self._collection_configs[CollectionType.GENERAL] = CollectionTypeConfig(
            collection_type=CollectionType.GENERAL,
            description="General-purpose knowledge collection",
            default_name_prefix="general",
            embedding_dimension=384,
            distance_metric="cosine",
            # Optimized for frequent updates
            hnsw_construction_ef=50,  # Lower for faster insertion
            hnsw_m=8,  # Lower for faster updates
            hnsw_search_ef=30,
            batch_insert_size=200,  # Larger batches for efficiency
            enable_auto_compaction=False,  # Manual compaction for control
            max_documents_per_collection=None,  # No limit
            default_permissions=[AccessPermission.READ, AccessPermission.WRITE],
            allow_public_access=False,
            content_type_patterns=["prose", "list", "table"],
            document_type_patterns=["text", "markdown"],
            metadata_filters={}
        )
        
        # REFERENCE: Reference materials, optimized for high-precision retrieval
        self._collection_configs[CollectionType.REFERENCE] = CollectionTypeConfig(
            collection_type=CollectionType.REFERENCE,
            description="Reference materials and lookup data",
            default_name_prefix="reference",
            embedding_dimension=384,
            distance_metric="cosine",
            # Optimized for high precision
            hnsw_construction_ef=300,  # Very high for maximum quality
            hnsw_m=48,  # Very high for maximum recall
            hnsw_search_ef=150,  # Very high for precision
            batch_insert_size=25,  # Small batches for careful processing
            enable_auto_compaction=True,
            max_documents_per_collection=2000,
            default_permissions=[AccessPermission.READ],
            allow_public_access=True,
            content_type_patterns=["reference", "table", "metadata-block"],
            document_type_patterns=["pdf", "docx", "presentation"],
            metadata_filters={"content_type": "reference"}
        )
        
        # TEMPORARY: Temporary or experimental collections
        self._collection_configs[CollectionType.TEMPORARY] = CollectionTypeConfig(
            collection_type=CollectionType.TEMPORARY,
            description="Temporary or experimental collections",
            default_name_prefix="temp",
            embedding_dimension=384,
            distance_metric="cosine",
            # Optimized for fast operations, lower quality acceptable
            hnsw_construction_ef=25,
            hnsw_m=4,
            hnsw_search_ef=20,
            batch_insert_size=500,  # Large batches for speed
            enable_auto_compaction=False,
            max_documents_per_collection=1000,
            default_permissions=[AccessPermission.READ, AccessPermission.WRITE, AccessPermission.ADMIN],
            allow_public_access=False,
            content_type_patterns=["unknown", "prose"],
            document_type_patterns=["text", "unknown"],
            metadata_filters={}
        )
        
        self.logger.info(f"Initialized {len(self._collection_configs)} default collection type configurations")
    
    def _load_custom_configurations(self) -> None:
        """Load custom collection type configurations from config file."""
        try:
            custom_configs = self.config_manager.get('collection_types', {})
            
            for type_name, config_data in custom_configs.items():
                try:
                    collection_type = CollectionType(type_name)
                    if collection_type in self._collection_configs:
                        # Update existing configuration
                        existing_config = self._collection_configs[collection_type]
                        for key, value in config_data.items():
                            if hasattr(existing_config, key):
                                setattr(existing_config, key, value)
                        self.logger.info(f"Updated configuration for collection type: {type_name}")
                except ValueError:
                    self.logger.warning(f"Unknown collection type in config: {type_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load custom config for {type_name}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to load custom collection configurations: {e}")
    
    def get_collection_config(self, collection_type: Union[CollectionType, str]) -> CollectionTypeConfig:
        """
        Get configuration for a specific collection type.
        
        Args:
            collection_type: Collection type enum or string
            
        Returns:
            Collection type configuration
            
        Raises:
            ValueError: If collection type is unknown
        """
        if isinstance(collection_type, str):
            try:
                collection_type = CollectionType(collection_type)
            except ValueError:
                raise ValueError(f"Unknown collection type: {collection_type}")
        
        if collection_type not in self._collection_configs:
            raise ValueError(f"No configuration found for collection type: {collection_type}")
        
        return self._collection_configs[collection_type]
    
    def get_all_collection_types(self) -> List[CollectionType]:
        """Get list of all available collection types."""
        return list(self._collection_configs.keys())
    
    def determine_collection_type(
        self,
        document_metadata: Dict[str, Any],
        chunk_metadata: Optional[Dict[str, Any]] = None,
        content_analysis: Optional[Dict[str, Any]] = None
    ) -> CollectionType:
        """
        Determine appropriate collection type based on document and chunk metadata.
        
        Args:
            document_metadata: Document-level metadata
            chunk_metadata: Chunk-level metadata (optional)
            content_analysis: Content analysis results (optional)
            
        Returns:
            Recommended collection type
        """
        # Analyze document characteristics
        document_type = document_metadata.get('document_type', 'unknown').lower()
        source_path = document_metadata.get('source_path', '').lower()
        team_id = document_metadata.get('team_id')
        
        # Check for project-specific content first (higher priority than reference)
        if team_id and team_id != 'public':
            return CollectionType.PROJECT_SPECIFIC
        
        if any(pattern in source_path for pattern in ['project/', 'src/', 'code/']):
            return CollectionType.PROJECT_SPECIFIC
        
        # Check for reference materials
        if any(pattern in document_type for pattern in ['reference', 'manual', 'specification']):
            return CollectionType.REFERENCE
        
        if any(pattern in source_path for pattern in ['reference/', 'docs/', 'manual/']):
            return CollectionType.REFERENCE
        
        # Check for fundamental knowledge
        if any(pattern in source_path for pattern in ['fundamental/', 'core/', 'standards/']):
            return CollectionType.FUNDAMENTAL
        
        if document_type in ['pdf', 'docx'] and 'standard' in source_path:
            return CollectionType.FUNDAMENTAL
        
        # Check for temporary content
        if any(pattern in source_path for pattern in ['temp/', 'tmp/', 'scratch/']):
            return CollectionType.TEMPORARY
        
        # Default to general collection
        return CollectionType.GENERAL
    
    def create_collection_name(
        self,
        collection_type: Union[CollectionType, str],
        project_name: Optional[str] = None,
        suffix: Optional[str] = None
    ) -> str:
        """
        Create standardized collection name based on type and context.
        
        Args:
            collection_type: Collection type
            project_name: Project name for project-specific collections
            suffix: Additional suffix for uniqueness
            
        Returns:
            Standardized collection name
        """
        config = self.get_collection_config(collection_type)
        
        # Start with prefix
        name_parts = [config.default_name_prefix]
        
        # Add project name for project-specific collections
        if config.collection_type == CollectionType.PROJECT_SPECIFIC and project_name:
            # Sanitize project name - keep underscores from space and dash replacement
            sanitized_project = project_name.lower().replace(' ', '_').replace('-', '_')
            # Remove all non-alphanumeric characters except underscores, then clean up multiple underscores
            sanitized_project = ''.join(c for c in sanitized_project if c.isalnum() or c == '_')
            # Remove multiple consecutive underscores
            import re
            sanitized_project = re.sub(r'_+', '_', sanitized_project).strip('_')
            if sanitized_project:
                name_parts.append(sanitized_project)
        
        # Add suffix if provided
        if suffix:
            sanitized_suffix = suffix.lower().replace(' ', '_').replace('-', '_')
            sanitized_suffix = ''.join(c for c in sanitized_suffix if c.isalnum() or c == '_')
            # Remove multiple consecutive underscores
            import re
            sanitized_suffix = re.sub(r'_+', '_', sanitized_suffix).strip('_')
            if sanitized_suffix:
                name_parts.append(sanitized_suffix)
        
        return '_'.join(name_parts)
    
    def validate_collection_for_type(
        self,
        collection_metadata: CollectionMetadata,
        collection_type: Union[CollectionType, str]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that collection metadata matches the expected configuration for its type.
        
        Args:
            collection_metadata: Collection metadata to validate
            collection_type: Expected collection type
            
        Returns:
            Tuple of (is_valid, list_of_validation_errors)
        """
        config = self.get_collection_config(collection_type)
        errors = []
        
        # Check collection type matches
        if collection_metadata.collection_type != config.collection_type:
            errors.append(f"Collection type mismatch: expected {config.collection_type}, got {collection_metadata.collection_type}")
        
        # Check embedding dimension
        if collection_metadata.embedding_dimension != config.embedding_dimension:
            errors.append(f"Embedding dimension mismatch: expected {config.embedding_dimension}, got {collection_metadata.embedding_dimension}")
        
        # Check distance metric
        if collection_metadata.distance_metric != config.distance_metric:
            errors.append(f"Distance metric mismatch: expected {config.distance_metric}, got {collection_metadata.distance_metric}")
        
        # Check HNSW parameters
        if collection_metadata.hnsw_construction_ef != config.hnsw_construction_ef:
            errors.append(f"HNSW construction_ef mismatch: expected {config.hnsw_construction_ef}, got {collection_metadata.hnsw_construction_ef}")
        
        if collection_metadata.hnsw_m != config.hnsw_m:
            errors.append(f"HNSW M parameter mismatch: expected {config.hnsw_m}, got {collection_metadata.hnsw_m}")
        
        return len(errors) == 0, errors
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all collection type configurations."""
        summary = {
            'total_types': len(self._collection_configs),
            'types': {}
        }
        
        for collection_type, config in self._collection_configs.items():
            summary['types'][str(collection_type)] = {
                'description': config.description,
                'embedding_dimension': config.embedding_dimension,
                'distance_metric': config.distance_metric,
                'hnsw_construction_ef': config.hnsw_construction_ef,
                'hnsw_m': config.hnsw_m,
                'max_documents': config.max_documents_per_collection,
                'default_permissions': [str(perm) for perm in config.default_permissions],
                'allow_public_access': config.allow_public_access,
                'content_patterns': config.content_type_patterns,
                'document_patterns': config.document_type_patterns
            }
        
        return summary


def create_collection_type_manager(config_manager: Optional[ConfigManager] = None) -> CollectionTypeManager:
    """
    Factory function to create CollectionTypeManager instance.
    
    Args:
        config_manager: Optional configuration manager
        
    Returns:
        Initialized CollectionTypeManager
    """
    return CollectionTypeManager(config_manager) 