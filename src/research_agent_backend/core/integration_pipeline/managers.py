"""
Supporting managers for data preparation and collection type management.

This module provides the DataPreparationManager and CollectionTypeManager
classes that support the integration pipeline with data processing and
collection configuration services.
"""

import logging
import time
from typing import Dict, Any, List

from .models import Collection


class DataPreparationManager:
    """
    Enhanced data preparation manager for integration testing.
    
    REFACTOR PHASE: More realistic data preparation with validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data preparation manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.normalization = config.get("normalization", "unit_vector")
        self.batch_size = config.get("batch_size", 100)
    
    def prepare_for_storage(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhanced data preparation with validation and normalization.
        
        REFACTOR PHASE: More sophisticated preparation logic.
        """
        self.logger.debug(f"Preparing {len(raw_data)} items for storage")
        
        prepared_data = []
        for i, item in enumerate(raw_data):
            try:
                # Validate item structure
                if not isinstance(item, dict) or not item.get("content"):
                    self.logger.warning(f"Skipping invalid item {i}")
                    continue
                
                # Prepare item with enhanced metadata
                prepared_item = item.copy()
                prepared_item.update({
                    "prepared": True,
                    "preparation_timestamp": time.time(),
                    "normalization_method": self.normalization,
                    "batch_id": f"batch_{time.time()}",
                    "item_index": i
                })
                
                # Add content statistics
                content = str(item.get("content", ""))
                prepared_item["content_stats"] = {
                    "length": len(content),
                    "word_count": len(content.split()),
                    "has_headers": "#" in content
                }
                
                prepared_data.append(prepared_item)
                
            except Exception as e:
                self.logger.error(f"Error preparing item {i}: {e}")
                continue
        
        self.logger.info(f"Data preparation complete: {len(prepared_data)} items prepared")
        return prepared_data


class CollectionTypeManager:
    """
    Enhanced collection type manager for integration testing.
    
    REFACTOR PHASE: More sophisticated collection management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize collection type manager."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.supported_types = [
            "FUNDAMENTAL", "PROJECT_SPECIFIC", "EXPERIMENTAL", 
            "documentation", "reference", "tutorial", "code", "notes"
        ]
    
    def get_collection_config(self, collection_type: str) -> Dict[str, Any]:
        """
        Enhanced collection configuration with type-specific settings.
        
        REFACTOR PHASE: Type-aware configuration generation.
        """
        # Handle unknown collection types
        original_type = collection_type
        if collection_type not in self.supported_types:
            self.logger.warning(f"Unknown collection type: {collection_type}")
            collection_type = "documentation"  # Default fallback
        
        # Type-specific configurations with max_documents
        base_config = {
            "type": collection_type,
            "embedding_function": "sentence-transformers",
            "distance_metric": "cosine",
            "created_at": time.time(),
            "version": "1.0.0",
            "max_documents": 1000  # Default max documents
        }
        
        # Add type-specific enhancements
        if collection_type in ["FUNDAMENTAL", "documentation"]:
            base_config.update({
                "chunk_strategy": "markdown_aware",
                "metadata_fields": ["section", "category", "tags"],
                "search_boost": 1.2,
                "max_documents": 5000  # Higher limit for fundamental docs
            })
        elif collection_type in ["PROJECT_SPECIFIC", "code"]:
            base_config.update({
                "chunk_strategy": "syntax_aware",
                "metadata_fields": ["language", "function", "class"],
                "search_boost": 1.0,
                "max_documents": 2000  # Medium limit for project-specific
            })
        elif collection_type in ["EXPERIMENTAL", "reference"]:
            base_config.update({
                "chunk_strategy": "semantic",
                "metadata_fields": ["api_version", "method", "parameters"],
                "search_boost": 1.1,
                "max_documents": 1500  # Medium limit for experimental
            })
        elif collection_type == "tutorial":
            base_config.update({
                "chunk_strategy": "structured",
                "metadata_fields": ["step", "difficulty", "topic"],
                "search_boost": 1.3,
                "max_documents": 800  # Lower limit for tutorials
            })
        elif collection_type == "notes":
            base_config.update({
                "chunk_strategy": "flexible",
                "metadata_fields": ["date", "author", "topic"],
                "search_boost": 0.9,
                "max_documents": 3000  # Higher limit for notes
            })
        
        self.logger.debug(f"Generated config for collection type: {collection_type} (originally: {original_type})")
        return base_config 