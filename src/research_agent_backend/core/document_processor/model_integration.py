"""
Model Change Detection Integration for Document Processing Pipeline

This module integrates the model change detection system with the document chunking
pipeline, ensuring that changes to embedding models trigger appropriate cache
invalidation and re-chunking operations.

Implements FR-KB-002 model change detection requirements for the hybrid chunking strategy.
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..model_change_detection import (
    ModelChangeDetector,
    ModelChangeEvent,
    ModelFingerprint
)
from .chunking import ChunkResult, ChunkConfig, RecursiveChunker
from .metadata import DocumentMetadata


@dataclass
class ModelAwareChunkResult(ChunkResult):
    """
    Enhanced ChunkResult that includes model fingerprint information.
    
    Extends the base ChunkResult with model tracking for cache invalidation.
    """
    model_fingerprint: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamps and model information."""
        super().__post_init__()
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class ModelChangeIntegration:
    """
    Manages integration between model change detection and document processing.
    
    This class provides:
    - Model fingerprint tracking for chunks
    - Cache invalidation on model changes  
    - Re-chunking trigger mechanisms
    - Model metadata integration with chunk results
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize model change integration.
        
        Args:
            embedding_service: Optional embedding service for model fingerprinting
        """
        self.logger = logging.getLogger(__name__)
        self.model_detector = ModelChangeDetector()
        self.embedding_service = embedding_service
        self._cached_fingerprint: Optional[ModelFingerprint] = None
        
    def get_current_model_fingerprint(self) -> Optional[ModelFingerprint]:
        """
        Get the current model fingerprint from the embedding service.
        
        Returns:
            ModelFingerprint if available, None otherwise
        """
        if not self.embedding_service:
            self.logger.warning("No embedding service configured for model fingerprinting")
            return None
            
        try:
            # Try to generate fingerprint from embedding service
            if hasattr(self.embedding_service, 'generate_model_fingerprint'):
                return self.embedding_service.generate_model_fingerprint()
            else:
                # Fallback: create fingerprint from service info
                model_info = getattr(self.embedding_service, 'get_model_info', lambda: {})()
                if model_info:
                    return ModelFingerprint(
                        model_name=model_info.get('model_name', 'unknown'),
                        model_type='local',  # Default to local for fallback
                        version=model_info.get('model_version', '1.0'),
                        checksum=self._hash_model_info(model_info)
                    )
        except Exception as e:
            self.logger.error(f"Failed to generate model fingerprint: {e}")
            
        return None
    
    def _hash_model_info(self, model_info: Dict[str, Any]) -> str:
        """Create a hash from model information."""
        info_str = str(sorted(model_info.items()))
        return hashlib.md5(info_str.encode()).hexdigest()
    
    def register_current_model(self) -> bool:
        """
        Register the current model with the change detector.
        
        Returns:
            True if successfully registered, False otherwise
        """
        fingerprint = self.get_current_model_fingerprint()
        if not fingerprint:
            return False
            
        try:
            self.model_detector.register_model(fingerprint)
            self._cached_fingerprint = fingerprint
            return True
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return False
    
    def check_for_model_changes(self) -> bool:
        """
        Check if the current model has changed since last registration.
        
        Returns:
            True if change detected, False otherwise
        """
        current_fingerprint = self.get_current_model_fingerprint()
        if not current_fingerprint:
            return False
            
        try:
            return self.model_detector.detect_change(current_fingerprint)
        except Exception as e:
            self.logger.error(f"Error checking for model changes: {e}")
            return False
    
    def should_invalidate_chunks(self, chunk_results: List[ChunkResult]) -> bool:
        """
        Determine if chunks should be invalidated due to model changes.
        
        Args:
            chunk_results: List of existing chunk results to check
            
        Returns:
            True if chunks should be invalidated, False otherwise
        """
        # Get current model fingerprint
        current_fingerprint = self.get_current_model_fingerprint()
        if not current_fingerprint:
            return False
            
        # Check if any chunks were created with a different model fingerprint
        for chunk in chunk_results:
            if isinstance(chunk, ModelAwareChunkResult):
                if chunk.model_fingerprint and chunk.model_fingerprint != current_fingerprint.checksum:
                    self.logger.info(f"Chunk invalidation required: chunk fingerprint {chunk.model_fingerprint} != current {current_fingerprint.checksum}")
                    return True
                
        return False
    
    def enhance_chunk_with_model_info(self, chunk: ChunkResult) -> ModelAwareChunkResult:
        """
        Enhance a chunk result with current model information.
        
        Args:
            chunk: Base chunk result to enhance
            
        Returns:
            Enhanced chunk with model tracking information
        """
        fingerprint = self.get_current_model_fingerprint()
        
        # Convert to ModelAwareChunkResult
        enhanced_chunk = ModelAwareChunkResult(
            content=chunk.content,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            chunk_index=chunk.chunk_index,
            overlap_with_previous=chunk.overlap_with_previous,
            overlap_with_next=chunk.overlap_with_next,
            boundary_type=chunk.boundary_type,
            source_section=chunk.source_section,
            language_detected=chunk.language_detected,
            content_type=chunk.content_type,
            quality_score=chunk.quality_score,
            processing_metadata=chunk.processing_metadata.copy() if chunk.processing_metadata else {},
            model_fingerprint=fingerprint.checksum if fingerprint else None,
            model_name=fingerprint.model_name if fingerprint else None,
            model_version=fingerprint.version if fingerprint else None
        )
        
        # Add model info to metadata
        if fingerprint:
            enhanced_chunk.processing_metadata.update({
                'model_fingerprint': fingerprint.checksum,
                'model_name': fingerprint.model_name,
                'model_version': fingerprint.version,
                'model_tracked_at': datetime.utcnow().isoformat()
            })
            
        return enhanced_chunk


class ChunkingPipelineWithModelDetection:
    """
    Enhanced chunking pipeline that integrates model change detection.
    
    This pipeline automatically:
    - Tracks model changes during chunking operations
    - Enhances chunks with model fingerprint information  
    - Provides cache invalidation recommendations
    - Maintains model consistency across chunk batches
    """
    
    def __init__(self, chunker: RecursiveChunker, embedding_service=None):
        """
        Initialize the model-aware chunking pipeline.
        
        Args:
            chunker: The recursive chunker to wrap
            embedding_service: Optional embedding service for model tracking
        """
        self.chunker = chunker
        self.model_integration = ModelChangeIntegration(embedding_service)
        self.logger = logging.getLogger(__name__)
        
        # Register current model on initialization
        self.model_integration.register_current_model()
    
    def chunk_text(self, text: str, **kwargs) -> List[ModelAwareChunkResult]:
        """
        Chunk text with model change detection integration.
        
        Args:
            text: Text to chunk
            **kwargs: Additional arguments for chunking
            
        Returns:
            List of model-aware chunk results
        """
        # Check for model changes before processing
        change_detected = self.model_integration.check_for_model_changes()
        if change_detected:
            self.logger.warning(f"Model change detected during chunking")
            # Re-register the new model
            self.model_integration.register_current_model()
        
        # Perform chunking
        base_chunks = self.chunker.chunk_text(text, **kwargs)
        
        # Enhance chunks with model information
        enhanced_chunks = []
        for chunk in base_chunks:
            enhanced_chunk = self.model_integration.enhance_chunk_with_model_info(chunk)
            enhanced_chunks.append(enhanced_chunk)
            
        self.logger.debug(f"Generated {len(enhanced_chunks)} model-aware chunks")
        return enhanced_chunks
    
    def chunk_document_sections(self, sections, **kwargs) -> List[ModelAwareChunkResult]:
        """
        Chunk document sections with model tracking.
        
        Args:
            sections: Document sections to chunk
            **kwargs: Additional chunking arguments
            
        Returns:
            List of model-aware chunk results
        """
        # Check for model changes
        change_detected = self.model_integration.check_for_model_changes()
        if change_detected:
            self.logger.warning(f"Model change detected during section chunking")
            self.model_integration.register_current_model()
        
        # Convert sections to text and perform chunking
        # For now, we'll concatenate sections and chunk as text
        # In a real implementation, this would preserve section boundaries
        if hasattr(sections, '__iter__') and not isinstance(sections, str):
            # If sections is a list/iterable, join them
            text = '\n\n'.join(str(section) for section in sections)
        else:
            # If sections is already a string
            text = str(sections)
            
        base_chunks = self.chunker.chunk_text(text, **kwargs)
        
        # Enhance with model information
        enhanced_chunks = []
        for chunk in base_chunks:
            enhanced_chunk = self.model_integration.enhance_chunk_with_model_info(chunk)
            enhanced_chunks.append(enhanced_chunk)
            
        return enhanced_chunks
    
    def invalidate_cache_if_needed(self, existing_chunks: List[ChunkResult]) -> bool:
        """
        Check if existing chunks should be invalidated due to model changes.
        
        Args:
            existing_chunks: Previously generated chunks to check
            
        Returns:
            True if cache invalidation is recommended, False otherwise
        """
        return self.model_integration.should_invalidate_chunks(existing_chunks)
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get current model status and change detection information.
        
        Returns:
            Dictionary with model status information
        """
        fingerprint = self.model_integration.get_current_model_fingerprint()
        change_detected = self.model_integration.check_for_model_changes()
        
        return {
            'current_model': {
                'name': fingerprint.model_name if fingerprint else None,
                'version': fingerprint.version if fingerprint else None,
                'fingerprint': fingerprint.checksum if fingerprint else None
            },
            'change_detected': change_detected,
            'last_change': None,  # Would need to track this separately
            'model_registered': fingerprint is not None
        } 