"""
Recursive Chunker Module

Contains the RecursiveChunker class - the main chunking engine that orchestrates
the entire chunking process using boundary detection and configuration.
"""

import logging
from typing import List, Optional

from .config import ChunkConfig
from .result import ChunkResult
from .boundary import ChunkBoundary

logger = logging.getLogger(__name__)


class RecursiveChunker:
    """
    Main recursive chunking engine that orchestrates the chunking process.
    
    Provides intelligent document chunking with configurable boundary detection,
    overlap management, and quality assessment. Uses the ChunkBoundary system
    for optimal splitting points and generates comprehensive ChunkResult objects.
    
    Key Features:
    - Recursive chunking with intelligent boundary detection
    - Configurable overlap between chunks with smart positioning
    - Content-aware processing for different text types
    - Quality assessment and validation of chunks
    - Performance optimization with caching and metrics
    - Comprehensive logging and debugging support
    
    Attributes:
        config: ChunkConfig instance with chunking preferences
        boundary_detector: ChunkBoundary instance for boundary detection
        _chunk_count: Running count of chunks created
        _total_processed: Total characters processed
    
    Example:
        >>> config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        >>> chunker = RecursiveChunker(config)
        >>> chunks = chunker.chunk_text("Your long text content here...")
        >>> for chunk in chunks:
        ...     print(f"Chunk {chunk.chunk_index}: {chunk.get_preview()}")
    """
    
    def __init__(self, config: ChunkConfig) -> None:
        """
        Initialize recursive chunker with configuration.
        
        Args:
            config: ChunkConfig instance with chunking preferences
            
        Raises:
            TypeError: If config is not a ChunkConfig instance
        """
        if not isinstance(config, ChunkConfig):
            raise TypeError(f"Config must be ChunkConfig, got: {type(config)}")
        
        self.config = config
        self.boundary_detector = ChunkBoundary(config)
        
        # Statistics tracking
        self._chunk_count = 0
        self._total_processed = 0
        
        logger.debug(f"RecursiveChunker initialized with chunk_size={config.chunk_size}, overlap={config.chunk_overlap}")
    
    def chunk_text(self, text: str, source_section: Optional[str] = None) -> List[ChunkResult]:
        """
        Chunk text into a list of ChunkResult objects with intelligent boundary detection.
        
        Main entry point for chunking operations. Handles the complete chunking process
        including boundary detection, overlap management, and quality assessment.
        
        Args:
            text: Text content to chunk
            source_section: Optional reference to source DocumentSection
            
        Returns:
            List of ChunkResult objects representing the chunks
            
        Raises:
            ValueError: If text is empty or invalid
            TypeError: If text is not a string
        """
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got: {type(text)}")
        
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        text_length = len(text)
        
        # Check if text is smaller than minimum chunk size
        if text_length <= self.config.min_chunk_size:
            logger.debug(f"Text length ({text_length}) smaller than min_chunk_size ({self.config.min_chunk_size}), returning single chunk")
            return [self._create_single_chunk(text, source_section)]
        
        # Check if text is smaller than chunk size
        if text_length <= self.config.chunk_size:
            logger.debug(f"Text length ({text_length}) smaller than chunk_size ({self.config.chunk_size}), returning single chunk")
            return [self._create_single_chunk(text, source_section)]
        
        logger.debug(f"Starting chunking: text_length={text_length}, target_chunks≈{text_length // self.config.chunk_size}")
        
        chunks = []
        current_position = 0
        chunk_index = 0
        
        while current_position < text_length:
            # Calculate target end position for this chunk
            target_end = min(current_position + self.config.chunk_size, text_length)
            
            # Find optimal boundary near target position
            if target_end < text_length:
                optimal_boundary = self.boundary_detector.find_optimal_boundary(text, target_end)
            else:
                optimal_boundary = text_length
            
            # Create chunk content
            chunk_content = text[current_position:optimal_boundary]
            
            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_content.strip()) < self.config.min_chunk_size and optimal_boundary < text_length:
                logger.debug(f"Skipping small chunk of size {len(chunk_content)} at position {current_position}")
                current_position = optimal_boundary
                continue
            
            # Calculate overlap information
            overlap_with_previous = 0
            if chunk_index > 0 and chunks:
                previous_chunk = chunks[-1]
                overlap_with_previous = self._calculate_overlap_with_previous(
                    current_position, 
                    previous_chunk.end_position,
                    text
                )
            
            # Create chunk result
            chunk_result = ChunkResult(
                content=chunk_content,
                start_position=current_position,
                end_position=optimal_boundary,
                chunk_index=chunk_index,
                overlap_with_previous=overlap_with_previous,
                source_section=source_section
            )
            
            # Set boundary type from boundary detector context if available
            # This could be enhanced by passing boundary type from boundary detector
            chunk_result.boundary_type = self._determine_boundary_type(text, optimal_boundary)
            
            chunks.append(chunk_result)
            
            # Record metrics if collector is available
            if self.config.metrics_collector:
                self.config.metrics_collector.record_chunk_created(
                    len(chunk_content),
                    chunk_result.boundary_type
                )
            
            logger.debug(
                f"Created chunk {chunk_index}: positions {current_position}-{optimal_boundary}, "
                f"size={len(chunk_content)}, overlap={overlap_with_previous}"
            )
            
            # Calculate next starting position with overlap
            if optimal_boundary >= text_length:
                break
            
            next_position = self._calculate_next_position(
                optimal_boundary,
                text,
                chunk_result
            )
            
            # Prevent infinite loops
            if next_position <= current_position:
                logger.warning(f"No progress made in chunking at position {current_position}, advancing by 1")
                next_position = current_position + 1
            
            current_position = next_position
            chunk_index += 1
        
        # Update overlap_with_next for all chunks except the last
        self._update_next_overlaps(chunks)
        
        # Update statistics
        self._chunk_count += len(chunks)
        self._total_processed += text_length
        
        logger.info(f"Chunking complete: {len(chunks)} chunks created from {text_length} characters")
        
        return chunks
    
    def _create_single_chunk(self, text: str, source_section: Optional[str] = None) -> ChunkResult:
        """Create a single chunk for text smaller than chunk size."""
        return ChunkResult(
            content=text,
            start_position=0,
            end_position=len(text),
            chunk_index=0,
            overlap_with_previous=0,
            overlap_with_next=0,
            source_section=source_section,
            boundary_type="complete_text"
        )
    
    def _calculate_overlap_with_previous(
        self, 
        current_start: int, 
        previous_end: int, 
        text: str
    ) -> int:
        """Calculate overlap with previous chunk."""
        if current_start >= previous_end:
            return 0
        
        # Overlap is the amount of text that appears in both chunks
        overlap = previous_end - current_start
        return max(0, overlap)
    
    def _calculate_next_position(
        self, 
        current_end: int, 
        text: str, 
        current_chunk: ChunkResult
    ) -> int:
        """Calculate the starting position for the next chunk considering overlap."""
        if not self.config.enable_smart_overlap:
            # Simple overlap: just go back by overlap amount
            return max(0, current_end - self.config.chunk_overlap)
        
        # Smart overlap: try to find a good boundary for overlap start
        target_overlap_start = current_end - self.config.chunk_overlap
        
        if target_overlap_start <= 0:
            return 0
        
        # Try to find a sentence or word boundary near the target overlap start
        search_range = min(50, self.config.chunk_overlap // 2)
        
        # Look for sentence boundaries first
        if self.config.preserve_sentences:
            for i in range(max(0, target_overlap_start - search_range), 
                          min(len(text), target_overlap_start + search_range)):
                if i > 0 and text[i-1:i+1] in ['. ', '! ', '? ']:
                    return i
        
        # Fall back to word boundaries
        for i in range(max(0, target_overlap_start - search_range), 
                      min(len(text), target_overlap_start + search_range)):
            if i > 0 and text[i] == ' ':
                return i
        
        # If no good boundary found, use target position
        return max(0, target_overlap_start)
    
    def _update_next_overlaps(self, chunks: List[ChunkResult]) -> None:
        """Update overlap_with_next for all chunks."""
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            if current_chunk.end_position > next_chunk.start_position:
                overlap = current_chunk.end_position - next_chunk.start_position
                current_chunk.overlap_with_next = overlap
    
    def _determine_boundary_type(self, text: str, position: int) -> str:
        """Determine what type of boundary was used at the given position."""
        if position >= len(text):
            return "end_of_text"
        
        # Check the character at and around the boundary position
        context_start = max(0, position - 5)
        context_end = min(len(text), position + 5)
        context = text[context_start:context_end]
        boundary_char = text[position] if position < len(text) else ""
        
        # Sentence boundary detection
        if any(punct in context for punct in ['. ', '! ', '? ']):
            return "sentence"
        
        # Paragraph boundary detection
        if '\n\n' in context or '\n\n' in text[max(0, position-3):position+3]:
            return "paragraph"
        
        # Word boundary detection
        if boundary_char == ' ' or (position > 0 and text[position-1] == ' '):
            return "word"
        
        # Markdown/markup boundary detection
        if any(marker in context for marker in ['#', '```', '|', '*', '-']):
            return "markup"
        
        return "character"
    
    def get_statistics(self) -> dict:
        """
        Get chunking statistics.
        
        Returns:
            Dictionary with chunking performance statistics
        """
        stats = {
            "total_chunks_created": self._chunk_count,
            "total_characters_processed": self._total_processed,
            "average_characters_per_chunk": (
                self._total_processed / self._chunk_count if self._chunk_count > 0 else 0
            ),
            "configuration": {
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "min_chunk_size": self.config.min_chunk_size,
                "boundary_strategy": self.config.boundary_strategy.value,
                "preserve_sentences": self.config.preserve_sentences,
                "preserve_paragraphs": self.config.preserve_paragraphs,
                "preserve_code_blocks": self.config.preserve_code_blocks
            }
        }
        
        # Add boundary detector statistics if available
        if hasattr(self.boundary_detector, 'get_performance_stats'):
            stats["boundary_detection"] = self.boundary_detector.get_performance_stats()
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset chunking statistics."""
        self._chunk_count = 0
        self._total_processed = 0
        
        # Clear boundary detector cache if available
        if hasattr(self.boundary_detector, 'clear_cache'):
            self.boundary_detector.clear_cache()
        
        logger.debug("Chunking statistics reset")
    
    def chunk_with_validation(self, text: str, source_section: Optional[str] = None) -> List[ChunkResult]:
        """
        Chunk text with additional validation and quality checks.
        
        Args:
            text: Text content to chunk
            source_section: Optional reference to source DocumentSection
            
        Returns:
            List of validated ChunkResult objects
            
        Raises:
            ValueError: If validation fails
        """
        # Validate input
        compatibility_warnings = self.config.validate_compatibility_with_text(text)
        if compatibility_warnings:
            for warning in compatibility_warnings:
                logger.warning(f"Text compatibility: {warning}")
        
        # Perform chunking
        chunks = self.chunk_text(text, source_section)
        
        # Validate results
        validation_errors = self._validate_chunks(chunks, text)
        if validation_errors:
            error_msg = f"Chunk validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Chunk validation passed for {len(chunks)} chunks")
        return chunks
    
    def _validate_chunks(self, chunks: List[ChunkResult], original_text: str) -> List[str]:
        """
        Validate that chunks properly represent the original text.
        
        Args:
            chunks: List of chunk results to validate
            original_text: Original text that was chunked
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not chunks:
            errors.append("No chunks were created")
            return errors
        
        # Check chunk sequence
        expected_position = 0
        total_content_length = 0
        
        for i, chunk in enumerate(chunks):
            # Check chunk index sequence
            if chunk.chunk_index != i:
                errors.append(f"Chunk {i} has incorrect index {chunk.chunk_index}")
            
            # Check position consistency
            if chunk.start_position < expected_position - self.config.chunk_overlap:
                errors.append(f"Chunk {i} starts too early: {chunk.start_position} < {expected_position - self.config.chunk_overlap}")
            
            # Check content matches original text
            actual_content = original_text[chunk.start_position:chunk.end_position]
            if chunk.content != actual_content:
                errors.append(f"Chunk {i} content doesn't match original text slice")
            
            # Update expected position for next chunk
            expected_position = chunk.end_position
            total_content_length += len(chunk.content)
        
        # Check that last chunk ends at text end
        if chunks and chunks[-1].end_position != len(original_text):
            errors.append(f"Last chunk doesn't end at text end: {chunks[-1].end_position} != {len(original_text)}")
        
        # Check for excessive overlaps
        total_overlap = sum(chunk.overlap_with_previous + chunk.overlap_with_next for chunk in chunks)
        if total_overlap > len(original_text) * 0.8:  # More than 80% overlap seems excessive
            errors.append(f"Excessive overlap detected: {total_overlap} vs text length {len(original_text)}")
        
        return errors
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RecursiveChunker(chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}, "
            f"strategy={self.config.boundary_strategy.value})"
        ) 