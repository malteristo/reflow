"""
Hybrid Chunker Module - Complete Implementation of FR-KB-002.1

This module implements the hybrid chunking strategy as specified in FR-KB-002.1:
- Markdown-aware splitting by headers combined with recursive chunking and atomic unit handling
- Special handling for atomic code blocks and tables
- Rich metadata extraction including header hierarchy and content types

The HybridChunker orchestrates multiple components:
- HeaderBasedSplitter for markdown structure
- RecursiveChunker for prose content 
- AtomicUnitHandler for code blocks and tables
- MetadataExtractor for rich metadata

Usage:
    >>> config = ChunkConfig(chunk_size=512, chunk_overlap=50)
    >>> chunker = HybridChunker(config)
    >>> result = chunker.chunk_document(markdown_text, document_id="doc1")
    >>> print(f"Created {len(result.chunks)} chunks with metadata")
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .config import ChunkConfig
from .result import ChunkResult
from .chunker import RecursiveChunker
from ..document_structure import HeaderBasedSplitter, DocumentSection, DocumentTree
from ..markdown_parser import MarkdownParser
from ..atomic_units import AtomicUnitHandler, AtomicUnit, AtomicUnitType
from ..metadata import MetadataExtractor, MetadataExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class HybridChunkResult:
    """
    Result of hybrid chunking operation with rich metadata.
    
    Contains chunks with their content, metadata, and structural information
    as required by FR-KB-002.3 for vector database storage.
    
    Attributes:
        chunks: List of chunk results with content and metadata
        document_metadata: Extracted frontmatter and inline metadata
        document_tree: Hierarchical structure of the document
        atomic_units: Detected code blocks, tables, and other atomic units
        processing_stats: Statistics about the chunking operation
    """
    chunks: List[ChunkResult] = field(default_factory=list)
    document_metadata: Optional[Any] = None  # DocumentMetadata type
    document_tree: Optional[DocumentTree] = None
    atomic_units: List[AtomicUnit] = field(default_factory=list)
    processing_stats: Dict[str, Any] = field(default_factory=dict)


class HybridChunker:
    """
    Hybrid document chunker implementing FR-KB-002.1 specification.
    
    Combines multiple chunking strategies in a coordinated pipeline:
    1. Metadata extraction from frontmatter and inline tags
    2. Markdown structure parsing with header-based sectioning
    3. Atomic unit detection for code blocks and tables
    4. Recursive character chunking for prose content
    5. Rich metadata attachment per chunk
    
    The chunker preserves document structure while ensuring manageable chunk sizes
    and handles edge cases like oversized atomic units and empty sections.
    
    Features:
    - Header hierarchy preservation in chunk metadata
    - Atomic unit boundary protection during chunking
    - Content type classification (prose, code_block, table)
    - Configurable chunking parameters and strategies
    - Rich metadata extraction and attachment
    - Performance metrics and debugging support
    
    Example:
        >>> config = ChunkConfig(
        ...     chunk_size=512,
        ...     chunk_overlap=50,
        ...     preserve_code_blocks=True,
        ...     preserve_tables=True
        ... )
        >>> chunker = HybridChunker(config)
        >>> result = chunker.chunk_document(markdown_content, document_id="example")
        >>> for chunk in result.chunks:
        ...     print(f"Chunk: {chunk.metadata['content_type']} in {chunk.metadata['header_hierarchy']}")
    """
    
    def __init__(self, config: ChunkConfig) -> None:
        """
        Initialize hybrid chunker with configuration and component setup.
        
        Args:
            config: ChunkConfig instance with chunking preferences
            
        Raises:
            TypeError: If config is not a ChunkConfig instance
        """
        if not isinstance(config, ChunkConfig):
            raise TypeError(f"Config must be ChunkConfig, got: {type(config)}")
        
        self.config = config
        
        # Initialize component pipeline
        self.markdown_parser = MarkdownParser()
        self.header_splitter = HeaderBasedSplitter(self.markdown_parser)
        self.recursive_chunker = RecursiveChunker(config)
        self.atomic_unit_handler = AtomicUnitHandler()
        self.metadata_extractor = MetadataExtractor()
        
        # Processing statistics
        self._stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'atomic_units_preserved': 0,
            'sections_processed': 0
        }
        
        logger.debug(f"HybridChunker initialized with chunk_size={config.chunk_size}")
    
    def chunk_document(
        self, 
        document_content: str, 
        document_id: Optional[str] = None,
        source_path: Optional[Union[str, Path]] = None
    ) -> HybridChunkResult:
        """
        Perform hybrid chunking on a markdown document.
        
        Implements the complete FR-KB-002.1 pipeline:
        1. Extract metadata (frontmatter, inline tags)
        2. Parse document structure (headers, sections)
        3. Detect atomic units (code blocks, tables)
        4. Chunk sections with atomic unit preservation
        5. Attach rich metadata to each chunk
        
        Args:
            document_content: Raw markdown document text
            document_id: Optional identifier for the document
            source_path: Optional path to source file for metadata
            
        Returns:
            HybridChunkResult with chunks and metadata
            
        Raises:
            ValueError: If document_content is empty or invalid
            TypeError: If document_content is not a string
        """
        if not isinstance(document_content, str):
            raise TypeError(f"Document content must be string, got: {type(document_content)}")
        
        if not document_content.strip():
            logger.warning("Empty document provided for hybrid chunking")
            return HybridChunkResult()
        
        logger.info(f"Starting hybrid chunking for document: {document_id or 'unnamed'}")
        start_time = self._get_timestamp()
        
        # Step 1: Extract metadata (frontmatter and inline tags)
        metadata_result = self.metadata_extractor.extract_all(
            document_content, 
            document_id=document_id or "unknown"
        )
        
        # Use content without frontmatter for further processing
        content_for_processing = metadata_result.content_without_frontmatter
        
        # Step 2: Parse document structure using header-based splitting
        document_tree = self.header_splitter.split_and_build_tree(content_for_processing)
        
        # Step 3: Detect atomic units across the entire document
        atomic_units = self.atomic_unit_handler.detect_atomic_units(content_for_processing)
        
        # Step 4: Process each section with hybrid chunking
        all_chunks = []
        section_index = 0
        
        for section in document_tree.get_all_sections():
            if not section.content.strip():
                continue
                
            logger.debug(f"Processing section: {section.title} (level {section.level})")
            
            # Get atomic units within this section
            section_atomic_units = self._get_atomic_units_in_section(
                atomic_units, section, content_for_processing
            )
            
            # Chunk this section with atomic unit preservation
            section_chunks = self._chunk_section_with_atomic_preservation(
                section, 
                section_atomic_units,
                section_index
            )
            
            # Attach rich metadata to each chunk
            for chunk in section_chunks:
                self._attach_rich_metadata(
                    chunk, 
                    section, 
                    document_tree,
                    metadata_result.document_metadata,
                    document_id,
                    source_path
                )
            
            all_chunks.extend(section_chunks)
            section_index += 1
        
        # Step 5: Handle content that doesn't fit in any section (rare edge case)
        if not all_chunks and content_for_processing.strip():
            logger.debug("Document has content but no sections, creating fallback chunks")
            fallback_chunks = self.recursive_chunker.chunk_text(content_for_processing)
            for chunk in fallback_chunks:
                self._attach_basic_metadata(
                    chunk, 
                    metadata_result.document_metadata,
                    document_id,
                    source_path
                )
            all_chunks.extend(fallback_chunks)
        
        # Update processing statistics
        processing_time = self._get_timestamp() - start_time
        self._update_stats(len(all_chunks), len(atomic_units), len(document_tree.get_all_sections()))
        
        result = HybridChunkResult(
            chunks=all_chunks,
            document_metadata=metadata_result.document_metadata,
            document_tree=document_tree,
            atomic_units=atomic_units,
            processing_stats={
                'processing_time_ms': processing_time,
                'total_chunks': len(all_chunks),
                'atomic_units_detected': len(atomic_units),
                'sections_processed': len(document_tree.get_all_sections()),
                'metadata_extracted': bool(metadata_result.document_metadata.frontmatter)
            }
        )
        
        logger.info(f"Hybrid chunking complete: {len(all_chunks)} chunks created in {processing_time}ms")
        return result
    
    def _get_atomic_units_in_section(
        self, 
        atomic_units: List[AtomicUnit], 
        section: DocumentSection,
        full_content: str
    ) -> List[AtomicUnit]:
        """
        Filter atomic units that belong to a specific section.
        
        Args:
            atomic_units: All atomic units detected in document
            section: DocumentSection to check
            full_content: Full document content for position calculation
            
        Returns:
            List of atomic units within the section boundaries
        """
        # Find section boundaries in full content
        section_start = full_content.find(section.content)
        if section_start == -1:
            return []
        
        section_end = section_start + len(section.content)
        
        # Filter atomic units that fall within section boundaries
        section_units = []
        for unit in atomic_units:
            if (unit.start_position >= section_start and 
                unit.end_position <= section_end):
                # Adjust positions relative to section start
                adjusted_unit = AtomicUnit(
                    unit_type=unit.unit_type,
                    content=unit.content,
                    start_position=unit.start_position - section_start,
                    end_position=unit.end_position - section_start,
                    metadata=unit.metadata.copy()
                )
                section_units.append(adjusted_unit)
        
        return section_units
    
    def _chunk_section_with_atomic_preservation(
        self,
        section: DocumentSection,
        atomic_units: List[AtomicUnit],
        section_index: int
    ) -> List[ChunkResult]:
        """
        Chunk a document section while preserving atomic unit boundaries.
        
        This implements the core hybrid strategy:
        1. Identify atomic units (code blocks, tables) that must stay intact
        2. Use recursive chunker for prose content
        3. Handle oversized atomic units appropriately
        
        Args:
            section: DocumentSection to chunk
            atomic_units: Atomic units within this section
            section_index: Index of section in document
            
        Returns:
            List of ChunkResult objects for this section
        """
        if not section.content.strip():
            return []
        
        # If no atomic units, use straight recursive chunking
        if not atomic_units:
            chunks = self.recursive_chunker.chunk_text(section.content, source_section=section)
            return chunks
        
        # Preserve atomic unit configuration
        if not (self.config.preserve_code_blocks or self.config.preserve_tables):
            # Configuration disabled atomic preservation, use recursive chunking
            chunks = self.recursive_chunker.chunk_text(section.content, source_section=section)
            return chunks
        
        # Strategy: Split content around atomic units, chunk prose parts separately
        content = section.content
        chunks = []
        current_position = 0
        
        # Sort atomic units by position
        sorted_units = sorted(atomic_units, key=lambda u: u.start_position)
        
        for unit in sorted_units:
            # Chunk prose content before this atomic unit
            if unit.start_position > current_position:
                prose_content = content[current_position:unit.start_position]
                if prose_content.strip():
                    prose_chunks = self.recursive_chunker.chunk_text(prose_content.strip())
                    
                    # Adjust positions and add to results
                    for chunk in prose_chunks:
                        chunk.start_position += current_position
                        chunk.end_position += current_position
                        chunk.metadata = chunk.metadata or {}
                        chunk.metadata['content_type'] = 'prose'
                        chunk.metadata['section_title'] = section.title
                        chunk.metadata['section_level'] = section.level
                    
                    chunks.extend(prose_chunks)
            
            # Handle the atomic unit
            if self._should_preserve_atomic_unit(unit):
                atomic_chunk = self._create_atomic_unit_chunk(
                    unit, 
                    current_position=unit.start_position,
                    section=section
                )
                chunks.append(atomic_chunk)
            else:
                # Atomic unit too large, apply special chunking
                large_unit_chunks = self._chunk_oversized_atomic_unit(unit, section)
                chunks.extend(large_unit_chunks)
            
            current_position = unit.end_position
        
        # Handle remaining prose content after last atomic unit
        if current_position < len(content):
            remaining_content = content[current_position:]
            if remaining_content.strip():
                remaining_chunks = self.recursive_chunker.chunk_text(remaining_content.strip())
                
                # Adjust positions
                for chunk in remaining_chunks:
                    chunk.start_position += current_position
                    chunk.end_position += current_position
                    chunk.metadata = chunk.metadata or {}
                    chunk.metadata['content_type'] = 'prose'
                    chunk.metadata['section_title'] = section.title
                    chunk.metadata['section_level'] = section.level
                
                chunks.extend(remaining_chunks)
        
        # If no atomic units were found or all were handled, ensure we have content
        if not chunks and section.content.strip():
            fallback_chunks = self.recursive_chunker.chunk_text(section.content)
            for chunk in fallback_chunks:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata['content_type'] = 'prose'
                chunk.metadata['section_title'] = section.title
                chunk.metadata['section_level'] = section.level
            chunks.extend(fallback_chunks)
        
        # Assign proper chunk indices within section
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata = chunk.metadata or {}
            chunk.metadata['section_index'] = section_index
        
        return chunks
    
    def _should_preserve_atomic_unit(self, unit: AtomicUnit) -> bool:
        """
        Determine if an atomic unit should be preserved intact.
        
        Args:
            unit: AtomicUnit to evaluate
            
        Returns:
            True if unit should be kept as single chunk
        """
        # Check configuration flags
        if unit.unit_type == AtomicUnitType.CODE_BLOCK and not self.config.preserve_code_blocks:
            return False
        if unit.unit_type == AtomicUnitType.TABLE and not self.config.preserve_tables:
            return False
        
        # Check size constraints - preserve if within reasonable size
        unit_size = len(unit.content)
        max_atomic_size = self.config.chunk_size * 2  # Allow up to 2x chunk size for atomic units
        
        return unit_size <= max_atomic_size
    
    def _create_atomic_unit_chunk(
        self, 
        unit: AtomicUnit, 
        current_position: int,
        section: DocumentSection
    ) -> ChunkResult:
        """
        Create a chunk for an atomic unit.
        
        Args:
            unit: AtomicUnit to convert to chunk
            current_position: Position in section content
            section: Parent DocumentSection
            
        Returns:
            ChunkResult for the atomic unit
        """
        chunk = ChunkResult(
            content=unit.content,
            start_position=unit.start_position,
            end_position=unit.end_position,
            chunk_index=0,  # Will be set by caller
            source_section=section,
            boundary_type='atomic_unit'
        )
        
        # Set atomic unit specific metadata
        chunk.metadata = {
            'content_type': unit.unit_type.value,
            'section_title': section.title,
            'section_level': section.level,
            'is_atomic_unit': True,
            **unit.metadata  # Include unit-specific metadata (e.g., code_language)
        }
        
        return chunk
    
    def _chunk_oversized_atomic_unit(
        self, 
        unit: AtomicUnit, 
        section: DocumentSection
    ) -> List[ChunkResult]:
        """
        Handle atomic units that are too large to preserve intact.
        
        For code blocks: Try to split by functions/classes if possible
        For tables: Split while preserving headers
        For other units: Fall back to recursive chunking
        
        Args:
            unit: Oversized AtomicUnit
            section: Parent DocumentSection
            
        Returns:
            List of ChunkResult objects for the split unit
        """
        logger.debug(f"Splitting oversized {unit.unit_type.value} of {len(unit.content)} characters")
        
        if unit.unit_type == AtomicUnitType.CODE_BLOCK:
            return self._split_code_block(unit, section)
        elif unit.unit_type == AtomicUnitType.TABLE:
            return self._split_table(unit, section)
        else:
            # Fall back to recursive chunking for other types
            recursive_chunks = self.recursive_chunker.chunk_text(unit.content)
            for chunk in recursive_chunks:
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.update({
                    'content_type': unit.unit_type.value,
                    'section_title': section.title,
                    'section_level': section.level,
                    'is_split_atomic_unit': True
                })
            return recursive_chunks
    
    def _split_code_block(self, unit: AtomicUnit, section: DocumentSection) -> List[ChunkResult]:
        """Split code block by logical boundaries (functions, classes, etc.)."""
        # For now, use recursive chunking - could be enhanced with AST parsing
        chunks = self.recursive_chunker.chunk_text(unit.content)
        
        for chunk in chunks:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata.update({
                'content_type': 'code_block',
                'section_title': section.title,
                'section_level': section.level,
                'is_split_atomic_unit': True,
                'code_language': unit.metadata.get('language', 'unknown')
            })
        
        return chunks
    
    def _split_table(self, unit: AtomicUnit, section: DocumentSection) -> List[ChunkResult]:
        """Split table while preserving header context."""
        # For now, use recursive chunking - could be enhanced with table-aware splitting
        chunks = self.recursive_chunker.chunk_text(unit.content)
        
        for chunk in chunks:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata.update({
                'content_type': 'table',
                'section_title': section.title,
                'section_level': section.level,
                'is_split_atomic_unit': True,
                'column_count': unit.metadata.get('column_count', 0),
                'row_count': unit.metadata.get('row_count', 0)
            })
        
        return chunks
    
    def _attach_rich_metadata(
        self,
        chunk: ChunkResult,
        section: DocumentSection,
        document_tree: DocumentTree,
        document_metadata: Any,  # DocumentMetadata type
        document_id: Optional[str],
        source_path: Optional[Union[str, Path]]
    ) -> None:
        """
        Attach rich metadata to chunk as required by FR-KB-002.3.
        
        Metadata includes:
        - source_document_id, document_title
        - header_hierarchy (structured)
        - chunk_sequence_id, content_type
        - code_language (if applicable)
        
        Args:
            chunk: ChunkResult to enhance
            section: Source DocumentSection
            document_tree: Full document structure
            document_metadata: Extracted document metadata
            document_id: Document identifier
            source_path: Source file path
        """
        # Initialize metadata if not exists
        if chunk.metadata is None:
            chunk.metadata = {}
        
        # Build header hierarchy for this section
        header_hierarchy = self._build_header_hierarchy(section, document_tree)
        
        # Required metadata per FR-KB-002.3
        rich_metadata = {
            'source_document_id': document_id or 'unknown',
            'document_title': getattr(document_metadata, 'title', None) or 
                             (Path(source_path).stem if source_path else 'untitled'),
            'header_hierarchy': header_hierarchy,
            'chunk_sequence_id': chunk.chunk_index,
            'content_type': chunk.metadata.get('content_type', 'prose'),
        }
        
        # Add code language if this is a code block
        if chunk.metadata.get('content_type') == 'code_block':
            rich_metadata['code_language'] = chunk.metadata.get('code_language', 'unknown')
        
        # Add source path if available
        if source_path:
            rich_metadata['source_path'] = str(source_path)
        
        # Add section-specific metadata
        rich_metadata.update({
            'section_title': section.title,
            'section_level': section.level,
            'section_line_number': section.line_number,
        })
        
        # Merge with existing metadata
        chunk.metadata.update(rich_metadata)
    
    def _attach_basic_metadata(
        self,
        chunk: ChunkResult,
        document_metadata: Any,  # DocumentMetadata type
        document_id: Optional[str],
        source_path: Optional[Union[str, Path]]
    ) -> None:
        """Attach basic metadata to chunks without section context."""
        if chunk.metadata is None:
            chunk.metadata = {}
        
        basic_metadata = {
            'source_document_id': document_id or 'unknown',
            'document_title': getattr(document_metadata, 'title', None) or 
                             (Path(source_path).stem if source_path else 'untitled'),
            'header_hierarchy': [],
            'chunk_sequence_id': chunk.chunk_index,
            'content_type': 'prose',
        }
        
        if source_path:
            basic_metadata['source_path'] = str(source_path)
        
        chunk.metadata.update(basic_metadata)
    
    def _build_header_hierarchy(self, section: DocumentSection, document_tree: DocumentTree) -> List[str]:
        """
        Build structured header hierarchy for a section.
        
        Args:
            section: DocumentSection to build hierarchy for
            document_tree: Full document tree
            
        Returns:
            List of header titles from root to current section
        """
        hierarchy = []
        
        # Find path from root to this section
        current = section
        while current:
            if current.title:  # Don't include empty titles
                hierarchy.insert(0, current.title)
            
            # Find parent section (section with lower level that comes before this one)
            parent = self._find_parent_section(current, document_tree)
            current = parent
        
        return hierarchy
    
    def _find_parent_section(self, section: DocumentSection, document_tree: DocumentTree) -> Optional[DocumentSection]:
        """Find the parent section for a given section based on header hierarchy."""
        all_sections = document_tree.get_all_sections()
        
        # Find current section index
        try:
            current_index = all_sections.index(section)
        except ValueError:
            return None
        
        # Look backwards for a section with lower level (higher in hierarchy)
        target_level = section.level - 1
        for i in range(current_index - 1, -1, -1):
            candidate = all_sections[i]
            if candidate.level == target_level:
                return candidate
        
        return None
    
    def _update_stats(self, chunks_created: int, atomic_units: int, sections: int) -> None:
        """Update processing statistics."""
        self._stats['documents_processed'] += 1
        self._stats['total_chunks_created'] += chunks_created
        self._stats['atomic_units_preserved'] += atomic_units
        self._stats['sections_processed'] += sections
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        import time
        return int(time.time() * 1000)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics for performance monitoring.
        
        Returns:
            Dictionary with processing statistics
        """
        return self._stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats = {
            'documents_processed': 0,
            'total_chunks_created': 0,
            'atomic_units_preserved': 0,
            'sections_processed': 0
        } 