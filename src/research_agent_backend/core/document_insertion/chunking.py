"""
Document chunking system and metadata creation algorithms.

This module provides intelligent document chunking functionality with 
metadata generation for the insertion pipeline.
"""

import logging
from typing import List, Optional, Tuple

from ...models.metadata_schema import (
    ChunkMetadata,
    DocumentMetadata,
    ContentType,
    create_chunk_metadata
)


class DocumentChunker:
    """Document chunking service for intelligent text segmentation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize chunker with optional logger."""
        self.logger = logger or logging.getLogger(__name__)
    
    def chunk_document(
        self, 
        text: str, 
        metadata: DocumentMetadata, 
        chunk_size: Optional[int] = None
    ) -> Tuple[List[str], List[ChunkMetadata]]:
        """
        Chunk document into smaller pieces with metadata.
        
        Args:
            text: Document text to chunk
            metadata: Document metadata for chunk creation
            chunk_size: Optional custom chunk size
            
        Returns:
            Tuple of (chunks, chunk_metadata_list)
        """
        # More predictable chunking implementation for REFACTOR PHASE
        if chunk_size is None:
            chunk_size = 500  # More reasonable default chunk size
        
        chunks = []
        chunk_metadata_list = []
        
        # More predictable sentence-based chunking
        sentences = text.split('. ')
        current_chunk = []
        current_chunk_size = 0
        chunk_sequence = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add period back if it was removed (except for last sentence)
            if not sentence.endswith('.') and sentence != sentences[-1]:
                sentence += '.'
                
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content, create a chunk
            if current_chunk_size + sentence_length > chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Create chunk metadata
                chunk_meta = create_chunk_metadata(
                    source_document_id=metadata.document_id,
                    document_title=metadata.title,
                    chunk_sequence_id=chunk_sequence,
                    content_type=ContentType.PROSE,
                    user_id=metadata.user_id
                )
                chunk_meta.chunk_size = len(chunk_text)
                chunk_metadata_list.append(chunk_meta)
                
                # Reset for next chunk
                current_chunk = [sentence]
                current_chunk_size = sentence_length
                chunk_sequence += 1
            else:
                current_chunk.append(sentence)
                current_chunk_size += sentence_length + 1  # +1 for space
        
        # Add final chunk if any content remains
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            
            chunk_meta = create_chunk_metadata(
                source_document_id=metadata.document_id,
                document_title=metadata.title,
                chunk_sequence_id=chunk_sequence,
                content_type=ContentType.PROSE,
                user_id=metadata.user_id
            )
            chunk_meta.chunk_size = len(chunk_text)
            chunk_metadata_list.append(chunk_meta)
        
        return chunks, chunk_metadata_list


class ChunkMetadataFactory:
    """Factory for creating chunk metadata from document metadata."""
    
    @staticmethod
    def create_chunk_metadata(
        doc_metadata: DocumentMetadata, 
        sequence_id: int
    ) -> ChunkMetadata:
        """
        Create chunk metadata from document metadata.
        
        Args:
            doc_metadata: Source document metadata
            sequence_id: Chunk sequence identifier
            
        Returns:
            ChunkMetadata instance
        """
        return create_chunk_metadata(
            source_document_id=doc_metadata.document_id,
            document_title=doc_metadata.title,
            chunk_sequence_id=sequence_id,
            content_type=ContentType.PROSE,
            user_id=doc_metadata.user_id
        ) 