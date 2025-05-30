"""
Chunk Boundary Detection Module

Contains the ChunkBoundary class for intelligent boundary detection in text chunking.
"""

import re
import time
import logging
from typing import List, Dict, Any, Tuple

from .config import ChunkConfig, BoundaryStrategy

logger = logging.getLogger(__name__)


class ChunkBoundary:
    """Advanced boundary detection system for chunking operations."""
    
    def __init__(self, config: ChunkConfig) -> None:
        if not isinstance(config, ChunkConfig):
            raise TypeError(f"Config must be ChunkConfig, got: {type(config)}")
        
        self.config = config
        self._compiled_patterns = self._compile_boundary_patterns()
        self._boundary_cache: Dict[str, int] = {} if config.cache_boundary_patterns else None
        
        logger.debug(f"ChunkBoundary initialized with strategy: {config.boundary_strategy.value}")
    
    def _compile_boundary_patterns(self) -> Dict[str, re.Pattern]:
        """Pre-compile all regex patterns for optimal performance."""
        patterns = {}
        
        # Sentence boundaries
        patterns['sentence_simple'] = re.compile(r'[.!?]+\s+')
        patterns['sentence_complex'] = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Paragraph boundaries
        patterns['paragraph_double_newline'] = re.compile(r'\n\s*\n')
        patterns['paragraph_markdown'] = re.compile(r'\n(?=\s*(?:#|\*|-|\d+\.))')
        
        # Word boundaries
        patterns['word_space'] = re.compile(r'\s+')
        
        # Code and markup boundaries
        patterns['code_block_fenced'] = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        patterns['table_row'] = re.compile(r'\|.*\|')
        patterns['markdown_header'] = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
        
        return patterns
    
    def find_optimal_boundary(self, text: str, target_position: int) -> int:
        """Find the optimal boundary position near the target position."""
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got: {type(text)}")
        
        if target_position < 0 or target_position > len(text):
            raise ValueError(f"target_position ({target_position}) must be between 0 and {len(text)}")
        
        if target_position >= len(text):
            return len(text)
        
        # Strategy-based boundary detection
        boundary_candidates = []
        
        if self.config.boundary_strategy == BoundaryStrategy.INTELLIGENT:
            boundary_candidates = self._find_intelligent_boundaries(text, target_position)
        elif self.config.boundary_strategy == BoundaryStrategy.SENTENCE_ONLY:
            boundary_candidates = self._find_sentence_boundaries(text, target_position)
        elif self.config.boundary_strategy == BoundaryStrategy.PARAGRAPH_ONLY:
            boundary_candidates = self._find_paragraph_boundaries(text, target_position)
        elif self.config.boundary_strategy == BoundaryStrategy.WORD_ONLY:
            boundary_candidates = self._find_word_boundaries(text, target_position)
        elif self.config.boundary_strategy == BoundaryStrategy.MARKUP_AWARE:
            boundary_candidates = self._find_markup_aware_boundaries(text, target_position)
        
        # Select best boundary from candidates
        optimal_boundary = self._select_optimal_boundary(
            boundary_candidates, 
            target_position, 
            text
        )
        
        return optimal_boundary
    
    def _find_intelligent_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Use intelligent multi-strategy approach to find optimal boundaries."""
        candidates = []
        
        # Check for protected elements first
        if self.config.preserve_code_blocks or self.config.preserve_tables:
            protected = self._find_protected_elements(text, target_position)
            candidates.extend(protected)
        
        # Add paragraph boundaries if enabled
        if self.config.preserve_paragraphs:
            para_boundaries = self._find_paragraph_boundaries(text, target_position)
            candidates.extend(para_boundaries)
        
        # Add sentence boundaries if enabled
        if self.config.preserve_sentences:
            sent_boundaries = self._find_sentence_boundaries(text, target_position)
            candidates.extend(sent_boundaries)
        
        # Add word boundaries as fallback
        word_boundaries = self._find_word_boundaries(text, target_position)
        candidates.extend(word_boundaries)
        
        return candidates
    
    def _find_protected_elements(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Find protected elements that should not be split."""
        protected = []
        search_range = (
            max(0, target_position - self.config.max_boundary_search_distance),
            min(len(text), target_position + self.config.max_boundary_search_distance)
        )
        
        # Code blocks
        if self.config.preserve_code_blocks:
            for match in self._compiled_patterns['code_block_fenced'].finditer(text):
                if match.start() <= target_position <= match.end():
                    # Inside code block - find boundary before or after
                    before_distance = target_position - match.start()
                    after_distance = match.end() - target_position
                    
                    boundary_pos = match.start() if before_distance < after_distance else match.end()
                    
                    protected.append({
                        'position': boundary_pos,
                        'type': 'code_block',
                        'quality_score': 0.9,
                        'reason': 'preserve_code_block'
                    })
        
        return protected
    
    def _find_sentence_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Find sentence boundaries with quality scoring."""
        candidates = []
        search_start = max(0, target_position - self.config.max_boundary_search_distance)
        search_end = min(len(text), target_position + self.config.max_boundary_search_distance)
        search_text = text[search_start:search_end]
        
        # Try complex sentence detection first
        for match in self._compiled_patterns['sentence_complex'].finditer(search_text):
            abs_pos = search_start + match.end()
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                quality = max(0.7, 1.0 - (distance / self.config.max_boundary_search_distance))
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'sentence_complex',
                    'quality_score': quality,
                    'reason': 'Sentence boundary'
                })
        
        # Fallback to simpler sentence detection
        if not candidates:
            for match in self._compiled_patterns['sentence_simple'].finditer(search_text):
                abs_pos = search_start + match.end()
                distance = abs(abs_pos - target_position)
                
                if distance <= self.config.max_boundary_search_distance:
                    quality = max(0.5, 1.0 - (distance / self.config.max_boundary_search_distance))
                    
                    candidates.append({
                        'position': abs_pos,
                        'type': 'sentence_simple',
                        'quality_score': quality,
                        'reason': 'Simple sentence boundary'
                    })
        
        return candidates
    
    def _find_paragraph_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Find paragraph boundaries."""
        candidates = []
        search_start = max(0, target_position - self.config.max_boundary_search_distance)
        search_end = min(len(text), target_position + self.config.max_boundary_search_distance)
        search_text = text[search_start:search_end]
        
        # Double newline paragraphs
        for match in self._compiled_patterns['paragraph_double_newline'].finditer(search_text):
            abs_pos = search_start + match.start()
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                quality = max(0.8, 1.0 - (distance / self.config.max_boundary_search_distance))
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'paragraph_double_newline',
                    'quality_score': quality,
                    'reason': 'Paragraph boundary'
                })
        
        return candidates
    
    def _find_word_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Find word boundaries as fallback option."""
        candidates = []
        
        # Search backwards for space
        for i in range(target_position, max(0, target_position - 50), -1):
            if i < len(text) and text[i] == ' ':
                distance = target_position - i
                quality = max(0.3, 1.0 - (distance / 50))
                
                candidates.append({
                    'position': i,
                    'type': 'word_space_before',
                    'quality_score': quality,
                    'reason': f'Word boundary before (distance: {distance})'
                })
                break
        
        # Search forwards for space
        for i in range(target_position, min(len(text), target_position + 50)):
            if text[i] == ' ':
                distance = i - target_position
                quality = max(0.3, 1.0 - (distance / 50))
                
                candidates.append({
                    'position': i,
                    'type': 'word_space_after',
                    'quality_score': quality,
                    'reason': f'Word boundary after (distance: {distance})'
                })
                break
        
        # Fallback to exact position if no word boundaries found
        if not candidates:
            candidates.append({
                'position': target_position,
                'type': 'exact_position',
                'quality_score': 0.1,
                'reason': 'No better boundary found'
            })
        
        return candidates
    
    def _find_markup_aware_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """Find boundaries that respect markdown and HTML markup."""
        candidates = []
        
        # First check for protected elements
        protected = self._find_protected_elements(text, target_position)
        candidates.extend(protected)
        
        # Add markup-specific boundaries
        search_start = max(0, target_position - self.config.max_boundary_search_distance)
        search_end = min(len(text), target_position + self.config.max_boundary_search_distance)
        search_text = text[search_start:search_end]
        
        # Header boundaries
        for match in self._compiled_patterns['markdown_header'].finditer(search_text):
            abs_pos = search_start + match.start()
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                quality = max(0.8, 1.0 - (distance / self.config.max_boundary_search_distance))
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'markdown_header',
                    'quality_score': quality,
                    'reason': 'Header boundary'
                })
        
        # If no markup boundaries found, fall back to other strategies
        if not candidates:
            candidates.extend(self._find_paragraph_boundaries(text, target_position))
            candidates.extend(self._find_sentence_boundaries(text, target_position))
            candidates.extend(self._find_word_boundaries(text, target_position))
        
        return candidates
    
    def _select_optimal_boundary(
        self, 
        candidates: List[Dict[str, Any]], 
        target_position: int, 
        text: str
    ) -> int:
        """Select the optimal boundary from candidates."""
        if not candidates:
            logger.warning(f"No boundary candidates found, using target position {target_position}")
            return min(target_position, len(text))

        # Sort candidates by quality and configuration preferences
        def score_candidate(candidate):
            distance = abs(candidate['position'] - target_position)
            quality = candidate['quality_score']
            boundary_type = candidate['type']
            
            # Base priority based on configuration preferences
            priority_bonus = 0.0
            
            if self.config.preserve_paragraphs and 'paragraph' in boundary_type:
                priority_bonus = 2.5
            elif self.config.preserve_sentences and 'sentence' in boundary_type:
                priority_bonus = 2.0
            elif boundary_type in ['code_block', 'table']:
                priority_bonus = 1.8
            
            # Distance penalty
            distance_penalty = distance / self.config.max_boundary_search_distance
            
            return quality + priority_bonus - (distance_penalty * 0.2)

        candidates.sort(key=score_candidate, reverse=True)
        
        optimal_candidate = candidates[0]
        
        logger.debug(
            f"Selected boundary: position={optimal_candidate['position']}, "
            f"type={optimal_candidate['type']}, quality={optimal_candidate['quality_score']:.2f}"
        )
        
        return optimal_candidate['position']
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ChunkBoundary(strategy={self.config.boundary_strategy.value})" 