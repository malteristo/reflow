"""
Utility functions for enhanced re-ranking features.

Provides keyword highlighting, source attribution extraction, and relevance
confidence analysis for improved result presentation per FR-RQ-008.
"""

import re
import html
from typing import List, Tuple, Optional, Dict, Any
from .models import HighlightedText, RelevanceIndicators, SourceAttribution
from ..integration_pipeline.models import SearchResult


class KeywordHighlighter:
    """Utility for highlighting keywords in search result content."""
    
    def __init__(self, highlight_tag: str = "mark", case_sensitive: bool = False):
        """
        Initialize keyword highlighter.
        
        Args:
            highlight_tag: HTML tag to use for highlighting (default: 'mark')
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.highlight_tag = highlight_tag
        self.case_sensitive = case_sensitive
    
    def highlight_keywords(self, text: str, keywords: List[str]) -> HighlightedText:
        """
        Highlight keywords in text with HTML markup.
        
        Args:
            text: Original text content
            keywords: List of keywords to highlight
            
        Returns:
            HighlightedText object with highlights and metadata
        """
        if not keywords:
            return HighlightedText(
                original_text=text,
                highlighted_text=html.escape(text),
                matched_keywords=[],
                highlight_positions=[]
            )
        
        # Escape HTML in original text
        escaped_text = html.escape(text)
        matched_keywords = []
        highlight_positions = []
        
        # Find all keyword matches first
        all_matches = []
        for keyword in keywords:
            if not keyword.strip():
                continue
                
            # Create regex pattern for keyword matching
            escaped_keyword = re.escape(keyword.strip())
            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = rf'\b{escaped_keyword}\b'
            
            # Find all matches for this keyword
            matches = list(re.finditer(pattern, text, flags))
            
            if matches:
                matched_keywords.append(keyword)
                for match in matches:
                    all_matches.append((match.start(), match.end(), keyword, match.group()))
        
        # Sort matches by position and resolve overlaps
        all_matches.sort(key=lambda x: x[0])
        non_overlapping_matches = []
        
        for start, end, keyword, matched_text in all_matches:
            # Check for overlap with previous matches
            overlaps = False
            for prev_start, prev_end, _, _ in non_overlapping_matches:
                if start < prev_end and end > prev_start:  # Overlap detected
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping_matches.append((start, end, keyword, matched_text))
                highlight_positions.append((start, end))
        
        # Apply highlighting in reverse order to maintain positions
        highlighted_text = escaped_text
        for start, end, keyword, matched_text in reversed(non_overlapping_matches):
            # Calculate positions in escaped text
            escaped_start = len(html.escape(text[:start]))
            escaped_end = escaped_start + len(html.escape(text[start:end]))
            
            # Create highlighted replacement
            highlighted_keyword = f'<{self.highlight_tag}>{html.escape(matched_text)}</{self.highlight_tag}>'
            
            # Replace in highlighted text
            highlighted_text = (
                highlighted_text[:escaped_start] + 
                highlighted_keyword + 
                highlighted_text[escaped_end:]
            )
        
        return HighlightedText(
            original_text=text,
            highlighted_text=highlighted_text,
            matched_keywords=list(set(matched_keywords)),  # Remove duplicates
            highlight_positions=highlight_positions
        )
    
    def extract_query_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from search query.
        
        Args:
            query: Search query string
            
        Returns:
            List of extracted keywords
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how',
            'when', 'where', 'why', 'who', 'which'
        }
        
        # Split query into words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Also include quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        keywords.extend([phrase.strip() for phrase in quoted_phrases if phrase.strip()])
        
        return keywords


class SourceAttributionExtractor:
    """Utility for extracting enhanced source attribution from search results."""
    
    def extract_attribution(self, search_result: SearchResult) -> SourceAttribution:
        """
        Extract enhanced source attribution from search result metadata.
        
        Args:
            search_result: SearchResult object with metadata
            
        Returns:
            SourceAttribution object with extracted information
        """
        metadata = search_result.metadata or {}
        
        # Extract document information
        document_title = metadata.get('title') or metadata.get('document_title')
        document_path = metadata.get('source') or metadata.get('file_path') or metadata.get('path')
        
        # Extract structural information
        section_title = metadata.get('section') or metadata.get('header') or metadata.get('section_title')
        chapter = metadata.get('chapter') or metadata.get('chapter_title')
        
        # Extract location information
        page_number = metadata.get('page') or metadata.get('page_number')
        if isinstance(page_number, str) and page_number.isdigit():
            page_number = int(page_number)
        
        # Extract line numbers
        line_numbers = None
        if 'line_start' in metadata and 'line_end' in metadata:
            try:
                line_numbers = (int(metadata['line_start']), int(metadata['line_end']))
            except (ValueError, TypeError):
                pass
        
        # Extract context and type
        context_snippet = metadata.get('context') or metadata.get('surrounding_text')
        document_type = metadata.get('type') or metadata.get('file_type') or self._infer_document_type(document_path)
        
        return SourceAttribution(
            document_title=document_title,
            document_path=document_path,
            section_title=section_title,
            chapter=chapter,
            page_number=page_number,
            line_numbers=line_numbers,
            context_snippet=context_snippet,
            document_type=document_type
        )
    
    def _infer_document_type(self, file_path: Optional[str]) -> Optional[str]:
        """Infer document type from file path."""
        if not file_path:
            return None
        
        extension_map = {
            '.md': 'markdown',
            '.txt': 'text',
            '.pdf': 'pdf',
            '.doc': 'word',
            '.docx': 'word',
            '.html': 'html',
            '.htm': 'html',
            '.py': 'python',
            '.js': 'javascript',
            '.json': 'json',
            '.csv': 'csv'
        }
        
        for ext, doc_type in extension_map.items():
            if file_path.lower().endswith(ext):
                return doc_type
        
        return 'unknown'


class RelevanceAnalyzer:
    """Utility for analyzing relevance confidence indicators."""
    
    def __init__(self, keyword_highlighter: KeywordHighlighter):
        """Initialize with keyword highlighter for analysis."""
        self.keyword_highlighter = keyword_highlighter
    
    def analyze_relevance(self, query: str, search_result: SearchResult, rerank_score: float) -> RelevanceIndicators:
        """
        Analyze relevance indicators for a search result.
        
        Args:
            query: Original search query
            search_result: SearchResult object
            rerank_score: Re-ranking score from cross-encoder
            
        Returns:
            RelevanceIndicators object with analysis
        """
        # Extract keywords and analyze matches
        keywords = self.keyword_highlighter.extract_query_keywords(query)
        highlighted = self.keyword_highlighter.highlight_keywords(search_result.content, keywords)
        
        # Calculate keyword density
        keyword_density = self._calculate_keyword_density(search_result.content, highlighted.matched_keywords)
        
        # Calculate structure relevance based on metadata
        structure_relevance = self._calculate_structure_relevance(search_result.metadata or {})
        
        # Use rerank_score as semantic similarity (cross-encoder provides semantic understanding)
        semantic_similarity = rerank_score
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(
            semantic_similarity, keyword_density, structure_relevance
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            confidence_level, semantic_similarity, keyword_density, 
            structure_relevance, len(highlighted.matched_keywords)
        )
        
        return RelevanceIndicators(
            confidence_level=confidence_level,
            semantic_similarity=semantic_similarity,
            keyword_density=keyword_density,
            structure_relevance=structure_relevance,
            explanation=explanation
        )
    
    def _calculate_keyword_density(self, content: str, matched_keywords: List[str]) -> float:
        """Calculate keyword density in content."""
        if not content or not matched_keywords:
            return 0.0
        
        total_words = len(content.split())
        keyword_occurrences = 0
        
        for keyword in matched_keywords:
            # Count case-insensitive occurrences
            keyword_occurrences += content.lower().count(keyword.lower())
        
        return min(1.0, keyword_occurrences / total_words) if total_words > 0 else 0.0
    
    def _calculate_structure_relevance(self, metadata: Dict[str, Any]) -> float:
        """Calculate relevance based on document structure metadata."""
        relevance_score = 0.0
        
        # Document title presence
        if metadata.get('title') or metadata.get('document_title'):
            relevance_score += 0.3
        
        # Section/header information
        if metadata.get('section') or metadata.get('header'):
            relevance_score += 0.2
        
        # Document type relevance
        doc_type = metadata.get('type', '').lower()
        if doc_type in ['markdown', 'text', 'documentation']:
            relevance_score += 0.2
        
        # Metadata richness
        if len(metadata) > 3:
            relevance_score += 0.1
        
        # Source attribution
        if metadata.get('source') or metadata.get('file_path'):
            relevance_score += 0.2
        
        return min(1.0, relevance_score)
    
    def _determine_confidence_level(self, semantic_sim: float, keyword_density: float, 
                                  structure_relevance: float) -> str:
        """Determine overall confidence level."""
        # Weighted average of factors
        overall_score = (
            semantic_sim * 0.6 +  # Cross-encoder score is most important
            keyword_density * 0.25 +  # Keyword matching is important
            structure_relevance * 0.15  # Structure provides context
        )
        
        if overall_score >= 0.80:
            return "very_high"
        elif overall_score >= 0.65:
            return "high"
        elif overall_score >= 0.45:
            return "medium"
        else:
            return "low"
    
    def _generate_explanation(self, confidence_level: str, semantic_sim: float,
                            keyword_density: float, structure_relevance: float,
                            keyword_matches: int) -> str:
        """Generate human-readable explanation of relevance."""
        explanations = []
        
        # Semantic similarity explanation
        if semantic_sim >= 0.8:
            explanations.append("High semantic similarity to query")
        elif semantic_sim >= 0.6:
            explanations.append("Good semantic relevance")
        elif semantic_sim >= 0.4:
            explanations.append("Moderate semantic match")
        else:
            explanations.append("Low semantic similarity")
        
        # Keyword explanation
        if keyword_matches > 0:
            if keyword_density >= 0.1:
                explanations.append(f"Strong keyword presence ({keyword_matches} matches)")
            else:
                explanations.append(f"Some keyword matches ({keyword_matches} found)")
        else:
            explanations.append("No direct keyword matches")
        
        # Structure explanation
        if structure_relevance >= 0.6:
            explanations.append("Well-structured source document")
        elif structure_relevance >= 0.3:
            explanations.append("Decent document structure")
        
        return "; ".join(explanations) + f" (Confidence: {confidence_level})" 