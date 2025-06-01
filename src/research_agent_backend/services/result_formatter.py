"""
Result Formatting Service for Research Agent.

This service provides rich formatting capabilities for search results including:
- Keyword highlighting in content
- Structural context display (header hierarchy)
- Relevance score visualization
- Rich metadata presentation
- User feedback UI elements
- Markdown formatting optimized for Cursor chat

Implements task requirement: Create rich result presentation with keyword highlighting,
structural context, and relevance scoring.
"""

import re
import html
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DisplayFormat(Enum):
    """Display format options for result formatting."""
    MARKDOWN = "markdown"
    HTML = "html"
    PLAIN_TEXT = "plain_text"
    RICH_CONSOLE = "rich_console"


class RelevanceLevel(Enum):
    """Relevance level indicators."""
    VERY_HIGH = "very_high"      # 0.9+
    HIGH = "high"                # 0.7-0.89
    MEDIUM = "medium"            # 0.5-0.69
    LOW = "low"                  # 0.3-0.49
    VERY_LOW = "very_low"        # 0.0-0.29


@dataclass
class FormattingOptions:
    """Configuration options for result formatting."""
    format_type: DisplayFormat = DisplayFormat.MARKDOWN
    highlight_keywords: bool = True
    show_metadata: bool = True
    show_header_path: bool = True
    show_relevance_scores: bool = True
    show_feedback_ui: bool = True
    max_content_length: int = 500
    include_source_links: bool = True
    expandable_content: bool = True
    show_collection_info: bool = True
    indent_level: int = 0
    highlight_style: str = "bold"  # bold, background, underline
    
    # Content filtering options
    exclude_empty_results: bool = True
    minimum_relevance: float = 0.0
    
    # UI customization
    use_icons: bool = True
    show_result_numbers: bool = True
    compact_mode: bool = False


@dataclass 
class HighlightedText:
    """Represents text with highlighted keywords."""
    original_text: str
    highlighted_text: str
    keyword_positions: List[Tuple[int, int]] = field(default_factory=list)
    keywords_found: Set[str] = field(default_factory=set)


@dataclass
class FormattedResult:
    """A search result formatted for display."""
    content: str
    relevance_info: Dict[str, Any]
    metadata_display: str
    source_info: str
    header_path: str
    feedback_ui: str
    raw_result: Dict[str, Any]
    
    # Additional formatting metadata
    content_truncated: bool = False
    highlights_count: int = 0
    formatting_options: FormattingOptions = field(default_factory=FormattingOptions)


class ResultFormatter:
    """Service for formatting search results with rich presentation features."""
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        """
        Initialize the result formatter.
        
        Args:
            options: Formatting configuration options
        """
        self.options = options or FormattingOptions()
        self.relevance_thresholds = {
            RelevanceLevel.VERY_HIGH: 0.9,
            RelevanceLevel.HIGH: 0.7,
            RelevanceLevel.MEDIUM: 0.5,
            RelevanceLevel.LOW: 0.3,
            RelevanceLevel.VERY_LOW: 0.0
        }
        
        # Icon mappings for different relevance levels
        self.relevance_icons = {
            RelevanceLevel.VERY_HIGH: "🎯",
            RelevanceLevel.HIGH: "🔥", 
            RelevanceLevel.MEDIUM: "✅",
            RelevanceLevel.LOW: "⚠️",
            RelevanceLevel.VERY_LOW: "❓"
        }
        
        # Collection type icons
        self.collection_icons = {
            "research": "📚",
            "documentation": "📖",
            "code": "💻",
            "notes": "📝",
            "papers": "📄",
            "default": "📁"
        }
    
    def format_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        options: Optional[FormattingOptions] = None
    ) -> List[FormattedResult]:
        """
        Format a list of search results with rich presentation.
        
        Args:
            results: List of raw search results
            query: Original search query for keyword highlighting
            options: Override formatting options
            
        Returns:
            List of formatted results ready for display
        """
        format_opts = options or self.options
        formatted_results = []
        
        # Extract keywords from query for highlighting
        keywords = self._extract_keywords(query)
        
        for i, result in enumerate(results):
            try:
                # Skip empty or invalid results if configured
                if format_opts.exclude_empty_results and not self._is_valid_result(result):
                    continue
                
                # Check minimum relevance threshold
                score = result.get("score", 0.0)
                if score < format_opts.minimum_relevance:
                    continue
                
                formatted_result = self._format_single_result(
                    result, keywords, i + 1, format_opts
                )
                formatted_results.append(formatted_result)
                
            except Exception as e:
                logger.warning(f"Failed to format result {i}: {e}")
                # Include a minimal formatted version to avoid losing data
                formatted_results.append(self._create_fallback_result(result, i + 1))
        
        return formatted_results
    
    def format_query_summary(
        self, 
        results: List[FormattedResult], 
        query: str,
        total_found: int,
        options: Optional[FormattingOptions] = None
    ) -> str:
        """
        Create a summary of the query results.
        
        Args:
            results: List of formatted results
            query: Original query
            total_found: Total number of results found
            options: Formatting options
            
        Returns:
            Formatted summary string
        """
        format_opts = options or self.options
        
        if format_opts.format_type == DisplayFormat.MARKDOWN:
            return self._format_markdown_summary(results, query, total_found)
        elif format_opts.format_type == DisplayFormat.HTML:
            return self._format_html_summary(results, query, total_found)
        else:
            return self._format_plain_summary(results, query, total_found)
    
    def _format_single_result(
        self, 
        result: Dict[str, Any], 
        keywords: List[str], 
        result_number: int,
        options: FormattingOptions
    ) -> FormattedResult:
        """Format a single search result."""
        
        # Extract basic information
        content = result.get("content", "")
        score = result.get("score", 0.0)
        document_id = result.get("document_id", "Unknown")
        collection = result.get("collection", "default")
        
        # Format content with highlighting
        highlighted_content = self._highlight_keywords(content, keywords, options)
        
        # Truncate content if needed
        content_truncated = False
        if len(highlighted_content.highlighted_text) > options.max_content_length:
            highlighted_content.highlighted_text = (
                highlighted_content.highlighted_text[:options.max_content_length] + "..."
            )
            content_truncated = True
        
        # Create relevance information
        relevance_info = self._create_relevance_info(score, options)
        
        # Format metadata
        metadata_display = self._format_metadata(result, options)
        
        # Create source information
        source_info = self._format_source_info(result, options)
        
        # Format header path
        header_path = self._format_header_path(result, options)
        
        # Create feedback UI
        feedback_ui = self._create_feedback_ui(result, options)
        
        return FormattedResult(
            content=highlighted_content.highlighted_text,
            relevance_info=relevance_info,
            metadata_display=metadata_display,
            source_info=source_info,
            header_path=header_path,
            feedback_ui=feedback_ui,
            raw_result=result,
            content_truncated=content_truncated,
            highlights_count=len(highlighted_content.keywords_found),
            formatting_options=options
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from a query for highlighting."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'what', 'how', 'where', 'when', 'why', 'who', 'which', 'that',
            'this', 'these', 'those', 'can', 'could', 'would', 'should',
            'show', 'me', 'find', 'search', 'about'
        }
        
        # Extract words, keeping quoted phrases together
        words = []
        
        # First, extract quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        words.extend(quoted_phrases)
        
        # Remove quoted content from query and extract individual words
        query_without_quotes = re.sub(r'"[^"]*"', '', query)
        individual_words = re.findall(r'\b\w+\b', query_without_quotes.lower())
        
        # Filter individual words
        meaningful_words = [
            word for word in individual_words 
            if word.lower() not in stop_words and len(word) > 2
        ]
        
        words.extend(meaningful_words)
        
        return list(set(words))  # Remove duplicates
    
    def _highlight_keywords(
        self, 
        text: str, 
        keywords: List[str], 
        options: FormattingOptions
    ) -> HighlightedText:
        """Apply keyword highlighting to text."""
        if not options.highlight_keywords or not keywords:
            return HighlightedText(text, text)
        
        highlighted_text = text
        keywords_found = set()
        keyword_positions = []
        
        # Sort keywords by length (longest first) to avoid partial matches
        sorted_keywords = sorted(keywords, key=len, reverse=True)
        
        for keyword in sorted_keywords:
            # Create case-insensitive pattern that matches whole words
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            
            matches = list(pattern.finditer(highlighted_text))
            if matches:
                keywords_found.add(keyword.lower())
                
                # Apply highlighting based on format - preserve original case
                if options.format_type == DisplayFormat.MARKDOWN:
                    def replace_match(match):
                        matched_text = match.group(0)
                        if options.highlight_style == "bold":
                            return f"**{matched_text}**"
                        elif options.highlight_style == "background":
                            return f"=={matched_text}=="
                        else:
                            return f"__{matched_text}__"
                    
                    highlighted_text = pattern.sub(replace_match, highlighted_text)
                    
                elif options.format_type == DisplayFormat.HTML:
                    def replace_match(match):
                        matched_text = match.group(0)
                        return f'<mark>{matched_text}</mark>'
                    
                    highlighted_text = pattern.sub(replace_match, highlighted_text)
                
                # Record positions for statistics
                for match in matches:
                    keyword_positions.append((match.start(), match.end()))
        
        return HighlightedText(
            original_text=text,
            highlighted_text=highlighted_text,
            keyword_positions=keyword_positions,
            keywords_found=keywords_found
        )
    
    def _create_relevance_info(
        self, 
        score: float, 
        options: FormattingOptions
    ) -> Dict[str, Any]:
        """Create relevance information display."""
        relevance_level = self._get_relevance_level(score)
        
        info = {
            "score": score,
            "level": relevance_level,
            "label": self._get_relevance_label(relevance_level),
            "description": self._get_relevance_description(relevance_level)
        }
        
        if options.use_icons:
            info["icon"] = self.relevance_icons.get(relevance_level, "📄")
        
        return info
    
    def _get_relevance_level(self, score: float) -> RelevanceLevel:
        """Determine relevance level from score."""
        if score >= 0.9:
            return RelevanceLevel.VERY_HIGH
        elif score >= 0.7:
            return RelevanceLevel.HIGH
        elif score >= 0.5:
            return RelevanceLevel.MEDIUM
        elif score >= 0.3:
            return RelevanceLevel.LOW
        else:
            return RelevanceLevel.VERY_LOW
    
    def _get_relevance_label(self, level: RelevanceLevel) -> str:
        """Get human-readable relevance label."""
        labels = {
            RelevanceLevel.VERY_HIGH: "Highly Relevant",
            RelevanceLevel.HIGH: "Very Relevant", 
            RelevanceLevel.MEDIUM: "Relevant",
            RelevanceLevel.LOW: "Somewhat Relevant",
            RelevanceLevel.VERY_LOW: "Low Relevance"
        }
        return labels.get(level, "Unknown")
    
    def _get_relevance_description(self, level: RelevanceLevel) -> str:
        """Get detailed relevance description."""
        descriptions = {
            RelevanceLevel.VERY_HIGH: "Excellent match for your query",
            RelevanceLevel.HIGH: "Strong match with good context",
            RelevanceLevel.MEDIUM: "Good match with relevant information",
            RelevanceLevel.LOW: "Partial match, may contain useful details", 
            RelevanceLevel.VERY_LOW: "Limited relevance to your query"
        }
        return descriptions.get(level, "Relevance unclear")
    
    def _format_metadata(self, result: Dict[str, Any], options: FormattingOptions) -> str:
        """Format metadata information for display."""
        if not options.show_metadata:
            return ""
        
        metadata = result.get("metadata", {})
        document_title = metadata.get("document_title", "")
        content_type = metadata.get("content_type", "prose")
        chunk_id = metadata.get("chunk_sequence_id", 0)
        
        if options.format_type == DisplayFormat.MARKDOWN:
            parts = []
            if document_title:
                parts.append(f"📄 **{document_title}**")
            if content_type != "prose":
                parts.append(f"Type: `{content_type}`")
            if chunk_id:
                parts.append(f"Section: {chunk_id}")
            
            return " | ".join(parts)
        
        return f"Document: {document_title}, Type: {content_type}, Section: {chunk_id}"
    
    def _format_source_info(self, result: Dict[str, Any], options: FormattingOptions) -> str:
        """Format source document information."""
        document_id = result.get("document_id", "Unknown")
        collection = result.get("collection", "default")
        
        if not options.include_source_links:
            return f"Source: {document_id}"
        
        # Get collection icon
        collection_icon = ""
        if options.use_icons:
            collection_icon = self.collection_icons.get(collection, "📁")
        
        if options.format_type == DisplayFormat.MARKDOWN:
            return f"{collection_icon} **{collection}** / `{document_id}`"
        
        return f"{collection_icon} {collection} / {document_id}"
    
    def _format_header_path(self, result: Dict[str, Any], options: FormattingOptions) -> str:
        """Format document header hierarchy path."""
        if not options.show_header_path:
            return ""
        
        header_path = result.get("header_path", "")
        if not header_path:
            return ""
        
        if options.format_type == DisplayFormat.MARKDOWN:
            return f"📍 {header_path}"
        
        return f"Location: {header_path}"
    
    def _create_feedback_ui(self, result: Dict[str, Any], options: FormattingOptions) -> str:
        """Create user feedback UI elements."""
        if not options.show_feedback_ui:
            return ""
        
        if options.format_type == DisplayFormat.MARKDOWN:
            return "👍 👎 💬 Feedback"
        
        return "[Like] [Dislike] [Comment]"
    
    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Check if a result has meaningful content."""
        content = result.get("content", "").strip()
        return len(content) > 0
    
    def _create_fallback_result(self, result: Dict[str, Any], result_number: int) -> FormattedResult:
        """Create a minimal formatted result when formatting fails."""
        return FormattedResult(
            content=result.get("content", "Content unavailable"),
            relevance_info={"score": result.get("score", 0.0), "label": "Unknown"},
            metadata_display="",
            source_info=result.get("document_id", "Unknown source"),
            header_path="",
            feedback_ui="",
            raw_result=result,
            formatting_options=self.options
        )
    
    def _format_markdown_summary(
        self, 
        results: List[FormattedResult], 
        query: str, 
        total_found: int
    ) -> str:
        """Format a markdown summary of query results."""
        if not results:
            return f"## No Results Found\n\nNo documents matched your query: **{query}**"
        
        summary_parts = [
            f"## Search Results for: *{query}*",
            f"Found **{len(results)}** relevant results" + 
            (f" (of {total_found} total)" if total_found > len(results) else ""),
            ""
        ]
        
        # Add relevance distribution
        relevance_counts = {}
        for result in results:
            level = result.relevance_info.get("level", RelevanceLevel.VERY_LOW)
            relevance_counts[level] = relevance_counts.get(level, 0) + 1
        
        if relevance_counts:
            summary_parts.append("### Relevance Distribution")
            for level in RelevanceLevel:
                count = relevance_counts.get(level, 0)
                if count > 0:
                    icon = self.relevance_icons.get(level, "")
                    label = self._get_relevance_label(level)
                    summary_parts.append(f"- {icon} {label}: {count}")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def _format_html_summary(
        self, 
        results: List[FormattedResult], 
        query: str, 
        total_found: int
    ) -> str:
        """Format an HTML summary of query results."""
        if not results:
            return f"<h3>No Results Found</h3><p>No documents matched your query: <strong>{html.escape(query)}</strong></p>"
        
        summary = f"""
        <h3>Search Results for: <em>{html.escape(query)}</em></h3>
        <p>Found <strong>{len(results)}</strong> relevant results
        {f" (of {total_found} total)" if total_found > len(results) else ""}</p>
        """
        
        return summary
    
    def _format_plain_summary(
        self, 
        results: List[FormattedResult], 
        query: str, 
        total_found: int
    ) -> str:
        """Format a plain text summary of query results."""
        if not results:
            return f"No Results Found\n\nNo documents matched your query: {query}"
        
        return f"Search Results for: {query}\nFound {len(results)} relevant results" + \
               (f" (of {total_found} total)" if total_found > len(results) else "")


# Convenience functions for common formatting tasks

def format_results_for_cursor(
    results: List[Dict[str, Any]], 
    query: str,
    compact: bool = False
) -> List[FormattedResult]:
    """
    Format results optimized for Cursor IDE display.
    
    Args:
        results: Raw search results
        query: Search query for highlighting
        compact: Use compact display mode
        
    Returns:
        Formatted results for Cursor
    """
    options = FormattingOptions(
        format_type=DisplayFormat.MARKDOWN,
        highlight_keywords=True,
        show_metadata=True,
        show_header_path=True,
        show_relevance_scores=True,
        show_feedback_ui=True,
        max_content_length=400 if compact else 500,
        include_source_links=True,
        expandable_content=True,
        use_icons=True,
        compact_mode=compact
    )
    
    formatter = ResultFormatter(options)
    return formatter.format_results(results, query)


def format_results_for_cli(
    results: List[Dict[str, Any]], 
    query: str,
    use_colors: bool = True
) -> List[FormattedResult]:
    """
    Format results optimized for CLI display.
    
    Args:
        results: Raw search results
        query: Search query for highlighting
        use_colors: Use colored output
        
    Returns:
        Formatted results for CLI
    """
    options = FormattingOptions(
        format_type=DisplayFormat.RICH_CONSOLE if use_colors else DisplayFormat.PLAIN_TEXT,
        highlight_keywords=True,
        show_metadata=True,
        show_header_path=True,
        show_relevance_scores=True,
        show_feedback_ui=False,  # Less relevant for CLI
        max_content_length=600,
        include_source_links=True,
        expandable_content=False,
        use_icons=use_colors,
        compact_mode=False
    )
    
    formatter = ResultFormatter(options)
    return formatter.format_results(results, query)


def create_result_markdown(formatted_result: FormattedResult, result_number: int) -> str:
    """
    Create a complete markdown representation of a formatted result.
    
    Args:
        formatted_result: Formatted result object
        result_number: Display number for the result
        
    Returns:
        Complete markdown string for the result
    """
    parts = []
    
    # Header with result number and relevance
    relevance_icon = formatted_result.relevance_info.get("icon", "")
    relevance_label = formatted_result.relevance_info.get("label", "")
    score = formatted_result.relevance_info.get("score", 0.0)
    
    parts.append(f"### {result_number}. {relevance_icon} {relevance_label} ({score:.2f})")
    parts.append("")
    
    # Source and metadata
    if formatted_result.source_info:
        parts.append(formatted_result.source_info)
    
    if formatted_result.metadata_display:
        parts.append(formatted_result.metadata_display)
    
    if formatted_result.header_path:
        parts.append(formatted_result.header_path)
    
    if formatted_result.source_info or formatted_result.metadata_display or formatted_result.header_path:
        parts.append("")
    
    # Content
    parts.append(formatted_result.content)
    
    # Feedback UI
    if formatted_result.feedback_ui:
        parts.append("")
        parts.append(formatted_result.feedback_ui)
    
    parts.append("")
    parts.append("---")
    parts.append("")
    
    return "\n".join(parts) 