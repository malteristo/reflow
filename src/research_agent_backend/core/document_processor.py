"""
Document Processing Module - Markdown Parser Implementation

This module implements the core Markdown parsing functionality that converts
Markdown syntax to HTML. It provides a flexible, rule-based system for
markdown transformation that serves as the foundation for the hybrid
document chunking strategy.

Key Components:
- Pattern: Regex matching with validation and error handling
- Rule: Transformation rules supporting both string and callable replacements
- MarkdownParser: Main parser with extensible rule system
- MarkdownParseError: Custom exception with debugging context

NEW: Header-based Document Splitting Components:
- DocumentSection: Represents a document section with header and content
- DocumentTree: Tree structure for document hierarchy
- HeaderBasedSplitter: Splits documents by headers into sections
- SectionExtractor: Extracts specific sections from document trees

Implements FR-KB-002.1: Hybrid chunking strategy with Markdown-aware processing.

Usage:
    >>> parser = MarkdownParser()
    >>> html = parser.parse("# Header\n\nThis is **bold** text.")
    >>> print(html)
    <h1>Header</h1>
    
    This is <strong>bold</strong> text.
    
    >>> # Header-based splitting
    >>> splitter = HeaderBasedSplitter(parser)
    >>> tree = splitter.split_and_build_tree("# Main\n\nContent\n\n## Sub\n\nMore content")
    >>> print(tree.root.title)
    Main
"""

import re
from typing import List, Dict, Any, Optional, Union, Callable, Match, Protocol, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)


class MarkdownParseError(Exception):
    """
    Custom exception for markdown parsing errors.
    
    Provides additional context about pattern failures for debugging
    and troubleshooting regex compilation or transformation issues.
    
    Attributes:
        message: Human-readable error description
        pattern_name: Name of the pattern that caused the error
        regex: The regex pattern string that failed
    """
    
    def __init__(
        self, 
        message: str, 
        pattern_name: Optional[str] = None, 
        regex: Optional[str] = None
    ) -> None:
        """
        Initialize MarkdownParseError with debugging context.
        
        Args:
            message: Primary error message
            pattern_name: Optional name of the failing pattern
            regex: Optional regex string that caused the error
        """
        super().__init__(message)
        self.pattern_name = pattern_name
        self.regex = regex
        
        # Log the error for debugging
        logger.error(
            f"MarkdownParseError: {message}",
            extra={
                "pattern_name": pattern_name,
                "regex": regex,
                "error_type": "markdown_parse"
            }
        )


@dataclass(frozen=True)
class MatchResult:
    """
    Immutable result container for pattern matching operations.
    
    Attributes:
        pattern_name: Name of the pattern that generated this result
        match_count: Number of matches found
        matches: List of match objects or strings
    """
    pattern_name: str
    match_count: int
    matches: List[Union[str, Match[str]]]


class Pattern:
    """
    A Pattern class for regex matching functionality.
    
    Handles regex compilation, validation, and provides match/findall operations
    for markdown element detection. Includes comprehensive error handling and
    performance optimizations through compiled regex caching.
    
    Attributes:
        name: Descriptive identifier for the pattern
        regex_pattern: Raw regex string
        compiled_regex: Pre-compiled regex object for performance
    """
    
    def __init__(self, name: str, regex_pattern: str) -> None:
        """
        Initialize a Pattern with name and regex validation.
        
        Args:
            name: Descriptive name for the pattern (e.g., 'header', 'bold')
            regex_pattern: Regular expression pattern string
            
        Raises:
            MarkdownParseError: If regex pattern is invalid or malformed
            ValueError: If name is empty or invalid
        """
        if not name or not name.strip():
            raise ValueError("Pattern name cannot be empty")
        
        if not regex_pattern:
            raise ValueError("Regex pattern cannot be empty")
        
        self.name = name.strip()
        self.regex_pattern = regex_pattern
        
        try:
            # Compile with MULTILINE for header patterns and optimization
            self.compiled_regex = re.compile(regex_pattern, re.MULTILINE)
            logger.debug(f"Compiled pattern '{self.name}': {regex_pattern}")
        except re.error as e:
            raise MarkdownParseError(
                f"Invalid regex pattern for '{name}': {e}",
                pattern_name=name,
                regex=regex_pattern
            )
    
    def match(self, text: str) -> Optional[Match[str]]:
        """
        Find the first match in text using compiled regex.
        
        Args:
            text: Input text to search
            
        Returns:
            Match object if pattern found, None otherwise
        """
        if not text:
            return None
        
        return self.compiled_regex.search(text)
    
    def findall(self, text: str) -> List[str]:
        """
        Find all matches in text using compiled regex.
        
        Args:
            text: Input text to search
            
        Returns:
            List of matched strings (empty list if no matches)
        """
        if not text:
            return []
        
        return self.compiled_regex.findall(text)
    
    def find_with_metadata(self, text: str) -> MatchResult:
        """
        Find all matches with additional metadata for debugging and analysis.
        
        Args:
            text: Input text to search
            
        Returns:
            MatchResult containing pattern name, count, and matches
        """
        matches = self.findall(text)
        return MatchResult(
            pattern_name=self.name,
            match_count=len(matches),
            matches=matches
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Pattern(name='{self.name}', regex='{self.regex_pattern}')"
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on name and regex pattern."""
        if not isinstance(other, Pattern):
            return NotImplemented
        return self.name == other.name and self.regex_pattern == other.regex_pattern


class Rule:
    """
    A Rule class for defining transformation rules.
    
    Associates a Pattern with a replacement strategy to convert markdown
    syntax to HTML. Supports both string-based replacements (with regex
    backreferences) and callable functions for complex transformations.
    
    Attributes:
        pattern: Pattern object for matching
        replacement: Replacement string or callable function
    """
    
    def __init__(
        self, 
        pattern: Pattern, 
        replacement: Union[str, Callable[[Match[str]], str]]
    ) -> None:
        """
        Initialize a Rule with pattern and replacement strategy.
        
        Args:
            pattern: Pattern object for matching markdown elements
            replacement: Replacement string (with \1, \2 backrefs) or callable
            
        Raises:
            TypeError: If pattern is not a Pattern instance
            ValueError: If replacement is neither string nor callable
        """
        if not isinstance(pattern, Pattern):
            raise TypeError("Pattern must be a Pattern instance")
        
        if not isinstance(replacement, (str, Callable)):
            raise ValueError("Replacement must be string or callable")
        
        self.pattern = pattern
        self.replacement = replacement
        
        logger.debug(
            f"Created rule for pattern '{pattern.name}' with replacement type: "
            f"{type(replacement).__name__}"
        )
    
    def apply(self, text: str) -> str:
        """
        Apply the transformation rule to text.
        
        Handles both string replacements (with regex backreferences) and
        callable replacements for dynamic transformations.
        
        Args:
            text: Input text to transform
            
        Returns:
            Transformed text with pattern matches replaced
            
        Raises:
            MarkdownParseError: If replacement function fails
        """
        if not text:
            return text
        
        try:
            if callable(self.replacement):
                # Handle function/lambda replacements
                def replace_func(match: Match[str]) -> str:
                    try:
                        return self.replacement(match)
                    except Exception as e:
                        raise MarkdownParseError(
                            f"Replacement function failed for pattern '{self.pattern.name}': {e}",
                            pattern_name=self.pattern.name
                        )
                
                return self.pattern.compiled_regex.sub(replace_func, text)
            else:
                # Handle string replacements with backreferences
                return self.pattern.compiled_regex.sub(self.replacement, text)
                
        except Exception as e:
            if isinstance(e, MarkdownParseError):
                raise
            
            raise MarkdownParseError(
                f"Rule application failed for pattern '{self.pattern.name}': {e}",
                pattern_name=self.pattern.name,
                regex=self.pattern.regex_pattern
            )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        repl_type = "callable" if callable(self.replacement) else "string"
        return f"Rule(pattern='{self.pattern.name}', replacement_type={repl_type})"


class MarkdownParser:
    """
    Main MarkdownParser class that applies rules sequentially to input text.
    
    Provides a complete markdown-to-HTML conversion system with support for
    headers, bold, italic, and link transformations. The parser uses a
    rule-based approach for flexibility and extensibility.
    
    Features:
    - Default rules for common markdown elements
    - Custom rule addition and management
    - Rule ordering for conflict resolution
    - Pattern lookup by name
    - Comprehensive error handling and logging
    
    Attributes:
        rules: List of transformation rules applied in order
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        """
        Initialize MarkdownParser with rules.
        
        Args:
            rules: Optional list of custom rules. If None, uses default rules
                  for headers, bold, italic, and links.
        """
        if rules is None:
            self.rules = self._create_default_rules()
            logger.info("MarkdownParser initialized with default rules")
        else:
            self.rules = rules.copy()  # Defensive copy
            logger.info(f"MarkdownParser initialized with {len(rules)} custom rules")
    
    def _create_default_rules(self) -> List[Rule]:
        """
        Create default rules for common markdown elements.
        
        The rules are ordered to prevent pattern conflicts:
        1. Headers (highest precedence)
        2. Bold text (before italic to prevent conflicts)
        3. Italic text (with lookbehind/lookahead for safety)
        4. Links (processed last)
        
        Returns:
            List of default Rule objects optimized for common markdown
        """
        rules = []
        
        # Header rule with lambda for dynamic header level detection
        header_pattern = Pattern("header", r"^(#{1,6})\s+(.+)$")
        header_rule = Rule(
            header_pattern,
            lambda match: f"<h{len(match.group(1))}>{match.group(2)}</h{len(match.group(1))}>"
        )
        rules.append(header_rule)
        
        # Bold rule - processed before italic to avoid conflicts
        bold_pattern = Pattern("bold", r"\*\*(.*?)\*\*")
        bold_rule = Rule(bold_pattern, r"<strong>\1</strong>")
        rules.append(bold_rule)
        
        # Italic rule with negative lookbehind/lookahead to prevent bold conflicts
        # Pattern explanation: (?<!\*) = not preceded by *, (?!\*) = not followed by *
        italic_pattern = Pattern("italic", r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)")
        italic_rule = Rule(italic_pattern, r"<em>\1</em>")
        rules.append(italic_rule)
        
        # Link rule for [text](url) format
        link_pattern = Pattern("link", r"\[([^\]]+)\]\(([^)]+)\)")
        link_rule = Rule(link_pattern, r'<a href="\2">\1</a>')
        rules.append(link_rule)
        
        logger.debug(f"Created {len(rules)} default rules")
        return rules
    
    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule to the parser's rule list.
        
        Rules are applied in the order they are added, so consider
        pattern conflicts when adding custom rules.
        
        Args:
            rule: Rule object to add to the transformation pipeline
            
        Raises:
            TypeError: If rule is not a Rule instance
        """
        if not isinstance(rule, Rule):
            raise TypeError("Rule must be a Rule instance")
        
        self.rules.append(rule)
        logger.debug(f"Added rule for pattern '{rule.pattern.name}'")
    
    def get_pattern_by_name(self, name: str) -> Optional[Pattern]:
        """
        Retrieve a pattern by its name for inspection or debugging.
        
        Args:
            name: Name of the pattern to find
            
        Returns:
            Pattern object if found, None otherwise
        """
        for rule in self.rules:
            if rule.pattern.name == name:
                return rule.pattern
        return None
    
    def get_rule_by_pattern_name(self, name: str) -> Optional[Rule]:
        """
        Retrieve a rule by its pattern name.
        
        Args:
            name: Name of the pattern to find
            
        Returns:
            Rule object if found, None otherwise
        """
        for rule in self.rules:
            if rule.pattern.name == name:
                return rule
        return None
    
    def parse(self, text: str) -> str:
        """
        Parse markdown text and convert to HTML.
        
        Applies all rules sequentially to transform markdown syntax.
        The order of rule application matters - rules are processed
        in the order they appear in the rules list.
        
        Args:
            text: Input markdown text to parse
            
        Returns:
            HTML-formatted text with markdown syntax converted
            
        Raises:
            MarkdownParseError: If any rule application fails
        """
        if not text:
            return text
        
        result = text
        rules_applied = 0
        
        logger.debug(f"Parsing text with {len(self.rules)} rules")
        
        # Apply each rule in sequence
        for rule in self.rules:
            try:
                previous_result = result
                result = rule.apply(result)
                
                # Log if rule made changes
                if result != previous_result:
                    rules_applied += 1
                    logger.debug(f"Rule '{rule.pattern.name}' applied transformations")
                    
            except MarkdownParseError:
                # Re-raise markdown parse errors as-is
                raise
            except Exception as e:
                # Wrap unexpected errors
                raise MarkdownParseError(
                    f"Unexpected error applying rule '{rule.pattern.name}': {e}",
                    pattern_name=rule.pattern.name
                )
        
        logger.debug(f"Parsing completed. {rules_applied} rules applied transformations")
        return result
    
    def parse_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Parse text and return both result and parsing metadata.
        
        Useful for debugging and analysis of the parsing process.
        
        Args:
            text: Input markdown text
            
        Returns:
            Dictionary containing:
            - 'result': Parsed HTML text
            - 'rules_applied': Number of rules that made changes
            - 'pattern_matches': Dict of pattern names to match counts
        """
        if not text:
            return {
                'result': text,
                'rules_applied': 0,
                'pattern_matches': {}
            }
        
        result = text
        rules_applied = 0
        pattern_matches = {}
        
        for rule in self.rules:
            # Check for matches before applying rule
            match_result = rule.pattern.find_with_metadata(result)
            pattern_matches[rule.pattern.name] = match_result.match_count
            
            previous_result = result
            result = rule.apply(result)
            
            if result != previous_result:
                rules_applied += 1
        
        return {
            'result': result,
            'rules_applied': rules_applied,
            'pattern_matches': pattern_matches
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        rule_names = [rule.pattern.name for rule in self.rules]
        return f"MarkdownParser(rules=[{', '.join(rule_names)}])"


# NEW CLASSES: Header-based Document Splitting

class DocumentSection:
    """
    Represents a document section with header information and content.
    
    A section corresponds to a markdown header and all content until the next
    header of equal or higher level. Supports hierarchical organization with
    parent-child relationships for building document trees.
    
    This class is designed to preserve the semantic structure of markdown documents
    while enabling efficient traversal and content extraction operations.
    
    Attributes:
        level: Header level (1-6 for h1-h6, 0 for non-header content)
        title: Section title extracted from header text
        content: Text content of the section (excluding header line)
        line_number: Line number where the section starts in original document
        children: List of child sections (subsections)
        parent: Reference to parent section (None for root sections)
    
    Example:
        >>> section = DocumentSection(
        ...     level=1, 
        ...     title="Introduction", 
        ...     content="This is the intro.",
        ...     line_number=1
        ... )
        >>> child = DocumentSection(level=2, title="Background", content="...", line_number=5)
        >>> section.add_child(child)
        >>> print(section.children[0].title)
        Background
    """
    
    def __init__(
        self,
        level: int,
        title: str,
        content: str,
        line_number: int,
        parent: Optional['DocumentSection'] = None
    ) -> None:
        """
        Initialize a DocumentSection with validation and setup.
        
        Args:
            level: Header level (0-6, where 0 is for non-header content, 1-6 for h1-h6)
            title: Section title from header text (empty string for non-header content)
            content: Section content text (excluding the header line itself)
            line_number: Starting line number in original document (1-indexed)
            parent: Parent section reference (None for root sections)
            
        Raises:
            ValueError: If level is not in valid range 0-6
            TypeError: If any required parameter has wrong type
        """
        # Input validation
        if not isinstance(level, int) or level < 0 or level > 6:
            raise ValueError(f"Header level must be integer 0-6, got: {level}")
        if not isinstance(title, str):
            raise TypeError(f"Title must be string, got: {type(title)}")
        if not isinstance(content, str):
            raise TypeError(f"Content must be string, got: {type(content)}")
        if not isinstance(line_number, int) or line_number < 1:
            raise ValueError(f"Line number must be positive integer, got: {line_number}")
        
        self.level = level
        self.title = title
        self.content = content
        self.line_number = line_number
        self.parent = parent
        self.children: List['DocumentSection'] = []
        
        # Log section creation for debugging
        logger.debug(
            f"Created DocumentSection: level={level}, title='{title}', "
            f"content_length={len(content)}, line={line_number}"
        )
    
    def add_child(self, child: 'DocumentSection') -> None:
        """
        Add a child section to this section with validation.
        
        Establishes bidirectional parent-child relationship and validates
        that the child has appropriate header level (higher than parent).
        
        Args:
            child: Child section to add
            
        Raises:
            TypeError: If child is not a DocumentSection
            ValueError: If child level is not greater than parent level
        """
        if not isinstance(child, DocumentSection):
            raise TypeError(f"Child must be DocumentSection, got: {type(child)}")
        
        # Validate header level hierarchy (child should have higher level than parent)
        if self.level > 0 and child.level > 0 and child.level <= self.level:
            logger.warning(
                f"Child level {child.level} is not greater than parent level {self.level}. "
                f"This may indicate improper document structure."
            )
        
        child.parent = self
        self.children.append(child)
        
        logger.debug(f"Added child '{child.title}' to section '{self.title}'")
    
    def remove_child(self, child: 'DocumentSection') -> bool:
        """
        Remove a child section from this section.
        
        Args:
            child: Child section to remove
            
        Returns:
            True if child was found and removed, False otherwise
        """
        if child in self.children:
            child.parent = None
            self.children.remove(child)
            logger.debug(f"Removed child '{child.title}' from section '{self.title}'")
            return True
        return False
    
    def get_depth(self) -> int:
        """
        Calculate the depth of this section in the document hierarchy.
        
        Depth is determined by traversing parent relationships up to the root.
        Root sections have depth 0, their children have depth 1, etc.
        
        Returns:
            Depth level (0 for root, 1 for first level children, etc.)
            
        Example:
            >>> root = DocumentSection(1, "Root", "", 1)
            >>> child = DocumentSection(2, "Child", "", 3)
            >>> grandchild = DocumentSection(3, "Grandchild", "", 5)
            >>> root.add_child(child)
            >>> child.add_child(grandchild)
            >>> print(grandchild.get_depth())
            2
        """
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth
    
    def get_all_content(self, include_headers: bool = False) -> str:
        """
        Get all content including content from child sections.
        
        Recursively collects content from this section and all its children,
        optionally including header text for better context.
        
        Args:
            include_headers: If True, include header titles in the output
            
        Returns:
            Combined content from this section and all children
            
        Example:
            >>> section = DocumentSection(1, "Main", "Main content", 1)
            >>> child = DocumentSection(2, "Sub", "Sub content", 3)
            >>> section.add_child(child)
            >>> print(section.get_all_content())
            Main content
            
            Sub content
        """
        content_parts = []
        
        # Add header if requested
        if include_headers and self.title:
            header_prefix = "#" * self.level if self.level > 0 else ""
            content_parts.append(f"{header_prefix} {self.title}" if header_prefix else self.title)
        
        # Add this section's content
        if self.content.strip():
            content_parts.append(self.content)
        
        # Recursively add children's content
        for child in self.children:
            child_content = child.get_all_content(include_headers)
            if child_content.strip():
                content_parts.append(child_content)
        
        return "\n\n".join(content_parts)
    
    def find_child_by_title(self, title: str, case_sensitive: bool = True) -> Optional['DocumentSection']:
        """
        Find a direct child section by title with optional case sensitivity.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Child section with matching title, or None if not found
            
        Example:
            >>> parent = DocumentSection(1, "Parent", "", 1)
            >>> child = DocumentSection(2, "Methods", "", 3)
            >>> parent.add_child(child)
            >>> found = parent.find_child_by_title("methods", case_sensitive=False)
            >>> print(found.title)
            Methods
        """
        for child in self.children:
            if case_sensitive:
                if child.title == title:
                    return child
            else:
                if child.title.lower() == title.lower():
                    return child
        return None
    
    def get_sibling_sections(self) -> List['DocumentSection']:
        """
        Get all sibling sections (sections with same parent).
        
        Returns:
            List of sibling sections (excluding self)
        """
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert section to dictionary for serialization.
        
        Returns:
            Dictionary representation with all section data
        """
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "line_number": self.line_number,
            "depth": self.get_depth(),
            "children_count": len(self.children),
            "children": [child.to_dict() for child in self.children]
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentSection(level={self.level}, title='{self.title}', "
            f"line={self.line_number}, children={len(self.children)})"
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on content and structure."""
        if not isinstance(other, DocumentSection):
            return NotImplemented
        return (
            self.level == other.level and
            self.title == other.title and
            self.content == other.content and
            self.line_number == other.line_number
        )


class DocumentTree:
    """
    Tree structure representing a complete document hierarchy.
    
    Manages the hierarchical organization of document sections and provides
    efficient methods for navigation, search, and manipulation of the document
    structure. Automatically maintains parent-child relationships and provides
    optimized lookup operations.
    
    The tree structure mirrors the logical organization of a markdown document,
    where headers create a natural hierarchy that can be traversed and analyzed.
    
    Attributes:
        root: Root section of the document (None for empty trees)
        _sections: Flat list of all sections for efficient O(1) lookup operations
        _title_index: Dictionary mapping titles to sections for fast title-based searches
    
    Example:
        >>> tree = DocumentTree()
        >>> tree.add_section(DocumentSection(1, "Chapter 1", "Content", 1))
        >>> tree.add_section(DocumentSection(2, "Section 1.1", "More content", 3))
        >>> print(tree.get_section_count())
        2
    """
    
    def __init__(self, root: Optional[DocumentSection] = None) -> None:
        """
        Initialize a DocumentTree with optional root section.
        
        Args:
            root: Optional root section to start the tree
        """
        self.root = root
        self._sections: List[DocumentSection] = []
        self._title_index: Dict[str, List[DocumentSection]] = {}
        
        if root:
            self._add_section_to_indexes(root)
        
        logger.debug(f"DocumentTree initialized with root: {root.title if root else 'None'}")
    
    def _add_section_to_indexes(self, section: DocumentSection) -> None:
        """
        Add section to internal indexes for efficient lookup.
        
        Args:
            section: Section to add to indexes
        """
        self._sections.append(section)
        
        # Update title index for fast title-based searches
        if section.title:
            if section.title not in self._title_index:
                self._title_index[section.title] = []
            self._title_index[section.title].append(section)
    
    def _remove_section_from_indexes(self, section: DocumentSection) -> None:
        """
        Remove section from internal indexes.
        
        Args:
            section: Section to remove from indexes
        """
        if section in self._sections:
            self._sections.remove(section)
        
        # Update title index
        if section.title in self._title_index:
            self._title_index[section.title] = [
                s for s in self._title_index[section.title] if s != section
            ]
            if not self._title_index[section.title]:
                del self._title_index[section.title]
    
    def add_section(self, section: DocumentSection) -> None:
        """
        Add a section to the tree, automatically determining proper hierarchy.
        
        Uses intelligent hierarchy detection based on header levels to place
        the section in the appropriate location within the tree structure.
        
        Args:
            section: Section to add to the tree
            
        Raises:
            TypeError: If section is not a DocumentSection
        """
        if not isinstance(section, DocumentSection):
            raise TypeError(f"Section must be DocumentSection, got: {type(section)}")
        
        self._add_section_to_indexes(section)
        
        if self.root is None:
            self.root = section
            logger.debug(f"Set '{section.title}' as root section")
            return
        
        # Find appropriate parent based on header levels and document order
        parent = self._find_parent_for_section(section)
        if parent:
            parent.add_child(section)
            logger.debug(f"Added '{section.title}' as child of '{parent.title}'")
        else:
            # Handle sections that should be siblings of root or new roots
            if section.level <= self.root.level:
                # For simplicity in this implementation, we keep the first root
                # In a more sophisticated version, we might create a virtual root
                logger.debug(f"'{section.title}' could be root-level, keeping as orphan")
            else:
                self.root.add_child(section)
                logger.debug(f"Added '{section.title}' as child of root '{self.root.title}'")
    
    def _find_parent_for_section(self, section: DocumentSection) -> Optional[DocumentSection]:
        """
        Find the appropriate parent for a section based on header levels and document order.
        
        Uses the most recent section with a lower header level as the parent,
        which matches the natural document hierarchy expectations.
        
        Args:
            section: Section needing a parent
            
        Returns:
            Appropriate parent section or None if no suitable parent found
        """
        # Find the most recent section with level < section.level
        for i in range(len(self._sections) - 1, -1, -1):
            candidate = self._sections[i]
            if candidate.level < section.level:
                return candidate
        return None
    
    def remove_section(self, section: DocumentSection) -> bool:
        """
        Remove a section from the tree.
        
        Args:
            section: Section to remove
            
        Returns:
            True if section was found and removed, False otherwise
        """
        if section not in self._sections:
            return False
        
        # Remove from parent if it has one
        if section.parent:
            section.parent.remove_child(section)
        
        # If this is the root, find a new root or set to None
        if section == self.root:
            self.root = self._sections[1] if len(self._sections) > 1 else None
        
        # Remove from indexes
        self._remove_section_from_indexes(section)
        
        logger.debug(f"Removed section '{section.title}' from tree")
        return True
    
    def get_section_count(self) -> int:
        """
        Get total number of sections in the tree.
        
        Returns:
            Total section count
        """
        return len(self._sections)
    
    def find_section_by_title(self, title: str, case_sensitive: bool = True) -> Optional[DocumentSection]:
        """
        Find a section by title with optimized lookup.
        
        Uses internal title index for O(1) average case performance.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            First section with matching title, or None if not found
        """
        if case_sensitive:
            sections = self._title_index.get(title, [])
            return sections[0] if sections else None
        else:
            # Fall back to linear search for case-insensitive
            for section in self._sections:
                if section.title.lower() == title.lower():
                    return section
            return None
    
    def find_all_sections_by_title(self, title: str, case_sensitive: bool = True) -> List[DocumentSection]:
        """
        Find all sections with a given title.
        
        Args:
            title: Title to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            List of sections with matching title
        """
        if case_sensitive:
            return self._title_index.get(title, []).copy()
        else:
            return [
                section for section in self._sections
                if section.title.lower() == title.lower()
            ]
    
    def get_sections_by_level(self, level: int) -> List[DocumentSection]:
        """
        Get all sections at a specific header level with validation.
        
        Args:
            level: Header level to filter by (0-6)
            
        Returns:
            List of sections at the specified level
            
        Raises:
            ValueError: If level is not in valid range
        """
        if not isinstance(level, int) or level < 0 or level > 6:
            raise ValueError(f"Level must be integer 0-6, got: {level}")
        
        return [section for section in self._sections if section.level == level]
    
    def get_sections_by_level_range(self, min_level: int, max_level: int) -> List[DocumentSection]:
        """
        Get sections within a level range.
        
        Args:
            min_level: Minimum level (inclusive)
            max_level: Maximum level (inclusive)
            
        Returns:
            List of sections within the level range
        """
        return [
            section for section in self._sections
            if min_level <= section.level <= max_level
        ]
    
    def traverse_breadth_first(self) -> List[DocumentSection]:
        """
        Traverse the tree in breadth-first order.
        
        Returns:
            List of sections in breadth-first order
        """
        if not self.root:
            return []
        
        result = []
        queue = [self.root]
        
        while queue:
            section = queue.pop(0)
            result.append(section)
            queue.extend(section.children)
        
        return result
    
    def traverse_depth_first(self) -> List[DocumentSection]:
        """
        Traverse the tree in depth-first order.
        
        Returns:
            List of sections in depth-first order
        """
        if not self.root:
            return []
        
        result = []
        
        def dfs(section: DocumentSection) -> None:
            result.append(section)
            for child in section.children:
                dfs(child)
        
        dfs(self.root)
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tree to dictionary for serialization with metadata.
        
        Returns:
            Dictionary representation with tree structure and metadata
        """
        return {
            "metadata": {
                "total_sections": len(self._sections),
                "max_depth": max(section.get_depth() for section in self._sections) if self._sections else 0,
                "levels_present": list(set(section.level for section in self._sections)),
                "has_root": self.root is not None
            },
            "root": self.root.to_dict() if self.root else None
        }
    
    def get_table_of_contents(self, max_level: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate table of contents with optional level filtering.
        
        Args:
            max_level: Maximum header level to include (None for all levels)
            
        Returns:
            List of TOC entries with enhanced metadata
        """
        toc = []
        for section in self._sections:
            if section.title and (max_level is None or section.level <= max_level):
                toc.append({
                    "title": section.title,
                    "level": section.level,
                    "line_number": section.line_number,
                    "depth": section.get_depth(),
                    "children_count": len(section.children),
                    "content_length": len(section.content)
                })
        return toc
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentTree(sections={len(self._sections)}, "
            f"root='{self.root.title if self.root else None}')"
        )


class HeaderBasedSplitter:
    """
    High-performance document splitter for markdown header-based sectioning.
    
    Provides efficient algorithms for parsing markdown documents and extracting
    logical sections based on header hierarchy. Uses optimized regex patterns
    and intelligent content extraction to build accurate document representations.
    
    The splitter handles various edge cases including documents without headers,
    nested header hierarchies, and mixed content types while preserving the
    semantic structure of the original document.
    
    Attributes:
        parser: MarkdownParser instance for header detection
        _header_pattern: Compiled regex for header matching (performance optimization)
    
    Example:
        >>> parser = MarkdownParser()
        >>> splitter = HeaderBasedSplitter(parser)
        >>> tree = splitter.split_and_build_tree("# Main\\n\\nContent\\n\\n## Sub\\n\\nMore")
        >>> print(tree.get_section_count())
        2
    """
    
    def __init__(self, parser: MarkdownParser) -> None:
        """
        Initialize HeaderBasedSplitter with parser and optimization setup.
        
        Args:
            parser: MarkdownParser instance to use for header detection
            
        Raises:
            TypeError: If parser is not a MarkdownParser instance
        """
        if not isinstance(parser, MarkdownParser):
            raise TypeError(f"Parser must be MarkdownParser, got: {type(parser)}")
        
        self.parser = parser
        # Pre-compile regex pattern for performance
        self._header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        logger.debug("HeaderBasedSplitter initialized with MarkdownParser")
    
    def split_by_headers(self, document: str, preserve_whitespace: bool = True) -> List[DocumentSection]:
        """
        Split a document into sections based on headers with enhanced processing.
        
        Parses the document line by line, identifying headers and extracting
        content between them. Handles edge cases and provides options for
        whitespace preservation.
        
        Args:
            document: Markdown document text
            preserve_whitespace: Whether to preserve original whitespace formatting
            
        Returns:
            List of DocumentSection objects in document order
            
        Raises:
            TypeError: If document is not a string
            
        Example:
            >>> splitter = HeaderBasedSplitter(MarkdownParser())
            >>> doc = "# Main\\n\\nContent\\n\\n## Sub\\n\\nMore content"
            >>> sections = splitter.split_by_headers(doc)
            >>> print(len(sections))
            2
        """
        if not isinstance(document, str):
            raise TypeError(f"Document must be string, got: {type(document)}")
        
        if not document.strip():
            logger.debug("Empty document provided, returning empty list")
            return []
        
        lines = document.split('\n')
        sections = []
        current_section = None
        content_lines = []
        
        logger.debug(f"Processing document with {len(lines)} lines")
        
        for line_num, line in enumerate(lines, 1):
            # Check if line is a header using pre-compiled regex
            header_match = self._header_pattern.match(line.strip())
            
            if header_match:
                # Save previous section if exists
                if current_section is not None:
                    content = self._process_content_lines(content_lines, preserve_whitespace)
                    current_section.content = content
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = DocumentSection(
                    level=level,
                    title=title,
                    content="",
                    line_number=line_num
                )
                content_lines = []
                
                logger.debug(f"Found header: level={level}, title='{title}', line={line_num}")
            else:
                content_lines.append(line)
        
        # Handle final section or content without headers
        if current_section is not None:
            content = self._process_content_lines(content_lines, preserve_whitespace)
            current_section.content = content
            sections.append(current_section)
        elif content_lines and any(line.strip() for line in content_lines):
            # Document has content but no headers
            content = self._process_content_lines(content_lines, preserve_whitespace)
            sections.append(DocumentSection(
                level=0,
                title="",
                content=content,
                line_number=1
            ))
            logger.debug("Created section for headerless content")
        
        logger.debug(f"Split document into {len(sections)} sections")
        return sections
    
    def _process_content_lines(self, lines: List[str], preserve_whitespace: bool) -> str:
        """
        Process content lines with whitespace handling options.
        
        Args:
            lines: List of content lines
            preserve_whitespace: Whether to preserve whitespace
            
        Returns:
            Processed content string
        """
        if preserve_whitespace:
            return '\n'.join(lines).strip()
        else:
            # Remove excessive whitespace while preserving paragraph breaks
            content = '\n'.join(lines).strip()
            # Normalize multiple newlines to double newlines (paragraph breaks)
            content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
            return content
    
    def build_tree(self, sections: List[DocumentSection]) -> DocumentTree:
        """
        Build a DocumentTree from a flat list of sections with validation.
        
        Args:
            sections: List of DocumentSection objects in document order
            
        Returns:
            DocumentTree with proper hierarchy established
            
        Raises:
            TypeError: If sections is not a list or contains non-DocumentSection items
        """
        if not isinstance(sections, list):
            raise TypeError(f"Sections must be list, got: {type(sections)}")
        
        for i, section in enumerate(sections):
            if not isinstance(section, DocumentSection):
                raise TypeError(f"Section {i} is not DocumentSection: {type(section)}")
        
        tree = DocumentTree()
        
        logger.debug(f"Building tree from {len(sections)} sections")
        
        for section in sections:
            tree.add_section(section)
        
        logger.debug(f"Built tree with {tree.get_section_count()} sections")
        return tree
    
    def split_and_build_tree(
        self, 
        document: str, 
        preserve_whitespace: bool = True
    ) -> DocumentTree:
        """
        Split document into sections and build tree in one optimized operation.
        
        Combines splitting and tree building with shared validation and
        performance optimizations for common use cases.
        
        Args:
            document: Markdown document text
            preserve_whitespace: Whether to preserve original whitespace
            
        Returns:
            DocumentTree representing the complete document structure
        """
        sections = self.split_by_headers(document, preserve_whitespace)
        return self.build_tree(sections)
    
    def analyze_document_structure(self, document: str) -> Dict[str, Any]:
        """
        Analyze document structure and provide detailed metrics.
        
        Args:
            document: Markdown document text
            
        Returns:
            Dictionary with structural analysis data
        """
        sections = self.split_by_headers(document)
        tree = self.build_tree(sections)
        
        if not sections:
            return {"structure": "empty", "sections": 0}
        
        levels = [section.level for section in sections if section.level > 0]
        content_lengths = [len(section.content) for section in sections]
        
        return {
            "structure": "hierarchical" if len(set(levels)) > 1 else "flat",
            "total_sections": len(sections),
            "header_sections": len([s for s in sections if s.level > 0]),
            "headerless_sections": len([s for s in sections if s.level == 0]),
            "max_depth": max([s.get_depth() for s in sections]) if sections else 0,
            "levels_used": sorted(set(levels)) if levels else [],
            "avg_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
            "total_content_length": sum(content_lengths),
            "table_of_contents": tree.get_table_of_contents()
        }


class SectionExtractor:
    """
    Advanced utility class for extracting and filtering document sections.
    
    Provides a comprehensive set of methods for targeted section extraction,
    content filtering, and document analysis. Supports various search criteria
    including title matching, level filtering, pattern matching, and custom
    predicate functions.
    
    Designed for high-performance operations on large document trees with
    caching and optimization features for repeated queries.
    
    Example:
        >>> extractor = SectionExtractor()
        >>> tree = build_sample_tree()
        >>> methods_section = extractor.extract_by_title(tree, "Methods")
        >>> all_level2 = extractor.extract_by_level_range(tree, 2, 2)
    """
    
    def __init__(self, cache_enabled: bool = True) -> None:
        """
        Initialize SectionExtractor with optional caching.
        
        Args:
            cache_enabled: Whether to enable result caching for performance
        """
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Any] = {}
        
        logger.debug(f"SectionExtractor initialized with caching: {cache_enabled}")
    
    def _get_cache_key(self, operation: str, *args: Any) -> str:
        """Generate cache key for operation and arguments."""
        return f"{operation}:{hash(str(args))}"
    
    def _cache_result(self, key: str, result: Any) -> None:
        """Cache result if caching is enabled."""
        if self.cache_enabled:
            self._cache[key] = result
    
    def _get_cached_result(self, key: str) -> Any:
        """Get cached result if available."""
        if self.cache_enabled:
            return self._cache.get(key)
        return None
    
    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
        logger.debug("SectionExtractor cache cleared")
    
    def extract_by_title(
        self, 
        tree: DocumentTree, 
        title: str, 
        case_sensitive: bool = True
    ) -> Optional[DocumentSection]:
        """
        Extract a section by title with caching support.
        
        Args:
            tree: DocumentTree to search
            title: Title to find
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Section with matching title, or None if not found
            
        Raises:
            TypeError: If tree is not a DocumentTree
        """
        if not isinstance(tree, DocumentTree):
            raise TypeError(f"Tree must be DocumentTree, got: {type(tree)}")
        
        cache_key = self._get_cache_key("extract_by_title", id(tree), title, case_sensitive)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = tree.find_section_by_title(title, case_sensitive)
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted section by title '{title}': {'found' if result else 'not found'}")
        return result
    
    def extract_by_level_range(
        self, 
        tree: DocumentTree, 
        min_level: int, 
        max_level: int
    ) -> List[DocumentSection]:
        """
        Extract sections within a level range with validation.
        
        Args:
            tree: DocumentTree to search
            min_level: Minimum header level (inclusive)
            max_level: Maximum header level (inclusive)
            
        Returns:
            List of sections within the level range, ordered by document position
            
        Raises:
            ValueError: If level range is invalid
        """
        if min_level > max_level:
            raise ValueError(f"min_level ({min_level}) cannot be greater than max_level ({max_level})")
        
        cache_key = self._get_cache_key("extract_by_level_range", id(tree), min_level, max_level)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = tree.get_sections_by_level_range(min_level, max_level)
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted {len(result)} sections in level range {min_level}-{max_level}")
        return result
    
    def extract_with_children(self, tree: DocumentTree, title: str) -> Optional[DocumentSection]:
        """
        Extract a section and all its children (returns section with full subtree).
        
        Args:
            tree: DocumentTree to search
            title: Title of section to extract
            
        Returns:
            Section with all children preserved, or None if not found
        """
        return tree.find_section_by_title(title)
    
    def extract_by_predicate(
        self, 
        tree: DocumentTree, 
        predicate: Callable[[DocumentSection], bool]
    ) -> List[DocumentSection]:
        """
        Extract sections matching a custom predicate function.
        
        Args:
            tree: DocumentTree to search
            predicate: Function that takes DocumentSection and returns bool
            
        Returns:
            List of sections matching the predicate
            
        Example:
            >>> # Extract sections with long content
            >>> long_sections = extractor.extract_by_predicate(
            ...     tree, lambda s: len(s.content) > 100
            ... )
        """
        if not callable(predicate):
            raise TypeError("Predicate must be callable")
        
        result = [section for section in tree._sections if predicate(section)]
        
        logger.debug(f"Extracted {len(result)} sections matching custom predicate")
        return result
    
    def extract_by_title_pattern(self, tree: DocumentTree, pattern: str) -> List[DocumentSection]:
        """
        Extract sections with titles matching a regex pattern.
        
        Args:
            tree: DocumentTree to search
            pattern: Regex pattern to match against titles
            
        Returns:
            List of sections with matching titles
            
        Raises:
            re.error: If pattern is invalid regex
        """
        try:
            regex = re.compile(pattern)
        except re.error as e:
            raise re.error(f"Invalid regex pattern '{pattern}': {e}")
        
        cache_key = self._get_cache_key("extract_by_title_pattern", id(tree), pattern)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            return cached
        
        result = []
        for section in tree._sections:
            if section.title and regex.search(section.title):
                result.append(section)
        
        self._cache_result(cache_key, result)
        
        logger.debug(f"Extracted {len(result)} sections matching pattern '{pattern}'")
        return result
    
    def extract_by_content_pattern(self, tree: DocumentTree, pattern: str) -> List[DocumentSection]:
        """
        Extract sections with content matching a regex pattern.
        
        Args:
            tree: DocumentTree to search
            pattern: Regex pattern to match against content
            
        Returns:
            List of sections with matching content
        """
        try:
            regex = re.compile(pattern, re.MULTILINE | re.DOTALL)
        except re.error as e:
            raise re.error(f"Invalid regex pattern '{pattern}': {e}")
        
        result = []
        for section in tree._sections:
            if section.content and regex.search(section.content):
                result.append(section)
        
        logger.debug(f"Extracted {len(result)} sections with content matching pattern")
        return result
    
    def generate_table_of_contents(
        self, 
        tree: DocumentTree,
        max_level: Optional[int] = None,
        include_line_numbers: bool = True,
        include_content_stats: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate enhanced table of contents with configurable options.
        
        Args:
            tree: DocumentTree to analyze
            max_level: Maximum header level to include
            include_line_numbers: Whether to include line numbers
            include_content_stats: Whether to include content statistics
            
        Returns:
            List of TOC entries with requested metadata
        """
        toc = []
        
        for section in tree._sections:
            if not section.title:  # Skip sections without titles
                continue
            
            if max_level is not None and section.level > max_level:
                continue
            
            entry = {
                "title": section.title,
                "level": section.level,
            }
            
            if include_line_numbers:
                entry["line_number"] = section.line_number
            
            if include_content_stats:
                entry.update({
                    "content_length": len(section.content),
                    "children_count": len(section.children),
                    "depth": section.get_depth()
                })
            
            toc.append(entry)
        
        logger.debug(f"Generated TOC with {len(toc)} entries")
        return toc
    
    def extract_outline(self, tree: DocumentTree, max_depth: Optional[int] = None) -> str:
        """
        Generate a textual outline of the document structure.
        
        Args:
            tree: DocumentTree to outline
            max_depth: Maximum depth to include in outline
            
        Returns:
            String representation of document outline
        """
        if not tree.root:
            return "Empty document"
        
        lines = []
        
        def add_section_to_outline(section: DocumentSection, depth: int = 0) -> None:
            if max_depth is not None and depth > max_depth:
                return
            
            indent = "  " * depth
            level_indicator = "#" * section.level if section.level > 0 else "•"
            title = section.title or "(untitled)"
            
            lines.append(f"{indent}{level_indicator} {title}")
            
            for child in section.children:
                add_section_to_outline(child, depth + 1)
        
        add_section_to_outline(tree.root)
        
        return "\n".join(lines)
    
    def find_section_path(self, tree: DocumentTree, target_title: str) -> List[str]:
        """
        Find the path from root to a section with given title.
        
        Args:
            tree: DocumentTree to search
            target_title: Title of target section
            
        Returns:
            List of section titles from root to target (empty if not found)
        """
        target = tree.find_section_by_title(target_title)
        if not target:
            return []
        
        path = []
        current = target
        while current:
            if current.title:  # Only include titled sections in path
                path.insert(0, current.title)
            current = current.parent
        
        return path


# NEW CLASSES: Recursive Chunking Algorithm

from typing import Protocol

class BoundaryStrategy(Enum):
    """
    Enumeration of available boundary detection strategies.
    
    Defines the different approaches for finding optimal chunk boundaries,
    allowing users to customize chunking behavior based on their specific needs.
    """
    INTELLIGENT = "intelligent"  # Use all available boundary types with smart prioritization
    SENTENCE_ONLY = "sentence_only"  # Only respect sentence boundaries
    PARAGRAPH_ONLY = "paragraph_only"  # Only respect paragraph boundaries
    WORD_ONLY = "word_only"  # Only use word boundaries (fastest)
    MARKUP_AWARE = "markup_aware"  # Respect markdown/HTML markup boundaries


class ChunkingMetrics(Protocol):
    """Protocol for chunking metrics collection."""
    
    def record_chunk_created(self, chunk_size: int, boundary_type: str) -> None:
        """Record a chunk creation event."""
        ...
    
    def record_boundary_search(self, search_time_ms: float, boundary_type: str) -> None:
        """Record boundary search performance."""
        ...


@dataclass
class ChunkConfig:
    """
    Advanced configuration class for recursive chunking parameters.
    
    Provides comprehensive control over the chunking algorithm with validation,
    serialization support, and performance tuning options. Supports both basic
    and advanced use cases with sensible defaults.
    
    Key Features:
    - Flexible boundary strategies with intelligent fallbacks
    - Performance tuning parameters for large documents
    - Content-aware options for different text types
    - Validation with detailed error messages
    - JSON serialization for configuration persistence
    - Metrics collection support for optimization
    
    Attributes:
        chunk_size: Maximum size of each chunk in characters (100-10000)
        chunk_overlap: Number of characters to overlap between chunks (0 to chunk_size-1)
        min_chunk_size: Minimum acceptable chunk size to prevent tiny chunks (1 to chunk_size/2)
        preserve_sentences: Whether to respect sentence boundaries when chunking
        preserve_paragraphs: Whether to respect paragraph boundaries when chunking
        preserve_code_blocks: Whether to keep code blocks intact (recommended: True)
        preserve_tables: Whether to keep markdown tables intact
        boundary_strategy: Strategy for boundary detection (see BoundaryStrategy enum)
        max_boundary_search_distance: Maximum characters to search for optimal boundary
        enable_smart_overlap: Use content-aware overlap positioning
        performance_mode: Optimize for speed over boundary quality
        content_type_hints: List of content type hints for specialized processing
        metrics_collector: Optional metrics collection interface
    
    Example:
        >>> # Basic configuration
        >>> config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        
        >>> # Advanced configuration for code documentation
        >>> config = ChunkConfig(
        ...     chunk_size=800,
        ...     chunk_overlap=150,
        ...     preserve_code_blocks=True,
        ...     boundary_strategy=BoundaryStrategy.MARKUP_AWARE,
        ...     content_type_hints=["technical", "code"]
        ... )
        
        >>> # Performance-optimized configuration
        >>> config = ChunkConfig(
        ...     chunk_size=2000,
        ...     chunk_overlap=100,
        ...     boundary_strategy=BoundaryStrategy.WORD_ONLY,
        ...     performance_mode=True
        ... )
    """
    
    # Core chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    
    # Content preservation options
    preserve_sentences: bool = True
    preserve_paragraphs: bool = True
    preserve_code_blocks: bool = True
    preserve_tables: bool = True
    preserve_links: bool = True
    
    # Advanced boundary detection
    boundary_strategy: BoundaryStrategy = BoundaryStrategy.INTELLIGENT
    max_boundary_search_distance: int = 100
    sentence_min_length: int = 10  # Minimum length to consider as sentence
    paragraph_min_length: int = 50  # Minimum length to consider as paragraph
    
    # Performance and optimization
    enable_smart_overlap: bool = True
    performance_mode: bool = False
    cache_boundary_patterns: bool = True
    
    # Content type awareness
    content_type_hints: List[str] = None
    language_code: Optional[str] = None  # For language-specific processing
    
    # Metrics and debugging
    metrics_collector: Optional[ChunkingMetrics] = None
    debug_mode: bool = False
    
    def __post_init__(self) -> None:
        """
        Validate configuration parameters with comprehensive checks.
        
        Performs extensive validation of all parameters to ensure they are
        within valid ranges and compatible with each other. Provides detailed
        error messages for debugging configuration issues.
        
        Raises:
            ValueError: If any parameter is invalid or incompatible
            TypeError: If parameter types are incorrect
        """
        # Initialize mutable defaults
        if self.content_type_hints is None:
            self.content_type_hints = []
        
        # Core parameter validation
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive integer, got: {self.chunk_size}")
        
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size should be at least 100 for meaningful chunks, got: {self.chunk_size}")
        
        if self.chunk_size > 10000:
            logger.warning(f"Large chunk_size ({self.chunk_size}) may impact performance")
        
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative integer, got: {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        
        if self.chunk_overlap > self.chunk_size * 0.5:
            logger.warning(
                f"Large overlap ratio ({self.chunk_overlap/self.chunk_size:.1%}) may cause excessive duplication"
            )
        
        # Min chunk size validation
        if not isinstance(self.min_chunk_size, int) or self.min_chunk_size <= 0:
            raise ValueError(f"min_chunk_size must be positive integer, got: {self.min_chunk_size}")
        
        if self.min_chunk_size > self.chunk_size // 2:
            raise ValueError(
                f"min_chunk_size ({self.min_chunk_size}) should not exceed half of chunk_size ({self.chunk_size//2})"
            )
        
        # Boundary strategy validation
        if not isinstance(self.boundary_strategy, BoundaryStrategy):
            raise TypeError(f"boundary_strategy must be BoundaryStrategy enum, got: {type(self.boundary_strategy)}")
        
        # Search distance validation
        if self.max_boundary_search_distance < 10:
            raise ValueError("max_boundary_search_distance should be at least 10")
        
        if self.max_boundary_search_distance > self.chunk_size:
            logger.warning(
                f"max_boundary_search_distance ({self.max_boundary_search_distance}) "
                f"exceeds chunk_size ({self.chunk_size}), limiting to chunk_size"
            )
            self.max_boundary_search_distance = self.chunk_size
        
        # Content type hints validation
        if not isinstance(self.content_type_hints, list):
            raise TypeError(f"content_type_hints must be list, got: {type(self.content_type_hints)}")
        
        # Language code validation
        if self.language_code is not None and not isinstance(self.language_code, str):
            raise TypeError(f"language_code must be string or None, got: {type(self.language_code)}")
        
        # Compatibility checks
        if self.performance_mode and self.boundary_strategy == BoundaryStrategy.INTELLIGENT:
            logger.info("Performance mode enabled, using simplified boundary detection")
            self.boundary_strategy = BoundaryStrategy.WORD_ONLY
        
        logger.debug(
            f"ChunkConfig validated: size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"strategy={self.boundary_strategy.value}, performance={self.performance_mode}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dictionary representation with all configuration parameters
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_chunk_size": self.min_chunk_size,
            "preserve_sentences": self.preserve_sentences,
            "preserve_paragraphs": self.preserve_paragraphs,
            "preserve_code_blocks": self.preserve_code_blocks,
            "preserve_tables": self.preserve_tables,
            "preserve_links": self.preserve_links,
            "boundary_strategy": self.boundary_strategy.value,
            "max_boundary_search_distance": self.max_boundary_search_distance,
            "sentence_min_length": self.sentence_min_length,
            "paragraph_min_length": self.paragraph_min_length,
            "enable_smart_overlap": self.enable_smart_overlap,
            "performance_mode": self.performance_mode,
            "cache_boundary_patterns": self.cache_boundary_patterns,
            "content_type_hints": self.content_type_hints.copy(),
            "language_code": self.language_code,
            "debug_mode": self.debug_mode
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChunkConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration parameters
            
        Returns:
            ChunkConfig instance
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Convert boundary strategy string back to enum
        if "boundary_strategy" in data and isinstance(data["boundary_strategy"], str):
            try:
                data["boundary_strategy"] = BoundaryStrategy(data["boundary_strategy"])
            except ValueError as e:
                raise ValueError(f"Invalid boundary_strategy '{data['boundary_strategy']}': {e}")
        
        # Create instance with validated parameters
        return cls(**data)
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert configuration to JSON string.
        
        Args:
            indent: Optional indentation for pretty printing
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkConfig':
        """
        Create configuration from JSON string.
        
        Args:
            json_str: JSON string with configuration
            
        Returns:
            ChunkConfig instance
            
        Raises:
            json.JSONDecodeError: If JSON is invalid
            ValueError: If configuration parameters are invalid
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def copy(self, **overrides) -> 'ChunkConfig':
        """
        Create a copy of this configuration with optional parameter overrides.
        
        Args:
            **overrides: Parameters to override in the copy
            
        Returns:
            New ChunkConfig instance with overrides applied
            
        Example:
            >>> config = ChunkConfig(chunk_size=1000)
            >>> small_config = config.copy(chunk_size=500, chunk_overlap=100)
        """
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data)
    
    def get_optimal_settings_for_content_type(self, content_type: str) -> 'ChunkConfig':
        """
        Get optimized configuration for specific content types.
        
        Args:
            content_type: Type of content ("technical", "narrative", "code", "academic")
            
        Returns:
            Optimized ChunkConfig for the content type
        """
        optimizations = {
            "technical": {
                "preserve_code_blocks": True,
                "preserve_tables": True,
                "boundary_strategy": BoundaryStrategy.MARKUP_AWARE,
                "sentence_min_length": 15
            },
            "narrative": {
                "preserve_sentences": True,
                "preserve_paragraphs": True,
                "boundary_strategy": BoundaryStrategy.SENTENCE_ONLY,
                "chunk_size": 1200
            },
            "code": {
                "preserve_code_blocks": True,
                "preserve_sentences": False,
                "boundary_strategy": BoundaryStrategy.MARKUP_AWARE,
                "chunk_overlap": 50
            },
            "academic": {
                "preserve_sentences": True,
                "preserve_paragraphs": True,
                "chunk_size": 800,
                "chunk_overlap": 150
            }
        }
        
        if content_type in optimizations:
            return self.copy(**optimizations[content_type])
        else:
            logger.warning(f"Unknown content type '{content_type}', using default settings")
            return self.copy()
    
    def validate_compatibility_with_text(self, text: str) -> List[str]:
        """
        Validate configuration compatibility with specific text.
        
        Args:
            text: Text to analyze for compatibility
            
        Returns:
            List of warning messages (empty if no issues)
        """
        warnings = []
        text_length = len(text)
        
        if text_length < self.chunk_size:
            warnings.append(f"Text length ({text_length}) is smaller than chunk_size ({self.chunk_size})")
        
        if text_length < self.min_chunk_size * 2:
            warnings.append(f"Text is very short, may not benefit from chunking")
        
        # Check for code blocks if preservation is enabled
        if self.preserve_code_blocks and "```" in text:
            code_blocks = text.count("```") // 2
            warnings.append(f"Found {code_blocks} code blocks, ensure chunk_size can accommodate them")
        
        # Check sentence structure if sentence preservation is enabled
        if self.preserve_sentences and text.count('.') < 3:
            warnings.append("Few sentence boundaries found, may not benefit from sentence preservation")
        
        return warnings
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ChunkConfig(size={self.chunk_size}, overlap={self.chunk_overlap}, "
            f"strategy={self.boundary_strategy.value}, performance={self.performance_mode})"
        )


@dataclass
class ChunkResult:
    """
    Comprehensive representation of a chunking operation result.
    
    Contains the chunk content along with detailed metadata about its position,
    relationships to other chunks, boundary information, and quality metrics.
    Provides methods for analysis, validation, and serialization.
    
    This class serves as the primary data container for chunk information and
    supports various operations needed for chunk management, analysis, and
    quality assessment.
    
    Attributes:
        content: The actual text content of the chunk
        start_position: Starting character position in original text (0-indexed)
        end_position: Ending character position in original text (exclusive)
        chunk_index: Sequential index of this chunk (0-based)
        overlap_with_previous: Number of characters overlapping with previous chunk
        overlap_with_next: Number of characters overlapping with next chunk
        boundary_type: Type of boundary used for this chunk (sentence, paragraph, word, etc.)
        source_section: Optional reference to source DocumentSection
        language_detected: Detected language code (if language detection is enabled)
        content_type: Detected content type (prose, code, table, list, etc.)
        quality_score: Quality assessment score (0.0-1.0)
        processing_metadata: Additional metadata from processing pipeline
    
    Example:
        >>> chunk = ChunkResult(
        ...     content="This is a sample chunk.",
        ...     start_position=0,
        ...     end_position=24,
        ...     chunk_index=0,
        ...     boundary_type="sentence"
        ... )
        >>> print(chunk.get_length())
        24
        >>> print(chunk.get_content_type())
        prose
    """
    
    # Core chunk data
    content: str
    start_position: int
    end_position: int
    chunk_index: int
    
    # Overlap information
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    # Boundary and processing metadata
    boundary_type: str = "word"
    source_section: Optional[str] = None  # Title of source DocumentSection
    language_detected: Optional[str] = None
    content_type: str = "prose"
    quality_score: float = 1.0
    
    # Advanced metadata
    processing_metadata: Dict[str, Any] = None
    creation_timestamp: Optional[float] = None
    
    def __post_init__(self) -> None:
        """
        Validate chunk data and initialize computed properties.
        
        Raises:
            ValueError: If chunk data is invalid
        """
        # Initialize mutable defaults
        if self.processing_metadata is None:
            self.processing_metadata = {}
        
        if self.creation_timestamp is None:
            self.creation_timestamp = time.time()
        
        # Validation
        if not isinstance(self.content, str):
            raise ValueError(f"Content must be string, got: {type(self.content)}")
        
        if self.start_position < 0:
            raise ValueError(f"start_position cannot be negative: {self.start_position}")
        
        if self.end_position < self.start_position:
            raise ValueError(f"end_position ({self.end_position}) cannot be less than start_position ({self.start_position})")
        
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index cannot be negative: {self.chunk_index}")
        
        if self.overlap_with_previous < 0 or self.overlap_with_next < 0:
            raise ValueError("Overlap values cannot be negative")
        
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(f"quality_score must be between 0.0 and 1.0, got: {self.quality_score}")
        
        # Auto-detect content type if not specified or is default
        if self.content_type == "prose":
            self.content_type = self._detect_content_type()
        
        logger.debug(f"ChunkResult validated: index={self.chunk_index}, length={self.get_length()}, type={self.content_type}")
    
    def get_length(self) -> int:
        """
        Get the length of the chunk content.
        
        Returns:
            Length in characters
        """
        return len(self.content)
    
    def get_word_count(self) -> int:
        """
        Get approximate word count of the chunk.
        
        Returns:
            Number of words (whitespace-separated tokens)
        """
        return len(self.content.split())
    
    def get_line_count(self) -> int:
        """
        Get number of lines in the chunk.
        
        Returns:
            Number of lines
        """
        return self.content.count('\n') + 1 if self.content else 0
    
    def has_overlap(self) -> bool:
        """
        Check if this chunk has overlap with neighboring chunks.
        
        Returns:
            True if chunk has any overlap, False otherwise
        """
        return self.overlap_with_previous > 0 or self.overlap_with_next > 0
    
    def get_overlap_ratio(self) -> float:
        """
        Calculate the overlap ratio as a percentage of chunk content.
        
        Returns:
            Overlap ratio (0.0-1.0)
        """
        total_overlap = self.overlap_with_previous + self.overlap_with_next
        content_length = self.get_length()
        
        if content_length == 0:
            return 0.0
        
        return min(1.0, total_overlap / content_length)
    
    def _detect_content_type(self) -> str:
        """
        Automatically detect content type based on content analysis.
        
        Returns:
            Detected content type string
        """
        content = self.content.strip()
        
        if not content:
            return "empty"
        
        # Code block detection
        if content.startswith("```") or "```" in content:
            return "code_block"
        
        # Inline code detection (high ratio of backticks)
        backtick_ratio = content.count("`") / len(content)
        if backtick_ratio > 0.05:
            return "code_heavy"
        
        # List detection
        lines = content.split('\n')
        list_indicators = sum(1 for line in lines if line.strip().startswith(('- ', '* ', '+ ', '1. ', '2. ')))
        if list_indicators > len(lines) * 0.5:
            return "list"
        
        # Table detection
        pipe_lines = sum(1 for line in lines if '|' in line and line.count('|') >= 2)
        if pipe_lines > len(lines) * 0.3:
            return "table"
        
        # Header detection
        header_lines = sum(1 for line in lines if line.strip().startswith('#'))
        if header_lines > 0:
            return "header_heavy"
        
        # Link-heavy content
        link_count = content.count('[') + content.count('http')
        if link_count > len(content.split()) * 0.1:
            return "link_heavy"
        
        # Mathematical content
        math_indicators = ['$$', '$', '\\(', '\\)', '\\[', '\\]']
        if any(indicator in content for indicator in math_indicators):
            return "mathematical"
        
        return "prose"
    
    def get_content_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the chunk content.
        
        Returns:
            Dictionary with content statistics
        """
        content = self.content
        lines = content.split('\n')
        
        # Character analysis
        char_counts = {
            'total': len(content),
            'letters': sum(1 for c in content if c.isalpha()),
            'digits': sum(1 for c in content if c.isdigit()),
            'whitespace': sum(1 for c in content if c.isspace()),
            'punctuation': sum(1 for c in content if not c.isalnum() and not c.isspace())
        }
        
        # Word analysis
        words = content.split()
        word_stats = {
            'count': len(words),
            'avg_length': sum(len(word) for word in words) / len(words) if words else 0,
            'unique_count': len(set(word.lower() for word in words))
        }
        
        # Line analysis
        line_stats = {
            'count': len(lines),
            'avg_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'empty_lines': sum(1 for line in lines if not line.strip())
        }
        
        # Markdown elements
        markdown_stats = {
            'headers': content.count('#'),
            'bold_markers': content.count('**'),
            'italic_markers': content.count('*') - content.count('**') * 2,
            'code_backticks': content.count('`'),
            'links': content.count('['),
            'images': content.count('![')
        }
        
        return {
            'characters': char_counts,
            'words': word_stats,
            'lines': line_stats,
            'markdown': markdown_stats,
            'content_type': self.content_type,
            'estimated_reading_time_seconds': word_stats['count'] * 0.25  # ~240 WPM
        }
    
    def assess_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of the chunk based on various metrics.
        
        Returns:
            Dictionary with quality assessment results
        """
        content = self.content.strip()
        
        if not content:
            return {
                'overall_score': 0.0,
                'issues': ['Empty content'],
                'recommendations': ['Content should not be empty']
            }
        
        issues = []
        recommendations = []
        score_factors = []
        
        # Length assessment
        length = len(content)
        if length < 50:
            issues.append(f"Very short content ({length} chars)")
            recommendations.append("Consider merging with adjacent chunks")
            score_factors.append(0.6)
        elif length > 2000:
            issues.append(f"Very long content ({length} chars)")
            recommendations.append("Consider splitting into smaller chunks")
            score_factors.append(0.8)
        else:
            score_factors.append(1.0)
        
        # Completeness assessment
        if content.endswith(('.', '!', '?', ':', ';')):
            score_factors.append(1.0)
        elif content.endswith(','):
            issues.append("Ends with comma (incomplete sentence)")
            recommendations.append("Adjust boundary to complete sentence")
            score_factors.append(0.7)
        else:
            issues.append("Ends abruptly (incomplete)")
            recommendations.append("Adjust boundary to natural stopping point")
            score_factors.append(0.5)
        
        # Structure assessment
        lines = content.split('\n')
        if len(lines) > 1:
            # Multi-line content should have reasonable structure
            empty_lines = sum(1 for line in lines if not line.strip())
            if empty_lines / len(lines) > 0.5:
                issues.append("Too many empty lines")
                score_factors.append(0.8)
            else:
                score_factors.append(1.0)
        else:
            score_factors.append(1.0)
        
        # Content type consistency
        if self.boundary_type == "sentence" and not any(content.endswith(p) for p in '.!?'):
            issues.append("Boundary type 'sentence' but doesn't end with sentence punctuation")
            score_factors.append(0.7)
        
        # Calculate overall score
        overall_score = sum(score_factors) / len(score_factors) if score_factors else 0.0
        
        return {
            'overall_score': round(overall_score, 2),
            'issues': issues,
            'recommendations': recommendations,
            'score_factors': score_factors
        }
    
    def to_dict(self, include_content: bool = True, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Convert chunk result to dictionary for serialization.
        
        Args:
            include_content: Whether to include the actual content text
            include_metadata: Whether to include processing metadata
            
        Returns:
            Dictionary representation
        """
        result = {
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "length": self.get_length(),
            "word_count": self.get_word_count(),
            "line_count": self.get_line_count(),
            "overlap_with_previous": self.overlap_with_previous,
            "overlap_with_next": self.overlap_with_next,
            "boundary_type": self.boundary_type,
            "content_type": self.content_type,
            "quality_score": self.quality_score,
            "has_overlap": self.has_overlap(),
            "overlap_ratio": self.get_overlap_ratio()
        }
        
        if include_content:
            result["content"] = self.content
        
        if self.source_section:
            result["source_section"] = self.source_section
        
        if self.language_detected:
            result["language_detected"] = self.language_detected
        
        if self.creation_timestamp:
            result["creation_timestamp"] = self.creation_timestamp
        
        if include_metadata and self.processing_metadata:
            result["processing_metadata"] = self.processing_metadata.copy()
        
        return result
    
    def get_preview(self, max_length: int = 100) -> str:
        """
        Get a preview of the chunk content.
        
        Args:
            max_length: Maximum length of preview
            
        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_length:
            return self.content
        
        truncated = self.content[:max_length - 3]
        
        # Try to truncate at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only if we don't lose too much
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    def compare_with(self, other: 'ChunkResult') -> Dict[str, Any]:
        """
        Compare this chunk with another chunk.
        
        Args:
            other: Another ChunkResult to compare with
            
        Returns:
            Dictionary with comparison results
        """
        if not isinstance(other, ChunkResult):
            raise TypeError(f"Can only compare with ChunkResult, got: {type(other)}")
        
        return {
            'length_difference': self.get_length() - other.get_length(),
            'word_count_difference': self.get_word_count() - other.get_word_count(),
            'quality_difference': self.quality_score - other.quality_score,
            'same_content_type': self.content_type == other.content_type,
            'same_boundary_type': self.boundary_type == other.boundary_type,
            'index_distance': abs(self.chunk_index - other.chunk_index),
            'position_distance': abs(self.start_position - other.start_position)
        }
    
    def is_adjacent_to(self, other: 'ChunkResult') -> bool:
        """
        Check if this chunk is adjacent to another chunk.
        
        Args:
            other: Another ChunkResult to check
            
        Returns:
            True if chunks are adjacent in the original text
        """
        return (self.end_position == other.start_position or 
                other.end_position == self.start_position)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        preview = self.get_preview(50)
        return (
            f"ChunkResult(index={self.chunk_index}, length={self.get_length()}, "
            f"type={self.content_type}, preview='{preview}')"
        )
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on content and position."""
        if not isinstance(other, ChunkResult):
            return NotImplemented
        return (
            self.content == other.content and
            self.start_position == other.start_position and
            self.end_position == other.end_position
        )
    
    def __hash__(self) -> int:
        """Hash based on content and position for use in sets/dicts."""
        return hash((self.content, self.start_position, self.end_position))


class ChunkBoundary:
    """
    Advanced intelligent boundary detection system for chunking operations.
    
    Provides sophisticated algorithms for finding optimal splitting points in text
    that preserve semantic meaning, respect content structure, and follow user
    configuration preferences. Includes performance optimizations, caching,
    and content-aware processing.
    
    Key Features:
    - Multi-strategy boundary detection with intelligent fallbacks
    - Content-aware processing for different text types
    - Performance optimization with compiled regex patterns and caching
    - Comprehensive logging and debugging support
    - Extensible architecture for custom boundary strategies
    - Quality scoring for boundary selection
    
    Attributes:
        config: ChunkConfig instance with boundary preferences
        _compiled_patterns: Pre-compiled regex patterns for performance
        _boundary_cache: Cache for boundary detection results
        _performance_stats: Performance tracking data
    
    Example:
        >>> config = ChunkConfig(preserve_sentences=True, preserve_code_blocks=True)
        >>> detector = ChunkBoundary(config)
        >>> boundary = detector.find_optimal_boundary(text, target_position=500)
    """
    
    def __init__(self, config: ChunkConfig) -> None:
        """
        Initialize boundary detector with configuration and optimization setup.
        
        Args:
            config: ChunkConfig instance with boundary preferences
            
        Raises:
            TypeError: If config is not a ChunkConfig instance
        """
        if not isinstance(config, ChunkConfig):
            raise TypeError(f"Config must be ChunkConfig, got: {type(config)}")
        
        self.config = config
        
        # Performance optimization: pre-compile all regex patterns
        self._compiled_patterns = self._compile_boundary_patterns()
        
        # Caching for repeated boundary searches
        self._boundary_cache: Dict[str, int] = {} if config.cache_boundary_patterns else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance tracking
        self._performance_stats = {
            'boundary_searches': 0,
            'total_search_time': 0.0,
            'strategy_usage': {}
        }
        
        logger.debug(f"ChunkBoundary initialized with strategy: {config.boundary_strategy.value}")
    
    def _compile_boundary_patterns(self) -> Dict[str, re.Pattern]:
        """
        Pre-compile all regex patterns for optimal performance.
        
        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {}
        
        # Sentence boundaries - enhanced for better detection with fixed-width lookbehind
        patterns['sentence_simple'] = re.compile(r'[.!?]+\s+')
        patterns['sentence_complex'] = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')  # Lookahead for capital letter
        
        # Fixed abbreviation-aware sentence detection using alternative approach
        # Instead of variable-width lookbehind, use a more robust method
        abbreviations = ['Mr', 'Mrs', 'Dr', 'Prof', 'vs', 'etc', 'i.e', 'e.g', 'Jr', 'Sr', 'Ph.D', 'M.D']
        # Create a pattern that matches abbreviations and then checks for sentence endings
        patterns['sentence_abbreviation_aware'] = re.compile(
            r'(?:^|[^A-Za-z])(?:' + '|'.join(re.escape(abbr) for abbr in abbreviations) + r')\.[A-Za-z]*'
        )
        patterns['sentence_end'] = re.compile(r'[.!?]+\s+(?=[A-Z])')
        
        # Paragraph boundaries with improved detection
        patterns['paragraph_double_newline'] = re.compile(r'\n\s*\n')
        patterns['paragraph_markdown'] = re.compile(r'\n(?=\s*(?:#|\*|-|\d+\.))')  # Before markdown elements
        patterns['paragraph_section'] = re.compile(r'\n(?=\s*[A-Z][^a-z]*:)')  # Section headers like "SECTION:"
        
        # Word boundaries with enhanced detection
        patterns['word_space'] = re.compile(r'\s+')
        patterns['word_punctuation'] = re.compile(r'(?<=[^\w\s])\s+')
        patterns['word_after_sentence'] = re.compile(r'(?<=[.!?])\s+(?=\w)')
        
        # Code and markup boundaries with comprehensive coverage
        patterns['code_block_fenced'] = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        patterns['code_block_indented'] = re.compile(r'(?:^|\n)(?: {4}|\t)[^\n]*(?:\n(?: {4}|\t)[^\n]*)*', re.MULTILINE)
        patterns['inline_code'] = re.compile(r'`[^`\n]+`')
        patterns['html_tag'] = re.compile(r'<[^>]+>')
        
        # Markdown structure patterns with performance optimization
        patterns['markdown_header'] = re.compile(r'^#{1,6}\s+.*$', re.MULTILINE)
        patterns['markdown_list_bullet'] = re.compile(r'^[\s]*[-*+]\s+.*$', re.MULTILINE)
        patterns['markdown_list_numbered'] = re.compile(r'^[\s]*\d+\.\s+.*$', re.MULTILINE)
        patterns['markdown_blockquote'] = re.compile(r'^[\s]*>\s+.*$', re.MULTILINE)
        
        # Table boundaries with robust detection
        patterns['table_row'] = re.compile(r'\|.*\|')
        patterns['table_separator'] = re.compile(r'^\s*\|?[-:\s\|]+\|?\s*$', re.MULTILINE)
        patterns['table_simple'] = re.compile(r'^\s*\|[^|]*\|[^|]*\|\s*$', re.MULTILINE)
        
        # Link and media boundaries
        patterns['markdown_link'] = re.compile(r'\[[^\]]*\]\([^)]*\)')
        patterns['markdown_image'] = re.compile(r'!\[[^\]]*\]\([^)]*\)')
        patterns['markdown_ref_link'] = re.compile(r'\[[^\]]*\]\[[^\]]*\]')
        patterns['url_standalone'] = re.compile(r'https?://[^\s<>"{}|\\^`[\]]+')
        
        # Mathematical expressions with enhanced support
        patterns['math_block'] = re.compile(r'\$\$[\s\S]*?\$\$')
        patterns['math_inline'] = re.compile(r'(?<!\$)\$[^$\n]+\$(?!\$)')  # Avoid matching $$ blocks
        patterns['latex_command'] = re.compile(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*')
        
        # Additional boundary markers for improved detection
        patterns['yaml_frontmatter'] = re.compile(r'^---\s*\n.*?\n---\s*\n', re.MULTILINE | re.DOTALL)
        patterns['horizontal_rule'] = re.compile(r'^[\s]*[-*_]{3,}[\s]*$', re.MULTILINE)
        patterns['comment_html'] = re.compile(r'<!--[\s\S]*?-->')
        patterns['comment_code'] = re.compile(r'//.*$|/\*[\s\S]*?\*/', re.MULTILINE)
        
        # Performance optimization: pre-compile search ranges
        patterns['whitespace_sequence'] = re.compile(r'\s{2,}')
        patterns['punctuation_sequence'] = re.compile(r'[.!?]{2,}')
        
        if self.config.debug_mode:
            logger.debug(f"Compiled {len(patterns)} optimized boundary detection patterns")
        
        return patterns
    
    def find_optimal_boundary(self, text: str, target_position: int) -> int:
        """
        Find the optimal boundary position near the target position using intelligent strategies.
        
        Uses multiple strategies based on configuration and content analysis to find
        the best splitting point. Includes performance optimization and quality scoring.
        
        Args:
            text: Text to analyze for boundaries
            target_position: Desired position to find boundary near
            
        Returns:
            Position of optimal boundary
            
        Raises:
            ValueError: If target_position is invalid
        """
        start_time = time.time()
        
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got: {type(text)}")
        
        if target_position < 0 or target_position > len(text):
            raise ValueError(f"target_position ({target_position}) must be between 0 and {len(text)}")
        
        if target_position >= len(text):
            return len(text)
        
        # Check cache first
        cache_key = None
        if self._boundary_cache is not None:
            cache_key = f"{hash(text[:target_position+100])}:{target_position}"
            if cache_key in self._boundary_cache:
                self._cache_hits += 1
                boundary_pos = self._boundary_cache[cache_key]
                logger.debug(f"Boundary cache hit: position {boundary_pos}")
                return boundary_pos
            self._cache_misses += 1
        
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
        
        # Cache result
        if self._boundary_cache is not None and cache_key:
            self._boundary_cache[cache_key] = optimal_boundary
        
        # Update performance stats
        search_time = time.time() - start_time
        self._performance_stats['boundary_searches'] += 1
        self._performance_stats['total_search_time'] += search_time
        strategy_name = self.config.boundary_strategy.value
        self._performance_stats['strategy_usage'][strategy_name] = (
            self._performance_stats['strategy_usage'].get(strategy_name, 0) + 1
        )
        
        # Record metrics if collector is available
        if self.config.metrics_collector:
            self.config.metrics_collector.record_boundary_search(
                search_time * 1000,  # Convert to milliseconds
                strategy_name
            )
        
        logger.debug(
            f"Found optimal boundary at position {optimal_boundary} "
            f"(target: {target_position}, strategy: {strategy_name}, time: {search_time:.3f}s)"
        )
        
        return optimal_boundary
    
    def _find_intelligent_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """
        Use intelligent multi-strategy approach to find optimal boundaries.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of boundary candidates with quality scores
        """
        candidates = []
        
        # Priority order: avoid splitting inside protected elements first
        protected_elements = self._find_protected_elements(text, target_position)
        if protected_elements:
            for element in protected_elements:
                candidates.append({
                    'position': element['boundary'],
                    'type': element['type'],
                    'quality_score': element['quality'],
                    'reason': f"Avoiding split inside {element['type']}"
                })
        
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
        """
        Find protected elements that should not be split.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of protected element boundaries
        """
        protected = []
        search_range = (
            max(0, target_position - self.config.max_boundary_search_distance),
            min(len(text), target_position + self.config.max_boundary_search_distance)
        )
        
        # Code blocks
        if self.config.preserve_code_blocks:
            for match in self._compiled_patterns['code_block_fenced'].finditer(text):
                if search_range[0] <= match.start() <= search_range[1] or search_range[0] <= match.end() <= search_range[1]:
                    if match.start() <= target_position <= match.end():
                        # Inside code block - find boundary before or after
                        before_distance = target_position - match.start()
                        after_distance = match.end() - target_position
                        
                        if before_distance < after_distance:
                            boundary_pos = match.start()
                            reason = "before_code_block"
                        else:
                            boundary_pos = match.end()
                            reason = "after_code_block"
                        
                        protected.append({
                            'position': boundary_pos,
                            'type': 'code_block',
                            'quality': 0.9,
                            'boundary': boundary_pos,
                            'reason': reason
                        })
        
        # Tables
        if self.config.preserve_tables:
            table_rows = list(self._compiled_patterns['table_row'].finditer(text))
            if table_rows:
                for row_match in table_rows:
                    if search_range[0] <= row_match.start() <= search_range[1]:
                        if row_match.start() <= target_position <= row_match.end():
                            # Inside table row - try to find table boundary
                            table_start = self._find_table_start(text, row_match.start())
                            table_end = self._find_table_end(text, row_match.end())
                            
                            if target_position - table_start < table_end - target_position:
                                boundary_pos = table_start
                            else:
                                boundary_pos = table_end
                            
                            protected.append({
                                'position': boundary_pos,
                                'type': 'table',
                                'quality': 0.85,
                                'boundary': boundary_pos,
                                'reason': 'table_boundary'
                            })
        
        # Links
        if self.config.preserve_links:
            for pattern_name in ['markdown_link', 'markdown_image', 'url_standalone']:
                for match in self._compiled_patterns[pattern_name].finditer(text):
                    if match.start() <= target_position <= match.end():
                        before_distance = target_position - match.start()
                        after_distance = match.end() - target_position
                        
                        boundary_pos = match.start() if before_distance < after_distance else match.end()
                        
                        protected.append({
                            'position': boundary_pos,
                            'type': pattern_name,
                            'quality': 0.8,
                            'boundary': boundary_pos,
                            'reason': f'preserve_{pattern_name}'
                        })
        
        return protected
    
    def _find_sentence_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """
        Find sentence boundaries with quality scoring.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of sentence boundary candidates
        """
        candidates = []
        search_start = max(0, target_position - self.config.max_boundary_search_distance)
        search_end = min(len(text), target_position + self.config.max_boundary_search_distance)
        search_text = text[search_start:search_end]
        
        # Try complex sentence detection first (highest quality)
        for match in self._compiled_patterns['sentence_complex'].finditer(search_text):
            abs_pos = search_start + match.end()  # Use end position for boundaries
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                # Check sentence length for quality assessment
                sentence_start = self._find_sentence_start(text, abs_pos)
                sentence_length = abs_pos - sentence_start
                
                quality = self._calculate_sentence_quality(sentence_length, distance)
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'sentence_complex',
                    'quality_score': quality,
                    'reason': f'Sentence boundary (length: {sentence_length})'
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
        """
        Find paragraph boundaries with enhanced detection.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of paragraph boundary candidates
        """
        candidates = []
        search_start = max(0, target_position - self.config.max_boundary_search_distance)
        search_end = min(len(text), target_position + self.config.max_boundary_search_distance)
        search_text = text[search_start:search_end]
        
        # Double newline paragraphs (highest quality)
        for match in self._compiled_patterns['paragraph_double_newline'].finditer(search_text):
            abs_pos = search_start + match.start()
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                # Check paragraph length for quality assessment
                para_start = self._find_paragraph_start(text, abs_pos)
                para_length = abs_pos - para_start
                
                quality = self._calculate_paragraph_quality(para_length, distance)
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'paragraph_double_newline',
                    'quality_score': quality,
                    'reason': f'Paragraph boundary (length: {para_length})'
                })
        
        # Markdown structure paragraphs
        for match in self._compiled_patterns['paragraph_markdown'].finditer(search_text):
            abs_pos = search_start + match.start()
            distance = abs(abs_pos - target_position)
            
            if distance <= self.config.max_boundary_search_distance:
                quality = max(0.7, 1.0 - (distance / self.config.max_boundary_search_distance))
                
                candidates.append({
                    'position': abs_pos,
                    'type': 'paragraph_markdown',
                    'quality_score': quality,
                    'reason': 'Markdown structure boundary'
                })
        
        return candidates
    
    def _find_word_boundaries(self, text: str, target_position: int) -> List[Dict[str, Any]]:
        """
        Find word boundaries as fallback option.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of word boundary candidates
        """
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
        """
        Find boundaries that respect markdown and HTML markup.
        
        Args:
            text: Text to analyze
            target_position: Target position
            
        Returns:
            List of markup-aware boundary candidates
        """
        candidates = []
        
        # First check for protected elements
        protected = self._find_protected_elements(text, target_position)
        for element in protected:
            candidates.append({
                'position': element['boundary'],
                'type': element['type'],
                'quality': element['quality'],
                'reason': element['reason']
            })
        
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
        
        # List boundaries
        for pattern_name in ['markdown_list_bullet', 'markdown_list_numbered']:
            for match in self._compiled_patterns[pattern_name].finditer(search_text):
                abs_pos = search_start + match.start()
                distance = abs(abs_pos - target_position)
                
                if distance <= self.config.max_boundary_search_distance:
                    quality = max(0.6, 1.0 - (distance / self.config.max_boundary_search_distance))
                    
                    candidates.append({
                        'position': abs_pos,
                        'type': pattern_name,
                        'quality_score': quality,
                        'reason': f'{pattern_name} boundary'
                    })
        
        # If no markup boundaries found, fall back to paragraph and sentence boundaries
        if not candidates:
            candidates.extend(self._find_paragraph_boundaries(text, target_position))
            candidates.extend(self._find_sentence_boundaries(text, target_position))
        
        # Final fallback to word boundaries
        if not candidates:
            candidates.extend(self._find_word_boundaries(text, target_position))
        
        return candidates
    
    def _select_optimal_boundary(
        self, 
        candidates: List[Dict[str, Any]], 
        target_position: int, 
        text: str
    ) -> int:
        """
        Select the optimal boundary from candidates based on quality scores and preferences.
        
        Args:
            candidates: List of boundary candidates with quality scores
            target_position: Original target position
            text: Original text
            
        Returns:
            Position of optimal boundary
        """
        if not candidates:
            logger.warning(f"No boundary candidates found, using target position {target_position}")
            return min(target_position, len(text))

        # Sort candidates by configuration preferences first, then quality
        def score_candidate(candidate):
            distance = abs(candidate['position'] - target_position)
            quality = candidate['quality_score']
            boundary_type = candidate['type']
            
            # Base priority based on configuration preferences
            priority_bonus = 0.0
            
            # Prioritize paragraph boundaries when preserve_paragraphs is enabled (HIGHEST priority)
            if self.config.preserve_paragraphs and 'paragraph' in boundary_type:
                priority_bonus = 2.5  # Highest preference for paragraph boundaries
            # Prioritize sentence boundaries when preserve_sentences is enabled
            elif self.config.preserve_sentences and 'sentence' in boundary_type:
                priority_bonus = 2.0  # Strong preference for sentence boundaries
            # Protected elements (code blocks, tables) get high priority
            elif boundary_type in ['code_block', 'table', 'markup']:
                priority_bonus = 1.8  # High preference for protected elements
            # Word boundaries get lower priority when other preservation is on
            elif 'word' in boundary_type and (self.config.preserve_sentences or self.config.preserve_paragraphs):
                priority_bonus = -0.5  # Lower preference for word boundaries
            
            # Distance penalty (prefer closer boundaries)
            distance_penalty = distance / self.config.max_boundary_search_distance
            
            # Combined score: priority + quality - distance penalty
            adjusted_score = quality + priority_bonus - (distance_penalty * 0.2)
            
            return adjusted_score

        candidates.sort(key=score_candidate, reverse=True)
        
        # Select best candidate
        optimal_candidate = candidates[0]
        
        if self.config.debug_mode:
            logger.debug(
                f"Selected boundary: position={optimal_candidate['position']}, "
                f"type={optimal_candidate['type']}, quality={optimal_candidate['quality_score']:.2f}, "
                f"reason={optimal_candidate['reason']}"
            )
            
            if len(candidates) > 1:
                logger.debug(f"Alternative candidates: {[(c['position'], c['type'], c['quality_score']) for c in candidates[1:3]]}")
        
        return optimal_candidate['position']
    
    def _calculate_sentence_quality(self, sentence_length: int, distance: int) -> float:
        """Calculate quality score for sentence boundaries."""
        # Prefer sentences of reasonable length
        length_score = 1.0
        if sentence_length < self.config.sentence_min_length:
            length_score = 0.5
        elif sentence_length > 200:
            length_score = 0.8
        
        # Distance penalty
        distance_score = max(0.2, 1.0 - (distance / self.config.max_boundary_search_distance))
        
        return (length_score + distance_score) / 2
    
    def _calculate_paragraph_quality(self, paragraph_length: int, distance: int) -> float:
        """Calculate quality score for paragraph boundaries."""
        # Prefer paragraphs of reasonable length
        length_score = 1.0
        if paragraph_length < self.config.paragraph_min_length:
            length_score = 0.6
        elif paragraph_length > 500:
            length_score = 0.9
        
        # Distance penalty
        distance_score = max(0.3, 1.0 - (distance / self.config.max_boundary_search_distance))
        
        return (length_score + distance_score) / 2
    
    def _find_sentence_start(self, text: str, position: int) -> int:
        """Find the start of the sentence containing the given position."""
        for i in range(position, max(0, position - 200), -1):
            if i == 0 or text[i-1:i+1] in ['. ', '! ', '? ', '\n\n']:
                return i
        return max(0, position - 200)
    
    def _find_paragraph_start(self, text: str, position: int) -> int:
        """Find the start of the paragraph containing the given position."""
        for i in range(position, max(0, position - 500), -1):
            if i == 0 or text[i-2:i] == '\n\n':
                return i
        return max(0, position - 500)
    
    def _find_table_start(self, text: str, position: int) -> int:
        """Find the start of a table containing the given position."""
        lines = text[:position].split('\n')
        for i in range(len(lines) - 1, -1, -1):
            if not self._compiled_patterns['table_row'].match(lines[i]):
                return sum(len(line) + 1 for line in lines[:i+1])
        return 0
    
    def _find_table_end(self, text: str, position: int) -> int:
        """Find the end of a table containing the given position."""
        lines = text[position:].split('\n')
        for i, line in enumerate(lines):
            if not self._compiled_patterns['table_row'].match(line):
                return position + sum(len(lines[j]) + 1 for j in range(i))
        return len(text)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for boundary detection.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self._performance_stats.copy()
        
        if stats['boundary_searches'] > 0:
            stats['average_search_time'] = stats['total_search_time'] / stats['boundary_searches']
        else:
            stats['average_search_time'] = 0.0
        
        if self._boundary_cache is not None:
            stats['cache_stats'] = {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0,
                'cache_size': len(self._boundary_cache)
            }
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the boundary detection cache."""
        if self._boundary_cache is not None:
            self._boundary_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.debug("Boundary detection cache cleared")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"ChunkBoundary(strategy={self.config.boundary_strategy.value}, cache={'enabled' if self._boundary_cache else 'disabled'})"


class RecursiveChunker:
    """
    Advanced recursive text chunking engine with intelligent boundary detection.
    
    Provides sophisticated methods to chunk text and document sections while preserving
    semantic meaning, respecting configuration constraints, and optimizing for various
    content types. Features comprehensive error handling, performance monitoring,
    and extensible architecture.
    
    Key Features:
    - Intelligent boundary-aware chunking with multiple strategies
    - Content-type specific optimization (code, prose, technical documentation)
    - Performance monitoring and optimization with caching
    - Quality assessment and validation for generated chunks
    - Comprehensive logging and debugging support
    - Extensible design for custom chunking strategies
    
    Attributes:
        config: ChunkConfig instance with chunking parameters
        boundary_detector: ChunkBoundary instance for finding optimal split points
        _chunking_stats: Performance and quality statistics
        _chunk_cache: Optional caching for repeated chunking operations
        
    Example:
        >>> config = ChunkConfig(chunk_size=1000, chunk_overlap=200)
        >>> chunker = RecursiveChunker(config)
        >>> chunks = chunker.chunk_text("Long document text...")
        >>> print(f"Created {len(chunks)} chunks")
        
        >>> # Advanced usage with document tree
        >>> sections = chunker.chunk_sections(document_tree)
        >>> stats = chunker.get_chunking_statistics(chunks)
    """
    
    def __init__(self, config: ChunkConfig) -> None:
        """
        Initialize recursive chunker with comprehensive setup.
        
        Args:
            config: ChunkConfig instance with chunking parameters
            
        Raises:
            TypeError: If config is not a ChunkConfig instance
        """
        if not isinstance(config, ChunkConfig):
            raise TypeError(f"Config must be ChunkConfig, got: {type(config)}")
        
        self.config = config
        self.boundary_detector = ChunkBoundary(config)
        
        # Performance and quality tracking
        self._chunking_stats = {
            'total_chunks_created': 0,
            'total_text_processed': 0,
            'total_processing_time': 0.0,
            'boundary_type_usage': {},
            'content_type_distribution': {},
            'quality_scores': []
        }
        
        # Optional caching for repeated operations
        self._chunk_cache: Optional[Dict[str, List[ChunkResult]]] = {} if config.cache_boundary_patterns else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.debug(f"RecursiveChunker initialized with chunk_size={config.chunk_size}, strategy={config.boundary_strategy.value}")
    
    def chunk_text(self, text: str, source_metadata: Optional[Dict[str, Any]] = None) -> List[ChunkResult]:
        """
        Chunk text into manageable pieces with intelligent boundary detection.
        
        Implements the core recursive chunking algorithm with advanced boundary detection,
        overlap management, and quality assessment. Handles edge cases and provides
        comprehensive error handling.
        
        Args:
            text: Text to chunk
            source_metadata: Optional metadata about the source of the text
            
        Returns:
            List of ChunkResult objects with detailed metadata
            
        Raises:
            ValueError: If text is invalid
            
        Example:
            >>> chunker = RecursiveChunker(ChunkConfig(chunk_size=500))
            >>> chunks = chunker.chunk_text("A long document...")
            >>> for chunk in chunks:
            ...     print(f"Chunk {chunk.chunk_index}: {len(chunk.content)} chars")
        """
        start_time = time.time()
        
        # Input validation
        if not isinstance(text, str):
            raise TypeError(f"Text must be string, got: {type(text)}")
        
        if not text or not text.strip():
            logger.debug("Empty or whitespace-only text provided")
            return []
        
        text = text.strip()
        
        # Check cache first
        cache_key = None
        if self._chunk_cache is not None:
            cache_key = f"{hash(text)}:{self.config.chunk_size}:{self.config.chunk_overlap}"
            if cache_key in self._chunk_cache:
                self._cache_hits += 1
                logger.debug(f"Chunk cache hit for text of length {len(text)}")
                return self._chunk_cache[cache_key].copy()
            self._cache_misses += 1
        
        # Handle single chunk case
        if len(text) <= self.config.chunk_size:
            chunk = ChunkResult(
                content=text,
                start_position=0,
                end_position=len(text),
                chunk_index=0,
                boundary_type="complete",
                source_section=source_metadata.get('section_title') if source_metadata else None,
                processing_metadata=source_metadata or {}
            )
            
            result = [chunk]
            
            # Cache result
            if self._chunk_cache is not None and cache_key:
                self._chunk_cache[cache_key] = result.copy()
            
            # Update statistics
            self._update_chunking_stats([chunk], time.time() - start_time)
            
            logger.debug(f"Single chunk created for text of length {len(text)}")
            return result
        
        # Multi-chunk processing
        chunks = []
        current_position = 0
        chunk_index = 0
        
        # Compatibility warnings
        warnings = self.config.validate_compatibility_with_text(text)
        if warnings:
            for warning in warnings:
                logger.warning(f"Chunking compatibility: {warning}")
        
        while current_position < len(text):
            try:
                chunk = self._create_next_chunk(
                    text=text,
                    current_position=current_position,
                    chunk_index=chunk_index,
                    source_metadata=source_metadata
                )
                
                if chunk is None:  # Safety check
                    logger.warning(f"Failed to create chunk at position {current_position}")
                    break
                
                chunks.append(chunk)
                
                # Move to next position (without overlap to avoid double-counting)
                next_position = chunk.end_position - chunk.overlap_with_next
                
                # Ensure progress to prevent infinite loops
                if next_position <= current_position:
                    next_position = current_position + 1
                    logger.warning(f"Forced progress from position {current_position} to {next_position}")
                
                current_position = next_position
                chunk_index += 1
                
                # Safety check to prevent infinite loops
                if chunk_index > 10000:
                    logger.error("Chunking stopped after 10000 chunks to prevent infinite loop")
                    break
                    
            except Exception as e:
                logger.error(f"Error creating chunk at position {current_position}: {e}")
                # Try to recover by moving forward
                current_position = min(current_position + self.config.chunk_size, len(text))
                chunk_index += 1
        
        # Post-process chunks
        self._post_process_chunks(chunks)
        
        # Cache result
        if self._chunk_cache is not None and cache_key:
            self._chunk_cache[cache_key] = chunks.copy()
        
        # Update statistics
        self._update_chunking_stats(chunks, time.time() - start_time)
        
        logger.debug(f"Chunked text of length {len(text)} into {len(chunks)} chunks")
        return chunks
    
    def _create_next_chunk(
        self,
        text: str,
        current_position: int,
        chunk_index: int,
        source_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ChunkResult]:
        """
        Create the next chunk in the sequence with optimal boundary detection.
        
        Args:
            text: Full text being chunked
            current_position: Starting position for this chunk
            chunk_index: Index of this chunk in the sequence
            source_metadata: Optional source metadata
            
        Returns:
            ChunkResult or None if chunk cannot be created
        """
        if current_position >= len(text):
            return None
        
        # Calculate target end position for this chunk
        target_end = current_position + self.config.chunk_size
        
        # Handle final chunk
        if target_end >= len(text):
            chunk_content = text[current_position:]
            end_position = len(text)
            boundary_type = "end_of_text"
        else:
            # Find optimal boundary position
            boundary_pos = self.boundary_detector.find_optimal_boundary(text, target_end)
            boundary_pos = max(boundary_pos, current_position + 1)  # Ensure progress
            
            chunk_content = text[current_position:boundary_pos]
            end_position = boundary_pos
            boundary_type = "intelligent"  # Will be refined based on actual boundary found
        
        # Skip if chunk is too small (except for final chunk)
        if (len(chunk_content.strip()) < self.config.min_chunk_size and 
            end_position < len(text) and 
            current_position > 0):
            logger.debug(f"Skipping small chunk of size {len(chunk_content)} at position {current_position}")
            return None
        
        # Calculate overlaps
        overlap_prev, overlap_next = self._calculate_overlaps(
            text=text,
            current_position=current_position,
            end_position=end_position,
            chunk_index=chunk_index
        )
        
        # Adjust chunk content to include overlaps
        final_start = max(0, current_position - overlap_prev)
        final_end = min(len(text), end_position + overlap_next)
        final_content = text[final_start:final_end]
        
        # Determine more specific boundary type
        boundary_type = self._determine_boundary_type(text, end_position)
        
        # Create chunk with validation
        try:
            chunk = ChunkResult(
                content=final_content,
                start_position=final_start,
                end_position=final_end,
                chunk_index=chunk_index,
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next,
                boundary_type=boundary_type,
                source_section=source_metadata.get('section_title') if source_metadata else None,
                processing_metadata=source_metadata or {}
            )
            
            # Quality assessment
            if self.config.debug_mode:
                quality = chunk.assess_quality()
                logger.debug(f"Chunk {chunk_index} quality: {quality['overall_score']:.2f}")
            
            return chunk
            
        except Exception as e:
            logger.error(f"Failed to create ChunkResult: {e}")
            return None
    
    def _calculate_overlaps(
        self,
        text: str,
        current_position: int,
        end_position: int,
        chunk_index: int
    ) -> Tuple[int, int]:
        """
        Calculate optimal overlap amounts for previous and next chunks.
        
        Args:
            text: Full text
            current_position: Current chunk start position
            end_position: Current chunk end position
            chunk_index: Index of current chunk
            
        Returns:
            Tuple of (overlap_with_previous, overlap_with_next)
        """
        overlap_prev = 0
        overlap_next = 0
        
        # Calculate overlap with previous chunk
        if chunk_index > 0 and self.config.chunk_overlap > 0:
            max_prev_overlap = min(self.config.chunk_overlap, current_position)
            
            if self.config.enable_smart_overlap:
                # Try to find sentence or word boundary for overlap
                overlap_boundary = self.boundary_detector.find_optimal_boundary(
                    text, 
                    current_position - max_prev_overlap // 2
                )
                overlap_prev = current_position - max(0, overlap_boundary)
            else:
                overlap_prev = max_prev_overlap
        
        # Calculate overlap with next chunk
        if end_position < len(text) and self.config.chunk_overlap > 0:
            max_next_overlap = min(self.config.chunk_overlap, len(text) - end_position)
            
            if self.config.enable_smart_overlap:
                # Try to find sentence or word boundary for overlap
                overlap_boundary = self.boundary_detector.find_optimal_boundary(
                    text, 
                    end_position + max_next_overlap // 2
                )
                overlap_next = max(0, overlap_boundary - end_position)
            else:
                overlap_next = max_next_overlap
        
        return overlap_prev, overlap_next
    
    def _determine_boundary_type(self, text: str, position: int) -> str:
        """
        Determine the type of boundary that was actually used.
        
        Args:
            text: Full text
            position: Boundary position
            
        Returns:
            String describing the boundary type
        """
        if position >= len(text):
            return "end_of_text"
        
        # Check character at boundary position
        if position > 0:
            boundary_char = text[position-1:position+1]
            
            if boundary_char in ['. ', '! ', '? ']:
                return "sentence"
            elif text[position-2:position] == '\n\n':
                return "paragraph"
            elif text[position] == ' ':
                return "word"
            elif text[position] == '\n':
                return "line"
        
        return "forced"
    
    def _post_process_chunks(self, chunks: List[ChunkResult]) -> None:
        """
        Post-process chunks to improve quality and consistency.
        
        Args:
            chunks: List of chunks to post-process
        """
        if not chunks:
            return
        
        # Update quality scores based on context
        for i, chunk in enumerate(chunks):
            quality_factors = []
            
            # Length factor
            length_ratio = len(chunk.content) / self.config.chunk_size
            if 0.5 <= length_ratio <= 1.2:
                quality_factors.append(1.0)
            else:
                quality_factors.append(max(0.3, 1.0 - abs(length_ratio - 1.0)))
            
            # Boundary quality factor
            if chunk.boundary_type in ["sentence", "paragraph"]:
                quality_factors.append(1.0)
            elif chunk.boundary_type in ["word", "line"]:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # Position factor (prefer chunks that aren't at forced boundaries)
            if i == 0 or i == len(chunks) - 1:
                quality_factors.append(1.0)  # First and last chunks are naturally bounded
            else:
                quality_factors.append(0.9)
            
            # Update quality score
            chunk.quality_score = sum(quality_factors) / len(quality_factors)
    
    def chunk_sections(self, tree: DocumentTree, section_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Chunk sections from a DocumentTree with comprehensive metadata preservation.
        
        Processes each section in the document tree individually, maintaining
        hierarchical relationships and providing detailed section-level statistics.
        
        Args:
            tree: DocumentTree to process
            section_metadata: Whether to include detailed section metadata
            
        Returns:
            List of section data with chunks and comprehensive metadata
            
        Example:
            >>> chunked_sections = chunker.chunk_sections(document_tree)
            >>> for section_data in chunked_sections:
            ...     print(f"Section: {section_data['section_title']}")
            ...     print(f"Chunks: {section_data['chunk_count']}")
        """
        if not tree.root:
            logger.debug("Empty document tree provided")
            return []
        
        chunked_sections = []
        
        for section in tree._sections:
            if not section.content or not section.content.strip():
                logger.debug(f"Skipping empty section: {section.title}")
                continue
            
            # Prepare source metadata
            source_metadata = {
                'section_title': section.title,
                'section_level': section.level,
                'section_line_number': section.line_number,
                'parent_title': section.parent.title if section.parent else None,
                'section_depth': section.get_depth()
            }
            
            # Chunk the section content
            chunks = self.chunk_text(section.content, source_metadata)
            
            if chunks:  # Only include sections with actual chunks
                section_data = {
                    "section_title": section.title,
                    "section_level": section.level,
                    "section_line_number": section.line_number,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "total_content_length": len(section.content),
                    "section_depth": section.get_depth()
                }
                
                if section_metadata:
                    # Add detailed section statistics
                    section_data.update({
                        "parent_title": section.parent.title if section.parent else None,
                        "child_count": len(section.children),
                        "avg_chunk_size": sum(len(chunk.content) for chunk in chunks) / len(chunks),
                        "quality_scores": [chunk.quality_score for chunk in chunks],
                        "boundary_types": [chunk.boundary_type for chunk in chunks],
                        "content_types": [chunk.content_type for chunk in chunks]
                    })
                
                chunked_sections.append(section_data)
        
        logger.debug(f"Chunked {len(chunked_sections)} sections from document tree")
        return chunked_sections
    
    def _update_chunking_stats(self, chunks: List[ChunkResult], processing_time: float) -> None:
        """Update internal statistics tracking."""
        self._chunking_stats['total_chunks_created'] += len(chunks)
        self._chunking_stats['total_text_processed'] += sum(len(chunk.content) for chunk in chunks)
        self._chunking_stats['total_processing_time'] += processing_time
        
        # Update boundary type usage
        for chunk in chunks:
            boundary_type = chunk.boundary_type
            self._chunking_stats['boundary_type_usage'][boundary_type] = (
                self._chunking_stats['boundary_type_usage'].get(boundary_type, 0) + 1
            )
            
            # Update content type distribution
            content_type = chunk.content_type
            self._chunking_stats['content_type_distribution'][content_type] = (
                self._chunking_stats['content_type_distribution'].get(content_type, 0) + 1
            )
            
            # Track quality scores
            self._chunking_stats['quality_scores'].append(chunk.quality_score)
    
    def get_chunking_statistics(self, chunks: Optional[List[ChunkResult]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about chunking operations.
        
        Args:
            chunks: Optional specific chunks to analyze, otherwise uses global stats
            
        Returns:
            Dictionary with detailed chunking statistics and performance metrics
        """
        if chunks is not None:
            # Analyze specific chunks
            return self._analyze_chunk_list(chunks)
        
        # Return global statistics
        stats = self._chunking_stats.copy()
        
        # Calculate derived metrics
        if stats['total_chunks_created'] > 0:
            stats['average_chunk_size'] = stats['total_text_processed'] / stats['total_chunks_created']
            stats['average_processing_time_per_chunk'] = stats['total_processing_time'] / stats['total_chunks_created']
        else:
            stats['average_chunk_size'] = 0
            stats['average_processing_time_per_chunk'] = 0
        
        # Quality statistics
        if stats['quality_scores']:
            stats['average_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['min_quality_score'] = min(stats['quality_scores'])
            stats['max_quality_score'] = max(stats['quality_scores'])
        else:
            stats['average_quality_score'] = 0
            stats['min_quality_score'] = 0
            stats['max_quality_score'] = 0
        
        # Cache statistics
        if self._chunk_cache is not None:
            stats['cache_stats'] = {
                'hits': self._cache_hits,
                'misses': self._cache_misses,
                'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0,
                'cached_results': len(self._chunk_cache)
            }
        
        return stats
    
    def _analyze_chunk_list(self, chunks: List[ChunkResult]) -> Dict[str, Any]:
        """Analyze a specific list of chunks."""
        if not chunks:
            return {
                "total_chunks": 0,
                "average_chunk_size": 0,
                "total_content_length": 0,
                "overlap_efficiency": 0
            }
        
        total_chunks = len(chunks)
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        total_content = sum(chunk_sizes)
        average_size = total_content / total_chunks if total_chunks > 0 else 0
        
        # Calculate overlap efficiency
        total_overlap = sum(chunk.overlap_with_next for chunk in chunks)
        overlap_efficiency = (total_overlap / total_content * 100) if total_content > 0 else 0
        
        # Quality analysis
        quality_scores = [chunk.quality_score for chunk in chunks]
        
        # Boundary type analysis
        boundary_types = {}
        content_types = {}
        for chunk in chunks:
            boundary_types[chunk.boundary_type] = boundary_types.get(chunk.boundary_type, 0) + 1
            content_types[chunk.content_type] = content_types.get(chunk.content_type, 0) + 1
        
        return {
            "total_chunks": total_chunks,
            "average_chunk_size": average_size,
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "total_content_length": total_content,
            "overlap_efficiency": overlap_efficiency,
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "boundary_type_distribution": boundary_types,
            "content_type_distribution": content_types
        }
    
    def clear_cache(self) -> None:
        """Clear the chunking cache and reset cache statistics."""
        if self._chunk_cache is not None:
            self._chunk_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.debug("Chunking cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the chunker and boundary detector."""
        chunker_stats = self.get_chunking_statistics()
        boundary_stats = self.boundary_detector.get_performance_stats()
        
        return {
            'chunker': chunker_stats,
            'boundary_detector': boundary_stats
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"RecursiveChunker(chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap}, strategy={self.config.boundary_strategy.value})"
        )


# ATOMIC UNIT HANDLER SYSTEM - GREEN PHASE IMPLEMENTATION

class AtomicUnitType(Enum):
    """Enumeration of atomic content unit types."""
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    BLOCKQUOTE = "blockquote"
    HORIZONTAL_RULE = "horizontal_rule"
    YAML_FRONTMATTER = "yaml_frontmatter"
    MATH_BLOCK = "math_block"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class AtomicUnit:
    """Represents a single atomic content unit with enhanced validation."""
    unit_type: AtomicUnitType
    content: str
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate atomic unit after creation."""
        if not isinstance(self.unit_type, AtomicUnitType):
            raise ValueError(f"unit_type must be an AtomicUnitType, got {type(self.unit_type)}")
        
        if not isinstance(self.content, str):
            raise ValueError(f"content must be a string, got {type(self.content)}")
        
        if not isinstance(self.start_position, int) or self.start_position < 0:
            raise ValueError(f"start_position must be a non-negative integer, got {self.start_position}")
        
        if not isinstance(self.end_position, int) or self.end_position < 0:
            raise ValueError(f"end_position must be a non-negative integer, got {self.end_position}")
        
        if self.start_position >= self.end_position:
            raise ValueError(f"start_position ({self.start_position}) must be less than end_position ({self.end_position})")
        
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata)}")
    
    def get_length(self) -> int:
        """Get the length of the content."""
        return len(self.content)
    
    def contains_position(self, position: int) -> bool:
        """Check if a position falls within this atomic unit."""
        if not isinstance(position, int):
            raise ValueError(f"position must be an integer, got {type(position)}")
        return self.start_position <= position < self.end_position
    
    def get_boundaries(self) -> Tuple[int, int]:
        """Get the start and end boundaries of this unit."""
        return (self.start_position, self.end_position)
    
    def overlaps_with_range(self, start: int, end: int) -> bool:
        """Check if this unit overlaps with a given range."""
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("start and end must be integers")
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        
        return not (self.end_position <= start or self.start_position >= end)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atomic unit to dictionary representation."""
        return {
            "unit_type": self.unit_type.value,
            "content": self.content,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "length": self.get_length(),
            "metadata": self.metadata.copy()
        }


class CodeBlockHandler:
    """Specialized handler for code blocks."""
    
    def detect(self, text: str) -> List[AtomicUnit]:
        """Detect code blocks in text - fenced and indented."""
        units = []
        lines = text.split('\n')
        
        # First pass: detect fenced code blocks and track their ranges
        fenced_ranges = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith('```') or stripped.startswith('~~~'):
                start_line = i
                start_pos = text.find(lines[i])
                
                # Find matching closing fence
                fence_marker = '```' if stripped.startswith('```') else '~~~'
                i += 1
                while i < len(lines) and not lines[i].strip().startswith(fence_marker):
                    i += 1
                
                if i < len(lines):  # Found closing fence
                    end_line = i
                    # Calculate actual text positions
                    end_pos = start_pos
                    for line_idx in range(start_line, end_line + 1):
                        end_pos += len(lines[line_idx]) + 1  # +1 for newline
                    end_pos -= 1  # Remove final newline
                    
                    fenced_content = '\n'.join(lines[start_line:end_line + 1])
                    metadata = self.extract_metadata(fenced_content)
                    
                    units.append(AtomicUnit(
                        unit_type=AtomicUnitType.CODE_BLOCK,
                        content=fenced_content,
                        start_position=start_pos,
                        end_position=end_pos,
                        metadata=metadata
                    ))
                    
                    # Track this range as fenced
                    fenced_ranges.append((start_line, end_line))
            i += 1
        
        # Second pass: detect indented code blocks, excluding fenced ranges
        current_indented_block = []
        block_start_line = None
        
        for i, line in enumerate(lines):
            # Skip lines that are within fenced code blocks
            in_fenced_range = any(start <= i <= end for start, end in fenced_ranges)
            if in_fenced_range:
                if current_indented_block:
                    # End current indented block if we hit a fenced block
                    self._add_indented_block(text, lines, current_indented_block, block_start_line, units)
                    current_indented_block = []
                    block_start_line = None
                continue
            
            # Check for indented lines (4+ spaces or tab)
            if line.startswith('    ') or line.startswith('\t'):
                if not current_indented_block:
                    block_start_line = i
                current_indented_block.append((i, line))
            elif line.strip() == '':
                # Empty line - continue current block if we have one
                if current_indented_block:
                    current_indented_block.append((i, line))
            else:
                # Non-indented, non-empty line - end current block
                if current_indented_block:
                    self._add_indented_block(text, lines, current_indented_block, block_start_line, units)
                    current_indented_block = []
                    block_start_line = None
        
        # Handle any remaining indented block
        if current_indented_block:
            self._add_indented_block(text, lines, current_indented_block, block_start_line, units)
        
        return units
    
    def _add_indented_block(self, text: str, lines: List[str], block_lines: List[tuple], start_line_idx: int, units: List[AtomicUnit]):
        """Helper to add an indented code block."""
        if not block_lines:
            return
        
        # Remove trailing empty lines
        while block_lines and block_lines[-1][1].strip() == '':
            block_lines.pop()
        
        if not block_lines:
            return
        
        # Calculate positions
        start_pos = 0
        for i in range(start_line_idx):
            start_pos += len(lines[i]) + 1  # +1 for newline
        
        end_line_idx = block_lines[-1][0]
        end_pos = start_pos
        for i, (line_idx, line_content) in enumerate(block_lines):
            end_pos += len(line_content)
            if i < len(block_lines) - 1:  # Add newline except for last line
                end_pos += 1
        
        block_content = '\n'.join(line_content for _, line_content in block_lines)
        metadata = self.extract_metadata(block_content)
        
        units.append(AtomicUnit(
            unit_type=AtomicUnitType.CODE_BLOCK,
            content=block_content,
            start_position=start_pos,
            end_position=end_pos,
            metadata=metadata
        ))
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from code block content."""
        metadata = {"line_count": content.count('\n') + 1}
        
        if content.startswith('```'):
            first_line = content.split('\n')[0]
            language = first_line.replace('```', '').strip()
            metadata["language"] = language
            metadata["block_type"] = "fenced"
            metadata["has_imports"] = "import " in content
        else:
            metadata["language"] = ""
            metadata["block_type"] = "indented"
            metadata["has_imports"] = False
        
        return metadata
    
    def validate(self, unit: AtomicUnit) -> bool:
        """Validate code block unit."""
        if unit.unit_type != AtomicUnitType.CODE_BLOCK:
            return False
        return "```" in unit.content or unit.content.startswith('    ')


class TableHandler:
    """Specialized handler for tables."""
    
    def detect(self, text: str) -> List[AtomicUnit]:
        """Detect tables in text."""
        units = []
        lines = text.split('\n')
        
        in_table = False
        table_start = 0
        table_lines = []
        
        for i, line in enumerate(lines):
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    table_start = sum(len(l) + 1 for l in lines[:i])
                table_lines.append(line)
            else:
                if in_table and table_lines:
                    content = '\n'.join(table_lines)
                    table_end = table_start + len(content)
                    
                    # Count columns from first row
                    col_count = table_lines[0].count('|') - 1 if table_lines[0].startswith('|') else table_lines[0].count('|') + 1
                    
                    # Check for header separator
                    has_separator = len(table_lines) > 1 and all(c in '-:|' for c in table_lines[1].replace(' ', ''))
                    
                    units.append(AtomicUnit(
                        unit_type=AtomicUnitType.TABLE,
                        content=content,
                        start_position=table_start,
                        end_position=table_end,
                        metadata={
                            "column_count": col_count,
                            "row_count": len(table_lines),
                            "has_header_separator": has_separator
                        }
                    ))
                    in_table = False
                    table_lines = []
        
        return units
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from table content."""
        lines = content.strip().split('\n')
        col_count = lines[0].count('|') - 1 if lines[0].startswith('|') else lines[0].count('|') + 1
        
        metadata = {
            "column_count": col_count,
            "row_count": len(lines),
            "has_header_separator": False,
            "column_alignments": ["left"] * col_count
        }
        
        # Check for header separator and alignment
        if len(lines) > 1:
            separator_line = lines[1].replace(' ', '')
            if all(c in '-:|' for c in separator_line):
                metadata["has_header_separator"] = True
                
                # Detect column alignments
                alignments = []
                parts = separator_line.split('|')[1:-1] if separator_line.startswith('|') else separator_line.split('|')
                for part in parts:
                    if part.startswith(':') and part.endswith(':'):
                        alignments.append("center")
                    elif part.endswith(':'):
                        alignments.append("right")
                    else:
                        alignments.append("left")
                metadata["column_alignments"] = alignments
        
        return metadata
    
    def validate(self, unit: AtomicUnit) -> bool:
        """Validate table unit."""
        if unit.unit_type != AtomicUnitType.TABLE:
            return False
        return '|' in unit.content and unit.content.count('|') >= 2


class ListHandler:
    """Specialized handler for lists."""
    
    def detect(self, text: str) -> List[AtomicUnit]:
        """Detect lists in text."""
        units = []
        lines = text.split('\n')
        
        in_list = False
        list_start = 0
        list_lines = []
        current_list_type = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            is_list_item = False
            list_type = None
            
            # Check for task lists FIRST (before bullet lists)
            if re.match(r'^- \[[xX \-]\]\s', stripped):
                is_list_item = True
                list_type = "task"
            # Check for bullet lists
            elif stripped.startswith(('- ', '* ', '+ ')):
                is_list_item = True
                list_type = "bullet"
            # Check for numbered lists
            elif re.match(r'^\d+[.)]\s', stripped):
                is_list_item = True
                list_type = "numbered"
            # Check for nested items (indented)
            elif in_list and (line.startswith('  ') or line.startswith('\t')):
                is_list_item = True
                list_type = current_list_type
            
            if is_list_item:
                if not in_list:
                    in_list = True
                    list_start = sum(len(l) + 1 for l in lines[:i])
                    current_list_type = list_type
                list_lines.append(line)
            else:
                if in_list and list_lines:
                    content = '\n'.join(list_lines)
                    list_end = list_start + len(content)
                    
                    metadata = self._analyze_list(list_lines, current_list_type)
                    
                    units.append(AtomicUnit(
                        unit_type=AtomicUnitType.LIST,
                        content=content,
                        start_position=list_start,
                        end_position=list_end,
                        metadata=metadata
                    ))
                    in_list = False
                    list_lines = []
                    current_list_type = None
        
        # Handle any remaining list at end of text
        if in_list and list_lines:
            content = '\n'.join(list_lines)
            list_end = list_start + len(content)
            
            metadata = self._analyze_list(list_lines, current_list_type)
            
            units.append(AtomicUnit(
                unit_type=AtomicUnitType.LIST,
                content=content,
                start_position=list_start,
                end_position=list_end,
                metadata=metadata
            ))
        
        return units
    
    def _analyze_list(self, lines: List[str], list_type: str) -> Dict[str, Any]:
        """Analyze list structure and extract metadata."""
        metadata = {
            "list_type": list_type,
            "item_count": 0,
            "has_nested_items": False,
            "max_nesting_depth": 1
        }
        
        if list_type == "bullet":
            # Detect marker type
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('- '):
                    metadata["marker"] = "-"
                    break
                elif stripped.startswith('* '):
                    metadata["marker"] = "*"
                    break
                elif stripped.startswith('+ '):
                    metadata["marker"] = "+"
                    break
        
        elif list_type == "numbered":
            # Detect marker style
            for line in lines:
                stripped = line.strip()
                if re.match(r'^\d+\.\s', stripped):
                    metadata["marker_style"] = "dot"
                    break
                elif re.match(r'^\d+\)\s', stripped):
                    metadata["marker_style"] = "paren"
                    break
        
        elif list_type == "task":
            completed = sum(1 for line in lines if re.search(r'\[[xX]\]', line))
            pending = sum(1 for line in lines if re.search(r'\[ \]', line))
            metadata.update({
                "completed_count": completed,
                "pending_count": pending,
                "total_tasks": completed + pending
            })
        
        # Count main items and detect nesting
        for line in lines:
            if not line.startswith(('  ', '\t')):
                metadata["item_count"] += 1
            else:
                metadata["has_nested_items"] = True
                # Count indentation depth
                depth = (len(line) - len(line.lstrip())) // 2 + 1
                metadata["max_nesting_depth"] = max(metadata["max_nesting_depth"], depth)
        
        return metadata
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from list content."""
        lines = content.split('\n')
        
        # Determine list type
        first_line = lines[0].strip()
        if re.match(r'^\d+[.)]\s', first_line):
            list_type = "numbered"
        elif re.match(r'^- \[[xX \-]\]\s', first_line):
            list_type = "task"
        else:
            list_type = "bullet"
        
        return self._analyze_list(lines, list_type)
    
    def validate(self, unit: AtomicUnit) -> bool:
        """Validate list unit."""
        if unit.unit_type != AtomicUnitType.LIST:
            return False
        
        first_line = unit.content.split('\n')[0].strip()
        return (first_line.startswith(('- ', '* ', '+ ')) or
                re.match(r'^\d+[.)]\s', first_line))


class BlockquoteHandler:
    """Specialized handler for blockquotes."""
    
    def detect(self, text: str) -> List[AtomicUnit]:
        """Detect blockquotes in text."""
        units = []
        lines = text.split('\n')
        
        in_quote = False
        quote_start = 0
        quote_lines = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('>'):
                if not in_quote:
                    in_quote = True
                    quote_start = sum(len(l) + 1 for l in lines[:i])
                quote_lines.append(line)
            else:
                if in_quote and quote_lines:
                    content = '\n'.join(quote_lines)
                    quote_end = quote_start + len(content)
                    
                    metadata = self._analyze_quote(quote_lines)
                    
                    units.append(AtomicUnit(
                        unit_type=AtomicUnitType.BLOCKQUOTE,
                        content=content,
                        start_position=quote_start,
                        end_position=quote_end,
                        metadata=metadata
                    ))
                    in_quote = False
                    quote_lines = []
        
        # Handle any remaining blockquote at end of text
        if in_quote and quote_lines:
            content = '\n'.join(quote_lines)
            quote_end = quote_start + len(content)
            
            metadata = self._analyze_quote(quote_lines)
            
            units.append(AtomicUnit(
                unit_type=AtomicUnitType.BLOCKQUOTE,
                content=content,
                start_position=quote_start,
                end_position=quote_end,
                metadata=metadata
            ))
        
        return units
    
    def _analyze_quote(self, lines: List[str]) -> Dict[str, Any]:
        """Analyze blockquote structure."""
        max_depth = 0
        nesting_levels = set()
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('>'):
                # Count depth by counting '>' characters, ignoring spaces between them
                depth = 0
                i = 0
                while i < len(stripped):
                    if stripped[i] == '>':
                        depth += 1
                        i += 1
                        # Skip any spaces after this '>'
                        while i < len(stripped) and stripped[i] == ' ':
                            i += 1
                    else:
                        break
                
                max_depth = max(max_depth, depth)
                nesting_levels.add(depth)
        
        # Check for attribution patterns
        has_attribution = any(':' in line or 'said' in line.lower() for line in lines)
        
        return {
            "max_depth": max_depth,
            "line_count": len(lines),
            "has_nested_quotes": max_depth > 1,
            "nesting_levels": sorted(list(nesting_levels)),
            "contains_attribution": has_attribution
        }
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from blockquote content."""
        lines = content.split('\n')
        return self._analyze_quote(lines)
    
    def validate(self, unit: AtomicUnit) -> bool:
        """Validate blockquote unit."""
        if unit.unit_type != AtomicUnitType.BLOCKQUOTE:
            return False
        return unit.content.strip().startswith('>')


class ParagraphHandler:
    """Specialized handler for paragraphs."""
    
    def detect(self, text: str) -> List[AtomicUnit]:
        """Detect paragraphs in text."""
        units = []
        
        # Split by double newlines to get paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        current_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                current_pos += len(para) + 2  # Account for newlines
                continue
            
            # Skip if it looks like other atomic units
            if (para.startswith(('```', '    ', '\t')) or  # Code
                para.startswith('>') or  # Blockquote
                '|' in para and para.count('|') >= 2 or  # Table
                re.match(r'^[\s]*[-*+]\s', para) or  # List
                re.match(r'^[\s]*\d+[.)]\s', para)):  # Numbered list
                current_pos = text.find(para, current_pos) + len(para) + 2
                continue
            
            start_pos = text.find(para, current_pos)
            end_pos = start_pos + len(para)
            
            metadata = self._analyze_paragraph(para)
            
            units.append(AtomicUnit(
                unit_type=AtomicUnitType.PARAGRAPH,
                content=para,
                start_position=start_pos,
                end_position=end_pos,
                metadata=metadata
            ))
            
            current_pos = end_pos + 2
        
        return units
    
    def _analyze_paragraph(self, content: str) -> Dict[str, Any]:
        """Analyze paragraph content."""
        sentences = re.split(r'[.!?]+\s+', content)
        words = content.split()
        
        return {
            "sentence_count": len([s for s in sentences if s.strip()]),
            "word_count": len(words),
            "line_count": content.count('\n') + 1,
            "has_formatting": '**' in content or '*' in content,
            "has_punctuation": any(p in content for p in '.,;:!?')
        }
    
    def extract_metadata(self, content: str) -> Dict[str, Any]:
        """Extract metadata from paragraph content."""
        return self._analyze_paragraph(content)
    
    def validate(self, unit: AtomicUnit) -> bool:
        """Validate paragraph unit."""
        if unit.unit_type != AtomicUnitType.PARAGRAPH:
            return False
        # Paragraphs are valid if they don't look like other atomic units
        content = unit.content.strip()
        return not (content.startswith(('```', '    ', '\t', '>')) or
                   '|' in content and content.count('|') >= 2 or
                   re.match(r'^[\s]*[-*+]\s', content) or
                   re.match(r'^[\s]*\d+[.)]\s', content))


class AtomicUnitRegistry:
    """Registry for managing atomic unit handlers."""
    
    def __init__(self) -> None:
        self._handlers: Dict[AtomicUnitType, Any] = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default handlers for all unit types."""
        self._handlers[AtomicUnitType.CODE_BLOCK] = CodeBlockHandler()
        self._handlers[AtomicUnitType.TABLE] = TableHandler()
        self._handlers[AtomicUnitType.LIST] = ListHandler()
        self._handlers[AtomicUnitType.BLOCKQUOTE] = BlockquoteHandler()
        self._handlers[AtomicUnitType.PARAGRAPH] = ParagraphHandler()
    
    def register_handler(self, unit_type: AtomicUnitType, handler: Any) -> None:
        """Register a handler for a specific unit type."""
        self._handlers[unit_type] = handler
    
    def get_handler(self, unit_type: AtomicUnitType) -> Any:
        """Get handler for a specific unit type."""
        return self._handlers.get(unit_type)
    
    def has_handler(self, unit_type: AtomicUnitType) -> bool:
        """Check if a handler exists for a unit type."""
        return unit_type in self._handlers
    
    def get_supported_types(self) -> List[AtomicUnitType]:
        """Get list of all supported unit types."""
        return list(self._handlers.keys())
    
    def unregister_handler(self, unit_type: AtomicUnitType) -> None:
        """Unregister a handler for a specific unit type."""
        if unit_type in self._handlers:
            del self._handlers[unit_type]


class AtomicUnitHandler:
    """Main handler for detecting and managing atomic content units with enhanced error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize handler with configuration and registry."""
        self.config = config or {}
        self.registry = AtomicUnitRegistry()
        self.logger = logging.getLogger(__name__)
    
    def detect_atomic_units(self, text: str) -> List[AtomicUnit]:
        """Detect all atomic units in text with comprehensive error handling."""
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text)}")
        
        if not text.strip():
            self.logger.debug("Empty or whitespace-only text provided")
            return []
        
        units = []
        
        try:
            # Detect each type of atomic unit
            for unit_type in AtomicUnitType:
                handler = self.registry.get_handler(unit_type)
                if handler:
                    try:
                        type_units = handler.detect(text)
                        units.extend(type_units)
                        self.logger.debug(f"Detected {len(type_units)} {unit_type.value} units")
                    except Exception as e:
                        self.logger.warning(f"Error detecting {unit_type.value} units: {e}")
                        # Continue with other unit types
                        continue
            
            # Sort units by start position for consistent ordering
            units.sort(key=lambda u: u.start_position)
            
            self.logger.debug(f"Total atomic units detected: {len(units)}")
            return units
            
        except Exception as e:
            self.logger.error(f"Critical error in atomic unit detection: {e}")
            raise RuntimeError(f"Failed to detect atomic units: {e}") from e
    
    def get_preservation_boundaries(self, text: str, atomic_units: List[AtomicUnit]) -> List[Dict[str, int]]:
        """Get boundaries that should be preserved during chunking.
        
        Returns boundaries as dictionaries with 'start' and 'end' keys.
        """
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text)}")
        
        if not isinstance(atomic_units, list):
            raise ValueError(f"atomic_units must be a list, got {type(atomic_units)}")
        
        try:
            boundaries = []
            
            for unit in atomic_units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in atomic_units list: {type(unit)}")
                    continue
                
                # Add boundary as dictionary with start and end keys
                boundaries.append({
                    'start': unit.start_position,
                    'end': unit.end_position
                })
            
            # Sort boundaries by start position
            boundaries.sort(key=lambda x: x['start'])
            
            self.logger.debug(f"Generated {len(boundaries)} preservation boundaries")
            return boundaries
            
        except Exception as e:
            self.logger.error(f"Error generating preservation boundaries: {e}")
            return []
    
    def get_atomic_units_in_range(self, atomic_units: List[AtomicUnit], start: int, end: int) -> List[AtomicUnit]:
        """Get atomic units that overlap with a specific range."""
        if not isinstance(atomic_units, list):
            raise ValueError(f"atomic_units must be a list, got {type(atomic_units)}")
        
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("start and end must be integers")
        
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        
        overlapping_units = []
        
        try:
            for unit in atomic_units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in atomic_units list: {type(unit)}")
                    continue
                
                if unit.overlaps_with_range(start, end):
                    overlapping_units.append(unit)
            
            return overlapping_units
            
        except Exception as e:
            self.logger.error(f"Error getting units in range {start}-{end}: {e}")
            return []
    
    def merge_overlapping_units(self, units: List[AtomicUnit]) -> List[AtomicUnit]:
        """Merge overlapping atomic units of the same type."""
        if not isinstance(units, list):
            raise ValueError(f"units must be a list, got {type(units)}")
        
        if not units:
            return []
        
        try:
            # Group by unit type
            type_groups = {}
            for unit in units:
                if not isinstance(unit, AtomicUnit):
                    self.logger.warning(f"Invalid unit in units list: {type(unit)}")
                    continue
                
                unit_type = unit.unit_type
                if unit_type not in type_groups:
                    type_groups[unit_type] = []
                type_groups[unit_type].append(unit)
            
            merged_units = []
            
            # Merge within each type group
            for unit_type, type_units in type_groups.items():
                # Sort by start position
                type_units.sort(key=lambda u: u.start_position)
                
                current_merged = type_units[0]
                
                for unit in type_units[1:]:
                    # Check if units overlap
                    if current_merged.end_position >= unit.start_position:
                        # Merge units - reconstruct content from unit boundaries
                        # Use the existing content plus additional content if needed
                        if unit.end_position > current_merged.end_position:
                            # Extend the merged content
                            merged_content = current_merged.content + unit.content[current_merged.end_position - unit.start_position:]
                        else:
                            # Unit is contained within current_merged
                            merged_content = current_merged.content
                        
                        current_merged = AtomicUnit(
                            unit_type=unit_type,
                            content=merged_content,
                            start_position=current_merged.start_position,
                            end_position=max(current_merged.end_position, unit.end_position),
                            metadata={**current_merged.metadata, **unit.metadata}
                        )
                    else:
                        # No overlap, add current and start new
                        merged_units.append(current_merged)
                        current_merged = unit
                
                merged_units.append(current_merged)
            
            # Sort final result by position
            merged_units.sort(key=lambda u: u.start_position)
            return merged_units
            
        except Exception as e:
            self.logger.error(f"Error merging overlapping units: {e}")
            return units  # Return original units if merge fails