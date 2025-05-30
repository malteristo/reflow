"""
Document Section Module - Individual Section Representation

This module provides the DocumentSection class for representing individual
sections of a markdown document with hierarchical relationships.

Key Components:
- DocumentSection: Represents a document section with header and content

Usage:
    >>> section = DocumentSection(
    ...     level=1, 
    ...     title="Introduction", 
    ...     content="This is the intro.",
    ...     line_number=1
    ... )
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


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
            >>> content = section.get_all_content(include_headers=True)
            >>> print("Main" in content and "Sub" in content)
            True
        """
        content_parts = []
        
        if include_headers and self.title:
            header_prefix = "#" * self.level if self.level > 0 else ""
            content_parts.append(f"{header_prefix} {self.title}")
        
        if self.content.strip():
            content_parts.append(self.content)
        
        # Recursively collect child content
        for child in self.children:
            child_content = child.get_all_content(include_headers)
            if child_content.strip():
                content_parts.append(child_content)
        
        return "\n\n".join(content_parts)
    
    def find_child_by_title(self, title: str) -> Optional['DocumentSection']:
        """
        Find immediate child section by title (case-sensitive).
        
        Args:
            title: Title to search for
            
        Returns:
            Child section with matching title, or None if not found
        """
        for child in self.children:
            if child.title == title:
                return child
        return None
    
    def find_descendant_by_title(self, title: str) -> Optional['DocumentSection']:
        """
        Find any descendant section by title (recursive search).
        
        Args:
            title: Title to search for
            
        Returns:
            First descendant section with matching title, or None if not found
        """
        # Check immediate children first
        direct_child = self.find_child_by_title(title)
        if direct_child:
            return direct_child
        
        # Recursively search in children
        for child in self.children:
            descendant = child.find_descendant_by_title(title)
            if descendant:
                return descendant
        
        return None
    
    def get_siblings(self) -> List['DocumentSection']:
        """
        Get all sibling sections (sections with same parent).
        
        Returns:
            List of sibling sections (excluding self)
        """
        if not self.parent:
            return []
        
        return [child for child in self.parent.children if child != self]
    
    def is_ancestor_of(self, other: 'DocumentSection') -> bool:
        """
        Check if this section is an ancestor of another section.
        
        Args:
            other: Section to check
            
        Returns:
            True if this section is an ancestor of other
        """
        current = other.parent
        while current:
            if current == self:
                return True
            current = current.parent
        return False
    
    def get_path_to_root(self) -> List['DocumentSection']:
        """
        Get path from this section to the root section.
        
        Returns:
            List of sections from root to this section (including self)
        """
        path = []
        current = self
        while current:
            path.insert(0, current)
            current = current.parent
        return path
    
    def to_dict(self) -> dict:
        """
        Convert section to dictionary representation.
        
        Returns:
            Dictionary with section data (excluding parent reference to avoid cycles)
        """
        return {
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "line_number": self.line_number,
            "children": [child.to_dict() for child in self.children],
            "depth": self.get_depth()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentSection(level={self.level}, title='{self.title}', "
            f"content_length={len(self.content)}, children={len(self.children)})"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.title:
            return f"{'#' * self.level} {self.title}" if self.level > 0 else self.title
        return f"Untitled section (level {self.level})" 