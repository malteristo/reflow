"""
Document Tree Module - Hierarchical Document Structure

This module provides the DocumentTree class for managing hierarchical
organization of document sections.

Key Components:
- DocumentTree: Tree structure for document hierarchy

Usage:
    >>> from .section import DocumentSection
    >>> tree = DocumentTree()
    >>> tree.add_section(DocumentSection(1, "Chapter 1", "Content", 1))
"""

from typing import List, Dict, Any, Optional
import logging

from .section import DocumentSection

logger = logging.getLogger(__name__)


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
            self.root = section.children[0] if section.children else None
            if self.root:
                self.root.parent = None
        
        # Remove section and all its children from indexes
        sections_to_remove = [section] + self._get_all_descendants(section)
        for sect in sections_to_remove:
            self._remove_section_from_indexes(sect)
        
        logger.debug(f"Removed section '{section.title}' and {len(sections_to_remove) - 1} descendants")
        return True
    
    def get_all_sections(self) -> List[DocumentSection]:
        """
        Get all sections in the tree in document order.
        
        Returns:
            List of all sections in the tree
        """
        return self._sections.copy()
    
    def _get_all_descendants(self, section: DocumentSection) -> List[DocumentSection]:
        """
        Get all descendants of a section recursively.
        
        Args:
            section: Section to get descendants for
            
        Returns:
            List of all descendant sections
        """
        descendants = []
        for child in section.children:
            descendants.append(child)
            descendants.extend(self._get_all_descendants(child))
        return descendants
    
    def find_section_by_title(self, title: str, case_sensitive: bool = True) -> Optional[DocumentSection]:
        """
        Find a section by title with case sensitivity option.
        
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
            title_lower = title.lower()
            for section in self._sections:
                if section.title.lower() == title_lower:
                    return section
        return None
    
    def get_sections_by_level(self, level: int) -> List[DocumentSection]:
        """
        Get all sections at a specific header level.
        
        Args:
            level: Header level to search for
            
        Returns:
            List of sections at the specified level
        """
        return [section for section in self._sections if section.level == level]
    
    def get_sections_by_level_range(self, min_level: int, max_level: int) -> List[DocumentSection]:
        """
        Get sections within a level range.
        
        Args:
            min_level: Minimum header level (inclusive)
            max_level: Maximum header level (inclusive)
            
        Returns:
            List of sections within the level range
        """
        return [
            section for section in self._sections 
            if min_level <= section.level <= max_level
        ]
    
    def get_section_count(self) -> int:
        """
        Get the total number of sections in the tree.
        
        Returns:
            Number of sections
        """
        return len(self._sections)
    
    def get_max_depth(self) -> int:
        """
        Get the maximum depth of the tree.
        
        Returns:
            Maximum depth (0 for empty tree or single root)
        """
        if not self._sections:
            return 0
        return max(section.get_depth() for section in self._sections)
    
    def get_levels_used(self) -> List[int]:
        """
        Get sorted list of header levels used in the document.
        
        Returns:
            Sorted list of unique header levels
        """
        levels = set(section.level for section in self._sections if section.level > 0)
        return sorted(levels)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the DocumentTree to a dictionary representation.
        
        Returns:
            Dictionary representation of the document tree
        """
        return {
            'root': self.root.to_dict() if self.root else None,
            'section_count': len(self._sections),
            'max_depth': self.get_max_depth(),
            'levels_used': self.get_levels_used()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DocumentTree(sections={len(self._sections)}, "
            f"root='{self.root.title if self.root else None}')"
        ) 