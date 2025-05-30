"""
Test module for Document Structure components

This module contains comprehensive tests for document structure functionality,
including DocumentSection, DocumentTree, HeaderBasedSplitter, and SectionExtractor classes.

Tests are extracted from the original monolithic test file and aligned with
the modular document processor architecture.
"""

import pytest
from typing import List, Dict, Any

from research_agent_backend.core.document_processor import (
    DocumentSection,
    DocumentTree,
    HeaderBasedSplitter,
    SectionExtractor,
    MarkdownParser
)


class TestDocumentSection:
    """Tests for DocumentSection class - represents a document section with header info."""
    
    def test_document_section_creation_with_basic_info(self):
        """Test creating a DocumentSection with header level, title, and content."""
        section = DocumentSection(
            level=1,
            title="Main Title",
            content="This is the main content.",
            line_number=1
        )
        assert section.level == 1
        assert section.title == "Main Title"
        assert section.content == "This is the main content."
        assert section.line_number == 1
        assert section.children == []
    
    def test_document_section_add_child_section(self):
        """Test adding child sections to a parent section."""
        parent = DocumentSection(level=1, title="Parent", content="Parent content", line_number=1)
        child = DocumentSection(level=2, title="Child", content="Child content", line_number=5)
        
        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent
    
    def test_document_section_get_depth(self):
        """Test calculating section depth in hierarchy."""
        root = DocumentSection(level=1, title="Root", content="", line_number=1)
        child = DocumentSection(level=2, title="Child", content="", line_number=3)
        grandchild = DocumentSection(level=3, title="Grandchild", content="", line_number=5)
        
        root.add_child(child)
        child.add_child(grandchild)
        
        assert root.get_depth() == 0
        assert child.get_depth() == 1
        assert grandchild.get_depth() == 2
    
    def test_document_section_get_all_content(self):
        """Test getting all content including children."""
        parent = DocumentSection(level=1, title="Parent", content="Parent content", line_number=1)
        child1 = DocumentSection(level=2, title="Child1", content="Child1 content", line_number=3)
        child2 = DocumentSection(level=2, title="Child2", content="Child2 content", line_number=5)
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        all_content = parent.get_all_content()
        assert "Parent content" in all_content
        assert "Child1 content" in all_content
        assert "Child2 content" in all_content
    
    def test_document_section_find_child_by_title(self):
        """Test finding child sections by title."""
        parent = DocumentSection(level=1, title="Parent", content="", line_number=1)
        child1 = DocumentSection(level=2, title="Introduction", content="", line_number=3)
        child2 = DocumentSection(level=2, title="Methods", content="", line_number=5)
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        found = parent.find_child_by_title("Methods")
        assert found == child2
        
        not_found = parent.find_child_by_title("Conclusion")
        assert not_found is None


class TestDocumentTree:
    """Tests for DocumentTree class - represents complete document hierarchy."""
    
    def test_document_tree_creation_with_root_section(self):
        """Test creating a DocumentTree with a root section."""
        root_section = DocumentSection(level=1, title="Document", content="", line_number=1)
        tree = DocumentTree(root_section)
        
        assert tree.root == root_section
        assert tree.get_section_count() == 1
    
    def test_document_tree_add_section_creates_hierarchy(self):
        """Test adding sections automatically creates proper hierarchy."""
        tree = DocumentTree()
        
        # Add sections in order
        tree.add_section(DocumentSection(level=1, title="Chapter 1", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Section 1.1", content="", line_number=3))
        tree.add_section(DocumentSection(level=2, title="Section 1.2", content="", line_number=5))
        tree.add_section(DocumentSection(level=1, title="Chapter 2", content="", line_number=7))
        
        assert tree.get_section_count() == 4
        
        # Verify hierarchy
        chapter1 = tree.find_section_by_title("Chapter 1")
        assert len(chapter1.children) == 2
        assert chapter1.children[0].title == "Section 1.1"
        assert chapter1.children[1].title == "Section 1.2"
    
    def test_document_tree_find_section_by_title(self):
        """Test finding sections by title throughout the tree."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Introduction", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Background", content="", line_number=3))
        
        found = tree.find_section_by_title("Background")
        assert found is not None
        assert found.title == "Background"
        assert found.level == 2
    
    def test_document_tree_get_sections_by_level(self):
        """Test getting all sections at a specific level."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Chapter 1", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Section 1.1", content="", line_number=3))
        tree.add_section(DocumentSection(level=2, title="Section 1.2", content="", line_number=5))
        tree.add_section(DocumentSection(level=1, title="Chapter 2", content="", line_number=7))
        
        level_1_sections = tree.get_sections_by_level(1)
        assert len(level_1_sections) == 2
        assert level_1_sections[0].title == "Chapter 1"
        assert level_1_sections[1].title == "Chapter 2"
        
        level_2_sections = tree.get_sections_by_level(2)
        assert len(level_2_sections) == 2
    
    def test_document_tree_to_dict_serialization(self):
        """Test converting tree to dictionary for serialization."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Main", content="Main content", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Sub", content="Sub content", line_number=3))
        
        tree_dict = tree.to_dict()
        assert isinstance(tree_dict, dict)
        assert "root" in tree_dict
        assert tree_dict["root"]["title"] == "Main"
        assert len(tree_dict["root"]["children"]) == 1


class TestHeaderBasedSplitter:
    """Tests for HeaderBasedSplitter class - main functionality for splitting documents."""
    
    def test_header_splitter_creation_with_parser(self):
        """Test creating HeaderBasedSplitter with MarkdownParser."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        assert splitter.parser == parser
    
    def test_split_by_headers_simple_document(self):
        """Test splitting a simple document with headers."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        document = """# Introduction

This is the introduction.

## Background

Some background information.

## Methods

The methods section."""
        
        sections = splitter.split_by_headers(document)
        assert len(sections) == 3
        assert sections[0].title == "Introduction"
        assert sections[0].level == 1
        assert sections[1].title == "Background"
        assert sections[1].level == 2
        assert sections[2].title == "Methods"
        assert sections[2].level == 2
    
    def test_split_by_headers_nested_hierarchy(self):
        """Test splitting document with nested header hierarchy."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        document = """# Chapter 1

Chapter introduction.

## Section 1.1

First section.

### Subsection 1.1.1

First subsection.

### Subsection 1.1.2

Second subsection.

## Section 1.2

Second section.

# Chapter 2

Second chapter."""
        
        sections = splitter.split_by_headers(document)
        
        # Should find all headers
        titles = [s.title for s in sections]
        assert "Chapter 1" in titles
        assert "Section 1.1" in titles
        assert "Subsection 1.1.1" in titles
        assert "Subsection 1.1.2" in titles
        assert "Section 1.2" in titles
        assert "Chapter 2" in titles
    
    def test_build_tree_from_sections(self):
        """Test building DocumentTree from flat list of sections."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        sections = [
            DocumentSection(level=1, title="Main", content="Main content", line_number=1),
            DocumentSection(level=2, title="Sub1", content="Sub1 content", line_number=3),
            DocumentSection(level=2, title="Sub2", content="Sub2 content", line_number=5),
            DocumentSection(level=3, title="SubSub", content="SubSub content", line_number=7),
        ]
        
        tree = splitter.build_tree(sections)
        assert tree.root.title == "Main"
        assert len(tree.root.children) == 2
        assert tree.root.children[1].children[0].title == "SubSub"
    
    def test_split_and_build_tree_end_to_end(self):
        """Test complete end-to-end splitting and tree building."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        document = """# Main Document

Introduction text.

## First Section

First section content.

### Subsection

Subsection content.

## Second Section

Second section content."""
        
        tree = splitter.split_and_build_tree(document)
        
        assert tree.root.title == "Main Document"
        assert len(tree.root.children) == 2
        assert tree.root.children[0].title == "First Section"
        assert len(tree.root.children[0].children) == 1
        assert tree.root.children[0].children[0].title == "Subsection"
    
    def test_split_document_without_headers(self):
        """Test handling documents without headers."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        document = """This is a document without headers.

It has multiple paragraphs.

But no markdown headers."""
        
        result = splitter.split_by_headers(document)
        assert len(result) == 1  # Should create single section for content
        assert result[0].title == ""  # No title for headerless content
        assert result[0].level == 0  # Level 0 for non-header content
    
    def test_extract_content_between_headers(self):
        """Test extracting content between consecutive headers."""
        parser = MarkdownParser()
        splitter = HeaderBasedSplitter(parser)
        
        document = """# Header 1

Content for header 1.
More content here.

## Header 2

Content for header 2.

# Header 3

Content for header 3."""
        
        sections = splitter.split_by_headers(document)
        
        assert "Content for header 1" in sections[0].content
        assert "More content here" in sections[0].content
        assert "Content for header 2" in sections[1].content
        assert "Content for header 3" in sections[2].content


class TestSectionExtractor:
    """Tests for SectionExtractor class - extracting specific sections from trees."""
    
    def test_section_extractor_creation(self):
        """Test creating SectionExtractor."""
        extractor = SectionExtractor()
        assert extractor is not None
    
    def test_extract_section_by_title(self):
        """Test extracting a specific section by title."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Introduction", content="Intro content", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Methods", content="Methods content", line_number=3))
        
        extractor = SectionExtractor()
        section = extractor.extract_by_title(tree, "Methods")
        
        assert section is not None
        assert section.title == "Methods"
        assert section.content == "Methods content"
    
    def test_extract_sections_by_level_range(self):
        """Test extracting sections within a level range."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Chapter", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Section", content="", line_number=3))
        tree.add_section(DocumentSection(level=3, title="Subsection", content="", line_number=5))
        tree.add_section(DocumentSection(level=4, title="Subsubsection", content="", line_number=7))
        
        extractor = SectionExtractor()
        sections = extractor.extract_by_level_range(tree, min_level=2, max_level=3)
        
        assert len(sections) == 2
        titles = [s.title for s in sections]
        assert "Section" in titles
        assert "Subsection" in titles
        assert "Chapter" not in titles
        assert "Subsubsection" not in titles
    
    def test_extract_section_with_children(self):
        """Test extracting section and all its children."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Main", content="Main content", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Sub1", content="Sub1 content", line_number=3))
        tree.add_section(DocumentSection(level=2, title="Sub2", content="Sub2 content", line_number=5))
        tree.add_section(DocumentSection(level=3, title="SubSub", content="SubSub content", line_number=7))
        
        extractor = SectionExtractor()
        section_with_children = extractor.extract_with_children(tree, "Main")
        
        assert section_with_children.title == "Main"
        assert len(section_with_children.children) == 2
        assert section_with_children.children[1].children[0].title == "SubSub"
    
    def test_extract_table_of_contents(self):
        """Test generating table of contents from document tree."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=1, title="Chapter 1", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Section 1.1", content="", line_number=3))
        tree.add_section(DocumentSection(level=2, title="Section 1.2", content="", line_number=5))
        tree.add_section(DocumentSection(level=1, title="Chapter 2", content="", line_number=7))
        
        extractor = SectionExtractor()
        toc = extractor.generate_table_of_contents(tree)
        
        assert isinstance(toc, list)
        assert len(toc) >= 4  # Should include all sections
        
        # Check structure contains titles and levels
        for item in toc:
            assert "title" in item
            assert "level" in item
            assert "line_number" in item
    
    def test_extract_sections_matching_pattern(self):
        """Test extracting sections with titles matching a pattern."""
        tree = DocumentTree()
        tree.add_section(DocumentSection(level=2, title="Introduction", content="", line_number=1))
        tree.add_section(DocumentSection(level=2, title="Methods", content="", line_number=3))
        tree.add_section(DocumentSection(level=2, title="Results", content="", line_number=5))
        tree.add_section(DocumentSection(level=2, title="Discussion", content="", line_number=7))
        
        extractor = SectionExtractor()
        
        # Extract sections ending with 's'
        sections = extractor.extract_by_title_pattern(tree, r".*s$")
        titles = [s.title for s in sections]
        assert "Methods" in titles
        assert "Results" in titles
        assert "Introduction" not in titles
        assert "Discussion" not in titles 