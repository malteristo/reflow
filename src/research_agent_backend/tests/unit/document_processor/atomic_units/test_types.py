"""Tests for AtomicUnitType and AtomicUnit classes - atomic content unit types and instances."""

import pytest
from research_agent_backend.core.document_processor.atomic_units import AtomicUnitType, AtomicUnit


class TestAtomicUnitType:
    """Tests for AtomicUnitType enum - defines different atomic content unit types."""
    
    def test_atomic_unit_type_enum_values(self):
        """Test AtomicUnitType contains all expected content unit types."""
        assert hasattr(AtomicUnitType, 'PARAGRAPH')
        assert hasattr(AtomicUnitType, 'CODE_BLOCK')
        assert hasattr(AtomicUnitType, 'TABLE')
        assert hasattr(AtomicUnitType, 'LIST')
        assert hasattr(AtomicUnitType, 'BLOCKQUOTE')
        assert hasattr(AtomicUnitType, 'HORIZONTAL_RULE')
        assert hasattr(AtomicUnitType, 'YAML_FRONTMATTER')
        assert hasattr(AtomicUnitType, 'MATH_BLOCK')
    
    def test_atomic_unit_type_string_representation(self):
        """Test AtomicUnitType values have proper string representations."""
        assert str(AtomicUnitType.PARAGRAPH) == "paragraph"
        assert str(AtomicUnitType.CODE_BLOCK) == "code_block"
        assert str(AtomicUnitType.TABLE) == "table"
        assert str(AtomicUnitType.LIST) == "list"
        assert str(AtomicUnitType.BLOCKQUOTE) == "blockquote"


class TestAtomicUnit:
    """Tests for AtomicUnit class - represents a single atomic content unit."""
    
    def test_atomic_unit_creation_with_basic_info(self):
        """Test creating AtomicUnit with type, content, and boundaries."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.PARAGRAPH,
            content="This is a paragraph of text.",
            start_position=0,
            end_position=28,
            metadata={"line_count": 1}
        )
        assert unit.unit_type == AtomicUnitType.PARAGRAPH
        assert unit.content == "This is a paragraph of text."
        assert unit.start_position == 0
        assert unit.end_position == 28
        assert unit.metadata["line_count"] == 1
    
    def test_atomic_unit_get_length(self):
        """Test AtomicUnit can calculate its content length."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.CODE_BLOCK,
            content="def hello():\n    return 'world'",
            start_position=10,
            end_position=43
        )
        assert unit.get_length() == 31
    
    def test_atomic_unit_contains_position(self):
        """Test AtomicUnit can check if it contains a specific position."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.TABLE,
            content="| Col1 | Col2 |\n|------|------|\n| A    | B    |",
            start_position=50,
            end_position=100
        )
        assert unit.contains_position(75) == True
        assert unit.contains_position(25) == False
        assert unit.contains_position(150) == False
    
    def test_atomic_unit_get_boundaries(self):
        """Test AtomicUnit can return its boundary positions."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.LIST,
            content="- Item 1\n- Item 2\n- Item 3",
            start_position=100,
            end_position=125
        )
        boundaries = unit.get_boundaries()
        assert boundaries == (100, 125)
    
    def test_atomic_unit_to_dict_serialization(self):
        """Test AtomicUnit can be serialized to dictionary."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.BLOCKQUOTE,
            content="> This is a quote\n> with multiple lines",
            start_position=200,
            end_position=240,
            metadata={"author": "Someone", "depth": 1}
        )
        unit_dict = unit.to_dict()
        
        assert unit_dict["unit_type"] == "blockquote"
        assert unit_dict["content"] == "> This is a quote\n> with multiple lines"
        assert unit_dict["start_position"] == 200
        assert unit_dict["end_position"] == 240
        assert unit_dict["metadata"]["author"] == "Someone"
        assert unit_dict["metadata"]["depth"] == 1
    
    def test_atomic_unit_overlaps_with_range(self):
        """Test AtomicUnit can detect overlaps with position ranges."""
        unit = AtomicUnit(
            unit_type=AtomicUnitType.PARAGRAPH,
            content="Sample paragraph content",
            start_position=50,
            end_position=75
        )
        
        # Test various overlap scenarios
        assert unit.overlaps_with_range(40, 60) == True  # Partial overlap start
        assert unit.overlaps_with_range(65, 85) == True  # Partial overlap end
        assert unit.overlaps_with_range(45, 80) == True  # Complete overlap
        assert unit.overlaps_with_range(55, 65) == True  # Inside unit
        assert unit.overlaps_with_range(10, 30) == False # No overlap before
        assert unit.overlaps_with_range(80, 100) == False # No overlap after 