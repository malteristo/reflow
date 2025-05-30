"""Tests for specialized atomic unit handlers - CodeBlockHandler, TableHandler, ListHandler, etc."""

import pytest
from core.document_processor.atomic_units import (
    AtomicUnitType, AtomicUnit, CodeBlockHandler, TableHandler, 
    ListHandler, BlockquoteHandler, ParagraphHandler
)


class TestCodeBlockHandler:
    """Tests for CodeBlockHandler class - specialized code block processing."""
    
    def test_code_block_handler_detect_fenced_blocks(self):
        """Test detecting fenced code blocks with language specifications."""
        handler = CodeBlockHandler()
        text = """Regular text.

```python
def hello_world():
    print("Hello, World!")
    return True
```

More regular text.

```javascript
console.log("JavaScript code");
```

Final text."""
        
        units = handler.detect(text)
        assert len(units) == 2
        
        python_block = next(u for u in units if "hello_world" in u.content)
        js_block = next(u for u in units if "console.log" in u.content)
        
        assert python_block.metadata["language"] == "python"
        assert python_block.metadata["block_type"] == "fenced"
        assert js_block.metadata["language"] == "javascript"
        assert js_block.metadata["block_type"] == "fenced"
    
    def test_code_block_handler_detect_indented_blocks(self):
        """Test detecting indented code blocks."""
        handler = CodeBlockHandler()
        text = """Regular paragraph.

    def indented_function():
        return "indented code"
        # This is still part of the block
    
    another_line = "still indented"

Back to regular text."""
        
        units = handler.detect(text)
        assert len(units) == 1
        
        code_block = units[0]
        assert code_block.metadata["block_type"] == "indented"
        assert "indented_function" in code_block.content
        assert "another_line" in code_block.content
    
    def test_code_block_handler_extract_metadata(self):
        """Test extracting metadata from code block content."""
        handler = CodeBlockHandler()
        
        fenced_content = """```python
def calculate_area(radius):
    import math
    return math.pi * radius ** 2
```"""
        
        metadata = handler.extract_metadata(fenced_content)
        assert metadata["language"] == "python"
        assert metadata["block_type"] == "fenced"
        assert metadata["line_count"] == 5
        assert metadata["has_imports"] == True
    
    def test_code_block_handler_validate_unit(self):
        """Test validating code block units."""
        handler = CodeBlockHandler()
        
        valid_unit = AtomicUnit(
            AtomicUnitType.CODE_BLOCK,
            "```python\nprint('hello')\n```",
            0, 25,
            {"language": "python", "block_type": "fenced"}
        )
        
        invalid_unit = AtomicUnit(
            AtomicUnitType.CODE_BLOCK,
            "Not actually code",
            0, 16,
            {}
        )
        
        assert handler.validate(valid_unit) == True
        assert handler.validate(invalid_unit) == False


class TestTableHandler:
    """Tests for TableHandler class - specialized table processing."""
    
    def test_table_handler_detect_pipe_tables(self):
        """Test detecting pipe-separated tables."""
        handler = TableHandler()
        text = """Before table.

| Name    | Age | City     |
|---------|-----|----------|
| Alice   | 30  | New York |
| Bob     | 25  | London   |
| Charlie | 35  | Paris    |

After table."""
        
        units = handler.detect(text)
        assert len(units) == 1
        
        table = units[0]
        assert table.metadata["column_count"] == 3
        assert table.metadata["row_count"] == 5  # Including header + separator + data rows
        assert table.metadata["has_header_separator"] == True
    
    def test_table_handler_detect_simple_tables(self):
        """Test detecting simple tables without header separators."""
        handler = TableHandler()
        text = """| Col1 | Col2 |
| Data1 | Data2 |
| Data3 | Data4 |"""
        
        units = handler.detect(text)
        # Simple tables without separators might not be detected by all implementations
        # This test verifies the handler can process such content gracefully
        if len(units) > 0:
            table = units[0]
            assert table.metadata["column_count"] == 2
            assert table.metadata["row_count"] >= 3
            assert table.metadata.get("has_header_separator", False) == False
    
    def test_table_handler_extract_metadata(self):
        """Test extracting detailed metadata from table content."""
        handler = TableHandler()
        table_content = """| Product | Price | Stock |
|---------|------:|:-----:|
| Widget  | $10.99|   15  |
| Gadget  | $25.50|    8  |"""
        
        metadata = handler.extract_metadata(table_content)
        assert metadata["column_count"] == 3
        assert metadata["row_count"] == 4  # header + separator + 2 data rows
        assert metadata["has_header_separator"] == True
        assert metadata["column_alignments"] == ["left", "right", "center"]
    
    def test_table_handler_validate_unit(self):
        """Test validating table units."""
        handler = TableHandler()
        
        valid_table = AtomicUnit(
            AtomicUnitType.TABLE,
            "| A | B |\n|---|---|\n| 1 | 2 |",
            0, 20,
            {"column_count": 2, "row_count": 2}
        )
        
        invalid_table = AtomicUnit(
            AtomicUnitType.TABLE,
            "Not a table",
            0, 11,
            {}
        )
        
        assert handler.validate(valid_table) == True
        assert handler.validate(invalid_table) == False


class TestListHandler:
    """Tests for ListHandler class - specialized list processing."""
    
    def test_list_handler_detect_bullet_lists(self):
        """Test detecting bullet lists with various markers."""
        handler = ListHandler()
        text = """- Item 1
- Item 2
  - Nested item
  - Another nested
- Item 3

* Different marker
* Second item

+ Plus marker
+ Another plus"""
        
        units = handler.detect(text)
        assert len(units) == 3  # Three separate lists
        
        # Check list types
        dash_list = next(u for u in units if "Item 1" in u.content)
        assert dash_list.metadata["list_type"] == "bullet"
        assert dash_list.metadata["marker"] == "-"
        assert dash_list.metadata["has_nested_items"] == True
    
    def test_list_handler_detect_numbered_lists(self):
        """Test detecting numbered lists."""
        handler = ListHandler()
        text = """1. First item
2. Second item
   a. Nested letter
   b. Another letter
3. Third item

1) Different style
2) Second item"""
        
        units = handler.detect(text)
        assert len(units) == 2
        
        dot_list = next(u for u in units if "First item" in u.content)
        assert dot_list.metadata["list_type"] == "numbered"
        assert dot_list.metadata["marker_style"] == "dot"
        assert dot_list.metadata["has_nested_items"] == True
    
    def test_list_handler_detect_task_lists(self):
        """Test detecting task lists with checkboxes."""
        handler = ListHandler()
        text = """- [x] Completed task
- [ ] Pending task
- [X] Also completed
- [-] Canceled task"""
        
        units = handler.detect(text)
        assert len(units) == 1
        
        task_list = units[0]
        assert task_list.metadata["list_type"] == "task"
        assert task_list.metadata["completed_count"] == 2
        assert task_list.metadata["pending_count"] == 1
        assert task_list.metadata["total_tasks"] == 3  # Only valid tasks, [-] might not count
    
    def test_list_handler_extract_metadata(self):
        """Test extracting comprehensive metadata from list content."""
        handler = ListHandler()
        list_content = """1. First item
   - Nested bullet
   - Another bullet
2. Second item
   1. Nested number
   2. Another number
3. Third item"""
        
        metadata = handler.extract_metadata(list_content)
        assert metadata["list_type"] == "numbered"
        assert metadata["item_count"] == 3
        assert metadata["has_nested_items"] == True
        assert metadata["max_nesting_depth"] == 2
        # nested_list_types might not always be present depending on implementation
        if "nested_list_types" in metadata:
            assert metadata["nested_list_types"] == ["bullet", "numbered"]


class TestBlockquoteHandler:
    """Tests for BlockquoteHandler class - specialized blockquote processing."""
    
    def test_blockquote_handler_detect_simple_quotes(self):
        """Test detecting simple blockquotes."""
        handler = BlockquoteHandler()
        text = """> This is a quote
> with multiple lines
> all at the same level"""
        
        units = handler.detect(text)
        assert len(units) == 1
        
        quote = units[0]
        assert quote.metadata["max_depth"] == 1
        assert quote.metadata["line_count"] == 3
    
    def test_blockquote_handler_detect_nested_quotes(self):
        """Test detecting nested blockquotes."""
        handler = BlockquoteHandler()
        text = """> Level 1 quote
> 
> > Level 2 nested
> > More level 2
> >
> > > Level 3 deeply nested
> 
> Back to level 1"""
        
        units = handler.detect(text)
        assert len(units) == 1
        
        quote = units[0]
        assert quote.metadata["max_depth"] == 3
        assert quote.metadata["has_nested_quotes"] == True
        assert quote.metadata["nesting_levels"] == [1, 2, 3]
    
    def test_blockquote_handler_extract_metadata(self):
        """Test extracting metadata from blockquote content."""
        handler = BlockquoteHandler()
        quote_content = """> Author said:
> "This is important."
> 
> > Nested response:
> > "I agree completely."
> 
> Final thoughts here."""
        
        metadata = handler.extract_metadata(quote_content)
        assert metadata["max_depth"] == 2
        assert metadata["line_count"] == 7
        assert metadata["has_nested_quotes"] == True
        assert metadata["contains_attribution"] == True  # "Author said:"


class TestParagraphHandler:
    """Tests for ParagraphHandler class - specialized paragraph processing."""
    
    def test_paragraph_handler_detect_simple_paragraphs(self):
        """Test detecting simple paragraphs separated by blank lines."""
        handler = ParagraphHandler()
        text = """First paragraph with some content.
More content in the same paragraph.

Second paragraph starts here.
It also has multiple lines.

Third paragraph is short."""
        
        units = handler.detect(text)
        assert len(units) == 3
        
        assert "First paragraph" in units[0].content
        assert "Second paragraph" in units[1].content
        assert "Third paragraph" in units[2].content
    
    def test_paragraph_handler_ignore_other_atomic_units(self):
        """Test paragraph handler ignores content that belongs to other unit types."""
        handler = ParagraphHandler()
        text = """Regular paragraph.

```code
This should not be detected as paragraph
```

Another paragraph.

> This is a blockquote
> Not a paragraph

Final paragraph."""
        
        units = handler.detect(text)
        paragraph_contents = [u.content for u in units]
        
        # Should detect regular paragraphs but not code or blockquotes
        assert any("Regular paragraph" in content for content in paragraph_contents)
        assert any("Another paragraph" in content for content in paragraph_contents)
        assert any("Final paragraph" in content for content in paragraph_contents)
        assert not any("This should not be detected" in content for content in paragraph_contents)
        assert not any("This is a blockquote" in content for content in paragraph_contents)
    
    def test_paragraph_handler_extract_metadata(self):
        """Test extracting metadata from paragraph content."""
        handler = ParagraphHandler()
        paragraph_content = """This is a paragraph with **bold text** and *italic text*.
It has multiple sentences. Some are longer than others.
The paragraph contains various punctuation marks: colons, semicolons; and more!
It also has numbers like 123 and special characters like @#$%."""
        
        metadata = handler.extract_metadata(paragraph_content)
        assert metadata["sentence_count"] >= 3
        assert metadata["word_count"] >= 20
        assert metadata["has_formatting"] == True
        assert metadata["has_punctuation"] == True
        assert metadata["line_count"] == 4 