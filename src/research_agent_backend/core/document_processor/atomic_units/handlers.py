"""
Atomic Unit Handlers

Specialized handlers for different types of atomic content units.
Each handler provides detection, metadata extraction, and validation for specific content types.
"""

import re
import logging
from typing import List, Dict, Any
from .types import AtomicUnit, AtomicUnitType

logger = logging.getLogger(__name__)


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