"""Tests for frontmatter parsing functionality (YAML and TOML)."""

import pytest
import datetime
from core.document_processor import FrontmatterParser, FrontmatterParseError


class TestFrontmatterParser:
    """Tests for frontmatter parsing functionality (YAML and TOML)."""
    
    def test_frontmatter_parser_creation(self):
        """Test creating a FrontmatterParser."""
        parser = FrontmatterParser()
        assert isinstance(parser, FrontmatterParser)
    
    def test_parse_yaml_frontmatter_basic(self):
        """Test parsing basic YAML frontmatter."""
        parser = FrontmatterParser()
        
        content = """---
title: "My Document"
author: "John Doe"
tags: ["python", "testing"]
date: 2024-01-15
---

# Document Content

This is the actual content."""
        
        result = parser.parse(content)
        assert result.has_frontmatter == True
        assert result.metadata["title"] == "My Document"
        assert result.metadata["author"] == "John Doe"
        assert result.metadata["tags"] == ["python", "testing"]
        # YAML automatically parses dates - check if it's parsed correctly
        expected_date = datetime.date(2024, 1, 15)
        assert result.metadata["date"] == expected_date
        assert result.content_without_frontmatter.startswith("# Document Content")
    
    def test_parse_yaml_frontmatter_complex(self):
        """Test parsing complex YAML frontmatter with nested structures."""
        parser = FrontmatterParser()
        
        content = """---
title: "Complex Document"
metadata:
  version: 1.2
  status: draft
  review:
    required: true
    reviewers: ["alice", "bob"]
categories:
  - technical
  - documentation
settings:
  toc: true
  numbered_headings: false
---

Content goes here."""
        
        result = parser.parse(content)
        assert result.metadata["title"] == "Complex Document"
        assert result.metadata["metadata"]["version"] == 1.2
        assert result.metadata["metadata"]["review"]["required"] == True
        assert result.metadata["categories"] == ["technical", "documentation"]
        assert result.metadata["settings"]["toc"] == True
    
    def test_parse_toml_frontmatter_basic(self):
        """Test parsing basic TOML frontmatter."""
        parser = FrontmatterParser()
        
        content = """+++
title = "TOML Document"
author = "Jane Smith"
tags = ["rust", "config"]
publish_date = 2024-02-01T10:30:00Z
+++

# TOML Content

This document uses TOML frontmatter."""
        
        result = parser.parse(content)
        assert result.has_frontmatter == True
        assert result.metadata["title"] == "TOML Document"
        assert result.metadata["author"] == "Jane Smith"
        assert result.metadata["tags"] == ["rust", "config"]
        assert "publish_date" in result.metadata
    
    def test_parse_toml_frontmatter_complex(self):
        """Test parsing complex TOML frontmatter with sections."""
        parser = FrontmatterParser()
        
        content = """+++
title = "Complex TOML"

[metadata]
version = "2.1"
status = "published"

[settings]
toc = true
math = false

[author]
name = "Research Team"
email = "team@example.com"
+++

Complex TOML content."""
        
        result = parser.parse(content)
        assert result.metadata["title"] == "Complex TOML"
        assert result.metadata["metadata"]["version"] == "2.1"
        assert result.metadata["settings"]["toc"] == True
        assert result.metadata["author"]["name"] == "Research Team"
    
    def test_parse_no_frontmatter(self):
        """Test parsing document without frontmatter."""
        parser = FrontmatterParser()
        
        content = """# Regular Document

This document has no frontmatter.
Just regular markdown content."""
        
        result = parser.parse(content)
        assert result.has_frontmatter == False
        assert result.metadata == {}
        assert result.content_without_frontmatter == content
    
    def test_parse_invalid_yaml_frontmatter(self):
        """Test handling invalid YAML frontmatter."""
        parser = FrontmatterParser()
        
        content = """---
title: "Invalid YAML
author: missing quote
- invalid list item
---

Content."""
        
        with pytest.raises(FrontmatterParseError):
            parser.parse(content)
    
    def test_parse_invalid_toml_frontmatter(self):
        """Test handling invalid TOML frontmatter."""
        parser = FrontmatterParser()
        
        content = """+++
title = "Invalid TOML
author = missing quote
invalid syntax here
+++

Content."""
        
        with pytest.raises(FrontmatterParseError):
            parser.parse(content)
    
    def test_detect_frontmatter_type_yaml(self):
        """Test detecting YAML frontmatter type."""
        parser = FrontmatterParser()
        
        content = """---
title: "YAML Doc"
---
Content"""
        
        result = parser.parse(content)
        assert result.frontmatter_type == "yaml"
    
    def test_detect_frontmatter_type_toml(self):
        """Test detecting TOML frontmatter type."""
        parser = FrontmatterParser()
        
        content = """+++
title = "TOML Doc"
+++
Content"""
        
        result = parser.parse(content)
        assert result.frontmatter_type == "toml" 