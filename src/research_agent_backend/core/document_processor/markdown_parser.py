"""
Markdown Parser Module - Core Parsing Functionality

This module implements the core Markdown parsing functionality that converts
Markdown syntax to HTML. It provides a flexible, rule-based system for
markdown transformation that serves as the foundation for the hybrid
document chunking strategy.

Key Components:
- Pattern: Regex matching with validation and error handling
- Rule: Transformation rules supporting both string and callable replacements
- MarkdownParser: Main parser with extensible rule system
- MarkdownParseError: Custom exception with debugging context

Implements FR-KB-002.1: Hybrid chunking strategy with Markdown-aware processing.

Usage:
    >>> parser = MarkdownParser()
    >>> html = parser.parse("# Header\n\nThis is **bold** text.")
    >>> print(html)
    <h1>Header</h1>
    
    This is <strong>bold</strong> text.
"""

import re
from typing import List, Dict, Any, Optional, Union, Callable, Match
import logging
from dataclasses import dataclass

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