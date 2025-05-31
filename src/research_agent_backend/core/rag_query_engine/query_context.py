"""
RAG Query Context - Query parsing and context data structures.

This module contains the data structures and enums for representing
parsed query context, intent classification, and contextual filters.
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .constants import (
    COMPARATIVE_INDICATORS, TROUBLESHOOTING_INDICATORS, 
    CODE_SEARCH_INDICATORS, TUTORIAL_INDICATORS, TECHNOLOGY_MAPPINGS
)


class QueryIntent(Enum):
    """Enumeration of different query intent types."""
    INFORMATION_SEEKING = "information_seeking"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TUTORIAL_SEEKING = "tutorial_seeking"
    CODE_SEARCH = "code_search"
    TROUBLESHOOTING = "troubleshooting"
    
    @classmethod
    def classify(cls, query: str) -> 'QueryIntent':
        """
        Classify the intent of a query based on its content.
        
        Args:
            query: Natural language query string
            
        Returns:
            Classified QueryIntent
        """
        query_lower = query.lower()
        
        # Comparative queries (highest priority for clear indicators)
        if any(word in query_lower for word in COMPARATIVE_INDICATORS):
            return cls.COMPARATIVE_ANALYSIS
        
        # Troubleshooting queries
        if any(word in query_lower for word in TROUBLESHOOTING_INDICATORS):
            return cls.TROUBLESHOOTING
        
        # Code search queries (specific patterns with technology context)
        if (any(word in query_lower for word in CODE_SEARCH_INDICATORS) and 
            any(tech in query_lower for tech in TECHNOLOGY_MAPPINGS.keys())):
            return cls.CODE_SEARCH
        
        # Tutorial/how-to queries (avoid triggering on collection references)
        if (any(word in query_lower for word in TUTORIAL_INDICATORS) and 
            'collection' not in query_lower):
            return cls.TUTORIAL_SEEKING
        
        # Default to information seeking
        return cls.INFORMATION_SEEKING


@dataclass
class ContextualFilter:
    """Represents a contextual filter extracted from a query."""
    field: str
    value: Any
    operator: str = "equals"
    confidence: float = 1.0
    
    @classmethod
    def extract_from_text(cls, text: str) -> List['ContextualFilter']:
        """
        Extract contextual filters from text using pattern matching.
        
        Args:
            text: Input text to analyze for filters
            
        Returns:
            List of extracted ContextualFilter objects
        """
        filters = []
        text_lower = text.lower()
        
        # Collection filters (high confidence pattern)
        collection_match = re.search(r'(?:from|in)\s+(\w+)\s+collection', text_lower)
        if collection_match:
            filters.append(cls(
                field="collection",
                value=collection_match.group(1),
                operator="equals",
                confidence=0.9
            ))
        
        # Year-specific patterns (more specific than general temporal)
        year_match = re.search(r'from\s+(\d{4})', text_lower)
        if year_match:
            filters.append(cls(
                field="year",
                value=year_match.group(1),
                operator="equals",
                confidence=0.8
            ))
        
        # General temporal patterns
        temporal_patterns = [
            (r'from\s+last\s+week', "date", "greater_than"),
            (r'from\s+last\s+month', "date", "greater_than"),
            (r'recent', "date", "greater_than"),
        ]
        
        for pattern, field, operator in temporal_patterns:
            if re.search(pattern, text_lower):
                filters.append(cls(
                    field=field,
                    value="recent",
                    operator=operator,
                    confidence=0.7
                ))
        
        return filters


@dataclass
class QueryContext:
    """Structured representation of query context and intent."""
    original_query: str
    intent: QueryIntent
    key_terms: List[str] = field(default_factory=list)
    filters: List[ContextualFilter] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    temporal_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert QueryContext to dictionary representation."""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "key_terms": self.key_terms,
            "filters": [
                {
                    "field": f.field,
                    "value": f.value,
                    "operator": f.operator,
                    "confidence": f.confidence
                }
                for f in self.filters
            ],
            "preferences": self.preferences,
            "entities": self.entities,
            "temporal_context": self.temporal_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        """Create QueryContext from dictionary representation."""
        filters = [
            ContextualFilter(
                field=f["field"],
                value=f["value"],
                operator=f["operator"],
                confidence=f["confidence"]
            )
            for f in data.get("filters", [])
        ]
        
        return cls(
            original_query=data["original_query"],
            intent=QueryIntent(data["intent"]),
            key_terms=data.get("key_terms", []),
            filters=filters,
            preferences=data.get("preferences", {}),
            entities=data.get("entities", {}),
            temporal_context=data.get("temporal_context")
        ) 