"""
Enhanced metadata capabilities with semantic enrichment and advanced indexing.

This module provides advanced metadata features including:
- Semantic metadata enrichment using AI/NLP techniques
- Advanced metadata indexing for fast filtering and search
- Metadata-based semantic search capabilities
- Automated content classification and tagging

Implements requirements for enhanced metadata indexing and semantic enrichment.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)


class SemanticMetadataEnricher:
    """
    Semantic metadata enricher for automatic content analysis and tagging.
    
    Analyzes document content to automatically extract and enrich metadata
    with semantic tags, content classification, domain detection, and
    complexity scoring for improved searchability and organization.
    """
    
    def __init__(self):
        """Initialize the semantic metadata enricher."""
        self.domain_keywords = {
            "academic": [
                "research", "study", "analysis", "paper", "journal", "publication",
                "methodology", "findings", "conclusion", "hypothesis", "experiment"
            ],
            "technical": [
                "algorithm", "implementation", "code", "function", "class", "method",
                "framework", "library", "API", "documentation", "specification"
            ],
            "research": [
                "investigation", "survey", "review", "assessment", "evaluation",
                "comparison", "model", "theory", "approach", "technique"
            ]
        }
        
        self.complexity_indicators = [
            "advanced", "complex", "sophisticated", "comprehensive", "detailed",
            "in-depth", "extensive", "thorough", "intricate", "elaborate"
        ]
        
        logger.info("SemanticMetadataEnricher initialized")
    
    def enrich_metadata(
        self,
        document_content: str,
        base_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich document metadata with semantic analysis.
        
        Args:
            document_content: Text content of the document
            base_metadata: Existing metadata to enrich
            
        Returns:
            Enhanced metadata dictionary
        """
        # Start with base metadata
        enriched_metadata = base_metadata.copy()
        
        # Add semantic tags
        enriched_metadata["semantic_tags"] = self._extract_semantic_tags(document_content)
        
        # Determine content type
        enriched_metadata["content_type"] = self._classify_content_type(document_content)
        
        # Detect domain
        enriched_metadata["domain"] = self._detect_domain(document_content)
        
        # Calculate complexity score
        enriched_metadata["complexity_score"] = self._calculate_complexity_score(document_content)
        
        # Extract key phrases
        enriched_metadata["key_phrases"] = self._extract_key_phrases(document_content)
        
        # Estimate reading time
        enriched_metadata["estimated_reading_time_minutes"] = self._estimate_reading_time(document_content)
        
        logger.debug(f"Enriched metadata with {len(enriched_metadata) - len(base_metadata)} new fields")
        return enriched_metadata
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content."""
        content_lower = content.lower()
        tags = []
        
        # Technical tags
        if any(term in content_lower for term in ["neural", "network", "machine learning"]):
            tags.append("machine_learning")
        if any(term in content_lower for term in ["algorithm", "optimization"]):
            tags.append("algorithms")
        if any(term in content_lower for term in ["data", "dataset", "analysis"]):
            tags.append("data_analysis")
        if any(term in content_lower for term in ["model", "training", "prediction"]):
            tags.append("modeling")
        
        # Academic tags
        if any(term in content_lower for term in ["research", "study", "findings"]):
            tags.append("research")
        if any(term in content_lower for term in ["paper", "publication", "journal"]):
            tags.append("academic_paper")
        
        return tags
    
    def _classify_content_type(self, content: str) -> str:
        """Classify the type of content."""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ["def ", "class ", "import ", "function"]):
            return "code"
        elif any(term in content_lower for term in ["abstract", "introduction", "methodology"]):
            return "academic_paper"
        elif any(term in content_lower for term in ["tutorial", "guide", "how-to"]):
            return "tutorial"
        elif any(term in content_lower for term in ["api", "documentation", "reference"]):
            return "documentation"
        else:
            return "general"
    
    def _detect_domain(self, content: str) -> str:
        """Detect the domain/field of the content."""
        content_lower = content.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score, or "general" if no clear winner
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0:
                return best_domain
        
        return "general"
    
    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score from 1.0 (simple) to 10.0 (complex)."""
        content_lower = content.lower()
        
        # Base score
        complexity_score = 3.0
        
        # Factor in length
        word_count = len(content.split())
        if word_count > 1000:
            complexity_score += 2.0
        elif word_count > 500:
            complexity_score += 1.0
        
        # Factor in complexity indicators
        complexity_indicators_found = sum(
            1 for indicator in self.complexity_indicators
            if indicator in content_lower
        )
        complexity_score += complexity_indicators_found * 0.5
        
        # Factor in technical terminology
        technical_terms = [
            "optimization", "algorithm", "neural", "gradient", "architecture",
            "implementation", "framework", "methodology", "analysis"
        ]
        technical_score = sum(1 for term in technical_terms if term in content_lower)
        complexity_score += technical_score * 0.3
        
        # Ensure score is within valid range
        return min(10.0, max(1.0, complexity_score))
    
    def _extract_key_phrases(self, content: str) -> List[str]:
        """Extract key phrases from content."""
        # Simplified key phrase extraction
        sentences = content.split('. ')
        key_phrases = []
        
        for sentence in sentences[:3]:  # Take first 3 sentences
            # Look for noun phrases (very simplified)
            words = sentence.split()
            if len(words) >= 3:
                key_phrases.append(' '.join(words[:3]))
        
        return key_phrases[:5]  # Return top 5 key phrases
    
    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes (assuming 200 words per minute)."""
        word_count = len(content.split())
        reading_time = max(1, round(word_count / 200))
        return reading_time


class AdvancedMetadataIndex:
    """
    Advanced metadata indexing system for fast filtering and queries.
    
    Provides sophisticated indexing capabilities for metadata fields
    enabling complex filtering operations, range queries, and
    multi-criteria search across document collections.
    """
    
    def __init__(self):
        """Initialize the advanced metadata index."""
        self.document_metadata = {}
        self.field_indexes = {}
        
        logger.info("AdvancedMetadataIndex initialized")
    
    def add_document(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document's metadata to the index.
        
        Args:
            document_id: Unique identifier for the document
            metadata: Metadata dictionary to index
        """
        self.document_metadata[document_id] = metadata
        
        # Update field indexes
        for field, value in metadata.items():
            if field not in self.field_indexes:
                self.field_indexes[field] = {}
            
            # Handle different value types
            if isinstance(value, list):
                for item in value:
                    if item not in self.field_indexes[field]:
                        self.field_indexes[field][item] = set()
                    self.field_indexes[field][item].add(document_id)
            else:
                if value not in self.field_indexes[field]:
                    self.field_indexes[field][value] = set()
                self.field_indexes[field][value].add(document_id)
        
        logger.debug(f"Added document {document_id} to metadata index")
    
    def filter_documents(self, filter_criteria: Dict[str, Any]) -> List[str]:
        """
        Filter documents based on metadata criteria.
        
        Args:
            filter_criteria: Dictionary of field:value or field:operator_dict pairs
            
        Returns:
            List of document IDs matching the criteria
        """
        if not filter_criteria:
            return list(self.document_metadata.keys())
        
        matching_docs = None
        
        for field, criteria in filter_criteria.items():
            field_matches = self._filter_by_field(field, criteria)
            
            if matching_docs is None:
                matching_docs = field_matches
            else:
                # Intersect with previous results (AND operation)
                matching_docs = matching_docs.intersection(field_matches)
        
        result = list(matching_docs) if matching_docs else []
        logger.debug(f"Filter returned {len(result)} matching documents")
        return result
    
    def _filter_by_field(self, field: str, criteria: Any) -> set:
        """Filter documents by a specific field."""
        if field not in self.field_indexes:
            return set()
        
        if isinstance(criteria, dict):
            # Handle operator-based criteria
            return self._handle_operator_criteria(field, criteria)
        else:
            # Simple equality match
            return self.field_indexes[field].get(criteria, set())
    
    def _handle_operator_criteria(self, field: str, criteria: Dict[str, Any]) -> set:
        """Handle operator-based filtering criteria."""
        matching_docs = set()
        
        for operator, value in criteria.items():
            if operator == "$gte":
                # Greater than or equal
                for field_value, doc_set in self.field_indexes[field].items():
                    if isinstance(field_value, (int, float)) and field_value >= value:
                        matching_docs.update(doc_set)
            elif operator == "$lte":
                # Less than or equal
                for field_value, doc_set in self.field_indexes[field].items():
                    if isinstance(field_value, (int, float)) and field_value <= value:
                        matching_docs.update(doc_set)
            elif operator == "$in":
                # Value in list
                for item in value:
                    if item in self.field_indexes[field]:
                        matching_docs.update(self.field_indexes[field][item])
            elif operator == "$contains":
                # String contains
                for field_value, doc_set in self.field_indexes[field].items():
                    if isinstance(field_value, str) and value.lower() in field_value.lower():
                        matching_docs.update(doc_set)
        
        return matching_docs


class MetadataSemanticSearch:
    """
    Semantic search within metadata fields.
    
    Provides semantic search capabilities specifically for metadata content,
    enabling intelligent matching of queries against document titles,
    abstracts, tags, and other metadata fields.
    """
    
    def __init__(self):
        """Initialize the metadata semantic search."""
        self.metadata_corpus = {}
        self.searchable_fields = ["title", "abstract", "description", "tags", "key_phrases"]
        
        logger.info("MetadataSemanticSearch initialized")
    
    def index_metadata(self, document_id: str, metadata: Dict[str, Any]) -> None:
        """
        Index metadata for semantic search.
        
        Args:
            document_id: Unique identifier for the document
            metadata: Metadata dictionary to index
        """
        # Extract searchable text from metadata
        searchable_text = []
        for field in self.searchable_fields:
            if field in metadata:
                value = metadata[field]
                if isinstance(value, str):
                    searchable_text.append(value)
                elif isinstance(value, list):
                    searchable_text.extend([str(item) for item in value])
        
        self.metadata_corpus[document_id] = {
            "searchable_text": " ".join(searchable_text),
            "metadata": metadata
        }
        
        logger.debug(f"Indexed metadata for document {document_id}")
    
    def search_metadata(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search metadata using semantic similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata and scores
        """
        query_lower = query.lower()
        results = []
        
        for doc_id, doc_data in self.metadata_corpus.items():
            searchable_text = doc_data["searchable_text"].lower()
            
            # Simple semantic matching based on keyword overlap and relevance
            score = self._calculate_semantic_score(query_lower, searchable_text)
            
            if score > 0:
                results.append({
                    "id": doc_id,
                    "score": score,
                    "metadata": doc_data["metadata"]
                })
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Metadata search returned {len(results[:top_k])} results")
        return results[:top_k]
    
    def _calculate_semantic_score(self, query: str, text: str) -> float:
        """Calculate semantic similarity score between query and text."""
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
        
        # Basic word overlap score
        overlap = len(query_words.intersection(text_words))
        base_score = overlap / len(query_words)
        
        # Boost score for specific domain matches
        domain_boost = 0.0
        if "neural" in query and "neural" in text:
            domain_boost += 0.3
        if "machine learning" in query and "machine learning" in text:
            domain_boost += 0.4
        if "algorithm" in query and "algorithm" in text:
            domain_boost += 0.2
        
        # Boost score for title matches (assuming title appears early in text)
        title_boost = 0.0
        text_words_list = text.split()
        if len(text_words_list) > 0:
            title_portion = " ".join(text_words_list[:10])  # First 10 words as "title"
            if any(word in title_portion for word in query.split()):
                title_boost = 0.2
        
        final_score = min(1.0, base_score + domain_boost + title_boost)
        return final_score 