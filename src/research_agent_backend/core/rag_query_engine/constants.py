"""
RAG Query Engine Constants - Configuration and mappings.

This module contains all constants, mappings, and configuration values
used throughout the RAG query processing pipeline.
"""

# Stop words for query processing
STOP_WORDS = {
    'what', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
    'to', 'for', 'of', 'with', 'by', 'from', 'show', 'me', 'find',
    'about', 'how', 'collection', 'research'
}

# Intent classification indicators
COMPARATIVE_INDICATORS = ['vs', 'versus', 'compare', 'difference between']
TROUBLESHOOTING_INDICATORS = ['error', 'bug', 'fix', 'problem', 'issue']
CODE_SEARCH_INDICATORS = ['function', 'code', 'example', 'implementation']
TUTORIAL_INDICATORS = ['how to', 'tutorial', 'step by step', 'guide']

# Technology mappings for entity extraction
TECHNOLOGY_MAPPINGS = {
    'python': 'Python',
    'javascript': 'JavaScript',
    'react': 'React', 
    'vue.js': 'Vue.js',
    'django': 'Django',
    'tensorflow': 'TensorFlow',
    'docker': 'Docker'
}

# Hardware mappings for entity extraction
HARDWARE_MAPPINGS = {
    'gpu': 'GPU',
    'cpu': 'CPU', 
    'ram': 'RAM',
    'ssd': 'SSD'
}

# Compound terms that should be treated as single entities
COMPOUND_TERMS = ['machine learning', 'data validation']

# Complexity level indicators
COMPLEXITY_BEGINNER_WORDS = ['simple', 'basic', 'beginner', 'easy']
COMPLEXITY_ADVANCED_WORDS = ['advanced', 'expert', 'complex']
EXAMPLE_INDICATORS = ['example', 'examples', 'with examples']

# Enhancement mappings for query embedding
INTENT_ENHANCEMENT_MAP = {
    "COMPARATIVE_ANALYSIS": "comparison analysis",
    "TUTORIAL_SEEKING": "tutorial guide how-to",
    "CODE_SEARCH": "code implementation example",
    "TROUBLESHOOTING": "troubleshooting solution fix"
}

TEMPORAL_ENHANCEMENT_MAP = {
    "recent": "recent latest",
    "last_month": "recent month",
    "last_week": "recent week"
}

COMPLEXITY_ENHANCEMENT_MAP = {
    "beginner": "beginner simple",
    "advanced": "advanced expert"
}

# Default search parameters
DEFAULT_TOP_K = 20
DEFAULT_DISTANCE_THRESHOLD = None

# Feedback generation constants
RELEVANCE_THRESHOLDS = {
    "HIGH": 0.8,
    "MODERATE": 0.5,
    "LOW": 0.0
}

SUGGESTION_TYPES = [
    "add_filter", "refine_terms", "expand_scope", 
    "change_collection", "add_context"
]

TECH_TERMS = [
    "python", "programming", "code", "function", 
    "javascript", "react", "django", "ai", "machine learning"
]

MISMATCH_COLLECTIONS = ["cooking", "recipes", "food"]

RELEVANCE_DESCRIPTIONS = {
    "high": "high similarity to query terms",
    "moderate": "moderate similarity to query terms", 
    "low": "low similarity to query terms"
}

PREFERENCE_KEYWORDS = {
    "beginner": ["simple", "basic", "beginner"],
    "examples": ["example"]
} 