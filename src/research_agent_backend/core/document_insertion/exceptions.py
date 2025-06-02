"""
Exception classes and result models for document insertion operations.

This module provides exception hierarchy and result dataclasses for the 
document insertion pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Custom Exception Classes
class InsertionError(Exception):
    """Base exception for document insertion failures."""
    pass


class ValidationError(InsertionError):
    """Exception for validation failures during insertion."""
    pass


class TransactionError(InsertionError):
    """Exception for transaction-related failures."""
    pass


@dataclass
class InsertionResult:
    """Result of single document insertion operation."""
    success: bool = False
    document_id: Optional[str] = None
    chunk_count: int = 0
    chunk_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    cache_hit: bool = False  # Track cache hits for optimization
    hybrid_chunking_stats: Optional[Dict[str, Any]] = None  # FR-KB-002.1 chunking statistics
    
    @property
    def has_errors(self) -> bool:
        """Check if result has errors."""
        return len(self.errors) > 0


@dataclass  
class BatchInsertionResult:
    """Result of batch document insertion operation."""
    total_documents: int = 0
    successful_insertions: int = 0
    failed_insertions: int = 0
    success: bool = False
    document_ids: List[str] = field(default_factory=list)
    failed_documents: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    transaction_id: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of batch insertion."""
        if self.total_documents == 0:
            return 0.0
        return self.successful_insertions / self.total_documents 