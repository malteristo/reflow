"""
Configuration management for re-ranking pipeline integration.

Provides configuration dataclass for managing pipeline parameters including
re-ranking settings, performance optimization, and monitoring options.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class PipelineConfig:
    """
    Configuration for re-ranking pipeline integration.
    
    Manages all parameters for pipeline integration including re-ranking
    settings, performance optimization, and monitoring options.
    
    Attributes:
        enable_reranking: Whether to apply re-ranking to search results
        rerank_top_k: Maximum candidates to consider for re-ranking
        rerank_top_n: Number of results to return after re-ranking
        rerank_threshold: Minimum re-ranking score threshold for filtering
        enable_logging: Whether to enable comprehensive pipeline logging
        enable_monitoring: Whether to collect performance metrics
        preserve_original_order: Whether to preserve order for passthrough
        data_optimization: Settings for data flow optimization
    """
    enable_reranking: bool = True
    rerank_top_k: int = 20
    rerank_top_n: int = 5
    rerank_threshold: float = 0.1
    enable_logging: bool = False
    enable_monitoring: bool = False
    preserve_original_order: bool = True
    
    # Data flow optimization settings
    data_optimization: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.rerank_top_k <= 0:
            raise ValueError(f"rerank_top_k must be positive, got: {self.rerank_top_k}")
        
        if self.rerank_top_n <= 0:
            raise ValueError(f"rerank_top_n must be positive, got: {self.rerank_top_n}")
        
        if self.rerank_threshold < 0:
            raise ValueError(f"rerank_threshold must be non-negative, got: {self.rerank_threshold}")
        
        if self.rerank_top_n > self.rerank_top_k:
            raise ValueError(f"rerank_top_n ({self.rerank_top_n}) cannot be greater than rerank_top_k ({self.rerank_top_k})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'enable_reranking': self.enable_reranking,
            'rerank_top_k': self.rerank_top_k,
            'rerank_top_n': self.rerank_top_n,
            'rerank_threshold': self.rerank_threshold,
            'enable_logging': self.enable_logging,
            'enable_monitoring': self.enable_monitoring,
            'preserve_original_order': self.preserve_original_order,
            'data_optimization': self.data_optimization.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def copy(self, **overrides) -> 'PipelineConfig':
        """Create a copy with optional parameter overrides."""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data) 