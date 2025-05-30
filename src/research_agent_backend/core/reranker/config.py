"""
Configuration management for re-ranking service.

Provides configuration dataclass and utilities for managing re-ranking
service parameters including model selection, batch processing, and
performance optimization settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class RerankerConfig:
    """
    Configuration for cross-encoder re-ranking service.
    
    Manages all parameters for re-ranking including model selection,
    batch processing, caching, and performance optimization.
    
    Attributes:
        model_name: Name of cross-encoder model to use
        batch_size: Number of query-document pairs to process in batch
        max_length: Maximum sequence length for model input
        device: Device to run model on ('cpu', 'cuda', 'mps')
        enable_caching: Whether to cache scores for repeated queries
        cache_size: Maximum number of cached query-document pairs
        temperature: Temperature for score normalization (if applicable)
        top_k_candidates: Maximum candidates to re-rank (performance limit)
    """
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L6-v2'
    batch_size: int = 32
    max_length: int = 512
    device: str = 'cpu'
    enable_caching: bool = True
    cache_size: int = 1000
    temperature: float = 1.0
    top_k_candidates: int = 100
    
    # Performance and optimization settings
    use_fp16: bool = False  # Half precision for GPU speedup
    num_threads: Optional[int] = None  # CPU thread count
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got: {self.batch_size}")
        
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got: {self.max_length}")
        
        if self.cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got: {self.cache_size}")
        
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got: {self.temperature}")
        
        if self.top_k_candidates <= 0:
            raise ValueError(f"top_k_candidates must be positive, got: {self.top_k_candidates}")
        
        if self.device not in ['cpu', 'cuda', 'mps', 'auto']:
            raise ValueError(f"device must be one of ['cpu', 'cuda', 'mps', 'auto'], got: {self.device}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'device': self.device,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'temperature': self.temperature,
            'top_k_candidates': self.top_k_candidates,
            'use_fp16': self.use_fp16,
            'num_threads': self.num_threads,
            'metadata': self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RerankerConfig':
        """Create config from dictionary."""
        # Filter out any unknown keys
        valid_keys = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def copy(self, **overrides) -> 'RerankerConfig':
        """Create a copy with optional parameter overrides."""
        data = self.to_dict()
        data.update(overrides)
        return self.from_dict(data) 