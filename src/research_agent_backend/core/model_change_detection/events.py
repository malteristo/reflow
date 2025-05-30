"""
ModelChangeEvent representation and assessment for change notifications.

This module provides event objects for representing model changes and
determining their impact on downstream operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from .fingerprint import ModelFingerprint
from .types import ChangeType


@dataclass
class ModelChangeEvent:
    """
    Event object representing a detected model change requiring action.
    
    ModelChangeEvent serves as a notification mechanism for model changes
    that may require downstream actions such as cache invalidation or
    vector database re-indexing. It follows the Observer pattern to
    enable loose coupling between change detection and response logic.
    
    Attributes:
        model_name: Name of the changed model
        change_type: Type of change that occurred
        old_fingerprint: Previous model state (None for new models)
        new_fingerprint: Current model state
        requires_reindexing: Whether vector database re-indexing is needed
        timestamp: When the change was detected
        metadata: Additional event-specific information
    
    Example:
        >>> event = ModelChangeEvent(
        ...     model_name="text-embedding-3-small",
        ...     change_type="version_update",
        ...     old_fingerprint=old_fp,
        ...     new_fingerprint=new_fp,
        ...     requires_reindexing=True
        ... )
        >>> if event.requires_reindexing:
        ...     trigger_reindexing(event.model_name)
    """
    
    model_name: str
    change_type: ChangeType
    old_fingerprint: Optional[ModelFingerprint]
    new_fingerprint: ModelFingerprint
    requires_reindexing: bool = True
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to dictionary for serialization and logging.
        
        Returns:
            JSON-serializable dictionary representation
        """
        return {
            "model_name": self.model_name,
            "change_type": self.change_type,
            "old_fingerprint": self.old_fingerprint.to_dict() if self.old_fingerprint else None,
            "new_fingerprint": self.new_fingerprint.to_dict(),
            "requires_reindexing": self.requires_reindexing,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def should_invalidate_cache(self) -> bool:
        """
        Determine if this change should trigger cache invalidation.
        
        Returns:
            True if cache invalidation is recommended
        """
        # Cache invalidation is recommended for all change types except metadata-only changes
        return self.change_type in ("new_model", "version_update", "checksum_change")
    
    def get_impact_level(self) -> Literal["low", "medium", "high"]:
        """
        Assess the impact level of this model change.
        
        Returns:
            Impact level classification for prioritization
        """
        if self.change_type == "new_model":
            return "medium"
        elif self.change_type in ("version_update", "checksum_change"):
            return "high"
        else:  # config_change
            return "low" 