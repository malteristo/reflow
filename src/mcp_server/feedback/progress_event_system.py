"""
Progress Event System for Structured Feedback and Progress Reporting.

Implements real-time progress tracking and event generation during long-running
operations like document ingestion and knowledge base queries.

Part of subtask 15.7: Implement Structured Feedback and Progress Reporting.
"""

import time
import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
import uuid

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be tracked."""
    DOCUMENT_INGESTION = "document_ingestion"
    QUERY_PROCESSING = "query_processing"
    COLLECTION_MANAGEMENT = "collection_management"
    KNOWLEDGE_AUGMENTATION = "knowledge_augmentation"
    EMBEDDING_GENERATION = "embedding_generation"
    VECTOR_SEARCH = "vector_search"
    RERANKING = "reranking"
    RESULT_FORMATTING = "result_formatting"


class ProgressPhase(Enum):
    """Phases of operation progress."""
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressEvent:
    """Individual progress event data structure."""
    
    # Event identification
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_id: str = ""
    operation_type: OperationType = OperationType.DOCUMENT_INGESTION
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Progress data
    progress_percentage: float = 0.0
    current_phase: ProgressPhase = ProgressPhase.INITIALIZING
    current_item: str = ""
    items_processed: int = 0
    total_items: int = 0
    
    # Time estimation
    estimated_time_remaining: float = 0.0
    elapsed_time: float = 0.0
    
    # Context information
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress event to dictionary for transmission."""
        return {
            "event_id": self.event_id,
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "timestamp": self.timestamp.isoformat(),
            "progress_percentage": self.progress_percentage,
            "current_phase": self.current_phase.value,
            "current_item": self.current_item,
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "estimated_time_remaining": self.estimated_time_remaining,
            "elapsed_time": self.elapsed_time,
            "message": self.message,
            "metadata": self.metadata
        }


@dataclass
class ProgressTracker:
    """Tracks progress for a specific operation."""
    
    # Operation identification
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: OperationType = OperationType.DOCUMENT_INGESTION
    
    # Progress tracking
    start_time: datetime = field(default_factory=datetime.utcnow)
    total_items: int = 0
    items_processed: int = 0
    current_phase: ProgressPhase = ProgressPhase.INITIALIZING
    current_item: str = ""
    
    # Time estimation
    _progress_history: List[tuple[float, float]] = field(default_factory=list, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    
    def update_progress(
        self,
        items_processed: Optional[int] = None,
        current_item: Optional[str] = None,
        phase: Optional[ProgressPhase] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressEvent:
        """Update progress and generate progress event."""
        with self._lock:
            # Update tracking data
            if items_processed is not None:
                self.items_processed = items_processed
            if current_item is not None:
                self.current_item = current_item
            if phase is not None:
                self.current_phase = phase
            
            # Calculate progress percentage
            progress_percentage = 0.0
            if self.total_items > 0:
                progress_percentage = min(100.0, (self.items_processed / self.total_items) * 100.0)
            
            # Calculate elapsed time
            elapsed_time = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Record progress for time estimation
            self._progress_history.append((elapsed_time, progress_percentage))
            
            # Calculate estimated time remaining
            estimated_time_remaining = self._calculate_eta(progress_percentage, elapsed_time)
            
            # Create progress event
            event = ProgressEvent(
                operation_id=self.operation_id,
                operation_type=self.operation_type,
                progress_percentage=progress_percentage,
                current_phase=self.current_phase,
                current_item=self.current_item,
                items_processed=self.items_processed,
                total_items=self.total_items,
                estimated_time_remaining=estimated_time_remaining,
                elapsed_time=elapsed_time,
                message=message,
                metadata=metadata or {}
            )
            
            return event
    
    def _calculate_eta(self, current_progress: float, elapsed_time: float) -> float:
        """Calculate estimated time remaining based on progress history."""
        if current_progress <= 0 or len(self._progress_history) < 2:
            return 0.0
        
        # Use recent progress data for more accurate estimation
        recent_history = self._progress_history[-10:]  # Last 10 data points
        
        if len(recent_history) < 2:
            return 0.0
        
        # Calculate average progress rate (percentage per second)
        time_diff = recent_history[-1][0] - recent_history[0][0]
        progress_diff = recent_history[-1][1] - recent_history[0][1]
        
        if time_diff <= 0 or progress_diff <= 0:
            return 0.0
        
        progress_rate = progress_diff / time_diff
        remaining_progress = 100.0 - current_progress
        
        return remaining_progress / progress_rate if progress_rate > 0 else 0.0


class ProgressEventSystem:
    """Manages progress tracking and event generation for all operations."""
    
    def __init__(
        self,
        update_interval: float = 1.0,
        batch_size: int = 10,
        max_events_in_memory: int = 1000
    ):
        """
        Initialize the progress event system.
        
        Args:
            update_interval: Minimum interval between progress updates (seconds)
            batch_size: Number of events to batch together for transmission
            max_events_in_memory: Maximum events to keep in memory
        """
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.max_events_in_memory = max_events_in_memory
        
        # Active trackers
        self._trackers: Dict[str, ProgressTracker] = {}
        
        # Event batching and transmission
        self._event_queue: List[ProgressEvent] = []
        self._event_callbacks: List[Callable[[List[ProgressEvent]], None]] = []
        self._last_update_time: Dict[str, float] = {}
        
        # Threading support
        self._lock = Lock()
        
        logger.info(f"ProgressEventSystem initialized with update_interval={update_interval}s, batch_size={batch_size}")
    
    def create_tracker(
        self,
        operation_type: OperationType,
        total_items: int = 0,
        operation_id: Optional[str] = None
    ) -> ProgressTracker:
        """Create a new progress tracker for an operation."""
        tracker = ProgressTracker(
            operation_id=operation_id or str(uuid.uuid4()),
            operation_type=operation_type,
            total_items=total_items
        )
        
        with self._lock:
            self._trackers[tracker.operation_id] = tracker
            self._last_update_time[tracker.operation_id] = 0.0
        
        logger.info(f"Created progress tracker for {operation_type.value} with {total_items} total items")
        return tracker
    
    def update_progress(
        self,
        operation_id: str,
        items_processed: Optional[int] = None,
        current_item: Optional[str] = None,
        phase: Optional[ProgressPhase] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        force_update: bool = False
    ) -> Optional[ProgressEvent]:
        """Update progress for an operation and potentially generate event."""
        with self._lock:
            tracker = self._trackers.get(operation_id)
            if not tracker:
                logger.warning(f"No tracker found for operation_id: {operation_id}")
                return None
            
            # Check if enough time has passed since last update
            current_time = time.time()
            last_update = self._last_update_time.get(operation_id, 0)
            
            if not force_update and (current_time - last_update) < self.update_interval:
                return None
            
            # Update progress and generate event
            event = tracker.update_progress(
                items_processed=items_processed,
                current_item=current_item,
                phase=phase,
                message=message,
                metadata=metadata
            )
            
            # Record update time
            self._last_update_time[operation_id] = current_time
            
            # Add to event queue
            self._event_queue.append(event)
            
            # Process event queue if needed
            self._process_event_queue()
            
            return event
    
    def complete_operation(self, operation_id: str, message: str = "Operation completed") -> Optional[ProgressEvent]:
        """Mark an operation as completed."""
        final_event = self.update_progress(
            operation_id=operation_id,
            phase=ProgressPhase.COMPLETED,
            message=message,
            force_update=True
        )
        
        # Clean up tracker
        with self._lock:
            if operation_id in self._trackers:
                del self._trackers[operation_id]
            if operation_id in self._last_update_time:
                del self._last_update_time[operation_id]
        
        logger.info(f"Completed operation: {operation_id}")
        return final_event
    
    def error_operation(self, operation_id: str, error_message: str) -> Optional[ProgressEvent]:
        """Mark an operation as failed with error."""
        error_event = self.update_progress(
            operation_id=operation_id,
            phase=ProgressPhase.ERROR,
            message=f"Error: {error_message}",
            force_update=True
        )
        
        # Clean up tracker
        with self._lock:
            if operation_id in self._trackers:
                del self._trackers[operation_id]
            if operation_id in self._last_update_time:
                del self._last_update_time[operation_id]
        
        logger.error(f"Operation failed: {operation_id} - {error_message}")
        return error_event
    
    def add_event_callback(self, callback: Callable[[List[ProgressEvent]], None]) -> None:
        """Add a callback to be called when events are ready for transmission."""
        self._event_callbacks.append(callback)
        logger.info(f"Added event callback: {callback.__name__}")
    
    def remove_event_callback(self, callback: Callable[[List[ProgressEvent]], None]) -> None:
        """Remove an event callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)
            logger.info(f"Removed event callback: {callback.__name__}")
    
    def _process_event_queue(self) -> None:
        """Process the event queue and trigger callbacks if batch is ready."""
        if len(self._event_queue) >= self.batch_size:
            # Extract batch of events
            batch = self._event_queue[:self.batch_size]
            self._event_queue = self._event_queue[self.batch_size:]
            
            # Trigger callbacks
            for callback in self._event_callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Error in event callback {callback.__name__}: {e}")
        
        # Prevent memory buildup
        if len(self._event_queue) > self.max_events_in_memory:
            overflow = len(self._event_queue) - self.max_events_in_memory
            self._event_queue = self._event_queue[overflow:]
            logger.warning(f"Event queue overflow: removed {overflow} oldest events")
    
    def flush_events(self) -> None:
        """Flush all pending events regardless of batch size."""
        if self._event_queue:
            batch = self._event_queue.copy()
            self._event_queue.clear()
            
            for callback in self._event_callbacks:
                try:
                    callback(batch)
                except Exception as e:
                    logger.error(f"Error in event callback {callback.__name__}: {e}")
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get information about all active operations."""
        with self._lock:
            operations = []
            for operation_id, tracker in self._trackers.items():
                operations.append({
                    "operation_id": operation_id,
                    "operation_type": tracker.operation_type.value,
                    "current_phase": tracker.current_phase.value,
                    "items_processed": tracker.items_processed,
                    "total_items": tracker.total_items,
                    "current_item": tracker.current_item,
                    "elapsed_time": (datetime.utcnow() - tracker.start_time).total_seconds()
                })
            return operations
    
    def cleanup(self) -> None:
        """Clean up resources and flush remaining events."""
        logger.info("Cleaning up ProgressEventSystem")
        self.flush_events()
        
        with self._lock:
            self._trackers.clear()
            self._last_update_time.clear()
            self._event_queue.clear()
            self._event_callbacks.clear() 