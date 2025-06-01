"""
Status Update Protocol for Structured Feedback and Progress Reporting.

Implements MCP-compliant status reporting and tracking for operations.
Part of subtask 15.7: Implement Structured Feedback and Progress Reporting.
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json

logger = logging.getLogger(__name__)


class OperationStatus(Enum):
    """Status values for operations."""
    STARTED = "started"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class StatusMessage:
    """Status message data structure for MCP protocol compliance."""
    
    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Status information
    status: OperationStatus = OperationStatus.STARTED
    message: str = ""
    progress_percentage: float = 0.0
    
    # MCP protocol fields
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Error information (when status is ERROR)
    error_code: Optional[str] = None
    error_details: Optional[str] = None
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert status message to MCP protocol format."""
        mcp_message = {
            "jsonrpc": "2.0",
            "method": "notifications/status",
            "params": {
                "message_id": self.message_id,
                "operation_id": self.operation_id,
                "timestamp": self.timestamp.isoformat(),
                "status": self.status.value,
                "message": self.message,
                "progress_percentage": self.progress_percentage,
                "metadata": self.metadata,
                "context": self.context
            }
        }
        
        # Add error information if present
        if self.status == OperationStatus.ERROR:
            mcp_message["params"]["error"] = {
                "code": self.error_code or "OPERATION_ERROR",
                "message": self.error_details or self.message
            }
        
        return mcp_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status message to dictionary."""
        return {
            "message_id": self.message_id,
            "operation_id": self.operation_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "message": self.message,
            "progress_percentage": self.progress_percentage,
            "metadata": self.metadata,
            "context": self.context,
            "error_code": self.error_code,
            "error_details": self.error_details
        }


@dataclass
class OperationTracker:
    """Tracks status for a specific operation."""
    
    # Operation identification
    operation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation_type: str = ""
    
    # Status tracking
    current_status: OperationStatus = OperationStatus.STARTED
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_update_time: datetime = field(default_factory=datetime.utcnow)
    completion_time: Optional[datetime] = None
    
    # Status history
    status_history: List[StatusMessage] = field(default_factory=list)
    
    # Operation context
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def update_status(
        self,
        status: OperationStatus,
        message: str = "",
        progress_percentage: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> StatusMessage:
        """Update operation status and create status message."""
        with self._lock:
            # Update tracker state
            self.current_status = status
            self.last_update_time = datetime.utcnow()
            
            if status in [OperationStatus.COMPLETED, OperationStatus.ERROR, OperationStatus.CANCELLED]:
                self.completion_time = self.last_update_time
            
            # Update context and metadata
            if metadata:
                self.metadata.update(metadata)
            if context:
                self.context.update(context)
            
            # Create status message
            status_msg = StatusMessage(
                operation_id=self.operation_id,
                status=status,
                message=message,
                progress_percentage=progress_percentage or 0.0,
                metadata=self.metadata.copy(),
                context=self.context.copy(),
                error_code=error_code,
                error_details=error_details
            )
            
            # Add to history
            self.status_history.append(status_msg)
            
            return status_msg
    
    def get_current_status(self) -> StatusMessage:
        """Get the current status message."""
        with self._lock:
            if self.status_history:
                return self.status_history[-1]
            else:
                # Create default status message
                return StatusMessage(
                    operation_id=self.operation_id,
                    status=self.current_status,
                    message="Operation initialized",
                    metadata=self.metadata.copy(),
                    context=self.context.copy()
                )
    
    def get_status_history(self) -> List[StatusMessage]:
        """Get complete status history."""
        with self._lock:
            return self.status_history.copy()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        end_time = self.completion_time or datetime.utcnow()
        return (end_time - self.start_time).total_seconds()


class StatusUpdateProtocol:
    """Manages status updates for operations with MCP protocol compliance."""
    
    def __init__(
        self,
        enable_persistence: bool = True,
        max_history_per_operation: int = 100,
        cleanup_completed_after_seconds: int = 3600  # 1 hour
    ):
        """
        Initialize the status update protocol.
        
        Args:
            enable_persistence: Whether to persist status across requests
            max_history_per_operation: Maximum status messages to keep per operation
            cleanup_completed_after_seconds: Cleanup completed operations after this time
        """
        self.enable_persistence = enable_persistence
        self.max_history_per_operation = max_history_per_operation
        self.cleanup_completed_after_seconds = cleanup_completed_after_seconds
        
        # Operation tracking
        self._operations: Dict[str, OperationTracker] = {}
        
        # Status message callbacks
        self._status_callbacks: List[Callable[[StatusMessage], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"StatusUpdateProtocol initialized with persistence={enable_persistence}")
    
    def start_operation(
        self,
        operation_id: Optional[str] = None,
        operation_type: str = "",
        initial_message: str = "Operation started",
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start tracking a new operation."""
        if not operation_id:
            operation_id = str(uuid.uuid4())
        
        with self._lock:
            # Create operation tracker
            tracker = OperationTracker(
                operation_id=operation_id,
                operation_type=operation_type,
                context=context or {},
                metadata=metadata or {}
            )
            
            # Update status to started
            status_msg = tracker.update_status(
                status=OperationStatus.STARTED,
                message=initial_message,
                context=context,
                metadata=metadata
            )
            
            # Store tracker
            self._operations[operation_id] = tracker
            
            # Notify callbacks
            self._notify_status_callbacks(status_msg)
            
            logger.info(f"Started operation tracking: {operation_id} ({operation_type})")
            return operation_id
    
    def update_operation_status(
        self,
        operation_id: str,
        status: OperationStatus,
        message: str = "",
        progress_percentage: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
        error_details: Optional[str] = None
    ) -> Optional[StatusMessage]:
        """Update status for an existing operation."""
        with self._lock:
            tracker = self._operations.get(operation_id)
            if not tracker:
                logger.warning(f"No operation found for ID: {operation_id}")
                return None
            
            # Update status
            status_msg = tracker.update_status(
                status=status,
                message=message,
                progress_percentage=progress_percentage,
                metadata=metadata,
                context=context,
                error_code=error_code,
                error_details=error_details
            )
            
            # Limit history size
            if len(tracker.status_history) > self.max_history_per_operation:
                tracker.status_history = tracker.status_history[-self.max_history_per_operation:]
            
            # Notify callbacks
            self._notify_status_callbacks(status_msg)
            
            return status_msg
    
    def complete_operation(
        self,
        operation_id: str,
        message: str = "Operation completed successfully",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[StatusMessage]:
        """Mark an operation as completed."""
        return self.update_operation_status(
            operation_id=operation_id,
            status=OperationStatus.COMPLETED,
            message=message,
            progress_percentage=100.0,
            metadata=metadata
        )
    
    def error_operation(
        self,
        operation_id: str,
        error_message: str,
        error_code: str = "OPERATION_ERROR",
        error_details: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[StatusMessage]:
        """Mark an operation as failed with error."""
        return self.update_operation_status(
            operation_id=operation_id,
            status=OperationStatus.ERROR,
            message=error_message,
            error_code=error_code,
            error_details=error_details,
            metadata=metadata
        )
    
    def cancel_operation(
        self,
        operation_id: str,
        message: str = "Operation cancelled",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[StatusMessage]:
        """Cancel an operation."""
        return self.update_operation_status(
            operation_id=operation_id,
            status=OperationStatus.CANCELLED,
            message=message,
            metadata=metadata
        )
    
    def get_operation_status(self, operation_id: str) -> Optional[StatusMessage]:
        """Get current status for an operation."""
        with self._lock:
            tracker = self._operations.get(operation_id)
            if tracker:
                return tracker.get_current_status()
            return None
    
    def get_operation_history(self, operation_id: str) -> List[StatusMessage]:
        """Get complete status history for an operation."""
        with self._lock:
            tracker = self._operations.get(operation_id)
            if tracker:
                return tracker.get_status_history()
            return []
    
    def get_active_operations(self) -> List[Dict[str, Any]]:
        """Get information about all active (non-completed) operations."""
        with self._lock:
            active_ops = []
            for operation_id, tracker in self._operations.items():
                if tracker.current_status not in [
                    OperationStatus.COMPLETED,
                    OperationStatus.ERROR,
                    OperationStatus.CANCELLED
                ]:
                    active_ops.append({
                        "operation_id": operation_id,
                        "operation_type": tracker.operation_type,
                        "status": tracker.current_status.value,
                        "start_time": tracker.start_time.isoformat(),
                        "elapsed_time": tracker.get_elapsed_time(),
                        "last_update": tracker.last_update_time.isoformat()
                    })
            return active_ops
    
    def get_all_operations(self) -> List[Dict[str, Any]]:
        """Get information about all operations."""
        with self._lock:
            all_ops = []
            for operation_id, tracker in self._operations.items():
                all_ops.append({
                    "operation_id": operation_id,
                    "operation_type": tracker.operation_type,
                    "status": tracker.current_status.value,
                    "start_time": tracker.start_time.isoformat(),
                    "elapsed_time": tracker.get_elapsed_time(),
                    "last_update": tracker.last_update_time.isoformat(),
                    "completion_time": tracker.completion_time.isoformat() if tracker.completion_time else None,
                    "status_count": len(tracker.status_history)
                })
            return all_ops
    
    def add_status_callback(self, callback: Callable[[StatusMessage], None]) -> None:
        """Add a callback to be called when status updates occur."""
        with self._lock:
            self._status_callbacks.append(callback)
            logger.info(f"Added status callback: {callback.__name__}")
    
    def remove_status_callback(self, callback: Callable[[StatusMessage], None]) -> None:
        """Remove a status callback."""
        with self._lock:
            if callback in self._status_callbacks:
                self._status_callbacks.remove(callback)
                logger.info(f"Removed status callback: {callback.__name__}")
    
    def _notify_status_callbacks(self, status_msg: StatusMessage) -> None:
        """Notify all registered status callbacks."""
        for callback in self._status_callbacks:
            try:
                callback(status_msg)
            except Exception as e:
                logger.error(f"Error in status callback {callback.__name__}: {e}")
    
    def cleanup_completed_operations(self) -> int:
        """Clean up old completed operations."""
        if not self.enable_persistence:
            return 0
        
        current_time = datetime.utcnow()
        cleanup_count = 0
        
        with self._lock:
            operations_to_remove = []
            
            for operation_id, tracker in self._operations.items():
                if (tracker.completion_time and 
                    tracker.current_status in [OperationStatus.COMPLETED, OperationStatus.ERROR, OperationStatus.CANCELLED]):
                    
                    elapsed = (current_time - tracker.completion_time).total_seconds()
                    if elapsed > self.cleanup_completed_after_seconds:
                        operations_to_remove.append(operation_id)
            
            for operation_id in operations_to_remove:
                del self._operations[operation_id]
                cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} completed operations")
        
        return cleanup_count
    
    def clear_all_operations(self) -> None:
        """Clear all operation tracking data."""
        with self._lock:
            operation_count = len(self._operations)
            self._operations.clear()
            logger.info(f"Cleared all operation tracking data ({operation_count} operations)")
    
    def get_operation_statistics(self) -> Dict[str, Any]:
        """Get statistics about operations."""
        with self._lock:
            stats = {
                "total_operations": len(self._operations),
                "active_operations": 0,
                "completed_operations": 0,
                "failed_operations": 0,
                "cancelled_operations": 0,
                "status_by_type": {},
                "average_completion_time": 0.0
            }
            
            completion_times = []
            
            for tracker in self._operations.values():
                # Count by status
                status = tracker.current_status.value
                if status == "started" or status == "processing" or status == "paused":
                    stats["active_operations"] += 1
                elif status == "completed":
                    stats["completed_operations"] += 1
                    if tracker.completion_time:
                        completion_times.append(tracker.get_elapsed_time())
                elif status == "error":
                    stats["failed_operations"] += 1
                elif status == "cancelled":
                    stats["cancelled_operations"] += 1
                
                # Count by operation type
                op_type = tracker.operation_type or "unknown"
                if op_type not in stats["status_by_type"]:
                    stats["status_by_type"][op_type] = {
                        "total": 0,
                        "active": 0,
                        "completed": 0,
                        "failed": 0,
                        "cancelled": 0
                    }
                
                stats["status_by_type"][op_type]["total"] += 1
                if status in ["started", "processing", "paused"]:
                    stats["status_by_type"][op_type]["active"] += 1
                elif status == "completed":
                    stats["status_by_type"][op_type]["completed"] += 1
                elif status == "error":
                    stats["status_by_type"][op_type]["failed"] += 1
                elif status == "cancelled":
                    stats["status_by_type"][op_type]["cancelled"] += 1
            
            # Calculate average completion time
            if completion_times:
                stats["average_completion_time"] = sum(completion_times) / len(completion_times)
            
            return stats 