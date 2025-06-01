"""
Asynchronous Progress Reporter for Structured Feedback and Progress Reporting.

This module provides non-blocking asynchronous progress reporting capabilities
with proper event loop handling, concurrent operation tracking, and error handling.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
import weakref
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class AsyncProgressEvent:
    """Asynchronous progress event data structure."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    operation_id: str = ""
    event_type: ProgressEventType = ProgressEventType.PROGRESS
    progress_percentage: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class AsyncOperationTracker:
    """Tracks an asynchronous operation's progress."""
    operation_id: str
    start_time: float = field(default_factory=time.time)
    current_progress: float = 0.0
    status: str = "running"
    last_update: float = field(default_factory=time.time)
    task: Optional[asyncio.Task] = None
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_cancelled: bool = False
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncProgressEventConsumer:
    """Handles asynchronous consumption of progress events."""
    
    def __init__(self, operation_id: str, consumer_id: Optional[str] = None):
        """Initialize the async progress event consumer."""
        self.operation_id = operation_id
        self.consumer_id = consumer_id or str(uuid4())
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.is_active = True
        self.consumed_events: List[AsyncProgressEvent] = []
        self.consumer_task: Optional[asyncio.Task] = None
        logger.debug(f"AsyncProgressEventConsumer initialized for operation {operation_id}")
    
    async def start_consuming(self, 
                            callback: Optional[Callable[[AsyncProgressEvent], None]] = None) -> None:
        """Start consuming progress events asynchronously."""
        if self.consumer_task and not self.consumer_task.done():
            return  # Already consuming
        
        self.consumer_task = asyncio.create_task(self._consume_events(callback))
        logger.info(f"Started consuming events for operation {self.operation_id}")
    
    async def _consume_events(self, 
                            callback: Optional[Callable[[AsyncProgressEvent], None]]) -> None:
        """Internal method to consume events from the queue."""
        try:
            while self.is_active:
                try:
                    # Wait for event with timeout to allow graceful shutdown
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                    
                    # Process the event
                    self.consumed_events.append(event)
                    
                    if callback:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(event)
                            else:
                                callback(event)
                        except Exception as e:
                            logger.error(f"Error in event callback: {e}")
                    
                    # Mark the task as done
                    self.event_queue.task_done()
                    
                    # Log event consumption
                    logger.debug(f"Consumed event {event.event_id} for operation {self.operation_id}")
                    
                except asyncio.TimeoutError:
                    # No event received within timeout, continue loop
                    continue
                
        except asyncio.CancelledError:
            logger.info(f"Event consumption cancelled for operation {self.operation_id}")
        except Exception as e:
            logger.error(f"Error in event consumption: {e}")
    
    async def add_event(self, event: AsyncProgressEvent) -> None:
        """Add an event to the consumer's queue."""
        if self.is_active:
            await self.event_queue.put(event)
    
    async def stop_consuming(self) -> None:
        """Stop consuming events."""
        self.is_active = False
        
        if self.consumer_task and not self.consumer_task.done():
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped consuming events for operation {self.operation_id}")
    
    def get_consumed_events(self) -> List[AsyncProgressEvent]:
        """Get all consumed events."""
        return self.consumed_events.copy()


class AsyncProgressReporter:
    """Provides asynchronous progress reporting capabilities with event loop handling."""
    
    def __init__(self, max_concurrent_operations: int = 100):
        """Initialize the asynchronous progress reporter."""
        self.max_concurrent_operations = max_concurrent_operations
        self.active_operations: Dict[str, AsyncOperationTracker] = {}
        self.event_consumers: Dict[str, List[AsyncProgressEventConsumer]] = {}
        self.global_callbacks: List[Callable[[AsyncProgressEvent], None]] = []
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_shutdown = False
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="async_progress")
        logger.info(f"AsyncProgressReporter initialized with max {max_concurrent_operations} operations")
    
    async def initialize_operation(self, 
                                 operation_id: str,
                                 metadata: Optional[Dict[str, Any]] = None) -> AsyncOperationTracker:
        """Initialize a new asynchronous operation for tracking."""
        if len(self.active_operations) >= self.max_concurrent_operations:
            raise RuntimeError(f"Maximum concurrent operations limit reached ({self.max_concurrent_operations})")
        
        if operation_id in self.active_operations:
            logger.warning(f"Operation {operation_id} already exists, returning existing tracker")
            return self.active_operations[operation_id]
        
        tracker = AsyncOperationTracker(
            operation_id=operation_id,
            metadata=metadata or {}
        )
        
        self.active_operations[operation_id] = tracker
        self.event_consumers[operation_id] = []
        
        # Create and emit started event
        start_event = AsyncProgressEvent(
            operation_id=operation_id,
            event_type=ProgressEventType.STARTED,
            progress_percentage=0.0,
            message="Operation started",
            metadata=metadata or {}
        )
        
        await self._emit_event(start_event)
        logger.info(f"Initialized async operation {operation_id}")
        
        return tracker
    
    async def report_progress(self,
                            operation_id: str,
                            progress: float,
                            message: str = "",
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Report progress for an async operation."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        tracker = self.active_operations[operation_id]
        
        if tracker.is_cancelled:
            logger.warning(f"Attempted to report progress for cancelled operation {operation_id}")
            return
        
        # Update tracker
        tracker.current_progress = max(0.0, min(100.0, progress))
        tracker.last_update = time.time()
        tracker.metadata.update(metadata or {})
        
        # Create progress event
        progress_event = AsyncProgressEvent(
            operation_id=operation_id,
            event_type=ProgressEventType.PROGRESS,
            progress_percentage=tracker.current_progress,
            message=message,
            metadata=metadata or {},
            duration=tracker.last_update - tracker.start_time
        )
        
        await self._emit_event(progress_event)
        logger.debug(f"Reported progress {progress:.1f}% for operation {operation_id}")
    
    async def complete_operation(self,
                               operation_id: str,
                               message: str = "Operation completed",
                               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark an operation as completed."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        tracker = self.active_operations[operation_id]
        tracker.status = "completed"
        tracker.current_progress = 100.0
        
        # Create completion event
        completion_event = AsyncProgressEvent(
            operation_id=operation_id,
            event_type=ProgressEventType.COMPLETED,
            progress_percentage=100.0,
            message=message,
            metadata=metadata or {},
            duration=time.time() - tracker.start_time
        )
        
        await self._emit_event(completion_event)
        
        # Clean up operation
        await self._cleanup_operation(operation_id)
        logger.info(f"Completed operation {operation_id}")
    
    async def report_error(self,
                         operation_id: str,
                         error: Exception,
                         message: str = "",
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Report an error for an async operation."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        tracker = self.active_operations[operation_id]
        tracker.status = "error"
        tracker.error = error
        
        # Create error event
        error_event = AsyncProgressEvent(
            operation_id=operation_id,
            event_type=ProgressEventType.ERROR,
            progress_percentage=tracker.current_progress,
            message=message or str(error),
            metadata=metadata or {},
            duration=time.time() - tracker.start_time,
            error_details={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_args": getattr(error, 'args', [])
            }
        )
        
        await self._emit_event(error_event)
        
        # Clean up operation
        await self._cleanup_operation(operation_id)
        logger.error(f"Error in operation {operation_id}: {error}")
    
    async def cancel_operation(self,
                             operation_id: str,
                             message: str = "Operation cancelled") -> None:
        """Cancel an async operation."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        tracker = self.active_operations[operation_id]
        tracker.status = "cancelled"
        tracker.is_cancelled = True
        
        # Cancel the task if it exists
        if tracker.task and not tracker.task.done():
            tracker.task.cancel()
        
        # Create cancellation event
        cancel_event = AsyncProgressEvent(
            operation_id=operation_id,
            event_type=ProgressEventType.CANCELLED,
            progress_percentage=tracker.current_progress,
            message=message,
            duration=time.time() - tracker.start_time
        )
        
        await self._emit_event(cancel_event)
        
        # Clean up operation
        await self._cleanup_operation(operation_id)
        logger.info(f"Cancelled operation {operation_id}")
    
    async def create_consumer(self, operation_id: str) -> AsyncProgressEventConsumer:
        """Create a consumer for progress events from a specific operation."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")
        
        consumer = AsyncProgressEventConsumer(operation_id)
        
        if operation_id not in self.event_consumers:
            self.event_consumers[operation_id] = []
        
        self.event_consumers[operation_id].append(consumer)
        logger.debug(f"Created consumer {consumer.consumer_id} for operation {operation_id}")
        
        return consumer
    
    async def _emit_event(self, event: AsyncProgressEvent) -> None:
        """Emit an event to all consumers and global callbacks."""
        # Send to operation-specific consumers
        if event.operation_id in self.event_consumers:
            for consumer in self.event_consumers[event.operation_id]:
                await consumer.add_event(event)
        
        # Send to global callbacks
        for callback in self.global_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    # Run sync callback in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(self.executor, callback, event)
            except Exception as e:
                logger.error(f"Error in global callback: {e}")
    
    async def _cleanup_operation(self, operation_id: str) -> None:
        """Clean up resources for a completed operation."""
        # Stop all consumers for this operation
        if operation_id in self.event_consumers:
            for consumer in self.event_consumers[operation_id]:
                await consumer.stop_consuming()
            del self.event_consumers[operation_id]
        
        # Remove operation tracker
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
    
    def add_global_callback(self, callback: Callable[[AsyncProgressEvent], None]) -> None:
        """Add a global callback for all progress events."""
        self.global_callbacks.append(callback)
        logger.debug(f"Added global callback: {callback}")
    
    def remove_global_callback(self, callback: Callable[[AsyncProgressEvent], None]) -> None:
        """Remove a global callback."""
        if callback in self.global_callbacks:
            self.global_callbacks.remove(callback)
            logger.debug(f"Removed global callback: {callback}")
    
    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an operation."""
        if operation_id not in self.active_operations:
            return None
        
        tracker = self.active_operations[operation_id]
        return {
            "operation_id": operation_id,
            "status": tracker.status,
            "progress": tracker.current_progress,
            "start_time": tracker.start_time,
            "last_update": tracker.last_update,
            "duration": tracker.last_update - tracker.start_time,
            "is_cancelled": tracker.is_cancelled,
            "error": str(tracker.error) if tracker.error else None,
            "metadata": tracker.metadata
        }
    
    async def get_all_operations(self) -> List[Dict[str, Any]]:
        """Get status of all active operations."""
        statuses = []
        for operation_id in self.active_operations:
            status = await self.get_operation_status(operation_id)
            if status:
                statuses.append(status)
        return statuses
    
    async def shutdown(self) -> None:
        """Shutdown the async progress reporter and clean up resources."""
        self.is_shutdown = True
        
        # Cancel all active operations
        for operation_id in list(self.active_operations.keys()):
            try:
                await self.cancel_operation(operation_id, "System shutdown")
            except Exception as e:
                logger.error(f"Error cancelling operation {operation_id} during shutdown: {e}")
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AsyncProgressReporter shutdown complete")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the async progress reporter."""
        return {
            "active_operations": len(self.active_operations),
            "max_concurrent_operations": self.max_concurrent_operations,
            "total_consumers": sum(len(consumers) for consumers in self.event_consumers.values()),
            "global_callbacks": len(self.global_callbacks),
            "background_tasks": len(self.background_tasks),
            "is_shutdown": self.is_shutdown
        } 