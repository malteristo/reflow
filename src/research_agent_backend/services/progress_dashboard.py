"""
Progress Tracking Dashboard for Research Agent.

This module provides a unified dashboard for tracking and displaying progress
across all Research Agent operations, including model changes, re-indexing,
document ingestion, and query processing.

Implements FR-KB-005: Progress tracking and status reporting for model changes.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from threading import Lock
import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, 
    TaskID,
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    ProgressColumn
)
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.columns import Columns

from ..core.model_change_detection import ModelChangeDetector, ModelChangeEvent
from .model_change_notifications import ModelChangeNotificationService, ReindexProgress

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be tracked."""
    MODEL_CHANGE_DETECTION = "model_change_detection"
    COLLECTION_REINDEXING = "collection_reindexing"
    DOCUMENT_INGESTION = "document_ingestion"
    QUERY_PROCESSING = "query_processing"
    BATCH_PROCESSING = "batch_processing"
    SYSTEM_MAINTENANCE = "system_maintenance"


class OperationStatus(Enum):
    """Status of tracked operations."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class OperationMetrics:
    """Metrics for a tracked operation."""
    operation_id: str
    operation_type: OperationType
    status: OperationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress_percentage: float = 0.0
    current_item: str = ""
    items_processed: int = 0
    total_items: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time for the operation."""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    @property
    def estimated_remaining_time(self) -> Optional[timedelta]:
        """Estimate remaining time based on current progress."""
        if self.progress_percentage <= 0 or self.status in [OperationStatus.COMPLETED, OperationStatus.FAILED]:
            return None
        
        elapsed = self.elapsed_time.total_seconds()
        if elapsed <= 0:
            return None
        
        rate = self.progress_percentage / elapsed
        remaining_progress = 100.0 - self.progress_percentage
        remaining_seconds = remaining_progress / rate if rate > 0 else 0
        
        return timedelta(seconds=remaining_seconds)
    
    @property
    def processing_rate(self) -> float:
        """Calculate items processed per second."""
        elapsed = self.elapsed_time.total_seconds()
        if elapsed <= 0:
            return 0.0
        return self.items_processed / elapsed


@dataclass
class DashboardConfig:
    """Configuration for the progress dashboard."""
    refresh_interval: float = 1.0
    max_operations_displayed: int = 10
    show_completed_operations: bool = True
    completed_operation_timeout: int = 300  # 5 minutes
    auto_cleanup_completed: bool = True
    show_performance_metrics: bool = True
    show_system_status: bool = True


class ProgressTrackingDashboard:
    """
    Unified progress tracking dashboard for all Research Agent operations.
    
    Provides real-time monitoring, status updates, and performance metrics
    for model changes, re-indexing, document processing, and other operations.
    """
    
    def __init__(
        self,
        config: Optional[DashboardConfig] = None,
        console: Optional[Console] = None
    ):
        """
        Initialize the progress tracking dashboard.
        
        Args:
            config: Dashboard configuration settings
            console: Rich console for output (creates default if None)
        """
        self.config = config or DashboardConfig()
        self.console = console or Console()
        self.logger = logging.getLogger(__name__)
        
        # Operation tracking
        self.active_operations: Dict[str, OperationMetrics] = {}
        self.completed_operations: Dict[str, OperationMetrics] = {}
        self.operation_callbacks: Dict[str, List[Callable]] = {}
        
        # Dashboard state
        self.is_running = False
        self.live_display: Optional[Live] = None
        self._lock = Lock()
        
        # Integration with existing services
        self.model_change_detector: Optional[ModelChangeDetector] = None
        self.notification_service: Optional[ModelChangeNotificationService] = None
        
        self.logger.info("Progress tracking dashboard initialized")
    
    def start_operation(
        self,
        operation_type: OperationType,
        operation_id: Optional[str] = None,
        total_items: int = 0,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start tracking a new operation.
        
        Args:
            operation_type: Type of operation being tracked
            operation_id: Unique identifier (generated if None)
            total_items: Total number of items to process
            description: Human-readable description
            metadata: Additional operation metadata
            
        Returns:
            Operation ID for tracking
        """
        if operation_id is None:
            operation_id = f"{operation_type.value}_{int(time.time() * 1000)}"
        
        with self._lock:
            metrics = OperationMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                status=OperationStatus.INITIALIZING,
                start_time=datetime.now(),
                total_items=total_items,
                current_item=description,
                metadata=metadata or {}
            )
            
            self.active_operations[operation_id] = metrics
        
        self.logger.info(f"Started tracking operation: {operation_id} ({operation_type.value})")
        return operation_id
    
    def update_operation(
        self,
        operation_id: str,
        status: Optional[OperationStatus] = None,
        progress_percentage: Optional[float] = None,
        current_item: Optional[str] = None,
        items_processed: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update progress for an existing operation.
        
        Args:
            operation_id: Operation to update
            status: New operation status
            progress_percentage: Progress as percentage (0-100)
            current_item: Currently processing item
            items_processed: Number of items processed
            error_message: Error message if operation failed
            metadata: Additional metadata to update
            
        Returns:
            True if operation was updated, False if not found
        """
        with self._lock:
            metrics = self.active_operations.get(operation_id)
            if not metrics:
                self.logger.warning(f"Operation not found for update: {operation_id}")
                return False
            
            # Update metrics
            if status is not None:
                metrics.status = status
            if progress_percentage is not None:
                metrics.progress_percentage = max(0.0, min(100.0, progress_percentage))
            if current_item is not None:
                metrics.current_item = current_item
            if items_processed is not None:
                metrics.items_processed = items_processed
            if error_message is not None:
                metrics.error_message = error_message
            if metadata:
                metrics.metadata.update(metadata)
            
            # Handle completion
            if status in [OperationStatus.COMPLETED, OperationStatus.FAILED, OperationStatus.CANCELLED]:
                metrics.end_time = datetime.now()
                
                # Move to completed operations
                self.completed_operations[operation_id] = metrics
                del self.active_operations[operation_id]
                
                self.logger.info(f"Operation completed: {operation_id} ({status.value})")
        
        return True
    
    def complete_operation(
        self,
        operation_id: str,
        success: bool = True,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation to complete
            success: Whether operation completed successfully
            message: Completion message
            metadata: Final metadata
            
        Returns:
            True if operation was completed, False if not found
        """
        status = OperationStatus.COMPLETED if success else OperationStatus.FAILED
        return self.update_operation(
            operation_id=operation_id,
            status=status,
            progress_percentage=100.0,
            current_item=message or ("Completed successfully" if success else "Failed"),
            error_message=None if success else message,
            metadata=metadata
        )
    
    def get_operation_status(self, operation_id: str) -> Optional[OperationMetrics]:
        """Get current status of an operation."""
        with self._lock:
            return (
                self.active_operations.get(operation_id) or 
                self.completed_operations.get(operation_id)
            )
    
    def get_active_operations(self) -> List[OperationMetrics]:
        """Get all currently active operations."""
        with self._lock:
            return list(self.active_operations.values())
    
    def get_completed_operations(self, limit: Optional[int] = None) -> List[OperationMetrics]:
        """Get completed operations, optionally limited."""
        with self._lock:
            operations = list(self.completed_operations.values())
            operations.sort(key=lambda x: x.end_time or datetime.min, reverse=True)
            return operations[:limit] if limit else operations
    
    def cleanup_old_operations(self) -> int:
        """Remove old completed operations based on timeout."""
        if not self.config.auto_cleanup_completed:
            return 0
        
        cutoff_time = datetime.now() - timedelta(seconds=self.config.completed_operation_timeout)
        removed_count = 0
        
        with self._lock:
            to_remove = [
                op_id for op_id, metrics in self.completed_operations.items()
                if metrics.end_time and metrics.end_time < cutoff_time
            ]
            
            for op_id in to_remove:
                del self.completed_operations[op_id]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.debug(f"Cleaned up {removed_count} old completed operations")
        
        return removed_count
    
    def create_dashboard_layout(self) -> Layout:
        """Create the dashboard layout for live display."""
        layout = Layout()
        
        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main section
        layout["main"].split_row(
            Layout(name="operations", ratio=2),
            Layout(name="metrics", ratio=1)
        )
        
        return layout
    
    def render_header(self) -> Panel:
        """Render the dashboard header."""
        active_count = len(self.active_operations)
        completed_count = len(self.completed_operations)
        
        header_text = Text()
        header_text.append("Research Agent Progress Dashboard", style="bold blue")
        header_text.append(f" | Active: {active_count} | Completed: {completed_count}", style="dim")
        
        return Panel(
            Align.center(header_text),
            title="🔄 Progress Tracking",
            border_style="blue"
        )
    
    def render_operations_panel(self) -> Panel:
        """Render the active operations panel."""
        if not self.active_operations:
            return Panel(
                Align.center(Text("No active operations", style="dim")),
                title="Active Operations",
                border_style="green"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Operation", style="cyan", width=20)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=15)
        table.add_column("Current Item", style="dim", width=30)
        table.add_column("Rate", width=10)
        
        for metrics in list(self.active_operations.values())[:self.config.max_operations_displayed]:
            # Status with color coding
            status_style = {
                OperationStatus.INITIALIZING: "yellow",
                OperationStatus.IN_PROGRESS: "blue",
                OperationStatus.PAUSED: "orange",
                OperationStatus.COMPLETED: "green",
                OperationStatus.FAILED: "red",
                OperationStatus.CANCELLED: "dim"
            }.get(metrics.status, "white")
            
            # Progress bar
            progress_text = f"{metrics.progress_percentage:.1f}%"
            if metrics.total_items > 0:
                progress_text += f" ({metrics.items_processed}/{metrics.total_items})"
            
            # Processing rate
            rate = metrics.processing_rate
            rate_text = f"{rate:.1f}/s" if rate > 0 else "-"
            
            table.add_row(
                metrics.operation_type.value.replace("_", " ").title(),
                Text(metrics.status.value.replace("_", " ").title(), style=status_style),
                progress_text,
                metrics.current_item[:30] + "..." if len(metrics.current_item) > 30 else metrics.current_item,
                rate_text
            )
        
        return Panel(table, title="Active Operations", border_style="green")
    
    def render_metrics_panel(self) -> Panel:
        """Render the system metrics panel."""
        if not self.config.show_performance_metrics:
            return Panel("Metrics disabled", title="System Metrics")
        
        # Calculate overall metrics
        total_active = len(self.active_operations)
        total_completed = len(self.completed_operations)
        
        # Recent completion rate (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_completions = [
            op for op in self.completed_operations.values()
            if op.end_time and op.end_time > one_hour_ago
        ]
        
        metrics_table = Table(show_header=False, box=None)
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value", style="cyan")
        
        metrics_table.add_row("Active Operations", str(total_active))
        metrics_table.add_row("Completed (Total)", str(total_completed))
        metrics_table.add_row("Completed (1h)", str(len(recent_completions)))
        
        if recent_completions:
            avg_duration = sum(
                op.elapsed_time.total_seconds() for op in recent_completions
            ) / len(recent_completions)
            metrics_table.add_row("Avg Duration", f"{avg_duration:.1f}s")
        
        return Panel(metrics_table, title="System Metrics", border_style="yellow")
    
    def render_footer(self) -> Panel:
        """Render the dashboard footer."""
        footer_text = Text()
        footer_text.append(f"Last updated: {datetime.now().strftime('%H:%M:%S')}", style="dim")
        footer_text.append(" | Press Ctrl+C to exit", style="dim")
        
        return Panel(
            Align.center(footer_text),
            border_style="dim"
        )
    
    def update_dashboard_display(self, layout: Layout) -> None:
        """Update the dashboard display with current data."""
        layout["header"].update(self.render_header())
        layout["operations"].update(self.render_operations_panel())
        layout["metrics"].update(self.render_metrics_panel())
        layout["footer"].update(self.render_footer())
    
    def start_live_dashboard(self) -> None:
        """Start the live dashboard display."""
        if self.is_running:
            self.logger.warning("Dashboard is already running")
            return
        
        self.is_running = True
        layout = self.create_dashboard_layout()
        
        try:
            with Live(layout, console=self.console, refresh_per_second=1/self.config.refresh_interval) as live:
                self.live_display = live
                
                while self.is_running:
                    self.update_dashboard_display(layout)
                    
                    # Cleanup old operations
                    if self.config.auto_cleanup_completed:
                        self.cleanup_old_operations()
                    
                    time.sleep(self.config.refresh_interval)
                    
        except KeyboardInterrupt:
            self.logger.info("Dashboard stopped by user")
        finally:
            self.is_running = False
            self.live_display = None
    
    def stop_live_dashboard(self) -> None:
        """Stop the live dashboard display."""
        self.is_running = False
    
    def display_summary(self) -> None:
        """Display a static summary of current operations."""
        self.console.print(self.render_header())
        self.console.print()
        
        # Active operations
        self.console.print(self.render_operations_panel())
        self.console.print()
        
        # Recent completed operations
        if self.config.show_completed_operations:
            recent_completed = self.get_completed_operations(limit=5)
            if recent_completed:
                completed_table = Table(title="Recent Completed Operations")
                completed_table.add_column("Operation", style="cyan")
                completed_table.add_column("Status", width=12)
                completed_table.add_column("Duration", width=12)
                completed_table.add_column("Items", width=10)
                
                for metrics in recent_completed:
                    status_style = "green" if metrics.status == OperationStatus.COMPLETED else "red"
                    duration = metrics.elapsed_time.total_seconds()
                    
                    completed_table.add_row(
                        metrics.operation_type.value.replace("_", " ").title(),
                        Text(metrics.status.value.title(), style=status_style),
                        f"{duration:.1f}s",
                        str(metrics.items_processed)
                    )
                
                self.console.print(Panel(completed_table, border_style="blue"))
                self.console.print()
        
        # System metrics
        self.console.print(self.render_metrics_panel())
    
    def integrate_with_model_change_detection(self, detector: ModelChangeDetector) -> None:
        """Integrate with model change detection system."""
        self.model_change_detector = detector
        self.logger.info("Integrated with model change detection system")
    
    def integrate_with_notification_service(self, service: ModelChangeNotificationService) -> None:
        """Integrate with model change notification service."""
        self.notification_service = service
        self.logger.info("Integrated with model change notification service")
    
    def track_reindexing_operation(
        self,
        collections: List[str],
        reindex_callback: Callable
    ) -> str:
        """
        Track a re-indexing operation with integrated progress reporting.
        
        Args:
            collections: List of collections to re-index
            reindex_callback: Callback function for re-indexing
            
        Returns:
            Operation ID for tracking
        """
        operation_id = self.start_operation(
            operation_type=OperationType.COLLECTION_REINDEXING,
            total_items=len(collections),
            description=f"Re-indexing {len(collections)} collection(s)",
            metadata={"collections": collections}
        )
        
        def enhanced_callback(collection_name: str, progress_callback: Callable) -> Dict[str, Any]:
            """Enhanced callback with dashboard integration."""
            
            def dashboard_progress_callback(percent: float, operation: str) -> None:
                """Update dashboard with collection progress."""
                self.update_operation(
                    operation_id=operation_id,
                    status=OperationStatus.IN_PROGRESS,
                    progress_percentage=percent,
                    current_item=f"{collection_name}: {operation}"
                )
                
                # Call original callback
                progress_callback(percent, operation)
            
            try:
                result = reindex_callback(collection_name, dashboard_progress_callback)
                return result
            except Exception as e:
                self.update_operation(
                    operation_id=operation_id,
                    status=OperationStatus.FAILED,
                    error_message=str(e)
                )
                raise
        
        return operation_id


# Global dashboard instance
_dashboard_instance: Optional[ProgressTrackingDashboard] = None


def get_dashboard() -> ProgressTrackingDashboard:
    """Get the global dashboard instance."""
    global _dashboard_instance
    if _dashboard_instance is None:
        _dashboard_instance = ProgressTrackingDashboard()
    return _dashboard_instance


def create_dashboard(config: Optional[DashboardConfig] = None) -> ProgressTrackingDashboard:
    """Create a new dashboard instance."""
    return ProgressTrackingDashboard(config=config) 