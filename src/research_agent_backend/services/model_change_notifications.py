"""
Model Change Notification Service for Research Agent.

This module provides user interaction patterns for model change detection,
including notifications, approval workflows, and progress reporting for
re-indexing operations.

Implements FR-KB-005: Model change detection and re-indexing workflows.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Literal, Tuple
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, 
    TaskID,
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn
)
from rich.table import Table
from rich.text import Text
from rich.prompt import Confirm, Prompt

from ..core.model_change_detection import ModelChangeEvent, ModelFingerprint
from ..models.metadata_schema.collection_metadata import CollectionMetadata

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class UserDecision(Enum):
    """User decision options for model change approval."""
    APPROVE = "approve"
    DENY = "deny"
    DEFER = "defer"
    CANCEL = "cancel"


@dataclass
class ModelChangeNotification:
    """
    Data structure for model change notifications.
    
    Contains all information needed to present a comprehensive
    notification to the user about detected model changes.
    """
    event: ModelChangeEvent
    affected_collections: List[str]
    impact_assessment: Dict[str, Any]
    notification_level: NotificationLevel
    timestamp: datetime = field(default_factory=datetime.now)
    user_decision: Optional[UserDecision] = None
    decision_timestamp: Optional[datetime] = None
    
    def get_impact_summary(self) -> str:
        """Generate a human-readable impact summary."""
        total_collections = len(self.affected_collections)
        total_documents = self.impact_assessment.get('total_documents', 0)
        estimated_time = self.impact_assessment.get('estimated_reindex_time_minutes', 0)
        
        if total_collections == 0:
            return "No existing collections affected"
        
        return (
            f"{total_collections} collection(s) affected • "
            f"{total_documents} documents • "
            f"~{estimated_time} min reindex time"
        )


@dataclass
class ReindexProgress:
    """Progress tracking for re-indexing operations."""
    collection_name: str
    total_documents: int
    processed_documents: int = 0
    current_operation: str = "Initializing..."
    start_time: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_documents == 0:
            return 100.0
        return (self.processed_documents / self.total_documents) * 100
    
    @property
    def elapsed_time_seconds(self) -> float:
        """Calculate elapsed time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def estimated_remaining_seconds(self) -> float:
        """Estimate remaining time based on current progress."""
        if self.processed_documents == 0:
            return 0.0
        
        rate = self.processed_documents / self.elapsed_time_seconds
        remaining_documents = self.total_documents - self.processed_documents
        return remaining_documents / rate if rate > 0 else 0.0


class ModelChangeNotificationService:
    """
    Service for managing model change notifications and user interactions.
    
    Provides comprehensive user interaction patterns for:
    - Model change detection notifications
    - Re-indexing approval workflows
    - Progress reporting and status updates
    - Impact assessment and recommendations
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        auto_approve_low_impact: bool = False,
        notification_threshold: NotificationLevel = NotificationLevel.INFO
    ):
        """
        Initialize the notification service.
        
        Args:
            console: Rich console for formatted output (creates default if None)
            auto_approve_low_impact: Automatically approve low-impact changes
            notification_threshold: Minimum level for showing notifications
        """
        self.console = console or Console()
        self.auto_approve_low_impact = auto_approve_low_impact
        self.notification_threshold = notification_threshold
        self.logger = logging.getLogger(__name__)
        
        # Track notification history
        self.notification_history: List[ModelChangeNotification] = []
        self.active_reindex_operations: Dict[str, ReindexProgress] = {}
    
    def assess_change_impact(
        self,
        event: ModelChangeEvent,
        collections: List[CollectionMetadata]
    ) -> Tuple[List[str], Dict[str, Any], NotificationLevel]:
        """
        Assess the impact of a model change on existing collections.
        
        Args:
            event: Model change event to assess
            collections: List of collection metadata to check
            
        Returns:
            Tuple of (affected_collections, impact_assessment, notification_level)
        """
        affected_collections = []
        total_documents = 0
        total_size_mb = 0.0
        
        # Find collections using the changed model
        for collection in collections:
            if collection.embedding_model == event.model_name:
                affected_collections.append(collection.collection_name)
                total_documents += collection.document_count
                total_size_mb += collection.total_size_bytes / (1024 * 1024)
        
        # Estimate re-indexing time (rough heuristic: 50 docs/minute)
        estimated_time_minutes = max(1, total_documents // 50)
        
        # Determine notification level based on impact
        if len(affected_collections) == 0:
            level = NotificationLevel.INFO
        elif total_documents < 100:
            level = NotificationLevel.WARNING
        else:
            level = NotificationLevel.CRITICAL
        
        impact_assessment = {
            'total_documents': total_documents,
            'total_size_mb': round(total_size_mb, 1),
            'estimated_reindex_time_minutes': estimated_time_minutes,
            'change_type': event.change_type,
            'requires_reindexing': event.requires_reindexing,
            'model_name': event.model_name,
            'old_version': event.old_fingerprint.version if event.old_fingerprint else None,
            'new_version': event.new_fingerprint.version
        }
        
        return affected_collections, impact_assessment, level
    
    def create_notification(
        self,
        event: ModelChangeEvent,
        collections: List[CollectionMetadata]
    ) -> ModelChangeNotification:
        """
        Create a structured notification for a model change event.
        
        Args:
            event: Model change event
            collections: Available collections for impact assessment
            
        Returns:
            ModelChangeNotification with complete impact analysis
        """
        affected_collections, impact_assessment, level = self.assess_change_impact(
            event, collections
        )
        
        notification = ModelChangeNotification(
            event=event,
            affected_collections=affected_collections,
            impact_assessment=impact_assessment,
            notification_level=level
        )
        
        # Store in history
        self.notification_history.append(notification)
        
        self.logger.info(
            f"Created {level.value} notification for model '{event.model_name}' "
            f"affecting {len(affected_collections)} collections"
        )
        
        return notification
    
    def display_change_notification(self, notification: ModelChangeNotification) -> None:
        """
        Display a formatted model change notification to the user.
        
        Args:
            notification: Notification to display
        """
        event = notification.event
        
        # Choose styling based on notification level
        if notification.notification_level == NotificationLevel.CRITICAL:
            panel_style = "red"
            title_emoji = "🚨"
        elif notification.notification_level == NotificationLevel.WARNING:
            panel_style = "yellow"
            title_emoji = "⚠️"
        else:
            panel_style = "blue"
            title_emoji = "ℹ️"
        
        # Create main panel content
        title = f"{title_emoji} Model Change Detected"
        
        content = f"""[bold]Model:[/bold] {event.model_name}
[bold]Change Type:[/bold] {event.change_type.replace('_', ' ').title()}
[bold]Impact:[/bold] {notification.get_impact_summary()}
[bold]Re-indexing Required:[/bold] {'Yes' if event.requires_reindexing else 'No'}

"""
        
        # Add version information if available
        if event.old_fingerprint:
            content += f"[bold]Version Change:[/bold] {event.old_fingerprint.version} → {event.new_fingerprint.version}\n"
        else:
            content += f"[bold]New Model Version:[/bold] {event.new_fingerprint.version}\n"
        
        # Add affected collections details
        if notification.affected_collections:
            content += f"\n[bold]Affected Collections:[/bold]\n"
            for collection in notification.affected_collections:
                content += f"  • {collection}\n"
        else:
            content += "\n[dim]No existing collections affected[/dim]\n"
        
        # Display the panel
        panel = Panel(
            content.strip(),
            title=title,
            style=panel_style,
            expand=False
        )
        
        self.console.print("\n")
        self.console.print(panel)
    
    def prompt_user_decision(
        self,
        notification: ModelChangeNotification,
        allow_defer: bool = True
    ) -> UserDecision:
        """
        Prompt user for decision on model change handling.
        
        Args:
            notification: Notification containing change details
            allow_defer: Whether to allow deferring the decision
            
        Returns:
            User's decision on how to handle the change
        """
        event = notification.event
        
        # Auto-approve low-impact changes if configured
        if (self.auto_approve_low_impact and 
            notification.notification_level == NotificationLevel.INFO and
            len(notification.affected_collections) == 0):
            
            self.console.print("[dim]Auto-approving low-impact change...[/dim]")
            return UserDecision.APPROVE
        
        # Show impact details
        if notification.affected_collections and event.requires_reindexing:
            total_docs = notification.impact_assessment['total_documents']
            est_time = notification.impact_assessment['estimated_reindex_time_minutes']
            
            self.console.print(f"\n[yellow]Re-indexing Impact:[/yellow]")
            self.console.print(f"  • {total_docs} documents will be re-processed")
            self.console.print(f"  • Estimated time: ~{est_time} minutes")
            self.console.print(f"  • Collections will be temporarily unavailable")
            
        # Build decision options
        options = []
        if event.requires_reindexing and notification.affected_collections:
            options.append("approve - Proceed with re-indexing")
            options.append("deny - Keep current model (may cause inconsistencies)")
        else:
            options.append("approve - Accept model change")
            options.append("deny - Reject model change")
        
        if allow_defer:
            options.append("defer - Decide later")
        
        options.append("cancel - Cancel operation")
        
        # Show options
        self.console.print(f"\n[bold]Available Actions:[/bold]")
        for option in options:
            self.console.print(f"  {option}")
        
        # Get user input
        while True:
            try:
                choice = Prompt.ask(
                    "\nHow would you like to proceed?",
                    choices=["approve", "deny", "defer", "cancel"] if allow_defer 
                           else ["approve", "deny", "cancel"],
                    default="approve" if notification.notification_level == NotificationLevel.INFO else None
                )
                
                decision = UserDecision(choice)
                
                # Update notification with decision
                notification.user_decision = decision
                notification.decision_timestamp = datetime.now()
                
                self.logger.info(f"User decision for model '{event.model_name}': {decision.value}")
                return decision
                
            except KeyboardInterrupt:
                self.console.print("\n[red]Operation cancelled by user[/red]")
                return UserDecision.CANCEL
            except Exception as e:
                self.console.print(f"[red]Invalid choice: {e}[/red]")
                continue
    
    def execute_reindexing_with_progress(
        self,
        collections: List[str],
        reindex_callback: Callable[[str, Callable[[float, str], None]], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute re-indexing operations with rich progress reporting.
        
        Args:
            collections: List of collection names to re-index
            reindex_callback: Function that performs re-indexing for a collection
                             Signature: (collection_name, progress_callback) -> result_dict
            
        Returns:
            Dictionary with overall re-indexing results
        """
        if not collections:
            return {"success": True, "message": "No collections to re-index"}
        
        start_time = datetime.now()
        results = {}
        errors = []
        
        # Create progress display
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        ) as progress:
            
            # Overall progress
            overall_task = progress.add_task(
                f"Re-indexing {len(collections)} collection(s)...",
                total=len(collections)
            )
            
            # Process each collection
            for i, collection_name in enumerate(collections):
                collection_task = progress.add_task(
                    f"Collection: {collection_name}",
                    total=100  # Progress as percentage
                )
                
                def update_progress(percent: float, operation: str) -> None:
                    """Update progress for current collection."""
                    progress.update(
                        collection_task,
                        completed=percent,
                        description=f"Collection: {collection_name} - {operation}"
                    )
                
                try:
                    # Execute re-indexing for this collection
                    result = reindex_callback(collection_name, update_progress)
                    results[collection_name] = result
                    
                    # Mark collection complete
                    progress.update(collection_task, completed=100)
                    progress.update(overall_task, advance=1)
                    
                except Exception as e:
                    error_msg = f"Failed to re-index {collection_name}: {str(e)}"
                    errors.append(error_msg)
                    results[collection_name] = {"success": False, "error": str(e)}
                    self.logger.error(error_msg)
                    
                    # Mark collection as failed
                    progress.update(
                        collection_task,
                        description=f"Collection: {collection_name} - [red]FAILED[/red]"
                    )
                    progress.update(overall_task, advance=1)
        
        # Compile overall results
        elapsed_time = (datetime.now() - start_time).total_seconds()
        successful_collections = [
            name for name, result in results.items() 
            if result.get('success', False)
        ]
        
        overall_result = {
            "success": len(errors) == 0,
            "total_collections": len(collections),
            "successful_collections": len(successful_collections),
            "failed_collections": len(errors),
            "elapsed_time_seconds": elapsed_time,
            "collection_results": results,
            "errors": errors
        }
        
        # Display summary
        self._display_reindex_summary(overall_result)
        
        return overall_result
    
    def _display_reindex_summary(self, results: Dict[str, Any]) -> None:
        """Display a summary of re-indexing results."""
        if results["success"]:
            style = "green"
            emoji = "✅"
            title = "Re-indexing Completed Successfully"
        else:
            style = "red"
            emoji = "❌"
            title = "Re-indexing Completed with Errors"
        
        # Create summary table
        table = Table(title=f"{emoji} {title}", style=style)
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        
        table.add_row("Total Collections", str(results["total_collections"]))
        table.add_row("Successful", str(results["successful_collections"]))
        table.add_row("Failed", str(results["failed_collections"]))
        table.add_row("Time Elapsed", f"{results['elapsed_time_seconds']:.1f} seconds")
        
        self.console.print("\n")
        self.console.print(table)
        
        # Show errors if any
        if results["errors"]:
            self.console.print("\n[red]Errors encountered:[/red]")
            for error in results["errors"]:
                self.console.print(f"  • {error}")
    
    def handle_model_change_workflow(
        self,
        event: ModelChangeEvent,
        collections: List[CollectionMetadata],
        reindex_callback: Optional[Callable[[str, Callable[[float, str], None]], Dict[str, Any]]] = None
    ) -> UserDecision:
        """
        Complete workflow for handling model change detection.
        
        Args:
            event: Model change event to handle
            collections: Available collections for impact assessment
            reindex_callback: Optional callback for performing re-indexing
            
        Returns:
            Final user decision after workflow completion
        """
        # Create and display notification
        notification = self.create_notification(event, collections)
        
        # Skip notification if below threshold
        if notification.notification_level.value < self.notification_threshold.value:
            self.logger.debug(f"Skipping notification below threshold: {notification.notification_level}")
            return UserDecision.APPROVE
        
        # Display the notification
        self.display_change_notification(notification)
        
        # Get user decision
        decision = self.prompt_user_decision(notification)
        
        # Execute re-indexing if approved and needed
        if (decision == UserDecision.APPROVE and 
            event.requires_reindexing and 
            notification.affected_collections and
            reindex_callback):
            
            self.console.print("\n[blue]Starting re-indexing operations...[/blue]")
            
            reindex_result = self.execute_reindexing_with_progress(
                notification.affected_collections,
                reindex_callback
            )
            
            if not reindex_result["success"]:
                self.console.print(
                    "\n[yellow]Warning:[/yellow] Some collections failed to re-index. "
                    "You may need to re-run the operation manually."
                )
        
        elif decision == UserDecision.DENY and event.requires_reindexing:
            self.console.print(
                "\n[yellow]Warning:[/yellow] Model change rejected. "
                "Existing embeddings may be inconsistent with the current model."
            )
        
        return decision
    
    def list_notification_history(self, limit: int = 10) -> None:
        """
        Display recent notification history.
        
        Args:
            limit: Maximum number of notifications to show
        """
        if not self.notification_history:
            self.console.print("[dim]No model change notifications found[/dim]")
            return
        
        table = Table(title="Recent Model Change Notifications")
        table.add_column("Timestamp", style="dim")
        table.add_column("Model", style="bold")
        table.add_column("Change Type")
        table.add_column("Level", justify="center")
        table.add_column("Collections Affected", justify="right")
        table.add_column("Decision", justify="center")
        
        # Show most recent notifications first
        recent_notifications = sorted(
            self.notification_history,
            key=lambda n: n.timestamp,
            reverse=True
        )[:limit]
        
        for notification in recent_notifications:
            level_color = {
                NotificationLevel.INFO: "blue",
                NotificationLevel.WARNING: "yellow",
                NotificationLevel.CRITICAL: "red"
            }[notification.notification_level]
            
            decision_text = notification.user_decision.value if notification.user_decision else "pending"
            decision_color = {
                "approve": "green",
                "deny": "red",
                "defer": "yellow",
                "cancel": "dim",
                "pending": "dim"
            }.get(decision_text, "white")
            
            table.add_row(
                notification.timestamp.strftime("%Y-%m-%d %H:%M"),
                notification.event.model_name,
                notification.event.change_type.replace("_", " ").title(),
                f"[{level_color}]{notification.notification_level.value.upper()}[/{level_color}]",
                str(len(notification.affected_collections)),
                f"[{decision_color}]{decision_text.upper()}[/{decision_color}]"
            )
        
        self.console.print(table) 