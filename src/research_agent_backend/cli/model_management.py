"""
Model Management CLI Commands for Research Agent.

This module provides CLI commands for managing embedding model changes,
including status checking, manual change detection, and re-indexing workflows.

Implements FR-KB-005: Model change detection and re-indexing CLI interface.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from ..core.model_change_detection import ModelChangeDetector, ModelChangeEvent, ModelFingerprint
from ..services.model_change_notifications import ModelChangeNotificationService, UserDecision
from ..core.vector_store import ChromaDBManager
from ..core.local_embedding_service import LocalEmbeddingService
from ..core.api_embedding_service import APIEmbeddingService
from ..core.document_insertion.manager import DocumentInsertionManager
from ..utils.config import ConfigManager
from ..services.backup_recovery_service import BackupRecoveryService
from ..services.migration_validation_service import (
    MigrationValidationService,
    ValidationConfig,
    ValidationStatus,
    ValidationReport
)

# Create CLI app
model_app = typer.Typer(
    name="model",
    help="Model change detection and management commands",
    no_args_is_help=True
)

console = Console()
logger = logging.getLogger(__name__)


def create_embedding_service():
    """Create embedding service based on current configuration."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Check if using local or API embedding service
        if hasattr(config.embedding_model, 'provider') and config.embedding_model.provider:
            # API service
            from ..core.api_embedding_service import APIConfiguration
            api_config = APIConfiguration(
                provider=config.embedding_model.provider,
                api_key=config.embedding_model.api_key or "",
                model_name=config.embedding_model.model_name_or_path
            )
            return APIEmbeddingService(api_config)
        else:
            # Local service
            return LocalEmbeddingService(
                model_name=config.embedding_model.model_name_or_path
            )
    except Exception as e:
        logger.error(f"Failed to create embedding service: {e}")
        return None


def create_chroma_manager():
    """Create ChromaDB manager."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        return ChromaDBManager(
            persist_directory=config.vector_store.persist_directory,
            collection_metadata={}
        )
    except Exception as e:
        logger.error(f"Failed to create ChromaDB manager: {e}")
        return None


def create_document_manager():
    """Create document insertion manager for re-indexing operations."""
    try:
        chroma_manager = create_chroma_manager()
        embedding_service = create_embedding_service()
        
        if not chroma_manager or not embedding_service:
            return None
            
        return DocumentInsertionManager(
            vector_store=chroma_manager,
            embedding_service=embedding_service
        )
    except Exception as e:
        logger.error(f"Failed to create document manager: {e}")
        return None


@model_app.command("status")
def model_status() -> None:
    """
    Display current model status and change detection information.
    
    Shows the current embedding model configuration, registered models,
    and any detected changes that may require user attention.
    
    Example:
        research-agent model status
    """
    try:
        # Get current embedding service and model info
        embedding_service = create_embedding_service()
        if not embedding_service:
            console.print("[red]Error:[/red] Could not access embedding service")
            raise typer.Exit(1)
        
        # Get model fingerprint
        try:
            current_fingerprint = embedding_service.generate_model_fingerprint()
            model_info = embedding_service.get_model_info()
        except Exception as e:
            console.print(f"[red]Error:[/red] Could not get model information: {e}")
            raise typer.Exit(1)
        
        # Get detector status
        detector = ModelChangeDetector()
        registered_fingerprint = detector.get_model_fingerprint(current_fingerprint.model_name)
        change_detected = detector.detect_change(current_fingerprint)
        
        # Create status table
        table = Table(title="🤖 Embedding Model Status", style="blue")
        table.add_column("Property", style="bold")
        table.add_column("Value")
        
        table.add_row("Model Name", current_fingerprint.model_name)
        table.add_row("Model Type", current_fingerprint.model_type.title())
        table.add_row("Version", current_fingerprint.version)
        table.add_row("Dimension", str(model_info.get('dimension', 'Unknown')))
        
        if current_fingerprint.model_type == "local":
            table.add_row("Max Sequence Length", str(model_info.get('max_seq_length', 'Unknown')))
            table.add_row("Library", model_info.get('library', 'Unknown'))
        else:
            table.add_row("Provider", model_info.get('provider', 'Unknown'))
        
        # Change detection status
        if registered_fingerprint:
            table.add_row("Registered", "✅ Yes")
            table.add_row("Change Detected", "⚠️ Yes" if change_detected else "✅ No")
            
            if change_detected:
                table.add_row("Action Required", "[yellow]Re-indexing may be needed[/yellow]")
        else:
            table.add_row("Registered", "❌ No")
            table.add_row("Status", "[yellow]New model detected[/yellow]")
        
        console.print("\n")
        console.print(table)
        
        # Show fingerprint details
        console.print(f"\n[dim]Model Fingerprint: {current_fingerprint.checksum}[/dim]")
        console.print(f"[dim]Created: {current_fingerprint.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
        
        # Show registered models summary
        all_models = detector.list_models()
        if all_models:
            console.print(f"\n[bold]Registered Models:[/bold] {len(all_models)}")
            for model_name in all_models[:5]:  # Show first 5
                console.print(f"  • {model_name}")
            if len(all_models) > 5:
                console.print(f"  ... and {len(all_models) - 5} more")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Model status command failed: {e}")
        raise typer.Exit(1)


@model_app.command("check-changes")
def check_model_changes(
    auto_register: bool = typer.Option(
        False,
        "--auto-register",
        help="Automatically register detected changes"
    ),
    show_collections: bool = typer.Option(
        True,
        "--show-collections/--no-collections",
        help="Show affected collections"
    )
) -> None:
    """
    Check for model changes and display impact assessment.
    
    Analyzes the current embedding model against registered state
    and shows which collections would be affected by changes.
    
    Args:
        auto_register: Automatically register detected changes without prompting
        show_collections: Whether to show affected collections
    
    Example:
        research-agent model check-changes
        research-agent model check-changes --auto-register
    """
    try:
        # Get current model state
        embedding_service = create_embedding_service()
        if not embedding_service:
            console.print("[red]Error:[/red] Could not access embedding service")
            raise typer.Exit(1)
        
        current_fingerprint = embedding_service.generate_model_fingerprint()
        detector = ModelChangeDetector()
        
        # Check for changes
        change_detected = detector.detect_change(current_fingerprint)
        
        if not change_detected:
            console.print("✅ [green]No model changes detected[/green]")
            console.print(f"Current model '{current_fingerprint.model_name}' is up to date")
            return
        
        # Get existing fingerprint for comparison
        old_fingerprint = detector.get_model_fingerprint(current_fingerprint.model_name)
        
        # Create change event
        if old_fingerprint is None:
            change_type = "new_model"
        elif old_fingerprint.version != current_fingerprint.version:
            change_type = "version_update"
        elif old_fingerprint.checksum != current_fingerprint.checksum:
            change_type = "checksum_change"
        else:
            change_type = "config_change"
        
        event = ModelChangeEvent(
            model_name=current_fingerprint.model_name,
            change_type=change_type,
            old_fingerprint=old_fingerprint,
            new_fingerprint=current_fingerprint,
            requires_reindexing=change_type in ("new_model", "version_update", "checksum_change")
        )
        
        # Get collection information for impact assessment
        collections = []
        if show_collections:
            try:
                chroma_manager = create_chroma_manager()
                if chroma_manager:
                    collection_infos = chroma_manager.list_collections()
                    for col_info in collection_infos:
                        try:
                            stats = chroma_manager.get_collection_stats(col_info.name)
                            # Create minimal collection metadata for impact assessment
                            collections.append(type('CollectionMetadata', (), {
                                'collection_name': col_info.name,
                                'embedding_model': current_fingerprint.model_name,  # Assume same model
                                'document_count': stats.document_count,
                                'total_size_bytes': stats.storage_size_bytes
                            })())
                        except Exception as e:
                            logger.warning(f"Could not get stats for collection {col_info.name}: {e}")
            except Exception as e:
                logger.warning(f"Could not access collections: {e}")
        
        # Create notification service and show the change
        notification_service = ModelChangeNotificationService(console=console)
        
        if auto_register:
            # Auto-register the change
            detector.register_model(current_fingerprint)
            console.print("✅ [green]Model change registered automatically[/green]")
            
            # Show impact assessment
            notification = notification_service.create_notification(event, collections)
            notification_service.display_change_notification(notification)
            
        else:
            # Interactive workflow
            decision = notification_service.handle_model_change_workflow(
                event, 
                collections,
                reindex_callback=None  # Just assessment, no actual re-indexing
            )
            
            if decision == UserDecision.APPROVE:
                detector.register_model(current_fingerprint)
                console.print("✅ [green]Model change registered[/green]")
            elif decision == UserDecision.DENY:
                console.print("❌ [yellow]Model change rejected[/yellow]")
            elif decision == UserDecision.DEFER:
                console.print("⏳ [blue]Decision deferred[/blue]")
            else:
                console.print("🚫 [dim]Operation cancelled[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Check changes command failed: {e}")
        raise typer.Exit(1)


@model_app.command("reindex")
def reindex_collections(
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to re-index (default: all)"
    ),
    parallel: bool = typer.Option(
        True,
        "--parallel/--sequential",
        help="Use parallel processing for embedding generation"
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of worker threads for parallel processing (default: auto)"
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        "-b",
        help="Batch size for processing documents (default: 50)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt"
    )
) -> None:
    """
    Re-index collections with new embedding model.
    
    This command rebuilds vector indices for the specified collections using the
    current embedding model. Supports parallel processing for improved performance.
    
    Examples:
        research-agent model reindex
        research-agent model reindex --collections=docs,research --parallel
        research-agent model reindex --workers=8 --batch-size=100 --force
    """
    try:
        console = Console()
        
        # Initialize services
        chroma_manager = create_chroma_manager()
        document_manager = create_document_manager()
        notification_service = ModelChangeNotificationService()
        
        # Determine collections to re-index
        if collections:
            collection_list = [c.strip() for c in collections.split(",")]
            # Validate collections exist
            existing_collections = [col.name for col in chroma_manager.list_collections()]
            invalid_collections = [c for c in collection_list if c not in existing_collections]
            if invalid_collections:
                console.print(f"[red]Error:[/red] Collections not found: {', '.join(invalid_collections)}")
                raise typer.Exit(1)
        else:
            collection_list = [col.name for col in chroma_manager.list_collections()]
        
        if not collection_list:
            console.print("[yellow]No collections found to re-index[/yellow]")
            return
        
        # Display re-indexing plan
        console.print("\n[blue]Re-indexing Plan[/blue]")
        console.print("=" * 50)
        console.print(f"Collections: [cyan]{', '.join(collection_list)}[/cyan]")
        console.print(f"Processing mode: [cyan]{'Parallel' if parallel else 'Sequential'}[/cyan]")
        if parallel and workers:
            console.print(f"Worker threads: [cyan]{workers}[/cyan]")
        if batch_size:
            console.print(f"Batch size: [cyan]{batch_size}[/cyan]")
        
        # Get collection statistics
        total_docs = 0
        collection_stats = {}
        for col_name in collection_list:
            try:
                stats = chroma_manager.get_collection_stats(col_name)
                collection_stats[col_name] = stats.document_count
                total_docs += stats.document_count
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not get stats for {col_name}: {e}")
                collection_stats[col_name] = 0
        
        console.print(f"Total documents: [cyan]{total_docs}[/cyan]")
        
        # Confirmation prompt
        if not force:
            console.print("\n[yellow]Warning:[/yellow] This will regenerate all embeddings and rebuild indices.")
            console.print("This operation may take several minutes for large collections.")
            
            if not typer.confirm("Continue with re-indexing?"):
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        # Define re-indexing callback with enhanced options
        def reindex_callback(collection_name: str, progress_callback: Callable[[float, str], None]) -> Dict[str, Any]:
            """Enhanced re-indexing callback with parallel processing support."""
            
            def progress_wrapper(percent: float) -> None:
                """Wrapper to convert progress format."""
                operation = "Initializing..."
                if percent < 20:
                    operation = "Extracting documents..."
                elif percent < 40:
                    operation = "Recreating collection..."
                elif percent < 90:
                    operation = f"Regenerating embeddings... ({percent:.1f}%)"
                else:
                    operation = "Finalizing..."
                
                progress_callback(percent, operation)
            
            # Configure document manager for this operation
            if batch_size and hasattr(document_manager, 'batch_size'):
                original_batch_size = document_manager.batch_size
                document_manager.batch_size = batch_size
            else:
                original_batch_size = None
            
            try:
                # Call enhanced rebuild method with parallel processing options
                result = document_manager.rebuild_collection_index(
                    collection_name=collection_name,
                    progress_callback=progress_wrapper,
                    enable_parallel=parallel,
                    max_workers=workers
                )
                
                # Add collection-specific metadata
                result['collection_stats'] = collection_stats.get(collection_name, 0)
                result['processing_mode'] = 'parallel' if parallel else 'sequential'
                
                return result
                
            finally:
                # Restore original batch size
                if original_batch_size is not None:
                    document_manager.batch_size = original_batch_size
        
        # Execute re-indexing with progress reporting
        console.print("\n[blue]Starting re-indexing operations...[/blue]")
        
        start_time = datetime.now()
        result = notification_service.execute_reindexing_with_progress(
            collection_list,
            reindex_callback
        )
        
        # Enhanced results display
        console.print("\n[blue]Re-indexing Results[/blue]")
        console.print("=" * 50)
        
        if result["success"]:
            console.print(f"[green]✓[/green] Successfully re-indexed {result['successful_collections']} collection(s)")
            
            # Performance summary
            total_time = result["elapsed_time_seconds"]
            total_processed = sum(
                res.get('documents_processed', 0) 
                for res in result['collection_results'].values() 
                if isinstance(res, dict)
            )
            
            if total_processed > 0 and total_time > 0:
                docs_per_second = total_processed / total_time
                console.print(f"Performance: [cyan]{docs_per_second:.1f} documents/second[/cyan]")
            
            console.print(f"Total time: [cyan]{total_time:.2f} seconds[/cyan]")
            
            # Per-collection results
            for col_name, col_result in result['collection_results'].items():
                if isinstance(col_result, dict) and col_result.get('success'):
                    docs_processed = col_result.get('documents_processed', 0)
                    processing_time = col_result.get('processing_time', 0)
                    mode = col_result.get('processing_mode', 'unknown')
                    workers_used = col_result.get('workers_used', 1)
                    
                    console.print(f"  • [cyan]{col_name}[/cyan]: {docs_processed} docs in {processing_time:.2f}s ({mode}, {workers_used} workers)")
        else:
            console.print(f"[red]✗[/red] Re-indexing completed with {result['failed_collections']} failure(s)")
            
            for error in result.get('errors', []):
                console.print(f"  [red]Error:[/red] {error}")
        
        # Register current model after successful re-indexing
        if result["success"] and result['successful_collections'] > 0:
            try:
                detector = ModelChangeDetector()
                embedding_service = create_embedding_service()
                
                if hasattr(embedding_service, 'generate_model_fingerprint'):
                    fingerprint = embedding_service.generate_model_fingerprint()
                    detector.register_model(fingerprint)
                    console.print("\n[green]✓[/green] Current model registered successfully")
                
            except Exception as e:
                console.print(f"\n[yellow]Warning:[/yellow] Could not register current model: {e}")
        
    except Exception as e:
        console.print(f"[red]Error during re-indexing:[/red] {e}")
        logger.error(f"Re-indexing failed: {e}")
        raise typer.Exit(1)


@model_app.command("history")
def show_notification_history(
    limit: int = typer.Option(
        10,
        "--limit",
        help="Maximum number of notifications to show"
    )
) -> None:
    """
    Show recent model change notification history.
    
    Displays a table of recent model change notifications including
    user decisions and outcomes.
    
    Args:
        limit: Maximum number of notifications to display
    
    Example:
        research-agent model history
        research-agent model history --limit=20
    """
    try:
        # Create notification service (this will be empty since it's a new instance)
        # In a real implementation, you'd want to persist notification history
        notification_service = ModelChangeNotificationService(console=console)
        
        # For now, show a message about implementation
        console.print("[blue]Model Change Notification History[/blue]")
        console.print("[dim]Note: Notification history is currently session-based[/dim]")
        
        notification_service.list_notification_history(limit=limit)
        
        # Show detector history as alternative
        detector = ModelChangeDetector()
        registered_models = detector.list_models()
        
        if registered_models:
            console.print(f"\n[bold]Registered Models ({len(registered_models)}):[/bold]")
            
            table = Table()
            table.add_column("Model Name", style="bold")
            table.add_column("Type")
            table.add_column("Version")
            table.add_column("Checksum", style="dim")
            
            for model_name in registered_models[:limit]:
                fingerprint = detector.get_model_fingerprint(model_name)
                if fingerprint:
                    table.add_row(
                        fingerprint.model_name,
                        fingerprint.model_type,
                        fingerprint.version,
                        fingerprint.checksum[:12] + "..."
                    )
            
            console.print(table)
        else:
            console.print("[dim]No registered models found[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"History command failed: {e}")
        raise typer.Exit(1)


@model_app.command("register")
def register_current_model(
    force: bool = typer.Option(
        False,
        "--force",
        help="Force registration even if model hasn't changed"
    )
) -> None:
    """
    Manually register the current embedding model.
    
    This command registers the current embedding model configuration
    in the change detection system. Useful for initial setup or
    after manual model changes.
    
    Args:
        force: Force registration even if no changes are detected
    
    Example:
        research-agent model register
        research-agent model register --force
    """
    try:
        embedding_service = create_embedding_service()
        if not embedding_service:
            console.print("[red]Error:[/red] Could not access embedding service")
            raise typer.Exit(1)
        
        current_fingerprint = embedding_service.generate_model_fingerprint()
        detector = ModelChangeDetector()
        
        # Check if registration is needed
        change_detected = detector.detect_change(current_fingerprint)
        
        if not change_detected and not force:
            console.print("✅ [green]Model is already registered and up to date[/green]")
            console.print(f"Model: {current_fingerprint.model_name}")
            console.print(f"Version: {current_fingerprint.version}")
            return
        
        # Register the model
        event = detector.register_model_with_event(current_fingerprint)
        
        # Show registration details
        console.print("✅ [green]Model registered successfully[/green]")
        
        table = Table(title="📝 Registration Details")
        table.add_column("Property", style="bold")
        table.add_column("Value")
        
        table.add_row("Model Name", event.model_name)
        table.add_row("Change Type", event.change_type.replace('_', ' ').title())
        table.add_row("Version", event.new_fingerprint.version)
        table.add_row("Requires Re-indexing", "Yes" if event.requires_reindexing else "No")
        table.add_row("Timestamp", event.timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        
        console.print("\n")
        console.print(table)
        
        if event.requires_reindexing:
            console.print(
                "\n[yellow]Note:[/yellow] This change may require re-indexing "
                "existing collections. Run 'research-agent model reindex' "
                "to update affected collections."
            )
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Register command failed: {e}")
        raise typer.Exit(1)


@model_app.command("dashboard")
def show_progress_dashboard(
    live: bool = typer.Option(
        False,
        "--live",
        "-l",
        help="Show live updating dashboard"
    ),
    refresh_interval: float = typer.Option(
        1.0,
        "--refresh",
        "-r",
        help="Refresh interval in seconds for live dashboard"
    ),
    max_operations: int = typer.Option(
        10,
        "--max-operations",
        "-m",
        help="Maximum number of operations to display"
    )
) -> None:
    """
    Display the progress tracking dashboard.
    
    Shows real-time progress for model changes, re-indexing operations,
    and other Research Agent activities.
    
    Examples:
        research-agent model dashboard
        research-agent model dashboard --live --refresh=0.5
        research-agent model dashboard --max-operations=20
    """
    try:
        from ..services.progress_dashboard import (
            ProgressTrackingDashboard, 
            DashboardConfig,
            get_dashboard
        )
        
        console = Console()
        
        # Configure dashboard
        config = DashboardConfig(
            refresh_interval=refresh_interval,
            max_operations_displayed=max_operations,
            show_completed_operations=True,
            show_performance_metrics=True
        )
        
        # Get or create dashboard instance
        dashboard = get_dashboard()
        dashboard.config = config
        dashboard.console = console
        
        if live:
            console.print("[blue]Starting live progress dashboard...[/blue]")
            console.print("[dim]Press Ctrl+C to exit[/dim]\n")
            
            try:
                dashboard.start_live_dashboard()
            except KeyboardInterrupt:
                console.print("\n[yellow]Dashboard stopped[/yellow]")
        else:
            # Show static summary
            dashboard.display_summary()
        
    except ImportError as e:
        console.print(f"[red]Error:[/red] Dashboard dependencies not available: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error displaying dashboard:[/red] {e}")
        logger.error(f"Dashboard display failed: {e}")
        raise typer.Exit(1)


@model_app.command("status")
def show_operation_status(
    operation_id: Optional[str] = typer.Option(
        None,
        "--operation-id",
        "-o",
        help="Show status for specific operation ID"
    ),
    list_active: bool = typer.Option(
        False,
        "--list-active",
        "-a",
        help="List all active operations"
    ),
    list_completed: bool = typer.Option(
        False,
        "--list-completed",
        "-c",
        help="List recent completed operations"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Limit number of operations to show"
    )
) -> None:
    """
    Show status of operations and model change detection.
    
    Displays current model status, active operations, and recent completions.
    Can show detailed information for specific operations.
    
    Examples:
        research-agent model status
        research-agent model status --list-active
        research-agent model status --operation-id=reindex_123456
        research-agent model status --list-completed --limit=5
    """
    try:
        from ..services.progress_dashboard import get_dashboard
        
        console = Console()
        dashboard = get_dashboard()
        
        # Show model change detection status
        console.print("[blue]Model Change Detection Status[/blue]")
        console.print("=" * 40)
        
        try:
            detector = ModelChangeDetector()
            embedding_service = create_embedding_service()
            
            if hasattr(embedding_service, 'generate_model_fingerprint'):
                current_fingerprint = embedding_service.generate_model_fingerprint()
                registered_models = detector.get_registered_models()
                
                console.print(f"Current model: [cyan]{current_fingerprint.model_name}[/cyan]")
                console.print(f"Model version: [cyan]{current_fingerprint.version}[/cyan]")
                console.print(f"Registered models: [cyan]{len(registered_models)}[/cyan]")
                
                # Check for changes
                change_detected = detector.detect_change(current_fingerprint)
                if change_detected:
                    console.print("[yellow]⚠️  Model change detected[/yellow]")
                else:
                    console.print("[green]✓ No model changes detected[/green]")
            else:
                console.print("[yellow]Model fingerprinting not available[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error checking model status:[/red] {e}")
        
        console.print()
        
        # Show specific operation status
        if operation_id:
            console.print(f"[blue]Operation Status: {operation_id}[/blue]")
            console.print("-" * 40)
            
            metrics = dashboard.get_operation_status(operation_id)
            if metrics:
                status_style = {
                    "completed": "green",
                    "failed": "red",
                    "in_progress": "blue",
                    "cancelled": "yellow"
                }.get(metrics.status.value, "white")
                
                console.print(f"Type: [cyan]{metrics.operation_type.value.replace('_', ' ').title()}[/cyan]")
                console.print(f"Status: [{status_style}]{metrics.status.value.replace('_', ' ').title()}[/{status_style}]")
                console.print(f"Progress: [cyan]{metrics.progress_percentage:.1f}%[/cyan]")
                console.print(f"Started: [dim]{metrics.start_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
                
                if metrics.end_time:
                    console.print(f"Completed: [dim]{metrics.end_time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
                    console.print(f"Duration: [cyan]{metrics.elapsed_time.total_seconds():.1f}s[/cyan]")
                
                if metrics.total_items > 0:
                    console.print(f"Items: [cyan]{metrics.items_processed}/{metrics.total_items}[/cyan]")
                
                if metrics.current_item:
                    console.print(f"Current: [dim]{metrics.current_item}[/dim]")
                
                if metrics.error_message:
                    console.print(f"Error: [red]{metrics.error_message}[/red]")
                
                if metrics.metadata:
                    console.print("Metadata:")
                    for key, value in metrics.metadata.items():
                        console.print(f"  {key}: [dim]{value}[/dim]")
            else:
                console.print(f"[yellow]Operation not found: {operation_id}[/yellow]")
            
            console.print()
        
        # Show active operations
        if list_active or not (operation_id or list_completed):
            active_ops = dashboard.get_active_operations()
            
            console.print(f"[blue]Active Operations ({len(active_ops)})[/blue]")
            console.print("-" * 30)
            
            if active_ops:
                for metrics in active_ops[:limit]:
                    status_style = {
                        "in_progress": "blue",
                        "initializing": "yellow",
                        "paused": "orange"
                    }.get(metrics.status.value, "white")
                    
                    console.print(f"• [{status_style}]{metrics.operation_id}[/{status_style}] - {metrics.operation_type.value.replace('_', ' ').title()}")
                    console.print(f"  Progress: {metrics.progress_percentage:.1f}% | {metrics.current_item}")
                    console.print()
            else:
                console.print("[dim]No active operations[/dim]")
                console.print()
        
        # Show completed operations
        if list_completed:
            completed_ops = dashboard.get_completed_operations(limit=limit)
            
            console.print(f"[blue]Recent Completed Operations ({len(completed_ops)})[/blue]")
            console.print("-" * 35)
            
            if completed_ops:
                for metrics in completed_ops:
                    status_style = "green" if metrics.status.value == "completed" else "red"
                    duration = metrics.elapsed_time.total_seconds()
                    
                    console.print(f"• [{status_style}]{metrics.operation_id}[/{status_style}] - {metrics.operation_type.value.replace('_', ' ').title()}")
                    console.print(f"  Status: {metrics.status.value.title()} | Duration: {duration:.1f}s | Items: {metrics.items_processed}")
                    if metrics.end_time:
                        console.print(f"  Completed: {metrics.end_time.strftime('%H:%M:%S')}")
                    console.print()
            else:
                console.print("[dim]No completed operations[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error showing status:[/red] {e}")
        logger.error(f"Status display failed: {e}")
        raise typer.Exit(1)


@model_app.command("backup")
def create_backup(
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to backup (default: all)"
    ),
    backup_type: str = typer.Option(
        "full",
        "--type",
        "-t",
        help="Type of backup: full, incremental, differential, snapshot"
    ),
    backup_id: Optional[str] = typer.Option(
        None,
        "--backup-id",
        "-i",
        help="Custom backup ID (generated if not provided)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Backup output directory (default: ./backups)"
    ),
    compress: bool = typer.Option(
        True,
        "--compress/--no-compress",
        help="Create compressed backup archive"
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Verify backup integrity with checksums"
    )
) -> None:
    """
    Create a backup of Research Agent collections and model state.
    
    Creates comprehensive backups including collection data, embeddings,
    metadata, and model fingerprints for recovery purposes.
    
    Examples:
        research-agent model backup
        research-agent model backup --collections=docs,research --type=snapshot
        research-agent model backup --backup-id=pre-migration --compress
    """
    try:
        from ..services.backup_recovery_service import (
            BackupRecoveryService,
            BackupConfig,
            BackupType
        )
        
        console = Console()
        
        # Validate backup type
        try:
            backup_type_enum = BackupType(backup_type.lower())
        except ValueError:
            console.print(f"[red]Error:[/red] Invalid backup type: {backup_type}")
            console.print("Valid types: full, incremental, differential, snapshot")
            raise typer.Exit(1)
        
        # Parse collections
        collection_list = None
        if collections:
            collection_list = [c.strip() for c in collections.split(",")]
        
        # Configure backup service
        config = BackupConfig(
            compression_enabled=compress,
            checksum_verification=verify
        )
        
        if output_dir:
            config.backup_directory = Path(output_dir)
        
        # Initialize services
        chroma_manager = create_chroma_manager()
        transaction_manager = TransactionManager(vector_store=chroma_manager)
        
        backup_service = BackupRecoveryService(
            vector_store=chroma_manager,
            config=config,
            transaction_manager=transaction_manager
        )
        
        # Progress tracking
        def progress_callback(percent: float, message: str) -> None:
            console.print(f"[blue]{percent:5.1f}%[/blue] {message}")
        
        console.print("[blue]Starting backup operation...[/blue]")
        
        if collection_list:
            console.print(f"Collections: [cyan]{', '.join(collection_list)}[/cyan]")
        else:
            console.print("Collections: [cyan]All collections[/cyan]")
        
        console.print(f"Type: [cyan]{backup_type_enum.value}[/cyan]")
        console.print(f"Compression: [cyan]{'Enabled' if compress else 'Disabled'}[/cyan]")
        console.print(f"Verification: [cyan]{'Enabled' if verify else 'Disabled'}[/cyan]")
        console.print()
        
        # Create backup
        start_time = datetime.now()
        backup_id_result = backup_service.create_backup(
            collections=collection_list,
            backup_type=backup_type_enum,
            backup_id=backup_id,
            progress_callback=progress_callback
        )
        
        # Get backup metadata
        backup_metadata = backup_service.get_backup_metadata(backup_id_result)
        
        # Display results
        console.print("\n[green]✓ Backup completed successfully[/green]")
        console.print(f"Backup ID: [cyan]{backup_id_result}[/cyan]")
        console.print(f"Duration: [cyan]{backup_metadata.duration_seconds:.2f} seconds[/cyan]")
        console.print(f"Collections: [cyan]{len(backup_metadata.collections)}[/cyan]")
        
        if backup_metadata.file_path:
            file_size_mb = backup_metadata.file_size_bytes / (1024 * 1024)
            console.print(f"File: [cyan]{backup_metadata.file_path}[/cyan]")
            console.print(f"Size: [cyan]{file_size_mb:.2f} MB[/cyan]")
        
        if backup_metadata.checksum:
            console.print(f"Checksum: [dim]{backup_metadata.checksum[:16]}...[/dim]")
        
        console.print(f"\nBackup saved to: [cyan]{config.backup_directory}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error creating backup:[/red] {e}")
        logger.error(f"Backup creation failed: {e}")
        raise typer.Exit(1)


@model_app.command("restore")
def restore_backup(
    backup_id: str = typer.Argument(
        help="ID of backup to restore"
    ),
    collections: Optional[str] = typer.Option(
        None,
        "--collections",
        "-c",
        help="Comma-separated list of collections to restore (default: all from backup)"
    ),
    recovery_id: Optional[str] = typer.Option(
        None,
        "--recovery-id",
        "-r",
        help="Custom recovery operation ID"
    ),
    backup_dir: Optional[str] = typer.Option(
        None,
        "--backup-dir",
        "-d",
        help="Backup directory location (default: ./backups)"
    ),
    use_transaction: bool = typer.Option(
        True,
        "--transaction/--no-transaction",
        help="Use transaction for rollback support"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompts"
    )
) -> None:
    """
    Restore collections from a backup.
    
    Restores collection data, embeddings, metadata, and model fingerprints
    from a previously created backup with optional transaction rollback support.
    
    Examples:
        research-agent model restore backup_1234567890
        research-agent model restore backup_1234567890 --collections=docs,research
        research-agent model restore backup_1234567890 --force --no-transaction
    """
    try:
        from ..services.backup_recovery_service import (
            BackupRecoveryService,
            BackupConfig
        )
        
        console = Console()
        
        # Parse collections
        collection_list = None
        if collections:
            collection_list = [c.strip() for c in collections.split(",")]
        
        # Configure backup service
        config = BackupConfig()
        if backup_dir:
            config.backup_directory = Path(backup_dir)
        
        # Initialize services
        chroma_manager = create_chroma_manager()
        transaction_manager = TransactionManager(vector_store=chroma_manager)
        
        backup_service = BackupRecoveryService(
            vector_store=chroma_manager,
            config=config,
            transaction_manager=transaction_manager
        )
        
        # Get backup metadata
        backup_metadata = backup_service.get_backup_metadata(backup_id)
        if not backup_metadata:
            console.print(f"[red]Error:[/red] Backup not found: {backup_id}")
            raise typer.Exit(1)
        
        # Display restore plan
        console.print(f"[blue]Restore Plan[/blue]")
        console.print("=" * 40)
        console.print(f"Backup ID: [cyan]{backup_id}[/cyan]")
        console.print(f"Backup Type: [cyan]{backup_metadata.backup_type.value}[/cyan]")
        console.print(f"Created: [cyan]{backup_metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")
        
        if collection_list:
            console.print(f"Collections to restore: [cyan]{', '.join(collection_list)}[/cyan]")
        else:
            console.print(f"Collections to restore: [cyan]{', '.join(backup_metadata.collections)}[/cyan]")
        
        console.print(f"Transaction support: [cyan]{'Enabled' if use_transaction else 'Disabled'}[/cyan]")
        
        if backup_metadata.file_path:
            file_size_mb = backup_metadata.file_size_bytes / (1024 * 1024)
            console.print(f"Backup size: [cyan]{file_size_mb:.2f} MB[/cyan]")
        
        # Confirmation
        if not force:
            console.print("\n[yellow]Warning:[/yellow] This will overwrite existing collections with backup data.")
            if not typer.confirm("Continue with restore operation?"):
                console.print("[yellow]Restore cancelled[/yellow]")
                return
        
        # Progress tracking
        def progress_callback(percent: float, message: str) -> None:
            console.print(f"[blue]{percent:5.1f}%[/blue] {message}")
        
        console.print("\n[blue]Starting restore operation...[/blue]")
        
        # Restore backup
        start_time = datetime.now()
        recovery_id_result = backup_service.restore_backup(
            backup_id=backup_id,
            target_collections=collection_list,
            recovery_id=recovery_id,
            progress_callback=progress_callback,
            use_transaction=use_transaction
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display results
        console.print("\n[green]✓ Restore completed successfully[/green]")
        console.print(f"Recovery ID: [cyan]{recovery_id_result}[/cyan]")
        console.print(f"Duration: [cyan]{duration:.2f} seconds[/cyan]")
        
        if collection_list:
            console.print(f"Restored collections: [cyan]{len(collection_list)}[/cyan]")
        else:
            console.print(f"Restored collections: [cyan]{len(backup_metadata.collections)}[/cyan]")
        
        console.print("\n[green]Collections have been restored from backup[/green]")
        
    except Exception as e:
        console.print(f"[red]Error during restore:[/red] {e}")
        logger.error(f"Restore operation failed: {e}")
        raise typer.Exit(1)


@model_app.command("list-backups")
def list_backups(
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by backup status: pending, in_progress, completed, failed, cancelled"
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        "-l",
        help="Maximum number of backups to display"
    ),
    backup_dir: Optional[str] = typer.Option(
        None,
        "--backup-dir",
        "-d",
        help="Backup directory location (default: ./backups)"
    )
) -> None:
    """
    List available backups with their metadata.
    
    Shows backup history including status, creation time, size,
    and collections included in each backup.
    
    Examples:
        research-agent model list-backups
        research-agent model list-backups --status=completed --limit=10
        research-agent model list-backups --backup-dir=/custom/backup/path
    """
    try:
        from ..services.backup_recovery_service import (
            BackupRecoveryService,
            BackupConfig,
            BackupStatus
        )
        
        console = Console()
        
        # Validate status filter
        status_filter = None
        if status:
            try:
                status_filter = BackupStatus(status.lower())
            except ValueError:
                console.print(f"[red]Error:[/red] Invalid status: {status}")
                console.print("Valid statuses: pending, in_progress, completed, failed, cancelled")
                raise typer.Exit(1)
        
        # Configure backup service
        config = BackupConfig()
        if backup_dir:
            config.backup_directory = Path(backup_dir)
        
        # Initialize services (minimal setup for listing)
        try:
            chroma_manager = create_chroma_manager()
        except Exception:
            # If ChromaDB is not available, create a minimal service for listing
            chroma_manager = None
        
        if chroma_manager:
            backup_service = BackupRecoveryService(
                vector_store=chroma_manager,
                config=config
            )
        else:
            # Create service without vector store for listing only
            backup_service = BackupRecoveryService.__new__(BackupRecoveryService)
            backup_service.config = config
            backup_service.backup_history = {}
            backup_service.active_backups = {}
            backup_service._lock = Lock()
            backup_service.metadata_file = config.backup_directory / "backup_metadata.json"
            backup_service.logger = logging.getLogger(__name__)
            backup_service._load_backup_metadata()
        
        # Get backups
        backups = backup_service.list_backups(
            status_filter=status_filter,
            limit=limit
        )
        
        if not backups:
            console.print("[yellow]No backups found[/yellow]")
            if status_filter:
                console.print(f"No backups with status: {status_filter.value}")
            return
        
        # Display backups table
        table = Table(title=f"Available Backups ({len(backups)} found)")
        table.add_column("Backup ID", style="cyan", width=20)
        table.add_column("Type", width=12)
        table.add_column("Status", width=12)
        table.add_column("Created", width=16)
        table.add_column("Duration", width=10)
        table.add_column("Collections", width=8)
        table.add_column("Size", width=10)
        
        for backup in backups:
            # Status with color coding
            status_style = {
                BackupStatus.COMPLETED: "green",
                BackupStatus.FAILED: "red",
                BackupStatus.IN_PROGRESS: "blue",
                BackupStatus.PENDING: "yellow",
                BackupStatus.CANCELLED: "dim"
            }.get(backup.status, "white")
            
            # File size
            if backup.file_size_bytes > 0:
                size_mb = backup.file_size_bytes / (1024 * 1024)
                size_text = f"{size_mb:.1f} MB"
            else:
                size_text = "-"
            
            # Duration
            duration_text = f"{backup.duration_seconds:.1f}s" if backup.duration_seconds > 0 else "-"
            
            table.add_row(
                backup.backup_id,
                backup.backup_type.value.title(),
                Text(backup.status.value.replace("_", " ").title(), style=status_style),
                backup.created_at.strftime("%Y-%m-%d %H:%M"),
                duration_text,
                str(len(backup.collections)),
                size_text
            )
        
        console.print(table)
        
        # Summary information
        console.print(f"\nBackup directory: [cyan]{config.backup_directory}[/cyan]")
        
        if status_filter:
            console.print(f"Filtered by status: [cyan]{status_filter.value}[/cyan]")
        
        if len(backups) == limit:
            console.print(f"[dim]Showing first {limit} backups. Use --limit to see more.[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error listing backups:[/red] {e}")
        logger.error(f"Backup listing failed: {e}")
        raise typer.Exit(1)


@model_app.command("delete-backup")
def delete_backup(
    backup_id: str = typer.Argument(
        help="ID of backup to delete"
    ),
    backup_dir: Optional[str] = typer.Option(
        None,
        "--backup-dir",
        "-d",
        help="Backup directory location (default: ./backups)"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt"
    )
) -> None:
    """
    Delete a backup and its associated files.
    
    Permanently removes backup files and metadata. This operation
    cannot be undone.
    
    Examples:
        research-agent model delete-backup backup_1234567890
        research-agent model delete-backup backup_1234567890 --force
    """
    try:
        from ..services.backup_recovery_service import (
            BackupRecoveryService,
            BackupConfig
        )
        
        console = Console()
        
        # Configure backup service
        config = BackupConfig()
        if backup_dir:
            config.backup_directory = Path(backup_dir)
        
        # Initialize services (minimal setup for deletion)
        try:
            chroma_manager = create_chroma_manager()
        except Exception:
            chroma_manager = None
        
        if chroma_manager:
            backup_service = BackupRecoveryService(
                vector_store=chroma_manager,
                config=config
            )
        else:
            # Create service without vector store for deletion only
            backup_service = BackupRecoveryService.__new__(BackupRecoveryService)
            backup_service.config = config
            backup_service.backup_history = {}
            backup_service.active_backups = {}
            backup_service._lock = Lock()
            backup_service.metadata_file = config.backup_directory / "backup_metadata.json"
            backup_service.logger = logging.getLogger(__name__)
            backup_service._load_backup_metadata()
        
        # Get backup metadata
        backup_metadata = backup_service.get_backup_metadata(backup_id)
        if not backup_metadata:
            console.print(f"[red]Error:[/red] Backup not found: {backup_id}")
            raise typer.Exit(1)
        
        # Display backup information
        console.print(f"[blue]Backup to Delete[/blue]")
        console.print("-" * 30)
        console.print(f"Backup ID: [cyan]{backup_id}[/cyan]")
        console.print(f"Type: [cyan]{backup_metadata.backup_type.value}[/cyan]")
        console.print(f"Created: [cyan]{backup_metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")
        console.print(f"Collections: [cyan]{len(backup_metadata.collections)}[/cyan]")
        
        if backup_metadata.file_path:
            file_size_mb = backup_metadata.file_size_bytes / (1024 * 1024)
            console.print(f"Size: [cyan]{file_size_mb:.2f} MB[/cyan]")
            console.print(f"File: [cyan]{backup_metadata.file_path}[/cyan]")
        
        # Confirmation
        if not force:
            console.print("\n[red]Warning:[/red] This will permanently delete the backup and cannot be undone.")
            if not typer.confirm("Are you sure you want to delete this backup?"):
                console.print("[yellow]Deletion cancelled[/yellow]")
                return
        
        # Delete backup
        success = backup_service.delete_backup(backup_id)
        
        if success:
            console.print(f"\n[green]✓ Backup {backup_id} deleted successfully[/green]")
        else:
            console.print(f"\n[red]✗ Failed to delete backup {backup_id}[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Error deleting backup:[/red] {e}")
        logger.error(f"Backup deletion failed: {e}")
        raise typer.Exit(1)


@model_app.command("snapshot")
def create_snapshot(
    collection: str = typer.Argument(
        help="Name of collection to snapshot"
    ),
    snapshot_id: Optional[str] = typer.Option(
        None,
        "--snapshot-id",
        "-i",
        help="Custom snapshot ID (generated if not provided)"
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Snapshot output directory (default: ./backups)"
    )
) -> None:
    """
    Create a quick snapshot of a single collection.
    
    Creates a lightweight backup of a specific collection for
    quick recovery or testing purposes.
    
    Examples:
        research-agent model snapshot docs
        research-agent model snapshot research --snapshot-id=pre-test
    """
    try:
        from ..services.backup_recovery_service import (
            BackupRecoveryService,
            BackupConfig
        )
        
        console = Console()
        
        # Configure backup service
        config = BackupConfig()
        if output_dir:
            config.backup_directory = Path(output_dir)
        
        # Initialize services
        chroma_manager = create_chroma_manager()
        
        # Verify collection exists
        existing_collections = [col.name for col in chroma_manager.list_collections()]
        if collection not in existing_collections:
            console.print(f"[red]Error:[/red] Collection not found: {collection}")
            console.print(f"Available collections: {', '.join(existing_collections)}")
            raise typer.Exit(1)
        
        backup_service = BackupRecoveryService(
            vector_store=chroma_manager,
            config=config
        )
        
        console.print(f"[blue]Creating snapshot of collection:[/blue] [cyan]{collection}[/cyan]")
        
        # Create snapshot
        start_time = datetime.now()
        snapshot_id_result = backup_service.create_collection_snapshot(
            collection_name=collection,
            snapshot_id=snapshot_id
        )
        
        # Get snapshot metadata
        snapshot_metadata = backup_service.get_backup_metadata(snapshot_id_result)
        
        # Display results
        console.print(f"\n[green]✓ Snapshot created successfully[/green]")
        console.print(f"Snapshot ID: [cyan]{snapshot_id_result}[/cyan]")
        console.print(f"Duration: [cyan]{snapshot_metadata.duration_seconds:.2f} seconds[/cyan]")
        
        if snapshot_metadata.file_path:
            file_size_mb = snapshot_metadata.file_size_bytes / (1024 * 1024)
            console.print(f"File: [cyan]{snapshot_metadata.file_path}[/cyan]")
            console.print(f"Size: [cyan]{file_size_mb:.2f} MB[/cyan]")
        
        console.print(f"\nSnapshot saved to: [cyan]{config.backup_directory}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Error creating snapshot:[/red] {e}")
        logger.error(f"Snapshot creation failed: {e}")
        raise typer.Exit(1)


@model_app.command("validate-migration")
def validate_migration(
    migration_id: str = typer.Argument(..., help="Migration ID to validate"),
    collections: Optional[str] = typer.Option(
        None, "--collections", "-c",
        help="Comma-separated list of collections to validate"
    ),
    pre_migration_backup: Optional[str] = typer.Option(
        None, "--pre-backup", "-b",
        help="Pre-migration backup ID for comparison"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output directory for validation reports"
    ),
    skip_performance: bool = typer.Option(
        False, "--skip-performance",
        help="Skip performance benchmark tests"
    ),
    skip_semantic: bool = typer.Option(
        False, "--skip-semantic",
        help="Skip semantic equivalence tests"
    ),
    skip_integrity: bool = typer.Option(
        False, "--skip-integrity",
        help="Skip collection integrity tests"
    )
):
    """Validate a migration operation with comprehensive testing."""
    try:
        # Initialize services
        config_manager = ConfigManager()
        vector_store = ChromaDBManager(config_manager)
        
        # Parse collections
        collection_list = None
        if collections:
            collection_list = [c.strip() for c in collections.split(',')]
        else:
            # Get all collections if none specified
            collection_list = vector_store.list_collections()
        
        # Create validation config
        validation_config = ValidationConfig(
            enable_performance_tests=not skip_performance,
            enable_semantic_tests=not skip_semantic,
            enable_integrity_tests=not skip_integrity
        )
        
        if output_dir:
            validation_config.report_output_directory = output_dir
        
        # Initialize validation service
        validation_service = MigrationValidationService(
            vector_store=vector_store,
            config=validation_config,
            config_manager=config_manager
        )
        
        # Get pre-migration data if backup provided
        pre_migration_data = None
        if pre_migration_backup:
            console.print(f"\n[cyan]📦 Loading pre-migration data from backup {pre_migration_backup}...[/cyan]")
            backup_service = BackupRecoveryService(
                vector_store=vector_store,
                config_manager=config_manager
            )
            # This would need additional implementation to extract embeddings from backup
            console.print("[yellow]⚠️ Pre-migration comparison not yet implemented[/yellow]")
        
        # Run validation with progress tracking
        console.print(f"\n[cyan]🔍 Starting migration validation for: {migration_id}[/cyan]")
        console.print(f"Collections: {', '.join(collection_list)}")
        
        # Progress callback
        def progress_callback(message: str, progress: float):
            if progress <= 100:
                console.print(f"[dim]{message} ({progress:.1f}%)[/dim]")
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running validation tests...", total=100)
            
            def update_progress(message: str, percent: float):
                progress.update(task, completed=percent, description=message)
            
            # Run validation
            report = validation_service.validate_migration(
                migration_id=migration_id,
                collections=collection_list,
                pre_migration_data=pre_migration_data,
                progress_callback=update_progress
            )
        
        # Display results
        _display_validation_report(report)
        
        # Exit with error code if validation failed
        if report.overall_status == ValidationStatus.FAILED:
            raise typer.Exit(1)
        elif report.overall_status == ValidationStatus.WARNING:
            console.print("\n[yellow]⚠️ Validation completed with warnings[/yellow]")
        else:
            console.print("\n[green]✅ Migration validation passed successfully![/green]")
        
    except Exception as e:
        console.print(f"\n[red]❌ Error validating migration: {e}[/red]")
        raise typer.Exit(1)


@model_app.command("list-validations")
def list_validations(
    limit: int = typer.Option(
        10, "--limit", "-l",
        help="Maximum number of validation reports to show"
    ),
    status_filter: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by validation status (passed, failed, warning)"
    )
):
    """List validation reports."""
    try:
        config_manager = ConfigManager()
        vector_store = ChromaDBManager(config_manager)
        validation_service = MigrationValidationService(
            vector_store=vector_store,
            config_manager=config_manager
        )
        
        # Parse status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = ValidationStatus(status_filter.lower())
            except ValueError:
                console.print(f"[red]Invalid status filter: {status_filter}[/red]")
                raise typer.Exit(1)
        
        reports = validation_service.list_validation_reports(
            limit=limit,
            status_filter=status_enum
        )
        
        if not reports:
            console.print("\n[yellow]No validation reports found.[/yellow]")
            return
        
        # Create validation reports table
        table = Table(title=f"Validation Reports (showing {len(reports)})")
        table.add_column("Validation ID", style="cyan")
        table.add_column("Migration ID", style="green")
        table.add_column("Started", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Duration", style="magenta")
        table.add_column("Collections", style="dim")
        table.add_column("Tests", style="dim")
        
        for report in reports:
            started_str = report.started_at.strftime("%Y-%m-%d %H:%M")
            duration_str = f"{report.duration_seconds:.1f}s" if report.completed_at else "Running..."
            
            collections_str = f"{len(report.collections_validated)} collections"
            tests_str = f"{len(report.passed_tests)}/{len(report.test_results)} passed"
            
            # Color code status
            status_str = report.overall_status.value
            if report.overall_status == ValidationStatus.PASSED:
                status_str = f"[green]{status_str}[/green]"
            elif report.overall_status == ValidationStatus.FAILED:
                status_str = f"[red]{status_str}[/red]"
            elif report.overall_status == ValidationStatus.WARNING:
                status_str = f"[yellow]{status_str}[/yellow]"
            
            table.add_row(
                report.validation_id,
                report.migration_id,
                started_str,
                status_str,
                duration_str,
                collections_str,
                tests_str
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"\n[red]❌ Error listing validations: {e}[/red]")
        raise typer.Exit(1)


@model_app.command("show-validation")
def show_validation(
    validation_id: str = typer.Argument(..., help="Validation ID to show details for"),
    show_metrics: bool = typer.Option(
        False, "--metrics", "-m",
        help="Show detailed metrics for each test"
    )
):
    """Show detailed validation report."""
    try:
        config_manager = ConfigManager()
        vector_store = ChromaDBManager(config_manager)
        validation_service = MigrationValidationService(
            vector_store=vector_store,
            config_manager=config_manager
        )
        
        report = validation_service.get_validation_report(validation_id)
        
        if not report:
            console.print(f"\n[red]Validation report not found: {validation_id}[/red]")
            raise typer.Exit(1)
        
        _display_validation_report(report, show_metrics=show_metrics)
        
    except Exception as e:
        console.print(f"\n[red]❌ Error showing validation: {e}[/red]")
        raise typer.Exit(1)


def _display_validation_report(report: ValidationReport, show_metrics: bool = False):
    """Display a comprehensive validation report."""
    # Header
    console.print(f"\n[bold cyan]📊 Migration Validation Report[/bold cyan]")
    console.print(f"Validation ID: [cyan]{report.validation_id}[/cyan]")
    console.print(f"Migration ID: [green]{report.migration_id}[/green]")
    console.print(f"Started: {report.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if report.completed_at:
        console.print(f"Completed: {report.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        console.print(f"Duration: {report.duration_seconds:.1f} seconds")
    
    # Overall status
    status_color = "green" if report.overall_status == ValidationStatus.PASSED else "red" if report.overall_status == ValidationStatus.FAILED else "yellow"
    console.print(f"\nOverall Status: [{status_color}]{report.overall_status.value.upper()}[/{status_color}]")
    console.print(f"Severity: {report.overall_severity.value}")
    
    # Summary statistics
    if report.summary:
        summary_panel = Panel(
            f"[green]✅ Passed: {report.summary.get('passed_tests', 0)}[/green]\n"
            f"[red]❌ Failed: {report.summary.get('failed_tests', 0)}[/red]\n"
            f"[yellow]⚠️ Warnings: {report.summary.get('warning_tests', 0)}[/yellow]\n"
            f"[blue]📊 Success Rate: {report.summary.get('success_rate', 0)}%[/blue]",
            title="Test Summary",
            border_style="blue"
        )
        console.print(summary_panel)
    
    # Test results
    if report.test_results:
        console.print(f"\n[bold]Test Results ({len(report.test_results)} tests):[/bold]")
        
        for test in report.test_results:
            # Test header
            status_icon = "✅" if test.passed else "❌" if test.failed else "⚠️"
            console.print(f"\n{status_icon} [bold]{test.test_name}[/bold] ({test.test_type.value})")
            console.print(f"   Status: {test.status.value} | Duration: {test.duration_seconds:.2f}s")
            
            if test.error_message:
                console.print(f"   [red]Error: {test.error_message}[/red]")
            
            if test.warnings:
                for warning in test.warnings:
                    console.print(f"   [yellow]Warning: {warning}[/yellow]")
            
            # Show metrics if requested
            if show_metrics and test.metrics:
                metrics_table = Table(show_header=True, header_style="bold magenta")
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                metrics_table.add_column("Status", style="blue")
                metrics_table.add_column("Message", style="dim")
                
                for metric in test.metrics:
                    status_str = metric.status.value
                    if metric.status == ValidationStatus.PASSED:
                        status_str = f"[green]{status_str}[/green]"
                    elif metric.status == ValidationStatus.FAILED:
                        status_str = f"[red]{status_str}[/red]"
                    elif metric.status == ValidationStatus.WARNING:
                        status_str = f"[yellow]{status_str}[/yellow]"
                    
                    metrics_table.add_row(
                        metric.name,
                        str(metric.value),
                        status_str,
                        metric.message
                    )
                
                console.print(metrics_table)
    
    # Collections validated
    if report.collections_validated:
        console.print(f"\n[bold]Collections Validated:[/bold]")
        for collection in report.collections_validated:
            console.print(f"  • {collection}")


if __name__ == "__main__":
    model_app() 