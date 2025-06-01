"""
Knowledge base augmentation CLI commands for Research Agent.

This module implements CLI commands for augmenting the knowledge base with
external content, research reports, user feedback, and duplicate management.

Implements FR-KB-001: Knowledge base augmentation functionality.
"""

import typer
import logging
import json
import tempfile
from pathlib import Path
from typing import Optional, List
from rich import print as rprint
from rich.progress import Progress
from rich.console import Console
from rich.table import Table
from datetime import datetime

from ..core.augmentation_service import AugmentationService, ExternalResult, ResearchReport, DocumentUpdate, AugmentationError
from ..core.feedback_service import FeedbackService, UserFeedback
from ..utils.config import ConfigManager
from ..exceptions.query_exceptions import QueryError

# Initialize console and logger for structured output
console = Console()
logger = logging.getLogger(__name__)

# Constants
DEFAULT_COLLECTION = "research"
DEFAULT_THRESHOLD = 0.85
DEFAULT_TOP_K = 10


def _get_augmentation_service() -> AugmentationService:
    """Get or create AugmentationService instance with RAG pipeline integration."""
    from ..core.augmentation_service import AugmentationService, AugmentationConfig
    from ..utils.config import ConfigManager
    
    try:
        # Initialize with real configuration manager
        config_manager = ConfigManager()
        
        # Create service with RAG pipeline integration
        # The service will automatically connect to ChromaDB and embedding services
        service = AugmentationService(config_manager=config_manager)
        
        logger.info("Initialized AugmentationService with RAG pipeline integration")
        return service
        
    except Exception as e:
        logger.error(f"Failed to create AugmentationService: {e}")
        
        # Fallback to basic configuration
        try:
            config = AugmentationConfig(
                quality_threshold=0.75,
                auto_categorize=True,
                enable_versioning=True
            )
            service = AugmentationService(config=config)
            logger.warning("Using fallback AugmentationService configuration")
            return service
        except Exception as fallback_error:
            logger.error(f"Fallback service creation also failed: {fallback_error}")
            raise AugmentationError(f"Service initialization failed: {e}")


def _get_feedback_service() -> FeedbackService:
    """Get or create FeedbackService instance with configuration."""
    from ..utils.config import ConfigManager
    
    try:
        config_manager = ConfigManager()
        return FeedbackService(config_manager=config_manager)
    except Exception as e:
        logger.warning(f"Failed to initialize FeedbackService with config: {e}")
        return FeedbackService()  # Use default configuration


def add_external_result(
    ctx: typer.Context,
    source: str = typer.Option(..., "--source", help="Source URL of the external content"),
    title: str = typer.Option(..., "--title", help="Title of the external content"),
    content: str = typer.Option(..., "--content", help="Main content text"),
    collection: str = typer.Option(DEFAULT_COLLECTION, "--collection", help="Target collection"),
    author: Optional[str] = typer.Option(None, "--author", help="Author name"),
    publication_date: Optional[str] = typer.Option(None, "--publication-date", help="Publication date (YYYY-MM-DD)"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    from_file: Optional[str] = typer.Option(None, "--from-file", help="Load data from JSON file"),
    detect_duplicates: bool = typer.Option(False, "--detect-duplicates", help="Check for duplicate content"),
    show_quality: bool = typer.Option(False, "--show-quality", help="Display detailed quality metrics"),
) -> None:
    """Add external search result or web content to the knowledge base with enhanced validation."""
    try:
        from ..core.augmentation_service import ExternalResult, QualityValidationError, DuplicateContentError
        
        if from_file:
            # Load from JSON file
            file_path = Path(from_file)
            if not file_path.exists():
                rprint(f"[red]Error:[/red] File not found: {from_file}")
                raise typer.Exit(1)
            
            with file_path.open('r') as f:
                data = json.load(f)
                
            external_result = ExternalResult(
                source_url=data['source'],
                title=data['title'],
                content=data['content'],
                author=data.get('author'),
                metadata=data.get('metadata', {})
            )
        else:
            # Parse publication date if provided
            pub_date = None
            if publication_date:
                try:
                    pub_date = datetime.strptime(publication_date, "%Y-%m-%d")
                except ValueError:
                    rprint(f"[red]Error:[/red] Invalid date format. Use YYYY-MM-DD")
                    raise typer.Exit(1)
            
            # Parse tags if provided
            tag_list = []
            if tags:
                tag_list = [tag.strip() for tag in tags.split(',')]
            
            external_result = ExternalResult(
                source_url=source,
                title=title,
                content=content,
                author=author,
                publication_date=pub_date,
                tags=tag_list
            )
        
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would add external result '{external_result.title}' to collection '{collection}'")
            rprint(f"Content hash: {external_result.content_hash[:8]}...")
            return
        
        # Add external result using enhanced service
        service = _get_augmentation_service()
        result = service.add_external_result(
            external_result, 
            collection=collection,
            detect_duplicates=detect_duplicates
        )
        
        if result['status'] == 'success':
            rprint(f"[green]✅ Success:[/green] Added external result: {result['document_id']}")
            rprint(f"Collection: {result['collection']}")
            
            # Show quality information if requested
            if show_quality and 'quality_score' in result:
                quality_score = result['quality_score']
                rprint(f"Quality Score: [cyan]{quality_score:.3f}[/cyan]")
                
            if 'source_attribution' in result:
                rprint(f"Source attribution: {result['source_attribution']['url']}")
                
            if result.get('auto_assigned'):
                rprint(f"[cyan]Auto-assigned[/cyan] to collection '{result['collection']}' "
                      f"(confidence: {result['assignment_confidence']:.2f})")
                      
        elif result['status'] == 'rejected':
            rprint(f"[red]❌ Rejected:[/red] {result['rejection_reason']}")
            if show_quality:
                rprint(f"Quality score: {result['quality_score']:.3f} (required: ≥ {service.config.quality_threshold})")
            raise typer.Exit(1)
            
        elif result['status'] == 'duplicate_detected':
            rprint(f"[yellow]⚠️  Duplicate detected:[/yellow] Similar to {len(result['similar_documents'])} existing documents")
            rprint(f"Similarity score: {result['similarity_score']:.3f}")
            rprint(f"Suggestion: {result['merge_suggestion']}")
            rprint("Use --detect-duplicates=false to force addition or merge manually")
            raise typer.Exit(1)
            
        else:
            rprint(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
            
    except (QualityValidationError, DuplicateContentError) as e:
        rprint(f"[red]Validation Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to add external result: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def add_research_report(
    ctx: typer.Context,
    file_path: str = typer.Argument(..., help="Path to research report file"),
    collection: str = typer.Option("research-reports", "--collection", help="Target collection"),
    category: Optional[str] = typer.Option(None, "--category", help="Report category"),
    auto_categorize: bool = typer.Option(False, "--auto-categorize", help="Automatically categorize the report"),
    batch: bool = typer.Option(False, "--batch", help="Process multiple files in directory"),
    pattern: str = typer.Option("*.md", "--pattern", help="File pattern for batch processing"),
) -> None:
    """Add research report(s) to the knowledge base."""
    try:
        path = Path(file_path)
        
        if batch and path.is_dir():
            # Batch processing mode
            files = list(path.glob(pattern))
            if not files:
                rprint(f"[yellow]Warning:[/yellow] No files found matching pattern '{pattern}'")
                return
            
            if ctx.obj and ctx.obj.get('dry_run'):
                rprint(f"[blue]DRY RUN:[/blue] Would process {len(files)} research reports")
                return
            
            service = _get_augmentation_service()
            result = service.add_research_reports_batch(
                folder_path=str(path),
                pattern=pattern,
                collection=collection
            )
            
            rprint(f"[green]Batch processing complete:[/green]")
            rprint(f"Processed: {result['processed']}")
            rprint(f"Successful: {result['successful']}")
            rprint(f"Failed: {result['failed']}")
            
        else:
            # Single file processing
            if not path.exists():
                rprint(f"[red]Error:[/red] File not found: {file_path}")
                raise typer.Exit(1)
            
            if ctx.obj and ctx.obj.get('dry_run'):
                rprint(f"[blue]DRY RUN:[/blue] Would add research report '{path.name}'")
                return
            
            research_report = ResearchReport(
                file_path=str(path),
                category=category,
                metadata={}
            )
            
            service = _get_augmentation_service()
            result = service.add_research_report(
                research_report, 
                collection=collection,
                auto_categorize=auto_categorize
            )
            
            if result['status'] == 'success':
                rprint(f"[green]Success:[/green] Added research report: {result['document_id']}")
                rprint(f"Chunks created: {result['chunks_created']}")
                if 'auto_category' in result:
                    rprint(f"Auto-categorized as: {result['auto_category']} (confidence: {result['confidence']})")
            else:
                rprint(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
                raise typer.Exit(1)
                
    except Exception as e:
        logger.error(f"Failed to add research report: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def update_document(
    ctx: typer.Context,
    document_id: str = typer.Argument(..., help="Document ID to update"),
    content: Optional[str] = typer.Option(None, "--content", help="New content for the document"),
    title: Optional[str] = typer.Option(None, "--title", help="New title"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    from_file: Optional[str] = typer.Option(None, "--from-file", help="Load content from file"),
    update_embeddings: bool = typer.Option(True, "--update-embeddings/--no-reembed", help="Update embeddings"),
) -> None:
    """Update an existing document in the knowledge base."""
    try:
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would update document '{document_id}'")
            return
        
        # Prepare metadata updates
        new_metadata = {}
        if title:
            new_metadata['title'] = title
        if tags:
            new_metadata['tags'] = [tag.strip() for tag in tags.split(',')]
        
        # Prepare content update
        new_content = content
        source_file = None
        
        if from_file:
            file_path = Path(from_file)
            if not file_path.exists():
                rprint(f"[red]Error:[/red] File not found: {from_file}")
                raise typer.Exit(1)
            source_file = str(file_path)
        
        update = DocumentUpdate(
            document_id=document_id,
            new_content=new_content,
            new_metadata=new_metadata if new_metadata else None,
            source_file=source_file,
            update_embeddings=update_embeddings
        )
        
        service = _get_augmentation_service()
        result = service.update_document(update)
        
        if result['status'] == 'updated':
            rprint(f"[green]Success:[/green] Updated document: {result['document_id']}")
            rprint(f"Version: {result['version']}")
            rprint(f"Changes: {', '.join(result['changes'])}")
        else:
            rprint(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Failed to update document: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def merge_duplicates(
    ctx: typer.Context,
    auto_detect: bool = typer.Option(False, "--auto-detect", help="Automatically detect duplicates"),
    threshold: float = typer.Option(DEFAULT_THRESHOLD, "--threshold", help="Similarity threshold for detection"),
    documents: Optional[str] = typer.Option(None, "--documents", help="Comma-separated document IDs to merge"),
    strategy: str = typer.Option("union", "--strategy", help="Merge strategy: union, latest-version"),
    keep_originals: bool = typer.Option(False, "--keep-originals", help="Keep original documents after merge"),
    preview_only: bool = typer.Option(False, "--preview-only", help="Show duplicates without merging"),
) -> None:
    """Detect and merge duplicate documents in the knowledge base."""
    try:
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would detect/merge duplicates")
            return
        
        service = _get_augmentation_service()
        
        if preview_only or auto_detect:
            # Detect duplicates
            if documents:
                doc_ids = [doc.strip() for doc in documents.split(',')]
                duplicates = service.detect_duplicates(document_ids=doc_ids, threshold=threshold)
            else:
                duplicates = service.detect_duplicates(threshold=threshold)
            
            if not duplicates:
                rprint("[green]No duplicates found.[/green]")
                return
            
            # Display duplicates
            table = Table(title="Detected Duplicate Groups")
            table.add_column("Group ID", style="cyan")
            table.add_column("Documents", style="green")
            table.add_column("Similarity", style="yellow")
            
            for group in duplicates:
                table.add_row(
                    str(group['group_id']),
                    ', '.join(group['documents']),
                    f"{group['similarity']:.3f}"
                )
            
            console.print(table)
            
            if preview_only:
                return
        
        # Perform merge if not preview only
        if documents:
            # Manual merge
            doc_ids = [doc.strip() for doc in documents.split(',')]
            # Create a duplicate group for the manual selection
            from ..core.augmentation_service import DuplicateGroup
            duplicate_group = DuplicateGroup(
                group_id=1,
                documents=doc_ids,
                similarity=1.0,  # Manual selection
                merge_strategy=strategy
            )
            result = service.merge_duplicates([duplicate_group], keep_originals=keep_originals)
        elif auto_detect:
            # Auto-detected merge
            result = service.merge_duplicates(duplicates, keep_originals=keep_originals)
        else:
            rprint("[red]Error:[/red] Must specify either --auto-detect or --documents")
            raise typer.Exit(1)
        
        rprint(f"[green]Merge complete:[/green]")
        rprint(f"Merged groups: {result['merged_groups']}")
        rprint(f"Documents merged: {result['documents_merged']}")
        if 'new_document_id' in result:
            rprint(f"New document ID: {result['new_document_id']}")
            
    except Exception as e:
        logger.error(f"Failed to merge duplicates: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def feedback(
    ctx: typer.Context,
    chunk_id: str = typer.Option(..., "--chunk-id", help="Chunk ID to provide feedback on"),
    rating: str = typer.Option(..., "--rating", help="Rating: positive, negative, neutral"),
    reason: str = typer.Option(..., "--reason", help="Reason for the rating"),
    comment: Optional[str] = typer.Option(None, "--comment", help="Additional comment"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="User ID (optional for anonymous feedback)"),
) -> None:
    """Submit feedback for a knowledge base chunk."""
    try:
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would submit {rating} feedback for chunk '{chunk_id}'")
            return
        
        # Validate rating
        valid_ratings = ['positive', 'negative', 'neutral']
        if rating not in valid_ratings:
            rprint(f"[red]Error:[/red] Invalid rating. Must be one of: {', '.join(valid_ratings)}")
            raise typer.Exit(1)
        
        user_feedback = UserFeedback(
            chunk_id=chunk_id,
            rating=rating,
            reason=reason,
            comment=comment,
            user_id=user_id,
            timestamp=datetime.now()
        )
        
        service = _get_feedback_service()
        result = service.submit_feedback(user_feedback)
        
        if result['status'] == 'recorded':
            rprint(f"[green]Success:[/green] Feedback recorded: {result['feedback_id']}")
            if result.get('impact'):
                rprint(f"Impact: {result['impact']}")
            if result.get('flagged_for_review'):
                rprint(f"[yellow]Note:[/yellow] Marked for review")
        else:
            rprint(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def feedback_analytics(
    ctx: typer.Context,
    period: str = typer.Option("30d", "--period", help="Time period: 7d, 30d, 90d, 1y"),
    collection: Optional[str] = typer.Option(None, "--collection", help="Filter by collection"),
) -> None:
    """Display feedback analytics and insights."""
    try:
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would show feedback analytics for period '{period}'")
            return
        
        service = _get_feedback_service()
        analytics = service.get_feedback_analytics(period=period, collection=collection)
        
        # Display analytics in a table
        table = Table(title=f"Feedback Analytics - {period}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Feedback", str(analytics['total_feedback']))
        table.add_row("Positive Ratio", f"{analytics['positive_ratio']:.2%}")
        table.add_row("Negative Ratio", f"{analytics['negative_ratio']:.2%}")
        
        if 'top_issues' in analytics:
            table.add_row("Top Issues", ', '.join(analytics['top_issues']))
        
        if 'quality_trends' in analytics:
            trends = analytics['quality_trends']
            if len(trends) >= 2:
                trend_direction = "📈" if trends[-1] > trends[0] else "📉"
                table.add_row("Quality Trend", f"{trend_direction} {trends[-1]:.2f}")
        
        console.print(table)
        
    except Exception as e:
        logger.error(f"Failed to get feedback analytics: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


def export_feedback(
    ctx: typer.Context,
    output: str = typer.Option(..., "--output", help="Output file path"),
    format: str = typer.Option("csv", "--format", help="Export format: csv, json, excel"),
    date_range: Optional[str] = typer.Option(None, "--date-range", help="Date range: YYYY-MM-DD,YYYY-MM-DD"),
    rating_filter: Optional[str] = typer.Option(None, "--rating-filter", help="Filter by rating: positive, negative, neutral"),
) -> None:
    """Export feedback data to file."""
    try:
        if ctx.obj and ctx.obj.get('dry_run'):
            rprint(f"[blue]DRY RUN:[/blue] Would export feedback to '{output}' in {format} format")
            return
        
        # Parse date range if provided
        date_range_tuple = None
        if date_range:
            try:
                start_date, end_date = date_range.split(',')
                date_range_tuple = (start_date.strip(), end_date.strip())
            except ValueError:
                rprint(f"[red]Error:[/red] Invalid date range format. Use: YYYY-MM-DD,YYYY-MM-DD")
                raise typer.Exit(1)
        
        from ..core.feedback_service import FeedbackExportOptions
        export_options = FeedbackExportOptions(
            format=format,
            date_range=date_range_tuple,
            rating_filter=rating_filter,
            include_comments=True,
            include_metadata=False
        )
        
        service = _get_feedback_service()
        result = service.export_feedback(output, export_options)
        
        rprint(f"[green]Success:[/green] Exported {result['records_exported']} records to {result['export_file']}")
        rprint(f"Format: {result['format']}")
        
    except Exception as e:
        logger.error(f"Failed to export feedback: {e}")
        rprint(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) 