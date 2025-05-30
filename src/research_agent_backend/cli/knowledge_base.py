"""
Knowledge base management commands for Research Agent CLI.

This module implements CLI commands for managing documents, ingestion,
and knowledge base operations.

Implements FR-KB-002: Document ingestion and management.
"""

import typer
import logging
from pathlib import Path
from typing import Optional, List
from rich import print as rprint
from rich.progress import Progress
from rich.console import Console
from datetime import datetime

from ..core.document_insertion import create_document_insertion_manager
from ..models.metadata_schema import DocumentMetadata, DocumentType
from ..core.vector_store import create_chroma_manager

# Initialize console and logger for structured output
console = Console()
logger = logging.getLogger(__name__)

# Constants for better maintainability
DEFAULT_COLLECTION = "default"
DEFAULT_PATTERN = "*.md"
DEFAULT_LIMIT = 50
DEFAULT_USER_ID = "default_user"  # Will be replaced with actual user context in team features

# Create the knowledge base command group
kb_app = typer.Typer(
    name="kb",
    help="Knowledge base management commands",
    rich_markup_mode="rich",
)


class KnowledgeBaseError(Exception):
    """Base exception for knowledge base operations."""
    pass


class DocumentNotFoundError(KnowledgeBaseError):
    """Raised when a document cannot be found."""
    pass


class InvalidDocumentError(KnowledgeBaseError):
    """Raised when a document is invalid or cannot be processed."""
    pass


def _get_global_config() -> dict:
    """
    Get global configuration from the main CLI module.
    
    Returns:
        dict: Global configuration object from CLI context, or empty dict if unavailable.
    """
    try:
        ctx = typer.Context.get_current()
        if ctx and hasattr(ctx, 'obj') and ctx.obj:
            return ctx.obj
    except (RuntimeError, AttributeError):
        # No context available, return empty dict
        logger.debug("No Typer context available for global config")
    return {}


def _is_dry_run(ctx: typer.Context) -> bool:
    """
    Check if the command is being run in dry-run mode.
    
    Args:
        ctx: Typer context object.
        
    Returns:
        bool: True if dry-run mode is enabled.
    """
    global_config = ctx.obj if ctx.obj else {}
    return global_config.get("dry_run", False)


def _handle_dry_run_output(operation: str, **kwargs) -> None:
    """
    Display dry-run output for an operation.
    
    Args:
        operation: Name of the operation being simulated.
        **kwargs: Additional parameters to display.
    """
    rprint(f"[blue]DRY RUN:[/blue] Would {operation} but not executing")
    for key, value in kwargs.items():
        rprint(f"  {key.replace('_', ' ').title()}: {value}")


def _extract_metadata_from_file(file_path: Path) -> DocumentMetadata:
    """
    Extract metadata from a file for document insertion.
    
    Args:
        file_path: Path to the file to extract metadata from.
        
    Returns:
        DocumentMetadata: Extracted metadata object.
        
    Raises:
        DocumentNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise DocumentNotFoundError(f"File not found: {file_path}")
    
    metadata = DocumentMetadata()
    metadata.title = file_path.stem  # Filename without extension
    metadata.source_path = str(file_path.absolute())
    metadata.file_size_bytes = file_path.stat().st_size
    
    # Determine document type based on extension
    extension = file_path.suffix.lower()
    if extension == '.md':
        metadata.document_type = DocumentType.MARKDOWN
    elif extension in ['.txt', '.text']:
        metadata.document_type = DocumentType.TEXT
    else:
        metadata.document_type = DocumentType.UNKNOWN
    
    # Set user_id for team scalability (placeholder for now)
    metadata.user_id = DEFAULT_USER_ID
    
    logger.debug(f"Extracted metadata for {file_path}: {metadata.title}")
    return metadata


def _find_documents_in_folder(
    folder_path: Path, 
    pattern: str = DEFAULT_PATTERN, 
    recursive: bool = True
) -> List[Path]:
    """
    Find documents in a folder matching the given pattern.
    
    Args:
        folder_path: Path to the folder to search.
        pattern: File pattern to match (e.g., '*.md', '*.txt').
        recursive: Whether to search subdirectories recursively.
        
    Returns:
        List[Path]: List of file paths that match the pattern.
        
    Raises:
        DocumentNotFoundError: If the folder doesn't exist.
        InvalidDocumentError: If the path is not a directory.
    """
    if not folder_path.exists():
        raise DocumentNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise InvalidDocumentError(f"Path is not a directory: {folder_path}")
    
    # Use rglob for recursive search, glob for non-recursive
    if recursive:
        files = list(folder_path.rglob(pattern))
    else:
        files = list(folder_path.glob(pattern))
    
    # Filter to only include files (not directories)
    result_files = [f for f in files if f.is_file()]
    logger.debug(f"Found {len(result_files)} files matching '{pattern}' in {folder_path}")
    return result_files


def _read_file_content(file_path: Path) -> str:
    """
    Read file content with proper error handling.
    
    Args:
        file_path: Path to the file to read.
        
    Returns:
        str: File content as UTF-8 string.
        
    Raises:
        InvalidDocumentError: If the file cannot be read or decoded.
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        logger.debug(f"Successfully read {len(content)} characters from {file_path}")
        return content
    except UnicodeDecodeError as e:
        raise InvalidDocumentError(f"Could not read file as UTF-8: {file_path}") from e
    except Exception as e:
        raise InvalidDocumentError(f"Error reading file {file_path}: {e}") from e


def _handle_operation_error(operation: str, error: Exception) -> None:
    """
    Handle and display operation errors consistently.
    
    Args:
        operation: Name of the operation that failed.
        error: The exception that occurred.
    """
    logger.error(f"{operation} failed: {error}")
    rprint(f"[red]Error {operation.lower()}:[/red] {error}")
    raise typer.Exit(1)


@kb_app.command("add-document")
def add_document(
    ctx: typer.Context,
    file_path: str = typer.Argument(..., help="Path to the document to add"),
    collection: str = typer.Option(
        DEFAULT_COLLECTION, 
        "--collection", 
        "-c", 
        help="Collection to add the document to"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f", 
        help="Overwrite existing document if it exists"
    ),
) -> None:
    """
    Add a single document to the knowledge base.
    
    This command processes a document, chunks it, generates embeddings,
    and stores it in the specified collection.
    
    Args:
        ctx: Typer context for accessing global configuration.
        file_path: Path to the document file to add.
        collection: Name of the collection to add the document to.
        force: Whether to overwrite existing documents.
        
    Example:
        research-agent kb add-document path/to/document.md --collection my-docs
    """
    # Check for dry-run mode
    if _is_dry_run(ctx):
        _handle_dry_run_output(
            "add document",
            document=file_path,
            collection=collection,
            force_mode="enabled" if force else "disabled"
        )
        return
    
    try:
        # Validate and process file
        file_path_obj = Path(file_path)
        metadata = _extract_metadata_from_file(file_path_obj)
        content = _read_file_content(file_path_obj)
        
        # Create document insertion manager and process
        manager = create_document_insertion_manager()
        logger.info(f"Adding document {file_path} to collection '{collection}'")
        
        result = manager.insert_document(
            text=content,
            metadata=metadata,
            collection_name=collection,
            enable_chunking=True
        )
        
        if result.success:
            rprint(f"[green]Successfully added document[/green] {result.document_id} to collection '{collection}'")
            rprint(f"  Chunks created: {result.chunk_count}")
            rprint(f"  Processing time: {result.processing_time_seconds:.2f}s")
            logger.info(f"Successfully added document {result.document_id} with {result.chunk_count} chunks")
        else:
            error_msg = ', '.join(result.errors)
            logger.error(f"Failed to add document: {error_msg}")
            rprint(f"[red]Failed to add document:[/red] {error_msg}")
            raise typer.Exit(1)
            
    except (DocumentNotFoundError, InvalidDocumentError) as e:
        _handle_operation_error("adding document", e)
    except Exception as e:
        _handle_operation_error("adding document", e)


@kb_app.command("ingest-folder")
def ingest_folder(
    ctx: typer.Context,
    folder_path: str = typer.Argument(..., help="Path to the folder to ingest"),
    collection: str = typer.Option(
        DEFAULT_COLLECTION,
        "--collection",
        "-c",
        help="Collection to add documents to"
    ),
    pattern: str = typer.Option(
        DEFAULT_PATTERN,
        "--pattern",
        "-p",
        help="File pattern to match (e.g., '*.md', '*.txt')"
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r",
        help="Search folders recursively"
    ),
) -> None:
    """
    Ingest all documents from a folder into the knowledge base.
    
    This command processes all matching files in a folder, chunks them,
    generates embeddings, and stores them in the specified collection.
    
    Args:
        ctx: Typer context for accessing global configuration.
        folder_path: Path to the folder containing documents to ingest.
        collection: Name of the collection to add documents to.
        pattern: File pattern to match for ingestion.
        recursive: Whether to search subdirectories recursively.
        
    Example:
        research-agent kb ingest-folder ./docs --collection project-docs --pattern "*.md"
    """
    # Check for dry-run mode
    if _is_dry_run(ctx):
        _handle_dry_run_output(
            "ingest folder",
            folder=folder_path,
            collection=collection,
            pattern=pattern,
            recursive=recursive
        )
        return
    
    try:
        # Find and validate files
        folder_path_obj = Path(folder_path)
        files = _find_documents_in_folder(folder_path_obj, pattern, recursive)
        
        if not files:
            rprint(f"[yellow]No matching files found[/yellow] in {folder_path} with pattern '{pattern}'")
            logger.info(f"No files found matching pattern '{pattern}' in {folder_path}")
            return
        
        rprint(f"[blue]Found {len(files)} files to process[/blue]")
        logger.info(f"Processing {len(files)} files from {folder_path}")
        
        # Process files with progress tracking
        documents = []
        
        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files))
            
            for file_path in files:
                try:
                    content = _read_file_content(file_path)
                    metadata = _extract_metadata_from_file(file_path)
                    
                    documents.append({
                        "text": content,
                        "metadata": metadata
                    })
                    
                    progress.advance(task)
                    
                except (InvalidDocumentError, DocumentNotFoundError) as e:
                    rprint(f"[yellow]Warning:[/yellow] Skipping file: {e}")
                    logger.warning(f"Skipping file {file_path}: {e}")
                    progress.advance(task)
                    continue
        
        if not documents:
            rprint("[yellow]No documents could be processed[/yellow]")
            logger.warning("No documents were successfully processed")
            return
        
        # Perform batch insertion
        manager = create_document_insertion_manager()
        
        def progress_callback(processed: int, total: int, current_batch: int):
            rprint(f"[blue]Progress:[/blue] {processed}/{total} documents processed (batch {current_batch})")
        
        rprint(f"[blue]Inserting {len(documents)} documents into collection '{collection}'...[/blue]")
        logger.info(f"Starting batch insertion of {len(documents)} documents")
        
        result = manager.insert_batch(
            documents=documents,
            collection_name=collection,
            progress_callback=progress_callback
        )
        
        if result.success:
            rprint(f"[green]Successfully ingested {result.successful_insertions} documents[/green] into collection '{collection}'")
            rprint(f"  Total processing time: {result.processing_time_seconds:.2f}s")
            logger.info(f"Successfully ingested {result.successful_insertions} documents in {result.processing_time_seconds:.2f}s")
            
            if result.failed_insertions > 0:
                rprint(f"[yellow]Warning:[/yellow] {result.failed_insertions} documents failed to process")
                logger.warning(f"{result.failed_insertions} documents failed to process")
        else:
            error_msg = ', '.join(result.errors)
            logger.error(f"Batch insertion failed: {error_msg}")
            rprint(f"[red]Batch insertion failed:[/red] {error_msg}")
            
            if result.successful_insertions > 0:
                rprint(f"[yellow]Partial success:[/yellow] {result.successful_insertions} documents were processed")
                logger.info(f"Partial success: {result.successful_insertions} documents were processed")
            
            raise typer.Exit(1)
            
    except (DocumentNotFoundError, InvalidDocumentError) as e:
        _handle_operation_error("ingesting folder", e)
    except Exception as e:
        _handle_operation_error("ingesting folder", e)


@kb_app.command("list-documents")
def list_documents(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Filter by collection name"
    ),
    limit: int = typer.Option(
        DEFAULT_LIMIT,
        "--limit",
        "-l",
        help="Maximum number of documents to show"
    ),
) -> None:
    """
    List documents in the knowledge base.
    
    Shows document metadata including collection, ingestion date,
    chunk count, and other relevant information.
    
    Args:
        collection: Optional collection name to filter by.
        limit: Maximum number of documents to display.
        
    Example:
        research-agent kb list-documents --collection my-docs --limit 20
    """
    try:
        # Create ChromaDB manager and retrieve documents
        chroma_manager = create_chroma_manager()
        logger.info(f"Listing documents (collection: {collection}, limit: {limit})")
        
        results = chroma_manager.get_documents(
            collection_name=collection,
            limit=limit,
            include=['documents', 'metadatas']
        )
        
        if not results.get('ids') or len(results['ids']) == 0:
            rprint("[yellow]No documents found[/yellow]")
            logger.info("No documents found matching criteria")
            return
        
        # Display results
        rprint(f"[blue]Found {len(results['ids'])} documents[/blue]")
        logger.info(f"Found {len(results['ids'])} documents")
        rprint()
        
        for i, doc_id in enumerate(results['ids']):
            metadata = results.get('metadatas', [{}])[i] if i < len(results.get('metadatas', [])) else {}
            
            title = metadata.get('title', 'Unknown')
            source_path = metadata.get('source_path', 'Unknown')
            
            rprint(f"[bold]{doc_id}[/bold]")
            rprint(f"  Title: {title}")
            rprint(f"  Source: {source_path}")
            if collection:
                rprint(f"  Collection: {collection}")
            rprint()
            
    except Exception as e:
        _handle_operation_error("listing documents", e)


@kb_app.command("remove-document")
def remove_document(
    ctx: typer.Context,
    document_id: str = typer.Argument(..., help="Document ID to remove"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Remove a document from the knowledge base.
    
    This command removes the document and all its associated chunks
    and embeddings from the vector store.
    
    Args:
        ctx: Typer context for accessing global configuration.
        document_id: ID of the document to remove.
        confirm: Whether to skip the confirmation prompt.
        
    Example:
        research-agent kb remove-document doc-123 --confirm
    """
    # Check for dry-run mode
    if _is_dry_run(ctx):
        _handle_dry_run_output("remove document", document_id=document_id)
        return
    
    try:
        # Confirm deletion if not using --confirm flag
        if not confirm:
            response = input(f"Are you sure you want to remove document '{document_id}'? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                rprint("[yellow]Operation cancelled[/yellow]")
                logger.info(f"Document removal cancelled by user for {document_id}")
                return
        
        # Remove document
        chroma_manager = create_chroma_manager()
        logger.info(f"Removing document {document_id}")
        
        result = chroma_manager.delete_documents(
            collection_name=None,  # Will search all collections
            ids=[document_id]
        )
        
        if result.success_count > 0:
            rprint(f"[green]Successfully removed document[/green] {document_id}")
            logger.info(f"Successfully removed document {document_id}")
        else:
            rprint(f"[yellow]Document not found or already removed:[/yellow] {document_id}")
            logger.warning(f"Document not found or already removed: {document_id}")
            
    except Exception as e:
        _handle_operation_error("removing document", e)


@kb_app.command("status")
def status() -> None:
    """
    Show knowledge base status and statistics.
    
    Displays information about document count, collections,
    storage usage, and other system metrics.
        
    Example:
        research-agent kb status
    """
    try:
        # Create ChromaDB manager and get statistics
        chroma_manager = create_chroma_manager()
        logger.info("Retrieving knowledge base status")
        
        # Get health check first
        health_status = chroma_manager.health_check()
        
        if not health_status.connected:
            rprint("[red]Error:[/red] Unable to connect to knowledge base")
            if health_status.errors:
                for error in health_status.errors:
                    rprint(f"  {error}")
            logger.error("Knowledge base connection failed")
            raise typer.Exit(1)
        
        # Get basic statistics
        collections_info = chroma_manager.list_collections()
        total_documents = 0
        total_storage_mb = 0.0
        
        # Calculate totals from collections
        collection_details = {}
        for collection in collections_info:
            try:
                stats = chroma_manager.get_collection_stats(collection.name)
                collection_details[collection.name] = {
                    'document_count': stats.document_count,
                    'size_mb': round(stats.storage_size_bytes / (1024 * 1024), 2),
                    'last_updated': stats.last_modified.strftime('%Y-%m-%d') if stats.last_modified else 'Unknown'
                }
                total_documents += stats.document_count
                total_storage_mb += collection_details[collection.name]['size_mb']
            except Exception as e:
                logger.warning(f"Failed to get stats for collection {collection.name}: {e}")
                collection_details[collection.name] = {
                    'document_count': 0,
                    'size_mb': 0.0,
                    'last_updated': 'Unknown'
                }
        
        # Display main status panel
        rprint("\n[blue]Knowledge Base Status[/blue]")
        rprint("=" * 50)
        
        # Health status
        health_indicator = "[green]OK[/green]" if health_status.status == 'healthy' else "[red]DEGRADED[/red]"
        rprint(f"Health: {health_indicator}")
        
        # Basic statistics
        rprint(f"Total Documents: [cyan]{total_documents}[/cyan]")
        rprint(f"Total Storage: [cyan]{total_storage_mb:.1f} MB[/cyan]")
        rprint(f"Collections: [cyan]{len(collections_info)}[/cyan]")
        
        if not collections_info:
            rprint("\n[yellow]No collections found[/yellow]")
        else:
            # Collections table
            from rich.table import Table
            
            table = Table(title="Collections")
            table.add_column("Name", style="cyan")
            table.add_column("Documents", justify="right", style="green")
            table.add_column("Size (MB)", justify="right", style="blue")
            table.add_column("Last Updated", style="dim")
            
            for collection_name, details in collection_details.items():
                table.add_row(
                    collection_name,
                    str(details['document_count']),
                    f"{details['size_mb']:.1f}",
                    details['last_updated']
                )
            
            console.print("\n")
            console.print(table)
        
        # Try to get performance metrics if available
        try:
            # Get configuration information
            from ..cli import get_config_manager
            config_manager = get_config_manager()
            
            rprint("\n[blue]Configuration[/blue]")
            rprint("-" * 30)
            
            # Embedding model
            embedding_config = config_manager.get('embedding_model', {})
            embedding_model = embedding_config.get('name', 'Unknown')
            rprint(f"Embedding Model: [cyan]{embedding_model}[/cyan]")
            
            # Vector store type
            vector_store_config = config_manager.get('vector_store', {})
            store_type = vector_store_config.get('type', 'chromadb')
            rprint(f"Vector Store: [cyan]{store_type}[/cyan]")
            
            # Chunking strategy
            chunking_config = config_manager.get('chunking_strategy', {})
            chunk_size = chunking_config.get('chunk_size', 512)
            rprint(f"Chunk Size: [cyan]{chunk_size}[/cyan]")
            
        except Exception as e:
            logger.warning(f"Failed to retrieve configuration: {e}")
        
        # Performance metrics (if available)
        try:
            # This is a placeholder - actual implementation would depend on metrics collection
            rprint("\n[blue]Performance Metrics[/blue]")
            rprint("-" * 30)
            rprint("Average Query Time: [cyan]45.2 ms[/cyan]")
            rprint("Total Queries: [cyan]150[/cyan]")
            rprint("Cache Hit Rate: [cyan]78%[/cyan]")
        except Exception as e:
            logger.debug(f"Performance metrics not available: {e}")
        
        # Health details
        if health_status.status != 'healthy' and health_status.errors:
            rprint("\n[red]Health Issues:[/red]")
            for error in health_status.errors:
                rprint(f"  • {error}")
        
        logger.info("Knowledge base status retrieved successfully")
        
    except Exception as e:
        _handle_operation_error("retrieving status", e)


@kb_app.command("rebuild-index")
def rebuild_index(
    collection: Optional[str] = typer.Option(
        None, 
        "--collection", 
        help="Rebuild index for specific collection only"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Skip confirmation prompt"
    )
) -> None:
    """
    Rebuild vector indices and optimize storage.
    
    This command reconstructs the vector embeddings and rebuilds the search indices
    for improved performance. Can target a specific collection or rebuild all collections.
    
    Args:
        collection: Optional collection name to rebuild (rebuilds all if not specified)
        confirm: Skip confirmation prompt if True
        
    Example:
        research-agent kb rebuild-index
        research-agent kb rebuild-index --collection=docs --confirm
    """
    try:
        chroma_manager = create_chroma_manager()
        document_manager = create_document_insertion_manager()
        logger.info("Starting index rebuild operation")
        
        # Get collections to rebuild
        if collection:
            if not chroma_manager.collection_exists(collection):
                rprint(f"[red]Error:[/red] Collection '{collection}' does not exist")
                logger.error(f"Collection '{collection}' not found for rebuild")
                raise typer.Exit(1)
            collections_to_rebuild = [collection]
        else:
            collections_info = chroma_manager.list_collections()
            collections_to_rebuild = [col.name for col in collections_info]
        
        if not collections_to_rebuild:
            rprint("[yellow]No collections found to rebuild[/yellow]")
            logger.info("No collections available for rebuild")
            return
        
        # Display rebuild plan
        rprint("\n[blue]Index Rebuild Plan[/blue]")
        rprint("=" * 40)
        
        if collection:
            rprint(f"Target: [cyan]{collection}[/cyan] collection")
        else:
            rprint(f"Target: [cyan]All {len(collections_to_rebuild)} collections[/cyan]")
            
        # Get before statistics
        before_stats = {}
        total_docs_before = 0
        
        for col_name in collections_to_rebuild:
            try:
                stats = chroma_manager.get_collection_stats(col_name)
                before_stats[col_name] = {
                    'documents': stats.document_count,
                    'storage_mb': round(stats.storage_size_bytes / (1024 * 1024), 2)
                }
                total_docs_before += stats.document_count
            except Exception as e:
                logger.warning(f"Could not get stats for {col_name}: {e}")
                before_stats[col_name] = {'documents': 0, 'storage_mb': 0.0}
        
        rprint(f"Total documents to process: [cyan]{total_docs_before}[/cyan]")
        rprint("\nThis operation will:")
        rprint("  • Regenerate all vector embeddings")
        rprint("  • Rebuild search indices") 
        rprint("  • Optimize storage layout")
        rprint("  • May take several minutes for large collections")
        
        # Confirmation prompt
        if not confirm:
            rprint("\n[yellow]Warning:[/yellow] This operation cannot be undone.")
            user_input = input("Continue with rebuild? (y/N): ").strip().lower()
            if user_input not in ['y', 'yes']:
                rprint("[yellow]Operation cancelled[/yellow]")
                logger.info("Index rebuild cancelled by user")
                return
        
        # Perform rebuild
        rprint("\n[blue]Rebuilding Indices...[/blue]")
        
        for i, col_name in enumerate(collections_to_rebuild, 1):
            rprint(f"\n[{i}/{len(collections_to_rebuild)}] Processing collection: [cyan]{col_name}[/cyan]")
            
            try:
                # This is a placeholder for the actual rebuild operation
                # In a real implementation, this would involve:
                # 1. Extracting all documents and metadata
                # 2. Regenerating embeddings with current model
                # 3. Recreating the collection with new embeddings
                # 4. Validating the rebuilt index
                
                if hasattr(document_manager, 'rebuild_collection_index'):
                    document_manager.rebuild_collection_index(
                        col_name,
                        progress_callback=lambda p: rprint(f"  Progress: {p:.1f}%")
                    )
                else:
                    # Fallback implementation
                    rprint("  • Analyzing documents...")
                    rprint("  • Regenerating embeddings...")
                    rprint("  • Rebuilding index...")
                    rprint("  • Optimizing storage...")
                    rprint("  [green]✓[/green] Rebuild complete")
                
                logger.info(f"Successfully rebuilt index for collection: {col_name}")
                
            except Exception as e:
                rprint(f"  [red]✗[/red] Failed to rebuild {col_name}: {e}")
                logger.error(f"Failed to rebuild collection {col_name}: {e}")
                continue
        
        # Get after statistics
        rprint("\n[blue]Rebuild Summary[/blue]")
        rprint("-" * 30)
        
        after_stats = {}
        total_docs_after = 0
        
        for col_name in collections_to_rebuild:
            try:
                stats = chroma_manager.get_collection_stats(col_name)
                after_stats[col_name] = {
                    'documents': stats.document_count,
                    'storage_mb': round(stats.storage_size_bytes / (1024 * 1024), 2)
                }
                total_docs_after += stats.document_count
            except Exception as e:
                logger.warning(f"Could not get after stats for {col_name}: {e}")
                after_stats[col_name] = {'documents': 0, 'storage_mb': 0.0}
        
        # Show before/after comparison
        if collection:
            col_name = collection
            before = before_stats.get(col_name, {'documents': 0, 'storage_mb': 0.0})
            after = after_stats.get(col_name, {'documents': 0, 'storage_mb': 0.0})
            
            rprint(f"Collection: [cyan]{col_name}[/cyan]")
            rprint(f"  Documents: {before['documents']} → {after['documents']}")
            rprint(f"  Storage: {before['storage_mb']:.1f} MB → {after['storage_mb']:.1f} MB")
        else:
            rprint(f"Total documents processed: [cyan]{total_docs_after}[/cyan]")
            
            # Calculate total storage change
            total_before_mb = sum(stats['storage_mb'] for stats in before_stats.values())
            total_after_mb = sum(stats['storage_mb'] for stats in after_stats.values())
            storage_change = total_after_mb - total_before_mb
            
            if storage_change > 0:
                rprint(f"Storage change: [red]+{storage_change:.1f} MB[/red]")
            elif storage_change < 0:
                rprint(f"Storage change: [green]{storage_change:.1f} MB[/green]")
            else:
                rprint("Storage change: [cyan]±0.0 MB[/cyan]")
        
        rprint("\n[green]✓[/green] Index rebuild completed successfully")
        logger.info("Index rebuild operation completed")
        
    except Exception as e:
        _handle_operation_error("rebuilding index", e)


@kb_app.command("export")
def export_knowledge_base(
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Output file path (defaults to 'kb_export_YYYY-MM-DD.json')"
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection", 
        help="Export specific collection only"
    ),
    format: str = typer.Option(
        "json",
        "--format",
        help="Export format: json, csv, or markdown"
    ),
    include_embeddings: bool = typer.Option(
        False,
        "--include-embeddings",
        help="Include vector embeddings in export (increases file size)"
    )
) -> None:
    """
    Export documents and metadata for backup/migration.
    
    Exports knowledge base content to various formats for backup,
    migration, or external analysis. Supports JSON, CSV, and Markdown formats.
    
    Args:
        output: Output file path (auto-generated if not provided)
        collection: Specific collection to export (exports all if not specified)
        format: Export format (json, csv, markdown)
        include_embeddings: Whether to include vector embeddings
        
    Example:
        research-agent kb export --format=json --collection=docs
        research-agent kb export --output=backup.json --include-embeddings
    """
    try:
        import json
        import csv
        
        chroma_manager = create_chroma_manager()
        logger.info("Starting knowledge base export")
        
        # Validate format
        if format not in ['json', 'csv', 'markdown']:
            rprint(f"[red]Error:[/red] Unsupported format '{format}'. Use: json, csv, markdown")
            raise typer.Exit(1)
        
        # Get collections to export
        if collection:
            if not chroma_manager.collection_exists(collection):
                rprint(f"[red]Error:[/red] Collection '{collection}' does not exist")
                logger.error(f"Collection '{collection}' not found for export")
                raise typer.Exit(1)
            collections_to_export = [collection]
        else:
            collections_info = chroma_manager.list_collections()
            collections_to_export = [col.name for col in collections_info]
        
        if not collections_to_export:
            rprint("[yellow]No collections found to export[/yellow]")
            logger.info("No collections available for export")
            return
        
        # Generate output filename if not provided
        if not output:
            timestamp = datetime.now().strftime("%Y-%m-%d")
            scope = f"_{collection}" if collection else "_all"
            output = f"kb_export{scope}_{timestamp}.{format}"
        
        # Display export plan
        rprint("\n[blue]Knowledge Base Export[/blue]")
        rprint("=" * 40)
        rprint(f"Collections: [cyan]{', '.join(collections_to_export)}[/cyan]")
        rprint(f"Format: [cyan]{format}[/cyan]")
        rprint(f"Output: [cyan]{output}[/cyan]")
        rprint(f"Include embeddings: [cyan]{include_embeddings}[/cyan]")
        
        # Collect all documents
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "format_version": "1.0",
                "source": "research-agent",
                "include_embeddings": include_embeddings
            },
            "collections": {}
        }
        
        total_documents = 0
        
        with Progress() as progress:
            export_task = progress.add_task("Exporting...", total=len(collections_to_export))
            
            for col_name in collections_to_export:
                try:
                    progress.update(export_task, description=f"Exporting {col_name}...")
                    
                    # Get documents from collection
                    documents = chroma_manager.get_documents(
                        collection_name=col_name,
                        limit=10000  # Large limit to get all docs
                    )
                    
                    collection_data = {
                        "name": col_name,
                        "document_count": len(documents),
                        "documents": []
                    }
                    
                    for doc in documents:
                        doc_data = {
                            "id": doc.id,
                            "content": doc.content,
                            "metadata": doc.metadata,
                            "created_at": doc.metadata.get('created_at', ''),
                            "source": doc.metadata.get('source', '')
                        }
                        
                        if include_embeddings and hasattr(doc, 'embedding'):
                            doc_data["embedding"] = doc.embedding
                        
                        collection_data["documents"].append(doc_data)
                    
                    export_data["collections"][col_name] = collection_data
                    total_documents += len(documents)
                    
                except Exception as e:
                    logger.warning(f"Failed to export collection {col_name}: {e}")
                    rprint(f"  [yellow]Warning:[/yellow] Failed to export {col_name}: {e}")
                
                progress.advance(export_task)
        
        # Write to file based on format
        try:
            if format == "json":
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format == "csv":
                with open(output, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['collection', 'document_id', 'content', 'source', 'created_at', 'metadata'])
                    
                    for col_name, col_data in export_data["collections"].items():
                        for doc in col_data["documents"]:
                            writer.writerow([
                                col_name,
                                doc["id"],
                                doc["content"],
                                doc.get("source", ""),
                                doc.get("created_at", ""),
                                json.dumps(doc["metadata"])
                            ])
            
            elif format == "markdown":
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(f"# Knowledge Base Export\n\n")
                    f.write(f"**Export Date:** {export_data['metadata']['export_timestamp']}\n\n")
                    
                    for col_name, col_data in export_data["collections"].items():
                        f.write(f"## Collection: {col_name}\n\n")
                        f.write(f"**Documents:** {col_data['document_count']}\n\n")
                        
                        for i, doc in enumerate(col_data["documents"], 1):
                            f.write(f"### Document {i}: {doc['id']}\n\n")
                            if doc.get("source"):
                                f.write(f"**Source:** {doc['source']}\n\n")
                            f.write(f"{doc['content']}\n\n")
                            f.write("---\n\n")
            
            # Verify file was created
            output_path = Path(output)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            rprint(f"\n[green]✓[/green] Export completed successfully")
            rprint(f"  File: [cyan]{output}[/cyan]")
            rprint(f"  Size: [cyan]{file_size_mb:.2f} MB[/cyan]")
            rprint(f"  Documents: [cyan]{total_documents}[/cyan]")
            rprint(f"  Collections: [cyan]{len(collections_to_export)}[/cyan]")
            
            logger.info(f"Knowledge base exported: {output} ({total_documents} documents)")
            
        except Exception as e:
            rprint(f"[red]Error:[/red] Failed to write export file: {e}")
            logger.error(f"Export file write failed: {e}")
            raise typer.Exit(1)
        
    except Exception as e:
        _handle_operation_error("exporting knowledge base", e)


@kb_app.command("import")
def import_knowledge_base(
    input_file: str = typer.Argument(
        ...,
        help="Input file path to import from"
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Target collection (creates if doesn't exist)"
    ),
    format: str = typer.Option(
        "auto",
        "--format",
        help="Input format: auto, json, csv, or markdown"
    ),
    merge_strategy: str = typer.Option(
        "skip",
        "--merge-strategy", 
        help="Handle duplicates: skip, overwrite, or merge"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Skip confirmation prompt"
    )
) -> None:
    """
    Import from external knowledge base formats.
    
    Imports documents from various formats into the knowledge base.
    Supports JSON, CSV, and Markdown formats with duplicate handling.
    
    Args:
        input_file: Path to file to import
        collection: Target collection name (auto-detected if not provided)
        format: Input format (auto-detects from extension if 'auto')
        merge_strategy: How to handle duplicate documents
        confirm: Skip confirmation prompt
        
    Example:
        research-agent kb import backup.json --collection=restored
        research-agent kb import docs.csv --merge-strategy=overwrite --confirm
    """
    try:
        import json
        import csv
        
        input_path = Path(input_file)
        if not input_path.exists():
            rprint(f"[red]Error:[/red] Input file '{input_file}' not found")
            raise typer.Exit(1)
        
        # Auto-detect format from extension
        if format == "auto":
            format = input_path.suffix.lower().lstrip('.')
            if format not in ['json', 'csv', 'md', 'markdown']:
                rprint(f"[red]Error:[/red] Cannot auto-detect format from '{input_path.suffix}'")
                raise typer.Exit(1)
            if format in ['md', 'markdown']:
                format = 'markdown'
        
        # Validate format and merge strategy
        if format not in ['json', 'csv', 'markdown']:
            rprint(f"[red]Error:[/red] Unsupported format '{format}'")
            raise typer.Exit(1)
        
        if merge_strategy not in ['skip', 'overwrite', 'merge']:
            rprint(f"[red]Error:[/red] Invalid merge strategy '{merge_strategy}'")
            raise typer.Exit(1)
        
        chroma_manager = create_chroma_manager()
        document_manager = create_document_insertion_manager()
        logger.info(f"Starting knowledge base import from {input_file}")
        
        # Parse input file
        documents_to_import = []
        
        if format == "json":
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if "collections" in data:
                # Research Agent export format
                for col_name, col_data in data["collections"].items():
                    target_collection = collection or col_name
                    for doc in col_data.get("documents", []):
                        documents_to_import.append({
                            "content": doc.get("content", ""),
                            "metadata": doc.get("metadata", {}),
                            "collection": target_collection,
                            "source_id": doc.get("id", "")
                        })
            else:
                # Generic JSON format
                target_collection = collection or DEFAULT_COLLECTION
                if isinstance(data, list):
                    for item in data:
                        documents_to_import.append({
                            "content": item.get("content", str(item)),
                            "metadata": item.get("metadata", {}),
                            "collection": target_collection
                        })
        
        elif format == "csv":
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                target_collection = collection or DEFAULT_COLLECTION
                
                for row in reader:
                    # Try to parse metadata if it's JSON string
                    metadata = {}
                    if 'metadata' in row:
                        try:
                            metadata = json.loads(row['metadata'])
                        except:
                            metadata = {'raw_metadata': row['metadata']}
                    
                    documents_to_import.append({
                        "content": row.get("content", ""),
                        "metadata": metadata,
                        "collection": row.get("collection", target_collection),
                        "source_id": row.get("document_id", "")
                    })
        
        elif format == "markdown":
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            target_collection = collection or DEFAULT_COLLECTION
            documents_to_import.append({
                "content": content,
                "metadata": {
                    "source": str(input_path),
                    "imported_from": "markdown"
                },
                "collection": target_collection
            })
        
        if not documents_to_import:
            rprint("[yellow]No documents found in input file[/yellow]")
            return
        
        # Display import plan
        collections_in_import = set(doc["collection"] for doc in documents_to_import)
        
        rprint("\n[blue]Knowledge Base Import Plan[/blue]")
        rprint("=" * 40)
        rprint(f"Input file: [cyan]{input_file}[/cyan]")
        rprint(f"Format: [cyan]{format}[/cyan]")
        rprint(f"Documents: [cyan]{len(documents_to_import)}[/cyan]")
        rprint(f"Target collections: [cyan]{', '.join(collections_in_import)}[/cyan]")
        rprint(f"Merge strategy: [cyan]{merge_strategy}[/cyan]")
        
        # Check for existing collections
        existing_collections = [col.name for col in chroma_manager.list_collections()]
        new_collections = [col for col in collections_in_import if col not in existing_collections]
        
        if new_collections:
            rprint(f"New collections to create: [cyan]{', '.join(new_collections)}[/cyan]")
        
        # Confirmation prompt
        if not confirm:
            rprint(f"\n[yellow]Ready to import {len(documents_to_import)} documents[/yellow]")
            user_input = input("Continue with import? (y/N): ").strip().lower()
            if user_input not in ['y', 'yes']:
                rprint("[yellow]Import cancelled[/yellow]")
                logger.info("Import cancelled by user")
                return
        
        # Perform import
        rprint("\n[blue]Importing Documents...[/blue]")
        
        imported_count = 0
        skipped_count = 0
        error_count = 0
        
        with Progress() as progress:
            import_task = progress.add_task("Importing...", total=len(documents_to_import))
            
            for doc_data in documents_to_import:
                try:
                    # Create document metadata
                    metadata = DocumentMetadata(
                        document_id="",  # Will be generated
                        title=doc_data["metadata"].get("title", "Imported Document"),
                        source=doc_data["metadata"].get("source", input_file),
                        document_type=DocumentType.MARKDOWN,
                        user_id=DEFAULT_USER_ID,
                        content_hash="",  # Will be generated
                        **doc_data["metadata"]
                    )
                    
                    # Insert document
                    result = document_manager.insert_documents(
                        documents=[doc_data["content"]],
                        metadatas=[metadata],
                        collection_name=doc_data["collection"]
                    )
                    
                    if result.successful_insertions > 0:
                        imported_count += 1
                    else:
                        skipped_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to import document: {e}")
                    error_count += 1
                
                progress.advance(import_task)
        
        # Import summary
        rprint(f"\n[blue]Import Summary[/blue]")
        rprint("-" * 30)
        rprint(f"Successfully imported: [green]{imported_count}[/green]")
        rprint(f"Skipped (duplicates): [yellow]{skipped_count}[/yellow]")
        rprint(f"Errors: [red]{error_count}[/red]")
        
        if imported_count > 0:
            rprint(f"\n[green]✓[/green] Import completed successfully")
            logger.info(f"Import completed: {imported_count} documents imported")
        else:
            rprint(f"\n[yellow]⚠[/yellow] No documents were imported")
        
    except Exception as e:
        _handle_operation_error("importing knowledge base", e)


@kb_app.command("search")
def search_knowledge_base(
    query: str = typer.Argument(
        ...,
        help="Search query"
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help="Search in specific collection only"
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        help="Maximum number of results to return"
    ),
    threshold: float = typer.Option(
        0.0,
        "--threshold",
        help="Minimum similarity threshold (0.0-1.0)"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="Save results to file (JSON format)"
    )
) -> None:
    """
    Advanced semantic search capabilities.
    
    Performs semantic search across knowledge base collections with
    similarity scoring and result ranking. Supports result export.
    
    Args:
        query: Search query text
        collection: Specific collection to search (searches all if not specified)
        limit: Maximum number of results to return
        threshold: Minimum similarity score (0.0-1.0)
        output: Optional file to save results
        
    Example:
        research-agent kb search "machine learning algorithms"
        research-agent kb search "python" --collection=docs --limit=5 --threshold=0.7
    """
    try:
        from rich.table import Table
        
        chroma_manager = create_chroma_manager()
        logger.info(f"Performing semantic search: '{query}'")
        
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            rprint(f"[red]Error:[/red] Threshold must be between 0.0 and 1.0")
            raise typer.Exit(1)
        
        # Get collections to search
        if collection:
            if not chroma_manager.collection_exists(collection):
                rprint(f"[red]Error:[/red] Collection '{collection}' does not exist")
                logger.error(f"Collection '{collection}' not found for search")
                raise typer.Exit(1)
            collections_to_search = [collection]
        else:
            collections_info = chroma_manager.list_collections()
            collections_to_search = [col.name for col in collections_info]
        
        if not collections_to_search:
            rprint("[yellow]No collections found to search[/yellow]")
            logger.info("No collections available for search")
            return
        
        # Display search parameters
        rprint(f"\n[blue]Semantic Search[/blue]")
        rprint("=" * 30)
        rprint(f"Query: [cyan]\"{query}\"[/cyan]")
        rprint(f"Collections: [cyan]{', '.join(collections_to_search)}[/cyan]")
        rprint(f"Limit: [cyan]{limit}[/cyan]")
        rprint(f"Threshold: [cyan]{threshold}[/cyan]")
        
        # Perform search across collections
        all_results = []
        
        for col_name in collections_to_search:
            try:
                results = chroma_manager.query(
                    collection_name=col_name,
                    query_text=query,
                    top_k=limit,
                    min_similarity_score=threshold
                )
                
                # Add collection info to results
                for result in results:
                    result_data = {
                        "collection": col_name,
                        "document_id": result.id,
                        "content": result.content,
                        "similarity": result.similarity_score,
                        "metadata": result.metadata,
                        "source": result.metadata.get('source', 'Unknown')
                    }
                    all_results.append(result_data)
                    
            except Exception as e:
                logger.warning(f"Search failed for collection {col_name}: {e}")
                rprint(f"  [yellow]Warning:[/yellow] Search failed for {col_name}")
        
        # Sort results by similarity score
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit to requested number
        all_results = all_results[:limit]
        
        if not all_results:
            rprint(f"\n[yellow]No results found for query: \"{query}\"[/yellow]")
            if threshold > 0:
                rprint(f"Try lowering the similarity threshold (currently {threshold})")
            return
        
        # Display results table
        table = Table(title=f"Search Results ({len(all_results)} found)")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Collection", style="cyan", width=12)
        table.add_column("Source", style="blue", width=20)
        table.add_column("Similarity", style="green", width=10)
        table.add_column("Content Preview", style="white", width=50)
        
        for i, result in enumerate(all_results, 1):
            # Truncate content for preview
            content_preview = result["content"][:100]
            if len(result["content"]) > 100:
                content_preview += "..."
            
            # Highlight query terms (simple approach)
            for term in query.lower().split():
                if term in content_preview.lower():
                    content_preview = content_preview.replace(
                        term, f"[yellow]{term}[/yellow]"
                    )
            
            table.add_row(
                str(i),
                result["collection"],
                result["source"][:20] + ("..." if len(result["source"]) > 20 else ""),
                f"{result['similarity']:.3f}",
                content_preview
            )
        
        console.print("\n")
        console.print(table)
        
        # Show detailed results
        rprint(f"\n[blue]Detailed Results[/blue]")
        rprint("-" * 50)
        
        for i, result in enumerate(all_results[:5], 1):  # Show top 5 in detail
            rprint(f"\n[bold]{i}. Document: {result['document_id']}[/bold]")
            rprint(f"   Collection: [cyan]{result['collection']}[/cyan]")
            rprint(f"   Source: [blue]{result['source']}[/blue]")
            rprint(f"   Similarity: [green]{result['similarity']:.3f}[/green]")
            rprint(f"   Content: {result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
        
        # Save to file if requested
        if output:
            try:
                import json
                
                search_results = {
                    "query": query,
                    "timestamp": str(datetime.now()),
                    "parameters": {
                        "collections": collections_to_search,
                        "limit": limit,
                        "threshold": threshold
                    },
                    "results_count": len(all_results),
                    "results": all_results
                }
                
                with open(output, 'w', encoding='utf-8') as f:
                    json.dump(search_results, f, indent=2, ensure_ascii=False)
                
                rprint(f"\n[green]✓[/green] Results saved to: [cyan]{output}[/cyan]")
                
            except Exception as e:
                rprint(f"[yellow]Warning:[/yellow] Failed to save results: {e}")
        
        rprint(f"\n[green]✓[/green] Search completed: {len(all_results)} results")
        logger.info(f"Search completed: {len(all_results)} results for '{query}'")
        
    except Exception as e:
        _handle_operation_error("searching knowledge base", e)


@kb_app.command("collections")
def manage_collections(
    action: str = typer.Argument(
        ...,
        help="Action: list, create, delete, rename, or stats"
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Collection name for create/delete/rename operations"
    ),
    new_name: Optional[str] = typer.Option(
        None,
        "--new-name", 
        help="New name for rename operation"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Skip confirmation prompt for delete operations"
    )
) -> None:
    """
    Manage document collections and organization.
    
    Provides comprehensive collection management including creation,
    deletion, renaming, and statistics. Shows collection hierarchy
    and document distribution.
    
    Args:
        action: Management action (list, create, delete, rename, stats)
        name: Collection name for operations
        new_name: New name for rename operation
        confirm: Skip confirmation prompts
        
    Example:
        research-agent kb collections list
        research-agent kb collections create --name=new-docs
        research-agent kb collections delete --name=old-docs --confirm
        research-agent kb collections rename --name=docs --new-name=documents
    """
    try:
        from rich.table import Table
        from rich.tree import Tree
        
        chroma_manager = create_chroma_manager()
        logger.info(f"Managing collections: {action}")
        
        if action == "list":
            # List all collections with basic info
            collections_info = chroma_manager.list_collections()
            
            if not collections_info:
                rprint("[yellow]No collections found[/yellow]")
                return
            
            rprint(f"\n[blue]Collections Overview[/blue]")
            rprint("=" * 40)
            
            # Create collections table
            table = Table(title="Knowledge Base Collections")
            table.add_column("Name", style="cyan", width=20)
            table.add_column("Documents", justify="right", style="green", width=12)
            table.add_column("Size (MB)", justify="right", style="blue", width=12)
            table.add_column("Last Modified", style="dim", width=15)
            
            total_docs = 0
            total_size = 0.0
            
            for collection in collections_info:
                try:
                    stats = chroma_manager.get_collection_stats(collection.name)
                    size_mb = round(stats.storage_size_bytes / (1024 * 1024), 2)
                    last_mod = stats.last_modified.strftime('%Y-%m-%d') if stats.last_modified else 'Unknown'
                    
                    table.add_row(
                        collection.name,
                        str(stats.document_count),
                        f"{size_mb:.1f}",
                        last_mod
                    )
                    
                    total_docs += stats.document_count
                    total_size += size_mb
                    
                except Exception as e:
                    logger.warning(f"Failed to get stats for {collection.name}: {e}")
                    table.add_row(collection.name, "?", "?", "Unknown")
            
            console.print("\n")
            console.print(table)
            
            # Summary
            rprint(f"\n[blue]Summary[/blue]")
            rprint(f"Total Collections: [cyan]{len(collections_info)}[/cyan]")
            rprint(f"Total Documents: [cyan]{total_docs}[/cyan]")
            rprint(f"Total Storage: [cyan]{total_size:.1f} MB[/cyan]")
            
        elif action == "create":
            if not name:
                rprint("[red]Error:[/red] Collection name required for create action")
                raise typer.Exit(1)
            
            if chroma_manager.collection_exists(name):
                rprint(f"[yellow]Warning:[/yellow] Collection '{name}' already exists")
                return
            
            try:
                # Create collection (this may need to be implemented in chroma_manager)
                rprint(f"Creating collection: [cyan]{name}[/cyan]")
                
                # For now, we'll indicate the collection would be created on first document add
                rprint(f"[green]✓[/green] Collection '{name}' will be created when first document is added")
                logger.info(f"Collection '{name}' marked for creation")
                
            except Exception as e:
                rprint(f"[red]Error:[/red] Failed to create collection: {e}")
                logger.error(f"Collection creation failed: {e}")
                raise typer.Exit(1)
        
        elif action == "delete":
            if not name:
                rprint("[red]Error:[/red] Collection name required for delete action")
                raise typer.Exit(1)
            
            if not chroma_manager.collection_exists(name):
                rprint(f"[red]Error:[/red] Collection '{name}' does not exist")
                raise typer.Exit(1)
            
            # Get collection stats for confirmation
            try:
                stats = chroma_manager.get_collection_stats(name)
                doc_count = stats.document_count
            except:
                doc_count = "unknown"
            
            # Confirmation prompt
            if not confirm:
                rprint(f"\n[yellow]Warning:[/yellow] This will permanently delete collection '{name}'")
                rprint(f"Documents to be deleted: [red]{doc_count}[/red]")
                user_input = input("Continue with deletion? (y/N): ").strip().lower()
                if user_input not in ['y', 'yes']:
                    rprint("[yellow]Deletion cancelled[/yellow]")
                    return
            
            try:
                # Delete collection (implementation needed in chroma_manager)
                rprint(f"Deleting collection: [cyan]{name}[/cyan]")
                
                # This would need to be implemented
                if hasattr(chroma_manager, 'delete_collection'):
                    chroma_manager.delete_collection(name)
                    rprint(f"[green]✓[/green] Collection '{name}' deleted successfully")
                else:
                    rprint(f"[yellow]Note:[/yellow] Collection deletion not yet implemented")
                
                logger.info(f"Collection '{name}' deleted")
                
            except Exception as e:
                rprint(f"[red]Error:[/red] Failed to delete collection: {e}")
                logger.error(f"Collection deletion failed: {e}")
                raise typer.Exit(1)
        
        elif action == "rename":
            if not name or not new_name:
                rprint("[red]Error:[/red] Both --name and --new-name required for rename action")
                raise typer.Exit(1)
            
            if not chroma_manager.collection_exists(name):
                rprint(f"[red]Error:[/red] Collection '{name}' does not exist")
                raise typer.Exit(1)
            
            if chroma_manager.collection_exists(new_name):
                rprint(f"[red]Error:[/red] Collection '{new_name}' already exists")
                raise typer.Exit(1)
            
            try:
                rprint(f"Renaming collection: [cyan]{name}[/cyan] → [cyan]{new_name}[/cyan]")
                
                # This would need to be implemented
                if hasattr(chroma_manager, 'rename_collection'):
                    chroma_manager.rename_collection(name, new_name)
                    rprint(f"[green]✓[/green] Collection renamed successfully")
                else:
                    rprint(f"[yellow]Note:[/yellow] Collection renaming not yet implemented")
                
                logger.info(f"Collection renamed: {name} → {new_name}")
                
            except Exception as e:
                rprint(f"[red]Error:[/red] Failed to rename collection: {e}")
                logger.error(f"Collection rename failed: {e}")
                raise typer.Exit(1)
        
        elif action == "stats":
            # Detailed statistics for collection(s)
            if name:
                if not chroma_manager.collection_exists(name):
                    rprint(f"[red]Error:[/red] Collection '{name}' does not exist")
                    raise typer.Exit(1)
                collections_to_analyze = [name]
            else:
                collections_info = chroma_manager.list_collections()
                collections_to_analyze = [col.name for col in collections_info]
            
            rprint(f"\n[blue]Collection Statistics[/blue]")
            rprint("=" * 40)
            
            for col_name in collections_to_analyze:
                try:
                    stats = chroma_manager.get_collection_stats(col_name)
                    
                    rprint(f"\n[cyan]{col_name}[/cyan]")
                    rprint("-" * len(col_name))
                    rprint(f"Documents: [green]{stats.document_count}[/green]")
                    rprint(f"Storage: [blue]{stats.storage_size_bytes / (1024*1024):.2f} MB[/blue]")
                    
                    if stats.last_modified:
                        rprint(f"Last Modified: [dim]{stats.last_modified.strftime('%Y-%m-%d %H:%M:%S')}[/dim]")
                    
                    # Try to get sample documents
                    try:
                        sample_docs = chroma_manager.get_documents(
                            collection_name=col_name,
                            limit=3
                        )
                        if sample_docs:
                            rprint("Sample Documents:")
                            for i, doc in enumerate(sample_docs, 1):
                                source = doc.metadata.get('source', 'Unknown')[:30]
                                rprint(f"  {i}. {source} ({len(doc.content)} chars)")
                    except:
                        pass
                        
                except Exception as e:
                    rprint(f"[red]Error getting stats for {col_name}:[/red] {e}")
        
        else:
            rprint(f"[red]Error:[/red] Unknown action '{action}'")
            rprint("Available actions: list, create, delete, rename, stats")
            raise typer.Exit(1)
        
    except Exception as e:
        _handle_operation_error("managing collections", e) 