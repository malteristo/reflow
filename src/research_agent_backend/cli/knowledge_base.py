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
    
    Note:
        This command is not yet implemented and will be completed in a future task.
        
    Example:
        research-agent kb status
    """
    rprint("[yellow]TODO:[/yellow] Show knowledge base status")
    rprint("[red]Not implemented yet - will be completed in Task 8[/red]")
    logger.info("Status command called but not yet implemented")


@kb_app.command("rebuild-index")
def rebuild_index(
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Rebuild index for specific collection only"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Rebuild the vector index for the knowledge base.
    
    This command regenerates embeddings and rebuilds the vector index.
    Useful after changing embedding models or fixing corrupted indexes.
    
    Args:
        collection: Optional collection name to rebuild index for.
        confirm: Whether to skip the confirmation prompt.
        
    Note:
        This command is not yet implemented and will be completed in a future task.
        
    Example:
        research-agent kb rebuild-index --collection my-docs --confirm
    """
    rprint("[yellow]TODO:[/yellow] Rebuild vector index")
    if collection:
        rprint(f"  Collection: {collection}")
    if not confirm:
        rprint("  Will prompt for confirmation (use --confirm to skip)")
    
    # Get global config for dry-run check
    global_config = _get_global_config()
    if global_config.get("dry_run"):
        rprint("[blue]DRY RUN:[/blue] Would rebuild index but not executing")
        logger.info("Rebuild index called in dry-run mode")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 8[/red]")
    logger.info("Rebuild index command called but not yet implemented") 