"""
Collection management commands for Research Agent CLI.

This module implements CLI commands for managing knowledge collections,
collection types, and collection-based operations.

Implements FR-KB-005: Collection and project management.
"""

import typer
from typing import Optional, List
from rich import print as rprint
from rich.table import Table
from rich.console import Console
from datetime import datetime

from ..core.vector_store import (
    create_chroma_manager,
    CollectionManager,
    CollectionInfo,
    CollectionStats,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    VectorStoreError
)
from ..models.metadata_schema import CollectionType

console = Console()

# Create the collections command group
collections_app = typer.Typer(
    name="collections",
    help="Collection management commands",
    rich_markup_mode="rich",
)


def _get_global_config() -> dict:
    """Get global configuration from the main CLI module."""
    from .cli import get_global_config
    return get_global_config()


def _handle_collection_error(operation: str, error: Exception) -> None:
    """Handle and display collection operation errors consistently."""
    if isinstance(error, CollectionAlreadyExistsError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, CollectionNotFoundError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, VectorStoreError):
        rprint(f"[red]Error:[/red] Failed to {operation}: {error}")
        raise typer.Exit(1)
    else:
        rprint(f"[red]Error:[/red] Unexpected error during {operation}: {error}")
        raise typer.Exit(1)


def _format_collection_size(count: int) -> str:
    """Format collection document count for display."""
    if count == 0:
        return "Empty"
    elif count == 1:
        return "1 document"
    else:
        return f"{count} documents"


@collections_app.command("create")
def create_collection(
    name: str = typer.Argument(..., help="Name of the collection to create"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the collection"
    ),
    collection_type: str = typer.Option(
        "general",
        "--type",
        "-t",
        help="Collection type (general, project-specific, fundamental, reference, temporary)"
    ),
) -> None:
    """
    Create a new knowledge collection.
    
    Collections organize documents by topic, project, or type.
    They enable focused queries and better knowledge organization.
    
    Example:
        research-agent collections create "machine-learning" --description "ML research papers"
    """
    # Get global config for dry-run check
    global_config = _get_global_config()
    if global_config.get("dry_run"):
        rprint("[blue]DRY RUN:[/blue] Would create collection but not executing")
        rprint(f"  Collection: {name}")
        rprint(f"  Type: {collection_type}")
        if description:
            rprint(f"  Description: {description}")
        return
    
    try:
        # Validate and convert collection type
        try:
            coll_type = CollectionType(collection_type)
        except ValueError:
            rprint(f"[red]Error:[/red] Invalid collection type '{collection_type}'")
            rprint(f"Valid types: {', '.join([t.value for t in CollectionType])}")
            raise typer.Exit(1)
        
        # Create vector store manager
        manager = create_chroma_manager()
        
        # Prepare metadata
        metadata = {}
        if description:
            metadata["description"] = description
        
        # Create collection
        collection = manager.collection_manager.create_collection(
            name=name,
            collection_type=coll_type,
            metadata=metadata,
            owner_id="default_user"  # TODO: Replace with actual user context
        )
        
        rprint(f"[green]Successfully created collection[/green] '{name}'")
        rprint(f"  Type: {collection_type}")
        if description:
            rprint(f"  Description: {description}")
        
    except Exception as e:
        _handle_collection_error("create collection", e)


@collections_app.command("list")
def list_collections(
    collection_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by collection type"
    ),
    show_stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show collection statistics (document count, size, etc.)"
    ),
) -> None:
    """
    List all collections in the knowledge base.
    
    Shows collection names, types, descriptions, and optionally
    statistics about document count and storage usage.
    
    Example:
        research-agent collections list --stats
    """
    try:
        # Create vector store manager
        manager = create_chroma_manager()
        
        # Get collections based on filter
        if collection_type:
            try:
                coll_type = CollectionType(collection_type)
                collections = manager.collection_manager.get_collections_by_type(coll_type)
            except ValueError:
                rprint(f"[red]Error:[/red] Invalid collection type '{collection_type}'")
                rprint(f"Valid types: {', '.join([t.value for t in CollectionType])}")
                raise typer.Exit(1)
        else:
            collections = manager.collection_manager.list_collections()
        
        if not collections:
            rprint("[yellow]No collections found[/yellow]")
            return
        
        # Create table for display
        table = Table(title="Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Description", style="white")
        if show_stats:
            table.add_column("Documents", justify="right", style="blue")
        
        # Add collection data to table
        for collection_info in collections:
            metadata = collection_info.metadata or {}
            coll_type = metadata.get('collection_type', 'unknown')
            description = metadata.get('description', 'No description')
            
            if show_stats:
                doc_count = _format_collection_size(collection_info.count)
                table.add_row(
                    collection_info.name,
                    coll_type,
                    description,
                    doc_count
                )
            else:
                table.add_row(
                    collection_info.name,
                    coll_type,
                    description
                )
        
        console.print(table)
        
    except Exception as e:
        _handle_collection_error("list collections", e)


@collections_app.command("info")
def collection_info(
    name: str = typer.Argument(..., help="Name of the collection"),
) -> None:
    """
    Show detailed information about a collection.
    
    Displays collection metadata, statistics, configuration,
    and recent activity.
    
    Example:
        research-agent collections info "machine-learning"
    """
    try:
        # Create vector store manager
        manager = create_chroma_manager()
        
        # Get collection statistics
        stats = manager.collection_manager.get_collection_stats(name)
        
        # Display collection information
        rprint(f"[bold cyan]Collection: {stats.name}[/bold cyan]")
        rprint(f"ID: {stats.id}")
        rprint(f"Document Count: [blue]{stats.document_count}[/blue]")
        
        # Display metadata
        metadata = stats.metadata or {}
        coll_type = metadata.get('collection_type', 'unknown')
        rprint(f"Type: [green]{coll_type}[/green]")
        
        description = metadata.get('description')
        if description:
            rprint(f"Description: {description}")
        
        created_at = metadata.get('created_at')
        if created_at:
            rprint(f"Created: {created_at}")
        
        owner_id = metadata.get('owner_id')
        if owner_id:
            rprint(f"Owner: {owner_id}")
        
        team_id = metadata.get('team_id')
        if team_id:
            rprint(f"Team: {team_id}")
            
    except Exception as e:
        _handle_collection_error("get collection info", e)


@collections_app.command("delete")
def delete_collection(
    name: str = typer.Argument(..., help="Name of the collection to delete"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
    keep_documents: bool = typer.Option(
        False,
        "--keep-documents",
        help="Keep documents but remove from collection"
    ),
) -> None:
    """
    Delete a collection from the knowledge base.
    
    By default, this removes the collection and all its documents.
    Use --keep-documents to preserve documents in the default collection.
    
    Example:
        research-agent collections delete "old-project" --confirm
    """
    # Get global config for dry-run check
    global_config = _get_global_config()
    if global_config.get("dry_run"):
        rprint("[blue]DRY RUN:[/blue] Would delete collection but not executing")
        rprint(f"  Collection: {name}")
        if keep_documents:
            rprint("  Keep documents: Yes")
        return
    
    try:
        # Get confirmation if not provided
        if not confirm:
            if keep_documents:
                message = f"Are you sure you want to remove collection '{name}' (documents will be preserved)?"
            else:
                message = f"Are you sure you want to delete collection '{name}' and all its documents?"
            
            if not typer.confirm(message):
                rprint("[yellow]Operation cancelled[/yellow]")
                return
        
        # Create vector store manager
        manager = create_chroma_manager()
        
        # TODO: Implement keep_documents functionality when document manager supports it
        if keep_documents:
            rprint("[yellow]Warning:[/yellow] Keep documents functionality not yet implemented")
            rprint("All documents will be deleted with the collection")
        
        # Delete collection
        manager.collection_manager.delete_collection(name)
        
        rprint(f"[green]Successfully deleted collection[/green] '{name}'")
        
    except Exception as e:
        _handle_collection_error("delete collection", e)


@collections_app.command("rename")
def rename_collection(
    old_name: str = typer.Argument(..., help="Current name of the collection"),
    new_name: str = typer.Argument(..., help="New name for the collection"),
) -> None:
    """
    Rename a collection.
    
    Changes the collection name while preserving all documents
    and metadata.
    
    Example:
        research-agent collections rename "old-name" "new-name"
    """
    # Get global config for dry-run check
    global_config = _get_global_config()
    if global_config.get("dry_run"):
        rprint("[blue]DRY RUN:[/blue] Would rename collection but not executing")
        rprint(f"  From: {old_name}")
        rprint(f"  To: {new_name}")
        return
    
    try:
        # Create vector store manager
        manager = create_chroma_manager()
        
        # Check if source collection exists
        try:
            source_collection = manager.collection_manager.get_collection(old_name)
        except CollectionNotFoundError:
            rprint(f"[red]Error:[/red] Collection '{old_name}' not found")
            raise typer.Exit(1)
        
        # Check if target name is available
        try:
            manager.collection_manager.get_collection(new_name)
            rprint(f"[red]Error:[/red] Collection '{new_name}' already exists")
            raise typer.Exit(1)
        except CollectionNotFoundError:
            pass  # Target name is available
        
        # Implement actual rename functionality
        rprint(f"[blue]Renaming collection '[/blue]{old_name}[blue]' to '[/blue]{new_name}[blue]'...[/blue]")
        rprint("1. Getting source collection information...")
        
        # Get all documents from source collection
        documents_data = manager.get_documents(
            collection_name=old_name,
            limit=10000,  # Large limit to get all documents
            include=['documents', 'metadatas', 'ids', 'embeddings']
        )
        
        document_count = len(documents_data.get('documents', []))
        rprint(f"   Found {document_count} documents to migrate")
        
        if document_count == 0:
            rprint("[yellow]Warning:[/yellow] Source collection is empty")
        
        # Get source collection metadata
        source_info = manager.collection_manager.get_collection(old_name)
        source_metadata = source_info.metadata or {}
        
        rprint("2. Creating target collection...")
        
        # Create new collection with same type and metadata
        collection_type = source_metadata.get('collection_type', 'general')
        try:
            coll_type = CollectionType(collection_type)
        except ValueError:
            coll_type = CollectionType.GENERAL
        
        # Create target collection with preserved metadata
        target_metadata = source_metadata.copy()
        target_metadata['renamed_from'] = old_name
        target_metadata['renamed_at'] = datetime.now().isoformat()
        
        new_collection = manager.collection_manager.create_collection(
            name=new_name,
            collection_type=coll_type,
            metadata=target_metadata,
            owner_id="default_user"  # TODO: Replace with actual user context
        )
        
        rprint(f"   Created collection '{new_name}' with type '{collection_type}'")
        
        if document_count > 0:
            rprint("3. Migrating documents...")
            
            # Get documents, metadata, and IDs
            documents = documents_data.get('documents', [])
            metadatas = documents_data.get('metadatas', [])
            ids = documents_data.get('ids', [])
            embeddings = documents_data.get('embeddings', [])
            
            # Prepare batch data for insertion
            try:
                if embeddings:
                    # Use existing embeddings if available
                    manager.add_documents_with_embeddings(
                        collection_name=new_name,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=embeddings
                    )
                else:
                    # Let the system generate new embeddings
                    manager.add_documents(
                        collection_name=new_name,
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                rprint(f"   Migrated {len(documents)} documents successfully")
                
            except Exception as e:
                rprint(f"[red]Error migrating documents:[/red] {e}")
                # Clean up the new collection if migration fails
                try:
                    manager.collection_manager.delete_collection(new_name)
                    rprint(f"   Cleaned up failed collection '{new_name}'")
                except Exception:
                    pass
                raise
        
        rprint("4. Deleting source collection...")
        
        # Delete original collection
        manager.collection_manager.delete_collection(old_name)
        rprint(f"   Deleted original collection '{old_name}'")
        
        rprint(f"[green]✓ Successfully renamed collection[/green] '{old_name}' to '{new_name}'")
        if document_count > 0:
            rprint(f"  Migrated: {document_count} documents")
        rprint(f"  Type: {collection_type}")
        if target_metadata.get('description'):
            rprint(f"  Description: {target_metadata['description']}")
        
    except Exception as e:
        _handle_collection_error("rename collection", e)


@collections_app.command("move-documents")
def move_documents(
    source: str = typer.Argument(..., help="Source collection name"),
    target: str = typer.Argument(..., help="Target collection name"),
    pattern: Optional[str] = typer.Option(
        None,
        "--pattern",
        "-p",
        help="Pattern to match document names (supports wildcards)"
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Move documents between collections.
    
    Moves all or matching documents from source to target collection.
    Use patterns to move specific subsets of documents.
    
    Example:
        research-agent collections move-documents "old-project" "archive" --pattern "*.pdf"
    """
    # Get global config for dry-run check
    global_config = _get_global_config()
    if global_config.get("dry_run"):
        rprint("[blue]DRY RUN:[/blue] Would move documents but not executing")
        rprint(f"  From: {source}")
        rprint(f"  To: {target}")
        if pattern:
            rprint(f"  Pattern: {pattern}")
        return
    
    try:
        # Create vector store manager
        manager = create_chroma_manager()
        
        # Verify both collections exist
        try:
            manager.collection_manager.get_collection(source)
        except CollectionNotFoundError:
            rprint(f"[red]Error:[/red] Source collection '{source}' not found")
            raise typer.Exit(1)
            
        try:
            manager.collection_manager.get_collection(target)
        except CollectionNotFoundError:
            rprint(f"[red]Error:[/red] Target collection '{target}' not found")
            raise typer.Exit(1)
        
        # Get confirmation if not provided
        if not confirm:
            if pattern:
                message = f"Move documents matching '{pattern}' from '{source}' to '{target}'?"
            else:
                message = f"Move all documents from '{source}' to '{target}'?"
            
            if not typer.confirm(message):
                rprint("[yellow]Operation cancelled[/yellow]")
                return
        
        # Implement actual document moving functionality
        rprint(f"[blue]Moving documents from '[/blue]{source}[blue]' to '[/blue]{target}[blue]'...[/blue]")
        
        # Get all documents from source collection
        rprint("1. Retrieving documents from source collection...")
        documents_data = manager.get_documents(
            collection_name=source,
            limit=10000,  # Large limit to get all documents
            include=['documents', 'metadatas', 'ids', 'embeddings']
        )
        
        documents = documents_data.get('documents', [])
        metadatas = documents_data.get('metadatas', [])
        ids = documents_data.get('ids', [])
        embeddings = documents_data.get('embeddings', [])
        
        if not documents:
            rprint(f"[yellow]No documents found in collection '{source}'[/yellow]")
            return
        
        # Apply pattern filtering if specified
        if pattern:
            import fnmatch
            
            rprint(f"2. Filtering documents by pattern: {pattern}")
            filtered_indices = []
            
            for i, (doc_id, metadata) in enumerate(zip(ids, metadatas)):
                # Check pattern against document ID
                if fnmatch.fnmatch(doc_id, pattern):
                    filtered_indices.append(i)
                # Also check against filename in metadata if available
                elif metadata and 'source_file' in metadata:
                    if fnmatch.fnmatch(metadata['source_file'], pattern):
                        filtered_indices.append(i)
                # Also check against title in metadata if available
                elif metadata and 'title' in metadata:
                    if fnmatch.fnmatch(metadata['title'], pattern):
                        filtered_indices.append(i)
            
            if not filtered_indices:
                rprint(f"[yellow]No documents matched pattern '{pattern}'[/yellow]")
                return
            
            # Filter all arrays by matching indices
            documents = [documents[i] for i in filtered_indices]
            metadatas = [metadatas[i] for i in filtered_indices]
            ids = [ids[i] for i in filtered_indices]
            if embeddings:
                embeddings = [embeddings[i] for i in filtered_indices]
            
            rprint(f"   Found {len(documents)} documents matching pattern")
        else:
            rprint(f"   Found {len(documents)} documents to move")
        
        # Add documents to target collection
        step_num = 3 if pattern else 2
        rprint(f"{step_num}. Adding documents to target collection...")
        
        try:
            # Add migration metadata
            for metadata in metadatas:
                if metadata is None:
                    metadata = {}
                metadata['moved_from'] = source
                metadata['moved_at'] = datetime.now().isoformat()
            
            if embeddings:
                # Use existing embeddings if available
                manager.add_documents_with_embeddings(
                    collection_name=target,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # Let the system generate new embeddings
                manager.add_documents(
                    collection_name=target,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            rprint(f"   Added {len(documents)} documents to '{target}'")
            
        except Exception as e:
            rprint(f"[red]Error adding documents to target collection:[/red] {e}")
            raise
        
        # Remove documents from source collection
        step_num += 1
        rprint(f"{step_num}. Removing documents from source collection...")
        
        try:
            # Remove documents by ID from source collection
            for doc_id in ids:
                try:
                    manager.delete_documents(
                        collection_name=source,
                        ids=[doc_id]
                    )
                except Exception as e:
                    rprint(f"   [yellow]Warning:[/yellow] Could not remove document {doc_id}: {e}")
            
            rprint(f"   Removed {len(ids)} documents from '{source}'")
            
        except Exception as e:
            rprint(f"[yellow]Warning:[/yellow] Some documents may not have been removed from source: {e}")
        
        rprint(f"[green]✓ Successfully moved[/green] {len(documents)} documents from '{source}' to '{target}'")
        if pattern:
            rprint(f"  Pattern: {pattern}")
        rprint(f"  Documents moved: {len(documents)}")
        
    except Exception as e:
        _handle_collection_error("move documents", e) 