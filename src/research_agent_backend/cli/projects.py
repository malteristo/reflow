"""
Project management commands for Research Agent CLI.

This module implements CLI commands for managing project-specific
knowledge spaces and project-based operations.

Implements FR-KB-005: Project and collection management.
"""

import typer
from typing import Optional, List
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from ..services.project_manager import create_project_manager
from ..exceptions.project_exceptions import (
    ProjectNotFoundError,
    ProjectAlreadyExistsError,
    CollectionAlreadyLinkedError,
    CollectionNotLinkedError,
    ProjectContextError,
    ProjectMetadataError
)
from ..core.vector_store import CollectionNotFoundError

console = Console()

# Create the projects command group
projects_app = typer.Typer(
    name="projects",
    help="Project-specific operations",
    rich_markup_mode="rich",
)


def _get_dry_run_status(ctx: typer.Context) -> bool:
    """
    Get dry-run status from the CLI context hierarchy.
    
    Args:
        ctx: Current Typer context
        
    Returns:
        True if dry-run mode is enabled, False otherwise
    """
    try:
        # Walk up the context hierarchy to find the root context with global options
        current_ctx = ctx
        while current_ctx:
            if hasattr(current_ctx, 'params') and 'dry_run' in current_ctx.params:
                return current_ctx.params.get('dry_run', False)
            current_ctx = getattr(current_ctx, 'parent', None)
        
        # Fallback: check the global config approach
        try:
            from .cli import get_global_config
            config = get_global_config()
            return config.get("dry_run", False)
        except Exception:
            pass
        
        return False
    except Exception:
        # Safe fallback if context access fails
        return False


def _get_global_config() -> dict:
    """Get global configuration from the main CLI module."""
    try:
        from .cli import get_global_config
        config = get_global_config()
        # If config is empty, we're likely not in CLI context
        # Return a default config that doesn't interfere with normal operation
        if not config:
            return {"dry_run": False, "verbose": False}
        return config
    except Exception:
        # Fallback to safe defaults if import fails
        return {"dry_run": False, "verbose": False}


def _handle_project_error(operation: str, error: Exception) -> None:
    """Handle and display project operation errors consistently."""
    if isinstance(error, ProjectNotFoundError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, ProjectAlreadyExistsError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, CollectionAlreadyLinkedError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, CollectionNotLinkedError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    elif isinstance(error, CollectionNotFoundError):
        rprint(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)
    else:
        rprint(f"[red]Error:[/red] Unexpected error during {operation}: {error}")
        raise typer.Exit(1)


@projects_app.command("create")
def create_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the project"
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        "-t",
        help="Comma-separated tags for the project"
    ),
) -> None:
    """
    Create a new project.
    
    Creates a new project with metadata storage for organizing
    collections and knowledge management.
    
    Example:
        research-agent projects create "my-research" --description "AI research project" --tags "ai,research"
    """
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would create project but not executing")
        rprint(f"  Project: {name}")
        if description:
            rprint(f"  Description: {description}")
        if tags:
            rprint(f"  Tags: {tags}")
        return
    
    try:
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        # Create project manager
        manager = create_project_manager()
        
        # Create project
        manager.create_project(
            name=name,
            description=description,
            tags=tag_list
        )
        
        rprint(f"[green]Successfully created project[/green] '{name}'")
        if description:
            rprint(f"  Description: {description}")
        if tag_list:
            rprint(f"  Tags: {', '.join(tag_list)}")
        
    except Exception as e:
        _handle_project_error("create project", e)


@projects_app.command("info")
def project_info(
    name: str = typer.Argument(..., help="Name of the project"),
) -> None:
    """
    Show detailed information about a project.
    
    Displays project metadata, collections, recent activity,
    and knowledge statistics.
    
    Example:
        research-agent projects info "my-research"
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Get project metadata
        metadata = manager.get_project_metadata(name)
        
        rprint(f"[cyan]Project:[/cyan] {metadata.name}")
        if metadata.description:
            rprint(f"[white]Description:[/white] {metadata.description}")
        if metadata.tags:
            rprint(f"[white]Tags:[/white] {', '.join(metadata.tags)}")
        rprint(f"[white]Status:[/white] {metadata.status.value}")
        rprint(f"[white]Created:[/white] {metadata.created_at}")
        rprint(f"[white]Collections:[/white] {metadata.linked_collections_count}")
        rprint(f"[white]Documents:[/white] {metadata.total_documents}")
        
    except Exception as e:
        _handle_project_error("get project info", e)


@projects_app.command("update")
def update_project(
    name: str = typer.Argument(..., help="Name of the project"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="New description for the project"
    ),
    add_tags: Optional[str] = typer.Option(
        None,
        "--add-tags",
        help="Comma-separated tags to add"
    ),
) -> None:
    """
    Update project metadata.
    
    Updates project description, tags, and other metadata.
    
    Example:
        research-agent projects update "my-research" --description "Updated description" --add-tags "new-tag"
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Prepare updates
        updates = {}
        if description:
            updates["description"] = description
        if add_tags:
            # For simplicity, just replace tags for now
            tag_list = [tag.strip() for tag in add_tags.split(",")]
            updates["tags"] = tag_list
        
        # Update project
        manager.update_project_metadata(name, **updates)
        
        rprint(f"[green]Successfully updated project[/green] '{name}'")
        
    except Exception as e:
        _handle_project_error("update project", e)


@projects_app.command("link-collection")
def link_collection(
    ctx: typer.Context,
    project_name: str = typer.Argument(..., help="Name of the project"),
    collection_name: str = typer.Argument(..., help="Name of the collection to link"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the link relationship"
    ),
) -> None:
    """
    Link a collection to a project.
    
    Creates a relationship between a project and a collection,
    allowing project-specific knowledge organization.
    
    Example:
        research-agent projects link-collection "my-research" "research-papers" --description "Core research papers"
    """
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would link collection but not executing")
        rprint(f"  Project: {project_name}")
        rprint(f"  Collection: {collection_name}")
        if description:
            rprint(f"  Description: {description}")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Link collection
        manager.link_collection(
            project_name=project_name,
            collection_name=collection_name,
            description=description
        )
        
        rprint(f"[green]Successfully linked collection[/green] '{collection_name}' to project '{project_name}'")
        if description:
            rprint(f"  Description: {description}")
        
    except Exception as e:
        _handle_project_error("link collection", e)


@projects_app.command("unlink-collection")
def unlink_collection(
    project_name: str = typer.Argument(..., help="Name of the project"),
    collection_name: str = typer.Argument(..., help="Name of the collection to unlink"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Unlink a collection from a project.
    
    Removes the relationship between a project and a collection.
    The collection itself is not deleted.
    
    Example:
        research-agent projects unlink-collection "my-research" "old-papers" --confirm
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Unlink collection
        manager.unlink_collection(
            project_name=project_name,
            collection_name=collection_name
        )
        
        rprint(f"[green]Successfully unlinked collection[/green] '{collection_name}' from project '{project_name}'")
        
    except Exception as e:
        _handle_project_error("unlink collection", e)


@projects_app.command("set-default-collections")
def set_default_collections(
    project_name: str = typer.Argument(..., help="Name of the project"),
    collections: str = typer.Argument(..., help="Comma-separated collection names (empty string to clear)"),
    append: bool = typer.Option(
        False,
        "--append",
        "-a",
        help="Append to existing default collections"
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        "-c",
        help="Clear existing default collections"
    ),
) -> None:
    """
    Set default collections for a project.
    
    Default collections are used automatically when querying
    within the project context.
    
    Example:
        research-agent projects set-default-collections "my-research" "papers,docs" --append
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        if clear or collections == "":
            # Clear default collections
            manager.clear_default_collections(project_name)
            rprint(f"[green]Cleared default collections[/green] for project '{project_name}'")
        else:
            # Parse collection names
            collection_names = [name.strip() for name in collections.split(",") if name.strip()]
            
            # Set default collections
            manager.set_default_collections(
                project_name=project_name,
                collection_names=collection_names,
                append=append
            )
            
            action = "Added to" if append else "Set"
            rprint(f"[green]Successfully set default collections[/green] for project '{project_name}'")
            rprint(f"  Collections: {', '.join(collection_names)}")
        
    except Exception as e:
        _handle_project_error("set default collections", e)


@projects_app.command("list-project-collections")
def list_project_collections(
    project_name: str = typer.Argument(..., help="Name of the project"),
    stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show collection statistics"
    ),
    defaults_only: bool = typer.Option(
        False,
        "--defaults-only",
        "-d",
        help="Show only default collections"
    ),
) -> None:
    """
    List collections linked to a project.
    
    Shows all collections associated with a project,
    optionally with statistics and filtering.
    
    Example:
        research-agent projects list-project-collections "my-research" --stats
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Get project collections
        project_info = manager.get_project_collections(project_name)
        
        if defaults_only:
            collections = project_info.default_collections
            title = f"Default Collections for '{project_name}'"
        else:
            collections = project_info.linked_collections
            title = f"Collections for '{project_name}'"
        
        if not collections:
            if defaults_only:
                rprint(f"[yellow]No default collections set[/yellow] for project '{project_name}'")
            else:
                rprint(f"[yellow]No collections linked[/yellow] to project '{project_name}'")
            return
        
        # Create table for display
        table = Table(title=title)
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Default", style="green")
        if stats:
            table.add_column("Documents", justify="right", style="blue")
        
        # Add collection data to table
        for coll in collections:
            default_marker = "✓" if coll.is_default else ""
            desc = coll.description or "No description"
            
            if stats:
                doc_count = str(coll.document_count) if coll.document_count is not None else "Unknown"
                table.add_row(coll.collection_name, desc, default_marker, doc_count)
            else:
                table.add_row(coll.collection_name, desc, default_marker)
        
        console.print(table)
        
    except Exception as e:
        _handle_project_error("list project collections", e)


@projects_app.command("detect-context")
def detect_context(
    path: Optional[str] = typer.Option(
        None,
        "--path",
        "-p",
        help="File path to analyze for project detection"
    ),
) -> None:
    """
    Detect project context from file path.
    
    Analyzes a file path to automatically determine
    which project it belongs to.
    
    Example:
        research-agent projects detect-context --path "/users/me/projects/my-research/docs/file.md"
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        if path:
            # Detect project from path
            detected_project = manager.detect_project_from_path(path)
            
            if detected_project:
                rprint(f"[green]Detected project:[/green] {detected_project}")
                rprint(f"  Path: {path}")
            else:
                rprint(f"[yellow]No project detected[/yellow] for path: {path}")
        else:
            rprint("[red]Error:[/red] Path is required for project detection")
            raise typer.Exit(1)
        
    except Exception as e:
        _handle_project_error("detect project context", e)


@projects_app.command("set-context")
def set_context(
    project_name: str = typer.Argument(..., help="Name of the project to activate"),
) -> None:
    """
    Set explicit project context.
    
    Manually sets the active project context for subsequent operations.
    
    Example:
        research-agent projects set-context "my-research"
    """
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Set active project
        manager.set_active_project(project_name)
        
        rprint(f"[green]Set project context[/green] to '{project_name}'")
        
    except Exception as e:
        _handle_project_error("set project context", e)


@projects_app.command("init")
def init_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project"),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Description of the project"
    ),
    template: Optional[str] = typer.Option(
        None,
        "--template",
        "-t",
        help="Project template to use"
    ),
) -> None:
    """
    Initialize a new project knowledge space.
    
    Creates project-specific collections, configurations,
    and knowledge organization structure.
    
    Example:
        research-agent projects init "my-research" --description "AI research project"
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Initialize project '{name}'")
    if description:
        rprint(f"  Description: {description}")
    if template:
        rprint(f"  Template: {template}")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would initialize project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("list")
def list_projects(
    active_only: bool = typer.Option(
        False,
        "--active-only",
        "-a",
        help="Show only active projects"
    ),
    show_stats: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show project statistics"
    ),
) -> None:
    """
    List all projects in the knowledge base.
    
    Shows project names, status, descriptions, and optionally
    statistics about collections and documents.
    
    Example:
        research-agent projects list --stats
    """
    # TODO: Implement in Task 10
    rprint("[yellow]TODO:[/yellow] List projects")
    if active_only:
        rprint("  Active only: Yes")
    if show_stats:
        rprint("  Include statistics: Yes")
    
    # Create example table for UI mockup
    table = Table(title="Projects (Example)")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description", style="white")
    if show_stats:
        table.add_column("Collections", justify="right", style="blue")
        table.add_column("Documents", justify="right", style="blue")
    
    # Example data
    if show_stats:
        table.add_row("research-agent", "active", "AI research agent project", "3", "127")
        table.add_row("old-project", "archived", "Archived research", "1", "42")
    else:
        table.add_row("research-agent", "active", "AI research agent project")
        table.add_row("old-project", "archived", "Archived research")
    
    console.print(table)
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("activate")
def activate_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project to activate"),
) -> None:
    """
    Activate a project as the current working context.
    
    Sets the project as the default for subsequent operations
    and configures the knowledge scope.
    
    Example:
        research-agent projects activate "my-research"
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Activate project '{name}'")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would activate project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("deactivate")
def deactivate_project(ctx: typer.Context) -> None:
    """
    Deactivate the current project.
    
    Returns to global knowledge base scope without
    project-specific filtering.
    
    Example:
        research-agent projects deactivate
    """
    # TODO: Implement in Task 10
    rprint("[yellow]TODO:[/yellow] Deactivate current project")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would deactivate project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("archive")
def archive_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project to archive"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
) -> None:
    """
    Archive a project.
    
    Marks a project as archived, hiding it from active lists
    but preserving all data and relationships.
    
    Example:
        research-agent projects archive "old-project" --confirm
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Archive project '{name}'")
    if confirm:
        rprint("  Confirmation: Skipped")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would archive project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("delete")
def delete_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project to delete"),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        "-y",
        help="Skip confirmation prompt"
    ),
    keep_collections: bool = typer.Option(
        False,
        "--keep-collections",
        help="Keep project collections as standalone collections"
    ),
) -> None:
    """
    Delete a project permanently.
    
    Removes a project and optionally its associated collections.
    This operation cannot be undone.
    
    Example:
        research-agent projects delete "old-project" --confirm --keep-collections
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Delete project '{name}'")
    if confirm:
        rprint("  Confirmation: Skipped")
    if keep_collections:
        rprint("  Keep collections: Yes")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would delete project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("export")
def export_project(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Name of the project to export"),
    output_path: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Output directory for exported project"
    ),
    format: str = typer.Option(
        "zip",
        "--format",
        "-f",
        help="Export format (zip, tar, folder)"
    ),
) -> None:
    """
    Export a project to an archive.
    
    Creates a portable archive containing project metadata,
    collections, and documents for backup or transfer.
    
    Example:
        research-agent projects export "my-research" --output "/backups" --format "zip"
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Export project '{name}'")
    rprint(f"  Output: {output_path}")
    rprint(f"  Format: {format}")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would export project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]")


@projects_app.command("import")
def import_project(
    ctx: typer.Context,
    archive_path: str = typer.Argument(..., help="Path to project archive"),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for imported project (default: from archive)"
    ),
    merge: bool = typer.Option(
        False,
        "--merge",
        help="Merge with existing project if name conflicts"
    ),
) -> None:
    """
    Import a project from an archive.
    
    Restores a project from a previously exported archive,
    recreating collections and metadata.
    
    Example:
        research-agent projects import "/backups/my-research.zip" --name "restored-research"
    """
    # TODO: Implement in Task 10
    rprint(f"[yellow]TODO:[/yellow] Import project from '{archive_path}'")
    if name:
        rprint(f"  Name: {name}")
    if merge:
        rprint("  Merge: Yes")
    
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would import project but not executing")
        return
    
    rprint("[red]Not implemented yet - will be completed in Task 10[/red]") 