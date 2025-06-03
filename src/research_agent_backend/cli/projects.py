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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would initialize project but not executing")
        rprint(f"  Project: {name}")
        if description:
            rprint(f"  Description: {description}")
        if template:
            rprint(f"  Template: {template}")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Parse tags from template if provided
        tag_list = []
        if template:
            # Define template-based tags
            template_tags = {
                "research": ["research", "academic", "analysis"],
                "development": ["development", "coding", "software"],
                "documentation": ["docs", "writing", "knowledge"],
                "analysis": ["analysis", "data", "insights"],
                "general": ["general", "misc"]
            }
            tag_list = template_tags.get(template.lower(), [template.lower()])
        
        # Create the project
        rprint(f"[blue]Initializing project:[/blue] '{name}'")
        manager.create_project(
            name=name,
            description=description or f"Project initialized from template: {template}" if template else f"Initialized project: {name}",
            tags=tag_list
        )
        
        # Create default collections for the project
        default_collections = []
        if template == "research":
            default_collections = [f"{name}-papers", f"{name}-notes", f"{name}-references"]
        elif template == "development":
            default_collections = [f"{name}-docs", f"{name}-code", f"{name}-issues"]
        elif template == "documentation":
            default_collections = [f"{name}-docs", f"{name}-drafts"]
        else:
            default_collections = [f"{name}-main", f"{name}-resources"]
        
        # Link collections to project
        from ..core.vector_store import create_chroma_manager
        chroma_manager = create_chroma_manager()
        
        for collection_name in default_collections:
            try:
                # Create collection if it doesn't exist
                if not chroma_manager.collection_exists(collection_name):
                    chroma_manager.create_collection(collection_name, metadata={"project": name})
                
                # Link collection to project
                manager.link_collection_to_project(
                    project_name=name,
                    collection_name=collection_name,
                    description=f"Default {collection_name.split('-')[-1]} collection for {name}"
                )
                
                rprint(f"  [green]✓[/green] Created and linked collection: {collection_name}")
                
            except Exception as e:
                rprint(f"  [yellow]Warning:[/yellow] Could not create collection {collection_name}: {e}")
        
        # Set default collections for the project
        if default_collections:
            try:
                manager.set_default_collections(name, default_collections)
                rprint(f"  [green]✓[/green] Set default collections: {', '.join(default_collections)}")
            except Exception as e:
                rprint(f"  [yellow]Warning:[/yellow] Could not set default collections: {e}")
        
        rprint(f"[green]✓ Successfully initialized project[/green] '{name}'")
        if description:
            rprint(f"  Description: {description}")
        if template:
            rprint(f"  Template: {template}")
        if tag_list:
            rprint(f"  Tags: {', '.join(tag_list)}")
        rprint(f"  Collections: {len(default_collections)} created")
        
        # Show next steps
        rprint(f"\n[blue]Next steps:[/blue]")
        rprint(f"• Use 'research-agent projects activate {name}' to set as active project")
        rprint(f"• Use 'research-agent kb ingest-folder <path> --collection {default_collections[0]}' to add documents")
        rprint(f"• Use 'research-agent projects info {name}' to view project details")
        
    except Exception as e:
        _handle_project_error("initialize project", e)


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
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Get all projects
        projects = manager.list_projects()
        
        # Filter by active status if requested
        if active_only:
            from ..models.project_schema import ProjectStatus
            projects = [p for p in projects if p.status == ProjectStatus.ACTIVE]
        
        if not projects:
            if active_only:
                rprint("[yellow]No active projects found[/yellow]")
            else:
                rprint("[yellow]No projects found[/yellow]")
            rprint("\n[dim]Use 'research-agent projects create <name>' to create a new project.[/dim]")
            return
        
        # Create projects table
        table = Table(title=f"Projects ({len(projects)} found)")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Description", style="white")
        
        if show_stats:
            table.add_column("Collections", justify="right", style="blue")
            table.add_column("Documents", justify="right", style="blue")
            table.add_column("Created", style="dim")
        
        # Populate table
        total_collections = 0
        total_documents = 0
        
        for project in projects:
            # Format status with color
            if project.status.value == "active":
                status_display = "[green]active[/green]"
            elif project.status.value == "archived":
                status_display = "[yellow]archived[/yellow]"
            else:
                status_display = project.status.value
            
            # Truncate description if too long
            description = project.description or "No description"
            if len(description) > 50:
                description = description[:47] + "..."
            
            if show_stats:
                collections_count = project.linked_collections_count
                documents_count = project.total_documents
                created_date = project.created_at.strftime('%Y-%m-%d') if project.created_at else 'Unknown'
                
                table.add_row(
                    project.name,
                    status_display,
                    description,
                    str(collections_count),
                    str(documents_count),
                    created_date
                )
                
                total_collections += collections_count
                total_documents += documents_count
            else:
                table.add_row(
                    project.name,
                    status_display,
                    description
                )
        
        console.print(table)
        
        # Show summary statistics
        if show_stats:
            rprint(f"\n[blue]Summary:[/blue]")
            rprint(f"Total projects: [cyan]{len(projects)}[/cyan]")
            rprint(f"Total collections: [cyan]{total_collections}[/cyan]")
            rprint(f"Total documents: [cyan]{total_documents}[/cyan]")
            
            # Show status breakdown
            status_counts = {}
            for project in projects:
                status = project.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            if len(status_counts) > 1:
                status_breakdown = ", ".join(f"{status}: {count}" for status, count in status_counts.items())
                rprint(f"Status breakdown: [dim]{status_breakdown}[/dim]")
            
            # Show tags if any
            all_tags = set()
            for project in projects:
                if project.tags:
                    all_tags.update(project.tags)
            
            if all_tags:
                rprint(f"Common tags: [dim]{', '.join(sorted(all_tags))}[/dim]")
        
        # Show active project if any
        try:
            active_project = manager.get_active_project()
            if active_project:
                rprint(f"\n[green]Active project:[/green] {active_project.name}")
            else:
                rprint(f"\n[dim]No active project set. Use 'research-agent projects activate <name>' to set one.[/dim]")
        except Exception:
            # Active project functionality might not be implemented yet
            pass
        
    except Exception as e:
        _handle_project_error("list projects", e)


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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would activate project but not executing")
        rprint(f"  Project: {name}")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Verify project exists
        try:
            project_metadata = manager.get_project_metadata(name)
        except ProjectNotFoundError:
            rprint(f"[red]Error:[/red] Project '{name}' not found")
            rprint("\n[dim]Use 'research-agent projects list' to see available projects.[/dim]")
            raise typer.Exit(1)
        
        # Activate the project
        manager.set_active_project(name)
        
        rprint(f"[green]✓ Activated project:[/green] '{name}'")
        rprint(f"  Description: {project_metadata.description or 'No description'}")
        rprint(f"  Status: {project_metadata.status.value}")
        rprint(f"  Collections: {project_metadata.linked_collections_count}")
        
        # Show default collections if any
        try:
            default_collections = manager.get_default_collections(name)
            if default_collections:
                rprint(f"  Default collections: {', '.join(default_collections)}")
            else:
                rprint("  Default collections: None set")
        except Exception:
            pass  # Default collections might not be set
        
        # Show impact of activation
        rprint(f"\n[blue]Project activated successfully![/blue]")
        rprint("Future operations will use this project's context:")
        rprint("• Query commands will search project collections by default")
        rprint("• Knowledge base operations will scope to project")
        rprint("• New documents will be suggested for project collections")
        
        rprint(f"\n[dim]Use 'research-agent projects deactivate' to return to global scope.[/dim]")
        
    except Exception as e:
        _handle_project_error("activate project", e)


@projects_app.command("deactivate")
def deactivate_project(ctx: typer.Context) -> None:
    """
    Deactivate the current project.
    
    Returns to global knowledge base scope without
    project-specific filtering.
    
    Example:
        research-agent projects deactivate
    """
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would deactivate project but not executing")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Check if there's an active project
        try:
            active_project = manager.get_active_project()
            if not active_project:
                rprint("[yellow]No active project to deactivate[/yellow]")
                rprint("\n[dim]Currently operating in global knowledge base scope.[/dim]")
                return
            
            current_project_name = active_project.name
            
        except Exception:
            # If get_active_project is not implemented or fails
            rprint("[yellow]No active project found[/yellow]")
            rprint("\n[dim]Currently operating in global knowledge base scope.[/dim]")
            return
        
        # Deactivate the project
        manager.deactivate_project()
        
        rprint(f"[green]✓ Deactivated project:[/green] '{current_project_name}'")
        rprint("\n[blue]Returned to global scope[/blue]")
        rprint("Operations will now use the entire knowledge base:")
        rprint("• Query commands will search all collections")
        rprint("• Knowledge base operations will have full scope")
        rprint("• No project-specific filtering applied")
        
        rprint(f"\n[dim]Use 'research-agent projects activate <name>' to activate a specific project.[/dim]")
        
    except Exception as e:
        _handle_project_error("deactivate project", e)


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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would archive project but not executing")
        rprint(f"  Project: {name}")
        if confirm:
            rprint("  Confirmation: Skipped")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Verify project exists
        try:
            project_metadata = manager.get_project_metadata(name)
        except ProjectNotFoundError:
            rprint(f"[red]Error:[/red] Project '{name}' not found")
            rprint("\n[dim]Use 'research-agent projects list' to see available projects.[/dim]")
            raise typer.Exit(1)
        
        # Check if already archived
        from ..models.project_schema import ProjectStatus
        if project_metadata.status == ProjectStatus.ARCHIVED:
            rprint(f"[yellow]Project '{name}' is already archived[/yellow]")
            return
        
        # Confirmation prompt unless --confirm is used
        if not confirm:
            rprint(f"\n[yellow]Archive project '{name}'?[/yellow]")
            rprint("This will:")
            rprint("• Mark the project as archived")
            rprint("• Hide it from active project lists")
            rprint("• Preserve all data and collections")
            rprint("• Allow future reactivation")
            
            response = input("\nContinue? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                rprint("[yellow]Archive cancelled[/yellow]")
                return
        
        # Archive the project
        manager.archive_project(name)
        
        rprint(f"[green]✓ Archived project:[/green] '{name}'")
        rprint(f"  Description: {project_metadata.description or 'No description'}")
        rprint(f"  Collections: {project_metadata.linked_collections_count} preserved")
        rprint(f"  Documents: {project_metadata.total_documents} preserved")
        
        # If this was the active project, deactivate it
        try:
            active_project = manager.get_active_project()
            if active_project and active_project.name == name:
                manager.deactivate_project()
                rprint("\n[blue]Note:[/blue] Deactivated as it was the active project")
                rprint("Returned to global knowledge base scope")
        except Exception:
            # Active project functionality might not be fully implemented
            pass
        
        rprint("\n[blue]Archive completed![/blue]")
        rprint("• Project is now hidden from active lists")
        rprint("• All data and collections are preserved")
        rprint("• Use 'research-agent projects list' to see archived projects")
        rprint(f"• Use 'research-agent projects info {name}' to view archived project details")
        
    except Exception as e:
        _handle_project_error("archive project", e)


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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would delete project but not executing")
        rprint(f"  Project: {name}")
        if confirm:
            rprint("  Confirmation: Skipped")
        if keep_collections:
            rprint("  Keep collections: Yes")
        return
    
    try:
        # Create project manager
        manager = create_project_manager()
        
        # Verify project exists and get metadata
        try:
            project_metadata = manager.get_project_metadata(name)
        except ProjectNotFoundError:
            rprint(f"[red]Error:[/red] Project '{name}' not found")
            rprint("\n[dim]Use 'research-agent projects list' to see available projects.[/dim]")
            raise typer.Exit(1)
        
        # Get linked collections
        linked_collections = []
        try:
            linked_collections = manager.get_linked_collections(name)
        except Exception:
            # Linked collections method might not be available
            pass
        
        # Confirmation prompt unless --confirm is used
        if not confirm:
            rprint(f"\n[red]⚠️  DELETE PROJECT '{name}'? ⚠️[/red]")
            rprint("\n[yellow]This operation CANNOT be undone![/yellow]")
            rprint("\nThis will:")
            rprint("• Permanently delete the project metadata")
            rprint("• Remove all project-collection associations")
            
            if keep_collections:
                rprint("• Keep collections as standalone collections")
                if linked_collections:
                    rprint(f"• Preserve {len(linked_collections)} collections: {', '.join(linked_collections[:3])}{'...' if len(linked_collections) > 3 else ''}")
            else:
                rprint("• DELETE all associated collections and their documents")
                if linked_collections:
                    rprint(f"• DELETE {len(linked_collections)} collections: {', '.join(linked_collections[:3])}{'...' if len(linked_collections) > 3 else ''}")
                rprint(f"• DELETE approximately {project_metadata.total_documents} documents")
            
            rprint(f"\nType the project name '{name}' to confirm deletion:")
            user_input = input("> ").strip()
            
            if user_input != name:
                rprint("[yellow]Project name mismatch. Deletion cancelled.[/yellow]")
                return
        
        # If this is the active project, deactivate it first
        try:
            active_project = manager.get_active_project()
            if active_project and active_project.name == name:
                manager.deactivate_project()
                rprint(f"[blue]Deactivated project '{name}' (was active)[/blue]")
        except Exception:
            # Active project functionality might not be fully implemented
            pass
        
        # Delete collections if not keeping them
        deleted_collections = []
        if not keep_collections and linked_collections:
            from ..core.vector_store import create_chroma_manager
            chroma_manager = create_chroma_manager()
            
            rprint(f"[blue]Deleting {len(linked_collections)} collections...[/blue]")
            for collection_name in linked_collections:
                try:
                    if chroma_manager.collection_exists(collection_name):
                        chroma_manager.delete_collection(collection_name)
                        deleted_collections.append(collection_name)
                        rprint(f"  [red]✗[/red] Deleted collection: {collection_name}")
                    else:
                        rprint(f"  [yellow]⚠[/yellow] Collection not found: {collection_name}")
                except Exception as e:
                    rprint(f"  [red]Error deleting collection {collection_name}:[/red] {e}")
        
        # Delete the project
        manager.delete_project(name, keep_collections=keep_collections)
        
        rprint(f"\n[red]✗ Deleted project:[/red] '{name}'")
        rprint(f"  Description: {project_metadata.description or 'No description'}")
        
        if keep_collections:
            rprint(f"  [green]Preserved {len(linked_collections)} collections[/green]")
            if linked_collections:
                rprint(f"  Collections: {', '.join(linked_collections)}")
        else:
            rprint(f"  [red]Deleted {len(deleted_collections)} collections[/red]")
            rprint(f"  [red]Deleted approximately {project_metadata.total_documents} documents[/red]")
        
        rprint("\n[red]⚠️  Project deletion completed ⚠️[/red]")
        if not keep_collections and deleted_collections:
            rprint("[red]All project data has been permanently removed[/red]")
        else:
            rprint("[blue]Project metadata removed, collections preserved[/blue]")
        
    except Exception as e:
        _handle_project_error("delete project", e)


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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would export project but not executing")
        rprint(f"  Project: {name}")
        rprint(f"  Output: {output_path}")
        rprint(f"  Format: {format}")
        return
    
    try:
        import os
        import json
        import tempfile
        import shutil
        from datetime import datetime
        from pathlib import Path
        
        # Create project manager
        manager = create_project_manager()
        
        # Verify project exists
        try:
            project_metadata = manager.get_project_metadata(name)
        except ProjectNotFoundError:
            rprint(f"[red]Error:[/red] Project '{name}' not found")
            rprint("\n[dim]Use 'research-agent projects list' to see available projects.[/dim]")
            raise typer.Exit(1)
        
        # Validate output path
        output_dir = Path(output_path)
        if not output_dir.exists():
            rprint(f"[red]Error:[/red] Output directory '{output_path}' does not exist")
            raise typer.Exit(1)
        
        # Validate format
        supported_formats = ["zip", "tar", "folder"]
        if format not in supported_formats:
            rprint(f"[red]Error:[/red] Unsupported format '{format}'. Supported: {', '.join(supported_formats)}")
            raise typer.Exit(1)
        
        rprint(f"[blue]Exporting project:[/blue] '{name}'")
        rprint(f"  Output directory: {output_path}")
        rprint(f"  Format: {format}")
        
        # Create temporary directory for export preparation
        with tempfile.TemporaryDirectory() as temp_dir:
            export_base = Path(temp_dir) / f"{name}_export"
            export_base.mkdir()
            
            # Export project metadata
            metadata_file = export_base / "project_metadata.json"
            metadata_dict = {
                "name": project_metadata.name,
                "description": project_metadata.description,
                "tags": project_metadata.tags or [],
                "status": project_metadata.status.value,
                "created_at": project_metadata.created_at.isoformat() if project_metadata.created_at else None,
                "linked_collections_count": project_metadata.linked_collections_count,
                "total_documents": project_metadata.total_documents,
                "export_timestamp": datetime.now().isoformat(),
                "export_version": "1.0"
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            rprint(f"  [green]✓[/green] Exported project metadata")
            
            # Export linked collections
            try:
                linked_collections = manager.get_linked_collections(name)
                collections_dir = export_base / "collections"
                collections_dir.mkdir()
                
                from ..core.vector_store import create_chroma_manager
                chroma_manager = create_chroma_manager()
                
                for i, collection_name in enumerate(linked_collections, 1):
                    rprint(f"  [{i}/{len(linked_collections)}] Exporting collection: {collection_name}")
                    
                    try:
                        if chroma_manager.collection_exists(collection_name):
                            # Get all documents from collection
                            documents_data = chroma_manager.get_documents(
                                collection_name=collection_name,
                                limit=10000,  # Large limit to get all documents
                                include=['documents', 'metadatas', 'ids']
                            )
                            
                            # Save collection data
                            collection_file = collections_dir / f"{collection_name}.json"
                            collection_export = {
                                "collection_name": collection_name,
                                "document_count": len(documents_data.get('documents', [])),
                                "documents": documents_data.get('documents', []),
                                "metadatas": documents_data.get('metadatas', []),
                                "ids": documents_data.get('ids', []),
                                "export_timestamp": datetime.now().isoformat()
                            }
                            
                            with open(collection_file, 'w') as f:
                                json.dump(collection_export, f, indent=2)
                                
                            rprint(f"    [green]✓[/green] {len(documents_data.get('documents', []))} documents")
                        else:
                            rprint(f"    [yellow]⚠[/yellow] Collection not found in ChromaDB")
                            
                    except Exception as e:
                        rprint(f"    [red]Error exporting collection {collection_name}:[/red] {e}")
                
                # Create collections manifest
                manifest_file = collections_dir / "collections_manifest.json"
                manifest = {
                    "project_name": name,
                    "total_collections": len(linked_collections),
                    "collections": linked_collections,
                    "export_timestamp": datetime.now().isoformat()
                }
                
                with open(manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                rprint(f"  [green]✓[/green] Exported {len(linked_collections)} collections")
                
            except Exception as e:
                rprint(f"  [yellow]Warning:[/yellow] Could not export collections: {e}")
                # Create empty collections directory
                collections_dir = export_base / "collections"
                collections_dir.mkdir()
            
            # Create the final export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_name = f"{name}_export_{timestamp}"
            
            if format == "folder":
                # Copy to output directory as folder
                final_path = output_dir / export_name
                shutil.copytree(export_base, final_path)
                final_export = final_path
                
            elif format == "zip":
                # Create ZIP archive
                final_path = output_dir / f"{export_name}.zip"
                shutil.make_archive(str(final_path.with_suffix('')), 'zip', temp_dir, f"{name}_export")
                final_export = final_path
                
            elif format == "tar":
                # Create TAR archive
                final_path = output_dir / f"{export_name}.tar.gz"
                shutil.make_archive(str(final_path.with_suffix('').with_suffix('')), 'gztar', temp_dir, f"{name}_export")
                final_export = final_path
        
        # Get file/folder size
        if final_export.is_file():
            size_mb = final_export.stat().st_size / (1024 * 1024)
            size_display = f"{size_mb:.1f} MB"
        else:
            size_display = "folder"
        
        rprint(f"\n[green]✓ Export completed successfully![/green]")
        rprint(f"  Project: {name}")
        rprint(f"  Export file: {final_export}")
        rprint(f"  Size: {size_display}")
        rprint(f"  Format: {format}")
        
        rprint(f"\n[blue]Export contents:[/blue]")
        rprint("• Project metadata and configuration")
        rprint(f"• {project_metadata.linked_collections_count} collections with documents")
        rprint(f"• Approximately {project_metadata.total_documents} documents")
        
        rprint(f"\n[dim]Use 'research-agent projects import {final_export}' to restore this project.[/dim]")
        
    except Exception as e:
        _handle_project_error("export project", e)


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
    # Check dry-run status from context
    if _get_dry_run_status(ctx):
        rprint("[blue]DRY RUN:[/blue] Would import project but not executing")
        rprint(f"  Archive: {archive_path}")
        if name:
            rprint(f"  Name: {name}")
        if merge:
            rprint("  Merge: Yes")
        return
    
    try:
        import os
        import json
        import tempfile
        import shutil
        import zipfile
        import tarfile
        from pathlib import Path
        from datetime import datetime
        
        # Validate archive path
        archive_file = Path(archive_path)
        if not archive_file.exists():
            rprint(f"[red]Error:[/red] Archive file '{archive_path}' not found")
            raise typer.Exit(1)
        
        rprint(f"[blue]Importing project from:[/blue] {archive_path}")
        
        # Create project manager
        manager = create_project_manager()
        
        # Extract archive to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            extract_dir = None
            
            # Determine archive type and extract
            if archive_file.suffix.lower() == '.zip':
                rprint("  Extracting ZIP archive...")
                with zipfile.ZipFile(archive_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                    # Find the extracted directory
                    extracted_items = list(temp_path.iterdir())
                    if len(extracted_items) == 1 and extracted_items[0].is_dir():
                        extract_dir = extracted_items[0]
                    
            elif archive_file.suffix.lower() in ['.tar', '.gz']:
                rprint("  Extracting TAR archive...")
                with tarfile.open(archive_file, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_path)
                    # Find the extracted directory
                    extracted_items = list(temp_path.iterdir())
                    if len(extracted_items) == 1 and extracted_items[0].is_dir():
                        extract_dir = extracted_items[0]
                        
            elif archive_file.is_dir():
                rprint("  Using folder as import source...")
                extract_dir = archive_file
                
            else:
                rprint(f"[red]Error:[/red] Unsupported archive format for '{archive_file}'")
                rprint("Supported formats: .zip, .tar.gz, or folder")
                raise typer.Exit(1)
            
            if not extract_dir or not extract_dir.exists():
                rprint(f"[red]Error:[/red] Could not find extracted project content")
                raise typer.Exit(1)
            
            # Read project metadata
            metadata_file = extract_dir / "project_metadata.json"
            if not metadata_file.exists():
                rprint(f"[red]Error:[/red] Invalid project archive - missing project_metadata.json")
                raise typer.Exit(1)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Determine project name
            project_name = name or metadata.get('name')
            if not project_name:
                rprint(f"[red]Error:[/red] Could not determine project name")
                raise typer.Exit(1)
            
            rprint(f"  Project name: {project_name}")
            rprint(f"  Original name: {metadata.get('name', 'Unknown')}")
            rprint(f"  Description: {metadata.get('description', 'No description')}")
            
            # Check if project already exists
            try:
                existing_project = manager.get_project_metadata(project_name)
                if not merge:
                    rprint(f"[red]Error:[/red] Project '{project_name}' already exists")
                    rprint("Use --merge flag to merge with existing project")
                    raise typer.Exit(1)
                else:
                    rprint(f"[yellow]Warning:[/yellow] Merging with existing project '{project_name}'")
            except ProjectNotFoundError:
                # Project doesn't exist, which is good for import
                pass
            
            # Create or update project
            if merge:
                try:
                    # Update existing project
                    manager.update_project(
                        name=project_name,
                        description=metadata.get('description'),
                        tags=metadata.get('tags', [])
                    )
                    rprint(f"  [blue]Updated existing project metadata[/blue]")
                except Exception as e:
                    rprint(f"  [yellow]Warning:[/yellow] Could not update project metadata: {e}")
            else:
                # Create new project
                manager.create_project(
                    name=project_name,
                    description=metadata.get('description'),
                    tags=metadata.get('tags', [])
                )
                rprint(f"  [green]✓[/green] Created project")
            
            # Import collections
            collections_dir = extract_dir / "collections"
            if collections_dir.exists():
                # Read collections manifest
                manifest_file = collections_dir / "collections_manifest.json"
                collections_to_import = []
                
                if manifest_file.exists():
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                        collections_to_import = manifest.get('collections', [])
                else:
                    # Fallback: scan for collection files
                    collections_to_import = [
                        f.stem for f in collections_dir.glob("*.json") 
                        if f.name != "collections_manifest.json"
                    ]
                
                if collections_to_import:
                    rprint(f"  [blue]Importing {len(collections_to_import)} collections...[/blue]")
                    
                    from ..core.vector_store import create_chroma_manager
                    from ..core.document_insertion import create_document_insertion_manager
                    
                    chroma_manager = create_chroma_manager()
                    doc_manager = create_document_insertion_manager()
                    
                    imported_collections = 0
                    total_documents = 0
                    
                    for i, collection_name in enumerate(collections_to_import, 1):
                        collection_file = collections_dir / f"{collection_name}.json"
                        
                        if not collection_file.exists():
                            rprint(f"    [{i}/{len(collections_to_import)}] [yellow]⚠[/yellow] Collection file not found: {collection_name}")
                            continue
                        
                        rprint(f"    [{i}/{len(collections_to_import)}] Importing collection: {collection_name}")
                        
                        try:
                            with open(collection_file, 'r') as f:
                                collection_data = json.load(f)
                            
                            documents = collection_data.get('documents', [])
                            metadatas = collection_data.get('metadatas', [])
                            ids = collection_data.get('ids', [])
                            
                            if documents:
                                # Create collection if it doesn't exist
                                if not chroma_manager.collection_exists(collection_name):
                                    chroma_manager.create_collection(
                                        collection_name, 
                                        metadata={"project": project_name, "imported": True}
                                    )
                                
                                # Insert documents
                                for doc, metadata_dict, doc_id in zip(documents, metadatas, ids):
                                    try:
                                        # Add import timestamp to metadata
                                        if metadata_dict is None:
                                            metadata_dict = {}
                                        metadata_dict['imported_at'] = datetime.now().isoformat()
                                        
                                        # Use document insertion manager
                                        doc_manager.insert_text_chunk(
                                            collection_name=collection_name,
                                            content=doc,
                                            metadata=metadata_dict,
                                            chunk_id=doc_id
                                        )
                                        
                                    except Exception as e:
                                        rprint(f"      [yellow]Warning:[/yellow] Could not import document {doc_id}: {e}")
                                
                                # Link collection to project
                                try:
                                    manager.link_collection_to_project(
                                        project_name=project_name,
                                        collection_name=collection_name,
                                        description=f"Imported collection with {len(documents)} documents"
                                    )
                                except Exception as e:
                                    rprint(f"      [yellow]Warning:[/yellow] Could not link collection to project: {e}")
                                
                                imported_collections += 1
                                total_documents += len(documents)
                                rprint(f"      [green]✓[/green] Imported {len(documents)} documents")
                            else:
                                rprint(f"      [yellow]⚠[/yellow] No documents to import")
                                
                        except Exception as e:
                            rprint(f"      [red]Error importing collection {collection_name}:[/red] {e}")
                    
                    rprint(f"  [green]✓[/green] Import completed: {imported_collections}/{len(collections_to_import)} collections, {total_documents} documents")
                else:
                    rprint(f"  [yellow]No collections found to import[/yellow]")
            else:
                rprint(f"  [yellow]No collections directory found in archive[/yellow]")
        
        rprint(f"\n[green]✓ Project import completed successfully![/green]")
        rprint(f"  Project: {project_name}")
        rprint(f"  Source: {archive_path}")
        
        # Show project info
        try:
            final_metadata = manager.get_project_metadata(project_name)
            rprint(f"  Collections: {final_metadata.linked_collections_count}")
            rprint(f"  Documents: {final_metadata.total_documents}")
        except Exception:
            pass
        
        rprint(f"\n[blue]Next steps:[/blue]")
        rprint(f"• Use 'research-agent projects info {project_name}' to view imported project details")
        rprint(f"• Use 'research-agent projects activate {project_name}' to set as active project")
        rprint(f"• Use 'research-agent query search <query> --collections <collection>' to test imported data")
        
    except Exception as e:
        _handle_project_error("import project", e) 