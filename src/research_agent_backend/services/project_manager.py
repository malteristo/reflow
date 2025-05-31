"""
Project Management Service for Research Agent.

This module provides the core project management functionality including
project-collection linking, metadata storage, and context detection.

Implements FR-KB-005: Project and collection management.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.vector_store import create_chroma_manager, CollectionNotFoundError
from ..models.metadata_schema.project_metadata import (
    ProjectMetadata,
    ProjectInfo,
    CollectionLinkMetadata,
    ProjectContext,
    ProjectStatus
)
from ..exceptions.project_exceptions import (
    ProjectNotFoundError,
    ProjectAlreadyExistsError,
    CollectionAlreadyLinkedError,
    CollectionNotLinkedError,
    ProjectContextError,
    ProjectMetadataError
)

# Set up logging
logger = logging.getLogger(__name__)


class ProjectManager:
    """
    Manages project-specific knowledge base operations.
    
    Provides functionality for linking collections to projects,
    managing default collections, and detecting project context.
    
    Implements FR-KB-005: Project and collection management.
    
    Attributes:
        storage_path (Path): Directory path for storing project metadata
        projects_file (Path): Path to projects.json file
        links_file (Path): Path to collection links file
        context_file (Path): Path to context configuration file
        
    Example:
        >>> manager = ProjectManager()
        >>> manager.create_project("my-research", "AI research project")
        >>> manager.link_collection("my-research", "papers")
    """
    
    def __init__(self, storage_path: str = "projects"):
        """
        Initialize the project manager.
        
        Args:
            storage_path: Directory path for storing project metadata.
                         Defaults to "projects" in current directory.
                         
        Raises:
            ProjectMetadataError: If storage initialization fails
        """
        try:
            self.storage_path = Path(storage_path)
            self.storage_path.mkdir(exist_ok=True)
            
            # File paths
            self.projects_file = self.storage_path / "projects.json"
            self.links_file = self.storage_path / "collection_links.json"
            self.context_file = self.storage_path / "context.json"
            
            # Initialize storage files if they don't exist
            self._ensure_storage_files()
            
            # Current context
            self._context = self._load_context()
            
            logger.info(f"ProjectManager initialized with storage path: {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProjectManager: {e}")
            raise ProjectMetadataError(f"Storage initialization failed: {e}")
    
    def _ensure_storage_files(self) -> None:
        """
        Ensure all storage files exist with default content.
        
        Creates the required JSON files with proper default structures
        if they don't already exist.
        
        Raises:
            ProjectMetadataError: If file creation fails
        """
        try:
            if not self.projects_file.exists():
                self._save_json(self.projects_file, {})
                logger.debug(f"Created projects file: {self.projects_file}")
            
            if not self.links_file.exists():
                self._save_json(self.links_file, {})
                logger.debug(f"Created links file: {self.links_file}")
            
            if not self.context_file.exists():
                default_context = {
                    "active_project": None,
                    "default_collections": [],
                    "project_paths": {}
                }
                self._save_json(self.context_file, default_context)
                logger.debug(f"Created context file: {self.context_file}")
                
        except Exception as e:
            logger.error(f"Failed to ensure storage files: {e}")
            raise ProjectMetadataError(f"Failed to create storage files: {e}")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """
        Load JSON data from file with error handling.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data, empty dict if file doesn't exist
            or contains invalid JSON
            
        Note:
            This method handles FileNotFoundError and JSONDecodeError gracefully
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Loaded JSON from {file_path}")
                return data
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {file_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load JSON from {file_path}: {e}")
            return {}
    
    def _save_json(self, file_path: Path, data: Dict[str, Any]) -> None:
        """
        Save JSON data to file with error handling.
        
        Args:
            file_path: Path where to save the JSON file
            data: Dictionary to save as JSON
            
        Raises:
            ProjectMetadataError: If file writing fails
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved JSON to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON to {file_path}: {e}")
            raise ProjectMetadataError(f"Failed to save data to {file_path}: {e}")
    
    def _validate_project_name(self, name: str) -> None:
        """
        Validate project name format and constraints.
        
        Args:
            name: Project name to validate
            
        Raises:
            ProjectMetadataError: If name is invalid
        """
        if not name:
            raise ProjectMetadataError("Project name cannot be empty")
        
        if not isinstance(name, str):
            raise ProjectMetadataError("Project name must be a string")
        
        if len(name.strip()) == 0:
            raise ProjectMetadataError("Project name cannot be only whitespace")
        
        if len(name) > 100:
            raise ProjectMetadataError("Project name too long (max 100 characters)")
        
        # Check for invalid characters that could cause file system issues
        invalid_chars = '<>:"/\\|?*'
        if any(char in name for char in invalid_chars):
            raise ProjectMetadataError(f"Project name contains invalid characters: {invalid_chars}")
    
    def _validate_collection_name(self, name: str) -> None:
        """
        Validate collection name format and constraints.
        
        Args:
            name: Collection name to validate
            
        Raises:
            ProjectMetadataError: If name is invalid
        """
        if not name:
            raise ProjectMetadataError("Collection name cannot be empty")
        
        if not isinstance(name, str):
            raise ProjectMetadataError("Collection name must be a string")
        
        if len(name.strip()) == 0:
            raise ProjectMetadataError("Collection name cannot be only whitespace")
        
        if len(name) > 100:
            raise ProjectMetadataError("Collection name too long (max 100 characters)")
    
    def _load_context(self) -> ProjectContext:
        """
        Load current project context from storage.
        
        Returns:
            ProjectContext object with current context data
        """
        try:
            data = self._load_json(self.context_file)
            context = ProjectContext(
                active_project=data.get("active_project"),
                default_collections=data.get("default_collections", []),
                project_paths=data.get("project_paths", {})
            )
            logger.debug(f"Loaded project context: active={context.active_project}")
            return context
        except Exception as e:
            logger.error(f"Failed to load context: {e}")
            return ProjectContext()
    
    def _save_context(self) -> None:
        """
        Save current project context to storage.
        
        Raises:
            ProjectMetadataError: If context saving fails
        """
        try:
            data = {
                "active_project": self._context.active_project,
                "default_collections": self._context.default_collections,
                "project_paths": self._context.project_paths
            }
            self._save_json(self.context_file, data)
            logger.debug("Saved project context")
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
            raise ProjectMetadataError(f"Failed to save context: {e}")
    
    def create_project(
        self, 
        name: str, 
        description: Optional[str] = None, 
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Create a new project with metadata storage.
        
        Implements FR-KB-005.1: Project creation and metadata management.
        
        Args:
            name: Unique project name (max 100 chars, no invalid filesystem chars)
            description: Optional project description
            tags: Optional list of tags for categorization
            
        Returns:
            True if project was created successfully
            
        Raises:
            ProjectAlreadyExistsError: If project with same name already exists
            ProjectMetadataError: If project creation fails due to validation or storage errors
            
        Example:
            >>> manager.create_project("ai-research", "Research project on AI", ["ai", "ml"])
            True
        """
        self._validate_project_name(name)
        
        try:
            projects = self._load_json(self.projects_file)
            
            if name in projects:
                logger.warning(f"Attempt to create existing project: {name}")
                raise ProjectAlreadyExistsError(name)
            
            # Create metadata with validation
            metadata = ProjectMetadata(
                name=name,
                description=description,
                tags=tags or [],
                owner_id="default_user"  # TODO: Replace with actual user context
            )
            
            # Convert to dict for JSON storage
            project_data = {
                "name": metadata.name,
                "description": metadata.description,
                "tags": metadata.tags,
                "status": metadata.status.value,
                "created_at": metadata.created_at,
                "updated_at": metadata.updated_at,
                "owner_id": metadata.owner_id,
                "team_id": metadata.team_id,
                "linked_collections": metadata.linked_collections,
                "default_collections": metadata.default_collections,
                "config": metadata.config
            }
            
            projects[name] = project_data
            self._save_json(self.projects_file, projects)
            
            logger.info(f"Created project: {name}")
            return True
            
        except ProjectAlreadyExistsError:
            raise
        except Exception as e:
            logger.error(f"Failed to create project {name}: {e}")
            raise ProjectMetadataError(f"Project creation failed: {e}", name, "create")
    
    def get_project_metadata(self, name: str) -> ProjectMetadata:
        """
        Get project metadata with computed statistics.
        
        Implements FR-KB-005.2: Project metadata retrieval and statistics.
        
        Args:
            name: Project name to retrieve metadata for
            
        Returns:
            ProjectMetadata object with computed statistics including
            linked collection count and total document count
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            ProjectMetadataError: If metadata retrieval fails
            
        Example:
            >>> metadata = manager.get_project_metadata("ai-research")
            >>> print(f"Project has {metadata.linked_collections_count} collections")
        """
        self._validate_project_name(name)
        
        try:
            projects = self._load_json(self.projects_file)
            
            if name not in projects:
                logger.warning(f"Attempt to get metadata for non-existent project: {name}")
                raise ProjectNotFoundError(name)
            
            data = projects[name]
            
            # Convert back to ProjectMetadata with validation
            metadata = ProjectMetadata(
                name=data["name"],
                description=data.get("description"),
                tags=data.get("tags", []),
                status=ProjectStatus(data.get("status", "active")),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                owner_id=data.get("owner_id"),
                team_id=data.get("team_id"),
                linked_collections=data.get("linked_collections", []),
                default_collections=data.get("default_collections", []),
                config=data.get("config", {})
            )
            
            # Update computed statistics
            metadata.linked_collections_count = len(metadata.linked_collections)
            
            # Get total documents from linked collections with error handling
            total_docs = 0
            try:
                vector_manager = create_chroma_manager()
                for coll_name in metadata.linked_collections:
                    try:
                        stats = vector_manager.collection_manager.get_collection_stats(coll_name)
                        total_docs += stats.document_count
                    except CollectionNotFoundError:
                        logger.warning(f"Linked collection not found: {coll_name}")
                        continue  # Collection may have been deleted
                metadata.total_documents = total_docs
            except Exception as e:
                logger.warning(f"Failed to compute document statistics for {name}: {e}")
                metadata.total_documents = 0
            
            logger.debug(f"Retrieved metadata for project: {name}")
            return metadata
            
        except ProjectNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get metadata for project {name}: {e}")
            raise ProjectMetadataError(f"Metadata retrieval failed: {e}", name, "get_metadata")
    
    def update_project_metadata(self, name: str, **updates) -> bool:
        """
        Update project metadata fields.
        
        Implements FR-KB-005.3: Project metadata modification.
        
        Args:
            name: Project name to update
            **updates: Field-value pairs to update. Supported fields:
                      description, tags, status, config
            
        Returns:
            True if update was successful
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            ProjectMetadataError: If update fails or invalid fields provided
            
        Example:
            >>> manager.update_project_metadata("ai-research", 
            ...                                  description="Updated description",
            ...                                  tags=["ai", "nlp", "research"])
            True
        """
        self._validate_project_name(name)
        
        if not updates:
            logger.warning(f"No updates provided for project: {name}")
            return True
        
        try:
            projects = self._load_json(self.projects_file)
            
            if name not in projects:
                logger.warning(f"Attempt to update non-existent project: {name}")
                raise ProjectNotFoundError(name)
            
            # Validate update fields
            allowed_fields = {"description", "tags", "status", "config"}
            invalid_fields = set(updates.keys()) - allowed_fields
            if invalid_fields:
                raise ProjectMetadataError(f"Invalid update fields: {invalid_fields}")
            
            # Apply updates with validation
            for key, value in updates.items():
                if key == "tags" and value is not None:
                    if not isinstance(value, list):
                        raise ProjectMetadataError("Tags must be a list")
                    if not all(isinstance(tag, str) for tag in value):
                        raise ProjectMetadataError("All tags must be strings")
                elif key == "status" and value is not None:
                    if not isinstance(value, str):
                        raise ProjectMetadataError("Status must be a string")
                    try:
                        ProjectStatus(value)  # Validate status value
                    except ValueError:
                        raise ProjectMetadataError(f"Invalid status value: {value}")
                elif key == "description" and value is not None:
                    if not isinstance(value, str):
                        raise ProjectMetadataError("Description must be a string")
                    if len(value) > 1000:
                        raise ProjectMetadataError("Description too long (max 1000 characters)")
                
                projects[name][key] = value
            
            # Update timestamp
            projects[name]["updated_at"] = datetime.utcnow().isoformat()
            
            self._save_json(self.projects_file, projects)
            logger.info(f"Updated project metadata: {name}")
            return True
            
        except ProjectNotFoundError:
            raise
        except ProjectMetadataError:
            raise
        except Exception as e:
            logger.error(f"Failed to update project {name}: {e}")
            raise ProjectMetadataError(f"Project update failed: {e}", name, "update")
    
    def link_collection(
        self, 
        project_name: str, 
        collection_name: str, 
        description: Optional[str] = None
    ) -> bool:
        """
        Link a collection to a project with metadata storage.
        
        Implements FR-KB-005.4: Project-collection relationship management.
        
        Args:
            project_name: Name of the project to link to
            collection_name: Name of the collection to link
            description: Optional description of the relationship
            
        Returns:
            True if collection was linked successfully
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            CollectionNotFoundError: If collection doesn't exist in vector store
            CollectionAlreadyLinkedError: If collection is already linked to this project
            ProjectMetadataError: If linking operation fails
            
        Example:
            >>> manager.link_collection("ai-research", "research-papers", 
            ...                          "Core research papers collection")
            True
        """
        self._validate_project_name(project_name)
        self._validate_collection_name(collection_name)
        
        try:
            # Check if project exists
            projects = self._load_json(self.projects_file)
            if project_name not in projects:
                logger.warning(f"Attempt to link collection to non-existent project: {project_name}")
                raise ProjectNotFoundError(project_name)
            
            # Check if collection exists in vector store
            try:
                vector_manager = create_chroma_manager()
                vector_manager.collection_manager.get_collection(collection_name)
                logger.debug(f"Verified collection exists: {collection_name}")
            except CollectionNotFoundError:
                logger.error(f"Collection not found in vector store: {collection_name}")
                raise CollectionNotFoundError(f"Collection '{collection_name}' not found")
            
            # Check if already linked
            links = self._load_json(self.links_file)
            link_key = f"{project_name}:{collection_name}"
            
            if link_key in links:
                logger.warning(f"Collection already linked: {collection_name} -> {project_name}")
                raise CollectionAlreadyLinkedError(collection_name, project_name)
            
            # Create the link metadata
            link_metadata = CollectionLinkMetadata(
                collection_name=collection_name,
                project_name=project_name,
                description=description,
                linked_by="default_user"  # TODO: Replace with actual user context
            )
            
            # Store link with validation
            link_data = {
                "collection_name": link_metadata.collection_name,
                "project_name": link_metadata.project_name,
                "description": link_metadata.description,
                "is_default": link_metadata.is_default,
                "linked_at": link_metadata.linked_at,
                "linked_by": link_metadata.linked_by
            }
            links[link_key] = link_data
            
            # Update project metadata
            if collection_name not in projects[project_name]["linked_collections"]:
                projects[project_name]["linked_collections"].append(collection_name)
                projects[project_name]["updated_at"] = datetime.utcnow().isoformat()
            
            # Atomic save operation
            self._save_json(self.links_file, links)
            self._save_json(self.projects_file, projects)
            
            logger.info(f"Linked collection {collection_name} to project {project_name}")
            return True
            
        except (ProjectNotFoundError, CollectionNotFoundError, CollectionAlreadyLinkedError):
            raise
        except Exception as e:
            logger.error(f"Failed to link collection {collection_name} to project {project_name}: {e}")
            raise ProjectMetadataError(f"Collection linking failed: {e}", project_name, "link_collection")
    
    def unlink_collection(self, project_name: str, collection_name: str) -> bool:
        """
        Remove a collection link from a project.
        
        Implements FR-KB-005.5: Project-collection relationship removal.
        
        Args:
            project_name: Name of the project to unlink from
            collection_name: Name of the collection to unlink
            
        Returns:
            True if collection was unlinked successfully
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            CollectionNotLinkedError: If collection is not linked to this project
            ProjectMetadataError: If unlinking operation fails
            
        Example:
            >>> manager.unlink_collection("ai-research", "old-papers")
            True
        """
        self._validate_project_name(project_name)
        self._validate_collection_name(collection_name)
        
        try:
            # Check if project exists
            projects = self._load_json(self.projects_file)
            if project_name not in projects:
                logger.warning(f"Attempt to unlink collection from non-existent project: {project_name}")
                raise ProjectNotFoundError(project_name)
            
            # Check if collection is linked
            links = self._load_json(self.links_file)
            link_key = f"{project_name}:{collection_name}"
            
            if link_key not in links:
                logger.warning(f"Collection not linked: {collection_name} -> {project_name}")
                raise CollectionNotLinkedError(collection_name, project_name)
            
            # Remove the link
            del links[link_key]
            
            # Update project metadata
            if collection_name in projects[project_name]["linked_collections"]:
                projects[project_name]["linked_collections"].remove(collection_name)
                projects[project_name]["updated_at"] = datetime.utcnow().isoformat()
            
            # Remove from default collections if present
            if collection_name in projects[project_name]["default_collections"]:
                projects[project_name]["default_collections"].remove(collection_name)
            
            # Atomic save operation
            self._save_json(self.links_file, links)
            self._save_json(self.projects_file, projects)
            
            logger.info(f"Unlinked collection {collection_name} from project {project_name}")
            return True
            
        except (ProjectNotFoundError, CollectionNotLinkedError):
            raise
        except Exception as e:
            logger.error(f"Failed to unlink collection {collection_name} from project {project_name}: {e}")
            raise ProjectMetadataError(f"Collection unlinking failed: {e}", project_name, "unlink_collection")
    
    def set_default_collections(
        self, 
        project_name: str, 
        collection_names: List[str], 
        append: bool = False
    ) -> bool:
        """
        Set or update default collections for a project.
        
        Implements FR-KB-005.6: Project default collection management.
        
        Args:
            project_name: Name of the project to update
            collection_names: List of collection names to set as defaults
            append: If True, append to existing defaults; if False, replace them
            
        Returns:
            True if default collections were updated successfully
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            CollectionNotLinkedError: If any collection is not linked to the project
            ProjectMetadataError: If operation fails
            
        Example:
            >>> manager.set_default_collections("ai-research", ["papers", "notes"])
            True
            >>> manager.set_default_collections("ai-research", ["extra"], append=True)
            True
        """
        self._validate_project_name(project_name)
        
        if not collection_names:
            logger.warning(f"Empty collection names list provided for project: {project_name}")
            return True
        
        for name in collection_names:
            self._validate_collection_name(name)
        
        try:
            projects = self._load_json(self.projects_file)
            
            if project_name not in projects:
                logger.warning(f"Attempt to set defaults for non-existent project: {project_name}")
                raise ProjectNotFoundError(project_name)
            
            project_data = projects[project_name]
            linked_collections = set(project_data.get("linked_collections", []))
            
            # Validate that all collections are linked to the project
            unlinked_collections = [name for name in collection_names if name not in linked_collections]
            if unlinked_collections:
                logger.warning(f"Collections not linked to project {project_name}: {unlinked_collections}")
                raise CollectionNotLinkedError(
                    f"Collections not linked to project: {unlinked_collections}", project_name
                )
            
            # Update default collections
            if append:
                current_defaults = set(project_data.get("default_collections", []))
                new_defaults = list(current_defaults.union(set(collection_names)))
                logger.debug(f"Appending to defaults for {project_name}: {collection_names}")
            else:
                new_defaults = list(set(collection_names))  # Remove duplicates
                logger.debug(f"Replacing defaults for {project_name}: {collection_names}")
            
            project_data["default_collections"] = new_defaults
            project_data["updated_at"] = datetime.utcnow().isoformat()
            
            self._save_json(self.projects_file, projects)
            
            logger.info(f"Updated default collections for project {project_name}: {new_defaults}")
            return True
            
        except (ProjectNotFoundError, CollectionNotLinkedError):
            raise
        except Exception as e:
            logger.error(f"Failed to set default collections for project {project_name}: {e}")
            raise ProjectMetadataError(f"Default collections update failed: {e}", project_name, "set_defaults")
    
    def get_project_collections(self, project_name: str) -> List[str]:
        """
        Get all collections linked to a project.
        
        Implements FR-KB-005.7: Project collection listing.
        
        Args:
            project_name: Name of the project to get collections for
            
        Returns:
            List of collection names linked to the project
            
        Raises:
            ProjectNotFoundError: If project doesn't exist
            ProjectMetadataError: If retrieval fails
            
        Example:
            >>> collections = manager.get_project_collections("ai-research")
            >>> print(f"Project has collections: {collections}")
        """
        self._validate_project_name(project_name)
        
        try:
            projects = self._load_json(self.projects_file)
            
            if project_name not in projects:
                logger.warning(f"Attempt to get collections for non-existent project: {project_name}")
                raise ProjectNotFoundError(project_name)
            
            collections = projects[project_name].get("linked_collections", [])
            logger.debug(f"Retrieved {len(collections)} collections for project {project_name}")
            return collections
            
        except ProjectNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get collections for project {project_name}: {e}")
            raise ProjectMetadataError(f"Collection retrieval failed: {e}", project_name, "get_collections")
    
    def detect_project_from_path(self, file_path: str) -> Optional[str]:
        """
        Detect project context based on file path.
        
        Implements FR-KB-005.8: Automatic project context detection.
        
        Uses stored project path mappings to automatically determine
        which project a file belongs to based on its path.
        
        Args:
            file_path: File path to analyze for project context
            
        Returns:
            Project name if detected, None if no project found
            
        Raises:
            ProjectContextError: If path detection logic fails
            
        Example:
            >>> project = manager.detect_project_from_path("/home/user/ai-research/paper.md")
            >>> print(f"Detected project: {project}")
        """
        if not file_path:
            logger.warning("Empty file path provided for project detection")
            return None
        
        try:
            normalized_path = os.path.abspath(file_path)
            logger.debug(f"Detecting project for path: {normalized_path}")
            
            # Check current context for path mappings
            project_paths = self._context.project_paths
            
            # Find the most specific matching path
            best_match = None
            best_match_length = 0
            
            for project_name, mapped_paths in project_paths.items():
                if not isinstance(mapped_paths, list):
                    continue
                    
                for mapped_path in mapped_paths:
                    try:
                        normalized_mapped = os.path.abspath(mapped_path)
                        
                        # Check if file path is within mapped path
                        if normalized_path.startswith(normalized_mapped):
                            # Use longest matching path as best match
                            if len(normalized_mapped) > best_match_length:
                                best_match = project_name
                                best_match_length = len(normalized_mapped)
                                
                    except Exception as e:
                        logger.warning(f"Error processing mapped path {mapped_path}: {e}")
                        continue
            
            if best_match:
                logger.info(f"Detected project {best_match} for path {normalized_path}")
            else:
                logger.debug(f"No project detected for path {normalized_path}")
                
            return best_match
            
        except Exception as e:
            logger.error(f"Failed to detect project from path {file_path}: {e}")
            raise ProjectContextError(f"Path detection failed: {e}")
    
    def set_current_project(self, project_name: Optional[str]) -> bool:
        """
        Set the current active project context.
        
        Implements FR-KB-005.9: Active project context management.
        
        Args:
            project_name: Name of project to set as active, or None to clear
            
        Returns:
            True if context was updated successfully
            
        Raises:
            ProjectNotFoundError: If project doesn't exist (when setting non-None)
            ProjectContextError: If context update fails
            
        Example:
            >>> manager.set_current_project("ai-research")
            True
            >>> manager.set_current_project(None)  # Clear active project
            True
        """
        if project_name is not None:
            self._validate_project_name(project_name)
            
            # Verify project exists
            projects = self._load_json(self.projects_file)
            if project_name not in projects:
                logger.warning(f"Attempt to set non-existent project as active: {project_name}")
                raise ProjectNotFoundError(project_name)
        
        try:
            self._context.active_project = project_name
            self._save_context()
            
            if project_name:
                logger.info(f"Set active project: {project_name}")
            else:
                logger.info("Cleared active project")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set current project to {project_name}: {e}")
            raise ProjectContextError(f"Context update failed: {e}")
    
    def get_current_project(self) -> Optional[str]:
        """
        Get the current active project.
        
        Implements FR-KB-005.10: Active project context retrieval.
        
        Returns:
            Name of currently active project, or None if no project is active
            
        Example:
            >>> current = manager.get_current_project()
            >>> print(f"Current project: {current or 'None'}")
        """
        try:
            active_project = self._context.active_project
            logger.debug(f"Current active project: {active_project or 'None'}")
            return active_project
        except Exception as e:
            logger.error(f"Failed to get current project: {e}")
            return None


def create_project_manager(storage_path: str = "projects") -> ProjectManager:
    """
    Create a ProjectManager instance with proper error handling.
    
    Factory function for creating ProjectManager instances with
    consistent error handling and logging setup.
    
    Args:
        storage_path: Directory path for storing project metadata.
                     Defaults to "projects" in current directory.
        
    Returns:
        Initialized ProjectManager instance
        
    Raises:
        ProjectMetadataError: If ProjectManager initialization fails
        
    Example:
        >>> manager = create_project_manager()
        >>> manager = create_project_manager("custom/project/path")
    """
    try:
        return ProjectManager(storage_path)
    except Exception as e:
        logger.error(f"Failed to create ProjectManager: {e}")
        raise ProjectMetadataError(f"ProjectManager creation failed: {e}") 