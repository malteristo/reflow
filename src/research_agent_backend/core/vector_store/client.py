"""
ChromaDB Client Connection and Health Management.

This module handles ChromaDB client initialization, connection management,
and health monitoring operations.

Implements FR-ST-002: Vector database connection management.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import chromadb
from chromadb.config import Settings

from ...utils.config import ConfigManager
from .types import (
    VectorStoreConfig,
    HealthStatus,
    VectorStoreError,
    DatabaseInitializationError,
    ConnectionError,
)


class ChromaDBClient:
    """
    ChromaDB Client Manager.
    
    Handles connection initialization, health monitoring, and client lifecycle.
    """
    
    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        persist_directory: Optional[str] = None,
        in_memory: bool = False
    ) -> None:
        """
        Initialize ChromaDB Client Manager.
        
        Args:
            config_manager: Configuration manager instance
            persist_directory: Custom persist directory (overrides config)
            in_memory: Use in-memory database (for testing)
        """
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Get vector store configuration
        vector_store_config = self.config_manager.get('vector_store', {})
        
        # Determine persist directory
        if in_memory:
            self.persist_directory = None
            self.logger.info("Using in-memory ChromaDB instance")
        else:
            self.persist_directory = persist_directory or vector_store_config.get('persist_directory', './data/chroma_db')
            # Resolve relative paths
            if self.persist_directory and not os.path.isabs(self.persist_directory):
                self.persist_directory = os.path.join(self.config_manager.project_root, self.persist_directory)
            self.logger.info(f"Using persistent ChromaDB at: {self.persist_directory}")
        
        # Initialize client
        self._client: Optional[chromadb.ClientAPI] = None
        self._connected = False
    
    @property
    def client(self) -> chromadb.ClientAPI:
        """Get ChromaDB client, initializing if necessary."""
        if self._client is None:
            self.initialize_database()
        return self._client
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected and healthy."""
        return self._connected and self._client is not None
    
    def initialize_database(self, db_path: Optional[str] = None) -> None:
        """
        Initialize ChromaDB database connection.
        
        Args:
            db_path: Custom database path (overrides instance setting)
            
        Raises:
            DatabaseInitializationError: If initialization fails
        """
        try:
            # Use provided path or instance setting
            persist_dir = db_path or self.persist_directory
            
            if persist_dir is None:
                # In-memory database
                self.logger.info("Initializing in-memory ChromaDB database")
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self._client = chromadb.Client(settings)
            else:
                # Persistent database
                persist_path = Path(persist_dir)
                
                # Check if path is writable before attempting creation
                try:
                    persist_path.mkdir(parents=True, exist_ok=True)
                except (OSError, PermissionError) as path_error:
                    error_msg = f"Cannot create database directory {persist_path}: {path_error}"
                    self.logger.error(error_msg)
                    raise DatabaseInitializationError(error_msg) from path_error
                
                self.logger.info(f"Initializing persistent ChromaDB database at: {persist_path}")
                
                # Configure ChromaDB settings
                settings = Settings(
                    persist_directory=str(persist_path),
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
                
                self._client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=settings
                )
            
            # Test connection
            self._test_connection()
            self._connected = True
            
            self.logger.info("ChromaDB database initialized successfully")
            
        except DatabaseInitializationError:
            # Re-raise our specific exceptions
            raise
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB database: {e}"
            self.logger.error(error_msg)
            raise DatabaseInitializationError(error_msg) from e
    
    def _test_connection(self) -> None:
        """Test database connection and basic operations."""
        try:
            # Test basic operations
            self._client.heartbeat()
            collections = self._client.list_collections()
            self.logger.debug(f"Database connection test successful. Found {len(collections)} collections.")
        except Exception as e:
            raise VectorStoreError(f"Database connection test failed: {e}") from e
    
    def health_check(self) -> HealthStatus:
        """
        Perform comprehensive health check.
        
        Returns:
            Health status with detailed information
        """
        health_status = HealthStatus(
            status='unknown',
            connected=False,
            persist_directory=self.persist_directory,
            collections_count=0,
            collections=[],
            timestamp=datetime.utcnow().isoformat(),
            errors=[]
        )
        
        try:
            if not self.is_connected:
                self.initialize_database()
            
            # Test heartbeat
            self._client.heartbeat()
            health_status.connected = True
            
            # Get collections info
            collections = self._client.list_collections()
            health_status.collections_count = len(collections)
            health_status.collections = [col.name for col in collections]
            
            health_status.status = 'healthy'
            self.logger.info("Database health check passed")
            
        except Exception as e:
            error_msg = f"Health check failed: {e}"
            health_status.errors.append(error_msg)
            health_status.status = 'unhealthy'
            self.logger.error(error_msg)
        
        return health_status
    
    def reset_database(self) -> None:
        """
        Reset the entire database (delete all collections).
        
        WARNING: This will permanently delete all data!
        """
        try:
            self.logger.warning("Resetting database - all data will be lost!")
            self.client.reset()
            self.logger.info("Database reset completed")
            
        except Exception as e:
            error_msg = f"Failed to reset database: {e}"
            self.logger.error(error_msg)
            raise VectorStoreError(error_msg) from e
    
    def close(self) -> None:
        """Close database connection and cleanup resources."""
        try:
            self._connected = False
            self._client = None
            self.logger.info("ChromaDB connection closed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}") 