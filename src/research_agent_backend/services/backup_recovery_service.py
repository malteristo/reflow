"""
Backup and Recovery Service for Research Agent.

This module provides comprehensive backup and recovery capabilities for
Research Agent collections, including automated backups, point-in-time
recovery, and integration with the existing transaction system.

Implements FR-KB-005: Backup and rollback mechanisms for model changes.
"""

import logging
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import zipfile
import hashlib
from threading import Lock
import asyncio

from ..core.vector_store import ChromaDBManager
from ..core.model_change_detection import ModelChangeDetector, ModelFingerprint
from ..core.document_insertion.transactions import TransactionManager
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups that can be created."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Status of backup operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecoveryStatus(Enum):
    """Status of recovery operations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupMetadata:
    """Metadata for a backup operation."""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    collections: List[str] = field(default_factory=list)
    model_fingerprints: Dict[str, ModelFingerprint] = field(default_factory=dict)
    file_path: Optional[Path] = None
    file_size_bytes: int = 0
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate backup duration in seconds."""
        if self.completed_at and self.created_at:
            return (self.completed_at - self.created_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "collections": self.collections,
            "model_fingerprints": {
                name: fp.to_dict() for name, fp in self.model_fingerprints.items()
            },
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "error_message": self.error_message,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackupMetadata':
        """Create from dictionary."""
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType(data["backup_type"]),
            status=BackupStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            collections=data.get("collections", []),
            model_fingerprints={
                name: ModelFingerprint.from_dict(fp_data)
                for name, fp_data in data.get("model_fingerprints", {}).items()
            },
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum=data.get("checksum"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {})
        )


@dataclass
class RecoveryOperation:
    """Metadata for a recovery operation."""
    recovery_id: str
    backup_id: str
    status: RecoveryStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    target_collections: List[str] = field(default_factory=list)
    recovery_point: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate recovery duration in seconds."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


@dataclass
class BackupConfig:
    """Configuration for backup operations."""
    backup_directory: Path = Path("backups")
    max_backup_age_days: int = 30
    max_backup_count: int = 100
    auto_cleanup_enabled: bool = True
    compression_enabled: bool = True
    checksum_verification: bool = True
    backup_model_fingerprints: bool = True
    backup_collection_metadata: bool = True
    parallel_backup_enabled: bool = True
    max_parallel_collections: int = 4


class BackupRecoveryService:
    """
    Comprehensive backup and recovery service for Research Agent.
    
    Provides automated backup creation, point-in-time recovery, and
    integration with the existing transaction and model change detection systems.
    """
    
    def __init__(
        self,
        vector_store: ChromaDBManager,
        config: Optional[BackupConfig] = None,
        transaction_manager: Optional[TransactionManager] = None
    ):
        """
        Initialize the backup and recovery service.
        
        Args:
            vector_store: ChromaDB manager instance
            config: Backup configuration settings
            transaction_manager: Transaction manager for rollback support
        """
        self.vector_store = vector_store
        self.config = config or BackupConfig()
        self.transaction_manager = transaction_manager
        self.logger = logging.getLogger(__name__)
        
        # Backup tracking
        self.active_backups: Dict[str, BackupMetadata] = {}
        self.backup_history: Dict[str, BackupMetadata] = {}
        self.active_recoveries: Dict[str, RecoveryOperation] = {}
        
        # Thread safety
        self._lock = Lock()
        
        # Initialize backup directory
        self.config.backup_directory.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.config.backup_directory / "backup_metadata.json"
        
        # Load existing backup metadata
        self._load_backup_metadata()
        
        self.logger.info(f"Backup and recovery service initialized with directory: {self.config.backup_directory}")
    
    def create_backup(
        self,
        collections: Optional[List[str]] = None,
        backup_type: BackupType = BackupType.FULL,
        backup_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> str:
        """
        Create a backup of specified collections.
        
        Args:
            collections: List of collections to backup (all if None)
            backup_type: Type of backup to create
            backup_id: Custom backup ID (generated if None)
            progress_callback: Optional progress callback
            
        Returns:
            Backup ID for tracking
        """
        if backup_id is None:
            backup_id = f"backup_{int(time.time() * 1000)}"
        
        # Get collections to backup
        if collections is None:
            collections = [col.name for col in self.vector_store.list_collections()]
        
        # Create backup metadata
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            collections=collections
        )
        
        with self._lock:
            self.active_backups[backup_id] = backup_metadata
        
        try:
            # Update status to in progress
            self._update_backup_status(backup_id, BackupStatus.IN_PROGRESS)
            
            if progress_callback:
                progress_callback(0.0, "Initializing backup...")
            
            # Create backup directory for this backup
            backup_dir = self.config.backup_directory / backup_id
            backup_dir.mkdir(exist_ok=True)
            
            # Backup model fingerprints if enabled
            if self.config.backup_model_fingerprints:
                self._backup_model_fingerprints(backup_id, backup_dir)
                if progress_callback:
                    progress_callback(10.0, "Backed up model fingerprints")
            
            # Backup collections
            total_collections = len(collections)
            for i, collection_name in enumerate(collections):
                if progress_callback:
                    progress = 10.0 + (i / total_collections) * 80.0
                    progress_callback(progress, f"Backing up collection: {collection_name}")
                
                self._backup_collection(backup_id, collection_name, backup_dir)
            
            # Create backup archive if compression enabled
            if self.config.compression_enabled:
                if progress_callback:
                    progress_callback(90.0, "Creating compressed archive...")
                
                archive_path = self._create_backup_archive(backup_id, backup_dir)
                backup_metadata.file_path = archive_path
                backup_metadata.file_size_bytes = archive_path.stat().st_size
                
                # Calculate checksum if verification enabled
                if self.config.checksum_verification:
                    backup_metadata.checksum = self._calculate_file_checksum(archive_path)
                
                # Remove uncompressed directory
                shutil.rmtree(backup_dir)
            else:
                backup_metadata.file_path = backup_dir
                backup_metadata.file_size_bytes = self._calculate_directory_size(backup_dir)
            
            # Complete backup
            backup_metadata.completed_at = datetime.now()
            self._update_backup_status(backup_id, BackupStatus.COMPLETED)
            
            if progress_callback:
                progress_callback(100.0, "Backup completed successfully")
            
            self.logger.info(f"Backup {backup_id} completed successfully")
            
            # Auto-cleanup old backups if enabled
            if self.config.auto_cleanup_enabled:
                self._cleanup_old_backups()
            
            return backup_id
            
        except Exception as e:
            error_msg = f"Backup failed: {e}"
            backup_metadata.error_message = error_msg
            self._update_backup_status(backup_id, BackupStatus.FAILED)
            self.logger.error(error_msg)
            raise
        finally:
            # Move from active to history
            with self._lock:
                if backup_id in self.active_backups:
                    self.backup_history[backup_id] = self.active_backups.pop(backup_id)
            
            # Save metadata
            self._save_backup_metadata()
    
    def restore_backup(
        self,
        backup_id: str,
        target_collections: Optional[List[str]] = None,
        recovery_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        use_transaction: bool = True
    ) -> str:
        """
        Restore collections from a backup.
        
        Args:
            backup_id: ID of backup to restore
            target_collections: Collections to restore (all if None)
            recovery_id: Custom recovery ID (generated if None)
            progress_callback: Optional progress callback
            use_transaction: Whether to use transaction for rollback support
            
        Returns:
            Recovery operation ID
        """
        if recovery_id is None:
            recovery_id = f"recovery_{int(time.time() * 1000)}"
        
        # Get backup metadata
        backup_metadata = self.get_backup_metadata(backup_id)
        if not backup_metadata:
            raise ValueError(f"Backup not found: {backup_id}")
        
        if backup_metadata.status != BackupStatus.COMPLETED:
            raise ValueError(f"Backup {backup_id} is not in completed state")
        
        # Determine collections to restore
        if target_collections is None:
            target_collections = backup_metadata.collections
        else:
            # Validate requested collections exist in backup
            missing_collections = set(target_collections) - set(backup_metadata.collections)
            if missing_collections:
                raise ValueError(f"Collections not found in backup: {missing_collections}")
        
        # Create recovery operation
        recovery_op = RecoveryOperation(
            recovery_id=recovery_id,
            backup_id=backup_id,
            status=RecoveryStatus.PENDING,
            started_at=datetime.now(),
            target_collections=target_collections
        )
        
        with self._lock:
            self.active_recoveries[recovery_id] = recovery_op
        
        try:
            # Update status to in progress
            recovery_op.status = RecoveryStatus.IN_PROGRESS
            
            if progress_callback:
                progress_callback(0.0, "Initializing recovery...")
            
            # Begin transaction if enabled
            if use_transaction and self.transaction_manager:
                self.transaction_manager.begin_transaction()
            
            # Extract backup if compressed
            backup_dir = self._prepare_backup_for_recovery(backup_metadata)
            
            if progress_callback:
                progress_callback(10.0, "Backup prepared for recovery")
            
            # Restore model fingerprints if available
            if self.config.backup_model_fingerprints:
                self._restore_model_fingerprints(backup_dir)
                if progress_callback:
                    progress_callback(20.0, "Restored model fingerprints")
            
            # Restore collections
            total_collections = len(target_collections)
            for i, collection_name in enumerate(target_collections):
                if progress_callback:
                    progress = 20.0 + (i / total_collections) * 70.0
                    progress_callback(progress, f"Restoring collection: {collection_name}")
                
                self._restore_collection(collection_name, backup_dir)
            
            # Commit transaction if enabled
            if use_transaction and self.transaction_manager:
                self.transaction_manager.commit_transaction()
            
            # Complete recovery
            recovery_op.completed_at = datetime.now()
            recovery_op.status = RecoveryStatus.COMPLETED
            
            if progress_callback:
                progress_callback(100.0, "Recovery completed successfully")
            
            self.logger.info(f"Recovery {recovery_id} completed successfully")
            return recovery_id
            
        except Exception as e:
            error_msg = f"Recovery failed: {e}"
            recovery_op.error_message = error_msg
            recovery_op.status = RecoveryStatus.FAILED
            
            # Rollback transaction if enabled
            if use_transaction and self.transaction_manager:
                try:
                    self.transaction_manager.rollback_transaction()
                    self.logger.info("Transaction rolled back due to recovery failure")
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            self.logger.error(error_msg)
            raise
        finally:
            # Clean up temporary extraction directory if needed
            if backup_metadata.file_path and backup_metadata.file_path.suffix == '.zip':
                temp_dir = self.config.backup_directory / f"temp_{backup_id}"
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
    
    def list_backups(
        self,
        status_filter: Optional[BackupStatus] = None,
        limit: Optional[int] = None
    ) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            status_filter: Filter by backup status
            limit: Maximum number of backups to return
            
        Returns:
            List of backup metadata
        """
        backups = list(self.backup_history.values())
        
        # Add active backups
        backups.extend(self.active_backups.values())
        
        # Filter by status
        if status_filter:
            backups = [b for b in backups if b.status == status_filter]
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply limit
        if limit:
            backups = backups[:limit]
        
        return backups
    
    def get_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get metadata for a specific backup."""
        return (
            self.backup_history.get(backup_id) or 
            self.active_backups.get(backup_id)
        )
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup and its associated files.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if backup was deleted, False if not found
        """
        backup_metadata = self.get_backup_metadata(backup_id)
        if not backup_metadata:
            return False
        
        try:
            # Delete backup files
            if backup_metadata.file_path and backup_metadata.file_path.exists():
                if backup_metadata.file_path.is_file():
                    backup_metadata.file_path.unlink()
                else:
                    shutil.rmtree(backup_metadata.file_path)
            
            # Remove from tracking
            with self._lock:
                self.backup_history.pop(backup_id, None)
                self.active_backups.pop(backup_id, None)
            
            # Save updated metadata
            self._save_backup_metadata()
            
            self.logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    def create_collection_snapshot(
        self,
        collection_name: str,
        snapshot_id: Optional[str] = None
    ) -> str:
        """
        Create a quick snapshot of a single collection.
        
        Args:
            collection_name: Name of collection to snapshot
            snapshot_id: Custom snapshot ID (generated if None)
            
        Returns:
            Snapshot ID
        """
        return self.create_backup(
            collections=[collection_name],
            backup_type=BackupType.SNAPSHOT,
            backup_id=snapshot_id
        )
    
    def restore_collection_snapshot(
        self,
        snapshot_id: str,
        target_collection_name: Optional[str] = None
    ) -> str:
        """
        Restore a collection from a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to restore
            target_collection_name: Target collection name (original if None)
            
        Returns:
            Recovery operation ID
        """
        backup_metadata = self.get_backup_metadata(snapshot_id)
        if not backup_metadata or backup_metadata.backup_type != BackupType.SNAPSHOT:
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        collections_to_restore = backup_metadata.collections
        if target_collection_name:
            collections_to_restore = [target_collection_name]
        
        return self.restore_backup(
            backup_id=snapshot_id,
            target_collections=collections_to_restore
        )
    
    def _backup_collection(
        self,
        backup_id: str,
        collection_name: str,
        backup_dir: Path
    ) -> None:
        """Backup a single collection."""
        try:
            # Get collection data
            collection_data = self.vector_store.export_collection(collection_name)
            
            # Save collection data
            collection_file = backup_dir / f"{collection_name}.json"
            with collection_file.open('w', encoding='utf-8') as f:
                json.dump(collection_data, f, indent=2, default=str)
            
            self.logger.debug(f"Backed up collection {collection_name} to {collection_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to backup collection {collection_name}: {e}")
            raise
    
    def _restore_collection(
        self,
        collection_name: str,
        backup_dir: Path
    ) -> None:
        """Restore a single collection."""
        try:
            collection_file = backup_dir / f"{collection_name}.json"
            if not collection_file.exists():
                raise FileNotFoundError(f"Collection backup file not found: {collection_file}")
            
            # Load collection data
            with collection_file.open('r', encoding='utf-8') as f:
                collection_data = json.load(f)
            
            # Restore collection
            self.vector_store.import_collection(collection_name, collection_data)
            
            self.logger.debug(f"Restored collection {collection_name} from {collection_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to restore collection {collection_name}: {e}")
            raise
    
    def _backup_model_fingerprints(
        self,
        backup_id: str,
        backup_dir: Path
    ) -> None:
        """Backup model fingerprints."""
        try:
            detector = ModelChangeDetector()
            fingerprints = detector.get_registered_models()
            
            # Save fingerprints
            fingerprints_file = backup_dir / "model_fingerprints.json"
            fingerprints_data = {
                name: fp.to_dict() for name, fp in fingerprints.items()
            }
            
            with fingerprints_file.open('w', encoding='utf-8') as f:
                json.dump(fingerprints_data, f, indent=2)
            
            # Update backup metadata
            backup_metadata = self.active_backups[backup_id]
            backup_metadata.model_fingerprints = fingerprints
            
            self.logger.debug(f"Backed up {len(fingerprints)} model fingerprints")
            
        except Exception as e:
            self.logger.warning(f"Failed to backup model fingerprints: {e}")
    
    def _restore_model_fingerprints(self, backup_dir: Path) -> None:
        """Restore model fingerprints."""
        try:
            fingerprints_file = backup_dir / "model_fingerprints.json"
            if not fingerprints_file.exists():
                self.logger.debug("No model fingerprints found in backup")
                return
            
            with fingerprints_file.open('r', encoding='utf-8') as f:
                fingerprints_data = json.load(f)
            
            # Restore fingerprints
            detector = ModelChangeDetector()
            fingerprints = [
                ModelFingerprint.from_dict(fp_data)
                for fp_data in fingerprints_data.values()
            ]
            
            detector.register_models_bulk(fingerprints)
            
            self.logger.debug(f"Restored {len(fingerprints)} model fingerprints")
            
        except Exception as e:
            self.logger.warning(f"Failed to restore model fingerprints: {e}")
    
    def _create_backup_archive(self, backup_id: str, backup_dir: Path) -> Path:
        """Create compressed archive of backup."""
        archive_path = self.config.backup_directory / f"{backup_id}.zip"
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in backup_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(backup_dir)
                    zipf.write(file_path, arcname)
        
        return archive_path
    
    def _prepare_backup_for_recovery(self, backup_metadata: BackupMetadata) -> Path:
        """Prepare backup for recovery (extract if compressed)."""
        if not backup_metadata.file_path:
            raise ValueError("Backup file path not found")
        
        if backup_metadata.file_path.suffix == '.zip':
            # Extract to temporary directory
            temp_dir = self.config.backup_directory / f"temp_{backup_metadata.backup_id}"
            temp_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(backup_metadata.file_path, 'r') as zipf:
                zipf.extractall(temp_dir)
            
            return temp_dir
        else:
            # Directory backup
            return backup_metadata.file_path
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory in bytes."""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _update_backup_status(self, backup_id: str, status: BackupStatus) -> None:
        """Update backup status."""
        with self._lock:
            if backup_id in self.active_backups:
                self.active_backups[backup_id].status = status
    
    def _cleanup_old_backups(self) -> None:
        """Clean up old backups based on configuration."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.max_backup_age_days)
            
            # Get backups to delete
            backups_to_delete = []
            all_backups = list(self.backup_history.values())
            all_backups.sort(key=lambda x: x.created_at, reverse=True)
            
            # Delete by age
            for backup in all_backups:
                if backup.created_at < cutoff_date:
                    backups_to_delete.append(backup.backup_id)
            
            # Delete by count (keep only max_backup_count newest)
            if len(all_backups) > self.config.max_backup_count:
                excess_backups = all_backups[self.config.max_backup_count:]
                for backup in excess_backups:
                    if backup.backup_id not in backups_to_delete:
                        backups_to_delete.append(backup.backup_id)
            
            # Delete backups
            for backup_id in backups_to_delete:
                self.delete_backup(backup_id)
            
            if backups_to_delete:
                self.logger.info(f"Cleaned up {len(backups_to_delete)} old backups")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old backups: {e}")
    
    def _load_backup_metadata(self) -> None:
        """Load backup metadata from disk."""
        try:
            if not self.metadata_file.exists():
                return
            
            with self.metadata_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            for backup_id, backup_data in data.items():
                try:
                    backup_metadata = BackupMetadata.from_dict(backup_data)
                    self.backup_history[backup_id] = backup_metadata
                except Exception as e:
                    self.logger.warning(f"Failed to load backup metadata for {backup_id}: {e}")
            
            self.logger.info(f"Loaded metadata for {len(self.backup_history)} backups")
            
        except Exception as e:
            self.logger.error(f"Failed to load backup metadata: {e}")
    
    def _save_backup_metadata(self) -> None:
        """Save backup metadata to disk."""
        try:
            data = {
                backup_id: backup.to_dict()
                for backup_id, backup in self.backup_history.items()
            }
            
            with self.metadata_file.open('w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save backup metadata: {e}")


# Global service instance
_backup_service_instance: Optional[BackupRecoveryService] = None


def get_backup_service(
    vector_store: Optional[ChromaDBManager] = None,
    config: Optional[BackupConfig] = None
) -> BackupRecoveryService:
    """Get the global backup service instance."""
    global _backup_service_instance
    if _backup_service_instance is None and vector_store:
        _backup_service_instance = BackupRecoveryService(
            vector_store=vector_store,
            config=config
        )
    return _backup_service_instance


def create_backup_service(
    vector_store: ChromaDBManager,
    config: Optional[BackupConfig] = None
) -> BackupRecoveryService:
    """Create a new backup service instance."""
    return BackupRecoveryService(vector_store=vector_store, config=config) 