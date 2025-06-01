"""
Migration Validation Framework for Research Agent.

This module provides comprehensive validation tests to ensure successful
re-indexing and migration operations, including comparison tests between
old and new embeddings, performance benchmarks, and semantic equivalence testing.

Implements FR-KB-005: Migration validation and integrity verification.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from enum import Enum
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr

from ..core.vector_store import ChromaDBManager
from ..core.model_change_detection import ModelChangeDetector, ModelFingerprint
from ..core.local_embedding_service import LocalEmbeddingService
from ..core.api_embedding_service import APIEmbeddingService
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)


class ValidationTestType(Enum):
    """Types of validation tests."""
    EMBEDDING_COMPARISON = "embedding_comparison"
    SEMANTIC_EQUIVALENCE = "semantic_equivalence"
    QUERY_PERFORMANCE = "query_performance"
    COLLECTION_INTEGRITY = "collection_integrity"
    MODEL_COMPATIBILITY = "model_compatibility"
    DATA_CONSISTENCY = "data_consistency"


class ValidationStatus(Enum):
    """Status of validation tests."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


class SeverityLevel(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationMetric:
    """Individual validation metric result."""
    name: str
    value: Union[float, int, str, bool]
    expected_value: Optional[Union[float, int, str, bool]] = None
    threshold: Optional[float] = None
    status: ValidationStatus = ValidationStatus.PENDING
    severity: SeverityLevel = SeverityLevel.INFO
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationTestResult:
    """Result of a validation test."""
    test_type: ValidationTestType
    test_name: str
    status: ValidationStatus
    severity: SeverityLevel
    duration_seconds: float
    metrics: List[ValidationMetric] = field(default_factory=list)
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        """Check if test passed."""
        return self.status == ValidationStatus.PASSED
    
    @property
    def failed(self) -> bool:
        """Check if test failed."""
        return self.status == ValidationStatus.FAILED
    
    @property
    def has_warnings(self) -> bool:
        """Check if test has warnings."""
        return self.status == ValidationStatus.WARNING or len(self.warnings) > 0


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    validation_id: str
    migration_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    collections_validated: List[str] = field(default_factory=list)
    test_results: List[ValidationTestResult] = field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.PENDING
    overall_severity: SeverityLevel = SeverityLevel.INFO
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Calculate validation duration."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    @property
    def passed_tests(self) -> List[ValidationTestResult]:
        """Get passed tests."""
        return [t for t in self.test_results if t.passed]
    
    @property
    def failed_tests(self) -> List[ValidationTestResult]:
        """Get failed tests."""
        return [t for t in self.test_results if t.failed]
    
    @property
    def warning_tests(self) -> List[ValidationTestResult]:
        """Get tests with warnings."""
        return [t for t in self.test_results if t.has_warnings]
    
    def calculate_overall_status(self) -> None:
        """Calculate overall validation status."""
        if not self.test_results:
            self.overall_status = ValidationStatus.PENDING
            return
        
        failed_count = len(self.failed_tests)
        warning_count = len(self.warning_tests)
        
        if failed_count > 0:
            self.overall_status = ValidationStatus.FAILED
            self.overall_severity = SeverityLevel.CRITICAL
        elif warning_count > 0:
            self.overall_status = ValidationStatus.WARNING
            self.overall_severity = SeverityLevel.MEDIUM
        else:
            self.overall_status = ValidationStatus.PASSED
            self.overall_severity = SeverityLevel.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "validation_id": self.validation_id,
            "migration_id": self.migration_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "collections_validated": self.collections_validated,
            "test_results": [
                {
                    "test_type": t.test_type.value,
                    "test_name": t.test_name,
                    "status": t.status.value,
                    "severity": t.severity.value,
                    "duration_seconds": t.duration_seconds,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.value,
                            "expected_value": m.expected_value,
                            "threshold": m.threshold,
                            "status": m.status.value,
                            "severity": m.severity.value,
                            "message": m.message,
                            "details": m.details
                        }
                        for m in t.metrics
                    ],
                    "error_message": t.error_message,
                    "warnings": t.warnings,
                    "recommendations": t.recommendations,
                    "metadata": t.metadata
                }
                for t in self.test_results
            ],
            "overall_status": self.overall_status.value,
            "overall_severity": self.overall_severity.value,
            "summary": self.summary,
            "metadata": self.metadata
        }


@dataclass
class ValidationConfig:
    """Configuration for migration validation."""
    # Embedding comparison thresholds
    cosine_similarity_threshold: float = 0.8
    embedding_dimension_tolerance: float = 0.0
    embedding_norm_tolerance: float = 0.1
    
    # Semantic equivalence thresholds
    semantic_correlation_threshold: float = 0.7
    semantic_ranking_threshold: float = 0.8
    query_result_overlap_threshold: float = 0.6
    
    # Performance thresholds
    query_performance_degradation_threshold: float = 2.0  # 2x slower max
    collection_size_variance_threshold: float = 0.05  # 5% variance
    embedding_generation_timeout: float = 300.0  # 5 minutes
    
    # Test configuration
    sample_size_for_testing: int = 100
    semantic_test_queries: List[str] = field(default_factory=lambda: [
        "What is machine learning?",
        "How does artificial intelligence work?",
        "Explain deep learning concepts",
        "What are neural networks?",
        "Define natural language processing"
    ])
    enable_performance_tests: bool = True
    enable_semantic_tests: bool = True
    enable_integrity_tests: bool = True
    
    # Output configuration
    save_detailed_reports: bool = True
    report_output_directory: Path = Path("validation_reports")
    include_embedding_samples: bool = False


class MigrationValidationService:
    """
    Comprehensive migration validation service for Research Agent.
    
    Provides validation tests to ensure successful re-indexing and migration
    operations, including embedding comparison, semantic equivalence testing,
    and performance benchmarks.
    """
    
    def __init__(
        self,
        vector_store: ChromaDBManager,
        config: Optional[ValidationConfig] = None,
        config_manager: Optional[ConfigManager] = None
    ):
        """
        Initialize the migration validation service.
        
        Args:
            vector_store: ChromaDB manager instance
            config: Validation configuration settings
            config_manager: Configuration manager for embedding services
        """
        self.vector_store = vector_store
        self.config = config or ValidationConfig()
        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize embedding services
        self.embedding_service = self._create_embedding_service()
        
        # Create output directory
        self.config.report_output_directory.mkdir(parents=True, exist_ok=True)
        
        # Active validations
        self.active_validations: Dict[str, ValidationReport] = {}
        self.validation_history: Dict[str, ValidationReport] = {}
        
        self.logger.info("Migration validation service initialized")
    
    def _create_embedding_service(self):
        """Create embedding service based on configuration."""
        try:
            embedding_config = self.config_manager.config.get('embedding_model', {})
            
            if embedding_config.get('provider'):
                # API service
                from ..core.api_embedding_service import APIConfiguration
                api_config = APIConfiguration(
                    provider=embedding_config['provider'],
                    api_key=embedding_config.get('api_key', ''),
                    model_name=embedding_config.get('model_name_or_path', '')
                )
                return APIEmbeddingService(api_config)
            else:
                # Local service
                return LocalEmbeddingService(
                    model_name=embedding_config.get('model_name_or_path', 'all-MiniLM-L6-v2')
                )
        except Exception as e:
            self.logger.warning(f"Failed to create embedding service: {e}")
            return None
    
    def validate_migration(
        self,
        migration_id: str,
        collections: List[str],
        pre_migration_data: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ValidationReport:
        """
        Perform comprehensive migration validation.
        
        Args:
            migration_id: Unique identifier for the migration
            collections: List of collections to validate
            pre_migration_data: Optional pre-migration data for comparison
            progress_callback: Optional progress callback
            
        Returns:
            ValidationReport with comprehensive test results
        """
        validation_id = f"validation_{migration_id}_{int(time.time() * 1000)}"
        
        # Create validation report
        report = ValidationReport(
            validation_id=validation_id,
            migration_id=migration_id,
            started_at=datetime.now(),
            collections_validated=collections
        )
        
        self.active_validations[validation_id] = report
        
        try:
            total_tests = self._count_enabled_tests()
            current_test = 0
            
            # Update progress
            if progress_callback:
                progress_callback("Starting validation tests...", 0.0)
            
            # Run collection integrity tests
            if self.config.enable_integrity_tests:
                current_test += 1
                if progress_callback:
                    progress = (current_test / total_tests) * 100
                    progress_callback("Running collection integrity tests...", progress)
                
                integrity_results = self._run_collection_integrity_tests(collections)
                report.test_results.extend(integrity_results)
            
            # Run embedding comparison tests
            if pre_migration_data:
                current_test += 1
                if progress_callback:
                    progress = (current_test / total_tests) * 100
                    progress_callback("Running embedding comparison tests...", progress)
                
                comparison_results = self._run_embedding_comparison_tests(
                    collections, pre_migration_data
                )
                report.test_results.extend(comparison_results)
            
            # Run semantic equivalence tests
            if self.config.enable_semantic_tests:
                current_test += 1
                if progress_callback:
                    progress = (current_test / total_tests) * 100
                    progress_callback("Running semantic equivalence tests...", progress)
                
                semantic_results = self._run_semantic_equivalence_tests(collections)
                report.test_results.extend(semantic_results)
            
            # Run performance tests
            if self.config.enable_performance_tests:
                current_test += 1
                if progress_callback:
                    progress = (current_test / total_tests) * 100
                    progress_callback("Running performance benchmark tests...", progress)
                
                performance_results = self._run_performance_tests(collections)
                report.test_results.extend(performance_results)
            
            # Run model compatibility tests
            current_test += 1
            if progress_callback:
                progress = (current_test / total_tests) * 100
                progress_callback("Running model compatibility tests...", progress)
            
            compatibility_results = self._run_model_compatibility_tests(collections)
            report.test_results.extend(compatibility_results)
            
            # Calculate overall status
            report.completed_at = datetime.now()
            report.calculate_overall_status()
            
            # Generate summary
            self._generate_validation_summary(report)
            
            # Save report if configured
            if self.config.save_detailed_reports:
                self._save_validation_report(report)
            
            if progress_callback:
                progress_callback("Validation completed", 100.0)
            
            self.logger.info(f"Migration validation completed: {validation_id}")
            return report
            
        except Exception as e:
            error_msg = f"Migration validation failed: {e}"
            self.logger.error(error_msg)
            
            # Add error to report
            error_result = ValidationTestResult(
                test_type=ValidationTestType.COLLECTION_INTEGRITY,
                test_name="validation_framework",
                status=ValidationStatus.FAILED,
                severity=SeverityLevel.CRITICAL,
                duration_seconds=0.0,
                error_message=error_msg
            )
            report.test_results.append(error_result)
            report.overall_status = ValidationStatus.FAILED
            report.overall_severity = SeverityLevel.CRITICAL
            report.completed_at = datetime.now()
            
            raise
        finally:
            # Move from active to history
            if validation_id in self.active_validations:
                self.validation_history[validation_id] = self.active_validations.pop(validation_id)
    
    def _count_enabled_tests(self) -> int:
        """Count the number of enabled test types."""
        count = 1  # Model compatibility always runs
        if self.config.enable_integrity_tests:
            count += 1
        if self.config.enable_semantic_tests:
            count += 1
        if self.config.enable_performance_tests:
            count += 1
        return count
    
    def _run_collection_integrity_tests(self, collections: List[str]) -> List[ValidationTestResult]:
        """Run collection integrity validation tests."""
        results = []
        
        for collection_name in collections:
            start_time = time.time()
            test_result = ValidationTestResult(
                test_type=ValidationTestType.COLLECTION_INTEGRITY,
                test_name=f"integrity_{collection_name}",
                status=ValidationStatus.RUNNING,
                severity=SeverityLevel.HIGH,
                duration_seconds=0.0
            )
            
            try:
                # Test collection existence
                collection = self.vector_store.get_collection(collection_name)
                if not collection:
                    test_result.status = ValidationStatus.FAILED
                    test_result.error_message = f"Collection {collection_name} not found"
                    test_result.duration_seconds = time.time() - start_time
                    results.append(test_result)
                    continue
                
                # Get collection statistics
                stats = self.vector_store.get_collection_stats(collection_name)
                
                # Test document count
                doc_count_metric = ValidationMetric(
                    name="document_count",
                    value=stats.document_count,
                    expected_value=None,
                    status=ValidationStatus.PASSED if stats.document_count > 0 else ValidationStatus.WARNING,
                    severity=SeverityLevel.MEDIUM if stats.document_count == 0 else SeverityLevel.INFO,
                    message=f"Collection contains {stats.document_count} documents"
                )
                test_result.metrics.append(doc_count_metric)
                
                # Test embedding dimensions consistency
                if hasattr(stats, 'embedding_dimension') and stats.embedding_dimension:
                    dim_metric = ValidationMetric(
                        name="embedding_dimension",
                        value=stats.embedding_dimension,
                        expected_value=None,
                        status=ValidationStatus.PASSED,
                        severity=SeverityLevel.INFO,
                        message=f"Embeddings have dimension {stats.embedding_dimension}"
                    )
                    test_result.metrics.append(dim_metric)
                
                # Test for null/empty embeddings
                sample_docs = self.vector_store.get_documents(
                    collection_name=collection_name,
                    limit=min(self.config.sample_size_for_testing, stats.document_count),
                    include=['embeddings']
                )
                
                if sample_docs.get('embeddings'):
                    embeddings = sample_docs['embeddings']
                    null_count = sum(1 for emb in embeddings if emb is None or len(emb) == 0)
                    
                    null_metric = ValidationMetric(
                        name="null_embeddings",
                        value=null_count,
                        expected_value=0,
                        threshold=0,
                        status=ValidationStatus.PASSED if null_count == 0 else ValidationStatus.FAILED,
                        severity=SeverityLevel.CRITICAL if null_count > 0 else SeverityLevel.INFO,
                        message=f"Found {null_count} null/empty embeddings"
                    )
                    test_result.metrics.append(null_metric)
                
                # Overall test status
                failed_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.FAILED]
                warning_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.WARNING]
                
                if failed_metrics:
                    test_result.status = ValidationStatus.FAILED
                    test_result.severity = SeverityLevel.CRITICAL
                elif warning_metrics:
                    test_result.status = ValidationStatus.WARNING
                    test_result.severity = SeverityLevel.MEDIUM
                else:
                    test_result.status = ValidationStatus.PASSED
                    test_result.severity = SeverityLevel.INFO
                
            except Exception as e:
                test_result.status = ValidationStatus.FAILED
                test_result.error_message = f"Integrity test failed: {e}"
                test_result.severity = SeverityLevel.CRITICAL
                self.logger.error(f"Collection integrity test failed for {collection_name}: {e}")
            
            test_result.duration_seconds = time.time() - start_time
            results.append(test_result)
        
        return results
    
    def _run_embedding_comparison_tests(
        self, 
        collections: List[str], 
        pre_migration_data: Dict[str, Any]
    ) -> List[ValidationTestResult]:
        """Run embedding comparison tests between old and new embeddings."""
        results = []
        
        for collection_name in collections:
            start_time = time.time()
            test_result = ValidationTestResult(
                test_type=ValidationTestType.EMBEDDING_COMPARISON,
                test_name=f"embedding_comparison_{collection_name}",
                status=ValidationStatus.RUNNING,
                severity=SeverityLevel.HIGH,
                duration_seconds=0.0
            )
            
            try:
                # Get pre-migration embeddings
                old_embeddings = pre_migration_data.get(collection_name, {}).get('embeddings', [])
                if not old_embeddings:
                    test_result.status = ValidationStatus.SKIPPED
                    test_result.warnings.append("No pre-migration data available for comparison")
                    test_result.duration_seconds = time.time() - start_time
                    results.append(test_result)
                    continue
                
                # Get new embeddings
                new_data = self.vector_store.get_documents(
                    collection_name=collection_name,
                    limit=len(old_embeddings),
                    include=['embeddings', 'ids']
                )
                new_embeddings = new_data.get('embeddings', [])
                
                if len(new_embeddings) != len(old_embeddings):
                    count_metric = ValidationMetric(
                        name="embedding_count_match",
                        value=len(new_embeddings),
                        expected_value=len(old_embeddings),
                        status=ValidationStatus.WARNING,
                        severity=SeverityLevel.MEDIUM,
                        message=f"Embedding count changed: {len(old_embeddings)} -> {len(new_embeddings)}"
                    )
                    test_result.metrics.append(count_metric)
                
                # Compare embedding dimensions
                if old_embeddings and new_embeddings:
                    old_dim = len(old_embeddings[0])
                    new_dim = len(new_embeddings[0])
                    
                    dim_metric = ValidationMetric(
                        name="embedding_dimension_match",
                        value=new_dim,
                        expected_value=old_dim,
                        status=ValidationStatus.PASSED if old_dim == new_dim else ValidationStatus.FAILED,
                        severity=SeverityLevel.CRITICAL if old_dim != new_dim else SeverityLevel.INFO,
                        message=f"Dimension: {old_dim} -> {new_dim}"
                    )
                    test_result.metrics.append(dim_metric)
                
                # Calculate cosine similarity for sample
                sample_size = min(self.config.sample_size_for_testing, len(old_embeddings), len(new_embeddings))
                if sample_size > 0:
                    old_sample = np.array(old_embeddings[:sample_size])
                    new_sample = np.array(new_embeddings[:sample_size])
                    
                    # Calculate pairwise cosine similarity
                    similarities = []
                    for i in range(sample_size):
                        sim = cosine_similarity([old_sample[i]], [new_sample[i]])[0][0]
                        similarities.append(sim)
                    
                    avg_similarity = np.mean(similarities)
                    min_similarity = np.min(similarities)
                    
                    similarity_metric = ValidationMetric(
                        name="average_cosine_similarity",
                        value=float(avg_similarity),
                        threshold=self.config.cosine_similarity_threshold,
                        status=ValidationStatus.PASSED if avg_similarity >= self.config.cosine_similarity_threshold else ValidationStatus.WARNING,
                        severity=SeverityLevel.MEDIUM if avg_similarity < self.config.cosine_similarity_threshold else SeverityLevel.INFO,
                        message=f"Average similarity: {avg_similarity:.3f} (min: {min_similarity:.3f})"
                    )
                    test_result.metrics.append(similarity_metric)
                
                # Overall test status
                failed_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.FAILED]
                warning_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.WARNING]
                
                if failed_metrics:
                    test_result.status = ValidationStatus.FAILED
                    test_result.severity = SeverityLevel.CRITICAL
                elif warning_metrics:
                    test_result.status = ValidationStatus.WARNING
                    test_result.severity = SeverityLevel.MEDIUM
                else:
                    test_result.status = ValidationStatus.PASSED
                    test_result.severity = SeverityLevel.INFO
                
            except Exception as e:
                test_result.status = ValidationStatus.FAILED
                test_result.error_message = f"Embedding comparison failed: {e}"
                test_result.severity = SeverityLevel.CRITICAL
                self.logger.error(f"Embedding comparison test failed for {collection_name}: {e}")
            
            test_result.duration_seconds = time.time() - start_time
            results.append(test_result)
        
        return results
    
    def _run_semantic_equivalence_tests(self, collections: List[str]) -> List[ValidationTestResult]:
        """Run semantic equivalence tests using query comparisons."""
        results = []
        
        start_time = time.time()
        test_result = ValidationTestResult(
            test_type=ValidationTestType.SEMANTIC_EQUIVALENCE,
            test_name="semantic_equivalence",
            status=ValidationStatus.RUNNING,
            severity=SeverityLevel.MEDIUM,
            duration_seconds=0.0
        )
        
        try:
            # Run test queries against each collection
            for collection_name in collections:
                for i, query in enumerate(self.config.semantic_test_queries):
                    try:
                        # Perform query
                        results_data = self.vector_store.query_collection(
                            collection_name=collection_name,
                            query_text=query,
                            n_results=10
                        )
                        
                        # Analyze results
                        if results_data.get('documents'):
                            result_count = len(results_data['documents'])
                            
                            query_metric = ValidationMetric(
                                name=f"query_{i+1}_results",
                                value=result_count,
                                expected_value=None,
                                status=ValidationStatus.PASSED if result_count > 0 else ValidationStatus.WARNING,
                                severity=SeverityLevel.INFO,
                                message=f"Query '{query[:30]}...' returned {result_count} results"
                            )
                            test_result.metrics.append(query_metric)
                        
                    except Exception as e:
                        query_metric = ValidationMetric(
                            name=f"query_{i+1}_error",
                            value=str(e),
                            status=ValidationStatus.FAILED,
                            severity=SeverityLevel.HIGH,
                            message=f"Query failed: {e}"
                        )
                        test_result.metrics.append(query_metric)
            
            # Overall semantic test status
            failed_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.FAILED]
            warning_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.WARNING]
            
            if failed_metrics:
                test_result.status = ValidationStatus.FAILED
                test_result.severity = SeverityLevel.HIGH
            elif warning_metrics:
                test_result.status = ValidationStatus.WARNING
                test_result.severity = SeverityLevel.MEDIUM
            else:
                test_result.status = ValidationStatus.PASSED
                test_result.severity = SeverityLevel.INFO
                
        except Exception as e:
            test_result.status = ValidationStatus.FAILED
            test_result.error_message = f"Semantic equivalence test failed: {e}"
            test_result.severity = SeverityLevel.HIGH
            self.logger.error(f"Semantic equivalence test failed: {e}")
        
        test_result.duration_seconds = time.time() - start_time
        results.append(test_result)
        
        return results
    
    def _run_performance_tests(self, collections: List[str]) -> List[ValidationTestResult]:
        """Run performance benchmark tests."""
        results = []
        
        for collection_name in collections:
            start_time = time.time()
            test_result = ValidationTestResult(
                test_type=ValidationTestType.QUERY_PERFORMANCE,
                test_name=f"performance_{collection_name}",
                status=ValidationStatus.RUNNING,
                severity=SeverityLevel.MEDIUM,
                duration_seconds=0.0
            )
            
            try:
                # Test query performance
                query_times = []
                test_query = "test performance query"
                
                for _ in range(5):  # Run 5 test queries
                    query_start = time.time()
                    try:
                        self.vector_store.query_collection(
                            collection_name=collection_name,
                            query_text=test_query,
                            n_results=10
                        )
                        query_time = time.time() - query_start
                        query_times.append(query_time)
                    except Exception as e:
                        self.logger.warning(f"Performance test query failed: {e}")
                
                if query_times:
                    avg_query_time = np.mean(query_times)
                    max_query_time = np.max(query_times)
                    
                    avg_time_metric = ValidationMetric(
                        name="average_query_time",
                        value=float(avg_query_time),
                        threshold=1.0,  # 1 second threshold
                        status=ValidationStatus.PASSED if avg_query_time < 1.0 else ValidationStatus.WARNING,
                        severity=SeverityLevel.MEDIUM if avg_query_time >= 1.0 else SeverityLevel.INFO,
                        message=f"Average query time: {avg_query_time:.3f}s (max: {max_query_time:.3f}s)"
                    )
                    test_result.metrics.append(avg_time_metric)
                
                # Test collection size
                stats = self.vector_store.get_collection_stats(collection_name)
                size_metric = ValidationMetric(
                    name="collection_size",
                    value=stats.document_count,
                    status=ValidationStatus.PASSED,
                    severity=SeverityLevel.INFO,
                    message=f"Collection contains {stats.document_count} documents"
                )
                test_result.metrics.append(size_metric)
                
                # Overall performance test status
                warning_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.WARNING]
                
                if warning_metrics:
                    test_result.status = ValidationStatus.WARNING
                    test_result.severity = SeverityLevel.MEDIUM
                else:
                    test_result.status = ValidationStatus.PASSED
                    test_result.severity = SeverityLevel.INFO
                
            except Exception as e:
                test_result.status = ValidationStatus.FAILED
                test_result.error_message = f"Performance test failed: {e}"
                test_result.severity = SeverityLevel.MEDIUM
                self.logger.error(f"Performance test failed for {collection_name}: {e}")
            
            test_result.duration_seconds = time.time() - start_time
            results.append(test_result)
        
        return results
    
    def _run_model_compatibility_tests(self, collections: List[str]) -> List[ValidationTestResult]:
        """Run model compatibility tests."""
        start_time = time.time()
        test_result = ValidationTestResult(
            test_type=ValidationTestType.MODEL_COMPATIBILITY,
            test_name="model_compatibility",
            status=ValidationStatus.RUNNING,
            severity=SeverityLevel.HIGH,
            duration_seconds=0.0
        )
        
        try:
            # Get current model fingerprint
            if self.embedding_service and hasattr(self.embedding_service, 'generate_model_fingerprint'):
                fingerprint = self.embedding_service.generate_model_fingerprint()
                
                model_metric = ValidationMetric(
                    name="current_model",
                    value=fingerprint.model_name,
                    status=ValidationStatus.PASSED,
                    severity=SeverityLevel.INFO,
                    message=f"Current model: {fingerprint.model_name} v{fingerprint.version}"
                )
                test_result.metrics.append(model_metric)
                
                # Check model registration
                detector = ModelChangeDetector()
                registered_fingerprint = detector.get_model_fingerprint(fingerprint.model_name)
                
                registration_metric = ValidationMetric(
                    name="model_registered",
                    value=registered_fingerprint is not None,
                    status=ValidationStatus.PASSED if registered_fingerprint else ValidationStatus.WARNING,
                    severity=SeverityLevel.MEDIUM if not registered_fingerprint else SeverityLevel.INFO,
                    message="Model registered" if registered_fingerprint else "Model not registered"
                )
                test_result.metrics.append(registration_metric)
            
            # Test embedding generation capability
            if self.embedding_service:
                try:
                    test_text = "This is a test for embedding generation"
                    if hasattr(self.embedding_service, 'embed_text'):
                        embedding = self.embedding_service.embed_text(test_text)
                    elif hasattr(self.embedding_service, 'generate_embedding'):
                        embedding = self.embedding_service.generate_embedding(test_text)
                    else:
                        embedding = None
                    
                    generation_metric = ValidationMetric(
                        name="embedding_generation",
                        value=embedding is not None and len(embedding) > 0,
                        status=ValidationStatus.PASSED if embedding and len(embedding) > 0 else ValidationStatus.FAILED,
                        severity=SeverityLevel.CRITICAL if not embedding or len(embedding) == 0 else SeverityLevel.INFO,
                        message=f"Embedding generation: {'OK' if embedding and len(embedding) > 0 else 'FAILED'}"
                    )
                    test_result.metrics.append(generation_metric)
                    
                except Exception as e:
                    generation_metric = ValidationMetric(
                        name="embedding_generation_error",
                        value=str(e),
                        status=ValidationStatus.FAILED,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Embedding generation failed: {e}"
                    )
                    test_result.metrics.append(generation_metric)
            
            # Overall compatibility test status
            failed_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.FAILED]
            warning_metrics = [m for m in test_result.metrics if m.status == ValidationStatus.WARNING]
            
            if failed_metrics:
                test_result.status = ValidationStatus.FAILED
                test_result.severity = SeverityLevel.CRITICAL
            elif warning_metrics:
                test_result.status = ValidationStatus.WARNING
                test_result.severity = SeverityLevel.MEDIUM
            else:
                test_result.status = ValidationStatus.PASSED
                test_result.severity = SeverityLevel.INFO
            
        except Exception as e:
            test_result.status = ValidationStatus.FAILED
            test_result.error_message = f"Model compatibility test failed: {e}"
            test_result.severity = SeverityLevel.CRITICAL
            self.logger.error(f"Model compatibility test failed: {e}")
        
        test_result.duration_seconds = time.time() - start_time
        return [test_result]
    
    def _generate_validation_summary(self, report: ValidationReport) -> None:
        """Generate summary statistics for validation report."""
        total_tests = len(report.test_results)
        passed_tests = len(report.passed_tests)
        failed_tests = len(report.failed_tests)
        warning_tests = len(report.warning_tests)
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Collect all metrics
        all_metrics = []
        for test in report.test_results:
            all_metrics.extend(test.metrics)
        
        # Summary statistics
        report.summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "warning_tests": warning_tests,
            "success_rate": round(success_rate, 2),
            "total_metrics": len(all_metrics),
            "duration_seconds": report.duration_seconds,
            "collections_count": len(report.collections_validated),
            "overall_status": report.overall_status.value,
            "overall_severity": report.overall_severity.value
        }
    
    def _save_validation_report(self, report: ValidationReport) -> None:
        """Save validation report to disk."""
        try:
            report_file = self.config.report_output_directory / f"{report.validation_id}.json"
            
            with report_file.open('w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
    
    def get_validation_report(self, validation_id: str) -> Optional[ValidationReport]:
        """Get validation report by ID."""
        return (
            self.active_validations.get(validation_id) or 
            self.validation_history.get(validation_id)
        )
    
    def list_validation_reports(
        self, 
        limit: Optional[int] = None,
        status_filter: Optional[ValidationStatus] = None
    ) -> List[ValidationReport]:
        """List validation reports with optional filtering."""
        reports = list(self.validation_history.values())
        reports.extend(self.active_validations.values())
        
        # Filter by status
        if status_filter:
            reports = [r for r in reports if r.overall_status == status_filter]
        
        # Sort by start time (newest first)
        reports.sort(key=lambda x: x.started_at, reverse=True)
        
        # Apply limit
        if limit:
            reports = reports[:limit]
        
        return reports


# Global service instance
_validation_service_instance: Optional[MigrationValidationService] = None


def get_validation_service(
    vector_store: Optional[ChromaDBManager] = None,
    config: Optional[ValidationConfig] = None
) -> MigrationValidationService:
    """Get the global validation service instance."""
    global _validation_service_instance
    if _validation_service_instance is None and vector_store:
        _validation_service_instance = MigrationValidationService(
            vector_store=vector_store,
            config=config
        )
    return _validation_service_instance


def create_validation_service(
    vector_store: ChromaDBManager,
    config: Optional[ValidationConfig] = None
) -> MigrationValidationService:
    """Create a new validation service instance."""
    return MigrationValidationService(vector_store=vector_store, config=config) 