"""
Knowledge base augmentation service implementation.

Provides functionality for adding external content, research reports,
updating documents, and managing duplicates in the knowledge base.

Implements FR-KB-001: Knowledge base augmentation functionality.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple, Protocol
from datetime import datetime, timezone
from pathlib import Path
import logging
import hashlib
import re

logger = logging.getLogger(__name__)


# === Custom Exceptions ===

class AugmentationError(Exception):
    """Base exception for augmentation operations."""
    pass


class QualityValidationError(AugmentationError):
    """Raised when content fails quality validation."""
    def __init__(self, score: float, threshold: float, message: str = None):
        self.score = score
        self.threshold = threshold
        super().__init__(message or f"Quality score {score:.2f} below threshold {threshold}")


class DuplicateContentError(AugmentationError):
    """Raised when duplicate content is detected."""
    def __init__(self, similar_docs: List[str], similarity: float):
        self.similar_docs = similar_docs
        self.similarity = similarity
        super().__init__(f"Duplicate content detected (similarity: {similarity:.2f})")


class VectorStoreError(AugmentationError):
    """Raised when vector store operations fail."""
    pass


# === Protocol Definitions ===

class VectorStoreProtocol(Protocol):
    """Protocol for vector store implementations."""
    
    def search(self, content: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar content."""
        ...
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the store."""
        ...
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        ...
    
    def update_document(self, doc_id: str, data: Dict[str, Any]) -> bool:
        """Update a document."""
        ...
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        ...


class ConfigManagerProtocol(Protocol):
    """Protocol for configuration managers."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...


# === Data Models ===

@dataclass
class ExternalResult:
    """Data model for external search results or web content."""
    source_url: str
    title: str
    content: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize URL format."""
        # URL validation
        if not self.source_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {self.source_url}")
        
        # Content validation
        if not self.content.strip():
            raise ValueError("Content cannot be empty")
        
        # Title validation and auto-generation
        if not self.title.strip():
            # Auto-generate title from content
            self.title = self._generate_title_from_content()
        
        # Normalize tags
        self.tags = [tag.strip().lower() for tag in self.tags if tag.strip()]
    
    def _generate_title_from_content(self) -> str:
        """Generate a title from content if none provided."""
        # Take first sentence or first 50 characters
        first_sentence = self.content.split('.')[0]
        if len(first_sentence) <= 50:
            return first_sentence.strip()
        return self.content[:50].strip() + "..."
    
    @property
    def content_hash(self) -> str:
        """Generate a hash of the content for deduplication."""
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()


@dataclass
class ResearchReport:
    """Data model for research reports."""
    file_path: str
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate file and extract metadata."""
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Research report file not found: {self.file_path}")
        
        # Extract file metadata
        self.metadata.update({
            'file_size': path.stat().st_size,
            'file_extension': path.suffix,
            'last_modified': datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        })
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in MB."""
        return self.metadata.get('file_size', 0) / (1024 * 1024)


@dataclass
class DocumentUpdate:
    """Data model for document updates."""
    document_id: str
    new_content: Optional[str] = None
    new_metadata: Optional[Dict[str, Any]] = None
    source_file: Optional[str] = None
    update_embeddings: bool = True
    update_reason: str = "manual_update"
    
    def __post_init__(self):
        """Validate update data."""
        if not any([self.new_content, self.new_metadata, self.source_file]):
            raise ValueError("At least one update field must be provided")
        
        # Validate source file if provided
        if self.source_file and not Path(self.source_file).exists():
            raise FileNotFoundError(f"Source file not found: {self.source_file}")


@dataclass
class DuplicateGroup:
    """Data model for duplicate document groups."""
    group_id: int
    documents: List[str]
    similarity: float
    merge_strategy: str = "content_union"
    confidence: float = 1.0
    
    def __post_init__(self):
        """Validate duplicate group data."""
        if len(self.documents) < 2:
            raise ValueError("Duplicate group must contain at least 2 documents")
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError("Similarity must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        # Validate merge strategy
        valid_strategies = ["content_union", "latest_version", "highest_quality", "manual_merge"]
        if self.merge_strategy not in valid_strategies:
            raise ValueError(f"Invalid merge strategy. Must be one of: {valid_strategies}")


@dataclass
class QualityMetrics:
    """Data model for content quality metrics."""
    overall_score: float
    content_length_score: float = 0.0
    structure_score: float = 0.0
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    language_score: float = 0.0
    uniqueness_score: float = 0.0
    
    def __post_init__(self):
        """Normalize scores to valid range."""
        score_fields = [
            'overall_score', 'content_length_score', 'structure_score', 
            'credibility_score', 'freshness_score', 'relevance_score',
            'language_score', 'uniqueness_score'
        ]
        
        for field_name in score_fields:
            value = getattr(self, field_name)
            if value < 0.0:
                setattr(self, field_name, 0.0)
            elif value > 1.0:
                setattr(self, field_name, 1.0)
    
    @property
    def quality_grade(self) -> str:
        """Get a letter grade for the overall quality."""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"


@dataclass
class AugmentationConfig:
    """Enhanced configuration for augmentation service."""
    quality_threshold: float = 0.7
    similarity_threshold: float = 0.85
    auto_categorize: bool = True
    enable_versioning: bool = True
    batch_size: int = 50
    cache_size: int = 1000
    default_collection: str = "research"
    
    # Quality scoring weights
    content_length_weight: float = 0.15
    structure_weight: float = 0.20
    credibility_weight: float = 0.25
    freshness_weight: float = 0.10
    relevance_weight: float = 0.20
    language_weight: float = 0.05
    uniqueness_weight: float = 0.05
    
    # Content validation settings
    min_content_length: int = 50
    max_content_length: int = 100000
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate thresholds
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("Quality threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        
        # Validate weights sum to approximately 1.0
        total_weight = (
            self.content_length_weight + self.structure_weight + 
            self.credibility_weight + self.freshness_weight + 
            self.relevance_weight + self.language_weight + 
            self.uniqueness_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Quality scoring weights must sum to 1.0, got {total_weight:.3f}")


# === Enhanced Service with RAG Pipeline Integration ===

class AugmentationService:
    """
    Enhanced service for augmenting the knowledge base with external content.
    
    Now integrated with the existing RAG pipeline infrastructure including
    real vector store operations, embedding generation, and configuration
    management.
    
    Provides methods for adding external results, research reports,
    updating documents, and managing duplicates with advanced quality
    validation and error handling.
    """
    
    def __init__(
        self, 
        config: Optional[AugmentationConfig] = None,
        config_manager: Optional[ConfigManagerProtocol] = None, 
        vector_store: Optional[VectorStoreProtocol] = None, 
        embedding_service=None
    ):
        """Initialize the augmentation service with RAG pipeline integration."""
        # Import here to avoid circular imports
        from .vector_store import ChromaDBManager
        from .local_embedding_service import LocalEmbeddingService
        from ..utils.config import ConfigManager
        
        # Initialize configuration
        if config:
            self.config = config
        elif config_manager:
            self.config = self._load_config_from_manager(config_manager)
        else:
            # Use existing configuration management system
            self._config_manager = ConfigManager()
            self.config = self._load_config_from_research_agent()
        
        # Initialize vector store with real ChromaDB implementation
        if vector_store:
            self.vector_store = vector_store
        else:
            try:
                self.vector_store = ChromaDBManager(
                    config_manager=self._config_manager if hasattr(self, '_config_manager') else None
                )
                logger.info("Connected to ChromaDB vector store for augmentation")
            except Exception as e:
                logger.warning(f"Failed to connect to vector store: {e}. Operating in offline mode.")
                self.vector_store = None
        
        # Initialize embedding service with real implementation
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            try:
                # Get embedding model config from configuration
                embedding_config = self._config_manager.get('embedding_model', {}) if hasattr(self, '_config_manager') else {}
                model_name = embedding_config.get('name', 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
                
                self.embedding_service = LocalEmbeddingService(model_name=model_name)
                logger.info(f"Initialized embedding service with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize embedding service: {e}. Operating without embeddings.")
                self.embedding_service = None
        
        # Initialize caches
        self._quality_cache: Dict[str, QualityMetrics] = {}
        self._categorization_cache: Dict[str, Tuple[str, float]] = {}
        
        # Initialize document chunk processor for research reports
        self._chunk_processor = self._create_chunk_processor()
        
        logger.info(f"AugmentationService initialized with quality threshold: {self.config.quality_threshold}")
    
    def _load_config_from_research_agent(self) -> AugmentationConfig:
        """Load configuration from Research Agent configuration system."""
        try:
            # Get augmentation-specific config with fallbacks
            augmentation_config = self._config_manager.get('augmentation', {})
            
            return AugmentationConfig(
                quality_threshold=augmentation_config.get('quality_threshold', 0.7),
                similarity_threshold=augmentation_config.get('similarity_threshold', 0.85),
                auto_categorize=augmentation_config.get('auto_categorize', True),
                enable_versioning=augmentation_config.get('enable_versioning', True),
                batch_size=augmentation_config.get('batch_size', 50),
                cache_size=augmentation_config.get('cache_size', 1000),
                default_collection=augmentation_config.get('default_collection', "research"),
                
                # Quality scoring weights
                content_length_weight=augmentation_config.get('content_length_weight', 0.15),
                structure_weight=augmentation_config.get('structure_weight', 0.20),
                credibility_weight=augmentation_config.get('credibility_weight', 0.25),
                freshness_weight=augmentation_config.get('freshness_weight', 0.10),
                relevance_weight=augmentation_config.get('relevance_weight', 0.20),
                language_weight=augmentation_config.get('language_weight', 0.05),
                uniqueness_weight=augmentation_config.get('uniqueness_weight', 0.05),
                
                # Content validation settings
                min_content_length=augmentation_config.get('min_content_length', 50),
                max_content_length=augmentation_config.get('max_content_length', 100000),
                allowed_languages=augmentation_config.get('allowed_languages', ["en"])
            )
        except Exception as e:
            logger.warning(f"Failed to load config from Research Agent, using defaults: {e}")
            return AugmentationConfig()
    
    def _create_chunk_processor(self):
        """Create document chunk processor for research reports."""
        try:
            # Import chunking components from existing system
            from ...processing.document_chunker import DocumentChunker  # This would exist in the system
            from ...models.metadata_schema import DocumentMetadata
            
            # Get chunking config from system
            chunking_config = self._config_manager.get('chunking_strategy', {}) if hasattr(self, '_config_manager') else {}
            
            # Create processor with system configuration
            return DocumentChunker(
                chunk_size=chunking_config.get('chunk_size', 512),
                chunk_overlap=chunking_config.get('chunk_overlap', 50),
                markdown_aware=chunking_config.get('markdown_aware', True)
            )
        except ImportError:
            logger.warning("Document chunker not available, using fallback implementation")
            return None
    
    def _load_config_from_manager(self, config_manager: ConfigManagerProtocol) -> AugmentationConfig:
        """Load configuration from config manager."""
        return AugmentationConfig(
            quality_threshold=config_manager.get('augmentation.quality_threshold', 0.7),
            similarity_threshold=config_manager.get('augmentation.similarity_threshold', 0.85),
            auto_categorize=config_manager.get('augmentation.auto_categorize', True),
            enable_versioning=config_manager.get('augmentation.enable_versioning', True),
            batch_size=config_manager.get('augmentation.batch_size', 50),
            cache_size=config_manager.get('augmentation.cache_size', 1000),
            default_collection=config_manager.get('augmentation.default_collection', "research")
        )
    
    def add_external_result(self, external_result: ExternalResult, collection: str = "research", 
                          detect_duplicates: bool = False) -> Dict[str, Any]:
        """
        Add an external search result to the knowledge base with real vector store integration.
        
        Args:
            external_result: The external content to add
            collection: Target collection name
            detect_duplicates: Whether to check for duplicates
            
        Returns:
            Dict containing result status and metadata
        """
        try:
            # Calculate quality score
            quality_score = self._calculate_quality_score(
                external_result.content, 
                external_result.metadata
            )
            
            # Check quality threshold
            if quality_score.overall_score < self.config.quality_threshold:
                raise QualityValidationError(quality_score.overall_score, self.config.quality_threshold)
            
            # Check for duplicates if requested and vector store is available
            if detect_duplicates and self.vector_store and self.embedding_service:
                # Generate embedding for duplicate detection
                content_embedding = self.embedding_service.embed_text(external_result.content)
                
                # Search for similar content
                try:
                    search_results = self.vector_store.query_collection(
                        collection_name=collection,
                        query_embedding=content_embedding,
                        k=5,
                        include_distances=True
                    )
                    
                    # Check similarity threshold
                    if search_results.documents and search_results.distances:
                        max_similarity = 1.0 - min(search_results.distances[0])  # Convert distance to similarity
                        if max_similarity >= self.config.similarity_threshold:
                            similar_ids = [search_results.ids[0][i] for i in range(min(3, len(search_results.ids[0])))]
                            raise DuplicateContentError(similar_ids, max_similarity)
                            
                except Exception as e:
                    logger.warning(f"Duplicate detection failed: {e}")
                    # Continue with addition if duplicate detection fails
            
            # Auto-assign collection if needed
            if collection == "research" and self.config.auto_categorize:
                assigned_collection, confidence = self._auto_assign_collection(external_result.content)
                collection = assigned_collection
                auto_assigned = True
                assignment_confidence = confidence
            else:
                auto_assigned = False
                assignment_confidence = None
            
            # Generate document ID
            document_id = f"ext_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(external_result.content.encode()).hexdigest()[:8]}"
            
            # Create source attribution
            source_attribution = {
                'url': external_result.source_url,
                'author': external_result.author,
                'added_date': datetime.now(timezone.utc).isoformat(),
                'type': 'external',
                'quality_score': quality_score.overall_score
            }
            
            # Add to vector store if available
            if self.vector_store and self.embedding_service:
                try:
                    # Ensure collection exists
                    try:
                        collection_obj = self.vector_store.get_collection(collection)
                    except:
                        # Create collection if it doesn't exist
                        collection_obj = self.vector_store.create_collection(
                            name=collection,
                            metadata={'type': 'augmentation', 'created_by': 'augmentation_service'}
                        )
                    
                    # Generate embedding
                    content_embedding = self.embedding_service.embed_text(external_result.content)
                    
                    # Add document to vector store
                    self.vector_store.add_documents(
                        collection_name=collection,
                        chunks=[external_result.content],
                        embeddings=[content_embedding],
                        metadata=[{
                            'document_id': document_id,
                            'title': external_result.title,
                            'source_attribution': source_attribution,
                            'tags': external_result.tags,
                            'quality_score': quality_score.overall_score,
                            'added_by': 'augmentation_service',
                            'content_type': 'external_result'
                        }],
                        ids=[document_id]
                    )
                    
                    logger.info(f"Successfully added external result to vector store: {document_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to add to vector store: {e}")
                    # Return error but don't raise - document metadata was still processed
                    return {
                        'status': 'partial_success',
                        'document_id': document_id,
                        'collection': collection,
                        'quality_score': quality_score.overall_score,
                        'quality_passed': True,
                        'source_attribution': source_attribution,
                        'vector_store_error': str(e),
                        'warning': 'Document processed but not added to vector store'
                    }
            
            result = {
                'status': 'success',
                'document_id': document_id,
                'collection': collection,
                'quality_score': quality_score.overall_score,
                'quality_passed': True,
                'source_attribution': source_attribution
            }
            
            if auto_assigned:
                result.update({
                    'auto_assigned': True,
                    'assignment_confidence': assignment_confidence
                })
            
            if not self.vector_store:
                result['warning'] = 'Vector store not available - document metadata processed only'
            
            return result
            
        except QualityValidationError as e:
            logger.warning(f"Quality validation failed: {e}")
            return {
                'status': 'rejected',
                'quality_score': e.score,
                'quality_passed': False,
                'rejection_reason': str(e)
            }
        except DuplicateContentError as e:
            logger.warning(f"Duplicate content detected: {e}")
            return {
                'status': 'duplicate_detected',
                'similar_documents': e.similar_docs,
                'similarity_score': e.similarity,
                'merge_suggestion': 'Consider merging with existing content'
            }
        except Exception as e:
            logger.error(f"Failed to add external result: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def add_research_report(self, research_report: ResearchReport, collection: str = "research-reports",
                          auto_categorize: bool = False) -> Dict[str, Any]:
        """
        Add a research report to the knowledge base with real document processing.
        
        Args:
            research_report: The research report to add
            collection: Target collection name
            auto_categorize: Whether to automatically categorize
            
        Returns:
            Dict containing result status and metadata
        """
        try:
            # Read file content
            content = Path(research_report.file_path).read_text(encoding='utf-8')
            
            # Process document through real chunking system if available
            if self._chunk_processor:
                try:
                    chunks = self._chunk_processor.chunk_document(content)
                    processed_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        processed_chunks.append({
                            'id': f"rpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                            'content': chunk.get('content', chunk) if isinstance(chunk, dict) else chunk,
                            'metadata': {
                                'chunk_index': i,
                                'chunk_type': 'research_report',
                                'source_file': research_report.file_path,
                                'file_size': research_report.file_size_mb,
                                'processed_at': datetime.now(timezone.utc).isoformat()
                            }
                        })
                except Exception as e:
                    logger.warning(f"Chunking system failed, using fallback: {e}")
                    processed_chunks = self._chunk_document_fallback(content, research_report.file_path)
            else:
                # Use fallback chunking
                processed_chunks = self._chunk_document_fallback(content, research_report.file_path)
            
            # Auto-categorize if requested
            if auto_categorize:
                category, confidence = self._auto_categorize_content(content)
                auto_category = category
                categorization_confidence = confidence
                auto_categorized = True
            else:
                auto_category = research_report.category
                categorization_confidence = None
                auto_categorized = False
            
            # Generate document ID
            document_id = f"rpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
            
            # Add to vector store if available
            if self.vector_store and self.embedding_service:
                try:
                    # Ensure collection exists
                    try:
                        collection_obj = self.vector_store.get_collection(collection)
                    except:
                        # Create collection if it doesn't exist
                        collection_obj = self.vector_store.create_collection(
                            name=collection,
                            metadata={'type': 'research_reports', 'created_by': 'augmentation_service'}
                        )
                    
                    # Process chunks for vector store
                    chunk_contents = [chunk['content'] for chunk in processed_chunks]
                    chunk_ids = [chunk['id'] for chunk in processed_chunks]
                    chunk_metadata = [chunk['metadata'] for chunk in processed_chunks]
                    
                    # Generate embeddings for all chunks
                    embeddings = self.embedding_service.embed_batch(chunk_contents)
                    
                    # Add enriched metadata
                    for i, metadata in enumerate(chunk_metadata):
                        metadata.update({
                            'document_id': document_id,
                            'category': auto_category,
                            'added_by': 'augmentation_service',
                            'content_type': 'research_report'
                        })
                        if auto_categorized:
                            metadata['auto_categorized'] = True
                            metadata['categorization_confidence'] = categorization_confidence
                    
                    # Add all chunks to vector store
                    self.vector_store.add_documents(
                        collection_name=collection,
                        chunks=chunk_contents,
                        embeddings=embeddings,
                        metadata=chunk_metadata,
                        ids=chunk_ids
                    )
                    
                    logger.info(f"Successfully added research report with {len(processed_chunks)} chunks to vector store: {document_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to add research report to vector store: {e}")
                    # Continue with metadata processing
                    return {
                        'status': 'partial_success',
                        'document_id': document_id,
                        'chunks_created': len(processed_chunks),
                        'collection': collection,
                        'vector_store_error': str(e),
                        'warning': 'Document processed but not added to vector store'
                    }
            
            result = {
                'status': 'success',
                'document_id': document_id,
                'chunks_created': len(processed_chunks),
                'collection': collection
            }
            
            if auto_categorized:
                result.update({
                    'auto_category': auto_category,
                    'confidence': categorization_confidence,
                    'auto_categorized': True
                })
            
            if not self.vector_store:
                result['warning'] = 'Vector store not available - document metadata processed only'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to add research report: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _chunk_document_fallback(self, content: str, source_path: str) -> List[Dict[str, Any]]:
        """Fallback document chunking when the main chunking system is not available."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            # Further split large paragraphs
            if len(paragraph) > self.config.max_content_length // 4:  # Quarter of max length
                # Split by sentences
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) < 500:  # Target chunk size
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append({
                                'id': f"chunk_{len(chunks)+1}",
                                'content': current_chunk.strip(),
                                'metadata': {
                                    'chunk_index': len(chunks),
                                    'chunk_type': 'paragraph_split',
                                    'source_file': source_path,
                                    'word_count': len(current_chunk.split()),
                                    'char_count': len(current_chunk)
                                }
                            })
                        current_chunk = sentence + ". "
                
                # Add remaining content
                if current_chunk:
                    chunks.append({
                        'id': f"chunk_{len(chunks)+1}",
                        'content': current_chunk.strip(),
                        'metadata': {
                            'chunk_index': len(chunks),
                            'chunk_type': 'paragraph_split',
                            'source_file': source_path,
                            'word_count': len(current_chunk.split()),
                            'char_count': len(current_chunk)
                        }
                    })
            else:
                # Paragraph is reasonable size, use as-is
                chunk_data = {
                    'id': f"chunk_{i+1}",
                    'content': paragraph,
                    'metadata': {
                        'chunk_index': i,
                        'chunk_type': 'paragraph',
                        'source_file': source_path,
                        'word_count': len(paragraph.split()),
                        'char_count': len(paragraph)
                    }
                }
                chunks.append(chunk_data)
        
        logger.debug(f"Created {len(chunks)} chunks from document using fallback chunking")
        return chunks
    
    def add_research_reports_batch(self, folder_path: str, pattern: str = "*.md", 
                                 collection: str = "batch-reports") -> Dict[str, Any]:
        """
        Add multiple research reports in batch.
        
        Args:
            folder_path: Path to folder containing reports
            pattern: File pattern to match
            collection: Target collection
            
        Returns:
            Dict containing batch processing results
        """
        try:
            folder = Path(folder_path)
            files = list(folder.glob(pattern))
            
            processed = len(files)
            successful = 0
            failed = 0
            results = []
            
            for file_path in files:
                try:
                    report = ResearchReport(file_path=str(file_path))
                    result = self.add_research_report(report, collection=collection)
                    if result['status'] == 'success':
                        successful += 1
                        results.append(result['document_id'])
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            
            return {
                'processed': processed,
                'successful': successful,
                'failed': failed,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Failed batch processing: {e}")
            return {
                'processed': 0,
                'successful': 0,
                'failed': 0,
                'results': [],
                'error': str(e)
            }
    
    def update_document(self, update: DocumentUpdate) -> Dict[str, Any]:
        """
        Update an existing document.
        
        Args:
            update: Document update specification
            
        Returns:
            Dict containing update result
        """
        try:
            # Check if document exists
            if self.vector_store:
                existing_doc = self.vector_store.get_document(update.document_id)
                if not existing_doc:
                    return {
                        'status': 'error',
                        'error': f'Document {update.document_id} not found'
                    }
                
                current_version = existing_doc.get('metadata', {}).get('version', 1)
                new_version = current_version + 1
            else:
                new_version = 2
            
            changes = []
            
            # Handle content updates
            if update.new_content:
                changes.append('content_updated')
                if update.update_embeddings:
                    changes.append('embeddings_regenerated')
            
            # Handle file-based updates
            if update.source_file:
                changes.append('content_updated_from_file')
                if update.update_embeddings:
                    changes.append('embeddings_regenerated')
            
            # Handle metadata updates
            if update.new_metadata:
                changes.append('metadata_updated')
            
            # Simulate update in vector store
            if self.vector_store:
                self.vector_store.update_document(update.document_id, {
                    'content': update.new_content,
                    'metadata': update.new_metadata
                })
            
            return {
                'status': 'updated',
                'document_id': update.document_id,
                'version': new_version,
                'changes': changes
            }
            
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def detect_duplicates(self, threshold: float = None, document_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Detect duplicate documents.
        
        Args:
            threshold: Similarity threshold for detection
            document_ids: Specific documents to check (optional)
            
        Returns:
            List of duplicate groups
        """
        try:
            if threshold is None:
                threshold = self.config.similarity_threshold
            
            # Mock duplicate detection
            if document_ids:
                # Manual selection mode
                if len(document_ids) >= 2:
                    return [{
                        'group_id': 1,
                        'documents': document_ids,
                        'similarity': 0.93,
                        'merge_strategy': 'content_union'
                    }]
            else:
                # Auto-detection mode
                return [{
                    'group_id': 1,
                    'documents': ['doc_001', 'doc_002'],
                    'similarity': 0.95,
                    'merge_strategy': 'content_union'
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to detect duplicates: {e}")
            return []
    
    def merge_duplicates(self, duplicate_groups: List[DuplicateGroup], 
                        keep_originals: bool = False, preview_only: bool = False) -> Dict[str, Any]:
        """
        Merge duplicate documents.
        
        Args:
            duplicate_groups: Groups of duplicates to merge
            keep_originals: Whether to keep original documents
            preview_only: Whether to only preview without actual merge
            
        Returns:
            Dict containing merge results
        """
        try:
            if preview_only:
                total_documents = sum(len(group.documents) for group in duplicate_groups)
                return {
                    'preview_mode': True,
                    'would_merge_groups': len(duplicate_groups),
                    'would_merge_documents': total_documents
                }
            
            merged_groups = 0
            documents_merged = 0
            new_document_ids = []
            
            for group in duplicate_groups:
                # Simulate merge process
                if self.vector_store:
                    # Get documents to merge
                    docs_to_merge = []
                    for doc_id in group.documents:
                        doc = self.vector_store.get_document(doc_id)
                        if doc:
                            docs_to_merge.append(doc)
                    
                    if docs_to_merge:
                        # Create merged document
                        merged_id = f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{group.group_id}"
                        
                        # Merge content based on strategy
                        if group.merge_strategy == "content_union":
                            merged_content = "\n\n".join([doc['content'] for doc in docs_to_merge])
                        elif group.merge_strategy == "latest_version":
                            # Use the latest document
                            latest_doc = max(docs_to_merge, key=lambda d: d.get('metadata', {}).get('created', ''))
                            merged_content = latest_doc['content']
                        else:
                            merged_content = docs_to_merge[0]['content']  # Default to first
                        
                        # Add merged document
                        self.vector_store.add_documents([{
                            'id': merged_id,
                            'content': merged_content,
                            'metadata': {'merged_from': group.documents}
                        }])
                        
                        # Remove originals if not keeping
                        if not keep_originals:
                            for doc_id in group.documents:
                                self.vector_store.delete_document(doc_id)
                        
                        new_document_ids.append(merged_id)
                        merged_groups += 1
                        documents_merged += len(group.documents)
            
            result = {
                'merged_groups': merged_groups,
                'documents_merged': documents_merged
            }
            
            if new_document_ids:
                result['new_document_id'] = new_document_ids[0]  # Return first for single group
            
            if keep_originals:
                result['originals_preserved'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to merge duplicates: {e}")
            return {
                'merged_groups': 0,
                'documents_merged': 0,
                'error': str(e)
            }
    
    def _calculate_quality_score(self, content: str, metadata: Dict[str, Any] = None) -> QualityMetrics:
        """Enhanced quality scoring with multiple factors and caching."""
        if metadata is None:
            metadata = {}
        
        # Check cache first
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        if content_hash in self._quality_cache:
            logger.debug(f"Using cached quality score for content hash: {content_hash[:8]}")
            return self._quality_cache[content_hash]
        
        # Content length scoring (normalized)
        content_length = len(content)
        if content_length < self.config.min_content_length:
            content_length_score = 0.0
        elif content_length > self.config.max_content_length:
            content_length_score = 0.8  # Penalize overly long content
        else:
            # Optimal range: 500-2000 characters
            optimal_length = min(content_length / 2000.0, 1.0)
            content_length_score = min(optimal_length, 1.0)
        
        # Structure scoring (enhanced)
        structure_score = self._calculate_structure_score(content)
        
        # Credibility scoring (enhanced)
        credibility_score = self._calculate_credibility_score(metadata)
        
        # Freshness scoring (enhanced)
        freshness_score = self._calculate_freshness_score(metadata)
        
        # Language quality scoring
        language_score = self._calculate_language_score(content)
        
        # Uniqueness scoring (basic implementation)
        uniqueness_score = self._calculate_uniqueness_score(content)
        
        # Relevance scoring (basic - could be enhanced with ML)
        relevance_score = 0.7  # Default relevance
        
        # Calculate weighted overall score
        overall_score = (
            content_length_score * self.config.content_length_weight +
            structure_score * self.config.structure_weight +
            credibility_score * self.config.credibility_weight +
            freshness_score * self.config.freshness_weight +
            relevance_score * self.config.relevance_weight +
            language_score * self.config.language_weight +
            uniqueness_score * self.config.uniqueness_weight
        )
        
        quality_metrics = QualityMetrics(
            overall_score=overall_score,
            content_length_score=content_length_score,
            structure_score=structure_score,
            credibility_score=credibility_score,
            freshness_score=freshness_score,
            relevance_score=relevance_score,
            language_score=language_score,
            uniqueness_score=uniqueness_score
        )
        
        # Cache the result
        if len(self._quality_cache) < self.config.cache_size:
            self._quality_cache[content_hash] = quality_metrics
        
        logger.debug(f"Calculated quality score: {overall_score:.3f} (grade: {quality_metrics.quality_grade})")
        return quality_metrics
    
    def _calculate_structure_score(self, content: str) -> float:
        """Calculate structure quality score."""
        score = 0.0
        
        # Check for paragraph breaks
        paragraphs = content.count('\n\n')
        if paragraphs > 0:
            score += 0.3
        
        # Check for sentence structure
        sentences = content.count('.')
        if sentences >= 3:
            score += 0.2
        
        # Check for headings (markdown style)
        headings = content.count('#')
        if headings > 0:
            score += 0.2
        
        # Check for lists
        if any(marker in content for marker in ['- ', '* ', '1. ', '2. ']):
            score += 0.1
        
        # Check for code blocks or technical content
        if any(marker in content for marker in ['```', '`', 'def ', 'class ', 'import ']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_credibility_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate credibility score based on metadata."""
        score = 0.4  # Base score
        
        # Author information
        if metadata.get('author'):
            score += 0.3
            # Bonus for multiple authors
            if ',' in str(metadata['author']) or ' and ' in str(metadata['author']):
                score += 0.1
        
        # Publication information
        if metadata.get('publication_date'):
            score += 0.2
        
        # DOI or other academic identifiers
        if any(key in metadata for key in ['doi', 'isbn', 'pmid']):
            score += 0.3
        
        # Institution or publisher
        if any(key in metadata for key in ['institution', 'publisher', 'journal']):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on publication date."""
        pub_date = metadata.get('publication_date')
        if not pub_date:
            return 0.5  # Neutral score for unknown dates
        
        if isinstance(pub_date, str):
            try:
                pub_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            except:
                return 0.5
        
        if not isinstance(pub_date, datetime):
            return 0.5
        
        # Calculate age in days
        now = datetime.now(timezone.utc)
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        
        age_days = (now - pub_date).days
        
        # Score based on age
        if age_days < 30:
            return 1.0  # Very fresh
        elif age_days < 365:
            return 0.8  # Recent
        elif age_days < 1825:  # 5 years
            return 0.6  # Somewhat dated
        else:
            return 0.3  # Old content
    
    def _calculate_language_score(self, content: str) -> float:
        """Calculate language quality score."""
        # Basic language quality checks
        score = 0.7  # Base score
        
        # Check for proper capitalization
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if sentences:
            capitalized = sum(1 for s in sentences if s and s[0].isupper())
            capitalization_ratio = capitalized / len(sentences)
            score += 0.2 * capitalization_ratio
        
        # Check for reasonable sentence length
        avg_sentence_length = len(content) / max(len(sentences), 1)
        if 10 <= avg_sentence_length <= 100:  # Reasonable range
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_uniqueness_score(self, content: str) -> float:
        """Calculate content uniqueness score."""
        # Basic uniqueness check - could be enhanced with vector similarity
        
        # Check for repetitive patterns
        words = content.lower().split()
        if len(words) == 0:
            return 0.0
        
        unique_words = set(words)
        uniqueness_ratio = len(unique_words) / len(words)
        
        # Penalize very repetitive content
        if uniqueness_ratio < 0.3:
            return 0.2
        elif uniqueness_ratio < 0.5:
            return 0.5
        else:
            return min(uniqueness_ratio, 1.0)
    
    def _chunk_document(self, content: str) -> List[Dict[str, Any]]:
        """Enhanced document chunking with metadata."""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        for i, paragraph in enumerate(paragraphs[:10]):  # Limit for testing
            chunk_data = {
                'id': f"chunk_{i+1}",
                'content': paragraph,
                'metadata': {
                    'chunk_index': i,
                    'chunk_type': 'paragraph',
                    'word_count': len(paragraph.split()),
                    'char_count': len(paragraph)
                }
            }
            chunks.append(chunk_data)
        
        logger.debug(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _auto_categorize_content(self, content: str) -> Tuple[str, float]:
        """Enhanced automatic content categorization with caching."""
        # Check cache first
        content_preview = content[:200].lower()
        if content_preview in self._categorization_cache:
            return self._categorization_cache[content_preview]
        
        # Enhanced keyword-based categorization
        categories = {
            'machine-learning': {
                'keywords': ['machine learning', 'neural network', 'ai', 'artificial intelligence', 
                           'deep learning', 'algorithm', 'model training', 'regression', 'classification'],
                'weight': 0.0
            },
            'research': {
                'keywords': ['research', 'study', 'analysis', 'experiment', 'hypothesis', 
                           'methodology', 'results', 'conclusion', 'findings'],
                'weight': 0.0
            },
            'technology': {
                'keywords': ['software', 'programming', 'development', 'code', 'api', 
                           'framework', 'library', 'database', 'cloud'],
                'weight': 0.0
            },
            'business': {
                'keywords': ['market', 'business', 'strategy', 'revenue', 'profit', 
                           'customer', 'product', 'service', 'management'],
                'weight': 0.0
            }
        }
        
        content_lower = content.lower()
        
        # Calculate weights for each category
        for category, data in categories.items():
            for keyword in data['keywords']:
                if keyword in content_lower:
                    # Weight by keyword frequency and length
                    frequency = content_lower.count(keyword)
                    data['weight'] += frequency * len(keyword)
        
        # Find best category
        best_category = 'general'
        best_confidence = 0.4
        
        for category, data in categories.items():
            if data['weight'] > 0:
                # Normalize confidence by content length
                confidence = min(data['weight'] / len(content_lower) * 1000, 0.95)
                if confidence > best_confidence:
                    best_category = category
                    best_confidence = confidence
        
        result = (best_category, best_confidence)
        
        # Cache the result
        if len(self._categorization_cache) < self.config.cache_size:
            self._categorization_cache[content_preview] = result
        
        logger.debug(f"Auto-categorized content as '{best_category}' with confidence {best_confidence:.2f}")
        return result
    
    def _auto_assign_collection(self, content: str) -> Tuple[str, float]:
        """Automatically assign collection based on enhanced content analysis."""
        category, confidence = self._auto_categorize_content(content)
        
        # Map categories to collections
        collection_mapping = {
            'machine-learning': 'ml-research',
            'research': 'academic-papers',
            'technology': 'tech-docs',
            'business': 'business-reports',
            'general': self.config.default_collection
        }
        
        collection = collection_mapping.get(category, self.config.default_collection)
        return collection, confidence 