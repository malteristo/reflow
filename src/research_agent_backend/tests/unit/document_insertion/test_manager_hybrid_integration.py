"""Tests for DocumentInsertionManager with HybridChunker integration - FR-KB-002.1."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from research_agent_backend.core.document_insertion.manager import DocumentInsertionManager
from research_agent_backend.core.document_insertion.exceptions import InsertionResult
from research_agent_backend.models.metadata_schema import DocumentMetadata, ContentType
from research_agent_backend.utils.config import ConfigManager


class TestDocumentInsertionManagerHybridIntegration:
    """Tests for DocumentInsertionManager with HybridChunker integration."""
    
    def test_manager_initializes_with_hybrid_chunker(self):
        """Test that DocumentInsertionManager initializes with HybridChunker."""
        # Create mock dependencies
        mock_vector_store = Mock()
        mock_config = Mock()
        
        # Ensure chunk_overlap < chunk_size for validation
        config_values = {
            "chunking_strategy.chunk_size": 512,
            "chunking_strategy.chunk_overlap": 50,  # Must be less than chunk_size
            "chunking_strategy.preserve_code_blocks": True,
            "chunking_strategy.preserve_tables": True
        }
        mock_config.get.side_effect = lambda key, default: config_values.get(key, default)
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Verify HybridChunker is initialized
        assert hasattr(manager, 'hybrid_chunker')
        assert manager.hybrid_chunker is not None
        
        # Verify configuration was passed to HybridChunker
        assert manager.hybrid_chunker.config.chunk_size == 512
        assert manager.hybrid_chunker.config.chunk_overlap == 50
        assert manager.hybrid_chunker.config.preserve_code_blocks == True
        assert manager.hybrid_chunker.config.preserve_tables == True
    
    @patch('research_agent_backend.core.document_insertion.validation.DocumentValidator')
    @patch('research_agent_backend.core.document_insertion.validation.DocumentPreparationService')
    @patch('research_agent_backend.core.document_insertion.embeddings.EmbeddingService')
    def test_hybrid_chunking_in_document_insertion(self, mock_embedding_svc, mock_prep_svc, mock_validator):
        """Test that document insertion uses HybridChunker for chunking."""
        # Setup mocks
        mock_vector_store = Mock()
        mock_config = Mock()
        mock_config.get.return_value = 300  # chunk_size
        
        # Mock preparation service
        mock_prep_svc_instance = Mock()
        mock_prep_svc.return_value = mock_prep_svc_instance
        mock_prep_svc_instance.prepare_document.return_value = (
            "# Test Document\n\nContent with code:\n\n```python\nprint('hello')\n```", 
            None, 
            {}
        )
        
        # Mock validator
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        mock_validator_instance.validate_document_input.return_value = None
        
        # Mock embedding service
        mock_embedding_svc_instance = Mock()
        mock_embedding_svc.return_value = mock_embedding_svc_instance
        mock_embedding_svc_instance.generate_embeddings_batch.return_value = [[0.1, 0.2], [0.3, 0.4]]
        
        # Mock vector store
        mock_vector_store.add_documents.return_value = None
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Create test document metadata
        doc_metadata = DocumentMetadata(
            document_id="test_doc",
            title="Test Document",
            user_id="test_user"
        )
        
        # Test document insertion with chunking enabled
        result = manager.insert_document(
            text="# Test Document\n\nContent with code:\n\n```python\nprint('hello')\n```",
            metadata=doc_metadata,
            collection_name="test_collection",
            enable_chunking=True,
            document_id="test_doc"
        )
        
        # Verify result structure
        assert isinstance(result, InsertionResult)
        assert result.success == True
        assert result.document_id == "test_doc"
        assert result.chunk_count > 0
        assert len(result.chunk_ids) == result.chunk_count
        
        # Verify hybrid chunking stats are included
        assert result.hybrid_chunking_stats is not None
        assert 'processing_time_ms' in result.hybrid_chunking_stats
        assert 'total_chunks' in result.hybrid_chunking_stats
        
        # Verify vector store was called with correct data
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        assert call_args[1]['collection_name'] == "test_collection"
        assert len(call_args[1]['chunks']) > 0
        assert len(call_args[1]['embeddings']) > 0
        assert len(call_args[1]['metadata']) > 0
        assert len(call_args[1]['ids']) > 0
    
    def test_hybrid_chunker_metadata_integration(self):
        """Test that hybrid chunker metadata is properly integrated into chunk metadata."""
        # Create mock dependencies
        mock_vector_store = Mock()
        mock_config = Mock()
        mock_config.get.return_value = 200  # small chunk size to ensure chunking
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Test document with markdown structure
        test_document = """---
title: "Test Document"
author: "Test Author"
---

# Introduction

This is the introduction section.

## Code Example

Here's some code:

```python
def hello_world():
    print("Hello, World!")
    return "success"
```

## Data Table

| Name | Value |
|------|-------|
| A    | 1     |
| B    | 2     |

More content here."""
        
        # Mock the hybrid chunker to return controlled results
        with patch.object(manager.hybrid_chunker, 'chunk_document') as mock_chunk:
            # Create mock hybrid chunk results
            from research_agent_backend.core.document_processor.chunking import ChunkResult, HybridChunkResult
            
            mock_chunks = [
                Mock(content="Introduction content", metadata={
                    'header_hierarchy': ['Introduction'],
                    'content_type': 'prose',
                    'section_title': 'Introduction',
                    'section_level': 1,
                    'source_document_id': 'test_doc'
                }),
                Mock(content="def hello_world():", metadata={
                    'header_hierarchy': ['Code Example'],
                    'content_type': 'code_block',
                    'section_title': 'Code Example', 
                    'section_level': 2,
                    'code_language': 'python',
                    'is_atomic_unit': True,
                    'source_document_id': 'test_doc'
                }),
                Mock(content="| Name | Value |", metadata={
                    'header_hierarchy': ['Data Table'],
                    'content_type': 'table',
                    'section_title': 'Data Table',
                    'section_level': 2,
                    'is_atomic_unit': True,
                    'source_document_id': 'test_doc'
                })
            ]
            
            mock_hybrid_result = HybridChunkResult(
                chunks=mock_chunks,
                processing_stats={
                    'processing_time_ms': 150,
                    'total_chunks': 3,
                    'atomic_units_detected': 2,
                    'sections_processed': 3
                }
            )
            mock_chunk.return_value = mock_hybrid_result
            
            # Mock other dependencies
            with patch.object(manager.validator, 'validate_document_input'), \
                 patch.object(manager.preparation_service, 'prepare_document') as mock_prep, \
                 patch.object(manager.embedding_svc, 'generate_embeddings_batch') as mock_embed, \
                 patch.object(manager.vector_store, 'add_documents'):
                
                mock_prep.return_value = (test_document, None, {})
                mock_embed.return_value = [[0.1] * 384] * 3  # Mock embeddings
                
                doc_metadata = DocumentMetadata(
                    document_id="test_doc",
                    title="Test Document",
                    user_id="test_user"
                )
                
                result = manager.insert_document(
                    text=test_document,
                    metadata=doc_metadata,
                    collection_name="test_collection",
                    enable_chunking=True,
                    document_id="test_doc"
                )
                
                # Verify hybrid chunking was called
                mock_chunk.assert_called_once()
                
                # Verify result includes hybrid stats
                assert result.hybrid_chunking_stats is not None
                assert result.hybrid_chunking_stats['total_chunks'] == 3
                assert result.hybrid_chunking_stats['atomic_units_detected'] == 2
                
                # Verify chunk count matches
                assert result.chunk_count == 3
                assert len(result.chunk_ids) == 3
    
    def test_hybrid_chunker_preserves_atomic_units(self):
        """Test that hybrid chunker atomic unit preservation works in full pipeline."""
        # Create manager with code and table preservation enabled
        mock_vector_store = Mock()
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default: {
            "chunking_strategy.chunk_size": 200,
            "chunking_strategy.chunk_overlap": 20,
            "chunking_strategy.preserve_code_blocks": True,
            "chunking_strategy.preserve_tables": True
        }.get(key, default)
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Verify chunker configuration
        assert manager.hybrid_chunker.config.preserve_code_blocks == True
        assert manager.hybrid_chunker.config.preserve_tables == True
        assert manager.hybrid_chunker.config.chunk_size == 200
        assert manager.hybrid_chunker.config.chunk_overlap == 20
    
    def test_error_handling_in_hybrid_chunking_pipeline(self):
        """Test error handling when hybrid chunking fails."""
        mock_vector_store = Mock()
        mock_config = Mock()
        mock_config.get.return_value = 512
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Mock hybrid chunker to raise an exception
        with patch.object(manager.hybrid_chunker, 'chunk_document') as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")
            
            # Mock other dependencies to allow test to reach chunking
            with patch.object(manager.validator, 'validate_document_input'), \
                 patch.object(manager.preparation_service, 'prepare_document') as mock_prep:
                
                mock_prep.return_value = ("test content", None, {})
                
                doc_metadata = DocumentMetadata(
                    document_id="test_doc",
                    title="Test Document", 
                    user_id="test_user"
                )
                
                # Expect InsertionError due to chunking failure
                from research_agent_backend.core.document_insertion.exceptions import InsertionError
                with pytest.raises(InsertionError, match="Unexpected error during document insertion"):
                    manager.insert_document(
                        text="test content",
                        metadata=doc_metadata,
                        collection_name="test_collection",
                        enable_chunking=True
                    )


class TestHybridChunkerConfigurationIntegration:
    """Test configuration integration between DocumentInsertionManager and HybridChunker."""
    
    def test_configuration_passed_to_hybrid_chunker(self):
        """Test that configuration is properly passed to HybridChunker."""
        mock_vector_store = Mock()
        mock_config = Mock()
        
        # Configure mock to return specific values
        config_values = {
            "chunking_strategy.chunk_size": 1024,
            "chunking_strategy.chunk_overlap": 100,
            "chunking_strategy.preserve_code_blocks": False,
            "chunking_strategy.preserve_tables": False
        }
        mock_config.get.side_effect = lambda key, default: config_values.get(key, default)
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Verify configuration was passed correctly
        assert manager.hybrid_chunker.config.chunk_size == 1024
        assert manager.hybrid_chunker.config.chunk_overlap == 100
        assert manager.hybrid_chunker.config.preserve_code_blocks == False
        assert manager.hybrid_chunker.config.preserve_tables == False
        assert manager.hybrid_chunker.config.boundary_strategy.value == "intelligent"
    
    def test_default_configuration_values(self):
        """Test default configuration values when config manager doesn't provide them."""
        mock_vector_store = Mock()
        mock_config = Mock()
        mock_config.get.side_effect = lambda key, default: default  # Always return default
        
        manager = DocumentInsertionManager(
            vector_store=mock_vector_store,
            config_manager=mock_config
        )
        
        # Verify default values are used
        assert manager.hybrid_chunker.config.chunk_size == 512
        assert manager.hybrid_chunker.config.chunk_overlap == 50
        assert manager.hybrid_chunker.config.preserve_code_blocks == True
        assert manager.hybrid_chunker.config.preserve_tables == True 