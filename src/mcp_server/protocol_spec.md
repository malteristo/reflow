# MCP Protocol Specification for Research Agent

## Overview

This document defines the Model Control Protocol (MCP) implementation for the Research Agent, specifying how the FastMCP server interfaces between Cursor IDE and the Python backend CLI.

**Implements:** FR-SI-001, FR-SI-002 from PRD

## Protocol Requirements

### 1. Communication Method
- **Transport:** STDIO (standard input/output)
- **Format:** JSON-RPC 2.0 over MCP protocol
- **Framework:** FastMCP (Python implementation)
- **Mode:** Server mode with client connections from Cursor IDE

### 2. Server Capabilities

#### 2.1 Tools (Required)
The MCP server must expose the following tools corresponding to CLI commands:

| Tool Name | CLI Command | Description | Priority |
|-----------|-------------|-------------|----------|
| `query_knowledge_base` | `research-agent query` | Search knowledge base with re-ranking | High |
| `manage_collections` | `research-agent collections` | CRUD operations for collections | High |
| `ingest_documents` | `research-agent kb` | Add documents to knowledge base | High |
| `manage_projects` | `research-agent projects` | Project-specific operations | Medium |
| `augment_knowledge` | `research-agent kb add-*` | Add external knowledge | Medium |

#### 2.2 Resources (Optional)
- Collection listings
- Document metadata
- Query history

#### 2.3 Prompts (Future)
- Query refinement templates
- Context extraction prompts

### 3. Message Format Specifications

#### 3.1 Tool Request Format
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "query_knowledge_base",
    "arguments": {
      "query": "string",
      "collections": "string|null",
      "top_k": "number",
      "document_context": "string|null"
    }
  },
  "id": "request-id"
}
```

#### 3.2 Tool Response Format
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "formatted_response"
      }
    ]
  },
  "id": "request-id"
}
```

#### 3.3 Error Response Format
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Tool execution failed",
    "data": {
      "details": "specific_error_description"
    }
  },
  "id": "request-id"
}
```

### 4. Tool Specifications

#### 4.1 query_knowledge_base
**Purpose:** Search the knowledge base with semantic similarity and re-ranking

**Parameters:**
- `query` (required, string): User's search query
- `collections` (optional, string): Comma-separated collection names
- `top_k` (optional, number, default=10): Number of results to return
- `document_context` (optional, string): Current document context for enhanced search

**CLI Mapping:** `research-agent query "{query}" --collections="{collections}" --top-k={top_k}`

**Response Format:**
```json
{
  "status": "success|error",
  "results": [
    {
      "content": "chunk_text",
      "relevance_score": 0.95,
      "relevance_label": "Highly Relevant",
      "source_document": "doc_name.md",
      "header_path": "Section > Subsection",
      "metadata": {
        "document_title": "title",
        "content_type": "prose|code_block|table",
        "chunk_sequence_id": 1
      }
    }
  ],
  "query_refinement": {
    "status": "optimal|clarification_needed|too_broad",
    "suggestions": ["keyword1", "keyword2"],
    "message": "user_guidance_message"
  }
}
```

#### 4.2 manage_collections
**Purpose:** Create, list, delete, and manage knowledge collections

**Parameters:**
- `action` (required, string): "create|list|delete|info"
- `collection_name` (conditional, string): Required for create/delete/info
- `collection_type` (conditional, string): "fundamental|project-specific" for create
- `project_name` (optional, string): For linking collections

**CLI Mapping:** `research-agent collections {action} {collection_name} --type={collection_type}`

#### 4.3 ingest_documents
**Purpose:** Add documents or folders to the knowledge base

**Parameters:**
- `path` (required, string): File or folder path
- `collection` (required, string): Target collection name
- `recursive` (optional, boolean, default=false): Process folders recursively

**CLI Mapping:** `research-agent kb add-document "{path}" --collection="{collection}"`

#### 4.4 manage_projects
**Purpose:** Project-specific knowledge operations

**Parameters:**
- `action` (required, string): "create|link|unlink|set-defaults"
- `project_name` (conditional, string): Project identifier
- `collections` (conditional, string): Collection names for linking

#### 4.5 augment_knowledge
**Purpose:** Add external research to knowledge base

**Parameters:**
- `content` (required, string): Research content to add
- `source` (required, string): Source identifier/URL
- `collection` (required, string): Target collection
- `title` (optional, string): Document title

### 5. Error Handling

#### 5.1 Error Categories
- **Configuration Errors** (-32001): Missing config, invalid settings
- **Collection Errors** (-32002): Collection not found, already exists
- **Document Errors** (-32003): File not found, parsing errors
- **Query Errors** (-32004): Invalid query, embedding failures
- **System Errors** (-32000): General execution failures

#### 5.2 Error Response Structure
All errors must include:
- Standard JSON-RPC 2.0 error format
- Specific error code from categories above
- Human-readable message
- Actionable guidance in error data

### 6. Protocol Compliance Requirements

#### 6.1 FastMCP Compliance
- Use FastMCP decorators for tool definition
- Implement proper parameter validation
- Return structured responses compatible with MCP protocol
- Handle STDIO communication automatically via FastMCP

#### 6.2 Response Formatting
- All text responses must be Markdown-formatted
- Include rich metadata for enhanced user experience
- Provide structured feedback for query refinement
- Support interactive flows for collection management

#### 6.3 Performance Requirements
- Tool calls should respond within 5 seconds for simple operations
- Query operations may take up to 30 seconds for complex searches
- Implement proper timeout handling
- Provide progress feedback for long-running operations

### 7. Testing Requirements

#### 7.1 Protocol Compliance Tests
- Validate all tool parameter schemas
- Test error response formats
- Verify JSON-RPC 2.0 compliance
- Test STDIO communication

#### 7.2 Integration Tests
- CLI command mapping verification
- End-to-end tool execution
- Error handling validation
- Response format validation

#### 7.3 Performance Tests
- Response time measurements
- Memory usage monitoring
- Concurrent request handling
- Large query processing

## Implementation Notes

### Dependencies
- `fastmcp` - FastMCP framework
- `subprocess` - CLI command execution
- `json` - Response formatting
- `typing` - Type annotations
- `logging` - Error tracking

### Configuration
- Server configuration via environment variables
- CLI path configuration
- Timeout settings
- Debug mode support

### Security Considerations
- Input sanitization for all parameters
- Path traversal prevention
- Command injection protection
- Proper error message sanitization 