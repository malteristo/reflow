# Research Agent

AI-powered research agent with local-first RAG (Retrieval-Augmented Generation) capabilities.

## Overview

Research Agent is a comprehensive tool that combines the power of AI with local knowledge management to provide intelligent research assistance. It features a hybrid architecture with a Python CLI backend and MCP (Model Context Protocol) server integration for seamless interaction with modern AI development environments like Cursor.

## Features

- **Local-First RAG Pipeline**: Vector database with ChromaDB, local embeddings, and intelligent re-ranking
- **Hybrid Document Processing**: Markdown-aware chunking with metadata extraction
- **MCP Server Integration**: Seamless integration with Cursor IDE and other MCP-compatible tools
- **Flexible Architecture**: Modular design supporting both CLI and programmatic access
- **Team-Ready Design**: Built with future team collaboration features in mind

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cursor IDE    │◄──►│   MCP Server     │◄──►│  Backend CLI    │
│   (Client)      │    │  (FastMCP)       │    │  (Core Logic)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                │               │  Vector Store   │
                                │               │  (ChromaDB)     │
                                │               └─────────────────┘
                                │                        │
                                │                        ▼
                                │               ┌─────────────────┐
                                └──────────────►│  Knowledge Base │
                                                │  (Documents)    │
                                                └─────────────────┘
```

## Technology Stack

- **Backend**: Python 3.11+ with ChromaDB, sentence-transformers, Typer
- **MCP Server**: FastMCP framework for Model Context Protocol integration
- **Vector Database**: ChromaDB (primary), SQLite+sqlite-vec (fallback)
- **Embeddings**: sentence-transformers/multi-qa-MiniLM-L6-cos-v1 (default)
- **Re-ranking**: cross-encoder/ms-marco-MiniLM-L6-v2

## Quick Start

### Prerequisites

- Python 3.11 or higher
- uv package manager (recommended) or pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd research-agent
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your API keys (optional, for external LLM providers):
```bash
# Edit .env with your preferred editor
nano .env
```

3. The main configuration is in `researchagent.config.json`. The default settings work for local-only usage.

### Basic Usage

#### CLI Usage

```bash
# Initialize knowledge base
research-agent kb init

# Add documents
research-agent kb add-document path/to/document.md
research-agent kb add-folder path/to/documents/

# Query knowledge base
research-agent query "What is the main concept?"

# List collections
research-agent collections list
```

#### MCP Server Usage

The MCP server provides integration with Cursor IDE and other MCP-compatible tools. Configuration details are in the project documentation.

## Development

### Project Structure

```
src/
├── research_agent_backend/     # Python CLI Backend
│   ├── cli/                   # CLI commands
│   ├── core/                  # Core business logic
│   ├── services/              # Service layer
│   ├── models/                # Data models
│   ├── utils/                 # Utilities
│   └── tests/                 # Backend tests
├── mcp_server/                # FastMCP Server
│   ├── tools/                 # MCP tool definitions
│   ├── handlers/              # Request handlers
│   └── tests/                 # MCP server tests
├── config/                    # Configuration management
└── shared/                    # Shared components
```

### Development Setup

1. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest
```

4. Code formatting:
```bash
black .
isort .
```

5. Type checking:
```bash
mypy src/
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the project coding standards
4. Add tests for new functionality
5. Run the test suite and ensure all tests pass
6. Submit a pull request

## Configuration

### Main Configuration File

The primary configuration is in `researchagent.config.json`. It extends `config/defaults/default_config.json` with project-specific settings.

Key configuration sections:
- `embedding_model`: Embedding model configuration
- `vector_store`: Vector database settings
- `chunking_strategy`: Document processing settings
- `rag_pipeline`: Query processing parameters
- `logging`: Logging configuration

### Environment Variables

Environment variables are used only for sensitive data like API keys. See `.env.example` for available options.

## Testing

The project includes comprehensive testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m e2e
```

## Documentation

- [Architecture Guide](.cursor/rules/architecture.mdc)
- [Development Workflow](.cursor/rules/dev_workflow.mdc)
- [Project Standards](.cursor/rules/ra-001-project-overview-and-standards.mdc)

## License

MIT License - see LICENSE file for details.

## Support

For questions, issues, or contributions, please visit the project repository or contact the development team. 