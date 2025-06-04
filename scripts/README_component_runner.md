# Component-Based Test Runner

A sophisticated test execution tool for the Research Agent project that provides component-level test isolation, parallel execution, and targeted debugging capabilities.

## Overview

The Component-Based Test Runner (`component_runner.py`) addresses the challenge of debugging and maintaining large test suites by organizing tests into logical components with proper dependency mapping and isolation.

## Features

### 🧩 Component Discovery
- **Automatic Detection**: Discovers 13 test components across the codebase
- **Smart Mapping**: Maps test files to functional components (core, cli, vector_store, etc.)
- **Dependency Graph**: Builds dependency relationships for optimal execution order
- **Test Estimation**: Provides test count estimates per component

### ⚡ Execution Modes
- **Single Component**: Focus on specific component for targeted debugging
- **Parallel Execution**: Run multiple components simultaneously for efficiency  
- **Failure Category**: Execute only tests matching specific failure patterns
- **Sequential Mode**: Traditional sequential execution with enhanced reporting

### 🔍 Integration with Test Analysis
- **Category Filtering**: Uses `test_analyzer.py` results for targeted testing
- **Smart Discovery**: Finds test files containing specific failing test functions
- **Pattern Matching**: Supports multiple failure categories (assertion_failure, mock_error, etc.)

### 📊 Comprehensive Reporting
- **JSON Reports**: Detailed execution results with timing and failure information
- **Component Statistics**: Per-component success rates and execution times
- **Failure Details**: Specific test names and error information
- **Dependency Information**: Component relationship mapping

## Usage

### List Available Components
```bash
python scripts/component_runner.py --list-components
```

### Run Specific Component
```bash
# Basic execution
python scripts/component_runner.py --component utils

# With verbose output
python scripts/component_runner.py --component core --verbose

# With custom timeout
python scripts/component_runner.py --component rag_pipeline --timeout 600
```

### Parallel Execution
```bash
# Run all components in parallel
python scripts/component_runner.py --parallel

# Control worker count
python scripts/component_runner.py --parallel --max-workers 8

# With isolation (recommended)
python scripts/component_runner.py --parallel --isolated
```

### Failure Category Testing
```bash
# Run tests with assertion failures
python scripts/component_runner.py --failure-category assertion_failure

# Run tests with configuration errors
python scripts/component_runner.py --failure-category configuration_error

# Available categories: assertion_failure, configuration_error, mock_error, 
# missing_method, not_implemented, type_error
```

### Report Generation
```bash
# Custom output location
python scripts/component_runner.py --component core --output core_results.json

# Full parallel run with detailed report
python scripts/component_runner.py --parallel --output full_analysis.json
```

## Component Architecture

### Base Components (No Dependencies)
- **models**: Data models and schema validation
- **utils**: Utility functions and helpers

### Infrastructure Components
- **config**: Configuration management system
- **vector_store**: Vector database integration
- **embedding**: Embedding generation services

### Processing Components  
- **document_processor**: Document processing and chunking
- **document_insertion**: Document insertion and transactions

### Interface Components
- **cli**: Command-line interface commands
- **cli_unit**: CLI unit tests

### Integration Components
- **core**: Core RAG functionality
- **rag_pipeline**: End-to-end RAG pipeline
- **integration**: Cross-component integration tests
- **performance**: Performance and load testing

## Command Reference

### Basic Options
- `--component, -c`: Run tests for specific component
- `--parallel, -p`: Run components in parallel
- `--list-components, -l`: List available components

### Execution Control
- `--max-workers, -w`: Maximum parallel workers (default: 4)
- `--timeout, -t`: Test timeout in seconds (default: 300)
- `--isolated, -i`: Run in isolated environment
- `--verbose, -v`: Verbose output

### Filtering Options
- `--failure-category, -f`: Run tests for specific failure category

### Output Options
- `--output, -o`: Output file for results (default: component_test_results.json)

## Integration Workflow

### With Test Analyzer
1. Run `python scripts/test_analyzer.py` to generate failure analysis
2. Use `--failure-category` to run only tests with specific failure patterns
3. Focus debugging efforts on specific failure types

### With TDD Workflow
1. Use `--component` for focused development on specific modules
2. Use parallel execution for full regression testing
3. Use failure categories for targeted issue resolution

### With CI/CD
```bash
# Quick component validation
python scripts/component_runner.py --parallel --max-workers 4 --timeout 120

# Detailed analysis for releases
python scripts/component_runner.py --parallel --output ci_results.json
```

## Performance Benefits

### Execution Speed
- **Parallel Execution**: Up to 4x faster with optimal worker configuration
- **Component Isolation**: Faster failure detection with `-x` flag
- **Smart Filtering**: Run only relevant tests for specific issues

### Debugging Efficiency
- **Focused Testing**: Target specific components for development
- **Failure Isolation**: Identify component-specific issues quickly
- **Pattern Analysis**: Group similar failures for batch resolution

### Resource Management
- **Configurable Timeouts**: Prevent long-running test hangs
- **Worker Control**: Balance speed vs system resources
- **Memory Isolation**: Independent component execution

## Output Examples

### Component List
```
📋 Available Components:
--------------------------------------------------------------------------------
🧩 core
   Description: Core RAG functionality (search, reranking, feedback)
   Test files: 5
   Estimated tests: 120
   Dependencies: models, utils
```

### Execution Summary
```
============================================================
📊 EXECUTION SUMMARY
============================================================
Components: 11/13 successful
Tests: 171/182 passed (94.0%)

❌ Components with failures:
   utils: 1 failures
     - test_status_command_shows_collection_details
```

### JSON Report Structure
```json
{
  "timestamp": "2025-06-04T09:38:06.572307",
  "summary": {
    "total_components": 13,
    "successful_components": 11,
    "total_tests": 182,
    "passed_tests": 171,
    "total_execution_time": 45.2
  },
  "component_results": { ... },
  "component_info": { ... }
}
```

## Best Practices

### Development Workflow
1. **Start with Components**: Use `--list-components` to understand test organization
2. **Focus Development**: Use `--component` for active development areas
3. **Regular Validation**: Use `--parallel` for full regression testing
4. **Issue Resolution**: Use `--failure-category` for systematic debugging

### Performance Optimization
1. **Worker Tuning**: Start with 2-4 workers, adjust based on system resources
2. **Timeout Settings**: Use shorter timeouts for CI, longer for local debugging
3. **Component Prioritization**: Focus on base components first (models, utils)

### Integration Points
1. **Test Analyzer**: Generate analysis first, then use category filtering
2. **CI Pipeline**: Include parallel execution for comprehensive validation
3. **Development Tools**: Integrate with IDE test runners and debuggers

## Troubleshooting

### Common Issues
1. **No Components Found**: Check test directory structure and patterns
2. **Timeout Errors**: Increase timeout or check for infinite loops
3. **Category Not Found**: Run test analyzer first to generate analysis results

### Performance Issues
1. **High Memory Usage**: Reduce worker count or use isolation
2. **Slow Execution**: Check component dependencies and execution order
3. **Hanging Tests**: Use shorter timeouts and investigate specific test files

---

**Part of Task 36.2: Test Failure Analysis and Debugging Tools**  
**Integrates with**: `test_analyzer.py`, pytest infrastructure, CI/CD pipelines 