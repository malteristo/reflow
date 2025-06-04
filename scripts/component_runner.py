#!/usr/bin/env python3
"""
Component-Based Test Runner for Research Agent

This script provides focused test execution by component with proper isolation,
dependency mapping, and parallel execution capabilities for efficient debugging.

Usage:
    python scripts/component_runner.py --component core
    python scripts/component_runner.py --component cli --parallel
    python scripts/component_runner.py --failure-category assertion_failure
    python scripts/component_runner.py --list-components
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from collections import defaultdict, namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Union
import tempfile
import shutil
import os


@dataclass
class ComponentInfo:
    """Information about a test component."""
    name: str
    path: Path
    test_files: List[str]
    dependencies: List[str]
    test_count: int
    description: str


@dataclass
class TestExecutionResult:
    """Result of running tests for a component."""
    component: str
    success: bool
    execution_time: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_message: Optional[str]
    detailed_output: str
    failed_test_names: List[str]


@dataclass
class DependencyNode:
    """Node in the component dependency graph."""
    component: str
    dependencies: Set[str]
    dependents: Set[str]
    test_files: List[str]
    execution_order: int


class ComponentTestRunner:
    """Main test runner class for component-based test execution."""
    
    def __init__(self, test_root: str = "src/research_agent_backend/tests"):
        self.test_root = Path(test_root)
        self.components = self._discover_components()
        self.dependency_graph = self._build_dependency_graph()
        
    def _discover_components(self) -> Dict[str, ComponentInfo]:
        """Discover all test components and their structure."""
        components = {}
        
        # Define component mappings based on source structure
        component_definitions = {
            'core': {
                'path': 'unit/core',
                'description': 'Core RAG functionality (search, reranking, feedback)',
                'dependencies': ['models', 'utils']
            },
            'vector_store': {
                'path': '.',
                'pattern': 'test_vector_store*.py',
                'description': 'Vector database integration and operations',
                'dependencies': ['models', 'config']
            },
            'embedding': {
                'path': 'unit',
                'pattern': '*embedding*.py',
                'description': 'Embedding generation services (local and API)',
                'dependencies': ['config', 'models']
            },
            'document_processor': {
                'path': 'unit/document_processor',
                'description': 'Document processing and chunking pipeline',
                'dependencies': ['models', 'config']
            },
            'document_insertion': {
                'path': 'unit/document_insertion',
                'description': 'Document insertion and transaction management',
                'dependencies': ['document_processor', 'vector_store', 'embedding']
            },
            'cli': {
                'path': 'cli',
                'description': 'Command-line interface commands',
                'dependencies': ['core', 'vector_store', 'document_processor']
            },
            'cli_unit': {
                'path': 'unit',
                'pattern': 'test_cli_*.py',
                'description': 'CLI unit tests',
                'dependencies': ['cli', 'config']
            },
            'config': {
                'path': 'unit',
                'pattern': 'test_config*.py',
                'description': 'Configuration management system',
                'dependencies': ['utils']
            },
            'models': {
                'path': 'unit',
                'pattern': 'test_*schema*.py,test_*model*.py',
                'description': 'Data models and schema validation',
                'dependencies': []
            },
            'utils': {
                'path': 'unit',
                'pattern': 'test_*util*.py',
                'description': 'Utility functions and helpers',
                'dependencies': []
            },
            'integration': {
                'path': 'integration',
                'description': 'Integration tests across multiple components',
                'dependencies': ['core', 'cli', 'vector_store']
            },
            'performance': {
                'path': 'performance',
                'description': 'Performance and load testing',
                'dependencies': ['core', 'vector_store']
            },
            'rag_pipeline': {
                'path': '.',
                'pattern': 'test_rag_*.py',
                'description': 'End-to-end RAG pipeline testing',
                'dependencies': ['core', 'vector_store', 'embedding', 'document_processor']
            }
        }
        
        for comp_name, comp_def in component_definitions.items():
            component_path = self.test_root / comp_def['path']
            test_files = self._find_test_files(component_path, comp_def.get('pattern'))
            
            if test_files:  # Only include components that have test files
                components[comp_name] = ComponentInfo(
                    name=comp_name,
                    path=component_path,
                    test_files=test_files,
                    dependencies=comp_def.get('dependencies', []),
                    test_count=self._count_tests_in_files(test_files),
                    description=comp_def['description']
                )
        
        return components
    
    def _find_test_files(self, component_path: Path, pattern: Optional[str] = None) -> List[str]:
        """Find test files in a component directory."""
        test_files = []
        
        if not component_path.exists():
            return test_files
        
        if pattern:
            patterns = pattern.split(',')
            for pat in patterns:
                test_files.extend(str(f) for f in component_path.rglob(pat.strip()) if f.is_file())
        else:
            # Default: find all test_*.py files
            test_files.extend(str(f) for f in component_path.rglob('test_*.py') if f.is_file())
        
        return sorted(test_files)
    
    def _count_tests_in_files(self, test_files: List[str]) -> int:
        """Estimate number of tests in the given files."""
        total_tests = 0
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                    # Simple heuristic: count function definitions starting with 'test_'
                    total_tests += content.count('def test_')
            except Exception:
                continue
        return total_tests
    
    def _build_dependency_graph(self) -> Dict[str, DependencyNode]:
        """Build component dependency graph for execution ordering."""
        graph = {}
        
        # Create nodes
        for comp_name, comp_info in self.components.items():
            graph[comp_name] = DependencyNode(
                component=comp_name,
                dependencies=set(comp_info.dependencies),
                dependents=set(),
                test_files=comp_info.test_files,
                execution_order=0
            )
        
        # Build reverse dependencies (dependents)
        for comp_name, node in graph.items():
            for dep in node.dependencies:
                if dep in graph:
                    graph[dep].dependents.add(comp_name)
        
        # Calculate execution order using topological sort
        self._calculate_execution_order(graph)
        
        return graph
    
    def _calculate_execution_order(self, graph: Dict[str, DependencyNode]):
        """Calculate optimal execution order using topological sort."""
        # Initialize all orders to 0
        in_degree = {comp: len(node.dependencies) for comp, node in graph.items()}
        queue = [comp for comp, degree in in_degree.items() if degree == 0]
        order = 0
        
        while queue:
            # Process all components at current level
            current_level = queue.copy()
            queue.clear()
            
            for comp in current_level:
                graph[comp].execution_order = order
                
                # Update in-degrees for dependents
                for dependent in graph[comp].dependents:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
            
            order += 1
    
    def list_components(self) -> List[ComponentInfo]:
        """List all discovered components with their information."""
        return sorted(self.components.values(), key=lambda c: c.name)
    
    def run_component(self, component_name: str, isolated: bool = True, 
                     verbose: bool = False, timeout: int = 300) -> TestExecutionResult:
        """Run tests for a specific component with optional isolation."""
        if component_name not in self.components:
            return TestExecutionResult(
                component=component_name,
                success=False,
                execution_time=0.0,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_message=f"Component '{component_name}' not found",
                detailed_output="",
                failed_test_names=[]
            )
        
        component = self.components[component_name]
        
        print(f"🧪 Running tests for component: {component_name}")
        print(f"   Description: {component.description}")
        print(f"   Test files: {len(component.test_files)}")
        print(f"   Estimated tests: {component.test_count}")
        
        start_time = time.time()
        
        # Prepare test execution environment
        if isolated:
            return self._run_isolated(component, verbose, timeout)
        else:
            return self._run_standard(component, verbose, timeout)
    
    def _run_isolated(self, component: ComponentInfo, verbose: bool, 
                     timeout: int) -> TestExecutionResult:
        """Run component tests in isolated environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create isolated test environment
            temp_test_dir = Path(temp_dir) / "isolated_tests"
            temp_test_dir.mkdir(parents=True)
            
            # Copy necessary files
            self._setup_isolated_environment(component, temp_test_dir)
            
            # Run tests in isolated environment
            cmd = self._build_pytest_command(component.test_files, verbose, temp_test_dir)
            return self._execute_tests(component.name, cmd, timeout)
    
    def _run_standard(self, component: ComponentInfo, verbose: bool, 
                     timeout: int) -> TestExecutionResult:
        """Run component tests in standard environment."""
        cmd = self._build_pytest_command(component.test_files, verbose)
        return self._execute_tests(component.name, cmd, timeout)
    
    def _setup_isolated_environment(self, component: ComponentInfo, temp_dir: Path):
        """Setup isolated test environment by copying necessary files."""
        # For now, use the standard environment but with isolated pytest execution
        # Full isolation would require copying source code and dependencies
        pass
    
    def _build_pytest_command(self, test_files: List[str], verbose: bool, 
                             temp_dir: Optional[Path] = None) -> List[str]:
        """Build pytest command for the given test files."""
        cmd = [sys.executable, '-m', 'pytest']
        
        # Add test files (convert to relative paths)
        if temp_dir:
            # For isolated runs, adjust paths
            cmd.extend(test_files)
        else:
            cmd.extend(test_files)
        
        # Add common pytest options
        cmd.extend([
            '--tb=short',
            '--no-header',
            '-x',  # Stop on first failure for component isolation
            '--disable-warnings'
        ])
        
        if verbose:
            cmd.append('-v')
        else:
            cmd.append('-q')
        
        return cmd
    
    def _execute_tests(self, component_name: str, cmd: List[str], 
                      timeout: int) -> TestExecutionResult:
        """Execute pytest command and parse results."""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=Path.cwd()
            )
            
            execution_time = time.time() - start_time
            
            # Parse test results
            stats = self._parse_test_output(result.stdout)
            failed_tests = self._extract_failed_test_names(result.stdout)
            
            return TestExecutionResult(
                component=component_name,
                success=result.returncode == 0,
                execution_time=execution_time,
                total_tests=stats.get('total', 0),
                passed_tests=stats.get('passed', 0),
                failed_tests=stats.get('failed', 0),
                skipped_tests=stats.get('skipped', 0),
                error_message=None if result.returncode == 0 else "Tests failed",
                detailed_output=result.stdout + "\n" + result.stderr,
                failed_test_names=failed_tests
            )
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestExecutionResult(
                component=component_name,
                success=False,
                execution_time=execution_time,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_message=f"Tests timed out after {timeout} seconds",
                detailed_output="",
                failed_test_names=[]
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestExecutionResult(
                component=component_name,
                success=False,
                execution_time=execution_time,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_message=str(e),
                detailed_output="",
                failed_test_names=[]
            )
    
    def _parse_test_output(self, output: str) -> Dict[str, int]:
        """Parse pytest output to extract test statistics."""
        stats = {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        # Look for pytest summary line
        import re
        
        # Pattern: "X failed, Y passed in Z.ZZs"
        pattern = r'(\d+) failed.*?(\d+) passed'
        match = re.search(pattern, output)
        if match:
            stats['failed'] = int(match.group(1))
            stats['passed'] = int(match.group(2))
            stats['total'] = stats['failed'] + stats['passed']
        
        # Pattern: "X passed in Z.ZZs" (no failures)
        pattern = r'(\d+) passed'
        match = re.search(pattern, output)
        if match and stats['total'] == 0:
            stats['passed'] = int(match.group(1))
            stats['total'] = stats['passed']
        
        # Pattern for skipped tests
        pattern = r'(\d+) skipped'
        match = re.search(pattern, output)
        if match:
            stats['skipped'] = int(match.group(1))
            stats['total'] += stats['skipped']
        
        return stats
    
    def _extract_failed_test_names(self, output: str) -> List[str]:
        """Extract names of failed tests from pytest output."""
        failed_tests = []
        lines = output.split('\n')
        
        for line in lines:
            if 'FAILED' in line and '::' in line:
                # Extract test name from line like "FAILED test_file.py::test_name"
                parts = line.split('FAILED')
                if len(parts) > 1:
                    test_path = parts[1].strip()
                    failed_tests.append(test_path)
        
        return failed_tests
    
    def run_components_parallel(self, component_names: List[str], 
                               max_workers: int = 4, timeout: int = 300) -> Dict[str, TestExecutionResult]:
        """Run multiple components in parallel."""
        results = {}
        
        if not component_names:
            component_names = list(self.components.keys())
        
        # Sort components by execution order for optimal dependency handling
        sorted_components = sorted(
            component_names,
            key=lambda c: self.dependency_graph.get(c, DependencyNode(c, set(), set(), [], 999)).execution_order
        )
        
        print(f"🚀 Running {len(sorted_components)} components in parallel (max {max_workers} workers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_component = {
                executor.submit(self.run_component, comp_name, isolated=True, timeout=timeout): comp_name
                for comp_name in sorted_components
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_component):
                comp_name = future_to_component[future]
                try:
                    result = future.result()
                    results[comp_name] = result
                    
                    # Print progress
                    status = "✅ PASSED" if result.success else "❌ FAILED"
                    print(f"   {status} {comp_name}: {result.passed_tests}/{result.total_tests} tests")
                    
                except Exception as e:
                    results[comp_name] = TestExecutionResult(
                        component=comp_name,
                        success=False,
                        execution_time=0.0,
                        total_tests=0,
                        passed_tests=0,
                        failed_tests=0,
                        skipped_tests=0,
                        error_message=str(e),
                        detailed_output="",
                        failed_test_names=[]
                    )
        
        return results
    
    def run_by_failure_category(self, category: str, max_workers: int = 2) -> Dict[str, TestExecutionResult]:
        """Run tests filtered by failure category using test analyzer."""
        print(f"🎯 Running tests for failure category: {category}")
        
        # Load previous analysis results
        analysis_file = Path('scripts/test_analysis_results.json')
        if not analysis_file.exists():
            print("❌ No analysis results found. Run test_analyzer.py first.")
            return {}
        
        with open(analysis_file) as f:
            analysis = json.load(f)
        
        # Find test files containing failures of this category
        category_tests = set()
        
        # Look through failure patterns to find tests of the specified category
        for pattern in analysis.get('failure_patterns', []):
            if pattern.get('pattern_id') == category:
                test_names = pattern.get('test_names', [])
                for test_name in test_names:
                    # Convert test name to file path
                    # Test names are just the test function names, we need to find which files contain them
                    category_tests.add(test_name)
                break
        
        if not category_tests:
            available_categories = [p.get('pattern_id', 'unknown') for p in analysis.get('failure_patterns', [])]
            print(f"❌ No tests found for category: {category}")
            print(f"🔍 Available categories: {', '.join(available_categories)}")
            return {}
        
        print(f"📁 Found {len(category_tests)} tests with {category} failures")
        
        # Now find which test files might contain these test names
        # We'll search through all test files to find matches
        matching_test_files = set()
        for comp_name, comp_info in self.components.items():
            for test_file in comp_info.test_files:
                try:
                    with open(test_file, 'r') as f:
                        content = f.read()
                        # Check if any of the failing test names are in this file
                        for test_name in category_tests:
                            if f"def {test_name}(" in content:
                                matching_test_files.add(test_file)
                                break
                except Exception:
                    continue
        
        if not matching_test_files:
            print(f"❌ No test files found containing {category} failures")
            return {}
        
        # Map test files to components
        components_to_run = []
        for comp_name, comp_info in self.components.items():
            if any(test_file in matching_test_files for test_file in comp_info.test_files):
                components_to_run.append(comp_name)
        
        print(f"🧪 Running {len(components_to_run)} components: {', '.join(components_to_run)}")
        
        return self.run_components_parallel(components_to_run, max_workers)
    
    def generate_execution_report(self, results: Dict[str, TestExecutionResult], 
                                 output_path: str = "component_test_results.json"):
        """Generate detailed execution report."""
        timestamp = datetime.now().isoformat()
        
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_components': len(results),
                'successful_components': sum(1 for r in results.values() if r.success),
                'failed_components': sum(1 for r in results.values() if not r.success),
                'total_tests': sum(r.total_tests for r in results.values()),
                'passed_tests': sum(r.passed_tests for r in results.values()),
                'failed_tests': sum(r.failed_tests for r in results.values()),
                'total_execution_time': sum(r.execution_time for r in results.values())
            },
            'component_results': {
                comp_name: asdict(result) for comp_name, result in results.items()
            },
            'component_info': {
                comp_name: asdict(comp_info) for comp_name, comp_info in self.components.items()
                if comp_name in results
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Report saved to: {output_path}")
        return report


def main():
    """Main entry point for component test runner."""
    parser = argparse.ArgumentParser(description='Component-Based Test Runner for Research Agent')
    
    parser.add_argument('--component', '-c', help='Run tests for specific component')
    parser.add_argument('--parallel', '-p', action='store_true', help='Run components in parallel')
    parser.add_argument('--max-workers', '-w', type=int, default=4, help='Max parallel workers')
    parser.add_argument('--timeout', '-t', type=int, default=300, help='Test timeout in seconds')
    parser.add_argument('--isolated', '-i', action='store_true', help='Run in isolated environment')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--list-components', '-l', action='store_true', help='List available components')
    parser.add_argument('--failure-category', '-f', help='Run tests for specific failure category')
    parser.add_argument('--output', '-o', default='component_test_results.json', 
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ComponentTestRunner()
    
    if args.list_components:
        print("📋 Available Components:")
        print("-" * 80)
        for component in runner.list_components():
            deps = ', '.join(component.dependencies) if component.dependencies else 'None'
            print(f"🧩 {component.name}")
            print(f"   Description: {component.description}")
            print(f"   Test files: {len(component.test_files)}")
            print(f"   Estimated tests: {component.test_count}")
            print(f"   Dependencies: {deps}")
            print()
        return
    
    # Run tests based on arguments
    results = {}
    
    if args.failure_category:
        results = runner.run_by_failure_category(args.failure_category, args.max_workers)
    elif args.component:
        result = runner.run_component(
            args.component, 
            isolated=args.isolated, 
            verbose=args.verbose,
            timeout=args.timeout
        )
        results[args.component] = result
    elif args.parallel:
        results = runner.run_components_parallel(
            list(runner.components.keys()),
            max_workers=args.max_workers,
            timeout=args.timeout
        )
    else:
        # Run all components sequentially
        for comp_name in sorted(runner.components.keys()):
            print(f"\n{'='*60}")
            result = runner.run_component(
                comp_name, 
                isolated=args.isolated, 
                verbose=args.verbose,
                timeout=args.timeout
            )
            results[comp_name] = result
    
    # Generate and display summary
    if results:
        runner.generate_execution_report(results, args.output)
        
        print(f"\n{'='*60}")
        print("📊 EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total_components = len(results)
        successful = sum(1 for r in results.values() if r.success)
        total_tests = sum(r.total_tests for r in results.values())
        passed_tests = sum(r.passed_tests for r in results.values())
        failed_tests = sum(r.failed_tests for r in results.values())
        
        print(f"Components: {successful}/{total_components} successful")
        print(f"Tests: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
        
        if failed_tests > 0:
            print(f"\n❌ Components with failures:")
            for comp_name, result in results.items():
                if not result.success:
                    print(f"   {comp_name}: {result.failed_tests} failures")
                    if result.failed_test_names:
                        for test_name in result.failed_test_names[:3]:  # Show first 3
                            print(f"     - {test_name}")
                        if len(result.failed_test_names) > 3:
                            print(f"     ... and {len(result.failed_test_names)-3} more")


if __name__ == '__main__':
    main() 