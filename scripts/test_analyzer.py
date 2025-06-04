#!/usr/bin/env python3
"""
Test Failure Analysis Script for Research Agent

This script provides comprehensive test failure analysis with pattern recognition,
clustering, and reporting capabilities to identify systemic issues that can be
fixed with batch solutions.

Usage:
    python scripts/test_analyzer.py [--output OUTPUT_DIR] [--format json|html|both]
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import xml.etree.ElementTree as ET


@dataclass
class TestFailure:
    """Represents a single test failure with analysis metadata."""
    test_name: str
    module: str
    error_type: str
    error_message: str
    full_traceback: str
    pattern_category: str
    severity: str
    fix_priority: int
    estimated_effort: str
    related_component: str


@dataclass
class FailurePattern:
    """Represents a pattern of similar test failures."""
    pattern_id: str
    description: str
    regex_pattern: str
    category: str
    frequency: int
    test_names: List[str]
    suggested_fix: str
    impact_score: int


@dataclass
class AnalysisReport:
    """Complete analysis report with all findings."""
    timestamp: str
    total_tests: int
    passing_tests: int
    failing_tests: int
    success_rate: float
    failure_patterns: List[FailurePattern]
    failures_by_category: Dict[str, int]
    failures_by_component: Dict[str, int]
    high_priority_fixes: List[str]
    systemic_issues: List[str]


class FailureAnalyzer:
    """Main analyzer class for test failure pattern recognition."""
    
    def __init__(self):
        self.failure_patterns = self._initialize_patterns()
        self.component_mapping = self._initialize_component_mapping()
        
    def _initialize_patterns(self) -> Dict[str, FailurePattern]:
        """Initialize known failure patterns with regex and fixes."""
        patterns = {
            'missing_method': FailurePattern(
                pattern_id='missing_method',
                description='Missing method or attribute errors',
                regex_pattern=r"AttributeError: '.*' object has no attribute '(.+)'",
                category='interface_mismatch',
                frequency=0,
                test_names=[],
                suggested_fix='Add missing method implementation or check interface compatibility',
                impact_score=8
            ),
            'import_error': FailurePattern(
                pattern_id='import_error',
                description='Module import failures',
                regex_pattern=r'ImportError: (.+)|ModuleNotFoundError: (.+)',
                category='import_issues',
                frequency=0,
                test_names=[],
                suggested_fix='Fix import paths or add missing dependencies',
                impact_score=9
            ),
            'type_error': FailurePattern(
                pattern_id='type_error',
                description='Type compatibility issues',
                regex_pattern=r'TypeError: (.+)',
                category='type_mismatch',
                frequency=0,
                test_names=[],
                suggested_fix='Fix type annotations or method signatures',
                impact_score=7
            ),
            'assertion_failure': FailurePattern(
                pattern_id='assertion_failure',
                description='Test assertion failures',
                regex_pattern=r'AssertionError: (.+)|assert (.+)',
                category='logic_errors',
                frequency=0,
                test_names=[],
                suggested_fix='Update test expectations or fix implementation logic',
                impact_score=5
            ),
            'not_implemented': FailurePattern(
                pattern_id='not_implemented',
                description='NotImplementedError placeholders',
                regex_pattern=r'NotImplementedError: (.+)|NotImplementedError$',
                category='incomplete_implementation',
                frequency=0,
                test_names=[],
                suggested_fix='Implement missing functionality',
                impact_score=6
            ),
            'mock_error': FailurePattern(
                pattern_id='mock_error',
                description='Mock setup and configuration issues',
                regex_pattern=r'Mock(.+)Error|mock\.(.+)|MagicMock(.+)',
                category='mock_issues',
                frequency=0,
                test_names=[],
                suggested_fix='Fix mock configuration and setup',
                impact_score=7
            ),
            'configuration_error': FailurePattern(
                pattern_id='configuration_error',
                description='Configuration and setup errors',
                regex_pattern=r'ConfigurationError|Configuration(.+)Error|config(.+)',
                category='configuration_issues',
                frequency=0,
                test_names=[],
                suggested_fix='Fix configuration setup and validation',
                impact_score=8
            )
        }
        return patterns
    
    def _initialize_component_mapping(self) -> Dict[str, str]:
        """Map test paths to system components."""
        return {
            'vector_store': 'Vector Storage System',
            'rag_query': 'RAG Query Engine',
            'embedding': 'Embedding Service',
            'document_processor': 'Document Processing',
            'api_embedding': 'API Embedding Service',
            'collection': 'Collection Management',
            'config': 'Configuration System',
            'cli': 'Command Line Interface',
            'integration': 'Integration Pipeline',
            'performance': 'Performance Testing'
        }
    
    def run_tests_and_capture_failures(self) -> Tuple[List[TestFailure], Dict[str, Any]]:
        """Run test suite and capture detailed failure information."""
        print("🔍 Running test suite and capturing failures...")
        
        # Run pytest with detailed output
        cmd = [
            sys.executable, '-m', 'pytest',
            'src/research_agent_backend/tests/',
            '--tb=short',
            '--no-header',
            '--quiet',
            '--junit-xml=test_results.xml'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse statistics from output
            stats = self._parse_test_statistics(result.stdout)
            
            # Parse detailed failures from XML if available
            failures = []
            if Path('test_results.xml').exists():
                failures = self._parse_junit_xml('test_results.xml')
            else:
                # Fallback to parsing stdout
                failures = self._parse_pytest_output(result.stdout)
            
            return failures, stats
            
        except subprocess.TimeoutExpired:
            print("⚠️  Test run timed out after 5 minutes")
            return [], {'error': 'timeout'}
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return [], {'error': str(e)}
    
    def _parse_test_statistics(self, output: str) -> Dict[str, Any]:
        """Extract test statistics from pytest output."""
        stats = {
            'total_tests': 0,
            'passing_tests': 0,
            'failing_tests': 0,
            'success_rate': 0.0
        }
        
        # Look for pytest summary line
        summary_pattern = r'(\d+) failed.*?(\d+) passed.*?in ([\d.]+)s'
        match = re.search(summary_pattern, output)
        if match:
            stats['failing_tests'] = int(match.group(1))
            stats['passing_tests'] = int(match.group(2))
            stats['total_tests'] = stats['failing_tests'] + stats['passing_tests']
            
        if stats['total_tests'] > 0:
            stats['success_rate'] = stats['passing_tests'] / stats['total_tests']
            
        return stats
    
    def _parse_junit_xml(self, xml_path: str) -> List[TestFailure]:
        """Parse JUnit XML for detailed failure information."""
        failures = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for testcase in root.findall('.//testcase'):
                test_name = testcase.get('name', '')
                module = testcase.get('classname', '')
                
                # Check for failures or errors
                failure_elem = testcase.find('failure')
                error_elem = testcase.find('error')
                
                if failure_elem is not None or error_elem is not None:
                    elem = failure_elem if failure_elem is not None else error_elem
                    error_type = elem.get('type', 'Unknown')
                    error_message = elem.get('message', '')
                    full_traceback = elem.text or ''
                    
                    failure = self._create_test_failure(
                        test_name, module, error_type, error_message, full_traceback
                    )
                    failures.append(failure)
                    
        except Exception as e:
            print(f"⚠️  Error parsing JUnit XML: {e}")
            
        return failures
    
    def _parse_pytest_output(self, output: str) -> List[TestFailure]:
        """Parse pytest stdout for failure information (fallback method)."""
        failures = []
        
        # Split by test failure sections
        sections = re.split(r'_{20,}|={20,}', output)
        
        for section in sections:
            if 'FAILED' in section and '::' in section:
                # Extract test name
                test_match = re.search(r'(\S+)::\S+', section)
                if test_match:
                    test_name = test_match.group(1)
                    
                    # Extract error information
                    error_lines = section.split('\n')
                    error_type = 'Unknown'
                    error_message = ''
                    
                    for line in error_lines:
                        if any(err in line for err in ['Error:', 'Exception:', 'AssertionError']):
                            error_type = line.split(':')[0].strip()
                            error_message = ':'.join(line.split(':')[1:]).strip()
                            break
                    
                    failure = self._create_test_failure(
                        test_name, '', error_type, error_message, section
                    )
                    failures.append(failure)
        
        return failures
    
    def _create_test_failure(self, test_name: str, module: str, error_type: str, 
                           error_message: str, full_traceback: str) -> TestFailure:
        """Create a TestFailure object with analysis metadata."""
        
        # Determine pattern category
        pattern_category = self._categorize_failure(error_type, error_message, full_traceback)
        
        # Determine component
        component = self._identify_component(test_name, module)
        
        # Calculate priority and effort
        priority = self._calculate_priority(pattern_category, error_type)
        effort = self._estimate_effort(pattern_category, error_type)
        severity = self._determine_severity(error_type, pattern_category)
        
        return TestFailure(
            test_name=test_name,
            module=module,
            error_type=error_type,
            error_message=error_message,
            full_traceback=full_traceback,
            pattern_category=pattern_category,
            severity=severity,
            fix_priority=priority,
            estimated_effort=effort,
            related_component=component
        )
    
    def _categorize_failure(self, error_type: str, error_message: str, traceback: str) -> str:
        """Categorize failure based on error patterns."""
        combined_text = f"{error_type} {error_message} {traceback}"
        
        for pattern_id, pattern in self.failure_patterns.items():
            if re.search(pattern.regex_pattern, combined_text, re.IGNORECASE):
                return pattern.category
                
        return 'unknown'
    
    def _identify_component(self, test_name: str, module: str) -> str:
        """Identify which system component the test belongs to."""
        test_path = f"{test_name} {module}".lower()
        
        for component_key, component_name in self.component_mapping.items():
            if component_key in test_path:
                return component_name
                
        return 'Other'
    
    def _calculate_priority(self, category: str, error_type: str) -> int:
        """Calculate fix priority (1-10, higher = more urgent)."""
        priority_map = {
            'import_issues': 9,
            'interface_mismatch': 8,
            'configuration_issues': 8,
            'type_mismatch': 7,
            'mock_issues': 7,
            'incomplete_implementation': 6,
            'logic_errors': 5,
            'unknown': 3
        }
        return priority_map.get(category, 5)
    
    def _estimate_effort(self, category: str, error_type: str) -> str:
        """Estimate fix effort level."""
        if category in ['import_issues', 'configuration_issues']:
            return 'LOW'
        elif category in ['interface_mismatch', 'type_mismatch']:
            return 'MEDIUM'
        elif category in ['incomplete_implementation', 'logic_errors']:
            return 'HIGH'
        else:
            return 'MEDIUM'
    
    def _determine_severity(self, error_type: str, category: str) -> str:
        """Determine severity level of the failure."""
        if category in ['import_issues', 'interface_mismatch']:
            return 'CRITICAL'
        elif category in ['type_mismatch', 'configuration_issues']:
            return 'HIGH'
        elif category in ['mock_issues', 'incomplete_implementation']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_failure_patterns(self, failures: List[TestFailure]) -> List[FailurePattern]:
        """Analyze failures to identify patterns and frequency."""
        
        # Reset pattern frequencies
        for pattern in self.failure_patterns.values():
            pattern.frequency = 0
            pattern.test_names = []
        
        # Count pattern occurrences
        for failure in failures:
            for pattern_id, pattern in self.failure_patterns.items():
                combined_text = f"{failure.error_type} {failure.error_message} {failure.full_traceback}"
                if re.search(pattern.regex_pattern, combined_text, re.IGNORECASE):
                    pattern.frequency += 1
                    pattern.test_names.append(failure.test_name)
        
        # Return patterns sorted by frequency
        sorted_patterns = sorted(
            self.failure_patterns.values(),
            key=lambda p: p.frequency,
            reverse=True
        )
        
        return [p for p in sorted_patterns if p.frequency > 0]
    
    def generate_analysis_report(self, failures: List[TestFailure], 
                               stats: Dict[str, Any]) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        
        patterns = self.analyze_failure_patterns(failures)
        
        # Count failures by category and component
        failures_by_category = Counter(f.pattern_category for f in failures)
        failures_by_component = Counter(f.related_component for f in failures)
        
        # Identify high-priority fixes and systemic issues
        high_priority_fixes = [
            f"Fix {pattern.description} ({pattern.frequency} tests affected)"
            for pattern in patterns[:5]  # Top 5 patterns
            if pattern.frequency >= 3
        ]
        
        systemic_issues = [
            pattern.suggested_fix
            for pattern in patterns
            if pattern.frequency >= 5  # Issues affecting 5+ tests
        ]
        
        return AnalysisReport(
            timestamp=datetime.now().isoformat(),
            total_tests=stats.get('total_tests', len(failures)),
            passing_tests=stats.get('passing_tests', 0),
            failing_tests=len(failures),
            success_rate=stats.get('success_rate', 0.0),
            failure_patterns=patterns,
            failures_by_category=dict(failures_by_category),
            failures_by_component=dict(failures_by_component),
            high_priority_fixes=high_priority_fixes,
            systemic_issues=systemic_issues
        )


def generate_json_report(report: AnalysisReport, output_path: Path):
    """Generate JSON format report."""
    report_dict = asdict(report)
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)
    
    print(f"📄 JSON report saved to: {output_path}")


def generate_html_report(report: AnalysisReport, output_path: Path):
    """Generate HTML format report with visualization."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Failure Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .pattern {{ background: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .high-priority {{ border-left: 4px solid #ff4444; }}
        .medium-priority {{ border-left: 4px solid #ffaa44; }}
        .low-priority {{ border-left: 4px solid #44ff44; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Failure Analysis Report</h1>
        <p>Generated: {report.timestamp}</p>
        <p>Total Tests: {report.total_tests} | Passing: {report.passing_tests} | Failing: {report.failing_tests}</p>
        <p>Success Rate: {report.success_rate:.1%}</p>
    </div>
    
    <div class="section">
        <h2>High Priority Fixes</h2>
        <ul>
"""
    
    for fix in report.high_priority_fixes:
        html_content += f"<li>{fix}</li>\n"
    
    html_content += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Failure Patterns by Frequency</h2>
"""
    
    for pattern in report.failure_patterns[:10]:  # Top 10 patterns
        priority_class = 'high-priority' if pattern.frequency >= 10 else 'medium-priority' if pattern.frequency >= 5 else 'low-priority'
        html_content += f"""
        <div class="pattern {priority_class}">
            <h3>{pattern.description} ({pattern.frequency} occurrences)</h3>
            <p><strong>Category:</strong> {pattern.category}</p>
            <p><strong>Suggested Fix:</strong> {pattern.suggested_fix}</p>
            <p><strong>Impact Score:</strong> {pattern.impact_score}/10</p>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="section">
        <h2>Failures by Component</h2>
        <table>
            <tr><th>Component</th><th>Failure Count</th></tr>
"""
    
    for component, count in sorted(report.failures_by_component.items(), key=lambda x: x[1], reverse=True):
        html_content += f"<tr><td>{component}</td><td>{count}</td></tr>\n"
    
    html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Systemic Issues Requiring Attention</h2>
        <ul>
"""
    
    for issue in report.systemic_issues:
        html_content += f"<li>{issue}</li>\n"
    
    html_content += """
        </ul>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"📊 HTML report saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Analyze test failures for systematic patterns')
    parser.add_argument('--output', '-o', type=str, default='analysis_reports', 
                       help='Output directory for reports (default: analysis_reports)')
    parser.add_argument('--format', '-f', choices=['json', 'html', 'both'], default='both',
                       help='Report format (default: both)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("🔬 Starting comprehensive test failure analysis...")
    
    # Initialize analyzer
    analyzer = FailureAnalyzer()
    
    # Run tests and capture failures
    failures, stats = analyzer.run_tests_and_capture_failures()
    
    if not failures and stats.get('error'):
        print(f"❌ Failed to analyze tests: {stats['error']}")
        return 1
    
    print(f"📊 Captured {len(failures)} test failures for analysis")
    
    # Generate analysis report
    report = analyzer.generate_analysis_report(failures, stats)
    
    # Generate output reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.format in ['json', 'both']:
        json_path = output_dir / f'test_analysis_{timestamp}.json'
        generate_json_report(report, json_path)
    
    if args.format in ['html', 'both']:
        html_path = output_dir / f'test_analysis_{timestamp}.html'
        generate_html_report(report, html_path)
    
    # Print summary
    print(f"\n🎯 ANALYSIS SUMMARY:")
    print(f"   • Total tests: {report.total_tests}")
    print(f"   • Success rate: {report.success_rate:.1%}")
    print(f"   • Unique failure patterns: {len(report.failure_patterns)}")
    print(f"   • High-priority fixes identified: {len(report.high_priority_fixes)}")
    print(f"   • Systemic issues found: {len(report.systemic_issues)}")
    
    if report.failure_patterns:
        top_pattern = report.failure_patterns[0]
        print(f"\n🔥 TOP ISSUE: {top_pattern.description}")
        print(f"   • Affects {top_pattern.frequency} tests")
        print(f"   • Suggested fix: {top_pattern.suggested_fix}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 