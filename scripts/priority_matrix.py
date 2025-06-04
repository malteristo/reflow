#!/usr/bin/env python3
"""
Test Priority Matrix with ROI Calculation

This script provides systematic prioritization of test failures based on impact,
effort, and ROI analysis to maximize test recovery efficiency.

Usage:
    python scripts/priority_matrix.py --analyze-failures
    python scripts/priority_matrix.py --component core --show-priorities
    python scripts/priority_matrix.py --export-report priority_analysis.json
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Union
import re


@dataclass
class TestFailureInfo:
    """Information about a failing test."""
    test_name: str
    component: str
    failure_pattern: str
    failure_category: str
    error_message: str
    test_file: str
    dependencies: List[str] = None


@dataclass
class FixImpactAnalysis:
    """Analysis of potential impact from fixing a test or group of tests."""
    direct_tests_fixed: int
    dependent_tests_unlocked: int
    component_health_improvement: float
    cascade_potential: int
    regression_risk: float


@dataclass
class FixEffortEstimate:
    """Estimate of effort required to fix a test or group of tests."""
    complexity_score: int  # 1-10 scale
    estimated_hours: float
    dependency_complexity: int
    code_change_scope: str  # 'minimal', 'moderate', 'extensive'
    risk_level: str  # 'low', 'medium', 'high'


@dataclass
class TestPriority:
    """Priority analysis for a test or group of tests."""
    test_name: str
    component: str
    failure_pattern: str
    impact_analysis: FixImpactAnalysis
    effort_estimate: FixEffortEstimate
    roi_score: float
    priority_rank: int
    recommended_action: str
    fix_suggestions: List[str]


class TestPriorityMatrix:
    """Main class for analyzing test failures and calculating fix priorities."""
    
    def __init__(self, test_analysis_path: str = "scripts/test_analysis_results.json"):
        self.test_analysis_path = Path(test_analysis_path)
        self.test_analysis_data = self._load_test_analysis()
        self.component_dependencies = self._load_component_dependencies()
        self.test_dependencies = self._analyze_test_dependencies()
        
    def _load_test_analysis(self) -> Dict[str, Any]:
        """Load existing test analysis results."""
        if not self.test_analysis_path.exists():
            raise FileNotFoundError(f"Test analysis file not found: {self.test_analysis_path}")
        
        with open(self.test_analysis_path, 'r') as f:
            return json.load(f)
    
    def _load_component_dependencies(self) -> Dict[str, List[str]]:
        """Load component dependency information."""
        # Component dependency mapping based on Research Agent architecture
        return {
            'models': [],
            'utils': [],
            'config': ['utils'],
            'vector_store': ['models', 'config'],
            'embedding': ['config', 'models'],
            'document_processor': ['models', 'config'],
            'document_insertion': ['document_processor', 'vector_store', 'embedding'],
            'core': ['vector_store', 'embedding', 'document_processor'],
            'cli': ['core', 'vector_store', 'document_processor'],
            'cli_unit': ['cli', 'config'],
            'integration': ['core', 'cli', 'vector_store'],
            'performance': ['core', 'vector_store'],
            'rag_pipeline': ['core', 'vector_store', 'embedding', 'document_processor']
        }
    
    def _analyze_test_dependencies(self) -> Dict[str, List[str]]:
        """Analyze test dependencies based on component structure."""
        test_deps = {}
        
        # For each test, determine its dependencies based on component and pattern
        for pattern in self.test_analysis_data.get('failure_patterns', []):
            for test_name in pattern.get('test_names', []):
                component = self._determine_test_component(test_name)
                test_deps[test_name] = self.component_dependencies.get(component, [])
        
        return test_deps
    
    def _determine_test_component(self, test_name: str) -> str:
        """Determine which component a test belongs to based on its name and patterns."""
        # Component classification rules based on test naming patterns
        component_patterns = {
            'cli': r'test_(cli_|command_|commands_)',
            'vector_store': r'test_(chroma|vector_store|collection)',
            'embedding': r'test_(embed|embedding)',
            'document_processor': r'test_(chunk|document_processor|atomic_unit)',
            'document_insertion': r'test_(insert|insertion|ingest)',
            'config': r'test_(config|configuration)',
            'models': r'test_(schema|model|pydantic)',
            'utils': r'test_(util|helper)',
            'core': r'test_(search|query|rag|rerank)',
            'integration': r'test_(integration|end_to_end|workflow)',
            'performance': r'test_(performance|stress|load|memory)',
            'rag_pipeline': r'test_(rag_pipeline|pipeline)'
        }
        
        for component, pattern in component_patterns.items():
            if re.search(pattern, test_name, re.IGNORECASE):
                return component
        
        # Default to 'other' if no pattern matches
        return 'other'
    
    def calculate_impact_analysis(self, failure_pattern: str, test_names: List[str]) -> FixImpactAnalysis:
        """Calculate the potential impact of fixing a failure pattern."""
        direct_tests = len(test_names)
        
        # Analyze potential dependent tests that could be unlocked
        dependent_tests = 0
        cascade_potential = 0
        
        for test_name in test_names:
            component = self._determine_test_component(test_name)
            
            # Count tests in dependent components
            for comp, deps in self.component_dependencies.items():
                if component in deps:
                    component_tests = self._count_component_tests(comp)
                    dependent_tests += int(component_tests * 0.1)  # 10% likely to be unlocked
            
            # Calculate cascade potential based on component importance
            cascade_potential += self._get_component_cascade_score(component)
        
        # Component health improvement based on component size and criticality
        avg_component_health_improvement = 0
        for test_name in test_names:
            component = self._determine_test_component(test_name)
            component_total_tests = self._count_component_tests(component)
            if component_total_tests > 0:
                avg_component_health_improvement += (1 / component_total_tests) * 100
        
        avg_component_health_improvement /= max(len(test_names), 1)
        
        # Regression risk assessment
        regression_risk = self._assess_regression_risk(failure_pattern, test_names)
        
        return FixImpactAnalysis(
            direct_tests_fixed=direct_tests,
            dependent_tests_unlocked=dependent_tests,
            component_health_improvement=avg_component_health_improvement,
            cascade_potential=cascade_potential,
            regression_risk=regression_risk
        )
    
    def calculate_effort_estimate(self, failure_pattern: str, test_names: List[str]) -> FixEffortEstimate:
        """Estimate the effort required to fix a failure pattern."""
        
        # Complexity scoring based on failure pattern type
        complexity_scores = {
            'assertion_failure': 6,  # Medium - may need logic changes
            'configuration_error': 4,  # Lower - config fixes
            'mock_error': 3,  # Lower - mock setup fixes
            'missing_method': 7,  # Higher - interface changes
            'not_implemented': 8,  # Higher - new implementation
            'type_error': 5  # Medium - type fixes
        }
        
        base_complexity = complexity_scores.get(failure_pattern, 5)
        
        # Adjust complexity based on number of tests affected
        test_count_multiplier = min(1.0 + (len(test_names) - 1) * 0.1, 2.0)
        complexity_score = int(base_complexity * test_count_multiplier)
        
        # Estimated hours based on complexity and test count
        base_hours = {
            'assertion_failure': 0.5,  # 30 min per test
            'configuration_error': 0.25,  # 15 min per test  
            'mock_error': 0.2,  # 12 min per test
            'missing_method': 1.0,  # 1 hour per test
            'not_implemented': 2.0,  # 2 hours per test
            'type_error': 0.3  # 18 min per test
        }.get(failure_pattern, 0.5)
        
        estimated_hours = base_hours * len(test_names)
        
        # Dependency complexity
        dep_complexity = 0
        for test_name in test_names:
            test_deps = self.test_dependencies.get(test_name, [])
            dep_complexity += len(test_deps)
        
        # Code change scope assessment
        if len(test_names) <= 5 and complexity_score <= 4:
            code_change_scope = 'minimal'
        elif len(test_names) <= 15 and complexity_score <= 7:
            code_change_scope = 'moderate'
        else:
            code_change_scope = 'extensive'
        
        # Risk level assessment
        if complexity_score <= 3 and dep_complexity <= 5:
            risk_level = 'low'
        elif complexity_score <= 6 and dep_complexity <= 10:
            risk_level = 'medium'
        else:
            risk_level = 'high'
        
        return FixEffortEstimate(
            complexity_score=complexity_score,
            estimated_hours=estimated_hours,
            dependency_complexity=dep_complexity,
            code_change_scope=code_change_scope,
            risk_level=risk_level
        )
    
    def calculate_roi_score(self, impact: FixImpactAnalysis, effort: FixEffortEstimate) -> float:
        """Calculate ROI score for prioritizing fixes."""
        
        # Impact score calculation (0-100)
        impact_score = (
            impact.direct_tests_fixed * 3 +  # Direct value
            impact.dependent_tests_unlocked * 1.5 +  # Cascade value
            impact.component_health_improvement * 0.5 +  # Health value
            impact.cascade_potential * 2  # Strategic value
        )
        
        # Effort penalty (higher effort = lower score)
        effort_penalty = (
            effort.complexity_score * 2 +
            effort.estimated_hours * 1 +
            effort.dependency_complexity * 0.5
        )
        
        # Risk adjustment
        risk_multiplier = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6
        }.get(effort.risk_level, 0.8)
        
        # ROI calculation with regression risk consideration
        roi_score = (impact_score / max(effort_penalty, 1)) * risk_multiplier
        roi_score *= (1 - impact.regression_risk)  # Reduce score for high regression risk
        
        return round(roi_score, 2)
    
    def generate_fix_suggestions(self, failure_pattern: str, test_names: List[str]) -> List[str]:
        """Generate specific actionable fix suggestions."""
        
        suggestions = []
        
        if failure_pattern == 'assertion_failure':
            suggestions.extend([
                "Review test expectations vs actual implementation behavior",
                "Check if tests expect outdated functionality",
                "Verify assertion logic and update if implementation is correct",
                "Consider batch updating tests with similar assertion patterns"
            ])
        
        elif failure_pattern == 'configuration_error':
            suggestions.extend([
                "Standardize test configuration setup patterns",
                "Create configuration fixtures for consistent test setup",
                "Review environment variable loading in tests",
                "Check for missing test configuration files"
            ])
        
        elif failure_pattern == 'mock_error':
            suggestions.extend([
                "Standardize mock setup patterns across test suite",
                "Create reusable mock fixtures for common dependencies",
                "Review mock configuration for external services",
                "Update mock patches to match current interface signatures"
            ])
        
        elif failure_pattern == 'missing_method':
            suggestions.extend([
                "Add missing method implementations to satisfy interface contracts",
                "Review interface compatibility between components",
                "Update method signatures to match current usage patterns",
                "Consider creating abstract base classes for clear interfaces"
            ])
        
        elif failure_pattern == 'not_implemented':
            suggestions.extend([
                "Identify if NotImplementedError is for performance tests only",
                "Implement missing functionality or mark tests as skip if not needed",
                "Prioritize implementation based on feature importance",
                "Consider creating placeholder implementations with proper behavior"
            ])
        
        elif failure_pattern == 'type_error':
            suggestions.extend([
                "Update method signatures to match current usage",
                "Add proper type annotations throughout codebase",
                "Review parameter passing conventions",
                "Check for outdated function call patterns in tests"
            ])
        
        # Add component-specific suggestions
        components = set(self._determine_test_component(test) for test in test_names)
        for component in components:
            if component == 'vector_store':
                suggestions.append("Check ChromaDB integration and metadata handling")
            elif component == 'embedding':
                suggestions.append("Verify embedding service configuration and API compatibility")
            elif component == 'cli':
                suggestions.append("Review CLI command implementations and argument parsing")
            elif component == 'config':
                suggestions.append("Check configuration loading and validation logic")
        
        return suggestions
    
    def _count_component_tests(self, component: str) -> int:
        """Count total tests in a component."""
        failures_by_component = self.test_analysis_data.get('failures_by_component', {})
        
        # Map component names to analysis data keys
        component_mapping = {
            'cli': 'Command Line Interface',
            'vector_store': 'Vector Storage System',
            'core': 'RAG Query Engine',
            'config': 'Configuration System',
            'embedding': 'Embedding Service',
            'performance': 'Performance Testing',
            'integration': 'Integration Pipeline',
            'document_processor': 'Document Processing',
            'collection': 'Collection Management'
        }
        
        mapped_component = component_mapping.get(component, component)
        return failures_by_component.get(mapped_component, 0)
    
    def _get_component_cascade_score(self, component: str) -> int:
        """Get cascade importance score for a component."""
        cascade_scores = {
            'models': 10,  # Foundational
            'utils': 8,    # Widely used
            'config': 9,   # Critical infrastructure
            'vector_store': 8,  # Core functionality
            'embedding': 7,     # Important but isolated
            'document_processor': 6,  # Specialized
            'core': 9,     # Central to RAG
            'cli': 5,      # User interface
            'integration': 4,   # Test-specific
            'performance': 3    # Test-specific
        }
        return cascade_scores.get(component, 5)
    
    def _assess_regression_risk(self, failure_pattern: str, test_names: List[str]) -> float:
        """Assess regression risk of fixing this pattern."""
        
        # Base risk by pattern type
        pattern_risks = {
            'assertion_failure': 0.3,  # Medium risk - logic changes
            'configuration_error': 0.1,  # Low risk - config changes
            'mock_error': 0.05,  # Very low risk - test setup
            'missing_method': 0.4,  # Higher risk - interface changes
            'not_implemented': 0.5,  # Highest risk - new code
            'type_error': 0.2   # Low-medium risk - signature changes
        }
        
        base_risk = pattern_risks.get(failure_pattern, 0.3)
        
        # Adjust based on test count (more tests = higher regression risk)
        test_count_factor = min(len(test_names) * 0.02, 0.3)
        
        # Adjust based on component criticality
        critical_components = {'core', 'vector_store', 'config', 'models'}
        component_risk = 0.1 if any(
            self._determine_test_component(test) in critical_components 
            for test in test_names
        ) else 0.0
        
        return min(base_risk + test_count_factor + component_risk, 0.8)
    
    def analyze_all_priorities(self) -> List[TestPriority]:
        """Analyze priorities for all failure patterns."""
        priorities = []
        
        for pattern in self.test_analysis_data.get('failure_patterns', []):
            pattern_id = pattern['pattern_id']
            test_names = pattern['test_names']
            
            impact = self.calculate_impact_analysis(pattern_id, test_names)
            effort = self.calculate_effort_estimate(pattern_id, test_names)
            roi_score = self.calculate_roi_score(impact, effort)
            suggestions = self.generate_fix_suggestions(pattern_id, test_names)
            
            # Determine recommended action based on ROI and risk
            if roi_score >= 8.0 and effort.risk_level in ['low', 'medium']:
                recommended_action = "High Priority - Fix Immediately"
            elif roi_score >= 5.0:
                recommended_action = "Medium Priority - Fix Soon"
            elif roi_score >= 2.0:
                recommended_action = "Low Priority - Fix When Time Permits"
            else:
                recommended_action = "Consider Deferring - High Effort, Low Impact"
            
            priority = TestPriority(
                test_name=f"{pattern_id} ({len(test_names)} tests)",
                component="Multiple" if len(set(self._determine_test_component(t) for t in test_names)) > 1 
                         else self._determine_test_component(test_names[0]),
                failure_pattern=pattern_id,
                impact_analysis=impact,
                effort_estimate=effort,
                roi_score=roi_score,
                priority_rank=0,  # Will be set after sorting
                recommended_action=recommended_action,
                fix_suggestions=suggestions
            )
            
            priorities.append(priority)
        
        # Sort by ROI score (descending) and assign ranks
        priorities.sort(key=lambda p: p.roi_score, reverse=True)
        for i, priority in enumerate(priorities, 1):
            priority.priority_rank = i
        
        return priorities
    
    def get_component_priorities(self, component: str) -> List[TestPriority]:
        """Get priorities for a specific component."""
        all_priorities = self.analyze_all_priorities()
        return [p for p in all_priorities if component.lower() in p.component.lower()]
    
    def export_priority_report(self, output_path: str = "test_priority_analysis.json"):
        """Export comprehensive priority analysis report."""
        priorities = self.analyze_all_priorities()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_summary': {
                'total_failure_patterns': len(priorities),
                'high_priority_patterns': len([p for p in priorities if 'High Priority' in p.recommended_action]),
                'medium_priority_patterns': len([p for p in priorities if 'Medium Priority' in p.recommended_action]),
                'low_priority_patterns': len([p for p in priorities if 'Low Priority' in p.recommended_action]),
                'avg_roi_score': sum(p.roi_score for p in priorities) / len(priorities) if priorities else 0
            },
            'prioritized_fixes': [asdict(p) for p in priorities],
            'quick_wins': [asdict(p) for p in priorities if p.roi_score >= 8.0 and p.effort_estimate.complexity_score <= 4],
            'strategic_fixes': [asdict(p) for p in priorities if p.impact_analysis.cascade_potential >= 3],
            'component_breakdown': self._generate_component_breakdown(priorities)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Priority analysis report exported to: {output_path}")
        return report
    
    def _generate_component_breakdown(self, priorities: List[TestPriority]) -> Dict[str, Any]:
        """Generate component-wise breakdown of priorities."""
        breakdown = defaultdict(lambda: {
            'patterns': [],
            'total_roi': 0,
            'avg_roi': 0,
            'total_tests_affected': 0,
            'recommended_order': []
        })
        
        for priority in priorities:
            component = priority.component
            breakdown[component]['patterns'].append(priority.failure_pattern)
            breakdown[component]['total_roi'] += priority.roi_score
            breakdown[component]['total_tests_affected'] += priority.impact_analysis.direct_tests_fixed
        
        # Calculate averages and sort recommendations
        for component, data in breakdown.items():
            if data['patterns']:
                data['avg_roi'] = data['total_roi'] / len(data['patterns'])
                
                # Get component priorities sorted by ROI
                component_priorities = [p for p in priorities if p.component == component]
                component_priorities.sort(key=lambda p: p.roi_score, reverse=True)
                data['recommended_order'] = [
                    {
                        'pattern': p.failure_pattern,
                        'roi_score': p.roi_score,
                        'action': p.recommended_action
                    }
                    for p in component_priorities
                ]
        
        return dict(breakdown)
    
    def print_priority_summary(self, top_n: int = 10):
        """Print a summary of top priority fixes."""
        priorities = self.analyze_all_priorities()
        
        print(f"\n{'='*80}")
        print(f"TEST FAILURE PRIORITY ANALYSIS - TOP {top_n} FIXES")
        print(f"{'='*80}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Failure Patterns: {len(priorities)}")
        print(f"Total Tests Affected: {sum(p.impact_analysis.direct_tests_fixed for p in priorities)}")
        
        for i, priority in enumerate(priorities[:top_n], 1):
            print(f"\n{'-'*60}")
            print(f"RANK #{i}: {priority.test_name}")
            print(f"Component: {priority.component}")
            print(f"ROI Score: {priority.roi_score}")
            print(f"Action: {priority.recommended_action}")
            print(f"Impact: {priority.impact_analysis.direct_tests_fixed} tests, "
                  f"{priority.impact_analysis.dependent_tests_unlocked} dependent unlocks")
            print(f"Effort: {priority.effort_estimate.estimated_hours:.1f} hours, "
                  f"{priority.effort_estimate.risk_level} risk")
            print(f"Top Suggestion: {priority.fix_suggestions[0] if priority.fix_suggestions else 'None'}")


def main():
    """Main CLI interface for priority matrix analysis."""
    parser = argparse.ArgumentParser(description='Test Priority Matrix with ROI Calculation')
    parser.add_argument('--analyze-failures', action='store_true',
                       help='Analyze all test failures and show priority matrix')
    parser.add_argument('--component', type=str,
                       help='Show priorities for specific component')
    parser.add_argument('--export-report', type=str,
                       help='Export detailed priority report to JSON file')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top priorities to show (default: 10)')
    parser.add_argument('--test-analysis-path', type=str,
                       default='scripts/test_analysis_results.json',
                       help='Path to test analysis results file')
    
    args = parser.parse_args()
    
    try:
        matrix = TestPriorityMatrix(args.test_analysis_path)
        
        if args.export_report:
            matrix.export_priority_report(args.export_report)
        
        if args.component:
            priorities = matrix.get_component_priorities(args.component)
            print(f"\nPRIORITIES FOR COMPONENT: {args.component.upper()}")
            print(f"{'='*50}")
            for priority in priorities:
                print(f"\nPattern: {priority.failure_pattern}")
                print(f"ROI Score: {priority.roi_score}")
                print(f"Action: {priority.recommended_action}")
                print(f"Tests Affected: {priority.impact_analysis.direct_tests_fixed}")
        
        if args.analyze_failures or not any([args.component, args.export_report]):
            matrix.print_priority_summary(args.top_n)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 