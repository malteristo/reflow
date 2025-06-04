#!/usr/bin/env python3
"""
Test Status Dashboard for Research Agent Test Suite.

Real-time monitoring dashboard for test execution, failure analysis, and progress
tracking toward 95% test success rate goal. Integrates with test analysis tools
to provide comprehensive test suite health monitoring.

Usage:
    python scripts/test_dashboard.py                    # Static summary
    python scripts/test_dashboard.py --live             # Live updating dashboard  
    python scripts/test_dashboard.py --analysis         # With failure analysis
    python scripts/test_dashboard.py --components       # Component health view
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from rich.columns import Columns
from rich.tree import Tree
from rich.rule import Rule

# Import test analysis tools
try:
    # Import classes directly from the script files
    sys.path.insert(0, str(project_root / "scripts"))
    from test_analyzer import FailureAnalyzer, AnalysisReport, TestFailure
    from component_runner import ComponentTestRunner, ComponentInfo, TestExecutionResult
    from priority_matrix import TestPriorityMatrix, TestPriority
    
    # Flag that imports were successful
    ANALYSIS_TOOLS_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️  Test analysis tools not fully available: {e}")
    print("    Some dashboard features may be limited.")
    FailureAnalyzer = ComponentTestRunner = TestPriorityMatrix = None
    ANALYSIS_TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TestStats:
    """Test statistics and metrics."""
    total_tests: int = 0
    passing_tests: int = 0
    failing_tests: int = 0
    success_rate: float = 0.0
    execution_time: float = 0.0
    coverage_percentage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def tests_to_95_percent(self) -> int:
        """Calculate tests needed to reach 95% success rate."""
        if self.total_tests == 0:
            return 0
        target_passing = int(self.total_tests * 0.95)
        return max(0, target_passing - self.passing_tests)


@dataclass
class ComponentHealth:
    """Health metrics for a test component."""
    name: str
    total_tests: int
    passing_tests: int
    failing_tests: int
    success_rate: float
    critical_failures: int = 0
    last_run: Optional[datetime] = None
    
    @property
    def health_status(self) -> str:
        """Determine health status based on success rate."""
        if self.success_rate >= 0.95:
            return "excellent"
        elif self.success_rate >= 0.90:
            return "good"
        elif self.success_rate >= 0.80:
            return "warning"
        else:
            return "critical"


@dataclass
class FailureBreakdown:
    """Breakdown of failure types and counts."""
    assertion_failures: int = 0
    configuration_errors: int = 0
    missing_method_errors: int = 0
    type_errors: int = 0
    mock_errors: int = 0
    not_implemented_errors: int = 0
    other_errors: int = 0
    
    @property
    def total_failures(self) -> int:
        """Total failure count."""
        return (self.assertion_failures + self.configuration_errors + 
                self.missing_method_errors + self.type_errors + 
                self.mock_errors + self.not_implemented_errors + 
                self.other_errors)


class TestDashboard:
    """
    Real-time test status dashboard with failure analysis integration.
    """
    
    def __init__(self, console: Optional[Console] = None):
        """Initialize the test dashboard."""
        self.console = console or Console()
        self.stats = TestStats()
        self.component_health: Dict[str, ComponentHealth] = {}
        self.failure_breakdown = FailureBreakdown()
        self.recent_runs: List[Dict[str, Any]] = []
        self.is_live = False
        
        # Integration with analysis tools - only create instances if available
        if ANALYSIS_TOOLS_AVAILABLE:
            try:
                self.analyzer = FailureAnalyzer()
                self.component_runner = ComponentTestRunner()
                self.priority_matrix = TestPriorityMatrix() if Path("scripts/test_analysis_results.json").exists() else None
                logger.info("Test dashboard initialized with full analysis tools")
            except Exception as e:
                logger.warning(f"Failed to initialize some analysis tools: {e}")
                self.analyzer = None
                self.component_runner = None
                self.priority_matrix = None
        else:
            self.analyzer = None
            self.component_runner = None
            self.priority_matrix = None
            logger.info("Test dashboard initialized with limited features")
        
        logger.info("Test dashboard initialized")
    
    def collect_test_metrics(self) -> TestStats:
        """Collect current test metrics from pytest runs."""
        try:
            # Run pytest with coverage to get comprehensive metrics
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "--tb=no", "--quiet", "--no-header",
                "--cov=src", "--cov-report=json",
                "src/research_agent_backend/tests/"
            ], 
            capture_output=True, text=True, cwd=project_root, timeout=300)
            
            # Parse pytest output for test counts
            lines = result.stdout.strip().split('\n')
            stats = TestStats()
            
            for line in lines:
                if " passed" in line or " failed" in line:
                    # Parse pytest summary line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            stats.passing_tests = int(parts[i-1])
                        elif part == "failed":
                            stats.failing_tests = int(parts[i-1])
            
            stats.total_tests = stats.passing_tests + stats.failing_tests
            if stats.total_tests > 0:
                stats.success_rate = (stats.passing_tests / stats.total_tests) * 100
            
            # Try to read coverage data
            coverage_file = project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                        stats.coverage_percentage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                except Exception as e:
                    logger.warning(f"Failed to read coverage data: {e}")
            
            stats.last_updated = datetime.now()
            return stats
            
        except subprocess.TimeoutExpired:
            logger.error("Test collection timed out")
            return TestStats()
        except Exception as e:
            logger.error(f"Failed to collect test metrics: {e}")
            return TestStats()
    
    def collect_component_health(self) -> Dict[str, ComponentHealth]:
        """Collect health metrics for each test component."""
        if not ANALYSIS_TOOLS_AVAILABLE or not self.component_runner:
            return {}
        
        try:
            # Get component list
            components = self.component_runner.list_components()
            health_data = {}
            
            for component in components:
                component_name = component.name
                
                # Use the component test count from discovery
                total_tests = component.test_count
                
                # Run quick test for this component to get current status
                try:
                    result = self.component_runner.run_component(
                        component_name, isolated=False, verbose=False, timeout=30
                    )
                    
                    health_data[component_name] = ComponentHealth(
                        name=component_name,
                        total_tests=result.total_tests,
                        passing_tests=result.passed_tests,
                        failing_tests=result.failed_tests,
                        success_rate=(result.passed_tests / result.total_tests * 100) if result.total_tests > 0 else 0,
                        last_run=datetime.now()
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to test component {component_name}: {e}")
                    # Create health data based on discovery info
                    health_data[component_name] = ComponentHealth(
                        name=component_name,
                        total_tests=total_tests,
                        passing_tests=0,
                        failing_tests=total_tests,
                        success_rate=0.0,
                        last_run=None
                    )
            
            return health_data
            
        except Exception as e:
            logger.error(f"Failed to collect component health: {e}")
            return {}
    
    def collect_failure_breakdown(self) -> FailureBreakdown:
        """Collect breakdown of failure types using analyzer."""
        if not ANALYSIS_TOOLS_AVAILABLE or not self.analyzer:
            return FailureBreakdown()
        
        try:
            # Run test analyzer to get failure data
            failures, stats = self.analyzer.run_tests_and_capture_failures()
            
            if not failures:
                return FailureBreakdown()
            
            breakdown = FailureBreakdown()
            
            # Count failure types based on pattern categories
            for failure in failures:
                category = failure.pattern_category
                
                if category == "logic_errors":
                    breakdown.assertion_failures += 1
                elif category == "configuration_issues":
                    breakdown.configuration_errors += 1
                elif category == "interface_mismatch":
                    breakdown.missing_method_errors += 1
                elif category == "type_mismatch":
                    breakdown.type_errors += 1
                elif category == "mock_issues":
                    breakdown.mock_errors += 1
                elif category == "incomplete_implementation":
                    breakdown.not_implemented_errors += 1
                else:
                    breakdown.other_errors += 1
            
            return breakdown
            
        except Exception as e:
            logger.error(f"Failed to collect failure breakdown: {e}")
            return FailureBreakdown()
    
    def update_all_metrics(self):
        """Update all dashboard metrics."""
        self.stats = self.collect_test_metrics()
        self.component_health = self.collect_component_health()
        self.failure_breakdown = self.collect_failure_breakdown()
        
        # Add to recent runs history
        run_data = {
            "timestamp": datetime.now(),
            "success_rate": self.stats.success_rate,
            "total_tests": self.stats.total_tests,
            "passing_tests": self.stats.passing_tests,
            "failing_tests": self.stats.failing_tests
        }
        self.recent_runs.append(run_data)
        
        # Keep only last 20 runs
        if len(self.recent_runs) > 20:
            self.recent_runs = self.recent_runs[-20:]
    
    def render_header(self) -> Panel:
        """Render dashboard header with key metrics."""
        header_text = Text()
        header_text.append("🧪 Test Suite Dashboard", style="bold blue")
        
        # Success rate with color coding
        success_rate = self.stats.success_rate
        if success_rate >= 95.0:
            rate_style = "bold green"
            rate_icon = "✅"
        elif success_rate >= 90.0:
            rate_style = "bold yellow"
            rate_icon = "⚠️"
        else:
            rate_style = "bold red"
            rate_icon = "❌"
        
        header_text.append(f" | {rate_icon} {success_rate:.1f}% Success", style=rate_style)
        header_text.append(f" | {self.stats.passing_tests}/{self.stats.total_tests} Tests", style="dim")
        
        if self.stats.coverage_percentage > 0:
            header_text.append(f" | {self.stats.coverage_percentage:.1f}% Coverage", style="cyan")
        
        return Panel(
            Align.center(header_text),
            title="Research Agent Test Status",
            border_style="blue"
        )
    
    def render_progress_to_goal(self) -> Panel:
        """Render progress toward 95% success rate goal."""
        current_rate = self.stats.success_rate
        target_rate = 95.0
        
        # Progress calculation
        if current_rate >= target_rate:
            progress_text = Text("🎉 TARGET ACHIEVED!", style="bold green")
        else:
            tests_needed = self.stats.tests_to_95_percent
            progress_text = Text()
            progress_text.append(f"Need {tests_needed} more passing tests to reach 95%\n", style="yellow")
            
            # Progress bar
            progress_bar = Progress()
            task = progress_bar.add_task("Progress to 95%", total=100)
            progress_bar.update(task, completed=current_rate)
            
        # Recent trend
        if len(self.recent_runs) >= 2:
            recent_rate = self.recent_runs[-2]["success_rate"]
            trend = current_rate - recent_rate
            if trend > 0:
                trend_text = f"↗️ +{trend:.1f}% improvement"
                trend_style = "green"
            elif trend < 0:
                trend_text = f"↘️ {trend:.1f}% decline"
                trend_style = "red"
            else:
                trend_text = "➡️ No change"
                trend_style = "dim"
            
            progress_text.append(f"\nRecent trend: ")
            progress_text.append(trend_text, style=trend_style)
        
        return Panel(progress_text, title="Progress to 95% Goal", border_style="yellow")
    
    def render_failure_breakdown(self) -> Panel:
        """Render breakdown of failure types."""
        breakdown = self.failure_breakdown
        
        if breakdown.total_failures == 0:
            return Panel(
                Align.center(Text("No test failures! 🎉", style="bold green")),
                title="Failure Breakdown",
                border_style="green"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Failure Type", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("Priority", justify="center")
        
        failure_types = [
            ("Assertion Failures", breakdown.assertion_failures, "🔍"),
            ("Configuration Errors", breakdown.configuration_errors, "⚡"),
            ("Missing Methods", breakdown.missing_method_errors, "🔧"), 
            ("Type Errors", breakdown.type_errors, "📝"),
            ("Mock Errors", breakdown.mock_errors, "🎭"),
            ("Not Implemented", breakdown.not_implemented_errors, "⏳"),
            ("Other Errors", breakdown.other_errors, "❓")
        ]
        
        for name, count, icon in failure_types:
            if count > 0:
                percentage = (count / breakdown.total_failures) * 100
                
                # Priority based on count and type
                if name == "Configuration Errors" and count > 10:
                    priority = "🔥 HIGH"
                elif name == "Missing Methods" and count > 5:
                    priority = "⚠️ MEDIUM"
                elif name == "Not Implemented" and count > 20:
                    priority = "📋 LOW"
                else:
                    priority = "➖"
                
                table.add_row(
                    f"{icon} {name}",
                    str(count),
                    f"{percentage:.1f}%",
                    priority
                )
        
        return Panel(table, title=f"Failure Breakdown ({breakdown.total_failures} total)", border_style="red")
    
    def render_component_health(self) -> Panel:
        """Render component health overview."""
        if not self.component_health:
            return Panel(
                Text("Component health data not available", style="dim"),
                title="Component Health",
                border_style="dim"
            )
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Success Rate", justify="right")
        table.add_column("Tests", justify="right")
        table.add_column("Last Run", style="dim")
        
        # Sort by health status
        components = sorted(
            self.component_health.values(),
            key=lambda x: x.success_rate,
            reverse=True
        )
        
        for health in components:
            # Status icon and color
            status_mapping = {
                "excellent": ("🟢", "green"),
                "good": ("🟡", "yellow"),
                "warning": ("🟠", "orange"),
                "critical": ("🔴", "red")
            }
            icon, color = status_mapping.get(health.health_status, ("⚪", "dim"))
            
            # Format last run time
            if health.last_run:
                time_ago = datetime.now() - health.last_run
                if time_ago.total_seconds() < 60:
                    last_run = "Just now"
                elif time_ago.total_seconds() < 3600:
                    last_run = f"{int(time_ago.total_seconds() / 60)}m ago"
                else:
                    last_run = f"{int(time_ago.total_seconds() / 3600)}h ago"
            else:
                last_run = "Never"
            
            table.add_row(
                health.name.replace("_", " ").title(),
                Text(f"{icon} {health.health_status.title()}", style=color),
                f"{health.success_rate:.1f}%",
                f"{health.passing_tests}/{health.total_tests}",
                last_run
            )
        
        return Panel(table, title="Component Health", border_style="blue")
    
    def render_recommendations(self) -> Panel:
        """Render actionable recommendations based on analysis."""
        recommendations = []
        
        # Priority recommendations based on failure analysis
        breakdown = self.failure_breakdown
        
        if breakdown.configuration_errors > 0:
            recommendations.append("🔧 Fix configuration errors for quick test recovery")
        
        if breakdown.missing_method_errors > 0:
            recommendations.append("⚡ Implement missing methods for interface compliance")
        
        if breakdown.assertion_failures > breakdown.total_failures * 0.5:
            recommendations.append("🔍 Review assertion failures - may indicate outdated test expectations")
        
        if self.stats.success_rate < 90:
            recommendations.append("🎯 Focus on systematic fixes using test analysis tools")
        
        if breakdown.not_implemented_errors > 20:
            recommendations.append("📋 Batch implement NotImplementedError placeholders")
        
        # Component-specific recommendations
        critical_components = [
            comp for comp in self.component_health.values()
            if comp.health_status == "critical"
        ]
        
        if critical_components:
            comp_names = ", ".join(comp.name for comp in critical_components[:3])
            recommendations.append(f"🚨 Critical attention needed: {comp_names}")
        
        if not recommendations:
            recommendations.append("✨ Great job! Test suite is in excellent condition")
        
        rec_text = Text()
        for i, rec in enumerate(recommendations[:5], 1):
            rec_text.append(f"{i}. {rec}\n")
        
        return Panel(rec_text, title="Recommended Actions", border_style="green")
    
    def render_footer(self) -> Panel:
        """Render dashboard footer."""
        footer_text = Text()
        footer_text.append(f"Last updated: {self.stats.last_updated.strftime('%H:%M:%S')}", style="dim")
        footer_text.append(" | Goal: 95% success rate", style="dim") 
        if self.is_live:
            footer_text.append(" | Press Ctrl+C to exit", style="dim")
        
        return Panel(
            Align.center(footer_text),
            border_style="dim"
        )
    
    def create_layout(self) -> Layout:
        """Create dashboard layout."""
        layout = Layout()
        
        # Main split
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main into left and right
        layout["main"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1)
        )
        
        # Split left column
        layout["left"].split_column(
            Layout(name="progress", size=8),
            Layout(name="failures", ratio=1),
            Layout(name="components", ratio=1)
        )
        
        # Right column for recommendations
        layout["right"].update(self.render_recommendations())
        
        return layout
    
    def update_layout(self, layout: Layout):
        """Update layout with current data."""
        layout["header"].update(self.render_header())
        layout["progress"].update(self.render_progress_to_goal())
        layout["failures"].update(self.render_failure_breakdown())
        layout["components"].update(self.render_component_health())
        layout["right"].update(self.render_recommendations())
        layout["footer"].update(self.render_footer())
    
    def show_static_summary(self):
        """Show static dashboard summary."""
        self.console.print("🔄 Collecting test metrics...")
        self.update_all_metrics()
        self.console.clear()
        
        # Display all panels
        self.console.print(self.render_header())
        self.console.print()
        self.console.print(self.render_progress_to_goal())
        self.console.print()
        
        # Two-column layout for failure breakdown and components
        columns = Columns([
            self.render_failure_breakdown(),
            self.render_component_health()
        ], equal=True)
        self.console.print(columns)
        self.console.print()
        
        self.console.print(self.render_recommendations())
        self.console.print()
        self.console.print(self.render_footer())
    
    def start_live_dashboard(self, refresh_interval: float = 2.0):
        """Start live updating dashboard."""
        self.is_live = True
        
        self.console.print("🚀 Starting live test dashboard...")
        self.console.print("📊 Initial data collection...")
        
        layout = self.create_layout()
        
        try:
            with Live(layout, console=self.console, refresh_per_second=1/refresh_interval) as live:
                while True:
                    # Update metrics periodically
                    self.update_all_metrics()
                    self.update_layout(layout)
                    time.sleep(refresh_interval)
                    
        except KeyboardInterrupt:
            self.console.print("\n✅ Dashboard stopped")
        finally:
            self.is_live = False


def main():
    """Main entry point for test dashboard."""
    parser = argparse.ArgumentParser(description="Research Agent Test Status Dashboard")
    parser.add_argument("--live", action="store_true", help="Start live updating dashboard")
    parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval for live mode (seconds)")
    parser.add_argument("--analysis", action="store_true", help="Include detailed failure analysis")
    parser.add_argument("--components", action="store_true", help="Focus on component health view")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Create dashboard
    console = Console()
    dashboard = TestDashboard(console=console)
    
    try:
        if args.live:
            dashboard.start_live_dashboard(refresh_interval=args.refresh)
        else:
            dashboard.show_static_summary()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Dashboard error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 