#!/usr/bin/env python3
"""
Test Analysis Framework - Orchestration Layer

Coordinates all test analysis tools to provide a unified workflow for systematic
test failure analysis, debugging, and resolution tracking.
"""

import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class FrameworkConfig:
    """Configuration for the test analysis framework."""
    project_root: str
    scripts_dir: str
    enable_parallel_execution: bool = True
    max_workers: int = 4
    dashboard_live_mode: bool = False
    export_reports: bool = True
    verbose: bool = False

@dataclass
class AnalysisSession:
    """Represents a complete analysis session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tests: int = 0
    passing_tests: int = 0
    failing_tests: int = 0
    success_rate: float = 0.0
    tools_used: List[str] = None
    reports_generated: List[str] = None

    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = []
        if self.reports_generated is None:
            self.reports_generated = []

class TestAnalysisFramework:
    """
    Main orchestration framework that coordinates all test analysis tools
    to provide a unified workflow for systematic test failure debugging.
    """
    
    def __init__(self, config: FrameworkConfig):
        self.config = config
        self.project_root = Path(config.project_root)
        self.scripts_dir = Path(config.scripts_dir)
        
        # Available tools (simplified integration)
        self.available_tools = [
            "test_analyzer.py",
            "component_runner.py", 
            "priority_matrix.py",
            "test_dashboard.py",
            "suggestion_engine.py"
        ]
        
        # Session management
        self.current_session: Optional[AnalysisSession] = None
        
    def _check_tool_availability(self):
        """Check which tools are available."""
        available = []
        for tool in self.available_tools:
            tool_path = self.scripts_dir / tool
            if tool_path.exists():
                available.append(tool)
        return available
    
    def start_session(self) -> str:
        """Start a new analysis session."""
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = AnalysisSession(
            session_id=session_id,
            start_time=datetime.now()
        )
        
        available_tools = self._check_tool_availability()
        
        print(f"\n🚀 Starting Analysis Session: {session_id}")
        print(f"   Project: {self.project_root}")
        print(f"   Available Tools: {', '.join(available_tools)}")
        
        return session_id
    
    def end_session(self):
        """End the current analysis session."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            
            # Export session summary
            self._export_session_summary()
            
            duration = (self.current_session.end_time - self.current_session.start_time).total_seconds()
            print(f"\n✅ Session {self.current_session.session_id} completed")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Tools Used: {', '.join(self.current_session.tools_used)}")
            print(f"   Reports Generated: {len(self.current_session.reports_generated)}")
            
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis workflow:
        1. Analyze test failures
        2. Run component analysis
        3. Generate priority matrix
        4. Create suggestions
        5. Update dashboard
        """
        session_id = self.start_session()
        results = {"session_id": session_id, "steps": {}}
        
        try:
            # Step 1: Test Failure Analysis
            print("\n📊 Step 1: Analyzing Test Failures...")
            if (self.scripts_dir / "test_analyzer.py").exists():
                analysis_result = self._run_test_analysis()
                results["steps"]["analysis"] = analysis_result
                print(f"   ✅ Found {analysis_result.get('failing_tests', 0)} failing tests")
            else:
                print("   ⚠️  Test analyzer not available, skipping...")
            
            # Step 2: Component Testing
            print("\n🧪 Step 2: Running Component Analysis...")
            if (self.scripts_dir / "component_runner.py").exists():
                component_result = self._run_component_analysis()
                results["steps"]["components"] = component_result
                print(f"   ✅ Analyzed {len(component_result.get('components', []))} components")
            else:
                print("   ⚠️  Component runner not available, skipping...")
            
            # Step 3: Priority Analysis
            print("\n📈 Step 3: Generating Priority Matrix...")
            if (self.scripts_dir / "priority_matrix.py").exists():
                priority_result = self._run_priority_analysis()
                results["steps"]["priority"] = priority_result
                print(f"   ✅ Prioritized {priority_result.get('total_patterns', 0)} failure patterns")
            else:
                print("   ⚠️  Priority matrix not available, skipping...")
            
            # Step 4: Fix Suggestions
            print("\n💡 Step 4: Generating Fix Suggestions...")
            if (self.scripts_dir / "suggestion_engine.py").exists():
                suggestion_result = self._run_suggestion_analysis()
                results["steps"]["suggestions"] = suggestion_result
                print(f"   ✅ Generated {len(suggestion_result.get('suggestions', []))} recommendations")
            else:
                print("   ⚠️  Suggestion engine not available, skipping...")
            
            # Step 5: Dashboard Update
            print("\n📋 Step 5: Updating Dashboard...")
            if (self.scripts_dir / "test_dashboard.py").exists():
                dashboard_result = self._update_dashboard()
                results["steps"]["dashboard"] = dashboard_result
                print("   ✅ Dashboard updated successfully")
            else:
                print("   ⚠️  Dashboard not available, skipping...")
            
            # Update session stats
            self._update_session_stats(results)
            
            # Generate comprehensive report
            self._generate_comprehensive_report(results)
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            results["error"] = str(e)
            import traceback
            if self.config.verbose:
                traceback.print_exc()
        
        finally:
            self.end_session()
        
        return results
    
    def _run_test_analysis(self) -> Dict[str, Any]:
        """Run test failure analysis."""
        self.current_session.tools_used.append("test_analyzer")
        
        cmd = [sys.executable, str(self.scripts_dir / "test_analyzer.py")]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Load results
        analysis_file = self.scripts_dir / "test_analysis_results.json"
        if analysis_file.exists():
            with open(analysis_file) as f:
                data = json.load(f)
                self.current_session.reports_generated.append(str(analysis_file))
                return data
        
        return {"error": "Test analysis failed", "output": result.stderr}
    
    def _run_component_analysis(self) -> Dict[str, Any]:
        """Run component-based testing analysis."""
        self.current_session.tools_used.append("component_runner")
        
        cmd = [sys.executable, str(self.scripts_dir / "component_runner.py"), 
               "--list-components"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        if result.returncode == 0:
            # Parse component list from output
            components = []
            for line in result.stdout.split('\n'):
                if 'Component' in line and ':' in line:
                    components.append(line.strip())
            
            return {
                "components": components,
                "total_components": len(components),
                "execution_mode": "parallel" if self.config.enable_parallel_execution else "sequential"
            }
        
        return {"error": "Component analysis failed", "output": result.stderr}
    
    def _run_priority_analysis(self) -> Dict[str, Any]:
        """Run priority matrix analysis."""
        self.current_session.tools_used.append("priority_matrix")
        
        cmd = [sys.executable, str(self.scripts_dir / "priority_matrix.py")]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Load priority results
        priority_file = self.scripts_dir / "test_priority_analysis.json"
        if priority_file.exists():
            with open(priority_file) as f:
                data = json.load(f)
                self.current_session.reports_generated.append(str(priority_file))
                return data
        
        return {"error": "Priority analysis failed", "output": result.stderr}
    
    def _run_suggestion_analysis(self) -> Dict[str, Any]:
        """Run suggestion engine analysis."""
        self.current_session.tools_used.append("suggestion_engine")
        
        cmd = [sys.executable, str(self.scripts_dir / "suggestion_engine.py")]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        # Find the latest suggestion file
        suggestion_files = list(self.scripts_dir.glob("fix_suggestions_*.json"))
        if suggestion_files:
            latest_file = max(suggestion_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file) as f:
                data = json.load(f)
                self.current_session.reports_generated.append(str(latest_file))
                return data
        
        return {"error": "Suggestion analysis failed", "output": result.stderr}
    
    def _update_dashboard(self) -> Dict[str, Any]:
        """Update the test dashboard."""
        self.current_session.tools_used.append("dashboard")
        
        cmd = [sys.executable, str(self.scripts_dir / "test_dashboard.py")]
        if self.config.dashboard_live_mode:
            cmd.append("--live")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
        
        return {
            "updated": result.returncode == 0,
            "live_mode": self.config.dashboard_live_mode,
            "output": result.stdout if self.config.verbose else "Dashboard updated"
        }
    
    def _update_session_stats(self, results: Dict[str, Any]):
        """Update session statistics from analysis results."""
        if "analysis" in results["steps"]:
            analysis = results["steps"]["analysis"]
            self.current_session.total_tests = analysis.get("total_tests", 0)
            self.current_session.passing_tests = analysis.get("passing_tests", 0)
            self.current_session.failing_tests = analysis.get("failing_tests", 0)
            self.current_session.success_rate = analysis.get("success_rate", 0.0)
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate a comprehensive analysis report."""
        report = {
            "session": asdict(self.current_session),
            "analysis_results": results,
            "summary": {
                "total_tests": self.current_session.total_tests,
                "success_rate": f"{self.current_session.success_rate:.1%}",
                "tools_used": len(self.current_session.tools_used),
                "reports_generated": len(self.current_session.reports_generated)
            },
            "recommendations": self._get_high_level_recommendations(results)
        }
        
        report_file = self.scripts_dir / f"comprehensive_report_{self.current_session.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.current_session.reports_generated.append(str(report_file))
        print(f"   📄 Comprehensive report: {report_file}")
    
    def _get_high_level_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate high-level recommendations from analysis results."""
        recommendations = []
        
        # Check success rate
        if self.current_session.success_rate < 0.8:
            recommendations.append("🔴 Test success rate below 80% - prioritize systematic debugging")
        elif self.current_session.success_rate < 0.95:
            recommendations.append("🟡 Test success rate below 95% - focus on remaining failures")
        else:
            recommendations.append("🟢 Test success rate good - maintain current quality")
        
        # Check failure patterns
        if "suggestions" in results["steps"]:
            suggestions = results["steps"]["suggestions"].get("suggestions", [])
            high_priority = [s for s in suggestions if s.get("priority") == "high"]
            if high_priority:
                recommendations.append(f"🔴 {len(high_priority)} high-priority issues identified")
        
        # Check component health
        if "components" in results["steps"]:
            total_components = results["steps"]["components"].get("total_components", 0)
            if total_components > 0:
                recommendations.append(f"✅ {total_components} components analyzed for focused debugging")
        
        return recommendations
    
    def _export_session_summary(self):
        """Export a brief session summary."""
        if not self.current_session:
            return
        
        summary = {
            "session_id": self.current_session.session_id,
            "duration_seconds": (self.current_session.end_time - self.current_session.start_time).total_seconds(),
            "test_metrics": {
                "total": self.current_session.total_tests,
                "passing": self.current_session.passing_tests,
                "failing": self.current_session.failing_tests,
                "success_rate": self.current_session.success_rate
            },
            "tools_used": self.current_session.tools_used,
            "reports_count": len(self.current_session.reports_generated)
        }
        
        summary_file = self.scripts_dir / f"session_summary_{self.current_session.session_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def run_quick_health_check(self) -> Dict[str, Any]:
        """Run a quick health check of the test suite."""
        print("\n🏥 Running Quick Health Check...")
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "tools_available": self._check_tool_availability(),
            "reports_exist": {},
            "last_analysis": None
        }
        
        # Check for existing reports
        reports = {
            "test_analysis": self.scripts_dir / "test_analysis_results.json",
            "priority_matrix": self.scripts_dir / "test_priority_analysis.json",
            "suggestions": list(self.scripts_dir.glob("fix_suggestions_*.json"))
        }
        
        for report_type, path in reports.items():
            if isinstance(path, list):
                health_status["reports_exist"][report_type] = len(path) > 0
                if path:
                    latest = max(path, key=lambda p: p.stat().st_mtime)
                    health_status[f"latest_{report_type}"] = latest.name
            else:
                health_status["reports_exist"][report_type] = path.exists()
                if path.exists():
                    health_status[f"last_modified_{report_type}"] = datetime.fromtimestamp(
                        path.stat().st_mtime).isoformat()
        
        # Quick test count check
        try:
            cmd = ["python", "-m", "pytest", "--collect-only", "-q"]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            if "collected" in result.stdout:
                import re
                match = re.search(r'(\d+) items collected', result.stdout)
                if match:
                    health_status["total_tests_discovered"] = int(match.group(1))
        except Exception as e:
            health_status["test_discovery_error"] = str(e)
        
        return health_status

def main():
    """Main CLI interface for the framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Analysis Framework - Comprehensive Test Debugging")
    parser.add_argument("--project-root", default=str(Path.cwd()), 
                       help="Project root directory")
    parser.add_argument("--mode", choices=["full", "health"], default="full",
                       help="Analysis mode")
    parser.add_argument("--live-dashboard", action="store_true", 
                       help="Enable live dashboard mode")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Enable parallel execution")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create configuration
    config = FrameworkConfig(
        project_root=args.project_root,
        scripts_dir=str(Path(args.project_root) / "scripts"),
        enable_parallel_execution=args.parallel,
        dashboard_live_mode=args.live_dashboard,
        verbose=args.verbose
    )
    
    # Initialize framework
    framework = TestAnalysisFramework(config)
    
    # Run based on mode
    if args.mode == "full":
        print("🔬 Running Complete Analysis Workflow...")
        results = framework.run_complete_analysis()
        print(f"\n📊 Analysis Complete - Session: {results['session_id']}")
        
    elif args.mode == "health":
        results = framework.run_quick_health_check()
        print("\n🏥 Health Check Results:")
        print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main() 