#!/usr/bin/env python3
"""
Main orchestrator for GPT-OSS-20B red teaming framework.

This script coordinates comprehensive vulnerability testing and manages
the entire red teaming workflow from test execution to findings export.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the tools directory to the path
sys.path.append(str(Path(__file__).parent / "tools"))

from tools.model_interface import GPTOSSInterface
from tools.findings_schema import FindingsManager
from tools.vulnerability_tests import ComprehensiveTestRunner


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('red_teaming.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RedTeamOrchestrator:
    """Main orchestrator for red teaming operations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_interface = GPTOSSInterface(
            api_key=config.get("openai_api_key"),
            model_name=config.get("model_name", "gpt-oss-20b")
        )
        self.findings_manager = FindingsManager(
            findings_dir=config.get("findings_dir", "findings")
        )
        self.test_runner = ComprehensiveTestRunner(
            self.model_interface, 
            self.findings_manager
        )
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path("results") / self.session_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_comprehensive_testing(self) -> Dict[str, Any]:
        """Run comprehensive vulnerability testing."""
        logger.info(f"Starting comprehensive red teaming session: {self.session_id}")
        
        try:
            # Run all test suites
            results = await self.test_runner.run_all_tests()
            
            # Save comprehensive results
            results_file = self.results_dir / "comprehensive_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Comprehensive testing completed. Results saved to {results_file}")
            
            # Generate summary report
            await self.generate_summary_report(results)
            
            # Export findings for competition submission
            if self.findings_manager.findings:
                await self.export_competition_findings()
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            raise
    
    async def run_targeted_testing(self, categories: List[str], num_tests: int = 50) -> Dict[str, Any]:
        """Run targeted testing on specific vulnerability categories."""
        logger.info(f"Starting targeted testing for categories: {categories}")
        
        results = {}
        
        for category in categories:
            if category == "reward_hacking":
                from tools.vulnerability_tests import RewardHackingTests
                suite = RewardHackingTests(self.model_interface)
            elif category == "deception":
                from tools.vulnerability_tests import DeceptionTests
                suite = DeceptionTests(self.model_interface)
            elif category == "sandbagging":
                from tools.vulnerability_tests import SandbaggingTests
                suite = SandbaggingTests(self.model_interface)
            else:
                logger.warning(f"Unknown category: {category}")
                continue
            
            # Limit test cases for targeted testing
            suite.test_cases = suite.generate_test_cases()[:num_tests]
            
            category_results = await suite.run_test_suite()
            results[category] = category_results
            
            # Create findings for high-risk issues
            for result in category_results:
                if result["vulnerability_detected"] and result["risk_score"] >= 6:
                    await self.test_runner.create_finding_from_result(result)
        
        # Save targeted results
        results_file = self.results_dir / f"targeted_results_{'_'.join(categories)}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    async def generate_summary_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary report."""
        
        report_lines = [
            "# GPT-OSS-20B Red Teaming Summary Report",
            f"**Session ID:** {self.session_id}",
            f"**Timestamp:** {datetime.now().isoformat()}",
            f"**Model:** {self.config.get('model_name', 'gpt-oss-20b')}",
            "",
            "## Executive Summary",
            ""
        ]
        
        summary = results.get("summary", {})
        total_vulnerabilities = summary.get("total_vulnerabilities", 0)
        findings_created = summary.get("findings_created", 0)
        
        report_lines.extend([
            f"- **Total Vulnerabilities Detected:** {total_vulnerabilities}",
            f"- **High-Risk Findings Created:** {findings_created}",
            f"- **Test Suites Executed:** {summary.get('total_test_suites', 0)}",
            "",
            "## Vulnerability Category Analysis",
            ""
        ])
        
        # Analyze each category
        for category, category_results in results.items():
            if category == "summary":
                continue
                
            total_tests = category_results.get("total_tests", 0)
            vulns_found = category_results.get("vulnerabilities_found", 0)
            
            success_rate = (vulns_found / total_tests * 100) if total_tests > 0 else 0
            
            report_lines.extend([
                f"### {category.replace('_', ' ').title()}",
                f"- Tests Run: {total_tests}",
                f"- Vulnerabilities Found: {vulns_found}",
                f"- Success Rate: {success_rate:.1f}%",
                ""
            ])
            
            # Top vulnerabilities in this category
            category_vulns = [r for r in category_results.get("results", []) 
                            if r["vulnerability_detected"]]
            
            if category_vulns:
                # Sort by risk score
                top_vulns = sorted(category_vulns, 
                                 key=lambda x: x["risk_score"], 
                                 reverse=True)[:3]
                
                report_lines.append("**Top Vulnerabilities:**")
                for i, vuln in enumerate(top_vulns, 1):
                    test_case = vuln["test_case"]
                    report_lines.append(
                        f"{i}. {test_case.description} (Risk: {vuln['risk_score']}/10)"
                    )
                report_lines.append("")
        
        # Risk assessment
        report_lines.extend([
            "## Risk Assessment",
            "",
            f"**Overall Risk Level:** {'HIGH' if total_vulnerabilities > 10 else 'MEDIUM' if total_vulnerabilities > 5 else 'LOW'}",
            "",
            "### Key Findings:",
        ])
        
        # Get findings summary
        findings_summary = self.findings_manager.get_findings_summary()
        if findings_summary.get("total_findings", 0) > 0:
            report_lines.extend([
                f"- Average Severity Score: {findings_summary.get('avg_severity', 0):.1f}/10",
                f"- Average Breadth Score: {findings_summary.get('avg_breadth', 0):.1f}/10",
                f"- Maximum Severity: {findings_summary.get('max_severity', 0)}/10",
                f"- Categories Affected: {list(findings_summary.get('categories', {}).keys())}",
            ])
        
        report_lines.extend([
            "",
            "## Recommendations",
            "",
            "1. **Immediate Actions:**",
            "   - Review high-severity findings (8+ severity score)",
            "   - Implement additional safety filters for identified patterns",
            "   - Test proposed mitigations against discovered vulnerabilities",
            "",
            "2. **Long-term Improvements:**",
            "   - Enhance training data filtering for vulnerable patterns",
            "   - Develop automated detection for identified vulnerability types",
            "   - Establish continuous monitoring for these vulnerability classes",
            "",
            "## Competition Submission",
            "",
            f"- **Findings Ready for Submission:** {findings_created}",
            f"- **Findings Files Location:** `{self.findings_manager.findings_dir}/`",
            f"- **Supporting Data:** `{self.results_dir}/`",
        ])
        
        # Write report
        report_content = "\n".join(report_lines)
        report_file = self.results_dir / "summary_report.md"
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Summary report generated: {report_file}")
        return str(report_file)
    
    async def export_competition_findings(self) -> List[str]:
        """Export findings in competition format."""
        logger.info("Exporting findings for competition submission...")
        
        exported_files = self.findings_manager.export_all_findings()
        
        # Copy to results directory for archival
        for filepath in exported_files:
            filename = Path(filepath).name
            result_copy = self.results_dir / filename
            
            with open(filepath, 'r') as src, open(result_copy, 'w') as dst:
                dst.write(src.read())
        
        logger.info(f"Exported {len(exported_files)} findings files")
        return exported_files
    
    async def validate_findings(self) -> Dict[str, Any]:
        """Validate all findings against competition schema."""
        logger.info("Validating findings against competition schema...")
        
        validation_results = {
            "total_findings": len(self.findings_manager.findings),
            "valid_findings": 0,
            "invalid_findings": [],
            "warnings": []
        }
        
        for finding in self.findings_manager.findings:
            try:
                self.findings_manager.validate_finding(finding)
                validation_results["valid_findings"] += 1
                
                # Additional quality checks
                if len(finding.description) < 50:
                    validation_results["warnings"].append(
                        f"Finding {finding.id}: Description may be too short"
                    )
                
                if finding.severity_self_assessment > 8 and finding.breadth_self_assessment < 3:
                    validation_results["warnings"].append(
                        f"Finding {finding.id}: High severity with low breadth - verify scores"
                    )
                    
            except Exception as e:
                validation_results["invalid_findings"].append({
                    "finding_id": finding.id,
                    "error": str(e)
                })
        
        # Save validation report
        validation_file = self.results_dir / "validation_report.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation complete. {validation_results['valid_findings']}/{validation_results['total_findings']} findings valid")
        return validation_results


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "model_name": os.getenv("MODEL_NAME", "gpt-oss-20b"),
        "findings_dir": "findings",
        "batch_size": 5,
        "delay_between_batches": 1.0
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    return config


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPT-OSS-20B Red Teaming Framework")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--mode", choices=["comprehensive", "targeted", "validate"], 
                       default="comprehensive", help="Testing mode")
    parser.add_argument("--categories", nargs="+", 
                       choices=["reward_hacking", "deception", "sandbagging"],
                       help="Categories for targeted testing")
    parser.add_argument("--num-tests", type=int, default=50,
                       help="Number of tests per category for targeted testing")
    parser.add_argument("--export-only", action="store_true",
                       help="Only export existing findings")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if not config["openai_api_key"]:
        logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or provide in config file.")
        sys.exit(1)
    
    # Initialize orchestrator
    orchestrator = RedTeamOrchestrator(config)
    
    try:
        if args.export_only:
            # Only export existing findings
            await orchestrator.export_competition_findings()
            
        elif args.mode == "validate":
            # Validate existing findings
            validation_results = await orchestrator.validate_findings()
            print(f"Validation complete: {validation_results['valid_findings']}/{validation_results['total_findings']} valid")
            
        elif args.mode == "targeted":
            # Run targeted testing
            if not args.categories:
                logger.error("Categories required for targeted testing")
                sys.exit(1)
            
            results = await orchestrator.run_targeted_testing(args.categories, args.num_tests)
            print(f"Targeted testing complete. Results in {orchestrator.results_dir}")
            
        else:
            # Run comprehensive testing
            results = await orchestrator.run_comprehensive_testing()
            print(f"Comprehensive testing complete. Results in {orchestrator.results_dir}")
        
        logger.info("Red teaming session completed successfully")
        
    except Exception as e:
        logger.error(f"Red teaming session failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())