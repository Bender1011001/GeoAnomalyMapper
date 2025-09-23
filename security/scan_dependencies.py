#!/usr/bin/env python3
# scan_dependencies.py - Security vulnerability scanner for GAM dependencies
# Scans requirements.txt and extras for known vulnerabilities using Safety.
# Generates report and optionally updates to safe versions.
# Usage: python security/scan_dependencies.py [OPTIONS]
# Requires: pip install safety pip-tools

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

from safety.client import SafetyClient
from safety.exceptions import InvalidRequirement
from piptools.scripts.compile import cli as compile_cli

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GAM_DIR = Path(__file__).parent.parent
REQUIREMENTS_FILE = GAM_DIR / "requirements.txt"
SETUP_PY = GAM_DIR / "setup.py"

class DependencyScanner:
    def __init__(self, requirements_file: Path = REQUIREMENTS_FILE, output_file: Path = Path("security_report.json"), threshold: str = "critical"):
        self.requirements_file = requirements_file
        self.output_file = output_file
        self.threshold = threshold.lower()
        self.severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        self.client = SafetyClient()
        self.vulnerabilities = []
        self.safe_requirements = []

    def parse_requirements(self) -> List[str]:
        """Parse requirements.txt for packages."""
        requirements = []
        with open(self.requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '>=' in line or '==' in line or '==' in line:
                    # Extract package name (before >=, ==, etc.)
                    pkg = line.split('>=')[0].split('==')[0].split('<=')[0].split('!=')[0].strip()
                    if pkg:
                        requirements.append(pkg)
        logger.info(f"Parsed {len(requirements)} packages from {self.requirements_file}")
        return requirements

    def parse_extras(self) -> List[str]:
        """Parse extras from setup.py."""
        extras = []
        with open(SETUP_PY, 'r') as f:
            content = f.read()
            # Extract extras_require dict
            import re
            extra_matches = re.findall(r'"(geophysics|visualization|dev)"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            for name, pkgs in extra_matches:
                pkg_list = re.findall(r'["\']([^"\']+)["\']', pkgs)
                extras.extend(pkg_list)
        unique_extras = list(set(extras))
        logger.info(f"Parsed {len(unique_extras)} extra packages from setup.py")
        return unique_extras

    def scan_vulnerabilities(self, packages: List[str]) -> List[Dict[str, Any]]:
        """Scan packages for vulnerabilities using Safety."""
        vulns = []
        for pkg in packages:
            try:
                results = self.client.check_package(pkg)
                for vuln in results:
                    vuln_dict = {
                        "package": pkg,
                        "vulnerability_id": vuln.vulnerability_id,
                        "severity": vuln.severity,
                        "description": vuln.description,
                        "installed_version": vuln.installed_version,
                        "affected_versions": vuln.affected_versions,
                        "fixed_version": vuln.fixed_version,
                        "more_info_url": vuln.more_info_url
                    }
                    if self.severity_levels.get(vuln.severity.lower(), 0) >= self.severity_levels[self.threshold]:
                        vulns.append(vuln_dict)
                    logger.debug(f"Scanned {pkg}: {vuln.severity} - {vuln.description[:50]}...")
            except InvalidRequirement:
                logger.warning(f"Invalid requirement: {pkg}")
            except Exception as e:
                logger.error(f"Error scanning {pkg}: {e}")
        return vulns

    def generate_report(self, vulns: List[Dict[str, Any]], all_packages: List[str]):
        """Generate security report."""
        severity_count = {}
        for v in vulns:
            sev = v['severity'].lower()
            severity_count[sev] = severity_count.get(sev, 0) + 1
        
        report = {
            "scan_timestamp": datetime.utcnow().isoformat(),
            "requirements_file": str(self.requirements_file),
            "threshold": self.threshold,
            "total_packages": len(all_packages),
            "vulnerable_packages": len(vulns),
            "severity_distribution": severity_count,
            "vulnerabilities": vulns,
            "recommendations": [
                "Review and update vulnerable packages to fixed versions.",
                "Use --update flag to generate updated requirements.in.",
                "Integrate this script into CI/CD for automated scanning.",
                "Consider using Dependabot or Safety CI for ongoing monitoring."
            ]
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report generated: {self.output_file}")
        logger.info(f"Found {len(vulns)} vulnerabilities above {self.threshold} threshold")
        for sev, count in severity_count.items():
            logger.info(f"  {sev}: {count}")
        
        return report

    def update_requirements(self, packages: List[str], vulns: List[Dict[str, Any]]):
        """Generate updated requirements with safe versions."""
        requirements_in = self.requirements_file.with_suffix('.in')
        with open(requirements_in, 'w') as f:
            for pkg in packages:
                vuln = next((v for v in vulns if v['package'] == pkg), None)
                if vuln and vuln['fixed_version']:
                    # Pin to fixed version
                    f.write(f"{pkg}=={vuln['fixed_version']}\n")
                    logger.info(f"Updated {pkg} to {vuln['fixed_version']}")
                else:
                    f.write(f"{pkg}\n")
        
        # Compile locked requirements
        compile_args = ['compile', str(requirements_in), '-o', str(self.requirements_file)]
        compile_cli(compile_args)
        logger.info("Updated requirements.txt generated")

    def run(self):
        """Run full scan."""
        packages = self.parse_requirements() + self.parse_extras()
        packages = list(set(packages))  # Dedupe
        
        vulns = self.scan_vulnerabilities(packages)
        report = self.generate_report(vulns, packages)
        
        if vulns and args.update:
            self.update_requirements(packages, vulns)
        
        if vulns:
            logger.warning(f"Security issues found! See {self.output_file}")
            return 1
        else:
            logger.info("No vulnerabilities found above threshold.")
            return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scan GAM dependencies for security vulnerabilities")
    parser.add_argument('--requirements', type=Path, default=REQUIREMENTS_FILE,
                        help='Path to requirements.txt')
    parser.add_argument('--output', type=Path, default=Path("security_report.json"),
                        help='Output report file')
    parser.add_argument('--threshold', choices=['low', 'medium', 'high', 'critical'], default='critical',
                        help='Minimum severity to report')
    parser.add_argument('--update', action='store_true',
                        help='Update requirements to safe versions')
    parser.add_argument('--extras', action='store_true', default=True,
                        help='Include extras from setup.py (default)')
    args = parser.parse_args()
    
    scanner = DependencyScanner(
        requirements_file=args.requirements,
        output_file=args.output,
        threshold=args.threshold
    )
    
    sys.exit(scanner.run())