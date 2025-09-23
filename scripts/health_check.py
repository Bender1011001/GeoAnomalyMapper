#!/usr/bin/env python3
# health_check.py - Production health check script for GeoAnomalyMapper (GAM)
# Verifies API, database, external services, system resources, and GAM functionality.
# Usage: python scripts/health_check.py [OPTIONS]
# Options:
#   --endpoint URL: GAM API endpoint (default: http://localhost:8000)
#   --db-url STRING: Database URL (default: from config)
#   --timeout SECONDS: Request timeout (default: 30)
#   --json: Output JSON summary
#   --help: Show help

import argparse
import json
import logging
import os
import sys
import time
import yaml
from datetime import datetime
from typing import Dict, Any

import psutil
import requests
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GAMHealthChecker:
    def __init__(self, endpoint: str = "http://localhost:8000", db_url: str = None, timeout: int = 30):
        self.endpoint = endpoint.rstrip('/')
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql://gam_user:password@localhost:5432/gam_db')
        self.timeout = timeout
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }

    def run_check(self, name: str, check_func, critical: bool = True):
        """Run a health check and record result."""
        try:
            status, message = check_func()
            self.results['checks'][name] = {
                'status': status,
                'message': message,
                'timestamp': datetime.utcnow().isoformat()
            }
            if status != 'healthy' and critical:
                self.results['overall_status'] = 'unhealthy'
                logger.warning(f"Critical check failed: {name} - {message}")
            else:
                logger.info(f"Check passed: {name}")
            return status == 'healthy'
        except Exception as e:
            self.results['checks'][name] = {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            if critical:
                self.results['overall_status'] = 'unhealthy'
            logger.error(f"Check error in {name}: {e}")
            return False

    def check_api(self) -> tuple:
        """Check GAM API endpoint."""
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'healthy':
                return 'healthy', f"API healthy: {data.get('version', 'unknown')}"
            else:
                return 'unhealthy', f"API reported unhealthy: {data}"
        except requests.RequestException as e:
            return 'unhealthy', f"API unreachable: {e}"

    def check_database(self) -> tuple:
        """Check database connectivity and basic query."""
        try:
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                if result.scalar() == 1:
                    return 'healthy', "DB connection successful"
                else:
                    return 'unhealthy', "DB query failed"
        except OperationalError as e:
            return 'unhealthy', f"DB connection failed: {e}"

    def check_external_apis(self) -> tuple:
        """Ping external data sources (non-destructive)."""
        checks = []
        try:
            # USGS example ping (replace with actual endpoint)
            response = requests.head("https://earthquake.usgs.gov/ws/rest/index.php", timeout=self.timeout)
            if response.status_code == 200:
                checks.append("USGS healthy")
            else:
                checks.append(f"USGS unhealthy: {response.status_code}")
        except requests.RequestException as e:
            checks.append(f"USGS error: {e}")

        # Add more: IRIS, ESA
        # IRIS: https://service.iris.edu/irisws/timeseries/1/query
        # ESA: https://scihub.copernicus.eu/dhus/

        if all("healthy" in check for check in checks):
            return 'healthy', f"External APIs: {', '.join(checks)}"
        else:
            return 'warning', f"External APIs issues: {', '.join(checks)}"

    def check_system_resources(self) -> tuple:
        """Check CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        mem_percent = memory.percent

        issues = []
        if cpu_percent > 80:
            issues.append(f"High CPU: {cpu_percent}%")
        if mem_percent > 90:
            issues.append(f"High Memory: {mem_percent}%")

        if issues:
            return 'warning', f"Resource warnings: {', '.join(issues)}"
        else:
            return 'healthy', f"Resources OK - CPU: {cpu_percent}%, Mem: {mem_percent}%"

    def check_gam_functionality(self) -> tuple:
        """Test basic GAM functionality (load config, small synthetic test)."""
        try:
            # Load production config
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'production', 'production.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config.get('global', {}).get('environment') == 'production':
                # Simple synthetic test (no real data)
                from gam.core.main import GAMPipeline
                pipeline = GAMPipeline(config=config)
                # Test a minimal run (e.g., validate config)
                pipeline.validate_config()
                return 'healthy', "GAM config loaded and validated"
            else:
                return 'unhealthy', "Config not in production mode"
        except ImportError:
            return 'warning', "GAM package not installed in current env"
        except Exception as e:
            return 'unhealthy', f"GAM functionality test failed: {e}"

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        self.run_check('api', self.check_api, critical=True)
        self.run_check('database', self.check_database, critical=True)
        self.run_check('external_apis', self.check_external_apis, critical=False)
        self.run_check('system_resources', self.check_system_resources, critical=False)
        self.run_check('gam_functionality', self.check_gam_functionality, critical=True)
        return self.results

def main():
    parser = argparse.ArgumentParser(description="GAM Production Health Check")
    parser.add_argument('--endpoint', default='http://localhost:8000', help='GAM API endpoint')
    parser.add_argument('--db-url', help='Database URL')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout seconds')
    parser.add_argument('--json', action='store_true', help='Output JSON summary')
    args = parser.parse_args()

    checker = GAMHealthChecker(endpoint=args.endpoint, db_url=args.db_url, timeout=args.timeout)
    results = checker.run_all_checks()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Health Check Summary ({results['timestamp']}):")
        print(f"Overall Status: {results['overall_status']}")
        for name, check in results['checks'].items():
            status_emoji = {'healthy': '✅', 'warning': '⚠️', 'unhealthy': '❌', 'error': '💥'}
            emoji = status_emoji.get(check['status'], '❓')
            print(f"  {emoji} {name}: {check['status']} - {check['message']}")

    sys.exit(0 if results['overall_status'] == 'healthy' else 1)

if __name__ == '__main__':
    main()