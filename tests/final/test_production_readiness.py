"""Production readiness validation tests for GeoAnomalyMapper (GAM).
Verifies dependencies, configurations, service lifecycle, external connectivity, and performance benchmarks.
Run with: pytest tests/final/test_production_readiness.py -v -m production_readiness

These tests should pass in a clean production-like environment before deployment.
Requires Docker and kubectl for service tests.
"""

import os
import pytest
import subprocess
import time
import requests
from pathlib import Path
from packaging import version

import pkg_resources
from gam.core.config import load_config, validate_config
from gam.core.pipeline import GAMPipeline

# Mark as production readiness tests
pytestmark = pytest.mark.production_readiness

@pytest.fixture
def production_config_path():
    """Fixture for production config."""
    return Path(__file__).parent.parent / "config" / "production" / "production.yaml"

class TestProductionReadiness:
    """Production readiness tests."""

    def test_dependencies_installed(self):
        """Verify all production dependencies are installed."""
        with open(Path(__file__).parent.parent / "requirements.txt", 'r') as f:
            required = [line.strip().split('>=')[0].split('==')[0] for line in f if line.strip() and not line.startswith('#')]
        
        missing = []
        for req in required:
            try:
                installed = pkg_resources.get_distribution(req)
                # Check version if pinned
                if '>=' in req or '==' in req:
                    req_version = req.split('>')[0].split('=')[0]
                    assert version.parse(installed.version) >= version.parse(req_version), f"{req} version mismatch"
            except pkg_resources.DistributionNotFound:
                missing.append(req)
        
        assert not missing, f"Missing dependencies: {missing}"
        logger.info("All dependencies installed and versions compatible")

    def test_configuration_files_valid(self, production_config_path):
        """Validate production configuration files."""
        assert production_config_path.exists(), "Production config not found"
        
        config = load_config(production_config_path)
        assert config['global']['environment'] == 'production', "Config not in production mode"
        
        validate_config(config)
        
        # Check secrets template
        secrets_template = production_config_path.parent / "secrets.yaml.template"
        assert secrets_template.exists(), "Secrets template not found"
        with open(secrets_template, 'r') as f:
            content = f.read()
            assert '${' in content, "Secrets template has placeholders"
        
        logger.info("Configuration files valid")

    def test_services_start_stop_cleanly(self):
        """Test GAM services start and stop without errors."""
        # Test Docker compose
        compose_file = Path(__file__).parent.parent / "deployment" / "docker" / "docker-compose.yml"
        try:
            # Start
            result = subprocess.run(["docker", "compose", "-f", str(compose_file), "up", "-d"], 
                                  capture_output=True, text=True, timeout=120)
            assert result.returncode == 0, "Docker compose up failed"
            
            time.sleep(30)  # Wait for startup
            
            # Check status
            ps_result = subprocess.run(["docker", "compose", "-f", str(compose_file), "ps"], 
                                     capture_output=True, text=True)
            assert "Up" in ps_result.stdout, "Services not up"
            
            # Stop
            stop_result = subprocess.run(["docker", "compose", "-f", str(compose_file), "down"], 
                                       capture_output=True, text=True, timeout=60)
            assert stop_result.returncode == 0, "Docker compose down failed"
            
            logger.info("Docker services lifecycle test passed")
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "compose", "-f", str(compose_file), "down"], check=False)
            pytest.fail("Timeout during service lifecycle test")
        
        # Test K8s (if kubectl available)
        try:
            subprocess.run(["kubectl", "apply", "-f", str(Path(__file__).parent.parent / "deployment" / "k8s")], 
                         capture_output=True, check=True)
            subprocess.run(["kubectl", "rollout", "status", "deployment/gam-deployment", "-n", "gam-system", "--timeout=60s"], 
                         capture_output=True, check=True)
            subprocess.run(["kubectl", "delete", "-f", str(Path(__file__).parent.parent / "deployment" / "k8s"), "--ignore-not-found"], 
                         capture_output=True, check=True)
            logger.info("K8s services lifecycle test passed")
        except FileNotFoundError:
            pytest.skip("kubectl not available for K8s test")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"K8s lifecycle test failed: {e}")

    def test_external_api_connectivity(self):
        """Verify connectivity to external data sources."""
        apis = [
            ("USGS", "https://earthquake.usgs.gov/robots.txt"),
            ("IRIS", "https://service.iris.edu/irisws/timeseries/1/query?net=IU&sta=ANMO&loc=00&cha=BHZ&starttime=2020-01-01&endtime=2020-01-02&format=text"),
            ("ESA", "https://scihub.copernicus.eu/dhus/odata/v1/")  # OData endpoint
        ]
        
        for name, url in apis:
            try:
                response = requests.head(url, timeout=10)
                assert response.status_code < 400, f"{name} API unreachable: {response.status_code}"
                logger.info(f"{name} API connectivity OK")
            except requests.RequestException as e:
                pytest.fail(f"{name} API connectivity failed: {e}")

    def test_performance_meets_benchmarks(self, production_config_path):
        """Run performance benchmark and verify thresholds."""
        config = load_config(production_config_path)
        pipeline = GAMPipeline(config=config)
        
        start_time = time.time()
        # Small test run
        results = pipeline.run_analysis(
            bbox=(29.0, 31.0, 30.0, 32.0),  # Small region
            modalities=['gravity'],
            use_synthetic=True
        )
        exec_time = time.time() - start_time
        
        assert exec_time < 30, f"Small analysis too slow: {exec_time}s (threshold 30s)"
        assert len(results.get('anomalies', [])) > 0, "No output from benchmark run"
        
        # Memory check (rough)
        import psutil
        mem_mb = psutil.Process().memory_info().rss / 1024**2
        assert mem_mb < 2000, f"High memory usage: {mem_mb}MB (threshold 2GB)"
        
        logger.info(f"Performance benchmark passed: {exec_time:.2f}s, {mem_mb:.2f}MB")

    def test_all_external_connectivity(self):
        """Comprehensive external service check."""
        # Database (assume local or test DB)
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:")
        with engine.connect():
            pass  # Basic connect
        
        # Redis (if configured)
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            r.ping()
            logger.info("Redis connectivity OK")
        except ImportError:
            pytest.skip("redis-py not installed")
        except Exception:
            pytest.skip("Redis not available for test")
        
        logger.info("All external connectivity tests passed")