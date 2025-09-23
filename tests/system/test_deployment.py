"""Deployment validation tests for GeoAnomalyMapper (GAM).
Tests Docker container functionality, configuration loading, service lifecycle, and security.
Run with: pytest tests/system/test_deployment.py -v -m deployment

Requires Docker daemon running. Uses docker-py for container management.
"""

import os
import pytest
import subprocess
import yaml
from pathlib import Path
from docker import from_env
from docker.errors import APIError, NotFound

from gam.core.config import load_config

# Mark as deployment tests
pytestmark = pytest.mark.deployment

@pytest.fixture(scope="session")
def docker_client():
    """Fixture for Docker client."""
    try:
        client = from_env()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")

@pytest.fixture
def gam_image(docker_client):
    """Build GAM Docker image."""
    dockerfile_path = Path(__file__).parent.parent.parent / "deployment" / "docker" / "Dockerfile"
    context_path = dockerfile_path.parent.parent  # GeoAnomalyMapper root
    
    try:
        image, _ = docker_client.images.build(
            path=str(context_path),
            dockerfile=str(dockerfile_path.relative_to(context_path)),
            tag="gam:test",
            rm=True,
            nocache=True
        )
        yield image
    finally:
        try:
            docker_client.images.remove("gam:test", force=True)
        except NotFound:
            pass

class TestDeployment:
    """Deployment validation tests."""

    def test_docker_image_build(self, docker_client, gam_image):
        """Test Docker image builds successfully."""
        assert gam_image.id is not None
        assert "gam:test" in [tag for tag in gam_image.tags]
        
        # Check image size reasonable (<2GB for geospatial)
        size_gb = gam_image.attrs['Size'] / (1024**3)
        assert size_gb < 2.0, f"Image too large: {size_gb}GB"
        
        # Inspect layers for security (no unnecessary packages)
        history = gam_image.history()
        assert len(history) < 50, "Too many layers - optimize Dockerfile"
        
        logger.info(f"Docker image built successfully: {gam_image.id}")

    def test_container_startup(self, docker_client, gam_image):
        """Test container starts and runs GAM CLI."""
        container = docker_client.containers.run(
            gam_image.id,
            command=["gam", "--help"],
            detach=False,
            remove=True,
            mem_limit="4g",
            environment={
                "GAM_ENV": "test",
                "DATABASE_URL": "sqlite:///:memory:"
            },
            volumes={
                str(Path(__file__).parent.parent.parent / "config" / "production"): {
                    'bind': '/app/config', 'mode': 'ro'
                }
            }
        )
        
        # Check exit code
        assert container.attrs['State']['ExitCode'] == 0, "Container failed to start"
        
        # Check logs for errors
        logs = container.logs().decode()
        assert "Error" not in logs, f"Errors in logs: {logs}"
        assert "Usage" in logs, "GAM CLI help not shown"
        
        logger.info("Container startup test passed")

    def test_config_loading_environments(self, tmp_path):
        """Test configuration loading in different environments."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        
        # Create test configs
        dev_config = config_dir / "dev.yaml"
        dev_config.write_text("""
global:
  environment: dev
database:
  url: sqlite:///:memory:
""")
        
        prod_config = config_dir / "prod.yaml"
        prod_config.write_text("""
global:
  environment: production
database:
  url: postgresql://user:pass@host/db
logging:
  level: INFO
""")
        
        # Load dev config
        dev = load_config(dev_config)
        assert dev['global']['environment'] == 'dev'
        assert 'logging' not in dev, "Dev config should be minimal"
        
        # Load prod config
        prod = load_config(prod_config)
        assert prod['global']['environment'] == 'production'
        assert prod['database']['url'].startswith('postgresql')
        assert prod['logging']['level'] == 'INFO'
        
        logger.info("Config loading for environments passed")

    def test_service_lifecycle(self, docker_client, gam_image):
        """Test service startup and shutdown."""
        container = docker_client.containers.run(
            gam_image.id,
            command=["gam", "run", "--config", "/app/config/production.yaml", "--daemon"],
            detach=True,
            mem_limit="4g",
            name="gam-test-lifecycle",
            environment={"GAM_ENV": "test"}
        )
        
        # Wait for startup
        time.sleep(10)
        
        # Check running
        status = container.status
        assert status == 'running', f"Container not running: {status}"
        
        # Check logs for successful start
        logs = container.logs().decode()
        assert "Pipeline started" in logs or "Server running" in logs, "Startup not confirmed in logs"
        
        # Graceful shutdown
        container.stop(timeout=30)
        container.wait()
        
        final_status = container.status
        assert final_status == 'exited', f"Container not stopped: {final_status}"
        
        # Clean up
        container.remove()
        
        logger.info("Service lifecycle test passed")

    def test_security_configurations(self, docker_client, gam_image):
        """Test security aspects of deployment."""
        container = docker_client.containers.run(
            gam_image.id,
            command=["id"],
            detach=False,
            remove=True,
            privileged=False,
            user="gam"  # Non-root user from Dockerfile
        )
        
        output = container.decode('utf-8')
        assert "uid=1000(gam) gid=1000(gam)" in output, "Not running as non-root user"
        
        # Test no privilege escalation
        logs = container.logs().decode()
        assert "permission denied" not in logs.lower(), "Unexpected permission issues"
        
        # Test secrets mounting (simulate with env)
        test_container = docker_client.containers.run(
            gam_image.id,
            command=["env | grep DB"],
            environment={"DATABASE_URL": "secret://db"},
            detach=False,
            remove=True
        )
        env_output = test_container.decode('utf-8')
        assert "DATABASE_URL=secret://db" in env_output, "Secrets not passed via env"
        
        logger.info("Security configurations test passed")

    def test_docker_compose_integration(self, docker_client):
        """Test full Docker Compose deployment."""
        compose_file = Path(__file__).parent.parent.parent / "deployment" / "docker" / "docker-compose.yml"
        
        # Start compose
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0, f"Docker Compose up failed: {result.stderr}"
        
        # Wait for healthy
        time.sleep(30)
        
        # Check services
        ps_result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "ps"],
            capture_output=True,
            text=True
        )
        assert "Up" in ps_result.stdout, "Services not up"
        
        # Check GAM service logs
        logs_result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "logs", "gam"],
            capture_output=True,
            text=True
        )
        assert "started" in logs_result.stdout.lower() or "running" in logs_result.stdout.lower(), "GAM not started"
        
        # Shutdown
        subprocess.run(["docker", "compose", "-f", str(compose_file), "down"], check=True)
        
        logger.info("Docker Compose integration test passed")