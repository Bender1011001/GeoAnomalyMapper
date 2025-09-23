"""Scalability and performance tests for GeoAnomalyMapper (GAM).
Tests parallel processing, memory usage, global benchmarks, and load handling.
Run with: pytest tests/system/test_scalability.py -v -m scalability

Requires Dask distributed for parallel tests and sufficient resources (8GB+ RAM).
Uses synthetic data scaled for load testing.
"""

import os
import pytest
import time
import psutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dask.distributed import Client, LocalCluster
from dask import delayed

from gam.core.parallel import setup_dask_cluster
from gam.core.pipeline import GAMPipeline
from gam.core.config import load_config
from gam.tests.conftest import synthetic_data_dir, test_region_geojson

# Mark as scalability tests
pytestmark = pytest.mark.scalability

@pytest.fixture(scope="session")
def dask_client():
    """Fixture for Dask distributed client."""
    with LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB') as cluster:
        with Client(cluster) as client:
            yield client

@pytest.fixture
def scalability_config(tmp_path):
    """Test config for scalability testing."""
    config_path = tmp_path / "scalability_config.yaml"
    config_path.write_text("""
global:
  environment: test
  debug: false

database:
  url: sqlite:///:memory:

ingestion:
  parallel_workers: 4
  cache_ttl_seconds: 60

preprocessing:
  parallel_workers: 8
  chunk_size_mb: 256

modeling:
  parallel:
    dask:
      n_workers: 8
      threads_per_worker: 2
      memory_limit_gb: 2
  fusion:
    method: simple

visualization:
  export_formats: ['csv']
  batch_size: 10000

performance:
  timeout: 600
""")
    return config_path

class TestScalability:
    """Scalability tests."""

    def test_parallel_ingestion_scaling(self, dask_client, synthetic_data_dir, scalability_config):
        """Test ingestion scaling with increasing workers."""
        config = load_config(scalability_config)
        pipeline = GAMPipeline(config=config)
        
        # Test with different worker counts
        worker_counts = [1, 2, 4, 8]
        times = []
        
        for n_workers in worker_counts:
            config['ingestion']['parallel_workers'] = n_workers
            start_time = time.time()
            
            # Ingest synthetic data (simulate large dataset)
            results = pipeline.ingest_data(
                data_dir=str(synthetic_data_dir),
                bbox=(0, 0, 10, 10),
                modalities=['gravity']
            )
            
            exec_time = time.time() - start_time
            times.append(exec_time)
            assert len(results) > 0, f"No data ingested with {n_workers} workers"
            
            logger.info(f"Ingestion with {n_workers} workers: {exec_time:.2f}s")
        
        # Check near-linear scaling (time decreases with workers)
        scaling_efficiency = times[0] / times[-1]  # First vs last
        assert scaling_efficiency > 2.0, f"Poor scaling: {scaling_efficiency:.2f}x speedup"
        
        logger.info("Parallel ingestion scaling test passed")

    def test_memory_usage_under_load(self, dask_client, synthetic_data_dir, scalability_config, tmp_path):
        """Test memory usage during modeling with large dataset."""
        config = load_config(scalability_config)
        pipeline = GAMPipeline(config=config)
        
        # Create large synthetic dataset (simulate global scale)
        large_data_dir = tmp_path / "large_synthetic"
        large_data_dir.mkdir()
        
        # Scale up synthetic data (copy multiple times)
        for i in range(10):  # 10x data
            for file in synthetic_data_dir.iterdir():
                (large_data_dir / f"{file.stem}_{i}.json").write_bytes(file.read_bytes())
        
        # Monitor memory before
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        start_time = time.time()
        results = pipeline.run_modeling(
            input_dir=str(large_data_dir),
            bbox=(0, 0, 180, 90),  # Global scale
            modalities=['gravity', 'magnetic']
        )
        exec_time = time.time() - start_time
        
        # Memory after
        mem_after = process.memory_info().rss / 1024**2
        
        mem_increase = mem_after - mem_before
        assert mem_increase < 6000, f"Excessive memory usage: {mem_increase}MB"  # <6GB threshold
        
        assert len(results) > 100, "Insufficient modeling output for large data"
        
        logger.info(f"Memory test: +{mem_increase:.2f}MB, time: {exec_time:.2f}s")

    def test_global_processing_benchmark(self, dask_client, scalability_config):
        """Benchmark global processing performance."""
        config = load_config(scalability_config)
        pipeline = GAMPipeline(config=config)
        
        # Global bbox
        global_bbox = (-180, -90, 180, 90)
        
        start_time = time.time()
        results = pipeline.run_analysis(
            bbox=global_bbox,
            modalities=['gravity'],  # Single modality for benchmark
            use_synthetic=True,  # Avoid real API calls
            output_dir=str(tmp_path)
        )
        exec_time = time.time() - start_time
        
        # Performance threshold (adjust based on hardware)
        assert exec_time < 300, f"Global processing too slow: {exec_time}s"  # <5 min for synthetic
        
        # Verify output scale
        anomalies = results.get('anomalies', [])
        assert len(anomalies) > 1000, "Too few global anomalies detected"
        
        logger.info(f"Global benchmark: {exec_time:.2f}s, {len(anomalies)} anomalies")

    def test_concurrent_user_load(self, synthetic_data_dir, scalability_config, tmp_path):
        """Load test with concurrent users via CLI/API simulation."""
        config = load_config(scalability_config)
        
        # Simulate 10 concurrent users running small analyses
        def run_single_analysis(user_id):
            output_dir = tmp_path / f"user_{user_id}"
            output_dir.mkdir()
            
            cmd = [
                sys.executable, "-m", "gam.core.cli", "run",
                "--config", str(scalability_config),
                "--bbox", "29,31,30,32",
                "--modalities", "gravity",
                "--input-dir", str(synthetic_data_dir),
                "--output-dir", str(output_dir),
                "--quick-test"  # Fast mode for load test
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(run_single_analysis, i) for i in range(10)]
            successes = [f.result() for f in futures]
        
        assert all(successes), f"Concurrent load test failed: {sum(successes)}/10 succeeded"
        
        # Check no resource exhaustion (system level)
        cpu_percent = psutil.cpu_percent(interval=1)
        mem_percent = psutil.virtual_memory().percent
        assert cpu_percent < 90, f"High CPU during load: {cpu_percent}%"
        assert mem_percent < 95, f"High memory during load: {mem_percent}%"
        
        logger.info("Concurrent load test passed with 10 users")

    def test_dask_distributed_scaling(self, dask_client, scalability_config):
        """Test Dask distributed processing scaling."""
        config = load_config(scalability_config)
        config['modeling']['parallel']['dask']['n_workers'] = len(dask_client.scheduler_info()['workers'])
        
        pipeline = GAMPipeline(config=config)
        
        # Delayed tasks for modeling (simulate distributed)
        @delayed
        def process_chunk(chunk_id):
            # Simulate chunk processing
            time.sleep(0.1 * chunk_id)  # Variable time
            return f"chunk_{chunk_id}_processed"
        
        chunks = range(20)  # 20 chunks
        delayed_tasks = [process_chunk(i) for i in chunks]
        results = dask_client.compute(delayed_tasks)
        processed = results.result()
        
        assert len(processed) == 20, "Not all chunks processed"
        assert all("processed" in str(r) for r in processed), "Processing incomplete"
        
        # Check distribution
        worker_stats = dask_client.scheduler_info()['workers']
        assert len(worker_stats) > 1, "Not distributed across workers"
        
        logger.info("Dask distributed scaling test passed")