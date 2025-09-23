"""
Performance benchmarks for GAM pipeline.

These tests measure memory usage, execution time, parallel scaling, and caching benefits.
Uses synthetic data of varying sizes and mocks for determinism. Thresholds are approximate
for CI environments (adjust for hardware). Run with: pytest tests/test_performance.py -v -m performance

Metrics:
- Memory: Peak RSS in MB (<500 for medium data)
- Time: Wall-clock seconds (<2s for medium)
- Scaling: Time decreases with workers
- Cache: 40%+ speedup with caching

Dependencies: pytest, psutil, gc (standard lib).
"""

import pytest
import time
import gc
import psutil
from typing import Callable

# GAM imports
from gam import GAMPipeline

# Fixtures from conftest.py
# performance_config, mock_external_apis, tmp_output_dir, test_bbox

def timer(func: Callable) -> Callable:
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"{func.__name__}: {elapsed:.2f}s")
        return result, elapsed
    return wrapper

def memory_monitor(func: Callable) -> Callable:
    """Decorator to measure peak memory usage (RSS in MB)."""
    def wrapper(*args, **kwargs):
        gc.collect()  # Clean before
        process = psutil.Process()
        start_mem = process.memory_info().rss / 1024**2
        result = func(*args, **kwargs)
        gc.collect()  # Clean after
        peak_mem = max(process.memory_info().rss / 1024**2 for _ in range(3))  # Sample peak
        print(f"{func.__name__}: Peak memory {peak_mem:.2f} MB (start: {start_mem:.2f} MB)")
        return result, peak_mem
    return wrapper

@memory_monitor
@timer
def run_benchmark_pipeline(config, bbox, modalities, output_dir, use_cache=False, workers=1):
    """Benchmark wrapper for pipeline run."""
    mod_config = config.copy()
    mod_config['pipeline']['use_cache'] = use_cache
    mod_config['pipeline']['parallel_workers'] = workers
    pipeline = GAMPipeline.from_config(mod_config)
    results = pipeline.run_analysis(
        bbox=bbox,
        modalities=modalities,
        output_dir=output_dir,
        use_cache=use_cache
    )
    return results

@pytest.mark.performance
class TestPerformanceBenchmarks:
    
    @pytest.fixture(scope="function")
    def scaled_synthetic_data(self, request):
        """Scaled synthetic data for different sizes."""
        size = request.param if hasattr(request, 'param') else 'small'
        if size == 'small':
            grid_size = 10
        elif size == 'medium':
            grid_size = 50
        elif size == 'large':
            grid_size = 100
        else:
            raise ValueError(f"Unknown size: {size}")
        
        # Generate larger RawData (override synthetic_raw_data logic)
        lat = np.linspace(30.0, 32.0, grid_size)
        lon = np.linspace(29.0, 31.0, grid_size)
        values = np.random.normal(9.8, 0.1, (grid_size, grid_size))  # Gravity-like
        da = xr.DataArray(values, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'])
        dataset = da.to_dataset(name='data')
        metadata = {'bbox': (29.0, 31.0, 30.0, 32.0), 'units': 'mGal', 'source': 'synthetic'}
        return RawData(values=dataset, metadata=metadata)
    
    def test_pipeline_memory_usage(self, mock_external_apis, performance_config, tmp_output_dir, test_bbox, scaled_synthetic_data):
        """
        Monitor memory consumption during processing (medium data).
        
        Verifies peak RSS < 500 MB for medium dataset.
        """
        # Use medium size
        request_param = type('Obj', (), {'param': 'medium'})()
        data = scaled_synthetic_data(request_param)
        
        # Run with memory monitor
        results, peak_mem = run_benchmark_pipeline(
            performance_config, test_bbox, ['gravity'], tmp_output_dir, use_cache=False
        )
        
        assert peak_mem < 500  # Threshold for test env
        assert results is not None
        
        logger.info(f"Memory test passed: {peak_mem:.2f} MB peak")
    
    @pytest.mark.parametrize('data_size', ['small', 'medium'])
    def test_pipeline_execution_time(self, data_size, mock_external_apis, performance_config, tmp_output_dir, test_bbox, scaled_synthetic_data):
        """
        Benchmark processing time for different data sizes.
        
        Verifies: small <0.5s, medium <2s.
        """
        # Get scaled data
        request_param = type('Obj', (), {'param': data_size})()
        data = scaled_synthetic_data(request_param)
        
        # Time run
        results, elapsed = run_benchmark_pipeline(
            performance_config, test_bbox, ['gravity'], tmp_output_dir, use_cache=False
        )
        
        if data_size == 'small':
            assert elapsed < 0.5
        elif data_size == 'medium':
            assert elapsed < 2.0
        
        logger.info(f"Time test for {data_size}: {elapsed:.2f}s")
    
    @pytest.mark.parametrize('workers', [1, 2, 4])
    def test_parallel_scaling(self, workers, mock_external_apis, performance_config, tmp_output_dir, test_bbox):
        """
        Test performance scaling with number of workers (medium data).
        
        Verifies time decreases ~linearly (time_1 > time_2 * 1.5 approx).
        """
        # Run for each worker count
        results, elapsed = run_benchmark_pipeline(
            performance_config, test_bbox, ['gravity'], tmp_output_dir, use_cache=False, workers=workers
        )
        
        # Relative scaling (compare to 1 worker baseline)
        if workers == 1:
            baseline = elapsed
            assert baseline < 3.0  # Baseline threshold
        else:
            speedup = baseline / elapsed
            assert speedup > 1.2  # At least 20% faster for 2+, 50% for 4+
            if workers == 4:
                assert speedup > 2.0
        
        logger.info(f"Scaling test for {workers} workers: {elapsed:.2f}s (speedup: {speedup:.2f}x)")
    
    def test_cache_performance(self, mock_cache_manager, tmp_cache_dir, mock_external_apis, performance_config, tmp_output_dir, test_bbox):
        """
        Benchmark caching overhead and benefits.
        
        Verifies: Cached run 40%+ faster than non-cached.
        """
        # Non-cached run
        config_no_cache = performance_config.copy()
        config_no_cache['pipeline']['use_cache'] = False
        results_no, time_no = run_benchmark_pipeline(
            config_no_cache, test_bbox, ['gravity'], tmp_output_dir / 'no_cache', use_cache=False
        )
        
        # Cached run (first populates cache)
        config_cache = performance_config.copy()
        config_cache['pipeline']['use_cache'] = True
        config_cache['ingestion']['cache_dir'] = str(tmp_cache_dir)
        results_cache1, time_cache1 = run_benchmark_pipeline(
            config_cache, test_bbox, ['gravity'], tmp_output_dir / 'cache1', use_cache=True
        )
        
        # Second cached run (resumption)
        results_cache2, time_cache2 = run_benchmark_pipeline(
            config_cache, test_bbox, ['gravity'], tmp_output_dir / 'cache2', use_cache=True
        )
        
        # Verify speedup
        overhead = time_cache1 / time_no  # First cache may have overhead
        assert overhead < 1.1  # <10% overhead for initial cache
        speedup = time_no / time_cache2
        assert speedup > 1.4  # 40%+ faster on resumption
        
        # Same results
        pd.testing.assert_frame_equal(results_no['anomalies'], results_cache2['anomalies'])
        
        logger.info(f"Cache test passed: Overhead {overhead:.2f}x, Speedup {speedup:.2f}x")