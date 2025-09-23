"""
CLI integration tests for GAM commands.

These tests verify end-to-end CLI functionality: parameter parsing, command execution, output generation,
and integration with core pipeline. Uses CliRunner for Click-based CLI (assumes gam.core.cli uses Click).
Mocks external dependencies to ensure isolation and speed.

Run with: pytest tests/test_cli_integration.py -v -m integration

Commands tested:
- gam run: End-to-end pipeline execution
- gam global: Tiled global processing
- gam config: Configuration management
- gam cache: Cache operations
- gam export: Export functionality

Expected: All tests pass, CLI handles params correctly, outputs match expectations.
"""

import pytest
import yaml
from pathlib import Path
from click.testing import CliRunner
import json

# GAM CLI import (assumes Click-based)
from gam.core.cli import cli

# Fixtures from conftest.py auto-available, but extend for CLI
# mock_external_apis, tmp_output_dir, test_config, tmp_cache_dir

@pytest.mark.integration
class TestCLIIntegration:
    
    def test_cli_run_command(self, mock_external_apis, test_config, tmp_output_dir):
        """
        Test the `gam run` command end-to-end.
        
        Verifies:
        - Parameter parsing (--bbox, --modalities, --output, --no-cache)
        - Pipeline execution via CLI
        - Output file generation (anomalies.csv, map.png)
        - Stdout success message
        """
        # Write temp config for CLI
        config_path = tmp_output_dir / 'test_cli_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Invoke CLI
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'run',
                '--config', str(config_path),
                '--bbox', '29.0 31.0 30.0 32.0',
                '--modalities', 'gravity',
                '--output', str(tmp_output_dir),
                '--no-cache'
            ],
            catch_exceptions=False
        )
        
        # Verify execution
        assert result.exit_code == 0
        assert 'Pipeline completed successfully' in result.stdout
        assert 'Error' not in result.stderr
        
        # Check outputs
        anomalies_file = tmp_output_dir / 'anomalies.csv'
        map_file = tmp_output_dir / 'gravity_map.png'
        assert anomalies_file.exists()
        assert map_file.exists()
        assert len(pd.read_csv(anomalies_file)) > 0
        
        logger.info("CLI run command test passed")
    
    @pytest.mark.parametrize('region', ['test', 'small_global'])
    def test_cli_global_command(self, mock_external_apis, performance_config, tmp_output_dir, region):
        """
        Test the `gam global` command with small region.
        
        Verifies:
        - Tiling and parallel processing params (--region, --tiles, --workers)
        - Merged output generation
        - Stdout reports tiles processed
        """
        # Temp config
        config_path = tmp_output_dir / 'global_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(performance_config, f)
        
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'global',
                '--config', str(config_path),
                '--region', region,
                '--tiles', '2',
                '--workers', '1',
                '--output', str(tmp_output_dir)
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0
        assert f'Tiles processed: 2' in result.stdout
        assert 'Global processing completed' in result.stdout
        
        # Outputs
        global_map = tmp_output_dir / 'global_map.png'
        global_anoms = tmp_output_dir / 'global_anomalies.csv'
        assert global_map.exists()
        assert global_anoms.exists()
        df = pd.read_csv(global_anoms)
        assert len(df) >= 4  # Merged from tiles
        
        logger.info(f"CLI global command test passed for {region}")
    
    def test_cli_config_commands(self, test_config, tmp_path):
        """
        Test configuration management commands: load, validate, modify, save.
        
        Verifies:
        - Profile loading from YAML
        - Validation (no errors)
        - Modification and saving
        """
        # Create temp config file
        config_path = tmp_path / 'original.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        runner = CliRunner()
        # Load and validate
        result_load = runner.invoke(cli, ['config', 'load', str(config_path)], catch_exceptions=False)
        assert result_load.exit_code == 0
        assert 'Config loaded successfully' in result_load.stdout
        assert 'Validation passed' in result_load.stdout
        
        # Modify and save
        modified_path = tmp_path / 'modified.yaml'
        result_set = runner.invoke(
            cli,
            ['config', 'set', 'pipeline.modalities', '["magnetic"]', 'save', str(modified_path)],
            env={'GAM_CONFIG_PATH': str(config_path)},
            catch_exceptions=False
        )
        assert result_set.exit_code == 0
        assert 'Config updated and saved' in result_set.stdout
        
        # Verify change
        with open(modified_path, 'r') as f:
            modified = yaml.safe_load(f)
        assert modified['pipeline']['modalities'] == ['magnetic']
        
        # Negative: Invalid key
        result_invalid = runner.invoke(cli, ['config', 'set', 'invalid.key', 'value'], catch_exceptions=True)
        assert result_invalid.exit_code != 0
        assert 'Invalid config key' in result_invalid.stderr
        
        logger.info("CLI config commands test passed")
    
    def test_cli_cache_commands(self, mock_cache_manager, tmp_cache_dir, tmp_path):
        """
        Test cache management commands: stats, cleanup.
        
        Verifies:
        - Cache statistics reporting
        - Cleanup by age (remove all for age=0)
        """
        # Populate mock cache with files
        (tmp_cache_dir / 'gravity.h5').touch()
        (tmp_cache_dir / 'magnetic.h5').touch()
        
        runner = CliRunner()
        # Stats
        result_stats = runner.invoke(
            cli,
            ['cache', 'stats', '--dir', str(tmp_cache_dir)],
            catch_exceptions=False
        )
        assert result_stats.exit_code == 0
        assert 'Cache files: 2' in result_stats.stdout
        assert 'Total size' in result_stats.stdout
        
        # Cleanup
        result_cleanup = runner.invoke(
            cli,
            ['cache', 'cleanup', '--dir', str(tmp_cache_dir), '--age', '0'],
            catch_exceptions=False
        )
        assert result_cleanup.exit_code == 0
        assert 'Cache cleaned' in result_cleanup.stdout
        assert len(list(tmp_cache_dir.glob('*.h5'))) == 0  # All removed
        
        logger.info("CLI cache commands test passed")
    
    @pytest.mark.parametrize('format', ['csv', 'geojson', 'shp'])
    def test_cli_export_commands(self, tmp_output_dir, format):
        """
        Test export functionality for different formats.
        
        Verifies:
        - Export from input file (assumes anomalies.csv from prior run)
        - Output file generation and validity
        - Format-specific content (e.g., GeoJSON geometry)
        """
        # Create mock input file (from synthetic)
        mock_anoms = pd.DataFrame({
            'lat': [30.5, 31.0], 'lon': [30.0, 30.5],
            'depth': [1.0, 2.0], 'confidence': [0.8, 0.9],
            'anomaly_type': ['test', 'test']
        })
        input_file = tmp_output_dir / 'anomalies.csv'
        mock_anoms.to_csv(input_file, index=False)
        
        output_file = tmp_output_dir / f'export.{format}'
        
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                'export',
                str(input_file),
                '--format', format,
                '--output', str(output_file)
            ],
            catch_exceptions=False
        )
        
        assert result.exit_code == 0
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        
        # Format-specific validation
        if format == 'csv':
            exported = pd.read_csv(output_file)
            assert len(exported) == 2
        elif format == 'geojson':
            with open(output_file, 'r') as f:
                geojson = json.load(f)
            assert 'features' in geojson
            assert len(geojson['features']) == 2
            assert 'geometry' in geojson['features'][0]
        elif format == 'shp':
            # For shapefile, check multiple files (.shp, .shx, .dbf)
            assert (tmp_output_dir / f'export.{format}.shp').exists()
            assert (tmp_output_dir / f'export.{format}.shx').exists()
        
        logger.info(f"CLI export command test passed for {format}")
    
    def test_cli_error_handling(self, tmp_output_dir):
        """Negative case: Invalid command/params raise UsageError."""
        runner = CliRunner()
        result = runner.invoke(cli, ['run', '--invalid-flag'], catch_exceptions=True)
        assert result.exit_code != 0
        assert 'Unknown option' in result.stderr or 'UsageError' in str(result.exception)
        
        logger.info("CLI error handling test passed")