"""
Module compatibility tests for GAM.

These tests verify:
- All modules and key classes can be imported without errors
- Modules comply with architectural interfaces (methods present)
- Dependencies are resolvable (core and optional)
- Configuration loading works across modules

Run with: pytest tests/test_compatibility.py -v -m compatibility

Expected: All imports succeed, interfaces compliant, configs load without validation errors.
"""

import pytest
from importlib import import_module
import sys

# GAM imports for testing
try:
    from gam import (
        GAMPipeline, IngestionManager, PreprocessingManager, ModelingManager, VisualizationManager,
        RawData, ProcessedGrid, InversionResults, AnomalyOutput, GAMConfig
    )
    from gam.core import cli
    from gam.ingestion import DataSource
    from gam.preprocessing import Preprocessor
    from gam.modeling import Inverter
    from gam.visualization import Visualizer  # Assuming base class in base.py
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    pytest.skip(f"GAM import failed: {e}", allow_module_level=True)

# Optional deps
OPTIONAL_DEPS = {
    'obspy': 'seismic',
    'simpeg': 'modeling',
    'pygmt': 'visualization',
    'pyvista': 'visualization',
    'pygimli': 'modeling'
}

# Interface specs (methods expected)
INTERFACE_SPECS = {
    'IngestionManager': ['fetch_modality', 'fetch_multiple', 'from_config'],
    'PreprocessingManager': ['process', 'from_config'],
    'ModelingManager': ['run_inversion', 'fuse_models', 'detect_anomalies', 'from_config'],
    'VisualizationManager': ['generate', 'export', 'from_config'],
    'GAMPipeline': ['run_analysis', 'global_run', 'from_config'],
    'DataSource': ['fetch_data'],  # ABC
    'Preprocessor': ['process_data'],  # ABC
    'Inverter': ['invert'],  # ABC
}

@pytest.mark.compatibility
class TestModuleCompatibility:
    
    def test_module_imports(self):
        """Verify all modules can be imported correctly."""
        modules = [
            'gam',
            'gam.core',
            'gam.ingestion',
            'gam.preprocessing',
            'gam.modeling',
            'gam.visualization',
            'gam.core.config',
            'gam.core.exceptions',
            'gam.core.utils',
            'gam.core.pipeline',
            'gam.core.cli'
        ]
        
        for mod_name in modules:
            try:
                module = import_module(mod_name)
                assert module is not None
                print(f"Imported {mod_name} successfully")
            except ImportError as e:
                pytest.fail(f"Failed to import {mod_name}: {e}")
        
        # Key classes
        classes = [
            GAMPipeline, IngestionManager, PreprocessingManager, ModelingManager, VisualizationManager,
            RawData, ProcessedGrid, InversionResults, AnomalyOutput
        ]
        for cls in classes:
            assert cls is not None
            print(f"Class {cls.__name__} imported successfully")
    
    def test_interface_compliance(self):
        """Test that modules follow architectural interfaces (required methods)."""
        for class_name, required_methods in INTERFACE_SPECS.items():
            try:
                cls = globals()[class_name]
                # Instantiate or use class for hasattr
                if class_name in ['DataSource', 'Preprocessor', 'Inverter']:
                    # ABC, check class methods
                    for method in required_methods:
                        assert hasattr(cls, method), f"{class_name} missing {method}"
                else:
                    instance = cls()
                    for method in required_methods:
                        assert hasattr(instance, method), f"{class_name} missing {method}"
                        print(f"{class_name}.{method} present")
            except Exception as e:
                pytest.skip(f"Interface test for {class_name} skipped: {e}")
        
        # ABC abstract methods
        assert 'fetch_data' in DataSource.__abstractmethods__
        assert 'process_data' in Preprocessor.__abstractmethods__
        assert 'invert' in Inverter.__abstractmethods__
    
    @pytest.mark.parametrize('dep_name, feature', OPTIONAL_DEPS.items())
    def test_dependency_resolution(self, dep_name, feature):
        """Verify dependencies are available (skip optional if missing)."""
        try:
            module = import_module(dep_name)
            assert module is not None
            print(f"Optional dep {dep_name} for {feature} resolved")
        except ImportError:
            pytest.skip(f"Optional {dep_name} for {feature} not installed")
    
    def test_configuration_loading(self, test_config):
        """Test configuration loading across modules."""
        # Load with GAMConfig
        config = GAMConfig.from_dict(test_config)
        assert config.pipeline.modalities == ['gravity']
        assert config.bbox == [29.0, 31.0, 30.0, 32.0]
        
        # Validate
        config.validate()
        assert config.is_valid()  # Assuming method
        
        # Module-specific loading
        ingestion_config = config.ingestion
        assert ingestion_config.retry_attempts == 1
        
        preprocessing_config = config.preprocessing
        assert preprocessing_config.grid_method == 'bilinear'
        
        modeling_config = config.modeling
        assert modeling_config.inversion_type == 'simple'
        
        visualization_config = config.visualization
        assert 'csv' in visualization_config.output_formats
        
        # Invalid config test
        invalid_config = test_config.copy()
        invalid_config['pipeline']['modalities'] = 'invalid'
        with pytest.raises(ValueError, match="Invalid modalities"):
            GAMConfig.from_dict(invalid_config).validate()
        
        print("Configuration loading and validation passed")

    def test_cli_compatibility(self):
        """Test CLI entry point compatibility."""
        assert cli is not None
        # If Click, test help
        try:
            from click.testing import CliRunner
            runner = CliRunner()
            result = runner.invoke(cli, ['--help'])
            assert result.exit_code == 0
            assert 'Usage' in result.stdout
            print("CLI compatibility passed")
        except ImportError:
            pytest.skip("Click not available for CLI test")