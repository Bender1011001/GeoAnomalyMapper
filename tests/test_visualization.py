"""Unit tests for GAM visualization module.

Tests cover all components: base classes, generators, exporters, utilities, manager,
and reports. Uses synthetic data for validation. Run with pytest.

Dependencies: pytest, numpy, pandas, xarray, matplotlib (for fig checks),
rasterio, pyvista, folium, pygmt (skipped if not installed).
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import os
import tempfile
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

# Import visualization module
from gam.visualization import (
    VisualizationManager, StaticMapGenerator, InteractiveMapGenerator,
    VolumeRenderer, CrossSectionGenerator, StatisticalPlots, ComparisonPlots, ProfilePlots,
    GeoTIFFExporter, VTKExporter, DatabaseExporter, JSONExporter, CSVExporter,
    ColorSchemes, SymbolStyles, LayoutManager, TextAnnotator, ShapeOverlay, ScaleIndicator,
    ReportGenerator, HTMLReportGenerator, Visualizer
)
from gam.preprocessing.data_structures import ProcessedGrid
from gam.modeling.data_structures import InversionResults, AnomalyOutput

# Synthetic data fixtures
@pytest.fixture
def mock_processed_grid():
    """Mock 2D ProcessedGrid."""
    data = np.random.rand(10, 10)
    lats = np.linspace(40, 41, 10)
    lons = np.linspace(-100, -99, 10)
    ds = xr.Dataset({'data': (['lat', 'lon'], data)}, coords={'lat': lats, 'lon': lons})
    ds.attrs = {'units': 'mGal', 'coordinate_system': 'EPSG:4326'}
    return ProcessedGrid(ds)

@pytest.fixture
def mock_inversion_results():
    """Mock InversionResults."""
    model = np.random.rand(5, 5, 3)
    uncertainty = np.random.rand(5, 5, 3) * 0.1
    metadata = {'units': 'kg/m³', 'converged': True, 'crs': 'EPSG:4326'}
    return InversionResults(model, uncertainty, metadata)

@pytest.fixture
def mock_anomaly_output():
    """Mock AnomalyOutput."""
    data = pd.DataFrame({
        'lat': np.random.uniform(40, 41, 5),
        'lon': np.random.uniform(-100, -99, 5),
        'depth': np.random.uniform(0, 1000, 5),
        'confidence': np.random.uniform(0.5, 1.0, 5),
        'anomaly_type': np.random.choice(['void', 'fault'], 5),
        'strength': np.random.uniform(-2, 2, 5)
    })
    return AnomalyOutput(data)

@pytest.fixture
def mock_manager():
    """Mock VisualizationManager."""
    return VisualizationManager()


def test_base_visualizer_abstract():
    """Test Visualizer is abstract."""
    with pytest.raises(TypeError):
        Visualizer()  # Should not instantiate


class TestMaps2D:
    def test_static_map_generator(self, mock_processed_grid):
        """Test StaticMapGenerator."""
        gen = StaticMapGenerator()
        fig = gen.generate(mock_processed_grid, projection='Mercator')
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_interactive_map_generator(self, mock_anomaly_output):
        """Test InteractiveMapGenerator."""
        gen = InteractiveMapGenerator()
        m = gen.generate(mock_anomaly_output)
        assert isinstance(m, folium.Map)
        assert len(m._children) > 0  # Markers added


class TestVolume3D:
    def test_volume_renderer(self, mock_inversion_results):
        """Test VolumeRenderer."""
        renderer = VolumeRenderer()
        plotter = renderer.generate(mock_inversion_results, mode='volume')
        assert isinstance(plotter, pv.Plotter)
        assert len(plotter.mesh) > 0

    def test_cross_section_generator(self, mock_inversion_results):
        """Test CrossSectionGenerator."""
        gen = CrossSectionGenerator()
        fig = gen.generate(mock_inversion_results, slice_type='vertical')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlots:
    def test_statistical_plots(self, mock_anomaly_output):
        """Test StatisticalPlots."""
        plots = StatisticalPlots()
        fig = plots.generate(mock_anomaly_output, types=['hist'])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_comparison_plots(self, mock_processed_grid):
        """Test ComparisonPlots with two identical datasets."""
        plots = ComparisonPlots()
        datasets = [mock_processed_grid, mock_processed_grid]
        fig = plots.generate(datasets, type='side_by_side')
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_profile_plots(self, mock_inversion_results):
        """Test ProfilePlots."""
        plots = ProfilePlots()
        fig = plots.generate(mock_inversion_results, line_start=(0,0,0), line_end=(4,4,0))
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestStyling:
    def test_color_schemes(self):
        """Test ColorSchemes."""
        schemes = ColorSchemes()
        cmap = schemes.get_cmap('gravity')
        assert isinstance(cmap, plt.cm.colors.LinearSegmentedColormap)
        with pytest.raises(ValueError):
            schemes.get_cmap('invalid')

    def test_symbol_styles(self):
        """Test SymbolStyles."""
        styles = SymbolStyles()
        style = styles.get_style('void', confidence=0.8, strength=1.0)
        assert 'marker' in style
        assert style['marker'] == 'o'
        assert style['size'] > 0

    def test_layout_manager(self):
        """Test LayoutManager."""
        manager = LayoutManager()
        fig, axs = manager.create_layout(1, 2)
        assert len(axs) == 2
        manager.apply_styles(axs)
        plt.close(fig)


class TestAnnotations:
    def test_text_annotator(self):
        """Test TextAnnotator."""
        fig, ax = plt.subplots()
        annot = TextAnnotator()
        title = annot.add_title(fig, 'Test Title')
        assert isinstance(title, plt.Text)
        label = annot.add_label(ax, (0.5, 0.5), 'Test Label')
        assert isinstance(label, plt.Text)
        plt.close(fig)

    def test_shape_overlay(self):
        """Test ShapeOverlay."""
        fig, ax = plt.subplots()
        overlay = ShapeOverlay()
        rect = overlay.add_bbox(ax, (0, 0, 1, 1))
        assert isinstance(rect, plt.patches.Rectangle)
        circle = overlay.add_circle(ax, (0.5, 0.5), 0.2)
        assert isinstance(circle, plt.patches.Circle)
        plt.close(fig)

    def test_scale_indicator(self):
        """Test ScaleIndicator."""
        fig, ax = plt.subplots()
        indicator = ScaleIndicator()
        bar, text = indicator.add_scale_bar(ax)
        assert isinstance(bar, plt.Line2D)
        assert isinstance(text, plt.Text)
        arrow = indicator.add_north_arrow(ax)
        assert isinstance(arrow, plt.patches.Polygon)
        plt.close(fig)


class TestExporters:
    @pytest.mark.skipif(not Path('test.tif').exists(), reason="Requires rasterio")
    def test_geotiff_exporter(self, mock_processed_grid, tmp_path):
        """Test GeoTIFFExporter."""
        path = tmp_path / 'test.tif'
        exporter = GeoTIFFExporter()
        exporter.export(mock_processed_grid, str(path))
        assert path.exists()
        with rasterio.open(path) as src:
            assert src.count == 1
            assert src.crs.to_string() == 'EPSG:4326'

    def test_vtk_exporter(self, mock_inversion_results, tmp_path):
        """Test VTKExporter."""
        path = tmp_path / 'test.vtk'
        exporter = VTKExporter()
        exporter.export(mock_inversion_results, str(path))
        assert path.exists()
        grid = pv.read(str(path))
        assert 'model' in grid.cell_data

    def test_database_exporter(self, mock_anomaly_output, tmp_path):
        """Test DatabaseExporter."""
        path = tmp_path / 'test.db'
        exporter = DatabaseExporter(str(path))
        exporter.export(mock_anomaly_output, str(path))
        assert path.exists()

    def test_json_exporter(self, mock_anomaly_output, tmp_path):
        """Test JSONExporter."""
        path = tmp_path / 'test.json'
        exporter = JSONExporter()
        exporter.export(mock_anomaly_output, str(path))
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
            assert 'type' in data
            assert 'data' in data

    def test_csv_exporter(self, mock_anomaly_output, tmp_path):
        """Test CSVExporter."""
        path = tmp_path / 'test.csv'
        exporter = CSVExporter()
        exporter.export(mock_anomaly_output, str(path))
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == len(mock_anomaly_output)


class TestManager:
    def test_generate_map(self, mock_manager, mock_anomaly_output):
        """Test VisualizationManager generate_map."""
        result = mock_manager.generate_map(mock_anomaly_output, map_type='interactive')
        assert isinstance(result, folium.Map)

    def test_export_data(self, mock_manager, mock_anomaly_output, tmp_path):
        """Test export_data."""
        path = tmp_path / 'test.csv'
        result = mock_manager.export_data(mock_anomaly_output, str(path), format='csv')
        assert Path(result).exists()

    def test_create_report(self, mock_manager, mock_anomaly_output):
        """Test create_report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pdf'
            result = mock_manager.create_report([mock_anomaly_output], str(path))
            assert Path(result).exists()


class TestReports:
    def test_report_generator(self, mock_anomaly_output):
        """Test ReportGenerator."""
        fig, ax = plt.subplots()
        ax.plot([1,2,3])
        gen = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.pdf'
            result = gen.generate([fig], str(path))
            assert Path(result).exists()

    def test_html_report_generator(self, mock_anomaly_output):
        """Test HTMLReportGenerator."""
        fig, ax = plt.subplots()
        ax.plot([1,2,3])
        gen = HTMLReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test.html'
            result = gen.generate([fig], str(path))
            assert Path(result).exists()
            with open(result) as f:
                assert '<html>' in f.read()


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])