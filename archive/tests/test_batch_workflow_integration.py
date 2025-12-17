import os
import sys
import shutil
import unittest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock torch and other dependencies that might be missing in the test environment
sys.modules['torch'] = MagicMock()
sys.modules['pinn_gravity_inversion'] = MagicMock()
sys.modules['fetch_lithology_density'] = MagicMock()

from batch_workflow import get_tiles, process_tile, merge_tiles

class TestBatchWorkflow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_batch_test")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.input_tif = self.test_dir / "input_gravity.tif"
        self.output_dir = self.test_dir / "output"
        self.temp_dir = self.output_dir / "temp"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy input GeoTIFF
        self.width = 100
        self.height = 100
        self.transform = from_origin(0, 0, 1, 1)
        self.data = np.random.rand(1, self.height, self.width).astype(np.float32)
        
        with rasterio.open(
            self.input_tif,
            'w',
            driver='GTiff',
            height=self.height,
            width=self.width,
            count=1,
            dtype=self.data.dtype,
            crs='EPSG:4326',
            transform=self.transform,
        ) as dst:
            dst.write(self.data)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_get_tiles(self):
        # Test tiling logic
        tile_size = 50
        overlap = 10
        tiles = get_tiles(self.width, self.height, tile_size, overlap)
        
        # Expected tiles:
        # Stride = 40
        # Rows: 0, 40, 80
        # Cols: 0, 40, 80
        # Total 9 tiles
        self.assertEqual(len(tiles), 9)
        
        # Check first tile
        self.assertEqual(tiles[0]['col_off'], 0)
        self.assertEqual(tiles[0]['row_off'], 0)
        self.assertEqual(tiles[0]['width'], 50)
        self.assertEqual(tiles[0]['height'], 50)
        
        # Check last tile (clipped)
        last_tile = tiles[-1]
        self.assertEqual(last_tile['col_off'], 80)
        self.assertEqual(last_tile['row_off'], 80)
        self.assertEqual(last_tile['width'], 20) # 100 - 80
        self.assertEqual(last_tile['height'], 20)

    @patch('batch_workflow.invert_gravity')
    @patch('batch_workflow.fetch_and_rasterize')
    def test_process_tile(self, mock_fetch, mock_invert):
        # Mock external dependencies
        mock_fetch.return_value = True
        
        # Create a dummy output file that invert_gravity would create
        def side_effect_invert(input_path, output_path, lith_path):
            with open(output_path, 'w') as f:
                f.write("dummy content")
        mock_invert.side_effect = side_effect_invert

        tile = {
            "id": "test_tile",
            "col_off": 0,
            "row_off": 0,
            "width": 50,
            "height": 50
        }
        
        success = process_tile(tile, str(self.input_tif), self.output_dir, self.temp_dir)
        
        self.assertTrue(success)
        self.assertTrue((self.output_dir / "test_tile_density.tif").exists())
        
        # Verify mocks called
        mock_fetch.assert_called()
        mock_invert.assert_called()

    @patch('subprocess.run')
    def test_merge_tiles_gdal(self, mock_subprocess):
        # Create dummy tile files
        (self.output_dir / "tile1_density.tif").touch()
        (self.output_dir / "tile2_density.tif").touch()
        
        final_output = str(self.test_dir / "final.tif")
        
        merge_tiles(self.output_dir, final_output)
        
        # Verify subprocess calls for gdalbuildvrt and gdal_translate
        self.assertEqual(mock_subprocess.call_count, 2)
        
        # Check arguments of first call (gdalbuildvrt)
        args, _ = mock_subprocess.call_args_list[0]
        self.assertEqual(args[0][0], "gdalbuildvrt")
        
        # Check arguments of second call (gdal_translate)
        args, _ = mock_subprocess.call_args_list[1]
        self.assertEqual(args[0][0], "gdal_translate")

if __name__ == '__main__':
    unittest.main()