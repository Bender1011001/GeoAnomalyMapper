#!/usr/bin/env python3
"""
Process Sentinel-1 InSAR Data for Subsurface Anomaly Detection

Processes downloaded Sentinel-1 SLC products to extract:
- Ground deformation (subsidence/uplift)
- Coherence (surface stability)
- Interferometric phase

Integrates with multi-resolution fusion pipeline for enhanced detection.
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
from typing import List, Dict, Optional, Tuple
import json

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling

from utils.insar_tools import (
    apply_gacos_correction,
    apply_coherence_mask,
    project_los_to_vertical
)
from utils.raster_io import write_raster_with_metadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InSARProcessor:
    """Process Sentinel-1 InSAR data for anomaly detection."""

    def __init__(
        self,
        data_dir: Path,
        interferogram_path: Optional[Path] = None,
        coherence_path: Optional[Path] = None,
        incidence_angle_path: Optional[Path] = None,
        gacos_grid: Optional[Path] = None,
        gacos_executable: str = 'gacos',
        coherence_threshold: float = 0.3,
        output_prefix: str = 'sentinel1_stack',
        skip_gacos: bool = False
    ):
        self.data_dir = data_dir
        self.insar_dir = data_dir / 'raw' / 'insar' / 'sentinel1'
        self.output_dir = data_dir / 'processed' / 'insar'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_file = self.insar_dir / '_manifest.jsonl'

        self.interferogram_path = interferogram_path
        self.coherence_path = coherence_path
        self.incidence_angle_path = incidence_angle_path
        self.gacos_grid = gacos_grid
        self.gacos_executable = gacos_executable
        self.coherence_threshold = coherence_threshold
        self.output_prefix = output_prefix
        self.skip_gacos = skip_gacos
    
    def load_manifest(self) -> List[Dict]:
        """Load downloaded scenes from manifest."""
        scenes = []
        if not self.manifest_file.exists():
            logger.warning("Manifest not found at %s", self.manifest_file)
            return scenes

        with open(self.manifest_file, 'r') as f:
            for line in f:
                if line.strip():
                    scenes.append(json.loads(line))
        return scenes
    
    def check_processing_software(self) -> Dict[str, bool]:
        """Check which InSAR processing software is available."""
        software = {}
        
        # Check for SNAP (ESA Sentinel Application Platform)
        try:
            result = subprocess.run(['gpt', '-h'], capture_output=True, timeout=5)
            software['snap'] = result.returncode == 0
        except:
            software['snap'] = False
        
        # Check for ISCE (InSAR Scientific Computing Environment)
        try:
            result = subprocess.run(['isce2', '--version'], capture_output=True, timeout=5)
            software['isce'] = result.returncode == 0
        except:
            software['isce'] = False
        
        # Check for Python InSAR libraries
        try:
            import mintpy
            software['mintpy'] = True
        except:
            software['mintpy'] = False
        
        return software
    
    def create_processing_guide(self, scenes: List[Dict]):
        """Create detailed processing instructions."""
        
        guide_path = self.output_dir / 'INSAR_PROCESSING_GUIDE.md'
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("# InSAR Processing Guide for Anomaly Detection\n\n")
            f.write(f"Downloaded scenes: {len(scenes)}\n")
            f.write(f"Location: {self.insar_dir}\n\n")
            
            f.write("---\n\n")
            f.write("## Processing Options\n\n")
            
            # Option 1: SNAP (Recommended for beginners)
            f.write("### Option 1: SNAP (ESA Sentinel Application Platform)\n\n")
            f.write("**Best for:** Beginners, graphical workflow\n\n")
            f.write("**Installation:**\n")
            f.write("1. Download: https://step.esa.int/main/download/snap-download/\n")
            f.write("2. Install SNAP Desktop\n")
            f.write("3. Install Sentinel-1 Toolbox\n\n")
            f.write("**Processing Steps:**\n")
            f.write("```\n")
            f.write("1. Open SNAP Desktop\n")
            f.write("2. File > Import > SAR Sensors > Sentinel-1 > SLC\n")
            f.write(f"3. Navigate to: {self.insar_dir}\n")
            f.write("4. Select two scenes (master + slave) from same track\n")
            f.write("5. Radar > Interferometric > Products > InSAR Stack Overview\n")
            f.write("6. Create interferogram:\n")
            f.write("   - TOPSAR Split (select same sub-swath and burst)\n")
            f.write("   - Apply Orbit File\n")
            f.write("   - Back-Geocoding\n")
            f.write("   - Interferogram Formation\n")
            f.write("   - TOPSAR Deburst\n")
            f.write("   - TopoPhase Removal (with SRTM DEM)\n")
            f.write("   - Goldstein Phase Filtering\n")
            f.write("   - Coherence Estimation\n")
            f.write("   - Terrain Correction\n")
            f.write("7. Export as GeoTIFF\n")
            f.write("```\n\n")
            
            # Option 2: ISCE (Advanced)
            f.write("### Option 2: ISCE2 (Advanced Processing)\n\n")
            f.write("**Best for:** Advanced users, batch processing\n\n")
            f.write("**Installation:**\n")
            f.write("```bash\n")
            f.write("conda install -c conda-forge isce2\n")
            f.write("```\n\n")
            f.write("**Processing Example:**\n")
            f.write("```python\n")
            f.write("from isce.applications import topsApp\n\n")
            f.write("# Create topsApp.xml config\n")
            f.write("# Run: topsApp.py topsApp.xml\n")
            f.write("```\n\n")
            
            # Option 3: Cloud processing
            f.write("### Option 3: Cloud Processing (Easiest)\n\n")
            f.write("**Best for:** Quick results without local processing\n\n")
            f.write("**COMET LiCSAR Portal:**\n")
            f.write("1. Visit: https://comet.nerc.ac.uk/COMET-LiCS-portal/\n")
            f.write("2. Search for your region\n")
            f.write("3. Download pre-processed interferograms\n")
            f.write("4. Already geocoded and phase-unwrapped!\n\n")
            
            f.write("---\n\n")
            f.write("## Expected Outputs\n\n")
            f.write("For integration with GeoAnomalyMapper, process to get:\n\n")
            f.write("1. **Coherence** (GeoTIFF): Measures surface stability\n")
            f.write("   - High coherence = stable surface\n")
            f.write("   - Low coherence = changes/decorrelation\n\n")
            f.write("2. **Unwrapped Phase** (GeoTIFF): Ground deformation\n")
            f.write("   - Converted to displacement (cm or mm)\n")
            f.write("   - Negative = subsidence (potential voids!)\n")
            f.write("   - Positive = uplift\n\n")
            f.write("3. **Line-of-Sight Displacement** (GeoTIFF):\n")
            f.write("   - Final deformation map\n")
            f.write("   - Resolution: ~5-20 meters\n\n")
            
            f.write("---\n\n")
            f.write("## Integration with Fusion Pipeline\n\n")
            f.write("Once processed, save outputs to:\n")
            f.write(f"```\n{self.output_dir}/\n")
            f.write("├── coherence.tif          # Surface stability\n")
            f.write("├── displacement.tif       # Ground deformation (mm/year)\n")
            f.write("└── processing_metadata.txt\n```\n\n")
            f.write("Then run fusion:\n")
            f.write("```powershell\n")
            f.write("python multi_resolution_fusion.py --include-insar --output with_insar\n")
            f.write("```\n\n")
            
            f.write("---\n\n")
            f.write("## Downloaded Scenes\n\n")
            f.write("```json\n")
            for i, scene in enumerate(scenes, 1):
                f.write(f"{i}. {scene.get('title', 'Unknown')}\n")
                if 'properties' in scene:
                    props = scene['properties']
                    f.write(f"   Start: {props.get('startDate', 'N/A')}\n")
                    if 'tileId' in scene:
                        f.write(f"   Tile: {scene['tileId']}\n")
            f.write("```\n\n")
            
            f.write("---\n\n")
            f.write("## Quick Start (Recommended)\n\n")
            f.write("**If you don't want to process locally:**\n\n")
            f.write("1. Use COMET LiCSAR pre-processed data:\n")
            f.write("   https://comet.nerc.ac.uk/COMET-LiCS-portal/\n\n")
            f.write("2. Download interferograms for your region\n\n")
            f.write("3. Place in data/processed/insar/\n\n")
            f.write("4. Run fusion pipeline\n\n")
            f.write("**Processing time estimates:**\n")
            f.write("- SNAP GUI: 2-4 hours per interferogram\n")
            f.write("- ISCE batch: 1-2 hours per pair\n")
            f.write("- LiCSAR download: 5 minutes\n")
        
        logger.info(f"✓ Processing guide created: {guide_path}")
        return guide_path
    
    def generate_snap_graph(self):
        """Generate SNAP GPT graph XML for batch processing."""

        graph_path = self.output_dir / 'snap_interferogram_graph.xml'
        
        graph_xml = """<?xml version="1.0" encoding="UTF-8"?>
<graph id="InSAR_Processing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${master}</file>
    </parameters>
  </node>
  
  <node id="Read(2)">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${slave}</file>
    </parameters>
  </node>
  
  <node id="TOPSAR-Split">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>IW1</subswath>
      <selectedPolarisations>VV</selectedPolarisations>
      <firstBurstIndex>1</firstBurstIndex>
      <lastBurstIndex>9</lastBurstIndex>
    </parameters>
  </node>
  
  <node id="TOPSAR-Split(2)">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="Read(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>IW1</subswath>
      <selectedPolarisations>VV</selectedPolarisations>
      <firstBurstIndex>1</firstBurstIndex>
      <lastBurstIndex>9</lastBurstIndex>
    </parameters>
  </node>
  
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Split"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
    </parameters>
  </node>
  
  <node id="Apply-Orbit-File(2)">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Split(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
    </parameters>
  </node>
  
  <node id="Back-Geocoding">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
      <sourceProduct.1 refid="Apply-Orbit-File(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>SRTM 1Sec HGT</demName>
      <demResamplingMethod>BICUBIC_INTERPOLATION</demResamplingMethod>
      <resamplingType>BISINC_5_POINT_INTERPOLATION</resamplingType>
    </parameters>
  </node>
  
  <node id="Interferogram">
    <operator>Interferogram</operator>
    <sources>
      <sourceProduct refid="Back-Geocoding"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
      <srpPolynomialDegree>5</srpPolynomialDegree>
      <srpNumberPoints>501</srpNumberPoints>
      <orbitDegree>3</orbitDegree>
      <includeCoherence>true</includeCoherence>
    </parameters>
  </node>
  
  <node id="TOPSAR-Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <sourceProduct refid="Interferogram"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <selectedPolarisations>VV</selectedPolarisations>
    </parameters>
  </node>
  
  <node id="TopoPhaseRemoval">
    <operator>TopoPhaseRemoval</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Deburst"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>SRTM 1Sec HGT</demName>
      <tileExtensionPercent>100</tileExtensionPercent>
      <outputTopoPhaseBand>false</outputTopoPhaseBand>
      <outputElevationBand>false</outputElevationBand>
    </parameters>
  </node>
  
  <node id="GoldsteinPhaseFiltering">
    <operator>GoldsteinPhaseFiltering</operator>
    <sources>
      <sourceProduct refid="TopoPhaseRemoval"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <alpha>1.0</alpha>
      <FFTSizeString>64</FFTSizeString>
      <windowSizeString>3</windowSizeString>
      <useCoherenceMask>false</useCoherenceMask>
      <coherenceThreshold>0.2</coherenceThreshold>
    </parameters>
  </node>
  
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="GoldsteinPhaseFiltering"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>SRTM 1Sec HGT</demName>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>AUTO:42001</mapProjection>
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${output}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
  
  <applicationData id="Presentation">
    <Description/>
    <node id="Read"/>
    <node id="Read(2)"/>
    <node id="TOPSAR-Split"/>
    <node id="TOPSAR-Split(2)"/>
    <node id="Apply-Orbit-File"/>
    <node id="Apply-Orbit-File(2)"/>
    <node id="Back-Geocoding"/>
    <node id="Interferogram"/>
    <node id="TOPSAR-Deburst"/>
    <node id="TopoPhaseRemoval"/>
    <node id="GoldsteinPhaseFiltering"/>
    <node id="Terrain-Correction"/>
    <node id="Write"/>
  </applicationData>
</graph>
"""
        
        with open(graph_path, 'w', encoding='utf-8') as f:
            f.write(graph_xml)
        
        logger.info(f"✓ SNAP graph created: {graph_path}")
        logger.info("  Usage: gpt snap_interferogram_graph.xml -Pmaster=scene1.zip -Pslave=scene2.zip -Poutput=result.tif")

        return graph_path

    def _read_raster(self, path: Path) -> Tuple[np.ndarray, dict]:
        """Read a raster and return array and profile."""

        with rasterio.open(path) as src:
            data = src.read(1)
            profile = src.profile
        return data, profile

    def _resample_to_match(self, source_path: Path, target_profile: dict) -> np.ndarray:
        """Resample a raster to match the target profile."""

        with rasterio.open(source_path) as src:
            destination = np.full(
                (target_profile['height'], target_profile['width']),
                np.nan,
                dtype=np.float32
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan
            )
        return destination

    def run_automated_pipeline(self) -> Optional[Dict[str, Path]]:
        """Execute automated InSAR processing chain when inputs are provided."""

        if not self.interferogram_path:
            logger.info("No interferogram provided; skipping automated InSAR pipeline.")
            return None

        if not Path(self.interferogram_path).exists():
            raise FileNotFoundError(f"Interferogram not found: {self.interferogram_path}")

        working_path = Path(self.interferogram_path)
        lineage = [str(working_path)]

        if self.gacos_grid and not self.skip_gacos:
            corrected_path = self.output_dir / f"{self.output_prefix}_gacos_corrected.tif"
            try:
                logger.info("Applying GACOS correction using %s", self.gacos_executable)
                working_path = apply_gacos_correction(
                    working_path,
                    Path(self.gacos_grid),
                    corrected_path,
                    self.gacos_executable
                )
                lineage.append(str(working_path))
            except FileNotFoundError:
                logger.warning(
                    "GACOS executable '%s' not found. Skipping atmospheric correction.",
                    self.gacos_executable
                )
            except subprocess.CalledProcessError as exc:
                logger.warning(
                    "GACOS correction failed (%s). Proceeding without correction.",
                    exc
                )

        los_data, base_profile = self._read_raster(working_path)
        los_data = los_data.astype(np.float32)

        coherence_data = None
        if self.coherence_path and Path(self.coherence_path).exists():
            logger.info("Loading coherence map: %s", self.coherence_path)
            coherence_data = self._resample_to_match(Path(self.coherence_path), base_profile)
        else:
            if self.coherence_path:
                logger.warning("Coherence map not found: %s", self.coherence_path)

        if coherence_data is not None:
            logger.info("Applying coherence mask with threshold %.2f", self.coherence_threshold)
            los_masked = apply_coherence_mask(los_data, coherence_data, self.coherence_threshold)
        else:
            los_masked = los_data

        incidence_data = None
        if self.incidence_angle_path and Path(self.incidence_angle_path).exists():
            logger.info("Loading incidence angle map: %s", self.incidence_angle_path)
            incidence_data = self._resample_to_match(Path(self.incidence_angle_path), base_profile)
        else:
            if self.incidence_angle_path:
                logger.warning("Incidence angle map not found: %s", self.incidence_angle_path)

        if incidence_data is not None:
            logger.info("Projecting LOS displacement to vertical component")
            vertical = project_los_to_vertical(los_masked, incidence_data)
        else:
            vertical = los_masked

        base_profile = base_profile.copy()
        base_profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': np.nan,
            'compress': 'DEFLATE',
            'predictor': 2,
            'tiled': True,
            'blockxsize': min(512, base_profile['width']),
            'blockysize': min(512, base_profile['height'])
        })

        los_output = self.output_dir / f"{self.output_prefix}_los_masked.tif"
        vertical_output = self.output_dir / f"{self.output_prefix}_vertical_displacement.tif"

        metadata_common = {
            'units': 'mm',
            'crs': str(base_profile['crs']),
            'effective_resolution_meters': abs(base_profile['transform'][0]) * 111000,
            'data_lineage': lineage,
            'raster_profile': base_profile.copy()
        }

        write_raster_with_metadata(
            los_masked,
            los_output,
            {**metadata_common, 'statistic': 'los_masked'}
        )

        write_raster_with_metadata(
            vertical,
            vertical_output,
            {**metadata_common, 'statistic': 'vertical_displacement'}
        )

        logger.info("Automated InSAR processing complete.")
        logger.info("Masked LOS displacement: %s", los_output)
        logger.info("Vertical displacement: %s", vertical_output)

        return {
            'los_masked': los_output,
            'vertical_displacement': vertical_output
        }

    def process(self):
        """Main processing workflow."""

        logger.info("=" * 70)
        logger.info("INSAR DATA PROCESSING")
        logger.info("=" * 70)
        
        # Load scenes
        scenes = self.load_manifest()
        logger.info(f"Found {len(scenes)} downloaded scenes")
        
        # Check software
        software = self.check_processing_software()
        logger.info("\nProcessing software availability:")
        for name, available in software.items():
            status = "✓ Installed" if available else "✗ Not found"
            logger.info(f"  {name.upper()}: {status}")
        
        # Create processing guide
        logger.info("\nCreating processing guide...")
        guide_path = self.create_processing_guide(scenes)
        
        # Generate SNAP graph
        if software.get('snap'):
            logger.info("\nGenerating SNAP processing graph...")
            self.generate_snap_graph()

        pipeline_outputs = self.run_automated_pipeline()
        if pipeline_outputs:
            logger.info("\nAutomated pipeline outputs:")
            for name, path in pipeline_outputs.items():
                logger.info("  %s: %s", name, path)

        logger.info("\n" + "=" * 70)
        logger.info("SETUP COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nProcessing guide: {guide_path}")
        logger.info(f"Downloaded scenes: {len(scenes)}")
        logger.info("\nNEXT STEPS:")
        logger.info("1. Review processing guide")
        logger.info("2. Choose processing method (SNAP/ISCE/LiCSAR)")
        logger.info("3. Process scenes to interferograms")
        logger.info("4. Save outputs to data/processed/insar/")
        logger.info("5. Run fusion: python multi_resolution_fusion.py --include-insar")
        logger.info("=" * 70)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(
        description="Automate Sentinel-1 InSAR processing for GeoAnomalyMapper"
    )
    default_data_dir = Path(__file__).parent.parent / 'data'
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=default_data_dir,
        help=f'Base data directory (default: {default_data_dir})'
    )
    parser.add_argument('--interferogram', type=Path, help='Path to LOS displacement GeoTIFF')
    parser.add_argument('--coherence', type=Path, help='Path to coherence GeoTIFF')
    parser.add_argument('--incidence-angle', type=Path, help='Path to incidence angle GeoTIFF')
    parser.add_argument('--gacos-grid', type=Path, help='Path to GACOS correction grid')
    parser.add_argument('--gacos-exec', type=str, default='gacos', help='GACOS executable name (default: gacos)')
    parser.add_argument('--coherence-threshold', type=float, default=0.3,
                        help='Coherence threshold for masking (default: 0.3)')
    parser.add_argument('--output-prefix', type=str, default='sentinel1_stack',
                        help='Prefix for processed output files (default: sentinel1_stack)')
    parser.add_argument('--skip-gacos', action='store_true', help='Skip GACOS correction even if provided')

    args = parser.parse_args()

    try:
        processor = InSARProcessor(
            data_dir=args.data_dir,
            interferogram_path=args.interferogram,
            coherence_path=args.coherence,
            incidence_angle_path=args.incidence_angle,
            gacos_grid=args.gacos_grid,
            gacos_executable=args.gacos_exec,
            coherence_threshold=args.coherence_threshold,
            output_prefix=args.output_prefix,
            skip_gacos=args.skip_gacos
        )
        processor.process()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Make sure you've downloaded Sentinel-1 data first:")
        logger.error("  python download_usa_lower48_FIXED.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()