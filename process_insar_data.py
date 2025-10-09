#!/usr/bin/env python3
"""
Process Sentinel-1 InSAR Data for Subsurface Anomaly Detection

Processes downloaded Sentinel-1 SLC products to extract:
- Ground deformation (subsidence/uplift)
- Coherence (surface stability)
- Interferometric phase

Integrates with multi-resolution fusion pipeline for enhanced detection.
"""

import logging
import sys
from pathlib import Path
import subprocess
from typing import List, Dict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InSARProcessor:
    """Process Sentinel-1 InSAR data for anomaly detection."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.insar_dir = data_dir / 'raw' / 'insar' / 'sentinel1'
        self.output_dir = data_dir / 'processed' / 'insar'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for manifest
        self.manifest_file = self.insar_dir / '_manifest.jsonl'
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")
    
    def load_manifest(self) -> List[Dict]:
        """Load downloaded scenes from manifest."""
        scenes = []
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
    base_dir = Path(__file__).parent.parent / 'data'
    
    try:
        processor = InSARProcessor(base_dir)
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