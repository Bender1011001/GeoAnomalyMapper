#!/usr/bin/env python3
"""
SNAP Installation and InSAR Batch Processing Script

Automates:
1. SNAP GPT installation check
2. Batch interferogram processing
3. Output organization for fusion pipeline
"""

import logging
import sys
from pathlib import Path
import subprocess
import json
from typing import List, Dict, Tuple
import zipfile
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SNAPInstaller:
    """Handles SNAP installation and configuration."""
    
    def __init__(self):
        self.snap_bin = None
        
    def find_snap(self) -> Path:
        """Locate SNAP GPT executable."""
        
        # Common SNAP installation paths (Windows paths use raw strings or forward slashes)
        possible_paths = [
            Path(r"C:\Program Files\esa-snap\bin\gpt.exe"),  # ESA SNAP (common name)
            Path(r"C:\Program Files\snap\bin\gpt.exe"),      # SNAP (alternate name)
            Path(r"C:\Program Files (x86)\esa-snap\bin\gpt.exe"),
            Path(r"C:\Program Files (x86)\snap\bin\gpt.exe"),
            Path.home() / "snap/bin/gpt.exe",
            Path.home() / "esa-snap/bin/gpt.exe",
            Path("/usr/local/snap/bin/gpt"),
            Path("/opt/snap/bin/gpt"),
        ]
        
        # Check if gpt is in PATH
        try:
            result = subprocess.run(['gpt', '--help'], capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.info("✓ SNAP GPT found in system PATH")
                self.snap_bin = 'gpt'
                return Path('gpt')
        except:
            pass
        
        # Check common paths
        for path in possible_paths:
            if path.exists():
                logger.info(f"✓ SNAP GPT found: {path}")
                self.snap_bin = str(path)
                return path
        
        return None
    
    def provide_installation_instructions(self):
        """Provide SNAP installation instructions."""
        
        logger.info("\n" + "=" * 70)
        logger.info("SNAP NOT FOUND - INSTALLATION REQUIRED")
        logger.info("=" * 70)
        logger.info("")
        logger.info("SNAP (Sentinel Application Platform) is required for InSAR processing.")
        logger.info("")
        logger.info("INSTALLATION STEPS:")
        logger.info("")
        logger.info("1. Download SNAP:")
        logger.info("   https://step.esa.int/main/download/snap-download/")
        logger.info("")
        logger.info("2. Choose your platform:")
        logger.info("   - Windows: snap-installer.exe (~1.5 GB)")
        logger.info("   - Linux: snap-installer.sh")
        logger.info("   - Mac: snap-installer.dmg")
        logger.info("")
        logger.info("3. Run installer:")
        logger.info("   - Accept default installation path")
        logger.info("   - Select 'Sentinel-1 Toolbox' during component selection")
        logger.info("   - Installation takes ~15-20 minutes")
        logger.info("")
        logger.info("4. Verify installation:")
        logger.info("   - Windows: C:\\Program Files\\snap\\bin\\gpt.exe --help")
        logger.info("   - Linux/Mac: /usr/local/snap/bin/gpt --help")
        logger.info("")
        logger.info("5. Re-run this script after installation")
        logger.info("")
        logger.info("=" * 70)
        
        return False


class InSARBatchProcessor:
    """Batch process Sentinel-1 scenes to interferograms."""
    
    def __init__(self, snap_gpt: str, data_dir: Path):
        self.gpt = snap_gpt
        self.data_dir = data_dir
        self.insar_dir = data_dir / 'raw' / 'insar' / 'sentinel1'
        self.output_dir = data_dir / 'processed' / 'insar'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Graph XML path
        self.graph_xml = Path(__file__).parent / 'snap_interferogram_graph.xml'
    
    def load_scenes(self) -> List[Path]:
        """Load list of downloaded Sentinel-1 scenes."""
        
        scenes = []
        if self.insar_dir.exists():
            # Find all .SAFE.zip files
            scenes = list(self.insar_dir.glob('*.SAFE.zip'))
            
            # Also check for extracted .SAFE directories
            scenes.extend([d for d in self.insar_dir.glob('*.SAFE') if d.is_dir()])
        
        logger.info(f"Found {len(scenes)} Sentinel-1 scenes")
        return sorted(scenes)
    
    def extract_scene_info(self, scene_path: Path) -> Dict:
        """Extract metadata from scene filename."""
        
        # Sentinel-1 filename format:
        # S1A_IW_SLC__1SDV_20241008T013439_20241008T013506_055464_06C16F_B190.SAFE
        
        name = scene_path.stem.replace('.SAFE', '')
        parts = name.split('_')
        
        if len(parts) >= 7:
            return {
                'satellite': parts[0],  # S1A or S1B or S1C
                'mode': parts[1],  # IW
                'product_type': parts[2],  # SLC
                'start_time': parts[5],
                'end_time': parts[6],
                'orbit': parts[7] if len(parts) > 7 else None,
                'filename': name
            }
        return {}
    
    def create_scene_pairs(self, scenes: List[Path]) -> List[Tuple[Path, Path]]:
        """Create master-slave pairs for interferometry."""
        
        pairs = []
        
        # Sort by acquisition time
        scene_info = [(s, self.extract_scene_info(s)) for s in scenes]
        scene_info.sort(key=lambda x: x[1].get('start_time', ''))
        
        # Create consecutive pairs
        for i in range(len(scene_info) - 1):
            master = scene_info[i][0]
            slave = scene_info[i + 1][0]
            pairs.append((master, slave))
        
        logger.info(f"Created {len(pairs)} interferometric pairs")
        return pairs
    
    def process_pair(self, master: Path, slave: Path, output_name: str) -> bool:
        """Process one interferometric pair."""
        
        output_path = self.output_dir / f"{output_name}.tif"
        
        if output_path.exists():
            logger.info(f"  ✓ Already processed: {output_name}")
            return True
        
        logger.info(f"  Processing: {master.name} + {slave.name}")
        logger.info(f"  Output: {output_name}.tif")
        logger.info(f"  This will take 1-2 hours...")
        
        # Check if graph XML exists
        if not self.graph_xml.exists():
            logger.error(f"  ✗ Graph XML not found: {self.graph_xml}")
            logger.info("  Creating graph XML...")
            self.create_graph_xml()
        
        # Build GPT command with hardware optimization
        # Use all available CPU cores and maximize memory
        cmd = [
            str(self.gpt),
            str(self.graph_xml),
            f'-Pmaster={master}',
            f'-Pslave={slave}',
            f'-Poutput={output_path}',
            '-c', '16G',  # Cache size (16GB - adjust based on your RAM)
            '-q', '8',    # Parallel threads (8 cores - adjust based on your CPU)
        ]
        
        try:
            logger.info(f"  Running: {' '.join(cmd)}")
            
            # Run GPT (this takes a long time)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            if result.returncode == 0:
                logger.info(f"  ✓ Success: {output_name}.tif")
                
                # Log processing info
                info_file = self.output_dir / f"{output_name}_info.txt"
                with open(info_file, 'w') as f:
                    f.write(f"Master: {master.name}\n")
                    f.write(f"Slave: {slave.name}\n")
                    f.write(f"Processed: {result.stderr if result.stderr else 'OK'}\n")
                
                return True
            else:
                logger.error(f"  ✗ Failed: {output_name}")
                logger.error(f"  Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"  ✗ Timeout: Processing took > 2 hours")
            return False
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            return False
    
    def create_graph_xml(self):
        """Create SNAP processing graph with hardware optimization."""
        
        # Full InSAR processing graph with optimized settings
        graph_content = '''<graph id="InSAR_Processing">
  <version>1.0</version>
  
  <!-- Read master scene -->
  <node id="Read_Master">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${master}</file>
    </parameters>
  </node>
  
  <!-- Read slave scene -->
  <node id="Read_Slave">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${slave}</file>
    </parameters>
  </node>
  
  <!-- Apply precise orbit files -->
  <node id="Apply-Orbit-File-Master">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read_Master"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  
  <node id="Apply-Orbit-File-Slave">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read_Slave"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  
  <!-- Back-geocoding (coregistration) -->
  <node id="Back-Geocoding">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct.1 refid="Apply-Orbit-File-Master"/>
      <sourceProduct.2 refid="Apply-Orbit-File-Slave"/>
    </sources>
    <parameters>
      <demName>SRTM 3Sec</demName>
      <demResamplingMethod>BICUBIC_INTERPOLATION</demResamplingMethod>
      <resamplingType>BISINC_5_POINT_INTERPOLATION</resamplingType>
      <maskOutAreaWithoutElevation>true</maskOutAreaWithoutElevation>
      <outputRangeAzimuthOffset>false</outputRangeAzimuthOffset>
      <outputDerampDemodPhase>false</outputDerampDemodPhase>
    </parameters>
  </node>
  
  <!-- Enhanced spectral diversity (improve coregistration) -->
  <node id="Enhanced-Spectral-Diversity">
    <operator>Enhanced-Spectral-Diversity</operator>
    <sources>
      <sourceProduct refid="Back-Geocoding"/>
    </sources>
    <parameters>
      <fineWinWidthStr>512</fineWinWidthStr>
      <fineWinHeightStr>512</fineWinHeightStr>
      <fineWinAccAzimuth>16</fineWinAccAzimuth>
      <fineWinAccRange>16</fineWinAccRange>
      <fineWinOversampling>128</fineWinOversampling>
      <xCorrThreshold>0.1</xCorrThreshold>
      <cohThreshold>0.3</cohThreshold>
      <numBlocksPerOverlap>10</numBlocksPerOverlap>
      <esdEstimator>Periodogram</esdEstimator>
      <weightFunc>Inv Quadratic</weightFunc>
      <useSuppliedRangeShift>false</useSuppliedRangeShift>
      <useSuppliedAzimuthShift>false</useSuppliedAzimuthShift>
    </parameters>
  </node>
  
  <!-- Interferogram formation -->
  <node id="Interferogram">
    <operator>Interferogram</operator>
    <sources>
      <sourceProduct refid="Enhanced-Spectral-Diversity"/>
    </sources>
    <parameters>
      <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
      <srpPolynomialDegree>5</srpPolynomialDegree>
      <srpNumberPoints>501</srpNumberPoints>
      <orbitDegree>3</orbitDegree>
      <includeCoherence>true</includeCoherence>
      <squarePixel>true</squarePixel>
      <subtractTopographicPhase>false</subtractTopographicPhase>
    </parameters>
  </node>
  
  <!-- TOPSAR deburst -->
  <node id="TOPSAR-Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <sourceProduct refid="Interferogram"/>
    </sources>
    <parameters>
      <selectedPolarisations>VV</selectedPolarisations>
    </parameters>
  </node>
  
  <!-- Topographic phase removal -->
  <node id="TopoPhaseRemoval">
    <operator>TopoPhaseRemoval</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Deburst"/>
    </sources>
    <parameters>
      <demName>SRTM 3Sec</demName>
      <tileExtensionPercent>100</tileExtensionPercent>
      <orbitDegree>3</orbitDegree>
    </parameters>
  </node>
  
  <!-- Goldstein phase filtering -->
  <node id="GoldsteinPhaseFiltering">
    <operator>GoldsteinPhaseFiltering</operator>
    <sources>
      <sourceProduct refid="TopoPhaseRemoval"/>
    </sources>
    <parameters>
      <alpha>1.0</alpha>
      <FFTSizeString>64</FFTSizeString>
      <windowSizeString>3</windowSizeString>
      <useCoherenceMask>false</useCoherenceMask>
      <coherenceThreshold>0.2</coherenceThreshold>
    </parameters>
  </node>
  
  <!-- Multilook (reduce speckle, improve coherence) -->
  <node id="Multilook">
    <operator>Multilook</operator>
    <sources>
      <sourceProduct refid="GoldsteinPhaseFiltering"/>
    </sources>
    <parameters>
      <nRgLooks>4</nRgLooks>
      <nAzLooks>1</nAzLooks>
      <outputIntensity>false</outputIntensity>
      <grSquarePixel>true</grSquarePixel>
    </parameters>
  </node>
  
  <!-- Terrain correction (geocoding) -->
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Multilook"/>
    </sources>
    <parameters>
      <sourceBands/>
      <demName>SRTM 3Sec</demName>
      <externalDEMFile/>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BICUBIC_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BICUBIC_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <pixelSpacingInDegree>9.0E-5</pixelSpacingInDegree>
      <mapProjection>AUTO:42001</mapProjection>
      <alignToStandardGrid>false</alignToStandardGrid>
      <standardGridOriginX>0.0</standardGridOriginX>
      <standardGridOriginY>0.0</standardGridOriginY>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <outputComplex>false</outputComplex>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
      <saveSigmaNought>false</saveSigmaNought>
      <saveGammaNought>false</saveGammaNought>
      <saveBetaNought>false</saveBetaNought>
      <incidenceAngleForSigma0>Use projected local incidence angle from DEM</incidenceAngleForSigma0>
      <incidenceAngleForGamma0>Use projected local incidence angle from DEM</incidenceAngleForGamma0>
      <auxFile>Latest Auxiliary File</auxFile>
    </parameters>
  </node>
  
  <!-- Write output -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters>
      <file>${output}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
  
  <!-- Application properties (hardware optimization) -->
  <applicationData id="Presentation">
    <Description/>
    <node id="Read_Master">
      <displayPosition x="50.0" y="50.0"/>
    </node>
    <node id="Read_Slave">
      <displayPosition x="50.0" y="150.0"/>
    </node>
  </applicationData>
</graph>'''
        
        # Write the graph XML file
        with open(self.graph_xml, 'w', encoding='utf-8') as f:
            f.write(graph_content)
        
        logger.info(f"  ✓ Graph XML created: {self.graph_xml}")
    
    def batch_process(self):
        """Process all scene pairs."""
        
        logger.info("\n" + "=" * 70)
        logger.info("BATCH INSAR PROCESSING")
        logger.info("=" * 70)
        
        # Load scenes
        scenes = self.load_scenes()
        
        if len(scenes) < 2:
            logger.error("Need at least 2 scenes for interferometry")
            logger.error(f"Found only {len(scenes)} scenes")
            return False
        
        # Create pairs
        pairs = self.create_scene_pairs(scenes)
        
        if not pairs:
            logger.error("No valid pairs could be created")
            return False
        
        # Process each pair
        logger.info(f"\nProcessing {len(pairs)} interferometric pairs...")
        logger.info("Each pair takes 1-2 hours to process\n")
        
        successful = 0
        failed = 0
        
        for i, (master, slave) in enumerate(pairs, 1):
            output_name = f"interferogram_{i:02d}"
            
            logger.info(f"\n[Pair {i}/{len(pairs)}] {output_name}")
            
            if self.process_pair(master, slave, output_name):
                successful += 1
            else:
                failed += 1
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Successful: {successful}/{len(pairs)}")
        logger.info(f"Failed: {failed}/{len(pairs)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("1. Review processed interferograms in output directory")
        logger.info("2. Run fusion with InSAR data:")
        logger.info("   python multi_resolution_fusion.py --include-insar --output with_insar")
        logger.info("3. Validate results:")
        logger.info("   python validate_against_known_features.py data/outputs/multi_resolution/with_insar.tif")
        logger.info("=" * 70)
        
        return successful > 0


def main():
    """Main installation and processing workflow."""
    
    logger.info("\n" + "=" * 70)
    logger.info("SNAP INSTALLATION & INSAR PROCESSING")
    logger.info("=" * 70)
    
    # Step 1: Check for SNAP
    logger.info("\nStep 1: Checking for SNAP installation...")
    installer = SNAPInstaller()
    snap_path = installer.find_snap()
    
    if not snap_path:
        installer.provide_installation_instructions()
        sys.exit(1)
    
    # Step 2: Initialize processor
    logger.info("\nStep 2: Initializing InSAR processor...")
    base_dir = Path(__file__).parent.parent / 'data'
    processor = InSARBatchProcessor(installer.snap_bin, base_dir)
    
    # Step 3: Batch process
    logger.info("\nStep 3: Starting batch processing...")
    logger.info("WARNING: This will take several hours!")
    logger.info("Each interferogram takes 1-2 hours to process")
    logger.info("")
    
    input("Press Enter to start processing, or Ctrl+C to cancel...")
    
    success = processor.batch_process()
    
    if success:
        logger.info("\n✓ Processing completed successfully!")
        sys.exit(0)
    else:
        logger.error("\n✗ Processing failed")
        sys.exit(1)


if __name__ == '__main__':
    main()