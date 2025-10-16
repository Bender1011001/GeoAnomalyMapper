#!/usr/bin/env python3
"""Dynamic SNAP graph generation for Sentinel-1 interferometry."""

import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

class GraphTemplateProcessor:
    def __init__(self, template_path: str, config: Optional[Dict] = None):
        """
        Initialize the template processor.

        :param template_path: Path to the SNAP graph XML template file.
        :param config: Optional dict for parameter overrides (e.g., {'subswath': 'IW2', 'polarization': 'VH'}).
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        self.config = config or {}
        self.template_content = self.template_path.read_text(encoding='utf-8')
        self._template = Template(self.template_content)
        logger.info(f"Loaded template from {template_path}")

    def extract_sentinel1_params(self, safe_path: str) -> Dict:
        """
        Extract processing parameters from Sentinel-1 .SAFE directory.

        Parses manifest.safe for mode/orbit/AOI, scans measurement/ for subswaths/pols via TIFF names,
        parses annotation XML for burst details. Computes defaults: first common subswath, pol, burst range.

        :param safe_path: Path to .SAFE directory.
        :return: Dict of params (e.g., {'mode': 'IW', 'subswath': 'IW1', 'polarization': 'VV', 'first_burst': 1, 'last_burst': 9}).
        :raises ValueError: If .SAFE structure invalid or key metadata missing.
        """
        safe_dir = Path(safe_path)
        if not safe_dir.exists() or not safe_dir.is_dir():
            raise ValueError(f"Invalid .SAFE directory: {safe_path}")

        params = {}
        logger.info(f"Extracting params from {safe_path}")

        # Parse manifest.safe for mode, orbit, AOI
        manifest_path = safe_dir / 'manifest.safe'
        if manifest_path.exists():
            try:
                tree = ET.parse(manifest_path)
                root = tree.getroot()
                # Mode from filename or <platformOperator> / acquisition mode (standard tag: <mode> in <acquisition>)
                filename_mode = self._extract_mode_from_filename(safe_dir.name)
                params['mode'] = filename_mode  # Fallback to filename if XML parse fails
                # Orbit from <orbitReference> or auxiliary/
                orbit_elem = root.find('.//{*}orbitReference')
                params['orbit_file'] = orbit_elem.text if orbit_elem is not None else 'Sentinel Precise (Auto Download)'
                # AOI from <frameSet> bounds (simplified to WKT or bbox tuple)
                frame_set = root.find('.//{*}frameSet')
                if frame_set is not None:
                    # Extract approx bbox from first <frame> <footprint>
                    footprint = frame_set.find('.//{*}footprint')
                    if footprint is not None:
                        coords = [float(c) for c in re.findall(r'[-+]?\d*\.?\d+', footprint.text or '')]
                        if len(coords) >= 4:
                            params['aoi_bbox'] = (min(coords[0::2]), min(coords[1::2]), max(coords[0::2]), max(coords[1::2]))
                logger.debug(f"Manifest parsed: mode={params.get('mode')}, orbit={params.get('orbit_file')}")
            except ET.ParseError as e:
                logger.warning(f"Failed to parse manifest.safe: {e}. Using filename heuristics.")
                params['mode'] = self._extract_mode_from_filename(safe_dir.name)
                params['orbit_file'] = 'Sentinel Precise (Auto Download)'
        else:
            raise ValueError(f"manifest.safe not found in {safe_path}")

        # Detect subswaths and polarizations from measurement TIFFs
        measurement_dir = safe_dir / 'measurement'
        if measurement_dir.exists():
            tiff_files = list(measurement_dir.glob('s1*-slc-*.tiff'))
            subswaths = set()
            pols = set()
            for tiff in tiff_files:
                match = re.match(r's1[a-z]-\s*(iw|ew|sm)(\d+)-slc-([vh]+)-\d+', tiff.name.lower())
                if match:
                    sub_type = match.group(1).upper()
                    sub_num = match.group(2)
                    pol = match.group(3).upper()
                    subswaths.add(f"{sub_type}{sub_num}")
                    pols.add(pol)
            params['available_subswaths'] = sorted(list(subswaths)) or ['IW1']  # Default
            params['available_polarizations'] = sorted(list(pols)) or ['VV']  # Default
            logger.debug(f"Detected subswaths: {params['available_subswaths']}, pols: {params['available_polarizations']}")
        else:
            raise ValueError(f"measurement/ dir not found in {safe_path}")

        # Select defaults (override with config)
        params['subswath'] = self.config.get('subswath', params['available_subswaths'][0] if params['available_subswaths'] else 'IW1')
        params['polarization'] = self.config.get('polarization', params['available_polarizations'][0] if params['available_polarizations'] else 'VV')

        # Extract bursts from annotation XML (per subswath/pol)
        subswath = params['subswath']
        pol = params['polarization']
        annotation_dir = safe_dir / 'annotation'
        burst_pattern = f"s1[a-z]-{subswath.lower()}-slc-{pol.lower()}-*.xml"
        annotation_files = list(annotation_dir.glob(burst_pattern))
        if not annotation_files:
            # Fallback: assume standard burst range based on mode
            if params['mode'] == 'IW':
                params['first_burst'] = self.config.get('first_burst', 1)
                params['last_burst'] = self.config.get('last_burst', 9)  # Typical IW1
            elif params['mode'] == 'EW':
                params['first_burst'] = self.config.get('first_burst', 1)
                params['last_burst'] = self.config.get('last_burst', 15)  # Wider
            else:  # SM
                params['first_burst'] = self.config.get('first_burst', 0)
                params['last_burst'] = self.config.get('last_burst', 0)
            logger.warning(f"No annotation for {subswath}/{pol}; using fallback bursts {params['first_burst']}-{params['last_burst']}")
            return params

        # Parse first annotation for burst list (all have same structure)
        try:
            tree = ET.parse(annotation_files[0])
            root = tree.getroot()
            burst_list = root.findall('.//{*}burst')
            bursts = []
            for burst in burst_list:
                index = int(burst.find('.//{*}burstIndex').text or '0')
                start_time_str = burst.find('.//{*}sensingStart').text
                end_time_str = burst.find('.//{*}sensingStop').text
                if start_time_str and end_time_str:
                    start = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                    end = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    bursts.append({'index': index, 'start': start, 'end': end})
            bursts.sort(key=lambda b: b['index'])
            params['available_bursts'] = bursts
            # Default range: all bursts (refine in overlap computation later)
            if bursts:
                params['first_burst'] = self.config.get('first_burst', bursts[0]['index'])
                params['last_burst'] = self.config.get('last_burst', bursts[-1]['index'])
            logger.debug(f"Extracted {len(bursts)} bursts for {subswath}/{pol}")
        except (ET.ParseError, AttributeError, ValueError) as e:
            logger.warning(f"Failed to parse bursts: {e}. Using config or defaults.")
            params['first_burst'] = self.config.get('first_burst', 1)
            params['last_burst'] = self.config.get('last_burst', 9)

        # Other defaults
        params['input_file'] = str(safe_dir)  # Full .SAFE as input
        params['output_paths'] = self.config.get('output_paths', 'interferogram')
        params['dem_name'] = self.config.get('dem_name', 'SRTM 1Sec HGT')

        valid_subswaths = {
            'IW1', 'IW2', 'IW3',
            'EW1', 'EW2', 'EW3', 'EW4', 'EW5',
            'SM'
        }
        if params['subswath'] not in valid_subswaths:
            raise ValueError(f"Invalid subswath detected: {params['subswath']}")
        logger.info(f"Extraction complete: { {k: v for k, v in params.items() if k != 'available_bursts'} }")
        return params

    def _extract_mode_from_filename(self, filename: str) -> str:
        """Heuristic mode extraction from .SAFE filename (e.g., 'S1A_IW_SLC' → 'IW')."""
        match = re.search(r'(IW|EW|SM)', filename.upper())
        return match.group(1) if match else 'IW'  # Default IW

    def validate_parameters(self, params: Dict, master_safe: str, slave_safe: str) -> bool:
        """
        Validate extracted parameters for processing compatibility.

        Checks: matching subswath/pol across pairs, valid burst range in available bursts,
        overlapping bursts (compute intersection), consistent mode/AOI overlap.

        :param params: Params from extraction (updated for pair).
        :param master_safe: Master .SAFE path.
        :param slave_safe: Slave .SAFE path.
        :return: True if valid.
        :raises ValueError: On inconsistencies (e.g., "Subswath IW1 not in slave").
        """
        master_params = self.extract_sentinel1_params(master_safe) if 'subswath' not in params else params
        slave_params = self.extract_sentinel1_params(slave_safe)

        # Check mode consistency
        if master_params['mode'] != slave_params['mode']:
            raise ValueError(f"Mode mismatch: master {master_params['mode']} vs slave {slave_params['mode']}")

        # Check subswath/pol availability
        if params.get('subswath') not in slave_params['available_subswaths']:
            raise ValueError(f"Subswath {params['subswath']} not available in slave: {slave_params['available_subswaths']}")
        if params.get('polarization') not in slave_params['available_polarizations']:
            raise ValueError(f"Polarization {params['polarization']} not available in slave: {slave_params['available_polarizations']}")

        # Compute burst overlap
        master_bursts = master_params.get('available_bursts', [])
        slave_bursts = slave_params.get('available_bursts', [])
        overlapping_bursts = self._compute_burst_overlap(master_bursts, slave_bursts)
        if not overlapping_bursts:
            raise ValueError("No overlapping bursts between master and slave scenes")

        # Update params with overlap
        params['first_burst'] = min(overlapping_bursts)
        params['last_burst'] = max(overlapping_bursts)
        if params['first_burst'] > params['last_burst']:
            raise ValueError("Invalid burst range after overlap computation")

        # AOI overlap check (simplified bbox intersection)
        if 'aoi_bbox' in master_params and 'aoi_bbox' in slave_params:
            if not self._bbox_overlap(master_params['aoi_bbox'], slave_params['aoi_bbox']):
                logger.warning("Minimal AOI overlap; processing may have low coverage")

        logger.info(f"Validation passed: bursts {params['first_burst']}-{params['last_burst']}, overlap OK")
        return True

    def _compute_burst_overlap(self, master_bursts: List[Dict], slave_bursts: List[Dict]) -> List[int]:
        """Compute overlapping burst indices by time intersection."""
        overlapping = []
        for m_burst in master_bursts:
            for s_burst in slave_bursts:
                if (m_burst['start'] <= s_burst['end'] and m_burst['end'] >= s_burst['start']):
                    overlapping.append(m_burst['index'])  # Assume indices align; refine if needed
        return sorted(set(overlapping))

    def _bbox_overlap(self, bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]) -> bool:
        """Check if two bboxes (minx, miny, maxx, maxy) overlap."""
        minx1, miny1, maxx1, maxy1 = bbox1
        minx2, miny2, maxx2, maxy2 = bbox2
        return not (maxx1 < minx2 or maxx2 < minx1 or maxy1 < miny2 or maxy2 < miny1)

    def generate_graph(self, template_params: Dict, output_path: str) -> str:
        """
        Generate SNAP graph with substituted parameters.

        :param template_params: Dict of params to substitute (e.g., {'SUBSWATH': 'IW1'}).
        :param output_path: Output XML path.
        :return: Path to generated graph.
        """
        # Map to template keys (upper case with _)
        subst_dict = {k.upper(): str(v) for k, v in template_params.items()}
        subst_dict['MASTER'] = template_params.get('input_file', '${master}')
        subst_dict['SLAVE'] = template_params.get('input_file', '${slave}')  # Update for slave in full process
        subst_dict['OUTPUT'] = f"{template_params.get('output_paths', 'interferogram')}.tif"

        try:
            generated = self._template.substitute(subst_dict)
            output = Path(output_path)
            output.write_text(generated, encoding='utf-8')
            logger.info(f"Generated graph: {output_path} with params {subst_dict}")
            return str(output)
        except KeyError as e:
            raise ValueError(f"Missing template param: {e}")

    def process_interferogram(self, master_safe: str, slave_safe: str, output_dir: str, manual_params: Optional[Dict] = None) -> Dict:
        """
        Complete interferogram processing with automatic parameter detection.

        Extracts params from master (assumes slave compatible), validates pair, generates/runs graph via GPT.

        :param master_safe: Master .SAFE path.
        :param slave_safe: Slave .SAFE path.
        :param output_dir: Output directory.
        :param manual_params: Optional overrides (takes precedence over auto).
        :return: Dict with results (graph_path, output_files, params_used).
        :raises ValueError: On extraction/validation/execution failure.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract (use manual if provided)
        params = manual_params or self.extract_sentinel1_params(master_safe)
        self.validate_parameters(params, master_safe, slave_safe)

        # Generate graph (template must have placeholders like ${SUBSWATH}, ${FIRST_BURST}, etc.)
        graph_path = output_dir / 'dynamic_snap_graph.xml'
        self.generate_graph(params, str(graph_path))

        # Run via GPT (assume gpt in PATH; add snap install check if needed)
        cmd = [
            'gpt', str(graph_path),
            '-Pmaster=' + master_safe,
            '-Pslave=' + slave_safe,
            '-Poutput=' + str(output_dir / f"{params['output_paths']}.tif")
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"GPT execution success: {result.stdout}")
            output_file = output_dir / f"{params['output_paths']}.tif"
            return {'graph_path': str(graph_path), 'output_file': str(output_file), 'params_used': params}
        except subprocess.CalledProcessError as e:
            logger.error(f"GPT failed: {e.stderr}")
            raise ValueError(f"SNAP processing failed: {e.stderr}")

# Example template content (save as snap_interferogram_template.xml)
TEMPLATE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<graph id="InSAR_Processing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${MASTER}</file>
    </parameters>
  </node>
  <node id="Read(2)">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${SLAVE}</file>
    </parameters>
  </node>
  <node id="TOPSAR-Split">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>${SUBSWATH}</subswath>
      <selectedPolarisations>${POLARIZATION}</selectedPolarisations>
      <firstBurstIndex>${FIRST_BURST}</firstBurstIndex>
      <lastBurstIndex>${LAST_BURST}</lastBurstIndex>
    </parameters>
  </node>
  <node id="TOPSAR-Split(2)">
    <operator>TOPSAR-Split</operator>
    <sources>
      <sourceProduct refid="Read(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <subswath>${SUBSWATH}</subswath>
      <selectedPolarisations>${POLARIZATION}</selectedPolarisations>
      <firstBurstIndex>${FIRST_BURST}</firstBurstIndex>
      <lastBurstIndex>${LAST_BURST}</lastBurstIndex>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Split"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>${ORBIT_FILE}</orbitType>
    </parameters>
  </node>
  <node id="Apply-Orbit-File(2)">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Split(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <orbitType>${ORBIT_FILE}</orbitType>
    </parameters>
  </node>
  <node id="Back-Geocoding">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
      <sourceProduct.1 refid="Apply-Orbit-File(2)"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>${DEM_NAME}</demName>
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
      <selectedPolarisations>${POLARIZATION}</selectedPolarisations>
    </parameters>
  </node>
  <node id="TopoPhaseRemoval">
    <operator>TopoPhaseRemoval</operator>
    <sources>
      <sourceProduct refid="TOPSAR-Deburst"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <demName>${DEM_NAME}</demName>
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
      <demName>${DEM_NAME}</demName>
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
      <file>${OUTPUT}</file>
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
</graph>"""

# To use, save TEMPLATE_XML to data/processed/insar/snap_interferogram_template.xml
# Then: processor = GraphTemplateProcessor('data/processed/insar/snap_interferogram_template.xml')
# Note: Add 'lxml' to requirements.txt if complex namespaces needed (ElementTree handles basic).