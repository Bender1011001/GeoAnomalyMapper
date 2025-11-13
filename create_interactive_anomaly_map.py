#!/usr/bin/env python3
"""
Interactive Anomaly Map Generator
Creates an interactive map with all detected anomalies marked with pins and explanations
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import folium
from folium import plugins
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies across multiple geophysical datasets."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.processed_dir = data_dir / 'processed'
        self.anomalies = []
    
    def load_insar_anomalies(self) -> List[Dict]:
        """Detect subsidence and uplift anomalies from InSAR data using percentile-based thresholds."""
        logger.info("Processing InSAR data for subsidence and uplift detection...")
        
        insar_dir = self.processed_dir / 'insar'
        displacement_files = list(insar_dir.glob('*_displacement.tif'))
        
        anomalies = []
        
        for disp_file in displacement_files:  # Process ALL files
            try:
                with rasterio.open(disp_file) as src:
                    displacement = src.read(1)
                    transform = src.transform
                    
                    # Remove NaN values for percentile calculation
                    valid_data = displacement[~np.isnan(displacement)]
                    if len(valid_data) < 100:
                        continue
                    
                    # Calculate percentiles for diverse anomaly detection
                    p1 = np.percentile(valid_data, 1)    # Extreme subsidence
                    p5 = np.percentile(valid_data, 5)    # Very high subsidence
                    p10 = np.percentile(valid_data, 10)  # High subsidence
                    p90 = np.percentile(valid_data, 90)  # High uplift
                    p95 = np.percentile(valid_data, 95)  # Very high uplift
                    p99 = np.percentile(valid_data, 99)  # Extreme uplift
                    
                    logger.info(f"  {disp_file.name}: Displacement range {p1:.1f} to {p99:.1f} mm")
                    
                    # Detect SUBSIDENCE anomalies (negative displacement) from different percentile ranges
                    extreme_subsidence = (displacement < p1) & ~np.isnan(displacement)
                    very_high_subsidence = (displacement >= p1) & (displacement < p5) & ~np.isnan(displacement)
                    high_subsidence = (displacement >= p5) & (displacement < p10) & ~np.isnan(displacement)
                    
                    # Detect UPLIFT anomalies (positive displacement) from different percentile ranges
                    extreme_uplift = (displacement > p99) & ~np.isnan(displacement)
                    very_high_uplift = (displacement <= p99) & (displacement > p95) & ~np.isnan(displacement)
                    high_uplift = (displacement <= p95) & (displacement > p90) & ~np.isnan(displacement)
                    
                    # Sample subsidence anomalies with geographic diversity
                    self._sample_anomalies(displacement, transform, extreme_subsidence, anomalies,
                                          'insar_subsidence', 3, 'extreme',
                                          'Extreme ground subsidence detected')
                    self._sample_anomalies(displacement, transform, very_high_subsidence, anomalies,
                                          'insar_subsidence', 4, 'high',
                                          'Significant ground subsidence detected')
                    self._sample_anomalies(displacement, transform, high_subsidence, anomalies,
                                          'insar_subsidence', 5, 'medium',
                                          'Moderate ground subsidence detected')
                    
                    # Sample uplift anomalies with geographic diversity
                    self._sample_anomalies(displacement, transform, extreme_uplift, anomalies,
                                          'insar_uplift', 2, 'extreme',
                                          'Extreme ground uplift detected')
                    self._sample_anomalies(displacement, transform, very_high_uplift, anomalies,
                                          'insar_uplift', 3, 'high',
                                          'Significant ground uplift detected')
                    self._sample_anomalies(displacement, transform, high_uplift, anomalies,
                                          'insar_uplift', 3, 'medium',
                                          'Moderate ground uplift detected')
                    
            except Exception as e:
                logger.warning(f"Error processing {disp_file.name}: {e}")
        
        subsidence_count = sum(1 for a in anomalies if a['type'] == 'insar_subsidence')
        uplift_count = sum(1 for a in anomalies if a['type'] == 'insar_uplift')
        logger.info(f"Found {len(anomalies)} InSAR anomalies: {subsidence_count} subsidence, {uplift_count} uplift")
        return anomalies
    
    def load_gravity_anomalies(self) -> List[Dict]:
        """Detect gravity anomalies using percentile-based thresholds."""
        logger.info("Processing gravity data for anomalies...")
        
        gravity_dir = self.processed_dir / 'gravity'
        gravity_files = list(gravity_dir.glob('*.tif'))
        
        anomalies = []
        
        for grav_file in gravity_files:  # Process ALL files
            try:
                with rasterio.open(grav_file) as src:
                    gravity = src.read(1)
                    transform = src.transform
                    
                    # Remove NaN values
                    valid_data = gravity[~np.isnan(gravity)]
                    if len(valid_data) < 100:
                        continue
                    
                    # Calculate percentiles for diverse anomaly detection
                    p1 = np.percentile(valid_data, 1)    # Extreme low
                    p5 = np.percentile(valid_data, 5)    # Very low
                    p10 = np.percentile(valid_data, 10)  # Low
                    p90 = np.percentile(valid_data, 90)  # High
                    p95 = np.percentile(valid_data, 95)  # Very high
                    p99 = np.percentile(valid_data, 99)  # Extreme high
                    
                    logger.info(f"  {grav_file.name}: Gravity range {p1:.1f} to {p99:.1f} mGal")
                    
                    # Detect LOW gravity anomalies from different percentile ranges
                    extreme_low = (gravity < p1) & ~np.isnan(gravity)
                    very_low = (gravity >= p1) & (gravity < p5) & ~np.isnan(gravity)
                    low = (gravity >= p5) & (gravity < p10) & ~np.isnan(gravity)
                    
                    # Detect HIGH gravity anomalies from different percentile ranges
                    extreme_high = (gravity > p99) & ~np.isnan(gravity)
                    very_high = (gravity <= p99) & (gravity > p95) & ~np.isnan(gravity)
                    high = (gravity <= p95) & (gravity > p90) & ~np.isnan(gravity)
                    
                    # Sample low gravity anomalies
                    self._sample_anomalies(gravity, transform, extreme_low, anomalies,
                                          'gravity_low', 2, 'extreme',
                                          'Extreme low gravity anomaly')
                    self._sample_anomalies(gravity, transform, very_low, anomalies,
                                          'gravity_low', 2, 'high',
                                          'Significant low gravity anomaly')
                    self._sample_anomalies(gravity, transform, low, anomalies,
                                          'gravity_low', 2, 'medium',
                                          'Moderate low gravity anomaly')
                    
                    # Sample high gravity anomalies
                    self._sample_anomalies(gravity, transform, extreme_high, anomalies,
                                          'gravity_high', 2, 'extreme',
                                          'Extreme high gravity anomaly')
                    self._sample_anomalies(gravity, transform, very_high, anomalies,
                                          'gravity_high', 2, 'high',
                                          'Significant high gravity anomaly')
                    self._sample_anomalies(gravity, transform, high, anomalies,
                                          'gravity_high', 2, 'medium',
                                          'Moderate high gravity anomaly')
                    
            except Exception as e:
                logger.warning(f"Error processing {grav_file.name}: {e}")
        
        low_count = sum(1 for a in anomalies if a['type'] == 'gravity_low')
        high_count = sum(1 for a in anomalies if a['type'] == 'gravity_high')
        logger.info(f"Found {len(anomalies)} gravity anomalies: {low_count} low, {high_count} high")
        return anomalies
    
    def load_magnetic_anomalies(self) -> List[Dict]:
        """Detect magnetic anomalies using percentile-based thresholds."""
        logger.info("Processing magnetic data for anomalies...")
        
        mag_dir = self.processed_dir / 'magnetic'
        mag_files = list(mag_dir.glob('*.tif'))
        
        anomalies = []
        
        for mag_file in mag_files:  # Process ALL files
            try:
                with rasterio.open(mag_file) as src:
                    magnetic = src.read(1)
                    transform = src.transform
                    
                    # Remove NaN values
                    valid_data = magnetic[~np.isnan(magnetic)]
                    if len(valid_data) < 100:
                        continue
                    
                    # Calculate percentiles for diverse anomaly detection
                    p1 = np.percentile(valid_data, 1)    # Extreme low
                    p5 = np.percentile(valid_data, 5)    # Very low
                    p10 = np.percentile(valid_data, 10)  # Low
                    p90 = np.percentile(valid_data, 90)  # High
                    p95 = np.percentile(valid_data, 95)  # Very high
                    p99 = np.percentile(valid_data, 99)  # Extreme high
                    
                    logger.info(f"  {mag_file.name}: Magnetic range {p1:.1f} to {p99:.1f} nT")
                    
                    # Detect LOW magnetic anomalies from different percentile ranges
                    extreme_low = (magnetic < p1) & ~np.isnan(magnetic)
                    very_low = (magnetic >= p1) & (magnetic < p5) & ~np.isnan(magnetic)
                    low = (magnetic >= p5) & (magnetic < p10) & ~np.isnan(magnetic)
                    
                    # Detect HIGH magnetic anomalies from different percentile ranges
                    extreme_high = (magnetic > p99) & ~np.isnan(magnetic)
                    very_high = (magnetic <= p99) & (magnetic > p95) & ~np.isnan(magnetic)
                    high = (magnetic <= p95) & (magnetic > p90) & ~np.isnan(magnetic)
                    
                    # Sample low magnetic anomalies
                    self._sample_anomalies(magnetic, transform, extreme_low, anomalies,
                                          'magnetic_low', 2, 'extreme',
                                          'Extreme low magnetic anomaly')
                    self._sample_anomalies(magnetic, transform, very_low, anomalies,
                                          'magnetic_low', 2, 'high',
                                          'Significant low magnetic anomaly')
                    self._sample_anomalies(magnetic, transform, low, anomalies,
                                          'magnetic_low', 1, 'medium',
                                          'Moderate low magnetic anomaly')
                    
                    # Sample high magnetic anomalies
                    self._sample_anomalies(magnetic, transform, extreme_high, anomalies,
                                          'magnetic_high', 2, 'extreme',
                                          'Extreme high magnetic anomaly')
                    self._sample_anomalies(magnetic, transform, very_high, anomalies,
                                          'magnetic_high', 3, 'high',
                                          'Significant high magnetic anomaly')
                    self._sample_anomalies(magnetic, transform, high, anomalies,
                                          'magnetic_high', 3, 'medium',
                                          'Moderate high magnetic anomaly')
                    
            except Exception as e:
                logger.warning(f"Error processing {mag_file.name}: {e}")
        
        low_count = sum(1 for a in anomalies if a['type'] == 'magnetic_low')
        high_count = sum(1 for a in anomalies if a['type'] == 'magnetic_high')
        logger.info(f"Found {len(anomalies)} magnetic anomalies: {low_count} low, {high_count} high")
        return anomalies
    
    def _sample_anomalies(self, data: np.ndarray, transform, mask: np.ndarray,
                         anomalies: List[Dict], anomaly_type: str,
                         num_samples: int, severity: str, description_prefix: str):
        """
        Sample anomalies from a mask with geographic diversity.
        
        Args:
            data: Data array (displacement, gravity, or magnetic)
            transform: Rasterio transform for coordinate conversion
            mask: Boolean mask of anomaly locations
            anomalies: List to append detected anomalies to
            anomaly_type: Type of anomaly (e.g., 'insar_subsidence', 'gravity_high')
            num_samples: Target number of samples to extract
            severity: Severity level ('extreme', 'high', 'medium')
            description_prefix: Prefix for anomaly description
        """
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            return
        
        # Ensure geographic diversity by dividing area into grid cells
        if len(rows) > num_samples:
            # Create spatial bins for geographic diversity
            row_bins = np.linspace(rows.min(), rows.max(), int(np.sqrt(num_samples)) + 1)
            col_bins = np.linspace(cols.min(), cols.max(), int(np.sqrt(num_samples)) + 1)
            
            selected_indices = []
            for i in range(len(row_bins) - 1):
                for j in range(len(col_bins) - 1):
                    # Find points in this grid cell
                    in_cell = ((rows >= row_bins[i]) & (rows < row_bins[i+1]) &
                              (cols >= col_bins[j]) & (cols < col_bins[j+1]))
                    cell_indices = np.where(in_cell)[0]
                    
                    if len(cell_indices) > 0:
                        # Select one random point from this cell
                        selected_indices.append(np.random.choice(cell_indices))
                    
                    if len(selected_indices) >= num_samples:
                        break
                if len(selected_indices) >= num_samples:
                    break
            
            # If we didn't get enough samples, add more randomly
            if len(selected_indices) < num_samples:
                remaining = num_samples - len(selected_indices)
                available = set(range(len(rows))) - set(selected_indices)
                if available:
                    additional = np.random.choice(list(available),
                                                 min(remaining, len(available)),
                                                 replace=False)
                    selected_indices.extend(additional)
        else:
            # If fewer points than samples requested, take all
            selected_indices = range(len(rows))
        
        # Create anomaly entries
        for idx in selected_indices:
            row, col = rows[idx], cols[idx]
            lon, lat = rasterio.transform.xy(transform, row, col)
            value = data[row, col]
            
            # Create type-specific descriptions and explanations
            if 'insar_subsidence' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} mm subsidence'
                explanation = f'The ground here is sinking at {abs(value):.1f} millimeters. This could indicate underground voids, water extraction, mining activity, or natural compaction.'
            elif 'insar_uplift' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} mm uplift'
                explanation = f'The ground here is rising at {value:.1f} millimeters. This could indicate volcanic activity, tectonic forces, groundwater recharge, or other subsurface processes.'
            elif 'gravity_high' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} mGal'
                explanation = 'Higher than normal gravity suggests denser rock underground, possibly indicating mineral deposits, intrusive igneous rocks, or other dense geological structures.'
            elif 'gravity_low' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} mGal'
                explanation = 'Lower than normal gravity suggests less dense material underground - possibly voids, caverns, porous rock formations, or sedimentary basins.'
            elif 'magnetic_high' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} nT'
                explanation = 'Strong magnetic field detected - could indicate iron-rich minerals, volcanic features, mafic intrusions, or other magnetically susceptible geological structures.'
            elif 'magnetic_low' in anomaly_type:
                description = f'{description_prefix}: {value:.1f} nT'
                explanation = 'Weak magnetic field detected - could indicate sedimentary rocks, altered zones, or areas with reduced magnetic mineral content.'
            else:
                description = f'{description_prefix}: {value:.2f}'
                explanation = 'Anomalous geophysical signature detected.'
            
            anomalies.append({
                'type': anomaly_type,
                'lat': lat,
                'lon': lon,
                'value': float(value),
                'severity': severity,
                'description': description,
                'explanation': explanation
            })
    
    def detect_all_anomalies(self) -> List[Dict]:
        """Detect all anomalies across all datasets."""
        logger.info("=" * 70)
        logger.info("ANOMALY DETECTION ACROSS ALL DATASETS")
        logger.info("=" * 70)
        
        all_anomalies = []
        
        # InSAR subsidence
        insar_anomalies = self.load_insar_anomalies()
        all_anomalies.extend(insar_anomalies)
        
        # Gravity anomalies
        gravity_anomalies = self.load_gravity_anomalies()
        all_anomalies.extend(gravity_anomalies)
        
        # Magnetic anomalies
        magnetic_anomalies = self.load_magnetic_anomalies()
        all_anomalies.extend(magnetic_anomalies)
        
        logger.info(f"\nTotal anomalies detected: {len(all_anomalies)}")
        logger.info(f"  - InSAR subsidence: {len(insar_anomalies)}")
        logger.info(f"  - Gravity anomalies: {len(gravity_anomalies)}")
        logger.info(f"  - Magnetic anomalies: {len(magnetic_anomalies)}")
        
        return all_anomalies


class InteractiveMapGenerator:
    """Generate interactive map with anomaly markers."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_map(self, anomalies: List[Dict]) -> str:
        """Create interactive Folium map with anomalies."""
        logger.info("\nCreating interactive map...")
        
        # Calculate center based on anomalies
        if anomalies:
            lats = [a['lat'] for a in anomalies]
            lons = [a['lon'] for a in anomalies]
            center_lat = np.mean(lats)
            center_lon = np.mean(lons)
        else:
            center_lat, center_lon = 37.0, -119.0  # Default to California
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Add additional tile layers
        folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(m)
        
        # Group anomalies by type
        insar_group = folium.FeatureGroup(name='🌍 Ground Movement (InSAR)', show=True)
        gravity_group = folium.FeatureGroup(name='⚖️ Gravity Anomalies', show=True)
        magnetic_group = folium.FeatureGroup(name='🧲 Magnetic Anomalies', show=True)
        
        # Color and icon mapping
        anomaly_config = {
            'insar_subsidence': {
                'color': 'red',
                'icon': 'arrow-down',
                'group': insar_group
            },
            'insar_uplift': {
                'color': 'green',
                'icon': 'arrow-up',
                'group': insar_group
            },
            'gravity_high': {
                'color': 'blue',
                'icon': 'arrow-up',
                'group': gravity_group
            },
            'gravity_low': {
                'color': 'orange',
                'icon': 'arrow-down',
                'group': gravity_group
            },
            'magnetic_high': {
                'color': 'purple',
                'icon': 'magnet',
                'group': magnetic_group
            },
            'magnetic_low': {
                'color': 'lightblue',
                'icon': 'resize-small',
                'group': magnetic_group
            }
        }
        
        # Add markers
        for anomaly in anomalies:
            config = anomaly_config.get(anomaly['type'], {'color': 'gray', 'icon': 'info-sign'})
            
            # Create popup with explanation
            popup_html = f"""
            <div style="width: 300px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: {config['color']};">
                    {anomaly['type'].replace('_', ' ').title()}
                </h4>
                <p style="margin: 5px 0;">
                    <strong>What we detected:</strong><br>
                    {anomaly['description']}
                </p>
                <p style="margin: 5px 0; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                    <strong>What this means:</strong><br>
                    {anomaly['explanation']}
                </p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">
                    <strong>Location:</strong> {anomaly['lat']:.4f}°N, {abs(anomaly['lon']):.4f}°W<br>
                    <strong>Severity:</strong> {anomaly['severity'].upper()}
                </p>
            </div>
            """
            
            folium.Marker(
                location=[anomaly['lat'], anomaly['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{anomaly['type'].replace('_', ' ').title()}: Click for details",
                icon=folium.Icon(
                    color=config['color'],
                    icon=config['icon'],
                    prefix='glyphicon'
                )
            ).add_to(config.get('group', m))
        
        # Add groups to map
        insar_group.add_to(m)
        gravity_group.add_to(m)
        magnetic_group.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 500px; height: 90px; 
                    background-color: white; border: 2px solid grey; 
                    z-index: 9999; font-size: 16px; padding: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <h3 style="margin: 0;">🗺️ Geophysical Anomaly Map</h3>
            <p style="margin: 5px 0; font-size: 12px;">
                Click any marker to learn what we found and what it means.<br>
                Use the layer control (top right) to filter anomaly types.
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Add statistics box
        stats_html = f'''
        <div style="position: fixed; 
                    top: 110px; right: 10px; width: 250px; 
                    background-color: white; border: 2px solid grey; 
                    z-index: 9999; font-size: 12px; padding: 10px;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
            <h4 style="margin: 0 0 10px 0;">📊 Detection Summary</h4>
            <p style="margin: 3px 0;"><strong>Total Anomalies:</strong> {len(anomalies)}</p>
            <p style="margin: 3px 0; color: red;">🔻 Subsidence: {sum(1 for a in anomalies if a['type'] == 'insar_subsidence')}</p>
            <p style="margin: 3px 0; color: green;">🔺 Uplift: {sum(1 for a in anomalies if a['type'] == 'insar_uplift')}</p>
            <p style="margin: 3px 0; color: blue;">⚖️ Gravity: {sum(1 for a in anomalies if 'gravity' in a['type'])}</p>
            <p style="margin: 3px 0; color: purple;">🧲 Magnetic: {sum(1 for a in anomalies if 'magnetic' in a['type'])}</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(stats_html))
        
        # Save map
        output_file = self.output_dir / 'interactive_anomaly_map.html'
        m.save(str(output_file))
        logger.info(f"✓ Interactive map saved: {output_file}")
        
        return str(output_file)
    
    def create_legend(self, anomalies: List[Dict]) -> str:
        """Create a separate legend document."""
        legend_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Anomaly Detection - User Guide</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .anomaly-type {{
            display: flex;
            align-items: center;
            margin: 15px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-left: 5px solid;
            border-radius: 5px;
        }}
        .icon {{
            font-size: 30px;
            margin-right: 15px;
        }}
        .red {{ border-left-color: #e74c3c; }}
        .blue {{ border-left-color: #3498db; }}
        .orange {{ border-left-color: #e67e22; }}
        .purple {{ border-left-color: #9b59b6; }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .stat {{
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            margin: 5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ Geophysical Anomaly Detection Map</h1>
        <p>Understanding What We Found Underground</p>
    </div>
    
    <div class="section">
        <h2>📊 What This Map Shows</h2>
        <p>We've analyzed satellite and geophysical data to detect unusual features underground. Each colored pin on the map represents a different type of anomaly - something that's different from the surrounding area.</p>
        
        <div style="text-align: center; margin: 20px 0;">
            <span class="stat">Total Anomalies: {len(anomalies)}</span>
            <span class="stat">Data Sources: 3</span>
            <span class="stat">Area Covered: USA</span>
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Types of Anomalies</h2>
        
        <div class="anomaly-type red">
            <div class="icon">🌍</div>
            <div>
                <h3 style="margin: 0;">Ground Subsidence (InSAR)</h3>
                <p style="margin: 5px 0;"><strong>What it is:</strong> Areas where the ground is sinking</p>
                <p style="margin: 5px 0;"><strong>What causes it:</strong> Underground voids, water extraction, mining, or natural caverns</p>
                <p style="margin: 5px 0;"><strong>Why it matters:</strong> Could indicate potential sinkhole risk or underground activity</p>
            </div>
        </div>
        
        <div class="anomaly-type blue">
            <div class="icon">⬆️</div>
            <div>
                <h3 style="margin: 0;">High Gravity Anomaly</h3>
                <p style="margin: 5px 0;"><strong>What it is:</strong> Areas with stronger gravitational pull</p>
                <p style="margin: 5px 0;"><strong>What causes it:</strong> Denser rock or mineral deposits underground</p>
                <p style="margin: 5px 0;"><strong>Why it matters:</strong> May indicate valuable mineral deposits or unique geological features</p>
            </div>
        </div>
        
        <div class="anomaly-type orange">
            <div class="icon">⬇️</div>
            <div>
                <h3 style="margin: 0;">Low Gravity Anomaly</h3>
                <p style="margin: 5px 0;"><strong>What it is:</strong> Areas with weaker gravitational pull</p>
                <p style="margin: 5px 0;"><strong>What causes it:</strong> Less dense material - voids, porous rock, or different composition</p>
                <p style="margin: 5px 0;"><strong>Why it matters:</strong> Could indicate caverns, ancient waterways, or subsurface voids</p>
            </div>
        </div>
        
        <div class="anomaly-type purple">
            <div class="icon">🧲</div>
            <div>
                <h3 style="margin: 0;">Magnetic Anomaly</h3>
                <p style="margin: 5px 0;"><strong>What it is:</strong> Areas with unusual magnetic field strength</p>
                <p style="margin: 5px 0;"><strong>What causes it:</strong> Iron-rich minerals, volcanic features, or geological structures</p>
                <p style="margin: 5px 0;"><strong>Why it matters:</strong> Helps identify rock types and geological history</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📖 How to Use the Map</h2>
        <ol>
            <li><strong>Click any marker</strong> to see details about that specific anomaly</li>
            <li><strong>Use the layer control</strong> (top right) to show/hide different anomaly types</li>
            <li><strong>Zoom and pan</strong> to explore different areas in detail</li>
            <li><strong>Switch map styles</strong> using the layer selector for different visualizations</li>
        </ol>
    </div>
    
    <div class="section">
        <h2>🔬 Data Sources</h2>
        <p><strong>InSAR (Interferometric Synthetic Aperture Radar):</strong> Sentinel-1 satellite data measuring ground movement with millimeter precision</p>
        <p><strong>Gravity Data:</strong> Global gravity field measurements showing density variations underground</p>
        <p><strong>Magnetic Data:</strong> Earth's magnetic field variations indicating different rock types</p>
    </div>
    
    <div class="section">
        <h2>⚠️ Important Notes</h2>
        <ul>
            <li>These are preliminary detections based on satellite and geophysical data</li>
            <li>Not all anomalies indicate immediate hazards or important features</li>
            <li>Many anomalies are natural geological formations</li>
            <li>Professional geologists should evaluate areas of concern</li>
        </ul>
    </div>
    
    <div class="section" style="text-align: center; background-color: #2c3e50; color: white;">
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>SAR-project Geophysical Anomaly Detection System</p>
    </div>
</body>
</html>
"""
        
        legend_file = self.output_dir / 'anomaly_guide.html'
        with open(legend_file, 'w', encoding='utf-8') as f:
            f.write(legend_html)
        
        logger.info(f"✓ User guide saved: {legend_file}")
        return str(legend_file)


def main():
    """Main execution."""
    logger.info("=" * 70)
    logger.info("INTERACTIVE ANOMALY MAP GENERATOR")
    logger.info("=" * 70)
    
    # Setup paths
    data_dir = Path('data')
    output_dir = data_dir / 'outputs' / 'final'
    
    # Detect anomalies
    detector = AnomalyDetector(data_dir)
    anomalies = detector.detect_all_anomalies()
    
    if not anomalies:
        logger.warning("No anomalies detected! Check data availability.")
        return
    
    # Save anomalies to JSON
    anomalies_file = output_dir / 'detected_anomalies.json'
    with open(anomalies_file, 'w') as f:
        json.dump(anomalies, f, indent=2)
    logger.info(f"\n✓ Anomalies saved to: {anomalies_file}")
    
    # Generate map
    map_gen = InteractiveMapGenerator(output_dir)
    map_file = map_gen.create_map(anomalies)
    guide_file = map_gen.create_legend(anomalies)
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\n📍 Interactive Map: {map_file}")
    logger.info(f"📖 User Guide: {guide_file}")
    logger.info(f"📊 Anomaly Data: {anomalies_file}")
    logger.info(f"\nTotal anomalies detected: {len(anomalies)}")
    logger.info("\n🌐 Open the interactive_anomaly_map.html file in your browser!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()