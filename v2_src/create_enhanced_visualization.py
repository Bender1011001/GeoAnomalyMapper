#!/usr/bin/env python3
"""
Enhanced Underground Visualization Tool
=========================================

Creates comprehensive multi-panel visualizations showing:
- Gravity anomalies (mass deficits = potential voids)
- Magnetic anomalies (structural features)
- Combined probability maps
- Cross-sections and depth profiles
- Interactive 3D models
- Comparison overlays

Usage:
    python create_enhanced_visualization.py --region "-105.0,32.0,-104.0,33.0"
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, LightSource
import rasterio
from rasterio.plot import show
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BASE_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs" / "visualizations"

# ============================================================================
# CUSTOM COLORMAPS
# ============================================================================

def create_void_colormap():
    """Create colormap optimized for void detection (purple=low, red=high)."""
    colors = ['#2b1d52', '#3d2d7d', '#5a4ca8', '#7a6dc2', 
              '#9b8dd4', '#bbaee5', '#ddd0f0',  # Purple shades
              '#f0d0dd', '#e5aebe', '#d98c9f',  # Pink transition
              '#cc6a80', '#bf4861', '#b22642',  # Red shades
              '#a50f15', '#8b0000']  # Dark red
    return LinearSegmentedColormap.from_list('void_detection', colors, N=256)

def create_anomaly_colormap():
    """Create diverging colormap for anomalies (blue=negative, red=positive)."""
    colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0',
              '#f7f7f7',  # White center
              '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    return LinearSegmentedColormap.from_list('anomaly', colors, N=256)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_raster_data(filepath):
    """Load raster data and return array and metadata."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            extent = [src.bounds.left, src.bounds.right, 
                     src.bounds.bottom, src.bounds.top]
            return data, extent, src.crs
    except Exception as e:
        logger.warning(f"Could not load {filepath}: {e}")
        return None, None, None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_multi_panel_view(gravity_data, magnetic_data, probability_data, 
                           extent, output_path):
    """Create comprehensive multi-panel visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    void_cmap = create_void_colormap()
    anomaly_cmap = create_anomaly_colormap()
    
    # ========== Panel 1: Gravity Anomaly ==========
    ax1 = fig.add_subplot(gs[0, 0])
    if gravity_data is not None:
        # Normalize gravity data for better visualization
        vmin, vmax = np.nanpercentile(gravity_data, [5, 95])
        im1 = ax1.imshow(gravity_data, extent=extent, cmap=anomaly_cmap,
                        vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im1, ax=ax1, label='Gravity Anomaly (mGal)', fraction=0.046)
        ax1.set_title('Gravity Anomaly\n(Negative = Mass Deficit/Potential Void)', 
                     fontsize=12, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'Gravity Data\nNot Available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Gravity Anomaly', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.grid(True, alpha=0.3)
    
    # ========== Panel 2: Magnetic Anomaly ==========
    ax2 = fig.add_subplot(gs[0, 1])
    if magnetic_data is not None:
        vmin, vmax = np.nanpercentile(magnetic_data, [5, 95])
        im2 = ax2.imshow(magnetic_data, extent=extent, cmap=anomaly_cmap,
                        vmin=vmin, vmax=vmax, aspect='auto')
        plt.colorbar(im2, ax=ax2, label='Magnetic Anomaly (nT)', fraction=0.046)
        ax2.set_title('Magnetic Anomaly\n(Structural Features)', 
                     fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Magnetic Data\nNot Available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Magnetic Anomaly', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.grid(True, alpha=0.3)
    
    # ========== Panel 3: Void Probability ==========
    ax3 = fig.add_subplot(gs[0, 2])
    if probability_data is not None:
        im3 = ax3.imshow(probability_data, extent=extent, cmap=void_cmap,
                        vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im3, ax=ax3, label='Void Probability', fraction=0.046)
        ax3.set_title('Void Probability Map\n(Combined Analysis)', 
                     fontsize=12, fontweight='bold')
        
        # Add contour lines for high-probability zones
        levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        contours = ax3.contour(probability_data, levels=levels, extent=extent,
                              colors='white', linewidths=1.5, alpha=0.7)
        ax3.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    else:
        ax3.text(0.5, 0.5, 'Probability Data\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Void Probability', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Longitude (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.grid(True, alpha=0.3)
    
    # ========== Panel 4: Hillshade Gravity (3D effect) ==========
    ax4 = fig.add_subplot(gs[1, 0])
    if gravity_data is not None:
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade = ls.hillshade(gravity_data, vert_exag=10)
        im4 = ax4.imshow(hillshade, extent=extent, cmap='gray', aspect='auto')
        ax4.set_title('Gravity Hillshade\n(3D Perspective)', 
                     fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Hillshade\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Gravity Hillshade', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Longitude (°)')
    ax4.set_ylabel('Latitude (°)')
    ax4.grid(True, alpha=0.3)
    
    # ========== Panel 5: Combined Overlay ==========
    ax5 = fig.add_subplot(gs[1, 1])
    if gravity_data is not None and probability_data is not None:
        # Show gravity as base with probability overlay
        vmin, vmax = np.nanpercentile(gravity_data, [5, 95])
        ax5.imshow(gravity_data, extent=extent, cmap='gray', 
                  vmin=vmin, vmax=vmax, aspect='auto', alpha=0.6)
        im5 = ax5.imshow(probability_data, extent=extent, cmap=void_cmap,
                        vmin=0, vmax=1, aspect='auto', alpha=0.7)
        plt.colorbar(im5, ax=ax5, label='Void Probability', fraction=0.046)
        ax5.set_title('Combined View\n(Gravity + Probability Overlay)', 
                     fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Combined View\nNot Available', 
                ha='center', va='center', transform=ax5.transAxes, fontsize=14)
        ax5.set_title('Combined Overlay', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Longitude (°)')
    ax5.set_ylabel('Latitude (°)')
    ax5.grid(True, alpha=0.3)
    
    # ========== Panel 6: High-Probability Zones ==========
    ax6 = fig.add_subplot(gs[1, 2])
    if probability_data is not None:
        # Threshold at 0.5 for high-probability zones
        high_prob_mask = probability_data >= 0.5
        masked_data = np.ma.masked_where(~high_prob_mask, probability_data)
        
        im6 = ax6.imshow(masked_data, extent=extent, cmap='hot_r',
                        vmin=0.5, vmax=1, aspect='auto')
        plt.colorbar(im6, ax=ax6, label='Probability > 0.5', fraction=0.046)
        ax6.set_title('High-Probability Zones Only\n(Threshold ≥ 0.5)', 
                     fontsize=12, fontweight='bold')
        
        # Calculate statistics
        high_prob_percentage = (np.sum(high_prob_mask) / high_prob_mask.size) * 100
        ax6.text(0.02, 0.98, f'Coverage: {high_prob_percentage:.1f}%',
                transform=ax6.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax6.text(0.5, 0.5, 'High-Probability\nZones Not Available', 
                ha='center', va='center', transform=ax6.transAxes, fontsize=14)
        ax6.set_title('High-Probability Zones', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Longitude (°)')
    ax6.set_ylabel('Latitude (°)')
    ax6.grid(True, alpha=0.3)
    
    # ========== Panel 7: Horizontal Cross-section (Latitude) ==========
    ax7 = fig.add_subplot(gs[2, :2])
    if gravity_data is not None:
        # Take middle row
        mid_row = gravity_data.shape[0] // 2
        x_coords = np.linspace(extent[0], extent[1], gravity_data.shape[1])
        
        ax7.plot(x_coords, gravity_data[mid_row, :], 'b-', linewidth=2, 
                label='Gravity', alpha=0.7)
        ax7.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax7.fill_between(x_coords, 0, gravity_data[mid_row, :], 
                        where=(gravity_data[mid_row, :] < 0),
                        color='red', alpha=0.3, label='Negative Anomaly (Potential Void)')
        ax7.set_xlabel('Longitude (°)', fontsize=10)
        ax7.set_ylabel('Gravity Anomaly (mGal)', fontsize=10)
        ax7.set_title(f'East-West Cross-Section (Latitude {(extent[2]+extent[3])/2:.2f}°)', 
                     fontsize=12, fontweight='bold')
        ax7.legend(loc='best')
        ax7.grid(True, alpha=0.3)
    else:
        ax7.text(0.5, 0.5, 'Cross-Section Not Available', 
                ha='center', va='center', transform=ax7.transAxes, fontsize=14)
        ax7.set_title('East-West Cross-Section', fontsize=12, fontweight='bold')
    
    # ========== Panel 8: Statistics and Legend ==========
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create statistics text
    stats_text = "SUBSURFACE ANALYSIS SUMMARY\n" + "="*40 + "\n\n"
    
    if gravity_data is not None:
        g_mean = np.nanmean(gravity_data)
        g_std = np.nanstd(gravity_data)
        g_min = np.nanmin(gravity_data)
        g_max = np.nanmax(gravity_data)
        stats_text += f"Gravity Statistics:\n"
        stats_text += f"  Mean: {g_mean:.2f} mGal\n"
        stats_text += f"  Std Dev: {g_std:.2f} mGal\n"
        stats_text += f"  Range: [{g_min:.2f}, {g_max:.2f}] mGal\n\n"
    
    if magnetic_data is not None:
        m_mean = np.nanmean(magnetic_data)
        m_std = np.nanstd(magnetic_data)
        m_min = np.nanmin(magnetic_data)
        m_max = np.nanmax(magnetic_data)
        stats_text += f"Magnetic Statistics:\n"
        stats_text += f"  Mean: {m_mean:.2f} nT\n"
        stats_text += f"  Std Dev: {m_std:.2f} nT\n"
        stats_text += f"  Range: [{m_min:.2f}, {m_max:.2f}] nT\n\n"
    
    if probability_data is not None:
        p_mean = np.nanmean(probability_data)
        p_max = np.nanmax(probability_data)
        high_prob = np.sum(probability_data >= 0.7)
        total_pixels = probability_data.size
        stats_text += f"Void Probability:\n"
        stats_text += f"  Mean: {p_mean:.3f}\n"
        stats_text += f"  Maximum: {p_max:.3f}\n"
        stats_text += f"  High Prob (≥0.7): {high_prob} pixels\n"
        stats_text += f"  Coverage: {(high_prob/total_pixels)*100:.2f}%\n\n"
    
    stats_text += "\nINTERPRETATION GUIDE:\n" + "-"*40 + "\n"
    stats_text += "• Negative gravity = mass deficit\n"
    stats_text += "  → Potential voids or low-density\n"
    stats_text += "     material\n\n"
    stats_text += "• Magnetic disruptions = structural\n"
    stats_text += "  changes or mineral variations\n\n"
    stats_text += "• High probability (>0.7) = strong\n"
    stats_text += "  indicators for subsurface voids\n\n"
    stats_text += "• Combined anomalies = increased\n"
    stats_text += "  confidence in detection"
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add main title
    fig.suptitle('Enhanced Subsurface Visualization - Multi-Panel Analysis',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved enhanced visualization: {output_path}")
    plt.close()


def create_depth_profile(gravity_data, extent, output_path):
    """Create estimated depth profile based on gravity anomalies."""
    
    if gravity_data is None:
        logger.warning("Cannot create depth profile without gravity data")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Panel 1: Gravity anomaly map
    anomaly_cmap = create_anomaly_colormap()
    vmin, vmax = np.nanpercentile(gravity_data, [5, 95])
    im1 = ax1.imshow(gravity_data, extent=extent, cmap=anomaly_cmap,
                    vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Gravity Anomaly (mGal)')
    ax1.set_title('Gravity Anomaly Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude (°)')
    ax1.set_ylabel('Latitude (°)')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Estimated depth profile
    # Simple depth estimation: depth ∝ -anomaly (negative = deeper)
    # This is a rough approximation
    depth_estimate = -gravity_data * 100  # Scale factor for visualization
    depth_estimate = np.clip(depth_estimate, 0, 300)  # Clip to 0-300 feet
    
    depth_cmap = plt.cm.YlOrRd  # Yellow to red
    im2 = ax2.imshow(depth_estimate, extent=extent, cmap=depth_cmap,
                    vmin=0, vmax=300, aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Estimated Depth (feet)')
    ax2.set_title('Estimated Void Depth Profile\n(Darker = Deeper, Approximate)', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Longitude (°)')
    ax2.set_ylabel('Latitude (°)')
    ax2.grid(True, alpha=0.3)
    
    # Add depth contours
    levels = [50, 100, 150, 200, 250]
    contours = ax2.contour(depth_estimate, levels=levels, extent=extent,
                          colors='black', linewidths=1, alpha=0.5)
    ax2.clabel(contours, inline=True, fontsize=8, fmt='%d ft')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved depth profile: {output_path}")
    plt.close()


def create_interactive_html(gravity_data, magnetic_data, probability_data, 
                           extent, output_path):
    """Create interactive HTML visualization."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Subsurface Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
        }}
        .interpretation {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        .interpretation h3 {{
            margin-top: 0;
            color: #856404;
        }}
        .interpretation ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .data-info {{
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🗺️ Interactive Subsurface Analysis</h1>
        
        <div class="data-info">
            <h3>📍 Analysis Region</h3>
            <p><strong>Longitude:</strong> {extent[0]:.4f}° to {extent[1]:.4f}°</p>
            <p><strong>Latitude:</strong> {extent[2]:.4f}° to {extent[3]:.4f}°</p>
            <p><strong>Area:</strong> ~{abs((extent[1]-extent[0]) * (extent[3]-extent[2])) * 111 * 111:.1f} km²</p>
        </div>
        
        <div class="stats-grid">
"""
    
    if gravity_data is not None:
        g_mean = np.nanmean(gravity_data)
        g_min = np.nanmin(gravity_data)
        g_max = np.nanmax(gravity_data)
        html_content += f"""
            <div class="stat-card">
                <h3>Gravity Anomaly</h3>
                <div class="value">{g_mean:.2f} mGal</div>
                <p>Mean Value</p>
                <small>Range: {g_min:.2f} to {g_max:.2f}</small>
            </div>
"""
    
    if magnetic_data is not None:
        m_mean = np.nanmean(magnetic_data)
        m_min = np.nanmin(magnetic_data)
        m_max = np.nanmax(magnetic_data)
        html_content += f"""
            <div class="stat-card">
                <h3>Magnetic Anomaly</h3>
                <div class="value">{m_mean:.2f} nT</div>
                <p>Mean Value</p>
                <small>Range: {m_min:.2f} to {m_max:.2f}</small>
            </div>
"""
    
    if probability_data is not None:
        p_mean = np.nanmean(probability_data)
        p_max = np.nanmax(probability_data)
        high_prob = np.sum(probability_data >= 0.7)
        total_pixels = probability_data.size
        html_content += f"""
            <div class="stat-card">
                <h3>Void Probability</h3>
                <div class="value">{p_mean:.3f}</div>
                <p>Average Probability</p>
                <small>Max: {p_max:.3f} | High: {(high_prob/total_pixels)*100:.1f}%</small>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="interpretation">
            <h3>🔍 Interpretation Guide</h3>
            <ul>
                <li><strong>Negative Gravity Anomalies:</strong> Indicate areas of mass deficit, which could represent:
                    <ul>
                        <li>Underground voids or caves</li>
                        <li>Low-density sediments</li>
                        <li>Karst features (dissolved limestone)</li>
                    </ul>
                </li>
                <li><strong>Magnetic Anomalies:</strong> Show structural variations:
                    <ul>
                        <li>Fault lines and fractures</li>
                        <li>Changes in rock type</li>
                        <li>Mineral deposits</li>
                    </ul>
                </li>
                <li><strong>High Probability Zones (>0.7):</strong> Areas with multiple converging anomalies suggesting strong evidence of subsurface voids</li>
            </ul>
        </div>
        
        <h2>📊 Detailed Analysis</h2>
        <table>
            <tr>
                <th>Measurement</th>
                <th>Value</th>
                <th>Significance</th>
            </tr>
"""
    
    if gravity_data is not None:
        neg_anomaly_count = np.sum(gravity_data < -5)
        html_content += f"""
            <tr>
                <td>Strong Negative Gravity Zones</td>
                <td>{neg_anomaly_count} locations</td>
                <td>Potential void sites (< -5 mGal)</td>
            </tr>
"""
    
    if probability_data is not None:
        very_high = np.sum(probability_data >= 0.8)
        high = np.sum((probability_data >= 0.6) & (probability_data < 0.8))
        medium = np.sum((probability_data >= 0.4) & (probability_data < 0.6))
        html_content += f"""
            <tr>
                <td>Very High Probability (≥0.8)</td>
                <td>{very_high} pixels ({(very_high/total_pixels)*100:.2f}%)</td>
                <td>Prime investigation targets</td>
            </tr>
            <tr>
                <td>High Probability (0.6-0.8)</td>
                <td>{high} pixels ({(high/total_pixels)*100:.2f}%)</td>
                <td>Secondary investigation areas</td>
            </tr>
            <tr>
                <td>Medium Probability (0.4-0.6)</td>
                <td>{medium} pixels ({(medium/total_pixels)*100:.2f}%)</td>
                <td>Monitoring recommended</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <div class="interpretation" style="background-color: #d4edda; border-left-color: #28a745;">
            <h3>✅ Next Steps</h3>
            <ol>
                <li>Review the multi-panel visualization for spatial patterns</li>
                <li>Investigate high-probability zones with field surveys</li>
                <li>Consider additional data sources (InSAR, LiDAR) for verification</li>
                <li>Cross-reference with known geological features</li>
                <li>Plan ground-truthing surveys for highest probability areas</li>
            </ol>
        </div>
        
        <div class="footer">
            <p>Generated by GeoAnomalyMapper Enhanced Visualization Tool</p>
            <p>For detailed images, see the visualization output directory</p>
        </div>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info(f"✓ Saved interactive HTML: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create enhanced visualizations of subsurface data"
    )
    parser.add_argument(
        '--region',
        type=str,
        default="-105.0,32.0,-104.0,33.0",
        help='Region bounds: "lon_min,lat_min,lon_max,lat_max"'
    )
    
    args = parser.parse_args()
    
    # Parse region
    try:
        coords = list(map(float, args.region.split(',')))
        if len(coords) != 4:
            raise ValueError("Need 4 coordinates")
    except Exception as e:
        logger.error(f"Invalid region format: {e}")
        return
    
    logger.info("="*70)
    logger.info("ENHANCED SUBSURFACE VISUALIZATION")
    logger.info("="*70)
    logger.info(f"Region: {coords}")
    logger.info("")
    
    # Load data
    logger.info("Loading processed data...")
    
    gravity_file = PROCESSED_DIR / "gravity" / "gravity_processed.tif"
    magnetic_file = PROCESSED_DIR / "magnetic" / "magnetic_processed.tif"
    probability_file = PROJECT_ROOT / "data" / "outputs" / "void_detection" / "void_probability.tif"
    
    gravity_data, extent, _ = load_raster_data(gravity_file)
    magnetic_data, _, _ = load_raster_data(magnetic_file)
    probability_data, _, _ = load_raster_data(probability_file)
    
    if extent is None:
        logger.error("Could not determine extent from data files")
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    logger.info("Creating multi-panel visualization...")
    create_multi_panel_view(
        gravity_data, magnetic_data, probability_data, extent,
        OUTPUT_DIR / "enhanced_multi_panel.png"
    )
    
    logger.info("Creating depth profile...")
    create_depth_profile(
        gravity_data, extent,
        OUTPUT_DIR / "depth_profile.png"
    )
    
    logger.info("Creating interactive HTML report...")
    create_interactive_html(
        gravity_data, magnetic_data, probability_data, extent,
        OUTPUT_DIR / "interactive_report.html"
    )
    
    logger.info("")
    logger.info("="*70)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  1. enhanced_multi_panel.png - Comprehensive multi-view analysis")
    logger.info(f"  2. depth_profile.png - Estimated void depth visualization")
    logger.info(f"  3. interactive_report.html - Interactive web report")
    logger.info("="*70)
    logger.info("")
    logger.info(f"📊 Open {OUTPUT_DIR / 'interactive_report.html'} in your browser")
    logger.info(f"🖼️  View {OUTPUT_DIR / 'enhanced_multi_panel.png'} for detailed analysis")


if __name__ == "__main__":
    main()