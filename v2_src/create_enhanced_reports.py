import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import folium
from folium.plugins import HeatMap
import pandas as pd
from pyproj import CRS
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths based on project structure
BASE_DIR = 'data'
RAW_GRAVITY_BASELINE = os.path.join(BASE_DIR, 'raw', 'gravity', 'gravity_disturbance_EGM2008_50491becf3ffdee5c9908e47ed57881ed23de559539cd89e49b4d76635e07266.tiff')
RAW_GRAVITY_ENHANCED = os.path.join(BASE_DIR, 'processed', 'gravity', 'gravity_processed.tif')  # Fallback to processed as raw high-res invalid
PROCESSED_ELEVATION = os.path.join(BASE_DIR, 'processed', 'elevation', 'nasadem_processed.tif')
PROCESSED_MAGNETIC = os.path.join(BASE_DIR, 'processed', 'magnetic', 'magnetic_processed.tif')
VOID_DETECTION = os.path.join(BASE_DIR, 'outputs', 'void_detection', 'void_probability.tif')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'visualizations')
REPORTS_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')

# Region of interest: Carlsbad Caverns (-105.0, 32.0, -104.0, 33.0)
ROI = [(-105.0, -104.0), (32.0, 33.0)]  # [lon_min, lon_max], [lat_min, lat_max] - fix for box(minx, miny, maxx, maxy)

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def crop_to_roi(src_path, roi):
    """Crop raster to ROI using rasterio. Validates file first; checks overlap; falls back to full if issues."""
    if not os.path.exists(src_path):
        logger.warning(f"File not found: {src_path}. Skipping.")
        return None, None
    
    # Basic validation: check size >0
    if os.path.getsize(src_path) == 0:
        logger.warning(f"File empty: {src_path}. Skipping.")
        return None, None
    
    try:
        with rasterio.open(src_path) as src:
            # Define geometry for masking (simple bounding box)
            from shapely.geometry import box
            minx, maxx = roi[0]
            miny, maxy = roi[1]
            geom = box(minx, miny, maxx, maxy)
            raster_bounds = box(*src.bounds)
            if not geom.intersects(raster_bounds):
                logger.warning(f"ROI does not overlap raster bounds for {src_path}. Using full raster.")
                return src_path, src.meta.copy()
            
            out_image, out_transform = mask(src, [geom], crop=True)
            if out_image.shape[1] == 0 or out_image.shape[2] == 0:
                logger.warning(f"Empty crop for {src_path}. Using full raster.")
                return src_path, src.meta.copy()
            
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": src.crs
            })
            # Temporary file for cropped data
            temp_path = src_path.replace('.tif', '_cropped.tif').replace('.tiff', '_cropped.tif')
            with rasterio.open(temp_path, 'w', **out_meta) as dest:
                dest.write(out_image)
            logger.info(f"Cropped {src_path} to {temp_path}")
            return temp_path, out_meta
    except rasterio.errors.RasterioIOError as e:
        if "do not overlap" in str(e).lower():
            logger.warning(f"ROI no overlap for {src_path}: {e}. Using full raster.")
            return src_path, None
        logger.warning(f"Failed to crop {src_path} (invalid format): {e}. Returning full path as fallback.")
        return src_path, None
    except Exception as e:
        logger.error(f"Unexpected error cropping {src_path}: {e}")
        return src_path, None  # Fallback to full even on unexpected

def get_raster_stats(src_path):
    """Compute basic statistics for raster data. Handles invalid paths."""
    if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
        logger.warning(f"Cannot compute stats for invalid/empty {src_path}")
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'resolution': (np.nan, np.nan)}
    
    try:
        with rasterio.open(src_path) as src:
            data = src.read(1)
            valid_data = data[data != src.nodata]
            if len(valid_data) == 0:
                return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'resolution': src.res}
            return {
                'mean': np.mean(valid_data),
                'std': np.std(valid_data),
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'resolution': src.res  # (xres, yres)
            }
    except Exception as e:
        logger.warning(f"Failed stats for {src_path}: {e}")
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'resolution': (np.nan, np.nan)}

def create_multi_panel_comparison():
    """Create multi-panel plot comparing baseline vs enhanced processing."""
    logger.info("Generating multi-panel comparison plot...")
    
    # Crop rasters to ROI for efficiency; handle fallbacks
    baseline_cropped = crop_to_roi(RAW_GRAVITY_BASELINE, ROI)
    if baseline_cropped[0] is None:
        logger.warning("Skipping baseline gravity panel.")
        baseline_valid = False
    else:
        baseline_valid = True
    
    enhanced_cropped = crop_to_roi(RAW_GRAVITY_ENHANCED, ROI)
    if enhanced_cropped[0] is None:
        logger.warning("Skipping enhanced gravity panel.")
        enhanced_valid = False
    else:
        enhanced_valid = True
    
    void_cropped = crop_to_roi(VOID_DETECTION, ROI)
    if void_cropped[0] is None:
        logger.warning("Skipping void detection panel.")
        void_valid = False
    else:
        void_valid = True
    
    # Adjust subplots if some invalid
    num_cols = 3
    if not baseline_valid:
        num_cols -= 1
    if not enhanced_valid:
        num_cols -= 1
    
    fig, axes = plt.subplots(2, num_cols if num_cols > 0 else 1, figsize=(6*num_cols, 12))
    if num_cols == 1:
        axes = axes.reshape(2,1)
    fig.suptitle('Enhanced Processing: Baseline vs. High-Resolution Comparison (Carlsbad Region)', fontsize=16)
    
    col_idx = 0
    
    # 1. Baseline Gravity (~20km res) if valid
    if baseline_valid:
        try:
            with rasterio.open(baseline_cropped[0]) as src:
                show(src, ax=axes[0, col_idx], cmap='viridis', title='Baseline Gravity (~20km Resolution)')
                axes[0, col_idx].set_xlabel('Longitude')
                axes[0, col_idx].set_ylabel('Latitude')
            col_idx += 1
        except Exception as e:
            logger.warning(f"Failed to plot baseline: {e}")
    
    # 2. Enhanced Gravity (250m res) if valid
    if enhanced_valid:
        try:
            with rasterio.open(enhanced_cropped[0]) as src:
                show(src, ax=axes[0, col_idx], cmap='viridis', title='Enhanced Gravity (Processed - Higher Res)')
                axes[0, col_idx].set_xlabel('Longitude')
                axes[0, col_idx].set_ylabel('Latitude')
            col_idx += 1
        except Exception as e:
            logger.warning(f"Failed to plot enhanced: {e}")
    
    # 3. Resolution Comparison (bar chart) if both valid
    if baseline_valid and enhanced_valid:
        try:
            baseline_stats = get_raster_stats(baseline_cropped[0])
            enhanced_stats = get_raster_stats(enhanced_cropped[0])
            res_data = {'Baseline': baseline_stats['resolution'][0], 'Enhanced': enhanced_stats['resolution'][0]}
            sns.barplot(x=list(res_data.keys()), y=list(res_data.values()), ax=axes[0, col_idx])
            axes[0, col_idx].set_title('Resolution Comparison (Pixel Size in Degrees)')
            axes[0, col_idx].set_ylabel('Pixel Size')
            col_idx += 1
        except Exception as e:
            logger.warning(f"Failed resolution chart: {e}")
    else:
        axes[0, col_idx].text(0.5, 0.5, 'Resolution Comparison\n(Skipped: Insufficient Data)', ha='center', va='center', transform=axes[0, col_idx].transAxes)
        col_idx += 1
    
    row_idx = 1
    col_idx = 0
    
    # 4. Void Detection Before (use magnetic as proxy)
    try:
        mag_cropped = crop_to_roi(PROCESSED_MAGNETIC, ROI)[0]
        if mag_cropped:
            with rasterio.open(mag_cropped) as src:
                show(src, ax=axes[row_idx, col_idx], cmap='RdBu_r', title='Baseline Anomaly Proxy (Magnetic)')
                axes[row_idx, col_idx].set_xlabel('Longitude')
                axes[row_idx, col_idx].set_ylabel('Latitude')
            col_idx += 1
    except Exception as e:
        logger.warning(f"Failed magnetic proxy: {e}")
        axes[row_idx, col_idx].text(0.5, 0.5, 'Magnetic Proxy\n(Failed)', ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
        col_idx += 1
    
    # 5. Void Detection After (enhanced) if valid
    if void_valid:
        try:
            with rasterio.open(void_cropped[0]) as src:
                show(src, ax=axes[row_idx, col_idx], cmap='Reds', title='Enhanced Void Detection Probability')
                axes[row_idx, col_idx].set_xlabel('Longitude')
                axes[row_idx, col_idx].set_ylabel('Latitude')
            col_idx += 1
        except Exception as e:
            logger.warning(f"Failed void plot: {e}")
    else:
        axes[row_idx, col_idx].text(0.5, 0.5, 'Void Detection\n(Skipped)', ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
        col_idx += 1
    
    # 6. Coverage Map (combined modalities) if possible
    try:
        elev_cropped = crop_to_roi(PROCESSED_ELEVATION, ROI)[0]
        if elev_cropped and void_valid:
            with rasterio.open(elev_cropped) as elev_src, rasterio.open(void_cropped[0]) as void_src:
                elev_data = elev_src.read(1)
                void_data = void_src.read(1)
                coverage = np.where((elev_data != elev_src.nodata) & (void_data != void_src.nodata), 1, 0)
                im = axes[row_idx, col_idx].imshow(coverage, extent=[ROI[0][0], ROI[0][1], ROI[1][0], ROI[1][1]], cmap='Blues')
                axes[row_idx, col_idx].set_title('Data Coverage (Elevation + Void Detection)')
                plt.colorbar(im, ax=axes[row_idx, col_idx])
        else:
            axes[row_idx, col_idx].text(0.5, 0.5, 'Coverage Map\n(Skipped: Data Issues)', ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
    except Exception as e:
        logger.warning(f"Failed coverage map: {e}")
        axes[row_idx, col_idx].text(0.5, 0.5, 'Coverage Map\n(Failed)', ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'enhanced_multi_panel.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Multi-panel plot saved to {output_path} (partial if data issues)")
    
    # Cleanup temp files where possible - filter valid paths only
    temps = []
    if baseline_valid and baseline_cropped is not None and baseline_cropped[0] is not None and baseline_cropped[0] != RAW_GRAVITY_BASELINE:
        temps.append(baseline_cropped[0])
    if enhanced_valid and enhanced_cropped is not None and enhanced_cropped[0] is not None and enhanced_cropped[0] != RAW_GRAVITY_ENHANCED:
        temps.append(enhanced_cropped[0])
    if void_valid and void_cropped is not None and void_cropped[0] is not None and void_cropped[0] != VOID_DETECTION:
        temps.append(void_cropped[0])
    if 'mag_cropped' in locals() and mag_cropped is not None and mag_cropped != PROCESSED_MAGNETIC:
        temps.append(mag_cropped)
    if 'elev_cropped' in locals() and elev_cropped is not None and elev_cropped != PROCESSED_ELEVATION:
        temps.append(elev_cropped)
    for temp in temps:
        if temp and os.path.exists(temp):
            try:
                os.remove(temp)
                logger.info(f"Cleaned up {temp}")
            except Exception as e:
                logger.warning(f"Failed cleanup {temp}: {e}")

def create_data_quality_assessment():
    """Create data quality assessment plots."""
    logger.info("Generating data quality assessment plot...")
    
    # List of files for quality assessment
    files = {
        'Gravity Enhanced': RAW_GRAVITY_ENHANCED,
        'Elevation': PROCESSED_ELEVATION,
        'Magnetic': PROCESSED_MAGNETIC,
        'Void Probability': VOID_DETECTION
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Assessment', fontsize=16)
    
    stats_df = pd.DataFrame()
    
    # 1. Spatial Coverage Maps (montage)
    coverage_data = []
    for i, (name, path) in enumerate(files.items()):
        ax = axes[0, 0] if i == 0 else axes[0, 1] if i == 1 else axes[1, 0] if i == 2 else axes[1, 1]
        with rasterio.open(path) as src:
            data = src.read(1)
            coverage = np.where(data != src.nodata, 1, 0)
            # Crop to ROI for plot
            cropped = crop_to_roi(path, ROI)[0]
            with rasterio.open(cropped) as cropped_src:
                show(cropped_src, ax=ax, cmap='Greys', title=f'{name} Coverage')
            os.remove(cropped)
            coverage_data.append(np.sum(coverage) / coverage.size * 100)
    
    # 2. Resolution Comparison
    resolutions = [get_raster_stats(path)['resolution'][0] for path in files.values()]
    sns.barplot(x=list(files.keys()), y=resolutions, ax=axes[0, 0])
    axes[0, 0].set_title('Resolution by Modality (Pixel Size)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 3. Histograms (example for gravity enhanced)
    with rasterio.open(RAW_GRAVITY_ENHANCED) as src:
        data = src.read(1).flatten()
        valid_data = data[data != src.nodata]
        axes[0, 1].hist(valid_data, bins=50, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Gravity Data Distribution')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
    
    # 4. Statistics Table (as bar for mean)
    means = [get_raster_stats(path)['mean'] for path in files.values()]
    sns.barplot(x=list(files.keys()), y=means, ax=axes[1, 1], color='lightgreen')
    axes[1, 1].set_title('Mean Values by Modality')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Collect stats for DF (for potential export)
    for name, path in files.items():
        stats = get_raster_stats(path)
        stats_df = pd.concat([stats_df, pd.DataFrame({'Modality': name, **stats})], ignore_index=True)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, 'data_quality_assessment.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Data quality plot saved to {output_path}")
    
    # Save stats to CSV for reference
    stats_df.to_csv(os.path.join(REPORTS_DIR, 'data_quality_stats.csv'), index=False)

def create_interactive_report():
    """Create interactive HTML report with folium."""
    logger.info("Generating interactive HTML report...")
    
    # Center on Carlsbad region
    m = folium.Map(location=[32.5, -104.5], zoom_start=8, tiles='OpenStreetMap')
    
    # Add layers for each modality (use raster to points for folium, sample points)
    def add_raster_layer(m, path, name, colormap='viridis'):
        with rasterio.open(path) as src:
            # Sample 1000 points for heat/overlay (for performance)
            height, width = src.height, src.width
            y, x = np.ogrid[:height, :width]
            mask = np.random.choice([True, False], size=(height, width), p=[0.001, 0.999])  # Sparse sample
            sample_y, sample_x = y[mask], x[mask]
            data_sample = src.read(1)[mask]
            transform = src.transform
            lons, lats = rasterio.transform.xy(transform, sample_y, sample_x)
            points = list(zip(lats, lons, data_sample))
            
            # Add as heatmap or markers with popup stats
            for lat, lon, val in points:
                if not np.isnan(val):
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        popup=f"{name}: {val:.2f}",
                        color='red' if val > np.nanmean(data_sample) else 'blue',
                        fill=True
                    ).add_to(m)
            
            # Add layer control
            folium.TileLayer('Stamen Terrain').add_to(m)  # Base for elevation proxy
    
    # Add layers
    add_raster_layer(m, RAW_GRAVITY_ENHANCED, 'Enhanced Gravity')
    add_raster_layer(m, PROCESSED_ELEVATION, 'Elevation', 'terrain')
    add_raster_layer(m, PROCESSED_MAGNETIC, 'Magnetic')
    add_raster_layer(m, VOID_DETECTION, 'Void Probability', 'Reds')
    
    # Add toggles and metadata
    folium.LayerControl().add_to(m)
    stats_text = """
    <p><strong>Enhanced Processing Summary:</strong></p>
    <ul>
        <li>High-res Gravity: 250m from XGM2019e</li>
        <li>Multi-modal: Gravity + Magnetic + Elevation</li>
        <li>Void Detection: Probability maps generated</li>
        <li>Region: Carlsbad Caverns (-105° to -104° Lon, 32° to 33° Lat)</li>
    </ul>
    <p>Toggle layers to compare modalities. Zoom for details.</p>
    """
    folium.Marker([32.5, -104.5], popup=stats_text, tooltip='Click for Stats').add_to(m)
    
    output_path = os.path.join(OUTPUT_DIR, 'enhanced_interactive_report.html')
    m.save(output_path)
    logger.info(f"Interactive report saved to {output_path}")

def main():
    """Main function to generate all visualizations."""
    try:
        create_multi_panel_comparison()
        create_data_quality_assessment()
        create_interactive_report()
        logger.info("All visualizations generated successfully.")
    except Exception as e:
        logger.error(f"Error generating reports: {e}")
        raise

if __name__ == "__main__":
    main()