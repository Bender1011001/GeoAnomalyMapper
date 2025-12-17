import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import gaussian_filter


def load_raster(path: str) -> tuple[np.ndarray, dict]:
    """
    Load a single-band raster as numpy array and its profile.

    Handles nodata as NaN with robust dtype conversion for integer rasters.
    Critical fix: Converts uint8/uint16 coherence maps to float32 before NaN filling.

    :param path: Path to the raster file.
    :return: (data: np.ndarray, profile: dict)
    :raises rasterio.errors.RasterioIOError: If file cannot be opened.
    :example:
        >>> data, profile = load_raster('coh_mean.tif')
    """
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        
        # Convert integer types to float32 before filling with NaN
        # This prevents "Cannot convert fill_value nan to dtype uint8" errors
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32, copy=False)
        
        data = data.filled(np.nan)
        return data, src.profile


def save_raster(path: str, data: np.ndarray, profile: dict) -> None:
    """
    Save a single-band float32 raster with optimized compression and tiling.

    :param path: Output path.
    :param data: 2D numpy array.
    :param profile: Rasterio profile dict.
    """
    profile = profile.copy()
    profile.update({
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'compress': 'DEFLATE',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
    })
    if data.nbytes > 4e9:
        profile['BIGTIFF'] = 'YES'

    data_clean = np.nan_to_num(data, nan=np.nan)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data_clean.astype('float32'), 1)


def compute_coherence_change_detection(
    coh_paths_list: list[str],
    output_ccd_path: str,
    window_size: int = 5  # Reserved for future local spatial smoothing
) -> None:
    """
    Compute temporal coherence change detection (CCD) from a stack of coherence rasters.

    Loads coherence rasters, stacks them, computes temporal standard deviation,
    applies Gaussian smoothing, and normalizes to 0-1 range.

    :param coh_paths_list: List of paths to coherence GeoTIFFs (T >= 2).
    :param output_ccd_path: Output path for normalized CCD raster (high = decorrelation/activity).
    :param window_size: Reserved parameter (not used in current temporal std implementation).
    :raises ValueError: If fewer than 2 rasters or shape mismatch.
    :example:
        coh_paths = ['coh_t1.tif', 'coh_t2.tif']
        compute_coherence_change_detection(coh_paths, 'coherence_change.tif')
        # For stack [[[0.9,0.8],[0.7,0.6]], [[0.8,0.7],[0.6,0.5]]],
        # std ≈ [[0.0707,0.0707],[0.0707,0.0707]]; normalized to 0 (constant) or scaled ~0.1 raw.
    """
    if len(coh_paths_list) < 2:
        raise ValueError("At least 2 coherence rasters are required for temporal analysis.")

    stack = []
    first_profile = None
    first_shape = None
    for i, path in enumerate(coh_paths_list):
        data, profile = load_raster(path)
        data = np.nan_to_num(data, nan=0.0)
        stack.append(data)
        if i == 0:
            first_profile = profile
            first_shape = data.shape
        elif data.shape != first_shape:
            raise ValueError(f"Raster shape mismatch at {path}: {data.shape} != {first_shape}")

    stack_array = np.stack(stack, axis=0)
    temporal_std = np.std(stack_array, axis=0, ddof=0)
    smoothed = gaussian_filter(temporal_std, sigma=1.0)

    min_val = np.nanmin(smoothed)
    max_val = np.nanmax(smoothed)
    if np.isfinite(max_val) and max_val > min_val:
        normalized = (smoothed - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(smoothed)

    save_raster(output_ccd_path, normalized, first_profile)


def compute_glcm_texture(
    coh_mean_path: str,
    output_homogeneity_path: str,
    output_entropy_path: str,
    distances: list[int] = [1, 2],
    angles: list[float] = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels: int = 32
) -> None:
    """
    Compute GLCM-based texture features (homogeneity, low-entropy proxy) on coherence mean raster.

    Quantizes input to specified levels, computes global GLCM, extracts mean properties,
    saves as constant-value rasters matching input geometry.

    :param coh_mean_path: Path to mean coherence raster (0-1 float).
    :param output_homogeneity_path: Output homogeneity raster (0-1, high=smooth/artificial).
    :param output_entropy_path: Output -entropy proxy raster (high=ordered/low disorder).
    :param distances: GLCM pixel distances.
    :param angles: GLCM orientation angles (radians).
    :param levels: Quantization levels (lower=faster).
    :raises ValueError: If levels < 2.
    :example:
        # Smooth uniform image → homogeneity ~1.0
        # Noisy varied image → homogeneity ~0.5, higher dissimilarity → lower orderedness
    """
    if levels < 2:
        raise ValueError("GLCM levels must be at least 2.")

    data, profile = load_raster(coh_mean_path)
    data_finite = np.nan_to_num(data, nan=0.0)
    data_clipped = np.clip(data_finite, 0.0, 1.0)
    image = np.round(data_clipped * (levels - 1)).astype(np.uint8)

    orig_h, orig_w = profile['height'], profile['width']
    h, w = image.shape

    # Pad if too small for GLCM (min ~5x5 for dist=2)
    min_size = 5
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')

    glcm = graycomatrix(
        image, distances=distances, angles=angles,
        levels=levels, symmetric=True, normed=True
    )

    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    max_dissimilarity = levels - 1
    entropy_proxy_norm = dissimilarity / max_dissimilarity if max_dissimilarity > 0 else 0.0
    orderedness = 1.0 - entropy_proxy_norm  # Proxy for -entropy (high=low disorder/ordered)

    homog_array = np.full((orig_h, orig_w), homogeneity, dtype=np.float32)
    entropy_array = np.full((orig_h, orig_w), orderedness, dtype=np.float32)

    save_raster(output_homogeneity_path, homog_array, profile)
    save_raster(output_entropy_path, entropy_array, profile)


def compute_structural_artificiality(
    ccd_path: str,
    homogeneity_path: str,
    output_art_path: str,
    alpha: float = 0.5
) -> None:
    """
    Compute structural artificiality probability from CCD and homogeneity.

    Weighted combination: alpha * homogeneity + (1-alpha) * ccd_norm
    (high Δγ decorrelation + high homogeneity → artificial structures/activity).

    :param ccd_path: Normalized CCD raster (high=decorrelation).
    :param homogeneity_path: Homogeneity raster (high=smooth).
    :param output_art_path: Output artificiality probability raster (0-1).
    :param alpha: Weight for homogeneity (0-1).
    :raises ValueError: If raster shapes mismatch.
    :example:
        homog=0.8, Δγ_norm=0.6, alpha=0.5 → 0.5*0.8 + 0.5*0.6 = 0.7
    """
    ccd_data, profile = load_raster(ccd_path)
    homog_data, h_profile = load_raster(homogeneity_path)

    if ccd_data.shape != homog_data.shape:
        raise ValueError(f"Shape mismatch: CCD {ccd_data.shape} vs homogeneity {homog_data.shape}")

    artificiality = alpha * homog_data + (1 - alpha) * ccd_data
    artificiality = np.clip(artificiality, 0.0, 1.0)

    save_raster(output_art_path, artificiality, profile)


def extract_insar_features(coh_paths: list[str], coh_mean_path: str) -> None:
    """
    Extract Phase 2 InSAR features: CCD, GLCM texture, structural artificiality.

    Orchestrates the three core functions, producing three output rasters.

    :param coh_paths: List of coherence raster paths.
    :param coh_mean_path: Mean coherence raster path.
    :raises ValueError: Propagated from sub-functions.
    :example:
        >>> extract_insar_features(['coh_t1.tif', 'coh_t2.tif'], 'coh_mean.tif')
        Phase 2 outputs: coherence_change.tif, texture_homogeneity.tif, structural_artificiality.tif
    """
    compute_coherence_change_detection(coh_paths, 'coherence_change.tif')
    compute_glcm_texture(coh_mean_path, 'texture_homogeneity.tif', 'texture_entropy.tif')
    compute_structural_artificiality(
        'coherence_change.tif', 'texture_homogeneity.tif', 'structural_artificiality.tif'
    )
    print("Phase 2 outputs: coherence_change.tif, texture_homogeneity.tif, structural_artificiality.tif")