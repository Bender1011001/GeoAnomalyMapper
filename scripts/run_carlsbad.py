import os
from pathlib import Path
import numpy as np
import rasterio as rio

from gam.visualization.globe_viewer import GlobeViewer

def robust_z(x: np.ndarray, clamp: float = 6.0) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    med = np.nanmedian(x)
    if np.isnan(med):
        return np.full_like(x, np.nan, dtype="float32")

    mad = np.nanmedian(np.abs(x - med))

    if mad == 0 or np.isnan(mad):
        z = np.zeros_like(x, dtype="float32")
        z[np.isnan(x)] = np.nan
        return z

    scale = np.float32(1.4826) * np.float32(mad)
    z = (x - np.float32(med)) / scale
    z = np.clip(z, -np.float32(clamp), np.float32(clamp), out=z)
    z[np.isnan(x)] = np.nan
    return z.astype("float32", copy=False)


def main() -> None:
    # Known site: Carlsbad Caverns, NM
    analysis_id = "voids_carlsbad"
    min_lon, min_lat, max_lon, max_lat = -104.8, 31.9, -104.3, 32.3
    bbox = (min_lon, min_lat, max_lon, max_lat)

    # EMAG2 global anomaly raster (mounted under /app/data inside containers)
    src_path = Path("/app/data/raw/emag2/EMAG2_V3_SeaLevel_DataTiff.tif")
    if not src_path.exists():
        raise FileNotFoundError(f"EMAG2 GeoTIFF not found at {src_path}")

    with rio.open(src_path) as ds:
        # Assuming EMAG2 is in EPSG:4326
        window = rio.windows.from_bounds(min_lon, min_lat, max_lon, max_lat, ds.transform)
        arr = ds.read(1, window=window, boundless=True, fill_value=np.nan).astype("float32")

    # Compute robust z-scores and normalize to 0..1, handling NaNs
    z = robust_z(arr)
    if np.isfinite(z).any():
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        anomalies = (z - zmin) / (zmax - zmin + 1e-12)
    else:
        anomalies = np.zeros_like(z, dtype="float32")

    # Build Cesium scene
    token = os.getenv("CESIUM_TOKEN", "")
    gv = GlobeViewer(cesium_token=token)
    gv.add_anomaly_heatmap(anomalies, bbox, cmap="hot", opacity=0.7, name="EMAG2 z-score")

    # Optional: mark top 1% anomaly pixels as cylinders
    try:
        thr = float(np.nanquantile(anomalies, 0.99))
        ys, xs = np.where(anomalies >= thr)
        pts = []
        h, w = anomalies.shape
        for y, x in list(zip(ys, xs))[:150]:
            val = float(anomalies[y, x])
            if not np.isfinite(val):
                continue
            lat = min_lat + (y / max(h - 1, 1)) * (max_lat - min_lat)
            lon = min_lon + (x / max(w - 1, 1)) * (max_lon - min_lon)
            pts.append({"lat": float(lat), "lon": float(lon), "value": val, "label": f"{val:.3f}"})
        if pts:
            gv.add_point_entities(pts, name="Top 1% anomalies")
    except Exception:
        # Non-fatal if quantile fails on degenerate arrays
        pass

    # Camera aimed at AOI center
    gv.set_camera((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0, height_m=150000.0)

    # Persist outputs into the shared data volume for dashboard/API
    out_dir = Path("/app/data/outputs/state") / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.json").write_text(gv.export_scene_config(), encoding="utf-8")
    (out_dir / "scene.html").write_text(gv.render_streamlit_html(), encoding="utf-8")

    print(f"Wrote: {out_dir / 'scene.json'}")
    print(f"Wrote: {out_dir / 'scene.html'}")
    if not token:
        print("Note: CESIUM_TOKEN not set; terrain will be disabled. The scene will still render.")

if __name__ == "__main__":
    main()