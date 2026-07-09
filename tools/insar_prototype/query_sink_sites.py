"""Query the hi-res stack at the published sink coordinates; list all bowls."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"E:\code.projects\GeoAnomalyMapper-1")
sys.path.insert(0, str(REPO))
WORK = REPO / "data" / "insar_wink"

import rasterio
from rasterio.warp import transform as warp_transform
from scipy.ndimage import label

SITES = {
    "Wink_Sink_1": (31.7775, -103.1129),
    "Wink_Sink_2": (31.7632, -103.1129),
    "fast_area_E_of_WS2_2016": (31.7632, -103.1020),
}

with rasterio.open(WORK / "wink_hires_stack_rate_m_yr.tif") as src:
    sm = src.read(1).astype(np.float64)
    tr = src.transform
    crs = src.crs

inv = ~tr
for name, (lat, lon) in SITES.items():
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    c, r = inv * (xs[0], ys[0])
    r, c = int(r), int(c)
    if not (0 <= r < sm.shape[0] and 0 <= c < sm.shape[1]):
        print(f"{name}: outside box")
        continue
    win = sm[max(0, r - 8):r + 9, max(0, c - 8):c + 9]  # ~680 m window
    finite = win[np.isfinite(win)]
    if finite.size == 0:
        print(f"{name}: no coherent pixels (decorrelated)")
        continue
    print(f"{name}: rate_at_px={sm[r, c]*100 if np.isfinite(sm[r, c]) else float('nan'):.1f} cm/yr, "
          f"win_min={finite.min()*100:.1f} win_p50={np.percentile(finite, 50)*100:.1f} cm/yr, "
          f"coherent={finite.size}/{win.size}")

print("\nAll bowls <= -5 cm/yr (>=3 px):")
mask = np.isfinite(sm) & (sm <= -0.05)
labeled, n = label(mask)
rows_all = []
for k in range(1, n + 1):
    px = labeled == k
    if px.sum() < 3:
        continue
    rows, cols = np.where(px)
    v = sm[rows, cols]
    pi = int(np.argmax(-v))
    x, y = tr * (int(cols[pi]) + 0.5, int(rows[pi]) + 0.5)
    lons, lats = warp_transform(crs, "EPSG:4326", [x], [y])
    la, lo = float(lats[0]), float(lons[0])
    dists = {nm: float(np.hypot((la - s[0]) * 111.32, (lo - s[1]) * 111.32 * np.cos(np.radians(la))))
             for nm, s in SITES.items()}
    nearest = min(dists, key=dists.get)
    print(f"  peak={v[pi]*100:6.1f} cm/yr px={px.sum():3d} at ({la:.5f}, {lo:.5f}) "
          f"nearest={nearest} ({dists[nearest]:.2f} km)")
